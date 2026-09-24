"""Weekend dislocation fade, re-measured for the actual paper venue.

Keyless public GETs only (live v2 candles already cached in ../comp/wb; UTA Demo v3 candles via the
`paptrading: 1` header). No account endpoint, no order, no LLM call.

Questions:
 1. Does UTA Demo price the US-equity perps like live (hourly close gap, mark vs index flash prints)?
 2. Restricted to the 11 names UTA Demo actually lists, does the -1% Saturday-morning fade survive?
 3. Does it survive an exit moved to Sunday 12:00 UTC (inside a 9/27 15:59 UTC cutoff)?
"""
import sys, json, os, math, random, statistics as st, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import candles_v3  # noqa: E402

H = 3600 * 1000
HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, 'wb')
DEMO11 = ['HOODUSDT', 'COINUSDT', 'MSTRUSDT', 'METAUSDT', 'AMZNUSDT', 'GOOGLUSDT', 'AAPLUSDT',
          'CRCLUSDT', 'NVDAUSDT', 'TSLAUSDT', 'SNDKUSDT']
ALL = sorted(f[:-5] for f in os.listdir(WB))

def load_live(s):
    b = json.load(open(os.path.join(WB, s + '.json')))
    return {int(x[0]): (float(x[1]), float(x[4])) for x in b}

live = {s: load_live(s) for s in ALL}
demo, mark, index = {}, {}, {}
for s in DEMO11:
    demo[s] = {int(r[0]): (r[1], r[4]) for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50)}
    mark[s] = {int(r[0]): r[4] for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50, kind='mark')}
    index[s] = {int(r[0]): r[4] for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50, kind='index')}

out = {'tracking': {}, 'fade': {}}
print('--- 1. Demo vs live pricing, per name ---')
for s in DEMO11:
    common = sorted(set(demo[s]) & set(live[s]))
    gaps = [abs(demo[s][t][1] / live[s][t][1] - 1) * 1e4 for t in common]
    wk = [abs(demo[s][t][1] / live[s][t][1] - 1) * 1e4 for t in common
          if dt.datetime.utcfromtimestamp(t / 1000).weekday() >= 5]
    mi = sorted(set(mark[s]) & set(index[s]))
    dev = [abs(mark[s][t] / index[s][t] - 1) for t in mi if index[s][t] > 0]
    first = dt.datetime.utcfromtimestamp(min(demo[s]) / 1000).date() if demo[s] else None
    row = dict(demo_hours=len(demo[s]), first=str(first), common=len(common),
               close_gap_median_bps=round(st.median(gaps), 2) if gaps else None,
               close_gap_p99_bps=round(sorted(gaps)[int(0.99 * (len(gaps) - 1))], 1) if gaps else None,
               close_gap_max_bps=round(max(gaps), 1) if gaps else None,
               weekend_gap_median_bps=round(st.median(wk), 2) if wk else None,
               mark_index_hours=len(dev), mark_index_over_3pct=sum(d > 0.03 for d in dev),
               mark_index_max_pct=round(max(dev) * 100, 2) if dev else None)
    out['tracking'][s] = row
    print(s.ljust(10), row)

def trades_for(bars, syms, exit_h, thr):
    tr = []
    for s in syms:
        b = bars[s]
        for t, (o, c) in b.items():
            d = dt.datetime.utcfromtimestamp(t / 1000)
            if d.weekday() == 4 and d.hour == 23:
                tb, te, tx = t + 12 * H, t + 13 * H, t + exit_h * H
                if tb in b and te in b and tx in b:
                    sig = b[tb][1] / c - 1
                    tr.append(dict(wk=str((d + dt.timedelta(hours=1)).date()), s=s, sig=sig,
                                   gross=b[tx][0] / b[te][0] - 1))
    return tr

def summarize(rows, cost, thr, label):
    g = [r for r in rows if r['sig'] < thr]
    if not g:
        print(label, 'no trades'); return {'label': label, 'trades': 0}
    by = {}
    for r in g: by.setdefault(r['wk'], []).append(r['gross'] - cost)
    m = [st.mean(v) for v in by.values()]
    allw = sorted({r['wk'] for r in rows})
    per = [r['gross'] - cost for r in g]
    t = st.mean(m) / (st.stdev(m) / math.sqrt(len(m))) if len(m) > 2 and st.stdev(m) > 0 else None
    # placebo: same weekends, same count, random names from the same universe
    byw = {}
    for r in rows: byw.setdefault(r['wk'], []).append(r['gross'] - cost)
    cnt = {}
    for r in g: cnt[r['wk']] = cnt.get(r['wk'], 0) + 1
    obs = st.mean(per); ge = 0; random.seed(11)
    for _ in range(2000):
        smp = []
        for w, k in cnt.items(): smp += random.sample(byw[w], min(k, len(byw[w])))
        if st.mean(smp) >= obs: ge += 1
    top3 = sorted(by.values(), key=len, reverse=True)[:3]
    res = dict(label=label, weekends_total=len(allw), weekends_triggered=len(by), trades=len(per),
               trade_mean_bps=round(obs * 1e4, 1), trade_win=round(sum(x > 0 for x in per) / len(per), 2),
               wk_mean_bps=round(st.mean(m) * 1e4, 1), wk_win=round(sum(x > 0 for x in m) / len(m), 2),
               wk_t=round(t, 2) if t else None, worst_trade_bps=round(min(per) * 1e4, 1),
               worst_wk_bps=round(min(m) * 1e4, 1), placebo_p=round(ge / 2000, 4),
               top3_weekend_trade_share=round(sum(len(v) for v in top3) / len(per), 2))
    print(res)
    return res

print('--- 2/3. Fade, threshold -1%, cost = 12bps taker + 4bps spread ---')
COST = 0.0016
for bars_name, bars, syms in [('LIVE all43', live, ALL), ('LIVE demo11', live, DEMO11), ('DEMO demo11', demo, DEMO11)]:
    for exit_h, ex in [(46, 'Sun21'), (37, 'Sun12'), (36, 'Sun11')]:
        rows = trades_for(bars, syms, exit_h, -0.01)
        out['fade'][f'{bars_name} {ex}'] = summarize(rows, COST, -0.01, f'{bars_name} exit {ex}')
json.dump(out, open(os.path.join(HERE, 'fade_demo.json'), 'w'), indent=1)
