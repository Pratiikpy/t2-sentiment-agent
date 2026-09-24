"""Risk envelope of the recommended Track 2 book over the ~3-day paper window, under the null of no edge.

Measured on live Bitget 1H candles (keyless), for instruments UTA Demo lists and that track live on Demo:
crypto majors 24/7, and the 11 Demo-listed US-equity perps Mon 00:00 -> Fri 20:00 UTC only (Demo freezes
them on weekends: see weekend_vol.json). No LLM, no orders. The 'LLM' here is a coin flip: this is the
honest expectation for an agent whose edge has not been demonstrated.
"""
import sys, os, json, math, random, statistics as st, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import candles_v2
H = 3600 * 1000
WB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wb')
CRYPTO = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
STOCKS = ['HOODUSDT','COINUSDT','MSTRUSDT','METAUSDT','AMZNUSDT','GOOGLUSDT','AAPLUSDT','CRCLUSDT','NVDAUSDT','TSLAUSDT','SNDKUSDT']
px = {}
for s in CRYPTO:
    px[s] = {int(r[0]): r[4] for r in candles_v2(s, 'USDT-FUTURES', '1H', n_pages=12)}
for s in STOCKS:
    px[s] = {int(x[0]): float(x[4]) for x in json.load(open(os.path.join(WB, s + '.json')))}
end = min(max(v) for v in px.values())
start = end - 90 * 24 * H
hours = list(range(start - start % H, end, H))

def tradable(s, t):
    if s in CRYPTO: return True
    d = dt.datetime.utcfromtimestamp(t / 1000)
    return d.weekday() < 4 or (d.weekday() == 4 and d.hour < 20)

SIDE_COST = 0.0008  # 6bps taker + 2bps half-spread per side

def run(t0, rng, per_name=0.05, k=5, trade_prob=1.0, stop=0.04, kill=0.015):
    eq = 1.0; peak = 1.0; mdd = 0.0; pos = {}; closed = []; marks = [1.0]
    day_start_eq = 1.0
    for i in range(72):
        t = t0 + i * H
        if i % 24 == 0:  # daily decision
            day_start_eq = eq
            for s, p in list(pos.items()):  # close all, pay exit
                eq -= p['w'] * SIDE_COST; closed.append(p['pnl'] - 2 * SIDE_COST * p['w']); del pos[s]
            if rng.random() < trade_prob:
                cands = [s for s in px if tradable(s, t) and t in px[s] and t + H in px[s]]
                for s in rng.sample(cands, min(k, len(cands))):
                    pos[s] = dict(side=rng.choice([1, -1]), w=per_name, entry=px[s][t], pnl=0.0)
                    eq -= per_name * SIDE_COST
        # mark hour t -> t+H
        for s, p in list(pos.items()):
            a, b = px[s].get(t), px[s].get(t + H)
            if a is None or b is None: continue
            if not tradable(s, t + H):  # forced flat before the Demo weekend freeze
                eq -= p['w'] * SIDE_COST; closed.append(p['pnl'] - 2 * SIDE_COST * p['w']); del pos[s]; continue
            r = p['side'] * (b / a - 1) * p['w']
            eq += r; p['pnl'] += r
            if p['side'] * (b / p['entry'] - 1) <= -stop:
                eq -= p['w'] * SIDE_COST; closed.append(p['pnl'] - 2 * SIDE_COST * p['w']); del pos[s]
        if eq / day_start_eq - 1 <= -kill:
            for s, p in list(pos.items()):
                eq -= p['w'] * SIDE_COST; closed.append(p['pnl'] - 2 * SIDE_COST * p['w']); del pos[s]
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1); marks.append(eq)
    for s, p in list(pos.items()):
        eq -= p['w'] * SIDE_COST; closed.append(p['pnl'] - 2 * SIDE_COST * p['w'])
    rets = [b / a - 1 for a, b in zip(marks, marks[1:])]
    sd = st.pstdev(rets)
    sh = st.mean(rets) / sd * math.sqrt(8760) if sd > 0 else float('nan')
    return eq - 1, mdd, sh, closed

def pct(v, q): v = sorted(v); return v[int(q * (len(v) - 1))]
res = {}
for label, kw in [('gross25_daily', {}), ('gross10_daily', dict(per_name=0.02)),
                  ('gross25_half_days', dict(trade_prob=0.5))]:
    R, M, S, W, N = [], [], [], [], []
    starts = [h for h in hours if dt.datetime.utcfromtimestamp(h / 1000).hour == 13 and h + 73 * H <= end]
    for seed in range(200):
        rng = random.Random(seed)
        for t0 in starts:
            r, m, s, c = run(t0, rng, **kw)
            R.append(r); M.append(m)
            if not math.isnan(s): S.append(s)
            if c: W.append(sum(x > 0 for x in c) / len(c))
            N.append(len(c))
    res[label] = dict(windows=len(R), ret_bps=dict(p05=round(pct(R, .05) * 1e4, 1), median=round(st.median(R) * 1e4, 1), mean=round(st.mean(R) * 1e4, 1), p95=round(pct(R, .95) * 1e4, 1)),
                      mdd_pct=dict(median=round(st.median(M) * 100, 2), p95_worst=round(pct(M, .05) * 100, 2), worst=round(min(M) * 100, 2)),
                      sharpe_ann=dict(p05=round(pct(S, .05), 1), median=round(st.median(S), 1), p95=round(pct(S, .95), 1)),
                      win_rate=dict(p05=round(pct(W, .05), 2), median=round(st.median(W), 2), p95=round(pct(W, .95), 2)),
                      closed_trades_median=st.median(N), share_positive=round(sum(x > 0 for x in R) / len(R), 2))
    print(label, json.dumps(res[label]))
json.dump(res, open('envelope.json', 'w'), indent=1)
