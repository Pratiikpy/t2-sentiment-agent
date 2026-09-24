"""UTA Demo venue integrity for candidate crypto instruments: mark vs index flash prints, and Demo vs live
last-price tracking. Keyless GETs (paptrading: 1). Last ~90 days of 1H candles."""
import sys, os, json, statistics as st, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import candles_v3
out = {}
for sym, cat in [('BTCUSDT','USDT-FUTURES'),('ETHUSDT','USDT-FUTURES'),('SOLUSDT','USDT-FUTURES'),('XRPUSDT','USDT-FUTURES'),
                 ('DOGEUSDT','USDT-FUTURES'),('BTCPERP','USDC-FUTURES'),('ETHPERP','USDC-FUTURES'),('SP500USDT','USDT-FUTURES'),('NDX100USDT','USDT-FUTURES')]:
    mk = {int(r[0]): (r[2], r[3], r[4]) for r in candles_v3(sym, cat, demo=True, n_pages=22, kind='mark')}
    ix = {int(r[0]): r[4] for r in candles_v3(sym, cat, demo=True, n_pages=22, kind='index')}
    dl = {int(r[0]): r[4] for r in candles_v3(sym, cat, demo=True, n_pages=22)}
    lv = {int(r[0]): r[4] for r in candles_v3(sym, cat, demo=False, n_pages=22)}
    ks = sorted(set(mk) & set(ix))
    dev = [(k, max(abs(mk[k][0] / ix[k] - 1), abs(mk[k][1] / ix[k] - 1))) for k in ks if ix[k] > 0]
    last30 = max(ks) - 30 * 86400000 if ks else 0
    bad = [(dt.datetime.utcfromtimestamp(k/1000).strftime('%m-%d %H'), round(d*100,1)) for k, d in dev if d > 0.03]
    cl = sorted(set(dl) & set(lv))
    gap = [abs(dl[k] / lv[k] - 1) * 1e4 for k in cl if lv[k] > 0]
    out[sym] = dict(hours=len(ks), first=dt.datetime.utcfromtimestamp(min(ks)/1000).strftime('%m-%d') if ks else None,
                    mark_hilo_vs_index_over3pct=len(bad), over3pct_last30d=sum(1 for k, d in dev if d > 0.03 and k >= last30),
                    worst=bad[:6], demo_vs_live_close_gap_median_bps=round(st.median(gap), 1) if gap else None,
                    gap_p99_bps=round(sorted(gap)[int(.99*(len(gap)-1))], 1) if gap else None)
    print(sym.ljust(11), out[sym])
json.dump(out, open('demo_integrity.json', 'w'), indent=1)
