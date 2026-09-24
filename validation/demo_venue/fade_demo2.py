import sys, os, json, datetime as dt, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import candles_v3
H = 3600 * 1000
WB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wb')
DEMO11 = ['HOODUSDT','COINUSDT','MSTRUSDT','METAUSDT','AMZNUSDT','GOOGLUSDT','AAPLUSDT','CRCLUSDT','NVDAUSDT','TSLAUSDT','SNDKUSDT']
for s in DEMO11:
    live = {int(x[0]): (float(x[1]), float(x[4])) for x in json.load(open(os.path.join(WB, s + '.json')))}
    d = {int(r[0]): (r[1], r[4], r[5]) for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50)}
    m = {int(r[0]): r[4] for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50, kind='mark')}
    fri = sorted(t for t in d if dt.datetime.utcfromtimestamp(t/1000).weekday()==4 and dt.datetime.utcfromtimestamp(t/1000).hour==23)
    line=[]
    for t in fri:
        tb, te, tx = t+12*H, t+13*H, t+37*H
        def f(b, k, i): return b[k][i] if k in b else None
        ds = (f(d,tb,1)/f(d,t,1)-1) if tb in d else None
        ms = (m[tb]/m[t]-1) if tb in m and t in m else None
        ls = (live[tb][1]/live[t][1]-1) if tb in live and t in live else None
        wkvol = sum(d[k][2] for k in d if t < k <= t+48*H)
        line.append(f"{dt.datetime.utcfromtimestamp(t/1000).date()} demo_last={ds and round(ds*1e4)} mark={ms and round(ms*1e4)} live={ls and round(ls*1e4)} demo_wkend_vol={round(wkvol)}")
    print(s, '|', ' ; '.join(line))
