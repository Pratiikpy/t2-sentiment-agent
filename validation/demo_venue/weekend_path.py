import sys, os, json, datetime as dt, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import candles_v3
H=3600*1000
WB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wb')
DEMO11 = ['HOODUSDT','COINUSDT','MSTRUSDT','METAUSDT','AMZNUSDT','GOOGLUSDT','AAPLUSDT','CRCLUSDT','NVDAUSDT','TSLAUSDT','SNDKUSDT']
res={}
for s in DEMO11:
    live = {int(x[0]): float(x[4]) for x in json.load(open(os.path.join(WB, s + '.json')))}
    d = {int(r[0]): r[4] for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50)}
    ix = {int(r[0]): r[4] for r in candles_v3(s, 'USDT-FUTURES', demo=True, n_pages=50, kind='index')}
    # hourly abs log-return volatility by session: weekend vs weekday, demo vs live vs demo index
    def vol(b, wkend):
        ks=sorted(b); r=[]
        for a,c in zip(ks,ks[1:]):
            if c-a!=H: continue
            w=dt.datetime.utcfromtimestamp(c/1000).weekday()>=5
            if w==wkend and b[a]>0: r.append(abs(b[c]/b[a]-1)*1e4)
        return round(st.mean(r),1) if r else None
    common=[k for k in d if k in live]
    lo=min(d); live_c={k:v for k,v in live.items() if k>=lo}
    res[s]=dict(weekend_absret_bps=dict(demo_last=vol(d,True),demo_index=vol(ix,True),live=vol(live_c,True)),
                weekday_absret_bps=dict(demo_last=vol(d,False),demo_index=vol(ix,False),live=vol(live_c,False)))
    print(s.ljust(10),res[s])
# live index for comparison (v3 index candles, live)
for s in ['MSTRUSDT','NVDAUSDT']:
    lix={int(r[0]): r[4] for r in candles_v3(s,'USDT-FUTURES',demo=False,n_pages=8,kind='index')}
    ks=sorted(lix); r=[abs(lix[c]/lix[a]-1)*1e4 for a,c in zip(ks,ks[1:]) if c-a==H and dt.datetime.utcfromtimestamp(c/1000).weekday()>=5]
    print(s,'LIVE index weekend mean |1h ret| bps',round(st.mean(r),1) if r else None, 'n',len(r))
json.dump(res,open('weekend_vol.json','w'),indent=1)
