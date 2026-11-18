import glob, numpy as np, datetime as dt, sys, re, json
src=open('/srv/quant/code/rag-cpp/mbo-engine/scripts/crossed_vs_status.py').read().replace('\nmain()\n','\n')
ns={}; exec(compile(src,'c','exec'),ns)
st=ns['load_status'](sorted(glob.glob('/srv/quant/data/raw/es-status/*/*.status.dbn.zst')))
D=ns['DUMP']
rows={}
for l in open('/srv/quant/results/phase2_book_check2.jsonl'):
    r=json.loads(l); rows[re.search(r'-(\d{8})\.',r['file']).group(1)]=r
tot_t=nt_t=pre_cross=pre_periods=0; mkt=[]
for day,r in sorted(rows.items()):
    iid=r['per_instrument'][0]['id']; s=st[st['instrument_id']==iid]
    d0=int(dt.datetime.strptime(day,'%Y%m%d').replace(tzinfo=dt.UTC).timestamp()*1e9); d1=d0+86400*10**9
    ts=s['ts_recv'].astype(np.int64)
    edges=[d0]+[int(t) for t in ts if d0<t<d1]+[d1]
    cross=np.fromfile(f'/srv/quant/results/crossed/{day}.ts',dtype=D)['t'].astype(np.int64)
    for a,b in zip(edges[:-1],edges[1:]):
        i=np.searchsorted(ts,a,side='right')-1
        if i<0: continue
        rec=s[i]; tot_t+=b-a
        if rec['is_trading'].decode()=='N': nt_t+=b-a
        if rec['action']==1:
            pre_periods+=1; has=bool(((cross>=a)&(cross<b)).any()); pre_cross+=has
            if rec['reason']==3: mkt.append((a,b,has))
print(f"share of calendar time with is_trading=N (front instrument): {100*nt_t/tot_t:.1f}%")
print(f"PreOpen periods: {pre_periods}; containing >=1 crossed boundary: {pre_cross}")
f=lambda n:dt.datetime.fromtimestamp(n/1e9,dt.UTC).strftime('%m-%d %H:%M:%S.%f')[:-3]
print(f"PreOpen/MarketEvent periods: {len(mkt)}; with a crossing: {sum(h for *_,h in mkt)}")
for a,b,h in mkt[:12]: print("  ",f(a),"->",f(b),f"dur={(b-a)/1e9:.2f}s crossed={'yes' if h else 'no'}")
import collections
print("durations (s) of MarketEvent pre-opens: min %.2f median %.2f max %.2f"%tuple(np.percentile([(b-a)/1e9 for a,b,_ in mkt],[0,50,100])))
