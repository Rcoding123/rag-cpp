"""Per-day check of data/features/trades/*.parquet against the Python databento reader (T records):
count, total size, buy/sell/none size, sum(price*size) in fixed point, and first/last ts_recv."""
import glob
import re
import sys

import databento as db
import numpy as np
import pyarrow.parquet as pq

bad = 0
tot = 0
files = sorted(glob.glob("/srv/quant/data/raw/es-mbo/*/*.mbo.dbn.zst"))
for f in files:
    d = re.search(r"-(\d{8})\.", f).group(1)
    a = db.DBNStore.from_file(f).to_ndarray()
    t = a[a["action"] == b"T"]
    p = pq.read_table(f"/srv/quant/data/features/trades/{d}.parquet").to_pandas()
    ok = len(p) == len(t) and int(p["size"].sum()) == int(t["size"].sum())
    for sd in (b"A", b"B", b"N"):
        ok &= int(p.loc[p.side == sd.decode(), "size"].sum()) == int(t["size"][t["side"] == sd].sum())
    ok &= int(np.round(p["price"].to_numpy() * 1e9).astype(np.int64).astype(object).dot(p["size"].to_numpy().astype(object))) == \
          int(t["price"].astype(object).dot(t["size"].astype(object)))
    ok &= len(p) == 0 or (int(p.ts_recv.iloc[0]) == int(t["ts_recv"][0]) and int(p.ts_recv.iloc[-1]) == int(t["ts_recv"][-1]))
    tot += len(p)
    if not ok:
        bad += 1
        print("MISMATCH", d, len(p), len(t))
print(f"{len(files)} days, {tot:,} trades, mismatching days: {bad}")
if bad:
    sys.exit(1)

# supplement: purchased ESZ6 roll-window trades (before the first MBO record of 2026-09-16)
import glob as _g
x = db.DBNStore.from_file(_g.glob("/srv/quant/data/raw/es-trades-esz6-roll/*/*.trades.dbn.zst")[0]).to_ndarray()
first = db.DBNStore.from_file(_g.glob("/srv/quant/data/raw/es-mbo/*/glbx-mdp3-20260916.mbo.dbn.zst")[0]).to_ndarray()["ts_recv"][0]
x = x[x["ts_recv"] < first]
q = pq.read_table("/srv/quant/data/features/trades/20260915_esz6_roll.parquet").to_pandas()
okx = (len(q) == len(x) and int(q["size"].sum()) == int(x["size"].sum()) and
       np.array_equal(q["ts_recv"].to_numpy(), x["ts_recv"].astype("int64")) and
       np.array_equal(np.round(q["price"].to_numpy() * 1e9).astype(np.int64), x["price"]) and
       np.array_equal(q["side"].to_numpy().astype("S1"), x["side"]))
print(f"supplement 20260915_esz6_roll: {len(q)} trades, identical to the purchased file: {okx}")
if not okx:
    sys.exit(1)
