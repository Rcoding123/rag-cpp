#!/usr/bin/env python3
"""Add the purchased ESZ6 roll-window trades (es-trades-esz6-roll) to data/features/trades.

ES.v.0 switched from ESU6 to ESZ6 at 00:00 UTC on 2026-09-16, but the 2026-09-16 CME session opened at 22:00 UTC the
evening before, so ESZ6's first two hours were missing from our MBO files. The purchased `trades` window starts at
22:00 UTC and overlaps our MBO trades after 00:00 UTC; the overlap was checked identical (1,441/1,441 trades:
ts_recv, price, size, side, sequence), so only trades BEFORE the first MBO record of that instrument are kept.
Output uses the same columns as trades_to_parquet.py. These trades have no book/bars (no MBO for the window).
"""
import glob
import sys

import databento as db
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S
from trades_to_parquet import early_closes

src = glob.glob("/srv/quant/data/raw/es-trades-esz6-roll/*/*.trades.dbn.zst")[0]
t = db.DBNStore.from_file(src).to_ndarray()
mbo = db.DBNStore.from_file(glob.glob("/srv/quant/data/raw/es-mbo/*/glbx-mdp3-20260916.mbo.dbn.zst")[0]).to_ndarray()
first_mbo = int(mbo["ts_recv"][0])
inst = np.unique(t["instrument_id"])
assert len(inst) == 1 and int(inst[0]) == int(mbo["instrument_id"][0]), "instrument mismatch with the MBO file"
a = t[t["ts_recv"] < first_mbo]
assert len(a) and (t["action"] == b"T").all()
ts = a["ts_recv"].astype(np.int64)
tab = pa.table({
    "ts_recv": ts, "price": a["price"] / 1e9, "size": a["size"],
    "side": a["side"].view("S1").astype(str), "instrument_id": a["instrument_id"],
    "trade_date": S.trade_date(ts).astype("datetime64[D]"),
    "rth": S.is_rth(ts, early_closes()), "halt": S.in_halt(ts),
})
out = "/srv/quant/data/features/trades/20260915_esz6_roll.parquet"
pq.write_table(tab, out, compression="zstd")
print(f"{len(a)} trades (of {len(t)} purchased; {len(t) - len(a)} overlap dropped) -> {out}")
