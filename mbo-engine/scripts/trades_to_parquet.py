#!/usr/bin/env python3
"""trades_extract output (.trades) -> Parquet with session tags.
Usage: trades_to_parquet.py IN.trades OUT.parquet
Columns: ts_recv (int64 ns UTC), price (points), size, side ('A' sell aggressor, 'B' buy, 'N' none),
instrument_id, trade_date (CME trade date, see sessions.py), rth (bool), halt (bool).
RTH uses sessions.is_rth with the early-close dates found in the status data (no RTH on shortened days).
"""
import glob
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S

DT = np.dtype([("ts_recv", "<u8"), ("price", "<i8"), ("size", "<u4"), ("side", "u1"), ("flags", "u1"), ("pad", "<u2"),
               ("sequence", "<u4"), ("instrument_id", "<u4")])
assert DT.itemsize == 32


def early_closes():
    import databento as db
    st = np.concatenate([db.DBNStore.from_file(f).to_ndarray()
                         for f in sorted(glob.glob("/srv/quant/data/raw/es-status/*/*.status.dbn.zst"))])
    out = set()
    for i in set(st["instrument_id"].tolist()):
        out |= S.early_close_dates(st[st["instrument_id"] == i])
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    early = early_closes()
    with open(src, "rb") as f:
        hdr = f.read(16)
        assert hdr[:8] == b"MBOTRDS1" and int.from_bytes(hdr[8:12], "little") == DT.itemsize
        a = np.fromfile(f, dtype=DT)
    ts = a["ts_recv"]
    t = pa.table({
        "ts_recv": ts.astype(np.int64), "price": a["price"] / 1e9, "size": a["size"],
        "side": a["side"].view("S1").astype(str),
        "instrument_id": a["instrument_id"],
        "trade_date": S.trade_date(ts).astype("datetime64[D]"),
        "rth": S.is_rth(ts, early),
        "halt": S.in_halt(ts),
    })
    pq.write_table(t, dst, compression="zstd")
    print(f"{len(a)} trades -> {dst}")


if __name__ == "__main__":
    main()
