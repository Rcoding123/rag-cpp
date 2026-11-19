#!/usr/bin/env python3
"""Convert bar_extract output (.bars) to Parquet.

Layout must match engine/include/mbo/bars.hpp BarRow (120 bytes, packed).
Prices are written in index points (1e-9 fixed-point / 1e9); 0 stays 0 (= empty).
Usage: bars_to_parquet.py IN.bars OUT.parquet
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DT = np.dtype([
    ("ts_end_ns", "<u8"), ("instrument_id", "<u4"), ("flags", "u1"), ("pad0", "u1", 3),
    ("bid_px", "<i8"), ("ask_px", "<i8"),
    ("bid_sz", "<u4", 5), ("ask_sz", "<u4", 5),
    ("buy_vol", "<u4"), ("sell_vol", "<u4"), ("n_trades", "<u4"),
    ("sold_at_bid", "<u4"), ("bought_at_ask", "<u4"),
    ("last_trade_px", "<i8"),
    ("n_add", "<u4"), ("n_cancel", "<u4"), ("n_modify", "<u4"),
    ("ofi", "<f4"), ("absorb", "<f4"),
])
assert DT.itemsize == 120, DT.itemsize


def read_bars(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        if hdr[:8] != b"MBOBARS1":
            raise ValueError("not a .bars file")
        row_bytes = int.from_bytes(hdr[8:12], "little")
        bucket_ms = int.from_bytes(hdr[12:16], "little")
        if row_bytes != DT.itemsize:
            raise ValueError(f"row size {row_bytes} != {DT.itemsize}")
        a = np.fromfile(f, dtype=DT)
    return a, bucket_ms


def to_table(a, bucket_ms):
    cols = {
        "ts_end_ns": a["ts_end_ns"].astype(np.int64),
        "instrument_id": a["instrument_id"], "flags": a["flags"],
        "bid_px": a["bid_px"] / 1e9, "ask_px": a["ask_px"] / 1e9,
        "last_trade_px": a["last_trade_px"] / 1e9,
    }
    for i in range(5):
        cols[f"bid_sz{i + 1}"] = a["bid_sz"][:, i]
        cols[f"ask_sz{i + 1}"] = a["ask_sz"][:, i]
    for k in ("buy_vol", "sell_vol", "n_trades", "sold_at_bid", "bought_at_ask",
              "n_add", "n_cancel", "n_modify", "ofi", "absorb"):
        cols[k] = a[k]
    t = pa.table(cols)
    return t.replace_schema_metadata({"bucket_ms": str(bucket_ms), "source": "mbo-engine bar_extract"})


if __name__ == "__main__":
    a, ms = read_bars(sys.argv[1])
    pq.write_table(to_table(a, ms), sys.argv[2], compression="zstd")
    print(f"{len(a)} rows, bucket {ms} ms -> {sys.argv[2]}")
