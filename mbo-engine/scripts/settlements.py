#!/usr/bin/env python3
"""Official daily settlement prices of ES outright futures from the statistics feed (stat_type 3).
Nothing is computed: values are the exchange's. Table columns:
  instrument_id, symbol, trade_date (from ts_ref, a DATE label, never localised), price (points),
  final (stat_flags bit0), actual (bit1), intraday (bit3), ts_recv (publication time, UTC ns).
asof(table, instrument_id, ts_ns, mode) -> the settlement a live consumer could know at ts_ns:
  the record with the latest trade_date strictly before trade_date(ts_ns) (prior-session level), and among
  that trade date's records the latest published one (final replaces preliminary once published).
Usage: settlements.py OUT.parquet   (also prints a sanity check vs trades if trades parquet exists)
"""
import glob
import sys

import databento as db
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S


def build():
    stat = np.concatenate([db.DBNStore.from_file(f).to_ndarray()
                           for f in sorted(glob.glob("/srv/quant/data/raw/es-fut-statistics/*/*.statistics.dbn.zst"))])
    dfn = np.concatenate([db.DBNStore.from_file(f).to_ndarray()
                          for f in sorted(glob.glob("/srv/quant/data/raw/es-fut-definition/*/*.definition.dbn.zst"))])
    sym = {}
    for r in dfn:
        s = r["raw_symbol"].decode()
        if len(s) == 4 and s.startswith("ES"):  # outrights only (no spreads)
            sym[int(r["instrument_id"])] = s
    m = (stat["stat_type"] == 3) & np.isin(stat["instrument_id"], list(sym)) & (stat["update_action"] == 1)
    s = stat[m]
    df = pd.DataFrame({
        "instrument_id": s["instrument_id"].astype(np.int64),
        "trade_date": (s["ts_ref"].astype(np.int64) // (86400 * 10**9)).astype("datetime64[D]"),
        "price": s["price"] / 1e9,
        "final": (s["stat_flags"] & 1) != 0, "actual": (s["stat_flags"] & 2) != 0, "intraday": (s["stat_flags"] & 8) != 0,
        "ts_recv": s["ts_recv"].astype(np.int64),
    })
    df["symbol"] = df.instrument_id.map(sym)
    return df.sort_values(["instrument_id", "ts_recv"]).reset_index(drop=True)


def asof(df, instrument_id, ts_ns):
    """Prior-session settlement known at ts_ns for instrument_id; None if none published yet."""
    td = S.trade_date([ts_ns])[0]
    d = df[(df.instrument_id == instrument_id) & (df.ts_recv <= ts_ns) & (df.trade_date < td) & (~df.intraday)]
    if d.empty:
        return None
    last_day = d.trade_date.max()
    return d[d.trade_date == last_day].sort_values("ts_recv").iloc[-1]


if __name__ == "__main__":
    df = build()
    df.to_parquet(sys.argv[1], compression="zstd")
    print(len(df), "settlement records,", df.symbol.nunique(), "outright instruments ->", sys.argv[1])
