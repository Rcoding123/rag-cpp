#!/usr/bin/env python3
"""Dense 100 ms RTH grid per trade date: mid (ticks), spread (ticks), valid mask, joined from bars + windowed features.

Used by the Phase 5 power analysis and signal tests. Mid/spread come from the bar touch (forward filled to the grid, only
within one instrument); `valid` is the windowed-features validity (exchange trading, both sides, not crossed, warm)
AND the bucket lies inside RTH (sessions.RTH_OPEN..RTH_CLOSE, America/Chicago; early-close days end at the last Trading
bucket because `trading` goes false)."""
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd

import sessions

TICK = 0.25
FEAT = "/srv/quant/data/features"


def rth_window_ns(day):
    o = dt.datetime.combine(day, sessions.RTH_OPEN, tzinfo=sessions.CHI)
    c = dt.datetime.combine(day, sessions.RTH_CLOSE, tzinfo=sessions.CHI)
    return (int(o.timestamp()) * 10**9, int(c.timestamp()) * 10**9)


def trade_days():
    """Trade dates that have both windowed features and a bars file for their RTH day (the last trade date, 2026-11-17, only
    has its evening session in the data: no RTH, no bars file)."""
    return sorted(os.path.basename(p)[:8] for p in glob.glob(f"{FEAT}/windowed/*.parquet")
                  if os.path.exists(f"{FEAT}/bars/{os.path.basename(p)}"))


def load_day(ymd, with_features=False):
    """Return DataFrame indexed 0..n-1 on the 100 ms grid for RTH: ts_end_ns, instrument_id, mid, spread, valid[, raw*/x*]."""
    day = dt.datetime.strptime(ymd, "%Y%m%d").date()
    lo, hi = rth_window_ns(day)
    w = pd.read_parquet(f"{FEAT}/windowed/{ymd}.parquet")
    w = w[(w.ts_end_ns > lo) & (w.ts_end_ns <= hi)].reset_index(drop=True)
    # the bars file is per UTC day; RTH of a trade date never crosses UTC midnight (13:30-21:00 UTC)
    b = pd.read_parquet(f"{FEAT}/bars/{ymd}.parquet", columns=["ts_end_ns", "instrument_id", "bid_px", "ask_px"])
    b = b[(b.bid_px > 0) & (b.ask_px > 0)]
    b = b.sort_values("ts_end_ns")
    m = pd.merge_asof(w[["ts_end_ns", "instrument_id"]], b, on="ts_end_ns", by="instrument_id", direction="backward")
    out = w[["ts_end_ns", "instrument_id"]].copy()
    out["mid"] = (m.bid_px.values + m.ask_px.values) / 2 / TICK  # in ticks
    out["spread"] = (m.ask_px.values - m.bid_px.values) / TICK
    out["valid"] = w["valid"].values.astype(bool) & np.isfinite(out["mid"].values)
    if with_features:
        for c in [c for c in w.columns if c.startswith("x") or c.startswith("raw")]:
            out[c] = w[c].values
    return out
