#!/usr/bin/env python3
"""Windowed order-flow features (13, scaled to ~unit size) on a dense 100 ms grid, per CME session.

Port of the original rag-cpp FeatureBuilder (include/rag/orderflow/features.hpp); the arithmetic is the same and is
pinned to the ORIGINAL C++ by scripts/verify_features.py (engine/build/feat_ref). Inputs: bars (touch, 5-level depth, L1
OFI, absorption; engine bar_extract), trades (aggressor volume, price*size) and the exchange status feed.

Features (raw, then x = scaled; see features.hpp):
 0 delta 5 s, 1 delta 30 s, 2 delta 120 s     signed aggressor volume
 3 OFI 5 s, 4 OFI 30 s                        L1 order-flow imbalance (level totals)
 5 volume 30 s vs its running mean            log((v30+1)/(mean+1))
 6 imbalance L1, 7 imbalance L1-5             (bid-ask)/(bid+ask)
 8,9,10 mid return 5/30/120 s                 TICKS
 11 mid - session VWAP                        ticks; VWAP over the session's side-known trades, from session start
 12 absorption                                running, from the bars
Scaling: every feature except 5,6,7 is divided by an EWMA RMS of itself (time constant 600 s), clipped to +-4; 5 is
clipped, 6/7 are doubled then clipped.

Differences from the simulator version (all deliberate, documented):
 * the book comes from the real per-order book at F_LAST boundaries; buckets are UTC-aligned on ts_recv;
 * volume/delta/VWAP use side-known trades only (buy+sell), as the original needs an aggressor;
 * windows, VWAP and absorption restart per (trade_date, instrument) session unit; the EWMA scalers (rms, volume mean) carry
   across units (causal, so no look-ahead);
 * sessions with no `Trading` state at all (weekends) are skipped; the final session is not padded past the last data row.
 * `valid` = quote usable (exchange says is_trading, both sides present, not crossed, no snapshot in the bucket) and
   warm (>= 120 s into the unit) and the returns/VWAP finite. Features are still computed on invalid rows (NaN where
   undefined) so consumers can decide.
 * bars are built per UTC-day file, so the absorption run counter and the first OFI restart at 00:00 UTC (a minor
   discontinuity at the file boundary).
"""
import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import numba as nb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessions as S

BUCKET_NS = 100_000_000
L5, L30, L120 = 50, 300, 1200
NORM_ALPHA = min(1.0, 0.1 / 600.0)
NF = 13
BAR_CROSSED, BAR_ONE_SIDE_EMPTY, BAR_SNAPSHOT = 1, 2, 4


@nb.njit(cache=True)
def kernel(buy, sell, pv_ticks, bid_t, ask_t, bsz, asz, ofi, absorb, qok, rms, vol_mean):
    """Dense per-bucket features. rms (13,) and vol_mean are the EWMA state, updated in place. Returns raw, x (n,13)."""
    n = len(buy)
    raw = np.full((n, NF), np.nan)
    x = np.full((n, NF), np.nan)
    cdel = np.zeros(n + 1); cvol = np.zeros(n + 1); cofi = np.zeros(n + 1)
    mid = np.empty(n)
    for i in range(n):
        cdel[i + 1] = cdel[i] + (buy[i] - sell[i])
        cvol[i + 1] = cvol[i] + (buy[i] + sell[i])
        cofi[i + 1] = cofi[i] + ofi[i]
        mid[i] = 0.5 * (bid_t[i] + ask_t[i]) if qok[i] else np.nan
    vnum = 0.0; vden = 0.0
    for i in range(n):
        vnum += pv_ticks[i]; vden += buy[i] + sell[i]
        lo5 = max(0, i + 1 - L5); lo30 = max(0, i + 1 - L30); lo120 = max(0, i + 1 - L120)
        vol30 = cvol[i + 1] - cvol[lo30]
        vol_mean += (vol30 - vol_mean) * NORM_ALPHA
        raw[i, 0] = cdel[i + 1] - cdel[lo5]
        raw[i, 1] = cdel[i + 1] - cdel[lo30]
        raw[i, 2] = cdel[i + 1] - cdel[lo120]
        raw[i, 3] = cofi[i + 1] - cofi[lo5]
        raw[i, 4] = cofi[i + 1] - cofi[lo30]
        raw[i, 5] = np.log((vol30 + 1.0) / (vol_mean + 1.0))
        if qok[i]:
            b1 = bsz[i, 0]; a1 = asz[i, 0]
            raw[i, 6] = (b1 - a1) / (b1 + a1) if b1 + a1 > 0 else 0.0
            b5 = 0.0; a5 = 0.0
            for j in range(5):
                b5 += bsz[i, j]; a5 += asz[i, j]
            raw[i, 7] = (b5 - a5) / (b5 + a5) if b5 + a5 > 0 else 0.0
            raw[i, 11] = (mid[i] - vnum / vden) if vden > 0 else 0.0
        for k, L in ((8, L5), (9, L30), (10, L120)):
            if i + 1 > L:
                raw[i, k] = mid[i] - mid[i - L]   # NaN if either end has no valid quote
            else:
                raw[i, k] = 0.0
        raw[i, 12] = absorb[i]
        for k in range(NF):
            r = raw[i, k]
            if np.isnan(r):
                continue
            if k == 5:
                x[i, k] = min(4.0, max(-4.0, r))
            elif k == 6 or k == 7:
                x[i, k] = min(4.0, max(-4.0, 2.0 * r))
            else:
                rms[k] = np.sqrt(rms[k] * rms[k] + (r * r - rms[k] * rms[k]) * NORM_ALPHA)
                x[i, k] = min(4.0, max(-4.0, r / max(1e-9, rms[k])))
    return raw, x, vol_mean


def _ct_ns(date, hhmm):
    d = dt.datetime.combine(date, dt.time(*hhmm), tzinfo=S.CHI)
    return int(d.timestamp() * 1e9)


def session_window(trade_date):
    """[open_ns, close_ns) of a CME trade date: 17:00 CT the previous calendar day to 16:00 CT."""
    d = pd.Timestamp(trade_date).date()
    return _ct_ns(d - dt.timedelta(days=1), (17, 0)), _ct_ns(d, (16, 0))


def load_status():
    import databento as db
    st = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-status/*/*.status.dbn.zst"))])
    st = st[st["action"] != 0]
    out = {}
    for i in np.unique(st["instrument_id"]):
        s = st[st["instrument_id"] == i]
        s = s[np.argsort(s["ts_recv"], kind="stable")]
        out[int(i)] = (s["ts_recv"].astype(np.int64), s["is_trading"] == b"Y")
    return out


def trading_mask(status, iid, start_ns_vec):
    if iid not in status:
        return np.zeros(len(start_ns_vec), bool)
    ts, ok = status[iid]
    k = np.searchsorted(ts, start_ns_vec, side="right") - 1
    return (k >= 0) & ok[np.clip(k, 0, None)]


def dense_unit(bars, trades_ts, trades_px_t, trades_sz, trades_side_buy, iid, b0, b1, status, prior_row):
    """Dense arrays for buckets [b0, b1). bars: DataFrame of this instrument's rows (sorted), bucket-start index column 'bk'.
    prior_row: last bars row before b0 (Series) or None. trades_*: this instrument's side-known trades in the window."""
    n = b1 - b0
    rows = bars[(bars.bk >= b0) & (bars.bk < b1)]
    k0 = 1 if prior_row is not None else 0
    parts = ([prior_row.to_frame().T] if prior_row is not None else []) + [rows]
    src = pd.concat(parts) if len(parts) > 1 else rows
    pos = (rows.bk.to_numpy() - b0).astype(np.int64)
    idx = np.full(n, -1, np.int64)
    idx[pos] = np.arange(k0, k0 + len(rows))
    if prior_row is not None and idx[0] < 0:
        idx[0] = 0
    idx = np.maximum.accumulate(idx)
    have = idx >= 0
    j = np.clip(idx, 0, None)

    def col(name):
        v = src[name].to_numpy()
        return v[j] if len(v) else np.zeros(n)
    flags = col("flags").astype(np.int64) if len(src) else np.zeros(n, np.int64)
    bid = np.round(col("bid_px").astype(np.float64) * 4).astype(np.int64)
    ask = np.round(col("ask_px").astype(np.float64) * 4).astype(np.int64)
    bsz = np.stack([col(f"bid_sz{i}") for i in range(1, 6)], 1).astype(np.float64)
    asz = np.stack([col(f"ask_sz{i}") for i in range(1, 6)], 1).astype(np.float64)
    absorb = col("absorb").astype(np.float64)
    ofi = np.zeros(n)
    ofi[pos] = rows.ofi.to_numpy().astype(np.float64)
    snap = np.zeros(n, bool)
    snap[pos] = (rows["flags"].to_numpy() & BAR_SNAPSHOT) != 0
    start_ns = (b0 + np.arange(n, dtype=np.int64)) * BUCKET_NS
    trading = trading_mask(status, iid, start_ns)
    qok = have & (bid > 0) & (ask > 0) & ((flags & BAR_CROSSED) == 0) & ~snap & trading & (ask > bid)
    # trades -> per-bucket aggregates
    tb = (trades_ts // BUCKET_NS - b0)
    m = (tb >= 0) & (tb < n)
    buy = np.bincount(tb[m & trades_side_buy], weights=trades_sz[m & trades_side_buy], minlength=n)
    sell = np.bincount(tb[m & ~trades_side_buy], weights=trades_sz[m & ~trades_side_buy], minlength=n)
    pv = np.bincount(tb[m], weights=trades_px_t[m] * trades_sz[m], minlength=n)
    return dict(buy=buy, sell=sell, pv=pv, bid=bid.astype(np.float64), ask=ask.astype(np.float64), bsz=bsz, asz=asz,
                ofi=ofi, absorb=absorb, qok=qok, trading=trading, start_ns=start_ns)


def main(out_dir, only=None):
    os.makedirs(out_dir, exist_ok=True)
    bars = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/srv/quant/data/features/bars/*.parquet"))], ignore_index=True)
    bars["bk"] = bars.ts_end_ns // BUCKET_NS - 1
    bars = bars.sort_values(["instrument_id", "bk"], kind="stable").reset_index(drop=True)
    tr = pd.concat([pd.read_parquet(f, columns=["ts_recv", "price", "size", "side", "instrument_id"]) for f in sorted(glob.glob("/srv/quant/data/features/trades/*.parquet"))])
    tr = tr[tr.side.isin(["A", "B"])].sort_values("ts_recv", kind="stable")
    status = load_status()
    by_inst = {int(i): g.reset_index(drop=True) for i, g in bars.groupby("instrument_id")}
    tr_ts = tr.ts_recv.to_numpy()
    # session units: (trade_date, instrument) from the bars (halt rows excluded by the session window)
    lo_all = int(bars.ts_end_ns.min()); hi_all = int(bars.ts_end_ns.max())
    first_td = pd.Timestamp(S.trade_date([lo_all])[0]); last_td = pd.Timestamp(S.trade_date([hi_all])[0])
    rms = np.ones(NF); vol_mean = 0.0
    n_rows = 0
    for td in pd.date_range(first_td, last_td):
        if only and td.strftime("%Y%m%d") not in only:
            continue
        o_ns, c_ns = session_window(td)
        ob, cb = o_ns // BUCKET_NS, c_ns // BUCKET_NS
        units = []
        for iid, br in by_inst.items():
            bk = br.bk.to_numpy()
            lo, hi = int(np.searchsorted(bk, ob, "left")), int(np.searchsorted(bk, cb, "left"))
            if hi > lo:
                units.append((int(bk[hi - 1]), iid, lo, hi))
        units.sort()                                   # in order of last activity: the last one runs to the session close
        frames = []
        for u, (last, iid, lo, hi) in enumerate(units):
            br = by_inst[iid]
            first = int(br.bk.iloc[lo])
            prior_row = br.iloc[lo - 1] if lo > 0 else None
            b0 = ob if prior_row is not None else first
            b1 = cb if u == len(units) - 1 else last + 1
            if hi == len(br):                              # the data ends inside this session (final stub): never pad past the last row
                b1 = min(b1, last + 1)
            if b1 - b0 < 10:
                continue
            i0, i1 = int(np.searchsorted(tr_ts, b0 * BUCKET_NS, "left")), int(np.searchsorted(tr_ts, b1 * BUCKET_NS, "left"))
            tt = tr.iloc[i0:i1]
            tt = tt[tt.instrument_id == iid]
            d = dense_unit(br.iloc[lo:hi], tt.ts_recv.to_numpy(), np.round(tt.price.to_numpy() * 4), tt["size"].to_numpy().astype(np.float64),
                           (tt.side == "B").to_numpy(), iid, b0, b1, status, prior_row)
            if not d["trading"].any():                   # weekend / holiday grids: the exchange never said Trading
                continue
            raw, x, vol_mean = kernel(d["buy"], d["sell"], d["pv"], d["bid"], d["ask"], d["bsz"], d["asz"], d["ofi"], d["absorb"], d["qok"], rms, vol_mean)
            n = b1 - b0
            warm = np.arange(n) >= L120                 # the original's warm(): count > n120
            ret_ok = np.isfinite(raw[:, 8:12]).all(1)
            cols = {"ts_end_ns": d["start_ns"] + BUCKET_NS, "instrument_id": np.full(n, iid, np.uint32), "trading": d["trading"], "qok": d["qok"],
                    "warm": warm, "valid": d["qok"] & warm & ret_ok}
            for k in range(NF):
                cols[f"raw{k}"] = raw[:, k].astype(np.float32)
            for k in range(NF):
                cols[f"x{k}"] = x[:, k].astype(np.float32)
            frames.append(pd.DataFrame(cols))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(f"{out_dir}/{td.strftime('%Y%m%d')}.parquet", compression="zstd")
            n_rows += len(out)
            print(f"{td.date()} {len(out):>8,} rows  valid {100 * out.valid.mean():5.1f}%  trading {100 * out.trading.mean():5.1f}%", flush=True)
    print("total rows", n_rows)


if __name__ == "__main__":
    main(sys.argv[1], set(sys.argv[2:]) or None)
