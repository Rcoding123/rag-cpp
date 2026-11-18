#!/usr/bin/env python3
"""Per-session price levels from the trades table (data/features/trades) and official settlements.

One row per (trade_date, instrument_id). Conventions (see sessions.py): trade date = 17:00 CT -> 16:00 CT,
ETH = the whole trading session, RTH = 08:30-15:00 CT (convention, unverified vs CME), overnight (ON) = ETH
before RTH open, opening range = first ORB_MINUTES of RTH (convention: 30).
Per scope (eth_, rth_): open, high, low, close, vwap, volume, poc, vah, val. Value area = 70% of volume built
around the point of control at tick resolution (standard expansion: add the larger of the next two levels
above/below); POC ties go to the price closest to the VWAP.
Completeness: `complete` is False if the data does not cover the session (first trade later than 5 min after the
open or last trade earlier than 5 min before the close; scheduled close is 12:00 CT on early-close dates).
Prior-session columns (prior_*): the previous trade date of the SAME contract; at a roll, the other contract's last
session, with prior_basis_adj = settle(new, prev date) - settle(old, prev date) from the official settlements,
so old-contract levels + adj are comparable to the new contract. prior_settle is always the new contract's own
official settlement (as-of, never adjusted).
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S

TICK = 0.25
ORB_MINUTES = 30
VALUE_AREA = 0.70


def value_area(prices, vols, vwap, frac=VALUE_AREA):
    """prices ascending unique tick prices, vols aligned. Returns (poc, vah, val)."""
    prices = np.asarray(prices, float); vols = np.asarray(vols, float)
    if vols.sum() <= 0:
        return np.nan, np.nan, np.nan
    mx = vols.max()
    cand = np.nonzero(vols == mx)[0]
    poc_i = int(cand[np.argmin(np.abs(prices[cand] - vwap))])
    lo = hi = poc_i
    total, got = vols.sum(), vols[poc_i]
    while got < frac * total and (lo > 0 or hi < len(vols) - 1):
        up = vols[hi + 1] + (vols[hi + 2] if hi + 2 < len(vols) else 0) if hi < len(vols) - 1 else -1
        dn = vols[lo - 1] + (vols[lo - 2] if lo - 2 >= 0 else 0) if lo > 0 else -1
        if up >= dn and hi < len(vols) - 1:  # ties expand upward
            take = 2 if hi + 2 < len(vols) else 1
            for _ in range(take):
                hi += 1; got += vols[hi]
        else:
            take = 2 if lo - 2 >= 0 else 1
            for _ in range(take):
                lo -= 1; got += vols[lo]
    return prices[poc_i], prices[hi], prices[lo]


def scope_stats(df):
    if df.empty or df["size"].sum() == 0:
        return dict(open=np.nan, high=np.nan, low=np.nan, close=np.nan, vwap=np.nan, volume=0, poc=np.nan, vah=np.nan, val=np.nan)
    v = df["size"].to_numpy(dtype=float); p = df["price"].to_numpy()
    vwap = float((p * v).sum() / v.sum())
    ticks = np.round(p / TICK).astype(np.int64)
    prof = pd.Series(v).groupby(ticks).sum().sort_index()
    poc, vah, val = value_area(prof.index.to_numpy() * TICK, prof.to_numpy(), vwap)
    return dict(open=float(p[0]), high=float(p.max()), low=float(p.min()), close=float(p[-1]), vwap=vwap,
                volume=int(v.sum()), poc=float(poc), vah=float(vah), val=float(val))


def session_rows(trades, early_close=frozenset()):
    """early_close: set of datetime.date with a 12:00 CT scheduled close. trades: DataFrame sorted by ts_recv with columns ts_recv, price, size, instrument_id, trade_date, rth, halt."""
    rows = []
    t = trades[~trades.halt]
    for (td, iid), g in t.groupby(["trade_date", "instrument_id"], sort=True):
        r = {"trade_date": td, "instrument_id": int(iid), "n_trades": len(g)}
        eth, rth = g, g[g.rth]
        for pre, d in (("eth_", eth), ("rth_", rth)):
            for k, v in scope_stats(d).items():
                r[pre + k] = v
        on = g[~g.rth & (pd.to_datetime(g.ts_recv, unit="ns", utc=True).dt.tz_convert(S.CHI).dt.hour * 60
                         + pd.to_datetime(g.ts_recv, unit="ns", utc=True).dt.tz_convert(S.CHI).dt.minute < S.RTH_OPEN.hour * 60 + S.RTH_OPEN.minute)]
        r["on_high"] = float(on.price.max()) if len(on) else np.nan
        r["on_low"] = float(on.price.min()) if len(on) else np.nan
        loc = pd.to_datetime(rth.ts_recv, unit="ns", utc=True).dt.tz_convert(S.CHI)
        mins = loc.dt.hour * 60 + loc.dt.minute - (S.RTH_OPEN.hour * 60 + S.RTH_OPEN.minute)
        orb = rth[(mins >= 0) & (mins < ORB_MINUTES)]
        r["orb_high"] = float(orb.price.max()) if len(orb) else np.nan
        r["orb_low"] = float(orb.price.min()) if len(orb) else np.nan
        first = pd.to_datetime(g.ts_recv.iloc[0], unit="ns", utc=True).tz_convert(S.CHI)
        last = pd.to_datetime(g.ts_recv.iloc[-1], unit="ns", utc=True).tz_convert(S.CHI)
        open_dt = dt.datetime.combine(pd.Timestamp(td).date() - dt.timedelta(days=1), S.SESSION_OPEN, tzinfo=S.CHI)
        # Sunday-evening sessions belong to Monday: the previous calendar day may be Sunday (correct), a Friday evening never opens
        close_t = dt.time(12, 0) if pd.Timestamp(td).date() in early_close else S.SESSION_CLOSE
        close_dt = dt.datetime.combine(pd.Timestamp(td).date(), close_t, tzinfo=S.CHI)
        r["first_trade_ct"] = first.strftime("%Y-%m-%d %H:%M:%S"); r["last_trade_ct"] = last.strftime("%Y-%m-%d %H:%M:%S")
        r["complete"] = bool((first - open_dt) <= pd.Timedelta(minutes=5) and (close_dt - last) <= pd.Timedelta(minutes=5))
        rows.append(r)
    return pd.DataFrame(rows)


def add_prior(levels, settle):
    """Attach prior-session levels (see module docstring)."""
    L = levels.sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
    cols = [c for c in L.columns if c.startswith(("eth_", "rth_")) and not c.endswith("volume")] + ["on_high", "on_low", "orb_high", "orb_low"]
    out = []
    for i, r in L.iterrows():
        same = L[(L.instrument_id == r.instrument_id) & (L.trade_date < r.trade_date)]
        prev = same.iloc[-1] if len(same) else None
        other = L[(L.instrument_id != r.instrument_id) & (L.trade_date < r.trade_date)]
        adj = 0.0
        if prev is None and len(other):
            prev = other.sort_values(["trade_date", "n_trades"]).iloc[-1]
        d = {"prior_trade_date": pd.NaT, "prior_instrument_id": -1, "prior_complete": False, "prior_basis_adj": np.nan}
        for c in cols:
            d["prior_" + c] = np.nan
        if prev is not None:
            d["prior_trade_date"] = prev.trade_date; d["prior_instrument_id"] = int(prev.instrument_id); d["prior_complete"] = bool(prev.complete)
            for c in cols:
                d["prior_" + c] = prev[c]
            if prev.instrument_id != r.instrument_id:
                a = settle[(settle.instrument_id == r.instrument_id) & (settle.trade_date == prev.trade_date) & (~settle.intraday)].sort_values("ts_recv")
                b = settle[(settle.instrument_id == prev.instrument_id) & (settle.trade_date == prev.trade_date) & (~settle.intraday)].sort_values("ts_recv")
                d["prior_basis_adj"] = float(a.price.iloc[-1] - b.price.iloc[-1]) if len(a) and len(b) else np.nan
            else:
                d["prior_basis_adj"] = 0.0
        # the contract's own official settlement of the previous trade date, known at the session open
        sd = settle[(settle.instrument_id == r.instrument_id) & (settle.trade_date < r.trade_date) & (~settle.intraday)]
        if len(sd):
            last = sd[sd.trade_date == sd.trade_date.max()].sort_values("ts_recv").iloc[-1]
            d["prior_settle"] = float(last.price); d["prior_settle_trade_date"] = last.trade_date; d["prior_settle_final"] = bool(last.final)
        out.append(d)
    return pd.concat([L, pd.DataFrame(out)], axis=1)


if __name__ == "__main__":
    import glob
    trades = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/srv/quant/data/features/trades/*.parquet"))]).sort_values("ts_recv", kind="stable")
    st = pd.read_parquet("/srv/quant/data/features/levels/settlement.parquet")
    import databento as db
    stat = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-status/*/*.status.dbn.zst"))])
    ec = set()
    for i in set(stat["instrument_id"].tolist()):
        ec |= S.early_close_dates(stat[stat["instrument_id"] == i])
    # normalise: trade dates as datetime64[ns] everywhere, early closes as python dates
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    ec = {pd.Timestamp(d).date() for d in ec}
    lv = add_prior(session_rows(trades, ec), st)
    lv.to_parquet(sys.argv[1], compression="zstd")
    print(len(lv), "session rows ->", sys.argv[1])
