#!/usr/bin/env python3
"""scid_to_bars.py -- OF18: 1-min RTH bars of the ES front month from Sierra Chart .scid trade files.

usage:
  python tools/scid_to_bars.py --start 2025-12-12 --end 2026-11-11
         [--scid-dir C:/SierraChart/Data] [--out-dir build-avx512/real/bars] [--contracts ESH26,ESM26,ESU26,ESZ26]
         [--manifest experiments/configs/of16_real_manifest.json]

Days on or after the manifest's confirm_rule.from (2026-11-12, the real-data confirm block) are refused
(exit 2, before any data file is opened). Output per RTH day (08:30-15:00 CT, the OF16 clock):
OUT/<date>.npz with arrays of length 390 (bar 1 = 08:30-08:31 CT):
  open, high, low, close   ticks (price / 0.25)
  volume, buy, sell        lots; buy = the records' ask volume (buy aggressor), sell = bid volume
  trades                   records' trade counts
  open_bid, open_ask       ticks: the bid and ask of the bar's first trade (tick-by-tick days; NaN otherwise)
and OUT/<date>.json (the sidecar): the contract, each file's RTH volume that day, the record style, gap
minutes, record and volume totals and the npz's sha256. A minute with no trade carries the last close
(open = high = low = close) with zero volume: a gap (OF18 plan section 2).

Front month: the file with the larger RTH volume that day (the OF13 rule) among --contracts.
Records (`<q4f4I`, 40 bytes after a 56-byte header): Sierra time (us since 1899-12-30 UTC), open, high, low,
close (price x100 here, as the files store them), trades, volume, bid volume, ask volume. Two styles
(surveyed 2026-11-12): 1-s bundles through August 2026 (OHLC of the second's trade prices; one record per
second) and tick by tick from September 2026 (one record per trade, several per second; high = ask,
low = bid, close = the trade price, open a marker). Style per day: "tick" when more than half of the
day's records share their second with the next record. Bars: for bundles the first open, the maximum
high, the minimum low and the last close of the minute's records; for ticks open, high, low and close all
from the close field. A bundle never spans a minute boundary (it covers one whole second).
Only each day's RTH byte range is read (binary search on the timestamps). Standard library + numpy.
"""
import argparse
import hashlib
import json
import mmap
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))
from ragrl import real  # noqa: E402  (stdlib only at import: the manifest and the CT clock)

REC = np.dtype([("t", "<i8"), ("o", "<f4"), ("h", "<f4"), ("l", "<f4"), ("c", "<f4"), ("n", "<u4"), ("v", "<u4"),
                ("bv", "<u4"), ("av", "<u4")])
HEADER = 56
PRICE_SCALE = 100.0  # the files store price x100
TICK = 0.25
MINUTES = 390
CONTRACTS = ("ESH26", "ESM26", "ESU26", "ESZ26")


class Scid:
    """A read-only view of one .scid file; records() returns a day's range as a numpy record array."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        hs = int.from_bytes(self.mm[4:8], "little")
        if hs != HEADER or (len(self.mm) - HEADER) % REC.itemsize:
            raise ValueError(f"{path}: header {hs}, size {len(self.mm)}: not a <q4f4I .scid file")
        self.n = (len(self.mm) - HEADER) // REC.itemsize

    def t(self, i: int) -> int:
        o = HEADER + i * REC.itemsize
        return int.from_bytes(self.mm[o:o + 8], "little", signed=True)

    def first_at_or_after(self, t: int) -> int:
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.t(mid) < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def records(self, t0: int, t1: int) -> np.ndarray:
        """Records with Sierra time in [t0, t1)."""
        a, b = self.first_at_or_after(t0), self.first_at_or_after(t1)
        return np.frombuffer(self.mm, dtype=REC, count=b - a, offset=HEADER + a * REC.itemsize).copy()

    def close(self):
        self.mm.close()
        self.f.close()


def bars_of(r: np.ndarray, t0: int) -> tuple:
    """(bars dict, style, tick share) from one day's RTH records; t0 = the open in Sierra microseconds."""
    sec = r["t"] // 1_000_000
    share = float((sec[1:] == sec[:-1]).mean()) if len(r) > 1 else 0.0
    style = "tick" if share > 0.5 else "bundle"
    m = ((r["t"] - t0) // 60_000_000).astype(np.int64)
    if len(m) and (m.min() < 0 or m.max() >= MINUTES):
        raise ValueError("records outside the RTH minutes")
    c = r["c"].astype(np.float64) / PRICE_SCALE / TICK
    if style == "tick":
        o = h = l = c
        bid, ask = r["l"].astype(np.float64) / PRICE_SCALE / TICK, r["h"].astype(np.float64) / PRICE_SCALE / TICK
    else:
        o = r["o"].astype(np.float64) / PRICE_SCALE / TICK
        h = r["h"].astype(np.float64) / PRICE_SCALE / TICK
        l = r["l"].astype(np.float64) / PRICE_SCALE / TICK
    out = {k: np.full(MINUTES, np.nan) for k in ("open", "high", "low", "close", "open_bid", "open_ask")}
    for k in ("volume", "buy", "sell", "trades"):
        out[k] = np.zeros(MINUTES, np.int64)
    np.add.at(out["volume"], m, r["v"].astype(np.int64))
    np.add.at(out["buy"], m, r["av"].astype(np.int64))
    np.add.at(out["sell"], m, r["bv"].astype(np.int64))
    np.add.at(out["trades"], m, r["n"].astype(np.int64))
    if len(m):
        starts = np.flatnonzero(np.r_[True, m[1:] != m[:-1]])  # records are time-ordered: one run per minute
        ends = np.r_[starts[1:], len(m)]
        mins = m[starts]
        out["open"][mins] = o[starts]
        out["close"][mins] = c[ends - 1]
        out["high"][mins] = np.maximum.reduceat(h, starts)
        out["low"][mins] = np.minimum.reduceat(l, starts)
        if style == "tick":
            out["open_bid"][mins] = bid[starts]
            out["open_ask"][mins] = ask[starts]
    # a minute without a trade carries the last close (no volume): a gap
    last = np.nan
    first = next((i for i in range(MINUTES) if not np.isnan(out["close"][i])), None)
    if first is not None:
        last = out["open"][first]
    for i in range(MINUTES):
        if out["volume"][i] == 0 or np.isnan(out["close"][i]):
            for k in ("open", "high", "low", "close"):
                out[k][i] = last
        else:
            last = out["close"][i]
    return out, style, share


def build_day(day: str, files: dict, manifest: dict) -> tuple:
    """(bars, sidecar) for one CT day, or (None, reason)."""
    o_unix, c_unix = real.rth_span(day, manifest)
    t0, t1 = real.sierra_us(o_unix), real.sierra_us(c_unix)
    per = {}
    for name, s in files.items():
        r = s.records(t0, t1)
        per[name] = r
    vols = {k: int(v["v"].astype(np.int64).sum()) for k, v in per.items()}
    if not vols or max(vols.values()) == 0:
        return None, "no RTH trades"
    front = max(vols, key=lambda k: vols[k])
    r = per[front]
    if np.any(np.diff(r["t"]) < 0):
        raise ValueError(f"{day} {front}: records out of time order")
    bars, style, share = bars_of(r, t0)
    gaps = int((bars["volume"] == 0).sum())
    live = bars["volume"] > 0
    viol = int((live & ((bars["high"] < np.maximum(bars["open"], bars["close"])) |
                        (bars["low"] > np.minimum(bars["open"], bars["close"])))).sum())
    if int(bars["volume"].sum()) != vols[front]:
        raise ValueError(f"{day}: bar volume {int(bars['volume'].sum())} != record volume {vols[front]}")
    side = {"schema": "of18_bars/1", "tool": "tools/scid_to_bars.py", "date": day, "contract": front, "rth_volume": vols,
            "style": style, "same_second_share": round(share, 4), "records": int(len(r)), "volume": vols[front],
            "gap_minutes": gaps, "ohlc_violations": viol, "rth_open_unix": o_unix, "price_unit": "ticks (0.25)"}
    return bars, side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--scid-dir", default="C:/SierraChart/Data")
    ap.add_argument("--out-dir", default=str(REPO / "build-avx512" / "real" / "bars"))
    ap.add_argument("--contracts", default=",".join(CONTRACTS))
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)
    m = real.load_manifest(a.manifest)
    cutoff = date.fromisoformat(m["confirm_rule"]["from"])
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    if d1 >= cutoff or d0 > d1:
        print(f"scid_to_bars: refused: days on or after {cutoff} are the real-data confirm block (or start > end)", file=sys.stderr)
        return 2
    files = {}
    for name in a.contracts.split(","):
        p = Path(a.scid_dir) / f"{name}_FUT_CME.scid"
        if p.exists():
            files[name] = Scid(p)
    if not files:
        print(f"scid_to_bars: no .scid files for {a.contracts} in {a.scid_dir}", file=sys.stderr)
        return 1
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    try:
        d = d0
        while d <= d1:
            day = d.isoformat()
            d += timedelta(days=1)
            if date.fromisoformat(day).weekday() >= 5:
                continue
            bars, side = build_day(day, files, m)
            if bars is None:
                skipped += 1
                print(f"{day}: skipped ({side})")
                continue
            npz = out / f"{day}.npz"
            tmp = out / f"{day}.tmp.npz"
            np.savez(tmp, **bars)
            tmp.replace(npz)
            side["npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
            (out / f"{day}.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
            made += 1
            print(f"{day} {side['contract']} {side['style']}: volume {side['volume']}, gaps {side['gap_minutes']}")
    finally:
        for s in files.values():
            s.close()
    print(f"{made} days written to {out}, {skipped} weekdays without RTH trades")
    return 0


if __name__ == "__main__":
    sys.exit(main())
