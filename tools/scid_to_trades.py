#!/usr/bin/env python3
"""scid_to_trades.py -- OF19: the RTH trades of the ES front month from Sierra Chart tick records.

usage:
  python tools/scid_to_trades.py --start 2026-09-08 --end 2026-11-11
         [--scid-dir C:/SierraChart/Data] [--bars build-avx512/real/bars] [--out-dir build-avx512/real/trades]

For each day with an OF18 bar file whose style is "tick" (records carry the bid and ask), the front month is the
bar sidecar's contract. Days on or after the manifest's confirm_rule.from (2026-11-12) are refused (exit 2, before
any data file is opened). Output: OUT/<date>.npz with one row per trade, in record order:
  t      seconds since 08:30 CT (float64, microseconds exact)
  price  ticks (0.25);  bid, ask  ticks: the quote in the trade's record (high = ask, low = bid)
  size   lots;  side  +1 buy aggressor (the record's ask volume), -1 sell aggressor (bid volume)
A record with both volumes gives two trades (buy first); one with neither takes its side from the quote (at or
above the ask: buy; at or below the bid: sell; else dropped and counted). OUT/<date>.json: counts, the volume
check against the OF18 bar file (must be equal) and the npz's sha256.
"""
import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO / "python")):
    if p not in sys.path:
        sys.path.insert(0, p)
from ragrl import real  # noqa: E402
from scid_to_bars import PRICE_SCALE, TICK, Scid  # noqa: E402


def trades_of(r: np.ndarray, t0_us: int) -> tuple:
    px = np.round(r["c"].astype(np.float64) / PRICE_SCALE / TICK)
    bid = np.round(r["l"].astype(np.float64) / PRICE_SCALE / TICK)
    ask = np.round(r["h"].astype(np.float64) / PRICE_SCALE / TICK)
    av, bv, v = r["av"].astype(np.int64), r["bv"].astype(np.int64), r["v"].astype(np.int64)
    t = (r["t"] - t0_us) / 1e6
    rows = []
    neither = (av == 0) & (bv == 0) & (v > 0)
    inferred_buy = neither & (px >= ask)
    inferred_sell = neither & (px <= bid) & ~inferred_buy
    drop = neither & ~inferred_buy & ~inferred_sell
    dropped, dropped_vol = int(drop.sum()), int(v[drop].sum())
    for mask, size, side in ((av > 0, av, 1), (bv > 0, bv, -1), (inferred_buy, v, 1), (inferred_sell, v, -1)):
        i = np.flatnonzero(mask)
        rows.append((i, side, size[i]))
    idx = np.concatenate([x[0] for x in rows])
    side = np.concatenate([np.full(len(x[0]), x[1], np.int8) for x in rows])
    size = np.concatenate([x[2] for x in rows])
    # record order, buy before sell within a record (stable sort on the record index, buy rows come first)
    o = np.lexsort((-side, idx))
    idx, side, size = idx[o], side[o], size[o]
    out = {"t": t[idx], "price": px[idx], "bid": bid[idx], "ask": ask[idx], "size": size, "side": side}
    return out, dropped, dropped_vol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--scid-dir", default="C:/SierraChart/Data")
    ap.add_argument("--bars", default=str(REPO / "build-avx512" / "real" / "bars"))
    ap.add_argument("--out-dir", default=str(REPO / "build-avx512" / "real" / "trades"))
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)
    m = real.load_manifest(a.manifest)
    cutoff = date.fromisoformat(m["confirm_rule"]["from"])
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    if d1 >= cutoff or d0 > d1:
        print(f"scid_to_trades: refused: days on or after {cutoff} are the real-data confirm block (or start > end)", file=sys.stderr)
        return 2
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, made = {}, 0
    try:
        d = d0
        while d <= d1:
            day = d.isoformat()
            d += timedelta(days=1)
            sp = Path(a.bars) / f"{day}.json"
            if not sp.exists():
                continue
            bside = json.loads(sp.read_text(encoding="utf-8"))
            if bside["style"] != "tick":
                continue
            c = bside["contract"]
            if c not in files:
                files[c] = Scid(Path(a.scid_dir) / f"{c}_FUT_CME.scid")
            o_unix, c_unix = real.rth_span(day, m)
            t0, t1 = real.sierra_us(o_unix), real.sierra_us(c_unix)
            r = files[c].records(t0, t1)
            tr, dropped, dropped_vol = trades_of(r, t0)
            vol = int(tr["size"].sum()) + dropped_vol
            if vol != bside["volume"]:
                raise ValueError(f"{day}: trade volume {vol} (+ dropped) != the OF18 bar volume {bside['volume']}")
            npz = out / f"{day}.npz"
            tmp = out / f"{day}.tmp.npz"
            np.savez(tmp, **tr)
            tmp.replace(npz)
            side = {"schema": "of19_trades/1", "tool": "tools/scid_to_trades.py", "date": day, "contract": c, "records": int(len(r)),
                    "trades": int(len(tr["t"])), "volume": int(tr["size"].sum()), "dropped_records": dropped,
                    "dropped_volume": dropped_vol, "bars_volume": bside["volume"],
                    "outside_quote": int(((tr["price"] > tr["ask"]) | (tr["price"] < tr["bid"])).sum()),
                    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
            (out / f"{day}.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
            made += 1
            print(f"{day} {c}: {side['trades']} trades, volume {side['volume']}, dropped {dropped}, outside quote {side['outside_quote']}")
    finally:
        for s in files.values():
            s.close()
    print(f"{made} days written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
