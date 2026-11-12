#!/usr/bin/env python3
"""sierra_to_ticks.py -- OF16: one RTH day of Sierra Chart ES data (.depth book + .scid trades)
into the tick_csv.hpp format, for ReplayScalpEnv / ragrl.real.

usage (standard library only):
  python tools/sierra_to_ticks.py DATE [--manifest experiments/configs/of16_real_manifest.json]
         [--depth-dir C:/SierraChart/Data/MarketDepthData] [--scid-dir C:/SierraChart/Data]
         [--out-dir build-avx512/real] [--role-ok test|confirm --i-am-the-orchestrator]

DATE must be a dev day of the manifest (experiments/configs/of16_real_manifest.json). Test and
confirm days are refused (exit 2, before any data file is opened) unless --role-ok names that
role AND --i-am-the-orchestrator is given: the flag is for the orchestrator's export step, never for development.

Output: OUT/DATE/ticks.csv and OUT/DATE/ticks.json (the sidecar: contract, date, RTH span,
counts, dropped quotes, gaps, trade-through counts and the CSV's sha256).

  T,<s>,<price>,<size>,<B|S>                         a trade; B/S = aggressor (scid ask / bid volume)
  Q,<s>,<bid>,<ask>,<bid size 1..5>,<ask size 1..5>  the book after a batch that changed it
Times are seconds since 08:30 CT of DATE (time_unit "s", exact microseconds), so bucket
boundaries stay exact over the day (Unix seconds in a double would drift ~20 ms by the close).
Levels are price levels 0..4 ticks from the best, 0 if empty (tools/real_vs_sim.py's book).

How the day is built (following real_vs_sim.task_depth):
  book    every .depth event from the start of the file is replayed (1 clear, 2/4 bid set,
          3/5 ask set, 6/7 delete); a snapshot is taken at the end of each batch (flag 1, or a
          change of timestamp) and written as a Q line when its top 5 levels differ from the last
          one written. Crossed or locked snapshots (ask <= bid) and one-sided books are dropped
          and counted.
  start   the tape starts at 08:30:00 CT with the book as it stood then (a Q line at time 0:
          ReplayScalpEnv's 150 s warm-up runs on the tape from there). If the book is not valid
          at the open (a recording that starts later), the tape starts at the first 100 ms grid
          point after the first valid snapshot.
  trades  .scid records (tick by tick: close = price x100, high/low = ask/bid at the trade,
          ask volume = buyer-aggressor lots, bid volume = seller-aggressor) with Sierra time in
          [08:30, 15:00) CT, read by byte range: the record range is found by binary search on
          the timestamps (8 bytes per probe), so no record outside the day's RTH is read. A
          record with both volumes gives two T lines; one with neither takes its side from the
          prevailing quote (at/above ask B, at/below bid S, else dropped and counted).
  merge   by timestamp, trades before book events at the same microsecond. Sierra writes the
          two files separately with millisecond times plus a within-millisecond counter, so
          sub-millisecond order between a trade and a book update is not recorded.
  end     the earlier of 15:00 CT and the last book event; trades after the book ends are
          dropped and counted. A day whose book reaches the close gets a last Q line at
          14:59:59.95 so the final 100 ms bucket closes on the grid.
Time zone: ragrl.real (zoneinfo America/Chicago when available, else the explicit US rule:
CDT before the first Sunday of November, CST after).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO / "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_vs_sim as rvs  # noqa: E402  (parsers: Sierra epoch, price -> ticks, record layouts)
from ragrl import real  # noqa: E402  (stdlib only at import: the manifest and the CT clock)

DEPTH_HEADER, DEPTH_REC, DEPTH_FMT = 64, 24, "<qBBHfII"
SCID_HEADER, SCID_REC, SCID_FMT = 56, 40, "<q4f4I"
LEVELS = 5
GRID_US = rvs.GRID_US  # 100 ms
TICK = 0.25
GAP_US = 1_000_000


def _header(f, path, want_header, want_rec):
    head = f.read(want_header)
    if len(head) < 12:
        raise ValueError(f"{path}: no header")
    hsize, rec = struct.unpack_from("<II", head, 4)
    if hsize != want_header or rec != want_rec:
        raise ValueError(f"{path}: header {hsize}/{rec}, expected {want_header}/{want_rec}")
    return hsize


def scid_range(path, t0_us, t1_us):
    """scid records with Sierra time in [t0_us, t1_us): binary search on timestamps by seek, then
    one read of exactly that byte range (the file may be 2 GB and hold other days)."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        hsize = _header(f, path, SCID_HEADER, SCID_REC)
        n = (size - hsize) // SCID_REC

        def t_at(i):
            f.seek(hsize + SCID_REC * i)
            return struct.unpack("<q", f.read(8))[0]

        def first_at_or_after(t):
            lo, hi = 0, n
            while lo < hi:
                m = (lo + hi) // 2
                if t_at(m) < t:
                    lo = m + 1
                else:
                    hi = m
            return lo

        i, j = first_at_or_after(t0_us), first_at_or_after(t1_us)
        f.seek(hsize + SCID_REC * i)
        data = f.read(SCID_REC * (j - i))
    recs = list(struct.iter_unpack(SCID_FMT, data))
    if any(r[0] < t0_us or r[0] >= t1_us for r in recs) or any(recs[k][0] > recs[k + 1][0] for k in range(len(recs) - 1)):
        raise ValueError(f"{path}: records in [{i}, {j}) are not time-ordered inside the day")
    return recs, {"first_record": i, "records": j - i, "bytes": len(data)}


def depth_events(path, chunk=1 << 16):
    """The .depth file's events in order (streamed; stops wherever the caller stops)."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size  # a file still being written: a snapshot of its size now
        hsize = _header(f, path, DEPTH_HEADER, DEPTH_REC)
        left = (size - hsize) // DEPTH_REC
        while left:
            k = min(left, chunk)
            buf = f.read(DEPTH_REC * k)
            if len(buf) < DEPTH_REC * k:
                k = len(buf) // DEPTH_REC
                buf = buf[:DEPTH_REC * k]
                left = k
            yield from struct.iter_unpack(DEPTH_FMT, buf)
            left -= k


def _rel(t_us, open_us):
    d = t_us - open_us
    return f"{d // 1_000_000}.{d % 1_000_000:06d}"


class Book:
    """real_vs_sim.task_depth's book (price ticks -> quantity), with the best prices kept incrementally."""

    def __init__(self):
        self.bids, self.asks = {}, {}
        self.bb = self.ba = None

    def event(self, cmd, p, qty):
        if cmd == 1:
            self.bids.clear()
            self.asks.clear()
            self.bb = self.ba = None
        elif cmd == 2 or cmd == 4:
            self.bids[p] = qty
            if self.bb is None or p > self.bb:
                self.bb = p
        elif cmd == 3 or cmd == 5:
            self.asks[p] = qty
            if self.ba is None or p < self.ba:
                self.ba = p
        elif cmd == 6:
            if self.bids.pop(p, None) is not None and p == self.bb:
                self.bb = max(self.bids) if self.bids else None
        elif cmd == 7:
            if self.asks.pop(p, None) is not None and p == self.ba:
                self.ba = min(self.asks) if self.asks else None

    def snapshot(self):
        """(bid, ask, bid sizes, ask sizes), 'one_sided' or 'crossed'."""
        bb, ba = self.bb, self.ba
        if bb is None or ba is None:
            return "one_sided"
        if ba <= bb:
            return "crossed"
        g, h = self.bids.get, self.asks.get
        return (bb, ba, tuple(g(bb - j, 0) for j in range(LEVELS)), tuple(h(ba + j, 0) for j in range(LEVELS)))


MIN_QUOTE_SHARE = 0.01   # Q lines per RTH depth event below which a tape is implausibly empty
FLAGS_UNSET_SHARE = 0.5  # timestamp changes inside an unflagged batch above which the flags are not trusted


def convert_day(depth_path, scid_path, open_unix, close_unix, out_csv):
    """convert() with end-of-batch flags; if Sierra left them unset on most timestamp changes, again with a
    batch per timestamp (reported as batching "timestamp"). Refuses a tape with implausibly few quotes."""
    res = convert(depth_path, scid_path, open_unix, close_unix, out_csv, use_flags=True)
    c = res["counts"]
    res["batching"] = "flags"
    share = c["batches_spanning_timestamps"] / max(1, c["timestamp_changes"])
    if share > FLAGS_UNSET_SHARE:
        res = convert(depth_path, scid_path, open_unix, close_unix, out_csv, use_flags=False)
        res["batching"] = f"timestamp (fallback: end-of-batch flag unset on {share:.0%} of timestamp changes)"
        c = res["counts"]
    if c["depth_events_rth"] >= 10_000 and c["quotes"] < MIN_QUOTE_SHARE * c["depth_events_rth"]:
        raise RuntimeError(f"implausibly few quotes: {c['quotes']} Q lines for {c['depth_events_rth']} RTH depth events "
                           f"({res['batching']}); not exported")
    return res


def convert(depth_path, scid_path, open_unix, close_unix, out_csv, use_flags=True):
    """Writes the day's CSV; returns the sidecar's measurements. use_flags=False: a batch per timestamp."""
    open_us, close_us = real.sierra_us(open_unix), real.sierra_us(close_unix)
    trades, scid_info = scid_range(scid_path, open_us, close_us)
    book = Book()
    out = open(out_csv, "wb")
    sha = hashlib.sha256()
    buf = []
    nbytes = 0
    c = {"trades": 0, "trade_records": 0, "volume": 0, "quotes": 0, "depth_events": 0, "depth_events_rth": 0, "batches_rth": 0,
         "book_clears_rth": 0}
    drop = {"crossed_or_locked": 0, "one_sided": 0, "trades_before_start": 0, "trades_after_book_end": 0, "trades_unknown_side": 0}
    tq = {"above_ask": 0, "below_bid": 0, "no_quote": 0}
    gaps, line_gaps = [], []  # (from, length) of book gaps (no depth event) and of output-line gaps > 1 s
    tq["outside_book_stale"] = 0
    c["trades_book_stale"] = 0
    c["batches_spanning_timestamps"] = 0
    c["timestamp_changes"] = 0
    st = {"started": False, "start_us": None, "last_line": None, "last_q": None, "pre_open_valid": False, "padded": False,
          "last_depth": None}

    def write(line, t):
        nonlocal nbytes
        b = line.encode("ascii")
        buf.append(b)
        nbytes += len(b)
        if st["last_line"] is not None and t - st["last_line"] > GAP_US:
            line_gaps.append((st["last_line"], t - st["last_line"]))
        st["last_line"] = t
        if len(buf) >= 4096:
            flush()

    def flush():
        data = b"".join(buf)
        sha.update(data)
        out.write(data)
        buf.clear()

    def q_line(t, snap, force=False):
        if not force and snap == st["last_q"]:
            return
        bb, ba, b5, a5 = snap
        write(f"Q,{_rel(t, open_us)},{bb * TICK:.2f},{ba * TICK:.2f}," + ",".join(map(str, b5)) + "," + ",".join(map(str, a5)) + "\n", t)
        st["last_q"] = snap
        c["quotes"] += 1

    def emit_book(t):
        s = book.snapshot()
        if isinstance(s, str):
            drop["crossed_or_locked" if s == "crossed" else "one_sided"] += 1
            return
        q_line(t, s)

    ti = 0

    def flush_trades(upto_us):
        """T lines for every trade with time <= upto_us (trades before the start are dropped)."""
        nonlocal ti
        while ti < len(trades) and trades[ti][0] <= upto_us:
            t, _o, _ask, _bid, close, _nt, vol, bv, av = trades[ti]
            ti += 1
            c["trade_records"] += 1
            if not st["started"] or t < st["start_us"]:
                drop["trades_before_start"] += 1
                continue
            p = rvs.ticks(close)
            q = st["last_q"]
            sides = []
            if av > 0:
                sides.append(("B", av if bv > 0 else vol))
            if bv > 0:
                sides.append(("S", bv if av > 0 else vol))
            if not sides:
                if q is not None and p >= q[1]:
                    sides = [("B", vol)]
                elif q is not None and p <= q[0]:
                    sides = [("S", vol)]
                else:
                    drop["trades_unknown_side"] += 1
                    continue
            stale = st["last_depth"] is None or t - st["last_depth"] > GAP_US
            c["trades_book_stale"] += stale
            if q is None:
                tq["no_quote"] += 1
            elif p > q[1] or p < q[0]:
                tq["above_ask" if p > q[1] else "below_bid"] += 1
                tq["outside_book_stale"] += stale
            for side, size in sides:
                write(f"T,{_rel(t, open_us)},{p * TICK:.2f},{size},{side}\n", t)
                c["trades"] += 1
                c["volume"] += size

    def start(t0):
        s = book.snapshot()
        if isinstance(s, str):  # the book went bad again before the start: wait for the next valid batch
            st["start_us"] = None
            st["pre_open_valid"] = False
            return
        st["started"] = True
        st["start_us"] = t0
        q_line(t0, s, force=True)  # the book at the start

    def batch_closed(tb):
        if tb < open_us:
            st["pre_open_valid"] = not isinstance(book.snapshot(), str)
            return
        c["batches_rth"] += 1
        if st["started"]:
            emit_book(tb)
        elif st["start_us"] is None and not isinstance(book.snapshot(), str):
            st["start_us"] = open_us + ((tb - open_us) // GRID_US + 1) * GRID_US

    # A batch ends at a change of timestamp after an event flagged end-of-batch: the book is
    # snapshotted once per timestamp, never in the middle of a batch.
    last_t, last_fl, in_batch = None, 1, False
    reached_close = False
    for t, cmd, fl, _nord, price, qty, _r in depth_events(depth_path):
        if t >= close_us:
            reached_close = True
            break
        c["depth_events"] += 1
        if in_batch and t != last_t:
            c["timestamp_changes"] += 1
            if last_fl & 1 or not use_flags:
                batch_closed(last_t)
                in_batch = False
            else:
                c["batches_spanning_timestamps"] += 1
        if t >= open_us:
            c["depth_events_rth"] += 1
            if cmd == 1:
                c["book_clears_rth"] += 1
            if not st["started"]:
                if st["start_us"] is None and st["pre_open_valid"]:
                    st["start_us"] = open_us
                if st["start_us"] is not None and t >= st["start_us"]:
                    start(st["start_us"])
            flush_trades(t)  # trades first at the same microsecond; their book age is measured before this event
            prev = st["last_depth"] if st["last_depth"] is not None else open_us
            if t - prev > GAP_US:
                gaps.append((prev, t - prev))
            st["last_depth"] = t
        book.event(cmd, rvs.ticks(price) if cmd != 1 else 0, qty)
        last_t, last_fl, in_batch = t, fl, True
    if in_batch:
        batch_closed(last_t)
    if reached_close and st["last_depth"] is not None and close_us - st["last_depth"] > GAP_US:
        gaps.append((st["last_depth"], close_us - st["last_depth"]))  # a silent book up to the close
    if not st["started"] and st["start_us"] is None and st["pre_open_valid"] and reached_close:
        st["start_us"] = open_us
    if not st["started"] and st["start_us"] is not None and (reached_close or (last_t is not None and last_t >= st["start_us"])):
        start(st["start_us"])
    end_us = close_us - 1 if reached_close else (last_t if last_t is not None else open_us)
    flush_trades(end_us)
    drop["trades_after_book_end"] = len(trades) - ti
    if reached_close and st["started"]:
        pad = close_us - GRID_US // 2
        s = book.snapshot()
        if st["last_line"] < pad and not isinstance(s, str):
            q_line(pad, s, force=True)
            st["padded"] = True
    flush()
    out.close()
    if not st["started"]:
        raise RuntimeError("no valid book inside RTH: nothing exported")
    span_end = close_us if reached_close else st["last_line"]
    missing = (st["start_us"] - open_us + close_us - span_end) / 1e6
    gaps.sort(key=lambda g: -g[1])
    line_gaps.sort(key=lambda g: -g[1])
    n_tr = c["trades"]
    return {
        "span": {"start_s": (st["start_us"] - open_us) / 1e6, "end_s": (span_end - open_us) / 1e6,
                 "start_ct": real.ct_clock(open_unix + (st["start_us"] - open_us) / 1e6),
                 "end_ct": real.ct_clock(open_unix + (span_end - open_us) / 1e6),
                 "seconds": (span_end - st["start_us"]) / 1e6, "book_reaches_close": reached_close},
        "rth_missing_seconds": missing,
        "counts": {**c, "lines": c["trades"] + c["quotes"]},
        "dropped": drop,
        "gaps_over_1s": {"what": "book gaps: no depth event for > 1 s inside the span (a recording outage when long)",
                         "count": len(gaps), "max_s": gaps[0][1] / 1e6 if gaps else 0.0,
                         "largest": [[real.ct_clock(open_unix + (g0 - open_us) / 1e6), g1 / 1e6] for g0, g1 in gaps[:10]],
                         "intervals_s": sorted([[(g0 - open_us) / 1e6, (g0 + g1 - open_us) / 1e6] for g0, g1 in gaps])},
        "line_gaps_over_1s": {"count": len(line_gaps), "max_s": line_gaps[0][1] / 1e6 if line_gaps else 0.0},
        "trades_outside_quote": {"what": "trade price above the prevailing ask or below the prevailing bid (the last Q line); "
                                         "outside_book_stale: of those, trades more than 1 s after the last depth event",
                                 "count": tq["above_ask"] + tq["below_bid"], "share": (tq["above_ask"] + tq["below_bid"]) / n_tr if n_tr else 0.0,
                                 "share_book_fresh": ((tq["above_ask"] + tq["below_bid"] - tq["outside_book_stale"]) /
                                                      max(1, n_tr - c["trades_book_stale"])), **tq},
        "padded_close": st["padded"],
        "scid": scid_info,
        "csv_bytes": nbytes,
        "csv_sha256": sha.hexdigest(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("date")
    ap.add_argument("--manifest", default=str(real.MANIFEST))
    ap.add_argument("--depth-dir")
    ap.add_argument("--scid-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--role-ok", choices=["test", "confirm"], help="orchestrator only: allow exporting a test / confirm day")
    ap.add_argument("--i-am-the-orchestrator", dest="orchestrator", action="store_true", help="required with --role-ok")
    a = ap.parse_args(argv)
    m = real.load_manifest(a.manifest)
    e = real.day_entry(a.date, m)
    role = e["role"]
    if a.role_ok and not a.orchestrator:
        print("sierra_to_ticks: refused: --role-ok is for the orchestrator only (--i-am-the-orchestrator)", file=sys.stderr)
        return 2
    if role != "dev" and not (role in ("test", "confirm") and a.role_ok == role):
        print(f"sierra_to_ticks: refused: {e['date']} has role {role!r} in {Path(a.manifest).name}; only dev days are exported "
              f"without --role-ok (orchestrator)", file=sys.stderr)
        return 2
    day = e["date"]
    depth = Path(a.depth_dir or m["data"]["depth_dir"]) / e["depth"]
    scid = Path(a.scid_dir or m["data"]["scid_dir"]) / e["scid"]
    out_dir = (Path(a.out_dir) if a.out_dir else real.export_root(m)) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    o, cl = real.rth_span(day, m)
    t = time.time()
    tmp = out_dir / "ticks.csv.part"
    try:
        res = convert_day(depth, scid, o, cl, tmp)
    except RuntimeError as err:
        tmp.unlink(missing_ok=True)
        print(f"sierra_to_ticks: {day}: {err}", file=sys.stderr)
        return 3
    os.replace(tmp, out_dir / "ticks.csv")
    side = {"schema": "of16_ticks/1", "tool": "tools/sierra_to_ticks.py", "date": day, "contract": e["contract"], "role": role,
            "manifest_sha256": real.manifest_sha256(a.manifest), "depth_file": depth.name, "depth_bytes": depth.stat().st_size,
            "scid_file": scid.name, "time_unit": real.TIME_UNIT, "time_origin": "seconds since 08:30 CT of date",
            "t0_unix": o, "rth_close_unix": cl, "utc_offset_hours": real.ct_offset_hours(o), "rth_ct": [m["rth"]["open"], m["rth"]["close"]],
            **res, "csv": "ticks.csv", "seconds_to_export": round(time.time() - t, 1)}
    (out_dir / "ticks.json").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
    c = res["counts"]
    print(f"{day} {e['contract']} ({role}): {res['span']['start_ct']}-{res['span']['end_ct']} CT, {c['trades']} trades, {c['quotes']} quotes, "
          f"{res['dropped']['crossed_or_locked']} crossed/locked dropped, {res['gaps_over_1s']['count']} gaps > 1 s, "
          f"{res['trades_outside_quote']['count']} trades outside the quote -> {out_dir / 'ticks.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
