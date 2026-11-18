#!/usr/bin/env python3
"""Classify every crossed F_LAST boundary (from book_check dumps) by the exchange
status of the SAME instrument at that ts_recv (databento `status` schema).
Codes are decoded from the dbn enums (StatusAction/StatusReason/TradingEvent), not assumed.
Usage: crossed_vs_status.py CROSSED_DIR STATUS_DBN [STATUS_DBN ...]
"""
import glob
import sys
from collections import Counter, defaultdict

import databento as db
import numpy as np

ACTION = {0: "None", 1: "PreOpen", 2: "PreCross", 3: "Quoting", 4: "Cross", 5: "Rotation",
          6: "NewPriceIndication", 7: "Trading", 8: "Halt", 9: "Pause", 10: "Suspend",
          11: "PreClose", 12: "Close", 13: "PostClose", 14: "SsrChange", 15: "NotAvailableForTrading"}
REASON = {0: "None", 1: "Scheduled", 2: "SurveillanceIntervention", 3: "MarketEvent",
          4: "InstrumentActivation", 5: "InstrumentExpiration", 6: "RecoveryInProcess"}
EVENT = {0: "None", 1: "NoCancel", 2: "ChangeTradingSession", 3: "ImpliedMatchingOn", 4: "ImpliedMatchingOff"}


DUMP = np.dtype([("i", "<u4"), ("t", "<u8")])
assert DUMP.itemsize == 12


def load_status(paths):
    parts = []
    for p in paths:
        a = db.DBNStore.from_file(p).to_ndarray()
        parts.append(a)
    a = np.concatenate(parts)
    a = a[a["action"] != 0]
    a = a[np.lexsort((a["ts_recv"], a["instrument_id"]))]
    return a


def main():
    cdir, spaths = sys.argv[1], sys.argv[2:]
    st = load_status(spaths)
    by = {}
    for iid in np.unique(st["instrument_id"]):
        s = st[st["instrument_id"] == iid]
        by[int(iid)] = s
    cnt = Counter(); clusters = defaultdict(list)
    total = 0
    for f in sorted(glob.glob(cdir + "/*.ts")):
        raw = np.fromfile(f, dtype=DUMP)  # packed 12-byte records: u32 instrument_id, u64 ts_recv
        day = f.rsplit("/", 1)[1][:8]
        for iid in np.unique(raw["i"]) if len(raw) else []:
            ts = raw["t"][raw["i"] == iid]
            s = by.get(int(iid))
            for t in ts:
                total += 1
                if s is None:
                    key = ("no status for instrument",)
                else:
                    k = np.searchsorted(s["ts_recv"], t, side="right") - 1
                    if k < 0:
                        key = ("before first status record",)
                    else:
                        r = s[k]
                        key = (ACTION.get(int(r["action"]), str(r["action"])), REASON.get(int(r["reason"]), str(r["reason"])),
                               EVENT.get(int(r["trading_event"]), str(r["trading_event"])),
                               "is_trading=" + r["is_trading"].decode(), "is_quoting=" + r["is_quoting"].decode())
                cnt[key] += 1
                if len(key) < 4 or key[3] == "is_trading=Y":
                    clusters[(day, key)].append(int(t))
    print(f"crossed boundaries total: {total:,}\n")
    nontr = sum(v for k, v in cnt.items() if len(k) >= 4 and k[3] == "is_trading=N")
    tr = sum(v for k, v in cnt.items() if len(k) >= 4 and k[3] == "is_trading=Y")
    other = total - nontr - tr
    print(f"in a NON-trading state (is_trading=N): {nontr:,}  ({100*nontr/total:.2f}%)")
    print(f"while status says TRADING (is_trading=Y): {tr:,}  ({100*tr/total:.2f}%)")
    print(f"no usable status: {other:,}\n")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"{v:>9,}  {' / '.join(k)}")
    print("\nCrossed boundaries while is_trading=Y or no status (clusters split on gaps > 2 s):")
    import datetime as dt
    for (day, key), ts in sorted(clusters.items()):
        ts = np.sort(np.array(ts, dtype=np.uint64)); cut = np.nonzero(np.diff(ts.astype(np.int64)) > 2_000_000_000)[0] + 1
        for c in np.split(ts, cut):
            f = lambda n: dt.datetime.fromtimestamp(int(n) / 1e9, dt.UTC).strftime("%H:%M:%S.%f")[:-3]
            print(f"  {day} {f(c[0])}-{f(c[-1])} UTC  n={len(c):>5}  {' / '.join(key)}")


main()
