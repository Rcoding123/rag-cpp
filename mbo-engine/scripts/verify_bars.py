#!/usr/bin/env python3
"""Independent check of bar_extract against the Python databento reader.

1. Whole file: per-bucket n_add/n_cancel/n_modify/n_trades/buy_vol/sell_vol/
   last_trade_px recomputed with numpy must equal the C++ rows (and the set of
   buckets with records must match, apart from buckets before the first F_LAST).
2. Prefix of N records: a separate dict-based Python order book is replayed and
   its touch and 5-level totals at every bucket end must equal the C++ rows.
Usage: verify_bars.py BARS_FILE DBN_FILE [PREFIX_RECORDS=300000]
"""
import sys
from collections import defaultdict

import databento as db
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bars_to_parquet import read_bars  # noqa: E402

UNDEF = 9223372036854775807


def check_counts(a, rec, bucket_ns):
    b = (rec["ts_recv"] // bucket_ns).astype(np.int64)
    act = rec["action"].view("u1"); side = rec["side"].view("u1")
    keys = np.unique(b)
    got = {int(r["ts_end_ns"]) // bucket_ns - 1: r for r in a}
    first_boundary = int(np.nonzero(rec["flags"] & 128)[0][0])
    first_bucket = int(b[first_boundary])
    exp_keys = [int(k) for k in keys if k >= first_bucket]
    if sorted(got) != exp_keys:
        return f"bucket sets differ: cpp {len(got)} vs py {len(exp_keys)}"
    sz = rec["size"].astype(np.int64)
    idx = np.searchsorted(keys, b)  # bucket ordinal
    n = len(keys)
    def agg(mask, w=None):
        return np.bincount(idx[mask], weights=None if w is None else w[mask], minlength=n)
    t = act == ord("T")
    cols = {"n_add": agg(act == ord("A")), "n_cancel": agg(act == ord("C")), "n_modify": agg(act == ord("M")),
            "n_trades": agg(t), "buy_vol": agg(t & (side == ord("B")), sz), "sell_vol": agg(t & (side == ord("A")), sz)}
    # last trade price per bucket
    lp = np.zeros(n, dtype=np.int64)
    ti = np.nonzero(t)[0]
    lp[idx[ti]] = rec["price"][ti]  # later assignment wins = last in order
    bad = []
    for ordinal, k in enumerate(keys):
        if int(k) < first_bucket:
            continue
        r = got[int(k)]
        for name, arr in cols.items():
            if int(r[name]) != int(arr[ordinal]):
                bad.append((int(k), name, int(r[name]), int(arr[ordinal])))
        if int(r["last_trade_px"]) != int(lp[ordinal]):
            bad.append((int(k), "last_trade_px"))
        if len(bad) > 5:
            break
    return None if not bad else f"count mismatches: {bad[:5]}"


def py_book_check(a, rec, bucket_ns, nrec):
    orders = {}  # id -> [side, price, size]
    lv = {"B": defaultdict(int), "A": defaultdict(int)}
    def rm(o, q):
        lv[o[0]][o[1]] -= q
        if lv[o[0]][o[1]] == 0:
            del lv[o[0]][o[1]]
    got = {int(r["ts_end_ns"]) // bucket_ns - 1: r for r in a}
    cur = None; snap = None; checked = 0
    for i in range(min(nrec, len(rec))):
        r = rec[i]
        bk = int(r["ts_recv"]) // bucket_ns
        if cur is not None and bk != cur:
            if snap is not None and cur in got:
                g = got[cur]
                exp_b = snap[0]; exp_a = snap[1]
                gb = [int(x) for x in g["bid_sz"]]; ga = [int(x) for x in g["ask_sz"]]
                eb = [s for _, s in exp_b[:5]] + [0] * (5 - len(exp_b[:5]))
                ea = [s for _, s in exp_a[:5]] + [0] * (5 - len(exp_a[:5]))
                ebp = exp_b[0][0] if exp_b else 0; eap = exp_a[0][0] if exp_a else 0
                if (int(g["bid_px"]), int(g["ask_px"]), gb, ga) != (ebp, eap, eb, ea):
                    return f"touch/depth mismatch bucket {cur}: cpp {int(g['bid_px'])},{int(g['ask_px'])},{gb},{ga} py {ebp},{eap},{eb},{ea}"
                checked += 1
        cur = bk
        a_, s_, oid, px, sz = chr(r["action"][0]) if isinstance(r["action"], bytes) else chr(r["action"]), None, int(r["order_id"]), int(r["price"]), int(r["size"])
        act = r["action"].decode(); side = r["side"].decode()
        if act == "A":
            if oid in orders:
                rm(orders[oid], orders[oid][2]); del orders[oid]
            orders[oid] = [side, px, sz]; lv[side][px] += sz
        elif act == "C":
            o = orders.get(oid)
            if o:
                q = min(sz, o[2]); rm(o, q); o[2] -= q
                if o[2] == 0: del orders[oid]
        elif act == "M":
            o = orders.get(oid)
            if o is None:
                orders[oid] = [side, px, sz]; lv[side][px] += sz
            else:
                if px == o[1] and sz <= o[2]:
                    lv[o[0]][o[1]] -= o[2] - sz; o[2] = sz
                else:
                    rm(o, o[2]); o[0] = side if side in "AB" else o[0]; o[1] = px; o[2] = sz; lv[o[0]][px] += sz
        elif act == "R":
            orders.clear(); lv["B"].clear(); lv["A"].clear()
        if int(r["flags"]) & 128:
            snap = (sorted(lv["B"].items(), reverse=True), sorted(lv["A"].items()))
    return None if checked else "nothing checked", checked


def main():
    bars, dbn = sys.argv[1], sys.argv[2]
    nrec = int(sys.argv[3]) if len(sys.argv) > 3 else 300000
    a, ms = read_bars(bars)
    bucket_ns = ms * 1_000_000
    rec = db.DBNStore.from_file(dbn).to_ndarray()
    err = check_counts(a, rec, bucket_ns)
    print("counts:", "OK" if err is None else "FAIL " + err, f"({len(a)} rows, {len(rec)} records)")
    res = py_book_check(a, rec, bucket_ns, nrec)
    if isinstance(res, tuple):
        err2, checked = res
    else:
        err2, checked = res, 0
    print("python-book touch/depth:", "OK" if err2 is None else "FAIL " + str(err2), f"({checked} buckets compared over first {nrec} records)")
    sys.exit(1 if (err or err2) else 0)


main()
