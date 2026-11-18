#!/usr/bin/env python3
"""Phase 1 acceptance: the C++ reader (engine/build/dbn_digest) must match the
Python `databento` reader exactly on record counts and every field.

Usage: verify_dbn.py DIGEST_BIN FILE [FILE ...]
Digest = count, per-action counts, and per-field uint64 wrap-around sums of
x and of x*(i+1) (position-sensitive, so reordering is caught).
"""
import json
import subprocess
import sys
import time

import databento as db
import numpy as np

FIELDS = ["publisher_id", "instrument_id", "ts_event", "order_id", "price", "size",
          "flags", "channel_id", "action", "side", "ts_recv", "ts_in_delta", "sequence"]
M = (1 << 64) - 1


def py_digest(path):
    store = db.DBNStore.from_file(path)
    a = store.to_ndarray()
    n = len(a)
    idx = np.arange(1, n + 1, dtype=np.uint64)
    out = {"count": n, "sum": {}, "wsum": {}, "actions": {},
           "version": store.metadata.version, "dataset": store.metadata.dataset}
    for f in FIELDS:
        col = a[f]
        if f in ("action", "side"):
            col = col.view("u1")
        if col.dtype.kind == "i":
            x = col.astype(np.int64).view(np.uint64)
        else:
            x = col.astype(np.uint64)
        out["sum"][f] = str(int(x.sum(dtype=np.uint64)))
        out["wsum"][f] = str(int((x * idx).sum(dtype=np.uint64)))
    acts, cnts = np.unique(a["action"], return_counts=True)
    out["actions"] = {k.decode(): int(v) for k, v in zip(acts, cnts)}
    return out


def main():
    binp, files = sys.argv[1], sys.argv[2:]
    bad = 0
    for p in files:
        t = time.time()
        py = py_digest(p)
        tp = time.time() - t
        cp = json.loads(subprocess.run([binp, p], check=True, capture_output=True, text=True).stdout)
        ok = (cp["count"] == py["count"] and cp["actions"] == py["actions"] and
              cp["sum"] == py["sum"] and cp["wsum"] == py["wsum"] and
              cp["version"] == py["version"] and cp["dataset"] == py["dataset"] and cp["skipped"] == 0)
        print(f"{'OK ' if ok else 'BAD'} {p.split('/')[-1]} n={py['count']} "
              f"py={tp:.1f}s cpp={cp['seconds']:.1f}s")
        if not ok:
            bad += 1
            for k in ("count", "actions", "sum", "wsum"):
                if cp[k] != py[k]:
                    print("  mismatch", k, cp[k], py[k])
    print("ALL MATCH" if not bad else f"{bad} FILE(S) MISMATCH")
    sys.exit(1 if bad else 0)


main()
