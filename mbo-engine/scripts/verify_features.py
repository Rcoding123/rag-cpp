"""Parity: scripts/features.py (numba) vs the ORIGINAL rag-cpp FeatureBuilder (engine/build/feat_ref) on whole UTC days.

For each day the dense bucket stream (quote forward-filled from the bars, individual side-known trades from the trades
table) is fed to the original C++ builder; the vectorised kernel runs on the same day with the bars' OFI/absorption.
All 13 raw and 13 scaled features must agree on every bucket. `qok` is forced True (every bucket has a quote) so that
the comparison is pure arithmetic: validity masking is tested separately (scripts/test_features.py).
This also checks bar_extract's per-bucket OFI and absorption against the original code (the reference recomputes them
from individual trades and quotes).
Usage: verify_features.py DAY [DAY ...]   (YYYYMMDD of the bars/trades file)
"""
import glob
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import features as F

REF = "/srv/quant/code/rag-cpp/mbo-engine/engine/build/feat_ref"
bad = 0
for day in sys.argv[1:]:
    bars = pd.read_parquet(f"/srv/quant/data/features/bars/{day}.parquet")
    bars["bk"] = bars.ts_end_ns // F.BUCKET_NS - 1
    tr = pd.read_parquet(f"/srv/quant/data/features/trades/{day}.parquet", columns=["ts_recv", "price", "size", "side", "instrument_id"])
    tr = tr[tr.side.isin(["A", "B"])]
    assert bars.instrument_id.nunique() == 1
    iid = int(bars.instrument_id.iloc[0])
    b0, b1 = int(bars.bk.iloc[0]), int(bars.bk.iloc[-1]) + 1
    n = b1 - b0
    d = F.dense_unit(bars, tr.ts_recv.to_numpy(), np.round(tr.price.to_numpy() * 4), tr["size"].to_numpy().astype(np.float64),
                     (tr.side == "B").to_numpy(), iid, b0, b1, {}, None)
    qok = np.ones(n, bool)
    # Known, deliberate deviation: bar_extract writes OFI = 0 when either quote it compares has an EMPTY side (bid or ask 0);
    # the original computes it arithmetically from zero prices (garbage, e.g. the empty book at the start of a Sunday file).
    # Prove the deviation is confined to such buckets, then use the original formula there so the rest stays an exact comparison.
    pb, pa = np.r_[d["bid"][0], d["bid"][:-1]], np.r_[d["ask"][0], d["ask"][:-1]]
    pbs, pas = np.r_[d["bsz"][0, 0], d["bsz"][:-1, 0]], np.r_[d["asz"][0, 0], d["asz"][:-1, 0]]
    e_orig = (d["bid"] >= pb) * d["bsz"][:, 0] - (d["bid"] <= pb) * pbs - (d["ask"] <= pa) * d["asz"][:, 0] + (d["ask"] >= pa) * pas
    e_orig[0] = 0.0                                   # the original has no previous quote at the first bucket
    side_empty = (d["bid"] == 0) | (d["ask"] == 0) | (pb == 0) | (pa == 0)
    differ = ~np.isclose(e_orig, d["ofi"])
    n_dev = int(differ.sum())
    assert not (differ & ~side_empty).any(), "OFI differs from the original formula at a bucket with both sides present"
    d["ofi"] = np.where(side_empty, e_orig, d["ofi"])
    import os
    if os.environ.get("PERTURB") == "ofi": d["ofi"][n // 2] += 1.0          # negative control: must be caught
    if os.environ.get("PERTURB") == "quote": d["bsz"][n // 3, 0] += 1.0    # negative control: must be caught
    if os.environ.get("PERTURB") == "absorb": d["absorb"][n // 4:] += 0.5   # negative control: must be caught
    raw, x, _ = F.kernel(d["buy"], d["sell"], d["pv"], d["bid"], d["ask"], d["bsz"], d["asz"], d["ofi"], d["absorb"], qok, np.ones(F.NF), 0.0)
    # reference input: Q only when the dense quote changes (the reference keeps its quote), then trades, then E
    chg = np.ones(n, bool)
    q = np.concatenate([d["bid"][:, None], d["ask"][:, None], d["bsz"], d["asz"]], 1)
    chg[1:] = (q[1:] != q[:-1]).any(1)
    tb = (tr.ts_recv.to_numpy() // F.BUCKET_NS - b0)
    order = np.argsort(tb, kind="stable")
    tb_s = tb[order]; tp = np.round(tr.price.to_numpy()[order] * 4).astype(np.int64); ts_ = tr["size"].to_numpy()[order]
    tside = np.where(tr.side.to_numpy()[order] == "B", 0, 1)
    starts = np.searchsorted(tb_s, np.arange(n), "left"); ends = np.searchsorted(tb_s, np.arange(n), "right")
    lines = []
    for i in range(n):
        if chg[i]:
            lines.append("Q %d %d %s" % (q[i, 0], q[i, 1], " ".join(str(int(v)) for v in q[i, 2:])))
        for j in range(starts[i], ends[i]):
            lines.append("T %d %d %d" % (tp[j], ts_[j], tside[j]))
        lines.append("E")
    out = subprocess.run([REF], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    ref = np.array(out.split(), dtype=np.float64).reshape(n, 26)
    dr = np.abs(ref[:, :13] - raw); dx = np.abs(ref[:, 13:] - x)
    scale = np.maximum(1.0, np.abs(ref[:, :13]))
    worst_raw = float((dr / scale).max()); worst_x = float(dx.max())
    per = {f"f{k}": float(max((dr[:, k] / scale[:, k]).max(), dx[:, k].max())) for k in range(13)}
    ok = worst_raw < 1e-9 and worst_x < 1e-9 and not np.isnan(ref).any() and not np.isnan(raw).any()
    bad += not ok
    print(f"{day}: OFI deviation (empty-side buckets only): {n_dev} bucket(s)")
    print(f"{day}: {n:,} buckets, {int(chg.sum()):,} quote changes, {len(tr):,} trades | max rel diff raw {worst_raw:.2e}, scaled {worst_x:.2e} | "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        k = int(np.argmax(np.maximum((dr / scale).max(0), dx.max(0))))
        i = int(np.argmax(dr[:, k] / scale[:, k])); print("   worst feature", k, "bucket", i, "ref", ref[i, k], "ours", raw[i, k], per)
sys.exit(1 if bad else 0)
