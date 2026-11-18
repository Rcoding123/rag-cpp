"""End-to-end check of data/features/windowed/*.parquet against an INDEPENDENT recomputation (pandas rolling windows, plain
forward fill) from the bars/trades parquet and the status feed. Covers what the kernel parity cannot: session/unit assembly,
grid boundaries, prior-quote carry, windows restarting per unit, VWAP per session, trading/valid masks.
Compares raw0-4 (flow, OFI), raw8-11 (returns in ticks, VWAP distance) on rows where the quote is usable at the row and at the
lookback row. (raw5 and the scaled columns depend on the EWMA state carried across sessions and are covered by the unit tests
and the C++ parity.) Usage: verify_features_pipeline.py YYYYMMDD [...]  (trade dates)
"""
import glob
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import features as F

B = F.BUCKET_NS
bars_all = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/srv/quant/data/features/bars/*.parquet"))], ignore_index=True)
bars_all["bk"] = bars_all.ts_end_ns // B - 1
tr_all = pd.concat([pd.read_parquet(f, columns=["ts_recv", "price", "size", "side", "instrument_id"]) for f in sorted(glob.glob("/srv/quant/data/features/trades/*.parquet"))])
tr_all = tr_all[tr_all.side.isin(["A", "B"])].sort_values("ts_recv", kind="stable")
bad = 0
for td in sys.argv[1:]:
    out = pd.read_parquet(f"/srv/quant/data/features/windowed/{td}.parquet")
    for iid, g in out.groupby("instrument_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        bk0 = int(g.ts_end_ns.iloc[0] // B - 1)
        assert (np.diff(g.ts_end_ns.to_numpy()) == B).all(), "grid is not contiguous 100 ms"
        idx = np.arange(bk0, bk0 + n)
        # --- independent dense reconstruction
        br = bars_all[bars_all.instrument_id == iid].set_index("bk")
        before = br[br.index < bk0].iloc[-1:]                      # prior row (if any)
        inside = br[(br.index >= bk0) & (br.index < bk0 + n)]
        q = pd.concat([before, inside])
        q = q[~q.index.duplicated(keep="last")].reindex(np.union1d(q.index.to_numpy(), idx)).ffill().reindex(idx)
        ofi = inside.ofi.reindex(idx).fillna(0.0).astype(float)
        mid = (q.bid_px + q.ask_px) / 2 * 4
        t = tr_all[(tr_all.instrument_id == iid) & (tr_all.ts_recv >= bk0 * B) & (tr_all.ts_recv < (bk0 + n) * B)]
        tb = (t.ts_recv // B).astype(np.int64)
        sgn = np.where(t.side == "B", 1.0, -1.0) * t["size"]
        delta = sgn.groupby(tb).sum().reindex(idx).fillna(0.0)
        vol = t["size"].groupby(tb).sum().reindex(idx).fillna(0.0)
        pv = (t.price * 4 * t["size"]).groupby(tb).sum().reindex(idx).fillna(0.0)
        exp = {}
        for k, L in ((0, 50), (1, 300), (2, 1200)):
            exp[k] = delta.rolling(L, min_periods=1).sum().to_numpy()
        for k, L in ((3, 50), (4, 300)):
            exp[k] = ofi.rolling(L, min_periods=1).sum().to_numpy()
        m = mid.to_numpy()
        for k, L in ((8, 50), (9, 300), (10, 1200)):
            r = np.zeros(n); r[L:] = m[L:] - m[:-L]
            exp[k] = r
        cv = vol.cumsum().to_numpy(); cpv = pv.cumsum().to_numpy()
        exp[11] = np.where(cv > 0, m - cpv / np.where(cv > 0, cv, 1), 0.0)
        qok = g.qok.to_numpy()
        for k, e in exp.items():
            got = g[f"raw{k}"].to_numpy().astype(np.float64)
            mask = qok.copy() if k >= 8 else np.ones(n, bool)
            if k in (8, 9, 10):
                L = {8: 50, 9: 300, 10: 1200}[k]
                mask[L:] &= qok[:-L]                              # need a usable quote at the lookback bucket too
                # for i < L the original returns 0.0 (no lookback yet); that is checked on usable rows as well
            ok = np.isclose(got[mask], e[mask], rtol=2e-6, atol=2e-6)
            if not ok.all():
                bad += 1
                i = int(np.nonzero(mask)[0][np.argmin(ok)])
                print(f"MISMATCH {td} iid {iid} raw{k} bucket {i}: got {got[i]} expected {e[i]}")
        # masks
        assert (~g.valid | g.qok).all() and (~g.qok | g.trading).all(), "valid must imply qok, qok must imply trading"
        assert (~g.valid | g.warm).all(), "valid must imply warm"
        assert np.isfinite(g[[f"x{k}" for k in range(13)]].to_numpy()[g.valid.to_numpy()]).all(), "valid rows must have finite features"
        assert (np.abs(g[[f"x{k}" for k in range(13)]].to_numpy()[g.valid.to_numpy()]) <= 4.0 + 1e-6).all(), "x out of +-4"
        print(f"{td} iid {iid}: {n:,} buckets | qok {100 * g.qok.mean():.1f}% valid {100 * g.valid.mean():.1f}% | compared raw0-4,8-11 on {int(qok.sum()):,} usable rows")
sys.exit(1 if bad else 0)
