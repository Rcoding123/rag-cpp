"""Statistics for per-episode results: mean, SE, t, Sharpe, paired differences
with a paired percentile bootstrap (fixed seed), and the multiplicity bar."""
import math

import numpy as np

# 39 ten-minute windows per RTH day, 252 days a year.
EPISODES_PER_YEAR = 39 * 252


def summary(res: dict) -> dict:
    x = np.asarray(res["net_pnl_ticks"], dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    sharpe = mean / sd if sd > 0 else float("nan")
    return {
        "n": n, "mean": mean, "sd": sd, "se": se, "t": mean / se if se and se > 0 else float("nan"),
        "win": float((x > 0).mean()), "orders": float(np.mean(res["orders"])), "fees": float(np.mean(res["fees_ticks"])),
        "sharpe_episode": sharpe, "sharpe_annual": sharpe * math.sqrt(EPISODES_PER_YEAR) if sd > 0 else float("nan"),
    }


def bootstrap_indices(n: int, b: int, seed: int) -> np.ndarray:
    """B x n resampling indices from a fixed seed: the same draws for every comparison."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def paired(a: dict, b: dict, idx: np.ndarray | None = None) -> dict:
    """a - b per episode (same episodes): mean, SE, t, and a 95% percentile bootstrap CI."""
    d = np.asarray(a["net_pnl_ticks"], dtype=np.float64) - np.asarray(b["net_pnl_ticks"], dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    out = {"mean": mean, "se": se, "t": mean / se if se and se > 0 else float("nan")}
    if idx is not None:
        boots = d[idx].mean(axis=1)
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def gross(res: dict) -> dict:
    """Per-episode P&L before fees (net + fees), in the per-episode result format (for paired())."""
    net = np.asarray(res["net_pnl_ticks"], dtype=np.float64)
    fees = np.asarray(res["fees_ticks"], dtype=np.float64)
    return {"net_pnl_ticks": net + fees, "orders": res["orders"], "fees_ticks": np.zeros_like(fees)}


def t_required(m: int) -> float:
    """Multiplicity-adjusted bar t_req = 2 + sqrt(2 ln M) for M validation choices (M = 1 -> 2)."""
    return 2.0 + math.sqrt(2.0 * math.log(max(1, m)))


def mean_of_seeds(results: list[dict]) -> dict:
    """Per-episode mean over seeds' results (the 'mean of seeds' headline, as of_train prints)."""
    return {k: np.mean([r[k] for r in results], axis=0) for k in ("net_pnl_ticks", "orders", "fees_ticks")}
