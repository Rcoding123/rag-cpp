#!/usr/bin/env python3
"""OF24: train the starting policy of the live sim agent and write it as a text weight file.

  python experiments/of24_export.py [--out build-avx512/of24/of24_weights.txt] [--seed 0]

Trains OF23's RL-free arm (skip + 12 exits; of23_mine.rl_train_test, unchanged and frozen) on every used day of
the OF23 cache (build-avx512/of23; run of23_mine.py --run first) and writes the policy, the value network and the
feature standardization in the format of include/rag/live/of24_kernel.hpp (Policy::load). The 6 live book inputs
get zero weights. The file goes outside the repository. docs/ai/OF24_LIVE_SIM_PLAN.md section 2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402

BOOK = 6


def train(C, seed):
    """rl_train_test on all rows; the two networks it builds are captured as they are constructed."""
    import torch
    made = []
    real = torch.nn.Sequential

    class Capture(real):
        def __init__(self, *a):
            super().__init__(*a)
            made.append(self)
    torch.nn.Sequential = Capture
    try:
        every = np.ones(len(C["day"]), bool)
        M.rl_train_test(C, "free", every, ~every, seed)
    finally:
        torch.nn.Sequential = real
    assert len(made) == 2 and made[0][-1].out_features == len(M.MENU_E) + 1 and made[1][-1].out_features == 1
    return made[0], made[1]


def write(path, C, pol, val):
    X = C["X"]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    F, H, A = X.shape[1] + BOOK, M.RL["hidden"], len(M.MENU_E) + 1
    lines = ["OF24W 1", f"in {F} hidden {H} actions {A} updates 0"]

    def put(name, a):
        a = np.asarray(a, np.float64).ravel()
        lines.append(f"{name} {a.size}")
        lines.append(" ".join(repr(float(x)) for x in a))
    put("mu", mu)
    put("sd", sd)
    for p, net in (("pol", pol), ("val", val)):
        lin = [m for m in net if hasattr(m, "weight")]
        w1 = lin[0].weight.detach().numpy()
        put(f"{p}_W1", np.concatenate([w1, np.zeros((w1.shape[0], BOOK))], 1))
        put(f"{p}_b1", lin[0].bias.detach().numpy())
        put(f"{p}_W2", lin[1].weight.detach().numpy())
        put(f"{p}_b2", lin[1].bias.detach().numpy())
        # OF23's critic predicts the rest-of-day return of an episode; the live study learns from one decision at a
        # time, so its baseline starts at zero (the value head's output layer) and learns the one-step mean online
        zero = p == "val"
        put(f"{p}_W3", lin[2].weight.detach().numpy() * (0.0 if zero else 1.0))
        put(f"{p}_b3", lin[2].bias.detach().numpy() * (0.0 if zero else 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return mu, sd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "build-avx512" / "of24" / "of24_weights.txt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    C, meta = M.load_cache()
    pol, val = train(C, a.seed)
    mu, sd = write(Path(a.out), C, pol, val)
    Z = np.nan_to_num((C["X"] - mu) / sd).astype(np.float32)
    with torch.no_grad():
        acts = pol(torch.from_numpy(Z)).argmax(1).numpy()
    share = np.bincount(acts, minlength=len(M.MENU_E) + 1) / len(acts)
    print(f"trained on {len(meta['days'])} days, {len(acts)} candidates, seed {a.seed}; greedy action shares: "
          + ", ".join(f"{'skip' if k == 0 else M.exit_name(M.MENU_E[k - 1])} {100 * s:.0f}%" for k, s in enumerate(share) if s >= 0.005))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    from ragrl.cli import hard_exit
    hard_exit(main())
