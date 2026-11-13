"""OF24 (docs/ai/OF24_LIVE_SIM_PLAN.md): the C++ live kernel gives OF23's candidates, features and policy outputs.
Synthetic bars only."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402
from test_of23 import day_bars  # noqa: E402

RAGOF_DIR = Path(os.environ.get("RAGOF_DIR", REPO / "build-avx512" / "Release"))
EXE = RAGOF_DIR / "of24_parity.exe"
F_IN, H, A = len(M.FEATS) + 6, 64, 13


def need_exe():
    if not EXE.exists():
        if os.environ.get("RAGOF_REQUIRED", "") not in ("", "0"):
            pytest.fail(f"{EXE} is missing (build the of24_parity target)")
        pytest.skip("of24_parity is not built")


def write_bars(path, prev, b, dow):
    rows = []
    for d in ([prev] if prev is not None else []) + [b]:
        rows += [" ".join(repr(float(d[k][i])) for k in ("open", "high", "low", "close", "volume", "buy", "sell")) for i in range(len(d["open"]))]
    path.write_text(f"{0 if prev is None else len(prev['open'])} {len(b['open'])} {dow}\n" + "\n".join(rows) + "\n", encoding="ascii")


def run(*args):
    p = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = {"L": [], "C": [], "P": [], "U": []}
    for line in p.stdout.splitlines():
        t = line.split()
        out[t[0]].append([float(x) for x in t[1:]])
    return out


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol * (1 + np.abs(b)))))


def exhaustion(b, m=200):
    b["high"][m] = b["high"][m - 29:m].max() + 40
    b["low"][m] = b["close"][m - 1] - 2
    b["close"][m] = b["low"][m] + 3
    b["volume"][m] = b["volume"][:m].max() * 2
    b["sell"][m] = b["volume"][m] - b["buy"][m]
    return b


def test_candidates_and_features_match_of23(tmp_path):
    need_exe()
    seen = set()
    for seed in range(6):
        b = day_bars(seed)
        if seed == 5:
            b = exhaustion(b)
        prev = day_bars(100 + seed) if seed % 3 else None
        lv = M.prior_levels(prev) if prev is not None else None
        py = M.candidates(b, lv, seed % 5)
        f = tmp_path / f"bars{seed}.txt"
        write_bars(f, prev, b, seed % 5)
        out = run(f)
        if lv is not None:
            assert close(out["L"][0], [lv[k] for k in M.LEVELS])
        assert [int(c[0]) for c in out["C"]] == [c["m"] for c in py], seed
        for cc, pc in zip(out["C"], py):
            assert int(cc[1]) == int(pc["dir"])
            got, want = cc[2:], [pc[k] for k in M.FEATS]
            assert close(got, want), (seed, pc["m"], [(k, g, w) for k, g, w in zip(M.FEATS, got, want) if not close([g], [w])])
            seen |= {k for k in M.FEATS[:8] if pc[k]}
    assert {"c1_1", "c1_5", "c1_15", "c2", "c3", "c4", "c5", "c6"} <= seen, seen


def write_weights(path, rng):
    arrs = {"mu": rng.normal(size=37), "sd": rng.uniform(0.5, 2.0, 37)}
    for p, out in (("pol", A), ("val", 1)):
        arrs[f"{p}_W1"], arrs[f"{p}_b1"] = rng.normal(0, 0.2, (H, F_IN)), rng.normal(0, 0.1, H)
        arrs[f"{p}_W2"], arrs[f"{p}_b2"] = rng.normal(0, 0.2, (H, H)), rng.normal(0, 0.1, H)
        arrs[f"{p}_W3"], arrs[f"{p}_b3"] = rng.normal(0, 0.2, (out, H)), rng.normal(0, 0.1, out)
    lines = ["OF24W 1", f"in {F_IN} hidden {H} actions {A} updates 0"]
    for k, a in arrs.items():
        lines += [f"{k} {a.size}", " ".join(repr(float(x)) for x in a.ravel())]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return arrs


def fwd(w, p, x):
    h1 = np.tanh(w[f"{p}_W1"] @ x + w[f"{p}_b1"])
    h2 = np.tanh(w[f"{p}_W2"] @ h1 + w[f"{p}_b2"])
    return w[f"{p}_W3"] @ h2 + w[f"{p}_b3"]


def test_policy_forward_and_update_match_torch(tmp_path):
    need_exe()
    import torch
    rng = np.random.default_rng(1)
    b = day_bars(2)
    write_bars(tmp_path / "bars.txt", None, b, 1)
    w = write_weights(tmp_path / "w.txt", rng)
    out = run(tmp_path / "bars.txt", tmp_path / "w.txt")
    py = M.candidates(b, None, 1)
    assert len(out["P"]) == len(py) > 10
    for k, pc in enumerate(py[:25]):
        f = np.array([pc[n] for n in M.FEATS])
        x = np.r_[np.nan_to_num((f - w["mu"]) / w["sd"]), np.zeros(6)]
        assert close(out["P"][k], np.r_[fwd(w, "pol", x), fwd(w, "val", x)], 1e-9)
        # one actor-critic step (lr 1e-4, entropy 0.01, clip 1 per network) with torch autograd
        T = {n: torch.tensor(a, dtype=torch.float64, requires_grad=True) for n, a in w.items() if n not in ("mu", "sd")}
        xt = torch.tensor(x)

        def net(p):
            return T[f"{p}_W3"] @ torch.tanh(T[f"{p}_W2"] @ torch.tanh(T[f"{p}_W1"] @ xt + T[f"{p}_b1"]) + T[f"{p}_b2"]) + T[f"{p}_b3"]
        logits, v = net("pol"), net("val")[0]
        lp = torch.log_softmax(logits, 0)
        adv = (0.5 - v).detach()
        lossp = -adv * lp[3] + 0.01 * (lp.exp() * lp).sum()
        lossv = 0.5 * (v - 0.5) ** 2
        w2 = dict(w)
        for p, loss in (("pol", lossp), ("val", lossv)):
            names = [f"{p}_{s}" for s in ("W1", "b1", "W2", "b2", "W3", "b3")]
            g = torch.autograd.grad(loss, [T[n] for n in names])
            norm = float(torch.sqrt(sum((x_ ** 2).sum() for x_ in g)))
            k_ = 1e-4 * (1.0 / norm if norm > 1.0 else 1.0)
            for n, gi in zip(names, g):
                w2[n] = w[n] - k_ * gi.numpy()
        assert close(out["U"][k], np.r_[fwd(w2, "pol", x), fwd(w2, "val", x)], 1e-9), k


def test_exit_menu_matches_of23():
    txt = (REPO / "include" / "rag" / "live" / "of24_kernel.hpp").read_text(encoding="utf-8")
    menu = txt.split("kMenu = {{")[1].split("}};")[0]
    import re
    got = re.findall(r"\{(\d+), (\d+), (\d+), (true|false)\}", menu)
    want = [(str(T), str(S), "0" if H_ is None else str(H_), "true" if tr else "false") for (T, S, H_, tr) in (M.EXITS[e] for e in M.MENU_E)]
    assert got == want
