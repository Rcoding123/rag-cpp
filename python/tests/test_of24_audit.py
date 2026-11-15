"""OF24 audit (docs/ai/OF24_AUDIT.md): the checks catch what they are for. Synthetic data only."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402
import of24_audit as A  # noqa: E402
import test_of24 as T  # noqa: E402

DIR = M.FEATS.index("dir")


def cache(n_days=6, per_day=30, seed=0):
    """A small cache: candidates every 10 minutes, random features and exit outcomes."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    X = rng.normal(size=(n, len(M.FEATS)))
    X[:, :8] = rng.random((n, 8)) < 0.3
    X[:, DIR] = rng.choice([-1, 1], n)
    X[:, M.FEATS.index("atr")] = rng.uniform(4, 30, n)
    C = {"X": X, "day": np.repeat(np.arange(n_days), per_day), "m": np.tile(20 + 10 * np.arange(per_day), n_days),
         "days": np.array([f"2026-01-{d + 5:02d}" for d in range(n_days)]),
         "net": rng.normal(0, 10, (n, M.NE)).astype(np.float32),
         "xmin": (np.tile(20 + 10 * np.arange(per_day), n_days)[:, None] + rng.integers(1, 40, (n, M.NE)))}
    return C


def log_of(C, rows, shift=0):
    flags = [M.FEATS.index(k) for k in ("c1_1", "c1_5", "c1_15", "c2", "c3", "c4", "c5", "c6")]
    dec = [(str(C["days"][C["day"][i]]), int(C["m"][i]) + shift, int(C["X"][i, DIR]), 0, 0.5, 0.0,
            "".join(str(int(C["X"][i, c] > 0)) for c in flags), [0.0] * 6) for i in rows]
    return {"log": ({"updates": "0"}, dec)}


def test_policy_matches_the_kernel_formula(tmp_path):
    rng = np.random.default_rng(3)
    w = T.write_weights(tmp_path / "w.txt", rng)
    W = A.load_weights(tmp_path / "w.txt")
    X = rng.normal(size=(5, 37))
    X[1, 4] = np.nan
    book = rng.normal(size=(5, 6))
    P, V = A.policy(W, X, book)
    for i in range(5):
        x = np.r_[np.nan_to_num((X[i] - w["mu"]) / w["sd"]), book[i]]
        y = T.fwd(w, "pol", x)
        want = np.exp(y - y.max()) / np.exp(y - y.max()).sum()
        assert np.allclose(P[i], want, atol=1e-12) and abs(V[i] - T.fwd(w, "val", x)[0]) < 1e-12


def test_ledger_and_exit_names(tmp_path):
    names = ["T8/S4/30", "T8/S4/30/trail", "T32/S12/EOD", "T48/S16/EOD/trail"]
    assert [A.exit_index(n) for n in names] == [M.MENU_E[0], M.MENU_E[1], M.MENU_E[8], M.MENU_E[11]]
    assert A.exit_index("T9/S5/30") == -1  # a scaled exit is not one of OF23's
    f = tmp_path / "profit_trades_replay.csv"
    f.write_text("date,entry_ct,side,exit_rule,entry_px,exit_px,move_ticks,exit,net_ticks,net_usd,day_net_ticks,cum_ticks,cum_trades,cum_usd,updates\n"
                 "2026-01-05,08:55,short,T48/S16/EOD,6901.75,6889.75,48.00,target,47.24,590.50,47.24,47.24,1,590.50,3\n", encoding="ascii")
    (d, m, direction, e, net, px, kind), = A.read_ledger(f)
    assert (d, m, direction, e, net, px, kind) == ("2026-01-05", 24, -1, M.MENU_E[10], 47.24, 6901.75, "target")


def test_candidate_check_catches_a_decision_made_a_bar_early():
    C = cache()
    key = A.index_cache(C)
    rows = np.arange(0, len(C["m"]), 3)
    assert A.check_candidates(C, key, log_of(C, rows), lambda s: None) == 1.0
    # the same decisions logged one bar before the pipeline could know them: none is a cached candidate
    assert A.check_candidates(C, key, log_of(C, rows, shift=-1), lambda s: None) == 0.0


def test_entry_check_catches_a_fill_outside_the_next_bar(tmp_path):
    n = 390
    mid = 28000.0 + 40.0 * np.arange(n)   # each bar far from its neighbours: only the right bar can contain a fill
    np.savez(tmp_path / "2026-01-05.npz", open=mid, high=mid + 4, low=mid - 4)
    tick = lambda m, off: (mid[m + 1] + off) * 0.25  # noqa: E731  (a price off ticks from the entry bar's open)
    ok = [("2026-01-05", 24, 1, 0, 0.0, tick(24, 3), "stop"), ("2026-01-05", 40, -1, 0, 0.0, tick(40, -5), "stop")]
    assert A.check_entries(ok, tmp_path, lambda s: None) == 1.0   # inside, and one tick outside (the quote)
    early = [("2026-01-05", 60, 1, 0, 0.0, (mid[60] + 1) * 0.25, "stop")]    # a fill in the decision bar itself
    late = [("2026-01-05", 60, 1, 0, 0.0, (mid[62] + 1) * 0.25, "stop")]     # a fill one bar too late
    assert A.check_entries(early, tmp_path, lambda s: None) == 0.0
    assert A.check_entries(late, tmp_path, lambda s: None) == 0.0
    assert abs(A.check_entries(ok + early, tmp_path, lambda s: None) - 2 / 3) < 1e-12
    # a ledger dated in the forward block opens no bar file, even if one exists
    np.savez(tmp_path / "2026-11-12.npz", open=mid, high=mid + 4, low=mid - 4)
    assert np.isnan(A.check_entries([("2026-11-12", 24, 1, 0, 0.0, tick(24, 0), "stop")], tmp_path, lambda s: None))


def test_sampled_walk_holds_one_position_and_reads_the_sampled_exit():
    day, m = np.zeros(3, int), np.array([20, 30, 40])
    net12 = np.array([[5.0], [7.0], [11.0]]) + np.arange(12)[None, :] * 100.0   # exit a - 1 pays 100 * (a - 1) more
    xmin12 = np.tile(np.array([[35], [45], [60]]), (1, 12))   # the first trade is still open at minute 30
    xmin12[0, 3] = 25                                          # ... unless it took exit 4, which is flat by 25
    U = np.full((3, 4), 0.5)

    def always(a):
        P = np.zeros((3, 13))
        P[:, a] = 1.0
        return A.sampled_walk(day, m, net12, xmin12, P, U)
    num, den = always(1)
    assert den.tolist() == [1.0, 0.0, 1.0] and num.tolist() == [5.0, 0.0, 11.0]
    num, den = always(3)   # action 3 is exit column 2
    assert den.tolist() == [1.0, 0.0, 1.0] and num.tolist() == [205.0, 0.0, 211.0]
    num, den = always(4)   # exit column 3 frees the position in time for the second candidate
    assert den.tolist() == [1.0, 1.0, 0.0] and num.tolist() == [305.0, 307.0, 0.0]
    assert always(0)[1].sum() == 0   # always skip
    # the inverse CDF: p(skip) 0.4, p(action 2) 0.6; a uniform below 0.4 skips, above takes action 2
    P = np.zeros((3, 13))
    P[:, 0], P[:, 2] = 0.4, 0.6
    num, den = A.sampled_walk(day, m, net12, xmin12, P, np.array([[0.39, 0.41]] * 3))
    assert den[0] == 0.5 and num[0] == 0.5 * 105.0


def test_first_decision_check_reproduces_and_catches_changed_features(tmp_path):
    C = cache()
    key = A.index_cache(C)
    rng = np.random.default_rng(7)
    T.write_weights(tmp_path / "w.txt", rng)
    W = A.load_weights(tmp_path / "w.txt")
    i = 40   # not on the cache's first day
    book = [0.1, -0.2, 0.0, 0.0, 0.0, 1.0]
    p, v = A.policy(W, C["X"][i:i + 1], np.array([book]))
    started = int(W["path"].stat().st_mtime) + 5
    name = f"replay-c1-{started}.csv"

    def run(prob, weights="replay-learned"):
        d = (str(C["days"][C["day"][i]]), int(C["m"][i]), int(C["X"][i, DIR]), 3, prob, float(v[0]), "0" * 8, book)
        return A.check_first_decision(C, key, {name: ({"updates": "0", "weights": weights}, [d])}, {"learned": W}, lambda s: None)
    assert run(float(p[0, 3])) < 1e-9
    assert run(float(p[0, 3]) + 0.02) > 0.019       # a decision the saved weights do not give
    assert np.isnan(run(float(p[0, 3]), "start"))   # a run from the starting file is not compared with a learned one
    old = {f"replay-c1-{started - 60}.csv": ({"updates": "0", "weights": "replay-learned"}, [])}
    assert np.isnan(A.check_first_decision(C, key, old, {"learned": W}, lambda s: None))


def test_lookahead_verdict_is_never_clean_when_nothing_ran():
    nan = float("nan")
    assert A.lookahead_verdict(nan, nan, nan).startswith("NOT CHECKED")
    assert A.lookahead_verdict(1.0, nan, nan).startswith("NOT CHECKED")
    assert A.lookahead_verdict(1.0, nan, 1.0) == "none found, first-decision check not available"
    assert A.lookahead_verdict(1.0, 1e-4, 0.999) == "none found"
    assert A.lookahead_verdict(0.9, 1e-4, 1.0) == A.lookahead_verdict(1.0, 0.01, 1.0) == A.lookahead_verdict(1.0, 0.0, 0.9) == "CHECK the A lines"
    C = cache()
    assert np.isnan(A.check_candidates(C, A.index_cache(C), {}, lambda s: None))


def test_unseen_days_are_exactly_the_days_the_holdout_did_not_train_on():
    days = np.array(["2026-09-10", "2026-09-11", "2026-09-14"])
    cut = "2026-09-11"
    trained = days <= cut            # experiments/of24_export.py --until
    assert (A.is_unseen(days, cut) ^ trained).all() and A.is_unseen(days, cut).tolist() == [False, False, True]


def test_block_ratio_and_forward_rule():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(10), 5)
    mean, lo, hi = A.block_ratio(np.full(50, 2.0), np.ones(50), days, 200, rng)
    assert mean == lo == hi == 2.0
    x = np.random.default_rng(1).normal(0, 5, 50)
    mean, lo, hi = A.block_ratio(x, np.ones(50), days, 500, rng)
    assert lo < mean < hi
    assert A.verdict(5, 500, 3.0, 0.0).startswith("too early") and A.verdict(30, 50, 3.0, 0.0).startswith("too early")
    assert A.verdict(30, 500, 0.4, 0.5) == "luck is not excluded"   # the haircut takes the lower bound below zero
    assert A.verdict(30, 500, 0.6, 0.5) == "skill is supported"


def test_shuffling_an_unused_input_changes_nothing(tmp_path):
    C = cache()
    rng = np.random.default_rng(5)
    T.write_weights(tmp_path / "w.txt", rng)
    W = A.load_weights(tmp_path / "w.txt")
    w1 = W["pol_W1"].reshape(W["hidden"], W["in"])
    for c in A.GROUPS["time"]:
        w1[:, M.FEATS.index(c)] = 0.0   # the policy cannot read the time inputs
    C["net"] = C["net"][:, :M.NE]
    U = rng.random((len(C["m"]), 4))
    res, _ = A.input_use(C, W, U, rng, lambda s: None, repeats=1)
    assert res["time"] == (0.0, 0.0) and res["momentum"][0] > 0.01


def test_scaled_exits_match_the_kernel(tmp_path):
    T.need_exe()
    rng = np.random.default_rng(1)
    b = T.day_bars(2)
    T.write_bars(tmp_path / "bars.txt", None, b, 1)
    py = M.candidates(b, None, 1)
    T.write_weights(tmp_path / "w.txt", rng)
    atr_i = M.FEATS.index("atr")
    typical = float(np.median([pc["atr"] for pc in py]))   # so that the factors spread inside the limits
    txt = (tmp_path / "w.txt").read_text(encoding="ascii").split("\n")
    assert txt[2] == "mu 37"
    mu = txt[3].split()
    mu[atr_i] = repr(typical)
    txt[3] = " ".join(mu)
    (tmp_path / "w.txt").write_text("\n".join(txt), encoding="ascii")
    out = T.run(tmp_path / "bars.txt", tmp_path / "w.txt")
    assert len(out["X"]) == len(py) > 10
    ks = []
    for row, pc in zip(out["X"], py):
        k = A.exit_scale(pc["atr"], typical)
        ks.append(k)
        assert abs(row[0] - k) < 1e-12
        assert [int(v) for v in row[1:]] == [v for e in A.scaled_menu(k) for v in e[:2]]
    inside = [k for k in ks if 0.5 < k < 2.0 and k != 1.0]
    assert len(set(inside)) >= 5, ks   # unclamped factors: the rounding of target and stop is exercised
    # the factor's limits and rounding
    assert A.exit_scale(10, 0) == 1.0 and A.exit_scale(float("nan"), 10) == 1.0
    assert A.exit_scale(1, 10) == 0.5 and A.exit_scale(100, 10) == 2.0
    assert A.scaled_menu(1.0) == [(Tg, S, H, tr) for (Tg, S, H) in M.MENU for tr in M.TRS]
    assert A.scaled_menu(0.5)[0][:2] == (4, 2) and A.scaled_menu(2.0)[-1][:2] == (96, 32)
    assert [e[:2] for e in A.scaled_menu(1.3)[::2]] == [(10, 5), (16, 8), (21, 10), (31, 10), (42, 16), (62, 21)]
    assert A.scaled_menu(1.0625)[0][:2] == (9, 4) and A.scaled_menu(1.0625)[6][:2] == (26, 9)   # halves round up
