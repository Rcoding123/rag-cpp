"""OF23 (docs/ai/OF23_SWING_MINING_PLAN.md): candidates, features, fills and the in-order walk on synthetic data.
No real market data is read here."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402


def day_bars(seed=0, n=M.N_MIN, drift=0.0, start=20000.0):
    """A synthetic RTH day of 1-min bars in ticks: a random walk with trends and volume."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 3.0, n) + 4.0 * np.sin(np.arange(n) / 25.0)
    c = np.round(start + np.cumsum(steps))
    o = np.r_[start, c[:-1]]
    h = np.maximum(o, c) + rng.integers(0, 4, n)
    l = np.minimum(o, c) - rng.integers(0, 4, n)
    v = rng.integers(200, 2000, n).astype(float)
    v[rng.random(n) < 0.03] *= 6
    buy = np.round(v * rng.uniform(0.3, 0.7, n))
    return {"open": o.astype(float), "high": h.astype(float), "low": l.astype(float), "close": c.astype(float),
            "volume": v, "buy": buy, "sell": v - buy}


def cut(b, n):
    return {k: a[:n].copy() for k, a in b.items()}


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


def test_no_lookahead_in_candidates_or_features():
    seen = set()
    for seed in range(4):
        b = day_bars(seed)
        lv = M.prior_levels(day_bars(100 + seed))
        full = M.candidates(b, lv, 2)
        assert len(full) > 20
        by_m = {c["m"]: c for c in full}
        for m in range(M.FIRST_M, M.LAST_M + 1, 3):
            tr = {c["m"]: c for c in M.candidates(cut(b, m + 1), lv, 2)}
            assert set(k for k in tr if k <= m) == set(k for k in by_m if k <= m), m
            if m in by_m:
                for f in M.FEATS:
                    assert same(tr[m][f], by_m[m][f]), (seed, m, f, tr[m][f], by_m[m][f])
        for c in full:
            seen |= {f for f in M.FEATS[:8] if c[f]}
    assert {"c1_1", "c1_5", "c1_15", "c2", "c4"} <= seen, seen


def flat_bars(price=1000.0, n=M.N_MIN):
    z = np.full(n, price)
    return {"open": z.copy(), "high": z.copy(), "low": z.copy(), "close": z.copy(), "volume": np.full(n, 100.0),
            "buy": np.full(n, 50.0), "sell": np.full(n, 50.0)}


def path(steps):
    """steps: [(minute, o, h, l)]"""
    a = np.array(steps, float)
    return {"minute": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3]}


def one(L, k=0):
    return float(L["move"][k]), int(L["kind"][k]), int(L["xmin"][k]), bool(L["both"][k])


def test_target_needs_a_trade_through():
    b = flat_bars()
    ex = [(8, 4, None, False)]
    L = M.labels(path([(51, 1000, 1008, 1000), (60, 1000, 1000, 1000)]), b, 50, 1, ex)
    assert one(L)[1] == M.K_TIME                     # touching the target is not a fill
    L = M.labels(path([(51, 1000, 1009, 1000)]), b, 50, 1, ex)
    assert one(L)[:3] == (8.0, M.K_TARGET, 51)
    L = M.labels(path([(51, 1000, 1000, 991)]), b, 50, -1, ex)     # short: mirror
    assert one(L)[:3] == (8.0, M.K_TARGET, 51)


def test_record_order_and_stop_first_in_one_step():
    b = flat_bars()
    ex = [(8, 4, None, False)]
    L = M.labels(path([(51, 1000, 1009, 1000), (51, 1000, 1000, 990)]), b, 50, 1, ex)
    assert one(L)[1] == M.K_TARGET                   # the target came first in the records
    L = M.labels(path([(51, 1000, 1000, 996), (51, 1000, 1009, 1000)]), b, 50, 1, ex)
    assert one(L)[:2] == (-5.0, M.K_STOP)            # stop one tick worse than the stop
    L = M.labels(path([(51, 1000, 1009, 996)]), b, 50, 1, ex)
    mv, k, x, both = one(L)
    assert (mv, k, both) == (-5.0, M.K_STOP, True)   # one step reached both: the stop counts first


def test_gap_through_the_stop_fills_at_the_open():
    b = flat_bars()
    L = M.labels(path([(51, 1000, 1001, 999), (52, 993, 994, 990)]), b, 50, 1, [(8, 4, None, False)])
    assert one(L)[:2] == (-8.0, M.K_STOP)            # opened 7 below: fill at the open, one tick worse


def test_time_cap_end_of_day_and_gap_bar():
    b = flat_bars()
    b["open"][81] = 1003.0
    b["close"][389] = 998.0
    L = M.labels(path([(51, 1000, 1002, 999)]), b, 50, 1, [(8, 4, 30, False), (8, 4, None, False)])
    assert one(L, 0)[:3] == (3.0, M.K_TIME, 81)      # open of bar j + 30
    assert one(L, 1)[:3] == (-2.0, M.K_TIME, 390)    # the close of the 14:59 bar
    g = flat_bars()
    g["volume"][70] = 0
    g["volume"][71] = 0
    g["open"][72] = 1001.0
    L = M.labels(path([(51, 1000, 1002, 999)]), g, 50, 1, [(8, 4, None, False)])
    assert one(L)[:3] == (1.0, M.K_TIME, 72)         # a gap bar: out at the next trading bar's open


def test_trailing_moves_from_the_next_bar_only():
    b = flat_bars()
    b["close"][51] = 1006.0                          # best close +6 >= S = 4 at the close of bar 51
    ex = [(48, 4, None, True)]
    # bar 51 itself dips to -3 (not the fixed -4 stop): no exit; from bar 52 the stop is at +2
    L = M.labels(path([(51, 1000, 1006, 997), (52, 1004, 1004, 1002)]), b, 50, 1, ex)
    assert one(L)[:3] == (1.0, M.K_STOP, 52)         # stop at +2, one tick worse
    L = M.labels(path([(51, 1000, 1006, 997), (52, 1004, 1005, 1003)]), b, 50, 1, [(48, 4, None, False), (48, 4, None, True)])
    assert one(L, 0)[1] == M.K_TIME and one(L, 1)[1] == M.K_TIME


def test_costs():
    mv = np.array([8.0, -5.0, 3.0])
    kd = np.array([M.K_TARGET, M.K_STOP, M.K_TIME], np.int8)
    assert np.allclose(M.net_of(mv, kd), [8 - 0.76, -5 - 1.28, 3 - 1.28])
    assert np.allclose(M.net_of(mv, kd, stress=True), [8 - 0.95, -6 - 1.5, 3 - 1.5])


def test_failed_extreme_level_fade_and_merge():
    b = flat_bars()
    b["high"][:40] = 1010.0
    b["high"][40], b["close"][40] = 1013.0, 1008.0   # new session high, closes back below: short (C2)
    c = {r["m"]: r for r in M.candidates(b, None, 0)}
    assert c[40]["c2"] == 1 and c[40]["dir"] == -1
    lv = {"pdh": 1005.0, "pdl": 900.0, "pdc": 950.0, "ppoc": 960.0, "pvah": 970.0, "pval": 940.0}
    b2 = flat_bars()
    b2["high"][45], b2["close"][45] = 1006.0, 1002.0  # through pdh by 1 tick from below, back below: short (C3)
    c = {r["m"]: r for r in M.candidates(b2, lv, 0)}
    assert c[45]["c3"] == 1 and c[45]["dir"] == -1 and c[45]["c3_name"] == 0 and c[45]["c3_count"] == 1
    b3 = flat_bars()
    b3["high"][:40] = 1010.0
    b3["low"][:40] = 990.0
    b3["high"][40], b3["low"][40], b3["close"][40] = 1013.0, 987.0, 1000.0  # failed high and failed low: both dropped
    assert 40 not in {r["m"] for r in M.candidates(b3, None, 0)}


def test_swing_ties_make_no_candidate():
    b = flat_bars()
    b["low"][30:35] = [995, 993, 990, 993, 995]
    assert any(r["m"] == 34 and r["c1_1"] == 1 and r["dir"] == 1 for r in M.candidates(b, None, 0))
    b["low"][30:35] = [995, 990, 990, 993, 995]
    assert not any(r["m"] in (33, 34) and r["c1_1"] for r in M.candidates(b, None, 0))


def test_walk_one_trade_at_a_time():
    day = np.array([0, 0, 0, 1, 1])
    m = np.array([20, 25, 40, 20, 30])
    take = np.ones((5, 2), bool)
    xmin = np.array([[30, 22], [50, 26], [60, 41], [25, 21], [35, 31]])
    net = np.array([[1, 2], [4, 8], [16, 32], [64, 128], [256, 512.0]])
    tot, cnt, taken = M.walk(day, m, take, xmin, net)
    assert taken[:, 0].tolist() == [True, False, True, True, True]   # 25 < 30: skipped while the trade is open
    assert taken[:, 1].tolist() == [True, True, True, True, True]
    assert tot.tolist() == [1 + 16 + 64 + 256, 2 + 8 + 32 + 128 + 512] and cnt.tolist() == [4, 5]


def test_choice_floor_and_ties():
    tot, cnt = np.array([10.0, 30.0, 30.0]), np.array([60.0, 10.0, 70.0])
    assert M.choose(tot, cnt, 100, [0, 1, 2]) == 2       # the floor removes 1
    assert M.choose(np.array([5.0, 5.0]), np.array([80.0, 90.0]), 100, [1, 0]) == 1  # tie: the key order
    assert M.choose(np.array([5.0, 9.0]), np.array([10.0, 20.0]), 100, [0, 1]) == 1  # none reaches the floor: most trades
    order = M.tie_order(True)
    assert len(order) == len(M.QS) * M.NE and M.QS[order[0][0]] == 0.10 and M.EXITS[order[0][1]] == (8, 4, 30, False)


def test_constants_match_the_plan():
    assert M.NE == 240 and len(M.MENU_E) == 12 and len(set(M.MENU_E)) == 12
    assert abs(M.T_REQ - 4.33) < 0.005
    assert M.TEST_MONTHS[0] == "2026-03" and M.TEST_MONTHS[-1] == "2026-11"


def test_refuses_the_confirm_block():
    with pytest.raises(RuntimeError):
        M.check_day("2026-11-12")


def test_runner_is_frozen():
    plan = (REPO / "docs" / "ai" / "OF23_SWING_MINING_PLAN.md").read_text(encoding="utf-8")
    mt = re.search(r"of23_mine\.py` sha256 `([0-9a-f]{64})`", plan)
    if mt is None:
        pytest.skip("the freeze amendment is not written yet")
    assert M.sha256_lf(REPO / "experiments" / "of23_mine.py") == mt.group(1)


def test_exhaustion_bar_is_causal():
    b = day_bars(7)
    m = 200
    b["high"][m] = b["high"][m - 29:m].max() + 40
    b["low"][m] = b["close"][m - 1] - 2
    b["close"][m] = b["low"][m] + 3
    b["volume"][m] = b["volume"][:m].max() * 2
    full = {c["m"]: c for c in M.candidates(b, None, 1)}
    assert full[m]["c6"] == 1 and full[m]["dir"] == -1
    tr = {c["m"]: c for c in M.candidates(cut(b, m + 1), None, 1)}
    for f in M.FEATS:
        assert same(tr[m][f], full[m][f]), f


def test_vwap_band_fade_and_fvg_retest():
    b = flat_bars()
    b["high"][100], b["close"][100] = 1030.0, 1001.0   # reaches far above vwap + 2 SD, closes inside: short (C4)
    b["volume"][100] = 100.0
    c = {r["m"]: r for r in M.candidates(b, None, 0)}
    assert c[100]["c4"] == 1 and c[100]["dir"] == -1
    g = flat_bars()
    # a bullish 5-min gap: 5-min bar 10 (minutes 50-54) high 1000, bar 12 (minutes 60-64) low 1010
    g["high"][50:55] = 1000.0
    for k in range(55, 65):
        g["open"][k] = g["close"][k] = g["low"][k] = 1012.0 if k >= 60 else 1006.0
        g["high"][k] = 1014.0 if k >= 60 else 1008.0
    g["low"][60:65] = 1010.0
    for k in range(65, 80):
        g["open"][k] = g["close"][k] = g["low"][k] = g["high"][k] = 1015.0
    g["low"][80], g["close"][80], g["high"][80] = 1008.0, 1011.0, 1015.0   # first trade into [1000, 1010]
    for k in range(81, 390):
        g["open"][k] = g["close"][k] = g["low"][k] = g["high"][k] = 1013.0
    c = {r["m"]: r for r in M.candidates(g, None, 0)}
    assert c[80]["c5"] == 1 and c[80]["dir"] == 1


def test_gap_at_the_time_cap_exits_at_the_next_trading_bar():
    b = flat_bars()
    b["volume"][81] = 0
    b["open"][82] = 1004.0
    L = M.labels(path([(51, 1000, 1002, 999)]), b, 50, 1, [(8, 4, 30, False)])
    assert one(L)[:3] == (4.0, M.K_TIME, 82)


def test_end_to_end_on_a_synthetic_cache(tmp_path, monkeypatch):
    """Workers and the report on a small synthetic cache: no crash after hours of compute."""
    rng = np.random.default_rng(3)
    days = [f"2026-{mo:02d}-{d:02d}" for mo in (1, 2, 3, 4) for d in range(5, 10)]
    rows = [(di, m) for di in range(len(days)) for m in sorted(rng.choice(np.arange(20, 360), 6, replace=False))]
    n = len(rows)
    X = rng.normal(size=(n, len(M.FEATS)))
    X[:, :8] = 0
    X[np.arange(n), rng.integers(0, 8, n)] = 1
    X[:, M.FEATS.index("dir")] = rng.choice([-1, 1], n)
    X[np.arange(n) % 7 == 0, 7] = 0          # C6 rare
    m_ = np.array([r[1] for r in rows])
    kind = rng.integers(0, 3, (n, M.NE)).astype(np.int8)
    xmin = np.minimum(m_[:, None] + rng.integers(2, 90, (n, M.NE)), 390).astype(np.int16)
    move = rng.normal(0, 8, (n, M.NE)).astype(np.float32)
    chk = np.flatnonzero(np.array([r[0] for r in rows]) >= 15)
    monkeypatch.setattr(M, "WORK", tmp_path)
    monkeypatch.setattr(M, "OUT", tmp_path / "results.txt")
    monkeypatch.setattr(M, "TEST_MONTHS", ["2026-03", "2026-04"])
    monkeypatch.setitem(M.RL, "iters", 2)
    np.savez(tmp_path / "cache.npz", X=X, day=np.array([r[0] for r in rows]), m=m_, move=move, kind=kind, xmin=xmin,
             both=rng.random((n, M.NE)) < 0.05, move_bar=move, kind_bar=kind, both_bar=rng.random((n, M.NE)) < 0.1,
             chk_row=chk, chk_move_sec=move[chk] + 0.1, chk_kind_sec=kind[chk])
    shape = (M.LAST_M - M.FIRST_M + 1, 2, M.NE)
    rz = {}
    for d in days[10:]:
        rz[f"{d}__move"] = rng.normal(0, 8, shape).astype(np.float32)
        rz[f"{d}__kind"] = rng.integers(0, 3, shape).astype(np.int8)
        rz[f"{d}__xmin"] = np.minimum(np.arange(M.FIRST_M, M.LAST_M + 1)[:, None, None] + 30, 390).repeat(2, 1).repeat(M.NE, 2).astype(np.int16)
    np.savez(tmp_path / "rand.npz", **rz)
    (tmp_path / "meta.json").write_text(__import__("json").dumps({"days": days, "dropped": [], "manifest_sha256": "x",
                                                                  "runner_sha256": "y", "n": n}), encoding="utf-8")
    for f in range(2):
        for p in M.PIPES:
            M.worker("gbm", [p, str(f), "1"])
        for arm in ("trade", "free"):
            for s_ in M.RL["seeds"]:
                M.worker("rl", [arm, str(f), str(s_)])
    for p in M.PIPES:
        M.worker("naive", [p])
    M.report(log=lambda s: None)
    txt = (tmp_path / "results.txt").read_text(encoding="utf-8")
    assert "PRIMARY:" in txt and "RL-trade" in txt and "deflated Sharpe" in txt and "resolution check" in txt
