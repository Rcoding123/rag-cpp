"""OF19 simulator pieces on synthetic data: the sparse-table first-hit search against brute force, and the exits
(stop at the worst of the event's bids and trades, target only on a trade one tick through, time cap)."""
import importlib.util

import numpy as np

from conftest import REPO

spec = importlib.util.spec_from_file_location("of19", REPO / "experiments" / "of19_footprint.py")
of19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(of19)


def _T(rows):
    t, px, bid, ask = (np.array(c, dtype=float if i == 0 else np.int64) for i, c in enumerate(zip(*rows)))
    return {"t": t, "px": px, "bid": bid, "ask": ask, "size": np.ones(len(t), np.int64), "side": np.ones(len(t), np.int64)}


def test_first_hit_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 3000
    px = 1000 + np.cumsum(rng.integers(-1, 2, n))
    T = {"t": np.arange(n, dtype=float) * 0.5, "px": px, "bid": px - 1, "ask": px + 1, "size": np.ones(n, np.int64),
         "side": np.ones(n, np.int64)}
    E = of19.Events(T)
    starts = rng.integers(0, n, 400)
    xs = px[starts] + rng.integers(-15, 16, 400)
    for below in (True, False):
        got = E.first(starts, xs, below)
        for s, x, g in zip(starts, xs, got):
            hit = np.flatnonzero(px[s:] <= x if below else px[s:] >= x)
            assert g == (s + hit[0] if len(hit) else n)


def test_exits_stop_target_cap():
    # events at t = 0, 1, 2, ... ; a long filled at event 0's ask (101)
    rows = [(0.0, 100, 100, 101), (1.0, 102, 101, 102), (2.0, 104, 103, 104), (3.0, 105, 104, 105), (4.0, 99, 98, 99)]
    E = of19.Events(_T(rows))
    e0 = np.array([0])
    one = np.array([1])
    # target 104: a trade at 104 is a touch, not through; the 105 trade at event 3 fills it
    g, _ = of19._exits(E, e0, one, np.array([101.0]), np.array([95.0]), np.array([104.0]), 100.0, False)
    assert g[0] == 3.0
    g, _ = of19._exits(E, e0, one, np.array([101.0]), np.array([95.0]), np.array([104.0]), 100.0, True)   # on touch: event 2
    assert g[0] == 3.0
    # stop 99: hit at event 4 (trade 99), exit at the worst of its bid (98) and trade (99) = 98
    g, _ = of19._exits(E, e0, one, np.array([101.0]), np.array([99.0]), np.array([np.nan]), 100.0, False)
    assert g[0] == 98 - 101
    # time cap 2.5 s: exit at event 3 (first at or after 2.5 s), at its worst bid / trade = 104
    g, _ = of19._exits(E, e0, one, np.array([101.0]), np.array([90.0]), np.array([np.nan]), 2.5, False)
    assert g[0] == 104 - 101
    # a short: fill 100 (event 0's bid), stop 103 hit at event 2 (trade 104 >= 103): buy back at the worst ask / trade = 104
    g, _ = of19._exits(E, e0, -one, np.array([100.0]), np.array([103.0]), np.array([np.nan]), 100.0, False)
    assert g[0] == 100 - 104


def test_sweep_is_one_event():
    # Sierra stamps one exchange message's records 1 us apart: they are one event, filled at its worst price
    rows = [(10.0, 100, 99, 100), (10.000001, 101, 100, 101), (10.000002, 102, 101, 102), (11.0, 102, 101, 102)]
    E = of19.Events(_T(rows))
    assert E.n == 2 and E.long_fill[0] == 102 and E.short_fill[0] == 99 and E.px_min[0] == 100
