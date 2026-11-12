"""The privileged channel: hidden state only in a separate array, only when
asked for, only on train and viz; never in the observation."""
import numpy as np
import pytest


def test_priv_columns(R):
    assert R.PRIV_COLUMNS == ("fair_value", "mid", "gap", "session_s", "fill_price", "orders", "cash_ticks")
    assert R.OBS_DIM == 16 and R.N_ACTIONS == 3 and R.N_FEATURES == 13


@pytest.mark.parametrize("block", ["validation", "pairing"])
def test_refused_on_selection_blocks(R, block):
    p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
    with pytest.raises(RuntimeError, match="privileged"):
        R.VecScalpEnv(p, R.EpisodeSource.block(block, 0, 2), privileged=True)


def test_refused_on_sealed_blocks(run_fresh):
    out = run_fresh("""
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        for b in ["test", "stress", "days", "confirm"]:
            R.open_sealed(b, "pytest")
        srcs = [R.EpisodeSource.block("test", 0, 1), R.EpisodeSource.stress("nominal", 0, 1), R.EpisodeSource.days(0, 1, 1),
                R.EpisodeSource.block("confirm", 0, 1, orchestrator="c")]
        n = 0
        for s in srcs:
            try:
                R.VecScalpEnv(p, s, privileged=True); print("ALLOWED", s)
            except RuntimeError:
                n += 1
        print("REFUSED", n)
    """)
    assert "ALLOWED" not in out and "REFUSED 4" in out


def test_priv_needs_the_flag_and_obs_do_not_depend_on_it(R):
    p = R.ScalpParams("discovery_v4", 1050.0, episode_seconds=30.0)
    plain = R.VecScalpEnv(p, R.EpisodeSource.block("viz", 10, 2), threads=2)
    priv = R.VecScalpEnv(p, R.EpisodeSource.block("viz", 10, 2), threads=2, privileged=True)
    assert not plain.privileged and priv.privileged
    o1, o2 = plain.reset(), priv.reset()
    with pytest.raises(RuntimeError):
        plain.priv
    with pytest.raises(RuntimeError):
        plain.final_priv
    assert o1.tobytes() == o2.tobytes()
    rng = np.random.default_rng(0)
    while True:
        a = rng.integers(0, 3, size=2).astype(np.int64)
        (o1, r1, d1), (o2, r2, d2) = plain.step(a), priv.step(a)
        assert o1.tobytes() == o2.tobytes() and r1.tobytes() == r2.tobytes()
        pv = priv.final_priv if d2[0] else priv.priv
        assert pv.shape == (2, 7)
        assert np.array_equal(pv[:, 2], pv[:, 0] - pv[:, 1])
        if d1[0]:
            break
    # gap is not a column of obs (no feature equals it)
    assert not any(np.array_equal(o2[:, k], pv[:, 2]) for k in range(16))


def test_train_priv_fill_and_orders(R):
    p = R.ScalpParams("default", 150.0, episode_seconds=60.0)
    env = R.VecScalpEnv(p, R.EpisodeSource.train(9), n_envs=4, threads=2, privileged=True)
    env.reset()
    pv = env.priv
    assert np.all(pv[:, 5] == 0) and np.all(pv[:, 6] == 0) and np.all(np.isnan(pv[:, 4]))
    fills = 0
    for t in range(60):
        a = np.full(4, 2 if t % 10 < 5 else 0, dtype=np.int64)  # long, then short: one order each switch
        o, r, d = env.step(a)
        pv = env.final_priv if d[0] else env.priv
        fills += int(np.isfinite(pv[:, 4]).sum())
        assert np.all(pv[:, 3] >= 150.0)  # session second, after the warm-up
    assert fills >= 4 * 11
    assert np.all(env.episode_stats["orders"] >= 12)
