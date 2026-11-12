"""The seed ledger as the binding enforces it: block names and ranges, sealed
blocks, the confirm claim, and the process roles (a trainer never opens a
sealed block; a scorer never trains or selects; forbid_sealed is final).
Role tests run in fresh processes (conftest.fresh)."""
import pytest

from conftest import REPO


def test_blocks_table(R):
    b = {x["name"]: x for x in R.BLOCKS}
    assert list(b) == ["train", "validation", "test", "stress", "days", "pairing", "confirm", "viz", "real"]
    assert (b["validation"]["base"], b["test"]["base"], b["stress"]["base"], b["days"]["base"]) == (700_000_000, 900_000_000, 920_000_000, 930_000_000)
    assert (b["pairing"]["base"], b["confirm"]["base"], b["viz"]["base"]) == (940_000_000, 950_000_000, 960_000_000)
    assert b["test"]["base"] + b["test"]["size"] == 910_000_000  # the tick test (910M+) is not exposed
    assert {n for n, x in b.items() if x["sealed"]} == {"test", "stress", "days", "confirm", "real"}
    assert {n for n, x in b.items() if x["privileged_ok"]} == {"train", "viz"}
    assert b["confirm"]["needs_claim"] and not b["test"]["needs_claim"]


def test_source_refusals(R):
    with pytest.raises(ValueError):
        R.EpisodeSource.train(0)
    with pytest.raises(ValueError):
        R.EpisodeSource.train(700_000_000)
    with pytest.raises(TypeError):
        R.EpisodeSource.train(-1)
    R.EpisodeSource.train(699_999_999)
    for name, first, count in [("validation", 199_999_999, 2), ("validation", 0, 0), ("test", 9_999_999, 2), ("pairing", 10_000_000, 1)]:
        with pytest.raises(ValueError):
            R.EpisodeSource.block(name, first, count)
    for name in ["train", "stress", "days", "real", "tick_test", "confirm"]:
        with pytest.raises(ValueError):  # not indexed, unknown, or confirm without a claim id
            R.EpisodeSource.block(name, 0, 1)
    with pytest.raises(ValueError):
        R.EpisodeSource.block("validation", 0, 1, orchestrator="claim")
    for f in [lambda: R.EpisodeSource.block("test", 0, 1), lambda: R.EpisodeSource.block("confirm", 0, 1, orchestrator="c1"),
              lambda: R.EpisodeSource.days(0, 1), lambda: R.EpisodeSource.stress("nominal", 0, 1)]:
        with pytest.raises(RuntimeError, match="sealed"):
            f()
    with pytest.raises(ValueError):
        R.open_sealed("validation", "not sealed")
    with pytest.raises(ValueError):
        R.open_sealed("nope", "unknown")
    s = R.EpisodeSource.block("validation", 5, 3)
    assert s.describe() == {"kind": "indexed", "block": "validation", "first": 5, "count": 3}
    assert R.EpisodeSource.train(4).describe() == {"kind": "train", "block": "train", "run_seed": 4}


def test_trainer_cannot_open_sealed(run_fresh):
    out = run_fresh("""
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        assert R.process_role() == "none"
        R.VecScalpEnv(p, R.EpisodeSource.block("viz", 0, 1))
        assert R.process_role() == "none"          # viz is neutral
        R.VecScalpEnv(p, R.EpisodeSource.train(1), n_envs=2)
        assert R.process_role() == "trainer"
        for b in ["test", "stress", "days", "confirm", "real"]:
            try:
                R.open_sealed(b, "should fail"); print("OPENED", b)
            except RuntimeError as e:
                assert "fresh process" in str(e)
        print("TRAINER_OK", R.sealed_openings())
    """)
    assert "OPENED" not in out and "TRAINER_OK []" in out


def test_selection_also_makes_a_trainer(run_fresh):
    out = run_fresh("""
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        R.run_builtin(p, R.EpisodeSource.block("pairing", 0, 2), "flat", threads=2)
        assert R.process_role() == "trainer"
        try:
            R.open_sealed("test", "x"); print("OPENED")
        except RuntimeError:
            print("REFUSED")
    """)
    assert out.strip() == "REFUSED"


def test_scorer_cannot_train_or_select(run_fresh):
    out = run_fresh("""
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        R.open_sealed("test", "pytest scorer")
        assert R.process_role() == "scorer"
        test = R.EpisodeSource.block("test", 0, 3)
        r = R.run_builtin(p, test, ("oracle", 0.5), threads=3)
        env = R.VecScalpEnv(p, test, threads=3)
        env.reset()
        R.VecScalpEnv(p, R.EpisodeSource.block("viz", 0, 1))
        refused = []
        for name, f in [("train", lambda: R.VecScalpEnv(p, R.EpisodeSource.train(1))),
                        ("validation", lambda: R.VecScalpEnv(p, R.EpisodeSource.block("validation", 0, 1))),
                        ("pairing", lambda: R.run_builtin(p, R.EpisodeSource.block("pairing", 0, 1), "flat")),
                        ("privileged test", lambda: R.VecScalpEnv(p, test, privileged=True))]:
            try:
                f(); print("ALLOWED", name)
            except RuntimeError:
                refused.append(name)
        print("SCORER_OK", len(refused), R.sealed_openings())
    """)
    assert "ALLOWED" not in out
    assert "SCORER_OK 4 [('test', 'pytest scorer')]" in out


def test_forbid_sealed_is_final(run_fresh):
    out = run_fresh("""
        R.open_sealed("test", "opened first")
        src = R.EpisodeSource.block("test", 0, 1)
        R.forbid_sealed()
        assert R.sealed_forbidden() and not R.is_open("test")
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        n = 0
        for f in [lambda: R.open_sealed("test", "again"), lambda: R.open_sealed("real", "x"),
                  lambda: R.VecScalpEnv(p, src), lambda: R.run_builtin(p, src, "flat")]:
            try:
                f(); print("ALLOWED")
            except RuntimeError:
                n += 1
        print("FORBID_OK", n)
    """)
    assert "ALLOWED" not in out and "FORBID_OK 4" in out


def test_confirm_needs_open_and_claim(run_fresh):
    out = run_fresh("""
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        R.open_sealed("confirm", "orchestrator claim c-1")
        try:
            R.EpisodeSource.block("confirm", 0, 2); print("NO CLAIM ALLOWED")
        except ValueError:
            pass
        s = R.EpisodeSource.block("confirm", 0, 2, orchestrator="c-1")
        assert s.describe()["orchestrator"] == "c-1"
        print("CONFIRM_OK", R.run_builtin(p, s, "flat")["net_pnl_ticks"].tolist())
    """)
    assert "NO CLAIM" not in out and "CONFIRM_OK [0.0, 0.0]" in out


def test_days_and_stress_sources(run_fresh):
    out = run_fresh("""
        R.open_sealed("days", "pytest"); R.open_sealed("stress", "pytest")
        p = R.ScalpParams("default", 150.0, episode_seconds=10.0)
        d = R.EpisodeSource.days(3, 2, sessions_per_day=4)
        env = R.VecScalpEnv(p, d, threads=4)
        env.reset()
        print("DAYS", env.n_envs, env.episode_ids.tolist())
        s = R.EpisodeSource.stress("fees $3/side", 0, 2)
        print("STRESS", R.run_builtin(p, s, "flat")["net_pnl_ticks"].tolist())
        v4 = R.ScalpParams("discovery_v4", 1050.0, episode_seconds=10.0)
        try:
            R.run_builtin(v4, s, "flat"); print("V4 NO LIST ALLOWED")
        except ValueError as e:
            print("V4 refused")
        R.run_builtin(v4, R.EpisodeSource.stress("nominal", 0, 1, list="discovery"), "flat")
        try:
            R.EpisodeSource.days(0, 1, sessions_per_day=40); print("40 ALLOWED")
        except ValueError:
            pass
    """)
    assert "DAYS 8 [300, 301, 302, 303, 400, 401, 402, 403]" in out
    assert "STRESS [0.0, 0.0]" in out and "V4 refused" in out and "ALLOWED" not in out


def test_open_sealed_only_in_ledger_module():
    """Only ragrl/ledger.py may call open_sealed (the scoring entry points import it)."""
    pkg = REPO / "python" / "ragrl"
    if not pkg.exists():
        pytest.skip("python/ragrl not present yet")
    bad = [str(f) for f in pkg.rglob("*.py") if f.name != "ledger.py" and "open_sealed" in f.read_text(encoding="utf-8")]
    assert not bad, bad
