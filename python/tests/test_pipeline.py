"""The ragrl pipeline (fast): configs, stats, frozen policies vs SB3's own predict,
the teacher vs the C++ oracle, the live-viewer schema vs of_train_live.exe,
red flags, write-once scoring and the confirm refusals."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import RAGOF_DIR, REPO

sys.path.insert(0, str(REPO / "python"))
CONFIGS = REPO / "experiments" / "configs"


def _raw(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_configs_resolve_and_round_trip():
    from ragrl import config
    files = sorted(CONFIGS.glob("*.json")) + sorted((CONFIGS / "smoke").glob("*.json"))
    assert len(files) >= 16
    for f in files:
        c = config.load(f)
        again = config.resolve(json.loads(json.dumps(c)))
        assert again == c and config.sha256(again) == config.sha256(c), f
        assert c["train"]["env_steps"] >= c["train"]["budget_steps"]
        assert c["train"]["episodes"] % c["train"]["n_envs"] == 0
    # of_train-sized defaults use exactly of_train's 2000 episodes (whole rounds)
    for f in CONFIGS.glob("of15_*_v4.json"):
        c = config.load(f)
        if c["agent"]["kind"] != "recurrent_ppo" or c["train"]["budget_steps"] == 1_200_000:
            assert c["train"]["episodes"] == c["train"]["budget_steps"] // 600, f


def test_config_refusals():
    from ragrl import config
    base = _raw("smoke/ppo.json")
    bad = []
    c = copy.deepcopy(base); c["train"]["typo"] = 1; bad.append(c)
    c = copy.deepcopy(base); c["agent"]["params"]["nope"] = 1; bad.append(c)
    c = copy.deepcopy(base); c["train"]["seeds"] = [700_000_000]; bad.append(c)
    c = copy.deepcopy(base); c["train"]["seeds"] = [1, 1]; bad.append(c)
    c = copy.deepcopy(base); c["train"]["workers"] = 99; bad.append(c)
    c = copy.deepcopy(base); c["val_select"] = {"first": 0, "count": 5}; bad.append(c)          # overlaps val_ckpt
    c = copy.deepcopy(base); c["score"]["block"] = "confirm"; bad.append(c)
    c = copy.deepcopy(base); c["score"]["headline"] = "best_seed"; bad.append(c)
    c = copy.deepcopy(base); c["agent"]["kind"] = "a3c"; bad.append(c)
    c = copy.deepcopy(base); c["schema"] = "ragrl/0"; bad.append(c)
    c = copy.deepcopy(base); c["hpo"] = {"space": {"not_a_param": ["log", 1e-5, 1e-3]}}; bad.append(c)
    c = copy.deepcopy(base); c["hpo"] = {"select": dict(base["val_select"])}; bad.append(c)     # hpo.select overlaps val_select
    c = copy.deepcopy(base); c["train"]["budget_steps"] = 1_200_000; bad.append(c)            # viz scoring beyond a smoke budget
    c = copy.deepcopy(base); c["score"]["block"] = "test"; del c["score"]["first"]; bad.append(c)  # sealed range not pre-registered
    c = copy.deepcopy(base); c["env"] = {"decide_seconds": 0.0}; bad.append(c)
    c = copy.deepcopy(base); c["env"] = {"fee_dollars": 2.0}
    c["score"]["baselines"] = [{"neuralq_dir": "build-avx512/rl/of14"}]; bad.append(c)        # of_train nets off their conditions
    c = _raw("smoke/neuralq_ref.json"); c["select"]["val_ckpt"]["first"] = 0; bad.append(c)   # inside of_train's [0, 40)
    for c in bad:
        with pytest.raises(config.ConfigError):
            config.resolve(c)


def test_stats():
    from ragrl import stats
    a = {"net_pnl_ticks": np.array([1.0, 2.0, 3.0, 4.0]), "orders": np.ones(4), "fees_ticks": np.zeros(4)}
    b = {"net_pnl_ticks": np.array([0.0, 2.0, 2.0, 4.0]), "orders": np.ones(4), "fees_ticks": np.zeros(4)}
    s = stats.summary(a)
    assert s["mean"] == 2.5 and abs(s["se"] - np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-15 and s["win"] == 1.0
    assert abs(s["sharpe_annual"] - s["sharpe_episode"] * np.sqrt(39 * 252)) < 1e-9
    idx = stats.bootstrap_indices(4, 500, 0)
    assert np.array_equal(idx, stats.bootstrap_indices(4, 500, 0))
    d = stats.paired(a, b, idx)
    assert d["mean"] == 0.5 and d["ci95"][0] <= 0.5 <= d["ci95"][1]
    assert stats.t_required(1) == 2.0 and stats.t_required(100) > 4.9


@pytest.fixture(scope="module")
def P(R):
    from ragrl import config
    return config.make_params(config.DEFAULTS["env"])


LAST_OBS64 = [None]


class Spy:
    """Wraps a VecScalpEnv and remembers the float64 observations it returned."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, k):
        return getattr(self.env, k)

    def reset(self):
        o = self.env.reset()
        LAST_OBS64[0] = o
        return o

    def step(self, a):
        o, r, d = self.env.step(a)
        LAST_OBS64[0] = o
        return o, r, d


@pytest.mark.parametrize("kind", ["ppo", "ppo_framestack", "recurrent_ppo", "dqn", "qrdqn"])
def test_frozen_policy_matches_sb3_predict(R, P, kind, tmp_path):
    """The frozen copy (and its save/load round trip) acts exactly like SB3's deterministic predict, including
    the frame stack and the LSTM state; DQN with polyak_eval=0 (the averaged copy is checked separately)."""
    from stable_baselines3.common.vec_env import VecFrameStack

    from ragrl import envs, policies
    from ragrl.agents import ZOO, sb3_agents
    p = dict(ZOO[kind].defaults)
    venv = envs.RagVecEnv(Spy(envs.train_env(P, 3, 4)))
    n_stack = p.get("n_stack", 1)
    env = VecFrameStack(venv, n_stack) if n_stack > 1 else venv
    if kind in ("ppo", "ppo_framestack"):
        from stable_baselines3 import PPO
        kw, pk = sb3_agents._ppo_kwargs(dict(p, n_steps=50, batch_size=100))
        model = PPO("MlpPolicy", env, policy_kwargs=pk, seed=1, device="cpu", **kw)
        model.learn(400)
        frozen = policies.SB3ActorCritic(sb3_agents._frozen_copy(model.policy), n_stack)
    elif kind == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        kw, pk = sb3_agents._ppo_kwargs(dict(p, n_steps=50, batch_size=100))
        pk.update(lstm_hidden_size=16, n_lstm_layers=1)
        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=pk, seed=1, device="cpu", **kw)
        model.learn(400)
        frozen = policies.SB3Recurrent(sb3_agents._frozen_copy(model.policy))
    else:
        DQNc, QRc = sb3_agents._dqn_classes()
        kw = sb3_agents._dqn_kwargs(dict(p, learning_starts=100, buffer_size=5000, exploration_fraction=0.1))
        if kind == "dqn":
            model = DQNc("MlpPolicy", env, double_q=True, polyak_eval=0.0, policy_kwargs={"net_arch": [32, 32]}, seed=1, device="cpu", **kw)
        else:
            model = QRc("MlpPolicy", env, polyak_eval=0.0, policy_kwargs={"net_arch": [32, 32], "n_quantiles": 8}, seed=1, device="cpu", **kw)
        model.learn(800)
        frozen = model.frozen()
    policies.save_policy(frozen, tmp_path)
    loaded = policies.load_policy(tmp_path)
    for pol in (frozen, loaded):
        # a fresh stream for each comparison (the model's env is consumed): run seed 4
        venv2 = envs.RagVecEnv(Spy(envs.train_env(P, 4, 4)))
        env2 = VecFrameStack(venv2, n_stack) if n_stack > 1 else venv2
        obs = env2.reset()
        pol.reset(4)
        state, start = None, np.ones(4, dtype=bool)
        for _ in range(120):
            a, state = model.predict(obs, state=state, episode_start=start, deterministic=True)
            b, s = pol.step(LAST_OBS64[0])
            assert np.array_equal(a, b)
            assert s.shape == (4, 3)
            obs, _, done, _ = env2.step(a)
            start = done


def test_dqn_polyak_average(R, P):
    """polyak_eval > 0: the frozen DQN uses the averaged weights, avg += tau (net - avg) after every update."""
    from ragrl import envs
    from ragrl.agents import ZOO, sb3_agents
    DQNc, _ = sb3_agents._dqn_classes()
    p = dict(ZOO["dqn"].defaults)
    env = envs.RagVecEnv(envs.train_env(P, 5, 2))
    kw = sb3_agents._dqn_kwargs(dict(p, learning_starts=50, buffer_size=2000, gradient_steps=1))
    m = DQNc("MlpPolicy", env, double_q=True, polyak_eval=0.5, policy_kwargs={"net_arch": [8]}, seed=1, device="cpu", **kw)
    import torch
    w0 = [x.clone() for x in m.q_net_avg.parameters()]
    m.learn(52)  # 26 vec steps x 2 envs: exactly one update after learning_starts
    net = list(m.q_net.parameters())
    for a, b, c in zip(m.q_net_avg.parameters(), w0, net):
        assert torch.allclose(a, 0.5 * b + 0.5 * c)
    fz = m.frozen()
    for a, b in zip(fz.policy.q_net.parameters(), m.q_net_avg.parameters()):
        assert torch.equal(a, b)


def test_teacher_is_the_cpp_oracle(R, P):
    """The imitation teacher (privileged gap column) plays exactly like the C++ oracle (viz block, privileged allowed)."""
    from ragrl import envs
    for th in (0.5, 1.0):
        env = R.VecScalpEnv(P, envs.indexed_source("viz", 10, 6), threads=2, privileged=True)
        obs = env.reset()
        while True:
            obs, _, done = env.step(envs.oracle_actions(obs, env.priv, th))
            if done[0]:
                break
        ref = R.run_builtin(P, envs.indexed_source("viz", 10, 6), ("oracle", th), threads=2)
        assert np.array_equal(env.episode_stats["net_pnl_ticks"], ref["net_pnl_ticks"])


def test_delta_rule_is_the_cpp_rule(R, P):
    from ragrl import envs
    from ragrl.evaluate import Builtin, run_policy
    from ragrl.policies import FrozenPolicy

    class D(FrozenPolicy):
        def step(self, obs):
            return envs.delta_rule_actions(obs), np.zeros((len(obs), 3))

    src = envs.indexed_source("viz", 20, 5)
    assert np.array_equal(run_policy(P, src, D())["net_pnl_ticks"], run_policy(P, src, Builtin("delta30"))["net_pnl_ticks"])


def test_live_header_matches_of_train_live(R, P, tmp_path):
    """Probe states, policy-map bins (obs[6] / 2), the rule's probe track: identical to of_train_live.exe's header."""
    exe = RAGOF_DIR / "of_train_live.exe"
    if not exe.exists():
        pytest.skip("of_train_live.exe not next to the module")
    subprocess.run([str(exe), "--episodes", "1", "--probe", "3", "--out", str(tmp_path / "ref.jsonl")], check=True, capture_output=True)
    ref = json.loads((tmp_path / "ref.jsonl").read_text().splitlines()[0])
    from ragrl.live import LiveWriter
    LiveWriter(tmp_path / "mine.jsonl", P, {"live": {"probe": 3, "every_episodes": 1}, "agent": {"kind": "x"}}, 1, 1)
    mine = json.loads((tmp_path / "mine.jsonl").read_text().splitlines()[0])
    assert set(ref) <= set(mine)
    assert mine["cell_states"] == ref["cell_states"] and mine["rule_pnl"] == ref["rule_pnl"]
    for k in ["tod", "mid", "fv", "pos", "pnl", "fill"]:
        assert mine["rule"][k] == ref["rule"][k], k


def test_registry_redflags_and_report(tmp_path, monkeypatch):
    """Synthetic sealed/selection records -> the expected flags and STATUS; report renders without an environment."""
    monkeypatch.setenv("RAGRL_ROOT", str(tmp_path))
    monkeypatch.setenv("RAGRL_LEDGER_FILE", str(tmp_path / "ledger_usage.jsonl"))
    from ragrl import redflags, registry, report

    def S(mean, se, orders=4.0, sh=0.1):
        return {"n": 300, "mean": mean, "sd": se * 17.3, "se": se, "t": mean / se, "win": 0.5, "orders": orders, "fees": 0.5,
                "sharpe_episode": sh, "sharpe_annual": sh * 99.1}

    run = tmp_path / "runs" / "x-1"
    (run / "seeds" / "seed1").mkdir(parents=True)
    registry.write_json(run / "config.json", {"name": "x", "agent": {"kind": "ppo", "params": {}}, "train": {
        "seeds": [1], "episodes": 8, "env_steps": 4800, "budget_rounded": False, "n_envs": 4},
        "env": {"market": "m", "warmup_seconds": 1, "latency_steps": 1, "fee_dollars": 1.5, "contracts": 1},
        "select": {"val_ckpt": {"first": 0, "count": 2}, "every_steps": 1}, "val_select": {"first": 9, "count": 2},
        "score": {"headline": "ensemble"}})
    registry.write_json(run / "provenance.json", {"git_sha": "g", "dirty": False, "dirty_diff_sha1": "", "module": "m",
                                                  "module_sha256": "s", "freeze_sha256": "f", "versions": {}, "python": "3"})
    (run / "config.sha256").write_text("c")
    registry.write_json(run / "seeds" / "seed1" / "val.json", {"val_ckpt_mean": 1.0, "offers": 3, "champion_step": 2,
                                                               "train_seconds": 1, "eval_seconds": 1, "extra": {}})
    good = {"seed1": S(1.5, 0.5, sh=0.06), "ensemble": S(1.2, 0.5), "mean_of_seeds": S(1.5, 0.5),
            "flat": S(0.0, 1e-9), "delta30": S(-4.7, 1.4), "oracle_0.5": S(-1.0, 0.3, orders=1.0)}
    paired = {"flat": {k: {"mean": good[k]["mean"], "se": 0.5, "t": good[k]["mean"] / 0.5, "ci95": [0, 1]} for k in ["seed1", "ensemble", "mean_of_seeds"]},
              "oracle_0.5": {k: {"mean": 2.5, "se": 0.5, "t": 5.0, "ci95": [1, 3]} for k in ["seed1", "ensemble", "mean_of_seeds"]}}
    sel = {"summary": {**good, "seed1": S(5.0, 1.0)}, "paired": {}, "best_oracle": "oracle_0.5", "multiplicity": 10,
           "determinism": {"seed": 1, "same_twice": True, "same_as_worker": True}, "val_ckpt": {}}
    registry.write_json(run / "select.json", sel)
    registry.write_json(run / "sealed.json", {"block": "test", "dry_run": False, "first": 0, "count": 300, "headline": "ensemble",
                                              "summary": good, "paired": paired, "learned": ["seed1", "ensemble", "mean_of_seeds"],
                                              "baselines": ["flat", "delta30", "oracle_0.5"], "bootstrap": {"B": 10, "seed": 0}})
    f = redflags.evaluate(run)
    ids = {x["id"] for x in f["hard"]}
    assert ids == {"R1", "R2"}, f   # seed1 Sharpe 0.06 * 99 > 5; beats the val-best oracle by 2 SE while > 0
    assert {x["id"] for x in f["claim"]} == {"R3"} and f["status"] == "RED_FLAGS"
    w = {x["id"] for x in f["warnings"]}
    assert {"R4", "R6", "R9", "R11"} <= w      # val 5.0 (t 5) -> test 1.5; oracle 1 order/episode; t 2.4 < t_req(10)
    assert "R1v" not in w
    # R1 on a near-flat row (< 2 orders/episode, the oracles' R6 rule) is vacuous: warning R1v, never hard;
    # a row trading >= 2 orders/episode still raises R1
    sealed = registry.read_json(run / "sealed.json")
    for seed1_orders, want_hard in ((0.1, []), (1.99, []), (2.0, ["seed1"])):
        lo = {k: dict(v, orders=0.1) for k, v in sealed["summary"].items()}
        lo["seed1"]["orders"] = seed1_orders
        (run / "sealed.json").unlink()
        registry.write_json(run / "sealed.json", dict(sealed, summary=lo))
        g = redflags.evaluate(run)
        assert [x["policy"] for x in g["hard"] if x["id"] == "R1"] == want_hard, g["hard"]
        vac = {x["policy"] for x in g["warnings"] if x["id"] == "R1v"}
        assert vac == {"ensemble", "mean_of_seeds", "flat", "delta30"} | ({"seed1"} if not want_hard else set()), g["warnings"]
    (run / "sealed.json").unlink()
    registry.write_json(run / "sealed.json", sealed)
    # a second test opening of the same run in the usage ledger: R10
    from ragrl import ledger
    for _ in range(2):
        ledger.log_usage({"run": "x-1", "block": "test", "dry_run": False})
    assert "R10" in {x["id"] for x in redflags.evaluate(run)["hard"]}
    report.write(run)
    txt = (run / "report.md").read_text(encoding="utf-8")
    assert "STATUS: RED_FLAGS" in txt and "| ensemble | flat |" in txt and "train s" in txt and "## Provenance" in txt
    det = (run / "report_results.md").read_text(encoding="utf-8")
    assert "train s" not in det and "## Provenance" not in det and "| ensemble | flat |" in det
    with pytest.raises(FileExistsError):
        registry.write_json(run / "sealed.json", {}, once=True)

    # R2g: gross above the highest-gross oracle by 2 SE fires even with a negative net; family size raises M;
    # a dry run never claims; a sealed opening outside the canonical ledger is R10
    sealed = registry.read_json(run / "sealed.json")
    G = {k: {"mean": -0.5, "se": 0.5, "t": -1.0} for k in sealed["summary"]}
    G.update({"oracle_0.5": {"mean": 3.0, "se": 0.5, "t": 6.0}, "oracle_1": {"mean": 1.0, "se": 0.5, "t": 2.0}})
    pg = {o: {k: {"mean": 0.1, "se": 1.0, "t": 0.1} for k in sealed["learned"]} for o in ("oracle_0.5", "oracle_1")}
    pg["oracle_0.5"]["seed1"] = {"mean": 2.5, "se": 1.0, "t": 2.5}
    pg["oracle_1"]["ensemble"] = {"mean": 9.0, "se": 1.0, "t": 9.0}   # beats a lower-gross oracle only: no flag
    neg = {k: dict(v, mean=-abs(v["mean"]), t=-abs(v["t"])) for k, v in good.items()}
    (run / "sealed.json").unlink()
    registry.write_json(run / "sealed.json", dict(sealed, summary=neg, gross=G, paired_gross=pg, family_size=9,
                                                  ledger_canonical=False))
    f = redflags.evaluate(run)
    r2g = [x for x in f["hard"] if x["id"] == "R2g"]
    assert [x["policy"] for x in r2g] == ["seed1"], f["hard"]
    assert f["multiplicity"] == 90 and any(x["id"] == "R10" and "canonical" in x["detail"] for x in f["hard"])
    (run / "sealed.json").unlink()
    registry.write_json(run / "sealed.json", dict(sealed, dry_run=True, block="viz"))
    f = redflags.evaluate(run)
    assert f["status"] == "DRY_RUN_RED_FLAGS" and not f["claim"] and any(x["id"] == "R3" for x in f["warnings"])


def test_ledger_range_reuse_and_confirm_offsets(tmp_path, monkeypatch):
    """Pre-viewed ranges, other experiments and family members beyond family_size are reuse; a regeneration
    (same experiment id) is not; confirmations never share 950M+ episodes."""
    monkeypatch.setenv("RAGRL_LEDGER_FILE", str(tmp_path / "u.jsonl"))
    from ragrl import ledger
    assert not ledger.is_canonical()
    r = ledger.range_reuse("test", 0, 300, "discovery_v4", "me", "e0", "fam", 3)
    assert r["reused"] and r["previewed"]
    assert not ledger.range_reuse("test", 1_000_000, 300, "discovery_v4", "me", "e0", "fam", 3)["reused"]
    log = lambda run, exp, fam, first=1_000_000, market="discovery_v4", dry=False: ledger.log_usage(
        {"run": run, "block": "test", "first": first, "count": 300, "market": market, "experiment": exp, "family": fam, "dry_run": dry})
    log("a", "e1", "fam")
    log("b", "e2", "fam")
    log("c", "e0", "fam")                         # an earlier regeneration of this experiment
    log("d", "x", "other", market="default")      # another market
    log("e", "y", None, dry=True)                 # a dry run
    r = ledger.range_reuse("test", 1_000_100, 300, "discovery_v4", "me", "e0", "fam", 3)
    assert not r["reused"] and r["family_experiments"] == 2
    r = ledger.range_reuse("test", 1_000_100, 300, "discovery_v4", "me", "e0", "fam", 2)
    assert r["reused"] and r["family_over"] == 1
    log("f", "z", "other")
    r = ledger.range_reuse("test", 1_000_000, 300, "discovery_v4", "me", "e0", "fam", 3)
    assert r["reused"] and [u["run"] for u in r["outside_family"]] == ["f"]
    assert ledger.next_confirm_first() == 0
    ledger.log_usage({"run": "a", "block": "confirm", "first": 0, "count": 300, "claim": "c1", "dry_run": False})
    with open(ledger.usage_file(), "a", encoding="utf-8") as f:
        f.write('{"run": "partial", "block": "co')    # a line still being appended is ignored
    assert ledger.read_usage()[-1]["claim"] == "c1"
    assert ledger.next_confirm_first() == 300


def test_confirm_refuses_dry_runs_and_non_claims(tmp_path):
    from ragrl import registry, runner
    run = tmp_path / "r"
    run.mkdir()
    registry.write_json(run / "sealed.json", {"block": "viz", "dry_run": True})
    with pytest.raises(PermissionError, match="dry run"):
        runner.confirm(run, "c1", True)
    (run / "sealed.json").unlink()
    registry.write_json(run / "sealed.json", {"block": "test", "dry_run": False})
    (run / "STATUS").write_text("NO_EDGE\n")
    with pytest.raises(PermissionError, match="CLAIM_PENDING"):
        runner.confirm(run, "c1", True)


def test_steps_per_episode_matches_cpp(R):
    from ragrl import config
    for decide, episode in [(1.0, 600.0), (3.0, 10.0), (0.1, 1.0), (0.3, 0.9), (7.0, 600.0), (0.25, 600.0)]:
        env = dict(config.DEFAULTS["env"], decide_seconds=decide, episode_seconds=episode)
        assert config.steps_per_episode(env) == config.make_params(env).steps_per_episode, (decide, episode)


def test_scorer_ledger_entry_and_refusals(run_fresh, tmp_path):
    """as_scorer opens and logs; confirm without a claim / with a used claim is refused before anything opens."""
    out = run_fresh(f"""
        import os, json
        os.environ["RAGRL_ROOT"] = {str(tmp_path)!r}
        os.environ["RAGRL_LEDGER_FILE"] = {str(tmp_path / "ledger_usage.jsonl")!r}
        sys.path.insert(0, {str(REPO / 'python')!r})
        from ragrl import ledger
        try:
            ledger.as_scorer("confirm", "r", "run", 0, 1)
        except PermissionError as e:
            print("NOCLAIM")
        ledger.log_usage({{"run": "old", "block": "confirm", "claim": "c7"}})
        try:
            ledger.as_scorer("confirm", "r", "run", 0, 1, claim="c7")
        except PermissionError as e:
            print("REUSED")
        print("ROLE", R.process_role(), R.is_open("confirm"))
        ledger.as_scorer("test", "unit test (opens, plays nothing)", "run", 0, 1)
        print("ROLE", R.process_role(), R.is_open("test"))
        u = ledger.read_usage()
        print("LOGGED", u[-1]["block"], u[-1]["run"], u[-1]["dry_run"])
    """)
    assert "NOCLAIM" in out and "REUSED" in out and "ROLE none False" in out
    assert "ROLE scorer True" in out and "LOGGED test run False" in out


def test_trainer_process_cannot_seal(run_fresh):
    out = run_fresh(f"""
        sys.path.insert(0, {str(REPO / 'python')!r})
        from ragrl import ledger
        ledger.as_trainer()
        try:
            ledger.as_scorer("test", "x", "run", 0, 1)
            print("OPENED")
        except Exception as e:
            print("REFUSED", type(e).__name__)
    """)
    assert "REFUSED" in out and "OPENED" not in out


def test_confirm_cli_refusal(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), RAGOF_DIR=str(RAGOF_DIR), RAGRL_ROOT=str(tmp_path),
               RAGRL_LEDGER_FILE=str(tmp_path / "ledger_usage.jsonl"))
    p = subprocess.run([sys.executable, "-m", "ragrl", "confirm", str(tmp_path / "nope"), "--claim", "x"], env=env,
                       capture_output=True, text=True)
    assert p.returncode != 0 and "orchestrator only" in p.stderr
    assert not (tmp_path / "ledger_usage.jsonl").exists()
