"""End-to-end smoke per agent (marked `smoke`; one ctest each: ragrl_smoke_<agent>).

Default market, warm-up 150, tiny budgets (experiments/configs/smoke/): train 2
seeds in worker processes, select on validation, 'seal' as a DRY RUN on the
visualization block (smoke runs never open the sealed test), red flags, report;
then retrain seed 1 and require a bit-identical champion (R8), and bake the
live-viewer JSONL with tools/of_train_live.py."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import RAGOF_DIR, REPO

SMOKE = REPO / "experiments" / "configs" / "smoke"
AGENTS = ["ppo", "ppo_framestack", "recurrent_ppo", "dqn", "qrdqn", "lgbm_threshold", "bc_dagger", "neuralq_ref"]


def _ragrl(tmp, *args, timeout=600):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), RAGOF_DIR=str(RAGOF_DIR), RAGRL_ROOT=str(tmp),
               RAGRL_LEDGER_FILE=str(tmp / "ledger_usage.jsonl"))
    p = subprocess.run([sys.executable, "-m", "ragrl", *args], env=env, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    assert p.returncode == 0, f"ragrl {args} failed:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}"
    return p.stdout


@pytest.mark.smoke
@pytest.mark.parametrize("agent", AGENTS)
def test_smoke(agent, tmp_path):
    t0 = time.perf_counter()
    _ragrl(tmp_path, "run", str(SMOKE / f"{agent}.json"))
    runs = list((tmp_path / "runs").iterdir())
    runs = [r for r in runs if r.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    status = (run / "STATUS").read_text().strip()
    assert status in ("DRY_RUN_OK", "DRY_RUN_RED_FLAGS"), status  # a dry run never claims
    sealed = json.loads((run / "sealed.json").read_text())
    assert sealed["dry_run"] and sealed["block"] == "viz" and sealed["source"]["block"] == "viz"
    assert all(len(v) for v in sealed["summary"].values())
    usage = [json.loads(l) for l in (tmp_path / "ledger_usage.jsonl").read_text().splitlines()]
    assert [u["block"] for u in usage] == ["viz"] and usage[0]["dry_run"]
    sel = json.loads((run / "select.json").read_text())
    assert sel["determinism"]["same_twice"] and sel["determinism"]["same_as_worker"]
    assert "report" in _ragrl(tmp_path, "report", str(run)).lower() or (run / "report.md").exists()
    det = (run / "report_results.md").read_text(encoding="utf-8")
    assert "train s" not in det and "## Provenance" not in det and str(run.name) not in det
    # a second seal is refused (write-once)
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), RAGOF_DIR=str(RAGOF_DIR), RAGRL_ROOT=str(tmp_path),
               RAGRL_LEDGER_FILE=str(tmp_path / "ledger_usage.jsonl"))
    p = subprocess.run([sys.executable, "-m", "ragrl", "seal", str(run)], env=env, capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "already sealed" in p.stderr
    t_run = time.perf_counter() - t0
    rc = json.loads(_ragrl(tmp_path, "recheck", str(run), "--seed", "1"))
    assert rc["same_actions"] and rc["same_val"], rc
    if agent != "neuralq_ref":
        assert rc["same_artifacts"], rc
    if (run / "live.jsonl").exists():
        out = subprocess.run([sys.executable, str(REPO / "tools" / "of_train_live.py"), "bake", str(run / "live.jsonl"), str(tmp_path / "live.html")],
                             capture_output=True, text=True, check=True).stdout
        assert "checkpoints" in out and (tmp_path / "live.html").stat().st_size > 10_000
        lines = [json.loads(l) for l in (run / "live.jsonl").read_text().splitlines()]
        assert lines[0]["type"] == "header" and lines[-1]["type"] == "done"
        ck = [l for l in lines if l["type"] == "ckpt"]
        assert len(ck) >= (1 if agent == "neuralq_ref" else 2) and all(len(c["policy"]) == 3 * 10 * 8 * 3 and len(c["track"]["mid"]) == 600 for c in ck)
    print(f"\nSMOKE {agent}: run+seal {t_run:.1f} s, total {time.perf_counter() - t0:.1f} s")


@pytest.mark.smoke
def test_smoke_hpo(tmp_path):
    t0 = time.perf_counter()
    _ragrl(tmp_path, "hpo", str(SMOKE / "ppo_hpo.json"))
    run = next(r for r in (tmp_path / "runs").iterdir() if r.is_dir())
    trials = [json.loads(l) for l in (run / "hpo.jsonl").read_text().splitlines()]
    assert len(trials) == 3 and any(t["state"] == "complete" for t in trials)
    best = json.loads((run / "best_config.json").read_text())
    assert best["train"]["seeds"] == [101, 102] and best["select"]["multiplicity_extra"] == 3 and best["hpo"] is None
    assert not (tmp_path / "ledger_usage.jsonl").exists()  # HPO never seals
    print(f"\nSMOKE hpo: {time.perf_counter() - t0:.1f} s")
