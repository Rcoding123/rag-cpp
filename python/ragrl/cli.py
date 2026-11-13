"""python -m ragrl <command> ...   (PYTHONPATH=python; RAGOF_DIR = the directory holding _ragof*.pyd)

  run CFG [--no-seal] [--reuse-ok]   train K seeds (workers), select on validation, then seal in a fresh process
  train RUN --seed K           one worker (spawned by run)
  select RUN                   validation-only selection record (select.json)
  seal RUN [--allow-env-change] [--reuse-ok]   the only test opener: paired scoring once (sealed.json is write-once);
                               refuses a range viewed before (pre-viewed / other experiments) unless --reuse-ok
  report RUN                   rewrite report.md (and redflags) from the run's files; no environment
  recheck RUN [--seed K]       retrain one seed and compare bit for bit (red flag R8)
  hpo CFG                      Optuna on validation only -> best_config.json (never seals)
  live CFG --seed K --out DIR/live.jsonl   train one seed with the live hook (serve: python tools/of_train_live.py serve DIR)
  diag RUN                     hidden-state correlation probe (train block)
  zoo                          the agents and their defaults
  confirm RUN --i-am-the-orchestrator --claim ID   950M+ confirmation (orchestrator only, once per claim)
  score-real RUN --days test|dev [--latency-buckets N] [--fee-dollars F] [--stress] [--threads T]
                               OF16: the sealed run on recorded ES (VecReplayEnv), in a fresh process; test days
                               once per run + manifest (ledger real_test); dev is "dev: not a score".
                               Results in RUN/real/<days>-<stamp>/, never sealed.json
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ragrl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--no-seal", action="store_true")
    p.add_argument("--reuse-ok", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("run")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir")
    for name in ("select", "report", "diag"):
        sub.add_parser(name).add_argument("run")
    p = sub.add_parser("seal")
    p.add_argument("run")
    p.add_argument("--allow-env-change", action="store_true")
    p.add_argument("--reuse-ok", action="store_true")
    p = sub.add_parser("recheck")
    p.add_argument("run")
    p.add_argument("--seed", type=int)
    sub.add_parser("hpo").add_argument("config")
    p = sub.add_parser("live")
    p.add_argument("config")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    sub.add_parser("zoo")
    p = sub.add_parser("confirm")
    p.add_argument("run")
    p.add_argument("--claim", default="")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--i-am-the-orchestrator", dest="orchestrator", action="store_true")
    p = sub.add_parser("score-real")
    p.add_argument("run")
    p.add_argument("--days", choices=["test", "dev"], required=True)
    p.add_argument("--latency-buckets", type=int, default=1)
    p.add_argument("--fee-dollars", type=float, default=1.5)
    p.add_argument("--stress", action="store_true", help="also latency 3 buckets and $3 per side, in the same opening")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--manifest")
    p.add_argument("--allow-env-change", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "zoo":
        from .agents import ZOO
        for k, s in ZOO.items():
            print(f"{k:16s} n_envs {s.n_envs:3d}  {s.description}\n{'':16s} {json.dumps(s.defaults)}")
        return 0
    if a.cmd == "score-real":
        from . import score_real
        score_real.score(Path(a.run), a.days, a.latency_buckets, a.fee_dollars, a.stress, a.threads, manifest=a.manifest,
                         allow_env_change=a.allow_env_change)
        return 0
    from . import runner
    if a.cmd == "run":
        runner.run(a.config, a.no_seal, reuse_ok=a.reuse_ok)
    elif a.cmd == "train":
        from .runner import use_run_module
        from .worker import train_worker
        use_run_module(Path(a.run))
        train_worker(Path(a.run), a.seed, out_dir=a.out_dir)
        # A worker's results are complete and closed once train_worker returns (every file is opened and closed per
        # write; no Python thread is left). Under the full gate's load, workers intermittently died with 0xC0000005
        # AFTER their last record (metrics "timing") was written: in interpreter finalization, the native libraries'
        # teardown (torch, _ragof, their OpenMP runtimes), where faulthandler is already off (empty worker.log;
        # LESSONS 58). A worker has nothing to finalize, so it leaves like a multiprocessing child: os._exit after
        # flushing. A crash during the work itself still exits nonzero and still fails the run.
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(0)
    elif a.cmd == "select":
        from . import ledger
        from . import registry
        runner.use_run_module(Path(a.run))
        ledger.as_trainer()
        runner.setup_torch(registry.read_json(Path(a.run) / "config.json")["train"]["torch_threads"])
        runner.select(Path(a.run))
    elif a.cmd == "seal":
        runner.seal(Path(a.run), a.allow_env_change, a.reuse_ok)
    elif a.cmd == "report":
        from . import redflags, report
        run = Path(a.run)
        if (run / "sealed.json").exists():
            redflags.evaluate(run)
        report.write(run)
        print((run / "report.md").read_text(encoding="utf-8"))
    elif a.cmd == "recheck":
        print(json.dumps(runner.recheck(Path(a.run), a.seed), indent=1))
    elif a.cmd == "diag":
        from . import ledger, registry
        run = Path(a.run)
        runner.use_run_module(run)
        ledger.as_trainer()
        print(json.dumps(runner.diag(run, registry.read_json(run / "config.json")), indent=1))
    elif a.cmd == "hpo":
        from . import hpo
        hpo.hpo(a.config)
    elif a.cmd == "live":
        live(a.config, a.seed, a.out)
    elif a.cmd == "confirm":
        print(json.dumps(runner.confirm(Path(a.run), a.claim, a.orchestrator, a.count), indent=1))
    return 0


def live(cfg_path, seed, out):
    """Trains one seed with the live hook into a scratch run (validation-only; never seals)."""
    import copy

    from . import config, ledger, registry
    from .runner import use_run_module
    from .worker import train_worker
    cfg = config.load(cfg_path)
    c = copy.deepcopy(cfg)
    c["name"] = cfg["name"] + "_live"
    c["train"]["seeds"] = [int(seed)]
    c["live"]["enabled"] = True
    c["live"]["seed"] = int(seed)
    c = config.resolve(c)
    run = registry.new_run(c, config.sha256(c))
    use_run_module(run)
    ledger.as_trainer()
    config.dump(c, run / "config.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"live: {run} -> {out}  (serve: python tools/of_train_live.py serve {out.parent} --file {out.name})", flush=True)
    train_worker(run, int(seed), live_path=out)


if __name__ == "__main__":
    sys.exit(main())
