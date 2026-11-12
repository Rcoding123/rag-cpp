"""The C++ NeuralQ reference (of_train): existing networks, or a fresh of_train run.

prepare (parent, before the workers): with nets_dir, check the networks exist;
otherwise run RUN/bin/of_train.exe once for seeds 1..max(seeds) (the OF3/OF14
recipe) into RUN/nets/. of_train always scores what it trains; here it scores 2
visualization episodes (--test 2 --test-seed 960000000: look-only, never read),
so the sealed test is not touched. Champion selection stays of_train's own
(700M+0..39, 20 evaluations: choices_per_seed = 20 in the multiplicity); the
worker loads the champion with check_net and offers it once, scored on val_ckpt,
which config.resolve keeps off [0, 40)."""
import subprocess
from pathlib import Path

from .. import native
from ..config import DEFAULTS, ConfigError, of_train_env_problems
from ..policies import NeuralQ


def nets_dir(cfg, run_dir: Path) -> Path:
    d = cfg["agent"]["params"]["nets_dir"]
    if d:
        d = Path(d)
        return d if d.is_absolute() else native.REPO / d
    return Path(run_dir) / "nets"


def prepare_neuralq_ref(cfg, run_dir):
    p = cfg["agent"]["params"]
    env = cfg["env"]
    for k in of_train_env_problems(env):
        raise ConfigError(f"neuralq_ref: of_train supports market and warmup only (env.{k} must stay {DEFAULTS['env'][k]!r})")
    d = nets_dir(cfg, run_dir)
    seeds = cfg["train"]["seeds"]
    if p["nets_dir"]:
        missing = [s for s in seeds if not (d / f"seed{s}.net").exists()]
        if missing:
            raise FileNotFoundError(f"{d}: no networks for seeds {missing}")
        return
    if not p["run_of_train"]:
        raise ConfigError("neuralq_ref: give nets_dir or run_of_train=true")
    exe = Path(run_dir) / "bin" / "of_train.exe"
    if not exe.exists():
        raise FileNotFoundError(f"{exe}: of_train is copied from RAGOF_DIR (snapshot) into the run's bin/")
    d.mkdir(parents=True, exist_ok=True)
    threads = min(max(seeds), cfg["train"]["workers"] * (cfg["train"]["env_threads"] + cfg["train"]["torch_threads"]))
    cmd = [str(exe), "--seeds", str(max(seeds)), "--threads", str(threads), "--episodes", str(cfg["train"]["episodes"]),
           "--market", env["market"], "--warmup", f"{env['warmup_seconds']:g}", "--hidden", str(p["hidden"]), "--gamma", f"{p['gamma']!r}",
           "--nstep", str(p["nstep"]), "--batch", str(p["batch"]), "--lr", f"{p['lr']!r}", "--test", "2", "--test-seed", "960000000",
           "--out", str(d)]
    with open(d / "of_train.log", "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def train_neuralq_ref(ctx):
    d = nets_dir(ctx.cfg, ctx.run_dir)
    net = NeuralQ.from_file(d / f"seed{ctx.seed}.net", ctx.cfg["env"]["market"], ctx.cfg["env"]["warmup_seconds"])
    ctx.extra["record"] = dict(net.net.record)
    ctx.ckpt.offer(net, ctx.env_steps, ctx.episodes, {"net": str(d / f"seed{ctx.seed}.net")})
    if ctx.live:
        ctx.live.ckpt(net, ctx.episodes, [], ctx.ckpt.pop_evals(), 0.0)
        ctx.live.done(ctx.ckpt.champion_score, str(d / f"seed{ctx.seed}.net"))
