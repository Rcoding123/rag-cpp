#!/usr/bin/env python3
"""of_zoo_trace.py - episode replays of any ragrl agent (OF15 zoo) for tools/of_viz.py.

    python tools/of_zoo_trace.py RUN [RUN ...] --out DIR [--episodes 20] [--policy ensemble|seedK]
                                 [--oracle best|TICKS] [--threads 4]

RUN is a ragrl run directory (build-avx512/rl/runs/<name>: config.json, bin/_ragof*.pyd,
seeds/seedK/policy.json). For each run this replays the chosen policy (default: the
run's headline ensemble) on visualization episodes 0..N-1 (seeds 960,000,000+: look,
never select or score on them) and writes DIR/<run>.csv in the of_trace schema, with
the 30 s delta rule and the fair-value oracle on the same episodes, then DIR/<run>.html
(tools/of_viz.py) and DIR/index.html: every agent side by side, P&L per episode.

Run it with the Python that trained the runs (the one with torch / stable-baselines3 /
lightgbm, e.g. C:/rag-cpp/.venv/Scripts/python.exe). ragrl is imported from the
directory the run's provenance.json names (ragrl_dir), and each run is replayed in its
own process with RAGOF_DIR=RUN/bin, so every agent sees the simulator binary it was
trained and scored with.

Differences from of_trace: VecScalpEnv exposes no quotes or L1 sizes, so bid, ask,
bid_sz1 and ask_sz1 are empty (of_viz then draws no spread band); imb_l1 is
feature 6 / 2, exactly as the live viewer bins it; q_* are the policy's display
scores (Q-values, log-probabilities or model scores: the header's score_kind);
open_pnl comes from the observation (ScalpEnv::features: open P&L / 4, clipped at
+-4), so it saturates at +-16 ticks. The oracle trades on the hidden gap at the run's
best_oracle threshold (select.json, chosen on validation; of_trace's fixed 2 ticks
never trades on discovery, where the gap rarely reaches 2) unless --oracle is given.
Checked against of_trace (NeuralQ seed, discovery, 5 episodes): every other column
matches exactly.
The policy sees observations only; the privileged channel (fair value, fills) is
written for display and drives only the oracle.
"""
import argparse
import csv
import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TICK_VALUE = 12.5  # ES dollars per tick (of_trace's kTickValue)
FEATURES = ["f_delta5", "f_delta30", "f_delta120", "f_ofi5", "f_ofi30", "f_relvol", "f_imb_l1",
            "f_imb_l5", "f_ret5", "f_ret30", "f_ret120", "f_vwap", "f_absorb"]
COLUMNS = (["episode", "seed", "step", "tod_s", "bid", "ask", "mid", "fair_value", "bid_sz1", "ask_sz1", "imb_l1"] + FEATURES +
           ["open_pnl", "time_left", "q_short", "q_flat", "q_long", "action", "pos", "fill_px", "orders", "fees", "pnl",
            "rule_pos", "rule_pnl", "oracle_pos", "oracle_pnl"])


def fmt(x, nd):
    if x is None:
        return ""
    x = float(x)
    return "" if not math.isfinite(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- one run (child process)

def keep_action(obs):
    import numpy as np
    return (np.sign(obs[:, 13]) + 1).astype(np.int64)


# ragrl.envs' delta_rule_actions / oracle_actions, copied: that module imports stable-baselines3
# (and so torch), which agents that need no torch (LightGBM, NeuralQ) shouldn't depend on.
def delta_rule_actions(obs, th: float = 1.5):
    import numpy as np
    x = obs[:, 1]
    return np.where(x > th, 2, np.where(x < -th, 0, keep_action(obs))).astype(np.int64)


def oracle_actions(obs, priv, th: float):
    import numpy as np
    gap = priv[:, 2]
    return np.where(gap > th, 2, np.where(gap < -th, 0, keep_action(obs))).astype(np.int64)


def load_run_policy(run: Path, cfg: dict, which: str):
    """runner.learned_policies without importing the runner (torch): seedK or the ensemble of every seed."""
    from ragrl.policies import Ensemble, load_policy
    seeds = [int(s) for s in cfg["train"]["seeds"]]
    dirs = {s: run / "seeds" / f"seed{s}" for s in seeds}
    if which == "ensemble":
        return Ensemble([load_policy(d) for d in dirs.values()]) if len(dirs) > 1 else load_policy(next(iter(dirs.values())))
    if which.startswith("seed") and which[4:].isdigit() and int(which[4:]) in dirs:
        return load_policy(dirs[int(which[4:])])
    raise SystemExit(f"{run.name}: no policy {which!r} (have ensemble, " + ", ".join(f"seed{s}" for s in seeds) + ")")


def trace_one(run: Path, out_csv: Path, episodes: int, which: str, oracle_gap: float, threads: int) -> dict:
    import numpy as np
    from ragrl import config, native

    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    params = config.make_params(cfg["env"])
    policy = load_run_policy(run, cfg, which)
    R = native.load()
    base = int(native.block("viz")["base"])

    def env():
        return R.VecScalpEnv(params, R.EpisodeSource.block("viz", 0, episodes), n_envs=episodes, threads=threads, privileged=True)

    ea, er, eo = env(), env(), env()
    oa, orl, oor = ea.reset(), er.reset(), eo.reset()
    policy.reset(episodes)
    n, steps = episodes, []
    cum = np.zeros((3, n))
    fee_ticks = params.fee_dollars / TICK_VALUE
    while True:
        pa = ea.priv.copy()
        a, s = policy.step(oa)
        q = policy.display_scores(oa, s)
        q = np.full((n, 3), np.nan) if q is None else np.asarray(q, dtype=np.float64)
        ar, ao = delta_rule_actions(orl), oracle_actions(oor, eo.priv, oracle_gap)
        pre = {"tod": pa[:, 3], "mid": pa[:, 1], "fv": pa[:, 0], "obs": oa.copy(), "q": q, "a": np.asarray(a, dtype=np.int64)}
        oa, ra, da = ea.step(a)
        orl, rr, _ = er.step(ar)
        oor, ro, _ = eo.step(ao)
        done = bool(da[0])
        cum += np.stack([ra, rr, ro])
        po = ea.final_priv if done else ea.priv
        obs_after = [e.final_obs if done else o for e, o in ((ea, oa), (er, orl), (eo, oor))]
        pos = [np.rint(o[:, 13] * params.contracts).astype(int) for o in obs_after]
        steps.append({**pre, "pos": pos, "fill": po[:, 4].copy(), "orders": po[:, 5].copy(), "pnl": cum.copy()})
        if done:
            break

    stats = [e.episode_stats for e in (ea, er, eo)]
    for k, st in enumerate(stats):  # integrity: the summed rewards are the episodes' net P&L
        dev = float(np.max(np.abs(cum[k] - np.asarray(st["net_pnl_ticks"]))))
        if dev > 1e-6:
            raise SystemExit(f"{run.name}: summed rewards differ from net P&L by {dev:g} ticks ({['agent', 'rule', 'oracle'][k]})")
    if float(np.max(np.abs(steps[-1]["orders"] - np.asarray(stats[0]["orders"])))) > 1e-9:
        raise SystemExit(f"{run.name}: the privileged order count disagrees with episode_stats")

    kind = cfg["agent"]["kind"]
    meta = (f"# of_zoo_trace run={run.name} agent={kind} policy={which} score_kind={policy.score_kind} market={params.market} "
            f"warmup_seconds={params.warmup_seconds:g} oracle_gap={oracle_gap:g} fee_dollars={params.fee_dollars:g} tick_value={TICK_VALUE:g} "
            f"seeds={base}+ (visualization block: look, never select)")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        f.write(meta + "\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for j in range(n):
            for t, st in enumerate(steps):
                o = st["obs"][j]
                w.writerow([j, base + j, t, fmt(st["tod"][j], 1), "", "", fmt(st["mid"][j], 1), fmt(st["fv"][j], 3), "", "",
                            fmt(o[6] / 2.0, 4)] + [fmt(o[k], 4) for k in range(13)] +
                           [fmt(o[14] * 4.0, 2), fmt(o[15], 4)] + [fmt(st["q"][j, k], 4) for k in range(3)] +
                           [int(st["a"][j]), int(st["pos"][0][j]), fmt(st["fill"][j], 2), int(round(st["orders"][j])),
                            fmt(st["orders"][j] * fee_ticks, 2), fmt(st["pnl"][0, j], 2),
                            int(st["pos"][1][j]), fmt(st["pnl"][1, j], 2), int(st["pos"][2][j]), fmt(st["pnl"][2, j], 2)])
    pos = np.stack([st["pos"][0] for st in steps])
    return {"run": run.name, "agent": kind, "policy": which, "score_kind": policy.score_kind, "market": params.market, "oracle_gap": oracle_gap,
            "warmup": params.warmup_seconds, "agent_pnl": cum[0].round(2).tolist(), "rule_pnl": cum[1].round(2).tolist(),
            "oracle_pnl": cum[2].round(2).tolist(), "orders": np.asarray(stats[0]["orders"]).tolist(),
            "long_share": float((pos > 0).mean()), "short_share": float((pos < 0).mean())}


# ---------------------------------------------------------------- parent: one child per run, then the pages

def ragrl_path(run: Path) -> str:
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    d = Path(prov.get("ragrl_dir", ""))
    if not (d / "__init__.py").is_file():
        raise SystemExit(f"{run.name}: provenance names ragrl_dir {d}, which does not exist")
    return str(d.parent)


def oracle_gap(run: Path, arg: str) -> float:
    if arg != "best":
        return float(arg)
    try:
        best = json.loads((run / "select.json").read_text(encoding="utf-8"))["best_oracle"]  # e.g. "oracle_1"
        return float(best.split("_", 1)[1])
    except (OSError, KeyError, ValueError, IndexError):
        print(f"{run.name}: no best_oracle in select.json; the oracle uses of_trace's 2 ticks", file=sys.stderr)
        return 2.0


def index_page(rows: list, episodes: int, failed: list) -> str:
    def cell(v):
        # diverging: blue (+) / red (−) around a neutral midpoint, |v| >= 20 ticks saturates
        a = min(1.0, abs(v) / 20.0)
        var = "--long" if v > 0 else "--short"
        return (f'<td class="c" style="background:color-mix(in srgb,var({var}) {round(100 * a)}%,var(--neutral))" '
                f'title="{v:+.2f} ticks">{v:+.0f}</td>')

    # one block per market (rows of different markets play different episodes): its agents, then its rule and oracle
    groups = {}
    for r in rows:
        groups.setdefault((r["market"], r["warmup"]), []).append(r)
    entries = []
    for (market, _), grp in groups.items():
        tag = f" · {html.escape(market)}" if len(groups) > 1 else ""
        ref = grp[0]
        entries += [(f'{html.escape(r["agent"])} · {html.escape(r["policy"])}{tag}', r["agent_pnl"], r["html"], r) for r in grp]
        entries += [(f"30 s delta rule{tag}", ref["rule_pnl"], None, None),
                    (f'fair-value oracle (gap &gt; {ref["oracle_gap"]:g} ticks){tag}', ref["oracle_pnl"], None, None)]
    lines = []
    for label, pnl, link, extra in entries:
        mean = sum(pnl) / len(pnl)
        name = f'<a href="{html.escape(link)}">{label}</a>' if link else f"<i>{label}</i>"
        info = (f'<td class="n">{sum(extra["orders"]) / len(extra["orders"]):.1f}</td><td class="n">{100 * extra["long_share"]:.0f}% / '
                f'{100 * extra["short_share"]:.0f}%</td>') if extra else '<td></td><td></td>'
        lines.append(f'<tr><th>{name}</th><td class="n">{mean:+.2f}</td>{info}' + "".join(cell(v) for v in pnl) + "</tr>")
    fails = ("<p>Not replayed: " + "; ".join(f"{html.escape(n)} ({html.escape(w)})" for n, w in failed) + "</p>") if failed else ""
    markets = sorted({f'{r["market"]} (warm-up {r["warmup"]:g} s)' for r in rows})
    heads = "".join(f"<th class=\"c\">{i}</th>" for i in range(episodes))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent zoo replays</title>
<style>
:root{{--surface:#fcfcfb;--ink:#1d1d1b;--ink2:#5d5c58;--axis:#d6d5d0;--long:#2a78d6;--short:#e34948;--neutral:#f0efec;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--surface:#1b1b1a;--ink:#ecebe7;--ink2:#a5a49e;--axis:#383835;--long:#3987e5;--short:#e66767;--neutral:#383835;color-scheme:dark}}}}
:root[data-theme="dark"]{{--surface:#1b1b1a;--ink:#ecebe7;--ink2:#a5a49e;--axis:#383835;--long:#3987e5;--short:#e66767;--neutral:#383835;color-scheme:dark}}
body{{margin:0;padding:16px;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:var(--ink2);margin:4px 0 12px;max-width:900px}}
.wrap{{overflow-x:auto}} table{{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-weight:500;white-space:nowrap;padding:3px 8px 3px 0}} td.n{{text-align:right;padding:3px 10px;white-space:nowrap}}
td.c,th.c{{min-width:30px;text-align:center;font-size:12px;padding:4px 2px;border-radius:3px}} th.c{{color:var(--ink2);font-weight:400}}
a{{color:var(--long)}} thead th{{color:var(--ink2);font-weight:400;font-size:12px}}
</style></head><body>
<h1>Agent zoo replays</h1>
<p>{html.escape(", ".join(markets))}. Visualization episodes 0–{episodes - 1} (seeds 960,000,000+): look as often as you like, never select or score on them.
Each cell is one episode's net P&amp;L in ticks (blue gain, red loss, saturating at ±20). Click an agent for its step-by-step replay.</p>
<div class="wrap"><table><thead><tr><th>agent · policy</th><th>mean</th><th>orders</th><th>long / short</th>{heads}</tr></thead>
<tbody>{"".join(lines)}</tbody></table></div>
{fails}</body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--policy", default="ensemble")
    ap.add_argument("--oracle", default="best", help="oracle gap threshold in ticks, or best: the run's select.json best_oracle")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not 1 <= a.episodes <= 500:
        ap.error("--episodes wants 1..500")
    if a.one:  # child: RAGOF_DIR and PYTHONPATH are set by the parent
        if sys.platform == "win32":
            # On this server (2026-11-11) every WMI query hangs ~20 s and then crashes the process, and
            # Python 3.12's platform module asks WMI for the OS version: torch and lightgbm call it on
            # import. Its own fallback (registry / `ver`) runs when the query raises OSError.
            import platform

            def _no_wmi(*_a, **_k):
                raise OSError("WMI bypassed by of_zoo_trace")
            platform._wmi_query = _no_wmi
        res =trace_one(a.one, a.out, a.episodes, a.policy, float(a.oracle), a.threads)
        print(json.dumps(res))
        return 0
    if not a.runs:
        ap.error("give at least one run directory")
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import of_viz
    rows, failed = [], []
    for run in a.runs:
        run = run.resolve()
        if not (run / "config.json").is_file() or not (run / "bin").is_dir():
            raise SystemExit(f"{run}: not a ragrl run directory (config.json, bin/)")
        env = os.environ.copy()
        env["RAGOF_DIR"] = str(run / "bin")
        env["PYTHONPATH"] = ragrl_path(run) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONHASHSEED"] = "0"
        out_csv = a.out / f"{run.name}.csv"
        gap = oracle_gap(run, a.oracle)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(run), "--out", str(out_csv), "--episodes", str(a.episodes),
               "--policy", a.policy, "--oracle", repr(gap), "--threads", str(a.threads)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
            hint = (" - a native crash (0xC0000005); check that `python -c \"import torch\"` works in this Python"
                    if p.returncode in (3221225477, -1073741819) else "")
            print(f"{run.name}: replay failed (exit {p.returncode}){hint}", file=sys.stderr)
            failed.append((run.name, f"exit {p.returncode}{hint}"))
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        if of_viz.main(["of_viz", str(out_csv), str(a.out / f"{run.name}.html")]) != 0:
            raise SystemExit(f"{run.name}: of_viz failed")
        res["html"] = f"{run.name}.html"
        rows.append(res)
        m = sum(res["agent_pnl"]) / a.episodes
        print(f"{run.name}: {res['agent']} {res['policy']} {m:+.2f} ticks/episode "
              f"(rule {sum(res['rule_pnl']) / a.episodes:+.2f}, oracle {sum(res['oracle_pnl']) / a.episodes:+.2f}), visualization block: not a score")
    if not rows:
        raise SystemExit("no run could be replayed")
    (a.out / "index.html").write_text(index_page(rows, a.episodes, failed), encoding="utf-8", newline="\n")
    print(f"wrote {a.out / 'index.html'}: {len(rows)} agents x {a.episodes} episodes" + (f", {len(failed)} not replayed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
