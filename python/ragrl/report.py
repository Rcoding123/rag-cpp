"""report.md, regenerated from config.json, provenance.json, select.json, sealed.json,
redflags.json and the seeds' val.json (no environment).

report.md (in the run) has wall-clock timings and provenance (git, module and venv
hashes); report_results.md leaves them out, so it is byte-identical when the same
experiment is regenerated: that is the copy committed as experiments/of15_*.txt."""
from pathlib import Path

from . import registry, stats


def _f(x, nd=2, sign=True):
    if x is None:
        return "-"
    try:
        return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _table(summary: dict, names) -> list[str]:
    out = ["| policy | mean ticks/ep | SE | t | win % | orders/ep | fees/ep | Sharpe (ann.) |", "|---|---|---|---|---|---|---|---|"]
    for k in names:
        s = summary[k]
        out.append(f"| {k} | {_f(s['mean'])} | {_f(s['se'], 2, False)} | {_f(s['t'], 1)} | {100 * s['win']:.0f} | {s['orders']:.1f} | "
                   f"{s['fees']:.2f} | {_f(s['sharpe_annual'], 1)} |")
    return out


def render(run_dir: Path, deterministic: bool = False) -> str:
    run_dir = Path(run_dir)
    cfg = registry.read_json(run_dir / "config.json")
    prov = registry.read_json(run_dir / "provenance.json")
    sel = registry.read_json(run_dir / "select.json")
    sealed = registry.read_json(run_dir / "sealed.json") if (run_dir / "sealed.json").exists() else None
    flags = registry.read_json(run_dir / "redflags.json") if (run_dir / "redflags.json").exists() else None
    L = [f"# {cfg['name']} ({cfg['agent']['kind']})", ""]
    if flags:
        L += [f"**STATUS: {flags['status']}**", ""]
        for grp, title in (("hard", "Red flags"), ("claim", "Claims (orchestrator: 950M+ confirmation)"), ("warnings", "Warnings")):
            if flags[grp]:
                L.append(f"{title}:")
                L += [f"- {x['id']} {x['policy']}: {x['detail']}" for x in flags[grp]]
                L.append("")
    if sealed and sealed.get("dry_run"):
        L += ["**DRY RUN: scored on the visualization block (960M+), not the sealed test. Not a result.**", ""]
    t, e = cfg["train"], cfg["env"]
    L += [f"market {e['market']}, warm-up {e['warmup_seconds']:g} s, latency {e['latency_steps']}, fee ${e['fee_dollars']:g}, "
          f"{e['contracts']} contract(s); seeds {t['seeds']}; {t['episodes']} episodes ({t['env_steps']} decisions) per seed"
          + (" (budget rounded up to whole rounds)" if t["budget_rounded"] else "") + f", n_envs {t['n_envs']}.",
          f"agent params: `{cfg['agent']['params']}`", "",
          f"Selection: champions on validation [{cfg['select']['val_ckpt']['first']}, +{cfg['select']['val_ckpt']['count']}) "
          f"every {cfg['select']['every_steps']} steps; comparisons on val_select [{cfg['val_select']['first']}, "
          f"+{cfg['val_select']['count']}); headline declared before training: **{cfg['score']['headline']}**; "
          f"multiplicity M = {sel['multiplicity']} per run"
          + (f", x family_size {flags['family_size']} -> M = {flags['multiplicity']} (t_req {flags['t_required']:.2f})."
             if flags and "family_size" in flags else f" (t_req {stats.t_required(sel['multiplicity']):.2f})."), ""]
    if sealed:
        names = [*sealed["learned"], *sealed["baselines"]]
        reused = bool((sealed.get("range_reuse") or {}).get("reused"))
        title = "Dry run (viz)" if sealed["dry_run"] else "REUSED test range (comparable only, not a fresh test)" if reused else "Sealed test"
        L += [f"## {title}: {sealed['block']} [{sealed['first']}, +{sealed['count']}), paired (every policy plays the same episodes)", ""]
        L += _table(sealed["summary"], names)
        L += ["", f"Paired differences (row - reference), 95% bootstrap CI (B = {sealed['bootstrap']['B']}, seed {sealed['bootstrap']['seed']}):", "",
              "| policy | reference | mean | SE | t | 95% CI |", "|---|---|---|---|---|---|"]
        for ref, rows in sealed["paired"].items():
            for k, d in rows.items():
                ci = d.get("ci95") or [None, None]
                L.append(f"| {k} | {ref} | {_f(d['mean'])} | {_f(d['se'], 2, False)} | {_f(d['t'], 1)} | [{_f(ci[0])}, {_f(ci[1])}] |")
        L.append("")
        if sealed.get("paired_gross"):
            G = sealed["gross"]
            L += ["Gross (before fees) per episode: " + ", ".join(f"{k} {_f(G[k]['mean'])}" for k in [*sealed["learned"], *sealed["paired_gross"]]),
                  "", "Paired gross differences against every oracle threshold (row - oracle):", "",
                  "| policy | oracle | mean | SE | t | 95% CI |", "|---|---|---|---|---|---|"]
            for o, rows in sealed["paired_gross"].items():
                for k, d in rows.items():
                    ci = d.get("ci95") or [None, None]
                    L.append(f"| {k} | {o} | {_f(d['mean'])} | {_f(d['se'], 2, False)} | {_f(d['t'], 1)} | [{_f(ci[0])}, {_f(ci[1])}] |")
            L.append("")
    L += [f"## Validation (val_select [{cfg['val_select']['first']}, +{cfg['val_select']['count']}))", ""]
    vnames = [k for k in sel["summary"]]
    L += _table(sel["summary"], vnames)
    L += ["", f"Validation-best oracle: {sel['best_oracle']}", "", "## Seeds", "",
          "| seed | val_ckpt (champion) | val_select | test | offers | champion step |" + ("" if deterministic else " train s | eval s |"),
          "|---|---|---|---|---|---|" + ("" if deterministic else "---|---|")]
    for s in t["seeds"]:
        v = registry.read_json(run_dir / "seeds" / f"seed{s}" / "val.json")
        k = f"seed{s}"
        test = _f(sealed["summary"][k]["mean"]) if sealed else "-"
        L.append(f"| {s} | {_f(v['val_ckpt_mean'])} | {_f(sel['summary'][k]['mean'])} | {test} | {v['offers']} | {v['champion_step']} |"
                 + ("" if deterministic else f" {v['train_seconds']:.0f} | {v['eval_seconds']:.0f} |"))
        if v.get("extra"):
            extra = {a: b for a, b in v["extra"].items() if a not in ("record", "net")}
            if extra:
                L.append(f"|  | extra: `{extra}` | | | | |" + ("" if deterministic else " | |"))
    det = sel.get("determinism", {})
    L += ["", f"Evaluation determinism (seed {det.get('seed')} champion on val_ckpt, twice and vs its worker): "
          f"{det.get('same_twice')} / {det.get('same_as_worker')}.", ""]
    if deterministic:
        L += [f"Config sha256 {(run_dir / 'config.sha256').read_text(encoding='utf-8').strip()}; versions {prov['versions']}; "
              "timings and binary provenance are in the run's report.md.", ""]
    else:
        L += ["## Provenance", "", f"- git {prov['git_sha']}{' (dirty, diff sha1 ' + prov['dirty_diff_sha1'] + ')' if prov['dirty'] else ''}",
              f"- ragrl code sha256 {prov.get('ragrl_sha256', '-')}",
              f"- module {prov['module']} sha256 {prov['module_sha256']} (from {prov.get('source_module_dir', '-')})",
              f"- venv freeze sha256 {prov['freeze_sha256']}",
              f"- versions {prov['versions']}, python {prov['python']}",
              f"- config sha256 {(run_dir / 'config.sha256').read_text(encoding='utf-8').strip()}", ""]
    L += ["Simulator results only: an edge must survive the orchestrator's 950M+ confirmation and recorded-tick replay "
          "(of_backtest / VecReplayEnv) before it is trusted."]
    return "\n".join(L) + "\n"


def write(run_dir: Path):
    (Path(run_dir) / "report.md").write_text(render(run_dir), encoding="utf-8", newline="\n")
    (Path(run_dir) / "report_results.md").write_text(render(run_dir, deterministic=True), encoding="utf-8", newline="\n")
