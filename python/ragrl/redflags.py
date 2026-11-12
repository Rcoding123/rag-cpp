"""Automatic red flags on a sealed run (redflags.json, STATUS, top of report.md).

hard (STATUS=RED_FLAGS)
  R1  annualized Sharpe > 5 for a non-oracle policy that trades >= 2 orders/episode (OF8); below
      2 orders/episode (the oracles' R6 rule) it is reported as the warning R1v (vacuous)
  R2  a learned policy above 0 AND above the validation-best oracle by 2 paired SE (OF9)
  R2g a learned policy's GROSS (net + fees) above the highest-gross oracle threshold by 2 paired SE,
      whatever the sign of its net (a leak that fees hide)
  R5  a naive built-in rule (delta30, rules, hold) net > 0 by 2 SE: the simulator is too easy (OF4)
  R8  nondeterminism: evaluation not bitwise repeatable, or a retrained seed differs (recheck.json)
  R10 ledger audit: more than one sealed opening of the test block for this run, or a sealed
      opening logged outside the canonical usage ledger ($RAGRL_LEDGER_FILE)
claim (STATUS=CLAIM_PENDING_950M_CONFIRMATION unless a hard flag fired)
  R3  a learned policy or the headline net > 0 by 2 SE on the sealed test: only the
      orchestrator may run `ragrl confirm` (950M+, once per claim); the pipeline never does
warnings
  R2w a learned policy's paired mean above the best oracle's (any margin)
  R4  validation -> test collapse (OF7): val_select net > 0 by 2 SE but test < 0.5 x val or test t < 1;
      or a val lead over delta30 (2 SE) not reproduced on test
  R1v R1's Sharpe > 5 on a near-flat policy (< 2 orders/episode): vacuous, not an edge
  R6  the best oracle trades < 2 orders/episode: the oracle bound is uninformative (OF10)
  R7  hidden-state leak probe: |corr(obs_k, gap)| > 0.5 on train episodes (diag.json)
  R9  the headline's t below the multiplicity bar t_req = 2 + sqrt(2 ln M), M = the run's validation
      choices x score.family_size (every run pre-registered on the range) x (1 + earlier outside views)
  R11 days / stress not run by the pipeline (orchestrator only)
  R12 the sealed range was viewed before (--reuse-ok): comparable only, not a fresh test
dry runs (viz) never claim: STATUS is DRY_RUN_OK or DRY_RUN_RED_FLAGS, R3 becomes a warning.
"""
import math

from . import ledger, registry, stats

NAIVE = ("delta30", "hold")
MIN_ORDERS = 2.0  # orders/episode below which a policy's Sharpe or an oracle's bound says nothing (R1v, R6)


def vacuous(s: dict) -> bool:
    """A summary row whose policy trades < MIN_ORDERS orders per episode (near-flat)."""
    return s.get("orders") is not None and s["orders"] < MIN_ORDERS


def _n(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _gt(x, y):
    return x is not None and y is not None and not (isinstance(x, float) and math.isnan(x)) and x > y


def evaluate(run_dir) -> dict:
    sealed = registry.read_json(run_dir / "sealed.json")
    sel = registry.read_json(run_dir / "select.json")
    S = sealed["summary"]
    learned = sealed["learned"]
    hard, claim, warn = [], [], []
    for k, s in S.items():
        if not k.startswith("oracle_") and _gt(s.get("sharpe_annual"), 5.0):
            if vacuous(s):  # a (near-)flat policy: a tiny mean over a tinier SD, not an edge
                warn.append({"id": "R1v", "policy": k, "detail": f"annualized Sharpe {s['sharpe_annual']:.2f} > 5 is vacuous: "
                                                                 f"{s['orders']:.2f} orders/episode < {MIN_ORDERS:g}"})
            else:
                hard.append({"id": "R1", "policy": k, "detail": f"annualized Sharpe {s['sharpe_annual']:.2f} > 5"})
    bo = sel.get("best_oracle")
    if bo and bo in sealed["paired"]:
        for k in learned:
            d = sealed["paired"][bo][k]
            if S[k]["mean"] > 0 and _gt(d["t"], 2.0):
                hard.append({"id": "R2", "policy": k, "detail": f"beats the validation-best oracle {bo} by {d['mean']:+.2f} (t {d['t']:.1f})"})
            elif d["mean"] > 0:
                warn.append({"id": "R2w", "policy": k, "detail": f"paired mean above {bo} ({d['mean']:+.2f}); the oracle loses after costs here"})
    G = sealed.get("gross") or {}
    og = [o for o in sealed.get("paired_gross", {}) if o in G]
    if og:
        top = max(og, key=lambda o: G[o]["mean"])
        for k in learned:
            d = sealed["paired_gross"][top][k]
            if _gt(d["t"], 2.0):
                hard.append({"id": "R2g", "policy": k, "detail": f"gross {G[k]['mean']:+.2f} beats the highest-gross oracle {top} "
                                                                  f"({G[top]['mean']:+.2f}) by {d['mean']:+.2f} (t {d['t']:.1f})"})
    for k in [*learned]:
        if S[k]["mean"] > 0 and _gt(S[k]["t"], 2.0):
            claim.append({"id": "R3", "policy": k, "detail": f"net {S[k]['mean']:+.2f} ticks/episode, t {S[k]['t']:.1f} on {sealed['block']}"})
    for k in S:
        if (k in NAIVE or k.startswith("rule_")) and S[k]["mean"] > 0 and _gt(S[k]["t"], 2.0):
            hard.append({"id": "R5", "policy": k, "detail": f"naive rule net {S[k]['mean']:+.2f} (t {S[k]['t']:.1f}): simulator too easy?"})
    V = sel["summary"]
    for k in learned:
        if k in V and V[k]["mean"] > 0 and _gt(V[k]["t"], 2.0) and (S[k]["mean"] < 0.5 * V[k]["mean"] or not _gt(S[k]["t"], 1.0)):
            warn.append({"id": "R4", "policy": k, "detail": f"validation {V[k]['mean']:+.2f} (t {V[k]['t']:.1f}) -> test {S[k]['mean']:+.2f}"})
        vp = sel.get("paired", {}).get("delta30", {}).get(k)
        tp = sealed["paired"].get("delta30", {}).get(k)
        if vp and tp and vp["mean"] > 0 and _gt(vp["t"], 2.0) and tp["mean"] <= 0:
            warn.append({"id": "R4", "policy": k, "detail": f"validation lead over delta30 {vp['mean']:+.2f} not reproduced on test ({tp['mean']:+.2f})"})
    if bo and bo in S and vacuous(S[bo]):
        warn.append({"id": "R6", "policy": bo, "detail": f"oracle trades {S[bo]['orders']:.1f} orders/episode: bound uninformative"})
    diag = run_dir / "diag.json"
    if diag.exists():
        dj = registry.read_json(diag)
        if dj["max_abs"] > 0.5:
            k = max(range(len(dj["corr_obs_gap"])), key=lambda i: abs(dj["corr_obs_gap"][i]))
            warn.append({"id": "R7", "policy": f"obs[{k}]", "detail": f"|corr(obs, hidden gap)| = {dj['max_abs']:.2f} > 0.5"})
    det = sel.get("determinism", {})
    if det and not (det.get("same_twice") and det.get("same_as_worker")):
        hard.append({"id": "R8", "policy": f"seed{det.get('seed')}", "detail": f"evaluation not bitwise repeatable: {det}"})
    rc = run_dir / "recheck.json"
    if rc.exists():
        r = registry.read_json(rc)
        if not (r["same_actions"] and r["same_val"]):
            hard.append({"id": "R8", "policy": f"seed{r['seed']}", "detail": "a retrained seed differs from the original (recheck.json)"})
    reuse = sealed.get("range_reuse") or {}
    outside = len(reuse.get("outside_family", [])) + len(reuse.get("previewed", [])) + int(reuse.get("family_over", 0))
    M_run = sel.get("multiplicity", 1)
    M = int(M_run * int(sealed.get("family_size", 1)) * (1 + outside))
    t_req = stats.t_required(M)
    h = S[sealed["headline"]]
    if not _gt(h["t"], t_req):
        warn.append({"id": "R9", "policy": sealed["headline"], "detail": f"headline t {_n(h['t'])} below t_req {t_req:.2f} (M = {M}): no edge claim"})
    if not sealed.get("dry_run"):
        opened = [u for u in ledger.read_usage() if u.get("run") == run_dir.name and u.get("block") == "test" and not u.get("dry_run")]
        if len(opened) > 1:
            hard.append({"id": "R10", "policy": "-", "detail": f"the test block was opened {len(opened)} times for this run"})
        if sealed.get("ledger_canonical") is False:
            hard.append({"id": "R10", "policy": "-", "detail": "the sealed opening was logged outside the canonical usage ledger"})
    if reuse.get("reused"):
        warn.append({"id": "R12", "policy": "-", "detail": f"the sealed range was viewed before ({len(reuse.get('previewed', []))} pre-viewed, "
                                                           f"{len(reuse.get('outside_family', []))} outside the family, "
                                                           f"{reuse.get('family_over', 0)} beyond family_size): comparable only"})
    warn.append({"id": "R11", "policy": "-", "detail": "days / stress not run (orchestrator only)"})
    if sealed.get("dry_run"):
        warn += [dict(c, id="R3", detail="(dry run: not a claim) " + c["detail"]) for c in claim]
        claim = []
        status = "DRY_RUN_RED_FLAGS" if hard else "DRY_RUN_OK"
    else:
        status = "RED_FLAGS" if hard else "CLAIM_PENDING_950M_CONFIRMATION" if claim else "NO_EDGE"
    out = {"status": status, "hard": hard, "claim": claim, "warnings": warn, "t_required": t_req, "multiplicity": M,
           "multiplicity_run": M_run, "family_size": int(sealed.get("family_size", 1)),
           "note": "R3 claims go to the orchestrator: `ragrl confirm RUN --i-am-the-orchestrator --claim ID` (950M+, once per claim)."}
    registry.write_json(run_dir / "redflags.json", registry.finite(out))
    (run_dir / "STATUS").write_text(status + "\n", encoding="utf-8")
    return out
