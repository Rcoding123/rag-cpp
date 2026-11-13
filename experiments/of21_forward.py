#!/usr/bin/env python3
"""OF21: forward test of exhaustion at the OF20 levels on days recorded from 2026-11-12 on (docs/ai/OF21_FORWARD_PLAN.md).

  python experiments/of21_forward.py --rehearse --out experiments/of21_rehearsal.txt
  python experiments/of21_forward.py --count --i-am-the-orchestrator
  python experiments/of21_forward.py --look K --i-am-the-orchestrator --out experiments/of21_lookK.txt

--rehearse  OF20's 37 scored days (2026-09-22 .. 2026-11-11) through this file's code path, straight from the .scid
            files, no ledger; must reproduce OF20's "exh any tol 2" row exactly.
--count     forward days: dates, front month, full / not full and a hash of the records only (no prices, no
            outcomes); logged as real_forward_scan. A day's status is fixed by its first scan.
--look K    the first N_K full forward days (plan section 4), logged as real_forward through ragrl.ledger.open_forward
            BEFORE any price is read. Refused unless every frozen file's sha256 is the one in the plan (section 8), the
            rehearsal reproduces, no scored day's records changed since its first scan, enough full days exist, and the
            ledger allows the look (in order, cumulative, never repeated except a crash rerun with the same fingerprint).
            From the deadline (plan section 4) look 3 takes every full day so far, with the remaining alpha.

The rule, fills, exits and statistics are OF19's and OF20's code, imported unchanged: exhaustion = OF19 P3 N 1 e 5
within 2 ticks of any OF20 level counted at the setup's formation time; shell = the same without the footprint
condition at the same levels.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "experiments"), str(REPO / "tools"), str(REPO / "python")):
    if p not in sys.path:
        sys.path.insert(0, p)
_spec = importlib.util.spec_from_file_location("of20_atlevels", REPO / "experiments" / "of20_atlevels.py")
A20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A20)
of19 = A20.of19
LV = A20.LV
import scid_to_bars as SB  # noqa: E402
import scid_to_trades as ST  # noqa: E402
from ragrl import ledger, real  # noqa: E402

CLAIM = "OF21"
PLAN = REPO / "docs" / "ai" / "OF21_FORWARD_PLAN.md"
CUTOFF = "2026-11-12"
DEADLINE = "2028-03-31"
LOOKS = {1: 60, 2: 125, 3: 250}
BOUNDS = {1: 4.504, 2: 3.022, 3: 2.007}  # Lan-DeMets O'Brien-Fleming, one-sided alpha 1 - Phi(2) (plan section 4)
ALPHA = 0.5 * math.erfc(2 / math.sqrt(2))
P3_KW = dict(N=1, e=5)
TOL = 2
GAP_US = 60_000_000
PRIOR_BACK = 7
ROLL_DAYS = 14
CONTRACTS = ("ESH26", "ESM26", "ESU26", "ESZ26", "ESH27", "ESM27", "ESU27", "ESZ27", "ESH28")
MONTHS = {"H": 3, "M": 6, "U": 9, "Z": 12}
FROZEN = ("experiments/of21_forward.py", "experiments/of20_atlevels.py", "experiments/of20_levels.py", "experiments/of19_footprint.py",
          "tools/scid_to_bars.py", "tools/scid_to_trades.py")
REPORTED = ("python/ragrl/ledger.py", "python/ragrl/real.py")
REHEARSE = ("2026-09-22", "2026-11-11")
REHEARSE_EXPECT = {"trades": 119, "net": "+1.44", "se": "2.17", "shell_trades": 925, "diff": "+2.61", "diff_se": "2.23"}


# ---------------------------------------------------------------- freeze
def file_sha256(rel: str) -> str:
    """sha256 of a file with CRLF read as LF (git's autocrlf may convert line endings on checkout)."""
    return hashlib.sha256((REPO / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_sha256(plan=PLAN) -> dict:
    """{path: sha256} from the plan's section 8 lines "`path` sha256: `hex`"."""
    return dict(re.findall(r"`([\w./]+\.py)` sha256: `([0-9a-f]{64})`", Path(plan).read_text(encoding="utf-8")))


def freeze_problems(plan=PLAN) -> list:
    want = frozen_sha256(plan)
    return [f for f in FROZEN if want.get(f) != file_sha256(f)]


# ---------------------------------------------------------------- sequential boundaries
def final_boundary(prev_ns, prev_bounds, n, paths=4_000_000, seed=0) -> float:
    """The last look's boundary that spends all alpha left after earlier looks at prev_ns days with prev_bounds (Brownian
    Monte Carlo, correlations sqrt(n_j / n_k)); with prev_ns (60, 125), their bounds and n = 250 it is BOUNDS[3]."""
    ns = np.array(list(prev_ns) + [n], float)
    t = ns / n
    Z = np.cumsum(np.random.default_rng(seed).standard_normal((paths, len(ns))) * np.sqrt(np.diff(np.r_[0, t])), 1) / np.sqrt(t)
    alive = np.ones(paths, bool)
    for k, b in enumerate(prev_bounds):
        alive &= Z[:, k] < b
    left = ALPHA - (1 - alive.mean())
    z = np.sort(Z[alive, -1])[::-1]
    return float(z[max(int(round(left * paths)) - 1, 0)])


# ---------------------------------------------------------------- days
def weekdays(d0: str, d1: str) -> list:
    out, d = [], date.fromisoformat(d0)
    while d <= date.fromisoformat(d1):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def expiry(contract: str) -> date:
    """The third Friday of the contract month (ES quarterly expiry)."""
    first = date(2000 + int(contract[3:5]), MONTHS[contract[2]], 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def next_contract(contract: str):
    i = CONTRACTS.index(contract)
    return CONTRACTS[i + 1] if i + 1 < len(CONTRACTS) else None


def open_files(scid_dir) -> dict:
    return {k: SB.Scid(Path(scid_dir) / f"{k}_FUT_CME.scid") for k in CONTRACTS if (Path(scid_dir) / f"{k}_FUT_CME.scid").exists()}


def rth_us(day: str, m: dict) -> tuple:
    o, c = real.rth_span(day, m)
    return real.sierra_us(o), real.sierra_us(c)


def style_of(r: np.ndarray) -> str:
    sec = r["t"] // 1_000_000
    return "tick" if len(r) > 1 and float((sec[1:] == sec[:-1]).mean()) > 0.5 else "bundle"


def full_day(r: np.ndarray, t0: int, t1: int, style: str) -> str:
    """'' for a full session, else the reason (plan section 2)."""
    if style != "tick":
        return f"{style} records"
    if not len(r):
        return "no records"
    t = r["t"]
    if t[0] - t0 > GAP_US:
        return "late first record"
    if t1 - t[-1] > GAP_US:
        return "early last record (half day or outage)"
    if len(t) > 1 and int(np.diff(t).max()) > GAP_US:
        return "a gap over 60 s"
    return ""


def prior_session(day: str, contract: str, files: dict, m: dict):
    """(date, records) of the latest earlier weekday, up to PRIOR_BACK calendar days back, on which `contract` has RTH
    records (a half day counts), or (None, None)."""
    d = date.fromisoformat(day)
    for k in range(1, PRIOR_BACK + 1):
        p = (d - timedelta(days=k)).isoformat()
        if date.fromisoformat(p).weekday() >= 5:
            continue
        r = files[contract].records(*rth_us(p, m))
        if len(r):
            return p, r
    return None, None


def scan(days: list, files: dict, m: dict) -> list:
    """Per day: (date, front month, reason or '' when scored, sha256 of the front month's overnight and RTH records).
    Uses only timestamps and volumes (build_day's bars are discarded)."""
    out = []
    for day in days:
        bars, side = SB.build_day(day, files, m)
        if bars is None:
            out.append((day, None, side, ""))
            continue
        c = side["contract"]
        t0, t1 = rth_us(day, m)
        r = files[c].records(t0, t1)
        why = full_day(r, t0, t1, side["style"])
        nx = next_contract(c)
        if not why and 0 <= (expiry(c) - date.fromisoformat(day)).days <= ROLL_DAYS:
            rn = files[nx].records(t0, t1) if nx in files else np.zeros(0, SB.REC)
            if style_of(rn) != "tick":
                why = f"roll window: {nx} not recorded tick by tick"
        if not why and prior_session(day, c, files, m)[0] is None:
            why = "no recorded prior session of its contract"
        prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        ro = files[c].records(real.sierra_us(real.ct_wall(prev, "17:00")), t0)  # the overnight OF20's levels read
        out.append((day, c, why, hashlib.sha256(ro.tobytes() + r.tobytes()).hexdigest()))
    return out


def fix_status(sc: list, scans: list) -> tuple:
    """(rows with each day's status fixed by its first logged scan, full days whose records changed since that scan)."""
    first = {}
    for u in scans:
        for d, e in u.get("days", {}).items():
            first.setdefault(d, e)
    rows, changed = [], []
    for d, c, w, h in sc:
        if d in first:
            e = first[d]
            if e["status"] == "" and e["sha256"] != h:
                changed.append(d)
            rows.append((d, e["contract"], e["status"], e["sha256"]))
        else:
            rows.append((d, c, w, h))
    return rows, changed


# ---------------------------------------------------------------- trades and statistics
def trades(r: np.ndarray, t0: int) -> dict:
    """OF19's trade arrays for one RTH record range: scid_to_trades.trades_of, then of19.load_trades (through an
    in-memory npz, the same bytes path as the OF19 files)."""
    tr, _dropped, _dvol, _inf = ST.trades_of(r, t0)
    buf = io.BytesIO()
    np.savez(buf, **tr)
    buf.seek(0)
    return of19.load_trades(buf)[0]


def day_rows(day: str, contract: str, files: dict, m: dict) -> dict:
    """Setup and shell trades of one scored day within TOL of a level: per kind arrays g (gross ticks), side, lt."""
    t0, t1 = rth_us(day, m)
    T = trades(files[contract].records(t0, t1), t0)
    pday, pr = prior_session(day, contract, files, m)
    Tp = trades(pr, rth_us(pday, m)[0])
    lv = LV.static_levels(day, contract, T, Tp)
    rv = LV.RunningVWAP(T)
    F, E = of19.footprint(T), of19.Events(T)
    out = {"prior": pday, "levels": lv}
    for kind, shell in (("setup", False), ("shell", True)):
        meta = []
        rows = of19.sig_P3(F, E, P3_KW["N"], P3_KW["e"], shell=shell, meta=meta)
        g, keep, _sig_t, side, _sd, _hold = of19.run_rows(E, rows, "P3", P3_KW)
        assert len(meta) == len(g)
        cl = [A20.classify(ft, px, lv, rv) for ft, px in meta]
        d = {"ds": np.array([c[0] for c in cl], float), "dv": np.array([c[1] for c in cl], float)}
        at = A20.mask_for(d, "any", TOL) if cl else np.zeros(0, bool)
        lt = np.array([c[2] or "" for c in cl], object)
        k = keep & at if len(g) else np.zeros(0, bool)
        out[kind] = {"g": g[k], "side": side[k], "lt": lt[k] if len(lt) else lt}
    return out


def evaluate(days: list, files: dict, m: dict, log=print) -> dict:
    n = len(days)
    acc = {f: np.zeros(n) for f in ("n", "gross", "net", "win", "sn", "sgross", "long_n", "long_net")}
    per_trade, types, contracts = [], {}, {}
    for i, (day, c) in enumerate(days):
        R = day_rows(day, c, files, m)
        g, sg = R["setup"]["g"], R["shell"]["g"]
        acc["n"][i], acc["gross"][i], acc["net"][i] = len(g), g.sum(), (g - of19.FEE).sum()
        acc["win"][i] = (g - of19.FEE > 0).sum()
        lg = R["setup"]["side"] > 0
        acc["long_n"][i], acc["long_net"][i] = lg.sum(), (g[lg] - of19.FEE).sum()
        acc["sn"][i], acc["sgross"][i] = len(sg), sg.sum()
        per_trade += [(i, x - of19.FEE) for x in g]
        for lt_, x in zip(R["setup"]["lt"], g):
            t_ = types.setdefault(lt_, [0, 0.0])
            t_[0] += 1
            t_[1] += x - of19.FEE
        contracts[c] = contracts.get(c, 0) + 1
        log(f"{day} {c}: prior session {R['prior']}, {len(g)} trades, {len(sg)} shell trades")
    idx = {b: of19.block_idx(n, b, np.random.default_rng(of19.SEED)) for b in of19.BLOCKS}
    N = acc["n"].sum()
    r = {"days": n, "trades": int(N), "shell_trades": int(acc["sn"].sum()), "types": types, "contracts": contracts}
    if N < 2:
        return {**r, **{f: float("nan") for f in ("net", "se", "t", "diff", "diff_se", "stress")}}
    net = acc["net"].sum() / N
    se = of19.ratio_se(acc["net"], acc["n"], idx)

    def diff(I=None):
        if I is None:
            n_, gs, sn, sg = (acc[f].sum() for f in ("n", "gross", "sn", "sgross"))
        else:
            n_, gs, sn, sg = (acc[f][I].sum(1) for f in ("n", "gross", "sn", "sgross"))
        return np.where(n_ > 0, gs / np.maximum(n_, 1), np.nan) - np.where(sn > 0, sg / np.maximum(sn, 1), np.nan)
    half = n // 2  # halves split at the look's middle scored day
    n1, n2 = acc["n"][:half].sum(), acc["n"][half:].sum()
    nets = np.array([x for _, x in per_trade])
    order = np.argsort(-nets)
    best2 = set(np.argsort(-acc["net"])[:2].tolist())
    keep2 = np.array([i not in best2 for i, _ in per_trade])
    ln = acc["long_n"].sum()
    r.update(net=net, se=se, t=net / se if se > 0 else float("nan"), gross=acc["gross"].sum() / N, win=acc["win"].sum() / N,
             median=float(np.median(nets)), sd=float(nets.std(ddof=1)), stress=net - of19.STRESS,
             no_top3=float(nets[order[3:]].mean()) if len(nets) > 3 else float("nan"),
             no_best2_days=float(nets[keep2].mean()) if keep2.any() else float("nan"),
             h1=acc["net"][:half].sum() / n1 if n1 else float("nan"), h2=acc["net"][half:].sum() / n2 if n2 else float("nan"),
             long_net=acc["long_net"].sum() / ln if ln else float("nan"),
             short_net=(acc["net"].sum() - acc["long_net"].sum()) / (N - ln) if N > ln else float("nan"),
             shell_net=acc["sgross"].sum() / acc["sn"].sum() - of19.FEE if acc["sn"].sum() else float("nan"),
             diff=float(diff()), diff_se=max(float(np.nanstd(diff(I), ddof=1)) for I in idx.values()))
    return r


def report(r: dict, head: list) -> list:
    n_ = of19.n_
    L = list(head)
    L.append(f"days {r['days']}; front months {r['contracts']}")
    L.append(f"  trades {r['trades']}  win {n_(r.get('win'), 2, False)}  gross {n_(r.get('gross'))}  net {n_(r['net'])}  se "
             f"{n_(r['se'], 2, False)}  t {n_(r['t'], 2)}  stress {n_(r.get('stress'))}")
    L.append(f"  shell trades {r['shell_trades']}  shell net {n_(r.get('shell_net'))}  setup - shell {n_(r.get('diff'))} "
             f"(se {n_(r.get('diff_se'), 2, False)})")
    L.append(f"  median {n_(r.get('median'))}  sd {n_(r.get('sd'), 2, False)}  without the top 3 trades {n_(r.get('no_top3'))}  "
             f"without the best 2 days {n_(r.get('no_best2_days'))}")
    L.append(f"  halves {n_(r.get('h1'))} / {n_(r.get('h2'))}  long / short {n_(r.get('long_net'))} / {n_(r.get('short_net'))}")
    L.append("  by nearest level type (trades, net per trade): " + "; ".join(
        f"{k or '-'}: {v[0]}, {n_(v[1] / v[0])}" for k, v in sorted(r["types"].items()) if v[0]))
    flags = []
    if math.isfinite(r["t"]) and r["t"] > 6:
        flags.append("t > 6")
    if r["trades"] >= 30 and r.get("win", 0) > 0.75:
        flags.append("win rate > 0.75")
    L.append(f"red flags: {flags or 'none'}")
    return L


def decision(r: dict, look: int, bound: float, final: bool) -> str:
    if r["trades"] > 1 and r["net"] > 0 and r["t"] >= bound and r["diff"] > 0 and r["stress"] > 0:
        return (f"EDGE at look {look} (t {r['t']:.2f} >= {bound:.3f}): the rule passed its pre-registered forward test (plan "
                "section 5). This does not show that the footprint condition beats a plain level touch.")
    if not final and not (r["net"] > 0):
        return f"FUTILITY at look {look}: net {r['net']:+.2f} <= 0, no edge so far; stopping is recommended (non-binding)."
    if not final:
        return f"CONTINUE: no edge at look {look} (t {r['t']:.2f} < {bound:.3f} or another condition failed); next look at {LOOKS[look + 1]} days."
    return (f"NO EDGE at the final look: net {r['net']:+.2f} per trade, upper bound net + {bound:.3f} x SE = "
            f"{r['net'] + bound * r['se']:+.2f}; {bound * r['se']:.2f} ticks per trade is the edge this test detects about half the time.")


def rehearse(files: dict, m: dict, log=print) -> tuple:
    sc = scan(weekdays(*REHEARSE), files, m)
    bad = [(d, w) for d, _c, w, _h in sc if w]
    if bad:
        raise RuntimeError(f"rehearsal days not full: {bad}")
    r = evaluate([(d, c) for d, c, _w, _h in sc], files, m, log=log)
    got = {"trades": r["trades"], "net": of19.n_(r["net"]), "se": of19.n_(r["se"], 2, False), "shell_trades": r["shell_trades"],
           "diff": of19.n_(r["diff"]), "diff_se": of19.n_(r["diff_se"], 2, False)}
    return r, got, got == REHEARSE_EXPECT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--look", type=int, choices=sorted(LOOKS))
    ap.add_argument("--i-am-the-orchestrator", dest="orch", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scid-dir", default="C:/SierraChart/Data")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--until", default=None, help="last day to consider (default: yesterday); its session must be over")
    a = ap.parse_args(argv)
    m = real.load_manifest(a.manifest)
    if m["confirm_rule"]["from"] != CUTOFF:
        raise RuntimeError(f"the manifest's confirm block starts {m['confirm_rule']['from']}, not {CUTOFF}")
    if (a.count or a.look) and not a.orch:
        print("of21_forward: refused: forward days are for the orchestrator only (--i-am-the-orchestrator)", file=sys.stderr)
        return 2
    if a.look and freeze_problems():
        print(f"of21_forward: refused: files differ from their sha256 frozen in {PLAN.name} section 8: {freeze_problems()}", file=sys.stderr)
        return 2
    until = a.until or (date.today() - timedelta(days=1)).isoformat()
    if real.rth_span(until, m)[1] >= time.time():
        print(f"of21_forward: refused: the session of {until} is not over", file=sys.stderr)
        return 2
    if not a.rehearse and until < CUTOFF:
        print("of21_forward: refused: no forward day yet", file=sys.stderr)
        return 2
    files = open_files(a.scid_dir)
    ok = True
    try:
        if a.rehearse:
            r, got, ok = rehearse(files, m)
            L = report(r, ["OF21 rehearsal: experiments/of21_forward.py on OF20's 37 scored days (2026-09-22 .. 2026-11-11), from the "
                           ".scid files", f"expected (OF20 'exh any tol 2'): {REHEARSE_EXPECT}", f"got: {got}"])
            L.append("REHEARSAL OK: the runner reproduces OF20 exactly." if ok else "REHEARSAL FAILED: the runner does not reproduce OF20.")
        else:
            if a.look:
                _r, got, rok = rehearse(files, m, log=lambda s: None)
                if not rok:
                    print(f"of21_forward: refused: the rehearsal does not reproduce OF20 ({got})", file=sys.stderr)
                    return 2
            sc = scan(weekdays(CUTOFF, until), files, m)
            ledger.log_forward_scan(CLAIM, {d: {"contract": c, "status": w, "sha256": h} for d, c, w, h in sc}, orchestrator=True,
                                    until=until, mode="look" if a.look else "count")
            rows, changed = fix_status(sc, ledger.forward_scans(CLAIM))
            used = {d for u in ledger.real_openings("confirm") for d in u.get("days", [])}
            full = [(d, c) for d, c, w, _h in rows if not w and d not in used]
            if a.count:
                L = [f"OF21 count to {until}: {len(rows)} weekdays scanned, {len(full)} full forward days "
                     f"(looks at {', '.join(str(v) for v in LOOKS.values())}; deadline {DEADLINE})"]
                L += [f"  {d} {c or '-'}: {w or ('used by a confirmation' if d in used else 'full')}" for d, c, w, _h in rows]
                if changed:
                    L.append(f"RED FLAG: records of full days changed since their first scan: {changed}")
            else:
                prev = sorted((u for u in ledger.forward_openings(CLAIM) if not u.get("retry")), key=lambda u: u["look"])
                deadline_look = a.look == max(LOOKS) and len(full) < LOOKS[a.look] and date.today().isoformat() >= DEADLINE
                need = len(full) if deadline_look else LOOKS[a.look]
                if len(full) < need or need == 0:
                    print(f"of21_forward: refused: {len(full)} full forward days, look {a.look} needs {LOOKS[a.look]}", file=sys.stderr)
                    return 2
                days = full[:need]
                bad = [d for d in changed if d <= days[-1][0]]
                if bad:
                    print(f"of21_forward: refused (red flag): records of full days changed since their first scan: {bad}", file=sys.stderr)
                    return 2
                hashes = {d: h for d, _c, _w, h in rows}
                code = {f: file_sha256(f) for f in FROZEN + REPORTED}
                fp = hashlib.sha256(json.dumps({"days": [(d, hashes[d]) for d, _ in days], "code": code}, sort_keys=True).encode()).hexdigest()
                ledger.open_forward(CLAIM, a.look, [d for d, _ in days], manifest_sha256=real.manifest_sha256(a.manifest),
                                    runner_sha256=code[FROZEN[0]], fingerprint=fp, orchestrator=True, code=code, deadline_look=deadline_look)
                if deadline_look:
                    earlier = [u for u in prev if u["look"] < a.look]
                    bound = final_boundary([len(u["days"]) for u in earlier], [BOUNDS[u["look"]] for u in earlier], need)
                else:
                    bound = BOUNDS[a.look]
                r = evaluate(days, files, m)
                skipped = [(d, w) for d, _c, w, _h in rows if w and d <= days[-1][0]]
                L = report(r, [f"OF21 look {a.look}: exhaustion (OF19 P3 N 1 e 5) within {TOL} ticks of any OF20 level, first {need} "
                               f"full forward days ({days[0][0]} .. {days[-1][0]}); boundary t >= {bound:.3f}"
                               + (" (deadline look: the remaining alpha)" if deadline_look else ""),
                               f"not scored up to the last look day: {skipped}", f"code sha256: {code}"])
                L += ["", "DECISION (plan section 5): " + decision(r, a.look, bound, final=a.look == max(LOOKS))]
    finally:
        for s in files.values():
            s.close()
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
