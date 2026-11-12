"""OF16 recorded ES: the Sierra converter, the RTH clock, the replay's fills, and the day
manifest's roles in the usage ledger. Synthetic data only: no test that runs here opens a
Sierra file or an export of a real test or confirm day.

Replays and ledger openings run in fresh processes (`fresh()`): opening the C++ "real"
block makes a process a scorer, and this pytest process may already be a trainer.
"""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))
TOOL = REPO / "tools" / "sierra_to_ticks.py"
SC_EPOCH_US = 25569 * 86400 * 1_000_000


# ---------------------------------------------------------------- the RTH clock
def test_dst_cdt_and_cst_days_cut_the_same_rth():
    from datetime import datetime, timezone

    from ragrl import real
    m = real.load_manifest()
    cdt, cst = real.rth_span("2026-09-14", m), real.rth_span("2026-11-03", m)
    utc = lambda s: datetime.fromtimestamp(s, timezone.utc).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    assert (utc(cdt[0]), utc(cdt[1])) == ("2026-09-14 13:30", "2026-09-14 20:00")   # CDT = UTC-5
    assert (utc(cst[0]), utc(cst[1])) == ("2026-11-03 14:30", "2026-11-03 21:00")   # CST = UTC-6
    for o, c in (cdt, cst):
        assert c - o == 23400
        assert real.ct_clock(o) == "08:30:00.000" and real.ct_clock(c) == "15:00:00.000"
    # the switch days themselves (02:00 local): RTH on the Monday after each switch
    assert real.ct_offset_hours(real.rth_span("2026-03-09")[0]) == -5
    assert real.ct_offset_hours(real.rth_span("2026-11-02")[0]) == -6
    assert real.ct_offset_hours(real.rth_span("2026-10-30")[0]) == -5
    # the explicit rule agrees with zoneinfo wherever a tz database exists
    if real._chicago() is not None:
        for d in ("2026-03-06", "2026-03-09", "2026-09-14", "2026-10-30", "2026-11-02", "2026-11-03", "2027-03-15"):
            o = real.rth_span(d)[0]
            assert real.ct_offset_hours_rule(o) == real.ct_offset_hours(o), d


# ---------------------------------------------------------------- the manifest
def test_manifest_roles():
    from ragrl import real
    m = real.load_manifest()
    assert real.listed_days("test", m) == ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-11-05", "2026-11-09",
                                           "2026-11-10", "2026-11-11"]
    assert real.listed_days("dev", m) == ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-11-03"]
    assert set(real.listed_days("excluded", m)) == {"2026-09-07", "2026-09-18", "2026-11-04", "2026-11-06"}
    e = real.day_entry("2026-11-12", m)
    assert e["role"] == "confirm" and e["contract"] == "ESZ26" and e["depth"] == "ESZ26_FUT_CME.2026-11-12.depth"
    assert real.day_entry("2026-12-21", m)["contract"] == "ESH27"
    assert real.role_of("2026-10-01", m) == "unlisted"
    assert real.day_entry("2026-11-03", m)["scid"] == "ESZ26_FUT_CME.scid"
    assert len(real.manifest_sha256()) == 64


def test_day_clustered_bootstrap():
    from ragrl.score_real import day_bootstrap
    x = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    d = np.array(["a", "a", "a", "b", "b", "b"])
    b = day_bootstrap(x, d)
    assert b["days"] == 2 and b["mean"] == 2.0
    assert b["ci95"] == [1.0, 3.0]                   # resampled days: {a,a} -> 1, {a,b} -> 2, {b,b} -> 3
    assert abs(b["se"] - np.sqrt(0.5)) < 0.02        # SD of {1, 2, 2, 3} equally likely
    assert day_bootstrap(x, d) == b                  # fixed seed
    assert np.isnan(day_bootstrap(x[:3], d[:3])["se"])


def test_rg_needs_real_gross_good_and_trades():
    """A80: real gross above the simulator's by 2 SE is a hard RG only if the real gross is > 0 by 2 SE on
    >= MIN_ORDERS orders/window; less bad than the simulator (OF16 phase 2's six flags) or too few trades is RGv."""
    from ragrl.score_real import red_flags

    def block(gross_mean, gross_se, orders):
        s = {"mean": gross_mean, "se": gross_se, "t": gross_mean / gross_se, "orders": orders, "n": 292, "sharpe_annual": None}
        return {"summary": {"seed1": s}, "gross": {"seed1": {"mean": gross_mean, "se": gross_se, "t": gross_mean / gross_se}}}

    sim = {"seed1": {"mean": -0.55, "se": 0.17}}
    ids = lambda r: ([f["id"] for f in r[0]], [f["id"] for f in r[1]])  # noqa: E731
    assert ids(red_flags(block(-0.30, 0.24, 0.7), sim, ["seed1"])) == ([], [])         # within 2 SE of the simulator: nothing
    assert ids(red_flags(block(0.08, 0.24, 0.7), sim, ["seed1"])) == ([], ["RGv"])     # OF16 neuralq_ref_v3 seed11: was a hard RG
    hard, warn = red_flags(block(0.15, 0.10, 3.0), sim, ["seed1"])                     # beats the simulator, real t 1.5: less bad
    assert not hard and [f["id"] for f in warn] == ["RGv"] and "less bad" in warn[0]["detail"]
    hard, warn = red_flags(block(0.50, 0.10, 0.5), sim, ["seed1"])                     # real t 5 on 0.5 orders/window
    assert not hard and [f["id"] for f in warn] == ["RGv"] and "too few trades" in warn[0]["detail"]
    hard, warn = red_flags(block(0.50, 0.10, 3.0), sim, ["seed1"])                     # real t 5 on 3 orders/window: a leak suspect
    assert [f["id"] for f in hard] == ["RG"] and not warn


# ---------------------------------------------------------------- the converter on a synthetic Sierra pair
def _depth_file(path, events):
    """events: (sierra_us, cmd, flags, price, qty) -> a .depth file (64-byte header, <qBBHfII records)."""
    head = b"SCDD" + struct.pack("<III", 64, 24, 1) + bytes(48)
    with open(path, "wb") as f:
        f.write(head)
        for t, cmd, fl, price, qty in events:
            f.write(struct.pack("<qBBHfII", t, cmd, fl, 1, price, qty, 0))


def _scid_file(path, records):
    """records: (sierra_us, price, volume, bid volume, ask volume) -> a .scid file (56-byte header, <q4f4I)."""
    head = b"SCID" + struct.pack("<II", 56, 40) + bytes(44)
    with open(path, "wb") as f:
        f.write(head)
        for t, p, vol, bv, av in records:
            f.write(struct.pack("<q4f4I", t, 0.0, p + 25, p - 25, p, 1, vol, bv, av))


P = 758000.0  # 7580.00 x 100


# 08:30 CT written down independently of ragrl.real (the code under test): CDT = UTC-5, CST = UTC-6
OPEN_UTC = {"2026-09-14": "2026-09-14T13:30:00+00:00", "2026-11-03": "2026-11-03T14:30:00+00:00"}


def _synthetic_day(day, tmp, flags=True, beat_changes=True):
    from datetime import datetime

    from ragrl import real
    e = real.day_entry(day, real.load_manifest())
    o = int(datetime.fromisoformat(OPEN_UTC[day]).timestamp())
    o_us = o * 1_000_000 + SC_EPOCH_US
    us = lambda s: o_us + int(round(s * 1e6))  # noqa: E731
    ev = [(us(-60), 1, 0, 0.0, 0)]
    ev += [(us(-60), 2, 0, P - 25 * j, q) for j, q in enumerate((10, 20, 30))]
    ev += [(us(-60), 3, 0, P + 25 + 25 * j, q) for j, q in enumerate((11, 21, 31))]
    ev[-1] = ev[-1][:2] + (1,) + ev[-1][3:]                  # end of the snapshot batch
    ev += [(us(0.5), 4, 1, P, 15),                           # bid size 10 -> 15
           (us(1.0), 3, 1, P, 7),                            # an ask at the bid: locked, dropped
           (us(1.2), 7, 1, P, 0),                            # gone again: the book equals the last Q, nothing written
           (us(1.3), 6, 1, P, 0), (us(1.3), 2, 1, P, 40),    # two flagged events, one timestamp: one Q
           (us(3.5), 5, 1, P + 25, 5)]                       # after a 2.2 s silent book
    t = 4.0
    k = 0
    while t < 23400:                                         # a heartbeat: level 5 of the bid alternates 50 / 51
        ev.append((us(t), 4, 1, P - 100, 50 + k % 2 if beat_changes else 50))
        t += 0.5
        k += 1
    ev.append((us(23401), 4, 1, P - 100, 60))                # after the close: not exported
    tr = [(us(-10), P, 1, 1, 0),                             # before the open: never read
          (us(0.2), P + 25, 2, 0, 2), (us(0.7), P, 3, 3, 0), (us(1.25), P + 75, 1, 0, 1), (us(1.3), P, 1, 1, 0),
          (us(2.0), P + 25, 4, 1, 3),                        # both volumes: two lines
          (us(2.5), P + 25, 1, 0, 0), (us(2.6), P, 1, 0, 0),  # no volume side: from the quote
          (us(23400), P, 9, 9, 0), (us(86400 * 2), P, 9, 9, 0)]  # at the close / another day: never read
    dd, sd = tmp / "depth", tmp / "scid"
    dd.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)
    if not flags:  # Sierra without end-of-batch flags
        ev = [x[:2] + (0,) + x[3:] for x in ev]
    _depth_file(dd / e["depth"], ev)
    _scid_file(sd / e["scid"], tr)
    return dd, sd, k


EXPECTED_HEAD = """Q,0.000000,7580.00,7580.25,10,20,30,0,0,11,21,31,0,0
T,0.200000,7580.25,2,B
Q,0.500000,7580.00,7580.25,15,20,30,0,0,11,21,31,0,0
T,0.700000,7580.00,3,S
T,1.250000,7580.75,1,B
T,1.300000,7580.00,1,S
Q,1.300000,7580.00,7580.25,40,20,30,0,0,11,21,31,0,0
T,2.000000,7580.25,3,B
T,2.000000,7580.25,1,S
T,2.500000,7580.25,1,B
T,2.600000,7580.00,1,S
Q,3.500000,7580.00,7580.25,40,20,30,0,0,5,21,31,0,0
Q,4.000000,7580.00,7580.25,40,20,30,0,50,5,21,31,0,0
Q,4.500000,7580.00,7580.25,40,20,30,0,51,5,21,31,0,0
"""


def _convert(day, tmp, *extra, **synth):
    dd, sd, beats = _synthetic_day(day, tmp, **synth)
    out = tmp / "out"
    p = subprocess.run([sys.executable, str(TOOL), day, "--depth-dir", str(dd), "--scid-dir", str(sd), "--out-dir", str(out), *extra],
                       capture_output=True, text=True, timeout=300)
    return p, out / day, beats


def test_converter_round_trip_and_dst(tmp_path):
    side = {}
    for day in ("2026-09-14", "2026-11-03"):  # CDT and CST: the same CT session, so the same bytes
        p, d, beats = _convert(day, tmp_path / day)
        assert p.returncode == 0, p.stderr
        text = (d / "ticks.csv").read_text(encoding="ascii")
        assert text.startswith(EXPECTED_HEAD), text[:800]
        lines = text.splitlines()
        assert lines[-2] == "Q,23399.500000,7580.00,7580.25,40,20,30,0,%d,5,21,31,0,0" % (50 + (beats - 1) % 2)
        assert lines[-1] == lines[-2].replace("23399.500000", "23399.950000")   # the close's bucket, on the grid
        s = json.loads((d / "ticks.json").read_text(encoding="utf-8"))
        assert s["csv_sha256"] == hashlib.sha256((d / "ticks.csv").read_bytes()).hexdigest()
        assert s["counts"]["trades"] == 8 and s["counts"]["trade_records"] == 7 and s["counts"]["volume"] == 13
        assert s["counts"]["quotes"] == 4 + beats + 1 == len(lines) - 8
        assert s["scid"]["records"] == 7                         # only [08:30, 15:00) CT was read
        assert s["dropped"]["crossed_or_locked"] == 1 and s["dropped"]["trades_unknown_side"] == 0
        assert s["gaps_over_1s"]["count"] == 1 and s["gaps_over_1s"]["intervals_s"] == [[1.3, 3.5]]
        assert s["trades_outside_quote"]["count"] == 1 and s["trades_outside_quote"]["above_ask"] == 1
        assert s["counts"]["trades_book_stale"] == 3  # > 1 s after the last book event: 0.2 s (the pre-open snapshot), 2.5, 2.6
        assert s["span"]["start_ct"] == "08:30:00.000" and s["span"]["end_ct"] == "15:00:00.000" and s["rth_missing_seconds"] == 0
        assert s["padded_close"] is True and s["time_unit"] == "s" and s["role"] == "dev" and s["batching"] == "flags"
        assert s["t0_unix"] == int(__import__("datetime").datetime.fromisoformat(OPEN_UTC[day]).timestamp())
        side[day] = s
    a, b = side["2026-09-14"], side["2026-11-03"]
    assert a["csv_sha256"] == b["csv_sha256"]
    assert (a["utc_offset_hours"], b["utc_offset_hours"]) == (-5, -6)
    assert a["contract"] == "ESU26" and b["contract"] == "ESZ26"


def test_converter_batching_fallback_and_empty_tape(tmp_path):
    """Without end-of-batch flags the converter batches by timestamp (same tape here) and says so; a book that
    changes almost never against many depth events is refused (exit 3, nothing written)."""
    ref, d0, _ = _convert("2026-11-03", tmp_path / "ref")
    p, d, _ = _convert("2026-11-03", tmp_path / "noflags", flags=False)
    assert ref.returncode == 0 and p.returncode == 0, p.stderr
    s = json.loads((d / "ticks.json").read_text(encoding="utf-8"))
    assert s["batching"].startswith("timestamp (fallback")
    assert s["csv_sha256"] == json.loads((d0 / "ticks.json").read_text(encoding="utf-8"))["csv_sha256"]
    p, d, _ = _convert("2026-11-03", tmp_path / "flat", beat_changes=False)
    assert p.returncode == 3 and "implausibly few quotes" in p.stderr, p.stderr
    assert not (d / "ticks.csv").exists() and not (d / "ticks.json").exists() and not (d / "ticks.csv.part").exists()


def test_converter_refuses_test_confirm_and_excluded_days(tmp_path):
    orch = ("--i-am-the-orchestrator",)
    for day, extra in (("2026-09-08", ()), ("2026-11-12", ()), ("2026-11-04", ()), ("2026-11-05", ("--role-ok", "confirm", *orch)),
                       ("2026-10-01", ()), ("2026-09-08", ("--role-ok", "test")), ("2026-11-12", ("--role-ok", "confirm")),
                       ("2026-11-03", ("--role-ok", "test"))):
        out = tmp_path / "out"
        p = subprocess.run([sys.executable, str(TOOL), day, "--depth-dir", str(tmp_path / "none"), "--scid-dir", str(tmp_path / "none"),
                            "--out-dir", str(out), *extra], capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "refused" in p.stderr, (day, p.stderr)
        assert not out.exists()


# ---------------------------------------------------------------- the replay (fresh scorer processes)
def _q(t, bid, ask, bs=10, asz=10):
    return f"Q,{t:.6f},{bid:.2f},{ask:.2f}," + ",".join([str(bs)] * 5) + "," + ",".join([str(asz)] * 5) + "\n"


REPLAY_PRELUDE = """
import sys
sys.path.insert(0, sys.argv[2])
from ragrl import score_real
R.open_sealed("real", "pytest synthetic tape")
tape = R.TickTape.from_csv(sys.argv[1], time_unit="s")
p = R.ReplayParams(decide_seconds=1.0, window_seconds=10.0, warmup_seconds=2.0, latency_buckets=1, fee_dollars=1.25)
assert score_real.windows_on_tape(tape, p) == 1 == R.VecReplayEnv(tape, p, [0]).windows_on_tape, tape.size
def scripted(actions):
    env = R.VecReplayEnv(tape, p, [0])
    env.reset()
    total = 0.0
    for a in actions:
        _, r, d = env.step(np.array([a], dtype=np.int64))
        total += float(r[0])
    assert d[0]
    st = env.episode_stats
    return round(float(st["net_pnl_ticks"][0]), 9), int(st["orders"][0]), round(float(st["fees_ticks"][0]), 9), round(total, 9)
"""


def test_replay_known_pnl(tmp_path, run_fresh):
    """Long 1 at 100.25, sell 2 at 100.75 (short 1), cover at 100.50: +2 + 1 ticks, 4 lots x $1.25 = 0.4 ticks of fees."""
    k = [0, 2, 4, 3, 1, 1, 0, 0, 0, 0]  # the bid in ticks above 100.00 from second s + 0.05 (bucket 20 + 10 s)
    csv = _q(0.0, 100.0, 100.25) + "".join(_q(2.05 + s, 100 + 0.25 * k[s], 100.25 + 0.25 * k[s]) for s in range(10))
    csv += _q(12.15, 100 + 0.25 * k[9], 100.25 + 0.25 * k[9])
    (tmp_path / "t.csv").write_text(csv)
    out = run_fresh(REPLAY_PRELUDE + """
print("RES", scripted([2, 2, 2, 0, 0, 1, 1, 1, 1, 1]))
r = score_real.play(score_real.Flat(), tape, p, [0]); print("FLAT", float(r["net_pnl_ticks"][0]), float(r["orders"][0]))
""", str(tmp_path / "t.csv"), str(PYDIR))
    assert "RES (2.6, 3, 0.4, 2.6)" in out, out
    assert "FLAT 0.0 0.0" in out, out


def test_no_lookahead_fill_is_one_bucket_after_the_decision(tmp_path, run_fresh):
    """The decision sees bucket 19 (ask 100.25); the fill must use bucket 20's end quote (ask 101.00), not
    bucket 19's (lookahead-free but stale: +6 ticks) or bucket 21's (ask 102.00: -1 tick). Exit at 101.75."""
    csv = _q(0.0, 100.0, 100.25) + _q(2.05, 100.75, 101.0) + _q(2.15, 101.75, 102.0) + _q(12.15, 101.75, 102.0)
    (tmp_path / "t.csv").write_text(csv)
    out = run_fresh(REPLAY_PRELUDE + """
print("RES", scripted([2] * 10))
""", str(tmp_path / "t.csv"), str(PYDIR))
    assert "RES (2.8, 2, 0.2, 2.8)" in out, out


def test_imbalance_rule_and_baselines_on_a_flat_book(tmp_path, run_fresh):
    """Bid size 30 > ask size 10 all day, prices fixed: the L1 rule goes long, flat, long, ... (5 round trips of
    1 tick of spread + 0.2 ticks of fees); flat and delta30 (no trades: no delta) never trade."""
    csv = _q(0.0, 100.0, 100.25, 30, 10) + _q(12.15, 100.0, 100.25, 30, 10)
    (tmp_path / "t.csv").write_text(csv)
    out = run_fresh(REPLAY_PRELUDE + """
for name, pol in score_real.baselines().items():
    r = score_real.play(pol, tape, p, [0])
    print("BL", name, round(float(r["net_pnl_ticks"][0]), 9), int(r["orders"][0]))
""", str(tmp_path / "t.csv"), str(PYDIR))
    assert "BL imbalance_l1 -6.0 10" in out, out
    assert "BL flat 0.0 0" in out and "BL delta30 0.0 0" in out, out


# ---------------------------------------------------------------- manifest roles in the usage ledger
def _tiny_manifest(tmp):
    m = json.loads((REPO / "experiments" / "configs" / "of16_real_manifest.json").read_text(encoding="utf-8"))
    m["export_dir"] = str(tmp / "exports")
    m["days"] = [
        {"date": "2030-01-07", "contract": "ESH30", "depth": "x.depth", "scid": "x.scid", "rth": "full", "role": "dev", "subset": "jan", "notes": ""},
        {"date": "2030-01-08", "contract": "ESH30", "depth": "x.depth", "scid": "x.scid", "rth": "full", "role": "test", "subset": "jan", "notes": ""},
        {"date": "2030-01-09", "contract": "ESH30", "depth": "x.depth", "scid": "x.scid", "rth": "none", "role": "excluded", "subset": "jan", "notes": ""},
    ]
    m["confirm_rule"].update({"from": "2030-02-01", "contracts": [{"contract": "ESH30", "from": "2030-02-01", "to": "2030-03-01"}]})
    mp = tmp / "manifest.json"
    mp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    csv = _q(0.0, 100.0, 100.25) + "T,0.500000,100.25,1,B\n" + _q(30.0, 100.0, 100.25)
    from ragrl import real
    for day in ("2030-01-07", "2030-01-08", "2030-01-09", "2030-02-03", "2030-02-04", "2030-02-05", "2030-02-06", "2030-02-07"):
        d = tmp / "exports" / day
        d.mkdir(parents=True)
        (d / "ticks.csv").write_bytes(csv.encode())
        (d / "ticks.json").write_text(json.dumps({
            "date": day, "contract": "ESH30", "csv_sha256": hashlib.sha256(csv.encode()).hexdigest(), "rth_missing_seconds": 0,
            "span": {"start_s": 0.0}, "line_gaps_over_1s": {"max_s": 29.5}, "time_unit": "s", "t0_unix": real.rth_span(day, m)[0],
            "gaps_over_1s": {"intervals_s": [[100.0, 200.0]] if day == "2030-02-03" else []}}), encoding="utf-8")
    return mp


LEDGER_SCRIPT = """
import json, os, sys
os.environ["RAGRL_LEDGER_FILE"] = sys.argv[3]
sys.path.insert(0, sys.argv[4])
from ragrl import ledger, real
man, root, action = sys.argv[1], sys.argv[2], sys.argv[5]
kw = dict(manifest=man, root=root)
def attempt(fn):
    try:
        t = fn()
        print("OPENED", sorted(t), R.process_role(), [x[0].size for x in t.values()])
    except (PermissionError, RuntimeError, FileNotFoundError, ValueError) as e:
        print("REFUSED", type(e).__name__, str(e)[:300], "open" if R.is_open("real") else "closed")
run_name = os.environ.get("RUN_NAME", "runA")
test = lambda: real.open_tapes("test", run=run_name, config_sha256="c0ffee", **kw)
if action == "dev":
    attempt(lambda: real.open_tapes("dev", **kw))
elif action == "test":
    attempt(test)
elif action == "test_after_trainer":
    ledger.as_trainer(); attempt(test)
elif action == "test_after_validation":
    R.run_builtin(R.ScalpParams("default", 150.0, episode_seconds=10.0), R.EpisodeSource.block("validation", 0, 1), "flat", threads=1)
    attempt(test)
elif action == "test_after_dev":
    real.open_tapes("dev", **kw); attempt(test)
elif action == "test_no_run":
    attempt(lambda: real.open_tapes("test", **kw))
elif action == "test_wrong_role":
    attempt(lambda: real.open_tapes("test", days=["2030-01-07"], run="runA", config_sha256="c0ffee", **kw))
elif action == "excluded":
    attempt(lambda: real.open_tapes("excluded", **kw))
elif action == "confirm_no_orchestrator":
    attempt(lambda: real.open_tapes("confirm", claim="c1", count=1, **kw))
elif action == "confirm_no_claim":
    attempt(lambda: real.open_tapes("confirm", orchestrator=True, count=1, **kw))
elif action == "dev_then_unopened_tape":
    real.open_tapes("dev", **kw)
    attempt(lambda: {"x": (real.load_tape(os.path.join(root, "2030-01-08", "ticks.csv")), None)})
elif action == "buckets":
    t = real.open_tapes("dev", **kw)
    tape, side = t["2030-01-07"]
    b = real.load_buckets(side["csv"])
    print("BUCKETS", len(b["bid"]), tape.size, int(b["bid"][0]), int(b["ask"][-1]), int(b["b1"][3]), int(b["buy"].sum()),
          int(np.nonzero(b["buy"])[0][0]))
    attempt(lambda: {"x": (real.load_buckets(os.path.join(root, "2030-01-08", "ticks.csv")), None)})
elif action == "unopened_tape":
    attempt(lambda: {"x": (real.load_tape(os.path.join(root, "2030-01-07", "ticks.csv")), None)})
elif action.startswith("confirm_"):
    claim = action.split("_")[1]
    days = [sys.argv[6]] if len(sys.argv) > 6 else None
    attempt(lambda: real.open_tapes("confirm", claim=claim, orchestrator=True, count=1, days=days, **kw))
"""


def test_real_tapes_by_role_and_ledger(tmp_path, run_fresh):
    from ragrl import real
    mp = _tiny_manifest(tmp_path)
    ledger_file = tmp_path / "ledger_usage.jsonl"
    root = tmp_path / "exports"

    def go(action, *extra, manifest=mp, run="runA"):
        import os
        os.environ["RUN_NAME"] = run
        try:
            return run_fresh(LEDGER_SCRIPT, str(manifest), str(root), str(ledger_file), str(PYDIR), action, *extra)
        finally:
            os.environ.pop("RUN_NAME", None)

    def entries(block):
        if not ledger_file.exists():
            return []
        return [json.loads(l) for l in ledger_file.read_text(encoding="utf-8").splitlines() if json.loads(l)["block"] == block]

    sha = real.manifest_sha256(mp)
    # the Python bucket view of a tape: TickTape's bucket count, the trade in bucket 5 (t = 0.5 s); unopened tapes refused
    out = go("buckets")
    assert "BUCKETS 300 300 400 401 10 1 5" in out and "REFUSED PermissionError" in out, out
    # dev: free, twice, logged for the audit only; the process ends up a scorer
    for _ in range(2):
        assert "OPENED ['2030-01-07'] scorer [300]" in go("dev")
    assert len(entries("real_dev")) == 3
    # test: never in a trainer, after a validation environment, or after dev tapes in the same process
    for a in ("test_after_trainer", "test_after_validation", "test_after_dev", "test_no_run", "test_wrong_role", "excluded"):
        out = go(a)
        assert "REFUSED" in out and "OPENED" not in out, (a, out)
    assert "closed" in go("test_after_trainer") and entries("real_test") == []
    # a fresh scorer: opened and logged once with the manifest sha256, run and config hash
    out = go("test")
    assert "OPENED ['2030-01-08'] scorer [300]" in out, out
    (e,) = entries("real_test")
    assert (e["run"], e["config_sha256"], e["manifest_sha256"], e["days"]) == ("runA", "c0ffee", sha, ["2030-01-08"])
    assert e["tapes"]["2030-01-08"] == hashlib.sha256((root / "2030-01-08" / "ticks.csv").read_bytes()).hexdigest()
    # the same experiment again: refused, whether the run is renamed / copied or the manifest edited; still one entry
    m2 = tmp_path / "manifest_edited.json"
    m2.write_text(mp.read_text(encoding="utf-8").replace('"written": "2026-11-12"', '"written": "2026-11-13"'), encoding="utf-8")
    assert real.manifest_sha256(m2) != sha
    for kw in ({}, {"run": "runA_copy"}, {"manifest": m2}, {"run": "runB", "manifest": m2}):
        out = go("test", **kw)
        assert "REFUSED PermissionError" in out and "once per experiment" in out and "closed" in out, (kw, out)
    assert len(entries("real_test")) == 1
    # tapes load only from this process's opening: never an unopened day, never without an opening
    for a in ("dev_then_unopened_tape", "unopened_tape"):
        out = go(a)
        assert "REFUSED PermissionError" in out and "not a tape opened in this process" in out, (a, out)
    # confirm: the orchestrator's claim, once per claim, never the same day twice
    assert "REFUSED" in go("confirm_no_orchestrator") and "REFUSED" in go("confirm_no_claim")
    assert "OPENED ['2030-02-04'] scorer" in go("confirm_c1")  # 02-03 has a 100 s book outage: not a confirm day
    assert "REFUSED" in go("confirm_c1")                       # the claim was used
    assert "OPENED ['2030-02-05'] scorer" in go("confirm_c2")  # the next unused confirm day
    assert "REFUSED" in go("confirm_c3", "2030-02-04")        # a day an earlier confirmation used
    assert "REFUSED" in go("confirm_c4", "2030-02-07")        # unused, but not the next due day (02-06)
    assert "REFUSED" in go("confirm_c4", "2030-02-03")        # the outage day
    assert "OPENED ['2030-02-06'] scorer" in go("confirm_c4", "2030-02-06")
    assert [x["claim"] for x in entries("real_confirm")] == ["c1", "c2", "c4"]
    # a sidecar with the wrong time unit or origin is refused before anything is opened
    sc = root / "2030-01-07" / "ticks.json"
    good = sc.read_text(encoding="utf-8")
    for key, bad in (("time_unit", "ns"), ("t0_unix", 0)):
        sc.write_text(json.dumps({**json.loads(good), key: bad}), encoding="utf-8")
        out = go("dev")
        assert "REFUSED RuntimeError" in out and key in out and "closed" in out, (key, out)
    sc.write_text(good, encoding="utf-8")
    # a changed export is refused before anything is opened
    with open(root / "2030-01-07" / "ticks.csv", "ab") as f:
        f.write(b"T,40.000000,100.25,1,B\n")
    out = go("dev")
    assert "REFUSED RuntimeError" in out and "sha256" in out and "closed" in out, out


# ---------------------------------------------------------------- score-real end to end (synthetic dev days)
def _random_tape(seed: int, seconds: float = 2300.0) -> str:
    rng = np.random.default_rng(seed)
    out, bid = [], 4000.0
    for i in range(int(seconds * 10)):
        t = i * 0.1
        if rng.random() < 0.3:
            bid += 0.25 * rng.choice([-1, 1])
        for _ in range(rng.poisson(1.5)):
            buy = rng.random() < 0.5
            out.append(f"T,{t + 0.01:.6f},{bid + 0.25 if buy else bid:.2f},{1 + rng.poisson(2)},{'B' if buy else 'S'}\n")
        sizes = [str(5 + rng.poisson(20)) for _ in range(10)]
        out.append(f"Q,{t + 0.05:.6f},{bid:.2f},{bid + 0.25:.2f}," + ",".join(sizes) + "\n")
    return "".join(out)


@pytest.mark.smoke
def test_score_real_dev_end_to_end(tmp_path):
    """`ragrl score-real` on a copy of a sealed OF15 run with two synthetic dev days: results under RUN/real/,
    sealed.json untouched, every policy and both blocks (base, stress), the day bootstrap and the subsets."""
    import os
    import shutil
    runs = sorted((REPO / "build-avx512" / "rl" / "runs").glob("of15_ppo_v4-*"))
    runs = [r for r in runs if (r / "sealed.json").exists()]
    if not runs:
        pytest.skip("no sealed of15_ppo_v4 run on this machine")
    run = tmp_path / runs[0].name
    shutil.copytree(runs[0], run, ignore=shutil.ignore_patterns("live.jsonl", "recheck", "*.npz"))
    sealed_sha = hashlib.sha256((run / "sealed.json").read_bytes()).hexdigest()
    m = json.loads((REPO / "experiments" / "configs" / "of16_real_manifest.json").read_text(encoding="utf-8"))
    m["days"] = [{"date": d, "contract": "ESH30", "depth": "x", "scid": "x", "rth": "full", "role": "dev", "subset": s, "notes": ""}
                 for d, s in (("2030-01-07", "sep"), ("2030-01-08", "nov"))]
    m["confirm_rule"]["from"] = "2030-02-01"
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    root = tmp_path / "exports"
    from ragrl import real
    for i, d in enumerate(("2030-01-07", "2030-01-08")):
        csv = _random_tape(i).encode()
        (root / d).mkdir(parents=True)
        (root / d / "ticks.csv").write_bytes(csv)
        (root / d / "ticks.json").write_text(json.dumps({"date": d, "contract": "ESH30", "csv_sha256": hashlib.sha256(csv).hexdigest(),
                                                         "span": {"start_s": 0.05}, "rth_missing_seconds": 0, "time_unit": "s",
                                                         "t0_unix": real.rth_span(d, m)[0],
                                                         "gaps_over_1s": {"intervals_s": []}, "line_gaps_over_1s": {"max_s": 0.1}}))
    env = {**os.environ, "PYTHONPATH": str(PYDIR), "RAGRL_REAL_DIR": str(root), "RAGRL_LEDGER_FILE": str(tmp_path / "ledger.jsonl"),
           "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, "-m", "ragrl", "score-real", str(run), "--days", "dev", "--manifest", str(mp), "--stress",
                        "--allow-env-change"], capture_output=True, text=True, timeout=1800, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "dev: not a score" in p.stdout
    (out,) = sorted((run / "real").glob("dev-*"))
    r = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert r["status"] == "DEV_NOT_A_SCORE" == (out / "STATUS").read_text().strip() and r["note"] == "dev: not a score"
    assert hashlib.sha256((run / "sealed.json").read_bytes()).hexdigest() == sealed_sha
    names = {*(f"seed{s}" for s in range(1, 13)), "ensemble", "mean_of_seeds", "flat", "delta30", "imbalance_l1"}
    for b, lat, fee in (("base", 1, 1.5), ("stress", 3, 3.0)):
        B = r["blocks"][b]
        assert B["params"]["latency_buckets"] == lat and B["params"]["fee_dollars"] == fee and B["params"]["warmup_seconds"] == 1050
        assert (out / f"windows_{b}.npz").exists()
        assert B["windows"] == 4 and set(B["summary"]) == names and B["summary"]["flat"]["mean"] == 0
        assert B["day_bootstrap"]["ensemble"]["days"] == 2 and set(B["paired"]) == {"flat", "delta30"}
        assert {k: v["windows"] for k, v in B["subsets"].items()} == {"sep": 2, "nov": 2}
    assert [e["block"] for e in map(json.loads, (tmp_path / "ledger.jsonl").read_text().splitlines())] == ["real_dev"]


# ---------------------------------------------------------------- recorded tapes load only through real.py
# These test files load synthetic tapes (or check a refusal) with the binding directly; nothing else may.
DIRECT_TAPE_TESTS = {"test_bit_identity.py": "refdump's synthetic tape", "test_real.py": "synthetic tapes",
                     "test_ledger.py": "open_sealed('real') refused after forbid_sealed (no tape)"}


def test_tapes_load_only_through_real_py():
    pats = ("TickTape.from_csv", 'open_sealed("real"', "open_sealed('real'")
    files = [*(REPO / "python").rglob("*.py"), *(REPO / "tools").rglob("*.py"), *(REPO / "experiments").rglob("*.py")]
    bad = []
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        if rel in ("python/ragrl/real.py", "python/ragrl/ledger.py") or (rel.startswith("python/tests/") and f.name in DIRECT_TAPE_TESTS):
            continue
        if any(p in f.read_text(encoding="utf-8", errors="replace") for p in pats):
            bad.append(rel)
    assert not bad, bad


# ---------------------------------------------------------------- every zoo agent through the score-real policy path
ZOO_SCRIPT = """
import os, sys
from pathlib import Path
os.environ["RAGRL_LEDGER_FILE"] = sys.argv[4]
import ragrl
import numpy as np
from ragrl import real, registry, runner, score_real
run = Path(sys.argv[1])
runner.use_run_module(run)
cfg = registry.read_json(run / "config.json")
warm = score_real.check_episodes(cfg["env"], real.load_manifest(sys.argv[2]))
allp = runner.learned_policies(run, cfg)
s0 = "seed%d" % cfg["train"]["seeds"][0]
pols = {s0: allp[s0], "ensemble": allp["ensemble"]}
score_real.check_policies({**pols, **score_real.baselines()})
tapes = real.open_tapes("dev", manifest=sys.argv[2], root=sys.argv[3])
res, labels, ex = score_real.play_days(pols, tapes, score_real.replay_params(warm))
ok = all(np.isfinite(v["net_pnl_ticks"]).all() and len(v["orders"]) == len(labels) for v in res.values())
print("ZOO", cfg["agent"]["kind"], len(labels), ok, {k: float(v["orders"].mean()) for k, v in res.items()})
"""


@pytest.mark.smoke
def test_every_zoo_agent_plays_through_score_real(tmp_path):
    """One sealed OF15 run per agent kind: its first seed and its ensemble are dry-run (check_policies) and then
    play a synthetic dev day through real.open_tapes + score_real.play_days, in a process importing RUN/bin."""
    import os

    from ragrl import real
    kinds = {}
    for r in sorted((REPO / "build-avx512" / "rl" / "runs").glob("of15_*")):
        if (r / "sealed.json").exists():
            kinds[json.loads((r / "config.json").read_text(encoding="utf-8"))["agent"]["kind"]] = r
    if not kinds:
        pytest.skip("no sealed OF15 runs on this machine")
    m = json.loads((REPO / "experiments" / "configs" / "of16_real_manifest.json").read_text(encoding="utf-8"))
    m["days"] = [{"date": "2030-01-07", "contract": "ESH30", "depth": "x", "scid": "x", "rth": "full", "role": "dev", "subset": "jan",
                  "notes": ""}]
    m["confirm_rule"]["from"] = "2030-02-01"
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    root = tmp_path / "exports"
    csv = _random_tape(7).encode()
    (root / "2030-01-07").mkdir(parents=True)
    (root / "2030-01-07" / "ticks.csv").write_bytes(csv)
    (root / "2030-01-07" / "ticks.json").write_text(json.dumps({
        "date": "2030-01-07", "contract": "ESH30", "csv_sha256": hashlib.sha256(csv).hexdigest(), "span": {"start_s": 0.05},
        "rth_missing_seconds": 0, "time_unit": "s", "t0_unix": real.rth_span("2030-01-07", m)[0],
        "gaps_over_1s": {"intervals_s": []}, "line_gaps_over_1s": {"max_s": 0.1}}))
    env = {k: v for k, v in os.environ.items() if k != "RAGOF_DIR"}
    env.update({"PYTHONPATH": str(PYDIR), "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    from ragrl.agents import ZOO
    assert set(kinds) == set(ZOO), set(ZOO) - set(kinds)
    for kind, run in sorted(kinds.items()):
        p = subprocess.run([sys.executable, "-c", ZOO_SCRIPT, str(run), str(mp), str(root), str(tmp_path / f"ledger_{kind}.jsonl")],
                           capture_output=True, text=True, timeout=900, env=env)
        assert p.returncode == 0, (kind, p.stdout[-2000:], p.stderr[-3000:])
        assert f"ZOO {kind} 2 True" in p.stdout, (kind, p.stdout)
