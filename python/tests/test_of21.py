"""OF21 forward test plumbing on synthetic data only (no Sierra file is read): the ledger's forward looks, the
overnight guard, the full-day rule, day selection around a roll, the sequential boundaries and the frozen runner."""
import importlib.util
import math
from datetime import date

import numpy as np
import pytest

from conftest import REPO

spec = importlib.util.spec_from_file_location("of21_forward", REPO / "experiments" / "of21_forward.py")
F21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F21)
ledger, real, SB, LV = F21.ledger, F21.real, F21.SB, F21.LV
SHA = "0" * 64


@pytest.fixture
def fresh_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGRL_LEDGER_FILE", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(ledger, "_forward_days", set())
    return tmp_path / "ledger.jsonl"


def _open(look, days, fp="f", **kw):
    ledger.open_forward("OF21", look, days, manifest_sha256=SHA, runner_sha256=SHA, fingerprint=fp, orchestrator=kw.pop("orchestrator", True),
                        **kw)


def test_open_forward_rules(fresh_ledger):
    d = ["2026-11-12", "2026-11-13", "2026-11-16"]
    with pytest.raises(PermissionError, match="orchestrator"):
        _open(1, d, orchestrator=False)
    with pytest.raises(PermissionError, match="needs look 1"):
        _open(2, d)
    with pytest.raises(ValueError):
        _open(1, d[::-1])
    assert not fresh_ledger.exists()  # refusals log nothing
    _open(1, d[:2])
    with pytest.raises(PermissionError, match="never repeated"):
        _open(1, d[:2], fp="other")  # a rerun is allowed only with the same fingerprint (a crash retry)
    _open(1, d[:2])
    assert [u["retry"] for u in ledger.forward_openings("OF21")] == [False, True]
    with pytest.raises(PermissionError, match="extend"):
        _open(2, ["2026-11-13", "2026-11-16", "2026-11-17"])
    with pytest.raises(PermissionError, match="extend"):
        _open(2, d[:2])
    _open(2, d)
    looks = ledger.forward_openings("OF21")
    assert [u["look"] for u in looks] == [1, 1, 2] and looks[2]["days"] == d
    assert all(ledger.forward_day_allowed(x) for x in d) and not ledger.forward_day_allowed("2026-11-17")
    # forward days count as used for any later confirmation
    assert set(d) <= ledger.used_real_confirm_days()


def test_open_forward_refuses_confirm_days(fresh_ledger):
    ledger.log_usage({"time": "t", "block": "real_confirm", "days": ["2026-11-13"], "claim": "X", "dry_run": False})
    with pytest.raises(PermissionError, match="real_confirm"):
        _open(1, ["2026-11-12", "2026-11-13"])
    _open(1, ["2026-11-12", "2026-11-16"])


def test_overnight_reads_past_the_cutoff_only_for_opened_days(fresh_ledger, tmp_path):
    with pytest.raises(RuntimeError, match="never read"):
        LV.overnight("2026-11-12", "ESZ26", scid_dir=tmp_path)
    _open(1, ["2026-11-12"])
    with pytest.raises(FileNotFoundError):  # past the guard; the synthetic folder has no file
        LV.overnight("2026-11-12", "ESZ26", scid_dir=tmp_path)
    with pytest.raises(RuntimeError, match="never read"):
        LV.overnight("2026-11-13", "ESZ26", scid_dir=tmp_path)


M = {"rth": {"open": "08:30", "close": "15:00"}}


def _records(day, end="15:00", gap=None, vol=1):
    """Tick-style records: a pair of trades (same second) every 20 s of RTH, optionally a hole [gap[0], gap[1]) s."""
    t0 = real.sierra_us(real.ct_wall(day, "08:30"))
    t1 = real.sierra_us(real.ct_wall(day, end))
    secs = np.arange(1, (t1 - t0) // 1_000_000, 20)
    if gap:
        secs = secs[(secs < gap[0]) | (secs >= gap[1])]
    t = np.repeat(t0 + secs * 1_000_000, 2) + np.tile([0, 1], len(secs))
    r = np.zeros(len(t), SB.REC)
    r["t"], r["c"], r["l"], r["h"] = t, 600_000, 600_000, 600_025
    r["n"], r["v"], r["av"] = 1, vol, vol
    return r


def _scid(path, recs):
    r = np.concatenate(recs)
    head = bytearray(56)
    head[4:8] = (56).to_bytes(4, "little")
    path.write_bytes(bytes(head) + r.tobytes())


def test_full_day_rule():
    day = "2026-12-07"
    t0, t1 = F21.rth_us(day, M)
    assert F21.full_day(_records(day), t0, t1, "tick") == ""
    assert "gap" in F21.full_day(_records(day, gap=(3600, 3700)), t0, t1, "tick")
    assert "half day" in F21.full_day(_records(day, end="12:15"), t0, t1, "tick")
    assert "bundle" in F21.full_day(_records(day), t0, t1, "bundle")
    late = _records(day)
    assert "late" in F21.full_day(late[late["t"] > t0 + 90_000_000], t0, t1, "tick")


def test_scan_roll_prior_session_and_half_day(tmp_path):
    # ESZ26 (expiry 2026-12-18) front Fri-Wed (Mon has a 2-min hole, Wed is a half day); ESH27 recorded from Mon (thin, none
    # on Fri: a roll-window day without the next contract is not scored), front Thu-Fri
    z = [_records("2026-12-04", vol=5), _records("2026-12-07", gap=(7200, 7330), vol=5), _records("2026-12-08", vol=5),
         _records("2026-12-09", end="12:15", vol=5), _records("2026-12-10", vol=1), _records("2026-12-11", vol=1)]
    h = [_records("2026-12-07", vol=1), _records("2026-12-08", vol=1), _records("2026-12-09", end="12:15", vol=1),
         _records("2026-12-10", vol=9), _records("2026-12-11", vol=9)]
    _scid(tmp_path / "ESZ26_FUT_CME.scid", z)
    _scid(tmp_path / "ESH27_FUT_CME.scid", h)
    files = F21.open_files(tmp_path)
    try:
        sc = F21.scan(F21.weekdays("2026-12-03", "2026-12-11"), files, M)
        assert F21.prior_session("2026-12-10", "ESH27", files, M)[0] == "2026-12-09"  # a half day is a prior session
        assert F21.prior_session("2026-12-07", "ESH27", files, M)[0] is None
    finally:
        for s in files.values():
            s.close()
    got = {d: (c, w) for d, c, w, _h in sc}
    assert list(got) == ["2026-12-03", "2026-12-04", "2026-12-07", "2026-12-08", "2026-12-09", "2026-12-10", "2026-12-11"]
    assert got["2026-12-03"] == (None, "no RTH trades")
    assert got["2026-12-04"] == ("ESZ26", "roll window: ESH27 not recorded tick by tick")
    assert got["2026-12-07"] == ("ESZ26", "a gap over 60 s")
    assert got["2026-12-08"] == ("ESZ26", "")
    assert got["2026-12-09"][0] == "ESZ26" and "half day" in got["2026-12-09"][1]
    assert got["2026-12-10"] == ("ESH27", "") and got["2026-12-11"] == ("ESH27", "")
    assert F21.expiry("ESZ26") == date(2026, 12, 18) and F21.expiry("ESH27") == date(2027, 3, 19)


def test_status_is_fixed_by_the_first_scan():
    sc = [("d1", "ESZ26", "", "h1"), ("d2", "ESZ26", "", "h2"), ("d3", "ESZ26", "", "h3")]
    scans = [{"days": {"d1": {"contract": "ESZ26", "status": "", "sha256": "old"},
                       "d2": {"contract": "ESZ26", "status": "a gap over 60 s", "sha256": "x"}}},
             {"days": {"d2": {"contract": "ESZ26", "status": "", "sha256": "h2"}}}]
    rows, changed = F21.fix_status(sc, scans)
    assert [w for _d, _c, w, _h in rows] == ["", "a gap over 60 s", ""]  # a backfilled day stays not full
    assert changed == ["d1"]  # a full day whose records changed is a red flag


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def test_sequential_boundaries():
    a = F21.ALPHA
    assert abs(a - (1 - _phi(2.0))) < 1e-12
    t = np.array([F21.LOOKS[k] for k in (1, 2, 3)]) / F21.LOOKS[3]
    lo, hi = 0.0, 10.0
    for _ in range(100):
        z = (lo + hi) / 2
        lo, hi = (z, hi) if _phi(z) < 1 - a / 2 else (lo, z)
    spend = [2 - 2 * _phi(z / math.sqrt(x)) for x in t]
    lo, hi = 0.0, 10.0
    for _ in range(100):  # look 1 is exact
        b = (lo + hi) / 2
        lo, hi = (b, hi) if 1 - _phi(b) > spend[0] else (lo, b)
    assert abs(b - F21.BOUNDS[1]) < 0.001
    rng = np.random.default_rng(7)
    n = 400_000
    S = np.cumsum(rng.standard_normal((n, 3)) * np.sqrt(np.diff(np.r_[0, t])), 1)
    Z = S / np.sqrt(t)
    hit = (Z >= np.array([F21.BOUNDS[k] for k in (1, 2, 3)])).cumsum(1) > 0
    # each look spends what Lan-DeMets O'Brien-Fleming allows (Monte Carlo tolerance)
    assert abs(hit[:, 1].mean() - spend[1]) < 0.0004
    assert abs(hit[:, 2].mean() - spend[2]) < 0.0015
    # the deadline look's boundary spends what is left; at the planned 250 days it is the planned final boundary
    assert abs(F21.final_boundary([60, 125], [F21.BOUNDS[1], F21.BOUNDS[2]], 250, paths=400_000, seed=3) - F21.BOUNDS[3]) < 0.02
    assert F21.final_boundary([60, 125], [F21.BOUNDS[1], F21.BOUNDS[2]], 180, paths=400_000, seed=3) < F21.BOUNDS[3] + 0.05


def test_runner_refusals_and_freeze(tmp_path, capsys):
    assert F21.main(["--count"]) == 2
    assert F21.main(["--look", "1"]) == 2
    tomorrow = date.fromordinal(date.today().toordinal() + 1).isoformat()  # a session not over is never scanned
    assert F21.main(["--count", "--i-am-the-orchestrator", "--until", tomorrow, "--scid-dir", str(tmp_path)]) == 2
    assert F21.main(["--count", "--i-am-the-orchestrator", "--until", "2026-11-11", "--scid-dir", str(tmp_path)]) == 2
    p = tmp_path / "plan.md"
    p.write_text("`experiments/of21_forward.py` sha256: `" + "a" * 64 + "`\n", encoding="utf-8")
    assert F21.frozen_sha256(p) == {"experiments/of21_forward.py": "a" * 64}
    assert set(F21.freeze_problems(p)) == set(F21.FROZEN)
    # the committed code is the code frozen in the plan (an edit to any of these files fails here and refuses every look)
    assert F21.freeze_problems() == []
