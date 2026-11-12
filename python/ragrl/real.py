"""Recorded ES (OF16): the day manifest, the RTH clock, and opening tapes by role.

The manifest (experiments/configs/of16_real_manifest.json) is the plan's day table
as data: per day the contract, the Sierra file names and the role (test, dev,
excluded); confirm days are a rule (every full front-month RTH day from
2026-11-12 on), not a list. tools/sierra_to_ticks.py exports one day to
EXPORT_DIR/<date>/ticks.csv (tick_csv.hpp format, time = seconds since 08:30 CT,
time_unit "s") with a ticks.json sidecar; open_tapes() checks every sidecar and
CSV sha256 BEFORE anything is opened, then opens through ledger.open_real().

Process roles (C++ seed ledger + ledger.open_real):
    every tape needs the C++ "real" block open, and opening it (ledger.py) makes the
    process a C++ "scorer": it can no longer build train / validation / pairing
    environments, and a trainer (or a forbid_sealed() process) can never open it.
    dev       a dev-diagnostic process opens dev tapes freely and so ends up a
              scorer. It must never train or select; it can still build viz-block
              simulator environments (neutral: look, never select), which is how a
              diagnostic compares real features with the simulator's. Logged as
              real_dev in the usage ledger (audit only, never refused).
    seen      a test day after its scoring (manifest "was": "test", A81): opened
              freely like dev for exploration and training (OF17, OF16 phase 3),
              logged as real_seen; never scored or claimed on again.
    test      `ragrl score-real RUN --days test` only: a fresh process (role
              "none", no real tape opened before), the run name and its config
              sha256; logged as real_test with the manifest sha256 and the tapes'
              sha256; a second opening for the same run + manifest is refused.
    confirm   orchestrator only with a claim id never used before (as 950M+);
              the days are the first full confirm-rule exports no earlier
              confirmation used.

Nothing at module level imports numpy or the binding (tools/sierra_to_ticks.py
imports this module for the manifest and the clock).
"""
import hashlib
import json
import os
from datetime import date as _date, datetime, timezone
from pathlib import Path

from . import ledger, native

MANIFEST = native.REPO / "experiments" / "configs" / "of16_real_manifest.json"
ROLES = ("test", "dev", "seen", "excluded", "confirm")
SC_EPOCH_US = 25569 * 86400 * 1_000_000  # Sierra time zero (1899-12-30) to the Unix epoch, microseconds
UTC = timezone.utc
TIME_UNIT = "s"  # export times: seconds since 08:30 CT of the day
TAPE = {"tick_size": 0.25, "bucket_seconds": 0.1, "max_gap_seconds": 60.0}
# A window (with its warm-up) that overlaps a book gap longer than this (no depth event: a recording
# outage, e.g. 2026-11-03 09:00:34-09:10:44 CT) is not played; ES RTH never pauses 5 s by itself.
BOOK_GAP_EXCLUDE_S = 5.0


# ---------------------------------------------------------------- manifest
def manifest_path(path=None) -> Path:
    return Path(path) if path else MANIFEST


def manifest_sha256(path=None) -> str:
    return hashlib.sha256(manifest_path(path).read_bytes()).hexdigest()


def load_manifest(path=None) -> dict:
    p = manifest_path(path)
    m = json.loads(p.read_text(encoding="utf-8"))
    for d in m["days"]:
        if d["role"] not in ROLES:
            raise ValueError(f"{p}: day {d['date']}: unknown role {d['role']!r}")
    dates = [d["date"] for d in m["days"]]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{p}: a date is listed twice")
    cf = m["confirm_rule"]["from"]
    if any(d >= cf for d in dates):
        raise ValueError(f"{p}: listed days must be before the confirm rule's start {cf}")
    return m


def export_root(manifest: dict, root=None) -> Path:
    if root:
        return Path(root)
    r = Path(manifest["export_dir"])
    return r if r.is_absolute() else native.REPO / r


def day_entry(day: str, manifest: dict) -> dict:
    """The manifest's entry for a CT calendar day: a listed day, a confirm-rule day, or role 'unlisted'."""
    day = str(_date.fromisoformat(day))
    for d in manifest["days"]:
        if d["date"] == day:
            return dict(d)
    cr = manifest["confirm_rule"]
    if day >= cr["from"]:
        for c in cr["contracts"]:
            if c["from"] <= day <= c["to"]:
                k = c["contract"]
                return {"date": day, "contract": k, "depth": f"{k}_FUT_CME.{day}.depth", "scid": f"{k}_FUT_CME.scid", "rth": "full",
                        "role": "confirm", "subset": cr.get("subset", "confirm"), "notes": cr["rule"]}
        return {"date": day, "contract": None, "role": "unlisted", "notes": "after the confirm rule's contract table"}
    return {"date": day, "contract": None, "role": "unlisted", "notes": "not in the manifest"}


def role_of(day: str, manifest: dict) -> str:
    return day_entry(day, manifest)["role"]


def listed_days(role: str, manifest: dict) -> list:
    return [d["date"] for d in manifest["days"] if d["role"] == role]


# ---------------------------------------------------------------- RTH clock (US Central)
def _nth_sunday(y: int, m: int, n: int) -> int:
    first = (6 - _date(y, m, 1).weekday()) % 7 + 1
    return first + 7 * (n - 1)


def ct_offset_hours_rule(unix_s: float) -> int:
    """US Central offset from UTC at a Unix time by the explicit US rule (CDT -5, CST -6)."""
    y = datetime.fromtimestamp(unix_s, UTC).year
    a = datetime(y, 3, _nth_sunday(y, 3, 2), 8, tzinfo=UTC).timestamp()   # 02:00 CST
    b = datetime(y, 11, _nth_sunday(y, 11, 1), 7, tzinfo=UTC).timestamp()  # 02:00 CDT
    return -5 if a <= unix_s < b else -6


def _chicago():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Chicago")
    except Exception:  # no tz database (Windows without the tzdata package)
        return None


def ct_offset_hours(unix_s: float) -> int:
    """US Central offset at a Unix time: zoneinfo America/Chicago when available, else the explicit rule."""
    z = _chicago()
    if z is None:
        return ct_offset_hours_rule(unix_s)
    return int(datetime.fromtimestamp(unix_s, UTC).astimezone(z).utcoffset().total_seconds() // 3600)


def ct_wall(day: str, hhmm: str) -> int:
    """Unix seconds of a CT wall-clock time on a CT calendar day (RTH hours are never ambiguous)."""
    h, m = (int(x) for x in hhmm.split(":"))
    local = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()) + 3600 * h + 60 * m
    # the offset at noon UTC of the day is the day's offset during RTH (DST switches at 02:00 local)
    return local - 3600 * ct_offset_hours(local + 6 * 3600)


def rth_span(day: str, manifest: dict | None = None) -> tuple:
    """(open, close) Unix seconds of RTH (08:30-15:00 CT) on a CT calendar day."""
    r = (manifest or {}).get("rth", {"open": "08:30", "close": "15:00"})
    return ct_wall(day, r["open"]), ct_wall(day, r["close"])


def ct_clock(unix_s: float) -> str:
    """HH:MM:SS.fff in US Central time."""
    t = datetime.fromtimestamp(unix_s + 3600 * ct_offset_hours(unix_s), UTC)
    return t.strftime("%H:%M:%S.%f")[:-3]


def sierra_us(unix_s: float) -> int:
    return int(round(unix_s * 1_000_000)) + SC_EPOCH_US


# ---------------------------------------------------------------- exports and tapes
def export_files(day: str, manifest: dict, root=None) -> tuple:
    d = export_root(manifest, root) / str(_date.fromisoformat(day))
    return d / "ticks.csv", d / "ticks.json"


def check_export(day: str, manifest: dict, root=None) -> dict:
    """The day's sidecar, with the CSV's sha256 verified (raises when missing or changed)."""
    csv, side = export_files(day, manifest, root)
    if not csv.exists() or not side.exists():
        raise FileNotFoundError(f"{day}: no export ({csv}); run tools/sierra_to_ticks.py {day}")
    s = json.loads(side.read_text(encoding="utf-8"))
    if s.get("date") != day:
        raise RuntimeError(f"{side}: sidecar is for {s.get('date')}, not {day}")
    got = native.sha256_file(csv)
    if got != s.get("csv_sha256"):
        raise RuntimeError(f"{csv}: sha256 {got[:12]} does not match its sidecar ({str(s.get('csv_sha256'))[:12]}): re-export")
    if s.get("time_unit") != TIME_UNIT:
        raise RuntimeError(f"{side}: time_unit {s.get('time_unit')!r}, expected {TIME_UNIT!r} (seconds since 08:30 CT)")
    if s.get("t0_unix") != rth_span(day, manifest)[0]:
        raise RuntimeError(f"{side}: t0_unix {s.get('t0_unix')} is not 08:30 CT of {day} ({rth_span(day, manifest)[0]})")
    lg = s.get("line_gaps_over_1s", {}).get("max_s", 0.0)
    if lg >= TAPE["max_gap_seconds"]:
        raise RuntimeError(f"{day}: a {lg:.0f} s gap between lines would restart tick_csv's bucket clock")
    e = day_entry(day, manifest)
    if s.get("contract") != e.get("contract"):
        raise RuntimeError(f"{day}: exported contract {s.get('contract')} is not the manifest's {e.get('contract')}")
    return {**s, "csv": str(csv)}


def confirm_days(manifest: dict, root=None, count: int | None = None, exclude=()) -> list:
    """Exported confirm-rule days with full RTH, in date order, minus `exclude` (earlier confirmations).
    A confirm day must be a whole clean session: a day with a book gap > BOOK_GAP_EXCLUDE_S anywhere in its
    span (a recording outage) is skipped as a day, not cut into windows (test days, a fixed list, keep the
    day and lose only the windows that overlap the gap)."""
    cr = manifest["confirm_rule"]
    base = export_root(manifest, root)
    out = []
    if base.exists():
        for d in sorted(p.name for p in base.iterdir() if p.is_dir()):
            try:
                _date.fromisoformat(d)
            except ValueError:
                continue
            if d in exclude or role_of(d, manifest) != "confirm" or not (base / d / "ticks.json").exists():
                continue
            s = json.loads((base / d / "ticks.json").read_text(encoding="utf-8"))
            outage = any(b - a > BOOK_GAP_EXCLUDE_S for a, b in s.get("gaps_over_1s", {}).get("intervals_s", []))
            if s.get("rth_missing_seconds", 1e9) <= cr["full_rth_max_missing_seconds"] and not outage:
                out.append(d)
    return out[:count] if count else out


def days_for(role: str, manifest: dict, root=None, count: int | None = None) -> list:
    if role in ("test", "dev", "seen"):
        return listed_days(role, manifest)
    if role == "confirm":
        return confirm_days(manifest, root, count, exclude=ledger.used_real_confirm_days())
    raise ValueError(f"no tapes are opened for role {role!r}")


def open_tapes(role: str, *, manifest=None, root=None, days=None, run: str = "", config_sha256: str = "", claim: str = "",
               orchestrator: bool = False, reason: str = "", count: int | None = None, **meta) -> dict:
    """{date: (TickTape, sidecar)} for every day of `role` (or the given `days`, all of that role).
    Checks every export first, then opens through ledger.open_real (the only opener)."""
    mpath = manifest_path(manifest)
    m = load_manifest(mpath)
    sha = manifest_sha256(mpath)
    if role == "confirm" and days is not None:
        due = confirm_days(m, root, count or len(days), exclude=ledger.used_real_confirm_days())
        if list(days) != due:
            raise PermissionError(f"confirm days must be the next unused full confirm days {due}, not {list(days)}")
    days = list(days) if days is not None else days_for(role, m, root, count)
    if not days:
        raise FileNotFoundError(f"no {role} days to open")
    wrong = [d for d in days if role_of(d, m) != role]
    if wrong:
        raise PermissionError(f"days {wrong} are not {role} days in {mpath.name}")
    side = {d: check_export(d, m, root) for d in days}
    ledger.open_real(role, reason or f"ragrl real {role} tapes", manifest_sha256=sha, days=days, run=run, config_sha256=config_sha256,
                     claim=claim, orchestrator=orchestrator, tapes={d: s["csv_sha256"] for d, s in side.items()},
                     paths={d: s["csv"] for d, s in side.items()}, manifest=str(mpath), **meta)
    return {d: (load_tape(s["csv"]), s) for d, s in side.items()}


def load_tape(path):
    """The one place a recorded tape is loaded: only a path ledger.open_real() registered in this process,
    and only if its bytes still have the sha256 that was logged (a pytest greps for TickTape.from_csv)."""
    want = ledger.real_tape_allowed(path)
    if want is None:
        raise PermissionError(f"{path}: not a tape opened in this process (real.open_tapes registers the opened days only)")
    got = native.sha256_file(path)
    if got != want:
        raise RuntimeError(f"{path}: sha256 {got[:12]} changed since it was opened ({want[:12]})")
    return native.load().TickTape.from_csv(str(path), time_unit=TIME_UNIT, **TAPE)


def clean_windows(n_windows: int, sidecar: dict, window_seconds: float, warmup_seconds: float, max_gap_s: float = BOOK_GAP_EXCLUDE_S):
    """(played, excluded): window indices k of a tape whose span [k W, k W + warm-up + W) (seconds from the
    tape's start) overlaps no book gap longer than max_gap_s, and the excluded ones with their gap."""
    t0 = sidecar["span"]["start_s"]
    bad = [(a - t0, b - t0) for a, b in sidecar.get("gaps_over_1s", {}).get("intervals_s", []) if b - a > max_gap_s]
    played, excluded = [], []
    for k in range(n_windows):
        lo, hi = k * window_seconds, k * window_seconds + warmup_seconds + window_seconds
        hit = [g for g in bad if g[0] < hi and lo < g[1]]
        (excluded if hit else played).append(k if not hit else {"window": k, "book_gap_s": [round(hit[0][0] + t0, 3), round(hit[0][1] + t0, 3)]})
    return played, excluded


def subset_of(day: str, manifest: dict) -> str:
    return day_entry(day, manifest).get("subset") or day[5:7]


def env_root():
    """$RAGRL_REAL_DIR overrides the manifest's export_dir (scratch exports)."""
    return os.environ.get("RAGRL_REAL_DIR") or None


def load_buckets(path) -> dict:
    """The tape's 100 ms buckets in Python, for diagnostics that need the quotes (TickTape exposes none):
    per bucket the end-of-bucket bid / ask (ticks), L1 sizes, and buy / sell aggressor volume. Same gate as
    load_tape (a path this process opened, unchanged), and tick_csv.hpp's bucket clock replayed exactly:
    the clock starts at the first event, an event at t first closes every bucket ending at or before t
    (bucket ends accumulated in a double, +0.1 each, as in C++), a bucket is emitted once a quote exists."""
    import numpy as np
    want = ledger.real_tape_allowed(path)
    if want is None:
        raise PermissionError(f"{path}: not a tape opened in this process (real.open_tapes registers the opened days only)")
    if native.sha256_file(path) != want:
        raise RuntimeError(f"{path}: changed since it was opened")
    bs, tick = TAPE["bucket_seconds"], TAPE["tick_size"]
    cols = {k: [] for k in ("bid", "ask", "b1", "a1", "buy", "sell")}
    q = None
    buy = sell = 0
    started = False
    end = 0.0

    def close():
        nonlocal buy, sell
        for k, v in zip(("bid", "ask", "b1", "a1"), q):
            cols[k].append(v)
        cols["buy"].append(buy)
        cols["sell"].append(sell)
        buy = sell = 0

    with open(path, encoding="ascii") as f:
        for line in f:
            if not line or line[0] not in "TQ":
                continue
            x = line.rstrip("\n").split(",")
            t = float(x[1])
            if not started:
                end, started = t + bs, True
            if t - end > TAPE["max_gap_seconds"]:
                raise RuntimeError(f"{path}: a {t - end:.0f} s gap (tick_csv would restart its clock)")
            while t >= end:
                if q is not None:  # trades before the first quote join the first bucket, as in TickTape
                    close()
                end += bs
            if x[0] == "T":
                if x[4] == "B":
                    buy += int(x[3])
                else:
                    sell += int(x[3])
            else:
                q = (round(float(x[2]) / tick), round(float(x[3]) / tick), int(x[4]) if len(x) > 4 else 0,
                     int(x[9]) if len(x) > 9 else 0)
    if q is not None:
        close()
    return {k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}
