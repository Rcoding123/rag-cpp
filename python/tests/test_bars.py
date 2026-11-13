"""OF18: tools/scid_to_bars.py on synthetic Sierra .scid files (never the user's data).

A CST day of 1-s bundles and a CDT day of tick records, two contract files with different RTH volume;
checks the RTH clock (08:30 CT = 14:30 / 13:30 UTC), the front month by volume, bar OHLC from each record
style, the aggressor mapping, gap minutes and the refusal of confirm-block days."""
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from conftest import REPO

TOOL = REPO / "tools" / "scid_to_bars.py"
SC_EPOCH_US = 25569 * 86400 * 1_000_000
REC = np.dtype([("t", "<i8"), ("o", "<f4"), ("h", "<f4"), ("l", "<f4"), ("c", "<f4"), ("n", "<u4"), ("v", "<u4"),
                ("bv", "<u4"), ("av", "<u4")])


def _us(iso_utc: str, extra_us: int = 0) -> int:
    t = datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc).timestamp()
    return int(round(t * 1_000_000)) + SC_EPOCH_US + extra_us


def _write(path, recs):
    a = np.array(recs, dtype=REC)
    a = a[np.argsort(a["t"], kind="stable")]
    head = b"SCID" + (56).to_bytes(4, "little") + bytes(48)
    path.write_bytes(head + a.tobytes())


def _p(price):
    return price * 100.0


def test_bars_from_bundles_and_ticks(tmp_path):
    front, back = [], []
    # 2026-01-06 (CST): 08:30 CT = 14:30 UTC. 1-s bundles: minute 1 has two seconds, minute 2 none (a gap), minute 3 one
    o = "2026-01-06 14:30:00"
    front.append((_us(o, 5_000_000), _p(6000.00), _p(6000.50), _p(5999.75), _p(6000.25), 3, 10, 4, 6))
    front.append((_us(o, 40_000_000), _p(6000.25), _p(6001.00), _p(6000.25), _p(6000.75), 2, 5, 1, 4))
    front.append((_us(o, 125_000_000), _p(6001.00), _p(6001.00), _p(6000.50), _p(6000.50), 1, 7, 7, 0))
    front.append((_us("2026-01-06 14:29:59"), _p(1.0), _p(1.0), _p(1.0), _p(1.0), 1, 99, 0, 99))   # before the open: ignored
    front.append((_us("2026-01-06 21:00:00"), _p(1.0), _p(1.0), _p(1.0), _p(1.0), 1, 99, 0, 99))   # at the close: ignored
    back.append((_us(o, 7_000_000), _p(6020.00), _p(6020.00), _p(6020.00), _p(6020.00), 1, 3, 0, 3))
    # 2026-09-15 (CDT): 08:30 CT = 13:30 UTC. Tick records: high = ask, low = bid, close = the trade, open = 0
    s = "2026-09-15 13:30:00"
    for k, (us, px, bid, ask, v, sell) in enumerate([(1_000_000, 6500.25, 6500.00, 6500.25, 2, False),
                                                     (1_000_500, 6500.00, 6500.00, 6500.25, 1, True),
                                                     (1_200_000, 6500.50, 6500.25, 6500.50, 4, False),
                                                     (61_000_000, 6499.75, 6499.75, 6500.00, 3, True)]):
        front.append((_us(s, us), 0.0, _p(ask), _p(bid), _p(px), 1, v, v if sell else 0, 0 if sell else v))
    back.append((_us(s, 2_000_000), 0.0, _p(6510.25), _p(6510.0), _p(6510.0), 1, 1, 1, 0))
    _write(tmp_path / "ESH26_FUT_CME.scid", front)
    _write(tmp_path / "ESM26_FUT_CME.scid", back)
    out = tmp_path / "bars"
    p = subprocess.run([sys.executable, str(TOOL), "--start", "2026-01-05", "--end", "2026-09-16", "--scid-dir", str(tmp_path),
                        "--out-dir", str(out), "--contracts", "ESH26,ESM26"], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert sorted(x.stem for x in out.glob("*.npz")) == ["2026-01-06", "2026-09-15"]
    a = np.load(out / "2026-01-06.npz")
    sa = json.loads((out / "2026-01-06.json").read_text(encoding="utf-8"))
    assert sa["contract"] == "ESH26" and sa["rth_volume"] == {"ESH26": 22, "ESM26": 3} and sa["style"] == "bundle"
    assert (a["open"][0], a["high"][0], a["low"][0], a["close"][0]) == (24000.0, 24004.0, 23999.0, 24003.0)
    assert (a["volume"][0], a["buy"][0], a["sell"][0], a["trades"][0]) == (15, 10, 5, 5)
    assert a["volume"][1] == 0 and a["open"][1] == a["close"][1] == 24003.0          # a gap carries the last close
    assert (a["open"][2], a["close"][2], a["sell"][2]) == (24004.0, 24002.0, 7)
    assert sa["gap_minutes"] == 390 - 2 and sa["ohlc_violations"] == 0 and np.isnan(a["open_bid"]).all()
    b = np.load(out / "2026-09-15.npz")
    sb = json.loads((out / "2026-09-15.json").read_text(encoding="utf-8"))
    assert sb["contract"] == "ESH26" and sb["style"] == "tick"
    assert (b["open"][0], b["high"][0], b["low"][0], b["close"][0]) == (26001.0, 26002.0, 26000.0, 26002.0)
    assert (b["open_bid"][0], b["open_ask"][0]) == (26000.0, 26001.0)
    assert (b["volume"][0], b["buy"][0], b["sell"][0]) == (7, 6, 1)
    assert b["close"][1] == 25999.0 and b["sell"][1] == 3
    # the confirm block is refused before anything is read
    p = subprocess.run([sys.executable, str(TOOL), "--start", "2026-11-10", "--end", "2026-11-12", "--scid-dir", str(tmp_path),
                        "--out-dir", str(tmp_path / "no")], capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "refused" in p.stderr and not (tmp_path / "no").exists()
