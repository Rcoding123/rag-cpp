"""OF20: key price levels for a day of ES (docs/ai/OF20_LEVELS_PLAN.md), from data before the level is used.

Static levels of day D (known at 08:30 CT, except the initial balance, active from 09:30 CT):
  pdh, pdl, pdc  prior RTH session high / low / last trade (the previous trades day in build-avx512/real/trades)
  ppoc         prior RTH volume POC (most traded price; ties: the one nearest the session's VWAP, then the lower)
  pvah, pval   prior value area: from the POC, add the adjacent price (above or below) with more volume, one tick at a
               time (ties: above), until it holds >= 70% of the session's volume
  onh, onl     overnight: trades of the same contract from 17:00 CT of the previous calendar day to 08:30 CT of D
  ibh, ibl     initial balance: RTH 08:30-09:30 CT high / low, counted only for setups formed at or after 09:30 CT
Running levels (at a formation time t >= 09:30 CT, from RTH trades strictly before t, rounded to the tick): vwap,
vwap +- 1 and 2 sigma (volume-weighted SD around it). Levels are judged at a setup's formation time (OF20 plan 2).
Prices in ticks (0.25). Nothing on or after 2026-11-12 is read.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "python"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from ragrl import real  # noqa: E402
from scid_to_bars import PRICE_SCALE, TICK, Scid  # noqa: E402

CUTOFF = "2026-11-12"
IB_END = 3600.0  # seconds after 08:30 CT
STATIC = ("pdh", "pdl", "pdc", "ppoc", "pvah", "pval", "onh", "onl", "ibh", "ibl")
RUNNING = ("vwap", "vwap+1s", "vwap-1s", "vwap+2s", "vwap-2s")


def value_area(px, size, share=0.70):
    lo = int(px.min())
    vol = np.bincount((px - lo).astype(np.int64), weights=size.astype(float))
    total = vol.sum()
    vw = float((px * size).sum() / size.sum()) - lo
    top = np.flatnonzero(vol == vol.max())
    poc = int(top[np.argmin(np.abs(top - vw))])  # ties: nearest the VWAP, then the lower (argmin takes the first)
    a = b = poc
    held = vol[poc]
    while held < share * total:
        up = vol[b + 1] if b + 1 < len(vol) else -1
        dn = vol[a - 1] if a - 1 >= 0 else -1
        if up >= dn:
            b += 1
            held += up
        else:
            a -= 1
            held += dn
    return poc + lo, b + lo, a + lo


def overnight(day: str, contract: str, scid_dir="C:/SierraChart/Data"):
    if day >= CUTOFF:
        raise RuntimeError("the confirm block is never read")
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    t0, t1 = real.sierra_us(real.ct_wall(prev, "17:00")), real.sierra_us(real.ct_wall(day, "08:30"))
    s = Scid(Path(scid_dir) / f"{contract}_FUT_CME.scid")
    try:
        r = s.records(t0, t1)
    finally:
        s.close()
    if not len(r):
        return None, None
    px = np.round(r["c"].astype(np.float64) / PRICE_SCALE / TICK)
    return int(px.max()), int(px.min())


def static_levels(day: str, contract: str, T, Tprev) -> dict:
    poc, vah, val = value_area(Tprev["px"], Tprev["size"])
    onh, onl = overnight(day, contract)
    ib = T["t"] < IB_END
    lv = {"pdh": int(Tprev["px"].max()), "pdl": int(Tprev["px"].min()), "pdc": int(Tprev["px"][-1]), "ppoc": poc, "pvah": vah, "pval": val,
          "onh": onh, "onl": onl, "ibh": int(T["px"][ib].max()) if ib.any() else None, "ibl": int(T["px"][ib].min()) if ib.any() else None}
    return lv


def active_static(lv: dict, t: float, names=STATIC) -> list:
    """The static levels known at time t (the initial balance only from 09:30 CT)."""
    return [lv[k] for k in names if lv.get(k) is not None and (k not in ("ibh", "ibl") or t >= IB_END)]


class RunningVWAP:
    """vwap and +-1 / 2 sigma at any time t from the RTH trades strictly before t."""

    def __init__(self, T):
        self.t = T["t"]
        w = T["size"].astype(float)
        p = T["px"].astype(float)
        self.cw, self.cwp, self.cwp2 = np.cumsum(w), np.cumsum(w * p), np.cumsum(w * p * p)

    def at(self, t: float) -> list:
        """The VWAP set at formation time t (scalar): [] before 09:30 CT or with no trade before t."""
        if t < IB_END:
            return []
        i = int(np.searchsorted(self.t, float(t), side="left")) - 1
        if i < 0:
            return []
        v = self.cwp[i] / self.cw[i]
        sd = float(np.sqrt(max(self.cwp2[i] / self.cw[i] - v * v, 0.0)))
        return [int(round(x)) for x in (v, v + sd, v - sd, v + 2 * sd, v - 2 * sd)]


def near(price, levels, tol) -> bool:
    return any(abs(price - L) <= tol for L in levels)
