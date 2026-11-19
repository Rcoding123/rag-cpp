"""CME ES session calendar in America/Chicago (DST handled by zoneinfo).

Conventions (owner, 2026-11-18) and what was verified:
  * Trade date D = session from 17:00 CT on the previous calendar day to 16:00 CT on D;
    16:00-17:00 CT is the daily halt (assigned to the session that just closed).
    VERIFIED against our 2026 status data: scheduled Close at 16:00 CT, Trading at 17:00 CT, no
    other scheduled pause (CME's 2020 FAQ mentions a 15:15-15:30 CT pause; it is absent in 2026).
    Early closes (e.g. Labor Day, 12:00 CT) exist; see early_close_dates().
  * Settlement time 15:00 CT (12:00 CT on shortened days), CME FAQ (archived copy); the settlement
    VALUE is taken from the statistics feed, never computed here.
  * RTH = 08:30-15:00 CT is the US cash-equity session (9:30-16:00 ET). This is a CONVENTION,
    NOT verified against CME (cmegroup.com returned 403); change RTH_OPEN/RTH_CLOSE if the owner
    prefers e.g. 15:15. Shortened days have no RTH (set via early_close_dates).
All functions take UTC nanosecond timestamps (numpy uint64/int64 arrays or ints).
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHI = ZoneInfo("America/Chicago")
SESSION_OPEN = dt.time(17, 0)
SESSION_CLOSE = dt.time(16, 0)
RTH_OPEN = dt.time(8, 30)
RTH_CLOSE = dt.time(15, 0)


def _local(ts_ns):
    return pd.to_datetime(np.asarray(ts_ns, dtype=np.int64), unit="ns", utc=True).tz_convert(CHI)


def trade_date(ts_ns):
    """CME trade date (datetime64[D]) for each UTC ns timestamp."""
    loc = _local(ts_ns)
    mins = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    base = np.asarray(loc.tz_localize(None).normalize(), dtype="datetime64[D]")
    return np.where(mins >= SESSION_OPEN.hour * 60 + SESSION_OPEN.minute, base + np.timedelta64(1, "D"), base)


def in_halt(ts_ns):
    """True during the daily halt 16:00-17:00 CT (no matching; books may be crossed)."""
    loc = _local(ts_ns)
    mins = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    return (mins >= SESSION_CLOSE.hour * 60) & (mins < SESSION_OPEN.hour * 60)


def is_rth(ts_ns, early_close=frozenset()):
    """True inside RTH on a normal full day. `early_close`: set of trade dates (datetime64[D]) with no RTH."""
    loc = _local(ts_ns)
    mins = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    inside = (mins >= RTH_OPEN.hour * 60 + RTH_OPEN.minute) & (mins < RTH_CLOSE.hour * 60 + RTH_CLOSE.minute)
    if early_close:
        td = trade_date(ts_ns)
        inside &= ~np.isin(td, np.array(sorted(early_close), dtype="datetime64[D]"))
    return inside


def early_close_dates(status_ndarray):
    """Trade dates whose scheduled close (status action Close/PreOpen with is_trading N, reason Scheduled)
    is not 16:00 CT, from a databento status ndarray of one outright instrument. Data-driven, no calendar."""
    a = status_ndarray
    sch = a[(a["reason"] == 1) & (a["is_trading"] == b"N") & np.isin(a["action"], [1, 12])]
    loc = _local(sch["ts_recv"])
    mins = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    out = set()
    for t, m in zip(sch["ts_recv"], mins):
        if m < SESSION_CLOSE.hour * 60 and m >= 9 * 60:  # a scheduled stop earlier in the day than 16:00 (ignore overnight/pre-open noise)
            out.add(trade_date([t])[0])
    return out
