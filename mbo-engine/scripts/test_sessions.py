import datetime as dt
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S

U = lambda y, mo, d, h, mi=0: int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.UTC).timestamp() * 1e9)
D = lambda s: np.datetime64(s)
fails = 0
def check(name, got, exp):
    global fails
    ok = bool(np.all(got == exp))
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name, "" if ok else f"got {got} expected {exp}")

# CDT (UTC-5) week: Thu 2026-10-29
check("Thu 16:59 CDT -> Thu", S.trade_date([U(2026, 10, 29, 21, 59)]), D("2026-10-29"))
check("Thu 17:00 CDT -> Fri", S.trade_date([U(2026, 10, 29, 22, 0)]), D("2026-10-30"))
check("Sun 17:00 CDT -> Mon", S.trade_date([U(2026, 10, 25, 22, 0)]), D("2026-10-26"))
# DST ends Sun 2026-11-01 02:00 CDT -> CST (UTC-6): same wall-clock 17:00 is one UTC hour later
check("Sun 17:00 CST -> Mon (23:00Z)", S.trade_date([U(2026, 11, 1, 23, 0)]), D("2026-11-02"))
check("Sun 22:00Z is 16:00 CST -> Sun (halt)", S.trade_date([U(2026, 11, 1, 22, 0)]), D("2026-11-01"))
check("Mon 22:00Z 11-02 = 16:00 CST -> Mon", S.trade_date([U(2026, 11, 2, 22, 0)]), D("2026-11-02"))
check("Mon 23:00Z 11-02 = 17:00 CST -> Tue", S.trade_date([U(2026, 11, 2, 23, 0)]), D("2026-11-03"))
# DST starts Sun 2026-03-08 02:00 CST -> CDT
check("Sun 17:00 CDT 03-08 (22:00Z) -> Mon", S.trade_date([U(2026, 3, 8, 22, 0)]), D("2026-03-09"))
check("Sun 17:00 CST 03-01 (23:00Z) -> Mon", S.trade_date([U(2026, 3, 1, 23, 0)]), D("2026-03-02"))
# halt
check("in_halt 16:00 CDT", S.in_halt([U(2026, 10, 29, 21, 0)]), True)
check("in_halt 16:59:59 CDT", S.in_halt([U(2026, 10, 29, 21, 59) + 59 * 10**9]), True)
check("not in_halt 17:00 CDT", S.in_halt([U(2026, 10, 29, 22, 0)]), False)
check("not in_halt 15:59 CDT", S.in_halt([U(2026, 10, 29, 20, 59)]), False)
# RTH 08:30-15:00 CT
check("RTH 08:29 CDT no", S.is_rth([U(2026, 10, 29, 13, 29)]), False)
check("RTH 08:30 CDT yes", S.is_rth([U(2026, 10, 29, 13, 30)]), True)
check("RTH 14:59 CDT yes", S.is_rth([U(2026, 10, 29, 19, 59)]), True)
check("RTH 15:00 CDT no", S.is_rth([U(2026, 10, 29, 20, 0)]), False)
check("RTH 08:30 CST (14:30Z) yes", S.is_rth([U(2026, 11, 2, 14, 30)]), True)
check("RTH early-close date excluded", S.is_rth([U(2026, 9, 7, 13, 30)], {D("2026-09-07")}), False)
print("FAILED" if fails else "sessions: all tests passed")
sys.exit(1 if fails else 0)
