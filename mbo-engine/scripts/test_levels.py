import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import levels as L
fails = 0
def check(n, got, exp):
    global fails
    ok = bool(np.all(np.isclose(got, exp, equal_nan=True))); fails += not ok
    print(("ok   " if ok else "FAIL ") + n, "" if ok else f"got {got} exp {exp}")
# value area: volumes by price 100..106: [1,2,10,20,12,3,2] total 50; 70% = 35
p = np.arange(100, 107.0); v = np.array([1, 2, 10, 20, 12, 3, 2.0])
poc, vah, val = L.value_area(p, v, vwap=103.0)
check("poc", poc, 103.0)
# start 20; up = 12+3=15, down = 10+2=12 -> up: 20+12+3=35 >= 35 -> stop. VA = 103..105
check("vah", vah, 105.0); check("val", val, 103.0)
# tie poc -> closest to vwap
poc, _, _ = L.value_area(np.array([100., 101., 102.]), np.array([5., 1., 5.]), vwap=101.9)
check("poc tie closest to vwap", poc, 102.0)
# single price
check("single price va", L.value_area(np.array([100.]), np.array([7.]), 100.0), (100.0, 100.0, 100.0))
# scope_stats
d = pd.DataFrame({"price": [100.0, 101.0, 99.5, 100.0], "size": [1, 2, 1, 4]})
s = L.scope_stats(d)
check("open/high/low/close", (s["open"], s["high"], s["low"], s["close"]), (100.0, 101.0, 99.5, 100.0))
check("vwap", s["vwap"], (100 + 202 + 99.5 + 400) / 8); check("volume", s["volume"], 8); check("poc busiest price", s["poc"], 100.0)
check("empty scope", L.scope_stats(d.iloc[0:0])["volume"], 0)
print("FAILED" if fails else "levels: all tests passed"); sys.exit(1 if fails else 0)
