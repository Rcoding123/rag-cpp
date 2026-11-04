import random, math, sys
from collections import deque
def run(k, h, xs, lag=50, rate=0.01, warm=200):
    # baseline from samples older than `lag`; CUSUM on standardized recent samples
    q = deque(); n = 0; m = 0.0; v = 0.0; S = 0.0; t_alarm = []
    for t, x in enumerate(xs):
        q.append(x)
        if len(q) > lag:
            y = q.popleft(); n += 1; r = max(rate, 1.0 / n); d = y - m; m += r * d; v = (1 - r) * (v + r * d * d)
        if n < warm: continue
        z = (x - m) / math.sqrt(max(v, 1e-12))
        S = min(0.0, S + z + k)
        if S < -h:
            t_alarm.append(t); S = 0.0; q.clear(); n = 0; m = v = 0.0
    return t_alarm
random.seed(1)
N = 400000
noise = [random.gauss(0, 1) for _ in range(N)]
for k, h, rate, warm in [(0.25, 12, 0.002, 1000), (0.25, 15, 0.002, 1000), (0.3, 14, 0.002, 1000), (0.25, 15, 0.001, 1500), (0.3, 16, 0.001, 1500)]:
    fa = run(k, h, noise, rate=rate, warm=warm)
    delays = []
    for trial in range(60):
        rs = random.Random(trial)
        xs = [rs.gauss(0, 1) for _ in range(3000)] + [rs.gauss(-0.55, 1) for _ in range(3000)]
        al = [a for a in run(k, h, xs, rate=rate, warm=warm) if a >= 3000]
        delays.append(al[0] - 3000 if al else 3000)
    delays.sort()
    print(f"k={k} h={h} rate={rate} warm={warm}: false alarms {len(fa)} / {N}; delay (0.55 sd): median {delays[30]}, 90th {delays[54]}, max {delays[-1]}")
