# Aggregate E21 leaderboard CSVs: mean, IQM, min-max and 95% percentile-bootstrap CI per learner x episodes.
import csv, glob, random, statistics, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + '/e21_*.csv')):
    for r in csv.reader(open(f)):
        if len(r) == 4: rows.append((r[0], int(r[1]), int(r[2]), 100 * float(r[3])))
def iqm(v):
    v = sorted(v); n = len(v); k = n // 4
    mid = v[k:n - k] if n - 2 * k > 0 else v
    return sum(mid) / len(mid)
def boot(v, B=20000, seed=1):
    rng = random.Random(seed); n = len(v); means = []
    for _ in range(B): means.append(sum(v[rng.randrange(n)] for _ in range(n)) / n)
    means.sort(); return means[int(0.025 * B)], means[int(0.975 * B)]
names = {'tileq_default': 'TileQ, default settings', 'tileq_tuned': 'TileQ, tuned by NLopt (E18)', 'tileq_coarse_al': 'TileQ, coarse tiles + advantage learning (E16, E19)',
         'actor_critic': 'ActorCritic', 'neuralq_double': 'NeuralQ + Double DQN (E16, E20)', 'model_based': 'Model-based (Shrunk)'}
order = ['tileq_default', 'tileq_tuned', 'tileq_coarse_al', 'actor_critic', 'neuralq_double', 'model_based']
print('| Learner | Episodes | Seeds | Mean | 95% CI (bootstrap) | IQM | Min–max |')
print('|---|---|---|---|---|---|---|')
for who in order:
    for ep in sorted({e for w, s, e, v in rows if w == who}):
        v = [x for w, s, e, x in rows if w == who and e == ep]
        if not v: continue
        lo, hi = boot(v)
        print(f'| {names[who]} | {ep:,} | {len(v)} | {statistics.mean(v):.1f}% | [{lo:.1f}, {hi:.1f}] | {iqm(v):.1f}% | {min(v):.0f}–{max(v):.0f}% |')
