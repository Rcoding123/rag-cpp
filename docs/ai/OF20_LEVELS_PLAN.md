# OF20 plan: absorption and exhaustion at key levels

Status: **done 2026-11-12: no edge** (EXPERIMENTS.md OF20, `experiments/of20_atlevels.txt`). Originally: plan only, nothing run. Written 2026-11-12 after OF19 (A85),
at the user's request; revised the same day after an independent design
review, before any outcome. Committing this file is the pre-registration:
levels, setups, statistics and the decision rule are fixed here before
any OF20 profit or loss is computed.

## 1. Question

Practitioners say absorption and exhaustion work "at a key level", not
everywhere (the user's KB, `practitioner_methods.md` F2, and its section
6: absorption's discretionary component is "at a key level"). OF19 tested
them everywhere and found no edge. The question here:

**At the standard auction-market levels, do the absorption and exhaustion
setups make money after costs, and do they beat a plain touch of the same
level without the footprint condition?**

What was looked at before writing this: OF19's results for these rules
over all occurrences (all negative), and the **counts** (not the outcomes)
of their signals near the levels below on the 37 OF19 scored days
(absorption W 15: 44 within 2 ticks of a static level, 91 of any level;
exhaustion N 1: 58 and 129). No outcome of any signal at a level has been
computed.

Expected answer: no. The test is coarse (§6).

## 2. Data and levels

The OF19 data: tick records of the front month, 47 days 2026-09-08 ..
2026-11-11; the same 37 scored days (2026-09-22 .. 2026-11-11), the first
10 days feeding the absorption threshold only. Nothing from 2026-11-12 on
is read. Levels (`experiments/of20_levels.py`, ticks), from data before
they are used:

- **Static** (known at 08:30 CT): the prior RTH session's high, low and
  close (its last trade), volume POC and 70% value area (from the POC, add
  the adjacent price with more volume, one tick at a time, ties above: a
  stated deviation from the classic two-row method, edges can differ by
  1-2 ticks); the overnight high and low (the same contract, 17:00 CT of
  the previous calendar day to 08:30 CT, the 08:30:00 print excluded); the
  initial balance high and low (08:30-09:30 CT RTH). RTH is 08:30-15:00 CT
  as in OF16-OF19 (not the 15:15 CT close some ES charts use). The prior
  session is the previous CME RTH session of day D's contract; on the 37
  scored days that is always the previous trades day (no gaps, no
  holidays).
- **VWAP set**: RTH VWAP and VWAP ± 1 and 2 volume-weighted SD, from RTH
  trades strictly before the formation time, each rounded to the nearest
  tick; active only from 09:30 CT (right after the open VWAP is about the
  price and the SD near 0, so every early signal would be "at VWAP").

**Formation time.** A level is judged at the setup's formation time:
absorption, the start of its window (t − W, with t the candidate second);
exhaustion, the start of its extreme bar. A level counts only if it was
known at that time: the initial balance only for setups formed at or after
09:30 CT, the VWAP set as above (none before the first trade).

A setup is **at a level** when its price is within `tol` ticks of a level
counted at its formation time: absorption, the absorbed price P;
exhaustion, the bar's extreme (the new low for a long, the new high for a
short).

## 3. Setups (OF19's definitions, exits and fills unchanged)

- **Absorption** = OF19 P2 with W = 15 s and target 4, at a level. W = 15
  was chosen for its count (the most frequent variant); it was also the
  least negative P2 variant in OF19 (net −1.66 against −2.10 for the
  primary), which was seen.
- **Exhaustion** = OF19 P3 with N = 1 and e = 5 (the study default), at a
  level.

Everything else is OF19 (§3 of `OF19_FOOTPRINT_PLAN.md`): trailing q95
for absorption, triggers, cancellation, stops, target, 15-min (P3) and
5-min (P2) caps, sweep-aware fills at the recorded touch, 100 ms latency,
$1.50 per side.

## 4. Tests

- **Primary (one test, t ≥ 2):** both setups pooled, at **any** level
  (static and VWAP sets), tol = 2 ticks: about 220 trades.
- **Secondary (8 tests, t_req = 2 + sqrt(2 ln 8) = 4.04):** each setup ×
  level set (static, VWAP) × tol ∈ {1, 2}.

## 5. Statistics and the decision rule (fixed now)

As OF19 §5: per-trade net at the recorded touch; SE the largest
moving-block bootstrap SE over blocks of 2, 5 and 10 scored days (10000
resamples, seed 0; total net / total trades); halves split at the middle
scored day; stress net − 0.5 ticks.

- **Edge:** net > 0, t ≥ its bar (2 primary, 4.04 secondary), positive in
  both halves and at stress, **and setup − shell ≥ 2 × the block SE of the
  difference** (the shell below). For the pooled primary, setup − shell =
  Σ_s w_s (setup_s − shell_s), with w_s the setup's share s of the
  primary's trades, its SE from resampling days jointly. Not a claim:
  OF20b, one confirmation on the next **40** full front-month RTH days
  from 2026-11-12 (the setups are rare: ~3-6 a day), claimed only if
  net > 0 with t ≥ 2 **and** setup − shell > 0. For OF20b the prior
  session is the previous recorded RTH session of the day's contract (read
  from that contract's .scid, so the December roll uses ESH27's own prior
  session); half days count as a prior session but are not scored; a day
  whose prior session is not recorded is skipped and counted.
- **If the setup passes net and t but not the shell condition**, the
  result reads "at these levels the touch, not the footprint condition,
  carries the result": the shell is not a pre-registered rule, so it is
  not a candidate and nothing is claimed.
- **Otherwise:** "At these levels the setups net no more than X ticks per
  trade (X = bar × SE)"; smaller edges are not excluded.

**The level-touch shell** (the comparison that answers the question): the
same setup without its footprint condition, at the same levels and
tolerance, with the same exits: absorption without S ≥ q95 (held price and
confirmation only); exhaustion as a new 1-bar low (high) then a break above
the bar's high (below its low), without the Bv (A) conditions. Reported
for every test: shell trades, shell net, and the setup minus the shell
with its block SE.

**Reported, not tested:** the interaction practitioners claim, (setup −
shell) at levels minus (setup − shell) away from all levels, with its block
SE; per level type (prior high/low, POC and value
area, overnight, initial balance, VWAP bands): trades and net; long and
short; trades by time of day; the absorption and exhaustion results away
from all levels (the complement), so the level effect is visible; X per
test.

**Red flags:** t > 6; a win rate above 75% on 30+ trades; a level computed
from data after its use (checked in code: static levels from the previous
day and the overnight before 08:30, the initial balance only after 09:30,
VWAP from trades before the signal time).

## 6. Known weaknesses (stated before the results)

- **Power.** About 220 trades in the primary over 37 days. Exhaustion's
  per-trade dispersion is about 3.5 times absorption's (OF19: SE 0.69 on
  788 trades against 0.23 on 574), so the pooled primary is mostly an
  exhaustion test: SE about 1.0, X about 2.1 ticks. The secondaries
  (25-60 trades) have X of roughly 4.5 (absorption) to 14 ticks
  (exhaustion): they cannot realistically pass, and a pass on 25 trades
  would itself be a red flag. A null says little below these sizes.
- **Seen data.** The same 37 days as OF19, whose overall results were
  seen. The at-level condition is the practitioners' a priori claim, not
  chosen from OF19's results, and only signal counts at levels were looked
  at.
- **Levels.** One standard set; practitioners also draw discretionary
  levels (prior swing points, composites, naked POCs, the official
  settlement rather than the last RTH trade) that are not tested.
- **No book.** As OF19.

## 7. Cost

Light: OF19's pass over the 47 days plus the levels (one overnight byte
range per day); minutes. Implementation, review: about 1.5-2 hours.
