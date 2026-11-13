# OF21 plan: forward test of exhaustion at key levels

Status: **plan only, nothing run, no day on or after 2026-11-12 read.**
Written 2026-11-12 after OF20 (A86), at the user's request ("Yes,
pre-register it"); revised the same day after an independent design
review, before any forward day was read. Committing this file with the
frozen code (§8) is the pre-registration: the rule, the days, the looks,
the statistics and the decision rule are fixed here.

## 1. Question and the power warning

OF20 found no edge, but one descriptive line was positive: exhaustion
(OF19 P3, N 1, e 5) within 2 ticks of any OF20 level ("exh any tol 2"):
119 trades on 37 days, net +1.44 ticks per trade, block SE 2.17, median
−9.24, −0.96 without its top 3 trades, −0.68 without its best 2 days. It
was not a pre-registered test, it was the best-looking of the descriptive
rows, and it rests on a handful of trades: the in-sample +1.44 is very
probably inflated. The expected forward answer is **no**.

**Question:** on days nobody has seen, does exhaustion at the OF20 levels
make money after costs?

**Power (stated before any data, the main weakness).** The in-sample
per-day dispersion gives a net-per-trade SE of about 13.2 / sqrt(days):
2.1 at 40 days, 1.7 at 60, 1.2 at 125, 0.83 at 250. A fixed test at
t ≥ 2 with 80% power for the in-sample +1.44 would need about **680 full
days (2.7 years)**. Under the design below (final look at 250 days) the
power is 0.39 if the true edge is +1.44, 0.65 at +2, 0.94 at +3 and
≥ 0.99 at +4 ticks per trade (Monte Carlo, §4; the design review
recomputed these). So this test is likely to find only an edge about
twice the in-sample estimate or larger. A null at the final look is
reported as an upper bound (net + 2.007 × SE); 2.007 × SE (about 1.7 ticks
at 250 days) is the edge this test detects only about half the time.

## 2. Data and days

- **Source:** Sierra Chart tick-by-tick .scid records (`<q4f4I`, as
  OF19), read on or after 2026-11-12 only by `experiments/of21_forward.py`
  in `--count` or `--look` mode with the orchestrator flag. The converters
  `tools/scid_to_bars.py` and `tools/scid_to_trades.py` keep refusing those
  days, and `experiments/of20_levels.py`'s `overnight()` reads one only for
  a day this process opened as a real_forward look.
- **The scan** (`--count`, and the first step of a look) reads each
  forward day's RTH records for timestamps and volumes only (front month,
  full or not, a sha256 of the front month's overnight and RTH records), logs that per
  day in the ledger (block `real_forward_scan`) and prints only dates,
  front months and reasons: no prices, levels or outcomes.
- **Contracts:** ESZ26, ESH27, ESM27, ESU27, ESZ27, ESH28 (whichever files
  exist). Front month: the file with the larger RTH volume that day (OF13
  rule, `scid_to_bars.build_day`). **Sierra must record each new
  contract tick by tick before its roll window** (ESH27 by 2026-12-04: no
  ESH27 file exists today).
- **Calendar:** weekdays from 2026-11-12 on, in date order, only after
  their session is over. RTH 08:30-15:00 CT (the OF16 clock).
- **A full day** (scored), all of:
  - the front month's records are tick style (OF18 rule);
  - its RTH records cover the session: the first within 60 s of 08:30,
    the last within 60 s of 15:00, no gap between records over 60 s;
  - **roll window:** within 14 calendar days before the front month's
    expiry (the third Friday of its month), the next contract also has
    tick-style RTH records that day (so a missing file can never make an
    expiring contract the "front month");
  - the day's contract has a recorded prior session (below).
  Half days (2026-11-27, 2026-12-24, ...), recording outages and trading
  halts over 60 s are not scored; they are counted and reported with the
  reason. Dropping halt days drops some of the most extreme days: stated
  here as a weakness (§7), not corrected.
- **A day's status is fixed by its first scan** after its session ended
  (the earliest `real_forward_scan` entry). A later backfill cannot turn
  a not-full day into a full one. If the records of a full day change
  after its first scan (a different sha256), that is a red flag and every
  later look is refused until an amendment, committed before the look,
  says how to proceed.
- **Prior session:** the most recent earlier weekday (up to 7 calendar
  days back) on which the **day's own contract** has RTH records, full or
  not (a half day counts). On a roll day this is the new front month's
  own, thinner, prior session (as written in OF20 §5 for OF20b). The first
  forward day's prior session (2026-11-11) is before the cutoff and is read
  as a prior session only.
- **Overnight:** the same contract, 17:00 CT of the previous calendar day
  to 08:30 CT (OF20's `overnight()`, unchanged).

## 3. Rule (OF20, unchanged)

Exhaustion = OF19 P3 with N = 1, e = 5 (`sig_P3`), at a level: the bar's
extreme (the new low for a long, the new high for a short) within 2 ticks
of any level known at the setup's formation time (the start of its extreme
bar): the ten static levels (prior high, low, close, POC, 70% value area,
overnight high and low, initial balance from 09:30 CT) or the VWAP set
(VWAP ± 1 and 2 SD from 09:30 CT). Fills, exits, stops, the 15-min cap,
100 ms latency, $1.50 per side (0.24 ticks round trip on top of the
spread), sweep grouping: OF19 §3 unchanged. **Shell:** the same setup
without its footprint (Bv/A) conditions at the same levels (OF20 §5).
The trade simulation is the OF19/OF20 code, frozen (§8).

## 4. Looks

Three looks, each on the first N_k full scored days from 2026-11-12, in
date order, minus any day a `real_confirm` opening used (cumulative: look
2 contains look 1's days):

| look | full days N_k | about when | boundary (t ≥) | cumulative α spent (one-sided) |
| --- | --- | --- | --- | --- |
| 1 | 60 | Feb 2027 | 4.504 | 0.0000033 |
| 2 | 125 | May 2027 | 3.022 | 0.0013 |
| 3 (final) | 250 | late 2027 | 2.007 | 0.0228 |

Lan-DeMets O'Brien-Fleming spending, α(τ) = 2 − 2Φ(z_{1−α/2}/sqrt(τ)) with
α = 1 − Φ(2) = 0.0228 (one-sided t ≥ 2) and τ = N_k / 250. Look 1's
boundary is exact (computed with erfc). Looks 2-3 come from 4,000,000
Monte Carlo Brownian paths (numpy, seed 0; correlations sqrt(N_j / N_k)),
Monte Carlo error about ±0.005 (the design review got 3.026 and 2.009).
The boundaries are constants in the runner. The bootstrap t is compared
with these normal boundaries, as OF19 and OF20 compared it with 2.

**Deadline.** If look 3's 250 days are not reached by **2028-03-31**,
look 3 is taken on that date or later on every full day so far (more than
look 2's days); its boundary then spends all α left (the runner's
`final_boundary`, the same Monte Carlo with the actual day counts). A
look not reached by the deadline is skipped.

Before a look, only the scan may run (§2). A look runs once the count
reaches N_k, on exactly the first N_k full days; it cannot be moved or
taken early. **Crash retry:** a look whose process died may be rerun only
with an identical fingerprint (the same days, the same per-day record
sha256 and the same code sha256); the output is deterministic (seed 0),
so a rerun reveals nothing new. It is logged as a retry.

## 5. Statistics and the decision rule (fixed now)

As OF20: per-trade net at the recorded touch; SE = the largest
moving-block bootstrap SE over blocks of 2, 5 and 10 scored days (10000
resamples, seed 0; statistic total net / total trades); stress = net −
0.5 ticks; setup − shell = setup gross per trade − shell gross per trade,
its SE by resampling days jointly; halves split at the look's middle
scored day.

At look k:
- **Edge (stop, then claim):** net > 0, t = net / SE ≥ the look's
  boundary, setup − shell > 0, and stress net > 0. A claim is only "the
  rule passed its pre-registered forward test"; trading it is a separate
  decision with its own costs and risk. Deliberate differences from
  OF20's in-sample rule: halves are not required, and setup − shell only
  needs to be positive, not 2 SE, because a forward test at this power
  cannot afford them. So a pass does **not** show that the footprint
  condition beats a plain touch of the level. The stress condition rarely
  binds once t clears the boundary; it stays for continuity.
- **Futility (looks 1-2, non-binding):** net ≤ 0. The result reads "no
  edge so far" and stopping is recommended; continuing is allowed because
  futility does not spend α (the boundaries assume no futility stop).
- **Otherwise:** continue to the next look. At the final look without an
  edge: "net N per trade, upper bound N + 2.007 × SE on fresh days; edges
  below 2.007 × SE are detected less than half the time".

**Reported at every look, not tested:** trades, win rate, median, net
without the top 3 trades and without the best 2 days (lesson 61), long
and short, halves, the shell's trades and net, trades per level type,
not-scored days with reasons, the front month per day, the code sha256.

**Red flags:** t > 6 at any look; a win rate above 75% on 30+ trades; any
level from data after its use (the OF20 checks); a look's days not the
first N_k full days; a full day whose records changed after its first
scan (§2); a frozen file whose sha256 differs from §8; the rehearsal not
reproducing OF20.

## 6. Ledger and guards

`ragrl.ledger.open_forward(claim, look, days, ...)` logs block
`real_forward` (claim `OF21`, look, days, fingerprint, the code and
manifest sha256) before any price of those days is read, and refuses: no
orchestrator flag; look k without look k−1; days that do not extend look
k−1's; any day an earlier `real_confirm` opening used; a look already
logged, except a crash retry with the same fingerprint (§4). Days read by
OF21 looks count as used for other confirmations
(`used_real_confirm_days`), so a later claim's confirmation takes other
days. `log_forward_scan` logs block `real_forward_scan` (§2).

Before every look, the runner (1) refuses unless every file in §8 has its
frozen sha256, (2) reruns the rehearsal (below) and refuses unless it
reproduces OF20 exactly, (3) scans and refuses on a changed full day. The
ledger and clock modules (`python/ragrl/ledger.py`, `python/ragrl/real.py`)
are not frozen (other work changes them); their sha256 is logged with
each look and the rehearsal checks the clock.

Usage (orchestrator only, through the kit):

    python experiments/of21_forward.py --count --i-am-the-orchestrator
    python experiments/of21_forward.py --look 1 --i-am-the-orchestrator --out experiments/of21_look1.txt

`--rehearse` reruns the same code path on OF20's 37 scored days
(2026-09-22 .. 2026-11-11, from the .scid files, no ledger) and must
reproduce OF20's "exh any tol 2" row exactly: 119 trades, net +1.44, SE
2.17, shell 925, setup − shell +2.61 ± 2.23. It did at the freeze
(`experiments/of21_rehearsal.txt`); all 37 days pass the full-day rule.

## 7. Known weaknesses

- **Power** (§1): only an edge of about 3 ticks per trade or more is
  likely to be found; the in-sample estimate is half that and inflated.
- **Selection:** the rule was the best-looking descriptive row of OF20;
  the forward days are what makes the test fair, nothing else.
- **Regime:** a year of new days is one market period; a pass would still
  need a check that it is not carried by a few trend days (reported).
- **Dropped days:** recording outages, missing contract files in a roll
  window and halt days are not scored (counted). Halt days are among the
  most extreme; excluding them can bias the result either way.
- **Costs:** market orders as in OF19; no queue position, no book.
- **The confirm block:** OF21 uses its days for about a year; another
  claim's confirmation in that time takes days OF21 has not yet read, and
  those days then leave OF21's sequence, so OF21's looks come later.

## 8. Frozen code

Every look refuses unless these files have exactly these sha256 (of the
file with CRLF read as LF; `python/tests/test_of21.py` fails on any
change):

- `experiments/of21_forward.py` sha256: `0adb0f61083aff1a38150b69e4d642b624fa26917d7a72be6268a07be80da0c7`
- `experiments/of20_atlevels.py` sha256: `075e691c368ec2002ff22b16b75792257aa068ace3e4ba0af23b9ab13572fcfc`
- `experiments/of20_levels.py` sha256: `7ec71ba08c38940226f898842219a6eb3f66648158d46374ec00d78bcc36690b`
- `experiments/of19_footprint.py` sha256: `47ca7794afedf6f81d99c634d216403a77a666c12c9a3b5b1ce95f2d0fe17470`
- `tools/scid_to_bars.py` sha256: `fd0f8118fc875b10bc0eddfa50596e58467fbc002f79516288a0079bf5b9eef3`
- `tools/scid_to_trades.py` sha256: `50b8c070043cc843a03d968254336159f1b19340896883b5c3792626bb4ba2e5`
