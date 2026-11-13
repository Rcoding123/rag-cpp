# OF19 plan: footprint order-flow patterns on true tick data

Status: **done 2026-11-12: no primary edge, no secondary candidate** (EXPERIMENTS.md OF19, `experiments/of19_footprint.txt`). Originally: plan only, nothing run. Written 2026-11-12 (revised the same day after an
independent design review, before any OF19 number) at the user's
request ("work on orderflow trends, exhaustion, absorptions") after OF18.
Committing this file is the pre-registration: data, pattern definitions,
directions, exits, statistics and the decision rule are fixed here before
any OF19 number is computed.

## 1. Question and what is already known

Do the classic footprint patterns (stacked imbalances, absorption,
exhaustion, unfinished auctions, trapped traders) make money on ES after
the real costs, when they are measured on true tick data with the
aggressor side and the bid and ask of every trade?

Already known (the user's knowledge base, `SierraChart-KB/reports/ES
order flow strategies.md`, `performance/orderflow-models/evidence/
strategies.txt`): the bar-level rules were tested out of sample on ESU26 /
ESZ26 at 1.4 ticks per round trip and all lost (delta divergence DD1-DD3
−2.2 to −2.7 ticks per trade, t down to −8.8; exhaustion fade S2 −1.8 to
−3.0; effort-vs-result absorption VA2 −1.68). Those bar-total rules are
exact on 1-s records and are **not retested** here. The per-price rules
(stacks, exhaustion at the extreme, unfinished auction, absorption at a
price) were only approximated there, from 1-s records in which most
volume spans several prices, and the KB's own verdict was to give them
"one clean run on true tick data". OF19 is that run. OF17 and OF18 (A82,
A83) found nothing in order-flow and trend features from 1 s to 15 min.

Expected answer: no.

## 2. Data

The user's Sierra `.scid` tick records (one record per trade, with the
trade price, size, aggressor side and the bid and ask at the trade) on the
front month by RTH volume: the **47** RTH days 2026-09-08 .. 2026-11-11
(ESU26 to 09-11, ESZ26 from 09-14; the OF18 bars' tick days, none with a
gap). **Nothing from 2026-11-12 on is read** (the confirm block). A new
tool writes per-day arrays of the RTH trades (time, price, size, side,
bid, ask) to `build-avx512/real/trades/` (gitignored, sha256 in a
sidecar), checking that the day's volume equals the OF18 bar file's.
A record with both bid and ask volume is split into two trades. The 8
seen OF16 days are inside the span; this study makes no claim on any day
of it (a candidate goes to fresh days, §5).

**Warm-up:** the first 10 days (to 2026-09-21) only feed the trailing
thresholds; **scored days are the other 37** (2026-09-22 .. 2026-11-11).
Every threshold that is a quantile comes from the 10 used days before the
day it is applied to.

Footprint bars: 1-min RTH bars (bar 1 = 08:30-08:31 CT). A(p) = buy-aggressor
volume traded at price p in the bar, Bv(p) = sell-aggressor volume. Lookbacks
use only bars of the same RTH session: a pattern needing N prior bars cannot
fire on bars 1..N.

**Quote check (before any statistic).** The share of buy-aggressor trades
priced at the record's ask and of sell-aggressor trades at its bid
(expected >= 95%; the OF18 review found ~2-3% of trades outside the quote,
stale quotes), and the count of records with bid >= ask or a zero side;
those use the last valid quote. Reported with the results; a share below
90% stops the study.

## 3. Patterns (direction fixed by the practitioner rule; long side shown, short is the mirror)

Primary settings are the user's Sierra study defaults where they exist
(`orderflow-footprint/footprint_core.hpp`: ratio 3, min volume 10,
stack 3, exhaustion threshold 5), else the KB's rule F1-F7
(`practitioner_methods.md`). Two stated deviations: Bv(p − 1) = 0 breaks a
stack (the study's rule; F1's vmin2 branch is not used), and P3's monotone
condition is from F3 (a), not the study.

**Signal time.** For bar-close patterns (P1, P4, P5) it is the bar's end
boundary (e.g. 08:31:00.000 CT), not its last trade. For P2 and P3 it is
the trigger trade's time.

- **P1 stacked imbalance (F1 a).** Diagonal buy imbalance at p:
  A(p) ≥ r · Bv(p − 1 tick), A(p) ≥ 10 and Bv(p − 1 tick) > 0. A buy stack is
  k consecutive prices with a buy imbalance. A bar with a buy stack and no
  sell stack: **long** at its close. Stop 2 ticks below the stack's lowest
  imbalance price. No target. Primary r = 3, k = 3.
- **P2 absorption (F2).** Seconds t are scanned in order; P = the bid of
  the last trade at or before t; S = sell-aggressor volume traded at P in
  (t − W, t]. A candidate when S ≥ the q95 of the same statistic (sampled
  every second, same 30-min time-of-day bucket, trailing 10 days) and no
  trade at or below P − 1 tick in (t − W, t]. It is confirmed if no trade
  at or below P − 1 prints in the next 10 s (the signal time is t + 10 s);
  a candidate that fails confirmation does not use up the slot. After a
  confirmed signal at t, the next t considered on that side is t + W.
  Trigger: **long** at the first trade at or above P + 1 tick within 60 s
  after the signal time; the pending entry is cancelled if a trade prints
  at or below the stop (P − 2 ticks) first. Target 4 ticks above the entry
  fill. Primary W = 30 s, target 4.
- **P3 exhaustion, sense (a) (F3 a, the footprint study's sense).** A bar
  makes a new low against the previous N bars, Bv(low) ≤ e and
  Bv(low) < Bv(low + 1 tick) < Bv(low + 2 ticks) (sellers dry up into the
  low). Trigger: **long** at the first trade above the bar's high within
  the next 5 bars, cancelled if a trade prints at or below the stop (1
  tick below the bar's low) first. No target. Primary N = 1, e = 5.
- **P4 unfinished auction (F4, PR3 trade).** The bar's high has A(H) > 0
  and Bv(H) > 0 and the bar closes at least d ticks below H: **long** at
  its close, target H (the revisit), stop 8 ticks below the entry fill. A
  bar unfinished at both extremes, each at least d from the close, is
  skipped. Primary d = 4.
- **P5 trapped traders (F7).** Bar t breaks below the 20-bar low L with
  the trigger, and one of bars t+1..t+M closes back above L: **long** at
  that close; stop 1 tick below bar t's low. No target. Triggers: *stack*
  = a sell stack (P1's rule, r = 3, k = 3) whose lowest price is in bar
  t's bottom 3 ticks; *delta* = bar t's delta ≤ the q10 of bar delta
  (trailing 10 days, same 30-min bucket; a bar-total rule like those §1
  does not retest, kept as a variant). Primary M = 2, trigger stack.

**Fills and exits (all patterns).** Records sharing one timestamp form
one event (a sweep). Entry: the first event at or after the signal time
+ 100 ms (for P2 and P3, the trigger trade's event + 100 ms); a long
fills at the worst (highest) of the event's asks and trade prices, a
short at the lowest of its bids and trade prices. A signal whose entry
fill is at or beyond its stop is skipped and counted. Stops, targets and
the time cap are checked on events after the entry event. Stop: the first
event with a trade at or through the stop; a long exits at the lowest of
that event's bids and trade prices. Target: a resting limit, filled only
when a trade prints one tick through it, at the target price. Otherwise
exit at the bid of the first event after the time cap: **15 min** (P2:
5 min), or at the bid of the last RTH trade before 15:00:00 CT. Fees $1.50
per side (0.24 ticks per round trip). Positions of a rule may overlap
(per-trade expectancy, not a strategy with position limits).

## 4. Tests

- **Primary:** the 5 patterns at their primary settings, M = 5,
  t_req = 2 + sqrt(2 ln 5) = 3.79.
- **Secondary** (the primaries included), 20 rules, t_req = 2 +
  sqrt(2 ln 20) = 4.45: P1 r ∈ {3, 4} × k ∈ {3, 4}; P2 W ∈ {15, 30, 60} s
  × target ∈ {4, 8}; P3 N ∈ {1, 20} × e ∈ {0, 5}; P4 d ∈ {4, 8}; P5
  M ∈ {1, 2} × trigger ∈ {stack, delta}.

Long and short signals of a rule are pooled. Nothing is tuned.

## 5. Statistics and the decision rule (fixed now)

Per rule over the 37 scored days: trades, skipped signals, win rate, mean
gross and mean net per trade (ticks, at the recorded touch, net of fees),
the first- and second-half net (split at the middle scored day), and the
SE: the largest moving-block bootstrap SE over blocks of 2, 5 and 10
scored days (10000 resamples, seed 0; statistic total net / total
trades). t = net / SE. Stress: net − 0.5 ticks (half a tick more slippage
per round trip).

- **Primary edge:** net > 0, t ≥ 3.79, positive in both halves and at
  stress. **Secondary candidate:** the same with t ≥ 4.45.
- Either goes to **OF19b**: one confirmation on the next 20 full
  front-month RTH days from 2026-11-12 (about four weeks of recording),
  rule frozen, thresholds still trailing; claimed only if net > 0 with
  t ≥ 2.
- **Otherwise:** "No footprint pattern at these settings nets more than
  X ticks per trade" (X = t_req × SE per rule, reported), "smaller edges
  are not excluded."

**Reported, not tested:**
- **Shell control** (does the footprint condition add anything?): each
  rule without its footprint condition, same entries and exits: P2
  without S ≥ q95; P3 a new N-bar low, then a break above the bar's high,
  without the Bv conditions; P4 finished highs (A(H) = 0 or Bv(H) = 0) at
  the same d; P5 break and reclaim without the trigger. P1 has no shell
  (its condition is the whole pattern) and is compared with the random
  control only. Rule gross minus shell gross, with the block SE of the
  difference.
- **Random control:** K = 10 random seconds per signal trade, same day,
  30-min bucket and side (seed 0); each uses the signal trade's stop and
  target as tick distances from its own entry fill and the same cap.
  Rule gross minus control gross, with its block SE.
- **Touch fills:** P2 and P4 with targets filled on touch (an optimistic
  bound).
- **Revisit rates** (F4's benchmark): the share of unfinished and of
  finished highs and lows revisited within 15 and 30 min, at the same
  distance d from the close.
- Trades by time of day; long and short separately.
- **Power:** X per rule.

**Red flags** (stop and check before believing): t > 6; a win rate above
75% on 50+ trades; gross above 3 ticks on trades held under a minute;
check that no record after the signal time (plus 100 ms) is used for the
entry decision, and that the day's trade volume equals the OF18 bars'.

## 6. Known weaknesses (stated before the results)

- **Few days.** 37 scored days, two months, two contracts. P5 is expected
  to fire a few times a day; its X will be several ticks and a null on it
  says little. Rough expectation (the review's, before any data): P4 tens
  to 100+ trades a day, P1 and P3 tens.
- **No targets for P1, P3, P5** (the practitioner rules give none): they
  exit on the stop or the time cap only.
- **Queue position.** Targets fill only on a trade-through, which is
  conservative and penalises P4's revisit thesis (the touch-fill bound is
  reported); stops and entries are market orders at the recorded touch,
  sweep-aware. 1 lot, no impact.
- **Levels.** Practitioners trade these patterns "at key levels" with
  discretion; this tests them everywhere they occur.
- **No book.** Absorption without the displayed bid size (the optional
  refill condition needs depth, recorded on only ~13 days).

## 7. Cost

Light: one pass over ~47 days of tick records (the ESU26 / ESZ26 byte
ranges, ~1.5 GB), per-day arrays, then numpy. Implementation, tests and
review: about 2-3 hours.
