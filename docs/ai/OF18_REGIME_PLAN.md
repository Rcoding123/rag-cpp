# OF18 plan: month-scale momentum and reversal regimes in ES (5-30 min)

Status: **plan only, nothing run.** Written 2026-11-12 after OF17 (A82),
revised the same day after an independent design review (before any OF18
number existed). Committing this file is the pre-registration: data,
features, rules, statistics and the decision rule are fixed here.

## 1. Question

OF17 found one lead. At 2-10 min, trend features predict the ES move
within a month, with a sign that flips between months. The clearest case
is vwap_dist at 600 s: the top-minus-bottom quintile move was +10.0 ticks
in November 2026 (momentum) and −7.3 in September (reversal). Eight days
could not say more. The question here:

**Is the sign of that relationship persistent enough that a rule
estimating it from the last few weeks, and trading it the next day, beats
a market-order round trip?**

One **primary** test aims at the lead itself (§5); 63 **secondary** rules
map the neighbourhood at a multiplicity-adjusted bar. Expected answer: no.
Intraday momentum and reversal in index futures are weak and unstable,
and the OF17 reviewer's walk-forward variant (direction from the previous
day) reached t 1.8.

**Power, stated before the results** (the design review's estimate, 1-min
SD ~9 ticks, ~40% of decisions trading): the smallest net per trade a
secondary rule can detect is about 1.5-1.9 ticks at 5 min, ~2.7 at 10 min,
~3.9 at 15 min; the primary test (t ≥ 2) about 1.1 ticks at 10 min. OF17's
lead implies at most ~4-5 ticks gross per trade at 10 min if the sign were
always right, so a null result here is an upper bound, never "no regimes".

## 2. Data

The user's Sierra `.scid` trade files (read only; nothing enters the repo):
ESH26, ESM26, ESU26, ESZ26 in `C:\SierraChart\Data`. Surveyed 2026-11-12:
records are 1-s bundles (open, high, low, close of trade prices, trades,
volume, sell-aggressor "bid" volume, buy-aggressor "ask" volume) through
August 2026, and tick by tick from early September 2026, where each
record's high is the ask, its low the bid and its close the trade price.

- **Days:** every RTH day (08:30-15:00 CT, the OF16 clock with the explicit
  US DST rule) from **2025-12-12 to 2026-11-11**. 2025-12-12 is the day
  after the CME roll date for ESZ25 (2025-12-11), the first day ESH26 is
  the front month; there is no ESZ25 file, so earlier days would be the
  back month. From then on the front month is the file with the larger
  RTH volume that day (the OF13 rule; both contracts of each later roll
  have files). **Nothing from 2026-11-12 on is read** (the confirm block;
  the bar builder refuses those days).
- **Bars:** 1-min bars built from the records by a new tool, one file per
  day in `build-avx512/real/bars/` (gitignored) with the file's sha256 in a
  sidecar. For 1-s bundles: the first bundle's open, the maximum high, the
  minimum low, the last bundle's close (a bundle never spans a minute
  boundary). For tick records: open, high, low and close all from the
  records' close field (the trade price). Volume = the sum of volume; buy
  aggressor = ask volume, sell aggressor = bid volume.
- **Outages:** a minute with zero volume is a gap. A decision is skipped if
  any minute of its feature window (the 30 min before it) or of its
  holding period is a gap; a day with more than 5 gap minutes is dropped
  and listed. Early closes drop out by the same rule.
- **Amendment 2026-11-12 (bars built, before any OF18 statistic).** The
  bars show ESH26 at 150,046 RTH lots on 2025-12-12 against 847,556 to
  1,664,448 on 12-15..12-18, and the March roll's volume crossing on the
  Monday after the CME roll date (03-16): 12-12 was still the back month
  by volume, the rule above. The span starts **2025-12-15**. Built: 236
  weekdays with trades; 7 holiday or half sessions have 165-180 gap
  minutes and drop out; 228 used days; 0 OHLC violations; tick records
  from 2026-09-08.
- The 13 OF16 days (8 seen, 5 dev) are inside the span. They helped form
  the hypothesis, so a sensitivity run without them is reported (§5).

## 3. Decisions, features, outcomes

Bars are numbered 1..390 (bar 1 = 08:30-08:31 CT); close(0) = open of
bar 1.

**Decisions** at the end of bars m = 30, 35, 40, ... while m + 1 + h ≤ 390:
71, 70, 69 and 66 decisions a day for h = 5, 10, 15 and 30 min.

**Features** at a decision (bars 1..m of the day only), 7 in all:
- ret_L = close(m) − close(m − L), L = 5, 15, 30 min;
- delta_L = Σ (buy − sell aggressor volume) over bars m − L + 1..m,
  L = 5, 15, 30;
- vwap_dist = close(m) − session VWAP, the VWAP over bars 1..m weighted by
  volume at each bar's (high + low + close) / 3.

**Outcomes** at h = 5, 10, 15 (tested) and 30 min (descriptive only), in
ticks: entry at the open of bar m + 1 (the first trade after the
decision), exit at the open of bar m + 1 + h; move = exit − entry. A
trade nets side × move − **1.28 ticks** (OF17's measured mean round trip on
the front month: the recorded spreads at entry and exit plus $1.50 per
side). Stress: 1.50 ticks. Nothing crosses a day.

## 4. Rules (walk-forward by day)

For a feature, a horizon h and a lookback W (trading days), on each scored
day D:
- from the W used days before D only: the feature's 20th and 80th
  percentiles, and the direction d = sign(mean move at h in the top
  quintile − mean move in the bottom quintile);
- on D: when d = +1, long above the 80th percentile and short below the
  20th; when d = −1 the reverse; when d = 0, no trade.

**Primary:** vwap_dist, h = 10 min, W = 20 days.
**Secondary:** the 7 features × h ∈ {5, 10, 15} × W ∈ {5, 20, 60} = 63
rules (the primary is one of them).
**Descriptive:** the same at h = 30 (21 rules), not tested.

Scored days: every used day with at least 60 used days before it, the same
days for every rule, about 165. Nothing is tuned.

## 5. Statistics and the decision rule (fixed now)

Per rule over the scored days: trades, mean gross and mean net per trade
(total net / total trades), the first-half and second-half net (split at
the middle scored day), and the SE: **the largest of the moving-block
bootstrap SEs with blocks of 5, 10 and 21 consecutive scored days**
(B = 10000, seed 0; each resample's statistic is total net / total trades
over the resampled days), since a regime that persists makes a
walk-forward rule's days correlated for up to the regime's length.
t = mean net / SE.

**Bias check (binding).** On the scored days from 2026-09-08 on, which
have bid and ask in the records, every trade is also priced at the mid
(the mean of the ask and bid in the record of the entry and exit trades):
bias = side × [(exit trade − exit mid) − (entry trade − entry mid)], per
trade, averaged per rule and reported separately for rules whose d agrees
with the feature's sign (momentum) and disagrees (reversal). A rule's
net after bias = mean net − its mean bias.

- **Primary edge:** net > 0, t ≥ 2, positive in both halves, positive at
  the stress cost and after its bias.
- **Secondary candidate:** the same with t ≥ t_req = 2 + sqrt(2 ln 63) =
  4.88.
- Either goes to **OF18b**, one confirmation on the next 40 full
  front-month RTH days from 2026-11-12 (about two months of recording):
  the rule frozen and still walk-forward, each trade at the recorded touch
  (buy at the ask, sell at the bid, plus $1.50 per side), the SE the
  largest of 5- and 10-day block bootstraps; claimed only if net > 0 with
  t ≥ 2.
- **Otherwise the conclusion is an upper bound:** "No walk-forward regime
  rule on these 7 trend features nets more than X_h ticks per trade at h"
  (X_h = the median t_req × SE over the rules at h, and 2 × SE for the
  primary), "smaller edges are not excluded."

**Reported, not tested:**
- **Persistence**, apart from any rule: per feature, h and W, the
  correlation over scored days between the trailing-W estimate of the
  top-minus-bottom quintile move and day D's own value, with the
  block-bootstrap SE (largest of 5, 10, 21). This is the regime measured
  directly.
- Power: X_h per h.
- Trades by time of day (the percentiles are pooled over the day, so
  trades may cluster after the open).
- Sensitivity: the primary and the 5 best secondary rules without the 13
  OF16 days.

**Red flags** (stop and check before believing): t > 8, or gross above
3 ticks per trade at h = 5 min (a candidate at 5 min would be near this
size; check before believing it): check that features use bars 1..m only
and the entry is bar m + 1's open. The bar builder's daily volume must
equal the sum of the records' volume. Every bar must have high ≥ max(open,
close) and low ≤ min(open, close), and the distribution of bar-to-bar
price changes must not shift at the switch from bundles to ticks (early
September).

## 6. Known weaknesses (stated before the results)

- **Power.** See §1. A null is an upper bound.
- **Trade prices, not quotes.** Through August, entries and exits are
  trade prices with a constant cost; the bias check measures the effect
  where quotes exist (about 45 scored days) and binds a candidate to it.
- **One year, one market.** About 165 scored days and a few regime
  periods; the block bootstrap only partly reflects how few.
- **Overlap.** At h = 15, decisions 5 min apart overlap three-fold; the SE
  comes from day blocks, not trade counts.
- **Features.** Seven trade-based trend features, no book (OF17 covered
  the book features up to 10 min on 8 days).
- **Market orders only.** No passive execution, no sizing.

## 7. Cost

Light. The bar builder reads each day's byte range of the `.scid` files
once (about 3 GB, the tick-by-tick months most of it) and caches small bar
files; the statistics are seconds of numpy (the bootstraps a few
minutes). Implementation and review: about 1.5-2 hours.
