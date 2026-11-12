# OF17 plan: a signal census on real ES (is there anything to trade?)

Status: **plan only, nothing run.** Written 2026-11-12 after OF16 phase 2
(the 16 OF15 runs: no edge on real ES, A79). Committing this file is the
pre-registration: data, predictors, statistics and the decision rule are
fixed here before any census number is computed.

## 1. Question

OF15 and OF16 tested agents. This tests the input they all share. Does any
of the 13 market features the agents observe (FeatureBuilder: delta5,
delta30, delta120, ofi5, ofi30, relvol30, imb_l1, imb_l5, ret5, ret30,
ret120, vwap_dist, absorb), alone or combined, predict the ES mid move by
more than the cost of a 1-lot market-order round trip, at any horizon from
1 s to 10 min?

If nothing does, no policy that trades market orders on these features can
make money on real ES, whatever learner picks it, and the next step is a
different information source or a different way of trading (passive
orders), not more RL. If something does, it is a candidate for a one-time
confirmation on fresh days.

Expected answer: no. OF16 phase 1 measured the strongest short-horizon
signal, the L1 imbalance, at +0.07 ticks over 1 s against a 1.37-tick round
trip. Longer horizons are the untested part.

## 2. Data

The 8 front-month RTH days scored once in OF16 phase 2 (2026-09-08..11,
ESU26; 11-05, 11-09, 11-10, 11-11, ESZ26). They are now role **seen** in
the day manifest (A81): opened freely, logged as `real_seen`, never scored
or claimed on again. The dev days (expiring back month, and one window of
11-03) are not used: they are a thinner market. Confirm days (front-month
RTH from 2026-11-12) are not opened.

Samples are the OF16 decisions exactly: per day, windows k = 0..36 (11-09:
0..32) of 600 decisions at 1 s, each window's features warmed up for
1050 s from k × 600 s (`VecReplayEnv`, flat policy), so each sample's
features are what an agent saw. Decision i of window k fills in bucket
f = 10500 + 6000 k + 10 i (100 ms latency, as in OF16). About 22,200
samples per day (11-09: 19,800), 175,200 in all. Alignment check (as in OF16 phase 1):
feature imb_l1 must equal 2 × the L1 imbalance of bucket f − 1 computed
from the tape's quotes; any mismatch stops the census.

## 3. Outcomes and costs

Horizons h ∈ {1, 5, 15, 30, 60, 120, 300, 600} s; exit bucket f + 10 h
(a sample whose exit is past the tape's end is dropped for that h). From
the tape's end-of-bucket quotes, in ticks:

- move: mid(f + 10 h) − mid(f);
- long net: bid(f + 10 h) − ask(f) − 0.24; short net: bid(f) − ask(f + 10 h) − 0.24
  (0.24 ticks = $1.50 per side; the spreads paid are the recorded ones).

## 4. Predictors (fit on one month, scored on the other)

Two folds: fit on the 4 Sep days and score the 4 Nov days, then the
reverse. Every number in §5 is out of fold (fit and score never share a
day). Within a fold nothing is tuned.

- **S (each feature alone, 13 × 8 = 104 tests).** On the fitting month,
  the feature's 20th and 80th percentiles, and the direction d = sign of
  (mean move in the top quintile − mean move in the bottom quintile). On
  the scored month: long when the feature is above its 80th percentile and
  d = +1 (short if d = −1), the opposite side below the 20th percentile,
  no trade in between.
- **G (all 13 features, LightGBM, 8 × 3 = 24 tests).** A regressor of the
  move at h, fixed parameters (200 trees, learning rate 0.05, 31 leaves,
  min_data_in_leaf 500, feature_fraction 1, bagging off, seed 0, 1
  thread). On the scored month: trade in the sign of the prediction when
  |prediction| is at or above the fitting month's 80th, 95th or 99th
  percentile of |prediction| (3 tests per h).

Each sample is a separate 1-lot trade held for h; trades overlap. This is
a census of per-trade expectancy, not a strategy with position limits.

## 5. Statistics and the decision rule (fixed now)

Per test (128 in all): the out-of-fold trades' mean net per trade, mean
gross (move in the trade's direction), trade count, and a **day-clustered**
SE (the OF16 day bootstrap, B = 10000, seed 0, over the 8 days), t = mean /
SE; the same per month. M = 128, t_req = 2 + sqrt(2 ln 128) = 5.11.

- **Candidate:** out-of-fold mean net > 0 with t ≥ 5.11, AND positive in
  both scored months. A candidate is not a claim: it goes to OF17b, a
  one-time confirmation on the next 8 full front-month confirm days
  (orchestrator, `real.open_tapes("confirm")`), with the rule refitted on
  all 8 seen days and nothing else changed, claimed only if net > 0 with
  t ≥ 2 there. That needs ~2 weeks of recording.
- **Otherwise: no signal to trade.** "None of the agents' 13 features,
  alone or combined, predicts the ES mid by more than a market-order round
  trip at 1 s to 10 min." Market-order RL on these features is closed; the
  next question needs other information or passive execution.

Reported beside the test, not tested: per h the best gross per trade
against the mean round-trip cost paid; LightGBM's out-of-fold R²; per
feature the top-minus-bottom quintile move.

**Red flags** (stop and check before believing): any test with t > 10, or
gross above 2 ticks per trade at h ≤ 5 s: check the fill-bucket alignment
first; the alignment check of §2 failing.

## 6. Known weaknesses (stated before the results)

- **8 days, 2 folds.** Day clustering with 8 clusters is coarse, and a
  month-to-month fit is a regime test as much as a signal test. That makes
  the census conservative: a signal must survive September to November.
- **Seen days.** These days were used to score OF16; nothing was fitted on
  them, and the census selects nothing for a claim on them (a candidate
  goes to fresh days).
- **Features only.** Other information (MBO queue data, other markets,
  news times) is out of scope. So are passive orders: the replay cannot
  model queue position.
- **Costs.** 1-lot market orders at the recorded touch, no impact (1 lot in
  ES), $1.50 per side. A signal that clears only zero-fee costs does not
  count.

## 7. Cost

Light: one process, single-threaded. The observation pass is a replay of
292 windows (minutes); 16 LightGBM fits on ~86k rows each (seconds each).
