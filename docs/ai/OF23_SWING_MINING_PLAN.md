# OF23 plan: mining turning-point entries with wide exits on real ES bars

Status: **plan only, nothing built or run.** Written 2026-11-13 at the
user's request ("let it hold trades, try to pick tops and bottoms, an
extended profit target ... train it like a data mining algorithm");
revised the same day after an independent design review (verdict
needs-work, 14 findings, all taken), before anything was built.
Committing this file is the pre-registration: data, candidates, exits,
features, the mining procedure, splits, statistics and the decision rule
are fixed here before any OF23 label, model or outcome is computed.
OF21 (forward test of exhaustion at levels) and OF22 (RL on real bars)
are separate pre-registrations; OF23 shares their data and conventions
but none of their code.

## 1. Question

OF18 found no walk-forward rule on trend features that pays at a fixed
5-15 min hold, and OF19-OF20 found no footprint setup that pays with a
4-tick target. No study here has let a trade run to a wide target, or
searched entries aimed at turning points. With a 24-tick target the
1.28-tick round trip is ~5% of the win instead of ~30%, so costs stop
being what decides.

The exit shape alone cannot make money: on a price path with no
information at the entry, any mix of target, stop and time cap has an
expected value of zero minus costs; it only trades win rate against
payoff. So the question is about entries:

1. **Primary:** a mining procedure that, inside each training range only,
   learns (gradient-boosted trees on bar features) which turning-point
   candidates to take and chooses one exit from a grid of 240: does it
   make money after costs on the months it never trained on, and more
   than the same exits on random entries?
2. **Secondary:** the same procedure per candidate family, and the same
   without the model (candidates of the family, exit chosen in
   training): does any family carry information?
3. **Descriptive:** what naive data mining would have claimed (the best
   combination chosen on all 228 days at once) and how much of it the
   deflated Sharpe ratio and the walk-forward remove.

Expected answer: no edge (OF18's null on the same bars, OF17's negative
out-of-fold R² at every horizon). The new information is whether entries
at turning points carry any edge with wide, asymmetric exits, and how
large an edge the design could have detected (section 7: about 1.5
ticks per trade for short exits, 4-6 for wide ones).

## 2. Data

- **Bars:** the OF18 cache `build-avx512/real/bars/<date>.npz` (built by
  `tools/scid_to_bars.py`): 390 one-minute RTH bars per day (08:30-15:00
  CT), OHLC in ticks, volume, buy and sell aggressor volume, trades; front
  month by RTH volume (OF13). **Days:** the 228 used days 2025-12-15 ..
  2026-11-11 of OF18's manifest (a day with more than 5 gap minutes is
  dropped, so the 7 holiday and half days are out); the manifest's
  sha256 is logged with the results. Real data never goes into the
  repository.
- **Path inside a bar:** the order in which a bar reached a target and a
  stop is taken from the day's `.scid` records of the front contract
  (the reader of `tools/scid_to_bars.py`): 1-s bundle records through
  2026-09-05, tick records from 2026-09-08 (47 days). Only when one
  second (bundles) or one record touches both does the stop count first.
- **Cutoff:** nothing dated on or after 2026-11-12 is read (the
  converter and `load_days` refuse; those days are OF21's and the confirm
  block's). OF23 makes no ledger opening: like OF18-OF20 it reads only
  days already seen; any OF23b on fresh days needs its own plan and
  ledger claim first (section 7).
- **Gaps:** a zero-volume minute is a gap. No candidate fires on a bar
  whose 30-bar window contains a gap; an open trade that meets a gap bar
  exits at the next bar's open at market (cost as a time exit).
- **Rolls:** the cache holds the front contract only, so on the first
  day after a front-month switch (about 4 days) there are no prior-day
  candidates (C3) and the prior-day features are missing (NaN, which
  LightGBM handles natively).

## 3. Candidates (entries)

All on **completed bars**: a candidate is decided at the close of bar m
and entered at the open of bar m + 1. **ATR** = mean true range of the
last 20 one-minute bars, where bar 0's true range is its high − low (no
overnight gap); candidates fire from bar 20 (08:50) to 14:30 (so a
30-min cap fits). Each candidate has a direction (long at a low, short
at a high). Comparisons are strict; ties make no candidate.

- **C1 swing reversal:** a 5-bar swing low at bar k (low(k) < low(j) for
  every j in k−2..k+2, j ≠ k) or high, on 1-min, 5-min and 15-min bars
  (5- and 15-min bars built from the 1-min bars on the clock, completed
  bars only), confirmed at the close of k + 2 of that timeframe; long
  after a swing low, short after a swing high. One indicator per
  timeframe.
- **C2 failed session extreme:** after 09:00, bar m's high exceeds the
  running session high of bars 0..m−1 and bar m closes below that prior
  high: short (mirror: long at the low).
- **C3 prior-day level fade:** bar m trades through a prior-day level of
  the same contract (pdh, pdl, pdc, ppoc, pvah, pval; value area as in
  OF20, from the prior day's bar volume spread evenly over each bar's
  range) by at least 1 tick and closes back on the side it came from:
  fade toward that side. Several levels on one bar make one candidate
  (features: count of levels, name of the nearest).
- **C4 VWAP band fade:** VWAP from 08:30 at typical price (H+L+C)/3
  weighted by bar volume, SD = sqrt(Σ v·(tp − vwap)² / Σ v), both over
  bars 0..m; from 09:30, bar m's high reaches vwap + 2 SD and it closes
  below it: short (mirror: long).
- **C5 FVG retest:** on 5-min and 15-min bars, bar i's low above bar
  i−2's high is a bullish gap [high(i−2), low(i)] (mirror: bearish), open
  from the close of bar i until a bar of its own timeframe closes beyond
  its far edge. The first 1-min bar after the gap opens whose range
  overlaps the gap is its one retest: long into a bullish gap, short into
  a bearish one, unless that 1-min bar closes beyond the far edge.
- **C6 bar exhaustion:** from 09:00, bar m's range ≥ 2 ATR, its volume ≥
  the 90th percentile of bars 0..m−1, its high > the highs of the 29
  bars before it, and it closes in its lower third: short (mirror: long).

Candidates on the same bar and direction merge into one (families become
indicator features); candidates of both directions on one bar are
dropped. In scoring only one OF23 trade is open at a time; a candidate
during an open trade is skipped.

## 4. Exits, fills and costs

**Exit grid (240):** target T ∈ {8, 12, 16, 24, 32, 48} ticks × stop
S ∈ {4, 6, 8, 12, 16} × time cap H ∈ {30, 60, 120 min, end of day} ×
trailing ∈ {off, on}. Trailing on: once the trade's best completed-bar
close is ≥ S ticks in profit, the stop moves to (best close − S), updated
at bar closes and effective from the next bar.

**Fills:**
- Entry: market at open(m + 1) (the first trade of that minute).
- Target: a resting limit, filled only when price trades through it by
  one tick (long: a trade at entry + T + 1 or higher), at the target.
- Stop: a stop-market order, triggered when price touches it, filled one
  tick worse than the stop; if a bar opens beyond the stop, at that open
  one tick worse.
- Order within a bar: from the records (section 2); the stop first only
  when one second (bundles) or one record reaches both.
- Time cap: market at the open of the first bar after H; end of day: the
  close of the 14:59 bar.

**Costs (per side, ticks):** market or stop 0.64 (half the measured
1.28-tick front-month round trip, $1.50 fee included), limit target 0.12
(OF22's convention). **Stress:** 0.75 / 0.20 and stop fills 2 ticks worse.
One tick = $12.50; results are in ticks per trade.

**Resolution check (binding):** on the 47 tick days, every candidate ×
the exits chosen in any fold is resolved twice, from the records
aggregated to 1-s bars (as the bundle months are) and from the tick
records, with the same costs and entry. The mean difference (1-s minus
tick) must have its two-sided 95% upper bound ≤ +0.25 ticks per trade,
or the edge claim fails. Reported: the share of bars that touch both
target and stop, per month and exit class, with bars alone and after
the records, and the stop-first bar-only net beside the resolved net.

## 5. Features (for the model; all known at the close of bar m)

Candidate family indicators (C1 per timeframe, C2-C6), C3 level count
and nearest level name, direction; distance in ATR to the session
extreme on the entry side and on the far side, to the session open and
to VWAP; VWAP z; ATR in ticks; bar range / ATR, close location in the
bar; relative volume (bar volume / mean of the 30 before it); delta (buy
− sell volume) over 1, 5, 15 and 30 bars, divided by their volume;
returns over 5, 15 and 60 bars and since the open, in ATR; EMA(8) slopes
of 5-, 15- and 60-min bars (completed only); distance to the nearest
prior-day level in ATR; count of open 5- and 15-min FVGs and distance
to the nearest; minutes since the open; day of week. Signed features
(delta, returns, slopes, VWAP z) are multiplied by the direction, so
"long into weakness" and "short into strength" share a sign.

**Leak test (binding, in the test suite):** every candidate and feature
computed on a day truncated at bar m equals the full-day value at m.

## 6. Mining procedure (nested walk-forward)

**Labels:** for each candidate and exit e, net ticks (section 4, base
costs). Labels never cross a day (every exit ends by the close), and
every feature looks only back, so inner blocks by month do not leak.

**Outer folds (test months):** 2026-03 .. 2026-11 (9 folds; November
ends 2026-11-11), as OF22. Each fold trains on all used days before the
test month. Folds 1-2 (March, April) train on ~50 and ~70 days and are
reported as low-power.

**Inside a fold, only training days:**
1. **Inner blocks** = calendar months of the training range, with
   December 2025 merged into January 2026. For each exit e, a LightGBM
   regressor on the label is fitted with one block left out at a time
   and predicts it, giving an out-of-fold prediction for every training
   candidate. Parameters fixed: objective l2 on the label clipped to ±60
   ticks, num_leaves 15, min_data_in_leaf 100 (lowered to n/20 when a
   fit has fewer than 2,000 rows), learning_rate 0.03, 200 rounds,
   feature_fraction 0.7, bagging 0.7 every 1, lambda_l2 10, seed 0,
   deterministic true, force_row_wise true, num_threads 4.
2. **Simulate in order:** for each (e, q), q ∈ {top 10%, 25%, 50%,
   100%} of training candidates by out-of-fold prediction (cut = that
   quantile of the training out-of-fold predictions), walk each training
   day in time order, taking a candidate above the cut only when no
   trade is open, exiting by e. This gives the non-overlapping training
   net per day and trades per day.
3. **Choose (e, q)** with the largest non-overlapping training net per
   day among those with ≥ 0.5 trades per day; if none reaches 0.5, the
   one with the most trades per day. Ties: smaller q, then smaller T,
   smaller S, shorter H, trailing off. The fold trades its choice even
   when the best training net is ≤ 0 (that is the procedure).
4. **Test month:** each candidate's prediction is the mean of the inner
   models for e (the same scale as the cut); take it when above the cut
   and no trade is open; exit by e.

The test month is touched once per fold by the fixed policy. Every
choice (exit, cut, models) is made inside training, so the 9 test
months together score **one** procedure: the primary has M = 1.

**Secondaries (M = 13):** the same procedure restricted to each family
(6); the same per family without the model (q = 100% only; e chosen by
step 2-3: 6); and the pooled procedure without the model (1).
t_req = 2 + sqrt(2 ln 13) = 4.26.

**Naive mining, descriptive only:** the same steps 1-3 run once on all
228 days (blocks = all months), and its chosen combination scored on
those same 228 days' out-of-fold predictions: net, t and **deflated
Sharpe ratio** (Bailey and López de Prado), with trials = 7 pipelines
(pooled + 6 families) × 240 exits × 4 q with the model, plus 7 × 240
without, reduced to an effective count from the trials' mean pairwise
correlation (N_eff = 1 + (1 − ρ̄)(N − 1)). Shown beside the walk-forward
result; never a claim.

## 7. Statistics and decision rule

**Unit:** net ticks per trade on the test months, pooled over folds;
also net per day. **SE:** moving-block bootstrap over test days, the
largest over blocks of 5, 10 and 21 days, B = 10,000, seed 0 (OF18).

**Random-time control (binding):** per test day, the procedure's number
of trades, placed in sequence (draw a bar from the empirical time-of-day
distribution of the procedure's entries in that fold, restricted to the
bars after the previous exit; hold to the exit; repeat), direction drawn
with the procedure's long share, exit e of that fold; 200 replicates,
seeds 1-200. **Random-candidate control (reported):** the same count
drawn at random from all that fold's candidates (own directions), in
sequence with the skip rule; it separates "the candidates carry
information" from "the model carries information".

**Edge (all required):**
1. Primary t = mean / SE ≥ 2.
2. Positive net in both halves (test months 03-07 and 08-11).
3. Positive net at stress costs.
4. Above the 95th percentile of the 200 random-time replicates' mean.
5. The resolution check passes (section 4).
6. ≥ 0.5 trades per test day on average.
7. Positive mean without the 5 best test days (LESSONS 61).

A secondary is an edge only under the same seven with t ≥ 4.26.

**A pass is never a claim:** it goes to OF23b, a fresh pre-registration
on days from 2026-11-12, written before any such day is read for it and
stating how its days are shared with OF21 (the ledger counts OF21's look
days as used confirm days).

**A null** is reported with its bound X = max(mean, 0) + 2 SE ticks per
trade and the detectable size at t = 2 for the exit actually chosen
(LESSONS 57). Expected detectable size by exit class (assumed per-trade
sd and trades per test day, ~180 test days, SE inflated 1.3× for
clustering): short (H ≤ 60 min, T ≤ 24): sd 10-15, ~3 a day, about
1.2-1.8 ticks; wide (T ≥ 32 or end of day or trailing): sd 25-40,
1-2 a day, about 4-6 ticks.

**Main table:** per fold the chosen exit and q, training non-overlapping
net per day, test net per day and per trade, trades per day; the pooled
test result beside the pooled training out-of-fold result (how much
selection inflated it).

**Reported, not tested:** win rate, average win and loss, holding time;
median trade and mean without the top 1% of trades; net by family, by
month and by direction; both controls' distributions; the two-touch
shares; the naive-mining table.

**Red flags (stop and check before believing):** t > 6; test net above
training out-of-fold net in most folds; one month carrying more than
half of the total; win rate more than 15 points above the random-time
control at the same exit; any candidate or feature failing the leak
test; resolution bias above 0.25 ticks.

## 8. Overlap, reuse and honesty

- **No test month is unseen.** OF17-OF18 read all 228 days and OF19-OF21
  the 47 tick days. OF23's walk-forward keeps its own choices out of its
  test months, but the families and features were proposed by people who
  have seen those studies' results on these days, and inner blocks train
  on months later than the one they predict (regime-optimistic training
  net, hence the main table). That is why a pass is only a lead for OF23b.
- **Shared with OF22:** the test months and the idea of FVG and level
  features; no code. OF23 imports no OF22 module and edits no file frozen
  by OF21 (its plan §8).
- **Build Alpha (descriptive only).** The user offered Build Alpha (a
  commercial strategy miner; user guide v3 and the v3.2 update page read
  2026-11-13). It is not installed on the server; installing it is the
  user's call. It cannot carry a claim here: the guide documents no random
  seed or deterministic rerun, no count of strategies tried, no correction
  beyond its Vs Random benchmark, no tick-denominated exits, and its
  same-bar target/stop rule on 1-min data is unstated. Its role is fixed:
  1. It runs only **after** OF23's nine test months are scored and
     `of23_results.txt` is committed, so it never touches a test month
     first; never on days from 2026-11-12.
  2. Input: the same 228 days as 1-min RTH bars of the front month
     (CSV, mm/dd/yyyy, hh:mm, written outside the repository), ES point
     value 50, tick 0.25, round-trip costs = the base 1.28 ticks ($16).
  3. Its in-sample/out-of-sample split and walk-forward are set to the
     OF23 fold boundaries; Vs Random, Noise and MC Permutation on; the
     run's settings file and the number of strategies it reports are
     kept with the results.
  4. Any strategy it surfaces is a lead only: exported as code, re-
     implemented in OF23's engine with OF23's fills and costs, and frozen
     in an OF23b plan before any forward day is read for it.
- **Amendments** are dated commits to this file before the first OF23
  outcome is computed; after that, only reported deviations.

## 9. Code and reproducibility

- `experiments/of23_mine.py`: day loading (OF18's loader and manifest),
  record paths, candidates, features, labels, the nested walk-forward,
  controls, report; one command regenerates
  `experiments/of23_results.txt`. Imports `ragrl` first (the WMI
  workaround needed before `import lightgbm` on this server); logs the
  LightGBM version, the manifest sha256 and its own sha256.
- **Freeze:** before the first full run, an amendment records the sha256
  of `of23_mine.py` (CRLF read as LF, LESSONS 62) and `test_of23.py`
  checks it, so nothing changes after test months are scored without
  showing.
- `python/tests/test_of23.py`: leak test (truncation); fill rules on
  synthetic bars and records (target trade-through, record order inside
  a bar, stop first in a same-second touch, gap through the stop,
  trailing update timing, end of day); each candidate family on
  synthetic bars, including ties and the merge rule; in-order simulation
  with the skip rule; refusal of days from 2026-11-12.
- **Compute:** about 15,000 small LightGBM fits for the primary and
  about 105,000 with the secondaries (240 exits × inner blocks × 9 folds
  × 7 pipelines), ~0.5 s each, 6 at a time at 4 threads, plus a record
  pass for the path inside bars: about 3-4 hours on the server. The user
  is told the estimate before the run.
