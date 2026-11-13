# OF22 plan: RL agents trained on real ES, aggressive vs passive, with order flow, multi-timeframe, FVG/IFVG and volume profile

Status: **plan only, nothing run.** Written 2026-11-13 at the user's
request ("the agent should learn based off of order-flow and
multi-time-frame analysis, along with things like FVG, IFVG, volume
profile analysis ... it needs to learn on real data"; then "I really want
to investigate the RL method, we need to test passive, aggressive trading
style"). Revised the same day after an independent design review, before
anything was built. Committing this file is the pre-registration: data,
features, environments, learners, splits, statistics and the decision
rule are fixed here before any OF22 model is trained or any outcome
computed. (OF21, the forward test of exhaustion at levels, is
pre-registered separately.)

## 1. Question

Every earlier agent was trained in the simulator (OF2-OF15), traded only
market orders, saw 13 short-window order-flow features, and scored no
edge on real ES (OF16 phase 2, A79). OF17-OF20 found no edge in
hand-written real-data rules. No learner here has seen multi-timeframe
structure, fair value gaps (FVG), inverse FVGs (IFVG) or session volume
profiles, been trained on real days, or traded with limit orders on real
data. The questions:

1. **RL on real data, by trading style:** three PPO agents with the same
   features and training (aggressive = market orders; passive = limit
   orders; mixed = the agent chooses each time), trained walk-forward only
   on real days before each test month: does any make money after costs on
   the months it never saw?
2. **Passive vs aggressive:** with fills judged conservatively, does
   trading passively beat trading aggressively, per test day?
3. **Information (secondary):** do the multi-timeframe, FVG/IFVG and
   volume-profile families predict the next 5-30 min better than order
   flow alone (a gradient-boosted model's out-of-sample rank IC)?

Expected answer: no edge for any style (OF17's negative out-of-fold R² at
every horizon; OF18's null on the same bars). The passive arm's result is
the new information: a passive style pays ~0.12 ticks per side instead of
~0.64, but its fills are adversely selected.

## 2. Data

`build-avx512/real/bars` (OF18, `tools/scid_to_bars.py`, unchanged): 1-min
RTH bars of the front month (08:30-15:00 CT, bars 1..390): open, high,
low, close (ticks), volume, buy and sell aggressor volume, and on tick
days the bid and ask of each bar's first trade. **Days, gaps and outages
as in OF18 §2 and its amendment:** span 2025-12-15 .. 2026-11-11, 228 used
days; a minute with zero volume is a gap; a day with more than 5 gap
minutes is dropped and listed (early closes drop out). In the GBM a
decision is skipped if any minute of its holding period is a gap; in the
RL environments gap bars stay in the episode (≤ 5 per day): **no order fills in a gap bar**; a market change waits for
the next non-gap open and a limit order lapses unfilled.
1-s bundles through August, tick by tick from September. Nothing on or
after 2026-11-12 is read.

**Contract rolls.** Earlier days' prices are shifted onto the current
contract by the roll offset: the new contract's volume-weighted trade
price minus the old contract's over 08:00-08:30 CT on the switch day (if
either has fewer than 50 lots there, the last 30 RTH minutes of the prior
session), both from the existing .scid files (a new small reader in
`experiments/of22_features.py`; dates before the cutoff only). Every
multi-day price feature (prior session, composite profile, naked POCs,
carried FVGs and IFVGs, daily bars) uses the shifted prices.

**Walk-forward split (fixed).** Test months 2026-03 .. 2026-11 (9 folds).
For test month k the learners use only days before its first day. The
first 5 used days are context only (never a sample). The day counts per
fold are written to the results file from the built list before any
training. Nothing crosses a day: positions and labels end by the close.

All 228 days were read before by OF17/OF18 (bars, other features); the 13
OF16 days helped form earlier hypotheses. A sensitivity run without those
13 days is reported.

## 3. Features (at the close of bar m, from bars 1..m of the day and earlier days only)

Python, `experiments/of22_features.py`, precomputed per day into arrays
(no C++ or environment change; the sealed runs are untouched). Scale:
`atr20` = the median 1-min range (high − low, ticks) over the last 20 used
days (fewer at the start, minimum 5); distances are in atr20 units.

- **OF (bar order flow):** delta = buy − sell over the last 1, 5, 15, 30,
  60 bars (fewer at the open), each over the volume of the same bars;
  session cumulative delta / session volume; delta divergence for L = 15
  and 60: +1 if close(m) is the highest close of the last L bars and the
  delta over the last L bars is below the mean of the delta over the 5
  preceding non-overlapping L-bar windows of the same day and earlier
  days, −1 for the mirror, else 0; relative volume = the last 5 bars'
  volume over the mean of the same 5 minutes on the last 20 used days
  (minimum 5).
- **MTF (multi-timeframe structure):** 5, 15 and 60-min bars resampled
  from the 1-min bars, session-aligned from 08:30 (the 14:30-15:00 piece is
  a 60-min bar of 30 min), plus daily bars of earlier days. **Only
  completed bars** of a timeframe enter its EMA, swings and returns; the
  forming bar enters only as "return since its open" and "close position
  in its range so far". Per timeframe: return over the last 1 and 3
  completed bars; close position in the last 3 completed bars' range;
  EMA(8) of completed closes: slope sign and distance; distance to the
  last confirmed swing high and low (a swing high at bar k: high(k) above
  the highs of k−2, k−1, k+1, k+2, confirmed at the close of k+2; mirror
  for lows). **Alignment** = the sum of the EMA(8) slope signs over the 1,
  5, 15, 60-min and daily timeframes (−5..5). Also: return since the open;
  close position in the prior day's range; the open's gap from the prior
  close.
- **FVG / IFVG** on 1, 5, 15 and 60-min bars. A bullish FVG forms at the
  close of completed bar k when low(k) ≥ high(k−2) + 1 tick (zone
  [high(k−2), low(k)]); bearish when high(k) ≤ low(k−2) − 1 tick. The three
  bars lie in one session. Fill and mitigation are tested on 1-min bars:
  **mitigated** when a 1-min bar first enters the zone; **filled** when a
  1-min bar trades through its far edge (bullish: a low ≤ high(k−2)), and
  it stops being an FVG. A filled gap **becomes an IFVG** if the first
  completed bar of its own timeframe that ends at or after the fill closes
  beyond the far edge (bullish FVG → bearish IFVG, mirror), else it is
  dead. An IFVG dies when a completed bar of its timeframe closes beyond
  its other edge. 15 and 60-min FVGs and IFVGs carry over from the last 5
  sessions (roll-shifted); 1 and 5-min ones live within the session. Per
  timeframe: signed distance to the nearest unfilled bullish FVG below and
  bearish FVG above, and to the nearest live IFVG above and below; their
  size and age (in bars of the timeframe); inside-a-zone flags; counts of
  unfilled FVGs and live IFVGs each way; "formed at this bar" and "first
  mitigated at this bar" flags (missing distance: a fixed sentinel of 50
  atr20 plus an "absent" flag).
- **VP (volume profile):** price-volume histograms from the 1-min bars,
  each bar's volume spread evenly over its low..high ticks (coarse VAP; one
  method for bundle and tick days). Prior session: POC (ties: nearest the
  session VWAP, then lower), 70% value area (OF20's one-tick expansion);
  HVNs and LVNs on the profile smoothed by a centred 5-tick mean: an HVN is
  a local maximum holding ≥ 1% of the session's volume at that tick, an
  LVN a local minimum holding ≤ 0.2% that lies between two HVNs. Developing
  session: POC and value area from bars 1..m. 5-day composite: POC and
  value area. Naked POCs: the last 5 sessions' POCs no later 1-min bar's
  range has covered. Features: signed distance to each POC and value-area
  edge, to the nearest HVN and LVN above and below, to the nearest naked
  POC above and below; position relative to the prior and the developing
  value area (−1 below, 0 inside, +1 above); the developing POC's change
  over the last 30 and 60 min; the share of session volume within 2 ticks
  of close(m).
- **Time:** minute of the session.

**No-lookahead test (unit test, binding):** on 3 dev-role days and on
synthetic data, every feature at every bar m equals the same feature
computed from a dataset truncated after day D's bar m (no later bar of D,
no later day loaded). The environments are tested the same way: the
observation at bar m must be identical when every bar after m is
perturbed. FVG/IFVG states are checked against a brute-force
scan; the roll shift against a hand-computed case.

**Switch check:** the mean and SD of each OF feature over August (bundles)
and September (ticks) are reported; a family whose SDs differ by more than
a factor of 1.5 is flagged in the results (descriptive).

## 4. RL environments (`experiments/of22_env.py`, pure Python, gymnasium)

One episode = one training day. Decisions at the close of bars m = 15 ..
388. Position −1, 0 or +1 lot. Equity is marked at bar opens. Costs, in
ticks per lot per side: **market 0.64** (half of OF17's measured 1.28-tick
front-month round trip: spread plus $1.50 per side), **limit 0.12** ($1.50
fee only). A flip from −1 to +1 is two lots.

- **Aggressive arm:** Discrete(3), the target position. A change fills at
  open(m+1) at market cost.
- **Passive arm:** Discrete(3), the target position. A change places a
  limit order for |Δ| lots, live during bar m+1 only: a buy at
  close(m) − 1 tick, a sell at close(m) + 1 tick (about joining the bid or
  offer). **Fill rule (primary, conservative, trade-through):** a buy at L
  fills at L if low(m+1) ≤ L − 1 tick; a sell if high(m+1) ≥ L + 1 tick
  (an order already marketable at open(m+1) also fills at L: slightly
  conservative).
  Unfilled orders lapse at the end of bar m+1 (the agent decides again).
  No queue model: a touch is not a fill.
- **Mixed arm:** Discrete(6), target position × {market, limit}; when the
  target equals the position nothing happens.
- **Reward at step m:** r = p_old × (F − open(m+1)) + p_new ×
  (open(m+2) − F) − costs, with F the fill price (open(m+1) for a market
  change, L for a limit fill; F = open(m+2) and p_new = p_old when nothing
  filled): the change in equity marked at open(m+2), net. Any position is closed at
  market at open(390). The reward is divided by the day's atr20 (fixed,
  for scale); results are reported in ticks.
- **Observation:** the ALL features (standardized with the means and SDs
  of the trained-on days only, not the 20 held out; clipped at ±5) +
  position + open P&L = position × (close(m) − average entry price) /
  atr20 (never the equity marked at an open after close(m)) + minutes left
  + whether the last limit order filled.
- **Optimistic sensitivity (reported, not tested):** the passive and mixed
  arms re-scored with "touch fills" (low(m+1) ≤ L), the same policies.
- **Fill checks on the 47 tick days (reported):** how often close(m) − 1
  equals the first trade's bid of bar m+1; the share of passive fills that
  are touch-only; adverse selection = mean move from the fill to 1 and 5
  min later, passive fills vs market entries; and OF18's bias check for
  market fills (each market fill re-priced at the recorded touch: bias =
  the mean difference, binding for an aggressive or mixed edge). **Binding
  for a passive or mixed edge:** net > 0 on the tick days with each limit
  price L re-set to the recorded first bid (buy) or ask (sell) of bar m+1
  whenever L is better than that touch.

## 5. Learners (hyperparameters fixed now; nothing tuned on test months)

- **PPO** (stable-baselines3 2.9, CPU, `torch.set_num_threads(1)`,
  DummyVecEnv with 8 envs, at most 12 training processes at once):
  MlpPolicy 64×64, n_steps 512, batch 1024, ent_coef 0.01, gamma 0.995,
  other parameters SB3 defaults; seeds 0, 1, 2. **Training and checkpoint
  choice:** the fold's training days minus its last 20 are trained on;
  checkpoints after 5, 10, 20 and 40 passes over those days (a pass = one
  episode per day, days in a seeded random order); the checkpoint with the
  best mean net per day on the 20 held-out days (deterministic policy) is
  the fold's policy (ties: the fewer passes). No refit. The test month is
  scored deterministically. Each arm's result per test day is the mean of
  its 3 seeds' nets.
- **GBM** (secondary, information): LightGBM 4.7 regression, Huber loss
  δ = 8 ticks, label = move at h ∈ {5, 15, 30} min (entry open(m+1), exit
  open(m+1+h), decisions m ≥ 15 with m + 1 + h ≤ 390, OF18), clipped at
  ±60; num_leaves 15, min_data_in_leaf 400, learning_rate 0.03,
  feature_fraction 0.7, bagging 0.7 every 1, lambda_l2 10; early stopping
  on the Huber loss of the last 20 training days (patience 50, max 2000),
  then refit on all training days with that round count; seeds 0-4,
  predictions averaged. Feature sets: TIME; OF; OF+MTF; OF+FVG; OF+VP;
  ALL. A GBM trading rule (|prediction| > 1.28 → trade sign for h, no
  overlap, market cost 1.28 round trip) is reported, not tested; with
  fewer than 100 trades or trades on fewer than 20 test days it reads "does
  not trade".

## 6. Statistics and the decision rule (fixed now)

SE: the largest moving-block bootstrap SE over blocks of **5, 10 and 21**
test days (10,000 resamples, seed 0) of the statistic as written, days
resampled jointly for paired differences; halves: test months
2026-03..07 and 2026-08..11.

**Primary edge tests (M = 3, t_req = 2 + sqrt(2 ln 3) = 3.48):** each RL
arm's net per test day (ticks, mean over seeds; passive and mixed with the
conservative fill rule). An **edge** needs net > 0, t ≥ 3.48, positive in
both halves, positive with costs raised to 0.75 (market) and 0.20 (limit)
ticks per side, and for an arm that uses market orders positive after the
bias check. Not a claim: an edge goes to OF22b, a separate
pre-registration of one confirmation on fresh days recorded from
2026-11-12 with the policy frozen. The fresh-day calendar is shared with
OF21's forward test; OF22b must say how the two divide or share days
before either reads them.

**Style comparison (one test, t ≥ 2), execution held to one policy:** the
aggressive arm's per-bar target path (each seed) is executed (a) at
market and (b) passively: a limit order (the §4 price) whenever the
target changes, re-posted each bar until filled or the target changes
again, market at open(390). Passive − aggressive net per test day
(mean over seeds, paired). |t| ≥ 2 is a statement about execution style
under the conservative fill rule, not an edge. The trained passive arm −
trained aggressive arm and the mixed comparisons are reported. If the
aggressive arm trades less than once a week the comparison reads "not
measurable".

**Information (secondary):** daily out-of-sample rank IC (Spearman of the
GBM prediction and the move over a test day's decisions; a day with
constant predictions has IC 0 and counts). **ALL − OF at h = 15 is the
single primary information test (t ≥ 2).** The family differences OF+MTF
− OF, OF+FVG − OF, OF+VP − OF at h = 15 use t_req = 3.48. Each set −
TIME is reported (so intraday seasonality is not read as information).

**Otherwise:** "trained on real days with these features, the agents net
no more than X ticks per day out of sample (X = max(mean, 0) + 2 × SE)";
smaller edges are not excluded.

**Reported, not tested:** per arm and fold: trades per day, holding times,
long/short split, market vs limit share (mixed), limit fill rate, the
chosen checkpoint, held-out vs test net; results by month; the optimistic
fill sensitivity; the run without the 13 OF16 days; GBM gain importance
by family, GBM IC per set and h, held-out (early-stopping) IC vs test IC
per fold; **random controls** per arm: a policy that changes its target
at random with the trained arm's per-day number of target changes (orders
placed, not fills) and mean holding time, same environment and fills, 200
replicates, seed 0 (its net is the cost and fill-selection floor).

**Red flags:** t > 6; a mean daily IC > 0.10 at h = 15; test IC above
held-out IC in most folds; any lookahead-test failure; an arm trading
less than once a week (that is "no edge", not a pass); a seed whose mean
net is opposite in sign to the other two; a passive arm that wins only
under touch fills (then "depends on queue position, unresolved").

## 7. Known weaknesses (stated before the results)

- **Seen days.** All test months were read by OF17/OF18; the new families
  and styles were not. Walk-forward keeps every model blind to its test
  month, not the researcher to the data.
- **Bars only.** No book, no queue: passive fills are judged by
  trade-through on 1-min bars (conservative; real fills at the touch
  happen too, and the optimistic bound brackets them). A limit order lives
  one minute; real passive traders rest orders longer and cancel faster.
  Order flow is aggressor volume per minute; volume profile is coarse; 1-s
  bundles before September.
- **Regime drift.** OF17 found long-horizon effects flipping sign between
  months; walk-forward measures exactly that.
- **RL sample size.** About 25 training days in the first fold to about
  190 in the last (after the 20 held out), 374 decisions a day: small for
  PPO; a 3-action policy facing costs can learn to stay flat.
- **Power.** OF18 on the same bars bounded 15-min rules at about 2.5
  ticks per trade; per-day RL results will have a comparable bound.
- **Practitioner definitions vary.** FVG/IFVG and profile nodes have
  several definitions; §3 fixes one each. Discretionary selection is not
  tested.

## 8. Cost

Build (features, rolls, FVG/IFVG, profiles, three environments, tests):
~3 hours. GBM: 6 sets × 3 h × 9 folds × 5 seeds, minutes. RL: 3 arms × 3
seeds × 9 folds = 81 runs of 40 passes over 25-190 days (~1.6M steps on
average, ~130M in all); at an assumed ~2,000 steps/s per process
including PPO's updates on 12 processes, ~1.5-2 hours. The speed is
measured on the largest fold (2026-11) before the full run; if it implies more than 4 hours the user is told before
starting, and the pass grid may be halved (5, 10, 20 → a cost-only
amendment logged before any test month is scored). Review and docs: ~1
hour. The browser pane shows one run live (the aggressive arm, seed 0,
of the fold being trained); the others log to files only.
