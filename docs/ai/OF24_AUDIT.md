# OF24 audit: look-ahead, and luck against skill

Status: **built and run 2026-11-15**, at the user's request ("create a
framework to make sure there is no look-ahead bias, make sure that it is
learning from patterns and proper order-flow analysis and not just sheer
luck"). Tool: `experiments/of24_audit.py`; tests
`python/tests/test_of24_audit.py`. Results of the first run are in
`EXPERIMENTS.md` (OF24) and decision A96.

```
python experiments/of24_audit.py            # report to the console and build-avx512/of24/audit.txt
python experiments/of24_export.py --until 2026-09-11 --out build-avx512/of24/of24_weights_holdout.txt
```

It reads only the study's own files (profit ledgers, decision logs,
weight files) and the OF23 cache (days through 2026-11-11). It never
reads the forward block: forward days are judged from the study's live
ledger alone.

## 1. What can go wrong, and the check for each

| Risk | Check | Passes when |
|---|---|---|
| The study decides with bars it could not have had yet | **A1** every decision in a replay log is a candidate of OF23's pipeline at the same day, minute and direction, with the same setup flags. That pipeline has its own truncation test, and the C++ kernel is only ever handed the bars so far | at least 98% match |
| The live features differ from the tested ones | **A2** the first decision of a run (before any learning step changes the weights) is recomputed from the saved weights and the cached features | probability and value agree to 0.005 |
| Fills at prices that did not exist after the decision | **A3** every entry fill lies in the range of the bar after the decision bar, or one tick outside it (a market order fills at the quote) | at least 98% |
| The policy has already seen the days it is scored on | **A4** share of replay trades on the starting policy's training days; **C** a holdout policy scored on days it never saw | reported, never "passed": only C and D count as evidence |
| The profit is the simulator's generous fills | **B1** the same trades re-priced with OF23's fills (a target needs a trade through it; costs 1.28 / 0.76 ticks) | the gap is reported and used as a haircut on live results |
| Any trader at these setups would have made it | **B2** random candidates on the same days, the same number of trades, the same exits, one at a time; **B3/C1** a random exit at every candidate | the agent is above 95% of the random draws |
| The online learning does nothing, or only fits the replayed days | **B3** each saved weight snapshot trades the cached days with the same random numbers; the change from one snapshot to the next is split into days that pass replayed and days it did not | reported with day-block intervals |
| It does not use order flow | **B4** one group of inputs is shuffled across candidates (setup flags, location, volatility and volume, buy minus sell volume, momentum, time): how many choices move and what the net does. For the live book inputs (depth, MBO, spread), which no cached day has, the size of their weights against a typical input | reported |
| A lucky run is read as skill | every figure has a 95% interval from a bootstrap over whole days (trades within a day are not independent); **D** a rule fixed before the first live session | see section 3 |

## 2. Unseen days (C)

`of24_export.py --until DAY` trains the same starting policy on the days
through DAY only and writes the cut beside the weights. The audit then
runs that policy over the later days, one position at a time, sampled as
the study samples, with OF23's fills (C1); the same after a Sierra
replay of those days with learning on (C2, `weights_replay_c*_holdout.txt`
and a ledger named `*holdout*`); and the fixed exit menu against the
volatility-scaled one (C3, labels from the trade records of those days,
cached in `build-avx512/of24/dyn_labels.npz`).

This is the only honest replay evidence: a replay of days the starting
policy was trained on measures memory, not skill.

**Ledger names.** The study always writes `profit_trades_replay.csv`.
After each replay run the ledger is renamed (`..._replay_<run>.csv`) and
the weights copied (`weights_replay_c1_after_<run>.txt`, or
`..._holdout.txt` for the holdout lineage); a ledger from the holdout
lineage must have `holdout` in its name, and only its trades after the
cut are counted as unseen. A ledger with volatility-scaled exits is not
re-priced (OF23's labels are for the menu's ticks; the scaled exits are
judged in C3 instead).

## 3. The forward rule (D), fixed 2026-11-15 before the first live session

From `profit_trades_live.csv` only:

- fewer than **20 sessions or 200 trades**: "too early";
- otherwise the lower end of the 95% day-block interval of net ticks per
  trade, minus the sim-fill haircut measured in B1: above zero is "skill
  is supported", else "luck is not excluded".

One look at the rule's outcome when the counts are first met; later looks
are reported as such. The rule is about the agent as it runs (learning
on, whatever exits are switched on); a change of exit rule or inputs
restarts the count.

## 4. What the audit cannot show

- **Order-book and MBO skill.** Recorded depth exists for a handful of
  days and MBO is never recorded, so nothing offline can say whether the
  book inputs help. Only the live ledger can, over many sessions; B4
  reports how much weight they carry.
- **Live fills.** Sim fills are immediate and queue-free. B1 measures
  the gap to OF23's stricter fills on replayed days and D subtracts it,
  but neither is a real fill.
- **Regime.** 228 days from one year.

## 5. First run (2026-11-15) and what follows

Figures in `EXPERIMENTS.md` (OF24, "Audit"). In short: no look-ahead;
the replay profit is in-sample; on unseen days the recipe loses like a
random exit; the online learning barely changes the policy.

**What the agent should be good at, and where it stands**

| Skill | Test | Today |
|---|---|---|
| Decide only on the past | A1-A3 | works |
| Read market context from bars | C1 against a random exit | no edge shown |
| Read the order book and MBO | weight of the book inputs; the live ledger | not testable: about five recorded depth days, MBO live only |
| Pick entries (take or skip) | C1, B2 | random on unseen days |
| Respect costs (skip without an edge) | trades a session at a negative expectancy | fails: 17-23 trades a session |
| Manage exits | C3 | scaled exits: no measurable gain |
| Keep risk limits | daily stop, one lot, sim guards | works |
| Adapt online | B3, C2 | under 0.5% of choices moved |
| Generalise | training days against unseen days | +1.6 against -2.3 |

**The three learners used so far**

| | Gradient-boosted trees (OF23 miner) | PPO (the starting policy) | Online actor-critic (the study) |
|---|---|---|---|
| Learns from | the labelled outcome of every exit | replayed training days, the one exit it sampled | each closed sim trade, once |
| Good at | small tabular data, walk-forward | sequences (one position, skip) | adapting live; the only way to learn from MBO |
| Weak at | no sequence | data-hungry, memorises | about 17 noisy lessons a session |
| Unseen days | -1.99 | -2.33 (must trade), -3.22 (may skip) | no measurable change |

All three lose on unseen days, which points at the inputs and the amount
of data rather than at the learner.

**Order of work that follows (not started; each needs its own plan)**

1. Signal first: a walk-forward predictor of the move after a setup,
   on order-book and MBO inputs at short horizons, judged against costs.
2. One learner with three properties: full feedback (every exit's known
   outcome), a distributional output (quantiles per exit), and offline
   caution (trade only when a pessimistic estimate beats costs, else
   skip), with an ensemble over seeds and resampled days whose
   disagreement means "skip".
3. Fewer exits, a simpler model, early stopping on a later block of
   days; nightly retraining checked on held-out days instead of one
   step per trade.
4. Sequence-aware offline RL (exit management, the cost of being busy)
   only on top of a signal that passes step 1.
