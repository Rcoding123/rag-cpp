# Session report, 2026-11-04 to 2026-11-05: price discovery (OF10) and the RL reruns (OF11)

## In one paragraph

The goal was plan item 1 (ORDERFLOW.md §7): make the ES simulator discover
prices in seconds instead of half a minute, then re-run the RL experiments
on the fixed market. It took three iterations. Along the way we found a
second, older bug: the simulator's trading activity got stuck at its floor,
so every earlier RL episode ran at about 4 trades/s instead of ES's 15–22.
Both are now fixed, opt-in, in a new `discovery` market (v3). On that
market **the RL agents find no edge**. All 24 trained seeds lose money on
sealed test episodes, the ensemble learns not to trade, and every stress
scenario and simulated trading day is at or below zero. The earlier RL
profits (OF3, OF8) came from the simulator's slow price discovery, not from
skill that would carry over to a real market.

## What was done (commits on `research/rl`)

| Commit | What |
|---|---|
| `10bfe03` | OF10 v1: makers jump their quotes when fair value moves (rejected: the book stopped carrying information) |
| `71ca8d6` | CI is Windows-only, as you decided; the local server gate decides when `main` moves |
| `8b2cfdc` | `of_trace` + `tools/of_viz.py`: a step-by-step viewer for RL episodes |
| `4ba0354` | OF10 v2: price discovery through the book (stale side drains, then the price ticks) |
| `d7a4ca9` | Docs for iterations 1–2 and the warm-up finding |
| `e229af4` | OF10 v3: a stationary activity model (the warm-up fix), the v3 `discovery` preset, a probe that checks every statistic in the regime the agent sees |
| `5bb7be3` | `--warmup` for the RL programs, a record next to each saved network, and an OF6 stress list fixed before any v3 network existed |
| `1a73bd6` | `experiments/of11_regen.sh`: the RL reruns, reproducible in one command |
| this commit | OF11 results, docs, this report, regenerated `handoff/` |

Every result file is regenerated from committed code by a script in
`experiments/` (`of10_regen.sh`, `of11_regen.sh`). Builds are warning-free
on MSVC; the full gate (build + 344 tests) passed before each commit.

## What happened, in order

1. **Iteration 1 (quote jumps).** The hidden-value gap stopped predicting
   the next 30 s, but book imbalance became useless. That isn't ES-like,
   so it was rejected.
2. **Iteration 2 (stale-side depletion).** Gap and imbalance were right,
   but the stylized facts and passive quoting failed, and the oracle bound
   was empty. A skeptic agent then found the **warm-up bug**: a fresh
   simulator starts its activity average at a wrong value, and the
   feedback drives activity to the floor for about 1.5 hours. Every
   OF2–OF9 episode ran in that state.
3. **Iteration 3.** The legacy activity loop turned out to be unstable:
   it has no steady state to start from, so starting it correctly is not
   enough. The fix is a new, opt-in activity model with a known stationary
   distribution, drawn at the start. v3 passes:
   * (a) the gap no longer predicts 30 s (+0.018 above the grid floor);
   * (b) imbalance predicts the next tick (0.21) and nothing after that;
   * (d) ES-like facts (18.5 trades/s, 1-min sd 4 ticks, kurtosis 5.6–7);
   * (e) no naive rule makes money after costs;
   * (f1) two pre-registered oracles that actually trade and win.

   It passed on tuning seeds, on held-out seeds (used once) and on the
   skeptic's fresh seeds.

   Still failing: a passive quote-both strategy loses 24–27 ticks per
   episode, mostly to "stale sweeps". That is a known limitation for the
   tick-scalping environments only (A64). The meta-order population also
   needed a longer warm-up: 1,050 s instead of 150 s (A63).
4. **Pre-RL fixes.** The `--warmup` option was added, with a record next
   to each saved network that refuses mismatched use. The stress list for
   discovery was written down before training. With the 1,050 s warm-up,
   episodes and long windows agree on 29/29 statistics.
5. **OF11, the RL reruns.** The OF3 recipe (NeuralQ, Double DQN, 24 seeds ×
   2,000 episodes) ran on v3 with the 1,050 s warm-up:
   * **Sealed test:** every seed is negative (−0.15 to −1.50
     ticks/episode, mean −0.54). The ensemble makes no trades. The 30 s
     delta rule makes −1.39.
   * **OF6 stresses:** every strategy is ≤ 0 in all 11 scenarios. The
     networks lose less than the rule.
   * **OF8, 20 simulated days:** the ensemble makes −3.3 ticks/day (Sharpe
     −3.1) and the rule −88.
   * **OF2 rules:** none is above 0 by 2 SE.
   * **Red flags:** none. There is no edge, so the fresh confirmation
     seeds (950M+) were not needed and stay unused.

## Process notes (the kit, the platform)

* **Windows only:** the Linux CI jobs were removed, as you decided. They
  can be restored from `e0a4c7a`.
* **Python 3.12** was installed with your approval. It isn't on PATH in
  older sessions; use the full path,
  `C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`.
* **The kit blocked work twice early on**, so two changes were made:
  * Experiments now run only from binary snapshots. A gate rebuild had
    collided with a running experiment's `.exe`.
  * A **hold** (`kit.ps1 hold` / `release`) defers the Stop-hook gate
    while a background workflow owns the checkout. It is said once, can
    last at most 12 h, and is never skipped: `release` runs the full
    gate.

  After these two changes the kit did not get in the way again.
* **Agents:** each work item ran as a workflow:
  1. two independent designs and a judge;
  2. an implementer;
  3. two adversarial reviewers (fake-edge and reproducibility);
  4. a fixer;
  5. an acceptance run;
  6. a skeptic.

  The skeptics caught the warm-up bug and the meta-order shortfall. The
  reviewers caught a relabel that could have hidden a real red flag.

## Where we are

* `discovery` (v3) is the best market so far for market-order RL: fast
  price discovery, realistic activity, ES-like facts, no naive edge.
* **Known remaining flaws:**
  * Passive quoting is badly adverse-selected (stale sweeps), so the tick
    environments (OF4, OF5, OF7) are not calibrated.
  * Own impact is linear and fully permanent, so multi-lot scalping is
    blocked.
  * Fair value has a small momentum (+0.013 at 10 s).
  * Two imbalance rows sit just outside the probe's own lines:
    corr(I1, r 30 s) is 0.076 against 0.05, and I5 is 0.080.
* **The old OF2–OF9 results** were measured at the activity floor and on
  a slow-discovery market. Read them as results about that simulator only.

## Next (ORDERFLOW.md §7)

1. **Calibrate to your ES ticks.** Export them to the CSV format and run
   `of_calibrate facts`, then `fit`. This is the step that can tell us
   whether v3 or real ES has the edge.
2. **Backtest on real ticks** with `of_backtest`, walk-forward by day.
3. Redesign the stale-sweep mechanism (or the L1 queue size) before any
   tick-scalping rerun. Make own impact concave or transient.
4. Paired evaluation on the same market path (`separate_streams`).
