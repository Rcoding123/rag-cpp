# RL experiment log

All experiments use `ExecutionEnv` with default `ExecutionParams` unless
noted: 100 lots, 20 steps, impact 0.01, alpha 1, rho 0.7, signal sd 0.1,
noise sd 0.05.

**Metric:** the *share of the optimal gain*. That's the mean per-episode gain
over TWAP, divided by the DP optimum's gain over TWAP, on the same held-out
scenarios (paired; `compare()`). The DP's gain is ≈ 15.2–16.1 depending on
the evaluation set. Each entry records the hypothesis, the setup, the result
and the conclusion; negative results stay in.

---

## E0. Baseline: tile-coded Q-learning (pre-existing)

* **Setup:** `AgentConfig` defaults (8×8×8 tiles, 8 tilings, replay 4 per
  step, alpha 0.1 annealed with floor 0.003), ε decayed over half the
  budget.
* **Results:**

  | Episodes | Share of optimal | Notes |
  |---|---|---|
  | 3,000 | 28% | |
  | 30,000 | 65% | constant alpha 0.03 |
  | 40,000 | 62–71% | seeds 1–6 |
  | 200,000 | 72% | plateau |
  | hand-written sign rule | 43% | for scale |

* **Representation sweep at 30k:**
  * 12×12×12 cells: 40%; 6×10×16: 43%;
  * 16 tilings: 59%;
  * replay 8 or 16 per step: 62%;
  * one cell per time step (20×8×8): 15%.
* **Double Q-learning:** 64.8% vs 64.5% at 30k; 64% vs 72% at 60k.
* **Episodic memory learner:** 14% at 1k episodes (vs 6% for TileQ), but
  only 33% at 10k.
* **Conclusion:** the plateau is the function approximator's bias, not a
  lack of data.

## E0b. Drift: the signal's meaning flips (pre-existing)

* **Setup:** train 40k episodes, then set alpha to −1.
* **Result:** detected 17 episodes after the flip. 40k episodes later the
  agent is at 50% with detection, and stays at −75% without.
* **Detector history:** the first detector false-alarmed in stationary
  worlds. Two-stage confirmation (champion re-score) fixed that; see
  DECISIONS A7/A8.

## E1. Model-based agent: learn the market, then plan

* **Hypothesis:** an agent that knows the *form* of the model, but not its
  numbers, can estimate them by regression and plan with the DP. It should
  reach near-optimal play in about 100 episodes instead of tens of
  thousands. A wrong form should make it worse than model-free learning.
* **Code:**
  * `include/rag/agent/model_based.hpp` (`ModelBasedExecutionAgent`,
    `CostModel`);
  * `ExecutionParams::impact_exponent` and `fixed_cost` for alternative
    markets;
  * `ExecutionDP` accepts learned cost tables.
* **Tests:** `tests/model_based_tests.cpp`.

### E1a. Correct model form (quadratic impact)

* **Estimation:**
  * (impact, alpha) by weighted least squares on step rewards, with weights
    1/max(1, I')²;
  * terminal impact from last steps;
  * (rho, sd) from the AR(1) regression of each signal on the previous one.
* **Parameter recovery:** after 1,000 episodes, impact 0.0100 (true 0.0100),
  terminal 0.0500 (0.0500), alpha 1.00 (1.00), rho 0.70–0.71 (0.70), sd
  0.099–0.101 (0.100).
* **Share of the optimal gain**, seeds 1–3:

  | Episodes | Share |
  |---|---|
  | 10 | 87–100% |
  | 20 | 99.6–100% |
  | ≥ 50 | 100% |

* **Model-free for comparison:** TileQ reaches 67% after 40,000 episodes.
* **Conclusion:** when the assumed form is right, model-based learning is
  roughly 1,000× more sample-efficient here. A handful of numbers to
  estimate, plus exact planning, beats learning a value for every state.

### E1b. Wrong model form

Markets where impact isn't quadratic, or every order pays a fixed cost.
Model-based agents trained 1,000 episodes, TileQ 40,000; shares of each
market's own optimum:

| Market | Quadratic model | Per-size model | TileQ |
|---|---|---|---|
| quadratic (correct form) | 100% | 100% | 66% |
| impact ~ q^1.5 | 99% | 100% | 73% |
| linear impact (exponent 1) | 99% | 100% | 69% |
| convex impact (exponent 3) | 83% | 100% | 38% |
| fixed cost 1.5 per order | 53% | 100% | 61% |
| exponent 1 + fixed 1.5 | 57% | 100% | 87% |
| exponent 3 + fixed 1.5 | 53% | 100% | 63% |

* **Mild misspecification is harmless.** With 5 discrete order sizes, a
  wrong-but-close fit rarely changes the best action.
* **Strong misspecification costs 17–47 points**, and the planner makes it
  worse by exploiting its model's errors: it trades the sizes its model
  underprices.
* **"Per-size"** learns a free cost for each order size traded, in closed
  form with alpha profiled out. It assumes no curve shape and gets 100%
  everywhere.
* **Measurement bug, caught and fixed:** the first run showed −64% for the
  quadratic model on convex impact. The agent's `structure` parameters had
  leaked the true `impact_exponent` and `fixed_cost` into its planner,
  combined with a coefficient fitted under the quadratic assumption.
  `replan()` now forces the assumed form, and the test
  `HiddenMarketParametersDoNotLeakIntoThePlan` feeds two agents identical
  experience with different hidden parameters and requires identical plans.

### E1c. Flexibility needs data; shrinkage makes it safe

Hard market (exponent 3 + fixed 1.5), mean over 5 seeds:

| Episodes | 5 | 10 | 20 | 50 | 100 | 300 |
|---|---|---|---|---|---|---|
| Quadratic | 43% | 47% | 52% | 56% | 55% | 55% |
| Per-size | −1601% | −641% | −198% | 100% | 100% | 100% |
| Shrunk (k = 30) | 78% | 87% | 93% | 97% | 98% | 99% |

In the correct quadratic market, Shrunk matches the quadratic model
(85 / 94 / 97 / 100%); per-size there is −535 / −168 / −22 / 100%.

* **Per-size is bimodal with little data.** At 10 episodes, seeds 1–10
  scored 97, −1701, 97, −1809, 97, 98, 98, −1782, 99 and −1440%. The planner
  exploits wild cost estimates for rarely traded sizes.
* **Shrinkage** (`CostModel::Shrunk`) blends each size's estimate with the
  quadratic prediction, weighted by its trade count; unseen sizes fall back
  to the prior. It was never catastrophic: 82–89% at 10 episodes on every
  seed, 91–95% at 20.
* **Conclusion:** the right amount of model structure depends on the data.
  Strong assumptions learn fast and plateau when wrong; flexible models are
  eventually right but dangerous early. Empirical-Bayes shrinkage gets most
  of both. The general lesson for planning agents: a planner exploits
  whatever its model gets wrong, so uncertain parts of the model need to be
  pessimistic or pulled toward a prior.

## E2. What limits model-free learning? (the ~72% plateau)

* **Hypothesis:** the plateau comes from bootstrapping. Multi-step or Monte
  Carlo targets should break it.
* **Code:** `AgentConfig::n_step` (n-step returns through the replay buffer,
  no off-policy correction) and `TileQ::blend`.
* **Tests:**
  * `AgentNStep.*`: the corridor is learned exactly with n = 1, 3 and 40,
    and the 3-step target is checked by hand;
  * `AgentRepresentation.TileCodingCanRepresentANearOptimalPolicy`.

### E2a. n-step sweep (share of the optimal gain, seeds 1–3)

| n | 10k episodes | 40k episodes |
|---|---|---|
| 1 | 36 / 41 / 45 | 66 / 71 / 62 |
| 2 | 49 / 56 / 45 | 47 / 66 / 69 |
| **3** | **50 / 48 / 47** | **70 / 69 / 69** |
| 5 | 38 / 42 / 43 | 67 / 66 / 65 |
| 10 | 46 / 47 / 45 | 62 / 61 / 65 |
| 20 (Monte Carlo) | 46 / 42 / 42 | 65 / 66 / 65 |

At 200k episodes, n = 3 reaches 75.8 / 76.8 / 74.9% (n = 1: 72%). **n = 3
is now the default.** The hypothesis is rejected, though: Monte Carlo
targets (no bootstrapping, no max-bias) plateau too.

### E2b. The representation ceiling: fit the true Q* directly

Fitting tile-coded Q to the DP's exact Q*(s, a) by plain regression, on
states from an exploring policy (80% optimal, 20% random), all 5 actions per
state, 100k episodes:

| Tiles | Weights per action | Greedy share |
|---|---|---|
| 8×8×8 × 8 tilings (default) | 5,832 | **99.5%** |
| 8×10×12 × 8 | 10,296 | 99.5% |
| 12×12×12 × 8 | 17,576 | 99.7% |
| 16×16×16 × 16 | 78,608 | 99.9% |

**The representation is not the bottleneck.** The earlier finding that finer
tiles do worse in RL is about learning (fewer samples per weight), not
capacity.

### E2c. Ablation: noise vs coverage (40k episodes)

| Targets | Actions updated per visited state | Share |
|---|---|---|
| exact Q* | all 5 | 99.4% |
| exact Q* | only the one taken | **−92.7%** |
| one sampled return (then optimal) | all 5 | **65.7%** |
| one sampled return | only the one taken | −51.9% |

* **Noise alone reproduces the plateau.** Sampled returns have sd ≈ 20–50,
  against action-value gaps that are often below 1.
* **Uneven coverage is catastrophic on its own.** Actions rarely taken in a
  region keep their initial value 0, which beats the true (negative) values,
  so the greedy policy picks the least-known action. In RL, ε-exploration
  partly repairs this.

### E2d. Fixes tried for the noise

| Idea | Result |
|---|---|
| Afterstates, Q = C[q] + W(t, I − q, g), sharing W across actions | 43–51% at 10k (better than n = 1), 61–67% at 40k (same plateau) |
| Polyak tail-averaging of weights (second half of training) | worse at 40k (n = 1: 31–48% vs 46–66%; n = 3: 64–71% vs 67–73%), because the iterates are still improving. At 200k it protected against one late dip (64% → 78%); the champion mechanism already covers that. |
| Lower step-size floor (0.001, 0.0003) | no effect up to 200k: the 1/(1 + ep/3000) schedule hadn't reached the floor |

### Conclusions

* **The gap is variance, not representation.** The model-free learner sees
  each state-action value through one noisy sample at a time.
* **Model-based learning (E1) closes the gap** because it estimates 5
  numbers from all transitions jointly and plans exactly: 100% in 20–50
  episodes.
* **For model-free learning, the remaining levers** are variance reduction:
  control variates, or counterfactual updates using the fact that the price
  move doesn't depend on the action.

## E3. Policy bank: recognizing a regime seen before

* **Hypothesis:** markets alternate between regimes. Keeping past champions
  and re-scoring them on a confirmed drift should let the agent redeploy a
  matching policy at once, instead of relearning.
* **Code:** `TrainConfig::policy_bank` (default on), `bank_capacity`,
  `recall_epsilon`, `recall_episodes`; `Trainer::recall_from_bank()`.
* **Mechanism:** on a confirmed drift,
  1. the outgoing champion is banked;
  2. every older banked policy is scored on the held-out scenarios;
  3. if one beats the re-scored champion by 2 standard errors, it becomes
     the champion, the learner warm-starts from it, and exploration is light
     (ε 0.05 for 500 episodes);
  4. otherwise, the agent re-explores normally (ε 0.5 for 5,000 episodes).
* **Tests:** `AgentPolicyBank.*`.
* **Setup:** 20k episodes in regime A (alpha +1), then the regime alternates
  B (alpha −1) / A / B / A every 10k episodes. Each cell is the deployed
  policy's share of the current regime's optimal gain:

  | Switch to | Bank on: +500 | +2,000 | +10,000 | Bank off: +500 | +2,000 | +10,000 |
  |---|---|---|---|---|---|---|
  | B (first) | −126% | −6% | 25% | −126% | −6% | 25% |
  | A | **59%** (recalled at +131) | 59% | 59% | −82% | 20% | 50% |
  | B | **25%** (recalled at +20) | 26% | 36% | −109% | 6% | 30% |
  | A | **59%** (recalled at +20) | 59% | 59% | −86% | 21% | 41% |

* **Conclusion:** on its first visit to a regime the bank changes nothing.
  On every return it's recognized within 20–131 episodes, and the deployed
  policy is back at its previous level instead of losing money for
  thousands of episodes. This only works because drift *confirmation*
  re-scores policies on fixed held-out scenarios; the same evaluation
  machinery answers "which known world is this?".
