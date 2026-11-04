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
