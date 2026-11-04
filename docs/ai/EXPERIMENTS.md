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
