# RL experiment log

All experiments use `ExecutionEnv` with default `ExecutionParams` unless
noted: 100 lots, 20 steps, impact 0.01, alpha 1, rho 0.7, signal sd 0.1,
noise sd 0.05.

**Metric:** the *share of the optimal gain*. That's the mean per-episode gain
over TWAP, divided by the DP optimum's gain over TWAP, on the same held-out
scenarios (paired; `compare()`). The DP's gain is ≈ 15.2–16.1 depending on
the evaluation set. Each entry records the hypothesis, the setup, the result
and the conclusion; negative results stay in.

## Summary so far

| # | Question | Answer in one line |
|---|---|---|
| E0 | How good is tile-coded Q-learning? | ~65% at 30k episodes, ~72% at 60k. |
| E1 | Learn a model and plan instead? | 99–100% after 20–50 episodes; robust to a misspecified form if per-size costs are shrunk (empirical Bayes). |
| E2 | Why does model-free plateau? | Target noise and coverage, not representation (which holds 99.5%). 3-step returns help most. |
| E3 | Recurring regimes? | A policy bank recalls a known regime within 20–131 episodes. |
| E4 | Risk aversion? | DP shows the Almgren–Chriss trade-off; the model-based agent learns the penalty to ~2%. |
| E5 | Policy gradient? | Actor-critic is the best model-free learner (86–89% at 200k), but needs a slow annealing schedule. |
| E6 | Robustness across markets? | Model-based 100% everywhere; TileQ can deploy policies *worse than TWAP* in weak markets. |
| E6b | Can deployment be made safe? | A baseline guard: never deploy a policy that hasn't beaten TWAP by 2 standard errors. Worst case −106% → 0%. |
| E7 | What if the market is outside the model class? | Model-based drops to 22–55% (model-free: 70–91%) and gets *worse* with data. A paired model check flags it after 100 episodes (z > 8 vs < 2.5). Fixes: expand the model (100%), or distill + fine-tune + guard (never below the plan, 78–82%). |
| E8 | What does learning cost online, with every episode real? | ε from 1 loses >1,000 optimal-episodes and breaks even at ~12k; ε 0.1 breaks even at ~2k and learns as well. Model-based earns 88–95% of the optimum while learning (95–100% with E11's defaults). |
| E9 | How fast can the model-based agent adapt? | Without forgetting: −166% 20 episodes after a flip. Constant forgetting tracks but costs up to 46 points when stable. Change detection + reset: 95–100% within 20 episodes, free when stable. |
| E10 | Can a policy be scored from logs before deployment? | Not by importance sampling at horizon 20 (errors 45–2,968%, ESS ≈ 0). A model can (2–4% when its class is right), but its bias under a wrong model doesn't shrink with data; doubly robust repairs it only near the logged policy. |
| E11 | Can system identification be cheaper? | How the agent explores at the start barely matters; ongoing ε 0.05 cost 6–10% of the online gain, so the default is now 0.01. Also fixed: the first plan came after 20 episodes, not 10. |
| E12 | Does the same agent work on another task? | Market making, unchanged agent: actor-critic 99–100% after 5,000 episodes, TileQ 97–99% at 50k; both discover the inventory skew. The task is 15× less noisy per unit of gain than execution (4.07 vs 0.27), which explains the execution plateau. |
| E13 | Does drift handling carry over to the second task? | Not at first: a market that got *better* went unnoticed (CUSUM watched only for drops, and a rejected alarm had blinded it: bug). Fixed with evaluation-time detection from the champion's exactly reproducible re-score, `rebase()`, and an optional two-sided test; the bank then recalls returning regimes at 99–100%. |
| E14 | What about the bad days? | Volatility isn't tail risk: the risk-neutral execution optimum is more volatile than TWAP but its worst 5% is −66 vs −117. A variance penalty halves execution's sd but barely moves its tail; in market making it lifts the worst 5% from −5.7 to +13.7. |

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

## E4. Risk-averse execution (mean-variance)

* **Hypothesis:** with a risk penalty, the optimal policy sells earlier;
  learners should discover how much earlier. The model-based agent needs a
  holding-cost term to stay optimal.
* **Code:**
  * `ExecutionParams::risk_aversion` λ: each step's reward is reduced by
    λ·noise_sd²·I'², the variance the held inventory adds. The penalty is in
    the reward only; cash accounting is unchanged. The DP includes it.
  * The model-based agent now solves one weighted normal-equations system
    (Gaussian elimination) over alpha, a holding cost κ (regressor −I'²), and
    either the quadratic impact or one cost per order size.
* **Tests:** `ModelBased.LearnsTheRiskPenalty`,
  `AgentRisk.FrontLoadsMoreWhenRiskAverse`.

| λ | DP optimum: mean cash / sd / lots sold in steps 0–4 | TileQ 40k: share, sd | Model-based 300 eps: share, fitted κ (true) |
|---|---|---|---|
| 0 | +9.3 / 58.7 / 26.5 | 70%, 52.1 | 100%, 0.00000 (0) |
| 0.01 | +9.3 / 57.7 / 28.6 | 66%, 50.7 | 100%, 0.00002 (0.000025) |
| 0.05 | +8.6 / 54.0 / 36.5 | 73%, 46.3 | 100%, 0.00011 (0.000125) |
| 0.2 | +2.8 / 42.9 / 55.0 | 91%, 39.2 | 100%, 0.00049 (0.0005) |
| 1.0 | −8.4 / 28.2 / 72.9 | 100%, 27.7 | 100%, 0.00249 (0.0025) |

"Share" is the share of the risk-adjusted optimal gain over TWAP.

* **The DP traces the classic Almgren–Chriss trade-off:** more risk aversion
  means selling earlier (26 → 73 lots in the first 5 steps), lower variance
  (sd 59 → 28) and a lower mean (+9.3 → −8.4).
* **Both learners front-load as λ grows.**
* **Strong risk aversion makes model-free learning easy.** The best policy
  becomes "sell fast", which leaves less to gain from the noisy signal:
  TileQ goes from 70% to 100%.
* **The model-based agent recovers κ to within ~2%** at λ ≥ 0.2, and stays at
  100%.

## E5. Policy gradient: actor-critic

* **Hypothesis:** learning a policy directly, with a critic's TD error as the
  advantage, copes better with target noise than learning Q-values, because
  the actor only needs relative preferences.
* **Code:**
  * `LearnerKind::ActorCritic`: a softmax policy over tile-coded logits plus
    a tile-coded critic V(s), with 1-step TD error as the advantage and
    `actor_alpha` / `critic_alpha` / `temperature` / `ac_decay_episodes`.
  * On-policy: exploration is sampling from the policy (ε applies only
    during re-exploration); no replay. `value()` is the critic, so drift
    detection and champions work unchanged.
* **Tests:** `AgentActorCritic.*`.

Standalone probe (seeds 1–5 at 10k/40k, 1–3 at 200k):

| Actor / critic step, temperature | 10k | 40k | 200k |
|---|---|---|---|
| 0.03 / 0.1, T = 1 | 55–76% (median 75) | 79–84% | **86–89%** |
| 0.01 / 0.1, T = 1 | −884% to −374% (policy still near-random) | 59–63% | |
| 0.003 / 0.1 | about −600% | about −600% | |
| 0.05–0.1 / 0.1 | 69–72% | 74–84% | |
| 0.03 / 0.03 (slow critic) | −427% to −310% | 73–77% | |

The step-size schedule matters (library Trainer, seeds 1–3):

| Annealing half-life | 10k | 40k |
|---|---|---|
| 3,000 episodes (the TileQ default) | −235%, 7%, −2% | −24%, 84%, 78% |
| 5,000 | 57–77% | 79–83% |
| **10,000 (default `ac_decay_episodes`)** | **75–78%** | **80–86%** |
| 20,000 | 75–79% | 82–84% |

* **Actor-critic is the best model-free learner here.** At 200k episodes it
  reaches 86–89%, against 75–79% for 3-step Q-learning, and it's far ahead
  early (75% vs 47–50% at 10k).
* **It's fragile in a different way.** Too small an actor step, or annealing
  too early, leaves the policy near-uniform, which in this task loses
  heavily. A learner whose failure mode is "stays random" needs its
  schedule chosen with care.

## E6. Robustness: how each learner fares when the market changes shape

* **Question:** are the defaults (tuned on one market) safe on others?
* **Setup:** one parameter varied at a time, others at their defaults.
  TileQ and ActorCritic train for 40k episodes (ε half-life 20k); the
  model-based agent (MB) sees 300 episodes. Cells are the deployed policy's
  share of that market's optimal gain over TWAP, on 1,500 shared held-out
  scenarios (probe `sweep.cpp`, seed 1).

| Market | DP gain over TWAP | TileQ | ActorCritic | Model-based |
|---|---|---|---|---|
| default (alpha 1, noise 0.05, rho 0.7, horizon 20) | +16.3 | 70% | 82% | 100% |
| alpha 0.25 (weak signal) | +1.4 | **−35%** | 34% | 100% |
| alpha 0.5 | | 43% | 76% | 100% |
| alpha 2 | | 72% | 79% | 100% |
| noise 0.025 | | 69% | 72% | 100% |
| noise 0.1 | | 62% | 81% | 100% |
| noise 0.2 | | 57% | 79% | 100% |
| rho 0 (no signal persistence) | +3.6 | 32% | 71% | 100% |
| rho 0.9 | | 68% | 78% | 100% |
| horizon 10 | | 68% | 85% | 100% |
| horizon 40 | | 61% | 76% | 100% |

* **The model-based agent is at 100% everywhere,** because every market
  here lies inside its model class.
* **ActorCritic is the more robust model-free learner** (34–85%); TileQ
  ranges from −35% to 72%.
* **The dangerous cell is TileQ in a weak-signal market.** When the gain
  available is small (+1.4 per episode against an episode sd of ~55), the
  learner fits noise and deploys a policy *worse than the trivial
  baseline*. A production system must never do that, which motivates E6b.

## E6b. A baseline guard: never deploy below TWAP

* **Code:** `Trainer::set_baseline(policy)`. With a baseline set,
  `deployed_act()` uses the baseline until the champion beats it on the
  held-out scenarios by more than 2 standard errors of the paired
  difference. The check reruns at every champion evaluation and after every
  confirmed drift, so a champion that stops working falls back to the
  baseline. `learned_policy_deployed()` reports which one is live;
  `EvalPoint::learned_deployed` logs it.
* **Test:** `AgentSafety.NeverDeploysAPolicyWorseThanTheBaseline`.
* **Measured** (TileQ, 20k episodes, seeds 1–5; deployed share of the
  optimal gain; probe `guard_probe.cpp`):

| Market | Unguarded | Guarded | Learned policy deployed |
|---|---|---|---|
| default | 52 / 61 / 59 / 60 / 58% | identical | 5 of 5 |
| alpha 0.25 | −58 / −66 / −75 / −35 / −106% | **0% (TWAP) on every seed** | 0 of 5 |
| rho 0 | −8 / 16 / 4 / 9 / 25% | 0 / 0 / 0 / 0 / 25% | 1 of 5 |

* **The guard removes the downside** (worst case −106% → 0%) and costs
  nothing where learning works.
* **It's conservative:** in the rho-0 market it held back three small gains
  (4–16%) that weren't statistically distinguishable from TWAP on 600
  held-out scenarios. More evaluation scenarios would release them sooner.

## E7. Outside the model class: permanent impact

* **Question:** E6 found the model-based agent at 100% everywhere, but every
  market there was inside its model class. What happens when the market has
  structure the model lacks, and can the agent tell?
* **Code:**
  * `ExecutionParams::permanent_impact` γ: each lot sold lowers the price for
    the rest of the episode, so the inventory still held loses γ·q·I'. The DP
    includes it.
  * `ExecutionDP::policy_value(policy)`: exact expected value of any policy
    under a model, by backward induction with the policy's action.
  * `ModelBasedExecutionAgent::Config::learn_permanent_impact` (regressor
    −q·I'), `check_model()`, `plan()`.
  * `Agent::distill(sim, teacher_q, episodes, seed)`: a supervised warm start
    of the tiles from a teacher's Q-values.
  * `evaluate()`/`compare()` moved to `evaluate.hpp`, so the model-based
    agent can use them without the trainer.
* **Tests:**
  * `ExecutionCosts.PermanentImpactKeepsTheAccountingIdentityAndTheDpIsExact`;
  * `ModelBased.LearnsPermanentImpactAndAWrongModelGetsWorseWithData`;
  * `ModelBased.CheckModelFlagsAMissingTerm`;
  * `AgentWarmStart.DistilledModelPlusGuardIsBestOfBoth`.

### E7a. A missing term reverses the ranking

Model-based agent (Shrunk, 300 episodes) without the permanent term vs the
model-free learners (40k episodes), seed 1 (probe `e7_probe.cpp`):

| γ | DP gain over TWAP | Model-based, no term | TileQ | ActorCritic |
|---|---|---|---|---|
| 0 | +16.3 | 100% | 70% | 82% |
| 0.005 | +17.9 | 47–55% (6 seeds) | 70% | 85% |
| 0.01 | +19.7 | 22–27% (6 seeds) | 80% | 82% |
| 0.02 | +23.9 | 22% | 83% | 91% |

* **This is the bias–variance trade-off of model-based RL, measured.** The
  model-free learners pay ~20–30 points of variance everywhere but no bias;
  the model-based agent has no variance to speak of, but its bias is
  unbounded once its model is wrong.
* **The omitted term goes into the wrong coefficients:** a fitted holding
  cost (0.0007–0.002 for a true 0) and an inflated alpha (1.18–1.28 for a
  true 1.0).
* **More data makes the wrong model worse** (5 seeds, γ = 0.01): 43% at 20
  episodes, 37% at 50, 25% at 300. The plan exploits the model's error, so
  the data it collects grows increasingly one-sided: a closed-loop
  identification bias.
* **A near-miss bug of the A11 kind:** the first version of this probe let
  the true γ leak from `structure` into the planner (replan() didn't
  override the new field). The leak regression test now scrambles every
  market field the planner reads.

### E7b. Expanding the model class fixes it, at almost no cost

With `learn_permanent_impact` (5 seeds; probe `e7_cost.cpp`):

| Market | Episodes | Without the term | With it | Fitted γ |
|---|---|---|---|---|
| γ = 0.01 | 20 | 43% | 93% | 0.0101 |
| γ = 0.01 | 50 | 37% | 99.8% | 0.0102 |
| γ = 0.01 | 300 | 25% | 99.9% | 0.0100 |
| default | 20 / 50 / 300 | 96.0 / 100.0 / 100.1% | 96.4 / 100.0 / 100.0% | ~0.0001 |
| λ = 0.2 | 20 / 50 / 300 | 99.0 / 99.7 / 100.0% | 99.0 / 99.4 / 99.9% | ~0.0001 |

The extra parameter costs at most 0.3 points where it isn't needed, and is
on by default. It does blur the holding-cost estimate on one test seed
(0.00057 for a true 0.0005), because front-loading plans make q·I' and I'²
correlated.

### E7c. The agent can tell when its model is wrong

`check_model(market, n, seed)` compares the gain over TWAP that the model
predicts for its own plan (exact, `policy_value`) with the gain measured on
n real scenarios, paired with TWAP. z = (predicted − realized) / standard
error, 8 training seeds:

| | n = 100 | n = 400 |
|---|---|---|
| Model right (γ = 0 or 0.005, term learned) | z ∈ [−2.4, +1.2] | z ∈ [−2.5, +1.6] |
| Term missing, γ = 0.005 | **z ∈ [+8.5, +12.9]** | z ∈ [+20.8, +25.5] |
| Term missing, γ = 0.01 | **z ∈ [+20.5, +27.4]** | z ∈ [+45.1, +55.2] |

* **A one-sided threshold z > 4 separates every case:** at n = 100 that's
  24 correct models (also γ = 0 without the term: z ≤ 1.1) and 16 wrong ones.
* **The wrong model is always optimistic.** The planner chooses the actions
  its errors make look best (the optimizer's curse), so "predicted >
  realized" is the signature to test for.
* **Pairing matters:** the unpaired version (predicted vs realized episode
  value) needed ~1,500 episodes to reach z ≈ 5, because episode returns
  have sd ≈ 55 against paired differences of ≈ 10.

### E7d. Unknown unknowns: warm start, fine-tune, guard

When the missing term isn't known, the model can still be useful as a
starting point. The recipe: distill the model's Q-values into TileQ (in the
model's own simulator, so no real episodes are spent), fine-tune on real
rewards with ε 0.1, and guard the deployment with the model's plan as the
baseline. Deployed share, seeds 1–3 (probe `e7_hybrid.cpp`):

| | Start | 2,500 | 5,000 | 10,000 | 20,000 | 40,000 |
|---|---|---|---|---|---|---|
| **γ = 0 (model right, plan 100%)** | | | | | | |
| TileQ from scratch | −3063% | 1–9% | 9–24% | 34–43% | 52–61% | 68–69% |
| Warm start, fine-tuned | 99% | 35–41% | 37–45% | 50–52% | 53–64% | 69–74% |
| Warm start + guard | 99% | 100% | 100% | 100% | 100% | 100% |
| **γ = 0.01 (model wrong, plan 25–28%)** | | | | | | |
| TileQ from scratch | −2286% | −69% to −167% | −24% to −100% | 16–45% | 68–73% | 76–79% |
| Warm start + guard | 25–27% | 27–38% | 27–49% | 48–65% | 69–74% | 78–82% |

* **RL fine-tuning destroys a correct model's policy.** 99% → 35–41% within
  2,500 episodes, and smaller step sizes don't save it (α 0.03: 51%; α
  0.01: 57%). A TD estimate's steady-state jitter is about √(α/2)·σ ≈ 2
  return units (σ ≈ 30), while the action gaps are below 1, so no practical
  step size holds a near-optimal policy against noisy targets. This is E2's
  finding seen from the other side.
* **Warm start + guard is best of both:** 100% when the model is right. When
  it's wrong, the deployed policy is never below the model's plan and ends
  where the model-free learner does (78–82%). Meanwhile TileQ from scratch
  deploys policies far worse than TWAP for its first ~5,000 episodes.
* **The production recipe:** learn a model; check it (`check_model`); if the
  check passes, deploy its plan; if it fails, either expand the model class
  (E7b) or fine-tune a distilled copy under the guard (E7d).

## E8. Online regret: what learning costs while it happens

* **Question:** every experiment so far scored the *deployed* policy, as if
  training were free (a simulator). If every training episode is real, what
  does learning cost compared with just running TWAP?
* **Metric:** cumulative gain over TWAP across all training episodes,
  exploration included, divided by the optimal gain per episode (exact:
  `ExecutionDP::expected_start_value()` − `policy_value(twap)`). The
  optimum would score +N after N episodes. *Break-even* is the episode
  after which the cumulative gain stays positive.
* **Test:** `AgentOnline.LowExplorationCostsLessOnlineAndLearnsAsWell`.
* **Results** (default market, 20,000 real episodes, seeds 1–3; probe
  `e8_regret.cpp`):

| Learner | after 1,000 | 5,000 | 10,000 | 20,000 | Break-even | Final greedy share |
|---|---|---|---|---|---|---|
| TileQ, ε 1 → 0.02 over 10k (as in E0–E7) | −283 to −474 | −1124 to −1504 | −585 to −1027 | +4002 to +4393 | 11.6k–12.8k | 58–61% |
| TileQ, ε 0.3 → 0.02 | −243 to −320 | −73 to +571 | +1353 to +2123 | +6659 to +7379 | 3.1k–5.7k | 56–62% |
| TileQ, ε 0.1 → 0.02 | −190 to +14 | +865 to +1078 | +2497 to +2971 | +7395 to +7652 | 1.5k–2.1k | 60–61% |
| TileQ, ε 0.02 constant | −6 to +14 | +1048 to +1561 | +3085 to +4192 | +7765 to +9918 | 1.0k–1.4k | 56–65% |
| TileQ, greedy (ε 0) | −48 to +52 | +811 to +1954 | +2480 to +4933 | +6968 to +11938 | 0.5k–1.3k | 52–73% |
| ActorCritic | −336 to −408 | −2514 to −4506 | −2629 to +119 | +3769 to +6585 | 9.8k–14.4k | 79–82% |
| Model-based (Shrunk) | +765 to +913 | +4292 to +4826 | +8517 to +9296 | **+17699 to +18902** | 1–42 | 100% |

With permanent impact γ = 0.01 the picture is the same; the model-based
agent *without* the permanent term earns +2326 to +3025 (12–15% of the
optimum online, 19% final), and TileQ with ε 0.1 earns +9551 to +9866.

Final quality at longer horizons (deployed share, seeds 1–3; probe
`e8_long.cpp`):

| Exploration | 10k | 40k | 100k |
|---|---|---|---|
| ε 1 → 0.02 | 32–43% | 65–70% | 72–80% |
| ε 0.1 → 0.02 | 44–52% | 65–70% | 75–77% |
| greedy | 44–67% | 55–81% | 61–85% |

* **Starting ε at 1 is a simulator habit.** Online it costs over a thousand
  optimal-episodes of losses and doesn't break even until ~12,000
  episodes. Starting at 0.1 breaks even 6–8× sooner, earns ~1.8× more over
  20,000 episodes, and learns as well at every horizon.
* **Why so little exploration suffices here:** Q-values start at 0, which
  is optimistic for the bad actions (selling nothing ends in a dump that
  costs up to 500), so greedy selection tries them and learns they're bad.
  Reward noise does the rest.
* **Greedy is a gamble:** cheapest online on average, but its final quality
  depends on luck (61% vs 85% across seeds). ε 0.1 keeps the cost low and
  the outcome predictable.
* **Actor-critic is the costliest online learner** (break-even at
  10k–14k), even though it ends with the best model-free policy. Its
  initial softmax is uniform, i.e. random selling with large dumps.
* **The model-based agent earns 88–95% of the optimal gain while
  learning.** Its 10 random exploration episodes are its only real cost.
  *(Corrected by E11: because of a bug, that phase lasted 20 episodes, and
  the larger cost was the ongoing ε 0.05. With E11's defaults it earns
  95–100%.)*
* **The library default stays ε 1 → 0.02** (A24): sparse-reward tasks
  such as the corridor have no optimistic values to explore with. For
  online learning on cost-like rewards, start at 0.1.

## E9. A changing market: forgetting vs change detection (model-based)

* **Question:** the model-free agent needed 5,000 episodes to get back to
  18% after the signal's meaning flipped (E0b). The model-based agent has
  only ~6 numbers to re-estimate. How fast can it adapt, and at what cost
  when nothing changes?
* **Code:**
  * `ModelBasedExecutionAgent::Config::forget`: per-episode decay of every
    sufficient statistic (exponential forgetting).
  * `detect_drift`, `drift` (`DriftParams`: lag 10, allowance 0.3,
    threshold 10, baseline rate 0.01, warmup 50) and `drift_forget` (0):
    a one-sided CUSUM test on surprise (episode return minus the plan's
    predicted value from the start state). On an alarm the statistics are
    reset and the plan is re-solved after `replan_every` new episodes.
  * `drift_alarms()`. `DriftDetector` moved to `drift.hpp` so both the
    trainer and the model-based agent use it.
* **Tests:** `ModelBased.DetectsAMarketFlipAndRecoversWithinThirtyEpisodes`,
  `ModelBased.ChangeDetectionIsFreeInAStableMarketButConstantForgettingIsNot`.
* **Setup:** 300 episodes with alpha = +1, then alpha = −1; replanning
  every 5 episodes. Share of the new regime's optimum (seeds 1–3; probes
  `e9_probe.cpp`, `e9b_probe.cpp`):

| Variant | +10 | +20 | +50 | +100 | +300 | Online gain over the 300 episodes after the flip (of 300) |
|---|---|---|---|---|---|---|
| No forgetting | −169% to −171% | −165% to −167% | −154% to −159% | −123% to −129% | 80–82% | −186 to −237 |
| Forget 0.99 (half-life ~70) | −160% to −162% | −142% to −146% | 0–5% | 94–97% | 99–100% | +114 to +200 |
| Forget 0.95 (half-life ~14) | −35% to −115% | 79–100% | 99% | 100% | 98–100% | +204 to +290 |
| **Detect + reset** | −169% to −171% | **95–100%** | 99–100% | 99–100% | 99–100% | +178 to +292 |

Alarms fired 7–13 episodes after the flip. In stable markets (3,000
episodes, seeds 1–3; share of the optimum at the end):

| Market | No forgetting | Forget 0.99 | Forget 0.95 | Detect + reset |
|---|---|---|---|---|
| default | 100.0–100.1% | 99.9–100.2% | 97.5–99.4% | 100.1–100.2% (1 false alarm in 9,000 episodes) |
| noise 0.2 | 99.8–100.2% | 98.1–99.9% | **54.4–93.0%** | 99.9–100.2% (2 in 9,000) |
| q³ impact + fixed cost | 99.8–100.0% | 96.6–97.4% | 89.9–93.0% | 99.8–100.0% (0) |

* **A model that never forgets is slow to adapt:** its estimates average
  both regimes, so alpha only changes sign after as many new episodes as
  old ones (−123% after 100 episodes).
* **Constant forgetting trades accuracy for tracking.** At 0.95 it recovers
  in ~20 episodes but always plans on ~20 episodes' worth of data, which
  costs up to 46 points in a noisy market. With 0.99 the cost is small but
  recovery takes ~100 episodes.
* **Change detection gives both:** full memory while the market is stable,
  a reset when it isn't. It recovered to 95–100% within 20 episodes, 250×
  faster than the model-free agent reached 18%, because a handful of
  parameters re-estimate from a few episodes while tile weights need
  thousands of noisy updates. A false alarm costs little: the agent
  relearns a stable market in ~20 episodes.

## E10. Off-policy evaluation: can a policy be scored before it's deployed?

* **Question:** E7's paired model check and the trainer's held-out
  evaluations both replay scenarios, which needs a simulator. In a live
  market each scenario happens once, under the policy that was deployed.
  Can logs of that policy say how a *different* policy would have done?
* **Code:**
  * `ope.hpp`: `LoggedStep`/`LoggedEpisode`, and `off_policy_estimates()`
    with per-decision importance sampling (PDIS), its self-normalized form
    (WPDIS), doubly robust (DR), and two diagnostics: `min_ess` (the
    effective sample size of the weights) and `uncovered_steps`.
  * `ExecutionDP::evaluate_policy()`: a fixed policy's full V/Q table under
    a model (DR needs Q and V; the direct method, DM, is the model's start
    value of the target).
* **Tests:** `OffPolicy.EstimatorArithmeticOnAHandMadeLog`,
  `OffPolicy.ImportanceSamplingCollapsesAtLongHorizonsAndTheModelDecides`,
  `AgentExecutionDP.PolicyValueTablesAreSelfConsistent`.
* **Setup:** the behaviour is TWAP with 20% uniformly random actions (known
  propensities). DM fits the model-based agent passively to the same logs.
  The truth is exact (`policy_value` under the true market). Errors are in
  % of the optimal gain; 20 replications (probe `e10_ope.cpp`).

Model class right (default market), RMSE:

| Target | Logged episodes | DM | PDIS | WPDIS | DR | Steps nobody followed (of 20) |
|---|---|---|---|---|---|---|
| TWAP (the behaviour's base) | 100 / 1,000 / 10,000 | 5.1 / 2.0 / 0.7 | 63 / 13 / 6.2 | 58 / 13 / 6.4 | 56 / 13 / 5.7 | 0–0.1 |
| near-TWAP nudge | 100 / 1,000 / 10,000 | 4.9 / 1.8 / 0.6 | 108 / 78 / 100 | 119 / 58 / 45 | 94 / 48 / 118 | 0–0.5 |
| sign rule | 100 / 1,000 / 10,000 | 6.6 / 2.9 / 2.1 | 2968 / 57 / 101 | 47 / 55 / 55 | 311 / 76 / 159 | 17–19 |
| DP optimum | 100 / 1,000 / 10,000 | 10.7 / 4.1 / 1.1 | 137 / 395 / 91 | 108 / 98 / 87 | 285 / 134 / 137 | 15–18 |

Permanent impact γ = 0.01 and a model without the term, bias (RMSE):

| Target | 1,000 logs: DM | DR | 10,000 logs: DM | DR |
|---|---|---|---|---|
| TWAP | −8.1 (8.6) | −1.6 (10.5) | −8.6 (8.6) | +1.1 (4.6) |
| near-TWAP nudge | +13.0 (13.2) | +13.0 (39.5) | +12.7 (12.7) | +19.6 (70.4) |
| sign rule | +27.1 (29.2) | +28.5 (63.9) | +27.7 (29.6) | +21.8 (112.8) |
| DP optimum | −62.4 (62.5) | +74.7 (376) | −62.2 (62.2) | +29.6 (314) |

* **Importance sampling doesn't work at this horizon.** A deterministic
  target that differs from the behaviour in many states is followed by
  almost no logged episode beyond a few steps (`uncovered_steps` 15–19,
  ESS ≈ 0). Even TWAP itself, the behaviour's base, is followed for all
  20 steps by only 0.84²⁰ ≈ 3% of episodes. Errors of 45–2,968% don't
  shrink usefully with 10× more data.
* **The model decides.** DM is accurate when the model class is right
  (2–4% RMSE at 1,000 episodes), and biased when it isn't, by an amount more
  data doesn't shrink (−62% at both 1,000 and 10,000 episodes).
* **DR repairs model bias only near the behaviour:** for TWAP −8.6% →
  +1.1% at 10,000 episodes; for anything further away its IS corrections
  are pure noise.
* **What this means for deployment:** offline, only policies close to
  what was logged can be certified from data alone; anything else rests on
  a model, so the model has to be checked first. The check that works on
  live data compares the model's predictions with the logged behaviour
  policy itself, whose propensities are known. Measured in E7, that
  unpaired form needed ~1,500 episodes to flag γ = 0.005, where the paired,
  simulator-based check needed 100. When evaluation scenarios can be replayed (a calibrated
  simulator), paired evaluation remains far more sample-efficient.

## E11. Cheaper system identification (and a bug in the first plan)

* **Question:** E8 put the model-based agent's online losses in its
  random exploration. Would exploring near TWAP, or exploring less, make
  identification cheaper without making the model worse?
* **Bug found while setting this up:** the agent acts randomly until it
  has a plan, and the first plan waited for `replan_every` (20) episodes on
  top of `explore_episodes`. So the documented "10 exploration episodes"
  were 20, and any `explore_episodes` below 20 had no effect. Fixed: the
  first plan is made as soon as exploration ends
  (`ModelBased.FirstPlanComesRightAfterExploration`).
* **Code:** `Config::explore_lo`/`explore_hi` (range of exploratory
  actions); the default `eps` is now 0.01 (was 0.05).
* **Setup:** 1,000 online episodes, 5 seeds, mean cumulative gain over TWAP
  in optimal-episodes (+1,000 is the optimum); final share after 1,000
  episodes. Probe `e11_probe.cpp`.

| Variant (after the fix) | default market | q³ impact + fixed cost | permanent impact 0.01 |
|---|---|---|---|
| uniform 0–4, 10 episodes, ε 0.05 (old default) | +842 | +844 | +855 |
| uniform 0–4, 3 episodes, ε 0.05 | +847 | +817 | +863 |
| near TWAP (0.5–2×), 10 episodes, ε 0.05 | +840 | +805 | +870 |
| near TWAP, 3 episodes, ε 0.05 | +861 | +840 | +870 |
| uniform, 10 episodes, ε 0.02 | +836 | +860 | +918 |
| **uniform, 10 episodes, ε 0.01 (new default)** | **+893** | **+908** | **+943** |
| uniform, 10 episodes, ε 0 | +911 | +908 | +917 |

Every variant reached 99.7–100.2% of the optimum by episode 1,000, and
92–97% by episode 50 (86% for the near-TWAP variant with no exploration
episodes on the q³ market).

* **How the agent explores at the start barely matters** (±3%, within seed
  noise): ten random episodes lose 3–7 episodes' worth of the optimal gain
  relative to TWAP, once.
* **Ongoing exploration is the cost:** ε 0.05 after identification gave up
  6–10% of the online gain. The plan's own actions already vary with the
  signal, so the regressions stay identified without random actions.
* **New defaults re-measured:** 97.5–100% of the optimum after 20
  episodes, 99.4–100.4% after 50 (5 seeds, both cost models), as before.
  Online over 20,000 episodes: +19,043 to +19,988 (seeds 1–3), against
  +17,699 to +18,902 in E8 (old defaults). These online sums carry about
  ±500 of noise (returns have sd ≈ 55 per episode), so read them to
  ~2.5%.

## E12. A second task: market making, with the same agent

* **Question:** the agent, trainer and safeguards are written against the
  `Environment` concept. Do they work unchanged on a different task? And
  why did model-free learning plateau at ~70% on execution?
* **Code:** `market_making.hpp`: `MarketMakingEnv` (quote a bid and an ask,
  each one of three distances from the mid; fills with probability
  0.9·e^(−d); inventory capped at ±10; mark-to-market reward with a
  per-step inventory-risk penalty and a liquidation penalty at the end) and
  `MarketMakingDP` (exact, over (t, q)). No other file changed.
* **Tests:** `MarketMaking.AccountingIdentityAndInventoryCap`,
  `MarketMaking.DpMatchesSimulationAndSkewsAgainstInventory`,
  `MarketMaking.TheSameAgentDiscoversTheSkew`.
* **The task:** symmetric quoting at the spread-maximizing distance *loses*
  money (−3.27 per episode), because inventory drifts and the risk penalty
  grows with q². The optimum (+27.38; simulated 27.43 ± 0.05) skews: when
  long it quotes a wide bid and a tight ask, and the mirror image when
  short (the discrete-time version of Avellaneda & Stoikov, 2008). A
  hand-written "lean against inventory beyond ±2" rule gets 86% of the
  optimal gain.
* **Results** (share of the optimal gain over symmetric quoting, deployed
  policy, seeds 1–3; probe `e12_probe.cpp`):

| Learner | 1,000 episodes | 5,000 | 20,000 | 50,000 |
|---|---|---|---|---|
| TileQ | 61–68% | 85–87% | 95–96% | 97–99% |
| ActorCritic | 94–95% | 99–100% | 100% | 100% |

  On seed 1, both learners' quotes at mid-horizon match the DP's for every
  inventory from −8 to +8, except TileQ at q = −8.

* **Why this task is so much easier than execution:** the optimal gain
  divided by the return's standard deviation is **4.07** here and **0.27**
  for execution (gain 30.4 vs 15.6 per episode, return sd 7.5 vs 57.4).
  That ratio, not the size of the state space, is what makes model-free
  learning slow: with 15× less noise per unit of gain, TileQ gets to 97–99%
  instead of plateauing near 70%, and actor-critic needs 5,000 episodes
  instead of 200,000 to reach its best.
* **Generality, checked:** `Agent`, `Trainer`, the champion, the baseline
  guard and `compare()` worked on the new task without a single change.

## E13. Regime switches on the second task: when the market gets *better*

* **Question:** do drift detection and the policy bank (E0b, E3) carry over
  to market making unchanged?
* **First attempt, a null result:** doubling volatility doesn't change the
  optimal quotes at this action granularity (each regime's optimum scores
  100% in the other), so there's nothing to adapt to. The regimes used
  instead differ in book depth: fill decay 1 (A) vs 0.3 (B, deep book).
  A's optimum gets 66% of B's optimal gain; B's gets 79% of A's.
* **What went wrong, and why:** switching to B was never detected, so the
  agent sat at 66% of the new optimum for 1,000+ episodes. Two causes:
  1. **An improvement isn't a "drop".** In the deep book *every* policy
     earns more (A's optimal policy: 27.4 → 63.8 per episode), and the CUSUM
     test only looked for falling returns.
  2. **Blindness after rejected alarms (a bug).** Every alarm, including
     one the champion re-score rejected, reset the detector, which then
     needed lag + warmup = 1,550 samples before it could fire again. A false
     alarm at episode 9,030 left it blind until ~10,580; the switch came at
     10,000. When it finally fired (at +1,337), the champion had already been
     re-scored in the new world at regular evaluations, so confirmation saw
     "no change" and rejected it too.
* **Fixes:**
  * `DriftParams::two_sided` (off by default): a mirror-image CUSUM flags
    sustained *rises* in surprise; confirmation then accepts a significant
    rise as well as a drop.
  * **Evaluation-time detection (always on with `detect_drift`):** at every
    evaluation the champion's re-score is compared with its previous score.
    The champion is frozen and the scenarios are fixed, so in an unchanged
    world the two are identical; a shift of more than 2 standard errors is
    a confirmed change in either direction, with latency ≤ `eval_every` and
    no false alarms. The old champion is kept for that evaluation so it can
    be banked. `EvalPoint::world_changed` records it.
  * **No blindness:** an alarm now only clears the accumulated evidence. The
    caller resets the baseline after a confirmed change, and calls
    `rebase()` after a false one, which re-centres on recent samples and
    stays armed (0–1 false alarms in 300 samples after a rebase, real changes
    caught within 1–3 samples; 10 seeds).
* **Tests:** `AgentDrift.RebaseAfterAFalseAlarmStaysArmed`,
  `MarketMaking.AnImprovedMarketIsNoticedAndAReturningOneRecalled`.
* **Results** (actor-critic, 10k episodes in A, then B/A/B/A every 5,000;
  deployed share of the current regime's optimum, seeds 1–3; probe
  `e13_probe.cpp`):

| Switch | Detection | One-sided + eval-time, bank on: +200 / +1,000 / +5,000 | Two-sided, bank on | One-sided, bank off |
|---|---|---|---|---|
| → B (first) | one-sided: +500 (evaluation); two-sided: +4–5 | 66–68 / 66–69 / 87–90% | 66–68 / 74–76 / 87–91% | as bank on |
| → A | +4–6 (CUSUM) | **99–100** / 100 / 99–100% (recalled) | 99–100 / 100 / 100% (recalled) | 89–90 / 89–90 / 89–90% |
| → B | one-sided +500; two-sided +4–8 | 65–68 / 88–90 / 89–92% (recalled at +500) | **87–91** / 87–91 / 88–93% (recalled) | 87–90% throughout |
| → A | +3–5 | 99–100 / 100 / 100% (recalled) | 99–100 / 100 / 100% (recalled) | 88–90% throughout |

* **A changed world can be good news,** and an agent that only watches for
  bad news misses it. The cheapest reliable detector here is the one the
  trainer already had: the champion's exactly reproducible held-out score.
* **The bank does the heavy lifting on returns:** with it, every return to
  A is back at 99–100% within 200 episodes; without it actor-critic stays
  at 88–90%.
* **Actor-critic re-exploration, a mixed result** (`ac_reexplore_temperature`,
  default off). Re-exploring with ε-uniform actions is off-policy for the
  actor's update: on market making, without the bank, it stalled at 88–92%
  even 10,000 episodes after a switch. Sampling the policy itself at triple
  temperature, decaying with the boost, reached 95–99%. On execution's
  signal flip, though, where the old policy is actively harmful (−140%),
  the flattened softmax didn't explore enough: 29–30% at +40,000 against
  37–52% for ε-uniform (probe `e13d_exec.cpp`). Neither is the default for
  every task; ε-uniform stays the default.

## E14. Tail risk: what the mean doesn't show

* **Question:** every score so far is a mean. How do the policies' bad days
  compare, and does a variance penalty protect against them?
* **Code:** `episode_returns()` (per-episode returns on the usual scenarios)
  and `cvar(returns, alpha)` (mean of the worst `alpha` fraction) in
  `evaluate.hpp`.
* **Tests:** `TailRisk.CvarArithmetic`,
  `TailRisk.VarianceIsNotTailRiskInExecution`,
  `MarketMaking.TheInventoryPenaltyCutsTheTail`.
* **Setup:** each policy is scored on *cash* outcomes (the same market with
  the risk penalty switched off), on 20,000 shared scenarios (probe
  `e14_probe.cpp`).

Execution:

| Policy | Mean | sd | CVaR 5% | CVaR 1% | Worst |
|---|---|---|---|---|---|
| TWAP | −5.6 | 54.5 | −117.1 | −150.8 | −215.4 |
| DP optimum, λ 0 (risk-neutral) | +10.0 | 57.4 | −66.0 | −78.1 | −118.1 |
| DP optimum, λ 0.05 | +9.2 | 52.6 | −62.5 | −76.6 | −117.5 |
| DP optimum, λ 0.2 | +3.1 | 41.8 | −60.8 | −75.7 | −117.5 |
| DP optimum, λ 1 | −8.2 | 27.5 | −61.1 | −75.7 | −116.6 |
| TileQ, 40k episodes | +5.1 | 51.1 | −72.7 | −90.8 | −135.2 |
| ActorCritic, 40k episodes | +7.2 | 52.3 | −64.9 | −77.2 | −116.4 |

Market making (default risk penalty 0.05 unless noted):

| Policy | Mean | sd | CVaR 5% | CVaR 1% | Worst |
|---|---|---|---|---|---|
| Symmetric quotes | +22.9 | 26.3 | −47.7 | −79.7 | −145.4 |
| DP optimum, risk penalty 0 | +31.4 | 16.6 | −5.7 | −21.3 | −75.1 |
| DP optimum, risk penalty 0.05 (and 0.2: same policy) | +29.0 | 7.6 | **+13.7** | +8.8 | −14.9 |
| ActorCritic, 20k episodes | +29.1 | 7.7 | +13.5 | +8.5 | −24.3 |

* **Volatility isn't tail risk.** The risk-neutral execution optimum is
  *more* volatile than TWAP (sd 57 vs 54) yet loses far less on its worst
  days (CVaR 5% −66 vs −117): the signal lets it avoid the worst outcomes,
  and the extra spread is on the upside.
* **In execution, a variance penalty buys variance, not tail protection.**
  λ = 1 halves the sd (57 → 28) but improves the worst 5% only from −66 to
  −61, while the mean falls from +10 to −8. The worst outcomes come from
  impact and early price moves that the variance term doesn't target.
* **In market making the same kind of penalty does protect the tail:**
  7% of the mean (31.4 → 29.0) buys a worst 5% of +13.7 instead of −5.7,
  because there the tail *is* inventory, and the penalty controls inventory.
* **Learned policies:** actor-critic's tails match the optimum's on both
  tasks; TileQ's execution policy has a noticeably worse tail (−73 vs −66)
  than its mean gap suggests.
* **Practical rule:** check the tail you care about directly (CVaR on cash
  outcomes); whether a variance penalty helps depends on where the tail
  comes from.
