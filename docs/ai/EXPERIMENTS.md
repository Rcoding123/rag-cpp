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
