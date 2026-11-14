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
| E15 | Does a model pay on the easy task too? | Yes: a model-based market maker that counts fills reaches 98–100% after 3 episodes and earns 99–100% of the optimal gain online, against 65–79% for actor-critic. |
| E16 | Would a neural network (PyTorch-style) do better? | Yes: a from-scratch MLP (DQN-style, no library) reaches 84–89% at 40k (deployed 86–92%) vs 65–69% for TileQ. Not the target network, not Adam: generalization. Coarser tiles alone reach 76–77%. |
| E17 | Does a control variate for the price noise help? | Not by default: the innovation is ~5% of the return variance (the signal path dominates). In a noisy market (noise_sd 0.2) it lifts TileQ from 49–61% to 67–71%. Decompose before reducing. |
| E18 | Can a proper optimizer tune the learner? | NLopt (CRS2 global + Subplex) lifts TileQ at 10k episodes from 37% [33, 41] to 59% [52, 66] on held-out seeds; the Eigen response surface rediscovers E8 and E16. Local search alone was start-dependent; winners were 0.3–18.5 points optimistic on their own seeds. |
| E20 | Do Double DQN and advantage learning help the neural learner? | Yes, mostly through stability: deployed 87–94% at 20k episodes vs 81–87% plain; the learner's own greedy policy 87–91% vs 70%. |
| E21 | How do all learners rank under one protocol (8 seeds, CIs)? | At 40k episodes: NeuralQ + Double DQN 91.5% [90.1, 92.7], ActorCritic 81.9%, TileQ coarse + AL 78.7%, tuned TileQ 73.4%, default TileQ 66.7%; model-based 99.5% after 50 episodes. |
| E22 | Can a lookup table replace the neural policy? | From 32 signal bins the table keeps the network's gain (within 0.25 points; never worse than 0.8) at ~3 ns per decision instead of ~1.7–3 µs; the exact DP policy compiled the same way keeps 99.6–100%. |
| E23 | Does a vote across seeds' tables beat one network? | Yes: the median of 8 seeds' tables keeps 97.5% (fresh scenarios: 97.9%, better than every member by 4.4–10.2 points), vs 91.4% for the average seed and 94.1% for the best; 3 members already beat the best seed; at equal compute, 4 × 40k voted (96.7%) beats 1 × 160k (93.5%). |
| E24 | Does averaging one network's weights help? | Yes, with a slow average: τ = 1e-4 deploys 94.1% (93.0–95.5) at 40k vs 91.5%, better on all 8 seeds (+1.0 to +5.6 points); ~ a 2–3-network vote for one training run. τ = 1e-3 is too fast. |
| E25 | Do the vote and weight averaging add up? | Yes, with diminishing returns: 8 averaged networks voted reach 97.8% (best model-free result), 3 reach 96.6% (vs 95.4% unaveraged); the gains overlap because both remove per-network noise. |
| E26 | Where do the learned policies err? | They over-trade: trading too much costs 3.5–11.2% of the optimal gain vs 0.7–5.8% for too little; the vote removes idiosyncratic errors, not the bias. |
| E27 | Does a lower-quantile vote fix the bias? | No: better on the selection set (98.1 vs 97.5%), not on fresh scenarios (97.8 vs 97.9%) — selection bias. |
| E28 | Do strategies survive a different market? (stress suite, sealed seeds) | Fixed strategies (DP plan, learned ensemble) only in their assumed market (−173% when the signal flips); the model-based agent re-learned in 50 episodes keeps 97–100% everywhere. |
| OF1–OF9 | ES order-flow trading | See ORDERFLOW.md §4: RL scalping beats every rule in the simulator (robust to 9 stresses) but the simulator is too predictable to trust absolute P&L; tick-scalping RL matches, not beats, the rule. Caveat (found in OF10): every OF2–OF9 episode ran at the simulator's activity floor (~4 trades/s instead of 15–22). |
| OF10 | Can the simulator discover prices in seconds? | Partly. Rigid quote jumps (v1) remove the 30 s gap predictability (0.57–0.64 → 0.11) but make book imbalance useless (+0.02). Stale-side depletion (v2) keeps both (gap 0.09–0.11, imbalance → next tick +0.25) but fails the stylized facts and passive quoting (−6.6 to −7.2 ticks); its oracle bound is vacuous. v3 (stationary activity model, no informed makers) meets the gap, imbalance (marginally), the facts and naive-rule breakeven, and its pre-registered oracles trade; it fails passive quoting (−24 to −27 ticks, stale sweeps), and meta-orders are 5% short after the 150 s warm-up (a 1,050 s warm-up fixes it: 29/29 AGREE). OF11: RL on v3 finds no edge (every seed negative on sealed episodes; ensemble flat). |
| E19 | Do gap-increasing targets help? | Advantage learning (k = 0.5) lifts TileQ from 65–69% to 70–75% at 40k, and to 76–80% with coarse tiles: the best tile-coded result, still below the neural learner. |

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

## E15. A model-based market maker

* **Question:** on execution, learning a model beat model-free learning by
  orders of magnitude (E1). Market making is the *easy* task for model-free
  learners (E12). Does a model still pay?
* **Code** (`market_making.hpp`):
  * `MarketMakingParams::fill_probs`: an optional fill probability per quote
    distance, so a market can have a fill curve that isn't exponential.
  * `MarketMakingEnv::last_bought()`, `last_sold()`, `last_mid_move()`: the
    agent's own fills and the mid move, which a real market maker sees as
    execution reports and market data.
  * `ModelBasedMarketMaker`: knows its objective (risk penalty, liquidation
    penalty, inventory cap, distances) but not the market. It estimates each
    distance's fill probability by counting (Beta(1, 1) prior, pooling bid
    and ask) and the volatility from observed moves, then plans with
    `MarketMakingDP`. The fill model is non-parametric over the three
    distances. Three random episodes first, then ε 0.01; re-plans after 5
    episodes, doubling. `train_model_based()` runs it online.
* **Test:** `MarketMaking.AModelBasedMarketMakerNeedsAFewEpisodes`.
* **Results** (seeds 1–3; probe `e15_probe.cpp`):

| Fill curve (distances 0.5 / 1 / 2) | Model-based after 3 / 10 / 50 episodes | Online gain over 1,000 episodes (of 1,000) | ActorCritic after 1,000 | ActorCritic online |
|---|---|---|---|---|
| exponential (0.55 / 0.33 / 0.12) | 100 / 100 / 99–100% | +986 to +1,009 | 94–96% | +655 to +678 |
| steep (0.85 / 0.15 / 0.10) | 98–100 / 98–99 / 99–100% | +989 to +995 | 99–100% | +770 to +788 |

  After 1,000 episodes the estimated fill probabilities match the truth to
  two decimals (one is 0.17 for 0.15) and the volatility to 1.00.

* **A model pays even on the easy task:** 3 episodes instead of 5,000, and
  99–100% of the optimal gain earned *while* learning instead of 65–79%.
* **Why it's so fast here:** the unknowns are four numbers, each observed
  directly (fills are counted, moves are seen), and the planner is exact.
  The expensive part of model-based RL is getting the model *class* right;
  here it's non-parametric over the three distances, so any fill curve fits.

## E16. A neural Q-learner, from scratch

* **Question:** you asked whether a framework like PyTorch would do better.
  Rather than add the dependency, the question behind it was tested with a
  small neural network written from scratch: does neural function
  approximation beat tile coding on execution, and if so, why?
* **Code:**
  * `mlp.hpp`: a multilayer perceptron (ReLU hidden layers, linear output),
    backpropagation, Adam and plain SGD, He-uniform initialization from the
    seeded `Rng`. No library.
  * `LearnerKind::NeuralQ` in `agent.hpp`: DQN-style. The MLP (default 2 ×
    64 hidden units) maps the features to five Q-values; each finished
    episode becomes 3-step transitions in a feature-space replay buffer
    (100k); one Adam minibatch (32) per step taken; a target network copied
    every 2,000 updates. `AgentConfig::nn_hidden`, `nn_lr`, `nn_batch`,
    `nn_target_sync`, `nn_warmup`. Trainer, champion, drift detection and
    guard work unchanged.
* **Tests:** `Mlp.GradientMatchesFiniteDifferences`,
  `Mlp.AdamFitsASmoothFunctionAndIsDeterministic`,
  `AgentNeural.LearnsFasterThanTileCodingOnExecution`,
  `AgentNeural.EvaluationDoesNotPerturbLearningAndSnapshotsAct`.
* **Verification:** backpropagation matches finite differences to
  1.4 × 10⁻¹¹; the network fits a nonlinear 3-D function to MSE 0.0004.

**Representation** (supervised fit to the exact Q*, greedy policy; probe
`e16_dqn.cpp fit`, `ceiling2.cpp`):

| Function approximator | Weights | Greedy on fitted Q* |
|---|---|---|
| MLP 3-32-32-5 | 1,349 | 83.6% (one step per episode) |
| MLP 3-64-64-5 | 4,741 | 92.8% (one step per episode); **98.6%** (one per state) |
| MLP 3-128-128-5 | 17,669 | 99.0% |
| tiles 8×8×8 × 8 (default) | 5,832 per action | 99.4% |
| tiles 4×4×4 × 8 | 1,000 per action | 98.2% |
| tiles 2×4×4 × 16 | 1,200 per action | 98.3% |

All of them can represent a ≥ 98% policy. The differences below are about
learning, not capacity.

**Learning from rewards** (default market, ε 1 → 0.02 over 10k episodes,
greedy learner's share of the optimal gain, seeds 1–3; probes
`e16_dqn.cpp`, `e16_tileq.cpp`, `e16_coarse.cpp`):

| Learner | 3,000 | 5,000 | 20,000 | 40,000 |
|---|---|---|---|---|
| TileQ, default tiles, 4 replays/step | −13 to +19% | 8–16% | 56–60% | 65–69% |
| TileQ, default tiles, 32 replays/step | −72 to −5% | 2–21% | 48–57% | 58–61% |
| TileQ, tiles 4×4×4 × 8 | | 7–46% | 57–67% | 72–75% |
| TileQ, tiles 2×4×4 × 16 | | 33–47% | 65–72% | **76–77%** |
| **DQN (MLP 64-64, Adam, target net)** | 67% (seed 1) | 66–81% | 76–86% | **84–89%** |

Ablations (DQN, 20,000 episodes, seeds 1–2):

| Variant | 5,000 | 10,000 | 20,000 |
|---|---|---|---|
| as above | 66–81% | 68–82% | 76–86% |
| no target network | 57–73% | 78–81% | 85–87% |
| SGD lr 10⁻³ instead of Adam | 63–75% | 78–80% | 67–83% |
| SGD lr 10⁻² | diverged | | |

* **The neural learner breaks the plateau:** 84–89% at 40k episodes
  against 65–69% for tile-coded Q-learning (and 75–77% after 200k, E2),
  with the same 3-step targets and the same exploration schedule.
* **It isn't the extra computation:** giving TileQ 32 replayed updates per
  step (DQN's minibatch) made it *worse* (58–61%).
* **It isn't the target network or Adam:** without the target network it
  did as well (85–87% at 20k); plain SGD reached 67–83% (less stable, but
  far above TileQ).
* **It's how the function generalizes.** Coarser tiles, which average the
  noisy targets over more states per weight, already lift TileQ from
  65–69% to 76–77% while keeping a 98% representation ceiling, so the
  default 8×8×8 tiling is too fine for this noise level (A3 concluded the
  same about 12³ vs 8³). The network goes further: it averages globally
  but can still put resolution where the values bend.
* **Through the library** (`LearnerKind::NeuralQ` with the standard
  `Trainer`, champion evaluated every 1,000 episodes on 500 scenarios,
  seeds 1–3; probe `e16_lib.cpp`): deployed 63–80% at 5k, 78–87% at 10k,
  81–88% at 20k, **86–92%** at 40k.
* **The champion matters more for neural learners:** the learner's greedy
  policy swings between checkpoints (seed 1: 81% → 70% → 90%; seed 2 ends
  at 73%), while the deployed champion holds (87%, 86%).
* **Costs:** about 7.5 ms of training per episode (vs ~0.1 ms for TileQ)
  and 1.7 µs per greedy decision (vs 39 ns), measured with the current
  scalar double-precision forward pass (not yet vectorized). For a latency-critical deployment the
  natural split is to train the network offline and distill it into tiles
  or a lookup table.
* **What this changes in the earlier conclusions:** E2 was right that the
  tiles could represent a near-optimal policy and that target noise was the
  problem, but wrong to imply that the representation choice didn't matter.
  How a function approximator *averages* noise is as important as what it
  can represent.

## E17. A control variate for the price noise (the research's top idea)

* **Question:** RESEARCH.md ranked first a control variate: each reward
  contains I′·(price innovation), which has zero mean given (s, a), so
  subtracting it leaves every policy's value (and the optimum) unchanged
  while removing noise. Does it break the tile-coding plateau?
* **Code:** `ExecutionEnv::last_move()` (the step's price move: market data);
  `variance_reduction.hpp`: `InnovationControlledEnv` wraps `ExecutionEnv`
  and returns r − I′·(ΔS − α̂·g), with α̂ supplied or estimated online by
  regressing observed moves on the signal (no market knowledge assumed). It
  satisfies the `Environment` concept, so learners and the trainer use it
  unchanged.
* **Tests:** `VarianceReduction.ControlledRewardsKeepValuesAndCutOnlyTheInnovation`,
  `VarianceReduction.EstimatesTheDriftFromMarketData`.
* **First, where the noise comes from** (20,000 episodes; probe
  `e17_decomp.cpp`):

| Market, policy | Innovation I′·ε | Drift I′·α·g (the signal path) | Costs | Total |
|---|---|---|---|---|
| default, TWAP | sd 12.5 | sd 52.9 | sd 0 | sd 54.4 |
| default, optimal | sd 13.2 | sd 56.1 | sd 1.4 | sd 57.4 |
| noise_sd 0.2, TWAP | sd 49.9 | sd 52.9 | sd 0 | sd 72.9 |
| noise_sd 0.2, optimal | sd 52.7 | sd 56.1 | sd 1.4 | sd 76.9 |

* **Learning** (TileQ, 40k episodes, seeds 1–3; α̂ estimated, it came out at
  0.999–1.001):

| Market | Plain rewards: 5k / 40k | Controlled: 5k / 40k |
|---|---|---|
| default | 8–16% / 65–69% | 2–23% / 67–68% |
| noise_sd 0.2 | −1–10% / 49–61% | **21–26% / 67–71%** |

* **In the default market it does nothing,** because the innovation is not
  the dominant noise. Most of the return's variance comes from the signal
  path (sd 53–56): the very information the agent trades on, realized along
  a random path. A reward-side control variate can't remove that without
  changing the optimal policy.
* **Where the innovation is large, it works exactly as theory says:** with
  noise_sd 0.2 it lifts TileQ from 49–61% to 67–71%, the level of the
  default market.
* **Lesson:** decompose the return variance before reducing it. The
  research's prediction ("the largest single gain; above 90%") assumed the
  innovation dominated; measured, it was 5% of the variance.

## E18. Proper optimization and data analysis: NLopt and Eigen

* **Question:** you asked for a real optimization and data-analysis library.
  Can a principled hyperparameter search beat the hand-chosen defaults, and
  what does a data analysis of the search say?
* **Libraries** (optional, like poppler; `RAG_OPTLIBS`, on when found):
  NLopt 2.7.1 (`libnlopt-cxx-dev`) and Eigen 3.4.0 (`libeigen3-dev`).
* **Code:**
  * `tuning.hpp`: `maximize()` (any NLopt algorithm, default Subplex) and
    `maximize_global()` (CRS2 global search, then Subplex), over parameters
    with log-scale and integer options, keeping every trial; seeded for
    reproducibility.
  * `analysis.hpp`: `least_squares()` (weighted, rank-revealing QR) and
    `quadratic_sensitivity()` (response surface on standardized parameters).
  * `evaluate.hpp`: `bootstrap_mean()` (percentile bootstrap intervals; no
    dependency).
  * `examples/tune_agent.cpp`: tunes six TileQ hyperparameters on
    ExecutionEnv, validates on held-out seeds and reports sensitivities.
* **Tests** (`tests/optional_libs_tests.cpp`, target `rag_optlib_tests`):
  the tuner finds a known optimum with log and integer parameters; the
  global stage escapes a local optimum a local search gets stuck in;
  Eigen's QR reproduces the model-based agent's hand-written
  normal-equations solve to 1e-6 (a cross-check of our own solver);
  the response surface recovers known effects.
* **Setup:** objective = mean share of the optimal gain after 10,000
  training episodes on fixed search seeds (deterministic); 120 evaluations;
  validation on seeds the search never saw.

| Search | Search seeds | Best on search seeds | Held-out: default | Held-out: tuned |
|---|---|---|---|---|
| Subplex, hand-set steps, 60 evals | 1, 2 | 71.4% | 40.9% (seeds 3–7) | 60.9% |
| Subplex, wider bounds, 120 evals | 1, 2, 8 | 70.1% | 40.9% (seeds 3–7) | 69.8%; at 40k: 64.0% → 75.6% |
| Subplex via the library (default steps), 120 evals | 1, 2, 8 | 57.0% | 37.1% [32.8, 40.8] (8 seeds) | 42.2% [27.6, 51.7] |
| **CRS2 global + Subplex (tool default), 120 evals** | 1, 2, 8 | 77.7% | 37.1% [32.8, 40.8] | **59.2% [52.4, 66.4]** |

Sensitivity across the 129 global-search trials (Eigen quadratic surface,
R² = 0.56; effect of +1 standard deviation of each parameter):
ε start −14.1%, bins −11.7%, α +8.3% (curvature −7.9%: an interior optimum),
α decay −3.1%, ε decay −2.7%, tilings +2.0%.

* **Tuning works when validated:** after 10,000 episodes the tuned learner
  reaches 59–70% on unseen seeds against 37–41% for the defaults, roughly
  what the defaults reach after 20,000–40,000 episodes.
* **A local search alone is unreliable here.** With the same budget,
  Subplex found 57% or 70% depending only on its initial step sizes; the
  first response surface explained only 17% of the variance. The landscape
  is rugged and seed-noisy, so the tool searches globally first.
* **The winner's curse is real:** the best configuration scores 0.3–18.5
  points higher on the seeds it was chosen on than on unseen ones. With 2 search
  seeds the gap was 10 points; with 3 seeds and a local search, 0.3; with
  the broader global search, 18. Always validate on held-out seeds.
* **The data analysis rediscovers the earlier hand-found lessons
  independently:** low starting ε matters most (E8), coarser tiles next
  (E16), and the step size has an interior optimum (A4).

## E19. Advantage learning: widening the action gaps

* **Question:** RESEARCH.md §2 suggests gap-increasing targets for tasks
  whose action-value gaps are tiny relative to target noise (here: gaps < 1,
  targets ±20–50). Advantage learning (Baird 1999; Bellemare et al. 2016)
  subtracts k × (max_b Q(s, b) − Q(s, a)) from each target, which widens
  the gaps while preserving the optimal policy.
* **Code:** `AgentConfig::advantage_learning` (k; default 0 = plain
  Q-learning), applied in TileQ's TD update.
* **Test:** `AgentLearning.AdvantageLearningTargetByHand`.
* **Results** (TileQ, ε 1 → 0.02 over 10k, seeds 1–3; probe `e19_al.cpp`):

| Tiles | k | 5,000 | 20,000 | 40,000 |
|---|---|---|---|---|
| default 8×8×8 × 8 | 0 | 8–16% | 56–60% | 65–69% |
| default 8×8×8 × 8 | **0.5** | **23–36%** | 62–65% | **70–75%** |
| default 8×8×8 × 8 | 0.9 | −4 to +36% | 64–68% | 62–75% |
| coarse 2×4×4 × 16 | 0 | 33–47% | 65–72% | 76–77% |
| coarse 2×4×4 × 16 | **0.5** | −123 to +46% | 70–76% | **76–80%** |
| coarse 2×4×4 × 16 | 0.9 | 30–47% | 70–78% | 68–80% |

* **k = 0.5 helps:** +5 points at 40k with the default tiles and roughly
  twice the share at 5k; with coarse tiles it gives the best tile-coded
  result so far (76–80%), still below the neural learner (84–89%).
* **k = 0.9 is too aggressive:** the spread across seeds widens (62–75%).
* **Early learning is noisier with gap-increasing targets** (one coarse
  seed at −123% at 5k), so the champion mechanism matters.
* Not made the default: the gain is moderate and other tasks' tests are
  tuned on k = 0.


## E20. Double DQN and advantage learning for the neural learner

* **Question:** the two target corrections that helped or should help
  (RESEARCH.md §2): does `NeuralQ` improve with Double DQN (the online
  network picks the next action, the target network values it, removing the
  upward bias of a max over noisy estimates) and with advantage learning?
* **Code:** in `NeuralQ`, `double_q` now means Double DQN and
  `advantage_learning` applies the E19 correction with the online network.
* **Test:** `AgentNeural.DoubleAndAdvantageTargetsStillLearn`.
* **Results** (library `NeuralQ` through the `Trainer`, champion evaluated
  every 1,000 episodes; seeds 1 / 2 at 20,000 episodes; probe `e20_nq.cpp`):

| Variant | Learner's greedy policy | Deployed champion |
|---|---|---|
| plain (E16) | 70% / 70% | 87% / 81% |
| Double DQN | 87% / 91% | **87% / 94%** |
| advantage learning 0.5 | 87% / 83% | 91% / 83% |
| both | 90% / 88% | 90% / 90% |

* **Both corrections help, and they mostly help stability.** The plain
  learner's greedy policy sat at 70% at 20k (its champion held 81–87%);
  with Double DQN or both corrections the learner itself is at 87–91%,
  and the deployed policy at 87–94%: the best model-free results on
  execution so far, at half the episodes E16 used.
* Two seeds only (each run is ~2.5 minutes): treat the ranking between the
  three corrected variants as unresolved; the gain over plain is
  consistent.


## E21. The leaderboard: every learner under one protocol

* **Question:** results so far came from different probes, seed counts and
  evaluation sets. Under one protocol, with confidence intervals, how do
  the learners rank? (RESEARCH.md §5, "evaluation hygiene".)
* **Protocol:** default execution market; 8 seeds per learner; ε 1 → 0.02
  over 10k episodes unless tuned; the `Trainer`'s deployed champion
  (evaluated every 1,000 episodes on 500 scenarios); share of the optimal
  gain on the same 1,000 held-out scenarios for every learner; mean,
  95% percentile-bootstrap interval, interquartile mean (IQM) and range.
  Probe `experiments/e21_board.cpp`, aggregation `experiments/e21_agg.py`
  (all probes are in `experiments/`).

| Learner | Episodes | Seeds | Mean | 95% CI (bootstrap) | IQM | Min–max |
|---|---|---|---|---|---|---|
| TileQ, default settings | 20,000 | 8 | 58.8% | [57.5, 60.3] | 58.2% | 57–63% |
| TileQ, default settings | 40,000 | 8 | 66.7% | [65.3, 68.1] | 66.7% | 63–70% |
| TileQ, tuned by NLopt (E18) | 20,000 | 8 | 67.8% | [59.4, 75.1] | 70.6% | 46–79% |
| TileQ, tuned by NLopt (E18) | 40,000 | 8 | 73.4% | [66.2, 80.0] | 75.1% | 57–84% |
| TileQ, coarse tiles + advantage learning (E16, E19) | 20,000 | 8 | 70.8% | [66.9, 74.5] | 71.9% | 62–80% |
| TileQ, coarse tiles + advantage learning (E16, E19) | 40,000 | 8 | 78.7% | [77.8, 79.6] | 78.6% | 76–81% |
| ActorCritic | 20,000 | 8 | 77.6% | [73.6, 80.3] | 79.6% | 65–81% |
| ActorCritic | 40,000 | 8 | 81.9% | [80.5, 83.1] | 82.2% | 78–84% |
| NeuralQ + Double DQN (E16, E20) | 20,000 | 8 | 86.3% | [84.2, 88.9] | 85.6% | 83–94% |
| NeuralQ + Double DQN (E16, E20) | 40,000 | 8 | 91.5% | [90.1, 92.7] | 91.6% | 88–94% |
| Model-based (Shrunk) | 20 | 8 | 98.8% | [97.5, 99.9] | 99.6% | 96–100% |
| Model-based (Shrunk) | 50 | 8 | 99.5% | [98.8, 100.0] | 99.9% | 97–100% |

* **Ranking at 40,000 episodes:** NeuralQ + Double DQN 91.5% > ActorCritic
  81.9% > TileQ with coarse tiles and advantage learning 78.7% > NLopt-tuned
  TileQ 73.4% > default TileQ 66.7%. The top three are separated (their
  bootstrap intervals don't overlap); the tuned configuration's wide
  interval overlaps both of its neighbours.
* **The model-based agent is in a different league:** 98.8% after 20
  episodes, 99.5% after 50 (2,000× fewer episodes).
* **Tuning for one budget doesn't transfer cleanly to another:** the E18
  configuration was chosen at 10k episodes; at 40k its mean (73.4%) beats
  the default but its spread is wide (57–84%), while coarse tiles plus
  advantage learning are both better and tight (76–81%).
* **The neural learner's worst seed beats every other model-free learner's
  best:** 88% at 40k, against at most 84% for ActorCritic and tuned TileQ.


## E22. Compile the neural policy into a lookup table

* **Question:** `NeuralQ` is the best model-free learner (E21) but costs
  1.7 µs per decision (about 3 µs with all cores busy), against 64 ns for
  TileQ. The execution state is small: 20 time steps × 101 inventories ×
  one continuous signal. Can a table of the network's greedy actions,
  sampled at the centre of each (t, inventory, signal bin) cell, keep its
  gain?
* **Setup:** `ExecutionPolicyTable` (`policy_table.hpp`); NeuralQ + Double
  DQN, 40,000 episodes, deployed champion, seeds 1–8 (the E21 protocol:
  the network scores match E21 to six digits); the same 1,000 held-out
  scenarios. "Same action" is the share of the decisions the network makes
  on those scenarios where the table picks the same action. Probes
  `experiments/e22_table.cpp`, `e22_dp_ceiling.cpp`.
* **The grid's own cost** (the exact DP policy, compiled): 82.1% of the
  optimum at 4 bins, 95.4% at 8, 98.9% at 16, and 99.6–100.4% from 32
  bins up (sampling noise around 100%).

| Signal bins | Range (sd) | Table size | Share (mean, min–max) | Change vs the network, points (mean, min–max) | Same action as the network | ns per decision |
|---|---|---|---|---|---|---|
| network | | | 91.5% (87.7–94.1) | | | ~3,000 (cores shared) |
| 4 | ±4 | 8 KB | 76.8% (75.0–78.7) | −14.71 (−16.08 to −11.13) | 63.8% | 2.6 |
| 8 | ±4 | 16 KB | 87.8% (83.3–91.3) | −3.69 (−4.75 to −1.73) | 76.5% | 2.8 |
| 16 | ±4 | 32 KB | 90.3% (86.3–93.6) | −1.21 (−1.71 to −0.53) | 86.7% | 2.5 |
| 32 | ±4 | 65 KB | 91.3% (86.9–94.1) | −0.16 (−0.80 to +0.22) | 93.2% | 2.7 |
| 64 | ±4 | 129 KB | 91.2% (87.2–93.9) | −0.25 (−0.52 to −0.02) | 96.6% | 2.9 |
| 64 | ±3 | 129 KB | 91.6% (87.5–94.2) | +0.15 (−0.22 to +0.44) | 97.3% | 3.1 |
| 128 | ±4 | 259 KB | 91.3% (87.5–93.9) | −0.16 (−0.47 to +0.11) | 98.3% | 2.7 |
| 256 | ±4 | 517 KB | 91.4% (87.2–94.1) | −0.05 (−0.48 to +0.14) | 99.1% | 3.0 |

* **From 32 bins, the table is the network:** within 0.25 points on
  average and never more than 0.8 points worse on any seed, about 1,000×
  faster (3 ns; 3.45 ns at random states), with no floating-point model at
  decision time.
* **Disagreement doesn't mean loss:** at 32 bins the table disagrees on
  6.8% of decisions yet loses 0.16 points. The disagreements are at bin
  edges, where the network's own action boundary lies and the two actions
  are worth nearly the same.
* **Bins matter more than range:** 64 bins over ±3 sd (finer bins, 0.3% of
  signals clamped) did slightly better than over ±4 sd.

## E23. Ensembles of compiled policies

* **Question:** a table costs the same whatever produced it. Does a vote
  across networks trained with different seeds beat a single network, at
  no extra decision cost?
* **Setup:** the 8 seeds' tables from E22 (256 and 64 bins, ±4 sd);
  `ExecutionPolicyTable::median` combines tables cell by cell with the
  lower median action (actions are ordered by trade size); a plurality
  vote (ties to the smaller action) for comparison. Every one of the 255
  non-empty subsets of the 8 seeds is scored, so no ensemble is selected on
  the test scenarios. Probes `experiments/e23_vote.cpp`, `e23_fresh.cpp`.

256 bins (64 bins: within 0.3 points of these everywhere):

| Members | Ensembles | Median vote: mean (min–max) | Plurality vote: mean (min–max) | Median vote beats its members' mean |
|---|---|---|---|---|
| 1 | 8 | 91.4% (87.2–94.1) | 91.4% (87.2–94.1) | — |
| 2 | 28 | 93.1% (89.0–96.3) | 93.1% (89.0–96.3) | 22/28 |
| 3 | 56 | 95.4% (93.5–97.2) | 94.5% (92.6–96.1) | 56/56 |
| 4 | 70 | 96.7% (95.4–97.9) | 95.5% (93.8–96.5) | 70/70 |
| 5 | 56 | 96.4% (95.2–97.4) | 96.2% (94.9–97.2) | 56/56 |
| 6 | 28 | 97.4% (96.5–98.1) | 96.7% (95.6–97.2) | 28/28 |
| 7 | 8 | 96.7% (96.2–97.2) | 96.8% (96.2–97.3) | 8/8 |
| 8 | 1 | 97.5% | 97.3% | 1/1 |

* **The ensemble is the best model-free policy in the project:** the median
  of all 8 tables keeps 97.5% of the optimal gain, against 91.4% for the
  average seed and 94.1% for the best one. Three members already beat the
  best single seed on average (95.4%).
* **Confirmed on fresh scenarios:** on 4,000 scenarios none of the earlier
  evaluations used, the 8-member ensemble scores 97.9% and beats every
  member, by 4.4 to 10.2 points (paired standard errors 0.3–0.5, so z ≥ 13).
* **Why it works:** the networks reach the same score by different routes:
  each one's errors sit in different cells, and the median removes errors
  that a majority doesn't share. This is the variance argument for bagging;
  the seeds' policies are diverse enough for it to pay.
* **Median vs plurality:** the median is better or equal at every size
  except 7 (within 0.1 points). With ordered actions, the median also
  accounts for how far apart the votes are, not only how many there are.
* **An unconfirmed hint:** at even sizes the lower median (which leans
  toward the smaller trade) does as well as or better than the next odd
  size (4 members 96.7% vs 5 members 96.4%; 6 vs 7: 97.4% vs 96.7%), and
  for two members it beats their mean in 22 of 28 pairs. That suggests the
  networks err toward trading too much; untested.
* **Not just more training (compute-matched):** one network trained 4×
  as long (160,000 episodes, seeds 1–4, same protocol) reaches 93.5%
  (93.0–94.0%); a median vote of 4 networks trained 40,000 episodes each
  reaches 96.7% on average over all 70 groups of 4 (95.4–97.9%; seeds 1–4:
  97.0%). The worst 4-member vote beats the best long-trained network:
  for the same compute, independent seeds are worth more than longer runs.
* **The cost is training, not deployment:** N members need N× the
  training compute; a decision stays one 3 ns lookup.


## E24. Weight averaging: one network, averaged

* **Question:** E23's vote needs N training runs. Averaging one network's
  weights over time (Polyak averaging, an exponential moving average; the
  idea behind stochastic weight averaging, RESEARCH.md §6.3) costs one
  extra copy of the weights. How much of the ensemble's gain does it get?
* **Setup:** `AgentConfig::nn_polyak = τ`: after every update,
  avg += τ (net − avg). Exploring actions use the online network, so
  training is bit-identical to E21 (a test checks this); greedy actions,
  snapshots and the deployed champion use the average. NeuralQ + Double
  DQN, the E21 protocol, seeds 1–8, paired against E21's runs of the same
  seeds. "Final" is the last greedy policy, without the Trainer's champion
  selection. Probe `experiments/e24_polyak.cpp`.

| | Deployed, 20k | Deployed, 40k | vs no averaging at 40k (paired) | Final greedy, 40k |
|---|---|---|---|---|
| No averaging (E21) | 86.3% (83.0–94.1) | 91.5% (87.7–94.1) | | |
| τ = 0.001 (~1,000-update window) | 90.2% (85.4–92.7) | 92.1% (90.9–93.1) | +0.6 points (−1.0 to +3.3), better on 4/8 | 88.2% (80.9–92.9) |
| τ = 0.0001 (~10,000 updates) | 92.5% (89.6–95.3) | **94.1% (93.0–95.5)** | **+2.7 points (+1.0 to +5.6), better on 8/8** | 93.1% (90.1–95.5) |

* **Slow averaging helps on every seed:** τ = 1e-4 raised the deployed
  policy on all 8 seeds, and the worst seed (93.0%) beats the average
  unaveraged one. It is worth most early: +6.2 points at 20k episodes.
* **The average is a usable policy by itself:** without champion
  selection, the averaged final policy scores 93.1%, close to the champion
  (94.1%). The online network's greedy policy jumps from update to update;
  the average smooths those jumps out.
* **Against the ensemble:** one averaged network (94.1%) is about as good
  as a vote of 2–3 unaveraged networks (93.1% and 95.4%, E23), for one
  training run. The 8-network vote (97.5%) is still better.
* **τ = 0.001 is too fast:** a ~1,000-update window still follows the
  online network's jumps (final greedy 88.2%).


## E25. Vote across averaged networks

* **Question:** do E23 (vote across seeds) and E24 (average each network's
  weights) add up?
* **Setup:** E22's table probe with `nn_polyak = 1e-4` (seeds 1–8, 40,000
  episodes, deployed champion compiled at 256 bins), then E23's vote over
  all 255 subsets. Probes `experiments/e22_table.cpp` (4th argument) and
  `e23_vote.cpp`. The single tables reproduce E24's deployed scores
  exactly (94.1%, 93.0–95.5).

| Members | Unaveraged networks (E23), median vote | Averaged networks, median vote | Averaged, plurality vote |
|---|---|---|---|
| 1 | 91.4% | 94.1% (93.0–95.5) | 94.1% |
| 2 | 93.1% | 94.1% (91.0–96.5) | 94.1% |
| 3 | 95.4% | 96.6% (95.4–97.8) | 95.9% |
| 4 | 96.7% | 97.1% (96.0–98.3) | 96.2% |
| 5 | 96.4% | 97.3% (96.5–98.0) | 97.2% |
| 6 | 97.4% | 97.7% (97.0–98.6) | 97.0% |
| 8 | 97.5% | **97.8%** | 96.9% |

* **They add up, with diminishing returns:** averaging lifts every
  ensemble size, most for small ones (+2.7 points alone, +1.2 at 3
  members, +0.3 at 8). Averaging removes much of the per-network noise that
  the vote also removes, so their gains overlap.
* **Best model-free result: 97.8%** of the optimal gain (8 averaged
  networks, median vote), at 3 ns per decision. Every vote of 3 or more
  averaged networks beats the best single averaged network on average.
* **Median again beats plurality** at every size above 2.
* **Practical recipe:** train 3–4 networks with `nn_polyak = 1e-4` and
  different seeds, compile each with `ExecutionPolicyTable`, deploy the
  median: 96.6–97.1% for 3–4× one network's training.


## E26. Where do the learned execution policies err? (they over-trade)

* **Setup:** each E22/E25 table's own trajectories on the 1,000 held-out
  scenarios; every decision's one-step regret V*(s) − Q*(s, a) from the
  exact DP, split by whether the table traded more or less than optimal.
  Summed along the policy's own path this is exactly its gap to the
  optimum (performance-difference identity). Probe `experiments/e26_errors.cpp`.
* **Result:** unaveraged networks agree with the optimum on 60–71% of
  decisions; trading too much costs 3.5–11.2% of the optimal gain, too
  little 0.7–5.8%. The 8-network median vote agrees on 80% and loses
  1.8% + 0.3%: it removes idiosyncratic errors, the over-trading bias
  remains. Weight averaging also reduces it (averaged: 2.8–5.1% vs
  1.0–3.1%).

## E27. A lower-quantile vote (negative)

* **Setup:** vote with the q-quantile of the members' actions instead of
  the median, to counter E26's over-trading. Held-out (77M) and fresh
  (99M) scenarios. Probe `experiments/e27_quantile.cpp`.
* **Result:** for unaveraged networks q = 0.375 scored 98.1% vs the
  median's 97.5% on the selection set, but 97.8% vs 97.9% on fresh
  scenarios; for averaged networks the median was best on both. The
  correction was fitted to the reused held-out set: selection bias, which
  motivated the sealed test seeds of E28.

## E28. Out-of-sample validation on a stress suite

* **Setup:** `validation.hpp`: the nominal market plus 11 one-change
  markets (impact ×2 / ×0.5, signal ×0.5 / ×1.5 / flipped, persistence 0.9 /
  0.3, volatility ×2, permanent impact, impact exponent 1.5, fees), each
  with its own exact optimum; 4,000 sealed episodes (900M+) per scenario,
  paired against TWAP. Probe `experiments/e28_validation.cpp`, results
  `e28_results.csv`.
* **Result:** fixed strategies work only in their assumed market: the
  nominal DP plan and the E25 ensemble keep 88–100% under moderate changes
  and lose to TWAP in nearly every episode when the signal flips (−177%,
  −173%); the model-based agent re-learned in 50 episodes keeps
  96.8–100.1% everywhere (97.3% flipped). No false alarms from the model
  check (z between −4.8 and −0.85).

## OF1–OF9. ES order-flow trading

The order-flow series (simulator stylized facts, scalping and
tick-scalping baselines and RL, stress, daily P&L, efficient market,
calibration, backtesting) is documented in [ORDERFLOW.md](ORDERFLOW.md) §4.
Caveat found in OF10: every OF2–OF9 episode ran at the activity floor of
a freshly started simulator (ORDERFLOW.md §5). Iteration 3 fixed this only
for the opt-in presets (activity model 1); the legacy results stand as
results at ~4 trades/s.

## OF10. Price discovery in seconds (plan item 1; ORDERFLOW.md §4)

* **Question:** can makers re-price fast enough that the hidden value gap
  stops predicting the next 30 s (it did at 0.57 by default and 0.64 in
  the efficient market), while book imbalance still predicts the next tick
  and the facts stay ES-like? Criteria (a)–(f): gap, imbalance, markout,
  facts, naive rules, oracle bound.
* **Setup:** probe `experiments/of10_discovery.cpp` (target
  `of_discovery`), opt-in presets, defaults bit-exact. Diagnostic simulator
  seeds 1–3 × 2 h (tuning), 4–6 × 2 h (held out), 7–12 × 6 h once (final);
  block-jackknife SEs; rules on `ScalpEnv`/`TickScalpEnv` episodes; the
  OF2/OF4 rule tables on 900M/910M. Results `of10_results.txt`,
  `of10_heldout.txt`, `of10_baselines.txt`, `of10_tick_baselines.txt`,
  `of10_calibrate_sim.txt` (v1 at `10bfe03`, v2 at `4ba0354`).
* **v1, rigid quote jumps** (`discovery_v1`): when |value − mid| > 0.65
  the whole book jumps and snipers take a clip of the stale quotes. Gap →
  next 30 s 0.11 (floor 0.09–0.10): met. Imbalance → next mid change +0.02
  (default +0.33): failed, the book moves as a block. Impact R² only via
  the sniper prints (0.80 vs 0.15 for the rest); 25.9 trades/s (3 h);
  markout flatness unresolved; the oracle never trades (the gap is capped
  at the band). Rejected.
* **v2, stale-side depletion** (`discovery`): stale levels drain lot by
  lot, nobody joins them, the price ticks when the stale best is empty;
  flow surprise moves value permanently. (a) gap → 30 s 0.111 / 0.092,
  within 0.02 of the grid floor: met. (b) imbalance → next mid change
  +0.25 / +0.24, nothing after: met on 2 h windows; +0.15 in 10-min
  episode windows. (c) markout 30 s − 5 s −0.08 ± 0.05 on both, final
  −0.015 ± 0.024: marginal. (d) trades/s 18–28 across seeds, kurtosis
  4.1–4.5 (target 5–10): failed. (e) quote both sides −6.6 to −7.2 ticks
  (target ≥ −4; default −2.8): failed. (f) passes only formally: the oracle
  at θ* = 0.70 trades 1.4–1.6 lots/episode and nets −0.5, while a 2 s
  clairvoyant makes +6.9 to +7.6. Impact of a 300 s TWAP buy is linear
  (exponent 0.99) and fully permanent (+54 ticks for 1,200 lots, default
  +1).
* **The warm-up finding** (skeptic's runs, not committed:
  `build-avx512/skeptic/s600.txt`, `s6h.txt`, `sclose.txt`): a fresh
  simulator's activity collapses to its floor (exp(−1.5)) within seconds
  and stays there ~1.5 h, because `move_avg_` starts at 0.1 tick/step with
  an 1,800 s time constant. In 600 s windows after the 150 s warm-up:
  activity 0.29, 4.2 trades/s (default) and 2.3 (v2), 1-min sd ~2.4. Over
  6 h: 21.6 and 33.1 trades/s. Every OF2–OF9 RL and rule result ran in the
  quiet regime; the OF1 facts on long windows. Present since `c808bd4`.
* **Conclusion (v2):** depletion is the right mechanism for (a) and (b),
  but the preset was tuned in the wrong regime.
* **Iteration 3** (`e229af4`; `of10_results.txt` seeds 1–3,
  `of10_heldout.txt` seeds 4–6 sanity, `of10_warmup_check.txt`; final
  check on 13–18 once by the acceptance agent, skeptic on fresh 19–21;
  every statistic in the episode regime (400 ScalpEnv-built 10-min
  windows per seed) and on 16 session-weighted sessions per seed, with an
  AGREE table between them):
  * *Warm-up root cause:* the legacy activity loop is unstable (loop gain
    0.045/s, R = 11.5 on default sessions); 10 h runs swing between the
    clamps, default episodes spend 41% of steps at the lower clamp (4.5
    trades/s). A correct `move_avg_` start cannot fix it. Fix: opt-in
    `activity_model 1` (centred log-OU, compensated news kicks, exact
    stationary law; warm start on a dedicated stream, `kStreams` = 9).
    Model 0 byte-identical; `default_warm()` = `SimParams{}` + model 1.
  * *v3 preset* (`discovery_market()`; v2 kept as `discovery_v2`):
    `maker_info 0`, retuned depletion, sniping, maker sizes and jumps.
  * (a) gap → 30 s minus floor +0.017 … +0.019 (≤ 0.02): pass. (b)
    imbalance → next mid change 0.206–0.216 (one seed 0.193 on 19–21),
    tail ≤ 0.024, leak corr(I1, g) 0.37 (≤ 0.40): marginal pass;
    corr(I1, r 30 s) ~0.07 misses the probe's own 0.05 line (not gated).
    (c) mk30 − mk5 fails only on tuning-seed episodes (+0.022 ± 0.009,
    2.4 SE), passes on 13–18, 19–21 and 4–6: chance-level per the
    skeptic. (d) all bands pass: 18.4–18.7 trades/s, 1-min sd 3.9–4.05,
    kurtosis 5.5–7.2, |r| ac1 0.19–0.22, R² 0.33–0.35.
  * (e) no naive rule nets > 0 after costs: pass. **Quote both sides
    FAILS:** −24 to −27 ticks/episode, adverse −0.17 … −0.20; ~60
    fills/episode are stale sweeps (1 s markout −0.38) vs ~25 flow fills
    (+0.30). Structural: snipers take the agent's lone stale lots before
    the price ticks (L1 queue ~108 lots); no parameter lever removed it.
  * (f1) pre-registered oracles trade and win: O1 (|g| > 0.62, latency 0)
    +24 … +30 ticks/episode on ~86–98 lots; O2 (EMO H 10 s, K 32, θ 1.24,
    decide 1 s, latency 100 ms) +5.4 … +7.2 on ~15–17.5 lots. (f2)
    untested: the RL smoke run collapsed to flat.
  * **AGREE fails on active meta-orders:** 10.46–10.59 in episodes vs
    11.07–11.18 on long windows (5–6 SE, all three seed sets); a 1,050 s
    burn gives 11.13. Activity is warm after 150 s, the meta-order
    population is not.
  * *Negative and side results:* fair value is not a martingale (10 s
    value change → next 10 s +0.013 ± 0.003, permanent-impact momentum;
    delta 5 s grosses +0.235/lot and loses only on costs). Own impact is
    linear and permanent: round trips of ≥ 30 lots profit from it, so
    `ScalpEnv` refuses `contracts > 1` with `value_impact > 0`. On
    `default_warm` naive rules profit strongly (OF2 delta 5 s > 1.5
    +15.19, t 22.8; imbalance L1–5 +7.31, t 10.9; quote both +1.3): the
    slow-discovery leak is real at realistic activity. Legacy OF6 stresses
    are partly empty on discovery (fade ×2 no-op, vol ×1.5 weak, informed
    ×0.5 ~4% of volume).
* **Conclusion (iteration 3):** accepted for market-order RL (OF3, OF6,
  OF8) with a 1,050 s warm-up; not a calibrated market for tick
  environments (A62–A66).
* **Pre-RL check** (`5bb7be3`, `experiments/of11_probe1050.txt`, fresh
  seeds 22–24, `--scalp-warmup 1050`): episode and long regimes agree on
  29/29 rows (active meta-orders 11.15 vs 11.18); (a)–(d) pass; (c) +0.000
  ± 0.008. Still outside the gate in both regimes: corr(I1, r 30 s) 0.076
  (probe line 0.05) and event corr(I5) 0.080. The held-out runs are now
  committed: `of10_final13_18.txt` (13–18) and `of10_skeptic19_21.txt`
  (19–21), reproduced from committed code by `of11_regen.sh`.

## OF11. RL reruns on the fixed market (OF10 v3, warm-up 1,050 s)

`bash experiments/of11_regen.sh` (`1a73bd6`; ~40 min), all on
`--market discovery --warmup 1050`:

* **OF3 recipe** (`of11_train.txt`, of_train 24 seeds × 2,000 episodes,
  NeuralQ + Double DQN, champion on validation, 300 sealed episodes): every
  seed negative, −0.15 … −1.50 ticks/episode (mean −0.54, t −1.0 … −5.0,
  0.4–3.3 orders); validation highs (+0.74) did not carry over. The mean-Q
  ensemble makes 0.0 orders. 30 s delta rule −1.39 (t −2.0).
* **OF2 rules** (`of11_baselines.txt`): none > 0 by 2 SE (best delta
  120 s > 2, +0.38, t 0.75); "oracle (sees fair value)" −0.21 on 0.3 lots.
* **OF6 stresses** (`of11_stress.txt`, the discovery list pre-registered
  in `5bb7be3`, nets seed1/seed2): all strategies ≤ 0 in all 11 scenarios;
  the nets lose less than the rule (best: seed1 on thinner book +0.007,
  t 0.03). Worst: session open (rule −5.7, nets −1.9 / −3.1).
* **OF8 days** (`of11_days.txt`, 20 days × 39 sessions): rule −88
  ticks/day (Sharpe −16.7), nets −9.1 / −24.8, ensemble −3.3 (55% winning
  days, Sharpe −3.1).
* **(f):** no learned policy comes near the pre-registered references (O2
  +5.4 … +7.7 ticks/episode on the probe's episodes; O1 +24 … +30; not
  paired, and both read the hidden state). No red flag; no edge, so the
  950M+ confirmation block stays unused.
* **Conclusion:** once prices are discovered in under a second at
  realistic activity, the OF3 recipe finds nothing after costs. The OF3 /
  OF8 profits came from the slow-discovery leak, which is real at realistic
  activity too (`default_warm`: naive delta 5 s > 1.5 +15.2 ticks/episode).
  The pre-registered gate (f) holds non-vacuously.

## OF12. Real ES vs the simulator, like for like (the user's Sierra Chart data)

`bash experiments/of12_regen.sh` (~17 s; `experiments/of12_real_vs_sim.txt`,
`tools/real_vs_sim.py`). One Python tool computes every statistic from
three data shapes (1-min bars, 1-s bundles, an L2 book stream) and applies
the same functions to the real data and to simulator sessions converted to
those shapes (12 × 6.5 h sessions per market from the open, seeds
960,012,001–012; `of_calibrate export` gained an opt-in market, seed and
start). Real data (read-only, outside the repo; only aggregates are
committed): 224 RTH days of 1-min bars with aggressor volume (Dec 2025 –
Nov 2026), Sierra `.scid` 1-s bundles on the front month, and Sierra
`.depth` L2 with only **43.9 min of RTH book** (Nov 3, 08:30–09:14 CT;
treat that column as one observation) plus ~15.7 h overnight.

* **v3 matches:** imbalance → next mid change (event corr(I1) +0.241
  real vs +0.238; direction right 61% vs 58%), R² of price change on delta
  (1 min 0.359 vs 0.352; 10 s 0.334 vs 0.300 without the outlier session),
  no 1-min delta predictability on either side, spread = 1 tick (95% vs
  100%).
* **Biggest mismatches** (ranked in the results file, with the parameter
  that most plausibly drives each):
  1. **Book depth:** real L1 17 ± 1 lots (median 15, ~1.2 lots per resting
     order, depth *rising* to ~36 at L5) vs v3 143 (falling with
     distance). v3's stationary L1 = maker_rate × size / cancel_rate = 240.
  2. **v3 flow predicts the mid after the fact:** corr(1-s delta, next 1 s
     mid move) −0.035 real vs **+0.151** v3 (+5 s markout −0.05 vs +0.13
     ticks); real ES moves within the same second (+0.66 vs +0.23 ticks).
     In v3 impact goes to fair value first and reaches the mid only as the
     stale queue depletes: a short-horizon leak, and the same mechanism as
     the stale sweeps.
  3. **1-s flow memory:** delta-sign ac at 1 s +0.017 real vs +0.236 v3
     (order splitting and momentum takers).
  4. **Volume:** v3 trades ~1.55× real ES's lots per second. Trade counts and
     lots/trade are *not* comparable: Sierra's count looks like one trade
     per resting order filled (1.31 lots/trade ≈ 1.2 lots per resting
     order), the simulator's one per aggressive order.
  5. **Volatility clustering:** ac30 of |1-min returns| +0.20 real vs 0.00
     (v3's activity memory is ~10 min).
  6. **Mid flicker:** real mid changes 168/min at 100 ms sampling vs 23
     (thin real L1 empties and refills within the second).
  7. **1-min volatility:** 9.3 ticks real vs 5.0 (4.2 without the outlier).
* **Simulator pathology found:** in one v3 session (seed 960,012,012,
  minute 314) a news jump empties the ask and takers print 1 lot on each
  of 151 ticks in one 100 ms step, then the price falls back within a
  minute. It drives v3's pooled 1-min kurtosis to 233 and inflates several
  v3 SEs. It stays in the table, with a per-session note.
* **Conclusion:** v3's imbalance signal and price-on-flow R² look like real
  ES, but its book is ~8× too deep and its mid lags its flow by a second or
  more, which real ES doesn't do. The next simulator iteration should
  target book depth and same-second impact; more RTH depth recording
  (Sierra market-depth recording on during RTH, tick-by-tick storage) is
  needed before fitting book parameters.

## OF13. Simulator v4 (`discovery_v4`), matched to real ES

`bash experiments/of13_regen.sh` (commit `31f51c4`; `tools/real_vs_sim.py
--of13`): real ES vs v4 with v3 as context (`of13_real_vs_v4.txt`; Table 4
holds the gates G1–G8, 25 rows, pre-registered before v4 was built), then
the efficiency probe (`of_discovery`, both regimes) and the naive rules
(`of2_baselines`) on v4 at warm-up 150 s and 1,050 s
(`of13_probe_w*.txt`, `of13_baselines_w*.txt`). `SEED0=… DIAG=… OUT=…
PROBE_PREFIX=…` select another block: final `of13_final_*` (sessions
960,013,001–012, diagnostic seeds 28–33, baselines episodes 960,013,000,000
+ i; run once), skeptic `of13_skeptic_*` (960,014,001–012, seeds 34–36,
960,014,000,000 + i). Tuning: sessions 960,012,001–012 (OF12's), seeds
25–27; its (e) ran on `of2_baselines`' sealed 900M episodes.

* **Design** (two designs, judged; book-first won): v3's activity
  process kept; new opt-in `SimParams` fields, every default off and
  checked byte-identical on every preset: `touch_cancel_mult` (thin,
  fast-turnover L1), `cancel_activity` (per-lot cancel hazard, depth
  independent of activity), `sweep_follow` (a sweep that empties the best
  level moves the hidden value to the new mid at once), `stale_binomial`
  (stale lots leave independently; Poisson residue leaked I1),
  `max_stale_levels` 3 and `max_step_ticks` 3 (bounded per-step moves; v3
  had a 151-tick step). Flat depth behind the touch (~1.2-lot orders),
  `join_ahead_mean` rescaled. **No meta-orders, no informed or momentum
  takers, `value_impact` 0.** v3 (`discovery`) frozen: golden test, and
  `of12_regen.sh` reproduces `of12_real_vs_sim.txt` byte for byte (A67).
* **Gates G1–G8 vs real ES** (tuning / final / skeptic; band):

  | Row | Tuning | Final | Skeptic | Band |
  |---|---|---|---|---|
  | G1 L1 mean, w / f (lots) | 20.2 / 20.9 | 20.2 / 21.2 | 20.6 / 21.3 | [10, 28] |
  | G1 L5/L1, w / f | 1.49 / 1.45 | 1.47 / 1.43 | 1.47 / 1.43 | ≥ 1.2 |
  | G2 corr(1-s delta, next 1 s), w / f | −0.022 / −0.019 | −0.033 / −0.027 | −0.029 / −0.027 | [−0.08, 0.04] |
  | G2 same-second move, w / f (ticks) | 0.594 / 0.536 | 0.589 / 0.512 | 0.568 / 0.508 | ≥ 0.45 |
  | G4 event corr(I1), w / f | 0.267 / 0.245 | 0.256 / 0.235 | 0.254 / 0.234 | [0.18, 0.32] |
  | G5 R² 1 min | 0.346 | 0.300 | 0.346 | [0.25, 0.50] |
  | G6 volume vs real RTH | 1.03 | 0.92 | 0.87 | [0.7, 1.4] |
  | G7 1-min sd (ticks; real 9.3) | 8.68 | 7.99 | 7.48 | [6.5, 12.5] |
  | G7 100 ms steps > 8 ticks (largest) | 0 (6) | 0 (6) | 0 (5) | 0 |
  | **G rows passed** | **25 / 25** | **25 / 25** | **25 / 25** | |

  (w = the matched first 44 min, f = whole sessions; every row in
  Table 4.) Soft targets: mid changes/min met; ac30 of |1-min r| missed
  in all blocks (no volatility clustering); 08:30–09:00 volume share met
  only on final.
* **Efficiency gates at warm-up 1,050 s** (A68, A69), all three blocks:
  (a) corr(g, r 30 s) − floor +0.002 … +0.003 (≤ 0.02); (b) leak
  corr(I1, g) 0.25 (≤ 0.40); (c) mk30 − mk5 within 2 SE of 0; AGREE 0 of
  29 rows disagree; (e) best naive rule −1.73 (t −1.46) tuning, −1.16
  (t −1.05) final, −0.49 (t −0.45) skeptic; (f1) O1/O2 trade 23 / 40–42
  lots/episode. The probe's legacy (d) bands, its combined (f1) line
  (which also wants O2 > 0) and `TickScalpEnv`'s (f) print FAIL; none
  judges v4.
* **Both oracles lose after costs:** O1 −8.23 ± 1.33 / −11.97 ± 1.06 /
  −11.11 ± 1.44, O2 −19.78 ± 2.50 / −18.33 ± 1.95 / −18.82 ± 3.10
  ticks/episode (tuning / final / skeptic). v4's fair value carries no
  edge after costs, so "RL finds no edge on v4" is close to true by
  construction for a 1-lot market-order scalper; v4 tests whether RL
  overfits the remaining small signals, not whether order flow carries
  information (LESSONS 48). Quote-both −198.9 / −199.9 / −203.8
  (~310 stale sweeps per episode at 1 s markout −0.45): A64 stays, and
  `TickScalpEnv` refuses v4.
* **Skeptic caveats** (ranked): (1) I1 predicts the next 1 s after the
  100 ms latency (+0.080 sim vs +0.002 / −0.008 real); the top decile
  gains +0.19 ticks/decision, below the 1.24-tick round trip, but can
  time entries; (2) a regression fitted on the sim keeps a top-2% edge on
  fresh sim seeds that does not transfer to real; (3) the 100 ms mid
  flicker reverts (−0.22 vs −0.02 real), giving small fade edges per lot
  (fade return 30 s +0.23); (4) the spread never widens (1 tick
  99.99–100% vs 92.7–94.7% real); (5) the real G2/G4 reference is one calm
  morning (Nov 3): a second, busy RTH morning (Nov 5) falls outside G4
  (event corr 0.149) and G2 (+5 s markout +0.139) itself; (6) margins
  shrink across blocks (1-min sd 8.68 → 7.99 → 7.48, volume 1.03 → 0.92
  → 0.87; spread between blocks ~3× the jackknife SE); (7) the 150 s
  warm-up is fragile (one AGREE miss in 2 of 3 blocks), 1,050 s
  confirmed; (8) "delta 30 s > 2.5" sometimes shows +0.45 per lot (watch
  item); (9) no volatility clustering, and G7's step bound holds by
  construction.
* **Smoke:** `of_train --market discovery_v4 --warmup 1050 --seeds 4
  --episodes 300`: sealed test −1.91 … −3.55 ticks/episode, mean-Q
  ensemble −0.48 (t −1.1).
* **Live training viewer** (merged `d0b672f`, `39a7a06`): `of_train_live`
  + `tools/of_train_live.py` (ORDERFLOW.md §3) runs alongside every RL
  run, probing on the 960M+ visualization block only (A70).

## OF14. RL reruns on simulator v4 (warm-up 1,050 s)

`bash experiments/of14_regen.sh` (`6d678c6`; ~20 min), the OF11 recipe and
seed blocks on `--market discovery_v4 --warmup 1050`:

* **OF3 recipe** (`of14_train.txt`, 24 seeds × 2,000 episodes, 300 sealed
  episodes): every seed negative, −0.75 … −6.29 ticks/episode (mean −3.18,
  t −1.4 … −9.2, 3.3–10 orders); validation highs (+1.15) did not carry
  over. The mean-Q ensemble makes 0 orders on the sealed episodes. 30 s
  delta rule −4.69 (t −3.2). On v4 the agents trade more than on v3
  (OF11: 0.4–3.3 orders, mean −0.54) and lose more.
* **OF8 days** (`of14_days.txt`, 20 days): rule −236 ticks/day, nets −164
  and −116 (0% and 10% winning days), ensemble −50 (Sharpe −14.7).
* Not run: OF6 (no stress list was pre-registered for v4); the OF2 rules at
  1,050 s are `of13_baselines_w1050.txt`.
* **Leak check:** an agent exploiting the skeptic's caveats (I1 persisting
  after latency, 100 ms mid flicker reverting) would show up as a profit on
  sealed episodes; none does, so no audit was triggered. 950M+ stays unused.
* **Watched live:** seed 1 trained under `of_train_live` alongside (same
  market, warm-up and recipe; probes 960M+), served in the browser pane;
  the OF14 nets were replayed with `of_trace` → `of_viz.py`.
* **Conclusion:** on a market matched to real ES at 1 s and below, the OF3
  recipe loses after costs, and more than on v3: the signals it learns in
  training (validation up to +1.15) don't survive sealed episodes. Combined
  with both oracles losing on v4 (OF13), a 1-lot market-order scalper at
  1 s decisions has nothing to find here; the next evidence has to come
  from real data (backtests on recorded ticks).


## OF15. The agent zoo through a leak-proof learning pipeline (v4 and v3)

`bash experiments/of15_regen.sh [agent ...]` (v4; `OF15_MARKET=v3` for
v3): builds with `-DRAG_PYTHON=ON` (warning-free), copies `_ragof` and the
programs to `build-avx512/runs/of15-<commit>/`, then `python -m ragrl run
experiments/configs/of15_<agent>_<v4|v3>.json` per agent; each run's
`report_results.md` (no timings or hashes) is copied to
`experiments/of15_<agent>[_v3].txt`. The write-up with the runtime
profile, crash history and audits is `of15_results.txt`. Runs and the
registry: `build-avx512/rl/runs/` (`index.jsonl`); sealed openings:
`build-avx512/rl/ledger_usage.jsonl`.

* **Question:** with a proper pipeline (leak-proof by construction, paired,
  with CIs and red flags), does any of eight agent families find an edge
  on v4 (`discovery_v4`) or v3 (`discovery`) where the C++ NeuralQ did not
  (OF11, OF14)?
* **Pipeline** (Design 2, rigor-first, judged over a faster Design 1;
  A71–A72): header-only C++ `seed_ledger.hpp` (the only seed-block table;
  a process-wide role guard: a process that built a train, validation or
  pairing env can't open a sealed block, a scorer can't train,
  `forbid_sealed()` is the first call of every worker; the privileged
  channel only on train and viz) and `vec_scalp_env.hpp` (`Pool`,
  `EpisodeSource`, `VecScalpEnv`, `VecReplayEnv`, `run_builtin`),
  bit-identical to serial `ScalpEnv` on the train stream and validation;
  the nanobind module `_ragof`; the Python package `python/ragrl` (runner,
  zoo registry, paired evaluation, bootstrap CIs, red flags, run registry,
  Optuna HPO, live-viewer hook). Seal is write-once, runs in a fresh
  process, refuses ranges viewed by another experiment (900M+[0,100000)
  by OF2–OF14) and logs every opening.
* **Zoo:** `neuralq_ref` (the C++ NeuralQ nets of OF14 for v4 and OF11
  for v3, loaded, not retrained), SB3 PPO, PPO with an 8-frame stack
  (`ppo_fs`), recurrent PPO (`rppo`), DQN, QR-DQN, a LightGBM threshold
  policy (`lgbm`) and behaviour cloning + DAgger of the privileged teacher
  (`dagger`). Venv `.venv` (gitignored): torch 2.15.0+cpu, SB3 and
  sb3-contrib 2.9.2, Optuna 5.0.0, LightGBM 4.7.0, nanobind 3.1.0,
  gymnasium 1.4.0.
* **Setup:** ES, 1 contract, market orders, fee $1.5, latency 1 step, 1 s
  decisions, 600 decisions/episode, warm-up 1,050 s (A63, A69). 12 run
  seeds × 2,000 episodes (1.2M decisions, `of_train`'s budget) per agent.
  Champion per seed on validation 700M+[0,200) every 60k decisions
  (`neuralq_ref`: `of_train`'s own champion on [0,40), re-scored on
  [40,240)); comparisons on 701M+[0,400). **Headline declared before
  training: the 12-seed ensemble** ("ens", A73); "mos" = mean of the 12
  seeds' per-episode P&L. One sealed scoring per config, all policies
  paired on the same 300 episodes: v4 test 901M+[0,300) (family
  `of15_v4`, 8 of 9 slots used; the HPO slot is unused), v3 test
  901M+[300,600) (family `of15_v3`, 8 of 8: full). 95% bootstrap CIs over
  episodes (B = 10,000). Module `_ragof` 9e3731fb for every row.
* **v4** (ticks/episode net of fees; 300 sealed episodes):

  | agent | val ens / mos | sealed ens [95% CI] | sealed mos [95% CI] | orders/ep ens / mos | ens − NeuralQ ens [CI] |
  |---|---|---|---|---|---|
  | neuralq_ref | −0.12 / −3.50 | −0.29 [−0.65, −0.04] | −3.80 [−4.29, −3.33] | 0.17 / 6.90 | – |
  | ppo | −1.23 / −1.27 | −1.10 [−3.72, +1.51] | −0.29 [−1.90, +1.34] | 2.00 / 2.17 | −0.82 [−3.45, +1.82] |
  | ppo_fs | −0.14 / −2.34 | −0.13 [−0.32, +0.05] | −2.66 [−3.17, −2.16] | 0.26 / 4.41 | +0.16 [−0.16, +0.55] |
  | rppo | +0.00 / −0.00 | +0.00 (flat) | −0.00 [−0.00, +0.00] | 0.00 / 0.00 | +0.29 [+0.04, +0.65] |
  | dqn | +0.00 / −0.20 | +0.00 (flat) | −0.10 [−0.19, −0.02] | 0.00 / 0.29 | +0.29 [+0.04, +0.65] |
  | qrdqn | +0.00 / −0.32 | +0.00 (flat) | −0.21 [−0.46, +0.02] | 0.00 / 0.45 | +0.29 [+0.04, +0.65] |
  | lgbm | +0.09 / −0.43 | +0.00 (flat) | −0.25 [−0.55, +0.02] | 0.00 / 0.34 | +0.29 [+0.04, +0.65] |
  | dagger | +0.00 / +0.01 | +0.00 (flat) | +0.01 [+0.00, +0.04] | 0.00 / 0.00 | +0.29 [+0.04, +0.65] |

* **v3:**

  | agent | val ens / mos | sealed ens [95% CI] | sealed mos [95% CI] | orders/ep ens / mos | ens − NeuralQ ens [CI] |
  |---|---|---|---|---|---|
  | neuralq_ref | +0.01 / −0.49 | +0.00 [+0.00, +0.01] | −0.60 [−0.72, −0.47] | 0.01 / 1.22 | – |
  | ppo | −1.17 / −1.27 | −0.60 [−2.04, +0.84] | −1.09 [−2.08, −0.15] | 2.38 / 2.69 | −0.60 [−2.04, +0.84] |
  | ppo_fs | −0.19 / −1.03 | −0.34 [−0.65, −0.06] | −0.83 [−1.05, −0.60] | 0.64 / 2.36 | −0.35 [−0.65, −0.07] |
  | rppo | +0.00 / −0.00 | +0.00 (flat) | −0.00 [−0.01, −0.00] | 0.00 / 0.00 | −0.00 [−0.01, +0.00] |
  | dqn | +0.00 / −0.04 | +0.00 (flat) | +0.02 [−0.04, +0.08] | 0.00 / 0.12 | −0.00 [−0.01, +0.00] |
  | qrdqn | +0.00 / −0.04 | +0.00 (flat) | −0.07 [−0.16, +0.01] | 0.00 / 0.30 | −0.00 [−0.01, +0.00] |
  | lgbm | −0.02 / −0.12 | +0.10 [+0.00, +0.29] | −0.15 [−0.33, +0.04] | 0.01 / 0.27 | +0.10 [−0.01, +0.29] |
  | dagger | +0.14 / +0.13 | −0.15 [−0.38, +0.05] | −0.23 [−0.48, −0.00] | 0.10 / 0.12 | −0.15 [−0.38, +0.04] |

  Every agent is "better" than the 30 s delta rule (+4.6 … +5.8 on v4,
  +1.3 … +2.0 on v3) only because the rule pays fees on ~6 orders/episode
  (−5.75 v4, −1.88 v3). The +0.29 vs NeuralQ on v4 is the NeuralQ
  ensemble's own loss. lgbm v3's +0.10 is a handful of trades in 300
  episodes (0.01 orders/episode); dagger v3's validation +0.14 turned
  into −0.15 on test.
* **Status:** all 16 configs NO_EDGE, no hard red flag. Warnings: R9
  (headline t far below t_req ≈ 5.1–5.2) and R11 (days/stress not run)
  everywhere; R6 (the best fair-value oracle trades 0.0–0.2
  orders/episode, so the oracle bound says nothing); R2w; R1v; R4 on qrdqn
  v3 seed 1 only (validation +0.12, t 2.4 → test −0.19). Nothing went to
  950M+.
* **Skeptic** (fresh viz episodes, look-only; nothing sealed): 600
  episodes at 960,015,000,000 + i per market, under base, latency 300 ms,
  fees $3 and both: ppo v4 ens −0.18 ± 1.06 base, −1.93 both; ppo v4 mos
  −0.75 ± 0.70; lgbm v3 ens (best sealed, +0.10) −0.00 ± 0.03; dagger v3
  ens +0.00; neuralq_ref v4 mos −3.91 (t −23.6). The ranking held in every
  condition; no seed is positive with t > 1.6 (best: qrdqn v4 seed 9,
  +0.45, t 1.2, gone at 300 ms). Quirk probe (400 episodes at
  960,015,001,000 + i): **PPO v4 does not scalp:** it enters once in the
  first second and holds ~597 s, a 10-minute directional coin flip
  (per-episode sd 26 ticks). NeuralQ, ppo v3, lgbm and ppo_fs enter with
  the I1 imbalance sign 60–80% of the time: +0.1 … +0.3 ticks at the 1 s
  markout, but ~0.55 ticks entry slippage, so gross per trade is negative.
  ppo_fs seeds 3, 8, 9 and 12 fade the 30 s return on 88–98% of entries
  and lose 2–5 ticks/episode. Pipeline checks held: privileged refusals on
  validation, pairing and sealed blocks; configs written before the first
  test opening (the v3 configs differ from v4 only in market, test range,
  family and reference nets); auto-reset and terminal semantics; the
  sealed table matches `sealed.json` and `sealed_episodes.npz` for all 16
  runs. One unintended `open_sealed('test')` call, in a process that had
  built a validation env, was refused by the guard; nothing was opened or
  logged.
* **Caveats:** (1) **The ensemble headline mostly measures "flat":** 5 of
  8 v4 and 4 of 8 v3 ensembles place 0 orders on test, so their NO_EDGE is
  true by construction; mos is the informative number, and it is ≤ 0
  everywhere on the fresh episodes (LESSONS 50). (2) **R1 → R1v was
  changed after seeing test results:** a non-oracle row with annualized
  Sharpe > 5 that trades < 2 orders/episode (`MIN_ORDERS` in
  `redflags.py`) is now the warning R1v. It only relaxed flags (7 runs
  went RED_FLAGS → NO_EDGE, all on rows trading 0.0–0.26 orders/episode)
  and no claim depends on it (A74). R1 is also miscalibrated: annualized
  per-episode Sharpe > 5 ≈ t > 0.87 at 300 episodes, so a t-based rule is
  the next step (LESSONS 51). `index.jsonl` is append-only and still lists
  RED_FLAGS for those 7 runs; their STATUS files and reports say NO_EDGE.
  (3) **Pairing tightens less than assumed:** the simulator reacts to our
  orders, so paths diverge after the first fill (PPO v4 P&L at 100 vs
  300 ms latency on the same episodes correlates 0.56, delta30 0.19;
  LESSONS 52). (4) **Heap crash 0xc0000374, not root-caused:** 5 worker
  crashes on 2026-11-05 (~0.15% of evaluations), all at `run_policy`
  teardown; each crashed run was retried whole, nothing reused. On
  2026-11-12 ~1,450 LightGBM evaluations under per-step `HeapValidate` and
  an ASan build of `_ragof` gave 0 crashes (P ≈ 0.2% at the old rate).
  Leading unconfirmed hypothesis: Python's `_wmi.exec_query` timeout path
  (WMI broke on this server ~2026-11-07; `ragrl/__init__.py` now bypasses
  `platform._wmi_query`, A76). Sealed results judged unaffected: crashes
  produce no output; seed 1 of every run retrained bit-identical; all 16
  runs' validation selections replay bit-identical (701M+[0,400)); the
  baselines are bit-identical across seal processes. Residual risk: silent
  damage in a seal process can't be excluded without reopening the test.
  Settling it needs WMI repaired (the user's call) and ~150 lgbm
  seed-runs with the bypass off and on. One rppo v4 run stopped after
  ~5 min with no worker crash; the cause is unrecorded because the retry
  overwrote the runner's log (LESSONS 53). (5) **Code versions:** 13
  configs trained under ragrl 3aa6572a, dqn/rppo/qrdqn v3 under 6aa7b15a
  (WMI bypass + R1v; same training code); qrdqn v4 was sealed with
  `--allow-env-change`, recorded in its `sealed.json` `env_problems`. Run
  records show git `e85dfe7`, which lacks the pipeline: until the OF15
  commit only `ragrl_sha256` and each run's `bin/` module copy identify
  the code. (6) Optuna was not exercised at acceptance scale; nothing ran
  on real ticks.
* **Where training time goes** (the GPU question; 10% profile, 12
  workers): PPO, frame-stack PPO and recurrent PPO spend 61–80% of
  training in the C++ simulator on v4 (~90% counting validation
  checkpoints); DQN ~60% network, QR-DQN 88–93% (2 gradient steps of batch
  256 × 32 quantiles per env step). Simulator ~9,000 decisions/s on v4
  with 12 in parallel (11,200 alone), 24,700/s on v3 alone. A full market
  takes ~2.0 h (v4) / 1.6 h (v3) of wall time; QR-DQN alone ~35 min. The
  networks are 64×64 MLPs, so a GPU is not worth renting now (A75).
* **Viewers:** every run writes `RUN/live.jsonl` for seed 1 in the
  `of_train_live` schema (baked replays `build-avx512/live/of15_*`);
  `tools/of_zoo_trace.py` replays any run's policy on viz episodes
  960M+[0,5) for `of_viz.py` (look only).
* **Conclusion:** on both simulators no agent family finds an edge after
  costs. What they learn is to stay flat; the ones that trade lose (PPO
  holds a 10-minute coin flip; I1 timing gains less than the slippage;
  fading loses). Consistent with OF11–OF14. The next evidence has to come
  from real ticks (`VecReplayEnv` exists) or a more realistic simulator.

## OF16 phase 1. Real ES plumbing and dev-day diagnostics (no score)

`kit.ps1 run of16_regen 'bash experiments/of16_regen.sh'`: exports the
dev days with `tools/sierra_to_ticks.py` (Sierra `.depth` + `.scid` →
`tick_csv` T/Q lines, 08:30-15:00 CT, sidecar JSON with checks and the
csv sha256) into `build-avx512/real/` (gitignored), then runs
`experiments/of16_dev_diag.py` → `experiments/of16_dev_diagnostics.txt`.
About 2.5 min single-threaded; the rerun through the kit reproduced the
file except its wall-time line. The plan, day manifest and decision rule
are in `OF16_REAL_ES_PLAN.md` (pre-registered, commit 121d48d; amended
before any test opening: warm-up 1050 s).

* **Built:** the converter; `experiments/configs/of16_real_manifest.json`
  (day roles); `python/ragrl/real.py` (manifest, CT clock with the explicit
  US DST rule since this machine has no tz database, tapes opened by role:
  dev free and logged as `real_dev`, test only in a fresh scorer process,
  once per config sha256 whatever the run name or manifest, confirm only
  with the orchestrator claim and only the next due days); `ragrl
  score-real RUN --days test|dev` (`score_real.py`: the run's ensemble and
  12 seeds plus flat, delta30 and an L1-imbalance rule on `VecReplayEnv`,
  paired stats, day-clustered bootstrap, Sep/Nov subsets, stress; every
  policy is stepped on a dummy batch before the gate opens and each block
  is written as it finishes); 13 tests in `python/tests/test_real.py`.
* **Review** (independent reviewer, before any test opening) found the
  replay warming features up for 150 s while every OF15 agent trained
  with 1050 s (features ~2x off scale at window start: a false "no edge"
  by construction), a once-per-run check keyed on the run name (bypassed
  by renaming), test tapes loadable after any dev opening, and a crash
  after opening burning the only opening. All fixed (A78); the plan's
  window count went from ~301 to 292.
* **Dev days** (diagnostics, not a score): 2026-09-14..17 (ESU26, the
  expiring back month after the roll: thinner, 1-tick spread 86.8% of
  buckets vs 94.7% front month, 1-min mid SD 7.2 vs 9.3 ticks) and the one
  usable window of 2026-11-03 (its book has a 610 s recording outage);
  149 windows.
  * **Covariate shift** vs v4 (viz 960,016,000,000+[0,200)): only
    `imb_l5` (9.3% of real values outside v4's 1-99% range) and `absorb`
    (6.2%; v4's range is exactly 0) exceed 5%. Setting them to the sim
    median changes any ensemble's action on <= 0.8% of decisions.
  * **Baselines** (ticks/window): flat 0 exactly; hold −2.5 ± 2.4; delta30
    −4.1 ± 2.2 (v4: −5.12); the imbalance rule (enter with the L1
    imbalance, flat at the next decision) −381.7 at 586 orders/window.
  * **I1 markout** in the imbalance's direction: real +0.022 / +0.072 /
    +0.090 ticks at 100 ms / 1 s / 5 s (strongest quintile +0.126 at 1 s),
    v4 +0.023 / +0.069 / +0.069, against a 1.37-tick real round trip
    (half-spread 0.57 + fees). v4 reproduces the real signal closely; on
    real ES it is about a tenth of the cost of trading it.
  * PPO ensembles (v3 and v4) enter on 100% of flat-state real decisions;
    the other ensembles on 0-0.4%.
* **Next:** phase 2 (score the 16 OF15 runs once on the 8 test days)
  needs the user's go. Expectation from the dev diagnostics: no edge (the
  one signal the agents use is a tenth of the round-trip cost on real ES).
  (Done: OF16 phase 2 below.)

## OF16 phase 2. The 16 OF15 runs on the real ES test days: no edge

`kit.ps1 run of16_phase2 'bash experiments/of16_phase2.sh test'`
(2026-11-12): exported the 8 test days (the orchestrator's flag), then ran
`ragrl score-real RUN --days test --stress` once per sealed OF15 run, each
in a fresh process, single-threaded, ~23 min in all. The same script with
`dev` ran first as a rehearsal on the dev days (all 16 runs, no failures).
Results: `experiments/of16_real_results.txt`, regenerated by
`experiments/of16_summary.py --days test` from `RUN/real/test-*/results.json`
(the opening itself cannot be repeated); a page at
`build-avx512/real/of16_viz/index.html` (`of16-real` in `.claude/launch.json`).

* 292 windows: 2026-09-08..11 (ESU26) and 11-05, 11-09, 11-10, 11-11
  (ESZ26), none excluded for book gaps. Warm-up 1050 s, 600 s windows,
  100 ms latency, $1.50 per side; the stress block (300 ms, $3 per side)
  in the same opening.
* **No headline meets the edge rule** (32 headlines, t_req 4.63). The best,
  lgbm_v4's ensemble, nets +0.40 ± 0.32 ticks/window, t 1.2 (day-clustered
  CI [+0.09, +0.80], Sep +0.24, Nov +0.56), from 6 round trips in 292
  windows, two of them +65 and +55 ticks: noise. The stress block has the
  same top (t 1.2). **Decision (plan §4): no edge on real ES.** The
  simulator-trained RL line is closed; phase 2b does not run.
* **Baselines** (ticks/window): flat 0; delta30 −2.71 ± 1.93 (Sep −3.82,
  Nov −1.57); the L1-imbalance rule −362.0 at 593.5 orders/window, gross
  −290.8, i.e. −0.49 ticks per order before fees: the +0.07-tick signal
  never pays the spread.
* **Transfer:** the agents behave on real ES as they did in their
  simulator. Across the 32 headlines, orders/window in sim and on real
  correlate 0.994 and net 0.88. The 9 headlines that trade more than 0.5
  orders/window lose −1.60 ticks/window on real against −1.24 in their
  simulator. Ensembles are flat or nearly (0-0.2 orders/window; ppo_v3/v4
  ~2); the mean of seeds loses wherever the seeds trade (neuralq_ref_v4
  −4.44, t −11.2; ppo_fs_v4 −2.37; ppo_fs_v3 −1.71).
* **Red flags:** 6 hard RG (3 policies × base and stress) and 2 R1v
  warnings (ppo_v3 seed10: Sharpe 7.8 as a point estimate, lower bound −3.9).
  The RG flags are not leaks. neuralq_ref_v3 seed11's real gross is +0.08
  (t 0.3) against −0.55 in v3; neuralq_ref_v4's ensemble +0.07 (t 1.2, 14
  of 292 windows traded) against −0.27; rppo_v3's mean of seeds +0.001 (one
  trade) against −0.003. Each fires because its simulator gross was
  negative and the real one is about 0. None has real gross above 0 by 2 SE,
  and 3 of the 224 learned policies at a one-sided 2 SE is chance level
  (~5 expected). The phase 1 no-lookahead test holds, and the imbalance
  rule pays its full spread, so fills do not see the decision bucket. The
  status files keep RED_FLAGS (not edited after the fact); the rule's flaw
  is LESSONS 56. A80 then tightened RG for later openings (hard only when
  the real gross is also > 0 by 2 SE on >= 2 orders/window); re-applied to
  these results it gives 6 RGv warnings and no hard flag.
* **Ledger:** 16 `real` test openings, one per OF15 config sha256
  (`build-avx512/rl/ledger_usage.jsonl`). The 8 test days have now been
  seen: a later real-data claim is scored on the confirm block (days from
  2026-11-12), never on them (A79).
* **Next:** plan §4 leaves phase 3 (training on real, walk-forward by day)
  or simulator realism. Phase 1's markout argues against phase 3 for
  directional 1 s market-order policies: the one signal is a tenth of the
  round trip. A different action space (passive orders, longer holds)
  would be a new question with its own pre-registration.

## OF17. A signal census of the agents' features on real ES: nothing to trade

Plan pre-registered in `OF17_SIGNAL_CENSUS_PLAN.md` (commit dea264c, before
any census number). `kit.ps1 run of17_census '... experiments/of17_signal_census.py
--out experiments/of17_signal_census.txt'` (2026-11-12, ~2 min, one thread;
a rerun by the reviewer reproduced the file byte for byte). Data: the 8
front-month days OF16 phase 2 scored, now manifest role `seen` (A81),
opened as `real_seen`; 175,200 samples, OF16's decisions exactly (1050 s
warm-up per 600 s window, fill one bucket after the decision; the imb_l1
alignment check against the tape is exact, 0). Each sample is a 1-lot
market-order round trip held 1, 5, 15, 30, 60, 120, 300 or 600 s at the
recorded touch plus $1.50 per side; rules are fitted on one month (Sep or
Nov) and scored on the other; 128 tests (13 single features × 8 horizons,
LightGBM on all 13 × 8 horizons × 3 prediction cut-offs), t_req 5.12,
day-clustered SE.

* **No candidate.** The highest t of a net is relvol30 at 600 s, +0.23
  ticks per trade, t 0.33. No red flag (t > 10, or > 2 ticks gross at
  h <= 5 s).
* **1-60 s: the signal is a fraction of the cost.** The mean round trip
  costs 1.28 ticks. The best gross per trade is +0.09 at 1 s, +0.10 at 5 s
  and +0.11 at 15 s (all imb_l1), +0.22 at 30 s (ret30) and +0.09 at
  60 s, against detectable nets of 0.12-0.79 ticks. The L1 imbalance is
  the one steady short-horizon signal, as OF16 phase 1 found, at a twelfth
  of the cost.
* **120-600 s: none detected.** Decisions 1 s apart overlap: about 730
  independent outcomes per month at 120 s and 143 at 600 s, so a rule
  needs 1.45-2.81 ticks net per trade to clear t_req. The long-horizon
  features (vwap_dist, delta120, ret120) do predict within a month, but
  the sign flips: vwap_dist's top-minus-bottom quintile move at 600 s is
  +10.0 ticks in November (momentum) and −7.3 in September (reversal);
  fitted on one month, they lose on the other (vwap_dist −3.9 gross per
  trade at 600 s). A walk-forward variant (direction from the previous
  day, the reviewer's exploration, not a test) peaked at +1.01 net, t 1.8.
* LightGBM's out-of-fold R² is negative at every horizon (Nov −0.0005 to
  −0.035; Sep −0.011 to −0.68): month-to-month instability plus, at long
  horizons, overfitting (a 500-row leaf holds about one independent 600 s
  outcome). Plain least squares is negative out of fold too.
* **Review** (independent skeptic): timing (features end at bucket f − 1,
  the fill at f's end quote; ret5 agrees with the f − 1 window in 100% of
  samples), costs, folds and statistics confirmed. It corrected the
  conclusion's wording, from "the features predict nothing" to "no rule
  whose direction holds from one month to the other", and pointed out
  the power limit at long horizons. Both are in the results file. Feature
  caveats, faithful to what the agents saw: features restart every
  window, so vwap_dist is a VWAP over the last 1050-1650 s, not the
  session's, and relvol30's running mean starts at 0 (a sawtooth within
  each window). The first 1050 s after 08:30 and the last ~150 s are never
  sampled.
* **Reading.** Market-order trading on these 13 features is closed at 1 s
  to 1 min, the agents' horizon. What is left is a regime question at
  minutes: a momentum-or-reversal state that holds for a month and flips.
  8 days cannot answer it; the user's ~224 days of `.scid` trades (no
  book) can, for the trade-based features (delta, returns, VWAP distance,
  relative volume), at 2-30 min horizons, with a pre-registered regime
  rule and a walk-forward by month.

## OF18. Month-scale momentum / reversal regimes at 5-30 min: no edge

Plan pre-registered in `OF18_REGIME_PLAN.md` (e9485da, revised after a
design review before any number; amended baf2a7b before any statistic:
the span starts 2025-12-15, since 12-12 was still the back month by
volume). Bars: `tools/scid_to_bars.py --start 2025-12-15 --end 2026-11-11`
(`kit.ps1 run of18_bars`): 1-min front-month RTH bars from the `.scid`
files (1-s bundles through August, tick records with bid and ask from
2026-09-08), into `build-avx512/real/bars/` (gitignored); 236 weekdays
with trades, 7 holiday or half sessions dropped (165-180 gap minutes),
228 used days, 0 OHLC violations, nothing read from the confirm block.
Study: `experiments/of18_regimes.py --out experiments/of18_regimes.txt`
(`kit.ps1 run of18_regimes`, ~30 s; the reviewer's rerun was byte
identical). Walk-forward by day over 168 scored days (2026-03-16 ..
2026-11-11): per feature (ret_5/15/30, delta_5/15/30, vwap_dist), horizon
and lookback W (5, 20, 60 days), day D trades the extreme quintiles in the
direction the W days before it showed; 1-lot market orders entered at the
next bar's open, 1.28 ticks per round trip.

* **Primary** (vwap_dist, 10 min, W 20, the OF17 lead): 4,926 trades,
  gross +0.50, net −0.78 ticks per trade (SE 0.43, t −1.83), negative in
  both halves (−1.13 / −0.44), −1.24 without the 13 OF16 days. No edge;
  the study could have detected a net of 0.85.
* **Secondary** (63 rules at 5, 10 and 15 min, t_req 4.88): no candidate.
  Every rule's net is negative; the best is ret_15 / delta_30 at 15 min,
  −0.23 / −0.24 (t −0.3). Upper bounds: no rule nets more than 1.55,
  2.53 and 3.46 ticks per trade at 5, 10 and 15 min. Gross per trade is
  −1.52 to +1.05: nothing near the cost. At 30 min (descriptive) the best
  is delta_30 W5, +0.95 net, t 0.7.
* **Trade-price bias** (47 days with bid and ask): |bias| <= 0.05 ticks
  for every rule; momentum trades −0.022, reversal +0.020 on average (the
  direction the design review predicted, and negligible).
* **Persistence** (descriptive): the correlation between the trailing
  estimate of a feature's top-minus-bottom move and day D's own value is
  within ±0.19 for most rules; vwap_dist W20 shows −0.19 to −0.41. The
  review showed that this is mostly mechanical: day D's own value exists
  only on the ~50 days that reach both trailing tails (they cross the
  VWAP), is negative by construction there (a no-regime random walk with
  the real volatility gives nearly the same), and scales with the day's
  volatility; with volatility clustering that yields the negative values,
  which after scaling by volatility sit within a day-shuffle null (SD
  ~0.15). No evidence of persistence or anti-persistence either way.
* **Review** (independent): code matches the plan with no lookahead; all
  390 bars rebuilt independently from the records on four days (bundle,
  roll, first tick day, CST tick day) with 0 mismatches; the CT clock and
  front-month choice right; bias formula right. It caught the omitted
  momentum / reversal bias split (added) and the persistence wording
  (corrected in the results file).
* **Reading.** With OF17, market-order trading on order-flow and trend
  features in ES is closed from 1 s to 15 min, within the stated power:
  OF17 tight at <= 1 min, OF18's walk-forward regime rules below 1.5-3.5
  ticks per trade not excluded but nothing points to them. What is left
  for this data is a different way of trading (passive orders, which the
  replay cannot fill honestly without queue data) or different
  information.

## OF19. Footprint patterns on true tick data: no edge

The user asked for order-flow trends, exhaustion and absorption. Their
knowledge base (`SierraChart-KB`, outside the repo) had already tested the
bar-total rules (delta divergence, exhaustion fade, effort-vs-result
absorption: all lost, t down to −8.8) and asked for "one clean run" of the
per-price rules on true tick data. OF19 is that run. Plan pre-registered
in `OF19_FOOTPRINT_PLAN.md` (d659252, revised after a design review
before any number). Data: `tools/scid_to_trades.py` (`kit.ps1 run
of19_trades`): every RTH trade of the front month with its aggressor side
and the record's bid and ask, 47 days (2026-09-08 .. 2026-11-11), ~1M
trades a day, volume equal to the OF18 bars, 0 dropped or zero-side
records, 96.8% of trades at the recorded touch. Study:
`experiments/of19_footprint.py --out experiments/of19_footprint.txt`
(`kit.ps1 run of19_footprint`); 10 warm-up days for trailing thresholds,
37 scored; 1-lot market entries and stops at the recorded touch,
sweep-aware, 100 ms latency, $1.50 per side; targets only on a trade one
tick through.

* **No primary edge, no secondary candidate** (5 primaries at the user's
  Sierra study defaults, t_req 3.79; 20 rules, t_req 4.45). Primaries,
  net per trade: stacked imbalance (r 3, k 3) −0.85 (353 trades, t −0.7);
  absorption (30 s, target 4) −2.10 (169, t −6.4); exhaustion (N 1,
  e 5) −2.19 (788, t −3.2); unfinished auction (d 4) −1.36 (6,169, t
  −9.0); trapped traders (stack trigger) −10.49 (8 trades). The only
  positive nets are P1 variants with 22-148 trades and SEs of 3-11 ticks
  (noise; one +105-tick trade checked as real).
* **Does the footprint condition add anything?** Against the same price
  action without it ("shell", block SE): absorption −0.28 to −1.39,
  exhaustion −0.49 ± 0.65, unfinished auction +0.11 ± 0.17, trapped
  traders (delta) −0.08 to −0.75: nothing distinguishable from zero. The
  KB's own benchmark for unfinished auctions: unfinished and finished
  extremes are revisited at indistinguishable rates (d = 4: 78.3% vs
  78.3% within 15 min, 83.8% vs 84.1% within 30 min, ~17,000 extremes).
* **Mechanics:** exhaustion with e = 0 never fires: on all 37 days, none
  of 6,405 new-low and 6,428 new-high 1-min bars has zero same-side
  aggressor volume at the extreme (a new low prints because someone sells
  into the bid). Absorption grosses 0.70 ± 0.34 below a random-time
  control: about 2 SE and uncorrected, not a finding.
* **Review** (independent): every pattern matches the plan with no
  lookahead; an independent per-record simulator matched 198 trades
  exactly; thresholds use only the 10 days before. It found that Sierra
  stamps the records of one exchange message 1 µs apart, so the first run
  grouped no sweeps; grouping 1-µs chains (the plan's intent) cost every
  rule and control 0.06-0.18 ticks per trade, so it could only make the
  result worse, and the run was redone with it. It also added the shell
  SEs, the short-hold red flag and the zero-side count, and corrected the
  wording ("indistinguishable", not "identical"; the absorption vs random
  difference is not a finding; "vs ctrl" for wide-stop rules measures the
  day's trend).
* **Reading.** With OF17 and OF18 and the KB's bar-level tests: in ES,
  2025-12 .. 2026-11, no order-flow pattern traded with market orders
  (features, regimes, delta divergence, exhaustion, absorption, stacked
  imbalances, unfinished auctions, trapped traders) pays its costs, within
  the stated power. What these tests do not cover: discretionary levels
  ("at a key level"), passive execution, and the book (depth recorded on
  ~13 days only).

## OF20. Absorption and exhaustion at key levels: no edge

Practitioners say these setups work "at a key level". Plan pre-registered
in `OF20_LEVELS_PLAN.md` (0898c06, revised after a design review before
any outcome; only signal counts at levels were looked at). Levels
(`experiments/of20_levels.py`): prior RTH high, low, close, volume POC and
70% value area; overnight high and low; initial balance (from 09:30 CT);
RTH VWAP and ± 1 and 2 SD (from 09:30 CT, rounded), each judged at the
setup's formation time. Study: `experiments/of20_atlevels.py --out
experiments/of20_atlevels.txt` (`kit.ps1 run of20_atlevels`, ~30 s; OF19's
data, signals, fills and statistics; OF19's signal functions gained an
optional `meta` list for the formation time and price, and OF19's results
file reproduced byte for byte after it). Setups: absorption = OF19 P2
(W 15, target 4), exhaustion = OF19 P3 (N 1, e 5); the shell = the same
setup without its footprint condition at the same levels.

* **Primary** (both setups within 2 ticks of any level, 222 trades): net
  +0.10 ticks per trade, t 0.08, halves −1.93 / +2.98; setup − shell
  +1.34 ± 1.21 (mix-weighted). No edge; the bound is 2.47 ticks per trade.
* **Secondary** (8 tests, t_req 4.04): none passes; their bounds are 1.9 to
  21 ticks (25-63 trades each).
* **Absorption** loses everywhere: −1.71 away from all levels (471
  trades, t −6.8), −1.44 at any level; at the VWAP bands −2.12 (59 trades)
  is only slightly worse than away, and its setup − shell −0.96 ± 0.45 is
  ~2 SE picked from 8 tests: not VWAP-specific.
* **Exhaustion** at any level nets +1.44 per trade (119 trades) against
  −2.83 away (669), but that mean rests on a few trades: median −9.24,
  SD 26.7, 34% winners; without its top 3 trades −0.96, without its best
  2 days −0.68. The interaction (setup − shell at levels minus away) is
  +3.64, about 1.6 joint-day SE: descriptive, not a finding. Exhaustion at
  the VWAP bands (+6.02, 62 trades) comes from 27 shorts at +13.54, three
  days contributing ~100 ticks each.
* **Review** (independent): all ten static levels recomputed on four days
  (incl. a Monday and the day after the DST change) matched exactly; no
  lookahead in 55,077 setup and shell trades (formation before signal,
  initial balance and VWAP matches only from 09:30 CT); four trades
  re-simulated from raw ticks matched; the rerun is byte identical. It
  corrected the wording of the exhaustion and absorption-at-VWAP lines
  (above) and recomputed the interaction SE with a joint resample (2.26;
  the results file's hypot gives 2.32).
* **Reading.** At the standard auction-market levels, absorption and
  exhaustion do not make money after costs on these 37 days, and the
  footprint condition does not beat a plain touch of the level. Within
  the power (2.5 ticks per trade for the primary), the "at a key level"
  claim is not supported. The only loose thread is exhaustion at levels,
  carried by a handful of trades; it would need many more days (tick
  records accumulate ~20 a month) and its own pre-registration.

## OF21. Forward test of exhaustion at key levels: pre-registered, not run

OF20's one positive descriptive row (exhaustion, OF19 P3 N 1 e 5, within 2
ticks of any OF20 level: +1.44 per trade on 119 trades, −0.96 without its
top 3) gets a forward test on days recorded from 2026-11-12 on, at the
user's request. Plan `OF21_FORWARD_PLAN.md`, revised after a design review
before any forward day was read; runner `experiments/of21_forward.py`,
frozen with OF19/OF20's code by sha256 (plan section 8, checked by
`python/tests/test_of21.py` and before every look).

* **Design:** looks at 60, 125 and 250 full front-month RTH days
  (Lan-DeMets O'Brien-Fleming, one-sided α 1 − Φ(2); boundaries t ≥ 4.504,
  3.022, 2.007), non-binding futility at net ≤ 0, deadline 2028-03-31.
  An edge needs net > 0, t ≥ the boundary, setup − shell > 0 and stress
  net > 0.
* **Power (the red flag):** the in-sample SE is about 13.2 / sqrt(days);
  the final look has power 0.39 if the true edge is +1.44 and 0.94 at +3.
  Confirming +1.44 at 80% power would take about 680 days.
* **Guards:** ledger blocks `real_forward` (one per look; cumulative, in
  order, never repeated except a crash rerun with the same fingerprint)
  and `real_forward_scan` (dates, front month, full or not and a records
  hash; no prices). A day's status is fixed by its first scan; a full day
  whose records later change refuses every later look. The converters
  still refuse days from 2026-11-12; OF20's `overnight()` reads one only
  for a day opened as a forward look. Roll windows need the next contract
  recorded tick by tick (ESH27 by 2026-12-04).
* **Rehearsal** (`experiments/of21_rehearsal.txt`, `kit.ps1 run
  of21_rehearsal`): the runner, reading the .scid files itself, reproduces
  OF20's row exactly on its 37 days (119 trades, +1.44, SE 2.17, shell
  925, setup − shell +2.61 ± 2.23); all 37 pass the full-day rule. OF20's
  results file still reproduces byte for byte after the `overnight()`
  guard change.

## OF23. Mining turning-point entries with wide exits: no edge

At the user's request ("let it hold trades, try to pick tops and bottoms,
an extended profit target ... train it like a data mining algorithm").
Plan `OF23_SWING_MINING_PLAN.md`, revised after a design review (needs-work,
14 findings, all taken) before anything was built.

* **Design:** six turning-point candidate families on the 228 days of
  1-min RTH bars (swings on 1/5/15 min, failed session extremes, prior-day
  level fades, VWAP ±2 SD fades, FVG retests, bar exhaustion), each
  labelled under 240 exits (targets 8-48 ticks, stops 4-16, caps 30 min
  to end of day, trailing on/off), with the order of target and stop
  inside a bar taken from the .scid records (1-s bundles, ticks from
  September).
* **Mining:** nested walk-forward by test month (March-November 2026): in
  each training range LightGBM out-of-fold predictions per exit, the
  exit and take fraction chosen by an in-order simulation with one trade
  at a time, the test month scored once by the fixed policy. All choices
  are inside training, so the primary has M = 1 (t >= 2); 13 secondaries
  (per family, with and without the model) at t >= 4.26.
* **Edge needs all of:** t >= 2, both halves positive, stress costs
  positive, above the 95th percentile of a random-time control, the 1-s
  vs tick resolution check, >= 0.5 trades a day, positive without the
  5 best days. A pass goes to OF23b on fresh days, never a claim.
  Detectable size about 1.2-1.8 ticks per trade for short exits, 4-6 for
  wide ones.
* **Build Alpha** (offered by the user; not installed): descriptive only,
  after OF23's test months are scored, on the same days and fold
  boundaries; any strategy it finds is a lead for OF23b.
* **RL arms** (amendment 6b, at the user's request "let the RL agent
  trade, let it figure out"): PPO agents on the same candidates as
  episodes, one that must trade and picks one of 12 exits, one that may
  skip; behaviour reported first; they join the secondaries (M = 15).
* **Built** (`experiments/of23_mine.py`, frozen by sha256 in plan 9b;
  `python/tests/test_of23.py`: truncation leak test, fills on synthetic
  records, each family, the in-order walk, an end-to-end run of the
  workers and report on a synthetic cache). Only candidate counts were
  computed on real bars: about 25,000 on 228 days (C1 most, C6 85).
  Run: `kit.ps1 run of23 'python experiments/of23_mine.py --run'`,
  about 1.5-2.5 hours on 24 threads.
* **Result (2026-11-13, `experiments/of23_results.txt`, A92): no edge.**
  24,983 candidates on 228 days; 178 test days (March-November 2026).
  - **Primary** (pooled miner): 945 trades, 5.3 a day, net −1.99 ticks
    per trade (SE 0.79, t −2.52), both halves negative, stress −2.87.
    It sits between the random-time control's mean (−2.65) and its 95th
    percentile (−1.29), so its entries carry no measurable information.
    Bound: an edge above +1.58 ticks per trade is excluded. Every fold
    chose q = 10%; only September was positive (+2.68).
  - **RL-trade** (must trade): 5,989 trades, 33.7 a day, −2.33 per trade;
    it learned to pick the widest exit, T48/S16 to the close, for 54% of
    trades and stop out 75% of the time (win 27%, average win +30.8,
    loss −14.8). That is the gambler's-ruin rate on a random walk: +49
    before −16 has probability 16/65 = 24.6%, which with the costs gives
    about −2.2 per trade. The wide exit minimizes forced trades per day
    (holding blocks later candidates), not loss per trade.
  - **RL-free** (may skip): 3,237 trades, 18.2 a day, −3.22 per trade; it
    learned to skip about half the candidates but not to stay flat.
  - **Secondaries:** none of 15 is positive at stress cost; no family
    beats its random-time control's 95th percentile.
  - **Naive mining:** the best of 8,400 combinations on all 228 days (C5,
    model, q 25%, T48/S12/60 min) made +5.42 ticks a day in-sample, naive
    t +2.07; deflated Sharpe 0.000 (effective trials 7,758), and the same
    pipeline walk-forward lost −0.60 per trade.
  - **Checks:** resolution check passes (1-s vs tick 0.00); bars alone with
    stop first would have said −2.09 instead of −1.99; no red flags.
* **Build Alpha (descriptive, after the commit):** the three agents' test
  trades exported by `experiments/of23_ba_export.py` (outside the repo,
  Build Alpha's external trade-list format) for its robustness tests.

## OF24. The OF23 agent live in Sierra Chart, simulated orders: built, not yet run

At the user's request (a separate Sierra sub-instance; the agent in C++
and Python; live ticks, depth and MBO; sim trading). Design
`OF24_LIVE_SIM_PLAN.md`. An observation, not a test of an edge: the
starting policy is OF23's RL-free arm, which lost on unseen months (A92).

* **Kernel** `include/rag/live/of24_kernel.hpp`: OF23's candidates and 37
  features, the 2 x 64 policy and value networks, 6 live book inputs
  (depth imbalance, MBO large-order share, spread; zero weights at the
  start), and one actor-critic step per closed sim trade. Parity with
  `of23_mine.py` and with torch autograd on synthetic days
  (`python/tests/test_of24.py` through `examples/of24_parity.cpp`).
* **Weights** `experiments/of24_export.py`: the RL-free arm trained on all
  228 days (seed 0), written outside the repo. Greedy: skip 52%, 48/16 to
  the close 29%.
* **Study** `sierra/RagAgent.cpp` (ACSIL, built with the SierraChart-KB
  script, a new DLL name per build): 1-min chart, market entry with an
  attached target and stop, time cap, OF23's trailing rule, daily stop;
  sim guards on every call, Arm input off by default, a dedicated Sim
  account. Reviewed; compiles without warnings. Live data stays inside
  Sierra (its data license); Python only trains offline on the OF23 cache.
* **Runtime check (2026-11-13, sub-instance #2, Sim9, build
  `RagAgent_vb_64`):** a chart replay of 2026-11-10 at 30x in sim. 89
  decisions, 26 entries, 26 closed trades, no error lines; every trade
  closed at its bracket's target or stop (fills match the attached
  orders; a few stops filled 1-3 ticks inside the offset because the
  bracket is priced from the order, not the fill), the trailing rule
  moved stops four times, a learning step ran after every decision.
  Net -93.7 ticks on the day by OF23's costs: one replayed day of an
  agent with no edge, not a result. Not yet observed at runtime: the
  time-cap and end-of-day flatten (no trade lasted that long), the
  daily stop, depth and MBO inputs (off in a replay; market closed).
* **Fix from the replay:** OF23's critic predicts the rest-of-day return
  of an episode (values of 3-7 in reward units), which is the wrong
  baseline for one-decision updates; the exported value head now starts
  at zero and learns the one-step mean online (`of24_export.py`).
* **State:** the study is in live mode, armed, on Sim9 in the
  sub-instance, waiting for Monday 2026-11-16 08:50 CT.

