# A self-learning agent, piece by piece

`include/rag/agent/` holds a complete self-learning agent in about 2,400 lines
of header-only C++20. Its task: sell 100 lots over 20 steps without being told
how the market works. It learns from rewards alone and gets within reach of
the mathematically optimal strategy. It also notices when the market changes
and relearns.

```bash
./build/agent_demo   # the whole story in about 5 seconds
```

This document explains the architecture. Every number quoted here comes from
`agent_demo`, the tests in `tests/agent_tests.cpp`, or the benchmarks.

---

## 1. The loop

Every self-learning agent, whether a robot, a game player or an LLM agent
with memory, is some version of this loop:

```
                  ┌──────────────────────── Trainer (supervision) ────────────────────────┐
                  │  held-out evaluation · champion/challenger · drift detection           │
                  └───────▲───────────────────────────────┬────────────────────────────────┘
                          │ scores, alarms                 │ reexplore(), forget_experience()
                          │                                ▼
 ┌─────────────┐  obs   ┌─────────┐ features ┌──────────┐ Q-values ┌──────────┐ action ┌─────────────┐
 │ Environment ├───────►│ Encoder ├─────────►│ Learner  ├─────────►│  Policy  ├───────►│ Environment │
 └──────┬──────┘        └─────────┘          └────▲─────┘          │ ε-greedy │        └──────┬──────┘
        │                                         │ update         └──────────┘               │
        │                                   ┌─────┴──────┐                                    │
        └────────── reward, next obs ──────►│   Memory   │◄───────────────────────────────────┘
                                            │ replay /   │
                                            │ episodic   │
                                            └────────────┘
```

| Stage | Question it answers | Here | File |
|---|---|---|---|
| Environment | What is the world, and what counts as good? | `ExecutionEnv`, `ChainEnv` | `env.hpp` |
| Encoder | How should situations be described so that similar ones are treated alike? | tile coding, random Fourier features | `features.hpp` |
| Learner | What is each action worth here? | `TileQ` (TD learning), `EpisodicMemory` (k-NN) | `learners.hpp` |
| Memory | What happened before? | `ReplayBuffer`, episodic slots | `learners.hpp` |
| Policy | What should I do now, and when should I try something new? | ε-greedy with schedules | `agent.hpp` |
| Supervision | Am I actually improving? Has the world changed? | `Trainer`, `DriftDetector` | `trainer.hpp` |
| Yardstick | How good could I possibly be? | `ExecutionDP` | `execution_dp.hpp` |

The agent (`Agent<Env>`) is generic. It sees only observations, an action
count, rewards, and the feature ranges an environment declares. The same
class learns the corridor (`ChainEnv`) and the trading task unchanged.

---

## 2. The task, and why it needs learning

`ExecutionEnv`: sell `lots = 100` over `horizon = 20` steps.

* **Impact:** selling `q` lots in one step costs `0.01·q²`. Fast selling is
  expensive, and the cheapest schedule ignoring everything else is TWAP
  (5 lots per step).
* **Signal:** the price moves by `alpha·signal + noise` after each step. The
  signal persists (AR(1), ρ = 0.7), so a negative signal means the price is
  about to fall and it pays to sell now.
* **Deadline:** anything unsold at the end is dumped at 5× the impact.
* **Reward:** the change in mark-to-market wealth. An episode's total reward
  is exactly cash received minus the arrival value. A test checks this to
  1e-9 for random policies.

The agent is never told the impact, the deadline rule, or what the signal
means. The demo's opening table gives the scale:

| Policy | Gain over TWAP per episode |
|---|---|
| TWAP | 0 |
| rule: "sell faster if signal < 0" | +6.5 (43% of optimal) |
| **DP optimum** (knows the model) | **+15.2** |
| learned agent after 40,000 episodes | **+10.1 (67% of optimal)** |

## 3. The yardstick: know what optimal looks like

`ExecutionDP` solves the task exactly (backward induction over time,
inventory and a 121-point signal grid, in 17 ms; 3.4 ms on the 61-point grid
the model-based agent replans with). The expectation over the next signal is
the same linear map of a value row for every time step, inventory and
action, so it's precomputed once as a banded matrix: each (time, remaining
inventory) costs one banded product, and each action then costs O(1). That
made the solve ~9× faster with identical decisions.

This matters for any learning system. "The reward went up" is weak evidence.
"It captures 67% of the gain the optimal policy achieves on the same unseen
scenarios" is a measurement. Before trusting the DP, the tests check:

* its predicted value matches simulation (paired gain 15.59 vs 15.61);
* it beats every heuristic;
* its last-step choices match a brute-force calculation;
* it is mirror-symmetric (flipping `alpha` and the signal together leaves
  the value unchanged).

**Evaluation method.** Single-episode returns have a standard deviation
around 55, far larger than the differences between policies, so comparisons
use *common random numbers*. Every policy faces the same scenarios
(`Rng(seed + i)`), and paired differences cancel the shared noise (see
`compare()` in `trainer.hpp`).

**Tails, not just means (E14).** Scored on cash outcomes, the risk-neutral
optimum is more volatile than TWAP (sd 57 vs 54) yet its worst 5% of days
average −66 against TWAP's −117: the signal lets it avoid the worst
outcomes. `episode_returns()` and `cvar()` in `evaluate.hpp` measure this.

**Risk aversion.** With `ExecutionParams::risk_aversion` λ > 0, each step's
reward also subtracts λ·σ²·I², the price variance the held inventory adds.
The DP stays exact for this mean-variance objective, and it shows the classic
trade-off: as λ goes from 0 to 1, the optimum sells 26 → 73 lots in its
first 5 steps, the sd of its cash falls 59 → 28, and its mean falls
+9.3 → −8.4 (E4). The tail barely improves, though: the worst 5% goes from
−66 to −61 (E14). On market making the same kind of penalty does lift the
tail (worst 5%: −5.7 → +13.7 for 7% of the mean), because there the tail is
inventory.

## 4. Encoders: deciding what "similar" means

Learning generalizes only through the encoder. Two are provided:

* **Tile coding** (for `TileQ`). There are 8 offset grids over
  (time, inventory, signal), and a state activates one cell in each. Nearby
  states share most cells, so learning in one transfers to its neighbours.
  Tests check that shared cells fall with distance, from all 8 for the same
  point to 0 for distant ones.
* **Random Fourier features** (for episodic memory). A 64-dimensional unit
  vector whose dot products approximate a Gaussian similarity
  (Rahimi & Recht). "Recall similar past situations" then becomes exactly
  the nearest-neighbour search the RAG engine runs over text embeddings, and
  it uses the same SIMD kernels (`simd::dot4`) and top-K selection.

*Lesson measured:* finer tiles made learning worse. With 12 cells per
dimension the agent captured 40% instead of 65%, and with one cell per time
step, 15% at 30,000 episodes. Each cell sees fewer samples, and neighbouring
states stop sharing what they learn. Generalization was worth more than
resolution.

## 5. Two ways to learn

| | `TileQ` (parametric) | `EpisodicMemory` (instance-based) |
|---|---|---|
| Stores | one weight per tile per action | up to 4,096 remembered situations, each with a value per action |
| Learns from | every step (3-step TD target `r + r' + r'' + max Q(s''')`) plus replay | each episode's actual returns (Monte Carlo) |
| Estimates Q by | summing 8 weights | similarity-weighted mean of the 8 nearest memories |
| Decision cost | **64 ns** | 19.8 µs (k-NN over 4,096 × 64 floats) |
| After 1,000 episodes | 5.8% of optimal | **13.6%** |
| After 40,000 episodes | **62-71%** (across seeds) | ~33% at 10,000 and improving slowly |

Episodic memory turns one good episode into a usable memory immediately.
The parametric learner generalizes smoothly and wins with data. This is the
"complementary learning systems" idea from neuroscience, and the reason some
agents combine both.

**Replay** (`ReplayBuffer`): every real step adds 4 extra updates from past
transitions, so each interaction with the world is used many times.

**Step size** (`alpha / (1 + episodes/3000)`, floor 0.003): rewards here
are noisy (about ±2.5 per step), while the difference between good and bad
actions is often below 1. A constant step size keeps estimates jittering,
and shrinking it averages the noise out. The floor keeps the agent able to
learn if the world changes.

*Lessons measured:*
* **Plateau:** with 1-step targets, learning levels off around 72%; with the
  default 3-step targets, 75–77% at 200,000 episodes. It is *not* the
  representation's limit. The same tiles fitted to the true optimal values
  give 99.5%. The gap is target noise: returns vary by ±20–50 while action
  values differ by less than 1. See `docs/ai/EXPERIMENTS.md` E2.
* **Model-based alternative:** an agent that learns the market's 5
  parameters and plans with the DP reaches 99–100% in 20–50 episodes
  (`model_based.hpp`, E1), as long as its assumed model form is right.
  See "When the model is wrong" below for what happens when it isn't.
* **Actor-critic (`LearnerKind::ActorCritic`) is the best model-free
  learner here.** It learns a softmax policy directly, with a tile-coded
  critic V(s) whose TD error is the advantage: 75–78% at 10,000 episodes,
  80–86% at 40,000 and 86–89% at 200,000. Its failure mode is different:
  too small an actor step, or annealing too early (`ac_decay_episodes`
  3,000), leaves the policy near-uniform, which loses money (E5).
* **Double Q-learning didn't help.** It removes the upward bias that
  `max()` puts into targets. Here, though, the two were level at 30,000
  episodes (64.8% with it, 64.5% without), and at 60,000 it was clearly worse
  (64% vs 72%): halving each estimator's updates cost more than the bias it
  removed. It's available as `double_q` but off by default.

### When the model is wrong
Add a market feature the model has no term for (permanent impact: every lot
sold lowers the price of everything still held) and the ranking flips. The
model-based agent falls to 22–55%, while TileQ and actor-critic stay at
70–91%, because they assumed nothing. Worse, the wrong model degrades *with*
data (43% → 25% from 20 to 300 episodes): its plan exploits the model's
error, so the data it collects grows more one-sided. Three tools address
this (`docs/ai/EXPERIMENTS.md` E7):

1. **Check the model** (`check_model()`): the gain over TWAP the model
   predicts for its plan (exact, by `ExecutionDP::policy_value`) against the
   gain measured on real scenarios, paired. A wrong model is always
   optimistic, because the planner picks the actions its errors flatter.
   After 100 episodes a correct model scores |z| ≤ 2.5 and a model missing a
   term scores z ≥ 8.5.
2. **Expand the model class** when the missing term is known
   (`learn_permanent_impact`): back to 99.9%, at a cost of ≤ 0.3 points
   where the term isn't needed.
3. **Warm start + guard** when it isn't known: `Agent::distill()` copies
   the model's Q-values into the tiles (in the model's own simulator, so no
   real episodes are spent), RL fine-tunes them on real rewards, and
   `set_baseline(plan)` keeps the plan deployed until the learner beats it.
   Fine-tuning alone wrecks a correct model's 99% policy (to 35–57%): TD
   noise (≈ √(α/2)·σ ≈ 2) is larger than the action gaps (< 1). With the
   guard, the result is 100% when the model is right, and 78–82% (never
   below the plan) when it's wrong.

### When the world changes, for a model
A model that never forgets averages the old and new worlds: 20 episodes
after the signal flips it's at −166% of the new optimum, and still −123%
after 100. Constant forgetting tracks the change but always plans on little
data (54–93% in a noisy, stable market). The model-based agent therefore
uses the same idea as the trainer: a CUSUM test on surprise (episode return
minus what its plan predicted), and on an alarm it starts its statistics
over. It recovers to 95–100% within 20 episodes, against 5,000 episodes
for the model-free agent to reach 18%, and costs nothing while the market
is stable (E9).

### Scoring a policy before deploying it
The trainer's held-out evaluations and the paired model check replay
scenarios, which needs a simulator. From live logs alone, importance
sampling reweights the logged episodes by how likely the target was to act
as the behaviour did. At a horizon of 20 that fails: a target that differs
from the logged policy in many states is followed by almost no logged
episode beyond a few steps, so the effective sample size collapses to ~0
and errors are 45–2,968% of the optimal gain even with 10,000 logs.
Evaluating the target under a fitted model works (2–4%) when the model
class is right; doubly robust estimation only repairs model bias for
targets close to the logged policy (`ope.hpp`, E10). Offline evaluation
therefore comes back to the same requirement as model-based control:
check the model.

## 6. Exploration

A fraction ε of decisions is random. ε falls linearly from 1 to 0.02, and
never reaches 0: an agent that never tries anything new can never notice that
the world changed. `reexplore(eps, n)` raises it temporarily and restarts
the step-size schedule, because relearning needs plasticity. Greedy decisions
(`act(o, false)`) are deterministic and don't touch the agent's random
number generator. A test shows that interleaving evaluations leaves training
bit-identical.

*Lesson measured (E8): exploration is expensive when the episodes are
real.* Counting every training episode as real trading, starting ε at 1
loses more than a thousand episodes' worth of the optimal gain before it
pays off (break-even after ~12,000 episodes). Starting at 0.1 breaks even
after ~2,000 and learns just as well. Little exploration is needed here
because Q-values start at 0, which is optimistic for the bad actions
(never selling ends in a dump costing up to 500), so the greedy choice
tries them by itself. The default stays at 1 for tasks where rewards are
sparse and nothing is optimistic. The model-based agent earns 95–100% of
the optimal gain *while* learning (E11: after its first plan it explores
only 1% of the time; exploring 5% of the time cost 6–10% of the online
gain).

## 7. Supervision: the part most tutorials skip

### Held-out evaluation and the champion
Training returns are measured while exploring, so they understate the
policy. The trainer scores the greedy policy on fixed scenarios it never
trains on, and keeps the best policy seen so far as the **champion**:

* A challenger is promoted only if it scores at least as well on the same
  scenarios.
* The champion is re-scored every time, so after a change it is judged on
  the new world.
* **The champion is what gets deployed** (`deployed_act()`), while the
  learner keeps its own weights.

*Lessons measured:*
* **Restoring the champion destroyed learning.** An early version restored
  the champion into the learner at the end of every `run()`, which silently
  undid learning between calls (recovery stalled at 16%). A test now
  requires that training in chunks is bit-identical to training in one go.
* **The champion check saves real dips.** At episode 30,000 in the demo the
  learner dipped to 16%. The champion (45%) stayed deployed, and the
  learner recovered to 63% and then 67%.
* **Small evaluation sets pick the wrong winner.** Choosing champions on
  300 scenarios and measuring on 1,000 others showed a winner's curse.
  Larger evaluation sets are cheap here.

### Drift detection: suspect fast, confirm exactly
The trainer watches **surprise** = return obtained − return predicted at the
episode start. It works in two stages:

1. **Suspect:** a one-sided CUSUM test (Page, 1954) on standardized surprise,
   with allowance 0.3σ and threshold 16σ, against a baseline built only from
   samples at least 50 episodes old. The parameters were chosen by
   simulation: about one false alarm per 200,000 episodes, and a 0.55σ shift
   (what the market flip causes) caught after a median of 51 episodes.
2. **Confirm:** re-score the frozen champion on the held-out scenarios. In an
   unchanged world that score is reproduced bit for bit; only a changed world
   moves it.

The confirmation stage exists because of a measured failure. During ordinary
learning, value estimates are biased and still moving: early on the agent
predicts +31 and gets −9.6. At episode ~34,000 the learner's policy dipped, a
real performance drop the detector correctly flagged, but the world hadn't
changed. Acting on it reset exploration and cost 18 percentage points (48% instead of
67% at 40,000 episodes).
A fast detector can't tell "the world changed" from "my learning wobbled";
the champion check can.

A confirmed drift makes the agent explore again (ε = 0.5, decaying over
5,000 episodes), restart its step size, and clear its replay buffer, unless
the policy bank recognizes the new world.

**Good news is a change too (E13).** On the market-making task, a deeper
book made *every* policy earn more, so a test that only looked for falling
returns stayed quiet while the agent captured just 66% of the new optimum.
Two fixes: every evaluation now compares the champion's re-score with its
previous score (identical in an unchanged world, so a shift of more than 2
standard errors is a confirmed change, in either direction, with no false
alarms); and `DriftParams::two_sided` adds a mirror-image CUSUM for rises.
A bug surfaced on the way: every alarm, even one the re-score rejected,
reset the detector, which was then blind for 1,550 episodes. Now a false
alarm only re-centres it (`rebase()`).

### Policy bank: worlds seen before
On every confirmed drift the outgoing champion is banked, and older banked
policies are scored on the new world. One that beats the champion by 2
standard errors is redeployed at once, and the learner warm-starts from it.
In a market alternating between two regimes, each return was recognized
within 20–131 episodes, and the deployed policy was back at its old level
500 episodes after the switch. Without the bank it was still at −82% to
−109% then (`docs/ai/EXPERIMENTS.md` E3).

### Baseline guard: never ship worse than the simple thing
`set_baseline(policy)` gives the trainer a safe fallback such as TWAP. The
learned champion is deployed only once it beats the baseline on the
held-out scenarios by 2 standard errors of the paired difference, and the
check reruns after every evaluation and every confirmed drift. In a
weak-signal market (alpha 0.25), unguarded Q-learning deployed policies at
−35% to −106% of the optimal gain, i.e. worse than TWAP. With the guard,
TWAP stayed deployed on every seed, and in the default market nothing
changed (E6b).

*Lessons measured:*
* **Re-exploration length matters most.** A short boost (ε 0.3 for 600
  episodes) recovered only 10% of the new optimum after 20,000 episodes; the
  longer boost recovered about 43%.
* **Clearing replay helps only early.** It speeds up the first 2,500
  episodes after the flip (−0.3% vs −59%) and makes no difference later.

**In the demo**, the signal's meaning flips after 40,000 episodes:

| Episodes after the flip | With drift detection | Without |
|---|---|---|
| 0 | −140% (old knowledge is harmful) | −140% |
| 5,000 (drift detected after 44) | 18% | −89% |
| 25,000 | 46% | −40% |
| 40,000 | 51% | −34% |

---

## 8. How the same architecture maps onto LLM agents

A "self-improving" LLM agent fills the same slots with different parts:

| Slot | Here | In an LLM agent |
|---|---|---|
| Environment | market simulator | tools, users, codebases, web pages |
| Encoder | tile coding / RFF | a text embedding model |
| Episodic memory | k-NN over situation embeddings | a vector store of past attempts and outcomes (this repo's RAG engine) |
| Parametric learner | TD-updated weights | fine-tuning, or learned prompts / rules |
| Reward | mark-to-market P&L | test results, user feedback, an evaluator model |
| Policy | ε-greedy over Q | the LLM, conditioned on retrieved memories |
| Champion/challenger | held-out scenario scores | regression evals before shipping a new prompt or model |
| Drift detection | CUSUM on surprise, confirmed by champion re-score | monitoring eval scores and user signals in production |

The hard problems are the same in both:
* measuring improvement against a fixed yardstick;
* not shipping a policy that got worse;
* noticing when old lessons stop being true.

## 9. Extending it

* **A new task:** implement the `Environment` concept: `reset`, `step`,
  `features`, `kNumActions`, `kStateDims` and `kLo`/`kHi`. Nothing else
  changes. `market_making.hpp` is a worked example: a market maker quoting
  a bid and an ask (9 actions, state (t, inventory)), with an exact DP.
  The unchanged agent learns the optimal inventory skew (actor-critic:
  99–100% of the optimal gain after 5,000 episodes). It's much easier than
  execution because its optimal gain is 4 return standard deviations per
  episode, against 0.27 for execution: the ratio of gain to return noise,
  not the size of the state space, sets how hard model-free learning is
  (E12). A model-based market maker that counts its fills per quote
  distance gets to 98–100% after 3 episodes (`ModelBasedMarketMaker`,
  E15).
* **A new learner:** follow `TileQ` / `EpisodicMemory` / the actor-critic
  branch in `agent.hpp` and add a `LearnerKind`.
* **Real market data:** replay historical signals and prices in an
  environment. Be aware that the price-impact model then becomes an
  assumption, and that a backtest can't show the agent's own impact on the
  market.

## 10. Tests (`tests/agent_tests.cpp` 41 cases, `tests/model_based_tests.cpp` 18, `tests/market_making_tests.cpp` 6)

* **Components:**
  * the random number generator's moments;
  * the reward accounting identity and the environment's invariants;
  * DP self-consistency (including the Bellman equation at every grid
    point, which pins the banded solve to the direct quadrature);
  * tile and Fourier-feature geometry;
  * `TileQ` update arithmetic;
  * the replay buffer ring;
  * episodic k-NN against brute force, plus LRU eviction.
* **Behaviour:**
  * learns the corridor exactly, with both learners and with random slips;
  * captures more than 55% of the optimal gain, and more than the
    hand-written rule plus 15 points;
  * sells more than twice as much when the signal says "falling";
  * the champion's score never decreases;
  * no false drift alarms in a stationary world, a real change detected
    within 500 episodes, and recovery that beats the non-adaptive agent by
    more than 20 points;
  * a returning regime is recalled from the policy bank;
  * a model's plan distilled into the tiles and fine-tuned under the guard
    is best of both;
  * n-step targets are the exact sum of rewards plus a bootstrap;
  * actor-critic learns the corridor and beats Q-learning at equal budget;
  * a risk-averse agent front-loads its selling;
  * the deployed policy is never significantly worse than the baseline.
* **Model-based (`model_based_tests.cpp`):**
  * generalized costs keep the accounting identity, and the DP with cost
    tables equals the DP with formulas;
  * it recovers the market's parameters (including the risk penalty) and is
    near-optimal after 50 episodes;
  * hidden market parameters (every field the planner reads) don't leak
    into the plan;
  * permanent impact is learned when modelled; without the term the model
    gets worse with more data, and `check_model()` flags it;
  * change detection catches a market flip and recovers within 30
    episodes, and is free in a stable market, where constant forgetting
    isn't;
  * off-policy estimators match a hand-computed log; importance sampling
    collapses at horizon 20 while a right model evaluates the target
    accurately, and a wrong model's bias doesn't shrink with data;
  * a wrong model form is costly, a flexible one fixes it, and shrinkage
    makes the flexible one safe with little data.
* **Engineering:**
  * deterministic runs, bit-identical across chunked training;
  * evaluation doesn't perturb learning;
  * snapshots restore the exact policy;
  * `act()` makes zero heap allocations.
