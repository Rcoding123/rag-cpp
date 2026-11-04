# Task and algorithm specifications

The exact definitions behind every number in EXPERIMENTS.md: the tasks
(dynamics, rewards, defaults), how the optimum is computed, how the learners
update, and how results are scored. Symbols match the code.

## 1. Optimal execution (`ExecutionEnv`, `env.hpp`)

**State** (observation): `t` (steps taken, 0…H−1), `I` (lots left),
`g` (signal). **Actions:** 5. Action k sells q = min(round(sizesₖ · L/H), I)
lots, with sizes = {0, 0.5, 1, 2, 3} × the TWAP rate; action 2 is TWAP.

**One step** from (t, I, g) with action a, q lots sold, I' = I − q held:

1. The sale executes at the current price S minus temporary impact:
   cost(q) = impact · q^impact_exponent + fixed_cost (if q > 0).
2. Permanent impact: the price drops by permanent_impact · q.
3. If this was the last step (t + 1 = H), the remaining I' lots are dumped
   at cost terminal_impact · I'² (+ fixed_cost) and the episode ends.
4. Otherwise the price moves: ΔS = alpha · g + noise_sd · Z₁, and the signal
   evolves as g' = signal_rho · g + √(1 − signal_rho²) · signal_sd · Z₂.

**Reward** (change in mark-to-market wealth, minus the risk penalty):

    r = −cost(q) − permanent_impact · q · I'                     (every step)
        − terminal_impact · I'² (− fixed_cost)                   (last step)
        + I' · ΔS − risk_aversion · noise_sd² · I'²              (other steps)

With risk_aversion = 0 an episode's total reward equals
cash_final − L · S₀ (minus the implementation shortfall); tests check
this identity. The episode also ends early once I = 0. The first signal is
drawn from the stationary distribution, g₀ = signal_sd · Z.

**Defaults** (`ExecutionParams`): L = 100 lots, H = 20 steps,
impact 0.01, terminal_impact 0.05, impact_exponent 2, fixed_cost 0,
risk_aversion 0, permanent_impact 0, alpha 1, signal_rho 0.7,
signal_sd 0.1, noise_sd 0.05, S₀ = 100.

**Features** given to learners: (t/H, I/L, g/signal_sd), with ranges
kLo = (0, 0, −3) and kHi = (1, 1, 3).

## 2. Market making (`MarketMakingEnv`, `market_making.hpp`)

**State:** `t` (0…H−1), `q` (inventory, −Q…Q). **Actions:** 9, coded as
3 · bid + ask, each indexing distances = {0.5, 1, 2} from the mid; action 4
is symmetric at distance 1.

**One step** from (t, q) with distances (d_b, d_a):

1. The bid fills with probability fill_rate · e^(−decay · d_b), unless
   q = Q; the ask fills with probability fill_rate · e^(−decay · d_a),
   unless q = −Q (independent draws). A bid fill buys one unit (q + 1,
   earning d_b against the mid), an ask fill sells one (q − 1, earning d_a).
2. On the last step the inventory is liquidated at the mid, paying
   terminal_penalty · q'², and the episode ends.
3. Otherwise the mid moves by volatility · Z.

**Reward:** d_b · [bid filled] + d_a · [ask filled] + (last step:
−terminal_penalty · q'²; otherwise q' · Δmid − risk_aversion ·
volatility² · q'²). With risk_aversion = 0 the total equals the final cash
(tested).

**Defaults** (`MarketMakingParams`): H = 50, Q = 10, fill_rate 0.9,
decay 1, volatility 1, risk_aversion 0.05, terminal_penalty 0.5.
Features: (t/H, q/Q) in [0, 1] × [−1, 1].

## 3. Corridor (`ChainEnv`)

Positions 0…length−1 (default 10), start at 0, actions left/right, reward −1
per step, episode ends at the goal or after max_steps (60); with slip > 0 a
move is reversed with that probability. Optimal return (slip 0):
−(length − 1). Used for exact unit tests of the learners.

## 4. The exact optima

**ExecutionDP:** backward induction over (t, I, g):
V(t, I, g) = maxₐ Q(t, I, g, a), with
Q = E[r] + E[V(t + 1, I', g')]. Since E[ΔS | g] = alpha · g and the noise
has mean 0, E[r] = −cost(q) − permanent_impact · q · I' + I' · alpha · g −
penalties. The expectation over g' uses a 31-point quadrature of the
standard normal on [−4, 4] (weights ∝ density, normalized); V is linearly
interpolated on a grid of 121 signal points spanning ±4 · signal_sd. On the
grid, the expectation is applied as a precomputed banded matrix (A28).
`expected_start_value()` averages V(0, L, g₀) over the stationary g₀.
`policy_value(π)` and `evaluate_policy(π)` run the same recursion with π's
action in place of the max. `ExecutionDP(params, trade_costs, dump_costs)`
plans with arbitrary cost tables.

**MarketMakingDP:** backward induction over (t, q) with the four fill
outcomes per quote pair (no signal, so no quadrature; exact).

## 5. Learners (`agent.hpp`, `learners.hpp`)

**Tile coding** (`TileCoder`): each feature range is cut into `bins` cells
(default 8 per dimension) by `tilings` (default 8) grids, each offset by a
fraction of a cell. A state activates one tile per tiling.

**TileQ:** Q(s, a) = Σ over active tiles of w[tile, a]. An update toward a
target y adds (α / tilings) · (y − Q(s, a)) to each active weight.

* Targets: n-step returns (default n = 3):
  y = Σ_{k<n} γᵏ rₖ + γⁿ · maxₐ Q(sₙ, a), truncated at episode end
  (γ = 1).
* Experience replay: capacity 100,000; 4 replayed updates per real step.
* Step size α(e) = max(alpha_min, alpha / (1 + e / alpha_decay_episodes)),
  defaults 0.1, 0.003, 3,000 (e counts episodes since the last
  re-exploration).
* Exploration: ε falls linearly from eps_start (1) to eps_end (0.02) over
  eps_decay_episodes (1,500), and never reaches 0. Optional double
  Q-learning (off).

**ActorCritic:** a softmax policy π(a | s) ∝ exp(h(s, a) / T) over
tile-coded logits h, and a tile-coded critic V(s).
δ = r + γ · V(s') − V(s); the critic moves toward r + γ · V(s'); every logit
moves by actor_alpha · δ · (1[b = a] − π(b | s)) / T. Defaults:
actor_alpha 0.03, critic_alpha 0.1, T = 1, both step sizes scaled by
1 / (1 + e / 10,000) (floor alpha_min / alpha). It explores by sampling π;
ε-uniform actions are used only during a re-exploration boost (or a raised
temperature, with `ac_reexplore_temperature`).

**EpisodicMemory:** k-NN (k = 8) over up to 4,096 remembered situations
(random-Fourier-feature embeddings, 64 dimensions), each with Monte Carlo
returns per action; LRU eviction.

**ModelBasedExecutionAgent** (`model_based.hpp`): fits, by weighted least
squares (weights 1 / max(1, I'²)) over every non-terminal step,

    r = alpha · (I' g) − holding · I'² − permanent · q I' − impact · q² [or a cost per order size]

plus terminal_impact from last steps and (rho, signal_sd) from the signal
path; then plans with ExecutionDP on the estimates. Schedule: 10 random
exploration episodes, then ε 0.01; re-plan after 20 episodes, doubling up
to 500. Cost models: Quadratic, PerSize, Shrunk (per-size costs shrunk to
the quadratic fit with 30 pseudo-observations). Optional forgetting and
change detection (E9).

## 6. Supervision (`trainer.hpp`, `drift.hpp`)

* **Held-out evaluation:** every eval_every (250) episodes, the greedy
  policy is scored on eval_episodes (300) fixed scenarios, Rng(9,000,000 + i).
* **Champion:** the best policy so far, re-scored at every evaluation; a
  challenger replaces it if it scores at least as well on the same
  scenarios. `deployed_act()` acts as the champion.
* **Drift:** a one-sided CUSUM test (optionally two-sided) on surprise =
  episode return − the agent's value of the start state, with allowance
  0.3σ, threshold 16σ, a baseline from samples ≥ 50 episodes old (EMA rate
  0.001, warm-up 1,500). An alarm is confirmed if the champion's re-score
  moved by more than 2 standard errors; a confirmed change also comes from
  any evaluation at which the champion's re-score moved that much.
* **On a confirmed change:** bank the champion (up to 8); recall a banked
  policy that beats it by 2 standard errors (then explore lightly: ε 0.05
  for 500 episodes), or else re-explore (ε 0.5 decaying over 5,000
  episodes), clear replay and restart the step-size schedule.
* **Baseline guard** (optional): deploy the baseline until the champion
  beats it by 2 standard errors of the paired difference.

## 7. Metrics

* **Share of the optimal gain:** (mean gain of the policy over the
  baseline) / (mean gain of the exact optimum over the baseline), both
  measured *paired* on the same scenarios (`compare()`: episode i of both
  policies uses Rng(seed + i)). The baseline is TWAP for execution and
  symmetric quoting for market making. 100% = optimal, 0% = baseline,
  negative = worse than the baseline.
* **Online gain** (E8, E11): Σ over training episodes of (return − exact
  value of the baseline), divided by the exact optimal gain per episode;
  +N after N episodes would be optimal.
* **Model check** (E7): z = (gain over TWAP the model predicts for its plan
  − gain measured on real scenarios, paired) / standard error; z > 4 means
  the model is wrong.
* **Tails** (E14): CVaR α = mean of the worst α fraction of episode
  returns, on cash outcomes (risk penalty off).
* **Seeds:** experiments report ranges over seeds (usually 1–3 or 1–5);
  tests pin one seed and assert with margins.
