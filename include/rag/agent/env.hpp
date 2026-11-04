// rag/agent/env.hpp — the agent's world: the Environment concept and two
// environments.
//
//   ExecutionEnv  sell an inventory over a fixed horizon. Trading fast costs
//                 market impact, trading slowly risks the price moving away,
//                 and a noisy signal predicts the next price move. The agent
//                 is never told what the signal means: discovering it is the
//                 point of the exercise.
//   ChainEnv      a corridor with a known optimal policy, for exact tests.
//
// An environment is the only part of the architecture that knows the task.
// The agent sees observations, an action count, rewards, and a feature view
// of each observation scaled into known ranges (kLo..kHi).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include "rag/agent/rng.hpp"

namespace rag::agent {

using Action = std::uint32_t;

template <class Obs>
struct Step {
  Obs obs;        // next observation
  double reward;  // reward for the action just taken
  bool done;      // episode over (obs is then terminal and never acted on)
};

template <class E>
concept Environment = requires(E& e, const E& ce, Action a, Rng& rng, const typename E::Observation& o) {
  typename E::Observation;
  requires E::kNumActions >= 2;
  requires E::kStateDims >= 1;
  { E::kLo } -> std::convertible_to<std::array<double, E::kStateDims>>;
  { E::kHi } -> std::convertible_to<std::array<double, E::kStateDims>>;
  { e.reset(rng) } -> std::same_as<typename E::Observation>;
  { e.step(a, rng) } -> std::same_as<Step<typename E::Observation>>;
  { ce.features(o) } -> std::same_as<std::array<double, E::kStateDims>>;
};

// ------------------------------- ChainEnv ------------------------------------
// Positions 0..length-1, start at 0, goal at length-1. Action 0 = left,
// 1 = right. Every step costs -1; reaching the goal ends the episode, as does
// running out of steps. With slip > 0 the chosen move is reversed with that
// probability. Optimal return (slip 0): -(length - 1).
struct ChainParams {
  std::uint32_t length = 10;
  std::uint32_t max_steps = 60;
  double slip = 0.0;
};

class ChainEnv {
 public:
  struct Observation {
    std::uint32_t pos;
    std::uint32_t steps;
  };
  static constexpr std::size_t kNumActions = 2;
  static constexpr std::size_t kStateDims = 1;
  static constexpr std::array<double, 1> kLo{0.0}, kHi{1.0};

  explicit ChainEnv(ChainParams p = {}) : p_(p) {}

  Observation reset(Rng&) noexcept {
    obs_ = {0, 0};
    return obs_;
  }

  Step<Observation> step(Action a, Rng& rng) noexcept {
    bool right = a == 1;
    if (p_.slip > 0 && rng.uniform() < p_.slip) right = !right;
    if (right && obs_.pos + 1 < p_.length) ++obs_.pos;
    if (!right && obs_.pos > 0) --obs_.pos;
    ++obs_.steps;
    const bool done = obs_.pos + 1 == p_.length || obs_.steps >= p_.max_steps;
    return {obs_, -1.0, done};
  }

  std::array<double, 1> features(const Observation& o) const noexcept {
    return {static_cast<double>(o.pos) / static_cast<double>(p_.length - 1)};
  }

  [[nodiscard]] const ChainParams& params() const noexcept { return p_; }

 private:
  ChainParams p_;
  Observation obs_{};
};

// ----------------------------- ExecutionEnv ----------------------------------
// Liquidate `lots` over `horizon` steps. At each step t the agent sells q
// lots at the current price S minus temporary impact (average fill price
// S - impact * q, so the step costs impact * q^2), then the price moves:
//     dS = alpha * signal_t + noise_sd * N(0,1)
// and the signal evolves as an AR(1) with persistence signal_rho. Anything
// still held after the last step is dumped at the much worse terminal_impact.
//
// Reward is the change in mark-to-market wealth (cash + inventory * S), so an
// episode's total reward is exactly cash_final - lots * S_0: minus the
// implementation shortfall against the arrival price. Higher is better.
//
// The signal says where the price is going. Selling faster when it is
// negative and slower when it is positive beats TWAP (selling evenly), but
// only an agent that has learned what the signal means can do that.
struct ExecutionParams {
  std::uint32_t lots = 100;
  std::uint32_t horizon = 20;
  double impact = 0.01;           // eta: selling q lots in one step costs eta * q^impact_exponent
  double terminal_impact = 0.05;  // leftover inventory is dumped at cost terminal_impact * rem^2
  // Alternative market forms (defaults reproduce the quadratic model exactly).
  // Used to test agents that assume the quadratic form against markets that
  // don't follow it.
  double impact_exponent = 2.0;   // e.g. 1.5: "square-root" style impact per lot
  double fixed_cost = 0.0;        // charged for every step that sells anything (fees, spread)
  // Risk aversion (Almgren-Chriss style mean-variance execution): every step
  // the agent holds I' lots through a price move, its reward is reduced by
  // risk_aversion * noise_sd^2 * I'^2, the variance that holding adds to the
  // final cash. 0 = risk-neutral. The penalty is part of the reward only;
  // cash and price accounting are unchanged.
  double risk_aversion = 0.0;
  // Permanent impact: every lot sold lowers the price by this much for the
  // rest of the episode (after the sale itself executes), so the inventory
  // still held loses permanent_impact * q * I'. Agents that model only
  // temporary impact don't have a term for it.
  double permanent_impact = 0.0;
  double alpha = 1.0;             // E[next price move] = alpha * signal
  double signal_rho = 0.7;        // signal persistence per step
  double signal_sd = 0.1;         // stationary standard deviation of the signal
  double noise_sd = 0.05;         // unpredictable part of each price move
  double start_price = 100.0;
  // Action k sells sizes[k] x the TWAP rate (lots / horizon), capped by inventory.
  std::array<double, 5> sizes = {0.0, 0.5, 1.0, 2.0, 3.0};
};

struct ExecutionObs {
  std::uint32_t t;          // steps taken so far, 0..horizon-1 while live
  std::uint32_t inventory;  // lots still to sell
  double signal;            // predicts the price move after this step's trade
};

class ExecutionEnv {
 public:
  using Observation = ExecutionObs;
  static constexpr std::size_t kNumActions = 5;
  static constexpr std::size_t kStateDims = 3;
  static constexpr Action kTwapAction = 2;  // sizes[2] == 1.0
  // features(): time and inventory as fractions, signal in standard deviations.
  static constexpr std::array<double, 3> kLo{0.0, 0.0, -3.0}, kHi{1.0, 1.0, 3.0};

  explicit ExecutionEnv(ExecutionParams p = {}) : p_(p) {}

  Observation reset(Rng& rng) noexcept {
    t_ = 0;
    inventory_ = p_.lots;
    price_ = p_.start_price;
    cash_ = 0.0;
    signal_ = p_.signal_sd * rng.normal();  // drawn from the stationary distribution
    return obs();
  }

  // Lots action `a` sells from `inventory` (deterministic, used by the DP too).
  [[nodiscard]] std::uint32_t lots_for(Action a, std::uint32_t inventory) const noexcept {
    const double rate = static_cast<double>(p_.lots) / static_cast<double>(p_.horizon);
    const auto q = static_cast<std::uint32_t>(std::lround(p_.sizes[a] * rate));
    return std::min(q, inventory);
  }

  // Execution cost of selling q lots in one step (excluding the price move).
  [[nodiscard]] double trade_cost(std::uint32_t q) const noexcept {
    if (q == 0) return 0.0;
    const double dq = static_cast<double>(q);
    const double impact = p_.impact_exponent == 2.0 ? dq * dq : std::pow(dq, p_.impact_exponent);
    return p_.impact * impact + p_.fixed_cost;
  }
  // Risk penalty for holding `held` lots through one price move.
  [[nodiscard]] double holding_penalty(std::uint32_t held) const noexcept {
    const double h = static_cast<double>(held);
    return p_.risk_aversion * p_.noise_sd * p_.noise_sd * h * h;
  }
  [[nodiscard]] double dump_cost(std::uint32_t rem) const noexcept {
    if (rem == 0) return 0.0;
    const double dr = static_cast<double>(rem);
    return p_.terminal_impact * dr * dr + p_.fixed_cost;
  }

  Step<Observation> step(Action a, Rng& rng) noexcept {
    const std::uint32_t q = lots_for(a, inventory_);
    const double cost = trade_cost(q);
    cash_ += static_cast<double>(q) * price_ - cost;
    inventory_ -= q;
    const double drop = p_.permanent_impact * static_cast<double>(q);
    price_ -= drop;
    double reward = -cost - drop * static_cast<double>(inventory_);
    ++t_;
    if (t_ >= p_.horizon) {  // last step: dump whatever is left
      const double dump = dump_cost(inventory_);
      cash_ += static_cast<double>(inventory_) * price_ - dump;
      reward -= dump;
      inventory_ = 0;
      return {obs(), reward, true};
    }
    // The price moves while the remaining inventory is held.
    const double move = p_.alpha * signal_ + p_.noise_sd * rng.normal();
    price_ += move;
    reward += static_cast<double>(inventory_) * move - holding_penalty(inventory_);
    signal_ = p_.signal_rho * signal_ + std::sqrt(1.0 - p_.signal_rho * p_.signal_rho) * p_.signal_sd * rng.normal();
    // Once flat, every later reward is exactly 0, so ending early changes nothing.
    return {obs(), reward, inventory_ == 0};
  }

  std::array<double, 3> features(const Observation& o) const noexcept {
    return {static_cast<double>(o.t) / static_cast<double>(p_.horizon),
            static_cast<double>(o.inventory) / static_cast<double>(p_.lots), o.signal / p_.signal_sd};
  }

  // Accounting (tests and reporting).
  [[nodiscard]] double price() const noexcept { return price_; }
  [[nodiscard]] double cash() const noexcept { return cash_; }
  [[nodiscard]] std::uint32_t inventory() const noexcept { return inventory_; }
  [[nodiscard]] const ExecutionParams& params() const noexcept { return p_; }

  // Change the market's dynamics mid-life (e.g. the signal's meaning flips),
  // to test whether a learning agent notices and adapts.
  void set_alpha(double a) noexcept { p_.alpha = a; }

 private:
  Observation obs() const noexcept { return {t_, inventory_, signal_}; }

  ExecutionParams p_;
  std::uint32_t t_ = 0, inventory_ = 0;
  double price_ = 0.0, cash_ = 0.0, signal_ = 0.0;
};

static_assert(Environment<ChainEnv>);
static_assert(Environment<ExecutionEnv>);

}  // namespace rag::agent
