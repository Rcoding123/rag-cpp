// rag/agent/market_making.hpp — a second task for the same agent: market
// making, with its exact optimum.
//
// Each step the agent quotes a bid at mid - d_b and an ask at mid + d_a, with
// d_b and d_a chosen from three distances (9 actions). Each side fills with
// probability fill_rate * exp(-decay * d): a wider quote earns more per fill
// but fills less often. A bid fill buys one unit (+1 inventory, earning d_b
// against the mid), an ask fill sells one (-1, earning d_a). Then the mid
// moves by volatility * N(0,1). Inventory is capped at +-max_inventory (the
// side that would breach the cap doesn't fill).
//
// Reward is the change in mark-to-market wealth, minus a risk penalty
// risk_aversion * volatility^2 * q^2 on the inventory carried through each
// price move; on the last step the inventory is liquidated at
// terminal_penalty * q^2. With risk_aversion = 0, an episode's total reward
// is exactly the final cash (the accounting identity the tests check).
//
// What makes it interesting: the risk-neutral spread-maximizing quote is the
// same on both sides, but holding inventory is penalized, so the optimal
// policy *skews*: when long it quotes a tighter ask (to sell) and a wider bid
// (to stop buying), and the other way round when short (the discrete-time
// version of Avellaneda & Stoikov, 2008). The agent sees only (t, q) and
// rewards, and has to discover the skew.
//
// MarketMakingDP solves it exactly: the state space is (t, q), and the price
// noise has mean zero, so the expected reward of a quote pair depends only
// on q.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

struct MarketMakingParams {
  std::uint32_t horizon = 50;
  std::int32_t max_inventory = 10;
  std::array<double, 3> distances = {0.5, 1.0, 2.0};  // quote distance from the mid
  double fill_rate = 0.9;          // fill probability at distance 0
  double decay = 1.0;              // P(fill) = fill_rate * exp(-decay * d)
  double volatility = 1.0;         // mid-price move per step: volatility * N(0,1)
  double risk_aversion = 0.05;     // per-step penalty risk_aversion * volatility^2 * q^2
  double terminal_penalty = 0.5;   // liquidation cost terminal_penalty * q^2 at the end
};

struct MarketMakingObs {
  std::uint32_t t;   // steps taken so far
  std::int32_t q;    // inventory (+ long, - short)
};

class MarketMakingEnv {
 public:
  using Observation = MarketMakingObs;
  static constexpr std::size_t kNumActions = 9;  // bid distance index * 3 + ask distance index
  static constexpr std::size_t kStateDims = 2;
  static constexpr std::array<double, 2> kLo{0.0, -1.0}, kHi{1.0, 1.0};
  static constexpr Action kSymmetricAction = 4;  // both sides at distances[1]

  explicit MarketMakingEnv(MarketMakingParams p = {}) : p_(p) {}

  [[nodiscard]] static constexpr Action action(std::uint32_t bid, std::uint32_t ask) noexcept { return bid * 3 + ask; }
  [[nodiscard]] double bid_distance(Action a) const noexcept { return p_.distances[a / 3]; }
  [[nodiscard]] double ask_distance(Action a) const noexcept { return p_.distances[a % 3]; }
  [[nodiscard]] double fill_probability(double d) const noexcept { return p_.fill_rate * std::exp(-p_.decay * d); }
  // Probability that the bid (ask) fills from inventory q: 0 at the cap.
  [[nodiscard]] double bid_fill(Action a, std::int32_t q) const noexcept {
    return q >= p_.max_inventory ? 0.0 : fill_probability(bid_distance(a));
  }
  [[nodiscard]] double ask_fill(Action a, std::int32_t q) const noexcept {
    return q <= -p_.max_inventory ? 0.0 : fill_probability(ask_distance(a));
  }
  [[nodiscard]] double risk_penalty(std::int32_t q) const noexcept {
    return p_.risk_aversion * p_.volatility * p_.volatility * static_cast<double>(q) * q;
  }

  Observation reset(Rng&) noexcept {
    t_ = 0;
    q_ = 0;
    mid_ = 100.0;
    cash_ = 0.0;
    return {t_, q_};
  }

  Step<Observation> step(Action a, Rng& rng) noexcept {
    const bool bought = rng.uniform() < bid_fill(a, q_);
    const bool sold = rng.uniform() < ask_fill(a, q_);
    double reward = 0;
    if (bought) {
      cash_ -= mid_ - bid_distance(a);
      reward += bid_distance(a);  // bought below the mid: marked at the mid
      ++q_;
    }
    if (sold) {
      cash_ += mid_ + ask_distance(a);
      reward += ask_distance(a);
      --q_;
    }
    ++t_;
    if (t_ >= p_.horizon) {  // liquidate at the mid, paying the terminal penalty
      const double cost = p_.terminal_penalty * static_cast<double>(q_) * q_;
      cash_ += q_ * mid_ - cost;
      reward -= cost;
      q_ = 0;
      return {{t_, q_}, reward, true};
    }
    const double move = p_.volatility * rng.normal();
    mid_ += move;
    reward += q_ * move - risk_penalty(q_);
    return {{t_, q_}, reward, false};
  }

  std::array<double, 2> features(const Observation& o) const noexcept {
    return {static_cast<double>(o.t) / static_cast<double>(p_.horizon), static_cast<double>(o.q) / p_.max_inventory};
  }

  [[nodiscard]] double cash() const noexcept { return cash_; }
  [[nodiscard]] std::int32_t inventory() const noexcept { return q_; }
  [[nodiscard]] const MarketMakingParams& params() const noexcept { return p_; }

  // Change the market mid-life (a volatility regime switch), to test whether
  // a learning agent notices and adapts (E13).
  void set_volatility(double v) noexcept { p_.volatility = v; }
  void set_decay(double d) noexcept { p_.decay = d; }  // book depth: how fast fills fall off with distance

 private:
  MarketMakingParams p_;
  std::uint32_t t_ = 0;
  std::int32_t q_ = 0;
  double mid_ = 100.0, cash_ = 0.0;
};

static_assert(Environment<MarketMakingEnv>);

// Exact optimum by backward induction over (t, q): four fill outcomes per
// quote pair, each with its spread earned, risk penalty and continuation.
class MarketMakingDP {
 public:
  explicit MarketMakingDP(const MarketMakingParams& p) : env_(p), p_(p), width_(2 * p.max_inventory + 1) {
    v_.assign(std::size_t{p.horizon + 1} * width_, 0.0);
    for (std::uint32_t t = p.horizon; t-- > 0;)
      for (std::int32_t q = -p.max_inventory; q <= p.max_inventory; ++q) {
        double best = -std::numeric_limits<double>::infinity();
        for (Action a = 0; a < MarketMakingEnv::kNumActions; ++a) best = std::max(best, q_value(t, q, a));
        v_[idx(t, q)] = best;
      }
  }

  // Expected reward-to-go of quoting `a` at (t, q), then acting optimally.
  [[nodiscard]] double q_value(std::uint32_t t, std::int32_t q, Action a) const noexcept {
    const double pb = env_.bid_fill(a, q), pa = env_.ask_fill(a, q);
    const double db = env_.bid_distance(a), da = env_.ask_distance(a);
    const bool last = t + 1 >= p_.horizon;
    auto after = [&](std::int32_t q2) {
      return last ? -p_.terminal_penalty * static_cast<double>(q2) * q2 : -env_.risk_penalty(q2) + v_[idx(t + 1, q2)];
    };
    return (1 - pb) * (1 - pa) * after(q) + pb * (1 - pa) * (db + after(q + 1)) + (1 - pb) * pa * (da + after(q - 1)) +
           pb * pa * (db + da + after(q));
  }

  [[nodiscard]] double value(std::uint32_t t, std::int32_t q) const noexcept { return t >= p_.horizon ? 0.0 : v_[idx(t, q)]; }

  [[nodiscard]] Action act(const MarketMakingObs& o) const noexcept {
    Action best = 0;
    for (Action a = 1; a < MarketMakingEnv::kNumActions; ++a)
      if (q_value(o.t, o.q, a) > q_value(o.t, o.q, best)) best = a;
    return best;
  }

  [[nodiscard]] double expected_start_value() const noexcept { return value(0, 0); }

  // Exact value of any policy of (t, q), from the start.
  template <class Policy>
  [[nodiscard]] double policy_value(Policy&& policy) const {
    MarketMakingDP copy = *this;
    for (std::uint32_t t = p_.horizon; t-- > 0;)
      for (std::int32_t q = -p_.max_inventory; q <= p_.max_inventory; ++q)
        copy.v_[idx(t, q)] = copy.q_value(t, q, policy(MarketMakingObs{t, q}));
    return copy.value(0, 0);
  }

 private:
  [[nodiscard]] std::size_t idx(std::uint32_t t, std::int32_t q) const noexcept {
    return std::size_t{t} * width_ + static_cast<std::size_t>(q + p_.max_inventory);
  }

  MarketMakingEnv env_;
  MarketMakingParams p_;
  std::uint32_t width_;
  std::vector<double> v_;  // V[t][q + max_inventory]
};

}  // namespace rag::agent
