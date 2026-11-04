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
#include <memory>
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
  // Optional fill curve: fill probability per distance index. Entries < 0
  // fall back to fill_rate * exp(-decay * d). Lets a market have a curve
  // that isn't exponential (E15).
  std::array<double, 3> fill_probs = {-1.0, -1.0, -1.0};
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
  // Fill probability at distance index k (0-2), honouring fill_probs.
  [[nodiscard]] double fill_probability_at(std::uint32_t k) const noexcept {
    return p_.fill_probs[k] >= 0 ? p_.fill_probs[k] : fill_probability(p_.distances[k]);
  }
  // Probability that the bid (ask) fills from inventory q: 0 at the cap.
  [[nodiscard]] double bid_fill(Action a, std::int32_t q) const noexcept {
    return q >= p_.max_inventory ? 0.0 : fill_probability_at(a / 3);
  }
  [[nodiscard]] double ask_fill(Action a, std::int32_t q) const noexcept {
    return q <= -p_.max_inventory ? 0.0 : fill_probability_at(a % 3);
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
    last_bought_ = bought;
    last_sold_ = sold;
    last_move_ = 0.0;
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
    last_move_ = move;
    mid_ += move;
    reward += q_ * move - risk_penalty(q_);
    return {{t_, q_}, reward, false};
  }

  std::array<double, 2> features(const Observation& o) const noexcept {
    return {static_cast<double>(o.t) / static_cast<double>(p_.horizon), static_cast<double>(o.q) / p_.max_inventory};
  }

  [[nodiscard]] double cash() const noexcept { return cash_; }
  // What a market maker observes after a step besides its reward: its own
  // fills (execution reports) and the mid-price move (market data).
  [[nodiscard]] bool last_bought() const noexcept { return last_bought_; }
  [[nodiscard]] bool last_sold() const noexcept { return last_sold_; }
  [[nodiscard]] double last_mid_move() const noexcept { return last_move_; }
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
  bool last_bought_ = false, last_sold_ = false;
  double last_move_ = 0.0;
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

// A model-based market maker: it knows its own objective (risk penalty,
// liquidation penalty, inventory cap, quote distances) but not the market.
// It estimates each distance's fill probability by counting fills (a
// Beta(1, 1) prior, pooled over bid and ask) and the volatility from the
// observed mid moves, then plans with MarketMakingDP on the estimates. The
// fill model is non-parametric over the three distances, so any fill curve
// is inside its model class. E15.
class ModelBasedMarketMaker {
 public:
  struct Config {
    std::uint32_t explore_episodes = 3;  // random quotes first: every distance gets tried
    double eps = 0.01;                   // exploration after that
    std::uint32_t replan_every = 5;      // episodes between re-plans, doubling up to replan_max_gap
    std::uint32_t replan_max_gap = 200;
    std::uint64_t seed = 1;
  };

  ModelBasedMarketMaker(const MarketMakingParams& objective, Config cfg)
      : objective_(objective), cfg_(cfg), rng_(cfg.seed), gap_(cfg.replan_every) {}

  Action act(const MarketMakingObs& o, bool explore) {
    // Greedy without a plan yet: symmetric quotes, without touching the RNG
    // (greedy evaluation must not change what training explores next).
    if (!plan_ && !explore) return MarketMakingEnv::kSymmetricAction;
    if (!plan_ || (explore && (episodes_ < cfg_.explore_episodes || rng_.uniform() < cfg_.eps)))
      return rng_.below(static_cast<std::uint32_t>(MarketMakingEnv::kNumActions));
    return plan_->act(o);
  }

  // One step's evidence: which quotes could fill (not at the cap), which did,
  // and the mid move (none on the last step).
  void observe(const MarketMakingObs& before, Action a, bool bought, bool sold, double mid_move, bool last) {
    const std::int32_t cap = objective_.max_inventory;
    if (before.q < cap) {
      attempts_[a / 3] += 1;
      fills_[a / 3] += bought ? 1 : 0;
    }
    if (before.q > -cap) {
      attempts_[a % 3] += 1;
      fills_[a % 3] += sold ? 1 : 0;
    }
    if (!last) {
      move2_ += mid_move * mid_move;
      moves_ += 1;
    }
  }

  void end_episode() {
    ++episodes_;
    // The gap doubles after each *re*-plan, so the first re-plan comes
    // replan_every episodes after the first plan (bug: it came 2x later).
    if (episodes_ >= cfg_.explore_episodes && (!plan_ || episodes_ - last_plan_ >= gap_)) {
      const bool first = !plan_;
      replan();
      if (!first) gap_ = std::min(gap_ * 2, cfg_.replan_max_gap);
    }
  }

  void replan() {
    MarketMakingParams p = objective_;
    for (std::uint32_t k = 0; k < 3; ++k) p.fill_probs[k] = estimated_fill(k);
    p.volatility = estimated_volatility();
    plan_ = std::make_unique<MarketMakingDP>(p);
    last_plan_ = episodes_;
  }

  [[nodiscard]] double estimated_fill(std::uint32_t k) const noexcept { return (fills_[k] + 1.0) / (attempts_[k] + 2.0); }
  [[nodiscard]] double estimated_volatility() const noexcept { return moves_ > 0 ? std::sqrt(move2_ / moves_) : 1.0; }
  [[nodiscard]] bool has_plan() const noexcept { return plan_ != nullptr; }
  [[nodiscard]] std::uint64_t episodes() const noexcept { return episodes_; }

 private:
  MarketMakingParams objective_;
  Config cfg_;
  Rng rng_;
  std::unique_ptr<MarketMakingDP> plan_;
  std::array<double, 3> fills_{}, attempts_{};
  double move2_ = 0, moves_ = 0;
  std::uint64_t episodes_ = 0, last_plan_ = 0;
  std::uint32_t gap_;
};

// Run `episodes` online episodes of the model-based market maker; returns
// the total reward per episode.
inline std::vector<double> train_model_based(ModelBasedMarketMaker& agent, MarketMakingEnv& env, std::uint32_t episodes, Rng& rng) {
  std::vector<double> returns;
  returns.reserve(episodes);
  for (std::uint32_t e = 0; e < episodes; ++e) {
    auto o = env.reset(rng);
    double total = 0;
    for (;;) {
      const Action a = agent.act(o, true);
      const auto st = env.step(a, rng);
      agent.observe(o, a, env.last_bought(), env.last_sold(), env.last_mid_move(), st.done);
      total += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    agent.end_episode();
    returns.push_back(total);
  }
  return returns;
}

}  // namespace rag::agent
