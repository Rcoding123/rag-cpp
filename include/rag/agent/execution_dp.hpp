// rag/agent/execution_dp.hpp — the optimal policy for ExecutionEnv, solved
// exactly (up to grid resolution) by backward dynamic programming.
//
// This is the yardstick for the learning agent. "It learned something" is a
// weak claim; "it captures 80% of the improvement over TWAP that the optimal
// policy achieves" is a measurable one. The DP knows the model (alpha, rho,
// impact); the agent knows none of it and must learn from rewards alone.
//
// The state is (t, inventory, signal). Rewards are mark-to-market, so the
// price level never enters: the expected reward of selling q from inventory I
// with signal g is  -impact*q^2 + (I - q) * alpha * g,  plus the terminal dump
// cost on the last step. The next signal is rho*g + sqrt(1-rho^2)*sd*Z; the
// expectation over Z uses a 31-point quadrature, and V is linearly
// interpolated on a signal grid spanning +-4 standard deviations.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rag/agent/env.hpp"

namespace rag::agent {

class ExecutionDP {
 public:
  static constexpr int kQuad = 31;

  explicit ExecutionDP(const ExecutionParams& p, std::uint32_t grid = 121, double grid_sds = 4.0)
      : ExecutionDP(p, cost_table(p, false), cost_table(p, true), grid, grid_sds) {}

  // Plan with arbitrary cost tables: trade_costs[q] for selling q lots in a
  // step, dump_costs[rem] for dumping rem lots at the end (both indexed
  // 0..lots). A model-based agent passes the costs it has learned.
  ExecutionDP(const ExecutionParams& p, std::vector<double> trade_costs, std::vector<double> dump_costs,
              std::uint32_t grid = 121, double grid_sds = 4.0)
      : env_(p), p_(p), m_(grid), lo_(-grid_sds * p.signal_sd), step_(2.0 * grid_sds * p.signal_sd / (grid - 1)),
        trade_cost_(std::move(trade_costs)), dump_cost_(std::move(dump_costs)) {
    if (trade_cost_.size() != p.lots + 1u || dump_cost_.size() != p.lots + 1u)
      throw std::invalid_argument("ExecutionDP: cost tables need lots + 1 entries");
    // Quadrature for E[f(Z)], Z ~ N(0,1): evenly spaced nodes on [-4, 4],
    // weights proportional to the density, normalised to sum to 1.
    double total = 0;
    for (int i = 0; i < kQuad; ++i) {
      nodes_[i] = -4.0 + 8.0 * i / (kQuad - 1);
      weights_[i] = std::exp(-0.5 * nodes_[i] * nodes_[i]);
      total += weights_[i];
    }
    for (auto& w : weights_) w /= total;
    innov_ = std::sqrt(1.0 - p.signal_rho * p.signal_rho) * p.signal_sd;

    v_.assign(std::size_t{p.horizon + 1} * (p.lots + 1) * m_, 0.0);
    for (std::uint32_t t = p.horizon; t-- > 0;)
      for (std::uint32_t inv = 0; inv <= p.lots; ++inv)
        for (std::uint32_t j = 0; j < m_; ++j) {
          const double g = lo_ + j * step_;
          double best = -std::numeric_limits<double>::infinity();
          for (Action a = 0; a < ExecutionEnv::kNumActions; ++a) best = std::max(best, q_value(t, inv, g, a));
          at(t, inv, j) = best;
        }
  }

  // Expected reward-to-go of action a, then acting optimally.
  [[nodiscard]] double q_value(std::uint32_t t, std::uint32_t inv, double g, Action a) const noexcept {
    const std::uint32_t q = env_.lots_for(a, inv);
    const std::uint32_t rem = inv - q;
    const double dr = rem;
    double r = -trade_cost_[q];
    if (t + 1 >= p_.horizon) return r - dump_cost_[rem];
    if (rem == 0) return r;  // flat: the episode ends
    r += dr * p_.alpha * g;
    double future = 0;
    for (int i = 0; i < kQuad; ++i) future += weights_[i] * value(t + 1, rem, p_.signal_rho * g + innov_ * nodes_[i]);
    return r + future;
  }

  // Optimal expected reward-to-go from (t, inventory, signal).
  [[nodiscard]] double value(std::uint32_t t, std::uint32_t inv, double g) const noexcept {
    if (t >= p_.horizon || inv == 0) return 0.0;
    const double x = std::clamp((g - lo_) / step_, 0.0, static_cast<double>(m_ - 1));
    const auto j = std::min(static_cast<std::uint32_t>(x), m_ - 2);
    const double f = x - j;
    return (1 - f) * at(t, inv, j) + f * at(t, inv, j + 1);
  }

  // The optimal action (greedy on the exact Q-values, signal not rounded).
  [[nodiscard]] Action act(const ExecutionObs& o) const noexcept {
    Action best = 0;
    double bv = -std::numeric_limits<double>::infinity();
    for (Action a = 0; a < ExecutionEnv::kNumActions; ++a) {
      const double q = q_value(o.t, o.inventory, o.signal, a);
      if (q > bv) { bv = q; best = a; }
    }
    return best;
  }

  // Expected total reward of the optimal policy from the start state,
  // averaged over the stationary distribution of the first signal.
  [[nodiscard]] double expected_start_value() const noexcept {
    double v = 0;
    for (int i = 0; i < kQuad; ++i) v += weights_[i] * value(0, p_.lots, p_.signal_sd * nodes_[i]);
    return v;
  }

 private:
  double& at(std::uint32_t t, std::uint32_t inv, std::uint32_t j) noexcept {
    return v_[(std::size_t{t} * (p_.lots + 1) + inv) * m_ + j];
  }
  double at(std::uint32_t t, std::uint32_t inv, std::uint32_t j) const noexcept {
    return v_[(std::size_t{t} * (p_.lots + 1) + inv) * m_ + j];
  }

  static std::vector<double> cost_table(const ExecutionParams& p, bool dump) {
    const ExecutionEnv env(p);
    std::vector<double> t(p.lots + 1);
    for (std::uint32_t q = 0; q <= p.lots; ++q) t[q] = dump ? env.dump_cost(q) : env.trade_cost(q);
    return t;
  }

  ExecutionEnv env_;  // only for lots_for()
  ExecutionParams p_;
  std::uint32_t m_;
  double lo_, step_, innov_ = 0;
  std::array<double, kQuad> nodes_{}, weights_{};
  std::vector<double> trade_cost_, dump_cost_;  // [lots]
  std::vector<double> v_;  // V[t][inventory][signal grid]
};

}  // namespace rag::agent
