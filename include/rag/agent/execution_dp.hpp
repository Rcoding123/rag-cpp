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
// with signal g is
//   -cost(q) - permanent*q*(I - q) + (I - q) * alpha * g - holding_penalty(I - q),
// plus the terminal dump cost on the last step. The next signal is
// rho*g + sqrt(1-rho^2)*sd*Z; the
// expectation over Z uses a 31-point quadrature, and V is linearly
// interpolated on a signal grid spanning +-4 standard deviations.
//
// Speed: on the grid, E[V(t+1, rem, g') | g_j] is the same linear map of the
// row V(t+1, rem, .) for every time step, inventory and action, so it's
// precomputed once as a banded matrix (quadrature weights x interpolation
// weights; ~45 nonzeros per row at the default parameters). The solve then
// costs one banded dot product per (t, rem, j) plus O(1) per action, instead
// of 31 interpolations per (t, inventory, j, action): 74 -> 3.5 ms at the
// 61-point grid. Off-grid queries (act, q_value) still use the quadrature
// directly.
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
    build_expectation_band();

    v_.assign(std::size_t{p.horizon + 1} * (p.lots + 1) * m_, 0.0);
    std::vector<double> cont(std::size_t{p.lots + 1} * m_, 0.0);
    for (std::uint32_t t = p.horizon; t-- > 0;) {
      const bool last = t + 1 >= p.horizon;
      if (!last) continuation(v_, t + 1, cont);
      for (std::uint32_t inv = 0; inv <= p.lots; ++inv)
        for (std::uint32_t j = 0; j < m_; ++j) {
          double best = -std::numeric_limits<double>::infinity();
          for (Action a = 0; a < ExecutionEnv::kNumActions; ++a) best = std::max(best, q_grid(cont, last, inv, j, a));
          at(t, inv, j) = best;
        }
    }
  }

  // Expected reward-to-go of action a, then acting optimally.
  [[nodiscard]] double q_value(std::uint32_t t, std::uint32_t inv, double g, Action a) const noexcept {
    return q_from(v_, t, inv, g, a);
  }

  // Optimal expected reward-to-go from (t, inventory, signal).
  [[nodiscard]] double value(std::uint32_t t, std::uint32_t inv, double g) const noexcept { return value_from(v_, t, inv, g); }

  [[nodiscard]] const ExecutionParams& params() const noexcept { return p_; }

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
  [[nodiscard]] double expected_start_value() const noexcept { return start_value_from(v_); }

  // Expected total reward of an arbitrary policy (a function of the
  // observation) under this model, from the start state: backward induction
  // with the policy's action in place of the max. The policy is queried on
  // the signal grid. A model's predicted gain of A over B is
  // policy_value(A) - policy_value(B); comparing that with the gain measured
  // in the real world, paired, checks the model (E7).
  template <class Policy>
  [[nodiscard]] double policy_value(Policy&& policy) const {
    return evaluate_policy(std::forward<Policy>(policy)).start_value();
  }

  // The full value table of a fixed policy under this model: V^pi(t, inv, g)
  // and Q^pi(t, inv, g, a) = take a now, follow the policy afterwards. Used
  // by doubly robust off-policy evaluation (E10). Holds a pointer to this
  // DP, which must outlive it.
  class PolicyValues {
   public:
    [[nodiscard]] double value(std::uint32_t t, std::uint32_t inv, double g) const noexcept { return dp_->value_from(v_, t, inv, g); }
    [[nodiscard]] double q(std::uint32_t t, std::uint32_t inv, double g, Action a) const noexcept {
      return dp_->q_from(v_, t, inv, g, a);
    }
    [[nodiscard]] double start_value() const noexcept { return dp_->start_value_from(v_); }

   private:
    friend class ExecutionDP;
    PolicyValues(const ExecutionDP* dp, std::vector<double> v) : dp_(dp), v_(std::move(v)) {}
    const ExecutionDP* dp_;
    std::vector<double> v_;
  };

  template <class Policy>
  [[nodiscard]] PolicyValues evaluate_policy(Policy&& policy) const {
    std::vector<double> w(v_.size(), 0.0);
    std::vector<double> cont(std::size_t{p_.lots + 1} * m_, 0.0);
    for (std::uint32_t t = p_.horizon; t-- > 0;) {
      const bool last = t + 1 >= p_.horizon;
      if (!last) continuation(w, t + 1, cont);
      for (std::uint32_t inv = 1; inv <= p_.lots; ++inv)
        for (std::uint32_t j = 0; j < m_; ++j)
          w[index(t, inv, j)] = q_grid(cont, last, inv, j, policy(ExecutionObs{t, inv, lo_ + j * step_}));
    }
    return PolicyValues(this, std::move(w));
  }

 private:
  [[nodiscard]] std::size_t index(std::uint32_t t, std::uint32_t inv, std::uint32_t j) const noexcept {
    return (std::size_t{t} * (p_.lots + 1) + inv) * m_ + j;
  }

  // Precompute, for every grid point j, the weights that map a value row
  // V(.) on the grid to E[V(rho*g_j + innov*Z)] (quadrature x interpolation).
  void build_expectation_band() {
    band_start_.assign(m_, 0);
    band_len_.assign(m_, 0);
    band_offset_.assign(m_ + 1, 0);
    band_w_.clear();
    std::vector<double> row(m_);
    for (std::uint32_t j = 0; j < m_; ++j) {
      std::fill(row.begin(), row.end(), 0.0);
      const double g = lo_ + j * step_;
      for (int i = 0; i < kQuad; ++i) {
        const double x = std::clamp((p_.signal_rho * g + innov_ * nodes_[i] - lo_) / step_, 0.0, static_cast<double>(m_ - 1));
        const auto k = std::min(static_cast<std::uint32_t>(x), m_ - 2);
        const double f = x - k;
        row[k] += weights_[i] * (1 - f);
        row[k + 1] += weights_[i] * f;
      }
      std::uint32_t lo = 0, hi = m_;
      while (lo < hi && row[lo] == 0.0) ++lo;
      while (hi > lo && row[hi - 1] == 0.0) --hi;
      band_start_[j] = lo;
      band_len_[j] = hi - lo;
      band_offset_[j] = static_cast<std::uint32_t>(band_w_.size());
      band_w_.insert(band_w_.end(), row.begin() + lo, row.begin() + hi);
    }
    band_offset_[m_] = static_cast<std::uint32_t>(band_w_.size());
  }

  // cont[rem * m + j] = E[V(t1, rem, g') | g = g_j] for the table v.
  void continuation(const std::vector<double>& v, std::uint32_t t1, std::vector<double>& cont) const noexcept {
    for (std::uint32_t rem = 1; rem <= p_.lots; ++rem) {
      const double* row = &v[index(t1, rem, 0)];
      double* out = &cont[std::size_t{rem} * m_];
      for (std::uint32_t j = 0; j < m_; ++j) {
        const double* w = &band_w_[band_offset_[j]];
        const double* r = row + band_start_[j];
        double s = 0;
        for (std::uint32_t k = 0; k < band_len_[j]; ++k) s += w[k] * r[k];
        out[j] = s;
      }
    }
  }

  // Q-value at grid point j, given the precomputed continuation.
  [[nodiscard]] double q_grid(const std::vector<double>& cont, bool last, std::uint32_t inv, std::uint32_t j, Action a) const noexcept {
    const std::uint32_t q = env_.lots_for(a, inv);
    const std::uint32_t rem = inv - q;
    const double dr = rem;
    double r = -trade_cost_[q] - p_.permanent_impact * static_cast<double>(q) * dr;
    if (last) return r - dump_cost_[rem];
    if (rem == 0) return r;
    return r + dr * p_.alpha * (lo_ + j * step_) - env_.holding_penalty(rem) + cont[std::size_t{rem} * m_ + j];
  }

  // Q-value of action a, then following the values in table `v`.
  [[nodiscard]] double q_from(const std::vector<double>& v, std::uint32_t t, std::uint32_t inv, double g, Action a) const noexcept {
    const std::uint32_t q = env_.lots_for(a, inv);
    const std::uint32_t rem = inv - q;
    const double dr = rem;
    double r = -trade_cost_[q] - p_.permanent_impact * static_cast<double>(q) * dr;
    if (t + 1 >= p_.horizon) return r - dump_cost_[rem];
    if (rem == 0) return r;  // flat: the episode ends
    r += dr * p_.alpha * g - env_.holding_penalty(rem);
    double future = 0;
    for (int i = 0; i < kQuad; ++i) future += weights_[i] * value_from(v, t + 1, rem, p_.signal_rho * g + innov_ * nodes_[i]);
    return r + future;
  }

  [[nodiscard]] double value_from(const std::vector<double>& v, std::uint32_t t, std::uint32_t inv, double g) const noexcept {
    if (t >= p_.horizon || inv == 0) return 0.0;
    const double x = std::clamp((g - lo_) / step_, 0.0, static_cast<double>(m_ - 1));
    const auto j = std::min(static_cast<std::uint32_t>(x), m_ - 2);
    const double f = x - j;
    return (1 - f) * v[index(t, inv, j)] + f * v[index(t, inv, j + 1)];
  }

  [[nodiscard]] double start_value_from(const std::vector<double>& v) const noexcept {
    double s = 0;
    for (int i = 0; i < kQuad; ++i) s += weights_[i] * value_from(v, 0, p_.lots, p_.signal_sd * nodes_[i]);
    return s;
  }

  double& at(std::uint32_t t, std::uint32_t inv, std::uint32_t j) noexcept { return v_[index(t, inv, j)]; }

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
  // Banded expectation operator on the grid (see build_expectation_band).
  std::vector<std::uint32_t> band_start_, band_len_, band_offset_;
  std::vector<double> band_w_;
};

}  // namespace rag::agent
