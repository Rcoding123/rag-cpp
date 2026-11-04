// rag/agent/model_based.hpp — a model-based agent for ExecutionEnv:
// learn the market's parameters from experience, then plan with the DP.
//
// Model-free learners (TileQ, EpisodicMemory) learn values directly and
// assume nothing about the world. This agent assumes the *form* of the model
// ("grey box"), and only its numbers are unknown:
//
//   reward  = -impact * q^2 + I' * (alpha * g + noise)     (I' = inventory held after the trade)
//   last    = -impact * q^2 - terminal_impact * rem^2
//   g'      = rho * g + innovation
//
// It estimates (impact, alpha) by weighted least squares over every
// non-terminal step (weights 1/max(1, I')^2: the noise term scales with I'),
// terminal_impact from last steps, and (rho, signal sd) by regressing each
// signal on the previous one. Every `replan_every` episodes it re-solves
// ExecutionDP with the current estimates and acts greedily on that plan,
// exploring with probability eps.
//
// Two cost models:
//   Quadratic  cost(q) = impact * q^2: 2 numbers, the textbook form.
//   PerSize    one learned cost per order size actually traded (closed-form
//              least squares); unseen sizes are interpolated. No assumption
//              about the shape of the impact curve; the dump at the end is
//              fitted as a * rem^2 + b.
//
// What it demonstrates (docs/ai/EXPERIMENTS.md, E1): with the right form, a
// few episodes pin down a handful of numbers and planning does the rest:
// 99-100% of the optimal gain after 10-20 episodes, against 67% after 40,000
// for the model-free learner. With a badly wrong form (impact ~ q^3) the
// quadratic planner exploits its model's errors and loses money (-64%);
// the per-size model, which assumes less, recovers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

enum class CostModel : std::uint8_t {
  Quadratic,  // cost(q) = impact * q^2
  PerSize,    // a free cost per order size: flexible, but wild with little data
  Shrunk,     // per-size costs shrunk toward the quadratic fit by data count (empirical Bayes)
};

struct ModelEstimate {
  double impact = 0, terminal_impact = 0, alpha = 0, rho = 0, signal_sd = 0;
  std::uint64_t steps = 0;  // transitions the estimate is based on
};

class ModelBasedExecutionAgent {
 public:
  struct Config {
    std::uint32_t explore_episodes = 10;  // purely random actions first: diverse data for the regressions
    double eps = 0.05;                    // exploration after that
    std::uint32_t replan_every = 20;      // episodes between DP re-solves...
    std::uint32_t replan_max_gap = 500;   // ...growing (doubling) up to this, as estimates settle
    std::uint32_t dp_grid = 61;           // signal grid of the planner (121 = the benchmark's)
    CostModel cost_model = CostModel::Quadratic;
    double shrink_steps = 30;  // Shrunk: pseudo-observations given to the quadratic prior per size
    std::uint64_t seed = 1;
  };

  // `structure` supplies only what the agent is assumed to know: lots,
  // horizon and the action sizes (what each action does). Its impact, alpha,
  // rho, signal_sd and terminal_impact are ignored; they are learned.
  ModelBasedExecutionAgent(const ExecutionParams& structure, Config cfg)
      : structure_(structure), cfg_(cfg), rng_(cfg.seed), env_(structure), gap_(cfg.replan_every),
        per_w_(structure.lots + 1, 0.0), per_x2_(structure.lots + 1, 0.0), per_y_(structure.lots + 1, 0.0),
        per_n_(structure.lots + 1, 0) {}
  explicit ModelBasedExecutionAgent(const ExecutionParams& structure) : ModelBasedExecutionAgent(structure, Config{}) {}

  Action act(const ExecutionObs& o, bool explore) {
    if (!plan_ || (explore && (episodes_ < cfg_.explore_episodes || rng_.uniform() < cfg_.eps)))
      return rng_.below(static_cast<std::uint32_t>(ExecutionEnv::kNumActions));
    return plan_->act(o);
  }

  void observe(const ExecutionObs& o, Action /*a*/, double r, const ExecutionObs& next, bool done) {
    const bool last = done && next.t >= structure_.horizon;
    // Lots sold this step, from the inventory change. On the last step the
    // env also dumps the remainder (next.inventory is then 0), so the sale is
    // taken from the action's known size, recorded by note_action().
    const std::uint32_t q = last ? sold_ : o.inventory - next.inventory;
    sold_ = 0;
    const double dq = q;
    if (last) {
      const double rem = static_cast<double>(o.inventory - q);
      if (rem > 0) {  // r = -impact q^2 - terminal_impact rem^2
        t_rem4_ += rem * rem * rem * rem;
        t_rem2_r_ += rem * rem * r;
        t_rem2_q2_ += rem * rem * dq * dq;
      }
      last_steps_.push_back({q, o.inventory - q, r});
      ++steps_;
      return;
    }
    // r = impact * (-q^2) + alpha * (I' g) + I' * noise
    const double held = next.inventory;
    const double x1 = -dq * dq, x2 = held * o.signal;
    const double w = 1.0 / std::max(1.0, held * held);
    s11_ += w * x1 * x1;
    s12_ += w * x1 * x2;
    s22_ += w * x2 * x2;
    s1y_ += w * x1 * r;
    s2y_ += w * x2 * r;
    ++per_n_[q];
    per_w_[q] += w;  // per-size model: r = -c[q] + alpha * x2
    per_x2_[q] += w * x2;
    per_y_[q] += w * r;
    if (!done) {  // signal transition g -> g'
      g_gg_ += o.signal * o.signal;
      g_gn_ += o.signal * next.signal;
      g_nn_ += next.signal * next.signal;
      ++g_n_;
    }
    ++steps_;
  }

  // Must be called right before observe() on the env's last step, with the
  // action taken there, so the agent can separate sale from dump. Trainers
  // that only call observe() still work: then the split comes from
  // lots_for(action, inventory) via note_action().
  void note_action(const ExecutionObs& o, Action a) noexcept { sold_ = env_.lots_for(a, o.inventory); }

  void end_episode() {
    ++episodes_;
    if (episodes_ >= cfg_.explore_episodes && episodes_ - last_plan_ >= gap_) {
      replan();
      gap_ = std::min(gap_ * 2, cfg_.replan_max_gap);
    }
  }

  // Re-estimate parameters and re-solve the plan now.
  void replan() {
    estimate_ = current_estimate();
    ExecutionParams p = structure_;
    p.impact = std::max(estimate_.impact, 1e-6);
    p.terminal_impact = std::max(estimate_.terminal_impact, p.impact);
    p.alpha = estimate_.alpha;
    p.signal_rho = std::clamp(estimate_.rho, -0.99, 0.99);
    p.signal_sd = std::max(estimate_.signal_sd, 1e-6);
    p.impact_exponent = 2.0;
    p.fixed_cost = 0.0;
    if (cfg_.cost_model != CostModel::Quadratic) {
      auto [trade, dump] = per_size_costs(p.alpha);
      if (cfg_.cost_model == CostModel::Shrunk) {
        // Blend each size's own estimate with the quadratic prediction,
        // weighted by how often that size was traded: unseen sizes fall back
        // to the quadratic fit instead of being extrapolated.
        const double k = cfg_.shrink_steps;
        for (std::uint32_t q = 1; q <= structure_.lots; ++q) {
          const double prior = p.impact * q * q, n = static_cast<double>(per_n_[q]);
          trade[q] = (n * (per_w_[q] > 0 ? trade[q] : prior) + k * prior) / (n + k);
        }
        const double n_last = static_cast<double>(last_steps_.size());
        for (std::uint32_t rem = 1; rem <= structure_.lots; ++rem) {
          const double prior = p.terminal_impact * rem * rem;
          dump[rem] = (n_last * dump[rem] + k * prior) / (n_last + k);
        }
      }
      plan_ = std::make_unique<ExecutionDP>(p, std::move(trade), std::move(dump), cfg_.dp_grid);
    } else {
      plan_ = std::make_unique<ExecutionDP>(p, cfg_.dp_grid);
    }
    last_plan_ = episodes_;
    ++replans_;
  }

  // Learned cost tables (PerSize): c[q] by closed-form least squares with
  // alpha profiled out; unseen sizes linearly interpolated (or extrapolated
  // along the last segment); dump cost fitted as a*rem^2 + b on last steps.
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> per_size_costs(double alpha) const {
    const std::uint32_t L = structure_.lots;
    std::vector<double> c(L + 1, 0.0);
    std::vector<std::uint32_t> seen;
    for (std::uint32_t q = 1; q <= L; ++q)
      if (per_w_[q] > 0) {
        c[q] = (alpha * per_x2_[q] - per_y_[q]) / per_w_[q];
        seen.push_back(q);
      }
    // c[0] = 0 by definition (no trade, no cost); fill the gaps.
    std::vector<std::uint32_t> knots{0};
    knots.insert(knots.end(), seen.begin(), seen.end());
    for (std::uint32_t q = 1; q <= L; ++q) {
      if (per_w_[q] > 0) continue;
      auto hi = std::upper_bound(knots.begin(), knots.end(), q);
      if (hi == knots.end()) {  // beyond the largest seen size: extend the last segment
        const std::uint32_t b = knots.back(), a = knots.size() > 1 ? knots[knots.size() - 2] : 0;
        c[q] = b == a ? c[b] : c[b] + (c[b] - c[a]) * (q - b) / static_cast<double>(b - a);
      } else {
        const std::uint32_t b = *hi, a = *(hi - 1);
        c[q] = c[a] + (c[b] - c[a]) * (q - a) / static_cast<double>(b - a);
      }
    }
    // Dump: -r - c[q] = a * rem^2 + b, least squares over last steps with rem > 0.
    double s44 = 0, s42 = 0, s22 = 0, s4y = 0, s2y = 0;
    for (const auto& ls : last_steps_) {
      if (ls.rem == 0) continue;
      const double x = static_cast<double>(ls.rem) * ls.rem, y = -ls.r - c[ls.q];
      s44 += x * x; s42 += x; s22 += 1; s4y += x * y; s2y += y;
    }
    double a = std::max(estimate_.terminal_impact, 1e-6), b = 0;
    const double det = s44 * s22 - s42 * s42;
    if (s22 >= 2 && det > 1e-9) {
      a = (s22 * s4y - s42 * s2y) / det;
      b = (s44 * s2y - s42 * s4y) / det;
    }
    std::vector<double> dump(L + 1, 0.0);
    for (std::uint32_t rem = 1; rem <= L; ++rem) dump[rem] = std::max(0.0, a * rem * rem + b);
    return {std::move(c), std::move(dump)};
  }

  [[nodiscard]] ModelEstimate current_estimate() const noexcept {
    ModelEstimate m;
    m.steps = steps_;
    const double det = s11_ * s22_ - s12_ * s12_;
    if (det > 1e-12) {
      m.impact = (s22_ * s1y_ - s12_ * s2y_) / det;
      m.alpha = (s11_ * s2y_ - s12_ * s1y_) / det;
    }
    if (cfg_.cost_model != CostModel::Quadratic) {
      // alpha with one free cost per size: alpha = (S2y - sum Sy_q Sx2_q / W_q) / (S22 - sum Sx2_q^2 / W_q),
      // where q = 0 (no trade, known zero cost) contributes to the S sums only.
      double num = s2y_, den = s22_;
      for (std::uint32_t q = 1; q <= structure_.lots; ++q)
        if (per_w_[q] > 0) {
          num -= per_y_[q] * per_x2_[q] / per_w_[q];
          den -= per_x2_[q] * per_x2_[q] / per_w_[q];
        }
      if (den > 1e-12) m.alpha = num / den;
    }
    if (t_rem4_ > 0) m.terminal_impact = -(t_rem2_r_ + m.impact * t_rem2_q2_) / t_rem4_;
    if (g_gg_ > 0 && g_n_ > 1) {
      m.rho = g_gn_ / g_gg_;
      const double innov_var = std::max(0.0, (g_nn_ - m.rho * g_gn_) / static_cast<double>(g_n_));
      const double r2 = std::min(m.rho * m.rho, 0.9999);
      m.signal_sd = std::sqrt(innov_var / (1.0 - r2));
    }
    return m;
  }

  [[nodiscard]] const ModelEstimate& model() const noexcept { return estimate_; }
  [[nodiscard]] bool has_plan() const noexcept { return plan_ != nullptr; }
  [[nodiscard]] std::uint64_t episodes() const noexcept { return episodes_; }
  [[nodiscard]] std::uint32_t replans() const noexcept { return replans_; }

 private:
  ExecutionParams structure_;
  Config cfg_;
  Rng rng_;
  ExecutionEnv env_;  // for lots_for() only
  std::unique_ptr<ExecutionDP> plan_;
  ModelEstimate estimate_{};
  std::uint64_t episodes_ = 0, last_plan_ = 0, steps_ = 0;
  std::uint32_t gap_, replans_ = 0, sold_ = 0;
  // Weighted least squares sufficient statistics for (impact, alpha).
  double s11_ = 0, s12_ = 0, s22_ = 0, s1y_ = 0, s2y_ = 0;
  // Terminal dump: sum rem^4, rem^2 r, rem^2 q^2.
  double t_rem4_ = 0, t_rem2_r_ = 0, t_rem2_q2_ = 0;
  // Per-size model: weighted sums per lot count q of w, w*x2, w*r.
  std::vector<double> per_w_, per_x2_, per_y_;
  std::vector<std::uint64_t> per_n_;  // steps observed per size
  struct LastStep {
    std::uint32_t q, rem;
    double r;
  };
  std::vector<LastStep> last_steps_;
  // Signal AR(1): sum g^2, g g', g'^2, count.
  double g_gg_ = 0, g_gn_ = 0, g_nn_ = 0;
  std::uint64_t g_n_ = 0;
};

// Train a model-based agent for `episodes` episodes on `env` (own RNG stream).
inline void train_model_based(ModelBasedExecutionAgent& agent, ExecutionEnv& env, std::uint32_t episodes, Rng& rng) {
  for (std::uint32_t e = 0; e < episodes; ++e) {
    auto o = env.reset(rng);
    for (;;) {
      const Action a = agent.act(o, true);
      agent.note_action(o, a);
      const auto st = env.step(a, rng);
      agent.observe(o, a, st.reward, st.obs, st.done);
      o = st.obs;
      if (st.done) break;
    }
    agent.end_episode();
  }
}

}  // namespace rag::agent
