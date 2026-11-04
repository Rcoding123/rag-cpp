// rag/agent/model_based.hpp — a model-based agent for ExecutionEnv:
// learn the market's parameters from experience, then plan with the DP.
//
// Model-free learners (TileQ, EpisodicMemory) learn values directly and
// assume nothing about the world. This agent assumes the *form* of the model
// ("grey box"), and only its numbers are unknown:
//
//   reward  = -cost(q) + alpha * I' * g - holding * I'^2 - permanent * q * I' + I' * noise
//             (I' = inventory held after the trade)
//   last    = -cost(q) - permanent * q * rem - terminal_impact * rem^2
//   g'      = rho * g + innovation
//
// `holding` is a per-step cost of carrying inventory: the risk penalty of a
// risk-averse objective (ExecutionParams::risk_aversion, E4), 0 if
// risk-neutral. `permanent` is permanent price impact (each lot sold lowers
// the price of everything still held), learned only with
// learn_permanent_impact (E7). All coefficients are estimated by weighted least squares over
// every non-terminal step (weights 1/max(1, I')^2: the noise term scales with
// I'), from one normal-equations system solved by Gaussian elimination;
// terminal_impact from last steps; (rho, signal sd) by regressing each signal
// on the previous one. Every `replan_every` episodes it re-solves ExecutionDP
// with the current estimates and acts greedily on that plan, exploring with
// probability eps.
//
// Cost models (CostModel):
//   Quadratic  cost(q) = impact * q^2: one number, the textbook form.
//   PerSize    one learned cost per order size actually traded; unseen sizes
//              are interpolated. No assumption about the shape of the impact
//              curve; the dump at the end is fitted as a * rem^2 + b.
//   Shrunk     per-size costs blended with the quadratic fit by trade count.
//
// What it demonstrates (docs/ai/EXPERIMENTS.md, E1): with the right form, a
// few episodes pin down a handful of numbers and planning does the rest:
// 99-100% of the optimal gain after 20-50 episodes, against ~70% after 40,000
// for the model-free learner. With a badly wrong form (impact ~ q^3, fixed
// costs) the quadratic planner exploits its model's errors (53-83%); per-size
// costs recover (100%) but are catastrophic on some seeds with little data;
// Shrunk is never catastrophic.
//
// Outside its model class (E7: permanent impact with learn_permanent_impact
// off) it falls to 22-55%, and *more data makes it worse*: its plan exploits
// the model's errors, so the data it collects is increasingly one-sided.
// check_model() detects this: the gain over TWAP the model predicts for its
// own plan (exact, by policy evaluation) against the gain realized on real
// scenarios, paired. |z| <= 2.5 when the model is right, 8.5-27 after 100
// episodes when a term is missing (the model is optimistic).
//
// In a market that changes (E9), statistics that never forget average the
// old and new regimes: -166% of the new optimum 20 episodes after a flip.
// Constant forgetting (`forget`) tracks but plans on less data (54-93% in a
// noisy stable market at 0.95). Change detection (`detect_drift`: CUSUM on
// the plan's surprise, reset on an alarm) recovers to 95-100% within 20
// episodes and costs nothing while the market is stable.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "rag/agent/drift.hpp"
#include "rag/agent/env.hpp"
#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

enum class CostModel : std::uint8_t {
  Quadratic,  // cost(q) = impact * q^2
  PerSize,    // a free cost per order size: flexible, but wild with little data
  Shrunk,     // per-size costs shrunk toward the quadratic fit by data count (empirical Bayes)
};

// Result of ModelBasedExecutionAgent::check_model().
struct ModelCheck {
  double predicted_gain = 0;  // plan's gain over TWAP under the learned model (exact)
  double realized_gain = 0;   // measured on real scenarios, paired with TWAP
  double std_error = 0;       // of realized_gain
  double z = 0;               // (predicted - realized) / std_error; > 4: the model is wrong
};

struct ModelEstimate {
  double impact = 0, terminal_impact = 0, alpha = 0, rho = 0, signal_sd = 0;
  double holding = 0;  // per-step cost of holding I' lots: holding * I'^2 (risk penalty)
  double permanent = 0;  // permanent impact: price drop per lot sold
  std::uint64_t steps = 0;  // transitions the estimate is based on
};

class ModelBasedExecutionAgent {
 public:
  struct Config {
    std::uint32_t explore_episodes = 10;  // purely random actions first: diverse data for the regressions
    double eps = 0.01;                    // exploration after that (E11: 0.05 cost 6-10% of the online gain)
    // Exploratory actions are drawn uniformly from [explore_lo, explore_hi].
    // The full range (0-4) includes selling nothing and 3x TWAP, which make
    // random episodes expensive; 1-3 explores near TWAP (E11).
    Action explore_lo = 0, explore_hi = ExecutionEnv::kNumActions - 1;
    std::uint32_t replan_every = 20;      // episodes between DP re-solves...
    std::uint32_t replan_max_gap = 500;   // ...growing (doubling) up to this, as estimates settle
    std::uint32_t dp_grid = 61;           // signal grid of the planner (121 = the benchmark's)
    CostModel cost_model = CostModel::Quadratic;
    double shrink_steps = 30;  // Shrunk: pseudo-observations given to the quadratic prior per size
    bool learn_holding_cost = true;  // estimate the -holding * I'^2 term (needed for risk-averse objectives)
    bool learn_permanent_impact = true;  // estimate the -permanent * q * I' term (E7)
    // Per-episode decay of every sufficient statistic: 1 keeps all data
    // (stationary markets); 0.99 gives old data a half-life of ~70 episodes,
    // so the estimates track a market that changes (E9). Pair it with a small
    // replan_max_gap, or the plan won't follow the estimates.
    double forget = 1.0;
    // Change detection: a one-sided CUSUM test on surprise (episode return
    // minus the plan's predicted value from the start state). On an alarm
    // the agent multiplies its statistics by drift_forget (0 = start over)
    // and replans after replan_every new episodes. Unlike constant
    // forgetting, it costs nothing while the market is stable (E9).
    bool detect_drift = false;
    DriftParams drift{.lag = 10, .allowance = 0.3, .threshold = 10.0, .baseline_rate = 0.01, .warmup = 50};
    double drift_forget = 0.0;
    std::uint64_t seed = 1;
  };

  // `structure` supplies only what the agent is assumed to know: lots,
  // horizon and the action sizes (what each action does). Its impact, alpha,
  // rho, signal_sd and terminal_impact are ignored; they are learned.
  ModelBasedExecutionAgent(const ExecutionParams& structure, Config cfg)
      : structure_(structure), cfg_(cfg), rng_(cfg.seed), env_(structure), gap_(cfg.replan_every),
        nparams_(kCost0 + structure.lots), xtx_(std::size_t{nparams_} * nparams_, 0.0), xty_(nparams_, 0.0),
        per_n_(structure.lots + 1, 0.0), detector_(cfg.drift) {}
  explicit ModelBasedExecutionAgent(const ExecutionParams& structure) : ModelBasedExecutionAgent(structure, Config{}) {}

  Action act(const ExecutionObs& o, bool explore) {
    // Greedy without a plan yet: TWAP, without touching the RNG (greedy
    // evaluation must not change what training explores next).
    if (!plan_ && !explore) return ExecutionEnv::kTwapAction;
    if (!plan_ || (explore && (episodes_ < cfg_.explore_episodes || rng_.uniform() < cfg_.eps)))
      return static_cast<Action>(cfg_.explore_lo + rng_.below(static_cast<std::uint32_t>(cfg_.explore_hi - cfg_.explore_lo + 1)));
    return plan_->act(o);
  }

  void observe(const ExecutionObs& o, Action a, double r, const ExecutionObs& next, bool done) {
    const bool last = done && next.t >= structure_.horizon;
    // Lots sold this step, from the inventory change. On the last step the
    // env also dumps the remainder (next.inventory is then 0), so the sale is
    // taken from the action's known size, lots_for(a, inventory).
    const std::uint32_t q = last ? env_.lots_for(a, o.inventory) : o.inventory - next.inventory;
    if (o.t == 0) {  // first step of an episode: what does the plan expect from here?
      has_prediction_ = plan_ != nullptr;
      predicted_ = has_prediction_ ? plan_->value(0, o.inventory, o.signal) : 0.0;
    }
    episode_return_ += r;
    const double dq = q;
    if (last) {
      const double rem = static_cast<double>(o.inventory - q);
      if (rem > 0) {  // r = -impact q^2 - permanent q rem - terminal_impact rem^2
        t_rem4_ += rem * rem * rem * rem;
        t_rem2_r_ += rem * rem * r;
        t_rem2_q2_ += rem * rem * dq * dq;
        t_rem3_q_ += rem * rem * rem * dq;
      }
      last_steps_.push_back({q, o.inventory - q, r, 1.0});
      ++steps_;
      return;
    }
    // r = alpha (I' g) + holding (-I'^2) + permanent (-q I') + impact (-q^2) [or c_q (-1)] + I' noise
    const double held = next.inventory;
    const double w = 1.0 / std::max(1.0, held * held);
    std::array<std::pair<std::uint32_t, double>, 5> x{};
    std::size_t nx = 0;
    x[nx++] = {kAlpha, held * o.signal};
    x[nx++] = {kHolding, -held * held};
    x[nx++] = {kPermanent, -dq * held};
    x[nx++] = {kImpact, -dq * dq};
    if (q > 0) x[nx++] = {kCost0 + q - 1, -1.0};
    for (std::size_t i = 0; i < nx; ++i) {
      xty_[x[i].first] += w * x[i].second * r;
      for (std::size_t j = 0; j < nx; ++j) xtx_[std::size_t{x[i].first} * nparams_ + x[j].first] += w * x[i].second * x[j].second;
    }
    per_n_[q] += 1.0;
    if (!done) {  // signal transition g -> g'
      g_gg_ += o.signal * o.signal;
      g_gn_ += o.signal * next.signal;
      g_nn_ += next.signal * next.signal;
      g_n_ += 1.0;
    }
    ++steps_;
  }

  // Kept for existing trainers; no longer needed. observe() separates the
  // last step's sale from the dump itself, from its action argument. (Bug:
  // it used to rely on this call, so trainers that only called observe()
  // fitted terminal_impact as if the whole last-step inventory was dumped.)
  void note_action(const ExecutionObs& /*o*/, Action /*a*/) noexcept {}

  void end_episode() {
    ++episodes_;
    if (cfg_.forget < 1.0) decay(cfg_.forget);
    const double surprise = episode_return_ - predicted_;
    const bool planned = has_prediction_;
    episode_return_ = predicted_ = 0.0;
    has_prediction_ = false;
    if (cfg_.detect_drift && planned && detector_.add(surprise)) {
      detector_.reset();  // the model starts over, so does the detector's baseline
      decay(cfg_.drift_forget);
      drift_alarms_.push_back(episodes_);
      last_plan_ = episodes_;
      gap_ = cfg_.replan_every;
      return;
    }
    // The first plan comes as soon as exploration ends. (Bug fixed in E11:
    // it used to wait for replan_every episodes too, so the agent acted
    // randomly for max(explore_episodes, replan_every) = 20 episodes, not 10.)
    // The gap doubles after each *re*-plan, so the first re-plan comes
    // replan_every episodes after the first plan. (Bug: it doubled after the
    // first plan too, giving plans at 10, 50, 130 instead of 10, 30, 70.)
    if (episodes_ >= cfg_.explore_episodes && (!plan_ || episodes_ - last_plan_ >= gap_)) {
      const bool first = !plan_;
      replan();
      if (!first) gap_ = std::min(gap_ * 2, cfg_.replan_max_gap);
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
    // holding_penalty(h) = risk_aversion * noise_sd^2 * h^2 = holding * h^2.
    p.risk_aversion = std::max(0.0, estimate_.holding);
    p.noise_sd = 1.0;
    p.permanent_impact = std::max(0.0, estimate_.permanent);
    if (cfg_.cost_model != CostModel::Quadratic) {
      auto [trade, dump] = per_size_costs();
      if (cfg_.cost_model == CostModel::Shrunk) {
        // Blend each size's own estimate with the quadratic prediction,
        // weighted by how often that size was traded: unseen sizes fall back
        // to the quadratic fit instead of being extrapolated.
        const double k = cfg_.shrink_steps;
        for (std::uint32_t q = 1; q <= structure_.lots; ++q) {
          const double prior = p.impact * q * q, n = per_n_[q];
          trade[q] = (n * (seen(q) ? trade[q] : prior) + k * prior) / (n + k);
        }
        double n_last = 0;
        for (const auto& ls : last_steps_) n_last += ls.w;
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
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> per_size_costs() const {
    const std::uint32_t L = structure_.lots;
    std::vector<double> c(L + 1, 0.0);
    std::vector<std::uint32_t> seen_sizes;
    const auto theta = solve_costs(/*per_size=*/true);
    for (std::uint32_t q = 1; q <= L; ++q)
      if (seen(q)) {
        c[q] = theta[kCost0 + q - 1];
        seen_sizes.push_back(q);
      }
    // c[0] = 0 by definition (no trade, no cost); fill the gaps.
    std::vector<std::uint32_t> knots{0};
    knots.insert(knots.end(), seen_sizes.begin(), seen_sizes.end());
    for (std::uint32_t q = 1; q <= L; ++q) {
      if (seen(q)) continue;
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
      const double x = static_cast<double>(ls.rem) * ls.rem;
      const double y = -ls.r - c[ls.q] - estimate_.permanent * ls.q * static_cast<double>(ls.rem);
      s44 += ls.w * x * x; s42 += ls.w * x; s22 += ls.w; s4y += ls.w * x * y; s2y += ls.w * y;
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

  [[nodiscard]] ModelEstimate current_estimate() const {
    ModelEstimate m;
    m.steps = steps_;
    const auto quad = solve_costs(/*per_size=*/false);
    m.impact = quad[kImpact];
    m.alpha = quad[kAlpha];
    m.holding = quad[kHolding];
    m.permanent = quad[kPermanent];
    if (cfg_.cost_model != CostModel::Quadratic) {  // alpha, holding, permanent with one free cost per size
      const auto per = solve_costs(/*per_size=*/true);
      m.alpha = per[kAlpha];
      m.holding = per[kHolding];
      m.permanent = per[kPermanent];
    }
    if (t_rem4_ > 0) m.terminal_impact = -(t_rem2_r_ + m.impact * t_rem2_q2_ + m.permanent * t_rem3_q_) / t_rem4_;
    if (g_gg_ > 0 && g_n_ > 1) {
      m.rho = g_gn_ / g_gg_;
      const double innov_var = std::max(0.0, (g_nn_ - m.rho * g_gn_) / g_n_);
      const double r2 = std::min(m.rho * m.rho, 0.9999);
      m.signal_sd = std::sqrt(innov_var / (1.0 - r2));
    }
    return m;
  }

  [[nodiscard]] const ModelEstimate& model() const noexcept { return estimate_; }
  [[nodiscard]] bool seen(std::uint32_t q) const noexcept { return q > 0 && q < per_n_.size() && per_n_[q] > 1e-6; }
  [[nodiscard]] bool has_plan() const noexcept { return plan_ != nullptr; }
  [[nodiscard]] const ExecutionDP* plan() const noexcept { return plan_.get(); }

  // Check the learned model against reality: the gain over TWAP it predicts
  // for its own greedy plan vs the gain measured on `episodes` scenarios of
  // `market`, paired. A model that exploits its own errors is optimistic, so
  // the test is one-sided: z > 4 means the model is wrong (E7: never above
  // 2.2 for a correct model, 8.5-27 with a missing term at 100 episodes).
  // Needs a plan (replan() first); costs 2 x `episodes` real episodes.
  [[nodiscard]] ModelCheck check_model(const ExecutionEnv& market, std::uint32_t episodes, std::uint64_t seed) const {
    ModelCheck c;
    if (!plan_) return c;
    const ExecutionDP& m = *plan_;
    auto plan = [&](const ExecutionObs& o) { return m.act(o); };
    auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
    // The plan is greedy on the model's own values, so its value under the
    // model is the DP's optimum (AgentExecutionDP.PolicyValueTablesAreSelfConsistent).
    c.predicted_gain = m.expected_start_value() - m.policy_value(twap);
    const EvalStats real = compare(market, plan, twap, episodes, seed);
    c.realized_gain = real.mean;
    c.std_error = real.std_error;
    c.z = real.std_error > 0 ? (c.predicted_gain - real.mean) / real.std_error : 0.0;
    return c;
  }
  [[nodiscard]] std::uint64_t episodes() const noexcept { return episodes_; }
  [[nodiscard]] std::uint32_t replans() const noexcept { return replans_; }
  // Episodes at which change detection fired (detect_drift).
  [[nodiscard]] const std::vector<std::uint64_t>& drift_alarms() const noexcept { return drift_alarms_; }

 private:
  // Parameter layout of the normal equations.
  static constexpr std::uint32_t kAlpha = 0, kHolding = 1, kImpact = 2, kPermanent = 3;
  static constexpr std::uint32_t kCost0 = 4;  // kCost0 + q - 1: cost of q lots

  // Weighted least squares over the chosen parameters: alpha, holding (if
  // learned) and either impact (quadratic) or one cost per seen size.
  // Returns a full-length parameter vector; unused entries are 0.
  [[nodiscard]] std::vector<double> solve_costs(bool per_size) const {
    std::vector<std::uint32_t> idx{kAlpha};
    if (cfg_.learn_holding_cost) idx.push_back(kHolding);
    if (cfg_.learn_permanent_impact) idx.push_back(kPermanent);
    if (per_size) {
      for (std::uint32_t q = 1; q <= structure_.lots; ++q)
        if (seen(q)) idx.push_back(kCost0 + q - 1);
    } else {
      idx.push_back(kImpact);
    }
    const std::size_t n = idx.size();
    std::vector<double> a(n * (n + 1));  // augmented [X'WX | X'Wy]
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) a[i * (n + 1) + j] = xtx_[std::size_t{idx[i]} * nparams_ + idx[j]];
      a[i * (n + 1) + n] = xty_[idx[i]];
      a[i * (n + 1) + i] += 1e-9 * (1.0 + a[i * (n + 1) + i]);  // tiny ridge: well-defined before data arrives
    }
    // Gaussian elimination with partial pivoting.
    for (std::size_t c = 0; c < n; ++c) {
      std::size_t piv = c;
      for (std::size_t r = c + 1; r < n; ++r)
        if (std::abs(a[r * (n + 1) + c]) > std::abs(a[piv * (n + 1) + c])) piv = r;
      if (std::abs(a[piv * (n + 1) + c]) < 1e-15) continue;
      if (piv != c)
        for (std::size_t k = 0; k <= n; ++k) std::swap(a[c * (n + 1) + k], a[piv * (n + 1) + k]);
      for (std::size_t r = 0; r < n; ++r) {
        if (r == c) continue;
        const double f = a[r * (n + 1) + c] / a[c * (n + 1) + c];
        if (f != 0)
          for (std::size_t k = c; k <= n; ++k) a[r * (n + 1) + k] -= f * a[c * (n + 1) + k];
      }
    }
    std::vector<double> theta(nparams_, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double d = a[i * (n + 1) + i];
      theta[idx[i]] = std::abs(d) < 1e-15 ? 0.0 : a[i * (n + 1) + n] / d;
    }
    return theta;
  }

  ExecutionParams structure_;
  Config cfg_;
  Rng rng_;
  ExecutionEnv env_;  // for lots_for() only
  std::unique_ptr<ExecutionDP> plan_;
  ModelEstimate estimate_{};
  std::uint64_t episodes_ = 0, last_plan_ = 0, steps_ = 0;
  std::uint32_t gap_, replans_ = 0;
  // Weighted least squares sufficient statistics: X'WX (nparams x nparams) and X'Wy.
  std::uint32_t nparams_;
  std::vector<double> xtx_, xty_;
  // Terminal dump: sum rem^4, rem^2 r, rem^2 q^2, rem^3 q.
  double t_rem4_ = 0, t_rem2_r_ = 0, t_rem2_q2_ = 0, t_rem3_q_ = 0;
  std::vector<double> per_n_;  // non-terminal steps observed per size (decayed by `forget`)
  struct LastStep {
    std::uint32_t q, rem;
    double r;
    double w;  // weight (decayed by `forget`)
  };
  std::vector<LastStep> last_steps_;
  // Signal AR(1): sum g^2, g g', g'^2, count.
  double g_gg_ = 0, g_gn_ = 0, g_nn_ = 0;
  double g_n_ = 0;
  // Change detection.
  DriftDetector detector_;
  double predicted_ = 0, episode_return_ = 0;
  bool has_prediction_ = false;
  std::vector<std::uint64_t> drift_alarms_;

  // Multiply every sufficient statistic by f (exponential forgetting).
  void decay(double f) {
    for (double& x : xtx_) x *= f;
    for (double& x : xty_) x *= f;
    t_rem4_ *= f; t_rem2_r_ *= f; t_rem2_q2_ *= f; t_rem3_q_ *= f;
    g_gg_ *= f; g_gn_ *= f; g_nn_ *= f; g_n_ *= f;
    for (double& n : per_n_) n *= f;
    for (auto& ls : last_steps_) ls.w *= f;
    std::erase_if(last_steps_, [](const LastStep& ls) { return ls.w < 1e-4; });
  }
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
