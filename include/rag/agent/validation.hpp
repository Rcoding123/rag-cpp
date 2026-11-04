// rag/agent/validation.hpp — out-of-sample validation of execution
// strategies (E28).
//
// A strategy that scores well in the simulator it was trained or tuned in
// can still fail in the market it meets. This header measures strategies the
// way a trading desk would before deploying one:
//
//   * on a stress suite: the nominal market and markets that differ from it
//     (impact, signal strength and sign, persistence, volatility, permanent
//     impact, impact shape, fees), each with its own exact optimum (DP);
//   * on sealed test scenarios: a seed range used for nothing else, so no
//     choice made while developing a strategy has seen them;
//   * with paired, per-episode trading metrics against TWAP (common random
//     numbers): mean gain with a bootstrap interval, share of what was
//     achievable in that market, the worst-5% tail (CVaR) and the
//     probability of doing worse than TWAP.
//
//   for (const auto& s : validation::stress_suite(ExecutionParams{})) {
//     const auto m = validation::measure(s.params, strategy, 4000, validation::kSealedSeed);
//   }
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"

namespace rag::agent::validation {

// Test scenarios start here; training, tuning and earlier evaluations use
// seeds far below (all experiments up to E27 used < 10^8).
inline constexpr std::uint64_t kSealedSeed = 900'000'000;

struct Scenario {
  std::string name;
  ExecutionParams params;
};

// The nominal market first, then one change at a time.
inline std::vector<Scenario> stress_suite(const ExecutionParams& base) {
  std::vector<Scenario> s;
  auto add = [&](std::string name, auto change) {
    ExecutionParams p = base;
    change(p);
    s.push_back({std::move(name), p});
  };
  add("nominal", [](ExecutionParams&) {});
  add("impact x2", [](ExecutionParams& p) { p.impact *= 2; p.terminal_impact *= 2; });
  add("impact x0.5", [](ExecutionParams& p) { p.impact *= 0.5; p.terminal_impact *= 0.5; });
  add("signal x0.5", [](ExecutionParams& p) { p.alpha *= 0.5; });
  add("signal x1.5", [](ExecutionParams& p) { p.alpha *= 1.5; });
  add("signal flipped", [](ExecutionParams& p) { p.alpha = -p.alpha; });
  add("persistence 0.9", [](ExecutionParams& p) { p.signal_rho = 0.9; });
  add("persistence 0.3", [](ExecutionParams& p) { p.signal_rho = 0.3; });
  add("volatility x2", [](ExecutionParams& p) { p.noise_sd *= 2; });
  add("permanent impact", [](ExecutionParams& p) { p.permanent_impact = 0.005; });
  add("impact exponent 1.5", [](ExecutionParams& p) { p.impact_exponent = 1.5; });
  add("fees 0.1 per trade", [](ExecutionParams& p) { p.fixed_cost = 0.1; });
  return s;
}

struct Metrics {
  double gain = 0;          // mean gain over TWAP per episode (price units)
  Interval gain_ci;         // 95% bootstrap interval of the mean gain
  double optimal_gain = 0;  // the scenario's exact optimum's mean gain on the same episodes
  double share = 0;         // gain / optimal_gain (meaningful when optimal_gain > 0)
  double cvar5 = 0;         // mean of the worst 5% of per-episode gains over TWAP
  double p_worse = 0;       // share of episodes that did worse than TWAP
};

// Paired per-episode gains of `policy` over TWAP on episodes seed .. seed+n-1.
template <class Policy>
std::vector<double> gains_over_twap(const ExecutionParams& p, Policy&& policy, std::uint32_t episodes, std::uint64_t seed) {
  const ExecutionEnv env(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const auto a = episode_returns(env, policy, episodes, seed);
  const auto b = episode_returns(env, twap, episodes, seed);
  std::vector<double> d(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
  return d;
}

// optimal_gain < 0 means "compute it": solve the scenario's DP and score it
// on the same episodes. Pass a precomputed value to measure many strategies.
template <class Policy>
Metrics measure(const ExecutionParams& p, Policy&& policy, std::uint32_t episodes, std::uint64_t seed, double optimal_gain = -1) {
  Metrics m;
  const auto g = gains_over_twap(p, policy, episodes, seed);
  double sum = 0;
  std::size_t worse = 0;
  for (double x : g) {
    sum += x;
    worse += x < 0;
  }
  m.gain = sum / static_cast<double>(g.size());
  m.gain_ci = bootstrap_mean(g, 2000, seed ^ 0x5eed);
  m.cvar5 = cvar(g, 0.05);
  m.p_worse = static_cast<double>(worse) / static_cast<double>(g.size());
  if (optimal_gain < 0) {
    const ExecutionDP dp(p);
    const auto o = gains_over_twap(p, [&](const ExecutionObs& obs) { return dp.act(obs); }, episodes, seed);
    double so = 0;
    for (double x : o) so += x;
    optimal_gain = so / static_cast<double>(o.size());
  }
  m.optimal_gain = optimal_gain;
  m.share = optimal_gain > 0 ? m.gain / optimal_gain : 0;
  return m;
}

}  // namespace rag::agent::validation
