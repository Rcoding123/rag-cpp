// rag/agent/evaluate.hpp — policy evaluation on common random numbers.
//
// evaluate() scores a policy on fixed scenarios; compare() scores two
// policies on the *same* scenarios and reports their paired difference.
// Episode returns here have a standard deviation (~55) much larger than the
// differences between good policies, so pairing is what makes comparisons
// precise. Used by the trainer (champions, drift confirmation, the baseline
// guard) and by the model-based agent's model check.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

struct EvalStats {
  double mean = 0;
  double std_error = 0;
  std::uint32_t episodes = 0;
};

// Run `episodes` greedy episodes of policy(obs) -> Action on copies of
// `proto`. Episode i always uses Rng(seed + i), so two policies evaluated
// with the same seed face exactly the same scenarios (common random numbers).
template <Environment Env, class Policy>
EvalStats evaluate(const Env& proto, Policy&& policy, std::uint32_t episodes, std::uint64_t seed) {
  double sum = 0, sum2 = 0;
  for (std::uint32_t i = 0; i < episodes; ++i) {
    Env env = proto;
    Rng rng(seed + i);
    auto o = env.reset(rng);
    double total = 0;
    for (;;) {
      const auto st = env.step(policy(o), rng);
      total += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    sum += total;
    sum2 += total * total;
  }
  EvalStats s;
  s.episodes = episodes;
  s.mean = episodes ? sum / episodes : 0;
  const double var = episodes > 1 ? std::max(0.0, (sum2 - episodes * s.mean * s.mean) / (episodes - 1)) : 0;
  s.std_error = episodes ? std::sqrt(var / episodes) : 0;
  return s;
}

// Mean per-episode difference policy_a - policy_b on the same scenarios.
// Pairing cancels the scenario-to-scenario noise both policies share, which
// usually dwarfs the difference between them.
template <Environment Env, class PolicyA, class PolicyB>
EvalStats compare(const Env& proto, PolicyA&& a, PolicyB&& b, std::uint32_t episodes, std::uint64_t seed) {
  double sum = 0, sum2 = 0;
  for (std::uint32_t i = 0; i < episodes; ++i) {
    const double ra = evaluate(proto, a, 1, seed + i).mean;
    const double rb = evaluate(proto, b, 1, seed + i).mean;
    sum += ra - rb;
    sum2 += (ra - rb) * (ra - rb);
  }
  EvalStats s;
  s.episodes = episodes;
  s.mean = episodes ? sum / episodes : 0;
  const double var = episodes > 1 ? std::max(0.0, (sum2 - episodes * s.mean * s.mean) / (episodes - 1)) : 0;
  s.std_error = episodes ? std::sqrt(var / episodes) : 0;
  return s;
}

// Every episode's return, on the same scenarios as evaluate() (episode i
// uses Rng(seed + i)): for distributions, tails and paired analyses.
template <Environment Env, class Policy>
std::vector<double> episode_returns(const Env& proto, Policy&& policy, std::uint32_t episodes, std::uint64_t seed) {
  std::vector<double> out;
  out.reserve(episodes);
  for (std::uint32_t i = 0; i < episodes; ++i) out.push_back(evaluate(proto, policy, 1, seed + i).mean);
  return out;
}

// Conditional value at risk: the mean of the worst `alpha` fraction of
// outcomes (alpha in (0, 1]; e.g. 0.05 = the worst 5%). At least one
// outcome is always included.
inline double cvar(std::vector<double> returns, double alpha) {
  if (returns.empty()) return 0;
  const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(alpha * static_cast<double>(returns.size())));
  std::nth_element(returns.begin(), returns.begin() + static_cast<std::ptrdiff_t>(k - 1), returns.end());
  std::sort(returns.begin(), returns.begin() + static_cast<std::ptrdiff_t>(k));
  double s = 0;
  for (std::size_t i = 0; i < k; ++i) s += returns[i];
  return s / static_cast<double>(k);
}

}  // namespace rag::agent
