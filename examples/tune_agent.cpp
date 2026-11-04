// tune_agent — tune the tile-coded learner's hyperparameters on ExecutionEnv
// with NLopt, validate on held-out seeds with bootstrap confidence intervals,
// and report which parameters matter (Eigen response surface). E18.
//
//   ./build/tune_agent [episodes=10000] [evaluations=120]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rag/agent/analysis.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
#include "rag/agent/tuning.hpp"

using namespace rag::agent;

namespace {

auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };

AgentConfig<3> config_from(const std::vector<double>& v, std::uint64_t seed) {
  AgentConfig<3> c;
  c.seed = seed;
  c.alpha = v[0];
  c.alpha_min = std::min(c.alpha_min, c.alpha);
  c.alpha_decay_episodes = static_cast<std::uint32_t>(v[1]);
  c.eps_start = v[2];
  c.eps_end = std::min(0.02, v[2]);
  c.eps_decay_episodes = static_cast<std::uint32_t>(v[3]);
  const auto bins = static_cast<std::uint32_t>(v[4]);
  c.bins = {bins, bins, bins};
  c.tilings = static_cast<std::uint32_t>(v[5]);
  return c;
}

// Share of the optimal gain after `episodes` episodes, one value per seed.
std::vector<double> shares(const std::vector<double>& v, const std::vector<std::uint64_t>& seeds, unsigned episodes,
                           const ExecutionEnv& env, double opt) {
  std::vector<double> out;
  for (auto seed : seeds) {
    ExecutionEnv e(env.params());
    Agent<ExecutionEnv> agent(e, config_from(v, seed));
    Trainer<ExecutionEnv> tr(agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
    tr.run(episodes);
    out.push_back(compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 1000, 61'000'000).mean / opt);
  }
  return out;
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

}  // namespace

int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 10000;
  const int evaluations = argc > 2 ? std::atoi(argv[2]) : 120;
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 61'000'000).mean;

  const std::vector<tuning::Param> space = {
      {"alpha", 0.001, 1.0, true, false},       {"alpha_decay_episodes", 300, 30000, true, true},
      {"eps_start", 0.02, 1.0, false, false},   {"eps_decay_episodes", 500, 20000, true, true},
      {"bins", 2, 10, false, true},             {"tilings", 4, 32, true, true}};
  const std::vector<double> defaults = {0.1, 3000, 1.0, 10000, 8, 8};
  const std::vector<std::uint64_t> search_seeds = {1, 2, 8}, held_out = {3, 4, 5, 6, 7, 9, 10, 11};

  std::printf("Searching %d configurations, %u training episodes each, on seeds 1, 2, 8 ...\n", evaluations, episodes);
  // Global search first (the landscape is rugged: a local search's result
  // depended on its initial step), then local refinement.
  const auto r = tuning::maximize_global(
      space, defaults, [&](const std::vector<double>& v) { return mean(shares(v, search_seeds, episodes, env, opt)); },
      evaluations * 2 / 3, evaluations - evaluations * 2 / 3, 1);
  std::printf("  %zu trials (%s); best on the search seeds: %.1f%% of the optimal gain\n\n", r.history.size(), r.status.c_str(),
              100 * r.best_score);
  std::printf("  %-22s %10s %10s\n", "parameter", "default", "tuned");
  for (std::size_t i = 0; i < space.size(); ++i) std::printf("  %-22s %10.4g %10.4g\n", space[i].name.c_str(), defaults[i], r.best[i]);

  std::printf("\nValidation on 8 held-out seeds (mean, 95%% bootstrap interval):\n");
  for (int which = 0; which < 2; ++which) {
    const auto s = shares(which ? r.best : defaults, held_out, episodes, env, opt);
    const auto ci = bootstrap_mean(s, 10000, 1);
    std::printf("  %-8s %5.1f%%  [%5.1f%%, %5.1f%%]\n", which ? "tuned" : "default", 100 * ci.mean, 100 * ci.lo, 100 * ci.hi);
  }

  std::vector<std::vector<double>> rows;
  std::vector<double> scores;
  for (const auto& t : r.history) {
    std::vector<double> row = t.values;
    for (std::size_t i = 0; i < space.size(); ++i)
      if (space[i].log_scale) row[i] = std::log(row[i]);  // analyse log-scaled parameters in log space
    rows.push_back(row);
    scores.push_back(t.score);
  }
  const auto sens = analysis::quadratic_sensitivity(rows, scores);
  std::printf("\nWhat mattered across the %zu trials (quadratic response surface, R^2 = %.2f):\n", rows.size(), sens.r2);
  std::printf("  %-22s %16s %12s\n", "parameter", "linear per sd", "curvature");
  for (std::size_t i = 0; i < space.size(); ++i)
    std::printf("  %-22s %+15.1f%% %+11.1f%%\n", space[i].name.c_str(), 100 * sens.linear[i], 100 * sens.quadratic[i]);
}
