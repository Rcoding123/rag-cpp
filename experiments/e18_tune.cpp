// E18: hyperparameter optimization of TileQ with NLopt (Subplex), validated on held-out seeds.
#include <nlopt.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
struct Ctx { const ExecutionEnv* env; double opt; std::vector<std::uint64_t> seeds; unsigned episodes; int evals = 0; double best = -1e9; std::FILE* hist = nullptr; };
AgentConfig<3> config_of(const std::vector<double>& x, std::uint64_t seed) {
  AgentConfig<3> c; c.seed = seed;
  c.alpha = std::exp(x[0]); c.alpha_decay_episodes = static_cast<std::uint32_t>(std::exp(x[1]));
  c.eps_start = x[2]; c.eps_end = std::min(0.02, x[2]); c.eps_decay_episodes = static_cast<std::uint32_t>(std::exp(x[3]));
  const auto bins = static_cast<std::uint32_t>(std::lround(x[4])); c.bins = {bins, bins, bins};
  c.tilings = static_cast<std::uint32_t>(std::lround(std::exp(x[5])));
  c.alpha_min = std::min(c.alpha_min, c.alpha);
  return c;
}
double score(const std::vector<double>& x, const Ctx& ctx, const std::vector<std::uint64_t>& seeds) {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  double s = 0;
  for (auto seed : seeds) {
    ExecutionEnv e(ctx.env->params()); Agent<ExecutionEnv> agent(e, config_of(x, seed));
    Trainer<ExecutionEnv> tr(agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
    tr.run(ctx.episodes);
    s += compare(*ctx.env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 1000, 61'000'000).mean / ctx.opt;
  }
  return s / static_cast<double>(seeds.size());
}
double objective(const std::vector<double>& x, std::vector<double>&, void* data) {
  auto& ctx = *static_cast<Ctx*>(data);
  const double v = score(x, ctx, ctx.seeds);
  ++ctx.evals; ctx.best = std::max(ctx.best, v);
  if (ctx.hist) { for (double xi : x) std::fprintf(ctx.hist, "%.6f ", xi); std::fprintf(ctx.hist, "%.6f\n", v); std::fflush(ctx.hist); }
  std::printf("eval %2d: alpha %.3f decay %5.0f eps0 %.2f epsdecay %5.0f bins %ld tilings %ld -> %5.1f%% (best %5.1f%%)\n", ctx.evals, std::exp(x[0]), std::exp(x[1]), x[2], std::exp(x[3]),
              std::lround(x[4]), std::lround(std::exp(x[5])), 100 * v, 100 * ctx.best);
  std::fflush(stdout);
  return v;
}
int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? std::atoi(argv[1]) : 10000;
  const int maxeval = argc > 2 ? std::atoi(argv[2]) : 60;
  ExecutionParams p; const ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  Ctx ctx{&env, compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 61'000'000).mean, {1, 2, 8}, episodes};
  ctx.hist = std::fopen(argc > 3 ? argv[3] : "e18_history.txt", "w");
  // log alpha, log alpha_decay, eps_start, log eps_decay, bins, log tilings
  std::vector<double> lo = {std::log(0.001), std::log(300.0), 0.02, std::log(500.0), 2.0, std::log(4.0)};
  std::vector<double> hi = {std::log(1.0), std::log(30000.0), 1.0, std::log(20000.0), 10.0, std::log(32.0)};
  std::vector<double> x0 = {std::log(0.1), std::log(3000.0), 1.0, std::log(10000.0), 8.0, std::log(8.0)};  // the defaults used in tests
  const auto t0 = std::chrono::steady_clock::now();
  nlopt::opt opt(nlopt::LN_SBPLX, 6);
  opt.set_lower_bounds(lo); opt.set_upper_bounds(hi);
  opt.set_max_objective(objective, &ctx);
  opt.set_maxeval(maxeval);
  opt.set_initial_step({0.7, 0.7, 0.3, 0.7, 2.0, 0.5});
  std::vector<double> x = x0; double fx = 0;
  try { opt.optimize(x, fx); } catch (const std::exception& e) { std::printf("nlopt: %s\n", e.what()); }
  std::printf("search: %d evaluations, %.0f s, best on training seeds %.1f%%\n", ctx.evals, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 100 * fx);
  const std::vector<std::uint64_t> held = {3, 4, 5, 6, 7};
  std::printf("held-out seeds 3-7: default %.1f%%  tuned %.1f%%\n", 100 * score(x0, ctx, held), 100 * score(x, ctx, held));
  std::printf("training seeds 1,2,8: default %.1f%%  tuned %.1f%%\n", 100 * score(x0, ctx, {1, 2, 8}), 100 * fx);
  {
    Ctx c40 = ctx;
    c40.episodes = 40000;
    std::printf("at 40k episodes, held-out seeds 3-7: default %.1f%%  tuned %.1f%%\n", 100 * score(x0, c40, held), 100 * score(x, c40, held));
  }
  std::printf("tuned: alpha %.4f alpha_decay %.0f eps_start %.3f eps_decay %.0f bins %ld tilings %ld\n", std::exp(x[0]), std::exp(x[1]), x[2], std::exp(x[3]), std::lround(x[4]), std::lround(std::exp(x[5])));
}
