#include <cstdio>
#include <string>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  struct Case { std::string name; ExecutionParams p; };
  std::vector<Case> cases;
  ExecutionParams base;
  for (double a : {0.25, 0.5, 2.0}) { auto p = base; p.alpha = a; cases.push_back({"alpha " + std::to_string(a).substr(0, 4), p}); }
  for (double n : {0.025, 0.1, 0.2}) { auto p = base; p.noise_sd = n; cases.push_back({"noise " + std::to_string(n).substr(0, 5), p}); }
  for (double r : {0.0, 0.9}) { auto p = base; p.signal_rho = r; cases.push_back({"rho " + std::to_string(r).substr(0, 3), p}); }
  for (unsigned h : {10u, 40u}) { auto p = base; p.horizon = h; cases.push_back({"horizon " + std::to_string(h), p}); }
  cases.insert(cases.begin(), Case{"default", base});
  std::printf("%-14s %9s %8s | %10s %10s %10s\n", "market", "DP gain", "rule", "TileQ 40k", "AC 40k", "MB 300");
  for (const auto& c : cases) {
    ExecutionEnv env(c.p); const ExecutionDP dp(c.p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1500, 5'000'000).mean;
    auto share = [&](auto&& pi) { return 100 * compare(env, pi, twap, 1500, 5'000'000).mean / opt; };
    double res[2];
    for (int k = 0; k < 2; ++k) {
      AgentConfig<3> cfg; cfg.learner = k ? LearnerKind::ActorCritic : LearnerKind::TileQ; cfg.eps_decay_episodes = 20000;
      ExecutionEnv e(c.p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
      Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 5000, .eval_episodes = 500, .detect_drift = false});
      tr.run(40000);
      res[k] = share([&](const ExecutionObs& o) { return tr.deployed_act(o); });
    }
    ModelBasedExecutionAgent mb(c.p, {.seed = 1}); Rng r(42); ExecutionEnv e3(c.p); train_model_based(mb, e3, 300, r); mb.replan();
    std::printf("%-14s %+9.2f %7.0f%% | %9.0f%% %9.0f%% %9.0f%%\n", c.name.c_str(), opt,
                share([](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; }), res[0], res[1],
                share([&](const ExecutionObs& o) { return mb.act(o, false); }));
  }
}
