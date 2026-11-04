#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base;
  auto with = [&](double e, double fixed) { ExecutionParams q = base; q.impact_exponent = e; q.impact = 0.01 * 25 / std::pow(5.0, e); q.fixed_cost = fixed; return q; };
  const Case cases[] = {{"quadratic (model is right)", base}, {"impact ~ q^1.5", with(1.5, 0)}, {"linear impact (e=1)", with(1.0, 0)}, {"convex impact (e=3)", with(3.0, 0)}, {"fixed cost 1.5/order", with(2.0, 1.5)},
                        {"e=1 + fixed 1.5", with(1.0, 1.5)}, {"e=3 + fixed 1.5", with(3.0, 1.5)}};
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  std::printf("%-28s %9s %9s %9s %9s %9s\n", "market", "optimal", "rule", "quadratic", "per-size", "TileQ");
  for (const auto& c : cases) {
    ExecutionEnv env(c.p);
    const ExecutionDP dp(c.p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
    const double rule = compare(env, [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; }, twap, 2000, 5'000'000).mean;
    ModelBasedExecutionAgent mb(c.p, {.seed = 1});
    Rng rng(42);
    train_model_based(mb, env, 1000, rng);
    mb.replan();
    const double gmb = compare(env, [&](const ExecutionObs& o) { return mb.act(o, false); }, twap, 2000, 5'000'000).mean;
    ModelBasedExecutionAgent ps(c.p, {.cost_model = CostModel::PerSize, .seed = 1});
    Rng rng2(42);
    train_model_based(ps, env, 1000, rng2);
    ps.replan();
    const double gps = compare(env, [&](const ExecutionObs& o) { return ps.act(o, false); }, twap, 2000, 5'000'000).mean;
    AgentConfig<3> cfg; cfg.eps_decay_episodes = 20000;
    ExecutionEnv env2(c.p);
    Agent<ExecutionEnv> agent(env2, cfg);
    Trainer<ExecutionEnv> tr(agent, env2, {.eval_every = 5000, .eval_episodes = 1000, .detect_drift = false});
    tr.run(40000);
    const double gq = compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 2000, 5'000'000).mean;
    const auto m = mb.model();
    std::printf("%-28s %+9.2f %8.0f%% %8.0f%% %8.0f%% %8.0f%%   (quadratic fit: impact %.4f alpha %.2f)\n", c.name, opt, 100 * rule / opt,
                100 * gmb / opt, 100 * gps / opt, 100 * gq / opt, m.impact, m.alpha);
  }
}
