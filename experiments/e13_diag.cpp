// E13 diagnosis: does the regime switch change what the *current* policy earns?
#include <cstdio>
#include <memory>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  MarketMakingParams pa, pb = pa; pb.decay = 0.3;
  const MarketMakingEnv ea(pa), eb(pb); const MarketMakingDP da(pa), db(pb);
  auto a_opt = [&](const MarketMakingObs& o) { return da.act(o); };
  auto b_opt = [&](const MarketMakingObs& o) { return db.act(o); };
  const auto a_in_a = evaluate(ea, a_opt, 4000, 9'000'000), a_in_b = evaluate(eb, a_opt, 4000, 9'000'000);
  const auto b_in_b = evaluate(eb, b_opt, 4000, 9'000'000);
  std::printf("A's optimal policy earns %.2f +- %.2f in A and %.2f +- %.2f in B; B's optimum earns %.2f in B\n",
              a_in_a.mean, a_in_a.std_error, a_in_b.mean, a_in_b.std_error, b_in_b.mean);
  for (int two = 0; two < 2; ++two) {
    AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000;
    MarketMakingEnv env(pa); auto agent = std::make_unique<Agent<MarketMakingEnv>>(env, cfg);
    TrainConfig tc{.eval_every = 500, .eval_episodes = 500}; tc.drift.two_sided = two == 1;
    Trainer<MarketMakingEnv> tr(*agent, env, tc);
    const auto pre = tr.run(10000);
    std::printf("before the switch: confirmed %zu, rejected %zu:", pre.drift_alarms.size(), pre.drift_rejected.size());
    for (auto r : pre.drift_rejected) std::printf(" %llu", (unsigned long long)r);
    std::printf("\n");
    env.set_decay(0.3);
    const auto rep = tr.run(2000);
    double mean_before = 0, mean_after = 0;
    std::printf("%s: alarms confirmed %zu, rejected %zu", two ? "two-sided" : "one-sided", rep.drift_alarms.size(), rep.drift_rejected.size());
    for (auto r : rep.drift_rejected) std::printf(" (rejected at +%llu)", (unsigned long long)(r - 10000));
    for (int i = 0; i < 200; ++i) mean_after += rep.returns[i] / 200;
    (void)mean_before;
    std::printf("; training returns in the first 200 episodes after the switch: %.2f\n", mean_after);
  }
}
