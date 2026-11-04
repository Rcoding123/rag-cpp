#include <cmath>
#include <cstdio>
#include <memory>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  MarketMakingParams pa;
  AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000;
  MarketMakingEnv env(pa); auto agent = std::make_unique<Agent<MarketMakingEnv>>(env, cfg);
  Trainer<MarketMakingEnv> tr(*agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(10000);
  Rng rng(77);
  auto block = [&](const char* name, int n) {
    double s = 0, s2 = 0, pred = 0;
    for (int i = 0; i < n; ++i) {
      auto o = env.reset(rng);
      const double p = agent->value(o);
      double total = 0;
      for (;;) { const Action a = agent->act(o, true); const auto st = env.step(a, rng); agent->observe(o, a, st.reward, st.obs, st.done); total += st.reward; o = st.obs; if (st.done) break; }
      agent->end_episode();
      s += total - p; s2 += (total - p) * (total - p); pred += p;
      if (i < 5 || i % 50 == 0) std::printf("  %s ep %3d: predicted %7.2f return %7.2f surprise %+7.2f\n", name, i, p, total, total - p);
    }
    std::printf("%s: mean surprise %+.2f sd %.2f, mean prediction %.2f\n", name, s / n, std::sqrt(s2 / n - (s / n) * (s / n)), pred / n);
  };
  block("A", 200);
  env.set_decay(0.3);
  block("B", 200);
}
