#include <cstdio>
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionEnv env{ExecutionParams{}};
  AgentConfig<3> cfg; cfg.eps_decay_episodes = 20000;
  Agent<ExecutionEnv> agent(env, cfg);
  Rng rng(1);
  DriftDetector det;
  double ssum = 0, gsum = 0, vsum = 0; int n = 0;
  for (int ep = 1; ep <= 60000; ++ep) {
    auto o = env.reset(rng);
    const double v = agent.value(o);
    double g = 0;
    for (;;) { const Action a = agent.act(o, true); auto st = env.step(a, rng); agent.observe(o, a, st.reward, st.obs, st.done); g += st.reward; o = st.obs; if (st.done) break; }
    agent.end_episode();
    ssum += g - v; gsum += g; vsum += v; ++n;
    bool alarm = ep > 20000 && det.add(g - v);
    if (alarm) std::printf("  ALARM at %d\n", ep);
    if (ep % 5000 == 0) { std::printf("ep %6d  mean return %7.2f  mean predicted %7.2f  mean surprise %7.2f  cusum evidence %5.1f\n", ep, gsum / n, vsum / n, ssum / n, det.evidence()); ssum = gsum = vsum = 0; n = 0; }
  }
}
