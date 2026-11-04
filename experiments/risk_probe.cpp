#include <cmath>
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
// Cash result (vs arrival) mean and sd, and lots sold in the first 5 steps, of a policy.
template <class P> void cash_stats(const ExecutionParams& p, P&& pi, double& mean, double& sd, double& early) {
  double s = 0, s2 = 0, e = 0; const int n = 2000;
  for (int i = 0; i < n; ++i) {
    ExecutionEnv env(p); Rng rng(5'000'000 + i); auto o = env.reset(rng);
    for (;;) { const auto inv = o.inventory; const auto st = env.step(pi(o), rng); if (o.t < 5) e += inv - st.obs.inventory - (st.done && st.obs.t >= p.horizon ? 0 : 0); o = st.obs; if (st.done) break; }
    const double c = env.cash() - p.lots * p.start_price; s += c; s2 += c * c;
  }
  mean = s / n; sd = std::sqrt(s2 / n - mean * mean); early = e / n;
}
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  std::printf("%-6s | %-26s | %-26s | %-26s | %s\n", "lambda", "DP optimum: cash mean/sd/early", "TileQ 40k", "model-based 300", "share of risk-adj. optimum (TileQ, MB)");
  for (double lam : {0.0, 0.01, 0.05, 0.2, 1.0}) {
    ExecutionParams p; p.risk_aversion = lam;
    ExecutionEnv env(p); const ExecutionDP dp(p);
    auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
    const double gopt = compare(env, opt, twap, 2000, 5'000'000).mean;
    AgentConfig<3> cfg; cfg.eps_decay_episodes = 20000;
    ExecutionEnv e2(p); Agent<ExecutionEnv> agent(e2, cfg);
    Trainer<ExecutionEnv> tr(agent, e2, {.eval_every = 5000, .eval_episodes = 1000, .detect_drift = false});
    tr.run(40000);
    auto rl = [&](const ExecutionObs& o) { return tr.deployed_act(o); };
    ModelBasedExecutionAgent mb(p, {.seed = 1}); Rng r(42); ExecutionEnv e3(p); train_model_based(mb, e3, 300, r); mb.replan();
    auto mbp = [&](const ExecutionObs& o) { return mb.act(o, false); };
    double m[3], sd[3], early[3];
    cash_stats(p, opt, m[0], sd[0], early[0]); cash_stats(p, rl, m[1], sd[1], early[1]); cash_stats(p, mbp, m[2], sd[2], early[2]);
    std::printf("%-6.2f | %+7.2f / %5.1f / %5.1f lots | %+7.2f / %5.1f / %5.1f lots | %+7.2f / %5.1f / %5.1f lots | %5.0f%% %5.0f%%  (MB holding est %.5f, true %.5f)\n",
                lam, m[0], sd[0], early[0], m[1], sd[1], early[1], m[2], sd[2], early[2],
                100 * compare(env, rl, twap, 2000, 5'000'000).mean / gopt, 100 * compare(env, mbp, twap, 2000, 5'000'000).mean / gopt,
                mb.model().holding, lam * p.noise_sd * p.noise_sd);
  }
}
