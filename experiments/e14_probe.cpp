// E14: tail risk of mean-optimal, risk-averse and learned policies (cash outcomes).
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
template <class Env, class P> void row(const char* name, const Env& env, P&& pi, double base_mean) {
  const auto r = episode_returns(env, pi, 20000, 7'000'000);
  const double m = std::accumulate(r.begin(), r.end(), 0.0) / r.size();
  double v = 0; for (double x : r) v += (x - m) * (x - m); v /= r.size() - 1;
  std::printf("  %-30s mean %+8.2f (gain %+6.2f)  sd %6.2f  CVaR5%% %+8.2f  CVaR1%% %+8.2f  worst %+8.2f\n", name, m, m - base_mean, std::sqrt(v),
              cvar(r, 0.05), cvar(r, 0.01), *std::min_element(r.begin(), r.end()));
}
int main() {
  std::printf("EXECUTION (cash outcome = reward with risk_aversion 0)\n");
  ExecutionParams p; const ExecutionEnv env(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const auto tw = episode_returns(env, twap, 20000, 7'000'000);
  const double twm = std::accumulate(tw.begin(), tw.end(), 0.0) / tw.size();
  row("TWAP", env, twap, twm);
  for (double lam : {0.0, 0.05, 0.2, 1.0}) {
    ExecutionParams q = p; q.risk_aversion = lam; const ExecutionDP dp(q);
    char name[64]; std::snprintf(name, sizeof name, "DP optimum, lambda %.2f", lam);
    row(name, env, [&](const ExecutionObs& o) { return dp.act(o); }, twm);
  }
  for (int k = 0; k < 2; ++k) {
    AgentConfig<3> cfg; cfg.learner = k ? LearnerKind::ActorCritic : LearnerKind::TileQ; cfg.eps_decay_episodes = 20000;
    ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 5000, .eval_episodes = 500, .detect_drift = false});
    tr.run(40000);
    row(k ? "ActorCritic 40k (lambda 0)" : "TileQ 40k (lambda 0)", env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twm);
  }
  std::printf("MARKET MAKING (cash outcome = reward with risk_aversion 0)\n");
  MarketMakingParams mp; MarketMakingParams cashp = mp; cashp.risk_aversion = 0; const MarketMakingEnv menv(cashp);
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  const auto sr = episode_returns(menv, sym, 20000, 7'000'000);
  const double sm = std::accumulate(sr.begin(), sr.end(), 0.0) / sr.size();
  row("symmetric quotes", menv, sym, sm);
  for (double ra : {0.0, 0.05, 0.2}) {
    MarketMakingParams q = mp; q.risk_aversion = ra; const MarketMakingDP dp(q);
    char name[64]; std::snprintf(name, sizeof name, "DP optimum, risk %.2f", ra);
    row(name, menv, [&](const MarketMakingObs& o) { return dp.act(o); }, sm);
  }
  {
    AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000;
    MarketMakingEnv e(mp); auto agent = std::make_unique<Agent<MarketMakingEnv>>(e, cfg);
    Trainer<MarketMakingEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .detect_drift = false});
    tr.run(20000);
    row("ActorCritic 20k (risk 0.05)", menv, [&](const MarketMakingObs& o) { return tr.deployed_act(o); }, sm);
  }
}
