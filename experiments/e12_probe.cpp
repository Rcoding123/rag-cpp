// E12: market making with the same agent and trainer.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  MarketMakingParams p;
  {  // accounting identity with risk_aversion = 0
    MarketMakingParams q = p; q.risk_aversion = 0; MarketMakingEnv env(q); Rng rng(3); double worst = 0;
    for (int e = 0; e < 200; ++e) { env.reset(rng); double sum = 0; for (;;) { auto st = env.step(rng.below(9), rng); sum += st.reward; if (st.done) break; }
      worst = std::max(worst, std::abs(sum - env.cash())); }
    std::printf("accounting identity worst error %.2e\n", worst);
  }
  const MarketMakingEnv env(p); const MarketMakingDP dp(p);
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  auto optimal = [&](const MarketMakingObs& o) { return dp.act(o); };
  auto skew_rule = [&](const MarketMakingObs& o) {  // hand-written: lean against inventory
    if (o.q > 2) return MarketMakingEnv::action(2, 0);
    if (o.q < -2) return MarketMakingEnv::action(0, 2);
    return MarketMakingEnv::kSymmetricAction; };
  const auto sim_opt = evaluate(env, optimal, 20000, 1);
  std::printf("DP value %.3f, simulated %.3f +- %.3f; symmetric exact %.3f\n", dp.expected_start_value(), sim_opt.mean, sim_opt.std_error, dp.policy_value(sym));
  const double opt = compare(env, optimal, sym, 2000, 5'000'000).mean;
  std::printf("optimal gain over symmetric quoting: %+.3f per episode (paired)\n", opt);
  std::printf("hand-written skew rule: %.0f%% of it\n", 100 * compare(env, skew_rule, sym, 2000, 5'000'000).mean / opt);
  for (int k = 0; k < 2; ++k) {
    AgentConfig<2> cfg; cfg.learner = k ? LearnerKind::ActorCritic : LearnerKind::TileQ; cfg.eps_decay_episodes = 5000; cfg.seed = seed;
    MarketMakingEnv e(p); auto agent = std::make_unique<Agent<MarketMakingEnv>>(e, cfg);
    Trainer<MarketMakingEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
    std::printf("%s:", k ? "ActorCritic" : "TileQ");
    unsigned done = 0;
    for (unsigned m : {1000u, 5000u, 20000u, 50000u}) {
      tr.run(m - done); done = m;
      std::printf("  %5u: %4.0f%%", m, 100 * compare(env, [&](const MarketMakingObs& o) { return tr.deployed_act(o); }, sym, 2000, 5'000'000).mean / opt);
    }
    std::printf("\n  quotes at t=25 by inventory (bid/ask distance index; DP below):\n   q:  ");
    for (int q = -8; q <= 8; q += 2) std::printf("%5d", q);
    std::printf("\n   agent");
    for (int q = -8; q <= 8; q += 2) { const Action a = tr.deployed_act({25, q}); std::printf("  %u/%u", a / 3, a % 3); }
    std::printf("\n   DP   ");
    for (int q = -8; q <= 8; q += 2) { const Action a = dp.act({25, q}); std::printf("  %u/%u", a / 3, a % 3); }
    std::printf("\n");
  }
}
