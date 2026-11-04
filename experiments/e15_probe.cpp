// E15: model-based market maker vs actor-critic; exponential and non-exponential fill curves.
#include <cstdio>
#include <memory>
#include <numeric>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  MarketMakingParams expo, odd = expo;
  odd.fill_probs = {0.85, 0.15, 0.10};  // steep: only the tightest quote fills often
  for (const auto* pp : {&expo, &odd}) {
    const MarketMakingParams& p = *pp;
    const MarketMakingEnv env(p); const MarketMakingDP dp(p);
    const double opt = compare(env, [&](const MarketMakingObs& o) { return dp.act(o); }, sym, 1000, 5'000'000).mean;
    const double v_sym = dp.policy_value(sym), gain = dp.expected_start_value() - v_sym;
    std::printf("%s fill curve (%.2f %.2f %.2f): optimal gain over symmetric %.1f\n", pp == &expo ? "exponential" : "steep",
                env.fill_probability_at(0), env.fill_probability_at(1), env.fill_probability_at(2), opt);
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      ModelBasedMarketMaker mb(p, {.seed = seed}); MarketMakingEnv e(p); Rng rng(seed);
      std::printf("  model-based seed %llu:", (unsigned long long)seed);
      unsigned done = 0; double cum = 0;
      for (unsigned m : {3u, 5u, 10u, 20u, 50u, 1000u}) {
        for (double r : train_model_based(mb, e, m - done, rng)) cum += r - v_sym;
        done = m;
        if (mb.has_plan()) std::printf("  %4u: %4.0f%%", m, 100 * compare(env, [&](const MarketMakingObs& o) { return mb.act(o, false); }, sym, 1000, 5'000'000).mean / opt);
        else std::printf("  %4u:   -  ", m);
      }
      std::printf("  | online %+.0f of 1000 | fills %.2f %.2f %.2f vol %.2f\n", cum / gain, mb.estimated_fill(0), mb.estimated_fill(1), mb.estimated_fill(2), mb.estimated_volatility());
      AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000; cfg.seed = seed;
      MarketMakingEnv e2(p); auto agent = std::make_unique<Agent<MarketMakingEnv>>(e2, cfg);
      Trainer<MarketMakingEnv> tr(*agent, e2, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
      const auto rep = tr.run(1000);
      double cum2 = 0; for (double r : rep.returns) cum2 += r - v_sym;
      std::printf("  actor-critic seed %llu: 1000: %4.0f%% (greedy learner) | online %+.0f of 1000\n", (unsigned long long)seed,
                  100 * compare(env, [&](const MarketMakingObs& o) { return agent->act(o, false); }, sym, 1000, 5'000'000).mean / opt, cum2 / gain);
    }
  }
}
