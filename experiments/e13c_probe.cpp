// E13c: actor-critic re-exploration after a regime switch (bank off): eps-uniform vs raised temperature.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  MarketMakingParams pa, pb = pa; pb.decay = 0.3;
  const MarketMakingEnv ea(pa), eb(pb); const MarketMakingDP da(pa), db(pb);
  const double oa = compare(ea, [&](const MarketMakingObs& o) { return da.act(o); }, sym, 1000, 5'000'000).mean;
  const double ob = compare(eb, [&](const MarketMakingObs& o) { return db.act(o); }, sym, 1000, 5'000'000).mean;
  for (double temp : {0.0, 2.0, 5.0})
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000; cfg.seed = seed; cfg.ac_reexplore_temperature = temp;
      MarketMakingEnv env(pa); auto agent = std::make_unique<Agent<MarketMakingEnv>>(env, cfg);
      TrainConfig tc{.eval_every = 500, .eval_episodes = 500, .train_seed = seed, .policy_bank = false}; tc.drift.two_sided = true;
      Trainer<MarketMakingEnv> tr(*agent, env, tc);
      tr.run(10000);
      std::printf("reexplore %s seed %llu:", temp == 0 ? "eps-uniform  " : (temp == 2 ? "temperature x3" : "temperature x6"), (unsigned long long)seed);
      for (int sw = 0; sw < 2; ++sw) {
        const bool to_b = sw == 0; env.set_decay(to_b ? 0.3 : 1.0);
        const auto& e = to_b ? eb : ea; const double o = to_b ? ob : oa;
        std::printf("  %s", to_b ? "B" : "A");
        unsigned done = 0;
        for (unsigned m : {1000u, 5000u, 10000u}) { tr.run(m - done); done = m;
          std::printf(" +%u:%4.0f%%", m, 100 * compare(e, [&](const MarketMakingObs& x) { return tr.deployed_act(x); }, sym, 1000, 5'000'000).mean / o); }
      }
      std::printf("\n");
    }
}
