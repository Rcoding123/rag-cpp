// E13: regime switches (volatility 1 <-> 2) in market making: drift detection + policy bank, unchanged.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  MarketMakingParams pa, pb = pa; pb.decay = 0.3;
  const MarketMakingEnv ea(pa), eb(pb); const MarketMakingDP da(pa), db(pb);
  const double oa = compare(ea, [&](const MarketMakingObs& o) { return da.act(o); }, sym, 1000, 5'000'000).mean;
  const double ob = compare(eb, [&](const MarketMakingObs& o) { return db.act(o); }, sym, 1000, 5'000'000).mean;
  // How different are the regimes? Each optimum scored in the other regime.
  std::printf("regime A (decay 1) optimum gain %.1f; B (decay 0.3, deep book) %.1f. A's optimum in B: %.0f%%; B's in A: %.0f%%\n", oa, ob,
              100 * compare(eb, [&](const MarketMakingObs& o) { return da.act(o); }, sym, 1000, 5'000'000).mean / ob,
              100 * compare(ea, [&](const MarketMakingObs& o) { return db.act(o); }, sym, 1000, 5'000'000).mean / oa);
  for (int two = 0; two < 2; ++two) for (int bank = 1; bank >= 0; --bank) {
    AgentConfig<2> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 5000; cfg.seed = seed;
    MarketMakingEnv env(pa); auto agent = std::make_unique<Agent<MarketMakingEnv>>(env, cfg);
    TrainConfig tc{.eval_every = 500, .eval_episodes = 500, .train_seed = seed, .policy_bank = bank == 1}; tc.drift.two_sided = two == 1;
    Trainer<MarketMakingEnv> tr(*agent, env, tc);
    tr.run(10000);
    std::printf("%s, bank %s, seed %llu: before switches %.0f%%\n", two ? "two-sided" : "one-sided", bank ? "on " : "off", (unsigned long long)seed,
                100 * compare(ea, [&](const MarketMakingObs& o) { return tr.deployed_act(o); }, sym, 1000, 5'000'000).mean / oa);
    for (int sw = 0; sw < 4; ++sw) {
      const bool to_b = sw % 2 == 0;
      env.set_decay(to_b ? 0.3 : 1.0);
      const auto& e = to_b ? eb : ea; const double o = to_b ? ob : oa;
      std::printf("  -> %s:", to_b ? "B" : "A");
      unsigned done = 0; std::string alarms, recalls;
      for (unsigned m : {200u, 1000u, 5000u}) {
        const auto rep = tr.run(m - done); done = m;
        for (auto x : rep.drift_alarms) alarms += " +" + std::to_string(static_cast<long>(x) - static_cast<long>(agent->episodes() - m));
        for (auto x : rep.bank_recalls) recalls += " r";
        std::printf("  +%u: %4.0f%%", m, 100 * compare(e, [&](const MarketMakingObs& ob2) { return tr.deployed_act(ob2); }, sym, 1000, 5'000'000).mean / o);
      }
      std::printf("   alarms:%s recalls:%s\n", alarms.c_str(), recalls.c_str());
    }
  }
}
