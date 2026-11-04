// E9: model-based agent with forgetting, market flips (alpha 1 -> -1) after 300 episodes.
#include <cstdio>
#include <cstdlib>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const unsigned seeds = argc > 1 ? std::atoi(argv[1]) : 3;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams pa, pb = pa; pb.alpha = -1.0;
  const ExecutionEnv ea(pa), eb(pb); const ExecutionDP da(pa), db(pb);
  const double oa = compare(ea, [&](const ExecutionObs& o) { return da.act(o); }, twap, 600, 5'000'000).mean;
  const double ob = compare(eb, [&](const ExecutionObs& o) { return db.act(o); }, twap, 600, 5'000'000).mean;
  const double gain_b = db.expected_start_value() - db.policy_value(twap);
  std::printf("%-7s | share before flip | share of new optimum after the flip at +N episodes        | online gain, 300 eps after flip\n", "forget");
  std::printf("%-7s | %16s | %7s %7s %7s %7s %7s %7s |\n", "", "(ep 300)", "+5", "+10", "+20", "+50", "+100", "+300");
  for (double f : {1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9}) {
    for (unsigned seed = 1; seed <= seeds; ++seed) {
      ModelBasedExecutionAgent a(pa, {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk, .forget = f, .seed = seed});
      ExecutionEnv env(pa); Rng rng(100 + seed);
      auto run = [&](unsigned n) {
        double cum = 0;
        for (unsigned i = 0; i < n; ++i) {
          auto o = env.reset(rng); double total = 0;
          for (;;) { const Action act = a.act(o, true); a.note_action(o, act); const auto st = env.step(act, rng);
                     a.observe(o, act, st.reward, st.obs, st.done); total += st.reward; o = st.obs; if (st.done) break; }
          a.end_episode(); cum += total;
        }
        return cum;
      };
      run(300);
      auto pol = [&](const ExecutionObs& o) { return a.act(o, false); };
      std::printf("%-7.3f | %15.0f%% |", f, 100 * compare(ea, pol, twap, 600, 5'000'000).mean / oa);
      env.set_alpha(-1.0);
      unsigned done = 0; double online = 0;
      for (unsigned m : {5u, 10u, 20u, 50u, 100u, 300u}) {
        online += run(m - done); done = m;
        std::printf(" %6.0f%%", 100 * compare(eb, pol, twap, 600, 5'000'000).mean / ob);
      }
      std::printf(" | %+7.0f of %u (TWAP = 0)\n", (online - 300 * db.policy_value(twap)) / gain_b, 300u);
    }
  }
}
