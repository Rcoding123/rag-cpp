// E9b: change detection + reset vs constant forgetting. Flip at 300; stationary false alarms.
#include <cstdio>
#include <cstdlib>
#include <string>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
using Cfg = ModelBasedExecutionAgent::Config;
struct Variant { const char* name; Cfg cfg; };
static Variant variants[] = {
  {"no forgetting", {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk}},
  {"forget 0.99", {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk, .forget = 0.99}},
  {"forget 0.95", {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk, .forget = 0.95}},
  {"detect + reset", {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk, .detect_drift = true}},
  {"detect, default gaps", {.cost_model = CostModel::Shrunk, .detect_drift = true}},
};
template <class F> double run(ModelBasedExecutionAgent& a, ExecutionEnv& env, Rng& rng, unsigned n, F&&) {
  double cum = 0;
  for (unsigned i = 0; i < n; ++i) {
    auto o = env.reset(rng); double total = 0;
    for (;;) { const Action act = a.act(o, true); a.note_action(o, act); const auto st = env.step(act, rng);
               a.observe(o, act, st.reward, st.obs, st.done); total += st.reward; o = st.obs; if (st.done) break; }
    a.end_episode(); cum += total;
  }
  return cum;
}
int main(int argc, char** argv) {
  const unsigned seeds = argc > 1 ? std::atoi(argv[1]) : 3;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams pa, pb = pa; pb.alpha = -1.0;
  const ExecutionEnv ea(pa), eb(pb); const ExecutionDP da(pa), db(pb);
  const double ob = compare(eb, [&](const ExecutionObs& o) { return db.act(o); }, twap, 600, 5'000'000).mean;
  const double gain_b = db.expected_start_value() - db.policy_value(twap);
  std::printf("FLIP at 300 (alpha 1 -> -1): share of new optimum at +N, online gain over 300 eps after (of 300), alarms\n");
  for (const auto& v : variants)
    for (unsigned seed = 1; seed <= seeds; ++seed) {
      Cfg c = v.cfg; c.seed = seed;
      ModelBasedExecutionAgent a(pa, c); ExecutionEnv env(pa); Rng rng(100 + seed);
      run(a, env, rng, 300, 0);
      env.set_alpha(-1.0);
      auto pol = [&](const ExecutionObs& o) { return a.act(o, false); };
      std::printf("  %-21s s%u", v.name, seed);
      unsigned done = 0; double online = 0;
      for (unsigned m : {10u, 20u, 50u, 100u, 300u}) { online += run(a, env, rng, m - done, 0); done = m;
        std::printf(" +%u:%5.0f%%", m, 100 * compare(eb, pol, twap, 600, 5'000'000).mean / ob); }
      std::string al; for (auto e : a.drift_alarms()) al += " " + std::to_string(static_cast<long>(e) - 300);
      std::printf(" | online %+5.0f | alarms (rel. to flip):%s\n", (online - 300 * db.policy_value(twap)) / gain_b, al.c_str());
    }
  std::printf("\nSTATIONARY 3000 episodes: false alarms, final share\n");
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base, noisy = base, hard = base;
  noisy.noise_sd = 0.2; hard.impact_exponent = 3.0; hard.impact = 0.01 * 25 / 125.0; hard.fixed_cost = 1.5;
  for (const Case& m : {Case{"default", base}, Case{"noise 0.2", noisy}, Case{"q^3 + fixed", hard}}) {
    ExecutionEnv env0(m.p); const ExecutionDP dp(m.p);
    const double opt = compare(env0, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 600, 5'000'000).mean;
    for (const auto& v : variants) {
      std::printf("  %-12s %-21s", m.name, v.name);
      for (unsigned seed = 1; seed <= seeds; ++seed) {
        Cfg c = v.cfg; c.seed = seed;
        ModelBasedExecutionAgent a(m.p, c); ExecutionEnv env(m.p); Rng rng(300 + seed);
        run(a, env, rng, 3000, 0);
        std::printf("  [%zu alarms, %5.1f%%]", a.drift_alarms().size(),
                    100 * compare(env0, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 600, 5'000'000).mean / opt);
      }
      std::printf("\n");
    }
  }
}
