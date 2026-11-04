// E10: off-policy evaluation. Log episodes under behaviour = TWAP + eps random;
// estimate a target policy's value without running it. Truth = exact DP evaluation.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
struct LogStep { ExecutionObs o; Action a; double r; ExecutionObs next; bool done; double b; };
using Episode = std::vector<LogStep>;
int main(int argc, char** argv) {
  const double gamma = argc > 1 ? std::atof(argv[1]) : 0.0;
  const double eps = argc > 2 ? std::atof(argv[2]) : 0.2;
  const unsigned reps = argc > 3 ? std::atoi(argv[3]) : 30;
  ExecutionParams p; p.permanent_impact = gamma;
  const ExecutionDP truth_dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto optimal = [&](const ExecutionObs& o) { return truth_dp.act(o); };
  auto rule = [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; };
  auto nudge = [&](const ExecutionObs& o) { return o.t < 5 && o.signal < -0.1 ? Action{3} : ExecutionEnv::kTwapAction; };  // near-TWAP
  const double scale = truth_dp.expected_start_value() - truth_dp.policy_value(twap);  // optimal gain: error unit
  std::printf("gamma %.3f eps %.2f: errors in %% of the optimal gain (%.2f); %u replications\n", gamma, eps, scale, reps);
  struct T { const char* name; std::function<Action(const ExecutionObs&)> pi; };
  std::vector<T> targets = {{"TWAP (=behaviour base)", twap}, {"near-TWAP nudge", nudge}, {"sign rule", rule}, {"DP optimum", optimal}};
  for (unsigned N : {100u, 1000u, 10000u}) {
    std::printf(" N = %u logged episodes\n", N);
    for (const auto& tg : targets) {
      const double truth = truth_dp.policy_value(tg.pi);
      double se[5] = {}, sb[5] = {}; unsigned zero_w = 0; double ess_sum = 0;
      for (unsigned rep = 0; rep < reps; ++rep) {
        // Log N episodes under the behaviour policy.
        ExecutionEnv env(p); Rng rng(1000 + 7919 * rep + N), pick(5 + rep);
        std::vector<Episode> logs(N);
        ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = false, .seed = 1});
        for (auto& ep : logs) {
          auto o = env.reset(rng);
          for (;;) {
            const Action base = ExecutionEnv::kTwapAction;
            const Action a = pick.uniform() < eps ? static_cast<Action>(pick.below(5)) : base;
            const double b = (a == base ? 1 - eps : 0.0) + eps / 5;
            mb.note_action(o, a);
            const auto st = env.step(a, rng);
            mb.observe(o, a, st.reward, st.obs, st.done);
            ep.push_back({o, a, st.reward, st.obs, st.done, b});
            o = st.obs;
            if (st.done) break;
          }
          mb.end_episode();
        }
        mb.replan();
        const auto model_values = mb.plan()->evaluate_policy(tg.pi);
        const double dm = model_values.start_value();
        std::vector<LoggedEpisode<ExecutionObs>> L;
        for (const auto& ep : logs) { LoggedEpisode<ExecutionObs> e; for (const auto& st : ep) e.push_back({st.o, st.a, st.r, st.b}); L.push_back(std::move(e)); }
        const auto ope = off_policy_estimates(L, tg.pi, [&](const ExecutionObs& o, Action a) { return model_values.q(o.t, o.inventory, o.signal, a); },
                                              [&](const ExecutionObs& o) { return model_values.value(o.t, o.inventory, o.signal); }, p.horizon);
        const double pdis = ope.pdis, wpdis = ope.wpdis, dr = ope.dr; zero_w += ope.uncovered_steps; ess_sum += ope.min_ess;
        const double est[4] = {dm, pdis, wpdis, dr};
        for (int k = 0; k < 4; ++k) { const double e = 100 * (est[k] - truth) / scale; sb[k] += e; se[k] += e * e; }
      }
      std::printf("   %-24s", tg.name);
      const char* names[] = {"DM", "PDIS", "WPDIS", "DR"};
      for (int k = 0; k < 4; ++k) std::printf("  %s bias %+7.1f rmse %7.1f |", names[k], sb[k] / reps, std::sqrt(se[k] / reps));
      std::printf(" uncovered steps %.1f of %u, min ESS %.1f\n", zero_w / double(reps), p.horizon, ess_sum / reps);
    }
  }
}
