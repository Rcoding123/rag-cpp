// E28: out-of-sample validation of execution strategies on the stress suite
// (validation.hpp), 4,000 sealed test episodes per scenario.
// Strategies:
//   plan        the DP plan for the *nominal* model (a trader's fixed model)
//   ensemble    median vote of the 8 averaged networks' tables (E25), trained in the nominal simulator only
//   single      one averaged network's table (seed 1)
//   model-50    model-based agent re-learned on 50 episodes of the scenario (training seeds, not test seeds)
//   model-200   the same with 200 episodes
// Usage: e28_validation TABLE_DIR. Prints CSV:
//   scenario,strategy,gain,ci_lo,ci_hi,optimal_gain,share,cvar5,p_worse,model_check_z
#include <cstdio>
#include <string>
#include <vector>

#include "rag/agent/model_based.hpp"
#include "rag/agent/policy_table.hpp"
#include "rag/agent/validation.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string dir = argv[1];
  const ExecutionParams nominal;
  const ExecutionDP plan(nominal);
  std::vector<ExecutionPolicyTable> members;
  for (int s = 1; s <= 8; ++s) members.push_back(ExecutionPolicyTable::load(dir + "/seed" + std::to_string(s) + "_bins256.tbl"));
  const auto ensemble = ExecutionPolicyTable::median(members);
  constexpr std::uint32_t kN = 4000;
  for (const auto& sc : validation::stress_suite(nominal)) {
    const ExecutionDP best(sc.params);
    const double optimal = validation::measure(sc.params, [&](const ExecutionObs& o) { return best.act(o); }, kN, validation::kSealedSeed, 1e300).gain;
    auto row = [&](const char* name, auto&& policy, double z) {
      const auto m = validation::measure(sc.params, policy, kN, validation::kSealedSeed, optimal);
      std::printf("%s,%s,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.4f,%.2f\n", sc.name.c_str(), name, m.gain, m.gain_ci.lo, m.gain_ci.hi,
                  m.optimal_gain, m.share, m.cvar5, m.p_worse, z);
      std::fflush(stdout);
    };
    row("plan", [&](const ExecutionObs& o) { return plan.act(o); }, 0);
    row("ensemble", [&](const ExecutionObs& o) { return ensemble.act(o); }, 0);
    row("single", [&](const ExecutionObs& o) { return members[0].act(o); }, 0);
    for (std::uint32_t budget : {50u, 200u}) {
      ModelBasedExecutionAgent mb(sc.params, {.cost_model = CostModel::Shrunk, .seed = 1});
      ExecutionEnv train(sc.params);
      Rng r(4242);  // training seeds: unrelated to the sealed test range
      train_model_based(mb, train, budget, r);
      mb.replan();
      const double z = mb.check_model(train, 200, 600'000'000).z;
      row(budget == 50 ? "model-50" : "model-200", [&](const ExecutionObs& o) { return mb.act(o, false); }, z);
    }
  }
}
