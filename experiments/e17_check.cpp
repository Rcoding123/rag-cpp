#include <cmath>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
#include "rag/agent/variance_reduction.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p; const ExecutionEnv plain(p); const ExecutionDP dp(p);
  const InnovationControlledEnv cv(ExecutionEnv(p), p.alpha);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
  for (int k = 0; k < 2; ++k) {
    auto run = [&](auto&& pi) {
      const auto a = evaluate(plain, pi, 20000, 1), b = evaluate(cv, pi, 20000, 1);
      std::printf("  %-8s plain %+7.3f (sd %5.2f)   controlled %+7.3f (sd %5.2f)\n", k ? "optimal" : "TWAP", a.mean, a.std_error * std::sqrt(20000.0), b.mean, b.std_error * std::sqrt(20000.0));
    };
    if (k) run(opt); else run(twap);
  }
  const auto g1 = compare(plain, opt, twap, 20000, 1), g2 = compare(cv, opt, twap, 20000, 1);
  std::printf("optimal gain over TWAP: plain %.3f +- %.3f, controlled %.3f +- %.3f; gain / return-sd: plain %.2f, controlled ?\n", g1.mean, g1.std_error, g2.mean, g2.std_error, 0.27);
}
