#include <cmath>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  const unsigned n = 20000;
  { ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
    auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
    auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
    const auto r = evaluate(env, opt, n, 1); const auto g = compare(env, opt, twap, n, 1);
    std::printf("execution:     optimal gain over baseline %.2f, return sd %.1f, paired-difference sd %.1f -> gain/return-sd %.2f\n",
                g.mean, r.std_error * std::sqrt(n), g.std_error * std::sqrt(n), g.mean / (r.std_error * std::sqrt(n))); }
  { MarketMakingParams p; MarketMakingEnv env(p); const MarketMakingDP dp(p);
    auto opt = [&](const MarketMakingObs& o) { return dp.act(o); };
    auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
    const auto r = evaluate(env, opt, n, 1); const auto g = compare(env, opt, sym, n, 1);
    std::printf("market making: optimal gain over baseline %.2f, return sd %.1f, paired-difference sd %.1f -> gain/return-sd %.2f\n",
                g.mean, r.std_error * std::sqrt(n), g.std_error * std::sqrt(n), g.mean / (r.std_error * std::sqrt(n))); }
}
