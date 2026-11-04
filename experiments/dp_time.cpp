#include <chrono>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p;
  for (unsigned grid : {61u, 121u}) {
    auto t0 = std::chrono::steady_clock::now();
    double sink = 0;
    const int reps = 5;
    for (int r = 0; r < reps; ++r) { const ExecutionDP dp(p, grid); sink += dp.expected_start_value(); }
    auto t1 = std::chrono::steady_clock::now();
    const ExecutionDP dp(p, grid);
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) sink += dp.policy_value([&](const ExecutionObs& o) { return dp.act(o); });
    auto t3 = std::chrono::steady_clock::now();
    Rng rng(1); std::uint64_t acts = 0; auto t4 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200000; ++i) acts += dp.act({rng.below(20), rng.below(101), 0.1 * rng.normal()});
    auto t5 = std::chrono::steady_clock::now();
    std::printf("grid %3u: solve %.1f ms   policy_value(greedy) %.1f ms   act %.0f ns   (%g %llu)\n", grid,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / reps,
                std::chrono::duration<double, std::milli>(t3 - t2).count() / reps,
                std::chrono::duration<double, std::nano>(t5 - t4).count() / 200000, sink, (unsigned long long)acts);
  }
}
