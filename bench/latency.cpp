// latency.cpp — tail-latency probe for the full query pipeline.
//
// Google Benchmark reports means; trading-style SLOs care about p99/p99.9.
// This harness records every query's wall time into a preallocated array
// (no allocation inside the timed loop), pins itself to one core to keep
// scheduler migrations out of the tail, and prints percentiles.
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "fixtures.hpp"

using namespace rag;
using namespace rag::bench;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const std::size_t iters = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 20000;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(1, &set);
  const bool pinned = sched_setaffinity(0, sizeof(set), &set) == 0;
  const auto qs = text_queries(1024);
  std::vector<std::int64_t> ns(iters);  // preallocated

  std::printf("corpus: %zu docs | pinned: %s | iters: %zu\n", corpus().docs.size(), pinned ? "cpu1" : "no", iters);
  std::printf("%-12s %8s %8s %8s %8s %8s %8s %9s\n", "mode", "chunks", "p50", "p90", "p99", "p99.9", "max", "QPS/core");
  for (auto mode : {DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf}) {
    auto& e = engine(mode);
    auto ctx = e.make_context();
    for (std::size_t i = 0; i < 2000; ++i) (void)e.query(qs[i & 1023], 8, ctx);  // warm caches + branch predictors
    volatile std::size_t sink = 0;
    const auto t_all = Clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
      const auto t0 = Clock::now();
      const auto r = e.query(qs[i & 1023], 8, ctx);
      const auto t1 = Clock::now();
      sink = sink + r.prompt.prompt.size();
      ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t_all).count();
    std::sort(ns.begin(), ns.end());
    auto pct = [&](double p) { return double(ns[std::min(iters - 1, std::size_t(p * double(iters)))]) / 1000.0; };
    std::printf("%-12s %8u %7.1fus %7.1fus %7.1fus %7.1fus %7.1fus %9.0f\n",
                mode == DenseMode::Exact ? "exact" : mode == DenseMode::Int8Rerank ? "int8+rerank" : "ivf",
                e.size(), pct(0.50), pct(0.90), pct(0.99), pct(0.999), double(ns.back()) / 1000.0, double(iters) / secs);
  }
}
