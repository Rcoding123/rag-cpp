// OF1: does the ES simulator reproduce the stylized facts of real futures
// tape? Simulates `hours` of trading (from the open) and reports, with
// typical ES targets in brackets where known:
//   activity      trades per second, mean trade size, volume per minute
//   book          share of time the spread is 1 tick, mean best-level size
//   returns       1-minute sd (ticks), excess kurtosis (fat tails, > 0),
//                 lag-1 autocorrelation (~0), |r| autocorrelation (clustering, > 0)
//   order flow    trade-sign autocorrelation (long memory, slowly decaying),
//                 R^2 of 10 s mid change on 10 s delta (order flow drives price)
//   predictability corr(30 s delta, next 30 s return) — what a strategy could use
// Usage: of1_stylized_facts [hours=2] [seed=1]
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "rag/orderflow/simulator.hpp"

using namespace rag::orderflow;

static double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
static double corr(const std::vector<double>& a, const std::vector<double>& b) {
  const double ma = mean(a), mb = mean(b);
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb); }
  return saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : 0;
}
static double autocorr(const std::vector<double>& v, std::size_t lag) {
  if (v.size() <= lag) return 0;
  return corr(std::vector<double>(v.begin(), v.end() - lag), std::vector<double>(v.begin() + lag, v.end()));
}

int main(int argc, char** argv) {
  const double hours = argc > 1 ? std::atof(argv[1]) : 2.0;
  const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  const SimParams p;
  EsSimulator sim(p, seed);
  const auto steps = static_cast<std::size_t>(hours * 3600 / p.dt);
  const std::size_t per10 = static_cast<std::size_t>(10 / p.dt), per30 = 3 * per10, per60 = 6 * per10;
  std::vector<double> mids, deltas_step, signs;
  double trades = 0, volume = 0, spread1 = 0, l1 = 0;
  for (std::size_t s = 0; s < steps; ++s) {
    sim.step();
    double d = 0;
    for (const auto& t : sim.trades()) {
      trades += 1;
      volume += t.size;
      const double sg = t.aggressor == Side::Buy ? 1 : -1;
      d += sg * t.size;
      signs.push_back(sg);
    }
    deltas_step.push_back(d);
    mids.push_back(sim.mid());
    spread1 += sim.spread() == 1;
    l1 += 0.5 * (sim.quote().bid_size[0] + sim.quote().ask_size[0]);
  }
  auto sum_window = [&](std::size_t from, std::size_t n) { double s = 0; for (std::size_t i = from; i < from + n; ++i) s += deltas_step[i]; return s; };
  std::vector<double> r1, r10, d10, d30, f30;
  for (std::size_t i = per60; i < mids.size(); i += per60) r1.push_back(mids[i] - mids[i - per60]);
  for (std::size_t i = per10; i < mids.size(); i += per10) { r10.push_back(mids[i] - mids[i - per10]); d10.push_back(sum_window(i - per10, per10)); }
  for (std::size_t i = per30; i + per30 < mids.size(); i += per30) { d30.push_back(sum_window(i - per30, per30)); f30.push_back(mids[i + per30] - mids[i]); }
  const double sd1 = std::sqrt(mean([&] { std::vector<double> v; for (double x : r1) v.push_back(x * x); return v; }()) - mean(r1) * mean(r1));
  double m4 = 0;
  for (double x : r1) m4 += std::pow(x - mean(r1), 4);
  m4 /= r1.size();
  std::vector<double> abs1;
  for (double x : r1) abs1.push_back(std::abs(x));
  const double c = corr(d10, r10);
  const double secs = hours * 3600;
  std::printf("simulated %.1f h (%zu steps of %.0f ms), seed %llu\n", hours, steps, p.dt * 1000, (unsigned long long)seed);
  std::printf("activity : %.1f trades/s, mean trade %.2f contracts, %.0f contracts/min\n", trades / secs, volume / trades, volume / secs * 60);
  std::printf("book     : spread = 1 tick %.1f%% of the time, mean best-level size %.0f\n", 100 * spread1 / steps, l1 / steps);
  std::printf("returns  : 1-min sd %.2f ticks, excess kurtosis %.2f, lag-1 autocorr %+.3f, |r| autocorr lag1 %+.3f lag5 %+.3f\n", sd1,
              m4 / std::pow(sd1, 4) - 3, autocorr(r1, 1), autocorr(abs1, 1), autocorr(abs1, 5));
  std::printf("flow     : trade-sign autocorr lag1 %+.3f lag10 %+.3f lag100 %+.3f; R^2(10 s dmid ~ 10 s delta) %.3f\n", autocorr(signs, 1),
              autocorr(signs, 10), autocorr(signs, 100), c * c);
  std::printf("predict  : corr(30 s delta, next 30 s return) %+.3f; range of the session %.0f ticks\n", corr(d30, f30),
              *std::max_element(mids.begin(), mids.end()) - *std::min_element(mids.begin(), mids.end()));
}
