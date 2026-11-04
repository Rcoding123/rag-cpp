// rag/agent/drift.hpp — change detection on a stream of per-episode
// "surprise" values (what happened minus what was predicted).
//
// Used by the trainer (surprise against the agent's own value estimate,
// confirmed by re-scoring the champion) and by the model-based agent
// (surprise against its plan's predicted value; E9).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>

namespace rag::agent {

// Defaults chosen by simulation on Gaussian noise: about one false alarm
// per 200,000 episodes, and a shift of 0.55 standard deviations (what the
// ExecutionEnv regime flip produces) detected after a median of ~50 episodes.
struct DriftParams {
  std::uint32_t lag = 50;         // samples held back before they join the baseline
  double allowance = 0.3;         // CUSUM k, in standard deviations: ignore shifts smaller than ~2k
  double threshold = 16.0;        // CUSUM h, in standard deviations of accumulated evidence
  double baseline_rate = 0.001;   // EMA rate of the baseline (memory of ~1000 samples)
  std::uint32_t warmup = 1500;    // baseline samples needed before alarms (also after one)
};

// Flags a sustained downward shift in "surprise" (return - predicted return)
// with a one-sided CUSUM test (Page, 1954):
//     S = min(0, S + z + k),   alarm when S < -h,   z = (x - mean) / sd
// Each sample adds evidence; ordinary noise drains it back toward 0 because
// of the allowance k, while a persistent shift accumulates until it crosses h.
//
// The baseline (mean, sd) is built only from samples at least `lag` old, so
// a shift is measured against the world before it, not partly against
// itself; and it has a long memory, because a noisy baseline is itself a
// source of false alarms.
class DriftDetector {
 public:
  using Params = DriftParams;
  explicit DriftDetector(Params p = {}) : p_(p) {}

  // Feed one episode's surprise; returns true on an alarm.
  bool add(double surprise) {
    held_.push_back(surprise);
    if (held_.size() > p_.lag) {  // the oldest sample graduates into the baseline
      absorb(held_.front());
      held_.pop_front();
    }
    if (baseline_n_ < p_.warmup) return false;
    const double z = (surprise - mean_) / std::sqrt(std::max(var_, 1e-12));
    cusum_ = std::min(0.0, cusum_ + z + p_.allowance);
    if (cusum_ < -p_.threshold) {
      reset();
      return true;
    }
    return false;
  }

  void reset() {
    held_.clear();
    baseline_n_ = 0;
    mean_ = var_ = cusum_ = 0;
  }

  [[nodiscard]] double evidence() const noexcept { return -cusum_; }  // how close to an alarm (alarm at threshold)

 private:
  // Exponential moving mean and variance; exact running moments while young.
  void absorb(double x) {
    ++baseline_n_;
    const double rate = std::max(p_.baseline_rate, 1.0 / static_cast<double>(baseline_n_));
    const double d = x - mean_;
    mean_ += rate * d;
    var_ = (1 - rate) * (var_ + rate * d * d);
  }

  Params p_;
  std::deque<double> held_;
  std::uint64_t baseline_n_ = 0;
  double mean_ = 0, var_ = 0, cusum_ = 0;
};

}  // namespace rag::agent
