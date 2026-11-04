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
  // Also flag sustained *upward* shifts. A world that got better is a change
  // too: the old policy still earns more than before, so a one-sided test
  // stays quiet while the agent leaves the new opportunity unused (E13).
  bool two_sided = false;
};

// Flags a sustained downward shift in "surprise" (return - predicted return)
// with a one-sided CUSUM test (Page, 1954):
//     S = min(0, S + z + k),   alarm when S < -h,   z = (x - mean) / sd
// and, with two_sided, an upward one with the mirror-image statistic
//     U = max(0, U + z - k),   alarm when U > h.
// Each sample adds evidence; ordinary noise drains it back toward 0 because
// of the allowance k, while a persistent shift accumulates until it crosses h.
//
// An alarm only clears the accumulated evidence. The caller decides what it
// meant: reset() after a confirmed change (the old baseline describes a world
// that's gone; rebuilding it takes `warmup` samples), or rebase() after a
// false alarm (re-centre on recent samples and stay armed). (E13: resetting
// on every alarm, including rejected ones, left the trainer blind for 1,550
// episodes, and a real change landed in that window.)
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
    if (p_.two_sided) up_ = std::max(0.0, up_ + z - p_.allowance);
    if (cusum_ < -p_.threshold || up_ > p_.threshold) {
      last_up_ = up_ > p_.threshold;
      cusum_ = up_ = 0;
      return true;
    }
    return false;
  }

  // After a false alarm: keep the baseline's spread, move its centre to the
  // recent (lagged-window) samples and let the next ones refine it as a
  // fresh running mean; clear the evidence; stay armed.
  void rebase() {
    if (!held_.empty()) {
      double s = 0;
      for (double x : held_) s += x;
      mean_ = s / static_cast<double>(held_.size());
      rate_n_ = held_.size();
    }
    cusum_ = up_ = 0;
  }

  void reset() {
    held_.clear();
    baseline_n_ = rate_n_ = 0;
    mean_ = var_ = cusum_ = up_ = 0;
  }

  // Direction of the last alarm: true if returns rose above expectations.
  [[nodiscard]] bool last_alarm_was_upward() const noexcept { return last_up_; }

  // How close to an alarm (alarm at threshold), in either direction with two_sided.
  [[nodiscard]] double evidence() const noexcept { return std::max(-cusum_, up_); }

 private:
  // Exponential moving mean and variance; exact running moments while young.
  void absorb(double x) {
    ++baseline_n_;
    ++rate_n_;
    const double rate = std::max(p_.baseline_rate, 1.0 / static_cast<double>(rate_n_));
    const double d = x - mean_;
    mean_ += rate * d;
    var_ = (1 - rate) * (var_ + rate * d * d);
  }

  Params p_;
  std::deque<double> held_;
  std::uint64_t baseline_n_ = 0;  // samples absorbed (arms the detector after `warmup`)
  std::uint64_t rate_n_ = 0;      // samples in the current running mean (restarted by rebase)
  double mean_ = 0, var_ = 0, cusum_ = 0, up_ = 0;
  bool last_up_ = false;
};

}  // namespace rag::agent
