// rag/orderflow/features.hpp — order-flow features from a tick stream.
//
// Feed it trades and quotes in time buckets (the simulator's steps, or
// recorded ticks grouped into the same bucket width):
//
//   FeatureBuilder fb(0.1);                 // 100 ms buckets
//   for each bucket: fb.add_trade(t)...; fb.end_bucket(quote);
//   const auto& x = fb.values();            // kFeatures numbers, each scaled to ~unit size
//
// Every feature is divided by a running RMS of itself (EWMA over
// `norm_seconds`), so the same code works on simulated and recorded ES
// without hand-tuned scales, and values are clipped to +-kClip.
//
//   0  delta 5 s        signed aggressor volume (buys - sells)
//   1  delta 30 s
//   2  delta 120 s
//   3  OFI 5 s          order-flow imbalance at the touch (Cont, Kukanov &
//   4  OFI 30 s         Stoikov 2014): includes limit-order adds/cancels
//   5  volume 30 s      log(volume / its running mean)
//   6  imbalance L1     (bid size - ask size) / (sum), best level
//   7  imbalance L1-5   the same over 5 levels
//   8  return 5 s       mid change, ticks
//   9  return 30 s
//  10  return 120 s
//  11  VWAP distance    mid - session VWAP
//  12  absorption       hidden liquidity absorbed at the touch: while the
//                       best bid holds, sell volume traded there beyond the
//                       size that was displayed when it became best; minus
//                       the same at the ask (> 0: a hidden buyer is
//                       absorbing sellers). Ordinary selling into a deep
//                       queue doesn't count; refills (icebergs) do.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rag/orderflow/simulator.hpp"

namespace rag::orderflow {

inline constexpr std::size_t kFeatures = 13;
inline constexpr double kClip = 4.0;

class FeatureBuilder {
 public:
  explicit FeatureBuilder(double bucket_seconds = 0.1, double norm_seconds = 600)
      : dt_(bucket_seconds), norm_alpha_(std::min(1.0, bucket_seconds / norm_seconds)) {
    n5_ = buckets(5);
    n30_ = buckets(30);
    n120_ = buckets(120);
    len_ = {n5_, n30_, n120_};
    ring_.resize(n120_ + 1);
    rms_.fill(1.0);
  }

  void add_trade(const Trade& t) {
    cur_.delta += sign(t.aggressor) * t.size;
    cur_.volume += t.size;
    if (t.aggressor == Side::Sell && t.price == last_.bid) run_sold_ += t.size;
    if (t.aggressor == Side::Buy && t.price == last_.ask) run_bought_ += t.size;
    vwap_num_ += static_cast<double>(t.price) * t.size;
    vwap_den_ += t.size;
  }

  void end_bucket(const Quote& q) {
    // OFI contribution of this quote change (best level only).
    if (have_quote_) {
      double e = 0;
      if (q.bid >= last_.bid) e += q.bid_size[0];
      if (q.bid <= last_.bid) e -= last_.bid_size[0];
      if (q.ask <= last_.ask) e -= q.ask_size[0];
      if (q.ask >= last_.ask) e += last_.ask_size[0];
      cur_.ofi = e;
      // Absorption: volume beyond what was displayed, while the level holds.
      if (q.bid != last_.bid) {
        run_sold_ = 0;
        bid_shown_ = q.bid_size[0];
      }
      if (q.ask != last_.ask) {
        run_bought_ = 0;
        ask_shown_ = q.ask_size[0];
      }
      absorb_ = std::max(0.0, run_sold_ - bid_shown_) - std::max(0.0, run_bought_ - ask_shown_);
    }
    cur_.mid = 0.5 * (q.bid + q.ask);
    ring_[head_] = cur_;
    head_ = (head_ + 1) % ring_.size();
    count_ = std::min(count_ + 1, ring_.size());
    // Running window sums: add the new bucket, drop the one that left.
    for (std::size_t w = 0; w < 3; ++w) {
      win_[w].add(cur_, 1);
      if (count_ > len_[w]) win_[w].add(at(len_[w]), -1);
    }
    cur_ = {};
    last_ = q;
    have_quote_ = true;
    compute(q);
  }

  [[nodiscard]] const std::array<double, kFeatures>& values() const noexcept { return x_; }
  [[nodiscard]] std::array<double, kFeatures> raw() const noexcept { return raw_; }
  // Enough history for the longest window.
  [[nodiscard]] bool warm() const noexcept { return count_ > n120_; }
  [[nodiscard]] double vwap() const noexcept { return vwap_den_ > 0 ? vwap_num_ / vwap_den_ : 0; }

 private:
  struct Bucket {
    double delta = 0, volume = 0, ofi = 0, sold_at_bid = 0, bought_at_ask = 0, mid = 0;
  };
  struct Sums {
    double delta = 0, volume = 0, ofi = 0, absorb = 0;
    void add(const Bucket& b, double w) {
      delta += w * b.delta;
      volume += w * b.volume;
      ofi += w * b.ofi;
      absorb += w * (b.sold_at_bid - b.bought_at_ask);
    }
  };
  [[nodiscard]] std::size_t buckets(double seconds) const { return std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(seconds / dt_))); }
  // The bucket `back` steps before the newest (0 = newest).
  [[nodiscard]] const Bucket& at(std::size_t back) const { return ring_[(head_ + ring_.size() - 1 - back) % ring_.size()]; }
  [[nodiscard]] double ret(std::size_t n) const { return count_ > n ? at(0).mid - at(n).mid : 0.0; }

  void compute(const Quote& q) {
    const Sums &w5 = win_[0], &w30 = win_[1], &w120 = win_[2];
    const double vol30 = w30.volume;
    vol_mean_ += (vol30 - vol_mean_) * norm_alpha_;
    double bid5 = 0, ask5 = 0;
    for (std::size_t i = 0; i < kDepth; ++i) {
      bid5 += q.bid_size[i];
      ask5 += q.ask_size[i];
    }
    const double b1 = q.bid_size[0], a1 = q.ask_size[0];
    raw_ = {w5.delta,
            w30.delta,
            w120.delta,
            w5.ofi,
            w30.ofi,
            std::log((vol30 + 1) / (vol_mean_ + 1)),
            b1 + a1 > 0 ? (b1 - a1) / (b1 + a1) : 0,
            bid5 + ask5 > 0 ? (bid5 - ask5) / (bid5 + ask5) : 0,
            ret(n5_),
            ret(n30_),
            ret(n120_),
            vwap_den_ > 0 ? 0.5 * (q.bid + q.ask) - vwap() : 0,
            absorb_};
    for (std::size_t k = 0; k < kFeatures; ++k) {
      // Imbalances and the volume ratio are already unit-free.
      const bool scaled = !(k == 5 || k == 6 || k == 7);
      if (scaled) {
        rms_[k] = std::sqrt(rms_[k] * rms_[k] + (raw_[k] * raw_[k] - rms_[k] * rms_[k]) * norm_alpha_);
        x_[k] = std::clamp(raw_[k] / std::max(1e-9, rms_[k]), -kClip, kClip);
      } else {
        x_[k] = std::clamp(k == 5 ? raw_[k] : 2 * raw_[k], -kClip, kClip);
      }
    }
  }

  double dt_, norm_alpha_;
  std::size_t n5_ = 1, n30_ = 1, n120_ = 1;
  std::array<std::size_t, 3> len_{};
  std::array<Sums, 3> win_{};
  std::vector<Bucket> ring_;
  std::size_t head_ = 0, count_ = 0;
  Bucket cur_;
  Quote last_{};
  bool have_quote_ = false;
  double vwap_num_ = 0, vwap_den_ = 0, vol_mean_ = 0;
  double run_sold_ = 0, run_bought_ = 0, bid_shown_ = 0, ask_shown_ = 0, absorb_ = 0;
  std::array<double, kFeatures> raw_{}, x_{}, rms_{};
};

}  // namespace rag::orderflow
