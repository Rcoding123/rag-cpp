// rag/orderflow/replay_env.hpp — the scalping environment on recorded ticks
// (backtesting), with the same observations and actions as ScalpEnv, so an
// agent trained in the simulator runs unchanged on real data.
//
//   TickTape tape = TickTape::from_csv("es_ticks.csv", {.time_unit = "ns"});
//   ReplayScalpEnv env(tape, {.window_seconds = 600});
//   env.seek(k);              // episode k = the k-th consecutive window (else reset() picks one at random)
//
// Fills: a market order reaches the book one bucket later (latency) and fills
// at the recorded best ask (buy) or bid (sell); an order larger than the
// recorded touch size walks the recorded depth, one tick per level. Fees per
// contract per side as in ScalpEnv.
//
// What a replay cannot do: the strategy's own orders don't move the recorded
// market (fine for 1-lot market orders in ES, whose touch holds hundreds of
// lots), and passive queue fills can't be replayed faithfully without
// order-by-order (MBO) data, so only directional market-order strategies are
// backtested here.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/orderflow/features.hpp"
#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/tick_csv.hpp"

namespace rag::orderflow {

// A recorded tick stream, bucketed: the trades of each bucket and the quote at its end.
class TickTape {
 public:
  struct Bucket {
    std::vector<Trade> trades;
    Quote quote;
  };
  static TickTape from_csv(const std::string& path, const TickCsvOptions& o = {}) {
    TickTape t;
    t.bucket_seconds_ = o.bucket_seconds;
    Collector c{&t, {}};
    t.stats_ = replay_tick_csv(path, c, o);
    return t;
  }
  static TickTape from_stream(std::istream& in, const TickCsvOptions& o = {}) {
    TickTape t;
    t.bucket_seconds_ = o.bucket_seconds;
    Collector c{&t, {}};
    t.stats_ = replay_tick_csv(in, c, o);
    return t;
  }
  [[nodiscard]] std::size_t size() const noexcept { return buckets_.size(); }
  [[nodiscard]] const Bucket& operator[](std::size_t i) const { return buckets_[i]; }
  [[nodiscard]] double bucket_seconds() const noexcept { return bucket_seconds_; }
  [[nodiscard]] const TickCsvStats& stats() const noexcept { return stats_; }

 private:
  struct Collector {
    TickTape* t;
    std::vector<Trade> pending;
    void add_trade(const Trade& tr) { pending.push_back(tr); }
    void end_bucket(const Quote& q) {
      t->buckets_.push_back({std::move(pending), q});
      pending.clear();
    }
  };
  std::vector<Bucket> buckets_;
  double bucket_seconds_ = 0.1;
  TickCsvStats stats_;
};

struct ReplayParams {
  double decide_seconds = 1.0;
  double window_seconds = 600;   // one episode
  double warmup_seconds = 150;   // features warm up on the tape before the window
  std::uint32_t latency_buckets = 1;
  std::uint32_t contracts = 1;
  double fee_dollars = 1.5;
};

class ReplayScalpEnv {
 public:
  using Observation = ScalpObs;
  static constexpr std::size_t kNumActions = ScalpEnv::kNumActions;
  static constexpr std::size_t kStateDims = ScalpEnv::kStateDims;
  static constexpr auto kLo = ScalpEnv::kLo;
  static constexpr auto kHi = ScalpEnv::kHi;

  ReplayScalpEnv(std::shared_ptr<const TickTape> tape, const ReplayParams& p = {}) : tape_(std::move(tape)), p_(p) {
    if (!tape_ || tape_->size() == 0) throw std::invalid_argument("ReplayScalpEnv: empty tape");
  }

  // Consecutive windows that fit on the tape after the warm-up.
  [[nodiscard]] std::size_t windows() const noexcept {
    const std::size_t need = buckets(p_.window_seconds);
    const std::size_t warm = buckets(p_.warmup_seconds);
    return tape_->size() > warm + need ? (tape_->size() - warm) / need : 0;
  }
  // Episode k = window k (0-based), for walk-forward evaluation.
  Observation seek(std::size_t k) {
    if (k >= windows()) throw std::out_of_range("ReplayScalpEnv: no such window");
    return start_at(k * buckets(p_.window_seconds));
  }
  // A random window (training on recorded data).
  Observation reset(rag::agent::Rng& rng) {
    const std::size_t w = windows();
    if (w == 0) throw std::out_of_range("ReplayScalpEnv: tape shorter than warm-up + window");
    return seek(rng.below(static_cast<std::uint32_t>(w)));
  }

  rag::agent::Step<Observation> step(rag::agent::Action a, rag::agent::Rng&) {
    const int target = (static_cast<int>(a) - 1) * static_cast<int>(p_.contracts);
    double fees = 0;
    const std::size_t n = buckets(p_.decide_seconds);
    for (std::size_t i = 0; i < n && pos_ < end_; ++i) {
      advance();
      if (i + 1 == std::min<std::size_t>(n, p_.latency_buckets) && target != position_) fees += trade_to(target);
    }
    if (p_.latency_buckets == 0 && target != position_) fees += trade_to(target);
    const bool done = pos_ >= end_;
    if (done && position_ != 0) fees += trade_to(0);
    const double eq = cash_ + position_ * mid();
    const double reward = eq - equity_ - fees;
    equity_ = eq;
    return {obs(done), reward, done};
  }

  [[nodiscard]] std::array<double, kStateDims> features(const Observation& o) const noexcept {
    std::array<double, kStateDims> x{};
    for (std::size_t i = 0; i < kFeatures; ++i) x[i] = o.f[i];
    x[kFeatures] = static_cast<double>(o.position) / static_cast<double>(p_.contracts);
    x[kFeatures + 1] = std::clamp(o.open_pnl / 4.0, -kClip, kClip);
    x[kFeatures + 2] = o.time_left;
    return x;
  }

  [[nodiscard]] double net_pnl_ticks() const noexcept { return equity_ - fees_; }
  [[nodiscard]] std::uint32_t orders() const noexcept { return orders_; }
  [[nodiscard]] double fees_ticks() const noexcept { return fees_; }

 private:
  [[nodiscard]] std::size_t buckets(double seconds) const {
    return std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(seconds / tape_->bucket_seconds())));
  }
  [[nodiscard]] double mid() const { return 0.5 * (quote().bid + quote().ask); }
  [[nodiscard]] const Quote& quote() const { return (*tape_)[pos_ == 0 ? 0 : pos_ - 1].quote; }
  void advance() {
    const auto& b = (*tape_)[pos_++];
    for (const auto& t : b.trades) fb_.add_trade(t);
    fb_.end_bucket(b.quote);
  }
  Observation start_at(std::size_t window_start) {
    fb_ = FeatureBuilder(tape_->bucket_seconds());
    pos_ = window_start;
    const std::size_t warm_end = window_start + buckets(p_.warmup_seconds);
    while (pos_ < warm_end) advance();
    end_ = std::min(tape_->size(), warm_end + buckets(p_.window_seconds));
    position_ = 0;
    cash_ = equity_ = fees_ = entry_ = 0;
    orders_ = 0;
    start_ = pos_;
    return obs(false);
  }
  // Fill against the recorded book: the touch, then one tick worse per level.
  double trade_to(int target) {
    const int diff = target - position_;
    std::uint32_t left = static_cast<std::uint32_t>(std::abs(diff));
    const Quote& q = quote();
    double notional = 0;
    for (std::size_t lvl = 0; left > 0; ++lvl) {
      const std::int32_t price = diff > 0 ? q.ask + static_cast<std::int32_t>(lvl) : q.bid - static_cast<std::int32_t>(lvl);
      const std::uint32_t shown = lvl < kDepth ? (diff > 0 ? q.ask_size[lvl] : q.bid_size[lvl]) : 0;
      // Without depth in the data (sizes 0), assume the touch holds the order.
      const std::uint32_t take = (shown == 0 && lvl == 0) || lvl + 1 >= kDepth ? left : std::min(left, shown);
      notional += static_cast<double>(take) * price;
      left -= take;
    }
    const std::uint32_t qty = static_cast<std::uint32_t>(std::abs(diff));
    const double avg = notional / qty;
    cash_ -= (diff > 0 ? 1.0 : -1.0) * qty * avg;
    const int before = position_;
    position_ = target;
    if (position_ != 0 && (before == 0 || (before > 0) != (position_ > 0))) entry_ = avg;
    if (position_ == 0) entry_ = 0;
    ++orders_;
    const double fee = qty * p_.fee_dollars / kTickValue;
    fees_ += fee;
    return fee;
  }
  [[nodiscard]] Observation obs(bool done) const {
    Observation o;
    o.f = fb_.values();
    o.position = position_;
    o.open_pnl = position_ != 0 ? position_ * (mid() - entry_) : 0;
    const double total = static_cast<double>(end_ - start_);
    o.time_left = done || total <= 0 ? 0 : std::max(0.0, static_cast<double>(end_ - pos_) / total);
    return o;
  }

  std::shared_ptr<const TickTape> tape_;
  ReplayParams p_;
  FeatureBuilder fb_{0.1};
  std::size_t pos_ = 0, end_ = 0, start_ = 0;
  int position_ = 0;
  double cash_ = 0, equity_ = 0, fees_ = 0, entry_ = 0;
  std::uint32_t orders_ = 0;
};

static_assert(rag::agent::Environment<ReplayScalpEnv>);

}  // namespace rag::orderflow
