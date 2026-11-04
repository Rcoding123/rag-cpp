// rag/orderflow/scalp_env.hpp — directional order-flow scalping on the ES
// simulator, as an Environment for rag::agent learners.
//
// Every `decide_seconds` the strategy picks a target position (short, flat
// or long `contracts`); a change is sent as a market order that reaches the
// book `latency_steps` simulator steps later, pays the spread through its
// fills, and pays `fee_dollars` per contract per side. The reward is the
// change in mark-to-market equity (marked at the mid) in ticks, net of fees;
// the episode ends flat (a final market order closes any position).
//
// Observation: the 13 order-flow features (features.hpp), the position, the
// open P&L and the time left. Each episode is a fresh market (a new
// simulator seed and a random time of day), warmed up so every window is
// full before the first decision.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>

#include "rag/agent/env.hpp"
#include "rag/orderflow/features.hpp"
#include "rag/orderflow/simulator.hpp"

namespace rag::orderflow {

struct ScalpParams {
  SimParams sim;
  double decide_seconds = 1.0;
  double episode_seconds = 600;
  double warmup_seconds = 150;
  std::uint32_t latency_steps = 1;  // 100 ms at the default dt
  std::uint32_t contracts = 1;
  double fee_dollars = 1.5;  // per contract per side (exchange, clearing, broker)
};

struct ScalpObs {
  std::array<double, kFeatures> f{};
  int position = 0;        // contracts
  double open_pnl = 0;     // ticks, marked at the mid
  double time_left = 1;    // fraction of the episode
};

class ScalpEnv {
 public:
  using Observation = ScalpObs;
  static constexpr std::size_t kNumActions = 3;  // 0 short, 1 flat, 2 long
  static constexpr std::size_t kStateDims = kFeatures + 3;
  static constexpr std::array<double, kStateDims> kLo = [] {
    std::array<double, kStateDims> a{};
    a.fill(-kClip);
    a[kFeatures] = -1;
    a[kFeatures + 1] = -kClip;
    a[kFeatures + 2] = 0;
    return a;
  }();
  static constexpr std::array<double, kStateDims> kHi = [] {
    std::array<double, kStateDims> a{};
    a.fill(kClip);
    a[kFeatures] = 1;
    a[kFeatures + 1] = kClip;
    a[kFeatures + 2] = 1;
    return a;
  }();

  explicit ScalpEnv(const ScalpParams& p = {}) : p_(p) {}

  Observation reset(rag::agent::Rng& rng) {
    const double latest_start = std::max(0.0, p_.sim.session_seconds - p_.episode_seconds - p_.warmup_seconds);
    sim_ = std::make_unique<EsSimulator>(p_.sim, rng.next(), rng.uniform() * latest_start);
    fb_ = FeatureBuilder(p_.sim.dt);
    for (std::uint32_t i = 0; i < steps(p_.warmup_seconds); ++i) advance();
    position_ = 0;
    cash_ = 0;
    equity_ = 0;
    entry_mid_ = 0;
    elapsed_ = 0;
    trades_ = 0;
    fees_ticks_ = 0;
    return obs();
  }

  rag::agent::Step<Observation> step(rag::agent::Action a, rag::agent::Rng&) {
    const int target = (static_cast<int>(a) - 1) * static_cast<int>(p_.contracts);
    const std::uint32_t n = steps(p_.decide_seconds);
    double fees = 0;
    if (p_.latency_steps == 0 && target != position_) fees += trade_to(target);
    for (std::uint32_t i = 0; i < n; ++i) {
      advance();
      if (p_.latency_steps > 0 && i + 1 == std::min(n, p_.latency_steps) && target != position_) fees += trade_to(target);
    }
    elapsed_ += p_.decide_seconds;
    const bool done = elapsed_ >= p_.episode_seconds - 1e-9;
    if (done && position_ != 0) fees += trade_to(0);
    const double eq = cash_ + position_ * sim_->mid();
    const double reward = eq - equity_ - fees;
    equity_ = eq;
    return {obs(), reward, done};
  }

  [[nodiscard]] std::array<double, kStateDims> features(const Observation& o) const noexcept {
    std::array<double, kStateDims> x{};
    for (std::size_t i = 0; i < kFeatures; ++i) x[i] = o.f[i];
    x[kFeatures] = static_cast<double>(o.position) / static_cast<double>(p_.contracts);
    x[kFeatures + 1] = std::clamp(o.open_pnl / 4.0, -kClip, kClip);
    x[kFeatures + 2] = o.time_left;
    return x;
  }

  // Episode statistics and the market, for evaluation and oracles.
  [[nodiscard]] double net_pnl_ticks() const noexcept { return equity_ - fees_ticks_; }  // marked at the mid, after fees
  [[nodiscard]] std::uint32_t round_trips() const noexcept { return trades_ / 2; }
  [[nodiscard]] std::uint32_t orders() const noexcept { return trades_; }
  [[nodiscard]] double fees_ticks() const noexcept { return fees_ticks_; }
  [[nodiscard]] const EsSimulator& market() const noexcept { return *sim_; }
  [[nodiscard]] const FeatureBuilder& feature_builder() const noexcept { return fb_; }
  [[nodiscard]] const ScalpParams& params() const noexcept { return p_; }

 private:
  [[nodiscard]] std::uint32_t steps(double seconds) const {
    return std::max<std::uint32_t>(1, static_cast<std::uint32_t>(std::lround(seconds / p_.sim.dt)));
  }
  void advance() {
    sim_->step();
    for (const auto& t : sim_->trades()) fb_.add_trade(t);
    fb_.end_bucket(sim_->quote());
  }
  // Market order to reach `target`; returns the fee in ticks.
  double trade_to(int target) {
    const int diff = target - position_;
    const auto qty = static_cast<std::uint32_t>(std::abs(diff));
    const Fill f = sim_->market_order(diff > 0 ? Side::Buy : Side::Sell, qty);
    cash_ -= (diff > 0 ? 1.0 : -1.0) * f.filled * f.avg_price;
    const int before = position_;
    position_ += diff > 0 ? static_cast<int>(f.filled) : -static_cast<int>(f.filled);
    if (position_ != 0 && (before == 0 || (before > 0) != (position_ > 0))) entry_mid_ = f.avg_price;  // new or flipped position
    if (position_ == 0) entry_mid_ = 0;
    ++trades_;
    const double fee = f.filled * p_.fee_dollars / kTickValue;
    fees_ticks_ += fee;
    return fee;
  }
  [[nodiscard]] Observation obs() const {
    Observation o;
    o.f = fb_.values();
    o.position = position_;
    o.open_pnl = position_ != 0 ? position_ * (sim_->mid() - entry_mid_) : 0;
    o.time_left = std::max(0.0, 1 - elapsed_ / p_.episode_seconds);
    return o;
  }

  ScalpParams p_;
  std::unique_ptr<EsSimulator> sim_;
  FeatureBuilder fb_{0.1};
  int position_ = 0;
  double cash_ = 0, equity_ = 0, entry_mid_ = 0, elapsed_ = 0, fees_ticks_ = 0;
  std::uint32_t trades_ = 0;
};

static_assert(rag::agent::Environment<ScalpEnv>);

}  // namespace rag::orderflow
