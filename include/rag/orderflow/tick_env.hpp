// rag/orderflow/tick_env.hpp — tick scalping (passive, queue-based) on the
// ES simulator, as an Environment for rag::agent learners.
//
// The strategy earns the spread by resting 1-lot limit orders at the best
// bid and/or ask and managing inventory. Every `decide_seconds` it picks a
// quoting mode:
//
//   0 no quotes        cancel everything
//   1 bid only         rest a buy at the best bid
//   2 ask only         rest a sell at the best ask
//   3 both sides
//   4 flatten          cancel quotes, market order back to flat
//
// Between decisions the environment keeps each quoted side at the touch
// every simulator step: if the best price moved away, the order is cancelled
// and rejoins at the back of the new best queue (queue priority is lost, as
// in a real FIFO book). A side is not quoted when a fill would take the
// inventory past +-max_position. Fills come from the simulator's FIFO model:
// an order fills only after the lots ahead of it trade, so adverse
// selection (being filled just before the price moves through you) is real.
//
// Reward: change in mark-to-market equity (at the mid) in ticks, net of
// fees; the episode ends flat. Observation: the 13 order-flow features plus
// queue position on each side, touch sizes, spread, position, open P&L and
// time left.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/orderflow/features.hpp"
#include "rag/orderflow/simulator.hpp"

namespace rag::orderflow {

struct TickParams {
  SimParams sim;
  double decide_seconds = 0.5;
  double episode_seconds = 300;
  double warmup_seconds = 150;
  double start_second = -1;  // session time of the warm-up start; < 0: random each episode
  int max_position = 1;
  double fee_dollars = 1.5;  // per contract per side, passive or aggressive
  // OF13: sweep_follow (discovery_v4) is refused unless set: a resting own quote changes
  // which side the takers sweep, and each sweep moves fair value (+0.77 tick mid shift 5 s
  // after a bid-only quote, OF13 review). Diagnostics only (of10_discovery's tick rows).
  bool allow_sweep_follow = false;
};

struct TickObs {
  std::array<double, kFeatures> f{};
  double bid_queue = -1, ask_queue = -1;  // lots ahead / displayed size at our level; -1: not quoting
  double bid_size = 0, ask_size = 0;      // log(1 + touch size)
  double spread = 1;                      // ticks
  int position = 0;
  double open_pnl = 0;  // ticks
  double time_left = 1;
};

class TickScalpEnv {
 public:
  using Observation = TickObs;
  static constexpr std::size_t kNumActions = 5;
  static constexpr std::size_t kStateDims = kFeatures + 8;
  static constexpr std::array<double, kStateDims> kLo = [] {
    std::array<double, kStateDims> a{};
    a.fill(-kClip);
    for (std::size_t i = kFeatures; i < kStateDims; ++i) a[i] = -1;
    return a;
  }();
  static constexpr std::array<double, kStateDims> kHi = [] {
    std::array<double, kStateDims> a{};
    a.fill(kClip);
    for (std::size_t i = kFeatures; i < kStateDims; ++i) a[i] = 1;
    return a;
  }();

  explicit TickScalpEnv(const TickParams& p = {}) : p_(p) {
    if (p_.sim.sweep_follow > 0 && !p_.allow_sweep_follow)
      throw std::invalid_argument("TickScalpEnv: sweep_follow > 0 (own quotes move fair value; allow_sweep_follow is for diagnostics only)");
  }

  // An episode's market before its warm-up: a fresh simulator seeded from
  // `rng` at a random time of day (or start_second). reset() and the
  // diagnostics probe (of10_discovery's episode regime) both build episodes
  // with it, so they see the same markets.
  [[nodiscard]] static EsSimulator make_market(const TickParams& p, rag::agent::Rng& rng) {
    const double latest_start = std::max(0.0, p.sim.session_seconds - p.episode_seconds - p.warmup_seconds);
    const double u = rng.uniform();  // drawn either way: episode seeds don't depend on the setting
    const std::uint64_t seed = rng.next();
    return EsSimulator(p.sim, seed, p.start_second >= 0 ? p.start_second : u * latest_start);
  }

  Observation reset(rag::agent::Rng& rng) {
    sim_.emplace(make_market(p_, rng));
    fb_ = FeatureBuilder(p_.sim.dt);
    for (std::uint32_t i = 0; i < steps(p_.warmup_seconds); ++i) advance();
    position_ = 0;
    cash_ = equity_ = fees_ = elapsed_ = 0;
    entry_ = 0;
    bid_id_ = ask_id_ = 0;
    mode_ = 0;
    fills_ = 0;
    return obs();
  }

  rag::agent::Step<Observation> step(rag::agent::Action a, rag::agent::Rng&) {
    step_fills_.clear();
    step_fill_values_.clear();
    double fees = 0;
    if (a == 4) {
      mode_ = 0;
      manage();
      if (position_ != 0) fees += take_to_flat();
    } else {
      mode_ = static_cast<int>(a);
      manage();
    }
    for (std::uint32_t i = 0, n = steps(p_.decide_seconds); i < n; ++i) {
      advance();
      fees += book_fills();
      manage();
    }
    elapsed_ += p_.decide_seconds;
    const bool done = elapsed_ >= p_.episode_seconds - 1e-9;
    if (done) {
      mode_ = 0;
      manage();
      if (position_ != 0) fees += take_to_flat();
    }
    const double eq = cash_ + position_ * sim_->mid();
    const double reward = eq - equity_ - fees;
    equity_ = eq;
    return {obs(), reward, done};
  }

  [[nodiscard]] std::array<double, kStateDims> features(const Observation& o) const noexcept {
    std::array<double, kStateDims> x{};
    for (std::size_t i = 0; i < kFeatures; ++i) x[i] = o.f[i];
    x[kFeatures + 0] = std::clamp(o.bid_queue, -1.0, 1.0);
    x[kFeatures + 1] = std::clamp(o.ask_queue, -1.0, 1.0);
    x[kFeatures + 2] = std::clamp(o.bid_size / 8 * 2 - 1, -1.0, 1.0);  // log sizes: ~0..8
    x[kFeatures + 3] = std::clamp(o.ask_size / 8 * 2 - 1, -1.0, 1.0);
    x[kFeatures + 4] = std::clamp(o.spread - 1, 0.0, 1.0);
    x[kFeatures + 5] = static_cast<double>(o.position) / p_.max_position;
    x[kFeatures + 6] = std::clamp(o.open_pnl / 4, -1.0, 1.0);
    x[kFeatures + 7] = o.time_left * 2 - 1;
    return x;
  }

  [[nodiscard]] double net_pnl_ticks() const noexcept { return equity_ - fees_; }
  [[nodiscard]] std::uint32_t fills() const noexcept { return fills_; }
  // Passive fills of the last step() (own-fill markouts); flattening market orders are not included.
  [[nodiscard]] const std::vector<OwnFill>& step_fills() const noexcept { return step_fills_; }
  // DIAGNOSTICS ONLY, hidden state (never an observation): the fair value when
  // each step_fills()[i] filled (EsSimulator::own_fill_hidden_values()).
  [[nodiscard]] const std::vector<double>& step_fill_hidden_values() const noexcept { return step_fill_values_; }
  [[nodiscard]] double fees_ticks() const noexcept { return fees_; }
  [[nodiscard]] const EsSimulator& market() const noexcept { return *sim_; }
  [[nodiscard]] const TickParams& params() const noexcept { return p_; }

 private:
  [[nodiscard]] std::uint32_t steps(double seconds) const {
    return std::max<std::uint32_t>(1, static_cast<std::uint32_t>(std::lround(seconds / p_.sim.dt)));
  }
  void advance() {
    sim_->step();
    for (const auto& t : sim_->trades()) fb_.add_trade(t);
    fb_.end_bucket(sim_->quote());
  }
  [[nodiscard]] const EsSimulator::Resting* find(std::uint64_t id) const {
    if (id == 0) return nullptr;
    for (const auto& r : sim_->resting())
      if (r.id == id) return &r;
    return nullptr;
  }
  void on_fill(Side side, std::int32_t price, std::uint32_t size) {
    const int before = position_;
    const int signed_size = side == Side::Buy ? static_cast<int>(size) : -static_cast<int>(size);
    cash_ -= signed_size * static_cast<double>(price);
    position_ += signed_size;
    if (position_ != 0 && (before == 0 || (before > 0) != (position_ > 0))) entry_ = price;
    if (position_ == 0) entry_ = 0;
    ++fills_;
  }
  // Passive fills from the last simulator step; returns their fees.
  double book_fills() {
    double fees = 0;
    const auto& values = sim_->own_fill_hidden_values();
    for (std::size_t i = 0; i < sim_->own_fills().size(); ++i) {
      const OwnFill& f = sim_->own_fills()[i];
      on_fill(f.side, f.price, f.size);
      step_fills_.push_back(f);
      step_fill_values_.push_back(values[i]);
      fees += f.size * p_.fee_dollars / kTickValue;
    }
    fees_ += fees;
    if (!find(bid_id_)) bid_id_ = 0;
    if (!find(ask_id_)) ask_id_ = 0;
    return fees;
  }
  // Keep each wanted side resting at the touch; cancel unwanted ones.
  void manage() {
    const bool want_bid = (mode_ == 1 || mode_ == 3) && position_ < p_.max_position;
    const bool want_ask = (mode_ == 2 || mode_ == 3) && position_ > -p_.max_position;
    auto keep = [&](std::uint64_t& id, bool want, Side side) {
      const auto* r = find(id);
      const std::int32_t best = side == Side::Buy ? sim_->bid() : sim_->ask();
      if (r && (!want || r->price != best)) {
        sim_->cancel(id);
        id = 0;
        r = nullptr;
      }
      if (want && !r) id = sim_->submit_limit(side, best, 1);
    };
    keep(bid_id_, want_bid, Side::Buy);
    keep(ask_id_, want_ask, Side::Sell);
  }
  double take_to_flat() {
    const int diff = -position_;
    const Fill f = sim_->market_order(diff > 0 ? Side::Buy : Side::Sell, static_cast<std::uint32_t>(std::abs(diff)));
    cash_ -= (diff > 0 ? 1.0 : -1.0) * f.filled * f.avg_price;
    position_ += diff > 0 ? static_cast<int>(f.filled) : -static_cast<int>(f.filled);
    if (position_ == 0) entry_ = 0;
    ++fills_;
    const double fee = f.filled * p_.fee_dollars / kTickValue;
    fees_ += fee;
    return fee;
  }
  [[nodiscard]] Observation obs() const {
    Observation o;
    o.f = fb_.values();
    const Quote q = sim_->quote();
    auto queue = [&](std::uint64_t id, std::uint32_t shown) {
      const auto* r = find(id);
      return r ? static_cast<double>(r->ahead) / std::max<std::uint32_t>(1, shown) : -1.0;
    };
    o.bid_queue = queue(bid_id_, q.bid_size[0]);
    o.ask_queue = queue(ask_id_, q.ask_size[0]);
    o.bid_size = std::log1p(q.bid_size[0]);
    o.ask_size = std::log1p(q.ask_size[0]);
    o.spread = q.ask - q.bid;
    o.position = position_;
    o.open_pnl = position_ != 0 ? position_ * (sim_->mid() - entry_) : 0;
    o.time_left = std::max(0.0, 1 - elapsed_ / p_.episode_seconds);
    return o;
  }

  TickParams p_;
  std::optional<EsSimulator> sim_;
  FeatureBuilder fb_{0.1};
  int position_ = 0, mode_ = 0;
  double cash_ = 0, equity_ = 0, fees_ = 0, elapsed_ = 0, entry_ = 0;
  std::uint64_t bid_id_ = 0, ask_id_ = 0;
  std::uint32_t fills_ = 0;
  std::vector<OwnFill> step_fills_;  // passive fills of the last step() (diagnostics)
  std::vector<double> step_fill_values_;  // step_fill_hidden_values(), parallel to step_fills_
};

static_assert(rag::agent::Environment<TickScalpEnv>);

}  // namespace rag::orderflow
