// rag/orderflow/simulator.hpp — a synthetic ES futures market at tick level.
//
// It produces what real ES tick data contains: trade prints (price, size,
// aggressor side) and quotes (best bid/ask and the size at the first kDepth
// levels), so features and strategies written against it run unchanged on
// recorded ticks later. A strategy can trade in it two ways:
//
//   market_order(side, qty)          take liquidity now (scalping / momentum)
//   submit_limit(side, price, qty)   join the FIFO queue at a price (tick
//   cancel(id)                       scalping): it fills only after the lots
//                                    ahead of it have traded or cancelled
//
// Prices are integer ticks (ES: 1 tick = 0.25 index points = $12.50 per
// contract). Time advances in steps of `dt` seconds. Each step:
//
//   1. activity: a log-OU process (volume and volatility clustering),
//      excited by price moves, times an intraday U-shape;
//   2. fair value: a hidden random walk with occasional jumps (news);
//   3. takers, all with heavy-tailed sizes:
//        noise       balanced single market orders;
//        meta-orders large parent orders split into child market orders
//                    over time, sizes power-law (the source of long memory
//                    in trade signs); informed ones start when the price is
//                    away from fair value and trade toward it, uninformed ones
//                    (rebalancing) pick a random side;
//        momentum    chase the last `momentum_lookback` seconds' move;
//   4. makers: limit-order arrivals and cancellations at every level; they
//      fade quotes against one-sided aggressive flow (cancel the side being
//      hit, add to the other), so prices adjust quickly and limit-order flow
//      carries information of its own;
//   5. icebergs: large passive orders with hidden reserve, placed by players
//      who know the fair value; aggressive flow into them trades without
//      moving the price (absorption), and the price tends to turn there;
//   6. quote jumps (opt-in, SimParams::reprice_band, discovery_market()):
//      when value has left the mid by more than the band, the whole book
//      re-prices at once and snipers take part of the stale quotes.
//
// All parameters are plain numbers in SimParams, meant to be calibrated
// against recorded ES ticks (experiments/of1_stylized_facts.cpp measures
// the stylized facts they produce).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/rng.hpp"

namespace rag::orderflow {

using rag::agent::Rng;

inline constexpr std::size_t kDepth = 5;     // book levels exposed in quotes
inline constexpr std::size_t kLevels = 16;   // book levels simulated per side
inline constexpr double kTickValue = 12.50;  // ES: dollars per tick per contract

enum class Side : std::uint8_t { Buy, Sell };
[[nodiscard]] inline constexpr double sign(Side s) noexcept { return s == Side::Buy ? 1.0 : -1.0; }

struct Trade {
  std::int64_t t_ns;
  std::int32_t price;  // ticks
  std::uint32_t size;  // contracts
  Side aggressor;
  bool own;  // the strategy's own market order
};

struct Quote {
  std::int64_t t_ns;
  std::int32_t bid, ask;  // ticks
  std::array<std::uint32_t, kDepth> bid_size, ask_size;
};

struct OwnFill {  // a fill of the strategy's resting limit order
  std::uint64_t id;
  Side side;  // Buy: our bid was hit
  std::int32_t price;
  std::uint32_t size;
};

struct SimParams {
  double dt = 0.1;                    // seconds per step
  double session_seconds = 23'400;    // regular trading hours, for the intraday shape
  std::int32_t start_price = 20'000;  // ticks (5,000.00 index points)
  // Activity: d(log A) = -log A / activity_tau dt + activity_vol dW, plus
  // activity_excite per tick the mid moved in a step *beyond its long-run
  // average move* (bursts after large moves, no drift), clamped to
  // [activity_min, activity_max]. Exciting on raw moves is supercritical:
  // busier markets move more, which makes them busier, without limit.
  double activity_tau = 300, activity_vol = 0.08, activity_excite = 0.15, activity_min = -1.5, activity_max = 2.0;
  double open_boost = 1.5, close_boost = 1.0, boost_minutes = 30;  // intraday U-shape
  // Fair value (ticks): diffusion per sqrt(second) at activity 1, scaled by
  // sqrt(activity) (information arrives with trading: volatility clusters
  // with volume, and the mixture fattens the tails); jumps.
  double value_vol = 0.33, jump_rate = 1.0 / 900, jump_sd = 10;
  // Noise takers: orders per second per side at activity 1.
  double noise_rate = 2.0;
  double taker_size_mean = 4.0, taker_size_logsd = 0.9;
  // Meta-orders: total size Pareto(meta_min, meta_alpha) capped at meta_max,
  // executed as child market orders at meta_child_rate per second.
  double meta_min = 20, meta_alpha = 1.5, meta_max = 3000, meta_child_rate = 2.0;
  double informed_meta_rate = 0.8, informed_scale = 2;  // starts per second at |value - mid| >= informed_scale
  double uninformed_meta_rate = 0.4;                    // starts per second, random side
  // Momentum takers.
  double momentum_rate = 0.8, momentum_scale = 8, momentum_lookback = 30;
  // Makers: limit orders per second at level i = maker_rate * exp(-i / maker_decay);
  // each resting lot is cancelled at cancel_rate per second.
  double maker_rate = 6.0, maker_decay = 3.0, maker_size_mean = 6.0, cancel_rate = 0.10;
  // Quote fading: with f = EWMA (fade_tau seconds) of signed taker volume per
  // second, the side being hit cancels at cancel_rate * L and the other side's
  // arrivals grow by L, L = min(fade_max, exp(fade * |f| / fade_norm)). The cap
  // matters: uncapped, fading feeds back on itself (the hit side vanishes,
  // the price runs, momentum adds flow) and the book blows up.
  double fade = 0.5, fade_norm = 25, fade_tau = 3, fade_max = 2.5;
  // Informed makers (0 = off, the default): liquidity providers that infer
  // where value is going re-price before the flow arrives. With g = fair
  // value - mid in ticks, the side the price is heading toward cancels at
  // cancel_rate * L and the other side's arrivals grow by L,
  // L = min(maker_info_max, exp(maker_info * |g| / informed_scale)). Prices
  // then move ahead of informed flow, so order flow predicts less (faster
  // price discovery, flatter markout curve).
  double maker_info = 0.0, maker_info_max = 4.0;
  double spread_fill_prob = 0.6;  // per step: an order inside a wide spread (repeats while it succeeds)
  // Icebergs.
  double iceberg_rate = 1.0 / 90, iceberg_min = 300, iceberg_max = 1500, iceberg_peak = 20;
  double iceberg_life = 300;            // seconds
  std::int32_t iceberg_max_offset = 4;  // ticks behind the best price
  std::uint32_t max_order = 300;        // size cap for a single order
  // The strategy's queue position (tick scalping). Cancellations land
  // mostly at the back of a queue (recent orders get pulled; the front is
  // long-standing size): of n lots cancelled by others, n * (ahead / others)^
  // cancel_back_bias come from ahead of our order (1 = uniform). And faster
  // participants beat a 100 ms order to a level: on joining, a lognormal
  // number of lots with mean join_ahead_mean lands ahead of it.
  // One random stream per exogenous process (value/activity, noise takers,
  // meta-orders, momentum, order shuffling, makers, icebergs, our queue
  // competition) instead of one shared stream. Then two strategies run on
  // the same seed see the same news and the same arriving traders even
  // after their own orders change the book (paired comparisons). Off by
  // default: the shared stream reproduces every earlier result exactly.
  bool separate_streams = false;
  double cancel_back_bias = 3.0, join_ahead_mean = 120;  // OF4: naive two-sided quoting ~breakeven gross, loses after fees
  // Quote jumps (0 = off, the default: every earlier result is unchanged).
  // A stale-quote race (Budish, Cramton & Shim 2015): when |fair value - mid|
  // exceeds reprice_band ticks at the end of a step, every maker re-prices
  // at once, so the whole book moves k = round(value - mid) ticks (shape and
  // spread kept) instead of drifting there through order rates. Fast takers
  // (snipers) first take a clip of reprice_snipe x the stationary best-level
  // size (at most the level) at each stale level the new book would cross,
  // printing in the direction of the jump (taker-sized prints; kept out of
  // the flow makers fade against: it is already priced). The sniped makers'
  // lots are re-posted by those who won the race, so the book keeps its shape.
  // The strategy's orders are too slow to move: crossed ones are filled at
  // their stale price, the rest keep their price at the back of the queue
  // there. A band in (0.5, 1) leaves 2 * band - 1 ticks of hysteresis; after
  // a jump the mid overshoots value by ~(1 - band) and tends to bounce back.
  double reprice_band = 0.0, reprice_snipe = 0.0;
};

// Informed makers: they re-price ahead of informed flow, so *trade* flow no
// longer predicts the next 30 s (3 seeds x 2 h: corr(30 s delta, next 30 s)
// -0.02 vs +0.12 by default; markout flat after 5 s). CAUTION, despite the
// name: the makers tilt their order *rates* toward the hidden value while
// the price adjusts slowly, so the *book imbalance* now leaks it (corr with
// the hidden gap 0.49 vs 0.20; with the next 30 s return 0.32 vs 0.25) and
// RL scalping earns more here than by default (OF9). Price discovery is too
// slow in both presets (the hidden gap predicts the next 30 s at 0.58-0.65);
// a truly efficient market needs makers that re-price (jump quotes) at once.
inline SimParams efficient_market() {
  SimParams p;
  p.maker_info = 1.0;
  return p;
}

// Fast price discovery (OF10): makers re-price (quote jumps) as soon as value
// moves more than reprice_band from the mid, and snipers take a clip of the
// stale quotes. The mid then tracks value within 0.65 tick, so the hidden
// gap, the book and the tape predict only the next tick or two. Informed
// traders must then trade fast (in the jump step) to be informed at all, so
// more of the flow is informed meta-orders and less is uninformed; quote
// jumps excite activity, so noise takers and excitation are scaled down to
// keep the stylized facts. Tuned on diagnostic seeds 1-3 x 2 h, checked on
// held-out seeds 4-6 (experiments/of10_discovery.cpp, of10_results.txt).
inline SimParams discovery_market() {
  SimParams p;
  p.reprice_band = 0.65;
  p.reprice_snipe = 0.2;
  p.informed_meta_rate = 6.0;
  p.uninformed_meta_rate = 0.15;
  p.noise_rate = 1.2;
  p.activity_excite = 0.08;
  return p;
}

// The market presets by name, for the command-line tools.
inline SimParams market_preset(const std::string& name) {
  if (name == "default") return SimParams{};
  if (name == "efficient") return efficient_market();
  if (name == "discovery") return discovery_market();
  throw std::invalid_argument("unknown market '" + name + "' (default, efficient or discovery)");
}

struct Fill {
  std::uint32_t filled = 0;
  double avg_price = 0;  // ticks
};

class EsSimulator {
 public:
  explicit EsSimulator(const SimParams& p = {}, std::uint64_t seed = 1, double start_second = 0) : p_(p), rng_(seed) {
    if (!(p_.dt > 0) || !(p_.cancel_rate >= 0) || !(p_.maker_decay > 0) || !(p_.meta_alpha > 0) || !(p_.fade_tau > 0))
      throw std::invalid_argument("SimParams: dt, maker_decay, meta_alpha and fade_tau must be positive");
    if (!(p_.reprice_band >= 0) || !(p_.reprice_snipe >= 0 && p_.reprice_snipe <= 1))
      throw std::invalid_argument("SimParams: reprice_band must be >= 0 and reprice_snipe in [0, 1]");
    seed_ = seed;
    if (p_.separate_streams)
      for (std::uint64_t k = 0; k < kStreams; ++k) streams_.emplace_back(stream_seed(k, 0));
    bid_ = p_.start_price;
    value_ = p_.start_price + 0.5;
    clock_ = start_second;
    for (std::size_t i = 0; i < kLevels; ++i) bid_q_[i] = ask_q_[i] = stationary_depth(i);
    history_.push_back(mid());
  }

  // Advance one step of dt seconds; trades() and own_fills() then hold this step's events.
  void step() {
    trades_.clear();
    fills_.clear();
    snipes_ = 0;
    // Counter-based streams: every step re-seeds each process from (seed,
    // process, step), so a different number of draws in one step (state-
    // dependent counts) can't shift any later step's randomness.
    ++steps_;
    for (std::size_t k = 0; k < streams_.size(); ++k) streams_[k].reseed(stream_seed(k, steps_));
    const double dt = p_.dt;
    t_ns_ += static_cast<std::int64_t>(std::llround(dt * 1e9));
    clock_ += dt;
    const double mid0 = mid();
    // 1. Activity.
    use(kValue);
    log_activity_ += -log_activity_ / p_.activity_tau * dt + p_.activity_vol * std::sqrt(dt) * r().normal();
    const double a = std::exp(log_activity_) * intraday(clock_);
    // 2. Fair value.
    value_ += p_.value_vol * std::sqrt(a * dt) * r().normal();
    if (r().uniform() < p_.jump_rate * dt) value_ += p_.jump_sd * r().normal();
    // 3. Takers.
    orders_.clear();
    use(kNoise);
    for (std::uint32_t n = poisson(p_.noise_rate * a * dt); n > 0; --n) orders_.push_back({Side::Buy, taker_size(), kNoiseTaker});
    for (std::uint32_t n = poisson(p_.noise_rate * a * dt); n > 0; --n) orders_.push_back({Side::Sell, taker_size(), kNoiseTaker});
    const double gap = value_ - mid();
    use(kMeta);
    if (std::abs(gap) > 0.5)
      for (std::uint32_t n = poisson(p_.informed_meta_rate * std::min(1.0, std::abs(gap) / p_.informed_scale) * a * dt); n > 0; --n) {
        metas_.push_back({gap > 0 ? Side::Buy : Side::Sell, meta_size(), true});
        ++meta_starts_[0];
      }
    for (std::uint32_t n = poisson(p_.uninformed_meta_rate * a * dt); n > 0; --n) {
      metas_.push_back({r().uniform() < 0.5 ? Side::Buy : Side::Sell, meta_size(), false});
      ++meta_starts_[1];
    }
    for (auto& m : metas_) {
      if (m.informed && sign(m.side) * (value_ - mid()) < -1) m.left = 0;  // the edge is gone: stop
      for (std::uint32_t n = poisson(p_.meta_child_rate * a * dt); n > 0 && m.left > 0; --n) {
        const std::uint32_t c = std::min(m.left, taker_size());
        orders_.push_back({m.side, c, m.informed ? kInformedMeta : kUninformedMeta});
        m.left -= c;
      }
    }
    std::erase_if(metas_, [](const Meta& m) { return m.left == 0; });
    use(kMomentum);
    const double move = mid() - history_.front();
    if (std::abs(move) >= 1)
      for (std::uint32_t n = poisson(p_.momentum_rate * std::min(1.0, std::abs(move) / p_.momentum_scale) * a * dt); n > 0; --n)
        orders_.push_back({move > 0 ? Side::Buy : Side::Sell, taker_size(), kMomentumTaker});
    use(kShuffle);
    for (std::size_t i = orders_.size(); i > 1; --i) std::swap(orders_[i - 1], orders_[r().below(static_cast<std::uint32_t>(i))]);
    double signed_volume = 0;
    for (const auto& o : orders_) {
      taker_volume_[o.kind] += execute(o.side, o.size, false).filled;
      signed_volume += sign(o.side) * o.size;
    }
    flow_ += (signed_volume / dt - flow_) * std::min(1.0, dt / p_.fade_tau);
    // 4. Makers, fading against the flow.
    use(kMaker);
    const double lean = std::min(p_.fade_max, std::exp(p_.fade * std::abs(flow_) / p_.fade_norm));
    double bid_cancel = p_.cancel_rate * (flow_ < 0 ? lean : 1.0), ask_cancel = p_.cancel_rate * (flow_ > 0 ? lean : 1.0);
    double bid_add = flow_ > 0 ? lean : 1.0, ask_add = flow_ < 0 ? lean : 1.0;
    if (p_.maker_info > 0) {
      const double g = value_ - mid();
      const double info = std::min(p_.maker_info_max, std::exp(p_.maker_info * std::abs(g) / p_.informed_scale));
      if (g > 0) {  // value above the price: asks get pulled, bids get added
        ask_cancel *= info;
        bid_add *= info;
      } else {
        bid_cancel *= info;
        ask_add *= info;
      }
    }
    for (std::size_t i = 0; i < kLevels; ++i) {
      const double rate = p_.maker_rate * std::exp(-static_cast<double>(i) / p_.maker_decay) * a * dt;
      bid_q_[i] += maker_lots(rate * bid_add);
      ask_q_[i] += maker_lots(rate * ask_add);
      cancel_level(Side::Buy, i, poisson(bid_cancel * others(Side::Buy, i) * dt));
      cancel_level(Side::Sell, i, poisson(ask_cancel * others(Side::Sell, i) * dt));
    }
    // 5. Icebergs.
    use(kIceberg);
    if (r().uniform() < p_.iceberg_rate * a * dt) place_iceberg();
    manage_icebergs();
    normalize_book();
    use(kMaker);
    while (spread_ > 1 && r().uniform() < p_.spread_fill_prob) improve(r().uniform() < 0.5 ? Side::Buy : Side::Sell);
    drop_far_orders();
    if (p_.reprice_band > 0) {  // opt-in: no draws and no state change when off
      use(kValue);
      reprice();
    }
    const double moved = std::abs(mid() - mid0);
    move_avg_ += (moved - move_avg_) * std::min(1.0, dt / 1800);
    log_activity_ = std::clamp(log_activity_ + p_.activity_excite * (moved - move_avg_), p_.activity_min, p_.activity_max);
    history_.push_back(mid());
    const auto keep = static_cast<std::size_t>(std::ceil(p_.momentum_lookback / dt)) + 1;
    while (history_.size() > keep) history_.pop_front();
  }

  // ---- the strategy's orders --------------------------------------------
  // Take liquidity now; prints are flagged own = true.
  Fill market_order(Side side, std::uint32_t qty) { return execute(side, qty, true); }

  // Join the back of the queue at `price`. A price at or through the
  // opposite best is rejected (returns 0): use market_order to take.
  std::uint64_t submit_limit(Side side, std::int32_t price, std::uint32_t qty) {
    if (qty == 0) return 0;
    if (side == Side::Buy ? price >= ask() : price <= bid_) return 0;
    // Inside the spread: the order becomes the new best level.
    while (side == Side::Buy ? price > bid_ : price < ask()) improve(side, false);
    const std::size_t i = level_index(side, price);
    if (i >= kLevels) return 0;  // too far from the market to simulate
    auto& q = side == Side::Buy ? bid_q_ : ask_q_;
    use(kOwn);
    const std::uint32_t faster = p_.join_ahead_mean > 0 ? lognormal_size(p_.join_ahead_mean, 0.8) : 0;
    q[i] += faster;
    resting_.push_back({next_id_, side, price, qty, q[i]});
    q[i] += qty;
    return next_id_++;
  }
  // Cancel a resting order; returns the unfilled size it removed.
  std::uint32_t cancel(std::uint64_t id) {
    for (auto it = resting_.begin(); it != resting_.end(); ++it)
      if (it->id == id) {
        const std::uint32_t left = it->qty;
        const std::size_t i = level_index(it->side, it->price);
        if (i < kLevels) {
          auto& q = it->side == Side::Buy ? bid_q_ : ask_q_;
          q[i] -= std::min(q[i], left);
        }
        resting_.erase(it);
        return left;
      }
    return 0;
  }
  struct Resting {
    std::uint64_t id;
    Side side;
    std::int32_t price;
    std::uint32_t qty;    // unfilled
    std::uint32_t ahead;  // lots ahead of it in the queue
  };
  [[nodiscard]] const std::vector<Resting>& resting() const noexcept { return resting_; }
  [[nodiscard]] const std::vector<OwnFill>& own_fills() const noexcept { return fills_; }

  // ---- market state -----------------------------------------------------
  [[nodiscard]] const std::vector<Trade>& trades() const noexcept { return trades_; }
  [[nodiscard]] Quote quote() const noexcept {
    Quote q{t_ns_, bid_, ask(), {}, {}};
    for (std::size_t i = 0; i < kDepth; ++i) {
      q.bid_size[i] = bid_q_[i];
      q.ask_size[i] = ask_q_[i];
    }
    return q;
  }
  [[nodiscard]] std::int32_t bid() const noexcept { return bid_; }
  [[nodiscard]] std::int32_t ask() const noexcept { return bid_ + spread_; }
  [[nodiscard]] std::int32_t spread() const noexcept { return spread_; }
  [[nodiscard]] double mid() const noexcept { return bid_ + 0.5 * spread_; }
  [[nodiscard]] std::int64_t t_ns() const noexcept { return t_ns_; }
  [[nodiscard]] const SimParams& params() const noexcept { return p_; }
  // Hidden from strategies; for oracles and diagnostics only.
  [[nodiscard]] double fair_value() const noexcept { return value_; }
  [[nodiscard]] double activity() const noexcept { return std::exp(log_activity_) * intraday(clock_); }
  [[nodiscard]] std::size_t active_icebergs() const noexcept { return icebergs_.size(); }
  [[nodiscard]] std::size_t active_meta_orders() const noexcept { return metas_.size(); }
  // Diagnostics (no effect on the market): who traded, how often quotes jumped.
  enum TakerKind : std::uint8_t { kNoiseTaker, kInformedMeta, kUninformedMeta, kMomentumTaker, kSniper, kTakerKinds };
  [[nodiscard]] std::uint64_t taker_volume(TakerKind k) const noexcept { return taker_volume_[k]; }  // lots filled since the start
  [[nodiscard]] std::uint64_t meta_starts(bool informed) const noexcept { return meta_starts_[informed ? 0 : 1]; }
  [[nodiscard]] std::uint64_t reprices() const noexcept { return reprices_; }            // quote jumps since the start
  [[nodiscard]] std::uint64_t reprice_ticks() const noexcept { return reprice_ticks_; }  // sum of |jump| in ticks
  // Prints of the last step() made by snipers: they are the last sniper_prints() entries of
  // trades() only until the strategy trades (market_order() appends its own prints after them).
  [[nodiscard]] std::size_t sniper_prints() const noexcept { return snipes_; }
  [[nodiscard]] double log_activity() const noexcept { return log_activity_; }  // clamped to [activity_min, activity_max]

 private:
  struct Order {
    Side side;
    std::uint32_t size;
    TakerKind kind;
  };
  struct Meta {
    Side side;
    std::uint32_t left;
    bool informed;
  };
  struct Iceberg {
    Side side;  // Buy: a hidden bid
    std::int32_t price;
    std::uint32_t reserve;
    double expires;
  };

  [[nodiscard]] double intraday(double sec) const noexcept {
    const double m = p_.boost_minutes * 60;
    const double to_close = std::max(0.0, p_.session_seconds - sec);
    return 1 + p_.open_boost * std::exp(-sec / m) + p_.close_boost * std::exp(-to_close / m);
  }
  [[nodiscard]] std::uint32_t stationary_depth(std::size_t i) const {
    const double lots = p_.maker_rate * std::exp(-static_cast<double>(i) / p_.maker_decay) * p_.maker_size_mean / std::max(1e-9, p_.cancel_rate);
    return static_cast<std::uint32_t>(std::min(1e6, lots));
  }
  [[nodiscard]] std::size_t level_index(Side side, std::int32_t price) const noexcept {
    const std::int32_t d = side == Side::Buy ? bid_ - price : price - ask();
    return d < 0 ? kLevels : static_cast<std::size_t>(d);
  }
  // Lots at a level that aren't the strategy's.
  [[nodiscard]] std::uint32_t others(Side side, std::size_t i) const noexcept {
    const auto& q = side == Side::Buy ? bid_q_ : ask_q_;
    std::uint32_t own = 0;
    for (const auto& r : resting_)
      if (r.side == side && level_index(side, r.price) == i) own += r.qty;
    return q[i] > own ? q[i] - own : 0;
  }

  std::uint32_t poisson(double lambda) {
    if (!(lambda > 0)) return 0;
    if (lambda > 30) return static_cast<std::uint32_t>(std::max(0.0, std::round(lambda + std::sqrt(lambda) * r().normal())));
    const double l = std::exp(-lambda);
    std::uint32_t k = 0;
    double prod = r().uniform();
    while (prod > l && k < 200) {
      ++k;
      prod *= r().uniform();
    }
    return k;
  }
  std::uint32_t lognormal_size(double mean, double logsd) {
    const double mu = std::log(mean) - 0.5 * logsd * logsd;
    return static_cast<std::uint32_t>(std::clamp(std::round(std::exp(mu + logsd * r().normal())), 1.0, static_cast<double>(p_.max_order)));
  }
  std::uint32_t taker_size() { return lognormal_size(p_.taker_size_mean, p_.taker_size_logsd); }
  std::uint32_t meta_size() {
    const double u = std::max(1e-12, r().uniform());
    return static_cast<std::uint32_t>(std::min(p_.meta_max, p_.meta_min * std::pow(u, -1.0 / p_.meta_alpha)));
  }
  std::uint32_t maker_lots(double rate) {
    std::uint32_t lots = 0;
    for (std::uint32_t n = poisson(rate); n > 0; --n) lots += lognormal_size(p_.maker_size_mean, 0.8);
    return lots;
  }

  // Others cancel `n` lots at a level; the strategy's queue position improves
  // in proportion (cancellations land uniformly in the queue).
  void cancel_level(Side side, std::size_t i, std::uint32_t n) {
    const std::uint32_t oth = others(side, i);
    n = std::min(n, oth);
    if (n == 0) return;
    auto& q = side == Side::Buy ? bid_q_ : ask_q_;
    q[i] -= n;
    for (auto& r : resting_)
      if (r.side == side && level_index(side, r.price) == i && oth > 0)
        r.ahead -= std::min(r.ahead, static_cast<std::uint32_t>(std::llround(
                                         static_cast<double>(n) * std::pow(static_cast<double>(r.ahead) / oth, p_.cancel_back_bias))));
  }

  Iceberg* iceberg_at(Side side, std::int32_t price) {
    for (auto& ib : icebergs_)
      if (ib.side == side && ib.price == price && ib.reserve > 0) return &ib;
    return nullptr;
  }

  // Take `take` lots from the best level of the passive side: the queue
  // ahead of each resting own order first, then the order itself (FIFO).
  void consume_best(Side passive, std::uint32_t take) {
    auto& q = passive == Side::Buy ? bid_q_ : ask_q_;
    const std::int32_t price = passive == Side::Buy ? bid_ : ask();
    q[0] -= std::min(q[0], take);
    for (auto it = resting_.begin(); it != resting_.end();) {
      if (it->side != passive || it->price != price) {
        ++it;
        continue;
      }
      const std::uint32_t past = std::min(take, it->ahead);
      it->ahead -= past;
      const std::uint32_t mine = std::min(take - past, it->qty);
      if (mine > 0) {
        fills_.push_back({it->id, passive, price, mine});
        it->qty -= mine;
      }
      if (it->qty == 0) {
        it = resting_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // consume_best at any level: take `take` lots from level `level` (price
  // `price`) of the passive side, FIFO through the strategy's orders there.
  void consume_at(Side passive, std::size_t level, std::int32_t price, std::uint32_t take) {
    auto& q = passive == Side::Buy ? bid_q_ : ask_q_;
    q[level] -= std::min(q[level], take);
    for (auto it = resting_.begin(); it != resting_.end();) {
      if (it->side != passive || it->price != price) {
        ++it;
        continue;
      }
      const std::uint32_t past = std::min(take, it->ahead);
      it->ahead -= past;
      const std::uint32_t mine = std::min(take - past, it->qty);
      if (mine > 0) {
        fills_.push_back({it->id, passive, price, mine});
        it->qty -= mine;
      }
      if (it->qty == 0) {
        it = resting_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Quote jumps (reprice_band > 0): see SimParams. Runs at the end of a step,
  // so every quote a strategy sees has |fair value - mid| <= max(band, 0.5).
  void reprice() {
    const double g = value_ - mid();
    if (!(std::abs(g) > p_.reprice_band)) return;
    const auto k = static_cast<std::int32_t>(std::lround(g));
    if (k == 0) return;
    const Side up = k > 0 ? Side::Buy : Side::Sell;      // the snipers' side
    const Side stale = k > 0 ? Side::Sell : Side::Buy;  // the quotes value just passed
    // The makers' lots per level (the strategy's taken out): the shape that moves.
    std::array<std::uint32_t, kLevels> mb = bid_q_, ma = ask_q_;
    for (const auto& o : resting_) {
      const std::size_t i = level_index(o.side, o.price);
      if (i >= kLevels) continue;
      auto& q = o.side == Side::Buy ? mb : ma;
      q[i] -= std::min(q[i], o.qty);
    }
    // Snipers race the cancels at each stale level the moved book would
    // cross; FIFO, so the strategy's orders there fill by queue position.
    const std::int32_t mag = k > 0 ? k : -k;
    const auto crossed = static_cast<std::size_t>(std::clamp<std::int32_t>(mag - spread_ + 1, 0, static_cast<std::int32_t>(kLevels)));
    const auto& sq = stale == Side::Sell ? ask_q_ : bid_q_;
    // Snipers take a fixed clip per level (reprice_snipe x the stationary
    // best-level size; their size is set by their risk limits, not by the
    // queue), so their prints scale with the jump, not with the book.
    const auto clip = static_cast<std::uint32_t>(std::llround(p_.reprice_snipe * static_cast<double>(stationary_depth(0))));
    for (std::size_t j = 0; j < crossed; ++j) {
      const auto off = static_cast<std::int32_t>(j);
      const std::int32_t px = stale == Side::Sell ? ask() + off : bid_ - off;
      auto take = std::min(sq[j], clip);
      while (take > 0) {
        const std::uint32_t c = std::min(take, taker_size());
        consume_at(stale, j, px, c);
        trades_.push_back({t_ns_, px, c, up, false});
        taker_volume_[kSniper] += c;
        ++snipes_;
        take -= c;
      }
    }
    // Every maker order moves k ticks (sniped lots are re-posted by the
    // makers who won the race: the shape is kept).
    bid_q_ = mb;
    ask_q_ = ma;
    bid_ += k;
    // Informed icebergs that value passed are pulled.
    std::erase_if(icebergs_, [&](const Iceberg& ib) { return ib.side == stale && (stale == Side::Sell ? ib.price <= bid_ : ib.price >= ask()); });
    // The strategy's orders keep their price: crossed ones are filled there
    // by the snipers; the rest rejoin at the back of the queue at their
    // price; beyond the simulated depth they are dropped.
    for (auto it = resting_.begin(); it != resting_.end();) {
      const bool through = it->side == Side::Sell ? it->price <= bid_ : it->price >= ask();
      if (through) {
        fills_.push_back({it->id, it->side, it->price, it->qty});
        trades_.push_back({t_ns_, it->price, it->qty, up, false});
        taker_volume_[kSniper] += it->qty;
        ++snipes_;
        it = resting_.erase(it);
        continue;
      }
      // Beyond the simulated depth, or left inside a wide spread (rare: the
      // spread is 1 tick > 99% of the time), the order is cancelled.
      const std::size_t i = level_index(it->side, it->price);
      if (i >= kLevels) {
        it = resting_.erase(it);
        continue;
      }
      auto& q = it->side == Side::Buy ? bid_q_ : ask_q_;
      it->ahead = q[i];
      q[i] += it->qty;
      ++it;
    }
    ++reprices_;
    reprice_ticks_ += static_cast<std::uint64_t>(mag);
  }

  // Take liquidity: a Buy consumes asks, a Sell consumes bids, level by level.
  Fill execute(Side side, std::uint32_t qty, bool own) {
    Fill f;
    double notional = 0;
    std::uint32_t left = qty;
    auto& q = side == Side::Buy ? ask_q_ : bid_q_;
    const Side passive = side == Side::Buy ? Side::Sell : Side::Buy;
    for (int guard = 0; left > 0 && guard < 1000; ++guard) {
      const std::int32_t price = side == Side::Buy ? ask() : bid_;
      if (q[0] == 0) {
        if (Iceberg* ib = iceberg_at(passive, price)) {  // hidden reserve refills the displayed queue
          const auto refill = std::min<std::uint32_t>(static_cast<std::uint32_t>(p_.iceberg_peak), ib->reserve);
          q[0] += refill;
          ib->reserve -= refill;
          continue;
        }
        shift_out(side);
        continue;
      }
      const std::uint32_t take = std::min(left, q[0]);
      consume_best(passive, take);
      left -= take;
      f.filled += take;
      notional += static_cast<double>(take) * price;
      trades_.push_back({t_ns_, price, take, side, own});
    }
    f.avg_price = f.filled ? notional / f.filled : 0;
    return f;
  }

  // The best level on the side being taken is gone: the next one becomes best.
  void shift_out(Side taker) {
    auto& q = taker == Side::Buy ? ask_q_ : bid_q_;
    for (std::size_t i = 0; i + 1 < kLevels; ++i) q[i] = q[i + 1];
    q[kLevels - 1] = stationary_depth(kLevels - 1);
    if (taker == Side::Sell) --bid_;
    ++spread_;
  }

  // A new best level inside the spread: a better bid (Buy) or ask (Sell),
  // opened by a maker's order or empty (the strategy's order joins it).
  void improve(Side maker, bool with_maker_order = true) {
    auto& q = maker == Side::Buy ? bid_q_ : ask_q_;
    for (std::size_t i = kLevels - 1; i > 0; --i) q[i] = q[i - 1];
    q[0] = with_maker_order ? lognormal_size(p_.maker_size_mean, 0.8) : 0;
    if (maker == Side::Buy) ++bid_;
    --spread_;
  }

  // Cancellations can empty the best level; the price then moves to the next.
  void normalize_book() {
    for (int g = 0; g < 4 && bid_q_[0] == 0 && !iceberg_at(Side::Buy, bid_); ++g) shift_out(Side::Sell);
    for (int g = 0; g < 4 && ask_q_[0] == 0 && !iceberg_at(Side::Sell, ask()); ++g) shift_out(Side::Buy);
  }

  // Own orders pushed beyond the simulated depth are cancelled.
  void drop_far_orders() {
    std::erase_if(resting_, [&](const Resting& r) { return level_index(r.side, r.price) >= kLevels; });
  }

  void place_iceberg() {
    // Informed passive player: below value, buyers wait under the price;
    // above value, sellers wait over it.
    const Side side = value_ >= mid() ? Side::Buy : Side::Sell;
    const auto offset = static_cast<std::int32_t>(1 + r().below(static_cast<std::uint32_t>(p_.iceberg_max_offset)));
    const std::int32_t price = side == Side::Buy ? bid_ - offset : ask() + offset;
    const auto reserve = static_cast<std::uint32_t>(p_.iceberg_min + r().uniform() * (p_.iceberg_max - p_.iceberg_min));
    icebergs_.push_back({side, price, reserve, clock_ + p_.iceberg_life});
  }

  void manage_icebergs() {
    std::erase_if(icebergs_, [&](const Iceberg& ib) {
      const bool wrong = ib.side == Side::Buy ? value_ < ib.price - 1 : value_ > ib.price + 1;
      return ib.reserve == 0 || clock_ > ib.expires || wrong;
    });
    for (const auto& ib : icebergs_) {  // a resting iceberg always shows its peak
      const std::size_t i = level_index(ib.side, ib.price);
      if (i >= kLevels) continue;
      auto& lvl = ib.side == Side::Buy ? bid_q_[i] : ask_q_[i];
      lvl = std::max<std::uint32_t>(lvl, static_cast<std::uint32_t>(p_.iceberg_peak));
    }
  }

  // Random streams (separate_streams): indices into streams_; the shared
  // stream rng_ otherwise. An index, not a pointer: simulators are copied.
  static constexpr std::size_t kValue = 0, kNoise = 1, kMeta = 2, kMomentum = 3, kShuffle = 4, kMaker = 5, kIceberg = 6, kOwn = 7, kStreams = 8;
  void use(std::size_t k) noexcept { cur_ = streams_.empty() ? kStreams : k; }
  [[nodiscard]] std::uint64_t stream_seed(std::uint64_t k, std::uint64_t step) const noexcept {
    std::uint64_t z = seed_ ^ (0x9E3779B97F4A7C15ull * (k + 1)) ^ (step * 0xD1B54A32D192ED03ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;  // splitmix64 finalizer
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  Rng& r() noexcept { return cur_ < streams_.size() ? streams_[cur_] : rng_; }

  SimParams p_;
  Rng rng_;
  std::vector<Rng> streams_;
  std::size_t cur_ = kStreams;
  std::uint64_t seed_ = 1, steps_ = 0;
  std::int32_t bid_ = 0, spread_ = 1;
  std::array<std::uint32_t, kLevels> bid_q_{}, ask_q_{};
  double value_ = 0, log_activity_ = 0, clock_ = 0, flow_ = 0, move_avg_ = 0.1;
  std::int64_t t_ns_ = 0;
  std::uint64_t next_id_ = 1;
  std::deque<double> history_;
  std::vector<Trade> trades_;
  std::vector<OwnFill> fills_;
  std::vector<Order> orders_;
  std::vector<Meta> metas_;
  std::vector<Iceberg> icebergs_;
  std::vector<Resting> resting_;
  std::array<std::uint64_t, kTakerKinds> taker_volume_{};
  std::array<std::uint64_t, 2> meta_starts_{};  // informed, uninformed
  std::uint64_t reprices_ = 0, reprice_ticks_ = 0;
  std::size_t snipes_ = 0;
};

}  // namespace rag::orderflow
