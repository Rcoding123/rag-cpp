// orderflow_tests.cpp — the ES order-flow simulator: book invariants,
// determinism, market orders, and the FIFO queue mechanics tick-scalping
// strategies depend on.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "rag/orderflow/simulator.hpp"

using namespace rag::orderflow;

TEST(EsSimulator, BookStaysConsistentAndPrintsAreAtTheTouch) {
  EsSimulator sim(SimParams{}, 7);
  std::size_t prints = 0;
  for (int s = 0; s < 20'000; ++s) {
    const std::int32_t bid0 = sim.bid(), ask0 = sim.ask();
    sim.step();
    ASSERT_GE(sim.spread(), 1);
    ASSERT_LT(sim.bid(), sim.ask());
    for (const auto& t : sim.trades()) {
      ++prints;
      ASSERT_GT(t.size, 0u);
      ASSERT_FALSE(t.own);
      // A buy prints at or above the ask it started from (sweeping moves it up),
      // a sell at or below the bid; never more than the simulated depth away.
      if (t.aggressor == Side::Buy) {
        ASSERT_GE(t.price, ask0 - static_cast<std::int32_t>(kLevels));
      } else {
        ASSERT_LE(t.price, bid0 + static_cast<std::int32_t>(kLevels));
      }
    }
    const Quote q = sim.quote();
    ASSERT_EQ(q.bid, sim.bid());
    ASSERT_EQ(q.ask, sim.ask());
  }
  EXPECT_GT(prints, 10'000u);  // ~2,000 s of trading
  EXPECT_LT(std::abs(sim.mid() - 20'000.0), 2'000.0);
}

TEST(EsSimulator, SameSeedSamePathDifferentSeedDifferentPath) {
  auto path = [](std::uint64_t seed) {
    EsSimulator sim(SimParams{}, seed);
    std::vector<double> m;
    for (int s = 0; s < 3000; ++s) {
      sim.step();
      m.push_back(sim.mid());
    }
    return m;
  };
  EXPECT_EQ(path(3), path(3));
  EXPECT_NE(path(3), path(4));
}

TEST(EsSimulator, MarketOrderTakesTheBestPrices) {
  EsSimulator sim(SimParams{}, 1);
  const std::int32_t ask = sim.ask();
  const std::uint32_t at_ask = sim.quote().ask_size[0];
  const Fill f = sim.market_order(Side::Buy, 3);
  EXPECT_EQ(f.filled, 3u);
  EXPECT_EQ(f.avg_price, ask);  // the best level has far more than 3 lots
  ASSERT_EQ(sim.trades().size(), 1u);
  EXPECT_TRUE(sim.trades()[0].own);
  EXPECT_EQ(sim.quote().ask_size[0], at_ask - 3);
}

TEST(EsSimulator, LimitOrderFillsOnlyAfterTheQueueAheadTrades) {
  EsSimulator sim(SimParams{.join_ahead_mean = 0}, 1);  // no faster participants: we join right behind the visible queue
  const std::int32_t bid = sim.bid();
  const std::uint32_t queue = sim.quote().bid_size[0];
  const std::uint64_t id = sim.submit_limit(Side::Buy, bid, 2);
  ASSERT_NE(id, 0u);
  ASSERT_EQ(sim.resting().size(), 1u);
  EXPECT_EQ(sim.resting()[0].ahead, queue);  // joined the back of the queue
  EXPECT_EQ(sim.quote().bid_size[0], queue + 2);
  // Selling exactly the lots ahead fills none of ours (FIFO)...
  sim.market_order(Side::Sell, queue);
  EXPECT_TRUE(sim.own_fills().empty());
  EXPECT_EQ(sim.resting()[0].ahead, 0u);
  EXPECT_EQ(sim.bid(), bid);  // our 2 lots still hold the level
  // ...one more lot fills one of ours, at our price.
  sim.market_order(Side::Sell, 1);
  ASSERT_EQ(sim.own_fills().size(), 1u);
  EXPECT_EQ(sim.own_fills()[0].id, id);
  EXPECT_EQ(sim.own_fills()[0].size, 1u);
  EXPECT_EQ(sim.own_fills()[0].price, bid);
  EXPECT_EQ(sim.resting()[0].qty, 1u);
  // Cancel returns what's left and takes it off the book.
  const std::uint32_t before = sim.quote().bid_size[0];
  EXPECT_EQ(sim.cancel(id), 1u);
  EXPECT_TRUE(sim.resting().empty());
  EXPECT_EQ(sim.quote().bid_size[0], before - 1);
  EXPECT_EQ(sim.cancel(id), 0u);
}

TEST(EsSimulator, QueuePositionImprovesAsOthersCancel) {
  SimParams p;
  // No takers and no icebergs: new maker orders only join behind ours, so
  // only cancellations can move it forward.
  p.noise_rate = 0;
  p.informed_meta_rate = 0;
  p.uninformed_meta_rate = 0;
  p.momentum_rate = 0;
  p.iceberg_rate = 0;
  EsSimulator sim(p, 2);
  const std::uint64_t id = sim.submit_limit(Side::Sell, sim.ask(), 1);
  ASSERT_NE(id, 0u);
  const std::uint32_t ahead0 = sim.resting()[0].ahead;
  for (int s = 0; s < 100; ++s) sim.step();  // 10 s at 10% cancellations per second
  ASSERT_EQ(sim.resting().size(), 1u);
  EXPECT_LT(sim.resting()[0].ahead, ahead0);
  EXPECT_GT(sim.resting()[0].ahead, ahead0 / 10);
  EXPECT_EQ(sim.resting()[0].qty, 1u);  // nobody traded
}

TEST(EsSimulator, LimitOrdersThroughTheMarketAreRejected) {
  EsSimulator sim(SimParams{}, 1);
  EXPECT_EQ(sim.submit_limit(Side::Buy, sim.ask(), 1), 0u);
  EXPECT_EQ(sim.submit_limit(Side::Sell, sim.bid(), 1), 0u);
  EXPECT_EQ(sim.submit_limit(Side::Buy, sim.bid(), 0), 0u);
  EXPECT_THROW(EsSimulator(SimParams{.dt = 0}), std::invalid_argument);
}

// --- features.hpp and scalp_env.hpp ------------------------------------------
#include "rag/orderflow/scalp_env.hpp"

TEST(FeatureBuilder, WindowSumsMatchTheTapeAndValuesStayBounded) {
  EsSimulator sim(SimParams{}, 5);
  FeatureBuilder fb(0.1);
  std::vector<double> deltas;
  for (int s = 0; s < 2000; ++s) {
    sim.step();
    double d = 0;
    for (const auto& t : sim.trades()) {
      fb.add_trade(t);
      d += sign(t.aggressor) * t.size;
    }
    fb.end_bucket(sim.quote());
    deltas.push_back(d);
    for (double x : fb.values()) {
      ASSERT_TRUE(std::isfinite(x));
      ASSERT_LE(std::abs(x), kClip);
    }
  }
  EXPECT_TRUE(fb.warm());  // 200 s > the 120 s window
  double d5 = 0, d30 = 0;
  for (std::size_t i = 0; i < 50; ++i) d5 += deltas[deltas.size() - 1 - i];
  for (std::size_t i = 0; i < 300; ++i) d30 += deltas[deltas.size() - 1 - i];
  EXPECT_NEAR(fb.raw()[0], d5, 1e-6);   // delta over 5 s = 50 buckets
  EXPECT_NEAR(fb.raw()[1], d30, 1e-6);  // 30 s
}

TEST(ScalpEnv, RewardsAddUpToNetPnlAndFlatEarnsNothing) {
  ScalpParams p;
  p.episode_seconds = 120;
  ScalpEnv env(p);
  for (rag::agent::Action fixed : {rag::agent::Action{1}, rag::agent::Action{2}}) {
    rag::agent::Rng r(11);
    auto o = env.reset(r);
    double total = 0;
    int steps = 0;
    for (;;) {
      const auto st = env.step(fixed, r);
      total += st.reward;
      ++steps;
      o = st.obs;
      if (st.done) break;
    }
    EXPECT_EQ(steps, 120);
    EXPECT_EQ(o.position, 0);  // the episode ends flat
    EXPECT_NEAR(total, env.net_pnl_ticks(), 1e-6);
    if (fixed == 1) {
      EXPECT_EQ(total, 0.0);
      EXPECT_EQ(env.orders(), 0u);
    } else {
      EXPECT_EQ(env.orders(), 2u);  // in once, out at the end
      EXPECT_NEAR(env.fees_ticks(), 2 * p.fee_dollars / kTickValue, 1e-12);
    }
  }
}

TEST(ScalpEnv, TradingEveryStepLosesTheSpreadAndFees) {
  ScalpParams p;
  p.episode_seconds = 60;
  ScalpEnv env(p);
  rag::agent::Rng r(3);
  env.reset(r);
  for (int i = 0; i < 60; ++i) env.step(static_cast<rag::agent::Action>(i % 2 == 0 ? 0 : 2), r);  // flip short/long every second
  // ~60 flips of 2 contracts: each pays about a spread (1 tick) per contract plus fees.
  EXPECT_LT(env.net_pnl_ticks(), -40.0);
}

// --- tick_env.hpp ------------------------------------------------------------
#include "rag/orderflow/tick_env.hpp"

TEST(TickScalpEnv, InventoryStaysWithinLimitsAndRewardsAddUp) {
  TickParams p;
  p.episode_seconds = 120;
  TickScalpEnv env(p);
  for (rag::agent::Action mode : {rag::agent::Action{0}, rag::agent::Action{3}, rag::agent::Action{1}}) {
    rag::agent::Rng r(21);
    auto o = env.reset(r);
    double total = 0;
    for (;;) {
      const auto st = env.step(mode, r);
      total += st.reward;
      o = st.obs;
      ASSERT_LE(std::abs(o.position), p.max_position);
      if (st.done) break;
    }
    EXPECT_EQ(o.position, 0);
    EXPECT_NEAR(total, env.net_pnl_ticks(), 1e-6);
    if (mode == 0) {
      EXPECT_EQ(env.fills(), 0u);
      EXPECT_EQ(total, 0.0);
    } else {
      EXPECT_GT(env.fills(), 0u);  // two minutes at the touch gets filled
    }
  }
}

TEST(TickScalpEnv, QueuePositionIsObservedWhileQuoting) {
  TickScalpEnv env;
  rag::agent::Rng r(5);
  env.reset(r);
  const auto st = env.step(3, r);  // quote both sides for 0.5 s
  // Unless already filled, each side reports a queue position in (0, ~2].
  if (st.obs.position == 0) {
    EXPECT_GT(st.obs.bid_queue, 0.0);
    EXPECT_GT(st.obs.ask_queue, 0.0);
  }
  const auto none = env.step(0, r);
  EXPECT_EQ(none.obs.bid_queue, -1.0);
  EXPECT_EQ(none.obs.ask_queue, -1.0);
}

// Why OF5's agent never learned to quote (and why OF7 guides exploration):
// a fill needs the quote to stay put while the ~120 lots ahead of it trade
// or cancel, and uniformly random modes cancel a side at most decisions, so
// the order rejoins at the back again and again. Measured on 120 s episodes
// (seeds 31-38): quoting both sides throughout got 6-14 fills each, uniform
// random modes got 0 every time (300 s validation episodes: 24.3 vs 0.10 per
// episode).
TEST(TickScalpEnv, OnlyQuotesThatStayPutReachTheFrontOfTheQueue) {
  TickParams p;
  p.episode_seconds = 120;
  std::uint32_t persistent = 0, flickering = 0;
  for (std::uint64_t seed = 31; seed < 35; ++seed) {
    for (const bool random_modes : {false, true}) {
      TickScalpEnv env(p);
      rag::agent::Rng r(seed), pick(seed + 1000);
      env.reset(r);
      while (!env.step(random_modes ? pick.below(5) : rag::agent::Action{3}, r).done) {
      }
      (random_modes ? flickering : persistent) += env.fills();
    }
  }
  EXPECT_GE(persistent, 20u);  // measured 34 (10 + 6 + 10 + 8)
  EXPECT_LE(flickering, 2u);   // measured 0
}

TEST(EsSimulator, SeparateStreamsKeepTheFairValuePathIndependentOfOurOrders) {
  // With one stream per exogenous process, our orders can't change the
  // news shocks. The fair-value *path* also depends on activity, which price
  // moves excite, so it is independent of our orders only without that
  // excitation. (Endogenous processes still diverge once the book differs;
  // see ORDERFLOW.md.)
  SimParams p;
  p.separate_streams = true;
  p.activity_excite = 0;
  EsSimulator quiet(p, 9), active(p, 9);
  for (int s = 0; s < 3000; ++s) {
    if (s % 20 == 0) active.submit_limit(Side::Buy, active.bid(), 1);
    if (s % 37 == 0) active.market_order(Side::Sell, 2);
    quiet.step();
    active.step();
    ASSERT_EQ(quiet.fair_value(), active.fair_value()) << "step " << s;
  }
  // The shared default stream does not have this property.
  EsSimulator q2(SimParams{.activity_excite = 0}, 9), a2(SimParams{.activity_excite = 0}, 9);
  bool diverged = false;
  for (int s = 0; s < 3000 && !diverged; ++s) {
    if (s % 20 == 0) a2.submit_limit(Side::Buy, a2.bid(), 1);
    q2.step();
    a2.step();
    diverged = q2.fair_value() != a2.fair_value();
  }
  EXPECT_TRUE(diverged);
}

// FNV-1a over every trade and quote of `steps` steps: a fingerprint of a run.
static std::uint64_t run_fingerprint(const SimParams& p, std::uint64_t seed, int steps = 20'000) {
  EsSimulator sim(p, seed);
  std::uint64_t h = 14695981039346656037ull;
  auto mix = [&h](std::uint64_t v) {
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xffu;
      h *= 1099511628211ull;
    }
  };
  for (int s = 0; s < steps; ++s) {
    sim.step();
    for (const auto& t : sim.trades()) {
      mix(static_cast<std::uint64_t>(t.t_ns));
      mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(t.price)));
      mix(t.size);
      mix(t.aggressor == Side::Buy ? 1u : 2u);
      mix(t.own ? 1u : 0u);
    }
    const Quote q = sim.quote();
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(q.bid)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(q.ask)));
    for (std::size_t i = 0; i < kDepth; ++i) {
      mix(q.bid_size[i]);
      mix(q.ask_size[i]);
    }
  }
  return h;
}

TEST(EsSimulator, GoldenRunsOfTheExistingPresetsAreUnchanged) {
  // Recorded with the code before the quote-jump mechanism (commit e0a4c7a),
  // MSVC x64. Every OF1-OF9 result depends on these paths staying
  // bit-identical. Floating-point library functions (exp, log, pow) and FMA
  // contraction differ between compilers, so the exact values are checked
  // only where they were recorded; the opt-in test below runs everywhere.
  SimParams sep;
  sep.separate_streams = true;
  const std::uint64_t a = run_fingerprint(SimParams{}, 1), b = run_fingerprint(efficient_market(), 2), c = run_fingerprint(sep, 3);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(a, 275146979745797631ull) << a;
  EXPECT_EQ(b, 8508263057256671083ull) << b;
  EXPECT_EQ(c, 12097387368164182254ull) << c;
#else
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
#endif
}

// --- quote jumps (SimParams::reprice_band, discovery_market()) ---------------

TEST(PriceDiscovery, OffByDefaultAndTheSniperSizeAloneChangesNothing) {
  // reprice_band = 0 switches the mechanism off: no draws, no state change.
  SimParams p;
  p.reprice_snipe = 0.5;
  EXPECT_EQ(run_fingerprint(p, 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  SimParams on;
  on.reprice_band = 0.65;
  EXPECT_NE(run_fingerprint(on, 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  EXPECT_THROW(EsSimulator(SimParams{.reprice_band = -1}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.reprice_snipe = 1.5}), std::invalid_argument);
  EXPECT_THROW(market_preset("fast"), std::invalid_argument);
  EXPECT_EQ(market_preset("discovery").reprice_band, discovery_market().reprice_band);
}

TEST(PriceDiscovery, QuotesTrackValueAndRequeuedOrdersSitBehindTheLevel) {
  SimParams p = discovery_market_v1();
  p.jump_rate = 1.0;  // news about every 10 steps
  EsSimulator sim(p, 11);
  rag::agent::Rng rng(3);
  std::uint64_t jumps = 0;
  for (int s = 0; s < 20'000; ++s) {
    if (s % 7 == 0) {
      const Side side = rng.uniform() < 0.5 ? Side::Buy : Side::Sell;
      const auto back = static_cast<std::int32_t>(rng.below(3));
      sim.submit_limit(side, side == Side::Buy ? sim.bid() - back : sim.ask() + back, 1 + rng.below(3));
    }
    if (s % 13 == 0 && !sim.resting().empty()) sim.cancel(sim.resting().front().id);
    const std::uint64_t before = sim.reprices();
    sim.step();
    // Every quote a strategy sees: value within max(band, 0.5) of the mid.
    ASSERT_LE(std::abs(sim.fair_value() - sim.mid()), std::max(p.reprice_band, 0.5) + 1e-9) << "step " << s;
    ASSERT_GE(sim.spread(), 1);
    if (sim.reprices() == before) continue;
    ++jumps;
    // Orders that kept their price rejoined the back of the queue there.
    const Quote q = sim.quote();
    for (const auto& r : sim.resting()) {
      const std::int32_t d = r.side == Side::Buy ? sim.bid() - r.price : r.price - sim.ask();
      ASSERT_GE(d, 0);
      if (d < static_cast<std::int32_t>(kDepth)) {
        const auto i = static_cast<std::size_t>(d);
        ASSERT_LE(r.ahead + r.qty, r.side == Side::Buy ? q.bid_size[i] : q.ask_size[i]) << "step " << s;
      }
    }
  }
  EXPECT_GT(jumps, 500u);
}

TEST(PriceDiscovery, AStaleOwnAskIsFilledAtItsPriceWhenValueJumpsThroughIt) {
  // Takers off, a news jump every step: the book moves only by quote jumps.
  SimParams p = discovery_market_v1();
  p.noise_rate = p.informed_meta_rate = p.uninformed_meta_rate = p.momentum_rate = 0;
  p.jump_rate = 10;
  p.jump_sd = 4;
  EsSimulator sim(p, 5);
  int crossed = 0;
  for (int s = 0; s < 2'000 && crossed < 20; ++s) {
    const std::int32_t ask = sim.ask();
    const std::uint64_t id = sim.submit_limit(Side::Sell, ask, 2);
    ASSERT_NE(id, 0u);
    sim.step();
    if (sim.bid() < ask) {
      sim.cancel(id);
      continue;
    }
    ++crossed;  // value jumped up through our ask: the snipers took it at our (stale) price
    std::uint32_t lots = 0;
    for (const auto& f : sim.own_fills())
      if (f.id == id) {
        EXPECT_EQ(f.price, ask);
        EXPECT_EQ(f.side, Side::Sell);
        lots += f.size;
      }
    EXPECT_EQ(lots, 2u);
    for (const auto& r : sim.resting()) EXPECT_NE(r.id, id);
    EXPECT_GT(sim.sniper_prints(), 0u);
  }
  EXPECT_GE(crossed, 10);
}

// corr(fair value - mid, mid change over the next 30 s), over `seconds` of trading.
static double gap_predicts_30s(const SimParams& p, std::uint64_t seed, double seconds) {
  EsSimulator sim(p, seed, 3600);
  const auto n = static_cast<std::size_t>(std::lround(seconds / p.dt)), h = static_cast<std::size_t>(std::lround(30 / p.dt));
  std::vector<double> g, m;
  for (std::size_t s = 0; s < n + h; ++s) {
    sim.step();
    g.push_back(sim.fair_value() - sim.mid());
    m.push_back(sim.mid());
  }
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = g[i], y = m[i + h] - m[i];
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
  }
  const double k = static_cast<double>(n);
  return (sxy - sx * sy / k) / std::sqrt((sxx - sx * sx / k) * (syy - sy * sy / k));
}

TEST(PriceDiscovery, TheHiddenGapNoLongerPredictsTheNext30Seconds) {
  // 2 h, seed 1. Measured (MSVC): default 0.62, discovery v1 0.13
  // (OF10 seeds 1-3: default 0.54-0.62, discovery ~0.1, the 1-tick grid's floor ~0.09-0.12).
  const double def = gap_predicts_30s(SimParams{}, 1, 7200), disc = gap_predicts_30s(discovery_market_v1(), 1, 7200);
  EXPECT_GT(def, 0.4) << def;
  EXPECT_LT(disc, 0.2) << disc;
}

TEST(PriceDiscovery, GoldenRunOfDiscoveryV1IsUnchanged) {
  // Recorded with the v1 code (commit 10bfe03, then discovery_market()), MSVC x64:
  // every v1 number in the OF10 history depends on it.
  const std::uint64_t d = run_fingerprint(discovery_market_v1(), 4);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(d, 13558066813598808490ull) << d;
#else
  EXPECT_NE(d, run_fingerprint(SimParams{}, 4));
#endif
  EXPECT_EQ(market_preset("discovery_v1").reprice_snipe, discovery_market_v1().reprice_snipe);
  EXPECT_GT(market_preset("discovery").reprice_cancel, 0);
}

TEST(PriceDiscovery, GoldenRunOfDiscoveryV2IsUnchanged) {
  // Recorded with the v2 code (commit 4ba0354, then discovery_market()), MSVC x64,
  // before iteration 3 changed discovery_market(): every v2 number depends on it.
  const std::uint64_t d = run_fingerprint(discovery_market_v2(), 5);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(d, 7055594995802652644ull) << d;  // = run_fingerprint(discovery_market(), 5) at 4ba0354
#else
  EXPECT_NE(d, run_fingerprint(SimParams{}, 5));
#endif
  EXPECT_EQ(market_preset("discovery_v2").maker_info, discovery_market_v2().maker_info);
}

TEST(PriceDiscovery, GoldenRunOfDiscoveryV3IsUnchanged) {
  // Recorded at 708b1c8 (OF12), MSVC x64, before OF13 added the v4 fields; with
  // EsSimulatorActivity.GoldenRunsOfTheIteration3Presets (seed 7) it pins v3.
  const std::uint64_t d = run_fingerprint(discovery_market(), 5);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(d, 16749299937159089722ull) << d;
#else
  EXPECT_NE(d, run_fingerprint(SimParams{}, 5));
#endif
}

// --- OF13 v4 (book-first): discovery_market_v4() and its opt-in fields ------

TEST(PriceDiscoveryV4, TheV4FieldsAreOffByDefaultAndInertWhereTheyCannotBind) {
  const SimParams d;
  EXPECT_EQ(d.touch_cancel_mult, 1.0);
  EXPECT_EQ(d.cancel_activity, 0.0);
  EXPECT_EQ(d.sweep_follow, 0.0);
  EXPECT_FALSE(d.stale_binomial);
  EXPECT_EQ(d.max_stale_levels, 0u);
  EXPECT_EQ(d.max_step_ticks, 0u);
  // Bounds wider than the simulated book never bind: v3's run is unchanged.
  SimParams wide = discovery_market();
  wide.max_stale_levels = static_cast<std::uint32_t>(kLevels);
  EXPECT_EQ(run_fingerprint(wide, 5, 5'000), run_fingerprint(discovery_market(), 5, 5'000));
  wide.max_step_ticks = 1'000'000;
  EXPECT_EQ(run_fingerprint(wide, 5, 5'000), run_fingerprint(discovery_market(), 5, 5'000));
  // Each mechanism, switched on alone, changes the run.
  for (int k = 0; k < 4; ++k) {
    SimParams on = discovery_market();
    if (k == 0) on.touch_cancel_mult = 2;
    if (k == 1) on.cancel_activity = 1;
    if (k == 2) on.sweep_follow = 1;
    if (k == 3) on.stale_binomial = true;
    EXPECT_NE(run_fingerprint(on, 5, 5'000), run_fingerprint(discovery_market(), 5, 5'000)) << k;
  }
  EXPECT_THROW(EsSimulator(SimParams{.touch_cancel_mult = 0}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.touch_cancel_mult = std::nan("")}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.cancel_activity = INFINITY}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.sweep_follow = 1.5}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.sweep_follow = -0.1}), std::invalid_argument);
  EXPECT_NO_THROW(EsSimulator(SimParams{.touch_cancel_mult = 0.5, .cancel_activity = -1, .sweep_follow = 0.5}));
}

TEST(PriceDiscoveryV4, GoldenRunOfDiscoveryV4) {
  // Recorded with the OF13 code (MSVC x64): every v4 number depends on it.
  const std::uint64_t d = run_fingerprint(discovery_market_v4(), 5);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(d, 8194920656928043571ull) << d;  // after the OF13 review (exact fills follow; was 8940020322986933354)
#else
  EXPECT_NE(d, run_fingerprint(discovery_market(), 5));
#endif
  const SimParams m = market_preset("discovery_v4"), v4 = discovery_market_v4();
  EXPECT_EQ(m.sweep_follow, v4.sweep_follow);
  EXPECT_EQ(m.max_step_ticks, v4.max_step_ticks);
  EXPECT_EQ(m.maker_rate, v4.maker_rate);
  EXPECT_EQ(m.activity_tau, discovery_market().activity_tau);  // v3's activity process
}

// Mean L1 and L2..L5 (lots per level and side) over `steps` steps.
static std::pair<double, double> mean_depth(const SimParams& p, std::uint64_t seed, int steps, double start = 0) {
  EsSimulator sim(p, seed, start);
  double l1 = 0, deep = 0;
  for (int s = 0; s < steps; ++s) {
    sim.step();
    const Quote q = sim.quote();
    l1 += 0.5 * (q.bid_size[0] + q.ask_size[0]);
    for (std::size_t i = 1; i < kDepth; ++i) deep += 0.5 * (q.bid_size[i] + q.ask_size[i]) / (kDepth - 1);
  }
  return {l1 / steps, deep / steps};
}

TEST(PriceDiscoveryV4, TheTouchIsThinnerThanTheFlatBookBehindIt) {
  // OF12: real ES L1 ~17 lots rising to ~35 at L5; v3 L1 ~140 falling with distance.
  const auto [l1, deep] = mean_depth(discovery_market_v4(), 3, 20'000);
  EXPECT_GT(l1, 10);
  EXPECT_LT(l1, 28);
  EXPECT_GT(deep / l1, 1.2) << l1 << " " << deep;
  SimParams one = discovery_market_v4();  // touch_cancel_mult 1: the touch as deep as the rest
  one.touch_cancel_mult = 1;
  const auto [l1b, deepb] = mean_depth(one, 3, 20'000);
  EXPECT_GT(l1b, 1.05 * l1) << l1b << " " << l1;
  // The starting book has the same shape (stationary_depth(0) / touch_cancel_mult).
  const Quote q0 = EsSimulator(discovery_market_v4(), 3).quote();
  EXPECT_LT(q0.bid_size[0], q0.bid_size[1]);
  EXPECT_LT(q0.ask_size[0], q0.ask_size[1]);
}

TEST(PriceDiscoveryV4, DepthNoLongerScalesWithActivity) {
  // Activity without noise: the open (intraday boost ~2.5x) against midday.
  auto quiet = [](double cancel_activity) {
    SimParams p = discovery_market_v4();
    p.activity_vol = 0;
    p.activity_news_excite = 0;
    p.cancel_activity = cancel_activity;
    return p;
  };
  for (const double ca : {0.0, 1.0}) {
    const SimParams p = quiet(ca);
    const double open = mean_depth(p, 4, 3'000, 0).second, midday = mean_depth(p, 4, 3'000, 12'000).second;
    if (ca == 0) EXPECT_GT(open / midday, 1.5) << open << " " << midday;  // cancel_rate fixed: depth ~ activity
    else EXPECT_NEAR(open / midday, 1.0, 0.15) << open << " " << midday;
  }
  SimParams busy = quiet(1);  // the per-lot hazard: however busy, cancels never exceed the level
  busy.cancel_rate = 1e6;
  EsSimulator sim(busy, 4);
  for (int s = 0; s < 2'000; ++s) {
    sim.step();
    ASSERT_GT(sim.quote().ask, sim.quote().bid);
  }
}

TEST(PriceDiscoveryV4, ASweptLevelMovesValueAndPriceInTheSameStep) {
  // sweep_follow 1: when takers empty one side's best, fair value moves to the
  // mid the book has once the other side follows: the step's mid move has the
  // sweep's sign, and it is not undone by value lagging behind.
  EsSimulator sim(discovery_market_v4(), 6);
  int moves = 0, agree = 0;
  for (int s = 0; s < 20'000; ++s) {
    const double shift0 = sim.sweep_shift(), mid0 = sim.mid();
    sim.step();
    const double d = sim.sweep_shift() - shift0, dm = sim.mid() - mid0;
    if (d == 0) continue;
    ++moves;
    agree += (d > 0) == (dm > 0) && dm != 0;
  }
  EXPECT_GT(moves, 1'000);
  EXPECT_GT(agree, 0.8 * moves) << agree << " of " << moves;  // measured 88%
  SimParams off = discovery_market();  // v3: sweep_follow off, no shift ever
  EsSimulator v3(off, 6);
  for (int s = 0; s < 5'000; ++s) v3.step();
  EXPECT_EQ(v3.sweep_shift(), 0.0);
}

// v4 without takers, diffusion or icebergs, with news jumps: the book moves by depletion only.
static SimParams v4_jumps_only(double jump_sd) {
  SimParams p = discovery_market_v4();
  p.noise_rate = p.informed_meta_rate = p.uninformed_meta_rate = p.momentum_rate = 0;
  p.value_vol = 0;
  p.iceberg_rate = 0;
  p.jump_rate = 0.3;
  p.jump_sd = jump_sd;
  return p;
}

TEST(PriceDiscoveryV4, BinomialDepletionClearsTheStaleQueueWhereThePoissonDrawLeftResidue) {
  // h = 1 (reprice_cancel huge), every stale level in reach: binomial depletion
  // takes every lot, so after each step value is inside the quote; the legacy
  // Poisson(h * lots) draw leaves ~sqrt(lots) lots about half the time.
  auto run = [](bool binomial) {
    SimParams p = v4_jumps_only(4);
    p.reprice_cancel = 1e9;
    p.max_stale_levels = 0;
    p.max_step_ticks = 0;
    p.stale_binomial = binomial;
    EsSimulator sim(p, 8);
    int outside = 0;
    for (int s = 0; s < 20'000; ++s) {
      sim.step();
      const Quote q = sim.quote();
      outside += sim.fair_value() > q.ask + 1e-9 || sim.fair_value() < q.bid - 1e-9;
    }
    return outside;
  };
  EXPECT_EQ(run(true), 0);
  EXPECT_GT(run(false), 50);
}

TEST(PriceDiscoveryV4, NewsMovesTheMidAtMostMaxStaleLevelsPerStep) {
  auto max_step = [](std::uint32_t max_stale_levels) {
    SimParams p = v4_jumps_only(20);
    p.max_stale_levels = max_stale_levels;
    EsSimulator sim(p, 9);
    double mx = 0, mid = sim.mid();
    for (int s = 0; s < 20'000; ++s) {
      sim.step();
      mx = std::max(mx, std::abs(sim.mid() - mid));
      mid = sim.mid();
    }
    return mx;
  };
  EXPECT_LE(max_step(3), 3 + 3 + 0.5);  // + max_step_ticks for cancel-emptied levels
  EXPECT_GT(max_step(0), 8);  // unbounded: a 20-tick jump clears up to kLevels levels in one step
}

TEST(PriceDiscoveryV4, NoTakerWalksTheBookFurtherThanMaxStepTicksButOwnOrdersDo) {
  // Huge market orders on the thin v4 book (OF12's pathology: a news jump
  // emptied the ask and takers printed on 146 ticks in one step on v3).
  auto run = [](std::uint32_t max_step_ticks, double* max_mid, std::int32_t* max_print) {
    SimParams p = discovery_market_v4();
    p.taker_size_mean = 300;
    p.taker_size_logsd = 0.3;
    p.max_order = 1000;
    p.max_step_ticks = max_step_ticks;
    EsSimulator sim(p, 10);
    Quote q = sim.quote();
    double mid = sim.mid();
    *max_mid = 0;
    *max_print = 0;
    for (int s = 0; s < 5'000; ++s) {
      sim.step();
      for (const auto& t : sim.trades())
        *max_print = std::max(*max_print, t.aggressor == Side::Buy ? t.price - q.ask : q.bid - t.price);
      *max_mid = std::max(*max_mid, std::abs(sim.mid() - mid));
      q = sim.quote();
      mid = sim.mid();
    }
    return sim.step_refills();
  };
  double mm = 0;
  std::int32_t mp = 0;
  run(3, &mm, &mp);
  EXPECT_LE(mp, 3);  // takers print <= max_step_ticks - 1 through the step's starting best; snipers at the stale best <= max_step_ticks
  EXPECT_LE(mm, 3 + 3 + 0.5) << mm;
  double mm0 = 0;
  std::int32_t mp0 = 0;
  run(0, &mm0, &mp0);
  EXPECT_GT(mp0, 3);  // off: the same orders walk the book
  // Between steps the strategy's own market orders are never cut short.
  EsSimulator sim(discovery_market_v4(), 11);
  for (int s = 0; s < 100; ++s) sim.step();
  const std::int32_t ask0 = sim.quote().ask;
  const Fill f = sim.market_order(Side::Buy, 200);
  EXPECT_EQ(f.filled, 200u);
  EXPECT_GT(sim.quote().ask - ask0, 3);
}

TEST(PriceDiscoveryV4, AnExactlyFilledBestLevelMovesValueLikeATradeThrough) {
  // One-lot takers only ever empty a level exactly (the order ends as the
  // level does): sweep_follow must still move value a full tick toward the
  // book's next price, as when an order trades through (OF13 review).
  SimParams p = discovery_market_v4();
  p.taker_size_mean = 1;
  p.taker_size_logsd = 0;
  EsSimulator sim(p, 13);
  int moves = 0, full = 0, agree = 0;
  for (int s = 0; s < 20'000; ++s) {
    const double shift0 = sim.sweep_shift(), mid0 = sim.mid();
    sim.step();
    const double d = sim.sweep_shift() - shift0, dm = sim.mid() - mid0;
    if (d == 0) continue;
    ++moves;
    full += std::abs(d) >= 0.5;
    agree += (d > 0) == (dm > 0) && dm != 0;
  }
  EXPECT_GT(moves, 100);  // measured 211: one-lot takers rarely empty a ~20-lot touch
  EXPECT_GT(full, 0.5 * moves) << full << " of " << moves;
  EXPECT_GT(agree, 0.8 * moves) << agree << " of " << moves;
}

TEST(PriceDiscoveryV4, OwnMarketOrdersNeverMoveFairValue) {
  // No takers, no diffusion, no news, no icebergs: only the strategy trades.
  SimParams p = discovery_market_v4();
  p.noise_rate = 0;
  p.value_vol = 0;
  p.jump_rate = 0;
  p.iceberg_rate = 0;
  EsSimulator sim(p, 14);
  for (int s = 0; s < 50; ++s) sim.step();
  const double v0 = sim.fair_value();
  for (int k = 0; k < 200; ++k) {
    const Quote q = sim.quote();
    const Side side = k % 2 ? Side::Sell : Side::Buy;
    sim.market_order(side, (side == Side::Buy ? q.ask_size[0] : q.bid_size[0]) + 1);  // clears the touch
    sim.step();
  }
  EXPECT_EQ(sim.sweep_shift(), 0.0);
  EXPECT_EQ(sim.fair_value(), v0);
}

// FNV-1a over a v4 run with scripted own orders (market orders that sweep a
// thin touch, joins and cancels): pins the own-order paths (sweep_follow
// ignores them, join_ahead_mean, no step bound for own orders).
static std::uint64_t own_fingerprint(const SimParams& p, std::uint64_t seed, int steps) {
  EsSimulator sim(p, seed);
  std::uint64_t h = 14695981039346656037ull;
  auto mix = [&h](std::uint64_t v) {
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xffu;
      h *= 1099511628211ull;
    }
  };
  std::uint64_t id = 0;
  for (int s = 0; s < steps; ++s) {
    const Quote q = sim.quote();
    if (s % 37 == 0) {
      const Side side = (s / 37) % 2 ? Side::Sell : Side::Buy;
      const Fill f = sim.market_order(side, 1 + static_cast<std::uint32_t>(s % 3));
      mix(f.filled);
      mix(static_cast<std::uint64_t>(std::llround(f.avg_price * 4)));
    }
    if (s % 53 == 0) {
      if (id) mix(sim.cancel(id));
      id = sim.submit_limit((s / 53) % 2 ? Side::Sell : Side::Buy, (s / 53) % 2 ? q.ask : q.bid, 1);
    }
    sim.step();
    for (const auto& t : sim.trades()) {
      mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(t.price)));
      mix(t.size);
      mix(t.own ? 1u : 0u);
    }
    mix(sim.own_fills().size());
    const Quote q1 = sim.quote();
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(q1.bid)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(q1.ask)));
    mix(q1.bid_size[0]);
    mix(q1.ask_size[0]);
    mix(static_cast<std::uint64_t>(std::llround(sim.fair_value() * 1e6)));
  }
  return h;
}

TEST(PriceDiscoveryV4, GoldenRunOfDiscoveryV4WithOwnOrders) {
  // Recorded with the OF13 code after review (MSVC x64).
  const std::uint64_t d = own_fingerprint(discovery_market_v4(), 15, 20'000);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(d, 209874786915369222ull) << d;
#else
  EXPECT_EQ(d, own_fingerprint(discovery_market_v4(), 15, 20'000));
#endif
}

TEST(PriceDiscoveryV4, TickScalpEnvRefusesSweepFollowOutsideDiagnostics) {
  TickParams p;
  p.sim = discovery_market_v4();
  EXPECT_THROW(TickScalpEnv{p}, std::invalid_argument);
  p.allow_sweep_follow = true;
  EXPECT_NO_THROW(TickScalpEnv{p});
  EXPECT_THROW(EsSimulator(SimParams{.max_step_ticks = 3'000'000'000u}), std::invalid_argument);
}

// --- stale-side depletion (v2: SimParams::reprice_cancel, value_impact) -----

TEST(PriceDiscovery, TheV2FieldsAreInertUnlessSwitchedOn) {
  // reprice_cancel needs reprice_band > 0; impact needs value_impact > 0.
  SimParams p;
  p.reprice_cancel = 5;
  p.reprice_take = 0.3;
  p.impact_forecast_tau = 3;
  p.impact_forecast_weight = 0.5;
  EXPECT_EQ(run_fingerprint(p, 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  SimParams v1 = discovery_market_v1();
  v1.reprice_take = 0.3;
  v1.impact_forecast_tau = 3;
  EXPECT_EQ(run_fingerprint(v1, 4, 5'000), run_fingerprint(discovery_market_v1(), 4, 5'000));
  SimParams sep;  // reprice_take alone (no reprice_band / reprice_cancel) is inert with separate streams too
  sep.separate_streams = true;
  SimParams sep2 = sep;
  sep2.reprice_take = 0.5;
  EXPECT_EQ(run_fingerprint(sep, 3, 5'000), run_fingerprint(sep2, 3, 5'000));
  {  // with separate streams, depletion (and its own random stream) leaves the value path alone
    SimParams on = sep;  // (activity_excite off: it feeds mid moves back into the value volatility)
    on.activity_excite = 0;
    on.reprice_band = 0.5;
    on.reprice_cancel = 15;
    on.reprice_take = 0.1;
    SimParams off = on;
    off.reprice_cancel = 0;
    EsSimulator a(on, 3), b(off, 3);
    for (int s = 0; s < 5'000; ++s) {
      a.step();
      b.step();
      ASSERT_EQ(a.fair_value(), b.fair_value()) << "step " << s;
    }
    EXPECT_GT(a.stale_cancelled_lots(), 0u);  // depletion did run
    EXPECT_EQ(b.stale_cancelled_lots(), 0u);
  }
  SimParams imp;
  imp.value_impact = 0.004;
  EXPECT_NE(run_fingerprint(imp, 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  EXPECT_THROW(EsSimulator(SimParams{.reprice_cancel = -1}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.reprice_take = 1.5}), std::invalid_argument);
  EXPECT_THROW(EsSimulator(SimParams{.value_impact = -1}), std::invalid_argument);
}

// Takers, diffusion and icebergs off, news jumps: the book moves only by depletion.
static SimParams jumps_only(double jump_rate) {
  SimParams p = discovery_market();
  p.noise_rate = p.informed_meta_rate = p.uninformed_meta_rate = p.momentum_rate = 0;
  p.value_vol = 0;
  p.iceberg_rate = 0;
  p.jump_rate = jump_rate;
  p.jump_sd = 4;
  return p;
}

TEST(PriceDiscovery, AfterNewsTheStaleSideDrainsNobodyJoinsItAndTheBookCatchesUp) {
  const SimParams p = jumps_only(0.3);
  EsSimulator sim(p, 7);
  int jumps = 0, slow = 0;
  double v = sim.fair_value();
  int since = -1;  // steps since the last jump, while the book hasn't caught up
  Quote before = sim.quote();
  for (int s = 0; s < 40'000; ++s) {
    sim.step();
    const Quote q = sim.quote();
    const double g = sim.fair_value() - sim.mid();
    // Nobody joins a stale price: while value has passed the best and the
    // price didn't move, the stale best only shrinks.
    if (q.bid == before.bid && q.ask == before.ask && sim.fair_value() == v) {
      if (sim.fair_value() - q.ask > p.reprice_band - 0.5) ASSERT_LE(q.ask_size[0], before.ask_size[0]) << "step " << s;
      if (q.bid - sim.fair_value() > p.reprice_band - 0.5) ASSERT_LE(q.bid_size[0], before.bid_size[0]) << "step " << s;
    }
    if (sim.fair_value() != v) {
      v = sim.fair_value();
      since = 0;
      ++jumps;
    } else if (since >= 0) {
      ++since;
    }
    if (since >= 0 && std::abs(g) <= p.reprice_band) {
      EXPECT_EQ(sim.spread(), 1) << "step " << s;  // caught up, with a 1-tick spread
      if (since > 20) ++slow;
      since = -1;
    }
    before = q;
  }
  EXPECT_GT(jumps, 300);
  EXPECT_LE(slow, jumps / 50) << slow;  // catch-up beyond 2 s is rare
  EXPECT_GT(sim.races(), 300u);
  EXPECT_GT(sim.stale_cancelled_lots(), 0u);
  EXPECT_GT(sim.taker_volume(EsSimulator::kSniper), 0u);
}

TEST(PriceDiscovery, AStaleOwnAskAtTheBackIsFilledOnlyAfterTheLotsAheadOfIt) {
  SimParams p = jumps_only(0.2);
  p.jump_sd = 1.5;  // small news: the stale queue drains over a few steps
  EsSimulator sim(p, 9);
  int cases = 0, fills = 0, queued = 0, early = 0;
  for (int s = 0; s < 200'000 && cases < 15; ++s) {
    if (!sim.resting().empty()) {
      sim.step();
      continue;
    }
    const std::int32_t ask = sim.ask();
    const std::uint64_t id = sim.submit_limit(Side::Sell, ask, 1);
    ASSERT_NE(id, 0u);
    sim.step();
    const bool stale = sim.fair_value() - ask > p.reprice_band - 0.5;
    for (const auto& f : sim.own_fills())
      if (stale && f.id == id) ++early;  // FIFO: the first depletion step can't reach the back of the queue
    if (!stale || sim.ask() != ask || sim.resting().empty()) {  // not stale: next try
      sim.cancel(id);
      continue;
    }
    // Stale: value passed our ask. We joined at the back (behind the level and
    // the faster participants), so the first depletion step didn't reach us.
    ++cases;
    ASSERT_EQ(sim.resting().front().id, id);
    std::uint32_t ahead = sim.resting().front().ahead;
    if (ahead > 0) ++queued;  // lots still ahead of us after the first depletion step
    bool filled = false;
    for (int k = 0; k < 100 && !filled; ++k) {
      sim.step();
      for (const auto& f : sim.own_fills())
        if (f.id == id) {
          EXPECT_EQ(f.price, ask);
          filled = true;
        }
      if (filled) break;
      ASSERT_FALSE(sim.resting().empty()) << "order vanished without a fill";
      ASSERT_LE(sim.resting().front().ahead, ahead);  // the queue ahead only shrinks
      ahead = sim.resting().front().ahead;
      ASSERT_EQ(sim.ask(), ask);                                      // the price can't tick past our unfilled order
      if (!(sim.fair_value() - ask > p.reprice_band - 0.5)) break;  // news back down: no longer stale
    }
    if (filled) ++fills;
    else sim.cancel(id);
  }
  EXPECT_GE(cases, 10);
  EXPECT_GE(fills, cases / 2);
  EXPECT_GE(queued, cases / 2);
  EXPECT_EQ(early, 0);
}

TEST(PriceDiscovery, V2BookInvariantsHoldUnderFrequentNews) {
  SimParams p = discovery_market();
  p.jump_rate = 1.0;
  EsSimulator sim(p, 11);
  rag::agent::Rng rng(3);
  for (int s = 0; s < 20'000; ++s) {
    if (s % 7 == 0) {
      const Side side = rng.uniform() < 0.5 ? Side::Buy : Side::Sell;
      const auto back = static_cast<std::int32_t>(rng.below(3));
      sim.submit_limit(side, side == Side::Buy ? sim.bid() - back : sim.ask() + back, 1 + rng.below(3));
    }
    if (s % 13 == 0 && !sim.resting().empty()) sim.cancel(sim.resting().front().id);
    if (s % 17 == 0) sim.market_order(rng.uniform() < 0.5 ? Side::Buy : Side::Sell, 1);
    sim.step();
    ASSERT_GE(sim.spread(), 1);
    const Quote q = sim.quote();
    for (const auto& r : sim.resting()) {
      const std::int32_t d = r.side == Side::Buy ? sim.bid() - r.price : r.price - sim.ask();
      ASSERT_GE(d, 0) << "step " << s;
      if (d < static_cast<std::int32_t>(kDepth)) {
        const auto i = static_cast<std::size_t>(d);
        ASSERT_LE(r.ahead + r.qty, r.side == Side::Buy ? q.bid_size[i] : q.ask_size[i]) << "step " << s;
      }
    }
  }
  EXPECT_GT(sim.reprices(), 500u);
}

// Event time: corr(L1 imbalance, the next mid change), over `seconds` of trading.
static double imbalance_predicts_next_change(const SimParams& p, std::uint64_t seed, double seconds) {
  EsSimulator sim(p, seed, 3600);
  const auto n = static_cast<std::size_t>(std::lround(seconds / p.dt));
  std::vector<double> x, m;
  for (std::size_t s = 0; s < n; ++s) {
    sim.step();
    const Quote q = sim.quote();
    const double b = q.bid_size[0], a = q.ask_size[0];
    x.push_back(b + a > 0 ? (b - a) / (b + a) : 0);
    m.push_back(sim.mid());
  }
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, k = 0;
  std::size_t c = 1;
  for (std::size_t i = 0; i < n; ++i) {
    if (c <= i) c = i + 1;
    while (c < n && m[c] == m[i]) ++c;
    if (c >= n) break;
    const double y = m[c] - m[i];
    sx += x[i];
    sy += y;
    sxx += x[i] * x[i];
    syy += y * y;
    sxy += x[i] * y;
    k += 1;
  }
  return (sxy - sx * sy / k) / std::sqrt((sxx - sx * sx / k) * (syy - sy * sy / k));
}

TEST(PriceDiscovery, V2ImbalancePredictsTheNextTickAndTheGapNotTheNext30Seconds) {
  // 2 h, seed 1. Measured (MSVC): v2 corr(g, r 30 s) 0.127 (the 1-tick grid floor ~0.10),
  // event-time corr(L1 imbalance, next mid change) 0.268 (v1: 0.057); v3 (no informed
  // makers) 0.104 and 0.229.
  for (const SimParams& d : {discovery_market_v2(), discovery_market()}) {
    const double gap = gap_predicts_30s(d, 1, 7200), imb = imbalance_predicts_next_change(d, 1, 7200);
    EXPECT_LT(gap, 0.2) << gap;
    EXPECT_GT(imb, 0.15) << imb;
  }
  const double imb_v1 = imbalance_predicts_next_change(discovery_market_v1(), 1, 7200);
  EXPECT_LT(imb_v1, 0.1) << imb_v1;  // v1: the rigid book carried no signal
}

TEST(EsSimulatorActivity, GoldenRunsOfTheIteration3Presets) {
  // Recorded with the iteration-3 code (OF10 v3), MSVC x64: every v3 and
  // default_warm number depends on them.
  const std::uint64_t w = run_fingerprint(default_warm(), 6), d = run_fingerprint(discovery_market(), 7);
#if defined(_MSC_VER) && !defined(__clang__)
  EXPECT_EQ(w, 1844421873489088197ull) << w;
  EXPECT_EQ(d, 5921645205393397081ull) << d;
#else
  EXPECT_NE(w, d);
#endif
  EXPECT_EQ(market_preset("discovery").maker_info, 0);
  EXPECT_EQ(market_preset("discovery").activity_model, 1);
}

// --- activity model 1 and the warm start (OF10 iteration 3) ------------------

TEST(EsSimulatorActivity, NewFieldsAreInertWithTheLegacyModelAndTheWarmStartNeedsModel1) {
  SimParams p;
  p.activity_news_excite = 0.7;  // model 0 ignores it
  EXPECT_EQ(run_fingerprint(p, 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  SimParams w = p;
  w.activity_warm_start = true;
  EXPECT_THROW(EsSimulator{w}, std::invalid_argument);  // legacy has no interior stationary law
  SimParams bad = default_warm();
  bad.activity_model = 2;
  EXPECT_THROW(EsSimulator{bad}, std::invalid_argument);
  EXPECT_NE(run_fingerprint(default_warm(), 1, 5'000), run_fingerprint(SimParams{}, 1, 5'000));
  EXPECT_EQ(market_preset("default_warm").activity_model, 1);
}

TEST(EsSimulatorActivity, TheWarmStartDrawsTheStationaryLaw) {
  // Activity in model 1 doesn't depend on the book, so a market without takers
  // and makers (cheap steps) has the same activity law as default_warm().
  SimParams p = default_warm();
  p.noise_rate = p.informed_meta_rate = p.uninformed_meta_rate = p.momentum_rate = 0;
  p.maker_rate = 0;
  p.iceberg_rate = 0;
  const ActivityLaw law = activity_law(p);
  const int n = 5'000;  // sd of x within 3%: ~3 SE (measured at 20,000: ratio 0.995-0.997, |mean - m0| <= 1.3 SE)
  std::vector<double> x0, x1;
  double ratio_num = 0, ratio_den = 0;
  std::vector<double> ratio;
  for (int i = 0; i < n; ++i) {
    EsSimulator sim(p, 5'000'000ull + static_cast<std::uint64_t>(i), 3600);
    x0.push_back(sim.log_activity());
    for (int s = 0; s < 3'000; ++s) sim.step();
    x1.push_back(sim.log_activity());
    ratio.push_back(sim.activity() / sim.intraday_factor());
    ratio_num += ratio.back();
    ratio_den += 1;
  }
  auto moments = [](const std::vector<double>& v) {
    double m = 0, s = 0;
    for (double x : v) m += x / static_cast<double>(v.size());
    for (double x : v) s += (x - m) * (x - m);
    return std::pair<double, double>{m, std::sqrt(s / static_cast<double>(v.size() - 1))};
  };
  for (const auto* v : {&x0, &x1}) {
    const auto [m, sd] = moments(*v);
    const double se = sd / std::sqrt(static_cast<double>(n));
    EXPECT_LE(std::abs(m - law.m0), 2 * se) << "mean " << m << " m0 " << law.m0 << " se " << se;
    EXPECT_NEAR(sd / law.sd, 1.0, 0.03) << "sd " << sd << " analytic " << law.sd;
  }
  const auto [rm, rsd] = moments(ratio);  // E[activity / intraday] = 1 (m0 centring)
  EXPECT_LE(std::abs(rm - 1), 2 * rsd / std::sqrt(static_cast<double>(n))) << rm;
  EXPECT_NEAR(law.sd, 0.531, 0.01);  // tau / 2 * (vol^2 + k^2 jump_rate), the design's value
}

// Share of steps at the lower activity clamp in ScalpEnv-built 10-minute episodes after the 150 s warm-up.
static std::pair<double, double> episode_activity(const SimParams& sim, int episodes) {
  ScalpParams sp;
  sp.sim = sim;
  double low = 0, act = 0, steps = 0;
  for (int i = 0; i < episodes; ++i) {
    rag::agent::Rng r(6'000'000ull + static_cast<std::uint64_t>(i));
    EsSimulator m = ScalpEnv::make_market(sp, r);
    for (int s = 0; s < 1'500 + 6'000; ++s) {
      m.step();
      if (s < 1'500) continue;
      low += m.log_activity() <= sim.activity_min + 1e-9 ? 1 : 0;
      act += m.activity() / m.intraday_factor();
      steps += 1;
    }
  }
  return {low / steps, act / steps};
}

TEST(EsSimulatorActivity, EpisodesAreNotStuckAtTheLowerClampWithTheWarmStart) {
  // 40 episodes. Measured (MSVC): legacy default 0.4 of the steps at the lower
  // clamp (activity / intraday ~0.3); default_warm and discovery v3 0.000
  // (activity / intraday ~1).
  const auto [low_def, act_def] = episode_activity(SimParams{}, 40);
  const auto [low_warm, act_warm] = episode_activity(default_warm(), 40);
  const auto [low_v3, act_v3] = episode_activity(discovery_market(), 40);
  EXPECT_GT(low_def, 0.2) << low_def << " " << act_def;
  EXPECT_LT(low_warm, 0.001) << low_warm;
  EXPECT_LT(low_v3, 0.001) << low_v3;
  EXPECT_GT(act_warm, 0.6) << act_warm;
  EXPECT_GT(act_v3, 0.6) << act_v3;
}

TEST(EsSimulatorActivity, MakeMarketBuildsTheSameEpisodeAsReset) {
  for (const SimParams& sim : {default_warm(), discovery_market()}) {
    ScalpParams sp;
    sp.sim = sim;
    ScalpEnv env(sp);
    rag::agent::Rng r1(77), r2(77);
    env.reset(r1);
    EsSimulator m = ScalpEnv::make_market(sp, r2);
    for (int s = 0; s < 1'500; ++s) m.step();
    for (int d = 0; d < 10; ++d) {  // 10 flat decisions: 100 steps
      env.step(1, r1);
      for (int s = 0; s < 10; ++s) m.step();
      ASSERT_EQ(env.market().t_ns(), m.t_ns());
      ASSERT_EQ(env.market().fair_value(), m.fair_value());
      ASSERT_EQ(env.market().bid(), m.bid());
      ASSERT_EQ(env.market().quote().bid_size, m.quote().bid_size);
      ASSERT_EQ(env.market().quote().ask_size, m.quote().ask_size);
    }
    TickParams tp;
    tp.sim = sim;
    TickScalpEnv tenv(tp);
    rag::agent::Rng t1(78), t2(78);
    tenv.reset(t1);
    EsSimulator tm = TickScalpEnv::make_market(tp, t2);
    for (int s = 0; s < 1'500; ++s) tm.step();
    ASSERT_EQ(tenv.market().fair_value(), tm.fair_value());
    ASSERT_EQ(tenv.market().quote().ask_size, tm.quote().ask_size);
  }
}

TEST(EsSimulatorActivity, ForkGivesACopyFreshRandomnessAndLeavesTheOriginalAlone) {
  for (bool sep : {false, true}) {
    SimParams p = discovery_market();
    p.separate_streams = sep;
    EsSimulator a(p, 9, 3600), b(p, 9, 3600);
    for (int s = 0; s < 500; ++s) {
      a.step();
      b.step();
    }
    EsSimulator f1 = a, f2 = a, f3 = a;
    f1.fork(1);
    f2.fork(1);
    f3.fork(2);
    bool differs = false;
    for (int s = 0; s < 300; ++s) {
      a.step();
      b.step();
      f1.step();
      f2.step();
      f3.step();
      ASSERT_EQ(a.fair_value(), b.fair_value());  // forking a copy doesn't touch the original
      ASSERT_EQ(f1.fair_value(), f2.fair_value());  // same salt: same future
      differs = differs || f1.fair_value() != f3.fair_value() || f1.fair_value() != a.fair_value();
    }
    EXPECT_TRUE(differs);
  }
}

// Review of iteration 3: the hidden fair value at a fill is diagnostics only.
// It must not ride in OwnFill (copied by agent-facing step_fills()); the
// simulator and TickScalpEnv keep it in parallel accessors named as hidden.
template <class T>
concept CarriesAValue = requires(T f) { f.value; };
static_assert(!CarriesAValue<OwnFill>, "OwnFill is agent-facing: keep the hidden value out of it");

TEST(EsSimulatorActivity, HiddenFillValuesStayOutOfOwnFillAndParallelToIt) {
  TickParams tp;
  tp.sim = discovery_market();
  tp.episode_seconds = 120;
  TickScalpEnv env(tp);
  rag::agent::Rng r(31);
  env.reset(r);
  std::size_t fills = 0;
  for (;;) {
    const auto st = env.step(rag::agent::Action{3}, r);  // quote both sides
    ASSERT_EQ(env.step_fill_hidden_values().size(), env.step_fills().size());
    ASSERT_EQ(env.market().own_fill_hidden_values().size(), env.market().own_fills().size());
    fills += env.step_fills().size();
    if (st.done) break;
  }
  EXPECT_GT(fills, 0u);
}

// The probe's ground-truth path (a longer burn before the same window) uses
// ScalpEnv::draw_market: the same draws, in the same order, as make_market.
TEST(EsSimulatorActivity, DrawMarketMatchesMakeMarket) {
  ScalpParams sp;
  sp.sim = discovery_market();
  for (std::uint64_t seed : {3ull, 4ull, 5ull}) {
    rag::agent::Rng r1(seed), r2(seed);
    const ScalpEnv::MarketDraw d = ScalpEnv::draw_market(sp, r1);
    EsSimulator a(sp.sim, d.seed, d.start), b = ScalpEnv::make_market(sp, r2);
    EXPECT_EQ(r1.next(), r2.next());  // the same number of draws consumed
    EXPECT_GE(d.start, 0.0);
    for (int s = 0; s < 200; ++s) {
      a.step();
      b.step();
    }
    EXPECT_EQ(a.t_ns(), b.t_ns());
    EXPECT_EQ(a.fair_value(), b.fair_value());
    EXPECT_EQ(a.quote().bid_size, b.quote().bid_size);
  }
}

// Review of iteration 3: with value_impact > 0 a block round trip profits
// from its own impact (own market orders fill at pre-impact prices), so
// multi-lot ScalpEnv use is refused on those presets.
TEST(EsSimulatorActivity, MultiLotScalpingIsRefusedWhereOwnImpactPays) {
  ScalpParams sp;
  sp.contracts = 2;
  EXPECT_NO_THROW(ScalpEnv{sp});  // no value_impact: the old markets are unchanged
  sp.sim = discovery_market();
  EXPECT_THROW(ScalpEnv{sp}, std::invalid_argument);
  sp.sim = discovery_market_v2();
  EXPECT_THROW(ScalpEnv{sp}, std::invalid_argument);
  sp.sim = discovery_market_v4();  // OF13: own orders never move value, but multi-lot ones are not validated
  EXPECT_THROW(ScalpEnv{sp}, std::invalid_argument);
  sp.contracts = 1;
  EXPECT_NO_THROW(ScalpEnv{sp});
}

// OF10 iteration 3: the --warmup option of the ScalpEnv programs
// (rag/orderflow/scalp_run.hpp), opt-in with ScalpEnv's 150 s default.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include "rag/orderflow/scalp_run.hpp"

TEST(ScalpEnvRun, TheWarmupOptionReachesScalpParamsAndMovesTheEpisodes) {
  EXPECT_EQ(kDefaultWarmup, ScalpParams{}.warmup_seconds);
  EXPECT_EQ(scalp_params("default").warmup_seconds, ScalpParams{}.warmup_seconds);
  ASSERT_TRUE(parse_warmup("1050").has_value());
  EXPECT_EQ(*parse_warmup("1050"), 1050.0);
  EXPECT_EQ(*parse_warmup("0"), 0.0);
  for (const char* bad : {"", "-1", "10x", "nan", "inf", "1e999"}) EXPECT_FALSE(parse_warmup(bad).has_value()) << bad;
  const ScalpParams p = scalp_params("discovery", *parse_warmup("1050"));
  EXPECT_EQ(p.warmup_seconds, 1050.0);
  EXPECT_EQ(p.sim.jump_rate, discovery_market().jump_rate);
  EXPECT_EQ(p.sim.activity_model, 1);
  // The warm-up start is uniform on [0, session - episode - warm-up]: every
  // episode starts >= 1050 s into the session and ends by the close.
  const double latest = p.sim.session_seconds - p.episode_seconds - p.warmup_seconds;
  double hi = 0;
  for (std::uint64_t s = 1; s <= 200; ++s) {
    rag::agent::Rng r(s);
    const double start = ScalpEnv::draw_market(p, r).start;
    EXPECT_GE(start, 0.0);
    EXPECT_LE(start, latest);
    hi = std::max(hi, start);
  }
  EXPECT_GT(hi, 0.9 * latest);
  // reset() runs the simulator for the whole warm-up before the first decision.
  ScalpEnv env(p);
  rag::agent::Rng r(7);
  (void)env.reset(r);
  EXPECT_EQ(env.market().t_ns(), 1'050'000'000'000LL);
  // At most session - episode (22,800 s): beyond that no episode fits before the close.
  EXPECT_EQ(warmup_error(p), "");
  EXPECT_EQ(warmup_error(scalp_params("default", 22'800)), "");
  EXPECT_NE(warmup_error(scalp_params("default", 22'801)), "");
  EXPECT_NE(warmup_error(scalp_params("discovery", 1e12)), "");
}

TEST(ScalpEnvRun, DiscoveryStressListDiffersAndTheLegacyListIsUnchanged) {
  auto names = [](const std::vector<StressScenario>& v) {
    std::vector<std::string> n;
    for (const auto& s : v) n.push_back(s.name);
    return n;
  };
  const std::vector<std::string> legacy = {"nominal",         "latency 300 ms",  "latency 1 s",     "fees $3/side", "informed flow x0.5",
                                           "uninformed flow x1.5", "volatility x1.5", "quote fading x2", "session open"};
  EXPECT_EQ(names(stress_scenarios(false)), legacy);
  const std::vector<std::string> disc = {"nominal",          "latency 300 ms",        "latency 1 s",      "fees $3/side",
                                         "uninformed flow x1.5", "slower discovery x0.5", "flow impact x1.5", "news x3",
                                         "informed meta-orders x4", "thinner book x0.5", "session open"};
  EXPECT_EQ(names(stress_scenarios(true)), disc);
  // The legacy scenarios change exactly what they did.
  const auto ls = stress_scenarios(false);
  ScalpParams q;
  ls[4].change(q);
  EXPECT_EQ(q.sim.informed_meta_rate, SimParams{}.informed_meta_rate * 0.5);
  q = {};
  ls[7].change(q);
  EXPECT_EQ(q.sim.fade, SimParams{}.fade * 2);
  // On discovery every new scenario changes the market (fade x2 would not: fade = 0).
  const ScalpParams base = scalp_params("discovery", 1050);
  ScalpParams f = base;
  ls[7].change(f);
  EXPECT_EQ(f.sim.fade, base.sim.fade);  // the no-op the discovery list drops
  const auto ds = stress_scenarios(true);
  auto changed = [&](std::size_t i) {
    ScalpParams x = base;
    ds[i].change(x);
    if (i < 10) EXPECT_EQ(x.warmup_seconds, 1050.0);  // the run's warm-up survives every scenario but the cold open
    return x;
  };
  EXPECT_EQ(changed(5).sim.reprice_cancel, base.sim.reprice_cancel * 0.5);
  EXPECT_EQ(changed(5).sim.reprice_take, base.sim.reprice_take * 0.5);
  EXPECT_EQ(changed(6).sim.value_impact, base.sim.value_impact * 1.5);
  EXPECT_EQ(changed(7).sim.jump_rate, base.sim.jump_rate * 3);
  EXPECT_EQ(changed(8).sim.informed_meta_rate, base.sim.informed_meta_rate * 4);
  EXPECT_EQ(changed(9).sim.maker_rate, base.sim.maker_rate * 0.5);
  for (std::size_t i = 5; i <= 9; ++i) EXPECT_NO_THROW(EsSimulator(changed(i).sim, 1)) << ds[i].name;
  // "Session open" is a cold open on both lists, whatever the run's warm-up:
  // the simulator starts at the open with a 150 s burn (the episode at 150 s),
  // which at the default warm-up is exactly the legacy start_second = 0.
  const ScalpParams disc_open = changed(10);
  ScalpParams legacy_open_1050 = base;
  ls[8].change(legacy_open_1050);
  for (const ScalpParams* o : std::initializer_list<const ScalpParams*>{&disc_open, &legacy_open_1050}) {
    EXPECT_EQ(o->start_second, 0.0);
    EXPECT_EQ(o->warmup_seconds, kDefaultWarmup);
    EXPECT_EQ(o->sim.jump_rate, base.sim.jump_rate);
  }
  ScalpParams d;
  ls[8].change(d);
  ScalpParams legacy_open;
  legacy_open.start_second = 0;
  EXPECT_EQ(d.start_second, legacy_open.start_second);
  EXPECT_EQ(d.warmup_seconds, legacy_open.warmup_seconds);
}

TEST(ScalpEnvRun, DaySessionsKeepTheirTimeOfDayAndNeverStartBeforeTheOpen) {
  for (unsigned k = 0; k < 39; ++k) {
    EXPECT_EQ(day_session_start(k), k * 600.0);  // the committed OF8 days
    const double s = day_session_start(k, 1050);
    EXPECT_GE(s, 0.0);
    EXPECT_EQ(s + 1050, std::max(1050.0, k * 600.0 + 150));  // the episode stays at k * 600 + 150 s (from the open: at 1050 s)
  }
}

TEST(ScalpEnvRun, EachNetworkRecordsItsMarketAndWarmupAndAMismatchIsRefused) {
  const auto dir = std::filesystem::temp_directory_path() / "rag_scalp_run_record_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string net = (dir / "seed1.net").string(), other = (dir / "seed2.net").string();
  EXPECT_EQ(record_path_of_net(net), (dir / "seed1.txt").string());
  EXPECT_EQ(record_path_of_net("a.b/net"), "a.b/net.txt");
  // No record: a legacy network (of3_rl, an older of_train), trained at 150 s
  // on an unknown market: fine at 150 s on any market, refused at 1050 s.
  EXPECT_EQ(check_net(net, "default", 150), "");
  EXPECT_EQ(check_net(net, "discovery", 150), "");
  EXPECT_NE(check_net(net, "discovery", 1050), "");
  ASSERT_TRUE(write_net_record(net, "discovery", 1050, "seeds=12\nseed=1\n"));
  EXPECT_EQ(net_record_value(net, "market").value_or(""), "discovery");
  EXPECT_EQ(net_record_value(net, "warmup_seconds").value_or(""), "1050");
  EXPECT_EQ(net_record_value(net, "seeds").value_or(""), "12");
  EXPECT_FALSE(net_record_value(net, "warmup").has_value());  // whole keys only
  EXPECT_EQ(check_net(net, "discovery", 1050), "");
  EXPECT_NE(check_net(net, "discovery", 150), "");   // another warm-up
  EXPECT_NE(check_net(net, "default", 1050), "");    // another market
  EXPECT_NE(check_net(other, "discovery", 1050), "");  // a record vouches for its own network only
  remove_net_record(net);
  EXPECT_FALSE(std::filesystem::exists(record_path_of_net(net)));
  {
    std::ofstream out(record_path_of_net(net), std::ios::binary);  // CRLF (an edited record) reads the same
    out << "market=discovery\r\nwarmup_seconds=150\r\n";
  }
  EXPECT_EQ(check_net(net, "discovery", 150), "");
  EXPECT_NE(check_net(net, "discovery", 1050), "");
  {
    std::ofstream out(record_path_of_net(net), std::ios::binary);  // a record without a readable warm-up is refused
    out << "market=discovery\nwarmup_seconds=abc\n";
  }
  EXPECT_NE(check_net(net, "discovery", 150), "");
  std::filesystem::remove_all(dir);
}
