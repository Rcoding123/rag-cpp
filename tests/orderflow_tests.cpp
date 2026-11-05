// orderflow_tests.cpp — the ES order-flow simulator: book invariants,
// determinism, market orders, and the FIFO queue mechanics tick-scalping
// strategies depend on.
#include <gtest/gtest.h>

#include <cmath>
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
