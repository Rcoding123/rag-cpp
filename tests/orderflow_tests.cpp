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
  EsSimulator sim(SimParams{}, 1);
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
