// stylized_tests.cpp — the tick CSV reader and the stylized-facts meter used
// to calibrate the ES simulator against recorded ticks.
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>

#include "rag/orderflow/stylized.hpp"
#include "rag/orderflow/tick_csv.hpp"

using namespace rag::orderflow;

namespace {
struct Recorder {
  std::vector<Trade> trades;
  std::vector<Quote> quotes;
  void add_trade(const Trade& t) { trades.push_back(t); }
  void end_bucket(const Quote& q) { quotes.push_back(q); }
};
}  // namespace

TEST(TickCsv, ParsesTradesQuotesAndBucketsThem) {
  std::istringstream in(
      "# header\n"
      "Q,0.00,5000.00,5000.25,10,11,12,13,14,20,21,22,23,24\n"
      "T,0.05,5000.25,3,B\n"
      "T,0.07,5000.00,2,S\n"
      "Q,0.12,5000.25,5000.50\n"
      "bad,line\n"
      "T,0.31,5000.50,1,X\n"  // bad side: skipped
      "T,100.0,5001.00,4,B\n");  // after a gap > 60 s
  Recorder r;
  const auto st = replay_tick_csv(in, r, TickCsvOptions{.time_unit = "s"});
  EXPECT_EQ(st.trades, 3u);
  EXPECT_EQ(st.quotes, 2u);
  EXPECT_EQ(st.skipped, 2u);
  EXPECT_EQ(st.gaps, 1u);
  ASSERT_EQ(r.trades.size(), 3u);
  EXPECT_EQ(r.trades[0].price, 20001);  // 5000.25 / 0.25
  EXPECT_EQ(r.trades[0].aggressor, Side::Buy);
  EXPECT_EQ(r.trades[1].aggressor, Side::Sell);
  ASSERT_FALSE(r.quotes.empty());
  EXPECT_EQ(r.quotes[0].bid, 20000);
  EXPECT_EQ(r.quotes[0].ask, 20001);
  EXPECT_EQ(r.quotes[0].bid_size[0], 10u);
  EXPECT_EQ(r.quotes[0].ask_size[4], 24u);
  EXPECT_EQ(r.quotes.back().bid, 20001);  // the second quote (no depth given: sizes 0)
  EXPECT_THROW(replay_tick_csv(in, r, TickCsvOptions{.tick_size = 0}), std::invalid_argument);
}

TEST(StylizedMeter, CsvRoundTripReproducesTheSimulatorsFacts) {
  // Simulator ticks written in the CSV format and read back must give the
  // same facts as measuring the simulator directly.
  const SimParams p;
  EsSimulator sim(p, 4, 3600);
  StylizedMeter direct(p.dt);
  std::ostringstream csv;
  csv.precision(12);
  for (int s = 0; s < 6000; ++s) {  // 10 minutes
    sim.step();
    const double t = static_cast<double>(sim.t_ns()) * 1e-9;
    for (const auto& tr : sim.trades()) {
      direct.add_trade(tr);
      csv << "T," << t << "," << tr.price * 0.25 << "," << tr.size << "," << (tr.aggressor == Side::Buy ? 'B' : 'S') << "\n";
    }
    const Quote q = sim.quote();
    direct.end_bucket(q);
    csv << "Q," << t + 0.0005 << "," << q.bid * 0.25 << "," << q.ask * 0.25;
    for (auto v : q.bid_size) csv << "," << v;
    for (auto v : q.ask_size) csv << "," << v;
    csv << "\n";
  }
  StylizedMeter replayed(p.dt);
  std::istringstream in(csv.str());
  const auto st = replay_tick_csv(in, replayed, TickCsvOptions{.time_unit = "s"});
  EXPECT_EQ(st.skipped, 0u);
  const auto a = direct.facts(), b = replayed.facts();
  EXPECT_NEAR(a.trades_per_sec, b.trades_per_sec, 0.02 * a.trades_per_sec);
  EXPECT_NEAR(a.mean_trade_size, b.mean_trade_size, 1e-9);
  EXPECT_NEAR(a.spread1_share, b.spread1_share, 0.01);
  EXPECT_NEAR(a.ret1m_sd, b.ret1m_sd, 0.1 * a.ret1m_sd + 0.05);
  EXPECT_NEAR(a.sign_ac1, b.sign_ac1, 1e-9);  // same prints in the same order
  EXPECT_GT(a.trades_per_sec, 1.0);
}

TEST(StylizedMeter, SimulatorFactsAreInARealisticRange) {
  const auto f = simulate_facts(SimParams{}, 1800, 3);
  EXPECT_GT(f.trades_per_sec, 3);
  EXPECT_LT(f.trades_per_sec, 80);
  EXPECT_GT(f.spread1_share, 0.9);
  EXPECT_GT(f.sign_ac1, 0.05);  // order splitting: long memory in trade signs
  EXPECT_GT(f.impact_r2, 0.1);  // order flow moves prices
}

// --- replay_env.hpp ------------------------------------------------------------
#include "rag/orderflow/replay_env.hpp"

TEST(ReplayScalpEnv, FillsAtTheRecordedTouchWithFeesAndWindowsTheTape) {
  // 20 minutes of a constant market: bid 5000.00 x 50, ask 5000.25 x 50, one print per second.
  std::ostringstream csv;
  for (int i = 0; i < 12000; ++i) {
    const double t = i * 0.1;
    if (i % 10 == 0) csv << "T," << t << ",5000.25,1,B\n";
    csv << "Q," << t + 0.0005 << ",5000.00,5000.25,50,50,50,50,50,50,50,50,50,50\n";
  }
  std::istringstream in(csv.str());
  auto tape = std::make_shared<const TickTape>(TickTape::from_stream(in, TickCsvOptions{.time_unit = "s"}));
  ReplayParams p;
  p.window_seconds = 300;
  ReplayScalpEnv env(tape, p);
  EXPECT_EQ(env.windows(), 3u);  // (1200 s - 150 s warm-up) / 300 s
  rag::agent::Rng r(1);
  for (rag::agent::Action a : {rag::agent::Action{1}, rag::agent::Action{2}}) {
    auto o = env.seek(1);
    double total = 0;
    for (;;) {
      const auto s = env.step(a, r);
      total += s.reward;
      o = s.obs;
      if (s.done) break;
    }
    EXPECT_NEAR(total, env.net_pnl_ticks(), 1e-9);
    if (a == 1) {
      EXPECT_EQ(env.net_pnl_ticks(), 0.0);
    } else {
      // Long once at the ask, out at the bid in an unchanged market: one tick of spread plus two fees.
      EXPECT_EQ(env.orders(), 2u);
      EXPECT_NEAR(env.net_pnl_ticks(), -1.0 - 2 * p.fee_dollars / kTickValue, 1e-9);
    }
  }
  EXPECT_THROW(env.seek(3), std::out_of_range);
}
