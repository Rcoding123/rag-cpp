// policy_table_tests.cpp — compiling a policy into a lookup table (E22).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"
#include "test_util.hpp"

using namespace rag::agent;

TEST(PolicyTable, ReproducesThePolicyAtBinCentresAndClampsOutsideTheGrid) {
  ExecutionParams p;
  p.lots = 10;
  p.horizon = 4;
  // A policy that depends on every coordinate.
  auto policy = [](const ExecutionObs& o) -> Action {
    return (o.t + o.inventory + (o.signal > 0.05 ? 2u : 0u)) % ExecutionEnv::kNumActions;
  };
  const ExecutionPolicyTable table(p, policy, {.signal_bins = 8, .signal_range_sd = 2.0});
  EXPECT_EQ(table.bytes(), 4u * 11u * 8u);
  for (std::uint32_t t = 0; t < 4; ++t)
    for (std::uint32_t inv = 0; inv <= 10; ++inv)
      for (std::uint32_t b = 0; b < 8; ++b) {
        const ExecutionObs o{t, inv, table.bin_centre(b)};
        EXPECT_EQ(table.act(o), policy(o));
      }
  // Bins are 0.05 wide over [-0.2, 0.2]: 0.07 lies in the bin centred at 0.075.
  EXPECT_EQ(table.act({1, 3, 0.07}), policy({1, 3, 0.075}));
  // Outside the grid: edge bins, last time step, full inventory.
  EXPECT_EQ(table.act({0, 0, 5.0}), table.act({0, 0, table.bin_centre(7)}));
  EXPECT_EQ(table.act({0, 0, -5.0}), table.act({0, 0, table.bin_centre(0)}));
  EXPECT_EQ(table.act({0, 0, std::numeric_limits<double>::quiet_NaN()}), table.act({0, 0, table.bin_centre(0)}));
  EXPECT_EQ(table.act({99, 99, 0.0}), table.act({3, 10, 0.0}));
}

TEST(PolicyTable, RejectsBadParametersAndInvalidActions) {
  const ExecutionParams p;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  EXPECT_THROW(ExecutionPolicyTable(p, twap, {.signal_bins = 0}), std::invalid_argument);
  EXPECT_THROW(ExecutionPolicyTable(p, twap, {.signal_range_sd = 0}), std::invalid_argument);
  EXPECT_THROW(ExecutionPolicyTable(p, twap, {.signal_range_sd = std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
  ExecutionParams zero = p;
  zero.horizon = 0;
  EXPECT_THROW(ExecutionPolicyTable(zero, twap), std::invalid_argument);
  EXPECT_THROW(ExecutionPolicyTable(p, [](const ExecutionObs&) { return Action{5}; }), std::out_of_range);
}

TEST(PolicyTable, RejectsAGridWhoseCellCountOverflows) {
  // horizon x (lots + 1) x bins = 2^64 wraps to 0 in size_t: the table must be
  // rejected before anything is allocated or sampled, not resized to nothing
  // and then written past its end.
  std::size_t calls = 0;
  auto counting = [&](const ExecutionObs&) {
    ++calls;
    return ExecutionEnv::kTwapAction;
  };
  ExecutionParams p;
  p.horizon = 1u << 16;
  p.lots = std::numeric_limits<std::uint32_t>::max();  // 2^32 inventory values
  EXPECT_THROW(ExecutionPolicyTable(p, counting, {.signal_bins = 1u << 16}), std::invalid_argument);
  p.horizon = 1u << 31;
  p.lots = (1u << 31) - 1;
  EXPECT_THROW(ExecutionPolicyTable(p, counting, {.signal_bins = 4}), std::invalid_argument);
  EXPECT_EQ(calls, 0u);
}

TEST(PolicyTable, EveryAcceptedSignalGridReproducesThePolicyAndRoundTrips) {
  // Individually valid, jointly extreme scales: signal_range_sd * signal_sd
  // can underflow to 0, overflow once doubled, or give a subnormal bin width
  // whose reciprocal is infinite. The constructor must reject such a grid or
  // build one that act() bins correctly and that load() reads back.
  struct Case {
    double range, sd;
    bool must_accept;
  };
  const Case cases[] = {{4.0, 0.1, true},       {4.0, 1e-300, true}, {1e300, 1e7, true},
                        {1e-300, 1e-300, false}, {1e308, 1.0, false}, {1e-150, 1e-160, false}};
  auto by_sign = [](const ExecutionObs& o) -> Action { return o.signal < 0 ? 0u : o.signal > 0 ? 4u : 2u; };
  const auto path = rag::testutil::temp_index("grid");
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message() << "range " << c.range << " sd " << c.sd);
    ExecutionParams p;
    p.lots = 2;
    p.horizon = 2;
    p.signal_sd = c.sd;
    std::optional<ExecutionPolicyTable> table;
    try {
      table.emplace(p, by_sign, PolicyTableParams{.signal_bins = 8, .signal_range_sd = c.range});
    } catch (const std::invalid_argument&) {
      EXPECT_FALSE(c.must_accept);
      continue;
    }
    for (std::uint32_t b = 0; b < 8; ++b) {
      const ExecutionObs o{1, 1, table->bin_centre(b)};
      EXPECT_EQ(table->act(o), by_sign(o)) << "bin " << b;
    }
    table->save(path);
    EXPECT_NO_THROW((void)ExecutionPolicyTable::load(path));
  }
}

TEST(PolicyTable, LoadValidatesTheStructureBehindTheChecksum) {
  // Edited files with a recomputed checksum reach the structural checks.
  ExecutionParams p;
  p.lots = 3;
  p.horizon = 2;
  const ExecutionPolicyTable table(p, [](const ExecutionObs& o) { return Action{o.inventory % 5}; }, {.signal_bins = 4});
  const auto path = rag::testutil::temp_index("structure");
  table.save(path);
  const auto good = rag::testutil::read_file(path);
  // Layout: magic[8], horizon@8, lots@12, bins@16, lo@20, width@28, length@36, cells@44, checksum.
  ASSERT_EQ(good.size(), 44u + 2u * 4u * 4u + 8u);
  auto expect_rejected = [&](auto edit, const char* what) {
    auto b = good;
    edit(b);
    rag::testutil::fix_checksum(b);
    rag::testutil::write_file(path, b);
    EXPECT_THROW((void)ExecutionPolicyTable::load(path), rag::io::FormatError) << what;
  };
  using rag::testutil::poke;
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  expect_rejected([&](auto& b) { poke<std::uint32_t>(b, 8, 0); }, "zero horizon");
  expect_rejected([&](auto& b) { poke<std::uint32_t>(b, 16, 0); }, "zero bins");
  expect_rejected([&](auto& b) { poke<std::uint32_t>(b, 12, 4); }, "lots disagree with the cell count");
  expect_rejected([&](auto& b) { poke<std::uint32_t>(b, 8, 1); }, "horizon disagrees with the cell count");
  expect_rejected([&](auto& b) { b[44 + 7] = 5; }, "invalid action");
  for (double lo : {inf, -inf, nan}) expect_rejected([&](auto& b) { poke<double>(b, 20, lo); }, "non-finite lo");
  for (double w : {0.0, -0.1, inf, nan, 1e-310})  // 1e-310 is subnormal: 1 / width is infinite
    expect_rejected([&](auto& b) { poke<double>(b, 28, w); }, "bad width");
  rag::testutil::write_file(path, good);
  EXPECT_NO_THROW((void)ExecutionPolicyTable::load(path));
}

TEST(PolicyTable, ACompiledOptimalPolicyKeepsNearlyAllOfItsGain) {
  // The DP's own policy, compiled at 64 bins, against the DP itself.
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
  const ExecutionPolicyTable table(p, opt);
  const double best = compare(env, opt, twap, 400, 5).mean;
  const double compiled = compare(env, [&](const ExecutionObs& o) { return table.act(o); }, twap, 400, 5).mean;
  ASSERT_GT(best, 0);
  EXPECT_GT(compiled / best, 0.97);
  EXPECT_LE(compiled / best, 1.0 + 1e-9 + 0.02);  // sampling noise only; a table can't beat the optimum by much
}

TEST(PolicyTable, SavesAndLoadsExactlyAndRejectsCorruptFiles) {
  ExecutionParams p;
  p.lots = 12;
  p.horizon = 5;
  auto policy = [](const ExecutionObs& o) -> Action { return (o.t * 7 + o.inventory * 3 + (o.signal > 0 ? 1u : 0u)) % 5; };
  const ExecutionPolicyTable table(p, policy, {.signal_bins = 6});
  const auto path = std::filesystem::path(::testing::TempDir()) / "policy_table_roundtrip.tbl";
  table.save(path);
  const auto back = ExecutionPolicyTable::load(path);
  EXPECT_EQ(back.bytes(), table.bytes());
  EXPECT_EQ(back.signal_bins(), 6u);
  for (std::uint32_t t = 0; t < 5; ++t)
    for (std::uint32_t inv = 0; inv <= 12; ++inv)
      for (double s : {-1.0, -0.13, -0.01, 0.0, 0.02, 0.2, 3.0}) EXPECT_EQ(back.act({t, inv, s}), table.act({t, inv, s}));

  std::vector<char> bytes;
  {
    std::ifstream f(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(f), {});
  }
  auto write = [&](const std::vector<char>& b) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(b.data(), static_cast<std::streamsize>(b.size()));
  };
  auto corrupt = bytes;
  corrupt[50] ^= 1;  // a table byte (they start at 44): the checksum catches it
  write(corrupt);
  EXPECT_THROW((void)ExecutionPolicyTable::load(path), rag::io::FormatError);
  write({bytes.begin(), bytes.end() - 3});  // truncated
  EXPECT_THROW((void)ExecutionPolicyTable::load(path), rag::io::FormatError);
  corrupt = bytes;
  corrupt[0] = 'X';  // wrong magic (the checksum would fail too; either rejects it)
  write(corrupt);
  EXPECT_THROW((void)ExecutionPolicyTable::load(path), rag::io::FormatError);
  std::filesystem::remove(path);
}

TEST(PolicyTable, MedianVotesCellByCell) {
  ExecutionParams p;
  p.lots = 4;
  p.horizon = 3;
  auto constant = [&](Action a) { return ExecutionPolicyTable(p, [a](const ExecutionObs&) { return a; }, {.signal_bins = 4}); };
  const auto ens = ExecutionPolicyTable::median({constant(0), constant(4), constant(2)});
  const auto even = ExecutionPolicyTable::median({constant(3), constant(1)});  // lower median
  // A table that varies by cell, voted with two copies of itself, is unchanged.
  auto varied = [](const ExecutionObs& o) -> Action { return (o.t + o.inventory) % 5; };
  const ExecutionPolicyTable v(p, varied, {.signal_bins = 4});
  const auto self = ExecutionPolicyTable::median({v, constant(0), v});
  for (std::uint32_t t = 0; t < 3; ++t)
    for (std::uint32_t inv = 0; inv <= 4; ++inv) {
      EXPECT_EQ(ens.act({t, inv, 0.0}), 2u);
      EXPECT_EQ(even.act({t, inv, 0.0}), 1u);
      EXPECT_EQ(self.act({t, inv, 0.0}), varied({t, inv, 0.0}));
    }
  EXPECT_THROW((void)ExecutionPolicyTable::median({}), std::invalid_argument);
  const ExecutionPolicyTable other(p, varied, {.signal_bins = 8});
  EXPECT_THROW((void)ExecutionPolicyTable::median({v, other}), std::invalid_argument);
}
