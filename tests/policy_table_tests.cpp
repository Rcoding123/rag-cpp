// policy_table_tests.cpp — compiling a policy into a lookup table (E22).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"

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
