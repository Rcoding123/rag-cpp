// fuzz_policy_table — ExecutionPolicyTable::load on arbitrary files.
//
// As in fuzz_load, the fuzzer's bytes are the file payload and a correct
// checksum is appended, so mutations get past the integrity check and reach
// the structural validation behind it.
//
// Invariants: load() either throws io::FormatError (anything else escapes and
// is reported) or returns a table that matches the payload's layout. Every
// lookup, including times, inventories and signals far outside the grid, NaN
// and infinities, is in bounds (ASan) and a valid action; NaN and -inf use
// bin 0, +inf the last bin, and on a grid whose bins are resolvable in double
// precision each bin centre maps to its own cell, all checked against the
// payload's bytes. median() of the table with itself is the table, and save()
// writes back exactly the bytes that were loaded.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "fuzz_common.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag;
using namespace rag::agent;
using namespace rag::fuzz;

namespace {
template <class T>
T peek(const std::uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto path = scratch_file("policy-table");
  io::Checksum c;
  c.update(data, size);
  const std::uint64_t h = c.value();
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    f.write(reinterpret_cast<const char*>(&h), sizeof h);
  }
  std::optional<ExecutionPolicyTable> loaded;
  try {
    loaded.emplace(ExecutionPolicyTable::load(path));
  } catch (const io::FormatError&) {
    return 0;  // rejected: fine
  }
  const ExecutionPolicyTable& table = *loaded;

  // Layout: magic[8], horizon, lots, bins (u32), lo, width (f64), u64 length, cells.
  constexpr std::size_t kHeader = 44;
  FUZZ_CHECK(size == kHeader + table.bytes());
  const auto horizon = peek<std::uint32_t>(data + 8), lots = peek<std::uint32_t>(data + 12),
             bins = peek<std::uint32_t>(data + 16);
  const double lo = peek<double>(data + 20), width = peek<double>(data + 28);
  FUZZ_CHECK(table.signal_bins() == bins && horizon > 0 && bins > 0);
  const std::uint64_t inventories = std::uint64_t{lots} + 1, cells = table.bytes();
  FUZZ_CHECK(cells % bins == 0 && (cells / bins) % inventories == 0 && cells / bins / inventories == horizon);
  const std::uint8_t* cell = data + kHeader;

  const ExecutionPolicyTable self3 = ExecutionPolicyTable::median({table, table, table});
  const ExecutionPolicyTable self2 = ExecutionPolicyTable::median({table, table});

  constexpr std::uint32_t kMax32 = std::numeric_limits<std::uint32_t>::max();
  constexpr double kInf = std::numeric_limits<double>::infinity(), kNaN = std::numeric_limits<double>::quiet_NaN(),
                   kMaxD = std::numeric_limits<double>::max(), kTiny = std::numeric_limits<double>::denorm_min();
  const std::uint32_t ts[] = {0, 1, horizon - 1, horizon, kMax32 - 1, kMax32};
  const std::uint32_t invs[] = {0, 1, lots, lots + 1 /* wraps at UINT32_MAX: fine */, kMax32};
  const double span = static_cast<double>(bins) * width;
  const double signals[] = {0.0,   -0.0,  1e-3,   -1e-3, 1.0,  -1.0,  1e300,       -1e300,
                            kMaxD, -kMaxD, kInf,  -kInf, kNaN, kTiny, -kTiny,      lo,
                            lo + span, std::nextafter(lo, -kInf), table.bin_centre(0), table.bin_centre(bins / 2),
                            table.bin_centre(bins - 1)};
  // Bin centres map back to their bins when the grid's coordinates are
  // resolvable: |lo| at most the grid's span (true of everything save()
  // writes) and no overflow at the top edge.
  const bool resolvable = std::isfinite(2 * span) && std::abs(lo) <= span;
  for (const std::uint32_t t : ts)
    for (const std::uint32_t inv : invs) {
      const std::uint64_t row = (std::uint64_t{std::min(t, horizon - 1)} * inventories + std::min(inv, lots)) * bins;
      for (const double s : signals) {
        const ExecutionObs o{t, inv, s};
        const Action a = table.act(o);
        FUZZ_CHECK(a < ExecutionEnv::kNumActions);
        FUZZ_CHECK(self3.act(o) == a && self2.act(o) == a);
      }
      const Action first{cell[row]}, last{cell[row + bins - 1]};
      FUZZ_CHECK(table.act({t, inv, kNaN}) == first);
      FUZZ_CHECK(table.act({t, inv, -kInf}) == first);
      FUZZ_CHECK(table.act({t, inv, -kMaxD}) == first);
      FUZZ_CHECK(table.act({t, inv, std::nextafter(lo, -kInf)}) == first);
      FUZZ_CHECK(table.act({t, inv, kInf}) == last);
      if (resolvable)
        for (const std::uint32_t b : {0u, bins / 2, bins - 1})
          FUZZ_CHECK(table.act({t, inv, table.bin_centre(b)}) == Action{cell[row + b]});
    }

  // save() writes back exactly what was loaded, checksum included.
  table.save(path);
  std::ifstream f(path, std::ios::binary);
  const std::vector<char> back{std::istreambuf_iterator<char>(f), {}};
  FUZZ_CHECK(back.size() == size + sizeof h);
  FUZZ_CHECK(std::memcmp(back.data(), data, size) == 0);
  FUZZ_CHECK(std::memcmp(back.data() + size, &h, sizeof h) == 0);
  return 0;
}
