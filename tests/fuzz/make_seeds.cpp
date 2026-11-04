// make_seeds — writes valid index payloads (file minus trailing checksum)
// for every dense mode, as a starting corpus for fuzz_load; given a second
// directory, also valid policy-table payloads for fuzz_policy_table.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "fuzz_common.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag;
using namespace rag::fuzz;

namespace {
// A saved file minus its trailing checksum: the fuzzers append a fresh one.
std::vector<char> payload_of(const std::filesystem::path& tmp) {
  std::vector<char> b;
  {
    std::ifstream f(tmp, std::ios::binary);
    b.assign(std::istreambuf_iterator<char>(f), {});
  }
  std::filesystem::remove(tmp);
  b.resize(b.size() - 8);
  return b;
}

void write_payload(const std::filesystem::path& p, const std::vector<char>& b) {
  std::ofstream o(p, std::ios::binary);
  o.write(b.data(), static_cast<std::streamsize>(b.size()));
}

// Small policy tables (each a few hundred bytes at most) over varied grids,
// including widths near both ends of the normal range (1e-307 is one byte
// edit from a subnormal width).
int policy_table_seeds(const std::filesystem::path& out) {
  using namespace rag::agent;
  std::filesystem::create_directories(out);
  struct Shape {
    std::uint32_t horizon, lots, bins;
    double range, sd;
  };
  const Shape shapes[] = {{1, 0, 1, 4.0, 0.1}, {2, 3, 4, 4.0, 0.1}, {5, 12, 6, 2.0, 0.1}, {3, 2, 32, 4.0, 1e-300},
                          {2, 1, 16, 1e300, 1e7}, {4, 9, 5, 0.5, 3.0},   {2, 2, 8, 4.0, 1e-307}};
  int idx = 0;
  for (const Shape& s : shapes) {
    ExecutionParams p;
    p.horizon = s.horizon;
    p.lots = s.lots;
    p.signal_sd = s.sd;
    const ExecutionPolicyTable table(
        p, [](const ExecutionObs& o) -> Action { return (o.t * 3 + o.inventory + (o.signal > 0 ? 2u : 0u)) % 5; },
        {.signal_bins = s.bins, .signal_range_sd = s.range});
    const auto tmp = out / "tmp.tbl";
    table.save(tmp);
    write_payload(out / ("seed-" + std::to_string(idx++)), payload_of(tmp));
  }
  return idx;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: make_seeds <out-dir> [<policy-table-out-dir>]\n"); return 2; }
  const std::filesystem::path out = argv[1];
  std::filesystem::create_directories(out);
  const char* docs[] = {"alpha beta gamma delta epsilon zeta eta theta iota kappa",
                        "order book imbalance predicts the next mid price move",
                        "the of and", "t0w1 t0w2 t1w3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14"};
  const DenseMode modes[] = {DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf, DenseMode::IvfInt8};
  int idx = 0;
  for (int m = 0; m < 4; ++m) {
    for (int ndocs : {0, 1, 4}) {
      Engine e(small_config(modes[m]));
      for (int d = 0; d < ndocs; ++d) {
        // Odd documents carry a source and two pages, so seeds cover every record field.
        const std::uint32_t pages[] = {0, static_cast<std::uint32_t>(std::string_view(docs[d]).size() / 2)};
        if (d % 2) e.ingest(docs[d], static_cast<std::uint32_t>(d), IngestOptions{.source = "seed.pdf", .page_starts = pages});
        else e.ingest(docs[d], static_cast<std::uint32_t>(d));
      }
      e.commit();
      if (ndocs == 4) e.ingest("uncommitted tail chunk", 9);
      const auto tmp = out / "tmp.idx";
      e.save(tmp);
      write_payload(out / ("seed-" + std::to_string(idx++)), payload_of(tmp));
    }
  }
  std::printf("wrote %d seeds to %s\n", idx, out.c_str());
  if (argc >= 3) std::printf("wrote %d policy-table seeds to %s\n", policy_table_seeds(argv[2]), argv[2]);
}
