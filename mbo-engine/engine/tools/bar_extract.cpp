// bar_extract FILE OUT.bars [bucket_ms=100]
// Writes: 16-byte header "MBOBARS1", u32 row_bytes, u32 bucket_ms; then packed BarRow rows.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>

#include "mbo/bars.hpp"

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: bar_extract FILE OUT [bucket_ms]\n"); return 2; }
  uint32_t ms = argc > 3 ? uint32_t(std::strtoul(argv[3], nullptr, 10)) : 100;
  if (ms == 0 || 1000 % ms != 0) { std::fprintf(stderr, "bucket_ms must divide 1000\n"); return 2; }
  try {
    mbo::DbnReader r(argv[1]);
    std::map<uint32_t, std::unique_ptr<mbo::BarBuilder>> bb;
    std::vector<mbo::BarRow> rows;
    rows.reserve(1 << 20);
    while (const mbo::MboMsg* m = r.next_mbo()) {
      auto& b = bb[m->hd.instrument_id];
      if (!b) b = std::make_unique<mbo::BarBuilder>(m->hd.instrument_id, uint64_t(ms) * 1000000ULL);
      b->add(*m, rows);
    }
    uint64_t dropped = 0;
    for (auto& [id, b] : bb) { b->finish(rows); dropped += b->rows_dropped_no_boundary(); }
    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    char hdr[16] = {'M', 'B', 'O', 'B', 'A', 'R', 'S', '1'};
    uint32_t rb = mbo::kBarRowBytes;
    std::memcpy(hdr + 8, &rb, 4); std::memcpy(hdr + 12, &ms, 4);
    bool ok = std::fwrite(hdr, 1, 16, f) == 16 &&
              std::fwrite(rows.data(), sizeof(mbo::BarRow), rows.size(), f) == rows.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::fprintf(stderr, "write failed\n"); return 1; }
    std::printf("{\"rows\":%zu,\"dropped_before_first_boundary\":%llu}\n", rows.size(), (unsigned long long)dropped);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
