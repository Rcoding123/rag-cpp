// tob_dump FILE OUT T0_NS T1_NS : replay the day, and for every A/C/M/T record with
// T0 <= ts_recv < T1 write our top of book AFTER applying it (to compare with mbp-1).
// Row (packed, 64 bytes): ts_recv u64, sequence u32, action u8, side u8, flags u8, pad u8,
// price i64, size u32, bid_ct u32, bid_px i64, ask_px i64, bid_sz u32, ask_sz u32, ask_ct u32, pad u32
#include <cstdio>
#include <cstdlib>

#include "mbo/book.hpp"

#pragma pack(push, 1)
struct Row {
  uint64_t ts_recv; uint32_t sequence; uint8_t action, side, flags, pad0;
  int64_t price; uint32_t size, bid_ct; int64_t bid_px, ask_px; uint32_t bid_sz, ask_sz, ask_ct, pad1;
};
#pragma pack(pop)
static_assert(sizeof(Row) == 64);

int main(int argc, char** argv) {
  if (argc != 5) { std::fprintf(stderr, "usage: tob_dump FILE OUT T0_NS T1_NS\n"); return 2; }
  uint64_t t0 = std::strtoull(argv[3], nullptr, 10), t1 = std::strtoull(argv[4], nullptr, 10);
  try {
    mbo::DbnReader r(argv[1]);
    mbo::Book b;
    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) return 1;
    uint64_t n = 0;
    while (const mbo::MboMsg* m = r.next_mbo()) {
      b.apply(*m);
      if (m->ts_recv < t0 || m->ts_recv >= t1) continue;
      if (m->action != 'A' && m->action != 'C' && m->action != 'M' && m->action != 'T') continue;
      Row w{};
      w.ts_recv = m->ts_recv; w.sequence = m->sequence; w.action = m->action; w.side = m->side; w.flags = m->flags;
      w.price = m->price; w.size = m->size;
      if (auto bb = b.best_bid()) { w.bid_px = bb->price; w.bid_sz = uint32_t(bb->size); w.bid_ct = bb->orders; }
      if (auto ba = b.best_ask()) { w.ask_px = ba->price; w.ask_sz = uint32_t(ba->size); w.ask_ct = ba->orders; }
      std::fwrite(&w, sizeof w, 1, f); ++n;
    }
    std::fclose(f);
    std::printf("%llu rows\n", (unsigned long long)n);
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
