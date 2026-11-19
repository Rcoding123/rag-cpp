// trades_extract FILE OUT.trades : every T record (trade) of the file, packed 32 bytes.
// Row: ts_recv u64, price i64, size u32, side u8 ('A','B','N'), flags u8, pad u16, sequence u32, instrument_id u32
// Header: "MBOTRDS1" + u32 row_bytes + u32 reserved.
#include <cstdio>
#include <vector>

#include "mbo/dbn.hpp"

#pragma pack(push, 1)
struct TradeRow { uint64_t ts_recv; int64_t price; uint32_t size; uint8_t side, flags; uint16_t pad; uint32_t sequence, instrument_id; };
#pragma pack(pop)
static_assert(sizeof(TradeRow) == 32);

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: trades_extract FILE OUT\n"); return 2; }
  try {
    mbo::DbnReader r(argv[1]);
    std::vector<TradeRow> rows;
    while (const mbo::MboMsg* m = r.next_mbo()) {
      if (m->action != 'T') continue;
      rows.push_back({m->ts_recv, m->price, m->size, uint8_t(m->side), m->flags, 0, m->sequence, m->hd.instrument_id});
    }
    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) return 1;
    char hdr[16] = {'M', 'B', 'O', 'T', 'R', 'D', 'S', '1'};
    uint32_t rb = sizeof(TradeRow);
    std::memcpy(hdr + 8, &rb, 4);
    bool ok = std::fwrite(hdr, 1, 16, f) == 16 && std::fwrite(rows.data(), sizeof(TradeRow), rows.size(), f) == rows.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::fprintf(stderr, "write failed\n"); return 1; }
    std::printf("{\"trades\":%zu}\n", rows.size());
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
