#include <cmath>
#include <cstdio>
#include "mbo/bars.hpp"
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
using namespace mbo;
static const int64_t T = 250000000, MS = 1000000, B = 100 * MS;  // 100 ms buckets
static MboMsg msg(uint64_t ts_ms, char act, char side, uint64_t id, int64_t px, uint32_t sz, uint8_t fl = flag::LAST) {
  MboMsg m{};
  m.hd.length = sizeof(MboMsg) / 4; m.hd.rtype = RTYPE_MBO; m.hd.instrument_id = 7;
  m.ts_recv = ts_ms * MS; m.action = act; m.side = side; m.order_id = id; m.price = px; m.size = sz; m.flags = fl;
  return m;
}
int main() {
  static_assert(sizeof(BarRow) == 120);
  std::vector<BarRow> out; BarBuilder b(7, B);
  // bucket 0 (0-100ms): build bid 100 x5 (id1), ask 101 x4 (id2); first event ends the bucket
  b.add(msg(10, 'A', 'B', 1, 100 * T, 5), out);
  b.add(msg(20, 'A', 'A', 2, 101 * T, 4), out);
  // bucket 1 (100-200): ask add 101 x6 (id3), a buyer lifts 2 at the ask: T + F + C as separate records
  b.add(msg(110, 'A', 'A', 3, 101 * T, 6), out);
  b.add(msg(150, 'T', 'B', 0, 101 * T, 2, 0), out);
  b.add(msg(150, 'F', 'A', 2, 101 * T, 2, 0), out);
  b.add(msg(150, 'C', 'A', 2, 101 * T, 2), out);
  // bucket 2 (200-300): bid 100 x3 more (id4); a seller hits the bid 4 (T side A at 100), cancel 4 of id1
  b.add(msg(210, 'A', 'B', 4, 100 * T, 3), out);
  b.add(msg(250, 'T', 'A', 0, 100 * T, 4, 0), out);
  b.add(msg(250, 'C', 'B', 1, 100 * T, 4), out);
  // bucket 5 (500-600): bid moves up to 100.25 (id5 x7): new best bid
  b.add(msg(510, 'A', 'B', 5, 100 * T + T / 1, 7), out);
  b.finish(out);
  CHECK(out.size() == 4);
  if (out.size() == 4) {
    const BarRow &r0 = out[0], &r1 = out[1], &r2 = out[2], &r3 = out[3];
    CHECK(r0.ts_end_ns == uint64_t(B) && r0.bid_px == 100 * T && r0.ask_px == 101 * T);
    CHECK(r0.bid_sz[0] == 5 && r0.ask_sz[0] == 4 && r0.n_add == 2 && r0.ofi == 0 && r0.flags == 0);
    // bucket 1: ask total 4-2+6 = 8, buyer volume 2 at the previous ask, count of C=1
    CHECK(r1.ask_sz[0] == 8 && r1.buy_vol == 2 && r1.bought_at_ask == 2 && r1.n_trades == 1 && r1.n_cancel == 1);
    CHECK(r1.last_trade_px == 101 * T && r1.sell_vol == 0);
    // OFI b1: bid same(+5-5=0); ask same price: -ask_sz_new(8) + ask_sz_old(4) = -4
    CHECK(std::fabs(r1.ofi + 4.f) < 1e-6);
    // bucket 2: bid total 5+3-4 = 4; seller 4 at bid => sold_at_bid 4; ofi: bid_new 4 - bid_old 5 = -1; ask same => 0
    CHECK(r2.bid_sz[0] == 4 && r2.sell_vol == 4 && r2.sold_at_bid == 4);
    CHECK(std::fabs(r2.ofi + 1.f) < 1e-6);
    // absorption: bid/ask prices never change after the first row, shown sizes stay 0 (as in
    // rag-cpp) so run_sold 4 - run_bought 2 (the lift in b1 still counts) = 2; b1 alone = -2
    CHECK(std::fabs(r1.absorb + 2.f) < 1e-6);
    CHECK(std::fabs(r2.absorb - 2.f) < 1e-6);
    // bucket 5: bid becomes 100.25 (new best): ofi += new size 7 (bid up) [no old term]; sizes visible
    CHECK(r3.ts_end_ns == uint64_t(6 * B) && r3.bid_px == 100 * T + T && r3.bid_sz[0] == 7 && r3.bid_sz[1] == 4);
    CHECK(std::fabs(r3.ofi - 7.f) < 1e-6);
    CHECK(r3.n_trades == 0 && r3.last_trade_px == 0);
    // bid changed => run_sold reset, bid_shown = 7; ask held => run_bought 2 stays => -2
    CHECK(std::fabs(r3.absorb + 2.f) < 1e-6);
  }
  { // no boundary yet => no row; crossed/one-side flags
    std::vector<BarRow> o2; BarBuilder c(7, B);
    c.add(msg(5, 'A', 'B', 1, 100 * T, 1, 0), o2); c.finish(o2);
    CHECK(o2.empty() && c.rows_dropped_no_boundary() == 1);
    std::vector<BarRow> o3; BarBuilder d(7, B);
    d.add(msg(5, 'A', 'B', 1, 101 * T, 1), o3); d.add(msg(6, 'A', 'A', 2, 100 * T, 1), o3); d.finish(o3);
    CHECK(o3.size() == 1 && (o3[0].flags & BAR_CROSSED) && !(o3[0].flags & BAR_ONE_SIDE_EMPTY));
    std::vector<BarRow> o4; BarBuilder e(7, B);
    e.add(msg(5, 'A', 'B', 1, 101 * T, 1, flag::LAST | flag::SNAPSHOT), o4); e.finish(o4);
    CHECK(o4.size() == 1 && (o4[0].flags & BAR_ONE_SIDE_EMPTY) && (o4[0].flags & BAR_SNAPSHOT) && o4[0].ask_px == 0);
  }
  std::printf(fails ? "FAILED (%d)\n" : "bars_test: all passed\n", fails);
  return fails ? 1 : 0;
}
