#include <cstdio>
#include <cstdlib>

#include "mbo/book.hpp"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

using namespace mbo;
static const int64_t T = 250000000;  // ES tick in 1e-9 units
static MboMsg msg(char act, char side, uint64_t id, int64_t px, uint32_t sz, uint8_t fl = 0) {
  MboMsg m{};
  m.hd.length = sizeof(MboMsg) / 4; m.hd.rtype = RTYPE_MBO; m.hd.instrument_id = 1;
  m.action = act; m.side = side; m.order_id = id; m.price = px; m.size = sz; m.flags = fl;
  return m;
}
static void ap(Book& b, MboMsg m) { b.apply(m); CHECK(b.validate()); }

int main() {
  { // add / best / FIFO queue position
    Book b;
    ap(b, msg('A', 'B', 1, 100 * T, 5)); ap(b, msg('A', 'B', 2, 100 * T, 3)); ap(b, msg('A', 'B', 3, 99 * T, 7));
    ap(b, msg('A', 'A', 4, 101 * T, 2));
    CHECK(b.best_bid()->price == 100 * T && b.best_bid()->size == 8 && b.best_bid()->orders == 2);
    CHECK(b.best_ask()->price == 101 * T && !b.crossed());
    uint64_t ahead, tot;
    CHECK(b.queue_position(2, ahead, tot) && ahead == 5 && tot == 8);
    CHECK(b.queue_position(1, ahead, tot) && ahead == 0);
    CHECK(!b.queue_position(99, ahead, tot));
    // bids best-first order
    int64_t px[2]; int n = 0;
    b.visit_bids(2, [&](int64_t p, uint64_t, uint32_t) { px[n++] = p; });
    CHECK(n == 2 && px[0] == 100 * T && px[1] == 99 * T);
  }
  { // partial and full cancel; level disappears
    Book b;
    ap(b, msg('A', 'B', 1, 100 * T, 5)); ap(b, msg('A', 'B', 2, 100 * T, 3));
    ap(b, msg('C', 'B', 1, 100 * T, 2));
    uint64_t a, t;
    CHECK(b.best_bid()->size == 6 && b.queue_position(1, a, t) && a == 0 && b.stats().partial_cancels == 1);
    ap(b, msg('C', 'B', 1, 100 * T, 3)); ap(b, msg('C', 'B', 2, 100 * T, 3));
    CHECK(!b.best_bid() && b.order_count() == 0 && b.bid_levels() == 0);
  }
  { // modify priority: size down keeps place, size up / price change goes to back
    Book b;
    ap(b, msg('A', 'A', 1, 101 * T, 5)); ap(b, msg('A', 'A', 2, 101 * T, 5)); ap(b, msg('A', 'A', 3, 101 * T, 5));
    uint64_t a, t;
    ap(b, msg('M', 'A', 1, 101 * T, 3));  // down: stays first
    CHECK(b.queue_position(1, a, t) && a == 0 && t == 13);
    ap(b, msg('M', 'A', 1, 101 * T, 9));  // up: to back
    CHECK(b.queue_position(1, a, t) && a == 10 && t == 19);
    ap(b, msg('M', 'A', 2, 102 * T, 5));  // reprice: new level, alone
    CHECK(b.queue_position(2, a, t) && a == 0 && t == 5 && b.ask_levels() == 2);
    CHECK(b.stats().modify_requeued == 2);
  }
  { // T/F/N do not change the book; R clears
    Book b;
    ap(b, msg('A', 'B', 1, 100 * T, 5));
    ap(b, msg('T', 'A', 0, 100 * T, 2)); ap(b, msg('F', 'B', 1, 100 * T, 2)); ap(b, msg('N', 'N', 0, 0, 0));
    CHECK(b.best_bid()->size == 5);
    ap(b, msg('R', 'N', 0, UNDEF_PRICE, 0));
    CHECK(b.order_count() == 0 && !b.best_bid() && b.stats().clears == 1);
  }
  { // anomalies are counted, book stays consistent
    Book b;
    ap(b, msg('C', 'B', 42, 100 * T, 1));
    CHECK(b.stats().unknown_cancel == 1);
    ap(b, msg('A', 'B', 1, 100 * T, 5)); ap(b, msg('A', 'B', 1, 100 * T, 4));
    CHECK(b.stats().duplicate_add == 1 && b.best_bid()->size == 4 && b.order_count() == 1);
    ap(b, msg('C', 'B', 1, 100 * T, 10));
    CHECK(b.stats().oversize_cancel == 1 && b.order_count() == 0);
    ap(b, msg('A', 'N', 5, 100 * T, 1)); ap(b, msg('A', 'B', 6, UNDEF_PRICE, 1)); ap(b, msg('A', 'B', 7, 100 * T, 0));
    CHECK(b.stats().bad_side == 1 && b.stats().bad_price_or_size == 2 && b.order_count() == 0);
    ap(b, msg('M', 'B', 8, 100 * T, 2));  // unknown modify => add + counted
    CHECK(b.stats().unknown_modify == 1 && b.order_count() == 1);
  }
  { // crossed detection; TOB/MBP records ignored
    Book b;
    ap(b, msg('A', 'B', 1, 101 * T, 1)); ap(b, msg('A', 'A', 2, 100 * T, 1));
    CHECK(b.crossed());
    ap(b, msg('A', 'B', 3, 50 * T, 1, flag::TOB));
    CHECK(b.stats().ignored_tob_or_mbp == 1 && b.order_count() == 2);
  }
  { // stress: random ops keep invariants, matches a naive model of order count
    Book b; std::srand(7); std::vector<uint64_t> live; uint64_t next = 1;
    for (int i = 0; i < 20000; ++i) {
      int r = std::rand() % 10;
      if (r < 5 || live.empty()) {
        uint64_t id = next++; live.push_back(id);
        b.apply(msg('A', (std::rand() & 1) ? 'B' : 'A', id, (90 + std::rand() % 20) * T, 1 + std::rand() % 9));
      } else {
        size_t k = std::rand() % live.size(); uint64_t id = live[k];
        if (r < 7) { b.apply(msg('M', 'B', id, (90 + std::rand() % 20) * T, 1 + std::rand() % 9)); }
        else { b.apply(msg('C', 'B', id, 0, 100)); live[k] = live.back(); live.pop_back(); }
      }
      if (i % 997 == 0) CHECK(b.validate());
    }
    CHECK(b.validate() && b.order_count() == live.size() && b.stats().unknown_cancel == 0 && b.stats().duplicate_add == 0 && b.stats().unknown_modify == 0);
  }
  std::printf(fails ? "FAILED (%d)\n" : "book_test: all passed\n", fails);
  return fails ? 1 : 0;
}
