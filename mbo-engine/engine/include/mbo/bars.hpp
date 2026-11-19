// Fixed-width time buckets of raw order-flow inputs from the per-order book.
// One BarRow per bucket that contains at least one record (sparse; downstream
// forward-fills). Windowed/normalised features are derived from these rows
// (Python/Polars now, C# at deployment), so the row is the parity contract.
//
// Clock: buckets are aligned on UTC nanoseconds of ts_recv (what a live
// consumer would have seen), not ts_event.
// Book reads happen only at F_LAST event boundaries: the touch/depth in a row is
// the book at the last boundary inside the bucket (or the one before it).
// OFI and absorption follow rag-cpp include/rag/orderflow/features.hpp
// (Cont-Kukanov-Stoikov L1 OFI; absorption = sold beyond displayed size at a
// holding bid minus bought beyond displayed size at a holding ask), but are
// computed per bucket from level TOTALS of the real book.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "mbo/book.hpp"

namespace mbo {

constexpr int kDepth = 5;
constexpr uint8_t BAR_CROSSED = 1, BAR_ONE_SIDE_EMPTY = 2, BAR_SNAPSHOT = 4, BAR_BAD_TS_RECV = 8,
                  BAR_MAYBE_BAD_BOOK = 16;

#pragma pack(push, 1)
struct BarRow {
  uint64_t ts_end_ns;       // bucket end, exclusive, UTC ns
  uint32_t instrument_id;
  uint8_t flags;            // BAR_* bits
  uint8_t pad0[3];
  int64_t bid_px, ask_px;   // 1e-9 units; 0 = that side empty at the boundary
  uint32_t bid_sz[kDepth], ask_sz[kDepth];  // level totals, best first
  uint32_t buy_vol, sell_vol;               // T records by aggressor side
  uint32_t n_trades;                        // all T records (incl. side N)
  uint32_t sold_at_bid, bought_at_ask;      // aggressor volume at the previous bucket's touch
  int64_t last_trade_px;                    // 0 = no trade in bucket
  uint32_t n_add, n_cancel, n_modify;       // A/C/M records (all levels)
  float ofi;                                // L1 OFI vs previous row's touch
  float absorb;                             // rag-cpp absorption, running
};
#pragma pack(pop)
static_assert(sizeof(BarRow) == 120, "BarRow layout (also in scripts/bars_to_parquet.py)");
constexpr size_t kBarRowBytes = sizeof(BarRow);

class BarBuilder {
 public:
  BarBuilder(uint32_t instrument_id, uint64_t bucket_ns) : id_(instrument_id), bucket_ns_(bucket_ns) {}

  // Feed every record of this instrument in file order. Rows are appended to out.
  void add(const MboMsg& m, std::vector<BarRow>& out) {
    const uint64_t bk = m.ts_recv / bucket_ns_;
    if (open_ && bk != cur_) finalize(out);
    if (!open_) { open_ = true; cur_ = bk; row_ = BarRow{}; }
    book_.apply(m);
    if (m.flags & flag::SNAPSHOT) row_.flags |= BAR_SNAPSHOT;
    if (m.flags & flag::BAD_TS_RECV) row_.flags |= BAR_BAD_TS_RECV;
    if (m.flags & flag::MAYBE_BAD_BOOK) row_.flags |= BAR_MAYBE_BAD_BOOK;
    switch (m.action) {
      case 'A': ++row_.n_add; break;
      case 'C': ++row_.n_cancel; break;
      case 'M': ++row_.n_modify; break;
      case 'T':
        ++row_.n_trades;
        row_.last_trade_px = m.price;
        if (m.side == 'B') {
          row_.buy_vol += m.size;
          if (have_q_ && m.price == q_.ask_px) { row_.bought_at_ask += m.size; run_bought_ += m.size; }
        } else if (m.side == 'A') {
          row_.sell_vol += m.size;
          if (have_q_ && m.price == q_.bid_px) { row_.sold_at_bid += m.size; run_sold_ += m.size; }
        }
        break;
      default: break;
    }
    if (m.flags & flag::LAST) { capture(); have_snap_ = true; }
  }
  void finish(std::vector<BarRow>& out) { if (open_) finalize(out); }
  uint64_t rows_dropped_no_boundary() const { return dropped_; }
  const Book& book() const { return book_; }

 private:
  struct Quote { int64_t bid_px = 0, ask_px = 0; uint32_t bid_sz[kDepth] = {}, ask_sz[kDepth] = {}; uint8_t flags = 0; };

  void capture() {
    snap_ = Quote{};
    int i = 0;
    book_.visit_bids(kDepth, [&](int64_t px, uint64_t sz, uint32_t) {
      if (i == 0) snap_.bid_px = px;
      snap_.bid_sz[i++] = uint32_t(std::min<uint64_t>(sz, 0xFFFFFFFFu));
    });
    i = 0;
    book_.visit_asks(kDepth, [&](int64_t px, uint64_t sz, uint32_t) {
      if (i == 0) snap_.ask_px = px;
      snap_.ask_sz[i++] = uint32_t(std::min<uint64_t>(sz, 0xFFFFFFFFu));
    });
    if (book_.crossed()) snap_.flags |= BAR_CROSSED;
    if (snap_.bid_px == 0 || snap_.ask_px == 0) snap_.flags |= BAR_ONE_SIDE_EMPTY;
  }

  void finalize(std::vector<BarRow>& out) {
    open_ = false;
    if (!have_snap_) { ++dropped_; return; }  // no complete event seen yet
    BarRow& r = row_;
    r.ts_end_ns = (cur_ + 1) * bucket_ns_;
    r.instrument_id = id_;
    r.flags |= snap_.flags;
    r.bid_px = snap_.bid_px; r.ask_px = snap_.ask_px;
    std::copy(snap_.bid_sz, snap_.bid_sz + kDepth, r.bid_sz);
    std::copy(snap_.ask_sz, snap_.ask_sz + kDepth, r.ask_sz);
    const bool both = have_q_ && q_.bid_px && q_.ask_px && snap_.bid_px && snap_.ask_px;
    if (both) {  // Cont-Kukanov-Stoikov L1 OFI on level totals
      double e = 0;
      if (snap_.bid_px >= q_.bid_px) e += snap_.bid_sz[0];
      if (snap_.bid_px <= q_.bid_px) e -= q_.bid_sz[0];
      if (snap_.ask_px <= q_.ask_px) e -= snap_.ask_sz[0];
      if (snap_.ask_px >= q_.ask_px) e += q_.ask_sz[0];
      r.ofi = float(e);
    }
    if (have_q_) {
      if (snap_.bid_px != q_.bid_px) { run_sold_ = 0; bid_shown_ = snap_.bid_sz[0]; }
      if (snap_.ask_px != q_.ask_px) { run_bought_ = 0; ask_shown_ = snap_.ask_sz[0]; }
      absorb_ = std::max(0.0, run_sold_ - bid_shown_) - std::max(0.0, run_bought_ - ask_shown_);
    }
    r.absorb = float(absorb_);
    q_ = snap_; have_q_ = true;
    out.push_back(r);
  }

  uint32_t id_;
  uint64_t bucket_ns_;
  Book book_;
  bool open_ = false, have_snap_ = false, have_q_ = false;
  uint64_t cur_ = 0, dropped_ = 0;
  BarRow row_{};
  Quote snap_, q_;
  double run_sold_ = 0, run_bought_ = 0, bid_shown_ = 0, ask_shown_ = 0, absorb_ = 0;
};

}  // namespace mbo
