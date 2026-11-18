// Per-order (L3) limit order book for CME MBO data, one Book per instrument.
//
// Semantics (Databento MBO docs, common-fields-enums-types "Action"):
//   A add      : new order joins the BACK of the FIFO queue at its price
//   C cancel   : full or partial cancel (size = quantity removed)
//   M modify   : price and/or size change. CME priority rule used here: a price
//                change or a size INCREASE sends the order to the back of the
//                queue; a size decrease at the same price keeps its place.
//   R clear    : drop every resting order of the instrument
//   T trade, F fill, N none : do NOT change the book (the matching resting-order
//                quantity is removed by a separate C record)
// The book may only be read at event boundaries (flags & F_LAST) because an
// event spans several records. Anything that contradicts the model (unknown
// order id, duplicate add, ...) is counted in Stats, never silently fixed.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "mbo/dbn.hpp"

namespace mbo {

struct BookStats {
  uint64_t adds = 0, cancels = 0, partial_cancels = 0, modifies = 0, modify_requeued = 0,
           clears = 0, trades = 0, fills = 0, nones = 0, ignored_tob_or_mbp = 0;
  // anomalies: each should be ~0 on a healthy feed
  uint64_t unknown_cancel = 0, unknown_modify = 0, duplicate_add = 0, bad_side = 0,
           bad_price_or_size = 0, oversize_cancel = 0;
  uint64_t anomalies() const {
    return unknown_cancel + unknown_modify + duplicate_add + bad_side + bad_price_or_size +
           oversize_cancel;
  }
};

struct Touch {
  int64_t price;
  uint64_t size;   // total resting size at the level
  uint32_t orders; // number of orders at the level
};

// What a modify does to queue priority. Cme is the documented CME rule and the default; the
// others exist only so tools/fifo_check can show empirically which rule the data follows.
enum class ModifyPolicy { Cme, KeepOnIncrease, AlwaysRequeue };

class Book {
 public:
  explicit Book(ModifyPolicy p = ModifyPolicy::Cme) : policy_(p) {}
  void apply(const MboMsg& m) {
    if (m.flags & (flag::TOB | flag::MBP)) { ++stats_.ignored_tob_or_mbp; return; }
    switch (m.action) {
      case 'A': add(m.order_id, m.side, m.price, m.size); break;
      case 'C': cancel(m.order_id, m.size); break;
      case 'M': modify(m.order_id, m.side, m.price, m.size); break;
      case 'R': clear(); ++stats_.clears; break;
      case 'T': ++stats_.trades; break;
      case 'F': ++stats_.fills; break;
      default: ++stats_.nones; break;
    }
  }

  std::optional<Touch> best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return touch(bids_.begin());
  }
  std::optional<Touch> best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return touch(asks_.begin());
  }
  bool crossed() const {
    return !bids_.empty() && !asks_.empty() && bids_.begin()->first >= asks_.begin()->first;
  }
  // Current price/size/side of a resting order; false if not resting.
  bool order_info(uint64_t id, int64_t& price, uint32_t& size, char& side) const {
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    const Order& o = pool_[it->second];
    price = o.price; size = o.size; side = o.side;
    return true;
  }
  size_t order_count() const { return index_.size(); }
  size_t bid_levels() const { return bids_.size(); }
  size_t ask_levels() const { return asks_.size(); }
  const BookStats& stats() const { return stats_; }

  // Quantity ahead of `id` in its level's FIFO queue, plus the level total.
  // O(orders in level). False if the order is not resting.
  bool queue_position(uint64_t id, uint64_t& ahead, uint64_t& level_total) const {
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    const Order& o = pool_[it->second];
    const Level* lv = o.side == 'B' ? find_level(bids_, o.price) : find_level(asks_, o.price);
    ahead = 0;
    for (uint32_t i = lv->head; i != NIL && i != it->second; i = pool_[i].next) ahead += pool_[i].size;
    level_total = lv->total;
    return true;
  }

  // Visit levels best-first up to n levels: f(price, total_size, order_count).
  template <class F> void visit_bids(size_t n, F&& f) const { visit(bids_, n, f); }
  template <class F> void visit_asks(size_t n, F&& f) const { visit(asks_, n, f); }

  // Visit orders at a price in queue order: f(order_id, size).
  template <class F> void visit_queue(char side, int64_t price, F&& f) const {
    const Level* lv = side == 'B' ? find_level(bids_, price) : find_level(asks_, price);
    if (!lv) return;
    for (uint32_t i = lv->head; i != NIL; i = pool_[i].next) f(pool_[i].id, pool_[i].size);
  }

  // Internal consistency (used by tests/tools): level totals == sum of orders,
  // every order reachable from its level, no empty levels.
  bool validate() const {
    size_t seen = 0;
    auto chk = [&](const auto& side_map, char side) {
      for (auto& [px, lv] : side_map) {
        uint64_t tot = 0; uint32_t cnt = 0, prev = NIL;
        for (uint32_t i = lv.head; i != NIL; i = pool_[i].next) {
          const Order& o = pool_[i];
          if (o.price != px || o.side != side || o.prev != prev) return false;
          auto it = index_.find(o.id);
          if (it == index_.end() || it->second != i) return false;
          tot += o.size; ++cnt; prev = i;
        }
        if (prev != lv.tail || tot != lv.total || cnt != lv.count || cnt == 0) return false;
        seen += cnt;
      }
      return true;
    };
    return chk(bids_, 'B') && chk(asks_, 'A') && seen == index_.size();
  }

 private:
  static constexpr uint32_t NIL = 0xFFFFFFFFu;
  struct Order {
    uint64_t id; int64_t price; uint32_t size; char side; uint32_t prev, next;
  };
  struct Level { uint32_t head = NIL, tail = NIL; uint64_t total = 0; uint32_t count = 0; };
  using Bids = std::map<int64_t, Level, std::greater<int64_t>>;
  using Asks = std::map<int64_t, Level>;

  static Touch touch(Asks::const_iterator it) {  // same iterator type for Bids and Asks
    return {it->first, it->second.total, it->second.count};
  }
  template <class M> static const Level* find_level(const M& m, int64_t px) {
    auto it = m.find(px);
    return it == m.end() ? nullptr : &it->second;
  }
  template <class M, class F> static void visit(const M& m, size_t n, F& f) {
    for (auto it = m.begin(); it != m.end() && n; ++it, --n) f(it->first, it->second.total, it->second.count);
  }

  uint32_t alloc() {
    if (!free_.empty()) { uint32_t i = free_.back(); free_.pop_back(); return i; }
    pool_.push_back({}); return uint32_t(pool_.size() - 1);
  }
  Level& level_ref(char side, int64_t px) { return side == 'B' ? bids_.find(px)->second : asks_.find(px)->second; }
  Level& level_for(char side, int64_t px) { return side == 'B' ? bids_[px] : asks_[px]; }

  void link_back(uint32_t i) {
    Order& o = pool_[i];
    Level& lv = level_for(o.side, o.price);
    o.prev = lv.tail; o.next = NIL;
    if (lv.tail != NIL) pool_[lv.tail].next = i; else lv.head = i;
    lv.tail = i; lv.total += o.size; ++lv.count;
  }
  void unlink(uint32_t i) {
    Order& o = pool_[i];
    auto drop = [&](auto& m) {
      auto it = m.find(o.price);
      Level& lv = it->second;
      if (o.prev != NIL) pool_[o.prev].next = o.next; else lv.head = o.next;
      if (o.next != NIL) pool_[o.next].prev = o.prev; else lv.tail = o.prev;
      lv.total -= o.size; --lv.count;
      if (lv.count == 0) m.erase(it);
    };
    if (o.side == 'B') drop(bids_); else drop(asks_);
  }
  void release(uint32_t i) { index_.erase(pool_[i].id); free_.push_back(i); }

  void add(uint64_t id, char side, int64_t px, uint32_t sz) {
    if (side != 'B' && side != 'A') { ++stats_.bad_side; return; }
    if (px == UNDEF_PRICE || sz == 0) { ++stats_.bad_price_or_size; return; }
    if (index_.count(id)) {  // keep the book consistent: replace the stale order
      ++stats_.duplicate_add;
      uint32_t old = index_[id]; unlink(old); release(old);
    }
    ++stats_.adds;
    uint32_t i = alloc();
    pool_[i] = {id, px, sz, side, NIL, NIL};
    index_[id] = i;
    link_back(i);
  }
  void cancel(uint64_t id, uint32_t sz) {
    auto it = index_.find(id);
    if (it == index_.end()) { ++stats_.unknown_cancel; return; }
    uint32_t i = it->second;
    Order& o = pool_[i];
    if (sz > o.size) ++stats_.oversize_cancel;
    if (sz >= o.size) { ++stats_.cancels; unlink(i); release(i); return; }
    ++stats_.partial_cancels;
    o.size -= sz;
    level_ref(o.side, o.price).total -= sz;
  }
  void modify(uint64_t id, char side, int64_t px, uint32_t sz) {
    auto it = index_.find(id);
    if (it == index_.end()) { ++stats_.unknown_modify; add(id, side, px, sz); return; }
    if (px == UNDEF_PRICE || sz == 0) { ++stats_.bad_price_or_size; return; }
    ++stats_.modifies;
    uint32_t i = it->second;
    Order& o = pool_[i];
    bool keep = px == o.price && sz <= o.size;  // CME: only a size decrease at the same price keeps place
    if (policy_ == ModifyPolicy::KeepOnIncrease) keep = px == o.price;
    if (policy_ == ModifyPolicy::AlwaysRequeue) keep = false;
    if (keep) {
      Level& lv = level_ref(o.side, o.price);
      lv.total = lv.total - o.size + sz;
      o.size = sz;
      return;
    }
    ++stats_.modify_requeued;
    unlink(i);
    o.price = px; o.size = sz;
    if (side == 'B' || side == 'A') o.side = side;
    link_back(i);
  }
  void clear() { bids_.clear(); asks_.clear(); pool_.clear(); free_.clear(); index_.clear(); }

  Bids bids_;
  Asks asks_;
  std::vector<Order> pool_;
  std::vector<uint32_t> free_;
  std::unordered_map<uint64_t, uint32_t> index_;
  BookStats stats_;
  ModifyPolicy policy_;
};

// One Book per instrument_id (ES.v.0 can switch contract on roll days).
class BookSet {
 public:
  Book& apply(const MboMsg& m) {
    Book& b = books_[m.hd.instrument_id];
    b.apply(m);
    return b;
  }
  Book* find(uint32_t instrument_id) {
    auto it = books_.find(instrument_id);
    return it == books_.end() ? nullptr : &it->second;
  }
  const std::unordered_map<uint32_t, Book>& all() const { return books_; }

 private:
  std::unordered_map<uint32_t, Book> books_;
};

}  // namespace mbo
