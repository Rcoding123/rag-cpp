// Phase 2 acceptance checks on a real DBN day. For every F_LAST boundary:
//   * the instrument's book must not be crossed
//   * book internal consistency (sampled)
// For every T (trade) record, vs the touch captured at the previous boundary:
//   * buy aggressor (side B) must print at >= best ask
//   * sell aggressor (side A) must print at <= best bid
// (a sweep prints at/through the touch; never on the wrong side of it)
// Also: anomaly counters of the book, replay speed, first-record shape.
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "mbo/book.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: book_check FILE [CROSSED_TS_DUMP]\n"); return 2; }
  // optional dump: u32 instrument_id, u64 ts_recv per crossed F_LAST boundary (packed, 12 bytes)
  std::FILE* dump = argc == 3 ? std::fopen(argv[2], "wb") : nullptr;
  if (argc == 3 && !dump) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
  try {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    mbo::DbnReader r(argv[1]);
    mbo::BookSet books;
    struct Pre { int64_t bid = 0, ask = 0; bool has_bid = false, has_ask = false; };
    std::unordered_map<uint32_t, Pre> pre;
    uint64_t n = 0, boundaries = 0, crossed = 0, trades = 0, trades_checked = 0, trade_viol = 0,
             inconsistent = 0, snap_recs = 0, empty_side = 0, crossed_outside_window = 0;
    uint64_t max_orders = 0;
    // crossing runs: consecutive crossed boundaries of one instrument (pre-open order
    // entry is not matched, so a crossed book is expected there). Window is UTC seconds of day.
    struct Run { uint64_t start = 0, end = 0, n = 0; };
    std::unordered_map<uint32_t, Run> open_run;
    std::vector<Run> runs;
    auto in_window = [](uint64_t ts) { uint64_t s = (ts / 1000000000ULL) % 86400; return s >= 21 * 3600 + 1800 && s < 23 * 3600; };
    while (const mbo::MboMsg* m = r.next_mbo()) {
      ++n;
      if (m->flags & mbo::flag::SNAPSHOT) ++snap_recs;
      if (m->action == 'T') {  // judge against the pre-event touch
        ++trades;
        auto it = pre.find(m->hd.instrument_id);
        if (it != pre.end() && m->price != mbo::UNDEF_PRICE) {
          if (m->side == 'B' && it->second.has_ask) { ++trades_checked; if (m->price < it->second.ask) ++trade_viol; }
          else if (m->side == 'A' && it->second.has_bid) { ++trades_checked; if (m->price > it->second.bid) ++trade_viol; }
        }
      }
      mbo::Book& b = books.apply(*m);
      if (m->flags & mbo::flag::LAST) {
        ++boundaries;
        Run& rn = open_run[m->hd.instrument_id];
        if (b.crossed()) {
          ++crossed;
          if (dump) { uint32_t i = m->hd.instrument_id; uint64_t t = m->ts_recv; std::fwrite(&i, 4, 1, dump); std::fwrite(&t, 8, 1, dump); }
          if (!in_window(m->ts_recv)) ++crossed_outside_window;
          if (rn.n == 0) rn.start = m->ts_recv;
          rn.end = m->ts_recv; ++rn.n;
        } else if (rn.n) { runs.push_back(rn); rn = Run{}; }
        auto bb = b.best_bid(); auto ba = b.best_ask();
        if (!bb || !ba) ++empty_side;
        Pre& p = pre[m->hd.instrument_id];
        p.has_bid = bb.has_value(); p.has_ask = ba.has_value();
        if (bb) p.bid = bb->price;
        if (ba) p.ask = ba->price;
        if ((boundaries & 0xFFFFF) == 0 && !b.validate()) ++inconsistent;
        if (b.order_count() > max_orders) max_orders = b.order_count();
      }
    }
    if (dump && std::fclose(dump) != 0) { std::fprintf(stderr, "dump write failed\n"); return 1; }
    for (auto& [id, rn] : open_run) if (rn.n) runs.push_back(rn);
    double sec = std::chrono::duration<double>(clk::now() - t0).count();
    uint64_t anomalies = 0, resting = 0;
    for (auto& [id, b] : books.all()) { if (!b.validate()) ++inconsistent; anomalies += b.stats().anomalies(); resting += b.order_count(); }
    std::printf("{\"file\":\"%s\",\"records\":%llu,\"instruments\":%zu,\"boundaries\":%llu,"
                "\"crossed_at_boundary\":%llu,\"crossed_outside_preopen_window\":%llu,\"boundaries_one_side_empty\":%llu,\"trades\":%llu,"
                "\"trades_checked\":%llu,\"trade_side_violations\":%llu,\"book_inconsistent\":%llu,"
                "\"anomalies\":%llu,\"snapshot_records\":%llu,\"max_resting_orders\":%llu,"
                "\"end_resting_orders\":%llu,\"seconds\":%.2f,\"msgs_per_sec\":%.0f",
                argv[1], (unsigned long long)n, books.all().size(), (unsigned long long)boundaries,
                (unsigned long long)crossed, (unsigned long long)crossed_outside_window, (unsigned long long)empty_side, (unsigned long long)trades,
                (unsigned long long)trades_checked, (unsigned long long)trade_viol,
                (unsigned long long)inconsistent, (unsigned long long)anomalies,
                (unsigned long long)snap_recs, (unsigned long long)max_orders, (unsigned long long)resting,
                sec, sec > 0 ? double(n) / sec : 0.0);
    std::printf(",\"crossing_runs\":[");
    for (size_t i = 0; i < runs.size(); ++i)
      std::printf("%s{\"start_ns\":%llu,\"end_ns\":%llu,\"boundaries\":%llu}", i ? "," : "",
                  (unsigned long long)runs[i].start, (unsigned long long)runs[i].end, (unsigned long long)runs[i].n);
    std::printf("],\"per_instrument\":[");
    bool first = true;
    for (auto& [id, b] : books.all()) {
      const auto& s = b.stats();
      std::printf("%s{\"id\":%u,\"adds\":%llu,\"cancels\":%llu,\"partial\":%llu,\"mods\":%llu,\"requeued\":%llu,"
                  "\"clears\":%llu,\"unk_cancel\":%llu,\"unk_mod\":%llu,\"dup_add\":%llu,\"bad_side\":%llu,"
                  "\"bad_px_sz\":%llu,\"oversize_cancel\":%llu}",
                  first ? "" : ",", id, (unsigned long long)s.adds, (unsigned long long)s.cancels,
                  (unsigned long long)s.partial_cancels, (unsigned long long)s.modifies,
                  (unsigned long long)s.modify_requeued, (unsigned long long)s.clears,
                  (unsigned long long)s.unknown_cancel, (unsigned long long)s.unknown_modify,
                  (unsigned long long)s.duplicate_add, (unsigned long long)s.bad_side,
                  (unsigned long long)s.bad_price_or_size, (unsigned long long)s.oversize_cancel);
      first = false;
    }
    std::printf("]}\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
