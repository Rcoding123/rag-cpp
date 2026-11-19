// fifo_check FILE [cme|keep|requeue|eq]
// ES matches price-time FIFO (to be confirmed from the exchange's definition: match_algorithm), so when an order
// is FILLED (action F) every order ahead of it in OUR queue (ranked by MESSAGE ORDER, as Databento's docs say priority
// is recoverable from message order; snapshot records are published in priority order) should have been filled too.
// Counts F records that break this, in layers:
//   lenient   = an unfilled order is ahead, allowing orders filled ANYWHERE in the same packet (same sequence)
//   residual  = lenient, and at least one such blocker was NOT removed (C) in the same packet either
//               (a removed blocker could be a fill whose F record is missing: "fill suppressed" cases)
// Fills of the AGGRESSING order (F with the same order_id as the packet's T) are excluded: they are not queue fills.
// and reports who the blockers are (snapshot or live) and how big the queue gap is.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mbo/book.hpp"

static double pct(std::vector<double>& v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: fifo_check FILE [cme|keep|requeue|eq]\n"); return 2; }
  mbo::ModifyPolicy pol = mbo::ModifyPolicy::Cme;
  if (argc > 2 && !std::strcmp(argv[2], "cme")) pol = mbo::ModifyPolicy::Cme;
  if (argc > 2 && !std::strcmp(argv[2], "keep")) pol = mbo::ModifyPolicy::KeepOnIncrease;
  if (argc > 2 && !std::strcmp(argv[2], "requeue")) pol = mbo::ModifyPolicy::AlwaysRequeue;
  if (argc > 2 && !std::strcmp(argv[2], "eq")) pol = mbo::ModifyPolicy::RequeueOnEqual;
  try {
    mbo::DbnReader r(argv[1]);
    mbo::Book b(pol);
    std::vector<mbo::MboMsg> grp;
    std::unordered_set<uint64_t> snap_ids;
    std::unordered_set<uint64_t> had_fill;           // resting orders that already received a passive fill earlier
    uint64_t res_blocker_prev_filled = 0, res_x_prev_filled = 0, res_blockers_total = 0, res_blockers_prev_filled = 0;
    uint64_t aggressor_fills = 0, fills = 0, unknown = 0, price_mismatch = 0, strict = 0, lenient = 0, residual = 0, removed_explains = 0;
    bool verbose = argc > 3 && !std::strcmp(argv[3], "verbose"); int ex_mismatch = 0;
    uint64_t res_x_snap = 0, res_blocker_snap = 0, res_both_live = 0, fills_x_snap = 0;
    std::vector<double> gap_size, gap_ratio, gap_orders;
    auto flush = [&]() {
      std::unordered_set<uint64_t> filled_any, removed_any, filled_before;
      for (auto& m : grp) { if (m.action == 'F') filled_any.insert(m.order_id); if (m.action == 'C') removed_any.insert(m.order_id); }
      uint64_t aggressor = 0;  // order_id of the latest T in this packet (the aggressing order, if it has one)
      for (auto& m : grp) {
        if (m.action == 'T') aggressor = m.order_id;
        if (m.action == 'F' && m.order_id != 0 && m.order_id == aggressor) {
          // Databento: if the aggressing order already rested on the book (modified to cross) it also gets an F record.
          // That is not a passive queue fill, so it is excluded from the FIFO test.
          ++aggressor_fills; filled_before.insert(m.order_id);
        } else if (m.action == 'F' && !(m.flags & mbo::flag::SNAPSHOT)) {
          ++fills;
          int64_t opx; uint32_t osz; char osd;
          if (!b.order_info(m.order_id, opx, osz, osd)) ++unknown;
          else {
            if (opx != m.price || osd != m.side) {
              ++price_mismatch;
              if (verbose && ex_mismatch < 6) { ++ex_mismatch; std::fprintf(stderr, "MISMATCH order=%llu F price=%.2f side=%c | book price=%.2f side=%c ts_recv=%llu seq=%u\n", (unsigned long long)m.order_id, m.price / 1e9, m.side, opx / 1e9, osd, (unsigned long long)m.ts_recv, m.sequence); }
            }
            bool xs = snap_ids.count(m.order_id) > 0; fills_x_snap += xs;
            bool bad_strict = false; double gsz = 0, gord = 0; bool unremoved = false, any_snap = false, seen = false, any_prev = false; uint64_t blk_total = 0, blk_prev = 0;
            uint64_t level_tot = 0, ahead_unused;
            b.queue_position(m.order_id, ahead_unused, level_tot);
            b.visit_queue(osd, opx, [&](uint64_t id, uint32_t sz) {
              if (seen) return;
              if (id == m.order_id) { seen = true; return; }
              if (!filled_before.count(id)) bad_strict = true;
              if (!filled_any.count(id)) {
                gsz += sz; gord += 1;
                if (!removed_any.count(id)) unremoved = true;
                if (snap_ids.count(id)) any_snap = true;
                ++blk_total; if (had_fill.count(id)) { ++blk_prev; any_prev = true; }
              }
            });
            strict += bad_strict;
            if (gord > 0) {
              ++lenient;
              if (!unremoved) ++removed_explains;
              else {
                ++residual; res_blocker_prev_filled += any_prev; res_x_prev_filled += had_fill.count(m.order_id) > 0; res_blockers_total += blk_total; res_blockers_prev_filled += blk_prev;
                if (xs) ++res_x_snap; else if (any_snap) ++res_blocker_snap; else ++res_both_live;
                gap_size.push_back(gsz); gap_orders.push_back(gord);
                gap_ratio.push_back(level_tot ? gsz / double(level_tot) : 0);
              }
            }
          }
          filled_before.insert(m.order_id); had_fill.insert(m.order_id);
        }
        if (m.action == 'C' || m.action == 'R') { if (m.action == 'R') had_fill.clear(); }
        if (m.action == 'A') { if (m.flags & mbo::flag::SNAPSHOT) snap_ids.insert(m.order_id); else snap_ids.erase(m.order_id); }
        b.apply(m);
      }
      grp.clear();
    };
    while (const mbo::MboMsg* m = r.next_mbo()) {
      if (!grp.empty() && m->sequence != grp.front().sequence) flush();
      grp.push_back(*m);
    }
    flush();
    auto P = [&](double x) { return fills ? 100.0 * x / fills : 0.0; };
    std::printf("{\"policy\":\"%s\",\"aggressor_F_excluded\":%llu,\"fills\":%llu,\"fill_order_unknown\":%llu,\"fill_price_or_side_differs_from_resting_order\":%llu,"
                "\"strict\":%llu,\"lenient\":%llu,\"lenient_explained_by_blocker_removed_in_packet\":%llu,\"residual\":%llu,"
                "\"pct_strict\":%.4f,\"pct_lenient\":%.4f,\"pct_residual\":%.4f,"
                "\"fills_of_snapshot_orders\":%llu,\"residual_by_kind\":{\"filled_order_from_snapshot\":%llu,\"blocker_from_snapshot\":%llu,\"all_live\":%llu},"
                "\"residual_with_a_previously_filled_blocker\":%llu,\"residual_where_filled_order_was_itself_previously_filled\":%llu,\"blockers_in_residual\":%llu,\"of_which_previously_filled\":%llu,\"residual_gap\":{\"blockers_median\":%.1f,\"blockers_p90\":%.1f,\"blocked_contracts_median\":%.1f,\"blocked_contracts_p90\":%.1f,"
                "\"blocked_contracts_p99\":%.1f,\"share_of_level_median\":%.3f,\"share_of_level_p90\":%.3f}}\n",
                argc > 2 ? argv[2] : "cme", (unsigned long long)aggressor_fills, (unsigned long long)fills, (unsigned long long)unknown, (unsigned long long)price_mismatch,
                (unsigned long long)strict, (unsigned long long)lenient, (unsigned long long)removed_explains, (unsigned long long)residual,
                P(strict), P(lenient), P(residual), (unsigned long long)fills_x_snap,
                (unsigned long long)res_x_snap, (unsigned long long)res_blocker_snap, (unsigned long long)res_both_live,
                (unsigned long long)res_blocker_prev_filled, (unsigned long long)res_x_prev_filled, (unsigned long long)res_blockers_total, (unsigned long long)res_blockers_prev_filled,
                pct(gap_orders, .5), pct(gap_orders, .9), pct(gap_size, .5), pct(gap_size, .9), pct(gap_size, .99), pct(gap_ratio, .5), pct(gap_ratio, .9));
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
