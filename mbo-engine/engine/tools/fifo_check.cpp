// fifo_check FILE [cme|keep|requeue]
// ES matches price-time FIFO, so when an order is FILLED (action F) every order ahead of it in
// our queue must itself have been filled earlier in the same event (its C follows the F records).
// Counts F records whose order is not at the front. Used to compare modify-priority policies.
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mbo/book.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: fifo_check FILE [cme|keep|requeue]\n"); return 2; }
  mbo::ModifyPolicy pol = mbo::ModifyPolicy::Cme;
  if (argc > 2 && !std::strcmp(argv[2], "keep")) pol = mbo::ModifyPolicy::KeepOnIncrease;
  if (argc > 2 && !std::strcmp(argv[2], "requeue")) pol = mbo::ModifyPolicy::AlwaysRequeue;
  try {
    mbo::DbnReader r(argv[1]);
    mbo::Book b(pol);
    // Process one packet (same sequence) at a time: ids filled ANYWHERE in the packet are allowed to
    // sit ahead of a later-listed fill (the F listing order need not equal queue order).
    std::vector<mbo::MboMsg> grp;
    uint64_t fills = 0, not_front = 0, unknown = 0, not_front_even_so = 0;
    // modification history per order id: bit0 = had a size-decrease-in-place M, bit1 = had a requeueing M
    std::unordered_map<uint64_t, uint8_t> hist;
    uint64_t cat[2][4] = {};  // [filled order X modified?][blocking order Y history 0..3] (lenient violations)
    std::unordered_set<uint64_t> snap_ids;           // orders created by the synthetic snapshot
    uint64_t snapcat[2][2] = {};                      // [X from snapshot][Y from snapshot] lenient violations
    uint64_t fills_x_snap = 0, blocker_id_larger = 0, blocker_id_smaller = 0;
    auto flush = [&]() {
      std::unordered_set<uint64_t> filled_any;
      for (auto& m : grp) if (m.action == 'F') filled_any.insert(m.order_id);
      std::unordered_set<uint64_t> filled_before;
      for (auto& m : grp) {
        if (m.action == 'F' && !(m.flags & mbo::flag::SNAPSHOT)) {
          ++fills; if (snap_ids.count(m.order_id)) ++fills_x_snap;
          uint64_t ahead, tot;
          if (!b.queue_position(m.order_id, ahead, tot)) ++unknown;
          else {
            bool bad_strict = false, bad_any = false, seen = false;
            uint64_t first_block = 0;
            b.visit_queue(m.side, m.price, [&](uint64_t id, uint32_t) {
              if (seen) return;
              if (id == m.order_id) { seen = true; return; }
              if (!filled_before.count(id)) bad_strict = true;
              if (!filled_any.count(id)) { if (!bad_any) first_block = id; bad_any = true; }
            });
            if (bad_any) {
              uint8_t hx = hist.count(m.order_id) ? hist[m.order_id] : 0, hy = hist.count(first_block) ? hist[first_block] : 0;
              ++cat[hx != 0][hy & 3];
              if (first_block > m.order_id) ++blocker_id_larger; else ++blocker_id_smaller;
              ++snapcat[snap_ids.count(m.order_id) ? 1 : 0][snap_ids.count(first_block) ? 1 : 0];
            }
            not_front += bad_strict; not_front_even_so += bad_any;
          }
          filled_before.insert(m.order_id);
        }
        if (m.action == 'M') {
          int64_t px; uint32_t sz; char sd;
          if (b.order_info(m.order_id, px, sz, sd)) hist[m.order_id] |= (m.price == px && m.size <= sz) ? 1 : 2;
        }
        if (m.action == 'A') { hist[m.order_id] = 0; if (m.flags & mbo::flag::SNAPSHOT) snap_ids.insert(m.order_id); else snap_ids.erase(m.order_id); }
        b.apply(m);
      }
      grp.clear();
    };
    while (const mbo::MboMsg* m = r.next_mbo()) {
      if (!grp.empty() && m->sequence != grp.front().sequence) flush();
      grp.push_back(*m);
    }
    flush();
    std::printf("{\"policy\":\"%s\",\"fills\":%llu,\"not_front_strict\":%llu,\"not_front_allowing_any_fill_in_packet\":%llu,"
                "\"fill_order_unknown\":%llu,\"pct_strict\":%.4f,\"pct_lenient\":%.4f,"
                "\"blocker_order_id_larger_than_filled\":%llu,\"blocker_order_id_smaller\":%llu,\"fills_of_snapshot_orders\":%llu,\"viol_by[X_snap][Y_snap]\":[[%llu,%llu],[%llu,%llu]],\"lenient_violations_by[X_modified][Y_hist(0 none,1 kept-priority M,2 requeued M,3 both)]\":[[%llu,%llu,%llu,%llu],[%llu,%llu,%llu,%llu]]}\n",
                argc > 2 ? argv[2] : "cme", (unsigned long long)fills, (unsigned long long)not_front,
                (unsigned long long)not_front_even_so, (unsigned long long)unknown,
                fills ? 100.0 * not_front / fills : 0.0, fills ? 100.0 * not_front_even_so / fills : 0.0,
                (unsigned long long)blocker_id_larger, (unsigned long long)blocker_id_smaller, (unsigned long long)fills_x_snap, (unsigned long long)snapcat[0][0], (unsigned long long)snapcat[0][1], (unsigned long long)snapcat[1][0], (unsigned long long)snapcat[1][1],
                (unsigned long long)cat[0][0], (unsigned long long)cat[0][1], (unsigned long long)cat[0][2], (unsigned long long)cat[0][3],
                (unsigned long long)cat[1][0], (unsigned long long)cat[1][1], (unsigned long long)cat[1][2], (unsigned long long)cat[1][3]);
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
