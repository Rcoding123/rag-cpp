// feat_ref: runs the ORIGINAL rag-cpp FeatureBuilder (include/rag/orderflow/features.hpp) on a bucket stream, as the
// reference for the vectorised pipeline in scripts/features.py (parity test).
// stdin text, one bucket per block:   Q bid ask b1..b5 a1..a5      (ticks, level totals)
//                                     T price size side           (side 0 = buy aggressor, 1 = sell), zero or more
//                                     E                           (end of bucket)
// stdout, per bucket: 13 raw values then 13 scaled values (%.17g).
#include <cstdio>
#include <cstring>

#include "rag/orderflow/features.hpp"

int main() {
  rag::orderflow::FeatureBuilder fb(0.1);  // 100 ms buckets, 600 s normalisation (the defaults)
  rag::orderflow::Quote q{};
  char tag[8];
  while (std::scanf("%7s", tag) == 1) {
    if (!std::strcmp(tag, "Q")) {
      int bid, ask; unsigned b[5], a[5];
      if (std::scanf("%d %d %u %u %u %u %u %u %u %u %u %u", &bid, &ask, &b[0], &b[1], &b[2], &b[3], &b[4], &a[0], &a[1], &a[2], &a[3], &a[4]) != 12) return 3;
      q.bid = bid; q.ask = ask;
      for (int i = 0; i < 5; ++i) { q.bid_size[i] = b[i]; q.ask_size[i] = a[i]; }
    } else if (!std::strcmp(tag, "T")) {
      int px, side; unsigned sz;
      if (std::scanf("%d %u %d", &px, &sz, &side) != 3) return 3;
      fb.add_trade({0, px, sz, side == 0 ? rag::orderflow::Side::Buy : rag::orderflow::Side::Sell, false});
    } else if (!std::strcmp(tag, "E")) {
      fb.end_bucket(q);
      auto r = fb.raw(); const auto& x = fb.values();
      for (size_t k = 0; k < rag::orderflow::kFeatures; ++k) std::printf("%.17g ", r[k]);
      for (size_t k = 0; k < rag::orderflow::kFeatures; ++k) std::printf("%.17g%c", x[k], k + 1 == rag::orderflow::kFeatures ? '\n' : ' ');
    } else return 3;
  }
  return 0;
}
