// Digest of every MBO record in a DBN file, for exact comparison with the
// Python reference (scripts/verify_dbn.py). Prints one JSON object.
#include <chrono>
#include <cstdio>
#include <map>

#include "mbo/dbn.hpp"

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: dbn_digest FILE\n"); return 2; }
  try {
    auto t0 = std::chrono::steady_clock::now();
    mbo::DbnReader r(argv[1]);
    constexpr int NF = 13;
    static const char* names[NF] = {"publisher_id", "instrument_id", "ts_event", "order_id", "price",
                                    "size", "flags", "channel_id", "action", "side", "ts_recv",
                                    "ts_in_delta", "sequence"};
    uint64_t s1[NF] = {0}, s2[NF] = {0}, n = 0;
    std::map<char, uint64_t> actions;
    while (const mbo::MboMsg* m = r.next_mbo()) {
      ++n;
      uint64_t v[NF] = {m->hd.publisher_id, m->hd.instrument_id, m->hd.ts_event, m->order_id,
                        uint64_t(m->price), m->size, m->flags, m->channel_id,
                        uint8_t(m->action), uint8_t(m->side), m->ts_recv,
                        uint64_t(int64_t(m->ts_in_delta)), m->sequence};
      for (int i = 0; i < NF; ++i) { s1[i] += v[i]; s2[i] += v[i] * n; }
      ++actions[m->action];
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("{\"version\":%d,\"dataset\":\"%s\",\"schema\":%u,\"ts_out\":%d,\"count\":%llu,\"skipped\":%llu,\"actions\":{",
                r.metadata().version, r.metadata().dataset.c_str(), r.metadata().schema,
                r.metadata().ts_out, (unsigned long long)n, (unsigned long long)r.skipped());
    bool first = true;
    for (auto& [a, c] : actions) { std::printf("%s\"%c\":%llu", first ? "" : ",", a, (unsigned long long)c); first = false; }
    std::printf("},\"sum\":{");
    for (int i = 0; i < NF; ++i) std::printf("%s\"%s\":\"%llu\"", i ? "," : "", names[i], (unsigned long long)s1[i]);
    std::printf("},\"wsum\":{");
    for (int i = 0; i < NF; ++i) std::printf("%s\"%s\":\"%llu\"", i ? "," : "", names[i], (unsigned long long)s2[i]);
    std::printf("},\"seconds\":%.3f}\n", sec);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
