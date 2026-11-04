#include <cstdio>
#include "rag/agent/drift.hpp"
#include "rag/agent/rng.hpp"
using namespace rag::agent;
int main() {
  for (std::uint64_t seed = 1; seed <= 10; ++seed) {
    DriftDetector d; Rng r(seed);
    for (int i = 0; i < 3000; ++i) d.add(10.0 * r.normal());
    int alarm = -1;
    for (int i = 0; i < 400 && alarm < 0; ++i) if (d.add(-8.0 + 10.0 * r.normal())) alarm = i;
    d.rebase();
    int alarms = 0; for (int i = 0; i < 300; ++i) alarms += d.add(-8.0 + 10.0 * r.normal());
    int fired = -1; for (int i = 0; i < 100 && fired < 0; ++i) if (d.add(-60.0 + 10.0 * r.normal())) fired = i;
    std::printf("seed %llu: first alarm after %d, alarms in 300 after rebase %d, real change caught after %d\n", (unsigned long long)seed, alarm, alarms, fired);
  }
}
