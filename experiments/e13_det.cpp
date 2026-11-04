#include <cstdio>
#include "rag/agent/drift.hpp"
#include "rag/agent/rng.hpp"
using namespace rag::agent;
int main() {
  DriftParams p; p.two_sided = true;
  DriftDetector d(p); Rng r(1);
  for (int i = 0; i < 3000; ++i) d.add(8 * r.normal());
  for (int i = 0; i < 50; ++i) if (d.add(25 + 8 * r.normal())) { std::printf("alarm after %d, upward %d\n", i + 1, d.last_alarm_was_upward()); return 0; }
  std::printf("no alarm\n");
}
