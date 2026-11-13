// of24_parity: runs the OF24 kernel on bars from a text file and prints every candidate, for the parity test
// against experiments/of23_mine.py (python/tests/test_of24.py).
//
//   of24_parity BARS.txt [WEIGHTS.txt]
// BARS.txt: "n_prev n dow", then n_prev + n lines "open high low close volume buy sell" (ticks; the prior day first;
// n_prev may be 0). Output per candidate: "C m dir f0 .. f36", then with weights "P logit0 .. logit12 value", and
// "U ..." the logits after one update(action 3, reward 0.5) on a copy of the policy.
#include <cstdio>
#include <fstream>
#include <iostream>

#include "rag/live/of24_kernel.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: of24_parity BARS.txt [WEIGHTS.txt]\n"); return 2; }
  std::ifstream f(argv[1]);
  int n_prev = 0, n = 0, dow = 0;
  if (!(f >> n_prev >> n >> dow)) { std::fprintf(stderr, "bad header\n"); return 2; }
  of24::Bars prev, day;
  for (int i = 0; i < n_prev + n; ++i) {
    double o, h, l, c, v, b, s;
    if (!(f >> o >> h >> l >> c >> v >> b >> s)) { std::fprintf(stderr, "bad bar %d\n", i); return 2; }
    (i < n_prev ? prev : day).push(o, h, l, c, v, b, s);
  }
  of24::Levels lv;
  if (n_prev > 0) lv = of24::prior_levels(prev);
  of24::Policy pol;
  const bool has_w = argc > 2;
  if (has_w && !pol.load(argv[2])) { std::fprintf(stderr, "bad weights\n"); return 2; }
  if (lv.ok) {
    std::printf("L");
    for (double x : lv.v) std::printf(" %.17g", x);
    std::printf("\n");
  }
  of24::Bars cut;
  for (int m = 0; m < n; ++m) {
    cut.push(day.o[m], day.h[m], day.l[m], day.c[m], day.v[m], day.buy[m], day.sell[m]);
    of24::Cand c;
    if (!of24::candidate_at(cut, lv.ok ? &lv : nullptr, dow, c)) continue;
    std::printf("C %d %d", c.m, c.dir);
    for (double x : c.f) std::printf(" %.17g", x);
    std::printf("\n");
    if (has_w) {
      const auto x = pol.input(c, {});
      const auto o = pol.forward(x);
      std::printf("P");
      for (double z : o.logit) std::printf(" %.17g", z);
      std::printf(" %.17g\n", o.value);
      of24::Policy q = pol;
      q.update(x, 3, 0.5);
      const auto o2 = q.forward(x);
      std::printf("U");
      for (double z : o2.logit) std::printf(" %.17g", z);
      std::printf(" %.17g\n", o2.value);
    }
  }
  return 0;
}
