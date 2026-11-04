#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
using C = std::chrono::steady_clock;
static double ms(C::time_point a, C::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
int main() {
  const size_t n = 200u << 20, al = 2u << 20;
  for (int mode = 0; mode < 3; ++mode) {
    auto t0 = C::now();
    void* p = std::aligned_alloc(al, n);
    if (mode == 1) madvise(p, n, MADV_HUGEPAGE);
    auto t1 = C::now();
    if (mode < 2) std::memset(p, 0, n);
    auto t2 = C::now();
    // second pass: steady-state write bandwidth once pages exist
    std::memset(p, 1, n);
    auto t3 = C::now();
    std::printf("%-22s alloc=%6.1f ms  first-touch memset=%7.1f ms  second memset=%6.1f ms\n",
                mode == 0 ? "no madvise" : mode == 1 ? "MADV_HUGEPAGE" : "no prefault (lazy)", ms(t0, t1), ms(t1, t2), ms(t2, t3));
    std::free(p);
  }
}
