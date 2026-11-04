#include <chrono>
#include <cstdio>
#include "fixtures.hpp"
using namespace rag; using namespace rag::bench;
using C = std::chrono::steady_clock;
static double ms(C::time_point a, C::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
int main(int, char** argv) {
  for (auto mode : {DenseMode::Exact, DenseMode::IvfInt8}) {
    auto cfg = engine(DenseMode::Exact).config(); cfg.dense_mode = mode;
    auto t0 = C::now(); auto l = Engine::load(argv[1], cfg); auto t1 = C::now();
    l->commit(); auto t2 = C::now();  // commit cost alone = BM25 freeze (+ IVF build)
    { io::Reader r(argv[1]); std::vector<char> buf(1 << 20); while (r.remaining()) { auto n = std::min<std::uint64_t>(buf.size(), r.remaining()); r.bytes(buf.data(), n); } r.finish(); }
    auto t3 = C::now();
    std::printf("%-8s load=%.0f ms | commit=%.0f ms | read+checksum only=%.0f ms\n", mode_name(mode), ms(t0, t1), ms(t1, t2), ms(t2, t3));
  }
}
