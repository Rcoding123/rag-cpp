#include <chrono>
#include <cstdio>
#include <filesystem>
#include "fixtures.hpp"
using namespace rag; using namespace rag::bench;
using C = std::chrono::steady_clock;
int main(int, char** argv) {
  const std::filesystem::path p = argv[1];
  for (auto mode : {DenseMode::Exact, DenseMode::IvfInt8}) {
    auto t0 = C::now();
    auto& e = engine(mode);  // builds from text: tokenize, chunk, embed, index
    auto t1 = C::now();
    e.save(p);
    auto t2 = C::now();
    auto l = Engine::load(p, e.config());
    auto t3 = C::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("%-8s chunks=%u file=%.1f MB  build-from-text=%.0f ms  save=%.0f ms  load=%.0f ms\n", mode_name(mode),
                l->size(), double(std::filesystem::file_size(p)) / 1e6, ms(t0, t1), ms(t1, t2), ms(t2, t3));
  }
}
