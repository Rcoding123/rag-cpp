#include <chrono>
#include <cstdio>
#include "rag.hpp"
using C = std::chrono::steady_clock;
int main(int, char** argv) {
  rag::EngineConfig cfg; cfg.max_chunks = 1u << 17; cfg.chunker = {128, 32};
  auto t0 = C::now();
  rag::RagEngine<rag::HashingEmbedder<384>> probe(cfg);  // construction cost alone
  auto t1 = C::now();
  auto l = rag::RagEngine<rag::HashingEmbedder<384>>::load(argv[1], cfg);
  auto t2 = C::now();
  std::printf("construct=%.0f ms load=%.0f ms chunks=%u\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count(), l->size());
}
