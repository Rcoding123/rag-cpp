#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
#include "rag.hpp"
using E = rag::RagEngine<rag::HashingEmbedder<384>>;
int main() {
  E e(rag::EngineConfig{.max_chunks = 1024});
  e.ingest("The limit order book aggregates resting bids and asks at each price level.", 0);
  e.ingest("VWAP execution slices a parent order across the trading day.", 1);
  e.commit();
  e.save("/tmp/w.idx");
  std::ifstream f("/tmp/w.idx", std::ios::binary);
  std::vector<char> good{std::istreambuf_iterator<char>(f), {}};
  std::printf("size=%zu\n", good.size());
  for (std::size_t pos : {good.size() / 2, std::size_t{100}, good.size() - 20}) {
    auto b = good; b[pos] ^= 0x10;
    std::ofstream o("/tmp/w2.idx", std::ios::binary); o.write(b.data(), b.size()); o.close();
    try { (void)E::load("/tmp/w2.idx", rag::EngineConfig{.max_chunks = 1024}); std::printf("pos %zu: loaded!\n", pos); }
    catch (const std::exception& ex) { std::printf("pos %zu: %s\n", pos, ex.what()); }
  }
}
