// Lazily-built, process-wide benchmark fixtures (built once, reused by every run).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "rag.hpp"
#include "../tests/synthetic.hpp"

namespace rag::bench {

inline constexpr std::size_t kD = 384;
using Engine = RagEngine<HashingEmbedder<kD>>;

inline VectorStore<kD>& store(std::uint32_t n) {
  static std::vector<std::pair<std::uint32_t, std::unique_ptr<VectorStore<kD>>>> cache;
  for (auto& [k, s] : cache) if (k == n) return *s;
  auto s = std::make_unique<VectorStore<kD>>(n);
  synth::fill_clustered(*s, n, std::max(16u, n / 200), 0.6f);
  cache.emplace_back(n, std::move(s));
  return *cache.back().second;
}

inline const std::vector<float>& queries(std::uint32_t n_store) {
  static std::vector<std::pair<std::uint32_t, std::vector<float>>> cache;
  for (auto& [k, q] : cache) if (k == n_store) return q;
  cache.emplace_back(n_store, synth::random_queries<kD>(256, 99, store(n_store), 0.4f));
  return cache.back().second;
}

inline const synth::TextCorpus& corpus() {
  static const auto c = synth::make_text_corpus(20000, 200, 256);
  return c;
}

inline std::vector<std::string> text_queries(std::size_t n) {
  std::mt19937 rng(5);
  const auto& c = corpus();
  std::vector<std::string> q;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& w = c.topic_words[rng() % c.topic_words.size()];
    q.push_back("what about " + w[rng() % 40] + " and " + w[rng() % 40] + " with " + w[rng() % 40]);
  }
  return q;
}

inline Engine& engine(DenseMode mode) {
  static std::unique_ptr<Engine> e[3];
  auto& slot = e[static_cast<int>(mode)];
  if (!slot) {
    EngineConfig cfg;
    cfg.max_chunks = 1u << 17;
    cfg.chunker = {128, 32};
    cfg.dense_mode = mode;
    cfg.nprobe = 16;
    slot = std::make_unique<Engine>(cfg);
    const auto& c = corpus();
    for (std::uint32_t d = 0; d < c.docs.size(); ++d) slot->ingest(c.docs[d], d);
    slot->commit();
  }
  return *slot;
}

}  // namespace rag::bench
