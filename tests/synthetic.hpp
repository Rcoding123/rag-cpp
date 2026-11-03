// Synthetic data shared by tests and benchmarks (deterministic, seeded).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "rag/simd.hpp"
#include "rag/vector_store.hpp"

namespace rag::synth {

// Clustered unit vectors: realistic for ANN (real embeddings are far from
// uniform on the sphere) and makes IVF recall meaningful. `noise` is the
// noise-to-signal norm ratio (per-dim sigma = noise / sqrt(D)).
template <std::size_t D>
void fill_clustered(VectorStore<D>& store, std::uint32_t n, std::uint32_t clusters, float noise,
                    std::uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> centers(std::size_t{clusters} * D);
  for (auto& c : centers) c = g(rng);
  for (std::uint32_t c = 0; c < clusters; ++c) simd::normalize(centers.data() + std::size_t{c} * D, D);
  std::vector<float> v(D);
  const float sigma = noise / std::sqrt(static_cast<float>(D));
  for (std::uint32_t i = 0; i < n; ++i) {
    const float* c = centers.data() + std::size_t{rng() % clusters} * D;
    for (std::size_t d = 0; d < D; ++d) v[d] = c[d] + sigma * g(rng);
    simd::normalize(v.data(), D);
    store.append(std::span<const float, D>(v.data(), D));
  }
}

template <std::size_t D>
std::vector<float> random_queries(std::uint32_t n, std::uint64_t seed, const VectorStore<D>& near, float noise) {
  // Perturbed copies of stored rows: queries land near real data, as in practice.
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> q(std::size_t{n} * D);
  const float sigma = noise / std::sqrt(static_cast<float>(D));
  for (std::uint32_t i = 0; i < n; ++i) {
    const float* r = near.row(static_cast<std::uint32_t>(rng() % near.size()));
    float* dst = q.data() + std::size_t{i} * D;
    for (std::size_t d = 0; d < D; ++d) dst[d] = r[d] + sigma * g(rng);
    simd::normalize(dst, D);
  }
  return q;
}

// Topic-structured text: each topic has its own vocabulary, plus shared
// filler words drawn Zipf-like, so BM25 and dense retrieval both have signal.
struct TextCorpus {
  std::vector<std::string> docs;
  std::vector<std::uint32_t> topic;  // topic of each doc
  std::vector<std::vector<std::string>> topic_words;
};

inline TextCorpus make_text_corpus(std::uint32_t ndocs, std::uint32_t topics, std::uint32_t words_per_doc,
                                   std::uint64_t seed = 7) {
  std::mt19937_64 rng(seed);
  TextCorpus c;
  c.topic_words.resize(topics);
  for (std::uint32_t t = 0; t < topics; ++t)
    for (int w = 0; w < 40; ++w) c.topic_words[t].push_back("t" + std::to_string(t) + "w" + std::to_string(w));
  std::vector<std::string> filler;
  for (int w = 0; w < 2000; ++w) filler.push_back("f" + std::to_string(w));
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (std::uint32_t d = 0; d < ndocs; ++d) {
    const std::uint32_t t = static_cast<std::uint32_t>(rng() % topics);
    std::string s;
    for (std::uint32_t i = 0; i < words_per_doc; ++i) {
      if (u(rng) < 0.35) s += c.topic_words[t][rng() % 40];
      else s += filler[static_cast<std::size_t>(std::pow(u(rng), 3.0) * 2000)];  // Zipf-ish
      s += (i % 17 == 16) ? ". " : " ";
    }
    c.docs.push_back(std::move(s));
    c.topic.push_back(t);
  }
  return c;
}

}  // namespace rag::synth
