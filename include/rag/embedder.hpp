// rag/embedder.hpp — the Embedder concept plus a deterministic, model-free
// reference implementation (signed feature hashing of unigrams + bigrams).
//
// The engine depends only on the concept, so a production deployment swaps
// HashingEmbedder for an ONNX Runtime / llama.cpp / TensorRT sentence
// encoder with the same signature. The hashing embedder is useful in its own
// right: zero model load time, ~100 ns per query, and fully reproducible
// tests.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "rag/common.hpp"
#include "rag/simd.hpp"
#include "rag/tokenizer.hpp"

namespace rag {

// Writes an L2-normalised D-dim embedding of `text` (whose tokens are
// pre-computed) into `out`. Must not allocate: it runs on the query path.
template <class E>
concept Embedder = requires(const E& e, std::string_view text, std::span<const Token> toks,
                            std::span<float, E::kDim> out) {
  requires ValidDim<E::kDim>;
  { e.embed(text, toks, out) } noexcept;
};

// Identity of an embedding model, stored in index files so that vectors made
// by one model are never searched with queries embedded by another. An
// embedder opts in with `static constexpr std::string_view kName`; bump the
// name whenever its output changes.
template <class E>
constexpr std::uint64_t embedder_tag() noexcept {
  std::uint64_t h = 14695981039346656037ull;
  if constexpr (requires { E::kName; })
    for (char c : std::string_view(E::kName)) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
  return detail::mix64(h ^ E::kDim);
}

template <std::size_t D>
  requires ValidDim<D>
class HashingEmbedder {
 public:
  static constexpr std::size_t kDim = D;
  static constexpr std::string_view kName = "rag.hashing-unigram-bigram.v1";

  void embed(std::string_view /*text*/, std::span<const Token> toks, std::span<float, D> out) const noexcept {
    for (auto& v : out) v = 0.f;
    std::uint64_t prev = 0;
    for (const Token& t : toks) {
      if (is_stopword(t.hash)) continue;
      add(out, t.hash, 1.0f);
      if (prev) add(out, detail::mix64(prev * 0x9E3779B97F4A7C15ull ^ t.hash), 0.5f);  // bigram
      prev = t.hash;
    }
    simd::normalize(out.data(), D);
  }

 private:
  static RAG_ALWAYS_INLINE void add(std::span<float, D> out, std::uint64_t h, float w) noexcept {
    // Lemire fast-range: maps 32 hash bits into [0, D) without a division.
    const auto idx = static_cast<std::size_t>((std::uint64_t{static_cast<std::uint32_t>(h)} * D) >> 32);
    out[idx] += (h >> 63) ? -w : w;  // signed hashing keeps collisions unbiased
  }
};

static_assert(Embedder<HashingEmbedder<384>>);

}  // namespace rag
