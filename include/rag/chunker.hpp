// rag/chunker.hpp — sliding-window chunking over token boundaries.
//
// Chunks are byte ranges into the original text, so chunk text is a
// zero-copy std::string_view into the arena that owns the document.
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

#include "rag/tokenizer.hpp"

namespace rag {

struct ChunkerConfig {
  std::uint32_t window_tokens = 128;
  std::uint32_t overlap_tokens = 32;

  constexpr void validate() const {
    if (window_tokens == 0 || overlap_tokens >= window_tokens)
      throw std::invalid_argument("ChunkerConfig: need 0 <= overlap < window");
  }
};

constexpr bool is_space(char c) noexcept { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

struct ChunkRange {
  std::uint32_t token_begin, token_end;  // [begin, end) into the token array
  std::uint32_t byte_begin, byte_end;    // [begin, end) into the source text
};

// Calls emit(ChunkRange) for every window over `toks` (the tokens of `text`).
// Windows advance by (window - overlap) tokens and the last window ends at the
// last token. A chunk's bytes run from its first token up to the next token
// (or end of text), minus trailing whitespace, so trailing punctuation such as
// "." or ")" stays with the sentence it ends.
template <class F>
constexpr void chunk_tokens(std::string_view text, std::span<const Token> toks, const ChunkerConfig& cfg, F&& emit) {
  const auto n = static_cast<std::uint32_t>(toks.size());
  if (n == 0) return;
  const std::uint32_t stride = cfg.window_tokens - cfg.overlap_tokens;
  for (std::uint32_t begin = 0;; begin += stride) {
    const std::uint32_t end = begin + cfg.window_tokens < n ? begin + cfg.window_tokens : n;
    std::uint32_t byte_end = end < n ? toks[end].offset : static_cast<std::uint32_t>(text.size());
    const std::uint32_t min_end = toks[end - 1].offset + toks[end - 1].length;
    while (byte_end > min_end && is_space(text[byte_end - 1])) --byte_end;
    emit(ChunkRange{begin, end, toks[begin].offset, byte_end});
    if (end == n) break;
  }
}

}  // namespace rag
