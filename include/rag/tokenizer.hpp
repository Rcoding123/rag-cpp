// rag/tokenizer.hpp — allocation-free ASCII/UTF-8 word tokenizer with
// compile-time hashing and a compile-time stopword table.
//
// Latency notes:
//  * One 256-entry constexpr lookup table classifies + lowercases each byte:
//    a single L1-resident load per byte, no locale, no branches on ranges.
//  * Terms are identified by a 64-bit hash computed in the same pass, so no
//    std::string is ever built. Hashes are constexpr, which lets the stopword
//    set be a sorted std::array baked into .rodata at compile time.
//  * Bytes >= 0x80 are treated as word characters, so UTF-8 words survive
//    intact (no Unicode case folding — documented limitation).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "rag/common.hpp"

namespace rag {

struct Token {
  std::uint32_t offset;  // byte offset into the source text
  std::uint32_t length;  // byte length
  std::uint64_t hash;    // term id (lower-cased FNV-1a + avalanche)
};

namespace detail {

// 0 => separator; otherwise the lower-cased byte value.
inline constexpr std::array<std::uint8_t, 256> kCharMap = [] {
  std::array<std::uint8_t, 256> t{};
  for (int c = 0; c < 256; ++c) {
    if (c >= 'a' && c <= 'z') t[c] = static_cast<std::uint8_t>(c);
    else if (c >= 'A' && c <= 'Z') t[c] = static_cast<std::uint8_t>(c - 'A' + 'a');
    else if (c >= '0' && c <= '9') t[c] = static_cast<std::uint8_t>(c);
    else if (c >= 0x80) t[c] = static_cast<std::uint8_t>(c);
  }
  return t;
}();

inline constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ull;

// splitmix64 finaliser: FNV-1a's low bits are weak, and we index hash tables
// and embedding buckets with them.
constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27; x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

}  // namespace detail

// Hash of a single term, case-insensitive. constexpr: usable in static_assert.
constexpr std::uint64_t hash_term(std::string_view s) noexcept {
  std::uint64_t h = detail::kFnvOffset;
  for (char ch : s) {
    const auto c = detail::kCharMap[static_cast<std::uint8_t>(ch)];
    h = (h ^ c) * detail::kFnvPrime;
  }
  return detail::mix64(h);
}

// Streaming tokenizer: calls f(Token) for every word. No allocation.
template <class F>
RAG_ALWAYS_INLINE void for_each_token(std::string_view text, F&& f) noexcept(noexcept(f(Token{}))) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(text.data());
  const std::uint32_t n = static_cast<std::uint32_t>(text.size());
  std::uint32_t i = 0;
  while (i < n) {
    while (i < n && detail::kCharMap[p[i]] == 0) ++i;  // skip separators
    if (i >= n) break;
    const std::uint32_t start = i;
    std::uint64_t h = detail::kFnvOffset;
    std::uint8_t c;
    while (i < n && (c = detail::kCharMap[p[i]]) != 0) {
      h = (h ^ c) * detail::kFnvPrime;
      ++i;
    }
    f(Token{start, i - start, detail::mix64(h)});
  }
}

// Bounded tokenizer into a caller-owned buffer (hot path: query text).
// Returns the number of tokens written; silently truncates when out is full.
inline std::size_t tokenize(std::string_view text, std::span<Token> out) noexcept {
  std::size_t n = 0;
  const std::size_t cap = out.size();
  // Early exit once full: overlong queries are truncated, never overflow.
  const auto* p = reinterpret_cast<const std::uint8_t*>(text.data());
  const std::uint32_t len = static_cast<std::uint32_t>(text.size());
  std::uint32_t i = 0;
  while (i < len && n < cap) {
    while (i < len && detail::kCharMap[p[i]] == 0) ++i;
    if (i >= len) break;
    const std::uint32_t start = i;
    std::uint64_t h = detail::kFnvOffset;
    std::uint8_t c;
    while (i < len && (c = detail::kCharMap[p[i]]) != 0) { h = (h ^ c) * detail::kFnvPrime; ++i; }
    out[n++] = Token{start, i - start, detail::mix64(h)};
  }
  return n;
}

// ---------------------------------------------------------------------------
// Compile-time stopword set: sorted hashes in .rodata, binary-searched.
// ---------------------------------------------------------------------------
inline constexpr auto kStopwords = [] {
  constexpr std::string_view words[] = {
      "a",    "an",   "and",  "are",   "as",   "at",    "be",   "but",  "by",    "for",
      "from", "has",  "have", "how",   "i",    "if",    "in",   "into", "is",    "it",
      "its",  "of",   "on",   "or",    "that", "the",   "their","then", "there", "these",
      "they", "this", "to",   "was",   "we",   "were",  "what", "when", "where", "which",
      "who",  "why",  "will", "with",  "you",  "your",  "does", "do",   "can",   "about"};
  std::array<std::uint64_t, std::size(words)> h{};
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = hash_term(words[i]);
  std::ranges::sort(h);
  return h;
}();

constexpr bool is_stopword(std::uint64_t h) noexcept {
  return std::ranges::binary_search(kStopwords, h);
}

static_assert(hash_term("Order") == hash_term("oRDER"), "hashing must be case-insensitive");
static_assert(is_stopword(hash_term("The")) && !is_stopword(hash_term("orderbook")));

}  // namespace rag
