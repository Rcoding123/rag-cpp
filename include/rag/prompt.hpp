// rag/prompt.hpp — context assembly into a caller-owned, fixed-size buffer.
//
// The prompt is written straight into a preallocated char buffer: no
// std::string, no ostringstream, no reallocation. Space for the question
// and footer is reserved up front, so a long context can never push the
// question out of the prompt. Chunks are packed greedily in rank order under
// a token budget; a chunk that does not fit is skipped and smaller,
// lower-ranked chunks may still be used.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include "rag/topk.hpp"

namespace rag {

struct ChunkView {
  std::string_view text;
  std::uint32_t doc_id;
  std::uint32_t ordinal;  // chunk index within the document
};

struct PromptParams {
  std::uint32_t max_context_tokens = 1024;
  std::string_view system =
      "You are a precise assistant. Answer using only the context below. "
      "Cite sources as [n]. If the context is insufficient, say so.";
};

struct PromptResult {
  std::string_view prompt;
  std::uint32_t chunks_used = 0;
  std::uint32_t context_tokens = 0;  // estimated
  bool truncated = false;            // some chunk(s) or the whole prompt did not fit
};

// ~4 bytes/token for English BPE vocabularies; swap for a real tokenizer count
// if the generator's limit is tight.
constexpr std::uint32_t estimate_tokens(std::size_t bytes) noexcept {
  return static_cast<std::uint32_t>((bytes + 3) / 4);
}

namespace detail {
class BufWriter {
 public:
  constexpr BufWriter(char* p, std::size_t cap) noexcept : p_(p), cap_(cap) {}
  void put(std::string_view s) noexcept { std::memcpy(p_ + len_, s.data(), s.size()); len_ += s.size(); }
  void put_u32(std::uint32_t v) noexcept { len_ = static_cast<std::size_t>(std::to_chars(p_ + len_, p_ + cap_, v).ptr - p_); }
  [[nodiscard]] std::size_t len() const noexcept { return len_; }
  [[nodiscard]] std::string_view view() const noexcept { return {p_, len_}; }
 private:
  char* p_;
  std::size_t cap_;
  std::size_t len_ = 0;
};
constexpr std::size_t digits(std::uint32_t v) noexcept { std::size_t d = 1; while (v >= 10) { v /= 10; ++d; } return d; }
}  // namespace detail

template <class Lookup>  // Lookup: (std::uint32_t chunk_id) -> ChunkView
PromptResult build_prompt(std::string_view question, std::span<const Hit> hits, Lookup&& lookup,
                          const PromptParams& p, std::span<char> buf) noexcept {
  using namespace std::string_view_literals;
  constexpr auto kCtx = "\n\n### Context\n"sv, kQ = "### Question\n"sv, kA = "\n\n### Answer\n"sv;
  PromptResult r;
  const std::size_t fixed = p.system.size() + kCtx.size() + kQ.size() + question.size() + kA.size();
  if (fixed > buf.size()) { r.truncated = true; return r; }

  detail::BufWriter w(buf.data(), buf.size());
  w.put(p.system);
  w.put(kCtx);
  const std::size_t budget_bytes = buf.size() - fixed;  // reserved for chunk entries
  std::size_t used_bytes = 0;
  for (const Hit& h : hits) {
    const ChunkView c = lookup(h.id);
    const std::uint32_t n = r.chunks_used + 1;
    // "[n] (doc D, chunk C)\n<text>\n\n"
    const std::size_t entry = 1 + detail::digits(n) + 7 + detail::digits(c.doc_id) + 8 +
                              detail::digits(c.ordinal) + 2 + c.text.size() + 2;
    const std::uint32_t tok = estimate_tokens(c.text.size());
    if (used_bytes + entry > budget_bytes || r.context_tokens + tok > p.max_context_tokens) {
      r.truncated = true;
      continue;
    }
    w.put("["sv); w.put_u32(n); w.put("] (doc "sv); w.put_u32(c.doc_id); w.put(", chunk "sv);
    w.put_u32(c.ordinal); w.put(")\n"sv); w.put(c.text); w.put("\n\n"sv);
    used_bytes += entry;
    r.context_tokens += tok;
    r.chunks_used = n;
  }
  w.put(kQ);
  w.put(question);
  w.put(kA);
  r.prompt = w.view();
  return r;
}

}  // namespace rag
