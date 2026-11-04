// rag/prompt.hpp — context assembly into a caller-owned, fixed-size buffer.
//
// The prompt is written straight into a preallocated char buffer: no
// std::string, no ostringstream, no reallocation. Space for the question
// and footer is reserved up front, so a long context can never push the
// question out of the prompt. Chunks are packed greedily in rank order under
// a token budget; a chunk that does not fit is skipped and smaller,
// lower-ranked chunks may still be used.
#pragma once

#include <algorithm>
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
  std::uint32_t ordinal;        // chunk index within the document
  std::string_view source{};    // e.g. "report.pdf"; empty if not given at ingest
  std::uint32_t page_first = 0; // 1-based pages the chunk spans; 0 = unknown
  std::uint32_t page_last = 0;
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
  // Writes are clamped to the buffer. Callers size their output up front, so
  // the clamp never truncates in practice; it guarantees that a budgeting
  // mistake can never become a buffer overflow.
  void put(std::string_view s) noexcept {
    const std::size_t n = std::min(s.size(), cap_ - len_);
    if (n == 0) return;  // an empty view's data() may be null: memcpy(dst, nullptr, 0) is UB
    std::memcpy(p_ + len_, s.data(), n);
    len_ += n;
  }
  void put_u32(std::uint32_t v) noexcept { len_ = static_cast<std::size_t>(std::to_chars(p_ + len_, p_ + cap_, v).ptr - p_); }
  [[nodiscard]] std::size_t len() const noexcept { return len_; }
  [[nodiscard]] std::string_view view() const noexcept { return {p_, len_}; }
 private:
  char* p_;
  std::size_t cap_;
  std::size_t len_ = 0;
};
// Longest source name a citation shows (longer names are cut at a UTF-8
// boundary and marked with "..."), so one odd file name can't eat the budget.
inline constexpr std::size_t kMaxCitedSource = 120;
inline constexpr std::size_t kMaxCitation = kMaxCitedSource + 64;

// Writes "[n] (<where>)\n" into out, where <where> is
//   "doc D, chunk C"          no source, no pages (plain text ingest)
//   "doc D, p. P" / "pp. P-Q" no source, pages
//   "SOURCE, chunk C"         source, no pages
//   "SOURCE, p. P" / "pp. P-Q" source and pages (e.g. PDFs)
// and returns its length. out must hold kMaxCitation bytes.
inline std::size_t format_citation(char* out, std::uint32_t n, const ChunkView& c) noexcept {
  BufWriter w(out, kMaxCitation);
  using namespace std::string_view_literals;
  w.put("["sv);
  w.put_u32(n);
  w.put("] ("sv);
  if (c.source.empty()) {
    w.put("doc "sv);
    w.put_u32(c.doc_id);
  } else if (c.source.size() <= kMaxCitedSource) {
    w.put(c.source);
  } else {
    std::size_t cut = kMaxCitedSource - 3;
    while (cut > 0 && (static_cast<unsigned char>(c.source[cut]) & 0xC0) == 0x80) --cut;  // don't split a UTF-8 sequence
    w.put(c.source.substr(0, cut));
    w.put("..."sv);
  }
  if (c.page_first == 0) {
    w.put(", chunk "sv);
    w.put_u32(c.ordinal);
  } else if (c.page_first == c.page_last) {
    w.put(", p. "sv);
    w.put_u32(c.page_first);
  } else {
    w.put(", pp. "sv);
    w.put_u32(c.page_first);
    w.put("-"sv);
    w.put_u32(c.page_last);
  }
  w.put(")\n"sv);
  return w.len();
}
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
    // "<citation>\n<text>\n\n"
    char cite[detail::kMaxCitation];
    const std::size_t cite_len = detail::format_citation(cite, n, c);
    const std::size_t entry = cite_len + c.text.size() + 2;
    const std::uint32_t tok = estimate_tokens(c.text.size());
    if (used_bytes + entry > budget_bytes || r.context_tokens + tok > p.max_context_tokens) {
      r.truncated = true;
      continue;
    }
    w.put({cite, cite_len}); w.put(c.text); w.put("\n\n"sv);
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
