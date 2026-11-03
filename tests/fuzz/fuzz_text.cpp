// fuzz_text — tokenizer, chunker, ingest, query and prompt assembly on
// arbitrary bytes, with invariants checked at every stage.
#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "fuzz_common.hpp"

using namespace rag;
using namespace rag::fuzz;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 3) return 0;
  // First bytes choose a chunker config and a split point; the rest is text.
  const std::uint32_t window = 1 + data[0] % 24;
  const std::uint32_t overlap = data[1] % window;  // always < window
  const std::string_view all(reinterpret_cast<const char*>(data + 3), size - 3);
  const std::size_t split = all.empty() ? 0 : data[2] % (all.size() + 1);
  const std::string_view doc = all.substr(0, split), query = all.substr(split);

  // Tokenizer: both APIs agree; tokens are in bounds, non-empty, ordered, and
  // never contain a separator byte.
  std::vector<Token> streamed;
  for_each_token(doc, [&](const Token& t) { streamed.push_back(t); });
  std::array<Token, 64> bounded;
  const std::size_t nb = tokenize(doc, bounded);
  FUZZ_CHECK(nb == std::min<std::size_t>(streamed.size(), bounded.size()));
  for (std::size_t i = 0; i < streamed.size(); ++i) {
    const Token& t = streamed[i];
    FUZZ_CHECK(t.length > 0 && std::size_t{t.offset} + t.length <= doc.size());
    FUZZ_CHECK(i == 0 || t.offset >= streamed[i - 1].offset + streamed[i - 1].length + 1);
    FUZZ_CHECK(t.hash == hash_term(doc.substr(t.offset, t.length)));
    if (i < nb) FUZZ_CHECK(bounded[i].offset == t.offset && bounded[i].hash == t.hash);
  }

  // Chunker: windows cover every token in order, with the configured overlap,
  // and byte ranges stay inside the text and contain their tokens.
  const ChunkerConfig cc{window, overlap};
  std::uint32_t prev_begin = 0, chunks = 0, last_end = 0;
  chunk_tokens(doc, streamed, cc, [&](const ChunkRange& r) {
    FUZZ_CHECK(r.token_begin < r.token_end && r.token_end <= streamed.size());
    FUZZ_CHECK(r.token_end - r.token_begin <= window);
    FUZZ_CHECK(chunks == 0 || r.token_begin == prev_begin + (window - overlap));
    FUZZ_CHECK(r.byte_begin == streamed[r.token_begin].offset);
    FUZZ_CHECK(r.byte_end <= doc.size());
    FUZZ_CHECK(r.byte_end >= streamed[r.token_end - 1].offset + streamed[r.token_end - 1].length);
    prev_begin = r.token_begin;
    last_end = r.token_end;
    ++chunks;
  });
  FUZZ_CHECK(streamed.empty() ? chunks == 0 : last_end == streamed.size());

  // Engine: ingest both halves, query with the other, in two dense modes.
  for (DenseMode mode : {DenseMode::Exact, DenseMode::IvfInt8}) {
    auto cfg = small_config(mode);
    cfg.chunker = cc;
    cfg.max_chunks = 64;  // small: also exercises the capacity limit
    Engine e(cfg);
    e.ingest(doc, 1);
    e.commit();
    e.ingest(query, 2);  // uncommitted: dense-only delta
    auto ctx = e.make_context();
    for (std::string_view q : {query, doc}) {
      const auto r = e.query(q, 16, ctx);
      const std::uint32_t n = e.size();
      for (const Hit& h : r.hits) FUZZ_CHECK(h.id < n && !std::isnan(h.score));
      for (const Hit& h : r.dense) FUZZ_CHECK(h.score > 0.f);  // dense_min_score = 0, strict
      const auto& p = r.prompt.prompt;
      FUZZ_CHECK(p.size() <= cfg.prompt_capacity);
      if (!p.empty()) FUZZ_CHECK(p.find("### Answer") != std::string_view::npos);
      FUZZ_CHECK(r.prompt.chunks_used <= r.hits.size());
    }
  }

  // Prompt builder with a tiny buffer: never writes past it.
  std::array<char, 96> buf;
  buf.fill('#');
  const std::array<Hit, 2> hits{{{1.f, 0}, {0.5f, 1}}};
  const auto r = build_prompt(query, hits, [&](std::uint32_t i) { return ChunkView{i ? doc : query, i, 0}; },
                              PromptParams{.max_context_tokens = 16, .system = "S"}, std::span(buf).first(64));
  FUZZ_CHECK(r.prompt.size() <= 64);
  FUZZ_CHECK(std::all_of(buf.begin() + 64, buf.end(), [](char c) { return c == '#'; }));
  return 0;
}
