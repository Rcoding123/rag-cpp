// demo.cpp — end-to-end usage: ingest, commit, query, hand prompt to a generator.
#include <cstdio>

#include "rag.hpp"

namespace {
// Stand-in for an LLM client (llama.cpp, vLLM/TGI over HTTP, ...). Anything
// with generate(std::string_view) satisfies rag::Generator.
struct PrintGenerator {
  void generate(std::string_view prompt) { std::printf("----- prompt sent to generator -----\n%.*s\n", int(prompt.size()), prompt.data()); }
};
}  // namespace

int main() {
  rag::EngineConfig cfg;
  cfg.max_chunks = 4096;
  cfg.chunker = {48, 12};
  cfg.dense_mode = rag::DenseMode::Int8Rerank;
  rag::RagEngine<rag::HashingEmbedder<384>> engine(cfg);

  const char* docs[] = {
      "Order flow imbalance (OFI) is the net change in queued size at the best bid minus the best ask. "
      "Positive OFI indicates buying pressure and tends to precede upticks in the mid price over short horizons.",
      "A feed handler normalises exchange multicast packets into book updates. Gaps in sequence numbers "
      "trigger a snapshot recovery from the replay channel before the book is marked healthy again.",
      "VWAP algorithms split a parent order according to the expected intraday volume profile, trading "
      "more during the open and close when liquidity is deepest.",
      "Pre-trade risk checks validate price collars, maximum order quantity and credit limits in the "
      "order gateway, typically in under a microsecond using precomputed limits.",
  };
  for (std::uint32_t i = 0; i < std::size(docs); ++i) engine.ingest(docs[i], i);
  engine.commit();

  auto ctx = engine.make_context();
  PrintGenerator gen;
  const auto r = engine.answer("What happens when the feed handler detects a sequence gap?", 3, ctx, gen);
  std::printf("----- hits -----\n");
  for (const auto& h : r.hits) {
    const auto c = engine.chunk(h.id);
    std::printf("  rrf=%.4f doc=%u chunk=%u  %.60s...\n", h.score, c.doc_id, c.ordinal, c.text.data());
  }
}
