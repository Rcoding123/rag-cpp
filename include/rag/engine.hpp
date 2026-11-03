// rag/engine.hpp — the RAG pipeline: ingest -> chunk -> embed -> index,
// and query -> embed -> {dense ANN, BM25} -> RRF -> prompt.
//
//   INGEST (single writer, cold-ish path)          QUERY (N readers, hot path, zero alloc)
//   ─────────────────────────────────────          ──────────────────────────────────────────
//   text ─► Arena copy (stable string_view)        query ─► tokenize (stack buffer)
//        ─► tokenize + sliding-window chunk              ─► embed (stack, L1-resident)
//        ─► Embedder ─► VectorStore.append ──┐          ├─► dense: Exact | Int8+rerank | IVF + delta scan
//        ─► Bm25Builder.add                  │          └─► BM25: impact-scored postings
//   commit() ─► freeze BM25, (re)build IVF   │               ─► RRF fusion (stack)
//                                            │               ─► prompt into preallocated buffer
//            release-store(size) ────────────┴─► acquire-load(size): new chunks are visible
//                                                 to dense search immediately (NRT), and to
//                                                 BM25/IVF after the next commit().
//
// Thread-safety contract
//   * ingest(): exactly one writer thread.
//   * query(): any number of threads concurrently with ingest(); each thread
//     owns one QueryContext.
//   * commit(): must not overlap query() (it swaps the frozen BM25/IVF
//     structures). Production builds would publish immutable segments
//     through an atomic pointer with epoch-based reclamation; this keeps the
//     reference implementation honest and small.
#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "rag/arena.hpp"
#include "rag/bm25.hpp"
#include "rag/chunker.hpp"
#include "rag/common.hpp"
#include "rag/embedder.hpp"
#include "rag/fusion.hpp"
#include "rag/ivf_index.hpp"
#include "rag/prompt.hpp"
#include "rag/tokenizer.hpp"
#include "rag/topk.hpp"
#include "rag/vector_store.hpp"

namespace rag {

enum class DenseMode : std::uint8_t {
  Exact,       // fp32 brute force (SIMD, register-blocked) — ground truth
  Int8Rerank,  // int8 scan -> fp32 rerank of a shortlist: ~4x less memory traffic
  Ivf,         // IVF-Flat over the committed snapshot + exact scan of the uncommitted delta
};

struct EngineConfig {
  std::uint32_t max_chunks = 1u << 16;
  ChunkerConfig chunker{};
  Bm25Params bm25{};
  RrfParams rrf{};
  PromptParams prompt{};
  DenseMode dense_mode = DenseMode::Exact;
  bool hybrid = true;                    // fuse BM25 with dense; false => dense only
  std::uint32_t retrieve_depth = 32;     // per-retriever list depth fed to RRF
  std::uint32_t rerank_candidates = 128; // Int8Rerank shortlist size
  std::uint32_t nprobe = 8;              // Ivf lists probed per query
  IvfParams ivf{};
  std::size_t prompt_capacity = 32 * 1024;
};

// A generator consumes the assembled prompt (llama.cpp, vLLM over HTTP, ...).
template <class G>
concept Generator = requires(G& g, std::string_view prompt) { g.generate(prompt); };

template <Embedder E, std::size_t MaxK = 64>
class RagEngine {
 public:
  static constexpr std::size_t D = E::kDim;
  static constexpr std::size_t kMaxK = MaxK;
  static constexpr std::size_t kMaxQueryTokens = 256;
  static constexpr std::size_t kMaxRerank = 1024;

  struct ChunkMeta {
    const char* text;
    std::uint32_t len;
    std::uint32_t doc_id;
    std::uint32_t ordinal;
  };

  // All per-query scratch. Create one per query thread, reuse forever.
  class QueryContext {
   public:
    QueryContext(QueryContext&&) noexcept = default;
   private:
    friend class RagEngine;
    QueryContext(std::uint32_t max_chunks, std::size_t prompt_cap)
        : bm25(max_chunks), prompt_buf(make_aligned_array<char>(prompt_cap)), prompt_cap(prompt_cap) {}
    alignas(kCacheLine) std::array<float, D> qvec{};
    alignas(kCacheLine) std::array<std::int8_t, D> qvec_i8{};
    std::array<Token, kMaxQueryTokens> tokens{};
    std::array<std::uint64_t, kMaxQueryTokens> terms{};
    std::array<Hit, MaxK> dense{}, lexical{}, fused{};
    TopK<kMaxRerank> cand{};
    Bm25Scratch bm25;
    AlignedArray<char> prompt_buf;
    std::size_t prompt_cap;
  };

  struct QueryResult {
    std::span<const Hit> hits;     // fused (or dense-only) ranking, best first
    std::span<const Hit> dense;    // dense retriever's list (cosine scores)
    std::span<const Hit> lexical;  // BM25 list
    PromptResult prompt;
  };

  explicit RagEngine(EngineConfig cfg = {}, E embedder = {})
      : cfg_(validated(cfg)), embedder_(std::move(embedder)), store_(cfg_.max_chunks),
        meta_(make_aligned_array<ChunkMeta>(cfg_.max_chunks)) {}

  [[nodiscard]] QueryContext make_context() const { return QueryContext(cfg_.max_chunks, cfg_.prompt_capacity); }

  // ---------------------------------------------------------------- ingest
  // Single writer. Returns the number of chunks added (fewer than the
  // document produced if capacity ran out).
  std::uint32_t ingest(std::string_view text, std::uint32_t doc_id) {
    const std::string_view doc = arena_.copy(text);
    tokens_.clear();
    for_each_token(doc, [&](const Token& t) { tokens_.push_back(t); });

    std::uint32_t added = 0, ordinal = 0;
    bool full = false;
    chunk_tokens(doc, tokens_, cfg_.chunker, [&](const ChunkRange& r) {
      if (full) return;
      const std::uint32_t id = store_.size();
      if (id >= cfg_.max_chunks) { full = true; return; }
      const std::string_view ctext = doc.substr(r.byte_begin, r.byte_end - r.byte_begin);

      // Rebase token offsets so they are relative to the chunk text (Embedder contract).
      chunk_toks_.clear();
      terms_.clear();
      for (std::uint32_t i = r.token_begin; i < r.token_end; ++i) {
        Token t = tokens_[i];
        t.offset -= r.byte_begin;
        chunk_toks_.push_back(t);
        if (!is_stopword(t.hash)) terms_.push_back(t.hash);
      }
      embedder_.embed(ctext, chunk_toks_, std::span<float, D>(vec_));
      builder_.add(id, terms_);
      meta_[id] = {ctext.data(), static_cast<std::uint32_t>(ctext.size()), doc_id, ordinal++};
      store_.append(vec_);  // release-publishes meta_[id] and the row together
      ++added;
    });
    return added;
  }

  // Freeze lexical index (and IVF) over everything ingested so far.
  // Must not run concurrently with query().
  void commit() {
    const std::uint32_t n = store_.size();
    bm25_ = builder_.freeze(cfg_.bm25);
    if (cfg_.dense_mode == DenseMode::Ivf) ivf_.build(store_, n, cfg_.ivf);
    committed_ = n;
  }

  // ----------------------------------------------------------------- query
  // Hot path: no heap allocation, no locks, no syscalls.
  QueryResult query(std::string_view q, std::uint32_t k, QueryContext& ctx) const noexcept {
    k = std::min<std::uint32_t>(k, MaxK);
    const std::size_t ntok = tokenize(q, ctx.tokens);
    std::size_t nterms = 0;
    for (std::size_t i = 0; i < ntok; ++i)
      if (!is_stopword(ctx.tokens[i].hash)) ctx.terms[nterms++] = ctx.tokens[i].hash;

    embedder_.embed(q, std::span<const Token>(ctx.tokens.data(), ntok), std::span<float, D>(ctx.qvec));
    const float* qv = ctx.qvec.data();
    const std::uint32_t live = store_.size();  // acquire: sees every published chunk
    const std::uint32_t depth = cfg_.hybrid ? cfg_.retrieve_depth : k;

    // Dense retrieval. An all-zero query vector carries no signal: skip it.
    TopK<MaxK> dense(depth);
    if (simd::dot(qv, qv, D) > 0.f) [[likely]] {
      switch (cfg_.dense_mode) {
        case DenseMode::Exact:
          store_.search_exact(qv, 0, live, dense);
          break;
        case DenseMode::Int8Rerank:
          simd::quantize_i8(qv, ctx.qvec_i8.data(), D);
          ctx.cand.reset(std::max(cfg_.rerank_candidates, depth));
          store_.search_two_stage(qv, ctx.qvec_i8.data(), 0, live, ctx.cand, dense);
          break;
        case DenseMode::Ivf: {
          const std::uint32_t base = std::min(ivf_.covered(), live);
          ivf_.search(qv, cfg_.nprobe, dense);
          store_.search_exact(qv, base, live, dense);  // uncommitted delta segment
          break;
        }
      }
    }
    const auto nd = copy_out(dense.sorted(), ctx.dense);

    QueryResult res;
    res.dense = {ctx.dense.data(), nd};
    if (cfg_.hybrid) {
      TopK<MaxK> lex(depth);
      bm25_.search(std::span<const std::uint64_t>(ctx.terms.data(), nterms), ctx.bm25, lex);
      const auto nl = copy_out(lex.sorted(), ctx.lexical);
      res.lexical = {ctx.lexical.data(), nl};
      TopK<MaxK> fused(k);
      rrf_fuse(res.dense, res.lexical, cfg_.rrf, fused);
      res.hits = {ctx.fused.data(), copy_out(fused.sorted(), ctx.fused)};
    } else {
      res.hits = res.dense.first(std::min<std::size_t>(k, nd));
    }

    res.prompt = build_prompt(q, res.hits, [this](std::uint32_t id) { return chunk(id); }, cfg_.prompt,
                              std::span<char>(ctx.prompt_buf.get(), ctx.prompt_cap));
    return res;
  }

  // Retrieve, then hand the prompt to a generator.
  template <Generator G>
  QueryResult answer(std::string_view q, std::uint32_t k, QueryContext& ctx, G& gen) const {
    auto r = query(q, k, ctx);
    gen.generate(r.prompt.prompt);
    return r;
  }

  [[nodiscard]] ChunkView chunk(std::uint32_t id) const noexcept {
    const ChunkMeta& m = meta_[id];
    return {{m.text, m.len}, m.doc_id, m.ordinal};
  }
  [[nodiscard]] std::uint32_t size() const noexcept { return store_.size(); }
  [[nodiscard]] std::uint32_t committed() const noexcept { return committed_; }
  [[nodiscard]] const VectorStore<D>& store() const noexcept { return store_; }
  [[nodiscard]] const Bm25Index& lexical_index() const noexcept { return bm25_; }
  [[nodiscard]] const IvfIndex<D>& ivf() const noexcept { return ivf_; }
  [[nodiscard]] const EngineConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] std::size_t arena_bytes() const noexcept { return arena_.bytes_used(); }

 private:
  static EngineConfig validated(EngineConfig c) {
    c.chunker.validate();
    if (c.max_chunks == 0) throw std::invalid_argument("max_chunks must be > 0");
    if (c.retrieve_depth == 0 || c.retrieve_depth > MaxK) throw std::invalid_argument("retrieve_depth must be in [1, MaxK]");
    if (c.rerank_candidates > kMaxRerank) throw std::invalid_argument("rerank_candidates > kMaxRerank");
    if (c.nprobe == 0) throw std::invalid_argument("nprobe must be > 0");
    return c;
  }

  static std::size_t copy_out(std::span<const Hit> src, std::array<Hit, MaxK>& dst) noexcept {
    const std::size_t n = std::min(src.size(), MaxK);
    std::copy_n(src.begin(), n, dst.begin());
    return n;
  }

  EngineConfig cfg_;
  E embedder_;
  VectorStore<D> store_;
  AlignedArray<ChunkMeta> meta_;
  Bm25Index bm25_;
  IvfIndex<D> ivf_;
  std::uint32_t committed_ = 0;

  // Writer-only state (ingest). Vectors are reused, so steady-state ingest
  // stops allocating once they reach the largest document's size.
  Arena arena_;
  Bm25Builder builder_;
  std::vector<Token> tokens_, chunk_toks_;
  std::vector<std::uint64_t> terms_;
  alignas(kCacheLine) std::array<float, D> vec_{};
};

}  // namespace rag
