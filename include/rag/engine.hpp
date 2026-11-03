// rag/engine.hpp — the RAG pipeline: ingest -> chunk -> embed -> index,
// and query -> embed -> {dense ANN, BM25} -> RRF -> prompt.
//
//   INGEST (single writer, cold-ish path)          QUERY (N readers, hot path, zero alloc)
//   ─────────────────────────────────────          ──────────────────────────────────────────
//   text ─► Arena copy (stable string_view)        query ─► tokenize (stack buffer)
//        ─► tokenize + sliding-window chunk              ─► embed (stack, L1-resident)
//        ─► Embedder ─► VectorStore.append ──┐          ├─► dense: Exact | Int8+rerank | IVF + delta scan
//        ─► Bm25Builder.add                  │          └─► BM25: impact-scored postings
//   commit() ─► build Snapshot{BM25, IVF}    │               ─► RRF fusion (stack)
//            ─► atomic exchange + EBR retire │               ─► prompt into preallocated buffer
//            release-store(size) ────────────┴─► acquire-load(size): new chunks are visible
//                                                 to dense search immediately (NRT), and to
//                                                 BM25/IVF after the next commit().
//
// Thread-safety contract
//   * Writer side, ingest() and commit(): one thread at a time (externally serialised).
//   * query(): any number of threads, concurrently with ingest() AND commit().
//     Each thread owns one QueryContext (at most EpochDomain::kMaxReaders live).
//   * commit() builds a new immutable Snapshot off to the side, publishes it
//     with one atomic exchange, and retires the old one through epoch-based
//     reclamation. Queries never block, and never see a half-built index.
//   * QueryContexts must be destroyed before the engine.
#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <filesystem>
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
#include "rag/epoch.hpp"
#include "rag/fusion.hpp"
#include "rag/io.hpp"
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
  IvfInt8,     // IVF-SQ8 shortlist -> fp32 rerank, + exact scan of the uncommitted delta
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
  std::uint32_t rerank_candidates = 128; // Int8Rerank / IvfInt8 shortlist size
  std::uint32_t nprobe = 8;              // Ivf / IvfInt8 lists probed per query
  IvfParams ivf{};
  std::size_t prompt_capacity = 32 * 1024;
};

// A generator consumes the assembled prompt (llama.cpp, vLLM over HTTP, ...).
template <class G>
concept Generator = requires(G& g, std::string_view prompt) { g.generate(prompt); };

template <Embedder E, std::size_t MaxK = 64>
class RagEngine {
 public:
  // Everything commit() publishes. Immutable once reachable from snap_.
  struct Snapshot {
    Bm25Index bm25;
    IvfIndex<E::kDim> ivf;
    std::uint32_t covered = 0;  // chunks [0, covered) are in bm25 / ivf
    std::uint64_t version = 0;  // number of commits that produced it
  };

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
    QueryContext(EpochDomain::Reader r, std::uint32_t max_chunks, std::size_t prompt_cap)
        : reader(std::move(r)), bm25(max_chunks), prompt_buf(make_aligned_array<char>(prompt_cap)),
          prompt_cap(prompt_cap) {}
    EpochDomain::Reader reader;  // this thread's reclamation slot
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
    std::uint64_t snapshot_version = 0;  // which commit the lexical/IVF results came from
  };

  // Pins the current snapshot for as long as it lives, e.g. to run several
  // reads against one consistent index version. Nestable with query() on the
  // same context. Keep it short-lived: it holds back reclamation.
  class PinnedSnapshot {
   public:
    PinnedSnapshot(const RagEngine& e, QueryContext& ctx) noexcept
        : guard_(e.epochs_, *ctx.reader.slot()), snap_(e.snap_.load(std::memory_order_seq_cst)) {}
    const Snapshot& operator*() const noexcept { return *snap_; }
    const Snapshot* operator->() const noexcept { return snap_; }

   private:
    EpochDomain::Guard guard_;  // must be constructed before snap_ is loaded
    const Snapshot* snap_;
  };

  explicit RagEngine(EngineConfig cfg = {}, E embedder = {})
      : cfg_(validated(cfg)), embedder_(std::move(embedder)), store_(cfg_.max_chunks),
        meta_(make_aligned_array<ChunkMeta>(cfg_.max_chunks, Init::Lazy)), snap_(new Snapshot{}) {}

  RagEngine(const RagEngine&) = delete;
  RagEngine& operator=(const RagEngine&) = delete;

  ~RagEngine() {
    delete snap_.load(std::memory_order_relaxed);
    for (auto& r : retired_) delete r.snap;
  }

  // Cold path: claims a reclamation slot and allocates per-thread scratch.
  [[nodiscard]] QueryContext make_context() const {
    return QueryContext(epochs_.register_reader(), cfg_.max_chunks, cfg_.prompt_capacity);
  }
  [[nodiscard]] PinnedSnapshot pin(QueryContext& ctx) const noexcept { return PinnedSnapshot(*this, ctx); }

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

  // Freeze lexical index (and IVF) over everything ingested so far and
  // publish it atomically. Safe to run while queries are in flight: the
  // expensive build happens before the swap, so the readers' only cost is
  // that their next query sees the new snapshot.
  void commit() {
    const std::uint32_t n = store_.size();
    const Snapshot* prev = snap_.load(std::memory_order_relaxed);  // writer owns snap_ stores
    auto next = std::make_unique<Snapshot>();
    next->bm25 = builder_.freeze(cfg_.bm25);
    if (uses_ivf(cfg_.dense_mode)) next->ivf.build(store_, n, effective_ivf(cfg_));
    next->covered = n;
    next->version = prev->version + 1;

    const Snapshot* old = snap_.exchange(next.release(), std::memory_order_seq_cst);
    retired_.push_back({old, epochs_.advance()});
    committed_.store(n, std::memory_order_release);
    reclaim();
  }

  // ------------------------------------------------------------ persistence
  // Writer side (same thread as ingest/commit). Queries may keep running.
  // Saves chunk text + metadata, fp32 vectors, BM25 builder state, and the
  // current committed snapshot (frozen BM25 + IVF) with the parameters it
  // was built with. load() reuses that snapshot when the new EngineConfig
  // asks for the same index parameters and it covers every chunk; otherwise
  // it rebuilds from the raw data. Either way the result is identical.
  struct LoadReport {
    bool reused_indexes = false;  // false => BM25 / IVF were rebuilt on load
  };

  void save(const std::filesystem::path& path) const {
    const std::uint32_t n = store_.size();
    io::Writer w(path);
    std::uint64_t text_bytes = 0;
    for (std::uint32_t i = 0; i < n; ++i) text_bytes += meta_[i].len;
    FileHeader hdr;  // magic comes from the default member initializer
    hdr.version = io::kFormatVersion;
    hdr.dim = static_cast<std::uint32_t>(D);
    hdr.embedder_tag = embedder_tag<E>();
    hdr.n_chunks = n;
    hdr.reserved = 0;
    hdr.text_bytes = text_bytes;
    w.pod(hdr);
    std::uint64_t off = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      const ChunkMeta& m = meta_[i];
      w.pod(ChunkRecord{off, m.len, m.doc_id, m.ordinal, 0});
      off += m.len;
    }
    for (std::uint32_t i = 0; i < n; ++i) w.bytes(meta_[i].text, meta_[i].len);
    for (std::uint32_t i = 0; i < n; ++i) w.bytes(store_.row(i), D * sizeof(float));
    builder_.save(w);

    // The writer thread is the only one that replaces or frees snapshots,
    // so it can read the current one without pinning.
    const Snapshot& snap = *snap_.load(std::memory_order_acquire);
    IndexRecord rec{};
    rec.k1 = cfg_.bm25.k1;
    rec.b = cfg_.bm25.b;
    rec.has_ivf = snap.ivf.nlist() > 0;
    const IvfParams ip = effective_ivf(cfg_);
    rec.nlist = ip.nlist;
    rec.train_iters = ip.train_iters;
    rec.max_train = ip.max_train_points_per_list;
    rec.storage = static_cast<std::uint8_t>(ip.storage);
    rec.seed = ip.seed;
    rec.covered = snap.covered;
    w.pod(rec);
    snap.bm25.save(w);
    snap.ivf.save(w);
    w.commit();
  }

  // Throws io::FormatError for a damaged file, std::invalid_argument for a
  // file that doesn't fit this engine type or configuration.
  [[nodiscard]] static std::unique_ptr<RagEngine> load(const std::filesystem::path& path, EngineConfig cfg = {},
                                                       E embedder = {}, LoadReport* report = nullptr) {
    io::Reader r(path);
    const auto h = r.pod<FileHeader>();
    if (std::memcmp(h.magic, io::kMagic, sizeof h.magic) != 0) throw io::FormatError("not a rag-cpp index file");
    if (h.version != io::kFormatVersion) throw io::FormatError("unsupported index format version");
    if (h.dim != D) throw std::invalid_argument("index dimension does not match the engine's embedder");
    if (h.embedder_tag != embedder_tag<E>()) throw std::invalid_argument("index was built with a different embedder");
    if (h.n_chunks > cfg.max_chunks) throw std::invalid_argument("index has more chunks than cfg.max_chunks");

    auto e = std::make_unique<RagEngine>(cfg, std::move(embedder));
    std::vector<ChunkRecord> recs(h.n_chunks);
    for (auto& rec : recs) {
      rec = r.pod<ChunkRecord>();
      if (rec.text_offset + rec.len > h.text_bytes) throw io::FormatError("chunk text out of range");
    }
    if (h.text_bytes > r.remaining()) throw io::FormatError("truncated text section");
    auto* text = static_cast<char*>(e->arena_.allocate(static_cast<std::size_t>(h.text_bytes ? h.text_bytes : 1), 1));
    r.bytes(text, static_cast<std::size_t>(h.text_bytes));  // one block for all chunk text
    alignas(kCacheLine) std::array<float, D> row;
    for (std::uint32_t i = 0; i < h.n_chunks; ++i) {
      r.bytes(row.data(), D * sizeof(float));
      e->meta_[i] = {text + recs[i].text_offset, recs[i].len, recs[i].doc_id, recs[i].ordinal};
      e->store_.append(row);
    }
    e->builder_.load(r);
    if (e->builder_.num_docs() != h.n_chunks) throw io::FormatError("lexical index does not match chunk count");

    const auto rec = r.pod<IndexRecord>();
    if (rec.covered > h.n_chunks) throw io::FormatError("saved snapshot covers more chunks than the file holds");
    auto snap = std::make_unique<Snapshot>();
    snap->bm25.load(r);
    snap->ivf.load(r, rec.covered);
    if (snap->bm25.num_docs() != rec.covered || snap->ivf.covered() > rec.covered)
      throw io::FormatError("saved snapshot is inconsistent");
    r.finish();  // checksum verified before anything is published

    const IvfParams ip = effective_ivf(e->cfg_);
    const bool bm25_ok = rec.covered == h.n_chunks && rec.k1 == e->cfg_.bm25.k1 && rec.b == e->cfg_.bm25.b;
    const bool ivf_ok = !uses_ivf(e->cfg_.dense_mode) ||
                        (rec.has_ivf && snap->ivf.covered() == h.n_chunks && rec.nlist == ip.nlist &&
                         rec.train_iters == ip.train_iters && rec.max_train == ip.max_train_points_per_list &&
                         rec.seed == ip.seed && rec.storage == static_cast<std::uint8_t>(ip.storage));
    const bool reuse = bm25_ok && ivf_ok;
    if (reuse) {
      if (!uses_ivf(e->cfg_.dense_mode)) snap->ivf = IvfIndex<D>{};  // not needed in this mode: free it
      snap->covered = h.n_chunks;
      snap->version = 1;
      delete e->snap_.exchange(snap.release(), std::memory_order_seq_cst);  // no readers exist yet
      e->committed_.store(h.n_chunks, std::memory_order_release);
    } else {
      e->commit();
    }
    if (report) report->reused_indexes = reuse;
    return e;
  }

  // Writer: free retired snapshots no reader can still see. commit() calls
  // this itself; call it directly to release memory without committing.
  // Returns the number of snapshots still waiting for readers to move on.
  std::size_t reclaim() {
    std::erase_if(retired_, [&](const Retired& r) {
      if (!epochs_.quiescent_since(r.epoch)) return false;
      delete r.snap;
      return true;
    });
    return retired_.size();
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
    const PinnedSnapshot snap(*this, ctx);    // EBR pin: snapshot can't be freed under us
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
          const std::uint32_t base = std::min(snap->ivf.covered(), live);
          snap->ivf.search(qv, cfg_.nprobe, dense);
          store_.search_exact(qv, base, live, dense);  // uncommitted delta segment
          break;
        }
        case DenseMode::IvfInt8: {
          const std::uint32_t base = std::min(snap->ivf.covered(), live);
          simd::quantize_i8(qv, ctx.qvec_i8.data(), D);
          ctx.cand.reset(std::max(cfg_.rerank_candidates, depth));
          snap->ivf.search_int8(qv, ctx.qvec_i8.data(), cfg_.nprobe, ctx.cand);
          for (const Hit& h : ctx.cand.sorted()) dense.push(simd::dot(qv, store_.row(h.id), D), h.id);  // fp32 rerank
          store_.search_exact(qv, base, live, dense);
          break;
        }
      }
    }
    const auto nd = copy_out(dense.sorted(), ctx.dense);

    QueryResult res;
    res.snapshot_version = snap->version;
    res.dense = {ctx.dense.data(), nd};
    if (cfg_.hybrid) {
      TopK<MaxK> lex(depth);
      snap->bm25.search(std::span<const std::uint64_t>(ctx.terms.data(), nterms), ctx.bm25, lex);
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
  [[nodiscard]] std::uint32_t committed() const noexcept { return committed_.load(std::memory_order_acquire); }
  [[nodiscard]] const VectorStore<D>& store() const noexcept { return store_; }
  [[nodiscard]] std::size_t retired_snapshots() const noexcept { return retired_.size(); }  // writer thread
  [[nodiscard]] std::size_t live_readers() const noexcept { return epochs_.readers(); }
  [[nodiscard]] const EngineConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] std::size_t arena_bytes() const noexcept { return arena_.bytes_used(); }

 private:
  static constexpr bool uses_ivf(DenseMode m) noexcept { return m == DenseMode::Ivf || m == DenseMode::IvfInt8; }
  static IvfParams effective_ivf(const EngineConfig& c) noexcept {
    IvfParams ip = c.ivf;
    ip.storage = c.dense_mode == DenseMode::IvfInt8 ? IvfStorage::Int8 : IvfStorage::Fp32;
    return ip;
  }

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
  struct Retired {
    const Snapshot* snap;
    std::uint64_t epoch;
  };

  // On-disk layout (see save()).
  struct FileHeader {
    char magic[8] = {'R', 'A', 'G', 'C', 'P', 'P', 'I', 'X'};
    std::uint32_t version;
    std::uint32_t dim;
    std::uint64_t embedder_tag;
    std::uint32_t n_chunks;
    std::uint32_t reserved;
    std::uint64_t text_bytes;
  };
  struct ChunkRecord {
    std::uint64_t text_offset;
    std::uint32_t len, doc_id, ordinal, reserved;
  };
  struct IndexRecord {  // parameters the saved snapshot was built with
    float k1, b;
    std::uint32_t nlist, train_iters, max_train;
    std::uint8_t storage, has_ivf, reserved8[2];
    std::uint64_t seed;
    std::uint32_t covered, reserved32;
  };
  static_assert(sizeof(FileHeader) == 40 && sizeof(ChunkRecord) == 24 && sizeof(IndexRecord) == 40,
                "file layout must not depend on padding");

  // Read by every query: its own line, never written by queries.
  alignas(kCacheLine) std::atomic<const Snapshot*> snap_;
  std::atomic<std::uint32_t> committed_{0};
  mutable EpochDomain epochs_;     // slots are written by readers (their own lines only)
  std::vector<Retired> retired_;  // writer only

  // Writer-only state (ingest). Vectors are reused, so steady-state ingest
  // stops allocating once they reach the largest document's size.
  Arena arena_;
  Bm25Builder builder_;
  std::vector<Token> tokens_, chunk_toks_;
  std::vector<std::uint64_t> terms_;
  alignas(kCacheLine) std::array<float, D> vec_{};
};

}  // namespace rag
