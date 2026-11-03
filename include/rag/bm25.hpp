// rag/bm25.hpp — lexical retrieval: BM25 over a frozen, impact-scored
// inverted index.
//
// Build (cold): Bm25Builder collects (term, doc, tf) triples; freeze() sorts
// them and precomputes, per posting, the full BM25 contribution
//     w = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
// ("impact scoring"). Query-time work per posting is then one gather-add —
// no division, no document-length lookup.
//
// Layout:
//   * term dictionary: open-addressing table (power-of-two capacity, linear
//     probing, load factor <= 0.5) of {hash, offset, len} — one or two
//     cache lines per lookup.
//   * postings: SoA arrays doc_[] (u32) and weight_[] (f32), contiguous per
//     term and sorted by doc id, so accumulation streams sequentially.
// Query scratch (dense accumulator + touched list) lives in Bm25Scratch,
// allocated once per query thread; queries never allocate.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "rag/common.hpp"
#include "rag/topk.hpp"

namespace rag {

struct Bm25Params {
  float k1 = 1.2f;
  float b = 0.75f;
};

// Per-thread query scratch. acc[] is all zero between queries (invariant).
struct Bm25Scratch {
  explicit Bm25Scratch(std::uint32_t max_docs)
      : acc(make_aligned_array<float>(max_docs)), touched(make_aligned_array<std::uint32_t>(max_docs)), capacity(max_docs) {}
  AlignedArray<float> acc;
  AlignedArray<std::uint32_t> touched;
  std::uint32_t capacity;
};

class Bm25Index {
 public:
  static constexpr std::size_t kMaxQueryTerms = 256;

  // `terms`: query term hashes (duplicates allowed; each distinct term counts once).
  // Documents with id >= scratch.capacity are ignored (cannot happen when the
  // scratch was sized from the engine's max_chunks).
  template <class TK>
  void search(std::span<const std::uint64_t> terms, Bm25Scratch& sc, TK& out) const noexcept {
    if (num_docs_ == 0 || terms.empty()) return;
    std::array<std::uint64_t, kMaxQueryTerms> uniq;
    const std::size_t nt = std::min(terms.size(), kMaxQueryTerms);
    std::copy_n(terms.begin(), nt, uniq.begin());
    std::sort(uniq.begin(), uniq.begin() + nt);
    const std::size_t nu = static_cast<std::size_t>(std::unique(uniq.begin(), uniq.begin() + nt) - uniq.begin());

    float* RAG_RESTRICT acc = sc.acc.get();
    std::uint32_t* RAG_RESTRICT touched = sc.touched.get();
    std::uint32_t ntouched = 0;
    for (std::size_t t = 0; t < nu; ++t) {
      const Slot* s = find(uniq[t]);
      if (!s) continue;
      const std::uint32_t* d = doc_.data() + s->offset;
      const float* w = weight_.data() + s->offset;
      for (std::uint32_t i = 0; i < s->len; ++i) {
        const std::uint32_t doc = d[i];
        if (doc >= sc.capacity) [[unlikely]] continue;
        // Weights are > 0, so acc == 0 means "first touch this query".
        if (acc[doc] == 0.f) touched[ntouched++] = doc;
        acc[doc] += w[i];
      }
    }
    for (std::uint32_t i = 0; i < ntouched; ++i) {  // select + restore the all-zero invariant
      const std::uint32_t doc = touched[i];
      out.push(acc[doc], doc);
      acc[doc] = 0.f;
    }
  }

  [[nodiscard]] std::uint32_t num_docs() const noexcept { return num_docs_; }
  [[nodiscard]] std::size_t num_terms() const noexcept { return num_terms_; }
  [[nodiscard]] std::size_t num_postings() const noexcept { return doc_.size(); }

  // Raw lookup (tests / debugging): document frequency of a term.
  [[nodiscard]] std::uint32_t doc_freq(std::uint64_t term) const noexcept {
    const Slot* s = find(term);
    return s ? s->len : 0;
  }

 private:
  friend class Bm25Builder;

  struct Slot {
    std::uint64_t key;  // 0 = empty (real hashes remapped away from 0)
    std::uint32_t offset;
    std::uint32_t len;
  };
  static constexpr std::uint64_t remap(std::uint64_t h) noexcept { return h ? h : 1; }

  const Slot* find(std::uint64_t h) const noexcept {
    if (table_.empty()) return nullptr;
    h = remap(h);
    for (std::size_t i = h & mask_;; i = (i + 1) & mask_) {
      const Slot& s = table_[i];
      if (s.key == h) return &s;
      if (s.key == 0) return nullptr;
    }
  }

  std::vector<Slot> table_;
  std::size_t mask_ = 0;
  std::size_t num_terms_ = 0;
  std::uint32_t num_docs_ = 0;
  std::vector<std::uint32_t> doc_;
  std::vector<float> weight_;
};

class Bm25Builder {
 public:
  // Add one document (chunk). `terms` may contain duplicates (that is the tf).
  void add(std::uint32_t doc, std::span<const std::uint64_t> terms) {
    scratch_.assign(terms.begin(), terms.end());
    std::sort(scratch_.begin(), scratch_.end());
    for (std::size_t i = 0; i < scratch_.size();) {
      std::size_t j = i;
      while (j < scratch_.size() && scratch_[j] == scratch_[i]) ++j;
      triples_.push_back({scratch_[i], doc, static_cast<std::uint32_t>(j - i)});
      i = j;
    }
    if (doc >= doc_len_.size()) doc_len_.resize(doc + 1, 0);
    doc_len_[doc] = static_cast<std::uint32_t>(terms.size());
    num_docs_ = std::max(num_docs_, doc + 1);
  }

  [[nodiscard]] Bm25Index freeze(Bm25Params p = {}) const {
    Bm25Index idx;
    idx.num_docs_ = num_docs_;
    if (triples_.empty()) return idx;

    std::vector<Triple> t = triples_;
    std::sort(t.begin(), t.end(), [](const Triple& a, const Triple& b) {
      return a.term != b.term ? a.term < b.term : a.doc < b.doc;
    });

    double total_len = 0;
    for (auto l : doc_len_) total_len += l;
    const double avgdl = std::max(1.0, total_len / std::max<std::uint32_t>(1, num_docs_));
    const double N = num_docs_;

    std::size_t nterms = 0;
    for (std::size_t i = 0; i < t.size(); ++i) nterms += (i == 0 || t[i].term != t[i - 1].term);
    std::size_t cap = 16;
    while (cap < nterms * 2) cap <<= 1;
    idx.table_.assign(cap, Bm25Index::Slot{0, 0, 0});
    idx.mask_ = cap - 1;
    idx.num_terms_ = nterms;
    idx.doc_.resize(t.size());
    idx.weight_.resize(t.size());

    for (std::size_t i = 0; i < t.size();) {
      std::size_t j = i;
      while (j < t.size() && t[j].term == t[i].term) ++j;
      const double df = static_cast<double>(j - i);
      const double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));  // Lucene BM25: always > 0
      for (std::size_t k = i; k < j; ++k) {
        const double tf = t[k].tf;
        const double norm = p.k1 * (1.0 - p.b + p.b * static_cast<double>(doc_len_[t[k].doc]) / avgdl);
        idx.doc_[k] = t[k].doc;
        idx.weight_[k] = static_cast<float>(idf * tf * (p.k1 + 1.0) / (tf + norm));
      }
      const std::uint64_t key = Bm25Index::remap(t[i].term);
      std::size_t s = key & idx.mask_;
      while (idx.table_[s].key != 0) s = (s + 1) & idx.mask_;
      idx.table_[s] = {key, static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j - i)};
      i = j;
    }
    return idx;
  }

  [[nodiscard]] std::uint32_t num_docs() const noexcept { return num_docs_; }

 private:
  struct Triple {
    std::uint64_t term;
    std::uint32_t doc;
    std::uint32_t tf;
  };
  std::vector<Triple> triples_;
  std::vector<std::uint32_t> doc_len_;
  std::vector<std::uint64_t> scratch_;  // reused across add() calls
  std::uint32_t num_docs_ = 0;
};

}  // namespace rag
