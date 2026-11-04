// rag_property_tests.cpp — property-based, differential and edge-case tests
// for the RAG engine (include/rag/*.hpp; the agent has its own suites).
//
// Every test is deterministic: a local splitmix64 generator with hand-written
// distributions (so results don't depend on the standard library's <random>
// distributions), fixed seeds, and a few hundred milliseconds at most in
// Release. References are written from the specification (std::string,
// std::map, double, full sorts), not from the optimized code.
//
// Bugs found by these tests. Each has a test in the PropRegression suite that
// fails on the old code (bug 6 in a sanitizer build, where it aborts):
//   1. IvfIndex::train divided by zero (SIGFPE) when
//      IvfParams::max_train_points_per_list == 0.
//   2. EngineConfig::bm25 was not validated: k1 < 0 (or b outside [0, 1]) can
//      give zero or negative posting weights, which broke Bm25Index::search's
//      first-touch test and overflowed Bm25Scratch::touched (heap overflow).
//   3. Bm25Index::load did not validate posting weights, so a forged index
//      with a valid checksum caused the same heap overflow at query time.
//   4. IvfIndex::search scored the tail rows of each list with simd::dot
//      instead of simd::dot1, so a row's score depended on its slot in the
//      list, and full-probe IVF did not reproduce the exact scan bitwise.
//   5. IvfIndex::build(store, 0) on a built index kept the old storage, so it
//      saved a file that its own load() rejects.
//   6. Prompt assembly passed a null pointer to memcpy (undefined behaviour)
//      for empty pieces such as an empty system prompt (found by UBSan).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag.hpp"
#include "rag/text_clean.hpp"
#include "test_util.hpp"

using namespace rag;
using namespace rag::testutil;

namespace {

// ================================ generator =================================
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : s_(seed) {}
  std::uint64_t next() noexcept {  // splitmix64
    std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, n), n > 0 (the modulo bias, < n / 2^64, doesn't matter here).
  std::uint32_t below(std::uint64_t n) noexcept { return static_cast<std::uint32_t>(next() % n); }
  bool coin(std::uint32_t num, std::uint32_t den) noexcept { return below(den) < num; }
  double unit() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
  float normal() noexcept {  // Box-Muller
    const double r = std::sqrt(-2.0 * std::log(1.0 - unit()));
    return static_cast<float>(r * std::cos(6.283185307179586 * unit()));
  }
  template <class T>
  const T& pick(const std::vector<T>& v) noexcept { return v[below(v.size())]; }

 private:
  std::uint64_t s_;
};

// ============================ specification =================================
namespace spec {

// Tokenizer spec: a word is a maximal run of ASCII letters/digits or bytes
// >= 0x80; ASCII letters are case-folded; every other byte separates words.
bool word_byte(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}
bool ascii_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

struct Word {
  std::string text;  // case-folded
  std::size_t offset, length;
};

std::vector<Word> words(std::string_view s) {
  std::vector<Word> out;
  for (std::size_t i = 0; i < s.size();) {
    if (!word_byte(static_cast<unsigned char>(s[i]))) { ++i; continue; }
    std::size_t j = i;
    std::string w;
    for (; j < s.size() && word_byte(static_cast<unsigned char>(s[j])); ++j) {
      const auto c = static_cast<unsigned char>(s[j]);
      w += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : s[j];
    }
    out.push_back({w, i, j - i});
    i = j;
  }
  return out;
}

const std::set<std::string>& stopwords() {
  static const std::set<std::string> s = {
      "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "i", "if", "in",
      "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "they", "this",
      "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
      "does", "do", "can", "about"};
  return s;
}

std::vector<std::string> terms(std::string_view s) {
  std::vector<std::string> t;
  for (const auto& w : words(s))
    if (!stopwords().count(w.text)) t.push_back(w.text);
  return t;
}

// HashingEmbedder spec: signed feature hashing of every non-stopword unigram
// (weight 1) and of each pair of consecutive non-stopword unigrams (weight
// 0.5), bucket = floor(h32 * D / 2^32), sign = top bit, then L2 normalisation.
std::vector<double> embed(std::string_view s, std::size_t D) {
  std::vector<double> v(D, 0.0);
  auto add = [&](std::uint64_t h, double w) {
    v[static_cast<std::size_t>((std::uint64_t{static_cast<std::uint32_t>(h)} * D) >> 32)] += (h >> 63) ? -w : w;
  };
  std::uint64_t prev = 0;
  for (const auto& t : terms(s)) {
    const std::uint64_t h = hash_term(t);
    add(h, 1.0);
    if (prev) add(rag::detail::mix64(prev * 0x9E3779B97F4A7C15ull ^ h), 0.5);
    prev = h;
  }
  double nn = 0;
  for (double x : v) nn += x * x;
  if (nn > 0)
    for (double& x : v) x /= std::sqrt(nn);
  return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

using Ranked = std::vector<std::pair<double, std::uint32_t>>;  // (score, id), best first
void rank(Ranked& r) {
  std::sort(r.begin(), r.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
}

// Lucene BM25: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), |d| = number of
// non-stopword terms, each distinct query term counted once, and at most the
// first Bm25Index::kMaxQueryTerms query terms considered.
struct Bm25 {
  std::vector<std::map<std::string, int>> tf;
  std::vector<double> len;
  std::map<std::string, int> df;
  double avgdl = 1;

  explicit Bm25(const std::vector<std::vector<std::string>>& docs) {
    double total = 0;
    for (const auto& d : docs) {
      std::map<std::string, int> m;
      for (const auto& t : d) ++m[t];
      for (const auto& kv : m) ++df[kv.first];
      tf.push_back(m);
      len.push_back(static_cast<double>(d.size()));
      total += static_cast<double>(d.size());
    }
    avgdl = std::max(1.0, docs.empty() ? 1.0 : total / static_cast<double>(docs.size()));
  }

  Ranked scores(std::vector<std::string> q, double k1, double b) const {
    if (q.size() > Bm25Index::kMaxQueryTerms) q.resize(Bm25Index::kMaxQueryTerms);
    const std::set<std::string> uq(q.begin(), q.end());
    const double N = static_cast<double>(tf.size());
    Ranked out;
    for (std::uint32_t d = 0; d < tf.size(); ++d) {
      double s = 0;
      bool any = false;
      for (const auto& t : uq) {
        const auto it = tf[d].find(t);
        if (it == tf[d].end()) continue;
        any = true;
        const double n = df.at(t), f = it->second;
        s += std::log(1 + (N - n + 0.5) / (n + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * len[d] / avgdl));
      }
      if (any) out.push_back({s, d});
    }
    rank(out);
    return out;
  }
};

Ranked rrf(std::span<const Hit> dense, std::span<const Hit> lex, const RrfParams& p) {
  std::map<std::uint32_t, double> s;
  for (std::size_t r = 0; r < dense.size(); ++r) s[dense[r].id] += double{p.w_dense} / (double{p.k0} + static_cast<double>(r + 1));
  for (std::size_t r = 0; r < lex.size(); ++r) s[lex[r].id] += double{p.w_lexical} / (double{p.k0} + static_cast<double>(r + 1));
  Ranked out;
  for (const auto& kv : s) out.push_back({kv.second, kv.first});
  rank(out);
  return out;
}

// Prompt spec: system, "\n\n### Context\n", then for each hit in rank order
// that fits both the byte capacity (question and footer reserved first) and
// the token budget (ceil(bytes / 4) per chunk text), "<citation><text>\n\n";
// then "### Question\n<q>\n\n### Answer\n". Citations: "[n] (<where>)\n".
std::string citation(std::uint32_t n, const ChunkView& c) {
  std::string s = "[" + std::to_string(n) + "] (";
  if (c.source.empty()) {
    s += "doc " + std::to_string(c.doc_id);
  } else if (c.source.size() <= 120) {
    s += c.source;
  } else {  // cut at a UTF-8 boundary at or before byte 117, then "..."
    std::size_t cut = 117;
    while (cut > 0 && (static_cast<unsigned char>(c.source[cut]) & 0xC0) == 0x80) --cut;
    s += std::string(c.source.substr(0, cut)) + "...";
  }
  if (c.page_first == 0) s += ", chunk " + std::to_string(c.ordinal);
  else if (c.page_first == c.page_last) s += ", p. " + std::to_string(c.page_first);
  else s += ", pp. " + std::to_string(c.page_first) + "-" + std::to_string(c.page_last);
  return s + ")\n";
}

struct Prompt {
  std::string text;
  std::uint32_t used = 0, tokens = 0;
  bool truncated = false;
};

template <class Lookup>
Prompt prompt(std::string_view q, std::span<const Hit> hits, Lookup&& lookup, const PromptParams& p, std::size_t cap) {
  Prompt r;
  const std::string head = std::string(p.system) + "\n\n### Context\n";
  const std::string tail = "### Question\n" + std::string(q) + "\n\n### Answer\n";
  if (head.size() + tail.size() > cap) { r.truncated = true; return r; }
  std::string body;
  for (const Hit& h : hits) {
    const ChunkView c = lookup(h.id);
    const std::string entry = citation(r.used + 1, c) + std::string(c.text) + "\n\n";
    const auto tok = static_cast<std::uint32_t>((c.text.size() + 3) / 4);
    if (head.size() + body.size() + entry.size() + tail.size() > cap ||
        std::uint64_t{r.tokens} + tok > p.max_context_tokens) {
      r.truncated = true;
      continue;
    }
    body += entry;
    r.tokens += tok;
    ++r.used;
  }
  r.text = head + body + tail;
  return r;
}

bool valid_utf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    const std::size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) return false;
    for (std::size_t k = 1; k < n; ++k)
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}

}  // namespace spec

// ================================ checkers ==================================
// `got` agrees with the spec's FULL ranking `want`: the score at every rank
// matches within tol, and the id matches wherever that score is not within
// 2 * tol of a neighbour (near-ties may legitimately come out either way).
::testing::AssertionResult same_ranking(const spec::Ranked& want, std::span<const Hit> got, double tol) {
  const std::size_t n = std::min(want.size(), got.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double w = want[i].first;
    if (std::abs(double{got[i].score} - w) > tol * std::max(1.0, std::abs(w)))
      return ::testing::AssertionFailure() << "rank " << i << ": score " << got[i].score << ", spec " << w;
    const bool tied = (i > 0 && std::abs(w - want[i - 1].first) < 2 * tol) ||
                      (i + 1 < want.size() && std::abs(w - want[i + 1].first) < 2 * tol);
    if (!tied && got[i].id != want[i].second)
      return ::testing::AssertionFailure() << "rank " << i << ": id " << got[i].id << ", spec " << want[i].second;
  }
  return ::testing::AssertionSuccess();
}

// Any retriever list: ids name chunks, are unique, and the list is strictly
// ordered by (score desc, id asc).
::testing::AssertionResult well_formed(std::span<const Hit> h, std::uint32_t nchunks) {
  std::set<std::uint32_t> seen;
  for (std::size_t i = 0; i < h.size(); ++i) {
    if (h[i].id >= nchunks) return ::testing::AssertionFailure() << "id " << h[i].id << " >= " << nchunks;
    if (std::isnan(h[i].score)) return ::testing::AssertionFailure() << "NaN score at " << i;
    if (!seen.insert(h[i].id).second) return ::testing::AssertionFailure() << "duplicate id " << h[i].id;
    if (i > 0 && !better(h[i - 1], h[i])) return ::testing::AssertionFailure() << "out of order at " << i;
  }
  return ::testing::AssertionSuccess();
}

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

::testing::AssertionResult same_hits(std::span<const Hit> a, std::span<const Hit> b) {
  if (a.size() != b.size()) return ::testing::AssertionFailure() << "sizes " << a.size() << " vs " << b.size();
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].id != b[i].id || bits(a[i].score) != bits(b[i].score))
      return ::testing::AssertionFailure() << "rank " << i << ": (" << a[i].id << ", " << a[i].score << ") vs ("
                                           << b[i].id << ", " << b[i].score << ")";
  return ::testing::AssertionSuccess();
}

// Bitwise equality of everything but the snapshot version.
template <class R>
::testing::AssertionResult same_result(const R& a, const R& b) {
  if (auto r = same_hits(a.hits, b.hits); !r) return r << " (hits)";
  if (auto r = same_hits(a.dense, b.dense); !r) return r << " (dense)";
  if (auto r = same_hits(a.lexical, b.lexical); !r) return r << " (lexical)";
  if (a.prompt.prompt != b.prompt.prompt || a.prompt.chunks_used != b.prompt.chunks_used ||
      a.prompt.context_tokens != b.prompt.context_tokens || a.prompt.truncated != b.prompt.truncated)
    return ::testing::AssertionFailure() << "prompts differ";
  return ::testing::AssertionSuccess();
}

// ================================ engine setup ==============================
constexpr std::size_t kDim = 192;  // an odd multiple of 64: no SIMD loop runs a power-of-two trip count
constexpr std::uint32_t kMaxK = 32;
using Emb = HashingEmbedder<kDim>;
using Engine = RagEngine<Emb, kMaxK>;

constexpr DenseMode kModes[] = {DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf, DenseMode::IvfInt8};
bool ivf_mode(DenseMode m) { return m == DenseMode::Ivf || m == DenseMode::IvfInt8; }

const std::vector<std::string>& vocab() {
  static const std::vector<std::string> v = [] {
    std::vector<std::string> w = {"Order", "order", "ORDER", "book", "VWAP", "vwap", "spread", "bid", "ask",
                                  "queue", "latency", "tick", "venue", "café", "Crème", "brûlée", "naïve",
                                  "\xE6\x97\xA5\xE6\x9C\xAC", "\xCE\xA9mega", "42", "x86", "L2", "it's", "ZX81Q",
                                  "a1b2", "fill", "cancel", "auction", "imbalance", "quote", "hedge", "delta",
                                  "gamma", "THE", "Of"};
    for (const auto& s : spec::stopwords()) w.push_back(s);
    return w;
  }();
  return v;
}

const std::vector<std::string>& separators() {
  static const std::vector<std::string> v = {" ", "  ", ", ", ". ", "\n", "\n\n", "\t", " -- ", "/", "(", ") ",
                                             "!?", "\r\n", "\f", std::string(1, '\0') + " ", "\xC2\xA0", " \v"};
  return v;
}

std::string random_text(Rng& rng, std::size_t nwords) {
  std::string s;
  if (rng.coin(1, 8)) s += rng.pick(separators());  // sometimes leading separators
  for (std::size_t i = 0; i < nwords; ++i) {
    s += rng.pick(vocab());
    s += rng.pick(separators());
  }
  return s;
}

const std::vector<std::string>& source_names() {
  static const std::vector<std::string> v = {
      "a.pdf", "notes.md", "",
      // 130 bytes with a multi-byte character straddling the 117-byte cut.
      std::string(116, 'x') + "\xC3\xBC" + std::string(12, 'y'),
      std::string(121, 'z'), std::string(120, 'w')};
  return v;
}

struct Doc {
  std::string text;
  std::uint32_t id = 0;
  std::string source;
  std::vector<std::uint32_t> pages;  // empty: no page structure
  IngestOptions opts() const { return IngestOptions{.source = source, .page_starts = pages}; }
};

std::vector<std::uint32_t> random_pages(Rng& rng, std::size_t len) {
  std::vector<std::uint32_t> p{0};
  for (std::uint32_t at = 0;;) {
    const std::uint32_t step = 1 + rng.below(60);
    if (at + step > len || rng.coin(1, 8)) break;
    at += step;
    p.push_back(at);
  }
  return p;
}

std::vector<Doc> random_docs(Rng& rng, std::size_t n, std::size_t max_words) {
  std::vector<Doc> docs;
  for (std::size_t d = 0; d < n; ++d) {
    Doc doc;
    doc.id = static_cast<std::uint32_t>(d);
    if (!docs.empty() && rng.coin(1, 8)) doc.text = docs[rng.below(docs.size())].text;  // duplicate: exact ties
    else if (rng.coin(1, 12)) doc.text = rng.coin(1, 2) ? "" : "the of and is it to";   // nothing to retrieve
    else doc.text = random_text(rng, rng.below(max_words + 1));
    if (rng.coin(1, 2)) doc.source = rng.pick(source_names());
    if (rng.coin(1, 2)) doc.pages = random_pages(rng, doc.text.size());
    docs.push_back(std::move(doc));
  }
  return docs;
}

std::string random_query(Rng& rng, const std::vector<Doc>& docs) {
  if (rng.coin(1, 12)) return rng.coin(1, 2) ? "" : "the of and";
  if (!docs.empty() && rng.coin(1, 3)) {  // a run of words from a document: many matches
    const auto w = spec::words(docs[rng.below(docs.size())].text);
    if (!w.empty()) {
      const std::size_t a = rng.below(w.size()), b = std::min(w.size(), a + 1 + rng.below(5));
      std::string q;
      for (std::size_t i = a; i < b; ++i) q += w[i].text + " ";
      return q;
    }
  }
  return random_text(rng, 1 + rng.below(6));
}

const std::string& long_system() {
  static const std::string s(300, 'S');
  return s;
}

EngineConfig random_config(Rng& rng, DenseMode mode) {
  static const std::vector<float> weights = {1.f, 1.f, 0.5f, 2.f};
  EngineConfig c;
  c.max_chunks = 4096;
  c.chunker.window_tokens = 1 + rng.below(24);
  c.chunker.overlap_tokens = rng.below(c.chunker.window_tokens);
  c.bm25.k1 = rng.coin(1, 6) ? 0.f : static_cast<float>(3.0 * rng.unit());
  c.bm25.b = rng.coin(1, 4) ? static_cast<float>(rng.below(2)) : static_cast<float>(rng.unit());
  c.rrf.k0 = static_cast<float>(1 + rng.below(100));
  c.rrf.w_dense = rng.pick(weights);
  c.rrf.w_lexical = rng.pick(weights);
  c.prompt.max_context_tokens = 8 + rng.below(600);
  switch (rng.below(4)) {
    case 0: c.prompt.system = {}; break;
    case 1: c.prompt.system = "SYS"; break;
    case 2: c.prompt.system = long_system(); break;
    default: break;  // the default system text
  }
  c.dense_mode = mode;
  c.hybrid = !rng.coin(1, 4);
  c.dense_min_score = rng.coin(1, 6) ? -2.f : rng.coin(1, 5) ? 0.15f : 0.f;
  c.retrieve_depth = 1 + rng.below(kMaxK);
  c.rerank_candidates = rng.coin(1, 2) ? 1024 : rng.below(64);
  c.nprobe = rng.coin(1, 2) ? 256 : 1 + rng.below(6);
  c.ivf.nlist = rng.coin(1, 3) ? 0 : 1 + rng.below(16);
  c.ivf.train_iters = rng.below(6);
  c.ivf.max_train_points_per_list = 1 + rng.below(32);
  c.ivf.seed = rng.next();
  c.prompt_capacity = rng.coin(1, 3) ? 32 * 1024 : 64 + rng.below(4096);
  return c;
}

std::array<float, kDim> embed_query(std::string_view q) {
  std::array<Token, Engine::kMaxQueryTokens> toks;
  const std::size_t n = tokenize(q, toks);
  std::array<float, kDim> v;
  Emb{}.embed(q, std::span<const Token>(toks.data(), n), std::span<float, kDim>(v));
  return v;
}

std::vector<ChunkRange> chunk_ranges(std::string_view text, const ChunkerConfig& cfg) {
  std::vector<Token> toks;
  for_each_token(text, [&](const Token& t) { toks.push_back(t); });
  std::vector<ChunkRange> out;
  chunk_tokens(text, toks, cfg, [&](const ChunkRange& r) { out.push_back(r); });
  return out;
}

// =================================== TopK ===================================
TEST(PropTopK, MatchesSortedReferenceWithTiesNaNInfinitiesFloorsAndInterleavedSorts) {
  Rng rng(11);
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> pool = {0.f, -0.f, 0.5f, 0.5f, -0.5f, 1.f, inf, -inf, nan, 0.25f, 0.75f};
  for (int iter = 0; iter < 400; ++iter) {
    const std::size_t n = rng.below(300), k = rng.below(80);  // k = 0 and k > capacity included
    std::vector<Hit> in;
    std::vector<std::uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0u);
    for (std::size_t i = n; i > 1; --i) std::swap(ids[i - 1], ids[rng.below(i)]);
    for (std::size_t i = 0; i < n; ++i)
      in.push_back({rng.coin(1, 2) ? rng.pick(pool) : static_cast<float>(rng.unit() * 2 - 1), ids[i] * 3 + 7});
    const std::uint32_t f = rng.below(5);
    const float floor = f == 0 ? -inf : f == 1 ? nan : f == 2 ? 0.5f : f == 3 ? -0.f : 2.f;

    TopK<64> tk;
    tk.reset(k, floor);
    const std::size_t kk = std::min<std::size_t>(k, 64);
    auto reference = [&](std::size_t upto) {
      std::vector<Hit> r;
      for (std::size_t i = 0; i < upto; ++i)
        if (in[i].score >= floor) r.push_back(in[i]);  // NaN scores and NaN floors admit nothing
      std::sort(r.begin(), r.end(), better);
      r.resize(std::min(r.size(), kk));
      return r;
    };
    for (std::size_t i = 0; i < n; ++i) {
      tk.push(in[i].score, in[i].id);
      if (rng.coin(1, 40)) {
        ASSERT_TRUE(same_hits(reference(i + 1), tk.sorted())) << "iter " << iter << " after " << i;
      }
    }
    const auto want = reference(n);
    const auto got = tk.sorted();
    ASSERT_TRUE(same_hits(want, got)) << "iter " << iter << " k " << k << " floor " << floor;
    ASSERT_TRUE(same_hits(got, tk.sorted()));  // sorted() is idempotent
    EXPECT_EQ(tk.k(), kk);
    // The threshold is the floor until the heap fills, then the k-th score.
    if (kk == 0) {
      EXPECT_TRUE(std::isnan(tk.threshold()));
    } else if (got.size() < kk) {
      EXPECT_EQ(bits(tk.threshold()), bits(floor));
    } else {
      EXPECT_EQ(bits(tk.threshold()), bits(got.back().score));
    }
  }
}

// =================================== SIMD ===================================
// Rounding bound for a length-n float dot product in any summation order,
// fused or not: |err| <= gamma_{n+1} * sum |a_i b_i| (Higham, eq. 3.4).
double dot_bound(std::size_t n, double abs_sum) {
  const double nu = static_cast<double>(n + 1) * 0x1.0p-24;
  return nu / (1 - nu) * abs_sum + 1e-30;
}

void fill_mixed(Rng& rng, float* p, std::size_t n) {  // signs and magnitudes 2^-6 .. 2^6
  for (std::size_t i = 0; i < n; ++i) p[i] = std::ldexp(rng.normal(), static_cast<int>(rng.below(13)) - 6);
}

TEST(PropSimd, FloatKernelsStayWithinTheRoundingBoundAtEveryMultipleOf64) {
  Rng rng(3);
  for (std::size_t n = 64; n <= 2048; n += 64) {
    const std::size_t stride = n + 16 * rng.below(3);  // a multiple of 16 floats keeps every row 64-byte aligned
    auto a = make_aligned_array<float>(n);
    auto rows = make_aligned_array<float>(4 * stride);
    fill_mixed(rng, a.get(), n);
    fill_mixed(rng, rows.get(), 4 * stride);
    double ref[4], tol[4];
    for (std::size_t j = 0; j < 4; ++j) {
      double s = 0, abs = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const double p = double{a[i]} * double{rows[j * stride + i]};
        s += p;
        abs += std::abs(p);
      }
      ref[j] = s;
      tol[j] = dot_bound(n, abs);
    }
    const float* b = rows.get();
    EXPECT_NEAR(simd::dot_scalar(a.get(), b, n), ref[0], tol[0]) << n;
    EXPECT_NEAR(simd::dot(a.get(), b, n), ref[0], tol[0]) << n;
    EXPECT_NEAR(simd::dot1(a.get(), b, n), ref[0], tol[0]) << n;
#if defined(__AVX2__) && defined(__FMA__)
    EXPECT_NEAR(simd::dot_avx2(a.get(), b, n), ref[0], tol[0]) << n;
#endif
#if defined(__AVX512F__)
    EXPECT_NEAR(simd::dot_avx512(a.get(), b, n), ref[0], tol[0]) << n;
#endif
    float out[4];
    simd::dot4(a.get(), rows.get(), stride, n, out);
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(out[j], ref[j], tol[j]) << n << " lane " << j;
      EXPECT_EQ(bits(out[j]), bits(simd::dot1(a.get(), rows.get() + j * stride, n))) << n << " lane " << j;
    }
    // A lane depends on its own row only: rewriting the other three rows
    // leaves it bit-identical.
    for (std::size_t j = 0; j < 4; ++j) {
      auto other = make_aligned_array<float>(4 * stride);
      fill_mixed(rng, other.get(), 4 * stride);
      std::copy_n(rows.get() + j * stride, n, other.get() + j * stride);
      float o[4];
      simd::dot4(a.get(), other.get(), stride, n, o);
      EXPECT_EQ(bits(o[j]), bits(out[j])) << n << " lane " << j;
    }
  }
  // The scalar kernels accept any length.
  for (std::size_t n : {1u, 2u, 3u, 5u, 7u, 31u, 33u, 63u, 65u, 100u, 127u, 129u}) {
    std::vector<float> a(n), b(n);
    fill_mixed(rng, a.data(), n);
    fill_mixed(rng, b.data(), n);
    double s = 0, abs = 0;
    for (std::size_t i = 0; i < n; ++i) { s += double{a[i]} * b[i]; abs += std::abs(double{a[i]} * b[i]); }
    EXPECT_NEAR(simd::dot_scalar(a.data(), b.data(), n), s, dot_bound(n, abs)) << n;
    std::vector<std::int8_t> qa(n), qb(n);
    std::int64_t want = 0;
    for (std::size_t i = 0; i < n; ++i) {
      qa[i] = static_cast<std::int8_t>(static_cast<int>(rng.below(255)) - 127);
      qb[i] = static_cast<std::int8_t>(static_cast<int>(rng.below(255)) - 127);
      want += std::int64_t{qa[i]} * qb[i];
    }
    EXPECT_EQ(simd::dot_i8_scalar(qa.data(), qb.data(), n), want) << n;
  }
}

TEST(PropSimd, Int8KernelsAreExactForEveryLengthAndPattern) {
  Rng rng(4);
  for (std::size_t n = 64; n <= 2048; n += 64) {
    std::vector<std::int8_t> a(n), b(n);
    for (int pattern = 0; pattern < 5; ++pattern) {
      for (std::size_t i = 0; i < n; ++i) {
        switch (pattern) {
          case 0: a[i] = static_cast<std::int8_t>(static_cast<int>(rng.below(255)) - 127);
                  b[i] = static_cast<std::int8_t>(static_cast<int>(rng.below(255)) - 127); break;
          case 1: a[i] = static_cast<std::int8_t>(i % 2 ? 127 : -127);                 // alternating extremes
                  b[i] = static_cast<std::int8_t>(i % 3 ? -127 : 127); break;
          case 2: a[i] = -127; b[i] = -127; break;                                     // max positive sum
          case 3: a[i] = 127; b[i] = -127; break;                                      // max negative sum
          default: a[i] = static_cast<std::int8_t>(i == n - 1 ? -127 : 0);             // lone tail value
                   b[i] = static_cast<std::int8_t>(i % 2 ? 127 : 1);
        }
      }
      std::int64_t want = 0;
      for (std::size_t i = 0; i < n; ++i) want += std::int64_t{a[i]} * b[i];
      EXPECT_EQ(simd::dot_i8_scalar(a.data(), b.data(), n), want) << n << " pattern " << pattern;
      EXPECT_EQ(simd::dot_i8(a.data(), b.data(), n), want) << n << " pattern " << pattern;
#if defined(__AVX2__)
      EXPECT_EQ(simd::dot_i8_avx2(a.data(), b.data(), n), want) << n << " pattern " << pattern;
#endif
#if defined(__AVX512BW__)
      EXPECT_EQ(simd::dot_i8_avx512(a.data(), b.data(), n), want) << n << " pattern " << pattern;
#endif
    }
  }
}

TEST(PropSimd, QuantizationAndNormalizationKeepTheirContracts) {
  Rng rng(5);
  for (int iter = 0; iter < 300; ++iter) {
    const std::size_t n = 64 * (1 + rng.below(8));
    std::vector<float> v(n);
    const int scale = static_cast<int>(rng.below(121)) - 60;  // magnitudes 2^-60 .. 2^60
    for (auto& x : v) x = rng.coin(1, 4) ? 0.f : std::ldexp(rng.normal(), scale);
    if (rng.coin(1, 10)) std::fill(v.begin(), v.end(), 0.f);
    if (rng.coin(1, 10)) v[rng.below(n)] = std::ldexp(100.f, scale);  // one outlier
    std::vector<std::int8_t> q(n, 99);
    const float s = simd::quantize_i8(v.data(), q.data(), n);
    float maxabs = 0;
    for (float x : v) maxabs = std::max(maxabs, std::abs(x));
    if (maxabs == 0) {
      EXPECT_EQ(s, 0.f);
      EXPECT_TRUE(std::all_of(q.begin(), q.end(), [](std::int8_t x) { return x == 0; }));
      continue;
    }
    ASSERT_GT(s, 0.f);
    int qmax = 0;
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_GE(q[i], -127) << "-128 would break the int8 kernels' sign trick";
      qmax = std::max(qmax, std::abs(int{q[i]}));
      // Dequantised within half a step (plus float rounding of the scale).
      EXPECT_LE(std::abs(double{v[i]} - double{s} * q[i]), 0.5002 * double{s}) << iter << " " << i;
    }
    EXPECT_EQ(qmax, 127) << "the largest component uses the full range";

    // normalize(): unit norm, or untouched if all zero.
    std::vector<float> u = v;
    simd::normalize(u.data(), n);
    double nn = 0;
    for (float x : u) nn += double{x} * x;
    if (scale > -50 && scale < 45) {  // |v|^2 neither underflows nor overflows
      EXPECT_NEAR(nn, 1.0, 1e-5) << iter;
    }
  }
  std::vector<float> z(128, 0.f);
  simd::normalize(z.data(), z.size());
  EXPECT_TRUE(std::all_of(z.begin(), z.end(), [](float x) { return bits(x) == 0; }));
}

TEST(PropSimd, VectorStoreInt8ScoresAreDequantisedDotProductsWithinTheirErrorBound) {
  Rng rng(6);
  constexpr std::size_t D = 192;
  VectorStore<D> s(301);
  std::array<float, D> v;
  for (int i = 0; i < 301; ++i) {
    for (auto& x : v) x = rng.coin(1, 3) ? 0.f : rng.normal();
    simd::normalize(v.data(), D);
    ASSERT_TRUE(s.append(v));
  }
  EXPECT_FALSE(s.append(v));  // full
  for (int qi = 0; qi < 20; ++qi) {
    std::array<float, D> q;
    for (auto& x : q) x = rng.normal();
    simd::normalize(q.data(), D);
    std::array<std::int8_t, D> qq;
    const float qs = simd::quantize_i8(q.data(), qq.data(), D);
    TopK<1024> all(1024);
    s.search_int8(qq.data(), 0, s.size(), all);
    ASSERT_EQ(all.size(), 301u);
    for (const Hit& h : all.sorted()) {
      // The stored int8 row is quantize_i8 of the fp32 row.
      std::array<std::int8_t, D> rq;
      EXPECT_EQ(bits(simd::quantize_i8(s.row(h.id), rq.data(), D)), bits(s.qscale(h.id)));
      ASSERT_EQ(std::memcmp(rq.data(), s.qrow(h.id), D), 0);
      EXPECT_EQ(bits(h.score), bits(static_cast<float>(simd::dot_i8_scalar(qq.data(), s.qrow(h.id), D)) * s.qscale(h.id)));
      // |x.y - sx sy qx.qy| <= (sy/2)|x|_1 + (sx/2)|y|_1 + D sx sy / 4.
      double l1q = 0, l1r = 0;
      for (std::size_t d = 0; d < D; ++d) { l1q += std::abs(double{q[d]}); l1r += std::abs(double{s.row(h.id)[d]}); }
      const double sr = s.qscale(h.id);
      const double bound = 0.5002 * (sr * l1q + double{qs} * l1r) + 0.26 * D * double{qs} * sr;
      double exact = 0;
      for (std::size_t d = 0; d < D; ++d) exact += double{q[d]} * double{s.row(h.id)[d]};
      EXPECT_LE(std::abs(double{h.score} * qs - exact), bound) << h.id;
    }
  }
}

// ================================ tokenizer =================================
TEST(PropTokenizer, MaximalWordRunsOnArbitraryBytesAndPrefixOnTruncation) {
  Rng rng(7);
  const std::vector<std::string> pieces = {"a", "Z", "9", " ", ",", "\n", "\t", "\xC3\xA9", "\xE2\x82\xAC",
                                           "\xF0\x9F\x98\x80", "\xC2\xA0", std::string(1, '\0'), "-", "'", "THE", "the"};
  for (int iter = 0; iter < 500; ++iter) {
    std::string s;
    const bool utf8 = iter % 2 == 0;  // valid UTF-8, or arbitrary bytes
    const std::size_t len = rng.below(120);
    for (std::size_t i = 0; i < len; ++i) {
      if (utf8) s += rng.pick(pieces);
      else s += static_cast<char>(rng.below(256));
    }
    ASSERT_TRUE(!utf8 || spec::valid_utf8(s));
    std::vector<Token> toks;
    for_each_token(s, [&](const Token& t) { toks.push_back(t); });
    const auto want = spec::words(s);
    ASSERT_EQ(toks.size(), want.size()) << iter;
    std::size_t word_bytes = 0;
    for (unsigned char c : s) word_bytes += spec::word_byte(c);
    std::size_t covered = 0;
    for (std::size_t i = 0; i < toks.size(); ++i) {
      const Token& t = toks[i];
      ASSERT_EQ(t.offset, want[i].offset);
      ASSERT_EQ(t.length, want[i].length);
      ASSERT_GT(t.length, 0u);
      const std::string_view w = std::string_view(s).substr(t.offset, t.length);
      EXPECT_EQ(t.hash, hash_term(w));
      EXPECT_EQ(t.hash, hash_term(want[i].text));  // case-insensitive
      std::string upper(w);
      for (char& c : upper) c = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
      EXPECT_EQ(hash_term(upper), t.hash);
      if (i > 0) {
        EXPECT_LT(toks[i - 1].offset + toks[i - 1].length, t.offset);  // separated, never adjacent
      }
      if (spec::valid_utf8(s)) {
        EXPECT_TRUE(spec::valid_utf8(w)) << "a token split a UTF-8 sequence";
      }
      covered += t.length;
    }
    EXPECT_EQ(covered, word_bytes) << "every word byte is in exactly one token";
    // The bounded tokenizer returns a prefix of the same tokens.
    std::array<Token, 64> buf;
    const std::size_t cap = rng.below(65);
    const std::size_t got = tokenize(s, std::span<Token>(buf.data(), cap));
    ASSERT_EQ(got, std::min(cap, toks.size()));
    for (std::size_t i = 0; i < got; ++i) {
      EXPECT_EQ(buf[i].offset, toks[i].offset);
      EXPECT_EQ(buf[i].length, toks[i].length);
      EXPECT_EQ(buf[i].hash, toks[i].hash);
    }
  }
}

TEST(PropTokenizer, LongTokensUnicodeSpacesAndStopwordOnlyText) {
  // A 1 MiB word is one token.
  const std::string big(std::size_t{1} << 20, 'q');
  std::vector<Token> toks;
  for_each_token(" " + big + " ", [&](const Token& t) { toks.push_back(t); });
  ASSERT_EQ(toks.size(), 1u);
  EXPECT_EQ(toks[0].offset, 1u);
  EXPECT_EQ(toks[0].length, big.size());
  EXPECT_EQ(toks[0].hash, hash_term(big));
  // Documented limitation: non-ASCII bytes are word bytes, so a no-break
  // space (U+00A0) glues words together (text_clean replaces it first).
  toks.clear();
  for_each_token("bid\xC2\xA0" "ask", [&](const Token& t) { toks.push_back(t); });
  EXPECT_EQ(toks.size(), 1u);
  toks.clear();
  for_each_token(rag::text::normalize_chars("bid\xC2\xA0" "ask"), [&](const Token& t) { toks.push_back(t); });
  EXPECT_EQ(toks.size(), 2u);

  // Stopword-only text (any case) gives chunks with tokens but an all-zero
  // embedding and no BM25 terms: a stopword-only query has no signal at all,
  // and the zero rows score exactly 0 against any other query (so only a
  // floor below 0, as here, lets them through).
  EngineConfig cfg;
  cfg.max_chunks = 64;
  cfg.chunker = {3, 1};
  cfg.dense_min_score = -2.f;
  Engine e(cfg);
  EXPECT_EQ(e.ingest("THE of AnD is IT to Was", 0), 3u);
  e.ingest(big, 1);  // and a chunk that is one huge token
  e.commit();
  for (std::uint32_t i = 0; i < 3; ++i)
    EXPECT_TRUE(std::all_of(e.store().row(i), e.store().row(i) + kDim, [](float x) { return x == 0.f; }));
  auto ctx = e.make_context();
  for (const char* q : {"the of and", "IT WAS", ""}) {
    const auto r = e.query(q, 8, ctx);
    EXPECT_TRUE(r.hits.empty()) << q;
    EXPECT_TRUE(r.dense.empty()) << q;
    EXPECT_TRUE(r.lexical.empty()) << q;
  }
  const auto r = e.query(std::string(big), 8, ctx);  // the huge token is retrievable by itself
  ASSERT_FALSE(r.hits.empty());
  EXPECT_EQ(e.chunk(r.hits[0].id).doc_id, 1u);
  ASSERT_EQ(r.dense.size(), 4u);
  EXPECT_EQ(r.dense[0].id, 3u);
  EXPECT_NEAR(r.dense[0].score, 1.f, 1e-6);
  for (std::uint32_t i = 1; i < 4; ++i) {
    EXPECT_EQ(r.dense[i].id, i - 1);  // the stopword chunks, tied at exactly 0, in id order
    EXPECT_EQ(bits(r.dense[i].score), 0u);
  }
  ASSERT_EQ(r.lexical.size(), 1u);
  EXPECT_EQ(r.lexical[0].id, 3u);
}

// ================================= chunker ==================================
TEST(PropChunker, WindowsCoverEveryTokenWithExactOverlapAndMonotoneBoundaries) {
  Rng rng(8);
  for (int iter = 0; iter < 800; ++iter) {
    std::string text;
    switch (iter % 8) {
      case 0: text = ""; break;
      case 1: text = rng.pick(vocab()); break;                       // one token
      case 2: text = std::string(rng.below(40), ' ') + "x" + std::string(rng.below(40), '\n'); break;
      case 3: text = " \t\r\n\f\v"; break;                           // whitespace only
      default: text = random_text(rng, rng.below(80));
    }
    ChunkerConfig cfg;
    switch (rng.below(6)) {
      case 0: cfg = {1, 0}; break;
      case 1: cfg = {0xFFFFFFFFu, 0xFFFFFFFEu}; break;  // huge window, stride 1: begin + window must not wrap
      case 2: cfg = {0x80000000u, 0}; break;
      case 3: { const std::uint32_t w = 2 + rng.below(20); cfg = {w, w - 1}; break; }  // stride 1
      default: { const std::uint32_t w = 1 + rng.below(30); cfg = {w, rng.below(w)}; }
    }
    ASSERT_NO_THROW(cfg.validate());
    std::vector<Token> toks;
    for_each_token(text, [&](const Token& t) { toks.push_back(t); });
    std::vector<ChunkRange> ch;
    chunk_tokens(text, toks, cfg, [&](const ChunkRange& r) { ch.push_back(r); });
    const auto n = static_cast<std::uint32_t>(toks.size());
    const std::string tag = "iter " + std::to_string(iter) + " n " + std::to_string(n) + " window " +
                            std::to_string(cfg.window_tokens) + " overlap " + std::to_string(cfg.overlap_tokens);
    if (n == 0) {
      EXPECT_TRUE(ch.empty()) << tag;
      continue;
    }
    const std::uint64_t window = cfg.window_tokens, stride = cfg.window_tokens - cfg.overlap_tokens;
    const std::uint64_t count = n <= window ? 1 : 1 + (n - window + stride - 1) / stride;
    ASSERT_EQ(ch.size(), count) << tag;
    std::vector<int> cover(n, 0);
    for (std::size_t i = 0; i < ch.size(); ++i) {
      const ChunkRange& c = ch[i];
      ASSERT_EQ(c.token_begin, i * stride) << tag;
      ASSERT_EQ(c.token_end, std::min<std::uint64_t>(c.token_begin + window, n)) << tag;
      ASSERT_LT(c.token_begin, c.token_end) << tag;
      if (i > 0) {
        EXPECT_EQ(ch[i - 1].token_end - c.token_begin, cfg.overlap_tokens) << tag << " chunk " << i;
        EXPECT_LT(ch[i - 1].byte_begin, c.byte_begin) << tag;
        EXPECT_LT(ch[i - 1].byte_end, c.byte_end) << tag;
      }
      for (std::uint32_t t = c.token_begin; t < c.token_end; ++t) ++cover[t];
      // Bytes: from the first token to the next token (or the end), minus
      // trailing ASCII whitespace; never cutting into the last token.
      const std::uint32_t min_end = toks[c.token_end - 1].offset + toks[c.token_end - 1].length;
      const std::size_t limit = c.token_end < n ? toks[c.token_end].offset : text.size();
      EXPECT_EQ(c.byte_begin, toks[c.token_begin].offset) << tag;
      EXPECT_GE(c.byte_end, min_end) << tag;
      EXPECT_LE(c.byte_end, limit) << tag;
      if (c.byte_end > min_end) {
        EXPECT_FALSE(spec::ascii_space(text[c.byte_end - 1])) << tag;  // trailing whitespace trimmed
      }
      if (c.byte_end < limit) {
        EXPECT_TRUE(spec::ascii_space(text[c.byte_end])) << tag;  // and only whitespace
      }
      // The chunk's text holds exactly its tokens.
      const auto inner = spec::words(std::string_view(text).substr(c.byte_begin, c.byte_end - c.byte_begin));
      ASSERT_EQ(inner.size(), c.token_end - c.token_begin) << tag;
      for (std::size_t t = 0; t < inner.size(); ++t) EXPECT_EQ(inner[t].offset + c.byte_begin, toks[c.token_begin + t].offset);
    }
    EXPECT_EQ(ch.back().token_end, n) << tag;
    EXPECT_TRUE(std::all_of(cover.begin(), cover.end(), [](int x) { return x >= 1; })) << tag;
  }
}

// ================================== BM25 ====================================
TEST(PropBm25, MatchesSpecForRandomParamsRoundTripsAndKeepsScratchZero) {
  Rng rng(9);
  for (int iter = 0; iter < 30; ++iter) {
    const std::uint32_t ndocs = 1 + rng.below(80);
    const Bm25Params p{rng.coin(1, 5) ? 0.f : static_cast<float>(3 * rng.unit()),
                       rng.coin(1, 4) ? static_cast<float>(rng.below(2)) : static_cast<float>(rng.unit())};
    std::vector<std::vector<std::string>> docs;
    Bm25Builder b;
    for (std::uint32_t d = 0; d < ndocs; ++d) {
      docs.push_back(spec::terms(random_text(rng, rng.below(30))));  // includes empty and stopword-only docs
      std::vector<std::uint64_t> h;
      for (const auto& t : docs.back()) h.push_back(hash_term(t));
      b.add(d, h);
    }
    const spec::Bm25 want_index(docs);
    const Bm25Index idx = b.freeze(p);
    EXPECT_EQ(idx.num_docs(), ndocs);
    // Round trip through a file (every third corpus: saves fsync): identical answers.
    Bm25Index loaded = b.freeze(p);
    Bm25Builder lb = b;
    if (iter % 3 == 0) {
      const auto path = temp_index("bm25");
      { io::Writer w(path); idx.save(w); b.save(w); w.commit(); }
      io::Reader r(path);
      loaded.load(r);
      lb.load(r);
      r.finish();
    }
    const Bm25Index refrozen = lb.freeze(p);

    Bm25Scratch sc(ndocs + rng.below(4));
    for (int qi = 0; qi < 12; ++qi) {
      std::vector<std::string> q = spec::terms(random_text(rng, 1 + rng.below(qi == 0 ? 400 : 6)));  // one overlong query
      std::vector<std::uint64_t> qh;
      for (const auto& t : q) qh.push_back(hash_term(t));
      TopK<64> tk(64);
      idx.search(qh, sc, tk);
      const auto want = want_index.scores(q, p.k1, p.b);
      ASSERT_EQ(tk.size(), std::min<std::size_t>(want.size(), 64)) << iter;
      EXPECT_TRUE(same_ranking(want, tk.sorted(), 1e-5)) << "iter " << iter << " k1 " << p.k1 << " b " << p.b;
      EXPECT_TRUE(well_formed(tk.sorted(), ndocs));
      for (const Hit& h : tk.sorted()) EXPECT_GT(h.score, 0.f);
      ASSERT_TRUE(std::all_of(sc.acc.get(), sc.acc.get() + sc.capacity, [](float x) { return bits(x) == 0; }))
          << "scratch accumulator must be all zero between queries";
      // Term order and repeats don't change a single bit (within the term limit).
      if (qh.size() <= Bm25Index::kMaxQueryTerms / 2) {
        std::vector<std::uint64_t> shuffled = qh;
        shuffled.insert(shuffled.end(), qh.begin(), qh.end());
        for (std::size_t i = shuffled.size(); i > 1; --i) std::swap(shuffled[i - 1], shuffled[rng.below(i)]);
        TopK<64> t2(64);
        idx.search(shuffled, sc, t2);
        EXPECT_TRUE(same_hits(tk.sorted(), t2.sorted()));
      }
      TopK<64> t3(64), t4(64);
      loaded.search(qh, sc, t3);
      refrozen.search(qh, sc, t4);
      EXPECT_TRUE(same_hits(tk.sorted(), t3.sorted())) << "loaded index";
      EXPECT_TRUE(same_hits(tk.sorted(), t4.sorted())) << "index refrozen from the loaded builder";
    }
  }
}

// ================================= prompt ===================================
TEST(PropPrompt, MatchesSpecAndNeverOverrunsItsBufferOrTokenBudget) {
  Rng rng(10);
  std::vector<std::string> texts, sources;
  std::vector<ChunkView> pool;
  for (std::uint32_t i = 0; i < 40; ++i) {
    texts.push_back(i % 9 == 0 ? std::string(500 + rng.below(800), 't') : random_text(rng, rng.below(25)));
    sources.push_back(rng.pick(source_names()));
  }
  for (std::uint32_t i = 0; i < 40; ++i) {
    const std::uint32_t pf = rng.coin(1, 2) ? 0 : 1 + rng.below(9);
    pool.push_back({texts[i], rng.coin(1, 8) ? 0xFFFFFFFFu : rng.below(1000), rng.below(100), sources[i], pf,
                    pf == 0 ? 0 : pf + rng.below(3)});
  }
  auto lookup = [&](std::uint32_t id) { return pool[id]; };
  for (int iter = 0; iter < 600; ++iter) {
    std::vector<Hit> hits(rng.below(20));
    for (auto& h : hits) h = {static_cast<float>(rng.unit()), rng.below(40)};  // repeats allowed
    const std::string q = random_text(rng, rng.below(12));
    const std::string sys = std::string(rng.below(3) == 0 ? 300 : rng.below(40), 's');
    const PromptParams p{.max_context_tokens = rng.coin(1, 8) ? 0 : rng.below(500), .system = sys};
    const std::size_t cap = rng.coin(1, 10) ? rng.below(64) : rng.below(4000);
    std::vector<char> buf(cap + 64, '#');
    const auto r = build_prompt(q, hits, lookup, p, std::span<char>(buf.data(), cap));
    const auto want = spec::prompt(q, hits, lookup, p, cap);
    ASSERT_EQ(r.prompt, want.text) << "iter " << iter;
    EXPECT_EQ(r.chunks_used, want.used);
    EXPECT_EQ(r.context_tokens, want.tokens);
    EXPECT_EQ(r.truncated, want.truncated);
    EXPECT_LE(r.prompt.size(), cap);
    EXPECT_LE(r.context_tokens, p.max_context_tokens);
    if (!r.prompt.empty()) {
      EXPECT_EQ(r.prompt.data(), buf.data());
    }
    EXPECT_TRUE(std::all_of(buf.begin() + static_cast<std::ptrdiff_t>(cap), buf.end(), [](char c) { return c == '#'; }))
        << "wrote past the buffer";
  }
  // Long source names are cut to at most 120 bytes at a UTF-8 boundary.
  for (const auto& s : source_names()) {
    char cite[detail::kMaxCitation];
    const ChunkView c{"t", 1, 2, s, 3, 4};
    const std::string_view got(cite, detail::format_citation(cite, 12345, c));
    EXPECT_EQ(got, spec::citation(12345, c));
    EXPECT_TRUE(spec::valid_utf8(got)) << got;
    EXPECT_LE(got.size(), std::string_view("[12345] (, pp. 3-4)\n").size() + 120);
  }
  // The longest possible citation fits its buffer.
  char cite[detail::kMaxCitation];
  const std::string longest(500, 'L');
  const ChunkView worst{"t", 0xFFFFFFFFu, 0xFFFFFFFFu, longest, 0xFFFFFFFEu, 0xFFFFFFFFu};
  const std::size_t n = detail::format_citation(cite, 0xFFFFFFFFu, worst);
  EXPECT_EQ(std::string_view(cite, n), spec::citation(0xFFFFFFFFu, worst));
}

// =============================== text clean =================================
TEST(PropTextClean, NormalizationKeepsValidUtf8IsIdempotentAndReflowIsStable) {
  Rng rng(12);
  // Code points the cleaner maps or drops, plus ones it must leave alone.
  const std::vector<std::string> pieces = {
      "a", "B", "7", " ", "\t", "\r", "\n", "\r\n", "\f", "\v", "\x01", "\x1B", "-", ".",
      "\xC2\xA0", "\xC2\xAD", "\xE2\x80\x80", "\xE2\x80\x8A", "\xE2\x80\x8B", "\xE2\x80\x8D", "\xE2\x80\x90",
      "\xE2\x80\x91", "\xE2\x80\xAF", "\xE2\x81\x9F", "\xE3\x80\x80", "\xEF\xBB\xBF", "\xEF\xAC\x80", "\xEF\xAC\x81",
      "\xEF\xAC\x86", "\xC3\xA9", "\xE2\x80\xA8", "\xE2\x80\x92", "\xE2\x82\xAC", "\xF0\x9F\x98\x80", "\xEF\xBF\xBD"};
  const std::vector<std::string> mapped = {"\xC2\xA0", "\xC2\xAD", "\xE2\x80\x80", "\xE2\x80\x8A", "\xE2\x80\x8B",
                                           "\xE2\x80\x8D", "\xE2\x80\x90", "\xE2\x80\x91", "\xE2\x80\xAF",
                                           "\xE2\x81\x9F", "\xE3\x80\x80", "\xEF\xBB\xBF", "\xEF\xAC\x80",
                                           "\xEF\xAC\x81", "\xEF\xAC\x86"};
  for (int iter = 0; iter < 400; ++iter) {
    std::string s;
    for (std::uint32_t i = 0, n = rng.below(60); i < n; ++i) s += rng.pick(pieces);
    const std::string once = text::normalize_chars(s);
    ASSERT_TRUE(spec::valid_utf8(once)) << iter;
    for (unsigned char c : once) EXPECT_TRUE(c >= 0x20 || c == '\n') << iter;
    for (const auto& m : mapped) EXPECT_EQ(once.find(m), std::string::npos) << iter;
    EXPECT_EQ(text::normalize_chars(once), once) << "normalize_chars is idempotent on valid UTF-8";
    for (bool join : {true, false})
      for (bool dehyph : {true, false}) {
        text::CleanOptions o;
        o.join_lines = join;
        o.dehyphenate = dehyph;
        const std::string r1 = text::reflow(once, o);
        EXPECT_EQ(text::reflow(r1, o), r1) << "reflow is idempotent: iter " << iter;
        EXPECT_TRUE(spec::valid_utf8(r1));
      }
  }
}

TEST(PropTextClean, JoinedPagesAlwaysIngestAndChunksCiteTheirOwnPages) {
  Rng rng(13);
  for (int iter = 0; iter < 60; ++iter) {
    std::vector<std::string> pages(rng.below(7));
    for (auto& p : pages) p = rng.coin(1, 5) ? "" : random_text(rng, rng.below(30));
    const auto cleaned = text::clean_pages(pages);
    const auto joined = text::join_pages(cleaned);
    ASSERT_EQ(joined.page_starts.size(), pages.size());
    EngineConfig cfg;
    cfg.max_chunks = 1024;
    cfg.chunker = {1 + rng.below(10), 0};
    Engine e(cfg);
    for (std::size_t i = 0; i < pages.size(); ++i)
      ASSERT_EQ(std::string_view(joined.text).substr(joined.page_starts[i], cleaned[i].size()), cleaned[i]);
    ASSERT_NO_THROW(e.ingest(joined.text, 0, IngestOptions{.source = "doc.pdf", .page_starts = joined.page_starts}));
    const auto want = chunk_ranges(joined.text, cfg.chunker);
    ASSERT_EQ(e.size(), want.size());
    for (std::uint32_t i = 0; i < e.size(); ++i) {
      const ChunkView c = e.chunk(i);
      ASSERT_GE(c.page_first, 1u);
      ASSERT_LE(c.page_first, c.page_last);
      ASSERT_LE(c.page_last, pages.size());
      // A chunk's first and last bytes are text of the pages it cites, never
      // the separators join_pages put between pages.
      const std::uint32_t first = want[i].byte_begin, last = want[i].byte_end - 1;
      EXPECT_LT(first - joined.page_starts[c.page_first - 1], cleaned[c.page_first - 1].size()) << iter;
      EXPECT_LT(last - joined.page_starts[c.page_last - 1], cleaned[c.page_last - 1].size()) << iter;
    }
  }
}

// =================================== IVF ====================================
template <std::size_t D>
void check_ivf_properties(std::uint64_t seed) {
  Rng rng(seed);
  const std::uint32_t n = 1 + rng.below(500);
  VectorStore<D> s(n);
  std::array<float, D> v;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (i > 0 && rng.coin(1, 10)) std::copy_n(s.row(rng.below(i)), D, v.begin());  // duplicate row
    else if (rng.coin(1, 30)) v.fill(0.f);                                          // empty-content row
    else { for (auto& x : v) x = rng.normal(); simd::normalize(v.data(), D); }
    s.append(v);
  }
  IvfParams p;
  p.nlist = rng.coin(1, 3) ? 0 : 1 + rng.below(40);
  p.train_iters = rng.below(6);
  p.max_train_points_per_list = 1 + rng.below(10);
  p.seed = rng.next();
  p.storage = rng.coin(1, 2) ? IvfStorage::Fp32 : IvfStorage::Both;
  IvfIndex<D> ivf;
  ivf.build(s, n, p);
  const std::string tag = "seed " + std::to_string(seed) + " D " + std::to_string(D) + " n " + std::to_string(n);
  ASSERT_GE(ivf.nlist(), 1u) << tag;
  ASSERT_LE(ivf.nlist(), n) << tag;
  std::uint32_t total = 0;
  for (std::uint32_t l = 0; l < ivf.nlist(); ++l) total += ivf.list_size(l);
  ASSERT_EQ(total, n) << tag;

  for (int qi = 0; qi < 6; ++qi) {
    std::array<float, D> q;
    for (auto& x : q) x = rng.normal();
    simd::normalize(q.data(), D);
    // Every list probed: each row exactly once, with exactly its exact-scan score.
    TopK<1024> all(1024);
    ivf.search(q.data(), ivf.nlist(), all);
    ASSERT_EQ(all.size(), n) << tag;
    std::vector<int> seen(n, 0);
    for (const Hit& h : all.sorted()) {
      ASSERT_LT(h.id, n);
      ++seen[h.id];
      EXPECT_EQ(bits(h.score), bits(simd::dot1(q.data(), s.row(h.id), D))) << tag << " row " << h.id;
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int x) { return x == 1; })) << tag;

    const std::size_t k = 1 + rng.below(40);
    TopK<64> exact(k);
    s.search_exact(q.data(), 0, n, exact);
    const auto ex = exact.sorted();
    TopK<64> full(k);
    ivf.search(q.data(), ivf.nlist(), full);
    EXPECT_TRUE(same_hits(ex, full.sorted())) << tag << ": full probe must equal the exact scan";
    // Probing more lists never makes any rank worse, and no rank beats exact.
    std::vector<Hit> prev;
    for (std::uint32_t probe = 1; probe <= ivf.nlist(); ++probe) {
      TopK<64> part(k);
      ivf.search(q.data(), probe, part);
      const auto got = part.sorted();
      ASSERT_TRUE(well_formed(got, n)) << tag;
      ASSERT_LE(got.size(), ex.size());
      ASSERT_GE(got.size(), prev.size());
      for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(bits(got[i].score), bits(simd::dot1(q.data(), s.row(got[i].id), D))) << tag;
        EXPECT_LE(got[i].score, ex[i].score) << tag << " nprobe " << probe << " rank " << i;
        if (i < prev.size()) {
          EXPECT_GE(got[i].score, prev[i].score) << tag << " nprobe " << probe << " rank " << i;
        }
      }
      prev.assign(got.begin(), got.end());
    }
    // SQ8 lists: dequantised scores of the store's own int8 rows.
    if (ivf.has_int8()) {
      std::array<std::int8_t, D> qq;
      simd::quantize_i8(q.data(), qq.data(), D);
      TopK<1024> cand(1024);
      ivf.search_int8(q.data(), qq.data(), ivf.nlist(), cand);
      ASSERT_EQ(cand.size(), n) << tag;
      for (const Hit& h : cand.sorted())
        EXPECT_EQ(bits(h.score), bits(static_cast<float>(simd::dot_i8(qq.data(), s.qrow(h.id), D)) * s.qscale(h.id)));
    }
  }
  // Save / load round trip: identical answers.
  const auto path = temp_index(("ivf" + std::to_string(seed)).c_str());
  { io::Writer w(path); ivf.save(w); w.commit(); }
  io::Reader r(path);
  IvfIndex<D> loaded;
  loaded.load(r, n);
  r.finish();
  EXPECT_EQ(loaded.nlist(), ivf.nlist());
  EXPECT_EQ(loaded.covered(), n);
  std::array<float, D> q;
  for (auto& x : q) x = rng.normal();
  TopK<64> a(64), b(64);
  ivf.search(q.data(), 3, a);
  loaded.search(q.data(), 3, b);
  EXPECT_TRUE(same_hits(a.sorted(), b.sorted())) << tag;
}

TEST(PropIvf, ProbesReturnExactScoresOfASubsetAndImproveMonotonicallyWithNprobe) {
  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    check_ivf_properties<64>(seed);
    check_ivf_properties<192>(100 + seed);
  }
}

// Clustered unit vectors (as real embeddings are): IVF recall is meaningful.
template <std::size_t D>
void fill_clusters(Rng& rng, VectorStore<D>& s, std::uint32_t n, std::uint32_t clusters, float noise) {
  std::vector<std::array<float, D>> centers(clusters);
  for (auto& c : centers) {
    for (auto& x : c) x = rng.normal();
    simd::normalize(c.data(), D);
  }
  const float sigma = noise / std::sqrt(static_cast<float>(D));
  std::array<float, D> v;
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto& c = centers[rng.below(clusters)];
    for (std::size_t d = 0; d < D; ++d) v[d] = c[d] + sigma * rng.normal();
    simd::normalize(v.data(), D);
    s.append(v);
  }
}

TEST(PropIvf, RecallAtTenOnClusteredDataStaysAboveItsMeasuredFloor) {
  // 4000 rows in 200 overlapping clusters (noise 1.0), nlist 32, queries near
  // stored rows. Measured recall@10 over seeds 0..3 (Release, AVX-512, g++):
  // nprobe 1: 0.560-0.687, 2: 0.670-0.767, 4: 0.795-0.862, 8: 0.900-0.935.
  // (With 40 tight clusters every setting measured 1.000, which tests nothing.)
  // The floors sit 0.05-0.07 below the minimum to absorb other ISAs' rounding.
  constexpr std::size_t D = 64;
  constexpr std::uint32_t kClusters = 200;
  constexpr float kNoise = 1.0f, kQueryNoise = 0.05f;
  for (std::uint64_t seed = 0; seed < 4; ++seed) {
    Rng rng(900 + seed);
    VectorStore<D> s(4000);
    fill_clusters(rng, s, 4000, kClusters, kNoise);
    IvfIndex<D> ivf;
    ivf.build(s, s.size(), {.nlist = 32, .train_iters = 8, .max_train_points_per_list = 64, .seed = seed});
    double r1 = 0, r2 = 0, r4 = 0, r8 = 0;
    constexpr int kQueries = 60;
    for (int qi = 0; qi < kQueries; ++qi) {
      std::array<float, D> q;
      const float* near = s.row(rng.below(s.size()));
      for (std::size_t d = 0; d < D; ++d) q[d] = near[d] + kQueryNoise * rng.normal();
      simd::normalize(q.data(), D);
      TopK<10> exact(10);
      s.search_exact(q.data(), 0, s.size(), exact);
      std::set<std::uint32_t> truth;
      for (const Hit& h : exact.sorted()) truth.insert(h.id);
      auto rec = [&](std::uint32_t nprobe) {
        TopK<10> p(10);
        ivf.search(q.data(), nprobe, p);
        double r = 0;
        for (const Hit& h : p.sorted()) r += static_cast<double>(truth.count(h.id)) / 10.0;
        return r;
      };
      r1 += rec(1); r2 += rec(2); r4 += rec(4); r8 += rec(8);
    }
    r1 /= kQueries;
    r2 /= kQueries;
    r4 /= kQueries;
    r8 /= kQueries;
    EXPECT_GE(r1, 0.50) << "seed " << seed;
    EXPECT_GE(r2, 0.60) << "seed " << seed;
    EXPECT_GE(r4, 0.72) << "seed " << seed;
    EXPECT_GE(r8, 0.85) << "seed " << seed;
    // Exact, not statistical: a true top-10 row that is among the candidates
    // is always in the candidates' top 10, so recall only grows with nprobe.
    EXPECT_LE(r1, r2) << "seed " << seed;
    EXPECT_LE(r2, r4) << "seed " << seed;
    EXPECT_LE(r4, r8) << "seed " << seed;
  }
}

// ============================ engine vs spec ================================
TEST(PropReference, EngineMatchesSpecForRandomCorporaConfigsAndAllDenseModes) {
  // How often each kind of check ran with something to check (asserted at
  // the end, so a change to the generators can't make the test vacuous).
  std::array<int, 4> exact_checks{}, approx_checks{};
  int lexical_checks = 0, dense_only_checks = 0, truncated_prompts = 0, delta_checks = 0;
  for (std::uint32_t iter = 0; iter < 48; ++iter) {
    Rng rng(0xA11CE + iter);
    const EngineConfig cfg = random_config(rng, kModes[iter % 4]);
    Engine e(cfg);
    const auto docs = random_docs(rng, 4 + rng.below(36), 60);
    std::uint64_t commits = 0;
    for (const Doc& d : docs) {
      e.ingest(d.text, d.id, d.opts());
      if (rng.coin(1, 10)) { e.commit(); ++commits; }
    }
    if (rng.coin(3, 4)) { e.commit(); ++commits; }
    const std::uint32_t live = e.size(), committed = e.committed();
    // The spec's view of every chunk, computed from its text alone.
    std::vector<std::vector<double>> vecs(live);
    std::vector<std::vector<std::string>> chunk_terms(committed);
    for (std::uint32_t i = 0; i < live; ++i) vecs[i] = spec::embed(e.chunk(i).text, kDim);
    for (std::uint32_t i = 0; i < committed; ++i) chunk_terms[i] = spec::terms(e.chunk(i).text);
    const spec::Bm25 bm25(chunk_terms);

    auto ctx = e.make_context();
    const std::uint32_t nlist = e.pin(ctx)->ivf.nlist();
    for (int qi = 0; qi < 16; ++qi) {
      const std::string q = random_query(rng, docs);
      const std::uint32_t k = rng.below(kMaxK + 6);  // 0 and k > MaxK (clamped) included
      const auto r = e.query(q, k, ctx);
      const std::uint32_t kk = std::min<std::uint32_t>(k, kMaxK), depth = cfg.hybrid ? cfg.retrieve_depth : kk;
      const std::size_t shortlist = std::min<std::size_t>(std::max(cfg.rerank_candidates, depth), Engine::kMaxRerank);
      const bool all_lists = nlist == 0 || std::min(cfg.nprobe, IvfIndex<kDim>::kMaxProbe) >= nlist;
      bool exhaustive = true;
      switch (cfg.dense_mode) {
        case DenseMode::Exact: break;
        case DenseMode::Int8Rerank: exhaustive = shortlist >= live; break;
        case DenseMode::Ivf: exhaustive = all_lists; break;
        case DenseMode::IvfInt8: exhaustive = all_lists && shortlist >= committed; break;
      }
      const std::string tag = "iter " + std::to_string(iter) + " q '" + q + "' k " + std::to_string(k);
      EXPECT_EQ(r.snapshot_version, commits) << tag;

      // Dense: every hit carries its true cosine; exhaustive modes return the
      // spec's full ranking (cosine strictly above dense_min_score).
      ASSERT_TRUE(well_formed(r.dense, live)) << tag;
      ASSERT_LE(r.dense.size(), depth) << tag;
      const auto qv = spec::embed(q, kDim);
      const bool signal = std::any_of(qv.begin(), qv.end(), [](double x) { return x != 0; });
      spec::Ranked want_dense;
      std::size_t strict = 0;
      for (std::uint32_t i = 0; i < live && signal; ++i) {
        const double c = spec::dot(qv, vecs[i]);
        if (c > cfg.dense_min_score - 1e-6) want_dense.push_back({c, i});
        strict += c > cfg.dense_min_score + 1e-6;
      }
      spec::rank(want_dense);
      for (const Hit& h : r.dense) {
        EXPECT_NEAR(h.score, spec::dot(qv, vecs[h.id]), 1e-5) << tag;
        EXPECT_GT(h.score, cfg.dense_min_score) << tag;
      }
      const auto mode = static_cast<std::size_t>(cfg.dense_mode);
      if (exhaustive) {
        EXPECT_GE(r.dense.size(), std::min<std::size_t>(strict, depth)) << tag;
        EXPECT_LE(r.dense.size(), std::min<std::size_t>(want_dense.size(), depth)) << tag;
        EXPECT_TRUE(same_ranking(want_dense, r.dense, 1e-5)) << tag << " mode " << mode;
      }
      if (!r.dense.empty()) ++(exhaustive ? exact_checks[mode] : approx_checks[mode]);

      // Lexical: BM25 over the committed chunks with the configured k1, b.
      if (cfg.hybrid) {
        ASSERT_TRUE(well_formed(r.lexical, committed)) << tag;
        const auto want = bm25.scores(spec::terms(q), cfg.bm25.k1, cfg.bm25.b);
        EXPECT_EQ(r.lexical.size(), std::min<std::size_t>(want.size(), depth)) << tag;
        EXPECT_TRUE(same_ranking(want, r.lexical, 1e-5)) << tag;
        // Fusion: RRF of the engine's own two lists.
        const auto fused = spec::rrf(r.dense, r.lexical, cfg.rrf);
        ASSERT_TRUE(well_formed(r.hits, live)) << tag;
        EXPECT_EQ(r.hits.size(), std::min<std::size_t>(fused.size(), kk)) << tag;
        EXPECT_TRUE(same_ranking(fused, r.hits, 1e-6)) << tag;
        lexical_checks += !r.lexical.empty();
      } else {
        EXPECT_TRUE(r.lexical.empty()) << tag;
        EXPECT_TRUE(same_hits(r.hits, r.dense.first(std::min<std::size_t>(kk, r.dense.size())))) << tag;
        dense_only_checks += !r.hits.empty();
      }
      truncated_prompts += r.prompt.truncated && r.prompt.chunks_used > 0;
      delta_checks += committed < live && !r.dense.empty();
      // Prompt: the spec's prompt for these hits, byte for byte.
      const auto sp = spec::prompt(q, r.hits, [&](std::uint32_t id) { return e.chunk(id); }, cfg.prompt, cfg.prompt_capacity);
      EXPECT_EQ(r.prompt.prompt, sp.text) << tag;
      EXPECT_EQ(r.prompt.chunks_used, sp.used) << tag;
      EXPECT_EQ(r.prompt.context_tokens, sp.tokens) << tag;
      EXPECT_EQ(r.prompt.truncated, sp.truncated) << tag;
    }
  }
  for (std::size_t m = 0; m < 4; ++m) {
    EXPECT_GT(exact_checks[m], 0) << "mode " << m << ": no exhaustive dense list was checked";
    if (m > 0) {
      EXPECT_GT(approx_checks[m], 0) << "mode " << m << ": no approximate dense list was checked";
    }
  }
  EXPECT_GT(lexical_checks, 0);
  EXPECT_GT(dense_only_checks, 0);
  EXPECT_GT(truncated_prompts, 0);
  EXPECT_GT(delta_checks, 0);
}

// Approximate modes made exhaustive reproduce their exact counterparts bit for
// bit: IVF probing every list scores each row exactly like the flat scan (a
// dot4 lane or dot1), and IVF-SQ8 probing every list builds the same int8
// shortlist as the flat int8 scan. Partial probes return true scores of a
// subset, so no rank can beat the exact list.
TEST(PropModes, FullProbeIvfModesEqualTheirFlatCounterpartsBitwise) {
  int deltas = 0, nonempty = 0, partial_misses = 0;
  for (std::uint32_t iter = 0; iter < 8; ++iter) {
    Rng rng(0xB0B + iter);
    EngineConfig base = random_config(rng, DenseMode::Exact);
    base.ivf.nlist = rng.coin(1, 3) ? 0 : 1 + rng.below(16);
    base.nprobe = IvfIndex<kDim>::kMaxProbe;
    std::array<EngineConfig, 5> cfgs;
    const DenseMode modes[5] = {DenseMode::Exact, DenseMode::Ivf, DenseMode::Int8Rerank, DenseMode::IvfInt8, DenseMode::Ivf};
    for (std::size_t m = 0; m < 5; ++m) { cfgs[m] = base; cfgs[m].dense_mode = modes[m]; }
    cfgs[4].nprobe = 1 + rng.below(3);  // partial probe
    std::vector<std::unique_ptr<Engine>> es;
    for (const auto& c : cfgs) es.push_back(std::make_unique<Engine>(c));
    const auto docs = random_docs(rng, 10 + rng.below(40), 60);
    for (const Doc& d : docs) {
      for (auto& e : es) e->ingest(d.text, d.id, d.opts());
      if (rng.coin(1, 8)) {
        for (auto& e : es) e->commit();
      }
    }
    for (auto& e : es) e->commit();
    if (rng.coin(1, 2)) {  // an uncommitted delta, scanned exactly by the IVF modes
      for (std::uint32_t d = 0; d < 3; ++d) {
        const std::string t = random_text(rng, 20);
        for (auto& e : es) e->ingest(t, 900 + d);
      }
    }
    std::vector<Engine::QueryContext> ctxs;
    for (auto& e : es) ctxs.push_back(e->make_context());
    const bool all_committed = es[0]->committed() == es[0]->size();
    deltas += !all_committed;
    for (int qi = 0; qi < 15; ++qi) {
      const std::string q = random_query(rng, docs);
      const std::uint32_t k = rng.below(kMaxK + 1);
      std::vector<Engine::QueryResult> r;
      for (std::size_t m = 0; m < 5; ++m) r.push_back(es[m]->query(q, k, ctxs[m]));
      const std::string tag = "iter " + std::to_string(iter) + " q '" + q + "'";
      EXPECT_TRUE(same_result(r[0], r[1])) << tag << ": Ivf (every list) vs Exact";
      if (all_committed) {
        EXPECT_TRUE(same_result(r[2], r[3])) << tag << ": IvfInt8 (every list) vs Int8Rerank";
      } else {
        EXPECT_TRUE(same_hits(r[2].lexical, r[3].lexical)) << tag;
      }
      const auto qv = embed_query(q);
      const auto& ex = r[0].dense;
      const auto& part = r[4].dense;
      ASSERT_TRUE(well_formed(part, es[4]->size())) << tag;
      ASSERT_LE(part.size(), ex.size()) << tag;
      bool worse = part.size() < ex.size();
      for (std::size_t i = 0; i < part.size(); ++i) {
        EXPECT_EQ(bits(part[i].score), bits(simd::dot1(qv.data(), es[4]->store().row(part[i].id), kDim))) << tag;
        EXPECT_LE(part[i].score, ex[i].score) << tag << " rank " << i;
        worse |= part[i].score < ex[i].score;
      }
      partial_misses += worse;
      nonempty += !ex.empty();
    }
  }
  EXPECT_GT(deltas, 0);
  EXPECT_GT(nonempty, 0);
  EXPECT_GT(partial_misses, 0) << "the partial probe never missed anything: it tests nothing";
}

// ================================ batching ==================================
TEST(PropBatch, RandomBatchSizesAndMismatchedSpansAreBitIdenticalToSingleQueries) {
  constexpr std::uint64_t kUntouched = 0xDEADBEEF;
  int answered = 0, untouched = 0;
  for (std::uint32_t iter = 0; iter < 8; ++iter) {
    Rng rng(0xBA7C4 + iter);
    const EngineConfig cfg = random_config(rng, kModes[iter % 4]);
    Engine e(cfg);
    const auto docs = random_docs(rng, 10 + rng.below(30), 50);
    for (const Doc& d : docs) e.ingest(d.text, d.id, d.opts());
    e.commit();
    if (rng.coin(1, 2)) e.ingest(random_text(rng, 30), 999);  // delta segment
    std::vector<Engine::QueryContext> ctxs;
    for (int i = 0; i < 20; ++i) ctxs.push_back(e.make_context());
    auto single = e.make_context();
    for (int trial = 0; trial < 6; ++trial) {
      const std::size_t nq = rng.below(22), nc = rng.below(21), no = rng.below(22);
      std::vector<std::string> qs;
      for (std::size_t i = 0; i < nq; ++i) qs.push_back(i > 0 && rng.coin(1, 6) ? qs[rng.below(i)] : random_query(rng, docs));
      const std::vector<std::string_view> views(qs.begin(), qs.end());
      std::vector<Engine::QueryResult> out(no);
      for (auto& o : out) o.snapshot_version = kUntouched;
      const std::uint32_t k = rng.below(kMaxK + 8);
      e.query_batch(views, k, std::span(ctxs).first(nc), out);
      const std::size_t n = std::min({nq, nc, no});
      for (std::size_t i = 0; i < n; ++i) {
        const auto ref = e.query(views[i], k, single);
        EXPECT_TRUE(same_result(ref, out[i])) << "iter " << iter << " trial " << trial << " q " << i << " of " << n;
        EXPECT_EQ(ref.snapshot_version, out[i].snapshot_version);
        answered += !ref.hits.empty();
      }
      for (std::size_t i = n; i < no; ++i) {
        EXPECT_EQ(out[i].snapshot_version, kUntouched) << "result past the shortest span was written";
        EXPECT_TRUE(out[i].hits.empty());
        ++untouched;
      }
      // The batch path allocates nothing, whatever the mode and shape.
      begin_alloc_count();
      e.query_batch(views, k, std::span(ctxs).first(nc), out);
      if (nq > 0) (void)e.query(views[0], k, single);
      EXPECT_EQ(end_alloc_count(), 0u) << "iter " << iter;
    }
  }
  EXPECT_GT(answered, 0);
  EXPECT_GT(untouched, 0);
}

// =============================== persistence ================================
bool reuse_expected(const EngineConfig& saved, bool saved_fully_committed, std::uint32_t n, const EngineConfig& load) {
  if (!saved_fully_committed || saved.bm25.k1 != load.bm25.k1 || saved.bm25.b != load.bm25.b) return false;
  if (!ivf_mode(load.dense_mode)) return true;
  const bool saved_has_ivf = ivf_mode(saved.dense_mode) && n > 0;
  return saved_has_ivf && (saved.dense_mode == DenseMode::IvfInt8) == (load.dense_mode == DenseMode::IvfInt8) &&
         saved.ivf.nlist == load.ivf.nlist && saved.ivf.train_iters == load.ivf.train_iters &&
         saved.ivf.max_train_points_per_list == load.ivf.max_train_points_per_list && saved.ivf.seed == load.ivf.seed;
}

::testing::AssertionResult same_engines(const Engine& a, const Engine& b, Rng& rng, const std::vector<Doc>& docs) {
  if (a.size() != b.size()) return ::testing::AssertionFailure() << "sizes " << a.size() << " vs " << b.size();
  for (std::uint32_t i = 0; i < a.size(); ++i) {
    const ChunkView x = a.chunk(i), y = b.chunk(i);
    if (x.text != y.text || x.doc_id != y.doc_id || x.ordinal != y.ordinal || x.source != y.source ||
        x.page_first != y.page_first || x.page_last != y.page_last)
      return ::testing::AssertionFailure() << "chunk " << i << " differs";
    if (std::memcmp(a.store().row(i), b.store().row(i), kDim * sizeof(float)) != 0)
      return ::testing::AssertionFailure() << "row " << i << " differs";
  }
  auto ca = a.make_context();
  auto cb = b.make_context();
  for (int qi = 0; qi < 10; ++qi) {
    const std::string q = random_query(rng, docs);
    const std::uint32_t k = rng.below(kMaxK + 1);
    if (auto r = same_result(a.query(q, k, ca), b.query(q, k, cb)); !r) return r << " for '" << q << "'";
  }
  return ::testing::AssertionSuccess();
}

TEST(PropPersist, RandomRoundTripsAndReloadsIntoOtherConfigsMatchFreshBuilds) {
  // Each save fsyncs twice (file and directory), 1-10 ms each on the test VM:
  // that, not the engine, is most of this test's time (130-530 ms measured at
  // 12 iterations), so it runs two configurations per dense mode.
  std::array<int, 2> reused{};  // [0] rebuilt on load, [1] reused the saved indexes
  for (std::uint32_t iter = 0; iter < 8; ++iter) {
    Rng rng(0x5A7E + iter);
    const EngineConfig cfg_a = random_config(rng, kModes[iter % 4]);
    const auto docs = random_docs(rng, rng.below(30), 50);  // includes an empty corpus
    Engine a(cfg_a);
    for (const Doc& d : docs) {
      a.ingest(d.text, d.id, d.opts());
      if (rng.coin(1, 10)) a.commit();
    }
    if (rng.coin(2, 3)) a.commit();
    const std::uint32_t n = a.size();
    const bool fully = a.committed() == n;
    const std::string tag = "iter " + std::to_string(iter) + " n " + std::to_string(n) + (fully ? " committed" : " delta");

    const auto path_a = temp_index("a");
    a.save(path_a);
    const IndexInfo info = Engine::inspect(path_a);
    EXPECT_EQ(info.format_version, io::kFormatVersion);
    EXPECT_EQ(info.dim, kDim);
    EXPECT_EQ(info.chunks, n);
    std::uint64_t text_bytes = 0;
    for (std::uint32_t i = 0; i < n; ++i) text_bytes += a.chunk(i).text.size();
    EXPECT_EQ(info.text_bytes, text_bytes);

    // Same config: reuses the saved indexes iff they cover everything, and
    // answers exactly like the original once that is committed too.
    Engine::LoadReport rep;
    const auto b = Engine::load(path_a, cfg_a, {}, &rep);
    EXPECT_EQ(rep.reused_indexes, reuse_expected(cfg_a, fully, n, cfg_a)) << tag;
    ++reused[rep.reused_indexes];
    EXPECT_EQ(b->committed(), n);
    if (!fully) a.commit();
    EXPECT_TRUE(same_engines(a, *b, rng, docs)) << tag;
    // save(load(save(A))) is byte-identical to save(A) once A is committed.
    const auto path_a2 = fully ? path_a : temp_index("a2"), path_b = temp_index("b");
    if (!fully) a.save(path_a2);
    b->save(path_b);
    EXPECT_TRUE(read_file(path_b) == read_file(path_a2)) << tag;

    // Another config (same chunker: it only shapes future ingests): the loaded
    // engine equals a fresh build with that config.
    EngineConfig cfg_c = random_config(rng, kModes[rng.below(4)]);
    cfg_c.chunker = cfg_a.chunker;
    if (rng.coin(1, 3)) cfg_c.bm25 = cfg_a.bm25;  // make index reuse likely in some iterations
    if (rng.coin(1, 2)) cfg_c.ivf = cfg_a.ivf;
    cfg_c.max_chunks = n + 1 + rng.below(64);
    const auto c = Engine::load(path_a, cfg_c, {}, &rep);
    EXPECT_EQ(rep.reused_indexes, reuse_expected(cfg_a, fully, n, cfg_c)) << tag;
    ++reused[rep.reused_indexes];
    Engine f(cfg_c);
    for (const Doc& d : docs) f.ingest(d.text, d.id, d.opts());
    f.commit();
    EXPECT_TRUE(same_engines(*c, f, rng, docs)) << tag;

    // ... and survives the trip back: reloading its file with the first
    // config gives the first engine's file again, byte for byte.
    const auto path_c = temp_index("c"), path_d = temp_index("d");
    c->save(path_c);
    Engine::load(path_c, cfg_a)->save(path_d);
    EXPECT_TRUE(read_file(path_d) == read_file(path_a2)) << tag;

    // Both keep ingesting identically (sources stay interned across the load).
    const Doc extra{random_text(rng, 40), 4242, rng.pick(source_names()), {}};
    EXPECT_EQ(c->ingest(extra.text, extra.id, extra.opts()), f.ingest(extra.text, extra.id, extra.opts()));
    c->commit();
    f.commit();
    EXPECT_TRUE(same_engines(*c, f, rng, docs)) << tag << " after a further ingest";
  }
  EXPECT_GT(reused[0], 0);
  EXPECT_GT(reused[1], 0);
}

// ========================= capacity, pages, errors ==========================
TEST(PropEngine, CapacityLimitsKeepAPrefixOfEachDocumentsChunks) {
  for (std::uint32_t iter = 0; iter < 40; ++iter) {
    Rng rng(0xCA9 + iter);
    EngineConfig cfg = random_config(rng, kModes[iter % 4]);
    cfg.max_chunks = 1 + rng.below(40);
    Engine e(cfg);
    const auto docs = random_docs(rng, 1 + rng.below(12), 40);
    for (const Doc& d : docs) {
      const auto want = chunk_ranges(d.text, cfg.chunker);
      const std::uint32_t before = e.size();
      const std::uint32_t added = e.ingest(d.text, d.id, d.opts());
      ASSERT_EQ(added, std::min<std::size_t>(want.size(), cfg.max_chunks - before)) << "iter " << iter;
      ASSERT_EQ(e.size(), before + added);
      for (std::uint32_t j = 0; j < added; ++j) {
        const ChunkView c = e.chunk(before + j);
        EXPECT_EQ(c.text, std::string_view(d.text).substr(want[j].byte_begin, want[j].byte_end - want[j].byte_begin));
        EXPECT_EQ(c.ordinal, j);
        EXPECT_EQ(c.doc_id, d.id);
      }
      if (rng.coin(1, 3)) e.commit();
    }
    e.commit();
    if (e.size() == cfg.max_chunks) {
      EXPECT_EQ(e.ingest("one more chunk of text", 77), 0u);
    }
    EXPECT_LE(e.size(), cfg.max_chunks);
    auto ctx = e.make_context();
    for (int qi = 0; qi < 5; ++qi) {
      const auto r = e.query(random_query(rng, docs), kMaxK, ctx);
      EXPECT_TRUE(well_formed(r.hits, e.size()));
      EXPECT_TRUE(well_formed(r.dense, e.size()));
      EXPECT_TRUE(well_formed(r.lexical, e.committed()));
    }
  }
}

TEST(PropEngine, PageSpansMatchByteOffsetsAndBadPageOffsetsChangeNothing) {
  for (std::uint32_t iter = 0; iter < 40; ++iter) {
    Rng rng(0x9A6E + iter);
    EngineConfig cfg;
    cfg.max_chunks = 2048;
    cfg.chunker = {1 + rng.below(12), 0};
    cfg.chunker.overlap_tokens = rng.below(cfg.chunker.window_tokens);
    Engine e(cfg);
    for (const Doc& d : random_docs(rng, 12, 60)) {
      const auto want = chunk_ranges(d.text, cfg.chunker);
      const std::uint32_t before = e.size();
      ASSERT_EQ(e.ingest(d.text, d.id, d.opts()), want.size());
      auto page_of = [&](std::uint32_t byte) -> std::uint32_t {
        if (d.pages.empty()) return 0;
        return static_cast<std::uint32_t>(std::upper_bound(d.pages.begin(), d.pages.end(), byte) - d.pages.begin());
      };
      for (std::uint32_t j = 0; j < want.size(); ++j) {
        const ChunkView c = e.chunk(before + j);
        EXPECT_EQ(c.page_first, page_of(want[j].byte_begin)) << "iter " << iter;
        EXPECT_EQ(c.page_last, page_of(want[j].byte_end - 1)) << "iter " << iter;
        EXPECT_EQ(c.source, d.source);
      }
      // Invalid page offsets throw before anything changes.
      std::vector<std::uint32_t> bad = d.pages.empty() ? std::vector<std::uint32_t>{0} : d.pages;
      switch (rng.below(3)) {
        case 0: bad[0] = 1 + rng.below(5); break;                                      // doesn't start at 0
        case 1: bad.push_back(bad.back()); break;                                      // not increasing
        default: bad.push_back(static_cast<std::uint32_t>(d.text.size()) + 1 + rng.below(3));  // past the end
      }
      const std::uint32_t size = e.size(), committed = e.committed();
      const std::size_t arena = e.arena_bytes();
      EXPECT_THROW(e.ingest(d.text, d.id, IngestOptions{.source = "never-interned", .page_starts = bad}), std::invalid_argument);
      EXPECT_EQ(e.size(), size);
      EXPECT_EQ(e.committed(), committed);
      EXPECT_EQ(e.arena_bytes(), arena);
    }
  }
}

TEST(PropEngine, QueriesBeyondTheTokenLimitEqualTheirFirst256Tokens) {
  for (std::uint32_t iter = 0; iter < 4; ++iter) {
    Rng rng(0x70C + iter);
    const EngineConfig cfg = random_config(rng, kModes[iter]);
    Engine e(cfg);
    const auto docs = random_docs(rng, 30, 60);
    for (const Doc& d : docs) e.ingest(d.text, d.id, d.opts());
    e.commit();
    auto c1 = e.make_context();
    auto c2 = e.make_context();
    for (int qi = 0; qi < 5; ++qi) {
      const std::string q = random_text(rng, 2 * Engine::kMaxQueryTokens + rng.below(300));  // no-break spaces glue words
      std::vector<Token> toks;
      for_each_token(q, [&](const Token& t) { toks.push_back(t); });
      ASSERT_GT(toks.size(), Engine::kMaxQueryTokens);
      const Token& last = toks[Engine::kMaxQueryTokens - 1];
      const std::string prefix = q.substr(0, last.offset + last.length);
      const auto a = e.query(q, kMaxK, c1), b = e.query(prefix, kMaxK, c2);
      EXPECT_TRUE(same_hits(a.hits, b.hits)) << "iter " << iter;
      EXPECT_TRUE(same_hits(a.dense, b.dense));
      EXPECT_TRUE(same_hits(a.lexical, b.lexical));
    }
  }
}

TEST(PropEngine, ReaderSlotsAreBoundedAndRecycled) {
  EngineConfig cfg;
  cfg.max_chunks = 8;
  cfg.prompt_capacity = 256;
  Engine e(cfg);
  e.ingest("alpha beta gamma", 0);
  e.commit();
  std::vector<Engine::QueryContext> ctxs;
  for (std::size_t i = 0; i < EpochDomain::kMaxReaders; ++i) ctxs.push_back(e.make_context());  // moves on regrowth
  EXPECT_EQ(e.live_readers(), EpochDomain::kMaxReaders);
  EXPECT_THROW((void)e.make_context(), std::runtime_error);
  for (auto& c : ctxs) EXPECT_EQ(e.query("alpha", 4, c).hits.size(), 1u);
  ctxs.pop_back();
  EXPECT_EQ(e.live_readers(), EpochDomain::kMaxReaders - 1);
  ctxs.push_back(e.make_context());
  EXPECT_THROW((void)e.make_context(), std::runtime_error);
  ctxs.clear();
  EXPECT_EQ(e.live_readers(), 0u);
  e.ingest("delta epsilon", 1);
  e.commit();
  EXPECT_EQ(e.reclaim(), 0u);  // nobody pins anything: every retired snapshot is freed
}

TEST(PropEngine, ConfigValidationAndLoadErrors) {
  EXPECT_THROW(Engine(EngineConfig{.retrieve_depth = 0}), std::invalid_argument);
  EXPECT_THROW(Engine(EngineConfig{.retrieve_depth = kMaxK + 1}), std::invalid_argument);
  EXPECT_NO_THROW(Engine(EngineConfig{.max_chunks = 1, .retrieve_depth = kMaxK}));
  EXPECT_THROW(Engine(EngineConfig{.rerank_candidates = Engine::kMaxRerank + 1}), std::invalid_argument);
  EXPECT_NO_THROW(Engine(EngineConfig{.max_chunks = 1, .rerank_candidates = Engine::kMaxRerank}));
  EXPECT_THROW(Engine(EngineConfig{.chunker = {0, 0}}), std::invalid_argument);
  // A one-chunk engine fills on its first chunk and keeps answering.
  Engine one(EngineConfig{.max_chunks = 1, .chunker = {2, 1}});
  EXPECT_EQ(one.ingest("alpha beta gamma delta", 3), 1u);
  EXPECT_EQ(one.ingest("epsilon", 4), 0u);
  one.commit();
  auto ctx = one.make_context();
  EXPECT_EQ(one.query("alpha", 1000, ctx).hits.size(), 1u);  // k far above MaxK is clamped
  EXPECT_TRUE(one.query("epsilon", 5, ctx).lexical.empty());  // the chunk that didn't fit isn't indexed
  const auto path = temp_index("one");
  one.save(path);
  EXPECT_THROW((void)Engine::load(path, EngineConfig{.max_chunks = 0}), std::invalid_argument);
  using SmallK = RagEngine<HashingEmbedder<kDim>, 16>;
  EXPECT_THROW((void)SmallK::load(path, EngineConfig{.retrieve_depth = 17}), std::invalid_argument);
  EXPECT_THROW((void)RagEngine<HashingEmbedder<64>>::load(path), std::invalid_argument);  // other dimension
  EXPECT_EQ(Engine::load(path, EngineConfig{.max_chunks = 1})->size(), 1u);
  EXPECT_THROW((void)Engine::inspect(temp_index("missing")), io::FormatError);
}

// ================================ regressions ===============================
// Bug 1: IvfIndex::train sampled m = min(n, nlist * max_train_points_per_list)
// points, so max_train_points_per_list = 0 gave m = 0, and reseeding an empty
// cluster from `perm[splitmix(rng) % m]` divided by zero: SIGFPE on the first
// commit() in an IVF mode. Fixed by sampling at least one point per list.
TEST(PropRegression, IvfWithZeroTrainingPointsPerListDoesNotDivideByZero) {
  Rng rng(1);
  const auto docs = random_docs(rng, 30, 40);
  for (DenseMode mode : {DenseMode::Ivf, DenseMode::IvfInt8}) {
    EngineConfig cfg;
    cfg.max_chunks = 1024;
    cfg.chunker = {8, 2};
    cfg.dense_mode = mode;
    cfg.ivf = {.nlist = 4, .train_iters = 5, .max_train_points_per_list = 0};
    cfg.nprobe = 4;
    EngineConfig flat = cfg;
    flat.dense_mode = mode == DenseMode::Ivf ? DenseMode::Exact : DenseMode::Int8Rerank;
    Engine e(cfg), ref(flat);
    for (const Doc& d : docs) { e.ingest(d.text, d.id, d.opts()); ref.ingest(d.text, d.id, d.opts()); }
    e.commit();
    ref.commit();
    auto c1 = e.make_context();
    auto c2 = ref.make_context();
    EXPECT_EQ(e.pin(c1)->ivf.nlist(), 4u);
    for (int qi = 0; qi < 8; ++qi) {
      const std::string q = random_query(rng, docs);
      const auto a = e.query(q, 10, c1), b = ref.query(q, 10, c2);
      if (mode == DenseMode::IvfInt8) {  // same int8 shortlist, same rerank
        EXPECT_TRUE(same_result(a, b)) << q;
        continue;
      }
      EXPECT_TRUE(same_hits(a.lexical, b.lexical)) << q;  // every list probed: the exact dense list
      ASSERT_EQ(a.dense.size(), b.dense.size()) << q;
      for (std::size_t i = 0; i < a.dense.size(); ++i) EXPECT_NEAR(a.dense[i].score, b.dense[i].score, 1e-6) << q;
    }
  }
  VectorStore<64> s(100);
  fill_clusters(rng, s, 100, 5, 0.5f);
  IvfIndex<64> ivf;
  ivf.build(s, 100, {.nlist = 8, .train_iters = 3, .max_train_points_per_list = 0});
  std::uint32_t total = 0;
  for (std::uint32_t l = 0; l < ivf.nlist(); ++l) total += ivf.list_size(l);
  EXPECT_EQ(total, 100u);
}

// Bug 2: EngineConfig::bm25 was not validated. Bm25Index::search detects a
// document's first touch with `acc[doc] == 0`, which needs every posting
// weight > 0. With k1 = -1 every weight is 0 (or -0), so a document was
// appended to Bm25Scratch::touched once per matching query term and the array
// (max_chunks entries) overflowed: a heap-buffer-overflow inside query().
// Fixed by rejecting k1 < 0, non-finite k1 and b outside [0, 1].
TEST(PropRegression, EngineRejectsBm25ParametersThatCannotKeepWeightsPositive) {
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  for (const Bm25Params p : {Bm25Params{-1.f, 0.75f}, Bm25Params{-0.01f, 0.75f}, Bm25Params{nan, 0.75f},
                             Bm25Params{inf, 0.75f}, Bm25Params{1.2f, -0.01f}, Bm25Params{1.2f, 1.01f},
                             Bm25Params{3.f, 3.f}, Bm25Params{1.2f, nan}})
    EXPECT_THROW(Engine(EngineConfig{.bm25 = p}), std::invalid_argument) << p.k1 << " " << p.b;
  // The edges of the valid range work and score like the spec.
  const std::vector<std::string> texts = {"alpha beta gamma delta", "alpha beta", "alpha", "beta gamma gamma gamma epsilon zeta",
                                          "alpha alpha beta"};
  for (const Bm25Params p : {Bm25Params{0.f, 0.f}, Bm25Params{0.f, 1.f}, Bm25Params{3.f, 1.f}, Bm25Params{1.2f, 0.f}}) {
    EngineConfig cfg;
    cfg.max_chunks = 8;
    cfg.chunker = {64, 0};
    cfg.bm25 = p;
    Engine e(cfg);
    std::vector<std::vector<std::string>> chunk_terms;
    for (std::uint32_t d = 0; d < 8; ++d) {
      e.ingest(texts[d % texts.size()], d);
      chunk_terms.push_back(spec::terms(texts[d % texts.size()]));
    }
    e.commit();
    auto ctx = e.make_context();
    const auto r = e.query("alpha beta gamma delta epsilon", kMaxK, ctx);
    const auto want = spec::Bm25(chunk_terms).scores(spec::terms("alpha beta gamma delta epsilon"), p.k1, p.b);
    ASSERT_EQ(r.lexical.size(), want.size());
    EXPECT_TRUE(same_ranking(want, r.lexical, 1e-5)) << p.k1 << " " << p.b;
    EXPECT_TRUE(well_formed(r.lexical, e.size()));
  }
}

// Offset of the first posting weight of the saved snapshot's BM25 index (see
// RagEngine::save and the Layout walker in forgery_tests.cpp).
std::size_t first_snapshot_weight(const std::vector<char>& b) {
  const auto n = peek<std::uint32_t>(b, 24);
  std::size_t at = 40 + std::size_t{40} * n + static_cast<std::size_t>(peek<std::uint64_t>(b, 32));
  at += 8 + 8 * static_cast<std::size_t>(peek<std::uint64_t>(b, at));   // source offsets
  at += 8 + static_cast<std::size_t>(peek<std::uint64_t>(b, at));       // source names
  at += std::size_t{n} * kDim * sizeof(float);                           // rows
  at += 8 + 16 * static_cast<std::size_t>(peek<std::uint64_t>(b, at));  // builder triples
  at += 8 + 4 * static_cast<std::size_t>(peek<std::uint64_t>(b, at));   // builder doc lengths
  at += 4 + 40;                                                          // builder doc count, IndexRecord
  at += 8 + 16 * static_cast<std::size_t>(peek<std::uint64_t>(b, at));  // term table
  at += 8 + 4;                                                           // term count, doc count
  at += 8 + 4 * static_cast<std::size_t>(peek<std::uint64_t>(b, at));   // posting doc ids
  return at + 8;                                                         // skip the weight vector's length
}

// Bug 3: Bm25Index::load checked every posting's doc id and range but not its
// weight. A forged index with a valid checksum and zero weights (here: one
// term whose 64 postings all name document 0) loaded fine, and the first
// query overflowed Bm25Scratch::touched for the reason above. Fixed by
// rejecting weights that are not finite and > 0 (freeze() never makes one).
TEST(PropRegression, Bm25LoaderRejectsNonPositiveOrNonFiniteWeights) {
  struct SlotBytes {
    std::uint64_t key;
    std::uint32_t offset, len;
  };
  const std::uint64_t key = hash_term("alpha");
  std::vector<SlotBytes> table(16, SlotBytes{0, 0, 0});
  table[key & 15] = {key, 0, 64};
  const std::vector<std::uint32_t> doc(64, 0);
  auto load_error = [&](float w) {
    const auto p = temp_index("weights");
    {
      io::Writer wr(p);
      wr.vec(table);
      wr.pod(std::uint64_t{1});
      wr.pod(std::uint32_t{1});
      wr.vec(doc);
      wr.vec(std::vector<float>(64, w));
      wr.commit();
    }
    return error_of([&] {
      io::Reader r(p);
      Bm25Index x;
      x.load(r);
      r.finish();
    });
  };
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  for (float w : {0.f, -0.f, -1.f, nan, inf, -inf})
    EXPECT_EQ(load_error(w), "Bm25Index::load: bad posting weight") << w;
  for (float w : {1.f, 1e-30f, std::numeric_limits<float>::denorm_min(), 1e30f}) EXPECT_EQ(load_error(w), "no exception") << w;

  // The same check guards a real engine file.
  Engine e(EngineConfig{.max_chunks = 16});
  e.ingest("alpha beta", 0);
  e.ingest("beta gamma", 1);
  e.commit();
  const auto path = temp_index("engine");
  e.save(path);
  auto bytes = read_file(path);
  const std::size_t w = first_snapshot_weight(bytes);
  ASSERT_EQ(peek<std::uint64_t>(bytes, w - 8), 4u);  // alpha:0, beta:0, beta:1, gamma:1
  ASSERT_GT(peek<float>(bytes, w), 0.f);
  poke<float>(bytes, w, 0.f);
  fix_checksum(bytes);
  write_file(path, bytes);
  EXPECT_EQ(error_of([&] { (void)Engine::load(path, EngineConfig{.max_chunks = 16}); }), "Bm25Index::load: bad posting weight");
}

// Bug 4: IvfIndex::search scored the last (list_size % 4) rows of each probed
// list with simd::dot (four accumulators) while every other row, and every
// row of the flat scan, gets a dot4 lane or the matching dot1. A row's score
// thus depended on its slot in its list: full-probe IVF did not reproduce the
// exact scan (46 of the 1280 hits below differed), and identical vectors
// scored differently, defeating the id tie-break (seven copies of one vector
// came back as ids 4, 5, 6 instead of 0, 1, 2).
// Fixed by scoring tail rows with simd::dot1, as VectorStore::search_exact does.
TEST(PropRegression, IvfScoresEveryRowExactlyLikeTheFlatScan) {
  Rng rng(2);
  constexpr std::size_t D = 64;
  VectorStore<D> s(203);
  std::array<float, D> v;
  for (int i = 0; i < 203; ++i) {
    for (auto& x : v) x = rng.normal();
    simd::normalize(v.data(), D);
    s.append(v);
  }
  IvfIndex<D> ivf;
  ivf.build(s, s.size(), {.nlist = 5});
  int differing = 0;
  for (int qi = 0; qi < 20; ++qi) {
    for (auto& x : v) x = rng.normal();
    simd::normalize(v.data(), D);
    TopK<64> exact(64), probed(64);
    s.search_exact(v.data(), 0, s.size(), exact);
    ivf.search(v.data(), ivf.nlist(), probed);
    const auto a = exact.sorted(), b = probed.sorted();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) differing += !(a[i].id == b[i].id && bits(a[i].score) == bits(b[i].score));
  }
  EXPECT_EQ(differing, 0);

  // Seven copies of one vector in a single list: block rows and tail rows tie
  // exactly, so the lowest ids win.
  VectorStore<D> dup(7);
  for (auto& x : v) x = rng.normal();
  simd::normalize(v.data(), D);
  for (int i = 0; i < 7; ++i) dup.append(v);
  IvfIndex<D> one;
  one.build(dup, 7, {.nlist = 1});
  for (int qi = 0; qi < 20; ++qi) {
    std::array<float, D> q;
    for (auto& x : q) x = rng.normal();
    simd::normalize(q.data(), D);
    TopK<3> top(3);
    one.search(q.data(), 1, top);
    const auto h = top.sorted();
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h[0].id, 0u);
    EXPECT_EQ(h[1].id, 1u);
    EXPECT_EQ(h[2].id, 2u);
  }
}

// Bug 6: detail::BufWriter::put passed an empty string_view's data() to
// memcpy. A default-constructed view's data() is null, and memcpy(dst,
// nullptr, 0) is undefined behaviour, so an empty system prompt
// (PromptParams{.system = {}}) or a `{}` question made every prompt build UB.
// Found by UBSan (prompt.hpp:60, "null pointer passed as argument 2"); in a
// sanitizer build (halt_on_error=1, as in CI) this test aborts on the old
// code. Fixed by returning early for empty writes, as io::Checksum does.
TEST(PropRegression, EmptyPromptPiecesWithNullDataAreNeverPassedToMemcpy) {
  const std::string_view none{};  // data() == nullptr
  ASSERT_EQ(none.data(), nullptr);
  const std::array<Hit, 1> hits{{{1.f, 0}}};
  auto lookup = [&](std::uint32_t) { return ChunkView{"text", 7, 0, none, 0, 0}; };
  std::array<char, 256> buf;
  const auto r = build_prompt(none, hits, lookup, PromptParams{.system = none}, buf);
  EXPECT_EQ(r.prompt, "\n\n### Context\n[1] (doc 7, chunk 0)\ntext\n\n### Question\n\n\n### Answer\n");
  // Through the engine: an empty system prompt and a null query.
  Engine e(EngineConfig{.max_chunks = 4, .prompt = {.system = none}});
  e.ingest("alpha beta", 0);
  e.commit();
  auto ctx = e.make_context();
  const auto q = e.query(none, 3, ctx);
  EXPECT_TRUE(q.hits.empty());
  EXPECT_EQ(q.prompt.prompt, "\n\n### Context\n### Question\n\n\n### Answer\n");
}

// Bug 5: IvfIndex::build(store, 0) on an index built before reset nlist and
// covered but kept the old centroids, lists and row storage, so has_fp32()
// stayed true and save() wrote "no lists, fp32 storage", which load() rejects
// as an inconsistent empty index. Fixed by resetting the whole index.
TEST(PropRegression, RebuildingAnIvfIndexOverNoRowsResetsItCompletely) {
  Rng rng(3);
  VectorStore<64> s(40);
  fill_clusters(rng, s, 40, 4, 0.4f);
  for (IvfStorage storage : {IvfStorage::Fp32, IvfStorage::Int8, IvfStorage::Both}) {
    IvfIndex<64> ivf;
    ivf.build(s, 40, {.nlist = 4, .storage = storage});
    ASSERT_GT(ivf.list_bytes(), 0u);
    ivf.build(s, 0);
    EXPECT_EQ(ivf.nlist(), 0u);
    EXPECT_EQ(ivf.covered(), 0u);
    EXPECT_FALSE(ivf.has_fp32());
    EXPECT_FALSE(ivf.has_int8());
    EXPECT_EQ(ivf.list_bytes(), 0u);
    const auto path = temp_index("ivf-empty");
    { io::Writer w(path); ivf.save(w); w.commit(); }
    EXPECT_EQ(error_of([&] {
                io::Reader r(path);
                IvfIndex<64> y;
                y.load(r, 40);
                r.finish();
              }),
              "no exception");
    ivf.build(s, 40, {.nlist = 4, .storage = storage});  // and it can be rebuilt again
    EXPECT_EQ(ivf.covered(), 40u);
  }
}

}  // namespace
