// reference_tests.cpp — differential tests against a deliberately naive
// reference implementation.
//
// The reference is written from the *specification* of each stage, not from
// the optimized code: std::string words, std::map counts, double math, full
// sorts. Randomized corpora (mixed case, punctuation, UTF-8, stopwords,
// duplicates, empty documents) and random configurations are then run through
// both, and every stage is compared: tokenization, chunking, embeddings, BM25
// scores and rankings, exact dense rankings, and RRF fusion.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "rag.hpp"
#include "test_util.hpp"

using namespace rag;

namespace ref {

// ------------------------------ tokenizer ------------------------------------
// Spec: a word is a maximal run of [A-Za-z0-9] or bytes >= 0x80; ASCII letters
// are case-folded; everything else separates words.
struct Word {
  std::string text;  // case-folded
  std::size_t offset, length;
};

inline bool word_byte(unsigned char c) { return std::isalnum(c) || c >= 0x80; }

inline std::vector<Word> tokenize(const std::string& s) {
  std::vector<Word> out;
  std::size_t i = 0;
  while (i < s.size()) {
    if (!word_byte(static_cast<unsigned char>(s[i]))) { ++i; continue; }
    std::size_t j = i;
    std::string w;
    while (j < s.size() && word_byte(static_cast<unsigned char>(s[j]))) {
      const auto c = static_cast<unsigned char>(s[j]);
      w += (c < 0x80) ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
      ++j;
    }
    out.push_back({w, i, j - i});
    i = j;
  }
  return out;
}

inline const std::set<std::string>& stopwords() {
  static const std::set<std::string> s = {
      "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "i", "if", "in",
      "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "they", "this",
      "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
      "does", "do", "can", "about"};
  return s;
}

// ------------------------------- chunker -------------------------------------
// Spec: windows of `window` words starting every (window - overlap) words; the
// last window ends at the last word. Chunk text runs from the first word to
// the start of the word after the window (or end of text), minus trailing
// ASCII whitespace.
struct Chunk {
  std::size_t first_word, end_word;
  std::string text;
};

inline std::vector<Chunk> chunk(const std::string& s, const std::vector<Word>& w, std::size_t window,
                                std::size_t overlap) {
  std::vector<Chunk> out;
  if (w.empty()) return out;
  for (std::size_t b = 0;; b += window - overlap) {
    const std::size_t e = std::min(b + window, w.size());
    std::size_t end = e < w.size() ? w[e].offset : s.size();
    while (end > w[e - 1].offset + w[e - 1].length && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    out.push_back({b, e, s.substr(w[b].offset, end - w[b].offset)});
    if (e == w.size()) break;
  }
  return out;
}

// -------------------------------- BM25 ---------------------------------------
// Spec: Lucene BM25, idf = ln(1 + (N - df + 0.5) / (df + 0.5)), document
// length = number of non-stopword terms, each distinct query term counted once.
struct Bm25 {
  std::vector<std::map<std::string, int>> tf;
  std::vector<double> len;
  double avgdl = 1;
  std::map<std::string, int> df;

  explicit Bm25(const std::vector<std::vector<std::string>>& docs) {
    double total = 0;
    for (const auto& d : docs) {
      std::map<std::string, int> m;
      for (const auto& t : d) ++m[t];
      for (const auto& [t, c] : m) ++df[t];
      tf.push_back(m);
      len.push_back(static_cast<double>(d.size()));
      total += static_cast<double>(d.size());
    }
    avgdl = std::max(1.0, docs.empty() ? 1.0 : total / static_cast<double>(docs.size()));
  }

  std::vector<std::pair<double, std::uint32_t>> scores(const std::vector<std::string>& q, double k1 = 1.2,
                                                       double b = 0.75) const {
    const std::set<std::string> uq(q.begin(), q.end());
    const double N = static_cast<double>(tf.size());
    std::vector<std::pair<double, std::uint32_t>> out;
    for (std::uint32_t d = 0; d < tf.size(); ++d) {
      double s = 0;
      bool any = false;
      for (const auto& t : uq) {
        const auto it = tf[d].find(t);
        if (it == tf[d].end()) continue;
        any = true;
        const double n = df.at(t), f = it->second;
        const double idf = std::log(1 + (N - n + 0.5) / (n + 0.5));
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len[d] / avgdl));
      }
      if (any) out.push_back({s, d});
    }
    std::sort(out.begin(), out.end(), [](auto& x, auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    return out;
  }
};

// ------------------------------ embedding ------------------------------------
// Spec: signed feature hashing of every non-stopword unigram (weight 1) and of
// each adjacent pair of non-stopword unigrams (weight 0.5), bucket =
// floor(h32 * D / 2^32), sign = top bit, then L2 normalisation.
inline std::vector<double> embed(const std::vector<Word>& words, std::size_t D) {
  std::vector<double> v(D, 0.0);
  auto add = [&](std::uint64_t h, double w) {
    const std::size_t idx = static_cast<std::size_t>((std::uint64_t{static_cast<std::uint32_t>(h)} * D) >> 32);
    v[idx] += (h >> 63) ? -w : w;
  };
  std::uint64_t prev = 0;
  for (const auto& w : words) {
    if (stopwords().count(w.text)) continue;
    const std::uint64_t h = hash_term(w.text);
    add(h, 1.0);
    if (prev) add(detail::mix64(prev * 0x9E3779B97F4A7C15ull ^ h), 0.5);
    prev = h;
  }
  double nn = 0;
  for (double x : v) nn += x * x;
  if (nn > 0)
    for (double& x : v) x /= std::sqrt(nn);
  return v;
}

// --------------------------------- RRF ---------------------------------------
inline std::vector<std::pair<double, std::uint32_t>> rrf(const std::vector<std::uint32_t>& a,
                                                         const std::vector<std::uint32_t>& b, double k0 = 60) {
  std::map<std::uint32_t, double> s;
  for (std::size_t r = 0; r < a.size(); ++r) s[a[r]] += 1.0 / (k0 + static_cast<double>(r + 1));
  for (std::size_t r = 0; r < b.size(); ++r) s[b[r]] += 1.0 / (k0 + static_cast<double>(r + 1));
  std::vector<std::pair<double, std::uint32_t>> out;
  for (auto [id, sc] : s) out.push_back({sc, id});
  std::sort(out.begin(), out.end(), [](auto& x, auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
  return out;
}

}  // namespace ref

namespace {

constexpr std::size_t kD = 384;
using Engine = RagEngine<HashingEmbedder<kD>>;

// Random documents built to stress tokenization and scoring.
std::string random_doc(std::mt19937& rng, std::size_t words) {
  static const std::vector<std::string> vocab = [] {
    std::vector<std::string> v = {"Order", "book", "VWAP", "vwap", "imbalance", "AND", "spread", "bid", "ask",
                                  "queue", "latency", "ZX81Q", "café", "crème", "brûlée", "naïve", "42", "x86",
                                  "L2", "tick", "tick", "TICK", "venue", "it's"};
    for (const auto& w : ref::stopwords()) v.push_back(w);  // every stopword appears in the corpora
    return v;
  }();
  static const std::vector<std::string> seps = {" ", "  ", ", ", ". ", "\n", "\t", " -- ", "/", "(", ") ", "!?"};
  std::string s;
  for (std::size_t i = 0; i < words; ++i) {
    s += vocab[rng() % vocab.size()];
    s += seps[rng() % seps.size()];
  }
  return s;
}

std::vector<std::string> nonstop_terms(const std::string& text) {
  std::vector<std::string> t;
  for (const auto& w : ref::tokenize(text))
    if (!ref::stopwords().count(w.text)) t.push_back(w.text);
  return t;
}

// Rankings agree up to near-ties: the score at every rank matches, and the
// ids match wherever the score isn't (nearly) tied with a neighbour. `want`
// must be the reference's FULL ranking, not cut to the engine's depth: a tie
// with the first candidate past the cut decides who makes the list too.
void expect_same_ranking(const std::vector<std::pair<double, std::uint32_t>>& want, std::span<const Hit> got,
                         double tol, const std::string& what) {
  const std::size_t n = std::min(want.size(), got.size());  // callers check the sizes themselves
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(got[i].score, want[i].first, tol * std::max(1.0, std::abs(want[i].first))) << what << " rank " << i;
    const bool tied = (i > 0 && std::abs(want[i].first - want[i - 1].first) < 2 * tol) ||
                      (i + 1 < want.size() && std::abs(want[i].first - want[i + 1].first) < 2 * tol);
    if (!tied) {
      EXPECT_EQ(got[i].id, want[i].second) << what << " rank " << i << " score " << want[i].first << " got "
                                           << got[i].score << " neighbours "
                                           << (i > 0 ? want[i - 1].first : -1) << " / "
                                           << (i + 1 < want.size() ? want[i + 1].first : -1);
    }
  }
}

}  // namespace

TEST(Reference, StopwordSetMatchesSpecExactly) {
  std::set<std::uint64_t> want;
  for (const auto& w : ref::stopwords()) want.insert(hash_term(w));
  const std::set<std::uint64_t> got(kStopwords.begin(), kStopwords.end());
  EXPECT_EQ(got, want);
  EXPECT_EQ(kStopwords.size(), ref::stopwords().size());  // no duplicates either
}

TEST(Reference, TokenizerMatchesSpecOnRandomBytes) {
  std::mt19937 rng(1);
  for (int iter = 0; iter < 300; ++iter) {
    std::string s(rng() % 200, '\0');
    for (auto& c : s) c = static_cast<char>(rng() % 256);  // arbitrary bytes, including NUL and UTF-8
    if (iter % 2) s = random_doc(rng, rng() % 40);
    const auto want = ref::tokenize(s);
    std::vector<Token> got;
    for_each_token(s, [&](const Token& t) { got.push_back(t); });
    ASSERT_EQ(got.size(), want.size()) << "iter " << iter;
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_EQ(got[i].offset, want[i].offset);
      EXPECT_EQ(got[i].length, want[i].length);
      EXPECT_EQ(got[i].hash, hash_term(want[i].text));  // hash of the case-folded word
    }
  }
}

TEST(Reference, ChunkerMatchesSpec) {
  std::mt19937 rng(2);
  for (int iter = 0; iter < 300; ++iter) {
    const std::string s = random_doc(rng, rng() % 60);
    const std::size_t window = 1 + rng() % 20, overlap = rng() % window;
    const auto words = ref::tokenize(s);
    const auto want = ref::chunk(s, words, window, overlap);
    std::vector<Token> toks;
    for_each_token(s, [&](const Token& t) { toks.push_back(t); });
    std::vector<ChunkRange> got;
    chunk_tokens(s, toks, ChunkerConfig{static_cast<std::uint32_t>(window), static_cast<std::uint32_t>(overlap)},
                 [&](const ChunkRange& r) { got.push_back(r); });
    ASSERT_EQ(got.size(), want.size()) << "window " << window << " overlap " << overlap;
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_EQ(got[i].token_begin, want[i].first_word);
      EXPECT_EQ(got[i].token_end, want[i].end_word);
      EXPECT_EQ(s.substr(got[i].byte_begin, got[i].byte_end - got[i].byte_begin), want[i].text);
    }
  }
}

TEST(Reference, EmbedderMatchesSpec) {
  std::mt19937 rng(3);
  HashingEmbedder<kD> emb;
  for (int iter = 0; iter < 200; ++iter) {
    const std::string s = random_doc(rng, rng() % 50);
    std::vector<Token> toks;
    for_each_token(s, [&](const Token& t) { toks.push_back(t); });
    std::array<float, kD> got;
    emb.embed(s, toks, got);
    const auto want = ref::embed(ref::tokenize(s), kD);
    for (std::size_t d = 0; d < kD; ++d) ASSERT_NEAR(got[d], want[d], 1e-6) << "iter " << iter << " dim " << d;
  }
}

TEST(Reference, Bm25MatchesSpecOnRandomCorpora) {
  std::mt19937 rng(4);
  for (int iter = 0; iter < 40; ++iter) {
    const std::size_t ndocs = 1 + rng() % 60;
    std::vector<std::vector<std::string>> docs;
    Bm25Builder b;
    for (std::uint32_t d = 0; d < ndocs; ++d) {
      const auto text = random_doc(rng, rng() % 30);  // includes empty and stopword-only docs
      docs.push_back(nonstop_terms(text));
      std::vector<std::uint64_t> h;
      for (const auto& t : docs.back()) h.push_back(hash_term(t));
      b.add(d, h);
    }
    const ref::Bm25 want_index(docs);
    const auto idx = b.freeze();
    Bm25Scratch sc(static_cast<std::uint32_t>(ndocs));
    for (int qi = 0; qi < 10; ++qi) {
      const auto q = nonstop_terms(random_doc(rng, 1 + rng() % 5));
      std::vector<std::uint64_t> qh;
      for (const auto& t : q) qh.push_back(hash_term(t));
      TopK<64> tk(64);
      idx.search(qh, sc, tk);
      const auto want = want_index.scores(q);
      expect_same_ranking(want, tk.sorted(), 1e-5, "bm25 iter " + std::to_string(iter));
    }
  }
}

TEST(Reference, EnginePipelineMatchesSpecStageByStage) {
  std::mt19937 rng(5);
  for (int iter = 0; iter < 12; ++iter) {
    EngineConfig cfg;
    cfg.max_chunks = 4096;
    cfg.chunker = {static_cast<std::uint32_t>(4 + rng() % 20), 0};
    cfg.chunker.overlap_tokens = static_cast<std::uint32_t>(rng() % cfg.chunker.window_tokens);
    cfg.retrieve_depth = static_cast<std::uint32_t>(1 + rng() % 32);
    Engine e(cfg);

    // Reference view of the same corpus.
    std::vector<std::string> chunk_text;
    std::vector<std::vector<double>> chunk_vec;
    std::vector<std::vector<std::string>> chunk_terms;
    const std::size_t ndocs = 5 + rng() % 40;
    for (std::uint32_t d = 0; d < ndocs; ++d) {
      const std::string doc = random_doc(rng, rng() % 80);
      e.ingest(doc, d);
      const auto words = ref::tokenize(doc);
      for (const auto& c : ref::chunk(doc, words, cfg.chunker.window_tokens, cfg.chunker.overlap_tokens)) {
        chunk_text.push_back(c.text);
        const std::vector<ref::Word> cw(words.begin() + static_cast<std::ptrdiff_t>(c.first_word),
                                        words.begin() + static_cast<std::ptrdiff_t>(c.end_word));
        chunk_vec.push_back(ref::embed(cw, kD));
        std::vector<std::string> t;
        for (const auto& w : cw)
          if (!ref::stopwords().count(w.text)) t.push_back(w.text);
        chunk_terms.push_back(t);
      }
    }
    e.commit();
    ASSERT_EQ(e.size(), chunk_text.size()) << "iter " << iter;
    for (std::uint32_t i = 0; i < e.size(); ++i) ASSERT_EQ(e.chunk(i).text, chunk_text[i]);
    const ref::Bm25 bm25(chunk_terms);

    auto ctx = e.make_context();
    for (int qi = 0; qi < 15; ++qi) {
      const std::string q = random_doc(rng, 1 + rng() % 6);
      const std::uint32_t k = 1 + static_cast<std::uint32_t>(rng() % 10);
      const auto r = e.query(q, k, ctx);
      const std::string tag = "iter " + std::to_string(iter) + " q '" + q + "'";

      // Dense: brute-force cosine in double over the reference embeddings,
      // keeping only strictly positive scores (dense_min_score = 0).
      const auto qv = ref::embed(ref::tokenize(q), kD);
      std::vector<std::pair<double, std::uint32_t>> dense_want;
      for (std::uint32_t i = 0; i < chunk_vec.size(); ++i) {
        double s = 0;
        for (std::size_t d = 0; d < kD; ++d) s += qv[d] * chunk_vec[i][d];
        if (s > 1e-6) dense_want.push_back({s, i});
      }
      std::sort(dense_want.begin(), dense_want.end(),
                [](auto& x, auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
      ASSERT_EQ(r.dense.size(), std::min<std::size_t>(dense_want.size(), cfg.retrieve_depth)) << tag;
      expect_same_ranking(dense_want, r.dense, 1e-5, "dense " + tag);

      // Lexical: the reference BM25 over the reference chunk terms.
      const auto lex_want = bm25.scores(nonstop_terms(q));
      ASSERT_EQ(r.lexical.size(), std::min<std::size_t>(lex_want.size(), cfg.retrieve_depth)) << tag;
      expect_same_ranking(lex_want, r.lexical, 1e-5, "lexical " + tag);

      // Fusion: the reference RRF of the engine's own two lists (so near-tie
      // flips upstream can't make this comparison flaky).
      std::vector<std::uint32_t> a, b;
      for (const Hit& h : r.dense) a.push_back(h.id);
      for (const Hit& h : r.lexical) b.push_back(h.id);
      const auto fused_want = ref::rrf(a, b);
      ASSERT_EQ(r.hits.size(), std::min<std::size_t>(fused_want.size(), k)) << tag;
      expect_same_ranking(fused_want, r.hits, 1e-6, "fused " + tag);
    }
  }
}
