// rag_tests.cpp — GoogleTest suite: unit, differential (SIMD vs scalar),
// recall, edge-case, concurrency and zero-allocation tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "rag.hpp"
#include "synthetic.hpp"

// ============================================================================
// Global allocation counter: replaces every operator new overload so the
// "zero heap allocations on the query path" claim is checked, not asserted.
// ============================================================================
namespace {
std::atomic<bool> g_track{false};
std::atomic<std::size_t> g_allocs{0};
void* counted(std::size_t n, std::size_t align) {
  if (g_track.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
  void* p = align > alignof(std::max_align_t) ? std::aligned_alloc(align, (n + align - 1) / align * align)
                                              : std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc{};
  return p;
}
struct AllocScope {
  AllocScope() { g_allocs = 0; g_track = true; }
  ~AllocScope() { g_track = false; }
  std::size_t count() const { return g_allocs.load(); }
};
}  // namespace
void* operator new(std::size_t n) { return counted(n, 0); }
void* operator new[](std::size_t n) { return counted(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) { return counted(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return counted(n, static_cast<std::size_t>(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return counted(n, 0); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return counted(n, 0); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

volatile std::size_t benchmark_sink;  // defeats dead-code elimination of query results

using namespace rag;
using namespace std::string_view_literals;

namespace {
std::vector<std::string> words_of(std::string_view text) {
  std::vector<std::string> out;
  for_each_token(text, [&](const Token& t) { out.emplace_back(text.substr(t.offset, t.length)); });
  return out;
}
template <class TK>
std::vector<std::uint32_t> ids(TK& tk) {
  std::vector<std::uint32_t> v;
  for (const Hit& h : tk.sorted()) v.push_back(h.id);
  return v;
}
double recall(const std::vector<std::uint32_t>& truth, const std::vector<std::uint32_t>& got) {
  const std::set<std::uint32_t> t(truth.begin(), truth.end());
  std::size_t hit = 0;
  for (auto g : got) hit += t.count(g);
  return truth.empty() ? 1.0 : double(hit) / double(truth.size());
}
}  // namespace

// ================================ Tokenizer =================================
TEST(Tokenizer, SplitsOnNonAlnumAndLowercases) {
  constexpr auto text = "Hello, WORLD! bid/ask 42x  -- L2"sv;
  EXPECT_EQ(words_of(text), (std::vector<std::string>{"Hello", "WORLD", "bid", "ask", "42x", "L2"}));
  std::array<Token, 16> buf;
  const auto n = tokenize(text, buf);
  ASSERT_EQ(n, 6u);
  EXPECT_EQ(buf[1].hash, hash_term("world"));
  EXPECT_EQ(buf[0].offset, 0u);
  EXPECT_EQ(buf[4].offset, 22u);
  EXPECT_EQ(buf[4].length, 3u);
}

TEST(Tokenizer, Utf8BytesStayInsideWords) {
  const auto w = words_of("café crème-brûlée");
  ASSERT_EQ(w.size(), 3u);
  EXPECT_EQ(w[0], "café");
  EXPECT_EQ(w[2], "brûlée");
}

TEST(Tokenizer, BoundedOutputTruncatesInsteadOfOverflowing) {
  std::array<Token, 2> buf;
  EXPECT_EQ(tokenize("one two three four five", buf), 2u);
  EXPECT_EQ(tokenize("", buf), 0u);
  EXPECT_EQ(tokenize("  ,,, !!! ", buf), 0u);
}

TEST(Tokenizer, CompileTimeStopwords) {
  static_assert(is_stopword(hash_term("THE")));
  EXPECT_FALSE(is_stopword(hash_term("vwap")));
  EXPECT_TRUE(std::ranges::is_sorted(kStopwords));
}

// ================================= Chunker ==================================
namespace {
std::vector<ChunkRange> chunks_of(std::string_view text, ChunkerConfig cfg) {
  std::vector<Token> toks;
  for_each_token(text, [&](const Token& t) { toks.push_back(t); });
  std::vector<ChunkRange> out;
  chunk_tokens(text, toks, cfg, [&](const ChunkRange& r) { out.push_back(r); });
  return out;
}
}  // namespace

TEST(Chunker, SlidingWindowWithOverlap) {
  const auto c = chunks_of("a0 a1 a2 a3 a4 a5 a6 a7 a8 a9", {4, 1});
  ASSERT_EQ(c.size(), 3u);  // [0,4) [3,7) [6,10)
  EXPECT_EQ(c[0].token_begin, 0u); EXPECT_EQ(c[0].token_end, 4u);
  EXPECT_EQ(c[1].token_begin, 3u); EXPECT_EQ(c[1].token_end, 7u);
  EXPECT_EQ(c[2].token_begin, 6u); EXPECT_EQ(c[2].token_end, 10u);
}

TEST(Chunker, ExactFitProducesNoDuplicateTail) {
  EXPECT_EQ(chunks_of("a b c d e f g", {4, 1}).size(), 2u);  // [0,4) [3,7)
}

TEST(Chunker, ShortDocumentIsOneChunkAndEmptyIsNone) {
  EXPECT_EQ(chunks_of("just three words", {128, 32}).size(), 1u);
  EXPECT_TRUE(chunks_of("", {128, 32}).empty());
  EXPECT_TRUE(chunks_of(" ... ", {128, 32}).empty());
}

TEST(Chunker, KeepsTrailingPunctuationTrimsWhitespace) {
  constexpr auto text = "Spread widened (sharply).  \n"sv;
  const auto c = chunks_of(text, {128, 32});
  ASSERT_EQ(c.size(), 1u);
  EXPECT_EQ(text.substr(c[0].byte_begin, c[0].byte_end - c[0].byte_begin), "Spread widened (sharply).");
}

TEST(Chunker, RejectsInvalidConfig) {
  EXPECT_THROW((ChunkerConfig{4, 4}.validate()), std::invalid_argument);
  EXPECT_THROW((ChunkerConfig{0, 0}.validate()), std::invalid_argument);
  EXPECT_NO_THROW((ChunkerConfig{4, 3}.validate()));
}

// ================================== Arena ===================================
TEST(Arena, ViewsStayValidAcrossBlockGrowth) {
  Arena a(256);
  std::vector<std::pair<std::string, std::string_view>> kept;
  for (int i = 0; i < 500; ++i) {
    std::string s = "chunk-" + std::to_string(i) + std::string(static_cast<std::size_t>(i % 37), 'x');
    kept.emplace_back(s, a.copy(s));
  }
  EXPECT_GT(a.blocks(), 1u);
  for (auto& [orig, view] : kept) EXPECT_EQ(orig, view);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.allocate(8, 64)) % 64, 0u);
  EXPECT_EQ(a.allocate(4096, 8) != nullptr, true);  // larger than a block
}

// =================================== SIMD ===================================
class SimdDims : public ::testing::TestWithParam<std::size_t> {};

TEST_P(SimdDims, FloatKernelsMatchScalarReference) {
  const std::size_t n = GetParam();
  std::mt19937 rng(1);
  std::normal_distribution<float> g;
  alignas(64) std::vector<float> a(n), b(4 * n);  // vector storage is 16B aligned; use AlignedArray for dot4
  auto rows = make_aligned_array<float>(4 * n);
  for (auto& x : a) x = g(rng);
  for (std::size_t i = 0; i < 4 * n; ++i) rows[i] = b[i] = g(rng);
  const float ref = simd::dot_scalar(a.data(), b.data(), n);
  const float tol = 1e-4f * static_cast<float>(n);
  EXPECT_NEAR(simd::dot(a.data(), b.data(), n), ref, tol);
#if defined(__AVX2__) && defined(__FMA__)
  EXPECT_NEAR(simd::dot_avx2(a.data(), b.data(), n), ref, tol);
#endif
#if defined(__AVX512F__)
  EXPECT_NEAR(simd::dot_avx512(a.data(), b.data(), n), ref, tol);
#endif
  float out[4];
  simd::dot4(a.data(), rows.get(), n, n, out);
  for (int j = 0; j < 4; ++j) EXPECT_NEAR(out[j], simd::dot_scalar(a.data(), rows.get() + j * n, n), tol);
}

TEST_P(SimdDims, Int8KernelsAreBitExact) {
  const std::size_t n = GetParam();
  std::mt19937 rng(2);
  std::uniform_int_distribution<int> u(-127, 127);
  std::vector<std::int8_t> a(n), b(n);
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<std::int8_t>(u(rng)); b[i] = static_cast<std::int8_t>(u(rng)); }
  a[0] = 127; b[0] = -127; a[1] = -127; b[1] = -127;  // extremes: worst case for maddubs saturation
  const auto ref = simd::dot_i8_scalar(a.data(), b.data(), n);
  EXPECT_EQ(simd::dot_i8(a.data(), b.data(), n), ref);
#if defined(__AVX2__)
  EXPECT_EQ(simd::dot_i8_avx2(a.data(), b.data(), n), ref);
#endif
#if defined(__AVX512BW__)
  EXPECT_EQ(simd::dot_i8_avx512(a.data(), b.data(), n), ref);
#endif
  // All-extreme vectors: maximum magnitude per lane.
  std::fill(a.begin(), a.end(), std::int8_t{-127});
  std::fill(b.begin(), b.end(), std::int8_t{-127});
  EXPECT_EQ(simd::dot_i8(a.data(), b.data(), n), static_cast<std::int32_t>(127 * 127 * n));
}

INSTANTIATE_TEST_SUITE_P(Dims, SimdDims, ::testing::Values(64, 128, 384, 768, 1024, 4096));

TEST(Simd, QuantizeRoundTripWithinHalfStep) {
  std::mt19937 rng(3);
  std::normal_distribution<float> g;
  std::vector<float> v(384);
  for (auto& x : v) x = g(rng);
  std::vector<std::int8_t> q(384);
  const float s = simd::quantize_i8(v.data(), q.data(), 384);
  for (std::size_t i = 0; i < 384; ++i) {
    EXPECT_GE(q[i], -127);
    EXPECT_LE(std::fabs(v[i] - s * q[i]), 0.5f * s + 1e-6f);
  }
  std::vector<float> z(64, 0.f);
  std::vector<std::int8_t> zq(64, 5);
  EXPECT_EQ(simd::quantize_i8(z.data(), zq.data(), 64), 0.f);
  EXPECT_TRUE(std::all_of(zq.begin(), zq.end(), [](auto x) { return x == 0; }));
  simd::normalize(z.data(), 64);  // zero vector must stay zero, not NaN
  EXPECT_TRUE(std::all_of(z.begin(), z.end(), [](float x) { return x == 0.f; }));
}

// =================================== TopK ===================================
TEST(TopK, MatchesFullSortOnRandomData) {
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<Hit> all;
  for (std::uint32_t i = 0; i < 10000; ++i) all.push_back({u(rng), i});
  std::vector<Hit> ref = all;
  std::sort(ref.begin(), ref.end(), better);
  for (std::size_t k : {1u, 10u, 64u}) {
    TopK<64> tk(k);
    for (const Hit& h : all) tk.push(h.score, h.id);
    const auto got = tk.sorted();
    ASSERT_EQ(got.size(), k);
    for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(got[i], ref[i]);
  }
}

TEST(TopK, EdgeCases) {
  TopK<8> few(8);
  few.push(0.5f, 1); few.push(0.9f, 2);
  ASSERT_EQ(few.sorted().size(), 2u);  // fewer candidates than k
  EXPECT_EQ(few.sorted()[0].id, 2u);

  TopK<8> none(0);
  none.push(1.f, 1); none.push(INFINITY, 2);
  EXPECT_EQ(none.size(), 0u);  // k == 0 admits nothing

  TopK<8> clamp(1000);
  EXPECT_EQ(clamp.k(), 8u);  // k clamps to capacity

  TopK<2> ties(2);
  ties.push(1.f, 9); ties.push(1.f, 3); ties.push(1.f, 5); ties.push(1.f, 1);
  const auto t = ties.sorted();
  EXPECT_EQ(t[0].id, 1u);  // equal scores -> lowest ids win, deterministically
  EXPECT_EQ(t[1].id, 3u);

  TopK<4> nan(4);
  nan.push(NAN, 1); nan.push(0.1f, 2);
  ASSERT_EQ(nan.size(), 1u);  // NaN scores are rejected
}

// =============================== VectorStore ================================
constexpr std::size_t kD = 384;

TEST(VectorStore, AppendStopsAtCapacity) {
  VectorStore<64> s(3);
  std::array<float, 64> v{};
  v[0] = 1.f;
  EXPECT_TRUE(s.append(v)); EXPECT_TRUE(s.append(v)); EXPECT_TRUE(s.append(v));
  EXPECT_FALSE(s.append(v));
  EXPECT_EQ(s.size(), 3u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(s.row(1)) % 64, 0u);
}

TEST(VectorStore, ExactSearchMatchesBruteForceIncludingRemainderRows) {
  VectorStore<kD> s(1003);  // not a multiple of 4: exercises the dot4 tail
  synth::fill_clustered(s, 1003, 20, 0.3f);
  const auto q = synth::random_queries<kD>(10, 9, s, 0.2f);
  for (int qi = 0; qi < 10; ++qi) {
    const float* qv = q.data() + qi * kD;
    TopK<16> tk(16);
    s.search_exact(qv, 0, s.size(), tk);
    std::vector<Hit> ref;
    for (std::uint32_t i = 0; i < s.size(); ++i) ref.push_back({simd::dot_scalar(qv, s.row(i), kD), i});
    std::sort(ref.begin(), ref.end(), better);
    const auto got = tk.sorted();
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(got[i].id, ref[i].id);
      EXPECT_NEAR(got[i].score, ref[i].score, 1e-4f);
    }
  }
}

TEST(VectorStore, Int8TwoStageRecallAbove95Percent) {
  VectorStore<kD> s(20000);
  synth::fill_clustered(s, 20000, 100, 0.6f);
  const auto q = synth::random_queries<kD>(50, 11, s, 0.4f);
  double r = 0;
  for (int qi = 0; qi < 50; ++qi) {
    const float* qv = q.data() + qi * kD;
    TopK<10> exact(10), approx(10);
    TopK<128> cand(128);
    s.search_exact(qv, 0, s.size(), exact);
    std::array<std::int8_t, kD> qq;
    simd::quantize_i8(qv, qq.data(), kD);
    s.search_two_stage(qv, qq.data(), 0, s.size(), cand, approx);
    r += recall(ids(exact), ids(approx));
  }
  EXPECT_GE(r / 50, 0.95);
}

TEST(VectorStore, ConcurrentReadersNeverSeeTornRows) {
  constexpr std::uint32_t N = 20000;
  VectorStore<kD> s(N);
  std::atomic<bool> bad{false};
  std::atomic<bool> done{false};
  auto reader = [&] {
    while (!done.load(std::memory_order_acquire)) {
      const std::uint32_t m = s.size();
      if (m == 0) continue;
      const std::uint32_t i = m - 1;  // the most recently published row
      const float expect = static_cast<float>(i % 997 + 1);
      const float* r = s.row(i);
      for (std::size_t d = 0; d < kD; ++d)
        if (r[d] != expect) bad = true;
      if (s.qrow(i)[kD - 1] != 127) bad = true;  // constant row quantises to all 127
    }
  };
  std::vector<std::thread> rs;
  for (int t = 0; t < 3; ++t) rs.emplace_back(reader);
  std::array<float, kD> v;
  for (std::uint32_t i = 0; i < N; ++i) {
    v.fill(static_cast<float>(i % 997 + 1));
    ASSERT_TRUE(s.append(v));
  }
  done = true;
  for (auto& t : rs) t.join();
  EXPECT_FALSE(bad.load());
}

// ================================== IVF =====================================
TEST(Ivf, EveryRowInExactlyOneListAndFullProbeIsExact) {
  VectorStore<kD> s(5000);
  synth::fill_clustered(s, 5000, 40, 0.5f);
  IvfIndex<kD> ivf;
  ivf.build(s, s.size(), {.nlist = 32});
  std::uint32_t total = 0;
  for (std::uint32_t l = 0; l < ivf.nlist(); ++l) total += ivf.list_size(l);
  EXPECT_EQ(total, 5000u);
  const auto q = synth::random_queries<kD>(20, 3, s, 0.3f);
  for (int qi = 0; qi < 20; ++qi) {
    TopK<10> exact(10), approx(10);
    s.search_exact(q.data() + qi * kD, 0, s.size(), exact);
    ivf.search(q.data() + qi * kD, ivf.nlist(), approx);
    EXPECT_EQ(ids(exact), ids(approx));
  }
}

TEST(Ivf, RecallAtTenWithEighthOfListsProbed) {
  VectorStore<kD> s(20000);
  synth::fill_clustered(s, 20000, 100, 0.6f);
  IvfIndex<kD> ivf;
  ivf.build(s, s.size(), {.nlist = 64});
  const auto q = synth::random_queries<kD>(50, 5, s, 0.4f);
  double r = 0;
  for (int qi = 0; qi < 50; ++qi) {
    TopK<10> exact(10), approx(10);
    s.search_exact(q.data() + qi * kD, 0, s.size(), exact);
    ivf.search(q.data() + qi * kD, 8, approx);
    r += recall(ids(exact), ids(approx));
  }
  EXPECT_GE(r / 50, 0.90);
}

TEST(Ivf, EmptyAndTinyBuilds) {
  VectorStore<64> s(3);
  IvfIndex<64> ivf;
  ivf.build(s, 0);
  TopK<4> tk(4);
  std::array<float, 64> q{};
  q[0] = 1.f;
  ivf.search(q.data(), 8, tk);
  EXPECT_EQ(tk.size(), 0u);
  s.append(q); s.append(q);
  ivf.build(s, 2, {.nlist = 16});  // nlist clamps to n
  EXPECT_EQ(ivf.nlist(), 2u);
  ivf.search(q.data(), 8, tk);
  EXPECT_EQ(tk.size(), 2u);
}

namespace {
// IVF-SQ8 search as the engine runs it: int8 shortlist, then fp32 rerank.
template <std::size_t D, class TK>
void ivf_sq8_search(const VectorStore<D>& s, const IvfIndex<D>& ivf, const float* q, std::uint32_t nprobe,
                    std::size_t shortlist, TK& out) {
  std::array<std::int8_t, D> qq;
  simd::quantize_i8(q, qq.data(), D);
  TopK<1024> cand(shortlist);
  ivf.search_int8(q, qq.data(), nprobe, cand);
  for (const Hit& h : cand.sorted()) out.push(simd::dot(q, s.row(h.id), D), h.id);
}
}  // namespace

TEST(Ivf, Int8ListsWithRerankMatchFp32ListRecall) {
  VectorStore<kD> s(20000);
  synth::fill_clustered(s, 20000, 100, 0.6f);
  IvfIndex<kD> both;
  both.build(s, s.size(), {.nlist = 64, .storage = IvfStorage::Both});
  ASSERT_TRUE(both.has_fp32() && both.has_int8());
  const auto q = synth::random_queries<kD>(50, 5, s, 0.4f);
  double r_f32 = 0, r_i8 = 0, r_full = 0;
  for (int qi = 0; qi < 50; ++qi) {
    const float* qv = q.data() + qi * kD;
    TopK<10> exact(10), f32(10), i8(10), full(10);
    s.search_exact(qv, 0, s.size(), exact);
    both.search(qv, 8, f32);
    ivf_sq8_search(s, both, qv, 8, 128, i8);
    ivf_sq8_search(s, both, qv, both.nlist(), 128, full);
    const auto t = ids(exact);
    r_f32 += recall(t, ids(f32));
    r_i8 += recall(t, ids(i8));
    r_full += recall(t, ids(full));
  }
  EXPECT_GE(r_i8 / 50, r_f32 / 50 - 0.02);  // quantisation costs at most 2 points of recall
  EXPECT_GE(r_full / 50, 0.95);             // probing every list approaches exact search
}

TEST(Ivf, Int8StorageIsAQuarterOfFp32) {
  VectorStore<kD> s(1000);
  synth::fill_clustered(s, 1000, 10, 0.5f);
  IvfIndex<kD> f, q;
  f.build(s, 1000, {.nlist = 8, .storage = IvfStorage::Fp32});
  q.build(s, 1000, {.nlist = 8, .storage = IvfStorage::Int8});
  EXPECT_FALSE(f.has_int8());
  EXPECT_FALSE(q.has_fp32());
  EXPECT_EQ(f.list_bytes(), 1000u * kD * 4);
  EXPECT_EQ(q.list_bytes(), 1000u * (kD + 4));
  // Searching a storage kind the index doesn't have is a safe no-op.
  TopK<4> tk(4);
  std::array<std::int8_t, kD> qq{};
  q.search(s.row(0), 4, tk);
  f.search_int8(s.row(0), qq.data(), 4, tk);
  EXPECT_EQ(tk.size(), 0u);
}

// ================================== BM25 ====================================
namespace {
std::vector<std::uint64_t> terms(std::string_view text) {
  std::vector<std::uint64_t> t;
  for_each_token(text, [&](const Token& tok) { if (!is_stopword(tok.hash)) t.push_back(tok.hash); });
  return t;
}
}  // namespace

TEST(Bm25, ScoreMatchesHandComputedFormula) {
  Bm25Builder b;
  b.add(0, terms("alpha beta beta"));
  b.add(1, terms("beta gamma"));
  b.add(2, terms("gamma delta epsilon zeta"));
  const Bm25Params p{1.2f, 0.75f};
  const auto idx = b.freeze(p);
  Bm25Scratch sc(3);
  TopK<4> tk(4);
  const auto q = terms("beta");
  idx.search(q, sc, tk);
  const auto h = tk.sorted();
  ASSERT_EQ(h.size(), 2u);
  const double N = 3, df = 2, avgdl = 9.0 / 3;
  const double idf = std::log(1 + (N - df + 0.5) / (df + 0.5));
  auto bm = [&](double tf, double dl) { return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl)); };
  EXPECT_EQ(h[0].id, 0u);  // tf=2 beats tf=1
  EXPECT_NEAR(h[0].score, bm(2, 3), 1e-5);
  EXPECT_NEAR(h[1].score, bm(1, 2), 1e-5);
  EXPECT_EQ(idx.doc_freq(hash_term("gamma")), 2u);
  EXPECT_EQ(idx.doc_freq(hash_term("missing")), 0u);
}

TEST(Bm25, RareTermsOutweighCommonOnesAndDuplicatesCountOnce) {
  Bm25Builder b;
  for (std::uint32_t d = 0; d < 50; ++d) b.add(d, terms(d == 7 ? "market vwap" : "market price"));
  const auto idx = b.freeze();
  Bm25Scratch sc(50);
  TopK<4> a(4), dup(4);
  idx.search(terms("market vwap"), sc, a);
  idx.search(terms("vwap vwap vwap market"), sc, dup);
  const auto ha = a.sorted(), hd = dup.sorted();
  EXPECT_EQ(ha[0].id, 7u);
  EXPECT_GT(ha[0].score, 3 * ha[1].score);
  EXPECT_EQ(ha[0].score, hd[0].score);
  // Scratch accumulator must be fully restored to zero after each query.
  for (std::uint32_t d = 0; d < 50; ++d) EXPECT_EQ(sc.acc[d], 0.f);
}

TEST(Bm25, UnknownTermsAndEmptyIndex) {
  Bm25Index empty;
  Bm25Scratch sc(4);
  TopK<4> tk(4);
  empty.search(terms("anything"), sc, tk);
  EXPECT_EQ(tk.size(), 0u);
  Bm25Builder b;
  b.add(0, terms("alpha"));
  b.freeze().search(terms("omega"), sc, tk);
  EXPECT_EQ(tk.size(), 0u);
}

// ================================== RRF =====================================
TEST(Rrf, FusesByReciprocalRank) {
  const std::array<Hit, 3> dense{{{0.9f, 1}, {0.8f, 2}, {0.7f, 3}}};
  const std::array<Hit, 3> lex{{{12.f, 3}, {11.f, 1}, {10.f, 4}}};
  TopK<8> out(8);
  rrf_fuse(dense, lex, RrfParams{}, out);
  const auto h = out.sorted();
  ASSERT_EQ(h.size(), 4u);
  EXPECT_EQ(h[0].id, 1u); EXPECT_NEAR(h[0].score, 1.f / 61 + 1.f / 62, 1e-7);
  EXPECT_EQ(h[1].id, 3u); EXPECT_NEAR(h[1].score, 1.f / 63 + 1.f / 61, 1e-7);
  EXPECT_EQ(h[2].id, 2u);
  EXPECT_EQ(h[3].id, 4u);
}

TEST(Rrf, EmptySidesPassThroughRankOrder) {
  const std::array<Hit, 2> lex{{{5.f, 8}, {4.f, 9}}};
  TopK<8> out(8);
  rrf_fuse({}, lex, RrfParams{}, out);
  EXPECT_EQ(ids(out), (std::vector<std::uint32_t>{8, 9}));
}

// ================================= Prompt ===================================
namespace {
struct FakeChunks {
  std::vector<std::string> text;
  ChunkView operator()(std::uint32_t id) const { return {text[id], id + 100, 0}; }
};
}  // namespace

TEST(Prompt, FormatsCitationsAndKeepsQuestion) {
  FakeChunks fc{{"alpha text.", "beta text."}};
  const std::array<Hit, 2> hits{{{1.f, 1}, {0.5f, 0}}};
  std::array<char, 1024> buf;
  const auto r = build_prompt("what?", hits, fc, PromptParams{.system = "SYS"}, buf);
  EXPECT_EQ(r.prompt,
            "SYS\n\n### Context\n[1] (doc 101, chunk 0)\nbeta text.\n\n[2] (doc 100, chunk 0)\nalpha text.\n\n"
            "### Question\nwhat?\n\n### Answer\n");
  EXPECT_EQ(r.chunks_used, 2u);
  EXPECT_FALSE(r.truncated);
}

TEST(Prompt, TokenBudgetSkipsLargeChunkButPacksSmallerOne) {
  FakeChunks fc{{std::string(400, 'x'), "short"}};
  const std::array<Hit, 2> hits{{{1.f, 0}, {0.5f, 1}}};
  std::array<char, 4096> buf;
  const auto r = build_prompt("q", hits, fc, PromptParams{.max_context_tokens = 50}, buf);
  EXPECT_EQ(r.chunks_used, 1u);
  EXPECT_TRUE(r.truncated);
  EXPECT_NE(r.prompt.find("[1] (doc 101"), std::string_view::npos);
  EXPECT_LE(r.context_tokens, 50u);
}

TEST(Prompt, NeverWritesPastBuffer) {
  FakeChunks fc{{std::string(100, 'y')}};
  const std::array<Hit, 1> hits{{{1.f, 0}}};
  std::array<char, 200> buf;
  buf.fill('#');
  const PromptParams p{.system = "S"};
  const auto r = build_prompt("q", std::span(hits), fc, p, std::span(buf).first(64));
  EXPECT_TRUE(r.truncated);  // chunk does not fit; question still does
  EXPECT_LE(r.prompt.size(), 64u);
  EXPECT_NE(r.prompt.find("### Question\nq"), std::string_view::npos);
  EXPECT_TRUE(std::all_of(buf.begin() + 64, buf.end(), [](char c) { return c == '#'; }));
  const auto tiny = build_prompt("q", std::span(hits), fc, p, std::span(buf).first(8));
  EXPECT_TRUE(tiny.truncated);
  EXPECT_TRUE(tiny.prompt.empty());
}

// ================================= Engine ===================================
namespace {
using Engine = RagEngine<HashingEmbedder<kD>>;

const std::vector<std::string> kDocs = {
    "The limit order book aggregates resting bids and asks at each price level. Queue position "
    "determines fill priority under price-time matching.",
    "VWAP execution slices a parent order across the trading day in proportion to historical "
    "volume curves to minimise market impact.",
    "Order flow imbalance measures the net pressure of aggressive buyers versus sellers and "
    "predicts short horizon mid price moves.",
    "Kernel bypass networking with DPDK or Onload removes syscalls and interrupt latency from "
    "the market data feed handler.",
    "Risk checks enforce position limits, fat finger price bands and maximum order notional "
    "before an order reaches the exchange gateway.",
    "Instrument ZX81Q settles T plus one and trades only in the closing auction.",
};

// The engine owns atomics and an arena, so it is deliberately non-movable.
std::unique_ptr<Engine> make_engine(DenseMode mode, std::uint32_t max_chunks = 1024) {
  EngineConfig cfg;
  cfg.max_chunks = max_chunks;
  cfg.dense_mode = mode;
  cfg.ivf.nlist = 4;
  cfg.nprobe = 4;
  auto e = std::make_unique<Engine>(cfg);
  for (std::uint32_t i = 0; i < kDocs.size(); ++i) e->ingest(kDocs[i], i);
  e->commit();
  return e;
}
}  // namespace

class EngineModes : public ::testing::TestWithParam<DenseMode> {};

TEST_P(EngineModes, RetrievesTheRelevantDocumentFirst) {
  auto ep = make_engine(GetParam());
  auto& e = *ep;
  auto ctx = e.make_context();
  const std::vector<std::pair<std::string, std::uint32_t>> cases = {
      {"how does queue position affect fill priority in the order book", 0},
      {"slicing a parent order by historical volume", 1},
      {"aggressive buyers sellers imbalance predicts mid price", 2},
      {"kernel bypass DPDK feed handler latency", 3},
      {"fat finger price bands position limits", 4},
      {"when does ZX81Q settle", 5},
  };
  for (const auto& [q, want] : cases) {
    const auto r = e.query(q, 3, ctx);
    ASSERT_FALSE(r.hits.empty()) << q;
    EXPECT_EQ(e.chunk(r.hits[0].id).doc_id, want) << q;
    EXPECT_NE(r.prompt.prompt.find("### Question\n" + q), std::string_view::npos);
    EXPECT_NE(r.prompt.prompt.find(e.chunk(r.hits[0].id).text), std::string_view::npos);
  }
}

TEST_P(EngineModes, ZeroHeapAllocationsOnQueryPath) {
  auto ep = make_engine(GetParam());
  auto& e = *ep;
  auto ctx = e.make_context();
  (void)e.query("warm up the order book", 5, ctx);
  std::size_t allocs;
  {
    AllocScope scope;
    for (int i = 0; i < 200; ++i) {
      const auto r = e.query(i % 2 ? "order flow imbalance and vwap" : "the of and", 5, ctx);
      ::benchmark_sink = r.hits.size();
    }
    allocs = scope.count();
  }
  EXPECT_EQ(allocs, 0u);
}

TEST_P(EngineModes, UncommittedChunksAreVisibleToDenseSearch) {
  auto ep = make_engine(GetParam());
  auto& e = *ep;
  e.ingest("Iceberg orders display only a small visible quantity while hiding the reserve.", 99);
  auto ctx = e.make_context();
  const auto r = e.query("iceberg orders hidden reserve quantity", 3, ctx);
  ASSERT_FALSE(r.hits.empty());
  EXPECT_EQ(e.chunk(r.hits[0].id).doc_id, 99u);
  EXPECT_TRUE(r.lexical.empty());  // BM25 snapshot predates the doc until commit()
  e.commit();
  EXPECT_FALSE(e.query("iceberg orders hidden reserve quantity", 3, ctx).lexical.empty());
}

INSTANTIATE_TEST_SUITE_P(Modes, EngineModes,
                         ::testing::Values(DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf,
                                           DenseMode::IvfInt8),
                         [](const auto& info) {
                           switch (info.param) {
                             case DenseMode::Exact: return std::string("Exact");
                             case DenseMode::Int8Rerank: return std::string("Int8Rerank");
                             case DenseMode::Ivf: return std::string("Ivf");
                             default: return std::string("IvfInt8");
                           }
                         });

TEST(Engine, HybridSurfacesExactKeywordViaLexicalList) {
  auto ep = make_engine(DenseMode::Exact);
  auto& e = *ep;
  auto ctx = e.make_context();
  const auto r = e.query("ZX81Q", 3, ctx);
  ASSERT_FALSE(r.lexical.empty());
  EXPECT_EQ(e.chunk(r.lexical[0].id).doc_id, 5u);
  EXPECT_EQ(e.chunk(r.hits[0].id).doc_id, 5u);
}

TEST(Engine, StopwordOnlyAndEmptyQueriesReturnNoHitsButValidPrompt) {
  auto ep = make_engine(DenseMode::Exact);
  auto& e = *ep;
  auto ctx = e.make_context();
  for (auto q : {"the of and is"sv, ""sv, "?!"sv}) {
    const auto r = e.query(q, 5, ctx);
    EXPECT_TRUE(r.hits.empty()) << q;
    EXPECT_NE(r.prompt.prompt.find("### Answer"), std::string_view::npos);
  }
}

TEST(Engine, OverlongQueryIsTruncatedSafely) {
  auto ep = make_engine(DenseMode::Exact);
  auto& e = *ep;
  auto ctx = e.make_context();
  std::string q;
  for (int i = 0; i < 2000; ++i) q += "vwap ";
  const auto r = e.query(q, 5, ctx);
  EXPECT_FALSE(r.hits.empty());
}

TEST(Engine, CapacityLimitIsRespected) {
  EngineConfig cfg;
  cfg.max_chunks = 3;
  cfg.chunker = {8, 2};
  Engine e(cfg);
  std::string big;
  for (int i = 0; i < 100; ++i) big += "word" + std::to_string(i) + " ";
  EXPECT_EQ(e.ingest(big, 0), 3u);
  EXPECT_EQ(e.ingest("more text", 1), 0u);
  EXPECT_EQ(e.size(), 3u);
  e.commit();
  auto ctx = e.make_context();
  EXPECT_LE(e.query("word5", 10, ctx).hits.size(), 3u);
}

TEST(Engine, RejectsInvalidConfig) {
  EXPECT_THROW(Engine(EngineConfig{.max_chunks = 0}), std::invalid_argument);
  EXPECT_THROW(Engine(EngineConfig{.retrieve_depth = 65}), std::invalid_argument);
  EXPECT_THROW(Engine(EngineConfig{.chunker = {4, 4}}), std::invalid_argument);
}

TEST(Engine, ConcurrentIngestAndQuery) {
  EngineConfig cfg;
  cfg.max_chunks = 8192;
  cfg.chunker = {32, 8};
  Engine e(cfg);
  const auto corpus = synth::make_text_corpus(1500, 30, 64);
  e.ingest(corpus.docs[0], 0);
  e.commit();
  std::atomic<bool> done{false}, bad{false};
  std::atomic<std::uint64_t> queries{0};
  auto reader = [&](int seed) {
    auto ctx = e.make_context();
    std::mt19937 rng(static_cast<unsigned>(seed));
    while (!done.load(std::memory_order_acquire)) {
      const auto& words = corpus.topic_words[rng() % 30];
      const std::string q = words[rng() % 40] + " " + words[rng() % 40];
      const auto r = e.query(q, 8, ctx);
      const std::uint32_t n = e.size();
      for (const Hit& h : r.hits) {
        if (h.id >= n || e.chunk(h.id).text.empty()) bad = true;
      }
      queries.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::vector<std::thread> ts;
  for (int t = 0; t < 3; ++t) ts.emplace_back(reader, t);
  for (std::uint32_t d = 1; d < corpus.docs.size(); ++d) e.ingest(corpus.docs[d], d);
  done = true;
  for (auto& t : ts) t.join();
  EXPECT_FALSE(bad.load());
  EXPECT_GT(queries.load(), 0u);
  EXPECT_GT(e.size(), 1500u);
}

TEST(Engine, TopicRetrievalPrecisionOnSyntheticCorpus) {
  EngineConfig cfg;
  cfg.max_chunks = 16384;
  cfg.chunker = {64, 16};
  Engine e(cfg);
  const auto corpus = synth::make_text_corpus(2000, 50, 128);
  for (std::uint32_t d = 0; d < corpus.docs.size(); ++d) e.ingest(corpus.docs[d], d);
  e.commit();
  auto ctx = e.make_context();
  std::mt19937 rng(17);
  int correct = 0, total = 0;
  for (int i = 0; i < 200; ++i) {
    const auto t = static_cast<std::uint32_t>(rng() % 50);
    const auto& w = corpus.topic_words[t];
    const auto r = e.query(w[rng() % 40] + " " + w[rng() % 40] + " " + w[rng() % 40], 5, ctx);
    for (const Hit& h : r.hits) { correct += corpus.topic[e.chunk(h.id).doc_id] == t; ++total; }
  }
  EXPECT_GE(double(correct) / total, 0.95);  // precision@5 by topic
}

// ======================= Epoch-based reclamation ============================
TEST(Epoch, NestedGuardsStayPinnedUntilOutermostExits) {
  EpochDomain d;
  auto r = d.register_reader();
  {
    EpochDomain::Guard outer(d, *r.slot());
    const auto e1 = d.advance();
    EXPECT_FALSE(d.quiescent_since(e1));
    {
      EpochDomain::Guard inner(d, *r.slot());
    }
    EXPECT_FALSE(d.quiescent_since(e1));  // inner exit must not unpin the outer section
  }
  EXPECT_TRUE(d.quiescent_since(d.advance()));
}

TEST(Epoch, ReaderEnteringAfterAdvanceDoesNotBlockOlderRetirement) {
  EpochDomain d;
  auto r = d.register_reader();
  const auto e = d.advance();
  EpochDomain::Guard g(d, *r.slot());  // entered at epoch >= e: cannot hold the retired object
  EXPECT_TRUE(d.quiescent_since(e));
  EXPECT_FALSE(d.quiescent_since(d.advance()));
}

TEST(Epoch, SlotsAreRecycledAndExhaustionThrows) {
  EpochDomain d;
  std::vector<EpochDomain::Reader> rs;
  for (std::size_t i = 0; i < EpochDomain::kMaxReaders; ++i) rs.push_back(d.register_reader());
  EXPECT_EQ(d.readers(), EpochDomain::kMaxReaders);
  EXPECT_THROW((void)d.register_reader(), std::runtime_error);
  rs.pop_back();
  EXPECT_NO_THROW((void)d.register_reader());  // released slot is reusable
}

TEST(Engine, PinnedSnapshotSurvivesCommitAndIsReclaimedAfterRelease) {
  auto ep = make_engine(DenseMode::Ivf);
  auto& e = *ep;
  auto ctx = e.make_context();
  {
    const auto pinned = e.pin(ctx);
    const auto v = pinned->version;
    const auto docs = pinned->bm25.num_docs();
    e.ingest("Latency arbitrage exploits stale quotes across venues.", 42);
    e.commit();
    EXPECT_EQ(e.retired_snapshots(), 1u);  // still pinned: must not be freed
    EXPECT_EQ(pinned->version, v);          // old snapshot still intact and readable
    EXPECT_EQ(pinned->bm25.num_docs(), docs);
    // Nested query on the same context sees the new snapshot.
    EXPECT_EQ(e.query("stale quotes latency arbitrage", 3, ctx).snapshot_version, v + 1);
    EXPECT_EQ(e.reclaim(), 1u);
  }
  EXPECT_EQ(e.reclaim(), 0u);  // pin released: freed
}

TEST(Engine, ContextsReleaseReaderSlots) {
  auto ep = make_engine(DenseMode::Exact);
  auto& e = *ep;
  {
    auto a = e.make_context();
    auto b = e.make_context();
    EXPECT_EQ(e.live_readers(), 2u);
    auto c = std::move(a);  // moving keeps exactly one owner
    EXPECT_EQ(e.live_readers(), 2u);
  }
  EXPECT_EQ(e.live_readers(), 0u);
}

TEST(Engine, ConcurrentCommitAndQuery) {
  EngineConfig cfg;
  cfg.max_chunks = 8192;
  cfg.chunker = {32, 8};
  cfg.dense_mode = DenseMode::Ivf;
  cfg.ivf.nlist = 16;
  Engine e(cfg);
  const auto corpus = synth::make_text_corpus(800, 30, 64);
  e.ingest(corpus.docs[0], 0);
  e.commit();
  std::atomic<bool> done{false}, bad{false};
  std::atomic<std::uint64_t> queries{0}, max_version{0};
  auto reader = [&](int seed) {
    auto ctx = e.make_context();
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uint64_t last_version = 0;
    while (!done.load(std::memory_order_acquire)) {
      const auto& w = corpus.topic_words[rng() % 30];
      const auto r = e.query(w[rng() % 40] + " " + w[rng() % 40], 8, ctx);
      const std::uint32_t n = e.size();
      if (r.snapshot_version < last_version) bad = true;  // versions never go backwards per reader
      last_version = r.snapshot_version;
      for (const Hit& h : r.lexical) if (h.id >= n) bad = true;
      for (const Hit& h : r.hits) if (h.id >= n || e.chunk(h.id).text.empty()) bad = true;
      std::uint64_t m = max_version.load();
      while (last_version > m && !max_version.compare_exchange_weak(m, last_version)) {}
      queries.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::vector<std::thread> ts;
  for (int t = 0; t < 3; ++t) ts.emplace_back(reader, t);
  std::uint32_t commits = 1;
  for (std::uint32_t d = 1; d < corpus.docs.size(); ++d) {
    e.ingest(corpus.docs[d], d);
    if (d % 50 == 0) { e.commit(); ++commits; }
  }
  done = true;
  for (auto& t : ts) t.join();
  EXPECT_FALSE(bad.load());
  EXPECT_GT(queries.load(), 0u);
  EXPECT_GT(max_version.load(), 1u);  // readers observed commits that happened mid-flight
  EXPECT_EQ(e.reclaim(), 0u);          // no readers left: everything retired is freed
  (void)commits;
}
