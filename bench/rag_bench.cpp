// rag_bench.cpp — Google Benchmark microbenchmarks, kernel -> component -> pipeline.
#include <benchmark/benchmark.h>

#include <random>

#if defined(__AVX512VNNI__)
#define RAG_VNNI_TAG "_vnni"
#else
#define RAG_VNNI_TAG ""
#endif

#include "fixtures.hpp"

using namespace rag;
using namespace rag::bench;

// ----------------------------- kernels --------------------------------------
template <float (*F)(const float*, const float*, std::size_t)>
static void BM_DotF32(benchmark::State& st) {
  const auto n = static_cast<std::size_t>(st.range(0));
  auto a = make_aligned_array<float>(n), b = make_aligned_array<float>(n);
  for (std::size_t i = 0; i < n; ++i) { a[i] = 0.001f * float(i); b[i] = 1.f - 0.0005f * float(i); }
  for (auto _ : st) {
    benchmark::DoNotOptimize(F(a.get(), b.get(), n));
    benchmark::ClobberMemory();
  }
  st.SetBytesProcessed(st.iterations() * std::int64_t(2 * n * sizeof(float)));
}
static float dot_scalar_noinline(const float* a, const float* b, std::size_t n) {
  // GCC auto-vectorises the plain loop at -O3 only with -ffast-math (FP
  // reassociation), so this is the honest "naive C++" baseline.
  return simd::dot_scalar(a, b, n);
}
BENCHMARK(BM_DotF32<dot_scalar_noinline>)->Name("Dot/f32/scalar")->Arg(384)->Arg(768);
#if defined(__AVX2__) && defined(__FMA__)
static float dot_avx2_f(const float* a, const float* b, std::size_t n) { return simd::dot_avx2(a, b, n); }
BENCHMARK(BM_DotF32<dot_avx2_f>)->Name("Dot/f32/avx2")->Arg(384)->Arg(768);
#endif
#if defined(__AVX512F__)
static float dot_avx512_f(const float* a, const float* b, std::size_t n) { return simd::dot_avx512(a, b, n); }
BENCHMARK(BM_DotF32<dot_avx512_f>)->Name("Dot/f32/avx512")->Arg(384)->Arg(768);
#endif

template <std::int32_t (*F)(const std::int8_t*, const std::int8_t*, std::size_t)>
static void BM_DotI8(benchmark::State& st) {
  const auto n = static_cast<std::size_t>(st.range(0));
  auto a = make_aligned_array<std::int8_t>(n), b = make_aligned_array<std::int8_t>(n);
  for (std::size_t i = 0; i < n; ++i) { a[i] = std::int8_t(i % 255 - 127); b[i] = std::int8_t((i * 7) % 255 - 127); }
  for (auto _ : st) {
    benchmark::DoNotOptimize(F(a.get(), b.get(), n));
    benchmark::ClobberMemory();
  }
  st.SetBytesProcessed(st.iterations() * std::int64_t(2 * n));
}
static std::int32_t dot_i8_scalar_f(const std::int8_t* a, const std::int8_t* b, std::size_t n) { return simd::dot_i8_scalar(a, b, n); }
BENCHMARK(BM_DotI8<dot_i8_scalar_f>)->Name("Dot/i8/scalar")->Arg(384);
#if defined(__AVX2__)
static std::int32_t dot_i8_avx2_f(const std::int8_t* a, const std::int8_t* b, std::size_t n) { return simd::dot_i8_avx2(a, b, n); }
BENCHMARK(BM_DotI8<dot_i8_avx2_f>)->Name("Dot/i8/avx2")->Arg(384);
#endif
#if defined(__AVX512BW__)
static std::int32_t dot_i8_avx512_f(const std::int8_t* a, const std::int8_t* b, std::size_t n) { return simd::dot_i8_avx512(a, b, n); }
BENCHMARK(BM_DotI8<dot_i8_avx512_f>)->Name("Dot/i8/avx512" RAG_VNNI_TAG)->Arg(384);
#endif

static void BM_TopKPush(benchmark::State& st) {
  std::vector<float> s(1 << 16);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u;
  for (auto& x : s) x = u(rng);
  for (auto _ : st) {
    TopK<64> tk(static_cast<std::size_t>(st.range(0)));
    for (std::uint32_t i = 0; i < s.size(); ++i) tk.push(s[i], i);
    benchmark::DoNotOptimize(tk.threshold());
  }
  st.SetItemsProcessed(st.iterations() * std::int64_t(s.size()));
}
BENCHMARK(BM_TopKPush)->Name("TopK/push")->Arg(10)->Arg(64);

// --------------------------- dense search -----------------------------------
static void BM_FlatExact(benchmark::State& st) {
  const auto n = static_cast<std::uint32_t>(st.range(0));
  auto& s = store(n);
  const auto& q = queries(n);
  std::size_t qi = 0;
  for (auto _ : st) {
    TopK<10> tk(10);
    s.search_exact(q.data() + (qi++ & 255) * kD, 0, n, tk);
    benchmark::DoNotOptimize(tk.threshold());
  }
  st.SetItemsProcessed(st.iterations() * n);
  st.SetBytesProcessed(st.iterations() * std::int64_t(n) * std::int64_t(kD * sizeof(float)));
}
BENCHMARK(BM_FlatExact)->Name("Dense/flat_f32")->Arg(10'000)->Arg(100'000)->Unit(benchmark::kMicrosecond);

static void BM_FlatInt8Rerank(benchmark::State& st) {
  const auto n = static_cast<std::uint32_t>(st.range(0));
  auto& s = store(n);
  const auto& q = queries(n);
  std::array<std::int8_t, kD> qq;
  std::size_t qi = 0;
  for (auto _ : st) {
    const float* qv = q.data() + (qi++ & 255) * kD;
    simd::quantize_i8(qv, qq.data(), kD);
    TopK<10> tk(10);
    TopK<128> cand(128);
    s.search_two_stage(qv, qq.data(), 0, n, cand, tk);
    benchmark::DoNotOptimize(tk.threshold());
  }
  st.SetItemsProcessed(st.iterations() * n);
  st.SetBytesProcessed(st.iterations() * std::int64_t(n) * std::int64_t(kD + sizeof(float)));
}
BENCHMARK(BM_FlatInt8Rerank)->Name("Dense/int8_rerank128")->Arg(10'000)->Arg(100'000)->Unit(benchmark::kMicrosecond);

static void BM_Ivf(benchmark::State& st) {
  constexpr std::uint32_t n = 100'000;
  auto& s = store(n);
  const auto& q = queries(n);
  static IvfIndex<kD> ivf = [&] { IvfIndex<kD> i; i.build(s, n, {.nlist = 256}); return i; }();
  // Measure recall@10 against exact search once, report it as a counter.
  double r = 0;
  for (int i = 0; i < 64; ++i) {
    TopK<10> e(10), a(10);
    s.search_exact(q.data() + i * kD, 0, n, e);
    ivf.search(q.data() + i * kD, static_cast<std::uint32_t>(st.range(0)), a);
    auto eh = e.sorted(); auto ah = a.sorted();
    for (auto& x : ah) r += std::any_of(eh.begin(), eh.end(), [&](const Hit& y) { return y.id == x.id; });
  }
  std::size_t qi = 0;
  for (auto _ : st) {
    TopK<10> tk(10);
    ivf.search(q.data() + (qi++ & 255) * kD, static_cast<std::uint32_t>(st.range(0)), tk);
    benchmark::DoNotOptimize(tk.threshold());
  }
  st.counters["recall@10"] = r / 640.0;
}
BENCHMARK(BM_Ivf)->Name("Dense/ivf256_100k/nprobe")->Arg(4)->Arg(16)->Arg(32)->Unit(benchmark::kMicrosecond);

// ------------------------------ lexical -------------------------------------
static void BM_Bm25(benchmark::State& st) {
  auto& e = engine(DenseMode::Exact);
  const auto qs = text_queries(256);
  std::vector<std::vector<std::uint64_t>> terms(qs.size());
  for (std::size_t i = 0; i < qs.size(); ++i)
    for_each_token(qs[i], [&](const Token& t) { if (!is_stopword(t.hash)) terms[i].push_back(t.hash); });
  Bm25Scratch sc(e.config().max_chunks);
  std::size_t qi = 0;
  for (auto _ : st) {
    TopK<32> tk(32);
    e.lexical_index().search(terms[qi++ & 255], sc, tk);
    benchmark::DoNotOptimize(tk.threshold());
  }
  st.counters["chunks"] = e.size();
  st.counters["postings"] = double(e.lexical_index().num_postings());
}
BENCHMARK(BM_Bm25)->Name("Lexical/bm25_query")->Unit(benchmark::kMicrosecond);

static void BM_TokenizeEmbed(benchmark::State& st) {
  HashingEmbedder<kD> emb;
  const auto qs = text_queries(256);
  std::array<Token, 256> toks;
  alignas(64) std::array<float, kD> v;
  std::size_t qi = 0;
  for (auto _ : st) {
    const auto& q = qs[qi++ & 255];
    const auto n = tokenize(q, toks);
    emb.embed(q, std::span<const Token>(toks.data(), n), v);
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(BM_TokenizeEmbed)->Name("Query/tokenize+embed");

// ----------------------------- end to end -----------------------------------
static void BM_EndToEnd(benchmark::State& st) {
  const auto mode = static_cast<DenseMode>(st.range(0));
  auto& e = engine(mode);
  auto ctx = e.make_context();
  const auto qs = text_queries(256);
  std::size_t qi = 0;
  for (auto _ : st) {
    auto r = e.query(qs[qi++ & 255], 8, ctx);
    benchmark::DoNotOptimize(r.prompt.prompt.data());
  }
  st.counters["chunks"] = e.size();
  st.SetLabel(mode == DenseMode::Exact ? "exact" : mode == DenseMode::Int8Rerank ? "int8+rerank" : "ivf");
}
BENCHMARK(BM_EndToEnd)->Name("Pipeline/query")->Arg(0)->Arg(1)->Arg(2)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
