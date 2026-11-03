// rag/simd.hpp — similarity kernels: fp32 dot, 4-row register-blocked dot,
// int8 dot (AVX-512 VNNI / AVX-512BW / AVX2), quantisation, normalisation.
//
// Dispatch is compile-time (build with -march=native or an explicit target):
// the best kernel is inlined straight into the scan loops, so there is no
// function-pointer call per row. Every ISA variant stays callable by name for
// benchmarking and for differential tests against the scalar reference.
//
// Preconditions: n % 64 == 0 for every SIMD variant (enforced at the type
// level by ValidDim<D>); scalar variants accept any n.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "rag/common.hpp"

namespace rag::simd {

// ------------------------------- scalar ------------------------------------
inline float dot_scalar(const float* RAG_RESTRICT a, const float* RAG_RESTRICT b, std::size_t n) noexcept {
  float s = 0.f;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline std::int32_t dot_i8_scalar(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
  std::int32_t s = 0;
  for (std::size_t i = 0; i < n; ++i) s += std::int32_t{a[i]} * std::int32_t{b[i]};
  return s;
}

// -------------------------------- AVX2 --------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
RAG_ALWAYS_INLINE float hsum256(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}
RAG_ALWAYS_INLINE std::int32_t hsum256_i32(__m256i v) noexcept {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

// 4 independent accumulators hide the 4-cycle FMA latency.
RAG_ALWAYS_INLINE float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  for (std::size_t i = 0; i < n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  return hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
}

// Register blocking: each query vector load is reused against 4 rows,
// cutting query-side load traffic 4x relative to 4 separate dot() calls.
RAG_ALWAYS_INLINE void dot4_avx2(const float* q, const float* r, std::size_t stride, std::size_t n,
                                 float* out) noexcept {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t i = 0; i < n; i += 8) {
    const __m256 qv = _mm256_loadu_ps(q + i);
    a0 = _mm256_fmadd_ps(_mm256_load_ps(r + i), qv, a0);
    a1 = _mm256_fmadd_ps(_mm256_load_ps(r + stride + i), qv, a1);
    a2 = _mm256_fmadd_ps(_mm256_load_ps(r + 2 * stride + i), qv, a2);
    a3 = _mm256_fmadd_ps(_mm256_load_ps(r + 3 * stride + i), qv, a3);
  }
  out[0] = hsum256(a0); out[1] = hsum256(a1); out[2] = hsum256(a2); out[3] = hsum256(a3);
}

// int8 x int8 via the sign trick: |a| (u8) * sign(b, a) (s8) -> maddubs.
// Requires values in [-127, 127] (quantize_i8 guarantees it) so that neither
// |a| nor the sign flip of b can overflow, and so maddubs' i16 pair sum
// (max 2*127*127 = 32258) never saturates.
RAG_ALWAYS_INLINE std::int32_t dot_i8_avx2(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
  for (std::size_t i = 0; i < n; i += 64) {
    const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
    const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
    const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(va0), _mm256_sign_epi8(vb0, va0));
    const __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(va1), _mm256_sign_epi8(vb1, va1));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
  }
  return hsum256_i32(_mm256_add_epi32(acc0, acc1));
}
#endif

// ------------------------------- AVX-512 ------------------------------------
#if defined(__AVX512F__)
RAG_ALWAYS_INLINE float dot_avx512(const float* a, const float* b, std::size_t n) noexcept {
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  for (std::size_t i = 0; i < n; i += 64) {
    s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
    s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

RAG_ALWAYS_INLINE void dot4_avx512(const float* q, const float* r, std::size_t stride, std::size_t n,
                                   float* out) noexcept {
  __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t i = 0; i < n; i += 16) {
    const __m512 qv = _mm512_loadu_ps(q + i);
    a0 = _mm512_fmadd_ps(_mm512_load_ps(r + i), qv, a0);
    a1 = _mm512_fmadd_ps(_mm512_load_ps(r + stride + i), qv, a1);
    a2 = _mm512_fmadd_ps(_mm512_load_ps(r + 2 * stride + i), qv, a2);
    a3 = _mm512_fmadd_ps(_mm512_load_ps(r + 3 * stride + i), qv, a3);
  }
  out[0] = _mm512_reduce_add_ps(a0); out[1] = _mm512_reduce_add_ps(a1);
  out[2] = _mm512_reduce_add_ps(a2); out[3] = _mm512_reduce_add_ps(a3);
}
#endif

#if defined(__AVX512BW__)
// Same sign trick as AVX2. AVX-512 has no vpsignb, so the sign transfer is a
// masked subtract driven by the sign bits of a.
RAG_ALWAYS_INLINE std::int32_t dot_i8_avx512(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
  const __m512i zero = _mm512_setzero_si512();
  __m512i acc = zero;
#if !defined(__AVX512VNNI__)
  const __m512i ones = _mm512_set1_epi16(1);
#endif
  for (std::size_t i = 0; i < n; i += 64) {
    const __m512i va = _mm512_loadu_si512(a + i);
    const __m512i vb = _mm512_loadu_si512(b + i);
    const __m512i ua = _mm512_abs_epi8(va);
    const __m512i sb = _mm512_mask_sub_epi8(vb, _mm512_movepi8_mask(va), zero, vb);
#if defined(__AVX512VNNI__)
    acc = _mm512_dpbusd_epi32(acc, ua, sb);  // u8*s8, 4-way sum into i32, one uop
#else
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(ua, sb), ones));
#endif
  }
  return _mm512_reduce_add_epi32(acc);
}
#endif

// ----------------------- best-available front ends --------------------------
RAG_ALWAYS_INLINE float dot(const float* a, const float* b, std::size_t n) noexcept {
#if defined(__AVX512F__)
  return dot_avx512(a, b, n);
#elif defined(__AVX2__) && defined(__FMA__)
  return dot_avx2(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

// out[j] = <q, r + j*stride>, j = 0..3. Rows must be 64-byte aligned.
RAG_ALWAYS_INLINE void dot4(const float* q, const float* r, std::size_t stride, std::size_t n, float* out) noexcept {
#if defined(__AVX512F__)
  dot4_avx512(q, r, stride, n, out);
#elif defined(__AVX2__) && defined(__FMA__)
  dot4_avx2(q, r, stride, n, out);
#else
  for (int j = 0; j < 4; ++j) out[j] = dot_scalar(q, r + j * stride, n);
#endif
}

// One lane of dot4: identical summation order (single accumulator, same
// vector width, same horizontal reduction), so dot1(q, r) is bit-identical to
// the matching dot4 output. Scans use it for tail rows so that a row's score
// never depends on where it fell in a 4-row block, or on batching. Note that
// dot() above uses four accumulators and does NOT match dot4 bitwise.
RAG_ALWAYS_INLINE float dot1(const float* q, const float* r, std::size_t n) noexcept {
#if defined(__AVX512F__)
  __m512 a = _mm512_setzero_ps();
  for (std::size_t i = 0; i < n; i += 16) a = _mm512_fmadd_ps(_mm512_loadu_ps(r + i), _mm512_loadu_ps(q + i), a);
  return _mm512_reduce_add_ps(a);
#elif defined(__AVX2__) && defined(__FMA__)
  __m256 a = _mm256_setzero_ps();
  for (std::size_t i = 0; i < n; i += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(r + i), _mm256_loadu_ps(q + i), a);
  return hsum256(a);
#else
  return dot_scalar(q, r, n);
#endif
}

RAG_ALWAYS_INLINE std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
#if defined(__AVX512BW__)
  return dot_i8_avx512(a, b, n);
#elif defined(__AVX2__)
  return dot_i8_avx2(a, b, n);
#else
  return dot_i8_scalar(a, b, n);
#endif
}

// Symmetric per-vector int8 quantisation into [-127, 127].
// Returns the dequantisation scale s such that x[i] ~= s * q[i].
inline float quantize_i8(const float* in, std::int8_t* out, std::size_t n) noexcept {
  float maxabs = 0.f;
  for (std::size_t i = 0; i < n; ++i) maxabs = std::fmax(maxabs, std::fabs(in[i]));
  if (maxabs == 0.f) {
    for (std::size_t i = 0; i < n; ++i) out[i] = 0;
    return 0.f;
  }
  const float inv = 127.f / maxabs;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = std::nearbyint(in[i] * inv);
    out[i] = static_cast<std::int8_t>(std::fmin(127.f, std::fmax(-127.f, v)));
  }
  return maxabs / 127.f;
}

// In-place L2 normalisation; leaves an all-zero vector untouched.
inline void normalize(float* v, std::size_t n) noexcept {
  const float nn = dot_scalar(v, v, n);
  if (nn <= 0.f) return;
  const float inv = 1.f / std::sqrt(nn);
  for (std::size_t i = 0; i < n; ++i) v[i] *= inv;
}

}  // namespace rag::simd
