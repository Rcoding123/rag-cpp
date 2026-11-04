// rag/common.hpp — shared primitives: cache-line constants, aligned buffers, hints.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(_WIN32)
#include <malloc.h>  // _aligned_malloc / _aligned_free (MSVC has no std::aligned_alloc)
#endif
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>  // _mm_prefetch
#endif

namespace rag {

// std::hardware_destructive_interference_size is 64 on every x86-64 part we
// target, but GCC warns that using it in a header is an ABI hazard
// (-Winterference-size), so we spell the value out and check it where we can.
inline constexpr std::size_t kCacheLine = 64;
#if defined(__cpp_lib_hardware_interference_size) && defined(__x86_64__) && !defined(__GNUC__)
static_assert(std::hardware_destructive_interference_size == kCacheLine);
#endif

inline constexpr std::size_t kHugePage = std::size_t{2} << 20;  // 2 MiB

#if defined(__GNUC__) || defined(__clang__)
#define RAG_ALWAYS_INLINE inline __attribute__((always_inline))
#define RAG_RESTRICT __restrict__
#define RAG_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#elif defined(_MSC_VER)
#define RAG_ALWAYS_INLINE __forceinline
#define RAG_RESTRICT __restrict
#if defined(_M_X64) || defined(_M_IX86)
#define RAG_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#else
#define RAG_PREFETCH(p) ((void)0)
#endif
#else
#define RAG_ALWAYS_INLINE inline
#define RAG_RESTRICT
#define RAG_PREFETCH(p) ((void)0)
#endif

// The AVX2 kernels also need FMA. GCC and Clang define __FMA__ (with
// -march=x86-64-v3 or native on any AVX2 CPU); MSVC's /arch:AVX2 generates
// FMA instructions but defines no macro for them.
#if defined(__AVX2__) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__)))
#define RAG_AVX2_FMA 1
#endif

// Dimensions must be a multiple of 64 so every SIMD kernel runs with no
// scalar tail: 64 fp32 = 4 AVX-512 regs; 64 int8 = 1 AVX-512 reg / 2 AVX2 regs.
template <std::size_t D>
concept ValidDim = (D > 0) && (D % 64 == 0);

// ---------------------------------------------------------------------------
// AlignedArray: one up-front allocation, cache-line (or huge-page) aligned.
//
// Init::Prefault zeroes the whole array now, so every page is mapped before
// the first query. Use it for memory that queries *write* at arbitrary
// offsets (scratch accumulators, prompt buffers), where a first-touch page
// fault would land on a query's latency.
//
// Init::Lazy leaves the memory uninitialised and unmapped until first write.
// Use it for storage the writer always fills before publishing it (vectors,
// chunk metadata, index arrays): readers then never touch an unwritten page,
// pages are faulted in by the writer as it goes, and memory for unused
// capacity is never committed at all.
// ---------------------------------------------------------------------------
enum class Init : std::uint8_t { Prefault, Lazy };

struct FreeDeleter {
  void operator()(void* p) const noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

template <class T>
using AlignedArray = std::unique_ptr<T[], FreeDeleter>;

template <class T>
[[nodiscard]] AlignedArray<T> make_aligned_array(std::size_t n, Init init = Init::Prefault) {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                "AlignedArray holds raw POD storage only");
  const std::size_t bytes = n * sizeof(T);
  // Big buffers (embedding matrices) get 2 MiB alignment so the kernel can
  // back them with transparent huge pages: one TLB entry per 2 MiB instead of
  // 512, which matters when a flat scan streams hundreds of MB.
  const std::size_t align = bytes >= kHugePage ? kHugePage : kCacheLine;
  const std::size_t rounded = ((bytes ? bytes : 1) + align - 1) / align * align;
#if defined(_WIN32)
  void* p = _aligned_malloc(rounded, align);
#else
  void* p = std::aligned_alloc(align, rounded);
#endif
  if (!p) throw std::bad_alloc{};
#if defined(__linux__) && defined(MADV_HUGEPAGE) && !defined(RAG_NO_HUGEPAGES)
  if (align == kHugePage) ::madvise(p, rounded, MADV_HUGEPAGE);
#endif
  if (init == Init::Prefault) std::memset(p, 0, rounded);  // map every page now (cold path)
  return AlignedArray<T>(static_cast<T*>(p));
}

}  // namespace rag
