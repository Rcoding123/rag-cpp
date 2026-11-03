# rag-cpp — header-only, low-latency Retrieval-Augmented Generation in C++20

[![CI](https://github.com/Rcoding123/rag-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Rcoding123/rag-cpp/actions/workflows/ci.yml)

Hybrid retrieval (dense ANN + BM25, fused by RRF) with a zero-allocation query
path, SIMD similarity kernels (AVX-512 VNNI / AVX2 / scalar), and lock-free
single-writer / multi-reader ingestion.

```
INGEST (1 writer)                                   QUERY (N readers, 0 allocations)
text ─► Arena (stable string_views)                 query ─► tokenize   (stack, constexpr LUT)
     ─► tokenize ─► sliding-window chunks                 ─► embed      (stack, L1)
     ─► Embedder ─► VectorStore.append ─┐                 ├─► dense:  Exact | Int8+rerank | IVF + delta scan
     ─► Bm25Builder.add                 │                 └─► BM25:   impact-scored postings, dense accumulator
commit() ─► build Snapshot ─► atomic swap  │                      ─► RRF fusion (stack)
                                        │                      ─► prompt into preallocated buffer ─► Generator
        size.store(n+1, release) ───────┴──► size.load(acquire): new chunks searchable immediately (dense)
```

| Header | Role | Key latency technique |
|---|---|---|
| `common.hpp` | aligned buffers | 64 B / 2 MiB alignment, `MADV_HUGEPAGE`, pre-faulted pages |
| `arena.hpp` | document text | bump-pointer, blocks never move → lock-free readers |
| `tokenizer.hpp` | terms | constexpr 256-byte class/lowercase LUT, constexpr FNV-1a, compile-time stopword table |
| `chunker.hpp` | chunking | zero-copy byte ranges, token-window with overlap |
| `embedder.hpp` | `Embedder` concept + hashing embedder | signed feature hashing, Lemire fast-range (no division) |
| `simd.hpp` | kernels | 4-accumulator FMA, 4-row register blocking, int8 sign-trick / VNNI `vpdpbusd` |
| `topk.hpp` | selection | stack `std::array` heap, cached threshold, `[[likely]]` reject |
| `vector_store.hpp` | dense storage | SoA fp32 + int8 + scale, SWMR via release/acquire, padded atomic |
| `ivf_index.hpp` | ANN | spherical k-means, CSR lists with contiguous reordered vectors |
| `bm25.hpp` | lexical | precomputed per-posting BM25 impact, open-addressing dictionary, touched-list reset |
| `fusion.hpp` | RRF | linear id-match over stack array (K ≤ 64) |
| `prompt.hpp` | context assembly | fixed buffer, reserved question space, greedy token-budget packing |
| `epoch.hpp` | snapshot reclamation | per-reader padded epoch slots, seq_cst publish/scan |
| `engine.hpp` | pipeline | per-thread `QueryContext` holds all scratch; atomic snapshot swap |

## Build, test, benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # -DRAG_ARCH=x86-64-v3 for portable AVX2
cmake --build build -j
./build/rag_tests          # GoogleTest suite
./build/rag_bench          # Google Benchmark
./build/rag_latency 20000  # p50/p99/p99.9 per dense mode
./build/rag_demo
```

Sanitizer builds: `-DRAG_SANITIZE=ON` (ASan+UBSan), `-DRAG_TSAN=ON` (ThreadSanitizer).

## Verified (Xeon @ 2.8 GHz, 4 vCPU, AVX-512 VNNI, GCC 13.3 `-O3 -march=native`)

* All tests pass on GCC 13 Release, GCC 13 ASan+UBSan, and Clang 18; the
  two concurrency tests are clean under ThreadSanitizer.
* Zero `operator new` calls across 200 queries in every dense mode (checked by
  a global allocation counter in the test binary).

See `bench_results.txt` and `latency_results.txt` for raw output.

| Benchmark (D = 384) | Result |
|---|---|
| fp32 dot: scalar / AVX2 / AVX-512 | 419 / 16.3 / 15.1 ns |
| int8 dot: scalar / AVX2 / AVX-512 VNNI | 46.1 / 12.8 / 7.3 ns |
| Flat fp32 scan, 100k rows (154 MB) | 16.3 ms (9.0 GB/s, DRAM-bound) |
| int8 scan + fp32 rerank of 128, 100k rows | 3.3 ms (4.9x faster) |
| IVF-256, nprobe=4, 100k rows | 259 µs |
| BM25 query, 60k chunks / 5.07M postings | 7.95 µs |
| tokenize + embed query | 0.67 µs |

End-to-end query (60k chunks, k = 8, pinned core, 3000 queries):

| mode | p50 | p99 | p99.9 | QPS/core |
|---|---|---|---|---|
| exact | 9.6 ms | 12.5 ms | 14.8 ms | 101 |
| int8 + rerank | 2.2 ms | 2.8 ms | 4.3 ms | 469 |
| ivf (nprobe 16) | 0.61 ms | 1.03 ms | 1.33 ms | 1587 |

Caveats: the IVF benchmark's recall@10 = 1.0 reflects well-separated
synthetic clusters. Real sentence embeddings need a larger nprobe for the same
recall, so measure on your own data. `HashingEmbedder` is a lexical-ish
stand-in. For semantic recall, plug a neural encoder into the `Embedder`
concept.

## Thread-safety contract

* **Writer side:** `ingest()` and `commit()` run on one thread at a time.
* **Readers:** `query()` runs on any number of threads, concurrently with both
  `ingest()` and `commit()`. Each thread owns a `QueryContext`, up to 128 live
  at once.
* **How `commit()` stays safe:**
  1. It builds a new immutable `Snapshot` (BM25 plus IVF) while queries keep
     running.
  2. It publishes the snapshot with one atomic exchange.
  3. It retires the old snapshot through epoch-based reclamation
     (`epoch.hpp`). Each query pins a snapshot by writing to its own
     cache-line slot, so pinning costs no read-modify-write and no shared-line
     write.
* **Consistent reads:** `engine.pin(ctx)` holds one snapshot across several
  reads. Each `QueryResult` reports the `snapshot_version` it was served from.
* **Lifetime:** destroy every `QueryContext` before the engine.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, with
`-Werror` on all six jobs:

| Job | Covers |
|---|---|
| gcc-native | the runner's best ISA (AVX-512 when present), plus a demo and benchmark smoke run |
| gcc-avx2 | the `-march=x86-64-v3` AVX2 + FMA kernels |
| gcc-scalar | the `-march=x86-64` scalar fallback |
| clang-native | Clang 18 |
| gcc-asan-ubsan | AddressSanitizer + UndefinedBehaviorSanitizer |
| gcc-tsan | ThreadSanitizer, with the concurrency tests repeated 25 times |
