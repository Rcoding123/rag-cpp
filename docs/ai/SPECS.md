# Specs: toolchain, machine, libraries, build

Everything here was read off the machine and the build files, not
remembered. If you rebuild elsewhere, re-measure before quoting numbers
(see CONVENTIONS.md).

## Language and build

| Item | Value |
|---|---|
| Language | C++20 (`CMAKE_CXX_STANDARD 20`, required, no GNU extensions) |
| Library form | header-only: `include/rag/` (engine) and `include/rag/agent/` (agent); CMake `INTERFACE` target `rag` |
| Build system | CMake ≥ 3.20 (measured with 3.28.3); default build type `Release` (`-O3 -DNDEBUG`) |
| Warnings | `-Wall -Wextra -Wpedantic -Wconversion -Wno-sign-conversion`; CI adds `-Werror` |
| ISA | `-march=${RAG_ARCH}`, default `native`; CI also builds `x86-64-v3` (AVX2) and `x86-64` (scalar) |
| SIMD paths | AVX-512 (+VNNI for int8), AVX2 + FMA, scalar fallback; chosen at compile time |

CMake options:

| Option | Default | Effect |
|---|---|---|
| `RAG_ARCH` | `native` | value of `-march` |
| `RAG_SANITIZE` | OFF | ASan + UBSan (`-fsanitize=address,undefined`) |
| `RAG_TSAN` | OFF | ThreadSanitizer |
| `RAG_HUGEPAGES` | ON | transparent huge pages for large arrays (`MADV_HUGEPAGE`) |
| `RAG_FUZZ` | OFF | libFuzzer targets (Clang only) |
| `RAG_PDF` | ON | PDF support, used only if poppler-cpp is found |

## Toolchains used

| Tool | Version |
|---|---|
| GCC | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| Clang | 18.1.3 (Ubuntu 1ubuntu1) |
| CMake | 3.28.3 |
| Python (tools only) | 3.13 |
| OS | Ubuntu 24.04.5 LTS, Linux 6.18 |
| CI runners | GitHub Actions `ubuntu-24.04` |

## Libraries

The engine and the agent use only the C++ standard library, POSIX
(`mmap`/`madvise`, `fsync`, `open`/`rename`) and compiler intrinsics
(`<immintrin.h>`). Everything else is optional and used for testing,
benchmarking or PDF input.

| Library | Version measured | Ubuntu package | Used by | Required? |
|---|---|---|---|---|
| C++ standard library (libstdc++) | GCC 13 | — | everything | yes |
| GoogleTest | 1.14.0 | `libgtest-dev` | `rag_tests`, `rag_pdf_tests` | for tests |
| Google Benchmark | 1.8.3 | `libbenchmark-dev` | `rag_bench` | optional (skipped if absent) |
| poppler-cpp | 24.02.0 | `libpoppler-cpp-dev` | `rag/pdf.hpp`, `rag_pdf`, `rag_pdf_tests` | optional (`RAG_PDF`) |
| libFuzzer + sanitizer runtimes | LLVM 18 | `libclang-rt-18-dev` | `fuzz_load`, `fuzz_text` | optional (`RAG_FUZZ`) |
| pthreads | glibc | — | concurrency tests, `rag_stress` | yes (via `Threads::Threads`) |

Install on Ubuntu 24.04:

```bash
sudo apt-get install -y cmake g++ libgtest-dev libbenchmark-dev libpoppler-cpp-dev
```

Not used, on purpose: no BLAS, no Eigen, no Boost, no JSON library, no
neural-network runtime. The embedder is a concept; `HashingEmbedder` is a
dependency-free stand-in (feature-hashed unigrams and bigrams). An ONNX or
llama.cpp encoder can be plugged in behind the same concept (STATUS.md).

## Measurement machine

All performance numbers in these docs come from one shared cloud VM:

| Item | Value |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, 4 vCPUs, 1 thread per core |
| ISA features | AVX2, FMA, AVX-512F, AVX-512 VNNI |
| Caches | L1d 48 KiB per core, L2 2 MiB per core, L3 260 MiB shared |
| Memory | 15 GiB |
| Kernel | Linux 6.18 |

A shared VM is noisy: timing differences under ~10% aren't meaningful
unless they reproduce.

## Test and CI matrix

| Job | Compiler | Flags | What it checks |
|---|---|---|---|
| gcc-native | GCC | Release, `-march=native` | everything, AVX-512 path |
| gcc-avx2 | GCC | Release, `x86-64-v3` | AVX2 path |
| gcc-scalar | GCC | Release, `x86-64` | scalar fallback |
| clang-native | Clang | Release | second compiler |
| gcc-asan-ubsan | GCC | Debug, ASan + UBSan | memory and UB errors |
| clang-asan-ubsan | Clang | Debug, ASan + UBSan | catches UB GCC's UBSan misses (R12) |
| gcc-tsan | GCC | RelWithDebInfo, TSan | data races (skips single-threaded learning suites) |
| fuzz-smoke | Clang | libFuzzer | the loader and text pipeline under fuzzing |

`ctest` registers every GoogleTest case plus `stress_all_modes` (the
concurrent stress run) and `rag_pdf_cli_smoke`. STATUS.md has the current
count.

## Numeric conventions

* Embeddings: `float` (fp32), L2-normalized; int8 copies with one scale per
  row.
* Agent: `double` throughout; actions are `std::uint32_t`.
* Randomness: xoshiro256** seeded through splitmix64 (`rng.hpp`);
  normal variates by the Marsaglia polar method. Every experiment is
  deterministic given its seeds; evaluation episode i always uses
  `Rng(seed + i)`.
* Floating point: results are bit-reproducible on one build. Across ISAs
  and compilers, FMA contraction can change the last bits, so tests assert
  tolerances or directions, not exact learned decisions.
