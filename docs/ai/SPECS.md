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

### No RL library

The reinforcement-learning stack is written from scratch in
`include/rag/agent/`; no Gymnasium, Stable-Baselines3, RLlib, PyTorch or
other framework is used. What each piece corresponds to:

| Piece | Header | Closest off-the-shelf equivalent |
|---|---|---|
| `Environment` concept (`reset`, `step`, `features`, action count, feature ranges) | `env.hpp` | Gymnasium `Env` |
| Tile coding, random Fourier features | `features.hpp` | Sutton's tile coder; scikit-learn `RBFSampler` |
| Q-learning (n-step, replay, annealed step size), episodic k-NN memory | `learners.hpp`, `agent.hpp` | SB3 DQN (with a linear model instead of a network) |
| Actor-critic (softmax policy, TD critic) | `agent.hpp` | SB3 A2C |
| Model-based agent (regression + DP planning) | `model_based.hpp` | none standard |
| Exact optima by dynamic programming | `execution_dp.hpp`, `market_making.hpp` | none (task-specific) |
| Training loop, held-out evaluation, champion, drift detection, policy bank, baseline guard | `trainer.hpp`, `drift.hpp` | partly RLlib / SB3 callbacks |
| Paired evaluation, CVaR | `evaluate.hpp` | none standard |
| Off-policy evaluation (PDIS, WPDIS, DR, ESS) | `ope.hpp` | SCOPE-RL, d3rlpy OPE |

Why from scratch:
* **Latency:** a greedy decision costs 64 ns (TileQ) with zero heap
  allocations, which a Python framework can't approach.
* **Transparency:** each update rule is a few lines (TASK_SPECS.md §5),
  which matters because the agent is meant for learning the architecture.
* **Reproducibility:** every random draw comes from seeded generators, so
  experiments reproduce bit for bit, and paired comparisons (common random
  numbers) are exact.
* **Exact yardsticks:** the tasks are built so a DP can compute the optimum;
  that integration is simplest when everything shares one codebase.

The trade-off: no neural-network function approximation (linear models on
tile features only), no vectorized or GPU environments, and no
distributed training.

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
