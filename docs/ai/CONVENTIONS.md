# Conventions and working rules

These are the rules the codebase has followed so far. Following them keeps
every claim in the docs measurable and every change verifiable.

## Code

* **Language:** C++20, header-only, `#pragma once`, namespace `rag` (agent:
  `rag::agent`).
* **Hot paths** (query, `act()`) don't allocate. Use fixed `std::array`
  scratch, `TopK<Cap>` on the stack, and per-thread contexts. Tests enforce
  this with the global `operator new` counter (`begin_alloc_count()` /
  `end_alloc_count()` in `tests/test_util.hpp`).
* **Cold paths** (ingest, commit, load, training setup) may allocate, and
  should allocate once.
* **Warnings:** builds use `-Wall -Wextra -Wpedantic -Wconversion` with
  `-Werror` in CI. Silence a narrowing conversion with an explicit
  `static_cast`, not a pragma.
* **Bounds checks** are written so they can't overflow, e.g.
  `len > total || off > total - len`, never `off + len > total`.
* **Untrusted input** (index files, PDFs, query text): validate everything
  before use and before mutating any state. Error messages name the failed
  check; tests match on them.
* **SIMD:** dispatch at compile time (`__AVX512F__`, `__AVX2__`, scalar
  fallback). Any score that must equal a `dot4` lane uses `simd::dot1`.
* **Determinism:** seeded `rag::agent::Rng` everywhere (no `std::mt19937`).
  Greedy evaluation must not consume the agent's RNG.
* **Comments** explain why, and quote measurements where a choice was made
  on evidence.

## Tests

* Every bug fix gets a regression test, and the test is shown to fail on the
  old code (mutation check) before it's trusted.
* Prefer exact checks (accounting identities, bit-exact equality, brute-force
  references) over loose ones.
* Statistical claims use paired comparisons on common random numbers, with
  thresholds set below the range measured across seeds and the measurement
  quoted in a comment.
* **Test files:**
  * `rag_tests.cpp`: engine;
  * `forgery_tests.cpp`: loader checks (add a case for every new file field);
  * `reference_tests.cpp`: spec-level reference;
  * `agent_tests.cpp`: agent;
  * `api_examples_tests.cpp`: the snippets in `docs/ai/API.md`.

## Verifying a change (what "done" means here)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j4
```

1. **Before pushing**, run the same 7 jobs as CI: gcc native, AVX2
   (`-DRAG_ARCH=x86-64-v3`) and scalar (`-DRAG_ARCH=x86-64`); clang; ASan+UBSan
   with both gcc and clang (`-DRAG_SANITIZE=ON`); and TSan (`-DRAG_TSAN=ON`).
   All use `-DCMAKE_CXX_FLAGS=-Werror`.
2. **For loader or parser changes**, fuzz:
   `-DRAG_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++`, `make_seeds`, then run
   `fuzz_load` / `fuzz_text` for a few minutes.
3. **For performance claims**, use `rag_bench` / `rag_latency`, and update
   `README.md` and `bench_results.txt` from the same run.

## Git

* Work on a branch. Push, wait for the 8 GitHub CI jobs, then fast-forward
  `main` to the green commit.
* Commit messages explain what changed and why, with measured results.

## Documentation

* Numbers in docs come from a recorded run (benchmark output, demo output,
  test thresholds). Re-measure rather than copy when the code changes.
* Experiments go in `docs/ai/EXPERIMENTS.md`, including the ones that
  failed.
