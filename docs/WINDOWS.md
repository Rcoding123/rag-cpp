# Building and running on Windows

The engine and the agent are header-only C++20 and build natively with
Visual Studio 2022 (MSVC). The optimized build uses the same AVX2 + FMA
SIMD kernels as Linux. GitHub CI builds this configuration and runs the
full test suite on Windows on every push (job `msvc-windows`).

## 1. Install (once)

1. **Visual Studio 2022** (the free Community edition is enough):
   <https://visualstudio.microsoft.com/vs/community/>. In the installer,
   tick the **"Desktop development with C++"** workload. It includes the
   MSVC compiler, CMake and the Windows SDK.
2. **Git for Windows**: <https://git-scm.com/download/win>.

Nothing else is required: GoogleTest (for the tests) is downloaded
automatically by CMake the first time you configure.

**On Windows Server, or without the IDE:** install **Build Tools for Visual
Studio 2022** instead (<https://visualstudio.microsoft.com/downloads/>,
under "Tools for Visual Studio"), with the **"Desktop development with
C++"** workload, or from an administrator PowerShell with `winget`:

```powershell
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"
```

Build Tools also install the "Developer PowerShell for VS 2022" shortcut
used below. (CI runs on Windows Server 2022 with Visual Studio
preinstalled; the `winget` lines themselves aren't exercised by CI.)

## 2. Build

Open **"Developer PowerShell for VS 2022"** from the Start menu (it puts
`cmake` and the compiler on the path), then:

```powershell
git clone https://github.com/Rcoding123/rag-cpp.git
cd rag-cpp
cmake --preset windows-msvc
cmake --build --preset windows-release --parallel
```

The first configure takes a minute (it downloads GoogleTest); the build
takes a few minutes. The programs land in `build\Release\`.

The same commands without presets, if you prefer:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DRAG_ARCH=x86-64-v3
cmake --build build --config Release --parallel
```

**Always build `Release`.** A Debug build is unoptimized and checks every
container access; it is many times slower and is meant for stepping
through code.

### Which preset for your CPU

| CPU | Preset (configure / build / test) | Kernels |
|---|---|---|
| AVX-512: AMD Zen 4 / Zen 5 (Ryzen 7000/9000, EPYC 9004/9005), Intel Xeon Skylake-SP or newer | `windows-msvc-avx512` / `windows-avx512-release` | AVX-512 (fastest) |
| AVX2: almost any x64 CPU since 2013 | `windows-msvc` / `windows-release` | AVX2 + FMA |
| older | `windows-msvc-scalar` / `windows-scalar-release` | scalar |

For example, on an **AMD Ryzen 9 9900X** (Zen 5, AVX-512):

```powershell
cmake --preset windows-msvc-avx512
cmake --build --preset windows-avx512-release --parallel
ctest --preset windows-avx512-release
```

The AVX-512 build puts its programs in `build-avx512\Release\`. CI compiles
it on Windows but can't run it (GitHub's Windows runners have no
AVX-512); the AVX-512 kernels are tested on Linux. Verified on Windows:
**285/285 tests on an AMD Ryzen 9 9900X (Zen 5, 12 cores, 128 GB) under
Windows Server 2025 Standard with Build Tools for Visual Studio 2022
17.14, in 13 s.** On a different CPU, run `ctest` once: if every test
passes, use this build; if any fails, use `windows-msvc` and tell us. Not sure what your CPU supports? Run
`Get-CimInstance Win32_Processor | Select-Object Name` and look it up.

## 3. Test

```powershell
ctest --preset windows-release
```

Every GoogleTest case runs as its own test, plus a concurrency stress test.
All should pass.

## 4. Run

```powershell
.\build\Release\rag_demo.exe          # RAG engine: index documents, retrieve, build a prompt
.\build\Release\agent_demo.exe        # the self-learning agent: trains, deploys, adapts to a regime change
.\build\Release\rag_latency.exe 20000 # query latency percentiles (p50 / p99 / p99.9) per search mode
```

### Order-flow trading tools on the server

The ES order-flow tools (`docs/ai/ORDERFLOW.md`) build with the presets
too and use every core:

```powershell
.\build-avx512\Release\of_train.exe --seeds 12 --episodes 2000 --out nets     # 12 scalping agents in parallel
.\build-avx512\Release\of_backtest.exe es_ticks.csv --time ns nets\seed1.net  # backtest on recorded ticks
.\build-avx512\Release\of_calibrate.exe facts es_ticks.csv --time ns          # calibration facts
1..12 | % { Start-Process -NoNewWindow .\build-avx512\Release\of_tick_train.exe "$_ 2000 0.99 0.5 0.25 0.02 200 100 tick_seed$_.net" }
```

`of_calibrate fit` needs NLopt (not installed on Windows by default; run
`facts` there and `fit` under WSL, or install NLopt with vcpkg).

### Train the best learned policy (`train_ensemble.exe`)

```powershell
.\build\Release\train_ensemble.exe 8 40000 8 ensemble.tbl
```

trains 8 neural networks (Double DQN with weight averaging, seeds 1–8)
in parallel, one per thread, compiles each to a lookup table, and saves
the cell-wise median vote to `ensemble.tbl`: the E25 recipe, 97.8% of the
optimal gain in the experiments. It prints each network's score and the
ensemble's. Arguments: networks, episodes per network, threads, output
file. Use up to one network per physical core (12 on a Ryzen 9 9900X).
Prefer 8 or an odd number: with an even count the median takes the lower
of the two middle votes, which for 2 networks is just the smaller trade.
With the AVX-512 build, run `.\build-avx512\Release\train_ensemble.exe` instead.
Load the result in your code with `ExecutionPolicyTable::load("ensemble.tbl")`.

To use the library in your own program, add `include\` to the include path
and `#include "rag.hpp"` (engine) or the headers under `rag\agent\`
(agent); see `docs/ai/API.md` for every API with examples. From CMake:
`add_subdirectory(rag-cpp)` and `target_link_libraries(your_app PRIVATE rag)`,
which also sets the optimization flags below.

## 5. In the Visual Studio IDE

**File → Open → Folder…**, pick the `rag-cpp` folder. Visual Studio reads
`CMakePresets.json`: choose the configure preset **"Windows x64, Visual
Studio 2022, AVX2 + FMA"** and the build configuration **Release** in the
toolbar, then **Build → Build All**. Run the tests with `ctest --preset windows-release`
from the Developer PowerShell.

## Optimization: what the Windows build does

| Setting | Why |
|---|---|
| `/O2` (CMake's Release) | full optimization |
| `/arch:AVX2` (presets `windows-msvc`, `RAG_ARCH=x86-64-v3` or `native`) | enables the AVX2 + FMA dot-product and int8 kernels; MSVC defines no `__FMA__`, so the code detects MSVC's AVX2 mode itself (`RAG_AVX2_FMA` in `common.hpp`) |
| `RAG_ARCH=x86-64-v4` → `/arch:AVX512` | the AVX-512 kernels, for CPUs that have AVX-512 (untested on Windows; AVX-512 is tested on Linux) |
| `__forceinline`, `__restrict`, `_mm_prefetch` | the same inlining, aliasing and prefetch hints GCC/Clang get |
| `_aligned_malloc` | cache-line (and 2 MiB for large arrays) alignment, as on Linux |
| `/fp:precise` (MSVC default) | results reproduce exactly for a given compiler; `/fp:fast` is not used |

Differences from Linux:

* **No AVX2?** Use the preset `windows-msvc-scalar` (or `-DRAG_ARCH=x86-64`):
  a build that runs on any x64 CPU, with scalar kernels. A CPU from 2013
  or later almost certainly has AVX2.
* **Transparent huge pages** are Linux-only, so very large scans can be a
  little slower on Windows (on Linux they gave 10–25%).
* **Durable saves:** index and policy-table files are flushed to disk with
  `_commit` (`FlushFileBuffers`) and renamed into place atomically. Windows
  has no directory flush, so durability of the rename is up to NTFS's
  journal.
* **Not available with MSVC:** ThreadSanitizer (`RAG_TSAN`), UBSan, and the
  libFuzzer targets; those run on Linux in CI. `-DRAG_SANITIZE=ON` gives
  MSVC's AddressSanitizer (untested in CI).
* **Optional extras** (PDF input via poppler, Google Benchmark, NLopt +
  Eigen tuning) are skipped automatically when not installed. They are
  tested on Linux only; with vcpkg they may be found, but that isn't
  tested.

## Measured on Windows

GitHub's `windows-2022` runner (AMD EPYC 7763, 2 cores / 4 threads, shared
VM), Visual Studio 2022, preset `windows-msvc` (AVX2 + FMA):

* **Tests:** 284 of 284 pass (`ctest --preset windows-release`).
* **Query latency** (`rag_latency.exe 20000`; 60,000 chunks, k = 8, pinned
  to one core):

| Mode | p50 | p90 | p99 | p99.9 | Queries/s per core |
|---|---|---|---|---|---|
| exact | 5.68 ms | 6.34 ms | 7.63 ms | 9.97 ms | 173 |
| int8 + rerank | 1.51 ms | 1.75 ms | 2.10 ms | 2.64 ms | 664 |
| ivf | 345 µs | 434 µs | 596 µs | 1.47 ms | 2,773 |
| ivf-sq8 | 154 µs | 186 µs | 251 µs | 479 µs | 6,190 |

The exact mode streams the whole 92 MB embedding matrix per query
(~16 GB/s), so the SIMD kernels are active. The README's Linux numbers come
from a different machine (a 4-vCPU Xeon with AVX-512), so the two tables
don't compare directly. The "max" column on a shared CI VM is dominated by
the scheduler (up to 78 ms), which is why it is left out here.

### On an AMD Ryzen 9 9900X (Zen 5, AVX-512)

Windows Server 2025 Standard, 128 GB, Build Tools for Visual Studio 2022
17.14, preset `windows-msvc-avx512`:

* **Tests:** 285/285 in 13 s.
* **Query latency** (`rag_latency.exe 20000`):

| Mode | p50 | p90 | p99 | p99.9 | Queries/s per core |
|---|---|---|---|---|---|
| exact | 1.72 ms | 2.43 ms | 3.91 ms | 9.30 ms | 526 |
| int8 + rerank | 398 µs | 563 µs | 971 µs | 3.33 ms | 2,259 |
| ivf | 119 µs | 151 µs | 239 µs | 1.64 ms | 7,840 |
| ivf-sq8 | 72 µs | 92 µs | 146 µs | 747 µs | 12,761 |

  The exact scan reads the 92 MB embedding matrix at ~54 GB/s, so it runs
  at memory bandwidth: 3.3× the CI runner and 5.9× the README's Linux VM.
* **`train_ensemble.exe 8 40000 8`:** 8 networks in parallel in **141 s**;
  the median vote scores **97.8%** of the optimal gain. Every network's
  score (93.0–95.5%) and the ensemble match E25, which was run on Linux
  with GCC, to the printed precision: the training is reproducible across
  operating systems, compilers and CPUs for these seeds.

## Alternative: WSL

Under Windows Subsystem for Linux (`wsl --install`, then Ubuntu 24.04) the
Linux instructions in the README apply unchanged, including the optional
libraries and sanitizers.
