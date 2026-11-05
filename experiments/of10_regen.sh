#!/bin/bash
# Regenerate every OF10 result file (and the OF2/OF9 tables the OF10 change
# touched) from the current code: build first, in the same command.
# Run from the repo root on Windows (Git Bash, windows-avx512-release preset):
#   bash experiments/of10_regen.sh
# Deterministic: the outputs don't depend on --threads.
set -euo pipefail
cd "$(dirname "$0")/.."
B=build-avx512/Release
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target of_discovery of2_baselines of4_tick_baselines of_calibrate > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
$B/of_discovery.exe --markets default,efficient,discovery --seeds 1-3 --seconds 7200 --threads 8 --rules 100 --csv experiments/of10_discovery.csv --out experiments/of10_results.txt > /dev/null
$B/of_discovery.exe --markets default,efficient,discovery --seeds 4-6 --seconds 7200 --threads 8 --rules 100 --csv experiments/of10_heldout.csv --out experiments/of10_heldout.txt > /dev/null
pids=()
$B/of2_baselines.exe 300 default > experiments/of2_results.txt & pids+=($!)
$B/of2_baselines.exe 300 efficient > experiments/of9_efficient_baselines.txt & pids+=($!)
$B/of2_baselines.exe 300 discovery > experiments/of10_baselines.txt & pids+=($!)
$B/of4_tick_baselines.exe 300 discovery > experiments/of10_tick_baselines.txt & pids+=($!)
$B/of_calibrate.exe sim 10800 discovery > experiments/of10_calibrate_sim.txt & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done  # set -e: any failing job stops here
echo regenerated
