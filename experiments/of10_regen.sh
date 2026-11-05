#!/bin/bash
# Regenerate every OF10 result file (and the OF2/OF9 tables the OF10 change
# touched) from the current code: build first, in the same command.
# Run from the repo root on Windows (Git Bash, windows-avx512-release preset):
#   bash experiments/of10_regen.sh
# Deterministic: the outputs don't depend on --threads. At most 12 processes /
# threads run at once.
#   of10_results.txt / of10_discovery.csv  tuning seeds 1-3 x 2 h, all markets, rules, tick rules, impact
#   of10_heldout.txt / of10_heldout.csv    held-out seeds 4-6 x 2 h, theta* per market pre-registered on seeds 1-3
#   of10_baselines.txt, of10_tick_baselines.txt, of10_calibrate_sim.txt   discovery (v2) market
#   of2_results.txt, of9_efficient_baselines.txt   default / efficient (rows appended in OF10 v2 only)
# The v1 tables of 10bfe03 (discovery market = discovery_v1 then) are reproduced, not regenerated here, by
#   of2_baselines 300 discovery_v1; of4_tick_baselines 300 discovery_v1; of_calibrate sim 10800 discovery_v1
# (identical except of2's rows appended in v2 and of_calibrate's title line); the v1 probe's own numbers
# (10bfe03 commit message): git show 10bfe03:experiments/of10_results.txt (and of10_heldout.txt).
# The final check on the fresh held-out seeds 7-12 x 6 h is run once, by hand, after the preset is frozen
# (theta* per market as in THETA below, unless the frozen preset's seeds 1-3 run pre-registers new ones):
#   build-avx512/Release/of_discovery.exe --markets default,discovery_v1,discovery --seeds 7-12 --seconds 21600 --threads 12
#     --rules 100 --tick 100 --impact 30 --theta default=1.00,discovery_v1=0.60,discovery=0.70
#     --csv build-avx512/tune/of10_final.csv --out build-avx512/tune/of10_final.txt
set -euo pipefail
cd "$(dirname "$0")/.."
B=build-avx512/Release
# theta* per market: the best non-vacuous gap oracle of each market on seeds 1-3 (of10_results.txt), pre-registered for 4-6
THETA=default=1.00,efficient=1.00,discovery_v1=0.60,discovery=0.70
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target of_discovery of2_baselines of4_tick_baselines of_calibrate > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
M=default,efficient,discovery_v1,discovery
$B/of_discovery.exe --markets $M --seeds 1-3 --seconds 7200 --threads 12 --rules 100 --tick 100 --impact 30 --csv experiments/of10_discovery.csv --out experiments/of10_results.txt > /dev/null
$B/of_discovery.exe --markets $M --seeds 4-6 --seconds 7200 --threads 12 --rules 100 --tick 100 --impact 30 --theta $THETA --csv experiments/of10_heldout.csv --out experiments/of10_heldout.txt > /dev/null
pids=()
$B/of2_baselines.exe 300 default > experiments/of2_results.txt & pids+=($!)
$B/of2_baselines.exe 300 efficient > experiments/of9_efficient_baselines.txt & pids+=($!)
$B/of2_baselines.exe 300 discovery > experiments/of10_baselines.txt & pids+=($!)
$B/of4_tick_baselines.exe 300 discovery > experiments/of10_tick_baselines.txt & pids+=($!)
$B/of_calibrate.exe sim 10800 discovery > experiments/of10_calibrate_sim.txt & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done  # set -e: any failing job stops here
echo regenerated
