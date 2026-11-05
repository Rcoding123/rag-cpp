#!/bin/bash
# Regenerate every OF10 result file (and the OF2/OF9 tables the OF10 change
# touched) from the current code: build first, in the same command.
# Run from the repo root on Windows (Git Bash, windows-avx512-release preset):
#   bash experiments/of10_regen.sh
# Deterministic: the outputs don't depend on --threads. At most 16 processes /
# threads run at once.
# Iteration 3 (discovery_market() = v3: activity model 1, no informed makers):
#   of10_results.txt / of10_discovery.csv  tuning seeds 1-3, every market in both regimes: 400 ScalpEnv-built
#                                          10-minute episodes and 16 session-weighted full sessions per seed
#                                          (AGREE table), rules 100, the pre-registered oracles (O1; O2 = EMO on
#                                          40 episodes), tick rules 100, impact 30
#   of10_heldout.txt / of10_heldout.csv    sanity seeds 4-6, the same, theta* per market pre-registered on seeds 1-3
#   of10_warmup_check.txt                  ground truth: the same episode windows after a 15-minute burn
#                                          (--episode-warmup 1050) instead of ScalpEnv's 150 s, seeds 1-3
#   of10_baselines.txt, of10_tick_baselines.txt, of10_calibrate_sim.txt   discovery (v3) market
#                                          (of10_calibrate_sim.txt is ONE 3-hour window: no error bar; its markouts
#                                          and delta predictability are not evidence for or against (c): use the
#                                          probe's per-regime (c) rows)
#   of2_results.txt                        default (rows unchanged), then the same table for default_warm()
#                                          appended (the old market at realistic activity)
#   of9_efficient_baselines.txt            efficient (unchanged)
# The v2 tables of 4ba0354 (discovery market = v2 then) came from that commit's probe and regen script
# (git show 4ba0354:experiments/of10_regen.sh); the v2 market itself is market_preset("discovery_v2")
# (golden hash in the PriceDiscovery tests), e.g. of2_baselines 300 discovery_v2 reproduces 4ba0354's
# of10_baselines.txt (and of4_tick_baselines 300 discovery_v2 its of10_tick_baselines.txt; of_calibrate sim
# 10800 discovery_v2 its of10_calibrate_sim.txt except the title line). The v1 tables (10bfe03): of2_baselines 300 discovery_v1, etc., as before.
# The final check on the reserved diagnostic seeds 13-18 is run once, by the acceptance agent only,
# after the preset is frozen (golden hash), e.g.:
#   build-avx512/Release/of_discovery.exe --markets discovery --seeds 13-18 --episodes 400 --long session
#     --sessions 16 --rules 100 --emo 40 --tick 100 --impact 30 --threads 16 --theta discovery=<theta* below>
#     --csv build-avx512/tune/of10_final.csv --out build-avx512/tune/of10_final.txt
set -euo pipefail
cd "$(dirname "$0")/.."
B=build-avx512/Release
# Legacy hysteresis gap-oracle theta* per market: the best non-vacuous one of each market on seeds 1-3
# (of10_results.txt), pre-registered for 4-6. Iteration 3's (f) gate uses the fixed oracles O1 / O2 instead.
THETA=default=1.00,default_warm=1.00,efficient=1.00,discovery_v1=0.60,discovery_v2=0.70,discovery=1.00
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target of_discovery of2_baselines of4_tick_baselines of_calibrate > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
M=default,default_warm,efficient,discovery_v1,discovery_v2,discovery
R="--episodes 400 --long session --sessions 16 --rules 100 --emo 40 --tick 100 --impact 30 --threads 16"
$B/of_discovery.exe --markets $M --seeds 1-3 $R --csv experiments/of10_discovery.csv --out experiments/of10_results.txt > /dev/null
$B/of_discovery.exe --markets $M --seeds 4-6 $R --theta $THETA --csv experiments/of10_heldout.csv --out experiments/of10_heldout.txt > /dev/null
$B/of_discovery.exe --markets default,default_warm,discovery --seeds 1-3 --episodes 400 --episode-warmup 1050 --long off --threads 16 \
  --out experiments/of10_warmup_check.txt > /dev/null
pids=()
{ $B/of2_baselines.exe 300 default; printf '\r\n=== default_warm (OF10 iteration 3): the same strategies and episodes on default_warm() (activity model 1)\r\n'; $B/of2_baselines.exe 300 default_warm; } > experiments/of2_results.txt & pids+=($!)
$B/of2_baselines.exe 300 efficient > experiments/of9_efficient_baselines.txt & pids+=($!)
$B/of2_baselines.exe 300 discovery > experiments/of10_baselines.txt & pids+=($!)
$B/of4_tick_baselines.exe 300 discovery > experiments/of10_tick_baselines.txt & pids+=($!)
$B/of_calibrate.exe sim 10800 discovery > experiments/of10_calibrate_sim.txt & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done  # set -e: any failing job stops here
echo regenerated
