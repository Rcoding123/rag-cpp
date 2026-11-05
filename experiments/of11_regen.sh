#!/bin/bash
# OF11: the RL reruns on the fixed market (OF10 v3, discovery_market(), warm-up
# 1050 s, DECISIONS A63) and the probe runs that back OF10 iteration 3's
# held-out claims. Regenerates every OF11 result file from the current code:
# build first (warning-free), copy the binaries to a run directory (so a later
# rebuild never swaps a binary under a running job), run, in one command.
# Run from the repo root on Windows (Git Bash, windows-avx512-release preset):
#   bash experiments/of11_regen.sh
# Deterministic: no output depends on --threads. Time on a Ryzen 9 9900X: ~40 min.
#
#   of11_train.txt         of_train: 24 seeds x 2000 episodes, NeuralQ + Double DQN (the OF3 recipe), on
#                          discovery with --warmup 1050, champion on validation (700M+), scored on 300 sealed
#                          episodes (900M+) against the 30 s delta rule; nets in build-avx512/rl/of11/
#   of11_stress.txt        of6_scalp_stress --market discovery --warmup 1050 seed1.net seed2.net: the discovery
#                          stress list fixed in 5bb7be3 before any v3 net existed (200 sealed episodes, 920M+)
#   of11_days.txt          of8_scalp_days 20 --market discovery --warmup 1050 seed1.net seed2.net (930M+)
#   of11_baselines.txt     of2_baselines 300 discovery --warmup 1050: the OF2 rules at the RL warm-up
#   of11_probe1050.txt     of_discovery, discovery, fresh diagnostic seeds 22-24, --scalp-warmup 1050: AGREE
#                          and (a)-(f1) in the regime the RL sees (the pre-RL check)
#   of10_final13_18.txt    of_discovery, discovery, seeds 13-18 at 150 s: OF10 iteration 3's held-out final
#                          check (first run once on e229af4; this re-run is deterministic and reproduces it)
#   of10_skeptic19_21.txt  the same on seeds 19-21 (the iteration-3 skeptic's fresh seeds)
set -euo pipefail
cd "$(dirname "$0")/.."
T="of_train of6_scalp_stress of8_scalp_days of2_baselines of_discovery"
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target $T > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
B=build-avx512/runs/of11-$(git rev-parse --short HEAD)
mkdir -p "$B"
for t in $T; do cp build-avx512/Release/$t.exe "$B/"; done
git status --short -- include examples experiments/*.cpp > "$B/DIRTY.txt"
N=build-avx512/rl/of11
mkdir -p "$N"
W="--market discovery --warmup 1050"
$B/of_train.exe --seeds 24 --threads 24 --episodes 2000 $W --test 300 --out $N > experiments/of11_train.txt
pids=()
$B/of6_scalp_stress.exe $W $N/seed1.net $N/seed2.net > experiments/of11_stress.txt & pids+=($!)
$B/of8_scalp_days.exe 20 $W $N/seed1.net $N/seed2.net > experiments/of11_days.txt & pids+=($!)
$B/of2_baselines.exe 300 discovery --warmup 1050 > experiments/of11_baselines.txt & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
P="--markets discovery --episodes 400 --long session --sessions 16 --rules 100 --emo 40 --tick 100 --impact 30 --threads 16 --theta discovery=1.00"
$B/of_discovery.exe $P --seeds 22-24 --scalp-warmup 1050 --out experiments/of11_probe1050.txt > /dev/null
$B/of_discovery.exe $P --seeds 13-18 --out experiments/of10_final13_18.txt > /dev/null
$B/of_discovery.exe $P --seeds 19-21 --out experiments/of10_skeptic19_21.txt > /dev/null
echo regenerated
