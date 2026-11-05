#!/bin/bash
# OF14: the RL reruns on simulator v4 (OF13, discovery_market_v4(): matched to real ES on book depth,
# same-second impact and flow memory; warm-up 1050 s, pre-registered in OF13). Same recipe and seed
# blocks as OF11, so the two compare directly. Build first (warning-free), copy the binaries to a run
# directory, run, in one command. From the repo root on Windows (Git Bash):
#   bash experiments/of14_regen.sh
# Deterministic: no output depends on --threads.
#
#   of14_train.txt   of_train: 24 seeds x 2000 episodes, NeuralQ + Double DQN (the OF3 recipe), on
#                    discovery_v4 with --warmup 1050, champion on validation (700M+), scored on 300 sealed
#                    episodes (900M+) against the 30 s delta rule; nets in build-avx512/rl/of14/
#   of14_days.txt    of8_scalp_days 20 --market discovery_v4 --warmup 1050 seed1.net seed2.net (930M+)
# Not run: OF6 stresses (no stress list was pre-registered for v4; of6 has one for discovery v3 only);
# the OF2 rules at 1050 s are experiments/of13_baselines_w1050.txt (OF13).
# Visualization (not part of the results): of_train_live on the same market and recipe, probes from the
# 960M+ block, served by tools/of_train_live.py (see docs/ai/ORDERFLOW.md).
set -euo pipefail
cd "$(dirname "$0")/.."
T="of_train of8_scalp_days"
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target $T > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi
rm -f "$build_log"
B=build-avx512/runs/of14-$(git rev-parse --short HEAD)
mkdir -p "$B"
for t in $T; do cp build-avx512/Release/$t.exe "$B/"; done
N=build-avx512/rl/of14
mkdir -p "$N"
W="--market discovery_v4 --warmup 1050"
$B/of_train.exe --seeds 24 --threads 23 --episodes 2000 $W --test 300 --out $N > experiments/of14_train.txt
$B/of8_scalp_days.exe 20 $W $N/seed1.net $N/seed2.net > experiments/of14_days.txt
echo regenerated
