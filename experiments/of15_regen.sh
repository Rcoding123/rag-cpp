#!/bin/bash
# OF15: the agent-zoo acceptance comparison on simulator v4 (discovery_v4, warm-up 1050), through the
# ragrl pipeline (python/ragrl) on the _ragof binding. Builds with -DRAG_PYTHON=ON (warning-free), copies
# the module and the programs to a run directory, then runs every config in one command. From the repo
# root on Windows (Git Bash):
#   bash experiments/of15_regen.sh [agent ...]        (default: all; e.g. ppo dqn)
#   OF15_MARKET=v3 bash experiments/of15_regen.sh    the same zoo on discovery (v3) (configs of15_<agent>_v3.json:
#                                                    OF11 nets, family of15_v3, test 901M+[300,600)); reports of15_<agent>_v3.txt
# Each config: 12 run seeds x 2000 episodes (of_train's budget) trained in parallel worker processes,
# champion on validation 700M+[0,200) (neuralq_ref: [40,240)), selection on validation 701M+[0,400), one
# retrained seed compared bit for bit (R8), ONE sealed scoring on 901M+[0,300) in a fresh process (the
# range pre-registered for the of15_v4 family: 900M+[0,100000) was viewed by OF2-OF14; seal refuses a
# range viewed by another experiment), paired against flat, the 30 s delta rule, the fair-value oracles
# and the OF14 NeuralQ nets (build-avx512/rl/of14: run experiments/of14_regen.sh first; copied into each
# run). Runs are registered in build-avx512/rl/runs/; each run's report_results.md (no timings or
# binary hashes: byte-identical on regeneration) is copied to experiments/of15_<agent>.txt.
# Deterministic given the seeds, the binary and the venv (provenance.json records all three); results
# don't depend on the number of workers. A STATUS of CLAIM_PENDING_950M_CONFIRMATION goes to the
# orchestrator (ragrl confirm, 950M+, once per claim); the pipeline never confirms.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
AGENTS=${*:-"neuralq_ref ppo ppo_fs rppo dqn qrdqn lgbm dagger"}
build_log=$(mktemp)
if ! { cmake --preset windows-msvc-avx512 -DRAG_PYTHON=ON > "$build_log" 2>&1 &&
       cmake --build --preset windows-avx512-release --parallel --target _ragof of15_refdump of_train of_train_live >> "$build_log" 2>&1; }; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi
rm -f "$build_log"
B=build-avx512/runs/of15-$(git rev-parse --short HEAD)
mkdir -p "$B"
cp build-avx512/Release/_ragof*.pyd build-avx512/Release/of_train.exe build-avx512/Release/of_train_live.exe "$B/"
export RAGOF_DIR=$(pwd -W)/$B PYTHONPATH=python  # C:/... (also under MSYS_NO_PATHCONV=1)
M=${OF15_MARKET:-v4}
case $M in v4) sfx= ;; v3) sfx=_v3 ;; *) echo "OF15_MARKET must be v4 or v3" >&2; exit 2 ;; esac
for a in $AGENTS; do
  cfg=experiments/configs/of15_${a}_$M.json
  $PY -m ragrl run "$cfg" | tee "$B/of15_$a$sfx.log"
  run=$(sed -n 's/^run //p' "$B/of15_$a$sfx.log" | head -1)
  cp "$run/report_results.md" "experiments/of15_$a$sfx.txt"
done
echo regenerated
