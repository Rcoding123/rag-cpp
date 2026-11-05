#!/bin/bash
# OF12: real ES order-flow statistics (the user's Sierra Chart data) vs the ES simulator, the same
# statistics from the same code (tools/real_vs_sim.py). One command, from the repo root on Windows
# (Git Bash, windows-avx512-release preset):
#   bash experiments/of12_regen.sh
# It builds of_calibrate, copies the binary to a snapshot (never runs build-avx512/Release), exports
# whole simulator sessions from the open, runs the tool and writes experiments/of12_real_vs_sim.txt.
#
# REAL DATA: read from the paths below, outside the repo (licensed market data: read only, never copied
# into the repo or committed; the result file holds aggregate statistics only). Edit them for another machine.
BARS=C:/Users/Administrator/Desktop/SierraChart-KB/performance/orderflow-models/cache/ES_continuous_1min.csv
SCID_DIR=C:/SierraChart/Data                                  # ESH26/ESM26/ESU26/ESZ26_FUT_CME.scid (1-s bundles)
DEPTH_DIR=C:/SierraChart/Data/MarketDepthData                 # full L2 .depth files
DEPTH_FILES="ESZ26_FUT_CME.2026-11-03.depth ESZ26_FUT_CME.2026-11-04.depth ESZ26_FUT_CME.2026-11-05.depth"
DEPTH_UNTIL="2026-11-05 12:38:00"                              # UTC: Sierra is still writing the Nov 5 file
PY=C:/Users/Administrator/AppData/Local/Programs/Python/Python312/python.exe   # Python 3.12, stdlib only
#
# SIMULATOR: 12 sessions x 6.5 h (23,400 s) from the open (start second 0, activity warm start where the
# preset has it) per market, seeds 960,012,001-960,012,012 (the visualization/diagnostic block,
# docs/ai/NEXT_SESSION.md section 4: look as often as you like, never select or score on it); the same
# seeds for every market. Exports (~0.9 GB) go to build-avx512/tune/of12/ (gitignored).
MARKETS="discovery default_warm default"   # discovery = v3 (the verdict column); then context
SEED0=960012000; SESSIONS=12
set -euo pipefail
cd "$(dirname "$0")/.."
T=build-avx512/tune/of12
mkdir -p "$T"
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target of_calibrate > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
if [ -f .claude/snap.sh ]; then S=$(bash .claude/snap.sh); else S=$T/bin; mkdir -p "$S"; cp build-avx512/Release/of_calibrate.exe "$S"/; fi
pids=()
for m in $MARKETS; do
  for i in $(seq 1 $SESSIONS); do
    seed=$((SEED0 + i))
    "$S"/of_calibrate.exe export "$T/${m}_${seed}.csv" 23400 1 "$m" "$seed" 0 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done; pids=()
done
depth=(); for f in $DEPTH_FILES; do depth+=("$DEPTH_DIR/$f"); done
sims=(); for m in $MARKETS; do sims+=(--sim "$m=$T/${m}_96*.csv"); done
"$PY" -I tools/real_vs_sim.py --bars "$BARS" --scid-dir "$SCID_DIR" --depth "${depth[@]}" --depth-until "$DEPTH_UNTIL" \
  "${sims[@]}" --jobs 16 --out experiments/of12_real_vs_sim.txt \
  --note "sim seeds $((SEED0 + 1))-$((SEED0 + SESSIONS)) (visualization block), of_calibrate export <out> 23400 1 <market> <seed> 0"
echo regenerated experiments/of12_real_vs_sim.txt
