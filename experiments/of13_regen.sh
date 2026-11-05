#!/bin/bash
# OF13: real ES (the user's Sierra Chart data) vs simulator v4 (discovery_v4), with v3 (discovery) as
# context: OF12's statistics from the same code (tools/real_vs_sim.py --of13), plus the 100 ms step
# rows and Table 4, the pre-registered v4 gates G1-G8. One command, from the repo root on Windows
# (Git Bash, windows-avx512-release preset):
#   bash experiments/of13_regen.sh
# It builds of_calibrate, of_discovery and of2_baselines, copies the binaries to a snapshot (never runs
# build-avx512/Release), exports whole simulator sessions from the open, runs the tool and writes
# experiments/of13_real_vs_v4.txt; then the market-efficiency probe of v4 in the RL episode regime
# (of10_discovery, both regimes) and the naive rules (of2_baselines), each at 150 s (ScalpEnv's default)
# and at OF11's 1050 s, the RL warm-up pre-registered for v4 (the only one whose AGREE core rows all
# pass after the OF13 review fixes): experiments/of13_probe_w150.txt, of13_probe_w1050.txt,
# of13_baselines_w150.txt, of13_baselines_w1050.txt. The probe's legacy (d) bands (1-min sd 3-4.5 etc.,
# from an older ES) do not judge v4: G1-G8 (Table 4 of the real-vs-sim file) replace them.
# experiments/of12_regen.sh (and of12_real_vs_sim.txt) are unchanged by OF13.
# Held-out and skeptic blocks (each run exactly once, same code, outputs renamed into experiments/):
#   SEED0=960013000 DIAG=28-33 bash experiments/of13_regen.sh  -> experiments/of13_final_*.txt   (final check)
#   SEED0=960014000 DIAG=34-36 bash experiments/of13_regen.sh  -> experiments/of13_skeptic_*.txt (skeptic)
#   (set OUT / PROBE_PREFIX to write them elsewhere; the defaults overwrite the tuning files).
#
# REAL DATA: the same files and cutoff as OF12 (read only, never copied into the repo; the result
# holds aggregate statistics only): the paths are read from experiments/of12_regen.sh.
#
# SIMULATOR: 12 sessions x 6.5 h (23,400 s) from the open per market. Seeds: SEED0 + 1 .. SEED0 + 12,
# default SEED0 = 960012000 (OF13's tuning block, = OF12's visualization block). The held-out
# check is SEED0=960013000 OUT=<another file> (once, by the orchestrator); 960014000 is the skeptic's.
# Exports (~0.6 GB) go to build-avx512/tune/of13_<SEED0>/ (gitignored). Probe seeds: DIAG (default 25-27,
# the tuning seeds; held-out 28-33, skeptic 34-36) with PROBE_PREFIX (default experiments/of13_).
# Baseline episodes: of2_baselines' sealed test episodes (900,000,000 + i) for the tuning block, as
# tuned (the brief's gate (e)); for any other SEED0 fresh episodes SEED0 * 1000 + i (--seed0), or
# BSEED0=<N> to choose.
MARKETS="discovery_v4 discovery"   # discovery_v4 = the verdict column; v3 for context
SEED0=${SEED0:-960012000}; SESSIONS=12
OUT=${OUT:-experiments/of13_real_vs_v4.txt}
DIAG=${DIAG:-25-27}; PROBE_PREFIX=${PROBE_PREFIX:-experiments/of13_}
if [ "$SEED0" != 960012000 ]; then BSEED0=${BSEED0:-$((SEED0 * 1000))}; fi
BARGS=(); if [ -n "${BSEED0:-}" ]; then BARGS=(--seed0 "$BSEED0"); fi
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(sed -n '/^BARS=/,/^PY=/p' experiments/of12_regen.sh)"   # BARS SCID_DIR DEPTH_DIR DEPTH_FILES DEPTH_UNTIL PY
T=build-avx512/tune/of13_$SEED0
mkdir -p "$T"
build_log=$(mktemp)
if ! cmake --build --preset windows-avx512-release --parallel --target of_calibrate of_discovery of2_baselines > "$build_log" 2>&1; then
  grep -E "error|warning" "$build_log" | head -40 || true
  exit 1
fi
if grep -E "warning (C|LNK|MSB)[0-9]" "$build_log"; then exit 1; fi  # results only from warning-free code
rm -f "$build_log"
if [ -f .claude/snap.sh ]; then S=$(bash .claude/snap.sh); else S=$T/bin; mkdir -p "$S"; cp build-avx512/Release/{of_calibrate,of_discovery,of2_baselines}.exe "$S"/; fi
pids=()
for m in $MARKETS; do
  for i in $(seq 1 $SESSIONS); do
    seed=$((SEED0 + i))
    "$S"/of_calibrate.exe export "$T/${m}_${seed}.csv" 23400 1 "$m" "$seed" 0 > /dev/null & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done; pids=()
done
depth=(); for f in $DEPTH_FILES; do depth+=("$DEPTH_DIR/$f"); done
sims=(); for m in $MARKETS; do sims+=(--sim "$m=$T/${m}_[0-9]*.csv"); done  # [0-9]: discovery_* would match discovery_v4_* too
"$PY" -I tools/real_vs_sim.py --of13 --bars "$BARS" --scid-dir "$SCID_DIR" --depth "${depth[@]}" --depth-until "$DEPTH_UNTIL" \
  "${sims[@]}" --jobs 16 --out "$OUT" \
  --note "sim seeds $((SEED0 + 1))-$((SEED0 + SESSIONS)), of_calibrate export <out> 23400 1 <market> <seed> 0"
echo "regenerated $OUT"
for w in 150 1050; do  # deterministic: the outputs don't depend on --threads
  "$S"/of_discovery.exe --markets discovery_v4 --seeds "$DIAG" --episodes 400 --long session --sessions 16 --rules 100 --emo 40 \
    --tick 100 --impact 30 --threads 12 --scalp-warmup $w --out "${PROBE_PREFIX}probe_w$w.txt" > /dev/null & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done; pids=()
for w in 150 1050; do "$S"/of2_baselines.exe 300 discovery_v4 --warmup $w "${BARGS[@]}" > "${PROBE_PREFIX}baselines_w$w.txt" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "regenerated ${PROBE_PREFIX}probe_w150.txt probe_w1050.txt baselines_w150.txt baselines_w1050.txt"
