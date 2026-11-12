#!/bin/bash
# OF16 phase 1 (docs/ai/OF16_REAL_ES_PLAN.md section 3): export the manifest's DEV days from the user's
# Sierra Chart data and write the dev-day diagnostics (experiments/of16_dev_diagnostics.txt: dev days,
# diagnostics, not a score). Test and confirm days are never exported here (tools/sierra_to_ticks.py
# refuses them without the orchestrator's flags). One command, from the repo root (Git Bash, Windows),
# through the kit:
#   MSYS_NO_PATHCONV=1 powershell -NoProfile -File .claude/kit.ps1 run of16_regen 'bash experiments/of16_regen.sh'
# Light: single-threaded, one day at a time. The _ragof module is copied to a snapshot first
# (build-avx512/real/bin-<sha>/), so a rebuild of build-avx512/Release never meets a running job.
# Exports go to build-avx512/real/<date>/ (gitignored); the market data never enters the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=C:/rag-cpp/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
MOD=$(ls build-avx512/Release/_ragof*.pyd | head -1)
SHA=$(sha256sum "$MOD" | cut -c1-12)
SNAP=build-avx512/real/bin-$SHA
mkdir -p "$SNAP"
[ -f "$SNAP/$(basename "$MOD")" ] || cp "$MOD" "$SNAP/"
DEV=$(PYTHONPATH=python "$PY" -c "import ragrl; from ragrl import real; print(' '.join(real.listed_days('dev', real.load_manifest())))")
for d in $DEV; do
  "$PY" tools/sierra_to_ticks.py "$d"
done
PYTHONPATH=python RAGOF_DIR="$SNAP" "$PY" -X faulthandler -u experiments/of16_dev_diag.py --out experiments/of16_dev_diagnostics.txt
