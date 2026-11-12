#!/bin/bash
# OF16 phase 2 (docs/ai/OF16_REAL_ES_PLAN.md section 3): the one zero-shot scoring of the 16 sealed
# OF15 runs on the 8 real ES test days, with the stress block (latency 300 ms, $3 per side) in the
# same opening. One command, from the repo root (Git Bash, Windows), through the kit:
#   MSYS_NO_PATHCONV=1 powershell -NoProfile -File .claude/kit.ps1 run of16_phase2 'bash experiments/of16_phase2.sh test'
# `bash experiments/of16_phase2.sh dev` is the rehearsal: the same scoring on the dev days (not a
# score; free to repeat). With `test` the script exports the test days (the orchestrator's step,
# tools/sierra_to_ticks.py --role-ok test --i-am-the-orchestrator) and then runs
# `ragrl score-real RUN --days test --stress` once per sealed run, each in a fresh process (ragrl
# refuses a second test opening of a config). A test opening cannot be redone: the regenerate step
# for the committed results is the summary alone (experiments/of16_summary.py, from the results
# under RUN/real/test-*/), which this script runs last.
# Light: single-threaded, one run at a time. Exports go to build-avx512/real/<date>/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
ROLE=${1:?usage: of16_phase2.sh test|dev}
PY=C:/rag-cpp/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
DAYS=$(PYTHONPATH=python "$PY" -c "import ragrl; from ragrl import real; print(' '.join(real.listed_days('$ROLE', real.load_manifest())))")
if [ "$ROLE" = test ]; then
  for d in $DAYS; do
    [ -f "build-avx512/real/$d/ticks.json" ] || "$PY" tools/sierra_to_ticks.py "$d" --role-ok test --i-am-the-orchestrator
  done
fi
for run in $(ls -d build-avx512/rl/runs/of15_*/ | sort); do
  [ -f "$run/sealed.json" ] || continue
  if [ "$ROLE" = test ] && ls -d "$run"real/test-* >/dev/null 2>&1; then
    echo "skip $run: already has a test opening"; continue
  fi
  PYTHONPATH=python "$PY" -X faulthandler -u -m ragrl score-real "$run" --days "$ROLE" --stress
done
OUT=experiments/of16_real_results.txt
[ "$ROLE" = dev ] && OUT=build-avx512/real/of16_dev_rehearsal.txt
PYTHONPATH=python "$PY" experiments/of16_summary.py --days "$ROLE" --out "$OUT" --html build-avx512/real/of16_viz/index.html
