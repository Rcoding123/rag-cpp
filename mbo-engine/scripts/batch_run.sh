#!/bin/bash
# Run a heavy job on the batch cores with the batch memory cap, WITHOUT sudo:
#   scripts/batch_run.sh <command> [args...]
# Uses a transient user scope (systemd --user) + taskset on the batch CPUs (4-11 and their
# hyper-thread siblings 16-23). Tested 2026-11-18: affinity applied, MemoryMax enforced (a
# 1.5 GB allocation under a 300 MB cap was killed, exit 137). Needs XDG_RUNTIME_DIR, which
# non-login shells do not set. Note: this puts the job in the user's app.slice, not in
# quant-batch.slice (that needs sudo); the CPU and memory limits are the same.
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
exec systemd-run --user --scope --expand-environment=no -q -p MemoryMax="${BATCH_MEM_MAX:-90G}" \
  taskset -c "${BATCH_CPUS:-4-11,16-23}" nice -n "${BATCH_NICE:-10}" "$@"
