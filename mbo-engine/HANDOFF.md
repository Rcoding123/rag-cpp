# Handoff: state as of 2026-11-18

This file is for the next session (human or AI) continuing the work **on the
server**. Read `CLAUDE.md` (working rules) and `PLAN.md` first.

## Server
- Headless Ubuntu 24.04.5, kernel 6.8.
- CPU: AMD Ryzen 9 9900X, 12 cores / 24 threads. RAM: 123 GB.
- Disk: **1 TB NVMe, single drive, no RAID.**
- Hostname `CHI416`. The IP is in the owner's SSH setup and is deliberately
  not recorded here, because this repo is public.
- User `quant`: key-only SSH, sudo needs the password. Root login and
  password login are disabled (`/etc/ssh/sshd_config.d/00-hardening.conf`,
  verified with `sshd -T`).
- Services:
  - ufw allows SSH only.
  - fail2ban is active. The owner's home IP is **not** on its ignore list;
    repeated failed logins can lock the owner out for about 10 minutes.
  - chrony keeps the clock in sync; the clock is UTC.
  - unattended-upgrades is on.
- CPU split (systemd slices from `server/bootstrap.sh`):

  | Slice | Logical CPUs | Physical cores | Limits |
  |---|---|---|---|
  | `quant-live.slice` | `0,12,1,13,2,14,3,15` | 0-3 | none |
  | `quant-batch.slice` | the rest | 4-11 | MemoryMax ~99 GB |

  Run heavy jobs in the batch slice:
  `sudo systemd-run --slice=quant-batch.slice --uid=quant -p WorkingDirectory=/srv/quant ...`
- Python venv: `/srv/quant/.venv` with Python 3.12.3, databento 0.90.0,
  numpy, pandas, polars, pyarrow, duckdb, numba, pybind11 and nanobind
  (versions in `scripts/requirements.txt`).
- No Node.js. The owner is installing their own AI coding tools; the scripts
  never install them.

## Layout on the server
```
/srv/quant/
  .env              DATABENTO_API_KEY (mode 600, never commit)
  .venv/
  estimate_cost.py, plan_budget.py, order_data.py, check_data.py   <- live copies
  data/raw/<label>/<job_id>/*.dbn.zst + manifest/metadata/condition.json
  data/raw/orders.json   <- record of paid Databento jobs (do not delete)
  data/features/, results/, models/, logs/, code/
```
The scripts in this repo (`scripts/`) are the source. The copies in
`/srv/quant` are the ones that run, because they look for `.env` and
`data/raw` next to themselves. If you change a script here, test it, then
copy it to `/srv/quant`.

## Data (all verified: sizes + sha256)
Range 2026-08-01 to 2026-11-16 (92 trading-day files per dataset), dataset
`GLBX.MDP3`.

| Folder | Request | Cost |
|---|---|---|
| `es-mbo` | `ES.v.0` (continuous, volume front month), schema `mbo` (15.76 GB compressed) | $96.68 |
| `es-fut-definition` / `es-fut-statistics` | `ES.FUT` (parent) | $0.01 |
| `es-opt-definition` / `es-opt-statistics` | `ES.OPT` (quarterly options only) | $1.56 |
| `es-wkopt-definition` / `es-wkopt-statistics` | 24 weekly/daily/EOM roots: `EW`, `EW1`-`EW4`, `E1A`-`E5D` without `E5A` | $16.92 |

Total: 665 files, 18.59 GB on disk. About $9.81 of Databento credit should
remain; check the portal.

**Added 2026-11-18 (owner-approved, $0.0810 total):** `es-status` (ES.FUT parent, status schema,
2026-08-01..11-17, $0.0031) and `es-mbp1-20260911-12h` (ES.v.0 mbp-1, 2026-09-11 12:00-13:00 UTC,
$0.0779), bought with `scripts/order_extra.py` (quote-only unless `--buy`, per-item caps $0.05 and
$1.00, total cap $0.20, records in orders.json, never re-orders). Credit left is about $9.73.

**data/raw is read-only** (`chmod -R a-w`, owner request 2026-11-18). To add data lift only the
directory (`chmod u+w /srv/quant/data/raw`), download, then `chmod -R a-w /srv/quant/data/raw`, and
tell the owner. It was lifted once (directory only) for the two purchases above and re-locked.

**Free re-download window ends around 2026-12-18** (30 days after the
orders). After that, replacing lost files costs money, and the disk has no
RAID. A backup copy is still to do.

## Scripts
- `scripts/order_data.py`: quote and order and download, safe by default.
  - Without `--buy` it only quotes. With `--buy` it asks the user to type
    `BUY` and has a `--max-usd` cap (default 100).
  - It never re-orders: it checks `orders.json`, then the account's jobs.
  - Downloads resume and are checksummed, and Databento 5xx/429 errors are
    retried. The order call itself is never retried.
  - Tested: retry logic, corrupt/truncated/missing-file repair, and a
    `--buy` re-run that charged $0 against the real paid jobs.
  - It uses the internal `databento.historical.api.batch._BatchJob` helper,
    tested with databento 0.90.0. Re-test after any upgrade.
- `scripts/check_data.py`: read-only content check of every dataset folder.
- `scripts/estimate_cost.py`, `scripts/plan_budget.py`: free price quotes.
- `server/bootstrap.sh`: sets up the research server (already run).
- `server/factory_bootstrap.sh`: separate build box (Docker, toolchains).
  Not used yet; it has no access to this server by design.

## Open issues
1. The hosting plan listed 2 TB+ NVMe in RAID1, but the server has one 1 TB
   drive. The owner is raising it with the provider.
2. Back up `data/raw` before the 2026-12-18 re-download window closes.
3. The owner should back up their SSH key and store the `quant` password.
4. Optionally add the owner's IP to fail2ban's `ignoreip`.
5. Queue position from MBO is not exact (3.6-7.4% of fills hit an order with an earlier unfilled
   order ahead; PLAN.md Phase 2). Matters for Phase 6.
6. RTH hours (08:30-15:00 CT) are a convention, unverified against CME.

## Engine (built 2026-11-18)
`engine/` (CMake, C++20, gcc 13): `include/mbo/dbn.hpp` (reader),
`include/mbo/book.hpp` (per-order book), `include/mbo/bars.hpp` (bucket rows), tools `dbn_digest`, `book_check`,
tools `bar_extract` (-> Parquet via `scripts/bars_to_parquet.py`), tests `book_test`, `bars_test`. Build: `cmake -S engine -B engine/build -G Ninja &&
cmake --build engine/build`. `scripts/verify_dbn.py` is the Phase 1 acceptance
(`.venv/bin/python -I scripts/verify_dbn.py engine/build/dbn_digest FILES`).
Heavy runs: `scripts/batch_run.sh <cmd>` (transient user scope, MemoryMax 90G + taskset on the batch
CPUs, no sudo; needs `XDG_RUNTIME_DIR=/run/user/1001`, which the script sets). Tested: affinity and
the memory cap are enforced (1.5 GB allocation under a 300 MB cap -> killed, exit 137). It runs in
the user's app.slice, not `quant-batch.slice` (that needs sudo), with the same CPU/memory limits.

Outputs on the server (not in git): `/srv/quant/data/features/{bars,trades}/*.parquet`,
`/srv/quant/data/features/levels/{settlement,session_levels}.parquet`,
`/srv/quant/results/crossed/*.ts` + `crossed_vs_status.txt`. Earlier list: `/srv/quant/results/phase2_book_check.jsonl`
(per-day book acceptance numbers) and `/srv/quant/data/features/bars/*.parquet`
(Phase 3 bar layer, 92 files). `scripts/verify_bars.py` re-checks a day
(`bar_extract FILE /tmp/x.bars 100` first).

## rag-cpp review summary (2026-11-18)
rag-cpp cannot be the base for this engine:
- no DBN reader;
- no per-order book;
- Windows/MSVC-only presets and CI since 2026-11-04;
- the live piece is a Sierra ACSIL study.

Reusable as patterns:
- the vectorised nanobind env (`vec_scalp_env.hpp`, `python/ragof_module.cpp`);
- the order-flow features (`features.hpp`);
- the validation tooling and `docs/ai/LESSONS.md`.
