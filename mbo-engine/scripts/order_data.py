r"""Order and download the starter ES dataset from Databento.

SAFE BY DEFAULT: without --buy this only prints a price quote (free, no charge).

    python order_data.py              # quote only - spends nothing
    python order_data.py --buy        # quote, ask you to type BUY, then order + download
    python order_data.py --buy        # run again any time: resumes, never re-orders

How it avoids double charges:
  * every submitted job is written to data/raw/orders.json straight away;
  * on every run, jobs already in orders.json (or already on your Databento
    account with identical parameters) are reused, not submitted again.

Downloads are per-file, resumable and sha256-checked. Files land in
    data/raw/<label>/<job_id>/
Databento keeps batch files downloadable for 30 days after the order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import databento as db
from databento.historical.api.batch import _BatchJob  # internal helper; tested with databento 0.90.0

from estimate_cost import load_env

DATASET = "GLBX.MDP3"
START = "2026-08-01"
END = "2026-11-17"  # exclusive: data through 2026-11-16 (same range as the quote)

# "ES.OPT" only covers the QUARTERLY options on ES. Weekly, daily and end-of-month
# options are separate CME products: EW (end of month), EW1-EW4 (Friday weeklies) and
# E<week><day> (A=Mon, B=Tue, C=Wed, D=Thu). E5A does not exist in this date range.
WEEKLY_OPTS = [f"{r}.OPT" for r in
               ["EW", "EW1", "EW2", "EW3", "EW4"]
               + [f"E{w}{d}" for d in "ABCD" for w in range(1, 6) if f"E{w}{d}" != "E5A"]]

# label -> request. ES.v.0 = the most-traded ES contract each day (handles rolls).
REQUESTS = {
    "es-mbo": dict(symbols=["ES.v.0"], stype_in="continuous", schema="mbo"),
    "es-fut-definition": dict(symbols=["ES.FUT"], stype_in="parent", schema="definition"),
    "es-fut-statistics": dict(symbols=["ES.FUT"], stype_in="parent", schema="statistics"),
    "es-opt-definition": dict(symbols=["ES.OPT"], stype_in="parent", schema="definition"),
    "es-opt-statistics": dict(symbols=["ES.OPT"], stype_in="parent", schema="statistics"),
    "es-wkopt-definition": dict(symbols=WEEKLY_OPTS, stype_in="parent", schema="definition"),
    "es-wkopt-statistics": dict(symbols=WEEKLY_OPTS, stype_in="parent", schema="statistics"),
}

ROOT = Path(__file__).resolve().parent / "data" / "raw"
STATE = ROOT / "orders.json"


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def norm_symbols(value) -> list[str]:
    if isinstance(value, str):
        return sorted(s for s in value.split(",") if s)
    return sorted(str(s) for s in (value or []))


def find_existing_job(client: db.Historical, req: dict) -> str | None:
    """Return the id of a job already on the account with identical parameters."""
    since = datetime.now(timezone.utc) - timedelta(days=29)
    # list_jobs only returns id/state/ts_received, so fetch each job's details to compare.
    for summary in retry(client.batch.list_jobs, since=since):
        try:
            job = retry(client.batch.get_job_details, summary["id"])
        except db.BentoError:
            continue
        if (
            job.get("dataset") == DATASET
            and job.get("schema") == req["schema"]
            and job.get("stype_in") == req["stype_in"]
            and norm_symbols(job.get("symbols")) == norm_symbols(req["symbols"])
            and str(job.get("start", ""))[:10] == START
            and str(job.get("end", ""))[:10] == END
            and job.get("encoding") == "dbn"
        ):
            return job["id"]
    return None


def retry(fn, *args, **kwargs):
    """Retry a call that is safe to repeat (quotes, status, resumable downloads) on
    Databento server errors (5xx) or rate limits (429). NEVER used for submit_job."""
    attempts = 8
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except db.BentoHttpError as exc:
            status = exc.http_status or 0
            if attempt == attempts or not (status >= 500 or status == 429):
                raise
            wait = min(10 * attempt, 60)
            print(f"  (Databento server busy: {status}; retry {attempt}/{attempts - 1} in {wait}s)", flush=True)
            time.sleep(wait)


def quote(client: db.Historical) -> dict[str, tuple[float, float]]:
    out = {}
    print(f"Dataset {DATASET}, {START} to {END} (end exclusive)\n")
    print(f"{'Item':<22} {'Size (GB)':>10} {'Cost (USD)':>11}")
    print("-" * 45)
    for label, req in REQUESTS.items():
        params = dict(dataset=DATASET, start=START, end=END, **req)
        gb = retry(client.metadata.get_billable_size, **params) / 1e9
        usd = retry(client.metadata.get_cost, **params)
        out[label] = (gb, usd)
        print(f"{label:<22} {gb:>10.2f} {usd:>11.2f}")
    print("-" * 45)
    print(f"{'TOTAL':<22} {sum(g for g, _ in out.values()):>10.2f} {sum(u for _, u in out.values()):>11.2f}")
    return out


def wait_until_done(client: db.Historical, job_ids: dict[str, str]) -> None:
    pending = dict(job_ids)
    while pending:
        states = {j["id"]: j.get("state") for j in retry(client.batch.list_jobs, states=["queued", "processing", "done", "expired"])}
        for label, jid in list(pending.items()):
            st = states.get(jid, "unknown")
            if st == "done":
                print(f"  {label}: ready")
                del pending[label]
            elif st == "expired":
                raise SystemExit(f"{label}: job {jid} has expired on Databento (30-day window passed).")
        if pending:
            print(f"  waiting for Databento to prepare: {', '.join(pending)} ...", flush=True)
            time.sleep(30)


def download_job(client: db.Historical, label: str, job_id: str) -> tuple[int, int]:
    out_dir = ROOT / label
    # List the job's files ONCE. (The client's public per-file download re-lists the job
    # for every single file, and that listing endpoint often times out with a 504.)
    job = retry(_BatchJob, client.batch, job_id=job_id, output_dir=out_dir)
    files = job._batch_files
    total = sum(f.size for f in files)
    done = 0
    print(f"\n{label}: {len(files)} files, {total / 1e9:.2f} GB compressed -> {out_dir / job_id}")
    for i, f in enumerate(files, 1):
        path = out_dir / job_id / f.filename
        # Skips finished files, resumes partial ones, verifies sha256 - so safe to retry.
        try:
            retry(job.download, [f.filename])
        except Warning as exc:  # checksum mismatch (made an error in main())
            print(f"  {f.filename}: {exc} - deleting and downloading again", flush=True)
            path.unlink(missing_ok=True)
            retry(job.download, [f.filename])  # a second mismatch stops the run
        if not path.exists() or path.stat().st_size != f.size:
            raise SystemExit(f"{f.filename}: size mismatch after download - run the script again to resume.")
        done += f.size
        print(f"  [{i}/{len(files)}] {f.filename}  ({done / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)
    return len(files), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buy", action="store_true", help="actually order and download (asks for confirmation)")
    ap.add_argument("--max-usd", type=float, default=100.0, help="refuse to order if the total is above this (default 100)")
    args = ap.parse_args()

    # A failed checksum must stop the run, not just print a warning.
    warnings.filterwarnings("error", message="Downloaded file failed checksum validation")

    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key or key == "paste_your_key_here":
        raise SystemExit("Put your key in .env as DATABENTO_API_KEY=... first.")
    client = db.Historical(key)

    state = load_state()
    quotes = quote(client)

    # Work out which items still need ordering (never re-order).
    to_order = {}
    for label, req in REQUESTS.items():
        if label in state:
            continue
        existing = find_existing_job(client, req)
        if existing:
            print(f"\nFound an existing identical job for {label} on your account ({existing}) - reusing it.")
            state[label] = {"job_id": existing, "adopted": True}
            save_state(state)
        else:
            to_order[label] = req

    already = [l for l in REQUESTS if l in state]
    if already:
        print(f"\nAlready ordered (will not be charged again): {', '.join(already)}")
    new_total = sum(quotes[l][1] for l in to_order)
    print(f"New charge if you proceed: ${new_total:.2f}")

    if not args.buy:
        print("\nQuote only - nothing ordered, nothing charged. Re-run with --buy to order.")
        return

    if to_order:
        if new_total > args.max_usd:
            raise SystemExit(f"Total ${new_total:.2f} is above the limit ${args.max_usd:.2f} (--max-usd). Nothing ordered.")
        if not sys.stdin.isatty():
            raise SystemExit("Refusing to order without an interactive terminal to confirm in.")
        print(f"\nThis will charge ${new_total:.2f} to your Databento account (credit first).")
        print("Billing happens at submission and cannot be undone by this script.")
        if input("Type BUY to confirm, anything else to cancel: ").strip() != "BUY":
            print("Cancelled - nothing ordered, nothing charged.")
            return
        for label, req in to_order.items():
            job = client.batch.submit_job(
                dataset=DATASET, start=START, end=END,
                encoding="dbn", compression="zstd", split_duration="day",
                stype_out="instrument_id", **req,
            )
            state[label] = {
                "job_id": job["id"],
                "quoted_usd": round(quotes[label][1], 2),
                "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request": {"dataset": DATASET, "start": START, "end": END, **req},
            }
            save_state(state)  # saved after every job, so a crash cannot cause a re-order
            print(f"  ordered {label}: job {job['id']}")

    job_ids = {label: state[label]["job_id"] for label in REQUESTS}
    print("\nChecking jobs are ready ...")
    wait_until_done(client, job_ids)

    n_files = n_bytes = 0
    for label, jid in job_ids.items():
        n, b = download_job(client, label, jid)
        n_files += n
        n_bytes += b
    print(f"\nDone: {n_files} files, {n_bytes / 1e9:.2f} GB on disk under {ROOT}")
    print("All files passed size and sha256 checks.")


if __name__ == "__main__":
    main()
