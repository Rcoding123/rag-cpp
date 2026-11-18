#!/usr/bin/env python3
"""Order the small extra Databento requests approved by the owner on 2026-11-18.

  es-status            ES.FUT (parent) schema "status", 2026-08-01 .. 2026-11-17   cap $0.05
  es-mbp1-20260911-12h ES.v.0 schema "mbp-1", 2026-09-11 12:00-13:00 UTC           cap $1.00
       (the hour contains the 12:30:01-12:30:06 crossed-book run)

Safe by default: without --buy it only quotes. With --buy it orders, with a hard
per-item cap (above it nothing is ordered) and a total cap (--max-usd, default 0.20).
It never re-orders: orders.json and then the account's own jobs are checked first.
Downloads reuse order_data.download_job (resume + sha256 + size check).
The owner's approval and caps are in the conversation of 2026-11-18, so --buy does
not prompt; there is no other way to order here.
Runtime copy lives in /srv/quant next to order_data.py (it imports it).
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import databento as db

import order_data as od
from estimate_cost import load_env

EXTRA = {
    "es-status": dict(symbols=["ES.FUT"], stype_in="parent", schema="status",
                      start="2026-08-01", end="2026-11-17", split_duration="month", cap_usd=0.05),
    "es-mbp1-20260911-12h": dict(symbols=["ES.v.0"], stype_in="continuous", schema="mbp-1",
                                 start="2026-09-11T12:00", end="2026-09-11T13:00",
                                 split_duration="none", cap_usd=1.00),
}


def params(req):
    return {k: v for k, v in req.items() if k not in ("split_duration", "cap_usd")}


def find_existing(client, req):
    from datetime import timedelta
    since = datetime.now(timezone.utc) - timedelta(days=29)
    for s in od.retry(client.batch.list_jobs, since=since):
        try:
            j = od.retry(client.batch.get_job_details, s["id"])
        except db.BentoError:
            continue
        n = len(req["start"])
        if (j.get("dataset") == od.DATASET and j.get("schema") == req["schema"]
                and j.get("stype_in") == req["stype_in"]
                and od.norm_symbols(j.get("symbols")) == od.norm_symbols(req["symbols"])
                and str(j.get("start", ""))[:n] == req["start"] and str(j.get("end", ""))[:len(req["end"])] == req["end"]
                and j.get("encoding") == "dbn"):
            return j["id"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buy", action="store_true")
    ap.add_argument("--max-usd", type=float, default=0.20)
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("error", message="Downloaded file failed checksum validation")
    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY missing")
    client = db.Historical(key)
    state = od.load_state()

    quotes, to_order = {}, {}
    print(f"{'item':<24}{'quote USD':>10}{'cap USD':>9}  status")
    for label, req in EXTRA.items():
        usd = od.retry(client.metadata.get_cost, dataset=od.DATASET, **params(req))
        quotes[label] = usd
        if label in state:
            st = "already ordered (orders.json)"
        elif (jid := find_existing(client, req)):
            state[label] = {"job_id": jid, "adopted": True}
            od.save_state(state)
            st = f"existing job on account {jid} - adopted"
        else:
            st = "OVER CAP - will not order" if usd > req["cap_usd"] else "to order"
            if usd <= req["cap_usd"]:
                to_order[label] = req
        print(f"{label:<24}{usd:>10.4f}{req['cap_usd']:>9.2f}  {st}")
    over = [l for l in EXTRA if l not in state and quotes[l] > EXTRA[l]["cap_usd"]]
    total = sum(quotes[l] for l in to_order)
    print(f"New charge: ${total:.4f} (limit ${args.max_usd:.2f})")
    if over:
        raise SystemExit(f"Quote over the owner's cap for {over}: stop and ask the owner. Nothing ordered.")
    if not args.buy:
        print("Quote only - nothing ordered, nothing charged. Re-run with --buy to order.")
        return
    if total > args.max_usd:
        raise SystemExit(f"Total ${total:.4f} above --max-usd ${args.max_usd:.2f}. Nothing ordered.")
    for label, req in to_order.items():
        job = client.batch.submit_job(  # never retried
            dataset=od.DATASET, encoding="dbn", compression="zstd", stype_out="instrument_id",
            split_duration=req["split_duration"], **params(req))
        state[label] = {"job_id": job["id"], "quoted_usd": round(quotes[label], 4),
                        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "request": {"dataset": od.DATASET, **params(req)}}
        od.save_state(state)  # after every job: a crash cannot cause a re-order
        print(f"  ordered {label}: job {job['id']}")
    ids = {l: state[l]["job_id"] for l in EXTRA}
    od.wait_until_done(client, ids)
    for label, jid in ids.items():
        od.download_job(client, label, jid)
    print("Done; all files passed size and sha256 checks.")


if __name__ == "__main__":
    main()
