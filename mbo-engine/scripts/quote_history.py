#!/usr/bin/env python3
"""QUOTE ONLY (metadata.get_cost / get_billable_size, free). Never downloads or orders. Prices extra ES history for Phase 5.
usage: quote_history.py   (reads DATABENTO_API_KEY from /srv/quant/.env)"""
import os
import databento as db

for line in open("/srv/quant/.env"):
    if line.startswith("DATABENTO_API_KEY="):
        os.environ["DATABENTO_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
c = db.Historical()
SPANS = [("2026-05-01", "2026-08-01", "3 months before"), ("2026-02-01", "2026-08-01", "6 months before"),
         ("2025-08-01", "2026-08-01", "12 months before")]
for schema in ("mbo", "mbp-10", "mbp-1", "trades"):
    for a, b, lab in SPANS:
        p = dict(dataset="GLBX.MDP3", symbols=["ES.v.0"], stype_in="continuous", schema=schema, start=a, end=b)
        try:
            print(f"{schema:<7} {lab:<17} {a}..{b}  {c.metadata.get_billable_size(**p)/1e9:9.2f} GB  ${c.metadata.get_cost(**p):9.2f}")
        except Exception as e:
            print(f"{schema:<7} {lab:<17} ERROR {type(e).__name__}: {str(e)[:120]}")
