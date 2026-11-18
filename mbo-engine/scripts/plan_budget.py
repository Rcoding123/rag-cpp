"""Quote month-by-month MBO costs for ES and NQ front months (no download, no charge).

Uses continuous volume-based front-month symbols (ES.v.0 / NQ.v.0), so contract
rolls are handled automatically.

Usage (from the project folder):
    .venv/Scripts/python.exe plan_budget.py [budget_usd]
"""

import os
import sys

import databento as db

from estimate_cost import load_env

DATASET = "GLBX.MDP3"
MONTHS = [
    ("2026-05", "2026-05-01", "2026-06-01"),
    ("2026-06", "2026-06-01", "2026-07-01"),
    ("2026-07", "2026-07-01", "2026-08-01"),
    ("2026-08", "2026-08-01", "2026-09-01"),
    ("2026-09", "2026-09-01", "2026-10-01"),
    ("2026-10", "2026-10-01", "2026-11-01"),
    ("2026-11*", "2026-11-01", "2026-11-17"),  # partial month to date
]
PRODUCTS = ["ES", "NQ"]


def quote(client, symbol, schema, start, end, stype_in="continuous"):
    params = dict(dataset=DATASET, symbols=[symbol], stype_in=stype_in,
                  schema=schema, start=start, end=end)
    size_gb = client.metadata.get_billable_size(**params) / 1e9
    cost = client.metadata.get_cost(**params)
    return size_gb, cost


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key or key == "paste_your_key_here":
        raise SystemExit("Put your key in .env as DATABENTO_API_KEY=... first.")
    client = db.Historical(key)

    rows = []  # (product, month, gb, usd)
    print(f"{'Product':<8} {'Month':<9} {'Size (GB)':>10} {'Cost (USD)':>11}")
    print("-" * 42)
    for product in PRODUCTS:
        for label, start, end in MONTHS:
            try:
                gb, usd = quote(client, f"{product}.v.0", "mbo", start, end)
            except Exception as exc:
                print(f"{product:<8} {label:<9} error: {exc}")
                continue
            rows.append((product, label, gb, usd))
            print(f"{product:<8} {label:<9} {gb:>10.2f} {usd:>11.2f}")
        print("-" * 42)

    # Options data for gamma over the whole window (small).
    first, last = MONTHS[0][1], MONTHS[-1][2]
    for product in PRODUCTS:
        for schema in ("statistics", "definition"):
            try:
                gb, usd = quote(client, f"{product}.OPT", schema, first, last, stype_in="parent")
                print(f"{product}.OPT {schema:<11} {first}..{last}: {gb:6.2f} GB  ${usd:7.2f}")
            except Exception as exc:
                print(f"{product}.OPT {schema}: error: {exc}")

    # Greedy: most recent months first, ES before NQ, until the budget runs out.
    print(f"\nMost-recent-first fill within ${budget:.0f}:")
    spent = 0.0
    for product in PRODUCTS:
        for p, label, gb, usd in reversed([r for r in rows if r[0] == product]):
            if spent + usd <= budget:
                spent += usd
                print(f"  + {p} {label:<9} ${usd:6.2f}   running total ${spent:6.2f}")
    print(f"  Total: ${spent:.2f}")


if __name__ == "__main__":
    main()
