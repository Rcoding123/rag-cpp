r"""Estimate Databento costs for a starter ES dataset WITHOUT downloading anything.

Metadata calls (get_cost / get_billable_size) only quote prices; they do not spend credit.

Usage (from the project folder):
    1. Put your key in .env:  DATABENTO_API_KEY=your_key
    2. .\.venv\Scripts\python.exe estimate_cost.py
"""

import os
from pathlib import Path

import databento as db


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))

DATASET = "GLBX.MDP3"

# (label, symbols, stype_in, schema, start, end)  -- end is exclusive
REQUESTS = [
    ("ES Dec-26 MBO, 1 day", ["ESZ6"], "raw_symbol", "mbo", "2026-11-13", "2026-11-14"),
    ("ES Dec-26 MBO, 1 week", ["ESZ6"], "raw_symbol", "mbo", "2026-11-09", "2026-11-14"),
    ("ES Dec-26 MBO, 2 weeks", ["ESZ6"], "raw_symbol", "mbo", "2026-11-02", "2026-11-14"),
    ("ES futures definitions, 1 day", ["ES.FUT"], "parent", "definition", "2026-11-13", "2026-11-14"),
    ("ES options definitions, 1 day", ["ES.OPT"], "parent", "definition", "2026-11-13", "2026-11-14"),
    ("ES options statistics (OI), 2 weeks", ["ES.OPT"], "parent", "statistics", "2026-11-02", "2026-11-14"),
]


def main() -> None:
    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key or key == "paste_your_key_here":
        raise SystemExit("Put your key in .env as DATABENTO_API_KEY=... first.")
    client = db.Historical(key)

    print(f"{'Request':<40} {'Size (GB)':>10} {'Cost (USD)':>11}")
    print("-" * 63)
    total = 0.0
    for label, symbols, stype_in, schema, start, end in REQUESTS:
        params = dict(
            dataset=DATASET,
            symbols=symbols,
            stype_in=stype_in,
            schema=schema,
            start=start,
            end=end,
        )
        try:
            size_gb = client.metadata.get_billable_size(**params) / 1e9
            cost = client.metadata.get_cost(**params)
        except Exception as exc:  # show the error and keep quoting the rest
            print(f"{label:<40} error: {exc}")
            continue
        total += cost
        print(f"{label:<40} {size_gb:>10.2f} {cost:>11.2f}")
    print("-" * 63)
    print(f"{'Sum of all rows (MBO rows overlap)':<40} {'':>10} {total:>11.2f}")


if __name__ == "__main__":
    main()
