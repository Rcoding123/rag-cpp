r"""Read-only sanity check of the downloaded Databento files. Changes nothing, costs nothing.

    python check_data.py

For each dataset folder under data/raw it reports the file count, size and date range,
then opens the smallest and largest data file to confirm the contents are readable and
are what was ordered.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import databento as db

# The databento library triggers a harmless pandas deprecation notice on every file read.
warnings.filterwarnings("ignore", message="The inplace keyword in DataFrame.set_index")

ROOT = Path(__file__).resolve().parent / "data" / "raw"
STATE = ROOT / "orders.json"


def describe(path: Path) -> None:
    store = db.DBNStore.from_file(path)
    md = store.metadata
    print(f"    file     {path.name}  ({path.stat().st_size / 1e6:.1f} MB compressed)")
    print(f"    schema   {store.schema}   dataset {store.dataset}   stype_in {md.stype_in}")
    print(f"    range    {store.start} -> {store.end}")
    print(f"    symbols  requested {store.symbols}")
    df = store.to_df(map_symbols=True)
    print(f"    records  {len(df):,}")
    if len(df) == 0:
        return
    if "symbol" in df.columns:
        syms = df["symbol"].dropna().unique()
        shown = ", ".join(map(str, syms[:6])) + (" ..." if len(syms) > 6 else "")
        print(f"    resolved {len(syms)} symbol(s): {shown}")
    if str(store.schema) == "mbo":
        counts = df["action"].value_counts().to_dict()
        names = {"A": "add", "C": "cancel", "M": "modify", "T": "trade", "F": "fill", "R": "clear", "N": "none"}
        print("    actions  " + ", ".join(f"{names.get(k, k)} {v:,}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        print(f"    orders   {df['order_id'].nunique():,} distinct order ids")
        px = df.loc[df["action"].isin(["T", "F"]), "price"]
        if len(px):
            print(f"    trades   price range {px.min():.2f} - {px.max():.2f}")
    if str(store.schema) == "statistics":
        oi = df[df["stat_type"] == 9]  # 9 = open interest
        print(f"    open interest records {len(oi):,} across {oi['symbol'].nunique() if 'symbol' in oi else '?'} instruments")


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"{ROOT} does not exist - nothing downloaded yet.")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    grand_files = grand_bytes = 0
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        files = sorted(folder.rglob("*.dbn.zst"))
        size = sum(f.stat().st_size for f in files)
        grand_files += len(files)
        grand_bytes += size
        job = state.get(folder.name, {}).get("job_id", "?")
        print(f"\n{folder.name}  (job {job})")
        if not files:
            print("    no data files")
            continue
        dates = [f.name.split("-")[2].split(".")[0] for f in files]
        empty = [f.name for f in files if f.stat().st_size == 0]
        print(f"    {len(files)} data files, {size / 1e9:.2f} GB, dates {min(dates)} to {max(dates)}")
        if empty:
            print(f"    WARNING: {len(empty)} empty file(s): {empty[:3]}")
        by_size = sorted(files, key=lambda f: f.stat().st_size)
        print("  smallest file:")
        describe(by_size[0])
        # Large MBO days hold tens of millions of rows; a 1 GB+ frame is not needed to check them.
        big = by_size[-1]
        if big != by_size[0] and big.stat().st_size < 60e6:
            print("  largest file:")
            describe(big)
        elif big != by_size[0]:
            print(f"  largest file: {big.name} ({big.stat().st_size / 1e6:.0f} MB compressed) - header only:")
            store = db.DBNStore.from_file(big)
            print(f"    schema   {store.schema}   range {store.start} -> {store.end}   symbols {store.symbols}")
    print(f"\nTOTAL: {grand_files} data files, {grand_bytes / 1e9:.2f} GB under {ROOT}")


if __name__ == "__main__":
    main()
