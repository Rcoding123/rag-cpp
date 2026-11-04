# Experiment probes

The programs behind `docs/ai/EXPERIMENTS.md`. Each experiment names its
probe (for example "probe `e7_hybrid.cpp`"); this directory holds them as
they were run.

* They are snapshots, **not built by CMake and not maintained**: some were
  written against an earlier version of the headers and may need small
  changes to compile today. The tests in `tests/` are the maintained
  versions of the findings.
* Build one against the headers:

  ```bash
  g++ -std=c++20 -O2 -Iinclude experiments/e21_board.cpp -o /tmp/e21_board
  ```

* Most print CSV or a table to stdout. `e21_agg.py` aggregates E21's CSVs
  (mean, bootstrap interval, IQM, range): `python3 experiments/e21_agg.py DIR`.
* Seeds are fixed in the sources, so results reproduce exactly on the same
  compiler and standard library.
