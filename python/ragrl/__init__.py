"""ragrl: the OF15 learning pipeline on top of the `_ragof` C++ simulator binding.

Config-driven experiments (experiments/configs/*.json): K seeds trained in
parallel worker processes, champions chosen on validation only, one sealed
scoring in a fresh process, paired evaluation against baselines, bootstrap
CIs, red flags and a run registry under build-avx512/rl/runs/.

    PYTHONPATH=python RAGOF_DIR=<snapshot dir> .venv/Scripts/python.exe -m ragrl --help

Seeds are named only through the seed ledger (docs/ai/NEXT_SESSION.md
section 4, enforced in C++ by rag/orderflow/seed_ledger.hpp): run seeds for
training, block-relative indices for validation/test/viz. Only ledger.py
opens sealed blocks.
"""
import sys as _sys

if _sys.platform == "win32":
    # On this server every WMI query crashes the calling process (0xC0000005)
    # since ~2026-11-07, and Python 3.12's `platform` module asks WMI for the
    # Windows version when torch or lightgbm is imported. Without WMI,
    # `platform` falls back to the registry. Must run before torch/lightgbm load.
    import platform as _platform

    def _no_wmi(*_a, **_k):
        raise OSError("WMI bypassed (ragrl)")

    _platform._wmi_query = _no_wmi

__version__ = "1.0"
SCHEMA = "ragrl/1"
