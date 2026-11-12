"""Shared fixtures for the OF15 binding tests.

The `_ragof` module is found through $RAGOF_DIR (default build-avx512/Release);
everything is skipped when it is not importable (RAG_PYTHON=OFF builds), unless
RAGOF_REQUIRED=1 (the ctests and the gate): then a module that does not import,
or a missing of15_refdump, is an error, so the suite cannot pass with nothing tested.

The usage ledger of sealed openings (ragrl.ledger) is redirected to a temporary
file for the whole session (RAGRL_LEDGER_FILE, inherited by the fresh processes),
so tests never write to the canonical build-avx512/rl/ledger_usage.jsonl.

The seed ledger is process-wide: once a test builds a train, validation or
pairing environment this pytest process is a "trainer" and can never open a
sealed block. Tests that need a scorer (or call forbid_sealed) therefore run
their code in a fresh interpreter through `fresh()`.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RAGOF_DIR = Path(os.environ.get("RAGOF_DIR", REPO / "build-avx512" / "Release"))
REQUIRED = os.environ.get("RAGOF_REQUIRED", "") not in ("", "0")
if not os.environ.get("RAGRL_LEDGER_FILE"):
    os.environ["RAGRL_LEDGER_FILE"] = str(Path(tempfile.mkdtemp(prefix="ragrl_ledger_")) / "ledger_usage.jsonl")
if str(RAGOF_DIR) not in sys.path:
    sys.path.insert(0, str(RAGOF_DIR))

try:
    import _ragof  # noqa: F401
    HAVE_RAGOF = True
    IMPORT_ERROR = ""
except ImportError as e:
    HAVE_RAGOF = False
    IMPORT_ERROR = repr(e)


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: slower end-to-end runs (one ctest each)")


def pytest_collection_modifyitems(config, items):
    if HAVE_RAGOF:
        return
    if REQUIRED:
        raise pytest.UsageError(f"RAGOF_REQUIRED=1 but _ragof does not import from {RAGOF_DIR}: {IMPORT_ERROR}")
    skip = pytest.mark.skip(reason=f"_ragof not importable from {RAGOF_DIR} (build with -DRAG_PYTHON=ON)")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def R():
    import _ragof
    return _ragof


@pytest.fixture(scope="session")
def refdump():
    exe = os.environ.get("RAGOF_REFDUMP") or str(RAGOF_DIR / ("of15_refdump.exe" if os.name == "nt" else "of15_refdump"))
    if not Path(exe).exists():
        if REQUIRED:
            pytest.fail(f"RAGOF_REQUIRED=1 but of15_refdump is missing ({exe})")
        pytest.skip(f"of15_refdump not built ({exe})")
    return exe


def fresh(code: str, *args: str, expect_ok: bool = True) -> str:
    """Runs `code` in a fresh Python process (its own seed-ledger role); returns stdout."""
    prelude = f"import sys; sys.path.insert(0, {str(RAGOF_DIR)!r}); import numpy as np; import _ragof as R\n"
    p = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code), *args], capture_output=True, text=True, timeout=600)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"fresh process failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    return p.stdout


@pytest.fixture(scope="session")
def run_fresh():
    return fresh
