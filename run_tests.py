#!/usr/bin/env python3
"""
Run the full test suite.

Usage:
    python run_tests.py

Thin wrapper around `pytest src/tests/` that also puts `src/` on sys.path,
so this works whether or not PYTHONPATH=src has been set manually (the
tests themselves import e.g. `pinn.model` and `common.classes`, which live
under src/).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    sys.exit(pytest.main([str(SRC / "tests"), "-v"]))
