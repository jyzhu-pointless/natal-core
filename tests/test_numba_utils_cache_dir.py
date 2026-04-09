#!/usr/bin/env python3
"""Tests for NUMBA_CACHE_DIR side effects and synchronization timing."""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_import_does_not_mutate_numba_cache_dir_env(monkeypatch):
    """Importing numba_utils should not force-write NUMBA_CACHE_DIR in environment."""
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    import natal.numba_utils as numba_utils  # noqa: E402

    importlib.reload(numba_utils)

    assert "NUMBA_CACHE_DIR" not in os.environ
