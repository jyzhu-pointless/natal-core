#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal import is_numba_enabled  # noqa: E402
from natal.spatial_simulation_kernels import run_spatial_tick  # noqa: E402


def test_run_spatial_tick_is_numba_dispatcher():
    # njit_switch should expose a dispatcher with py_func when numba is enabled.
    assert not is_numba_enabled() or hasattr(run_spatial_tick, "py_func")
