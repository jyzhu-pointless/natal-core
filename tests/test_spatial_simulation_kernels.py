#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal import is_numba_enabled  # noqa: E402
from natal.spatial_simulation_kernels import (  # noqa: E402
    apply_spatial_adjacency_migration,
    run_spatial_tick,
)


def test_run_spatial_tick_is_numba_dispatcher():
    # njit_switch should expose a dispatcher with py_func when numba is enabled.
    assert not is_numba_enabled() or hasattr(run_spatial_tick, "py_func")


def test_apply_spatial_adjacency_migration_deterministic_preserves_totals() -> None:
    ind = np.zeros((2, 2, 2, 1), dtype=np.float64)
    sperm = np.zeros((2, 2, 1, 1), dtype=np.float64)
    ind[0, 0, 1, 0] = 10.0
    ind[0, 1, 1, 0] = 6.0
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    kernel = np.zeros((1, 1), dtype=np.float64)

    ind_next, sperm_next = apply_spatial_adjacency_migration(
        ind_count_all=ind,
        sperm_store_all=sperm,
        adjacency=adjacency,
        migration_mode=0,
        topology_rows=0,
        topology_cols=0,
        topology_wrap=False,
        migration_kernel=kernel,
        kernel_include_center=False,
        rate=0.25,
        is_stochastic=False,
        use_continuous_sampling=False,
    )

    assert np.isclose(ind_next.sum(), ind.sum())
    assert np.isclose(sperm_next.sum(), sperm.sum())
    assert np.isclose(ind_next[0].sum(), 12.0)
    assert np.isclose(ind_next[1].sum(), 4.0)


def test_apply_spatial_adjacency_migration_stochastic_preserves_totals() -> None:
    ind = np.zeros((3, 2, 2, 1), dtype=np.float64)
    sperm = np.zeros((3, 2, 1, 1), dtype=np.float64)
    ind[0, 0, 1, 0] = 10.0
    ind[0, 1, 1, 0] = 6.0
    ind[1, 0, 1, 0] = 4.0
    sperm[0, 1, 0, 0] = 3.0
    adjacency = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    kernel = np.zeros((1, 1), dtype=np.float64)

    ind_next, sperm_next = apply_spatial_adjacency_migration(
        ind_count_all=ind,
        sperm_store_all=sperm,
        adjacency=adjacency,
        migration_mode=0,
        topology_rows=0,
        topology_cols=0,
        topology_wrap=False,
        migration_kernel=kernel,
        kernel_include_center=False,
        rate=0.4,
        is_stochastic=True,
        use_continuous_sampling=False,
    )

    assert np.isclose(ind_next.sum(), ind.sum())
    assert np.isclose(sperm_next.sum(), sperm.sum())
    assert np.all(ind_next >= 0.0)
    assert np.all(sperm_next >= 0.0)
