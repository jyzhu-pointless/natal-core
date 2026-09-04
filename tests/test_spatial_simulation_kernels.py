#!/usr/bin/env python3

import sys
from pathlib import Path

import pytest  # type: ignore  # module under test is resolved dynamically (backend layout moved in slice 6)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.backends.reference.spatial_simulator import run_spatial_tick  # noqa: E402
from natal.backends.reference.spatial_migrator import run_spatial_migration  # noqa: E402
from natal.backends.reference.migration.adjacency import apply_csr_migration  # noqa: E402
from natal.frontend.spatial.migration import fold_migration_csr  # noqa: E402


def _fold_dense(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold a dense adjacency matrix into the slice-5 CSR triple."""
    n_demes = adjacency.shape[0]
    csr = fold_migration_csr(
        n_demes=n_demes,
        topology=None,
        adjacency_dense=adjacency,
        migration_kernel=None,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="adjacency",
    )
    return csr.indptr, csr.dest_idx, csr.weights


def test_run_spatial_tick_is_reference():
    """The spatial tick kernel is a plain Python reference function."""
    assert callable(run_spatial_tick)


def test_apply_csr_migration_deterministic_preserves_totals() -> None:
    ind = np.zeros((2, 2, 2, 1), dtype=np.float64)
    sperm = np.zeros((2, 2, 1, 1), dtype=np.float64)
    ind[0, 0, 1, 0] = 10.0
    ind[0, 1, 1, 0] = 6.0
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    indptr, dest_idx, weights = _fold_dense(adjacency)
    rate = np.full((2, 2, 2), 0.25, dtype=np.float64)

    ind_next, sperm_next = apply_csr_migration(
        ind_count_all=ind,
        sperm_store_all=sperm,
        indptr=indptr,
        dest_idx=dest_idx,
        weights=weights,
        rate=rate,
        stochastic=False,
        continuous_sampling=False,
    )

    assert np.isclose(ind_next.sum(), ind.sum())
    assert np.isclose(sperm_next.sum(), sperm.sum())
    assert np.isclose(ind_next[0].sum(), 12.0)
    assert np.isclose(ind_next[1].sum(), 4.0)


def test_apply_csr_migration_stochastic_preserves_totals() -> None:
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
    indptr, dest_idx, weights = _fold_dense(adjacency)
    rate = np.full((3, 2, 2), 0.4, dtype=np.float64)

    ind_next, sperm_next = apply_csr_migration(
        ind_count_all=ind,
        sperm_store_all=sperm,
        indptr=indptr,
        dest_idx=dest_idx,
        weights=weights,
        rate=rate,
        stochastic=True,
        continuous_sampling=False,
    )

    assert np.isclose(ind_next.sum(), ind.sum())
    assert np.isclose(sperm_next.sum(), sperm.sum())
    assert np.all(ind_next >= 0.0)
    assert np.all(sperm_next >= 0.0)
    assert np.all(np.isfinite(ind_next))
    assert np.all(np.isfinite(sperm_next))


def test_run_spatial_migration_zero_rate_is_identity() -> None:
    ind = np.ones((2, 2, 2, 1), dtype=np.float64)
    sperm = np.ones((2, 2, 1, 1), dtype=np.float64)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    indptr, dest_idx, weights = _fold_dense(adjacency)
    rate = np.zeros((2, 2, 2), dtype=np.float64)

    ind_next, sperm_next = run_spatial_migration(
        ind, sperm, indptr, dest_idx, weights, rate, False, False, False
    )

    assert np.array_equal(ind_next, ind)
    assert np.array_equal(sperm_next, sperm)
