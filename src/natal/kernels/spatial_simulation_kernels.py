"""Spatial simulation kernels.

Core multi-deme kernels live under ``natal.kernels`` and are intended to be
called by generated wrappers (see ``natal.kernels.codegen``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from natal.kernels.simulation_kernels import run_aging, run_reproduction, run_survival
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig

__all__ = [
    # No user-facing API for now
]


@njit_switch(cache=True)
def apply_spatial_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: float,
) -> NDArray[np.float64]:
    """Apply adjacency-based migration over the first axis (deme axis)."""
    n_demes = ind_count_all.shape[0]
    flat = ind_count_all.reshape(n_demes, -1)
    inflow = np.zeros_like(flat)

    for src in range(n_demes):
        for dst in range(n_demes):
            w = adjacency[src, dst]
            if w == 0.0:
                continue
            inflow[dst, :] += w * flat[src, :]

    out_flat = (1.0 - rate) * flat + rate * inflow
    return out_flat.reshape(ind_count_all.shape)


@njit_switch(cache=True)
def run_spatial_reproduction(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run reproduction stage for all demes."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    for deme_id in range(ind.shape[0]):
        ind_d, sperm_d = run_reproduction(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d
    return ind, sperm


@njit_switch(cache=True)
def run_spatial_survival(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run survival stage for all demes."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    for deme_id in range(ind.shape[0]):
        ind_d, sperm_d = run_survival(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d
    return ind, sperm


@njit_switch(cache=True)
def run_spatial_aging(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run aging stage for all demes."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    for deme_id in range(ind.shape[0]):
        ind_d, sperm_d = run_aging(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d
    return ind, sperm


@njit_switch(cache=True)
def run_spatial_migration(
    ind_count_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    migration_rate: float,
) -> NDArray[np.float64]:
    """Run migration stage for all demes via adjacency matrix."""
    if migration_rate <= 0.0 or adjacency.size == 0:
        return ind_count_all
    return apply_spatial_adjacency_migration(ind_count_all, adjacency, migration_rate)


@njit_switch(cache=True)
def run_spatial_tick(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick without hook dispatch.

    Stage order is strict:
    reproduction -> survival -> aging.
    """
    ind, sperm = run_spatial_reproduction(ind_count_all, sperm_store_all, config)
    ind, sperm = run_spatial_survival(ind, sperm, config)
    ind, sperm = run_spatial_aging(ind, sperm, config)
    return ind, sperm, int(tick) + 1


@njit_switch(cache=True)
def run_spatial_tick_with_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_rate: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick with migration stage at the end."""
    ind, sperm, tick_next = run_spatial_tick(ind_count_all, sperm_store_all, config, tick)
    ind = run_spatial_migration(ind, adjacency, migration_rate)
    return ind, sperm, tick_next


@njit_switch(cache=True)
def run_spatial_tick_with_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_rate: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Backward-compatible alias for migration-enabled spatial tick."""
    return run_spatial_tick_with_migration(
        ind_count_all,
        sperm_store_all,
        config,
        tick,
        adjacency,
        migration_rate,
    )
