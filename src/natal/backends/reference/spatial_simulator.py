"""Spatial simulation engine.

Core multi-deme lifecycle engine live under ``natal.backends.reference``.
Migration engine were split into ``natal.backends.reference.spatial_migrator``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from natal.backends.reference.age_structured_simulator import (
    run_aging,
    run_reproduction,
    run_survival,
)
from natal.backends.reference.spatial_migrator import run_spatial_migration
from natal.frontend.data import ModelDraft, PopulationState

__all__ = [
    # No user-facing API for now
]
def run_spatial_tick(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: ModelDraft,
    tick: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick without hook dispatch.

    Stage order is strict:
    reproduction -> survival -> aging.

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config: Shared population configuration used by every deme.
        tick: Current simulation tick.

    Returns:
        A tuple ``(ind_next, sperm_next, tick_next)``.

    This kernel executes one full lifecycle per deme inside a single
    ``range`` region. Compared with stage-by-stage spatial passes, it
    reduces synchronization points between parallel sections while
    preserving per-deme lifecycle ordering.
    """
    # Spatial ticks intentionally reuse the single-deme lifecycle ordering.
    # Offspring tensor is precomputed and shared across all demes.
    for deme_id in range(ind_count_all.shape[0]):
        # Work on one deme-local pair of arrays; there are no cross-deme reads
        # until the migration stage, so this section is parallel-safe.
        deme_state = PopulationState(
            n_tick=tick,
            individual_count=ind_count_all[deme_id],
            sperm_storage=sperm_store_all[deme_id],
        )
        deme_state = run_reproduction(deme_state, config)
        # Keep lifecycle order identical to non-spatial single-population engine.
        deme_state = run_survival(deme_state, config)
        deme_state = run_aging(deme_state, config)
        ind_count_all[deme_id] = deme_state.individual_count
        sperm_store_all[deme_id] = deme_state.sperm_storage

    return ind_count_all, sperm_store_all, int(tick) + 1
def run_spatial_tick_heterogeneous(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: Any,
    deme_config_ids: NDArray[np.int64],
    tick: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick with per-deme heterogeneous configs.

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config_bank: List of unique configs.
        deme_config_ids: Per-deme config id into ``config_bank``.
        tick: Current simulation tick.

    Returns:
        A tuple ``(ind_next, sperm_next, tick_next)``.

    Note:
        This kernel keeps deme-level ``range`` parallelism while allowing
        each deme to use a different configuration object.
    """
    for deme_id in range(ind_count_all.shape[0]):
        cfg = config_bank[int(deme_config_ids[deme_id])]

        deme_state = PopulationState(
            n_tick=tick,
            individual_count=ind_count_all[deme_id],
            sperm_storage=sperm_store_all[deme_id],
        )
        deme_state = run_reproduction(deme_state, cfg)
        deme_state = run_survival(deme_state, cfg)
        deme_state = run_aging(deme_state, cfg)

        ind_count_all[deme_id] = deme_state.individual_count
        sperm_store_all[deme_id] = deme_state.sperm_storage

    return ind_count_all, sperm_store_all, int(tick) + 1
def run_spatial_tick_with_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: ModelDraft,
    tick: int,
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    migration_rate: NDArray[np.float64],
    stochastic: bool,
    continuous_sampling: bool,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick with migration applied after aging.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked sperm storage arrays.
        config: Shared population configuration.
        tick: Current simulation tick.
        indptr: CSR migration row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        migration_rate: ``(n_demes, n_sexes, n_ages)`` rate column.
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
            approximations.

    Returns:
        A tuple ``(ind_next, sperm_next, tick_next)``.

    Note:
        Local lifecycle and migration are intentionally separated into two
        kernels: the first phase is embarrassingly parallel per deme, while
        migration introduces cross-deme coupling and is handled afterwards.
    """
    # First finish the within-deme lifecycle for every deme, then apply one
    # synchronized migration step on the post-aging state.
    ind, sperm, tick_next = run_spatial_tick(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        config=config,
        tick=tick,
    )
    ind, sperm = run_spatial_migration(
        ind_count_all=ind,
        sperm_store_all=sperm,
        indptr=indptr,
        dest_idx=dest_idx,
        weights=weights,
        migration_rate=migration_rate,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
    )
    return ind, sperm, tick_next
def run_spatial_steps_with_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: ModelDraft,
    tick: int,
    n_steps: int,
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    migration_rate: NDArray[np.float64],
    stochastic: bool,
    continuous_sampling: bool,
    record_interval: int = 0,
) -> Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], Optional[NDArray[np.float64]], bool]:
    """Execute multiple spatial ticks with migration and optional history recording.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked sperm storage arrays.
        config: Shared population configuration.
        tick: Starting simulation tick.
        n_steps: Number of ticks to execute.
        indptr: CSR migration row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        migration_rate: ``(n_demes, n_sexes, n_ages)`` rate column.
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
            approximations.
        record_interval: History recording interval (0 = no recording).
    Returns:
        A tuple ``(state_tuple, history, was_stopped)``.
    """
    was_stopped = False
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    tick_cur = tick

    flatten_size = 1 + ind.size + sperm.size

    if record_interval > 0:
        estimated_size = (n_steps // record_interval) + 2
        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
    history_count = 0

    if record_interval > 0 and (tick_cur % record_interval == 0):
        flat_state = np.zeros(flatten_size, dtype=np.float64)
        flat_state[0] = tick_cur
        flat_state[1:1 + ind.size] = ind.flatten()
        flat_state[1 + ind.size:] = sperm.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_steps):
        ind, sperm, tick_cur = run_spatial_tick_with_migration(
            ind_count_all=ind,
            sperm_store_all=sperm,
            config=config,
            tick=int(tick_cur),
            indptr=indptr,
            dest_idx=dest_idx,
            weights=weights,
            migration_rate=migration_rate,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
        )

        if record_interval > 0 and (tick_cur % record_interval == 0):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick_cur
            flat_state[1:1 + ind.size] = ind.flatten()
            flat_state[1 + ind.size:] = sperm.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind, sperm, tick_cur), history_result, was_stopped
