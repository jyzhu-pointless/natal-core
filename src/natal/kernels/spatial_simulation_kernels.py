"""Spatial simulation kernels.

Core multi-deme lifecycle kernels live under ``natal.kernels`` and are
intended to be called by generated wrappers (see ``natal.kernels.codegen``).
Migration kernels were split into ``natal.kernels.spatial_migration_kernels``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from numba import prange  # pyright: ignore
except ImportError:
    prange = range  # type: ignore[assignment]

import natal.algorithms as alg
from natal.kernels.simulation_kernels import (
    run_aging,
    run_reproduction_with_precomputed_offspring_probability,
    run_survival,
)
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig

__all__ = [
    # No user-facing API for now
]

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

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config: Shared population configuration used by every deme.
        tick: Current simulation tick.

    Returns:
        A tuple ``(ind_next, sperm_next, tick_next)``.

    Design:
        This kernel executes one full lifecycle per deme inside a single
        ``prange`` region. Compared with stage-by-stage spatial passes, it
        reduces synchronization points between parallel sections while
        preserving per-deme lifecycle ordering.
    """
    # Spatial ticks intentionally reuse the single-deme lifecycle ordering.
    # To avoid stage-by-stage global synchronization, run a full local tick
    # for each deme inside one prange pass, then apply migration separately.
    # The offspring tensor depends only on static config, so compute once and
    # reuse for every deme in this tick.
    offspring_probability = alg.compute_offspring_probability_tensor(
        meiosis_f=config.genotype_to_gametes_map[0],
        meiosis_m=config.genotype_to_gametes_map[1],
        haplo_to_genotype_map=config.gametes_to_zygote_map,
        n_genotypes=config.n_genotypes,
        n_haplogenotypes=config.n_haploid_genotypes,
        n_glabs=config.n_glabs,
    )

    for deme_id in prange(ind_count_all.shape[0]):
        # Work on one deme-local pair of arrays; there are no cross-deme reads
        # until the migration stage, so this section is parallel-safe.
        ind_d, sperm_d = run_reproduction_with_precomputed_offspring_probability(
            ind_count=ind_count_all[deme_id],
            sperm_store=sperm_store_all[deme_id],
            config=config,
            offspring_probability=offspring_probability,
        )
        # Keep lifecycle order identical to non-spatial single-population kernels.
        ind_d, sperm_d = run_survival(
            ind_count=ind_d,
            sperm_store=sperm_d,
            config=config,
        )
        ind_d, sperm_d = run_aging(
            ind_count=ind_d,
            sperm_store=sperm_d,
            config=config,
        )
        ind_count_all[deme_id] = ind_d
        sperm_store_all[deme_id] = sperm_d

    return ind_count_all, sperm_store_all, int(tick) + 1


@njit_switch(cache=True, parallel=True)
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
        config_bank: Numba-typed list of unique configs.
        deme_config_ids: Per-deme config id into ``config_bank``.
        tick: Current simulation tick.

    Returns:
        A tuple ``(ind_next, sperm_next, tick_next)``.

    Note:
        This kernel keeps deme-level ``prange`` parallelism while allowing
        each deme to use a different configuration object.
    """
    for deme_id in prange(ind_count_all.shape[0]):
        cfg = config_bank[int(deme_config_ids[deme_id])]
        offspring_probability = alg.compute_offspring_probability_tensor(
            meiosis_f=cfg.genotype_to_gametes_map[0],
            meiosis_m=cfg.genotype_to_gametes_map[1],
            haplo_to_genotype_map=cfg.gametes_to_zygote_map,
            n_genotypes=cfg.n_genotypes,
            n_haplogenotypes=cfg.n_haploid_genotypes,
            n_glabs=cfg.n_glabs,
        )

        ind_d, sperm_d = run_reproduction_with_precomputed_offspring_probability(
            ind_count=ind_count_all[deme_id],
            sperm_store=sperm_store_all[deme_id],
            config=cfg,
            offspring_probability=offspring_probability,
        )
        ind_d, sperm_d = run_survival(
            ind_count=ind_d,
            sperm_store=sperm_d,
            config=cfg,
        )
        ind_d, sperm_d = run_aging(
            ind_count=ind_d,
            sperm_store=sperm_d,
            config=cfg,
        )

        ind_count_all[deme_id] = ind_d
        sperm_store_all[deme_id] = sperm_d

    return ind_count_all, sperm_store_all, int(tick) + 1


@njit_switch(cache=True)
def run_spatial_tick_with_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    migration_rate: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick with migration applied after aging.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked sperm storage arrays.
        config: Shared population configuration.
        tick: Current simulation tick.
        adjacency: Dense outbound adjacency matrix.
        migration_mode: Backend selector. ``0`` for adjacency, ``1`` for
            topology/kernel routing.
        topology_rows: Number of topology rows for kernel routing.
        topology_cols: Number of topology columns for kernel routing.
        topology_wrap: Whether kernel routing wraps around the topology border.
        migration_kernel: Kernel used when ``migration_mode == 1``.
        kernel_include_center: Whether the kernel center contributes outbound
            mass to the source deme.
        migration_rate: Probability that each scalar bucket attempts to
            migrate.

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
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        config=config,
        migration_rate=migration_rate,
    )
    return ind, sperm, tick_next


@njit_switch(cache=True)
def run_spatial_tick_with_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    migration_rate: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Backward-compatible alias for migration-enabled spatial tick."""
    return run_spatial_tick_with_migration(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        config=config,
        tick=tick,
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        migration_rate=migration_rate,
    )


def run_spatial_tick_codegen_wrapper(
    run_tick_fn: Any,
    registry: Any,
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    migration_rate: float,
) -> Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], bool]:
    """Execute one spatial tick through one generated codegen wrapper."""
    final_state_tuple, result = run_tick_fn(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        config=config,
        registry=registry,
        tick=tick,
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        migration_rate=migration_rate,
    )
    return final_state_tuple, int(result) != 0


def run_spatial_steps_codegen_wrapper(
    run_fn: Any,
    registry: Any,
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    n_steps: int,
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    migration_rate: float,
    record_every: int,
) -> Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], bool]:
    """Execute multiple spatial ticks through one generated codegen wrapper."""
    final_state_tuple, _history, was_stopped = run_fn(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        config=config,
        registry=registry,
        tick=tick,
        n_ticks=n_steps,
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        migration_rate=migration_rate,
        record_interval=record_every,
    )
    return final_state_tuple, bool(was_stopped)
