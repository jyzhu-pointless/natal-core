"""Spatial migration kernels compatibility facade.

Core implementations are split by backend under ``natal.kernels.migration``:

- ``adjacency``: dense/sparse adjacency-row routing backend.
- ``kernel``: topology + migration-kernel routing backend.

This module keeps the legacy public API surface stable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from natal.kernels.migration.adjacency import apply_spatial_adjacency_mode
from natal.kernels.migration.kernel import (
    apply_spatial_kernel_migration,
    apply_spatial_kernel_migration_heterogeneous,
)
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig

__all__ = [
    "apply_spatial_adjacency_migration",
    "run_spatial_migration",
]


@njit_switch(cache=True)
def apply_spatial_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
    adjust_migration_on_edge: bool = False,
    deme_kernel_ids: NDArray[np.int64] | None = None,
    kernel_d_row: NDArray[np.int64] | None = None,
    kernel_d_col: NDArray[np.int64] | None = None,
    kernel_weights: NDArray[np.float64] | None = None,
    kernel_nnzs: NDArray[np.int64] | None = None,
    kernel_total_sums: NDArray[np.float64] | None = None,
    max_nnz: int = 0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one synchronized migration step over the deme axis.

    Compatibility note:
        ``migration_mode == 0`` dispatches to the adjacency backend;
        ``migration_mode == 1`` dispatches to the topology-kernel backend.

    When ``deme_kernel_ids`` is provided, per-deme kernel selection is active
    and the pre-built per-kernel arrays are used instead of a single kernel.
    """
    if rate <= 0.0:
        return ind_count_all, sperm_store_all

    if migration_mode == 1:
        if (
            deme_kernel_ids is not None
            and kernel_d_row is not None
            and kernel_d_col is not None
            and kernel_weights is not None
            and kernel_nnzs is not None
            and kernel_total_sums is not None
            and max_nnz > 0
        ):
            return apply_spatial_kernel_migration_heterogeneous(
                ind_count_all=ind_count_all,
                sperm_store_all=sperm_store_all,
                topology_rows=topology_rows,
                topology_cols=topology_cols,
                topology_wrap=topology_wrap,
                rate=rate,
                is_stochastic=is_stochastic,
                use_continuous_sampling=use_continuous_sampling,
                adjust_on_edge=adjust_migration_on_edge,
                kernel_d_row=kernel_d_row,
                kernel_d_col=kernel_d_col,
                kernel_weights=kernel_weights,
                kernel_nnzs=kernel_nnzs,
                kernel_total_sums=kernel_total_sums,
                deme_kernel_ids=deme_kernel_ids,
                max_nnz=max_nnz,
            )
        return apply_spatial_kernel_migration(
            ind_count_all=ind_count_all,
            sperm_store_all=sperm_store_all,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=topology_wrap,
            migration_kernel=migration_kernel,
            kernel_include_center=kernel_include_center,
            rate=rate,
            is_stochastic=is_stochastic,
            use_continuous_sampling=use_continuous_sampling,
            adjust_on_edge=adjust_migration_on_edge,
        )

    return apply_spatial_adjacency_mode(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        rate=rate,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )


@njit_switch(cache=True)
def run_spatial_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    config: PopulationConfig,
    migration_rate: float,
    adjust_migration_on_edge: bool = False,
    deme_kernel_ids: NDArray[np.int64] | None = None,
    kernel_d_row: NDArray[np.int64] | None = None,
    kernel_d_col: NDArray[np.int64] | None = None,
    kernel_weights: NDArray[np.float64] | None = None,
    kernel_nnzs: NDArray[np.int64] | None = None,
    kernel_total_sums: NDArray[np.float64] | None = None,
    max_nnz: int = 0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run migration stage for all demes with config-carried stochastic flags."""
    if migration_rate <= 0.0:
        return ind_count_all, sperm_store_all

    return apply_spatial_adjacency_migration(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
        rate=migration_rate,
        is_stochastic=bool(config.is_stochastic),
        use_continuous_sampling=bool(config.use_continuous_sampling),
        adjust_migration_on_edge=adjust_migration_on_edge,
        deme_kernel_ids=deme_kernel_ids,
        kernel_d_row=kernel_d_row,
        kernel_d_col=kernel_d_col,
        kernel_weights=kernel_weights,
        kernel_nnzs=kernel_nnzs,
        kernel_total_sums=kernel_total_sums,
        max_nnz=max_nnz,
    )
