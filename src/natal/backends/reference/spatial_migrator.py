"""Spatial migration kernels compatibility facade (slice 5).

Runtime migration consumes the CSR routing table folded onto the
:class:`~natal.contracts.blueprint.Blueprint` at build time plus the
``(n_demes, n_sexes, n_ages)`` migration-rate column from
:class:`~natal.contracts.params.Params`.  All topology / adjacency /
kernel normalization happened at build time
(:mod:`natal.frontend.spatial.migration`); this module only multiplies
the rate column by the fixed CSR.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from natal.backends.reference.migration.adjacency import apply_csr_migration

__all__ = [
    "run_spatial_migration",
]
def run_spatial_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    migration_rate: NDArray[np.float64],
    stochastic: bool,
    continuous_sampling: bool,
    stay_after_send: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the migration stage for all demes over the frozen CSR.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        indptr: CSR row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        migration_rate: ``(n_demes, n_sexes, n_ages)`` rate column.
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
            approximations.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one migration step.
    """
    if np.all(migration_rate <= 0.0):
        return ind_count_all, sperm_store_all

    return apply_csr_migration(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        indptr=indptr,
        dest_idx=dest_idx,
        weights=weights,
        rate=migration_rate,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
        stay_after_send=stay_after_send,
    )
