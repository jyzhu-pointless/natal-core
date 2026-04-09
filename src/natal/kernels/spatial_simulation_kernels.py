"""Spatial simulation kernels.

Core multi-deme kernels live under ``natal.kernels`` and are intended to be
called by generated wrappers (see ``natal.kernels.codegen``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from numba import get_num_threads, prange  # pyright: ignore
    from numba.np.ufunc.parallel import get_thread_id  # pyright: ignore
    numba_max_threads = int(get_num_threads())
except ImportError:
    prange = range  # type: ignore[assignment]
    def get_thread_id() -> int:
        return 0

    numba_max_threads = 1

import natal.algorithms as alg
from natal import numba_compat as nbc
from natal.kernels.simulation_kernels import (
    run_aging,
    run_reproduction_with_precomputed_offspring_probability,
    run_survival,
)
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig

__all__ = [
    # No user-facing API for now
]


@njit_switch(cache=True, parallel=True)
def _apply_spatial_adjacency_migration_deterministic_parallel(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_nnz: NDArray[np.int64],
    row_total: NDArray[np.float64],
    rate: float,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one deterministic migration step using sparse per-source rows.

    This implementation avoids scanning all destination demes for every source
    bucket. Instead, each source deme iterates only over its valid outbound
    destinations precomputed in ``row_dst_idx``/``row_dst_prob``.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        row_dst_idx: Destination indices for each source row.
        row_dst_prob: Normalized destination probabilities for each source row.
        row_nnz: Number of valid destinations in each source row.
        row_total: Sum of unnormalized row weights in each source row.
        rate: Migration probability.
        n_threads: Number of thread lanes reserved for thread-local buffers.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one deterministic migration.
    """
    # Read the leading dimensions once to avoid repeated shape indexing.
    n_demes = ind_count_all.shape[0]
    # Sex axis: 0=female, 1=male (and potentially more in generalized models).
    n_sexes = ind_count_all.shape[1]
    # Number of age buckets per deme.
    n_ages = ind_count_all.shape[2]
    # Number of genotype buckets per sex/age.
    n_genotypes = ind_count_all.shape[3]
    # Thread-local accumulation avoids write conflicts across ``prange`` source
    # lanes. A final reduction merges all thread-local tensors.
    # Thread-local individual tensor accumulator.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    # Thread-local sperm tensor accumulator.
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)

    # Parallelize by source deme so each lane processes one source row at a time.
    for src in prange(n_demes):
        # Identify current thread lane id.
        thread_id = get_thread_id()
        # Alias current lane's individual accumulator.
        out_ind = out_ind_by_thread[thread_id]
        # Alias current lane's sperm accumulator.
        out_sperm = out_sperm_by_thread[thread_id]

        # ``src_row_total`` answers whether this source has any outbound path.
        # ``src_nnz`` is the sparse row length (effective destination count).
        # Sum of outbound probabilities for this source row.
        src_row_total = row_total[src]
        # Number of valid sparse destinations for this source row.
        src_nnz = int(row_nnz[src])

        # Iterate all age buckets for female virgin/sperm-coupled migration.
        for age in range(n_ages):
            for female_genotype in range(n_genotypes):
                # Recompute stored sperm total so virgin females can be separated.
                stored_total = 0.0
                for male_genotype in range(n_genotypes):
                    # Aggregate all sperm contributors for one female bucket.
                    stored_total += sperm_store_all[src, age, female_genotype, male_genotype]

                # Total females in this bucket before splitting.
                female_total = ind_count_all[src, 0, age, female_genotype]
                # Virgin females are total females minus mated females implied by sperm.
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    # Clamp tiny negative drift from float arithmetic.
                    virgin_count = 0.0

                if src_row_total > 0.0:
                    # Split source bucket into stay part and outbound part.
                    outbound = virgin_count * rate
                    stay = virgin_count - outbound
                    # Keep non-migrating mass at source.
                    out_ind[src, 0, age, female_genotype] += stay
                    for nnz_idx in range(src_nnz):
                        # Sparse traversal: only touch real destinations.
                        dst = int(row_dst_idx[src, nnz_idx])
                        prob = row_dst_prob[src, nnz_idx]
                        # Add migrated virgin-female mass to destination bucket.
                        out_ind[dst, 0, age, female_genotype] += outbound * prob
                else:
                    # No outbound edges: everything stays at source.
                    out_ind[src, 0, age, female_genotype] += virgin_count

                for male_genotype in range(n_genotypes):
                    # One sperm sub-bucket value.
                    sperm_value = sperm_store_all[src, age, female_genotype, male_genotype]
                    if src_row_total > 0.0:
                        # Outbound sperm mass selected by migration rate.
                        outbound_sperm = sperm_value * rate
                        # Residual sperm mass kept locally.
                        stay_sperm = sperm_value - outbound_sperm
                        # Keep local sperm state.
                        out_sperm[src, age, female_genotype, male_genotype] += stay_sperm
                        # Keep matching mated-female count in female individual tensor.
                        out_ind[src, 0, age, female_genotype] += stay_sperm
                        for nnz_idx in range(src_nnz):
                            # Keep sperm and mated-female mass synchronized.
                            dst = int(row_dst_idx[src, nnz_idx])
                            prob = row_dst_prob[src, nnz_idx]
                            # Destination share of outbound sperm.
                            moved_sperm = outbound_sperm * prob
                            # Update destination sperm tensor.
                            out_sperm[dst, age, female_genotype, male_genotype] += moved_sperm
                            # Update destination female count consistently.
                            out_ind[dst, 0, age, female_genotype] += moved_sperm
                    else:
                        # No outbound edges: sperm stays local.
                        out_sperm[src, age, female_genotype, male_genotype] += sperm_value
                        # Matching female mass also stays local.
                        out_ind[src, 0, age, female_genotype] += sperm_value

        # Migrate all non-female-virgin buckets (e.g., males and extra sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for genotype in range(n_genotypes):
                    # Scalar bucket value before migration split.
                    value = ind_count_all[src, sex, age, genotype]
                    if src_row_total > 0.0:
                        # Outbound expectation in deterministic mode.
                        outbound = value * rate
                        # Residual local part.
                        stay = value - outbound
                        # Keep local part at source.
                        out_ind[src, sex, age, genotype] += stay
                        for nnz_idx in range(src_nnz):
                            dst = int(row_dst_idx[src, nnz_idx])
                            prob = row_dst_prob[src, nnz_idx]
                            # Add destination share of outbound mass.
                            out_ind[dst, sex, age, genotype] += outbound * prob
                    else:
                        # No route: full bucket remains at source.
                        out_ind[src, sex, age, genotype] += value

    # Merge thread-local partial sums to one deterministic final state.
    # Allocate final merged individual tensor.
    out_ind = np.zeros_like(ind_count_all)
    # Allocate final merged sperm tensor.
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        # Reduce one thread lane into global output.
        out_ind += out_ind_by_thread[thread_id]
        # Reduce sperm lane as well.
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm


@njit_switch(cache=True, parallel=True)
def _apply_spatial_adjacency_migration_stochastic_parallel(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_nnz: NDArray[np.int64],
    rate: float,
    use_continuous_sampling: bool,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stochastic migration step using sparse routing rows.

    Each source deme samples outbound mass from its scalar buckets, then
    distributes the sampled outbound amount only to valid destinations listed
    in its sparse row.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        row_dst_idx: Sparse destination indices per source row.
        row_dst_prob: Sparse destination probabilities per source row.
        row_nnz: Number of valid destinations per source row.
        rate: Migration probability.
        use_continuous_sampling: Whether to use continuous sampling.
        n_threads: Number of thread lanes reserved for thread-local buffers.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one stochastic migration.
    """
    # Leading dimensions for loop bounds.
    n_demes = ind_count_all.shape[0]
    n_sexes = ind_count_all.shape[1]
    n_ages = ind_count_all.shape[2]
    n_genotypes = ind_count_all.shape[3]
    # Maximum sparse row length allocated in compact table.
    max_nnz = row_dst_idx.shape[1]

    # Keep per-thread accumulation buffers to avoid data races in ``prange``.
    # Per-thread individual-state output accumulator.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    # Per-thread sperm-state output accumulator.
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)
    # Per-thread scratch space for sparse outbound allocation.
    distributed_by_thread = np.zeros((n_threads, max_nnz), dtype=np.float64)

    # Parallelize by source deme in stochastic mode as well.
    for src in prange(n_demes):
        # Active thread lane id.
        thread_id = get_thread_id()
        # Scratch vector reused across buckets for this source/thread.
        distributed = distributed_by_thread[thread_id]
        # Lane-local individual output alias.
        out_ind = out_ind_by_thread[thread_id]
        # Lane-local sperm output alias.
        out_sperm = out_sperm_by_thread[thread_id]
        # Read sparse row metadata once per source for reuse in all buckets.
        src_nnz = int(row_nnz[src])

        # Handle female virgin + sperm-coupled buckets.
        for age in range(n_ages):
            for female_genotype in range(n_genotypes):
                # Aggregate sperm contributors to derive virgin females.
                stored_total = 0.0
                for male_genotype in range(n_genotypes):
                    stored_total += sperm_store_all[src, age, female_genotype, male_genotype]

                # Total females for this source/age/genotype bucket.
                female_total = ind_count_all[src, 0, age, female_genotype]
                # Derived virgin count.
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    # Clamp minor floating-point drift.
                    virgin_count = 0.0

                _migrate_scalar_bucket(
                    # Virgin female scalar bucket to migrate.
                    value=virgin_count,
                    row_dst_idx=row_dst_idx[src],
                    row_dst_prob=row_dst_prob[src],
                    row_dst_count=src_nnz,
                    rate=rate,
                    is_stochastic=True,
                    use_continuous_sampling=use_continuous_sampling,
                    distributed=distributed,
                    out_ind=out_ind,
                    source_idx=src,
                    sex_idx=0,
                    age_idx=age,
                    genotype_idx=female_genotype,
                )

                for male_genotype in range(n_genotypes):
                    _migrate_sperm_bucket(
                        # One sperm-storage scalar bucket.
                        value=sperm_store_all[src, age, female_genotype, male_genotype],
                        row_dst_idx=row_dst_idx[src],
                        row_dst_prob=row_dst_prob[src],
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=True,
                        use_continuous_sampling=use_continuous_sampling,
                        distributed=distributed,
                        out_ind=out_ind,
                        out_sperm=out_sperm,
                        source_idx=src,
                        age_idx=age,
                        female_genotype_idx=female_genotype,
                        male_genotype_idx=male_genotype,
                    )

        # Handle remaining individual buckets (male and other sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for genotype in range(n_genotypes):
                    _migrate_scalar_bucket(
                        # Scalar source bucket for this axis tuple.
                        value=ind_count_all[src, sex, age, genotype],
                        row_dst_idx=row_dst_idx[src],
                        row_dst_prob=row_dst_prob[src],
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=True,
                        use_continuous_sampling=use_continuous_sampling,
                        distributed=distributed,
                        out_ind=out_ind,
                        source_idx=src,
                        sex_idx=sex,
                        age_idx=age,
                        genotype_idx=genotype,
                    )

    # Deterministic reduction from thread-local partial states.
    # Allocate global merged individual tensor.
    out_ind = np.zeros_like(ind_count_all)
    # Allocate global merged sperm tensor.
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        # Merge lane-local contributions.
        out_ind += out_ind_by_thread[thread_id]
        # Merge sperm contributions.
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm


@njit_switch(cache=True)
def _build_kernel_offset_table(
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], int]:
    """Build compact non-zero kernel offsets once per migration call.

    The migration kernel is source-relative. This helper converts kernel matrix
    coordinates into flat offset vectors ``(d_row, d_col, weight)`` and drops
    zero-weight entries (and optionally the center). The returned vectors are
    reused for every source deme in the same migration call.

    Args:
        migration_kernel: Odd-sized kernel whose center corresponds to the
            source deme.
        kernel_include_center: Whether the center kernel cell contributes to
            outbound routing.

    Returns:
        A tuple ``(d_row, d_col, weights, nnz)`` where:

        - ``d_row`` stores row offsets relative to each source.
        - ``d_col`` stores column offsets relative to each source.
        - ``weights`` stores raw positive kernel weights.
        - ``nnz`` is the number of valid entries in those arrays.
    """
    # Kernel shape and center index are used to convert matrix coordinates
    # into source-relative offsets.
    kernel_rows = migration_kernel.shape[0]
    kernel_cols = migration_kernel.shape[1]
    center_row = kernel_rows // 2
    center_col = kernel_cols // 2
    # The maximum possible non-zero count is the full kernel size.
    max_nnz = kernel_rows * kernel_cols

    # Preallocate compact vectors. Only the first ``nnz`` entries are valid.
    d_row = np.zeros(max_nnz, dtype=np.int64)
    d_col = np.zeros(max_nnz, dtype=np.int64)
    weights = np.zeros(max_nnz, dtype=np.float64)

    # Write cursor of the compact representation.
    nnz = 0
    for kernel_row in range(kernel_rows):
        for kernel_col in range(kernel_cols):
            # Optionally exclude self-loop from the kernel center.
            if (not kernel_include_center) and kernel_row == center_row and kernel_col == center_col:
                continue
            # Ignore non-positive entries so sparse traversal stays compact.
            weight = migration_kernel[kernel_row, kernel_col]
            if weight <= 0.0:
                continue

            # Convert kernel coordinates into source-relative offsets.
            d_row[nnz] = kernel_row - center_row
            d_col[nnz] = kernel_col - center_col
            weights[nnz] = weight
            nnz += 1

    return d_row, d_col, weights, nnz


@njit_switch(cache=True)
def _build_source_kernel_sparse_row(
    source_idx: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    d_row: NDArray[np.int64],
    d_col: NDArray[np.int64],
    weights: NDArray[np.float64],
    kernel_nnz: int,
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
) -> int:
    """Build one source row in compact sparse form from kernel offsets.

    This helper applies the compact kernel offset table to exactly one source
    deme. It resolves valid destination demes under either wrapped or clipped
    borders, writes destination indices and raw weights, then normalizes the
    row probabilities in place.

    Args:
        source_idx: Flattened source-deme index.
        topology_rows: Number of rows in the topology grid.
        topology_cols: Number of columns in the topology grid.
        topology_wrap: Whether offsets wrap around topology borders.
        d_row: Compact row offsets produced by
            ``_build_kernel_offset_table``.
        d_col: Compact column offsets produced by
            ``_build_kernel_offset_table``.
        weights: Compact raw kernel weights produced by
            ``_build_kernel_offset_table``.
        kernel_nnz: Number of valid compact offsets.
        row_dst_idx: Preallocated destination-index output buffer.
        row_dst_prob: Preallocated destination-probability output buffer.

    Returns:
        The number of valid destinations written into
        ``row_dst_idx``/``row_dst_prob``.
    """
    # Clear output row so any unused tail slots remain deterministic.
    for idx in range(row_dst_idx.shape[0]):
        row_dst_idx[idx] = -1
        row_dst_prob[idx] = 0.0

    # Invalid topology or empty kernel means no outbound destinations.
    if topology_rows <= 0 or topology_cols <= 0 or kernel_nnz <= 0:
        return 0

    # Decode flattened source index into grid coordinates.
    src_row = source_idx // topology_cols
    src_col = source_idx % topology_cols

    # ``total`` tracks raw weight mass for later normalization.
    total = 0.0
    # ``count`` is the number of valid destinations written so far.
    count = 0
    for idx in range(kernel_nnz):
        # Apply source-relative offset.
        dst_row = src_row + int(d_row[idx])
        dst_col = src_col + int(d_col[idx])

        if topology_wrap:
            # Periodic topology: wrap both axes.
            dst_row %= topology_rows
            dst_col %= topology_cols
        elif dst_row < 0 or dst_row >= topology_rows or dst_col < 0 or dst_col >= topology_cols:
            # Non-wrapping topology: drop out-of-bounds offsets.
            continue

        # Write one sparse destination entry.
        row_dst_idx[count] = dst_row * topology_cols + dst_col
        row_dst_prob[count] = weights[idx]
        total += weights[idx]
        count += 1

    if total > 0.0:
        # Normalize only the written prefix so row probabilities sum to one.
        inv_total = 1.0 / total
        for idx in range(count):
            row_dst_prob[idx] *= inv_total

    return count


@njit_switch(cache=True, parallel=True)
def _apply_spatial_kernel_migration_deterministic_parallel(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    rate: float,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply deterministic migration in kernel/topology mode.

    Unlike adjacency-mode kernels that may prebuild full sparse rows for all
    source demes, this function constructs one sparse row on-the-fly per source
    inside ``prange``. For fixed small kernels this keeps routing work close to
    ``O(kernel_nnz * n_demes)``.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        topology_rows: Number of rows in the topology grid.
        topology_cols: Number of columns in the topology grid.
        topology_wrap: Whether topology wraps around borders.
        migration_kernel: Odd-sized kernel interpreted as source-relative
            offsets.
        kernel_include_center: Whether center kernel cell contributes.
        rate: Deterministic migration probability for each scalar bucket.
        n_threads: Number of thread lanes used for thread-local accumulation.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one deterministic migration
        step.
    """
    # Leading dimensions are read once for tighter inner loops.
    n_demes = ind_count_all.shape[0]
    n_sexes = ind_count_all.shape[1]
    n_ages = ind_count_all.shape[2]
    n_genotypes = ind_count_all.shape[3]

    # Build compact kernel offsets once and reuse across all source demes.
    d_row, d_col, weights, kernel_nnz = _build_kernel_offset_table(
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
    )

    # Use thread-local output tensors to avoid write races in ``prange``.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)
    # Per-thread sparse-row buffers avoid cross-thread mutation conflicts.
    row_dst_idx_by_thread = np.full((n_threads, max(1, kernel_nnz)), -1, dtype=np.int64)
    row_dst_prob_by_thread = np.zeros((n_threads, max(1, kernel_nnz)), dtype=np.float64)
    # Per-thread scratch vector for outbound destination split.
    distributed_by_thread = np.zeros((n_threads, max(1, kernel_nnz)), dtype=np.float64)

    for src in prange(n_demes):
        # Resolve current thread lane and lane-local buffers.
        thread_id = get_thread_id()
        out_ind = out_ind_by_thread[thread_id]
        out_sperm = out_sperm_by_thread[thread_id]
        row_dst_idx = row_dst_idx_by_thread[thread_id]
        row_dst_prob = row_dst_prob_by_thread[thread_id]
        distributed = distributed_by_thread[thread_id]

        # Build sparse outbound row for this source only.
        src_nnz = _build_source_kernel_sparse_row(
            source_idx=src,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=topology_wrap,
            d_row=d_row,
            d_col=d_col,
            weights=weights,
            kernel_nnz=kernel_nnz,
            row_dst_idx=row_dst_idx,
            row_dst_prob=row_dst_prob,
        )

        # Female buckets are split into virgin + sperm-coupled parts.
        for age in range(n_ages):
            for female_genotype in range(n_genotypes):
                # Aggregate stored sperm to recover virgin female mass.
                stored_total = 0.0
                for male_genotype in range(n_genotypes):
                    stored_total += sperm_store_all[src, age, female_genotype, male_genotype]

                female_total = ind_count_all[src, 0, age, female_genotype]
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    # Clamp tiny numerical drift.
                    virgin_count = 0.0

                # Migrate virgin female scalar bucket.
                _migrate_scalar_bucket(
                    value=virgin_count,
                    row_dst_idx=row_dst_idx,
                    row_dst_prob=row_dst_prob,
                    row_dst_count=src_nnz,
                    rate=rate,
                    is_stochastic=False,
                    use_continuous_sampling=False,
                    distributed=distributed,
                    out_ind=out_ind,
                    source_idx=src,
                    sex_idx=0,
                    age_idx=age,
                    genotype_idx=female_genotype,
                )

                for male_genotype in range(n_genotypes):
                    # Migrate sperm bucket and synchronized mated-female mass.
                    _migrate_sperm_bucket(
                        value=sperm_store_all[src, age, female_genotype, male_genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=False,
                        use_continuous_sampling=False,
                        distributed=distributed,
                        out_ind=out_ind,
                        out_sperm=out_sperm,
                        source_idx=src,
                        age_idx=age,
                        female_genotype_idx=female_genotype,
                        male_genotype_idx=male_genotype,
                    )

        # Migrate remaining individual buckets (male and other sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for genotype in range(n_genotypes):
                    _migrate_scalar_bucket(
                        value=ind_count_all[src, sex, age, genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=False,
                        use_continuous_sampling=False,
                        distributed=distributed,
                        out_ind=out_ind,
                        source_idx=src,
                        sex_idx=sex,
                        age_idx=age,
                        genotype_idx=genotype,
                    )

    # Deterministically merge thread-local partial tensors.
    out_ind = np.zeros_like(ind_count_all)
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        out_ind += out_ind_by_thread[thread_id]
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm


@njit_switch(cache=True, parallel=True)
def _apply_spatial_kernel_migration_stochastic_parallel(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    rate: float,
    use_continuous_sampling: bool,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply stochastic migration in kernel/topology mode.

    This function uses the same on-the-fly sparse row construction as the
    deterministic kernel path, but samples outbound mass for each scalar bucket
    (discrete Binomial/Multinomial or continuous Beta/Dirichlet approximation
    depending on flags).

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        topology_rows: Number of rows in the topology grid.
        topology_cols: Number of columns in the topology grid.
        topology_wrap: Whether topology wraps around borders.
        migration_kernel: Odd-sized kernel interpreted as source-relative
            offsets.
        kernel_include_center: Whether center kernel cell contributes.
        rate: Migration probability for each scalar bucket.
        use_continuous_sampling: Whether stochastic routing uses continuous
            approximation samplers.
        n_threads: Number of thread lanes used for thread-local accumulation.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one stochastic migration
        step.
    """
    # Leading dimensions are read once for tighter inner loops.
    n_demes = ind_count_all.shape[0]
    n_sexes = ind_count_all.shape[1]
    n_ages = ind_count_all.shape[2]
    n_genotypes = ind_count_all.shape[3]

    # Build compact kernel offsets once and reuse across all source demes.
    d_row, d_col, weights, kernel_nnz = _build_kernel_offset_table(
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
    )

    # Use thread-local output tensors to avoid write races in ``prange``.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)
    # Per-thread sparse-row buffers avoid cross-thread mutation conflicts.
    row_dst_idx_by_thread = np.full((n_threads, max(1, kernel_nnz)), -1, dtype=np.int64)
    row_dst_prob_by_thread = np.zeros((n_threads, max(1, kernel_nnz)), dtype=np.float64)
    # Per-thread scratch vector for outbound destination split.
    distributed_by_thread = np.zeros((n_threads, max(1, kernel_nnz)), dtype=np.float64)

    for src in prange(n_demes):
        # Resolve current thread lane and lane-local buffers.
        thread_id = get_thread_id()
        out_ind = out_ind_by_thread[thread_id]
        out_sperm = out_sperm_by_thread[thread_id]
        row_dst_idx = row_dst_idx_by_thread[thread_id]
        row_dst_prob = row_dst_prob_by_thread[thread_id]
        distributed = distributed_by_thread[thread_id]

        # Build sparse outbound row for this source only.
        src_nnz = _build_source_kernel_sparse_row(
            source_idx=src,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=topology_wrap,
            d_row=d_row,
            d_col=d_col,
            weights=weights,
            kernel_nnz=kernel_nnz,
            row_dst_idx=row_dst_idx,
            row_dst_prob=row_dst_prob,
        )

        # Female buckets are split into virgin + sperm-coupled parts.
        for age in range(n_ages):
            for female_genotype in range(n_genotypes):
                # Aggregate stored sperm to recover virgin female mass.
                stored_total = 0.0
                for male_genotype in range(n_genotypes):
                    stored_total += sperm_store_all[src, age, female_genotype, male_genotype]

                female_total = ind_count_all[src, 0, age, female_genotype]
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    # Clamp tiny numerical drift.
                    virgin_count = 0.0

                # Migrate virgin female scalar bucket with stochastic routing.
                _migrate_scalar_bucket(
                    value=virgin_count,
                    row_dst_idx=row_dst_idx,
                    row_dst_prob=row_dst_prob,
                    row_dst_count=src_nnz,
                    rate=rate,
                    is_stochastic=True,
                    use_continuous_sampling=use_continuous_sampling,
                    distributed=distributed,
                    out_ind=out_ind,
                    source_idx=src,
                    sex_idx=0,
                    age_idx=age,
                    genotype_idx=female_genotype,
                )

                for male_genotype in range(n_genotypes):
                    # Migrate sperm bucket and synchronized mated-female mass.
                    _migrate_sperm_bucket(
                        value=sperm_store_all[src, age, female_genotype, male_genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=True,
                        use_continuous_sampling=use_continuous_sampling,
                        distributed=distributed,
                        out_ind=out_ind,
                        out_sperm=out_sperm,
                        source_idx=src,
                        age_idx=age,
                        female_genotype_idx=female_genotype,
                        male_genotype_idx=male_genotype,
                    )

        # Migrate remaining individual buckets (male and other sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for genotype in range(n_genotypes):
                    _migrate_scalar_bucket(
                        value=ind_count_all[src, sex, age, genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=True,
                        use_continuous_sampling=use_continuous_sampling,
                        distributed=distributed,
                        out_ind=out_ind,
                        source_idx=src,
                        sex_idx=sex,
                        age_idx=age,
                        genotype_idx=genotype,
                    )

    # Deterministically merge thread-local partial tensors.
    out_ind = np.zeros_like(ind_count_all)
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        out_ind += out_ind_by_thread[thread_id]
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm


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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one synchronized migration step over the deme axis.

    This kernel handles both migration backends currently exposed by
    ``SpatialPopulation``:

    - ``migration_mode == 0``: use the provided dense outbound adjacency row.
    - ``migration_mode == 1``: derive the outbound row on the fly from the
      kernel and topology metadata.

    Migration is synchronized across demes. Every source deme reads from the
    pre-migration state and contributes to fresh output buffers, which avoids
    order-dependent bias from in-place updates. In stochastic mode, outbound
    mass is sampled at the smallest scalar bucket level:

    - ``individual_count[sex, age, genotype]`` for ordinary individuals
    - ``sperm_store[age, female_genotype, male_genotype]`` for stored sperm
    - virgin females are inferred as ``female_total - stored_sperm_total`` and
      sampled separately so mated and unmated females are not merged

    Implementation note:
        The kernel first compacts each source row into sparse destinations and
        then iterates only those destinations. For kernel-topology migration
        with small fixed kernel support, this makes migration close to
        ``O(n_demes * kernel_nonzero)`` instead of ``O(n_demes^2)`` scanning.
        In adjacency mode the complexity becomes ``O(total_nonzero_edges)``.

    Args:
        ind_count_all: Stacked individual counts with shape
            ``(n_demes, n_sexes, n_ages, n_genotypes)``.
        sperm_store_all: Stacked sperm storage with shape
            ``(n_demes, n_ages, n_genotypes, n_genotypes)``.
        adjacency: Dense outbound migration matrix. Used directly when
            ``migration_mode == 0``.
        migration_mode: Integer backend selector. ``0`` means dense adjacency;
            ``1`` means topology/kernel-driven routing.
        topology_rows: Number of topology rows for kernel migration.
        topology_cols: Number of topology columns for kernel migration.
        topology_wrap: Whether topology coordinates wrap at the borders.
        migration_kernel: Odd-sized kernel used when ``migration_mode == 1``.
        kernel_include_center: Whether the kernel center contributes outbound
            mass to the source deme.
        rate: Fraction of each scalar bucket that attempts to migrate.
        is_stochastic: Whether migration should use stochastic sampling.
        use_continuous_sampling: Whether stochastic mode should use the
            continuous Dirichlet/Beta approximation instead of discrete
            Binomial/Multinomial draws.

    Returns:
        A tuple ``(ind_next, sperm_next)`` containing the migrated state.
    """
    if rate <= 0.0:
        return ind_count_all, sperm_store_all

    # Kernel-topology mode can route with O(kernel_nnz * n_demes) complexity
    # without building n_demes x n_demes sparse tables every tick.
    if migration_mode == 1:
        if not is_stochastic:
            return _apply_spatial_kernel_migration_deterministic_parallel(
                ind_count_all=ind_count_all,
                sperm_store_all=sperm_store_all,
                topology_rows=topology_rows,
                topology_cols=topology_cols,
                topology_wrap=topology_wrap,
                migration_kernel=migration_kernel,
                kernel_include_center=kernel_include_center,
                rate=rate,
                n_threads=numba_max_threads,
            )

        return _apply_spatial_kernel_migration_stochastic_parallel(
            ind_count_all=ind_count_all,
            sperm_store_all=sperm_store_all,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=topology_wrap,
            migration_kernel=migration_kernel,
            kernel_include_center=kernel_include_center,
            rate=rate,
            use_continuous_sampling=use_continuous_sampling,
            n_threads=numba_max_threads,
        )

    row_dst_idx, row_dst_prob, row_nnz, row_total = _build_sparse_migration_rows(
        adjacency=adjacency,
        migration_mode=migration_mode,
        topology_rows=topology_rows,
        topology_cols=topology_cols,
        topology_wrap=topology_wrap,
        migration_kernel=migration_kernel,
        kernel_include_center=kernel_include_center,
    )

    if not is_stochastic:
        return _apply_spatial_adjacency_migration_deterministic_parallel(
            ind_count_all=ind_count_all,
            sperm_store_all=sperm_store_all,
            row_dst_idx=row_dst_idx,
            row_dst_prob=row_dst_prob,
            row_nnz=row_nnz,
            row_total=row_total,
            rate=rate,
            n_threads=numba_max_threads,
        )

    return _apply_spatial_adjacency_migration_stochastic_parallel(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        row_dst_idx=row_dst_idx,
        row_dst_prob=row_dst_prob,
        row_nnz=row_nnz,
        rate=rate,
        use_continuous_sampling=use_continuous_sampling,
        n_threads=numba_max_threads,
    )


@njit_switch(cache=True)
def _build_sparse_migration_rows(
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Build sparse outbound rows for all source demes.

    The output stores, for each source deme, the list of valid destination
    indices and their normalized probabilities. Only non-zero destinations are
    stored in the leading ``row_nnz[src]`` slots.

    Args:
        adjacency: Dense adjacency matrix for adjacency mode.
        migration_mode: Migration backend selector.
        topology_rows: Topology row count for kernel mode.
        topology_cols: Topology column count for kernel mode.
        topology_wrap: Topology wrap flag for kernel mode.
        migration_kernel: Migration kernel for kernel mode.
        kernel_include_center: Whether the center kernel element is included.

    Returns:
        A tuple ``(row_dst_idx, row_dst_prob, row_nnz, row_total)``.
    """
    # Number of source demes equals adjacency row count.
    n_demes = adjacency.shape[0]
    # Destination index table; unused slots are -1.
    row_dst_idx = np.full((n_demes, n_demes), -1, dtype=np.int64)
    # Destination probability table parallel to ``row_dst_idx``.
    row_dst_prob = np.zeros((n_demes, n_demes), dtype=np.float64)
    # Effective sparse row length per source.
    row_nnz = np.zeros(n_demes, dtype=np.int64)
    # Sum of positive weights in each source row.
    row_total = np.zeros(n_demes, dtype=np.float64)

    # ``row_probs`` is only needed when migration rows are synthesized from
    # topology/kernel metadata. For dense adjacency mode we compact directly
    # from ``adjacency[src, :]`` in a single pass.
    row_probs = np.zeros(n_demes, dtype=np.float64)
    for src in range(n_demes):
        # Adjacency mode can be compacted in one pass: iterate dense row once
        # and write non-zero destinations directly to sparse buffers.
        if migration_mode == 0:
            src_total = 0.0
            src_nnz = 0
            for dst in range(n_demes):
                prob = adjacency[src, dst]
                if prob <= 0.0:
                    continue
                row_dst_idx[src, src_nnz] = dst
                row_dst_prob[src, src_nnz] = prob
                src_nnz += 1
                src_total += prob

            row_nnz[src] = src_nnz
            row_total[src] = src_total
            continue

        # Fill dense row for this source according to migration backend.
        _populate_migration_row(
            adjacency=adjacency,
            migration_mode=migration_mode,
            topology_rows=topology_rows,
            topology_cols=topology_cols,
            topology_wrap=topology_wrap,
            migration_kernel=migration_kernel,
            kernel_include_center=kernel_include_center,
            source_idx=src,
            row_probs=row_probs,
        )

        # Compact dense row into sparse (index, probability) pairs.
        # Running sum of kept probabilities in this row.
        src_total = 0.0
        # Running sparse length for this source.
        src_nnz = 0
        for dst in range(n_demes):
            # Candidate probability in dense scratch row.
            prob = row_probs[dst]
            if prob <= 0.0:
                # Skip zero-weight destination.
                continue
            # Store destination index in compact row.
            row_dst_idx[src, src_nnz] = dst
            # Store probability in compact row.
            row_dst_prob[src, src_nnz] = prob
            # Advance sparse write cursor.
            src_nnz += 1
            # Accumulate row probability mass.
            src_total += prob

        # Persist sparse row length.
        row_nnz[src] = src_nnz
        # Persist row mass for fast no-edge checks.
        row_total[src] = src_total

    return row_dst_idx, row_dst_prob, row_nnz, row_total


@njit_switch(cache=True)
def _populate_migration_row(
    adjacency: NDArray[np.float64],
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    source_idx: int,
    row_probs: NDArray[np.float64],
) -> None:
    """Fill one normalized outbound migration row for a single source deme.

    Args:
        adjacency: Dense outbound migration matrix.
        migration_mode: Backend selector. ``0`` reads directly from adjacency;
            ``1`` computes one row from topology-aware kernel offsets.
        topology_rows: Number of rows in the grid used by kernel migration.
        topology_cols: Number of columns in the grid used by kernel migration.
        topology_wrap: Whether out-of-bounds kernel offsets wrap around.
        migration_kernel: Odd-sized migration kernel.
        kernel_include_center: Whether the kernel center contributes to the
            source deme itself.
        source_idx: Flattened source-deme index.
        row_probs: Preallocated output buffer that receives the normalized
            outbound weights for ``source_idx``.
    """
    n_demes = row_probs.shape[0]
    for idx in range(n_demes):
        row_probs[idx] = 0.0

    if migration_mode == 0:
        # Dense adjacency mode already stores one outbound row per source
        # deme, so this helper just copies the precomputed probabilities.
        for dst_idx in range(n_demes):
            row_probs[dst_idx] = adjacency[source_idx, dst_idx]
        return

    if topology_rows <= 0 or topology_cols <= 0:
        return

    src_row = source_idx // topology_cols
    src_col = source_idx % topology_cols
    kernel_rows = migration_kernel.shape[0]
    kernel_cols = migration_kernel.shape[1]
    center_row = kernel_rows // 2
    center_col = kernel_cols // 2
    total = 0.0

    for kernel_row in range(kernel_rows):
        for kernel_col in range(kernel_cols):
            # The kernel is interpreted as source-relative offsets in the
            # flattened topology grid rather than as a matrix that directly
            # mixes neighboring rows of the state tensor.
            if (not kernel_include_center) and kernel_row == center_row and kernel_col == center_col:
                continue
            weight = migration_kernel[kernel_row, kernel_col]
            if weight <= 0.0:
                continue

            dst_row = src_row + kernel_row - center_row
            dst_col = src_col + kernel_col - center_col

            if topology_wrap:
                # Periodic boundaries wrap offsets onto the opposite edge.
                dst_row %= topology_rows
                dst_col %= topology_cols
            elif dst_row < 0 or dst_row >= topology_rows or dst_col < 0 or dst_col >= topology_cols:
                # Non-wrapping topologies simply drop invalid offsets. The
                # remaining valid weights are renormalized below.
                continue

            # Kernel routing is constructed from source-relative offsets in the
            # topology index space, then normalized over valid destinations
            # only. This preserves total migrating mass at borders.
            dst_idx = dst_row * topology_cols + dst_col
            row_probs[dst_idx] += weight
            total += weight

    if total > 0.0:
        for dst_idx in range(n_demes):
            row_probs[dst_idx] /= total


@njit_switch(cache=True)
def _sample_outbound_count(
    value: float,
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
) -> float:
    """Compute the outbound amount for one scalar bucket.

    Args:
        value: Source bucket mass before migration.
        rate: Migration probability applied to this bucket.
        is_stochastic: Whether to sample rather than use the expectation.
        use_continuous_sampling: Whether stochastic mode should use the
            continuous Beta approximation.

    Returns:
        The amount of mass that leaves the source bucket.
    """
    if value <= 0.0 or rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return float(value)
    if not is_stochastic:
        # Deterministic migration moves the expectation directly.
        return float(value) * rate
    if use_continuous_sampling:
        # Continuous mode keeps the state real-valued while still injecting
        # stochasticity into the outbound amount.
        return float(alg.continuous_binomial(float(value), float(rate)))
    # Discrete mode treats each scalar bucket as a Bernoulli family and keeps
    # the migrated amount integer-valued.
    return float(nbc.binomial(int(round(float(value))), float(rate)))


@njit_switch(cache=True)
def _distribute_outbound_count(
    outbound: float,
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    is_stochastic: bool,
    use_continuous_sampling: bool,
    distributed: NDArray[np.float64],
) -> None:
    """Distribute one outbound amount across sparse destinations.

    Args:
        outbound: Total mass already selected to leave the source bucket.
        row_dst_prob: Destination probabilities for one source deme.
        row_dst_count: Number of valid destination entries in ``row_dst_prob``.
        is_stochastic: Whether to sample rather than use expectations.
        use_continuous_sampling: Whether stochastic mode should use the
            continuous Dirichlet approximation.
        distributed: Preallocated output vector. On return, contains the
            destination-wise migrated mass in the first ``row_dst_count`` slots.
    """
    # Always clear the scratch vector before filling.
    for idx in range(distributed.shape[0]):
        distributed[idx] = 0.0

    # Nothing selected to migrate.
    if outbound <= 0.0:
        return

    # No valid destinations in sparse row.
    if row_dst_count <= 0:
        return

    # Compute row mass in case probabilities are not perfectly normalized.
    total = 0.0
    for idx in range(row_dst_count):
        total += row_dst_prob[idx]
    if total <= 0.0:
        # No effective destination weight means all outbound mass is treated as
        # staying at source by the caller's ``value - moved_total`` logic.
        return

    # Deterministic path: direct expected-value split.
    if not is_stochastic:
        for idx in range(row_dst_count):
            distributed[idx] = outbound * (row_dst_prob[idx] / total)
        return

    # Build normalized probability vector for stochastic samplers.
    probs = np.zeros(row_dst_count, dtype=np.float64)
    for idx in range(row_dst_count):
        # Normalize defensively here as well. Adjacency rows should already
        # sum to one, but kernel rows may have been built from a subset of
        # valid border offsets.
        probs[idx] = row_dst_prob[idx] / total

    # Continuous stochastic split using Dirichlet-like sampler.
    if use_continuous_sampling:
        # Continuous multinomial keeps real-valued buckets while conserving
        # the sampled outbound total.
        alg.continuous_multinomial(float(outbound), probs, distributed)
        return

    # Discrete multinomial allocates an integer outbound count to
    # destination demes while preserving the total exactly.
    # Discrete stochastic split preserving integer outbound total.
    sampled = nbc.multinomial(int(round(float(outbound))), probs)
    for idx in range(row_dst_count):
        distributed[idx] = float(sampled[idx])


@njit_switch(cache=True)
def _migrate_scalar_bucket(
    value: float,
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
    distributed: NDArray[np.float64],
    out_ind: NDArray[np.float64],
    source_idx: int,
    sex_idx: int,
    age_idx: int,
    genotype_idx: int,
) -> None:
    """Migrate one ``individual_count`` scalar bucket into the output buffer.

    Args:
        value: Scalar source bucket to migrate.
        row_dst_idx: Destination indices for one source deme.
        row_dst_prob: Destination probabilities for one source deme.
        row_dst_count: Number of valid destination entries.
        rate: Migration probability for this bucket.
        is_stochastic: Whether outbound mass is sampled.
        use_continuous_sampling: Whether stochastic mode uses continuous
            approximations.
        distributed: Scratch buffer reused for destination allocations.
        out_ind: Destination individual-count buffer updated in place.
        source_idx: Source deme index.
        sex_idx: Sex index of the migrating bucket.
        age_idx: Age index of the migrating bucket.
        genotype_idx: Genotype index of the migrating bucket.
    """
    # Sample/compute how much mass leaves this scalar source bucket.
    outbound = _sample_outbound_count(
        value=value,
        rate=rate,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )
    # Split outbound mass across sparse destinations.
    _distribute_outbound_count(
        outbound=outbound,
        row_dst_prob=row_dst_prob,
        row_dst_count=row_dst_count,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
        distributed=distributed,
    )

    # Track total moved mass so residual can stay at source.
    moved_total = 0.0
    for dst_pos in range(row_dst_count):
        # Destination contribution selected for this sparse position.
        moved = distributed[dst_pos]
        moved_total += moved
        # Resolve destination deme index from compact table.
        dst_idx = int(row_dst_idx[dst_pos])
        # Apply migrated mass to destination scalar bucket.
        out_ind[dst_idx, sex_idx, age_idx, genotype_idx] += moved

    # Any mass not assigned to outbound destinations stays in the source
    # bucket. This keeps the update synchronized and avoids in-place bias.
    # Keep any non-moved remainder at the source bucket.
    out_ind[source_idx, sex_idx, age_idx, genotype_idx] += value - moved_total


@njit_switch(cache=True)
def _migrate_sperm_bucket(
    value: float,
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
    distributed: NDArray[np.float64],
    out_ind: NDArray[np.float64],
    out_sperm: NDArray[np.float64],
    source_idx: int,
    age_idx: int,
    female_genotype_idx: int,
    male_genotype_idx: int,
) -> None:
    """Migrate one sperm-storage entry and keep female counts consistent.

    Stored sperm and the matching mated-female mass must move together. This
    helper therefore writes into both ``out_sperm`` and the female slice of
    ``out_ind``.

    Args:
        value: Scalar sperm-storage entry to migrate.
        row_dst_idx: Destination indices for one source deme.
        row_dst_prob: Destination probabilities for one source deme.
        row_dst_count: Number of valid destination entries.
        rate: Migration probability for this bucket.
        is_stochastic: Whether outbound mass is sampled.
        use_continuous_sampling: Whether stochastic mode uses continuous
            approximations.
        distributed: Scratch buffer reused for destination allocations.
        out_ind: Destination individual-count buffer updated in place.
        out_sperm: Destination sperm-storage buffer updated in place.
        source_idx: Source deme index.
        age_idx: Female age index.
        female_genotype_idx: Female genotype index for the sperm bucket.
        male_genotype_idx: Male genotype index for the sperm bucket.
    """
    # Compute/sampling outbound sperm mass for this scalar sperm bucket.
    outbound = _sample_outbound_count(
        value=value,
        rate=rate,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )
    # Split outbound sperm mass across sparse destinations.
    _distribute_outbound_count(
        outbound=outbound,
        row_dst_prob=row_dst_prob,
        row_dst_count=row_dst_count,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
        distributed=distributed,
    )

    # Accumulate moved sperm mass for source residual computation.
    moved_total = 0.0
    for dst_pos in range(row_dst_count):
        # Migrated mass for this destination entry.
        moved = distributed[dst_pos]
        moved_total += moved
        # Resolve concrete destination deme index.
        dst_idx = int(row_dst_idx[dst_pos])
        # Move the sperm entry and the corresponding mated-female mass
        # together, so downstream stages still see consistent sperm ownership.
        out_sperm[dst_idx, age_idx, female_genotype_idx, male_genotype_idx] += moved
        out_ind[dst_idx, 0, age_idx, female_genotype_idx] += moved

    # Residual sperm mass stays at source.
    stay = value - moved_total
    out_sperm[source_idx, age_idx, female_genotype_idx, male_genotype_idx] += stay
    out_ind[source_idx, 0, age_idx, female_genotype_idx] += stay


@njit_switch(cache=True, parallel=True)
def run_spatial_reproduction(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the reproduction stage independently for every deme.

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config: Shared population configuration used by every deme.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after reproduction.
    """
    # Precompute offspring probability tensor (config-dependent, not deme-dependent)
    # to avoid redundant recomputation in the prange loop
    offspring_probability = alg.compute_offspring_probability_tensor(
        meiosis_f=config.genotype_to_gametes_map[0],
        meiosis_m=config.genotype_to_gametes_map[1],
        haplo_to_genotype_map=config.gametes_to_zygote_map,
        n_genotypes=config.n_genotypes,
        n_haplogenotypes=config.n_haploid_genotypes,
        n_glabs=config.n_glabs,
    )
    # Modify input arrays in-place within prange loop.
    # Safe because callers do not expect inputs to remain unmodified.
    for deme_id in prange(ind_count_all.shape[0]):
        # Each deme runs the ordinary single-population kernel independently,
        # then the stacked spatial state is rebuilt deme by deme.
        ind_d, sperm_d = run_reproduction_with_precomputed_offspring_probability(
            ind_count=ind_count_all[deme_id],
            sperm_store=sperm_store_all[deme_id],
            config=config,
            offspring_probability=offspring_probability,
        )
        ind_count_all[deme_id] = ind_d
        sperm_store_all[deme_id] = sperm_d
    return ind_count_all, sperm_store_all


@njit_switch(cache=True, parallel=True)
def run_spatial_survival(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the survival stage independently for every deme.

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config: Shared population configuration used by every deme.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after survival.
    """
    # Modify input arrays in-place within prange loop.
    # Safe because callers do not expect inputs to remain unmodified.
    for deme_id in prange(ind_count_all.shape[0]):
        # Survival remains local to each deme; spatial coupling only happens
        # in the migration stage.
        ind_d, sperm_d = run_survival(
            ind_count=ind_count_all[deme_id],
            sperm_store=sperm_store_all[deme_id],
            config=config,
        )
        ind_count_all[deme_id] = ind_d
        sperm_store_all[deme_id] = sperm_d
    return ind_count_all, sperm_store_all


@njit_switch(cache=True, parallel=True)
def run_spatial_aging(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the aging stage independently for every deme.

    Args:
        ind_count_all: Stacked individual counts for all demes.
        sperm_store_all: Stacked sperm-storage arrays for all demes.
        config: Shared population configuration used by every deme.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after aging.
    """
    # Modify input arrays in-place within prange loop.
    # Safe because callers do not expect inputs to remain unmodified.
    for deme_id in prange(ind_count_all.shape[0]):
        # Aging is also purely within-deme. Migration happens only after the
        # full local lifecycle has finished for the tick.
        ind_d, sperm_d = run_aging(
            ind_count=ind_count_all[deme_id],
            sperm_store=sperm_store_all[deme_id],
            config=config,
        )
        ind_count_all[deme_id] = ind_d
        sperm_store_all[deme_id] = sperm_d
    return ind_count_all, sperm_store_all


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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the migration stage for all demes.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked sperm storage arrays.
        adjacency: Dense outbound adjacency matrix used directly in adjacency
            mode and ignored otherwise.
        migration_mode: Backend selector. ``0`` means adjacency routing and
            ``1`` means topology/kernel routing.
        topology_rows: Number of topology rows for kernel routing.
        topology_cols: Number of topology columns for kernel routing.
        topology_wrap: Whether kernel routing wraps around the topology border.
        migration_kernel: Kernel used when ``migration_mode == 1``.
        kernel_include_center: Whether the kernel center routes outbound mass
            back to the source deme.
        config: Shared population configuration carrying stochastic flags.
        migration_rate: Probability that each scalar bucket attempts to migrate.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after migration.
    """
    if migration_rate <= 0.0:
        return ind_count_all, sperm_store_all
    # Migration is factored into a dedicated helper because it has to keep
    # individual counts and sperm storage synchronized under both dense and
    # topology/kernel routing modes.
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
    )


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
