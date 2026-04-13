"""Kernel-topology-mode spatial migration kernels."""

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

from natal.kernels.migration.adjacency import (
    migrate_scalar_bucket,
    migrate_sperm_bucket,
)
from natal.numba_utils import njit_switch

__all__ = ["apply_spatial_kernel_migration"]


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
    """Populate migration probability distribution for a single source deme (in-place).

    This function applies the precomputed kernel offset table to a specific source deme,
    writing the migration probability distribution directly into the provided output
    buffers. The function operates in-place and returns only the count of valid destinations.

    Key operations:
    1. Clear output buffers for deterministic behavior
    2. Convert flattened source index to grid coordinates
    3. Apply kernel offsets and handle boundary conditions
    4. Write valid destination indices and weights to buffers
    5. Normalize probabilities to sum to 1.0

    Note: The actual migration distribution is written to row_dst_idx and row_dst_prob
    buffers in-place. Only the first [return_value] entries in these buffers contain valid data.

    Args:
        source_idx: Flattened index of the source deme (0-based)
        topology_rows: Number of rows in the topology grid
        topology_cols: Number of columns in the topology grid
        topology_wrap: Boundary handling mode (True=wrap, False=clip)
        d_row: Row offset vector from _build_kernel_offset_table
        d_col: Column offset vector from _build_kernel_offset_table
        weights: Raw weight vector from _build_kernel_offset_table
        kernel_nnz: Number of valid non-zero kernel offsets
        row_dst_idx: Destination index output buffer (modified in-place)
        row_dst_prob: Migration probability output buffer (modified in-place)

    Returns:
        Number of valid destinations written to the output buffers
        (only the first [return_value] entries contain valid data)
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
def apply_spatial_kernel_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: NDArray[np.float64],
    kernel_include_center: bool,
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one synchronized migration step in kernel-topology backend mode.

    Unlike adjacency-mode kernels that may prebuild full sparse rows for all
    source demes, this function constructs one sparse row on-the-fly per source
    inside ``prange``. For fixed small kernels this keeps routing work close to
    ``O(kernel_nnz * n_demes)``.

    The migration behavior is controlled by the is_stochastic and
    use_continuous_sampling parameters:
    - Deterministic mode: exact mathematical calculation
    - Stochastic mode: probabilistic sampling with discrete/continuous options

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        topology_rows: Number of rows in the topology grid.
        topology_cols: Number of columns in the topology grid.
        topology_wrap: Whether topology wraps around borders.
        migration_kernel: Odd-sized kernel interpreted as source-relative offsets.
        kernel_include_center: Whether center kernel cell contributes.
        rate: Migration probability for each scalar bucket.
        is_stochastic: Whether to use stochastic migration sampling.
        use_continuous_sampling: Whether to use continuous approximation samplers.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one migration step.
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
    out_ind_by_thread = np.zeros((numba_max_threads,) + ind_count_all.shape, dtype=np.float64)
    out_sperm_by_thread = np.zeros((numba_max_threads,) + sperm_store_all.shape, dtype=np.float64)
    # Per-thread sparse-row buffers avoid cross-thread mutation conflicts.
    row_dst_idx_by_thread = np.full((numba_max_threads, max(1, kernel_nnz)), -1, dtype=np.int64)
    row_dst_prob_by_thread = np.zeros((numba_max_threads, max(1, kernel_nnz)), dtype=np.float64)
    # Per-thread scratch vector for outbound destination split.
    distributed_by_thread = np.zeros((numba_max_threads, max(1, kernel_nnz)), dtype=np.float64)

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
                if virgin_count < 0.0 and abs(virgin_count) < 1e-10:
                    # Clamp tiny numerical drift.
                    virgin_count = 0.0

                # Migrate virgin female scalar bucket.
                migrate_scalar_bucket(
                    value=virgin_count,
                    row_dst_idx=row_dst_idx,
                    row_dst_prob=row_dst_prob,
                    row_dst_count=src_nnz,
                    rate=rate,
                    is_stochastic=is_stochastic,
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
                    migrate_sperm_bucket(
                        value=sperm_store_all[src, age, female_genotype, male_genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=is_stochastic,
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
                    migrate_scalar_bucket(
                        value=ind_count_all[src, sex, age, genotype],
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=rate,
                        is_stochastic=is_stochastic,
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
    for thread_id in range(numba_max_threads):
        out_ind += out_ind_by_thread[thread_id]
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm
