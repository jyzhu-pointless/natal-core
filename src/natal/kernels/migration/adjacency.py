"""Adjacency-mode spatial migration kernels."""

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

from natal import numba_compat as nbc
from natal.numba_utils import njit_switch

__all__ = [
    "apply_spatial_adjacency_mode",
    "migrate_scalar_bucket",
    "migrate_sperm_bucket",
]


@njit_switch(cache=True, parallel=True)
def _apply_spatial_adjacency_migration_internal(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_nnz: NDArray[np.int64],
    rate: float,
    is_stochastic: bool,
    use_continuous_sampling: bool,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Internal implementation for spatial adjacency migration.

    This function contains the shared logic for both deterministic and stochastic
    migration modes using precomputed sparse adjacency rows.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        row_dst_idx: Sparse destination indices per source row.
        row_dst_prob: Sparse destination probabilities per source row.
        row_nnz: Number of valid destinations per source row.
        rate: Migration probability.
        is_stochastic: Whether to use stochastic migration sampling.
        use_continuous_sampling: Whether to use continuous sampling.
        n_threads: Number of thread lanes reserved for thread-local buffers.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one migration step.
    """
    # Read the leading dimensions once to avoid repeated shape indexing.
    n_demes = ind_count_all.shape[0]
    n_sexes = ind_count_all.shape[1]
    n_ages = ind_count_all.shape[2]
    n_genotypes = ind_count_all.shape[3]
    max_nnz = row_dst_idx.shape[1]

    # Thread-local accumulation avoids write conflicts across ``prange`` source lanes.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)
    distributed_by_thread = np.zeros((n_threads, max_nnz), dtype=np.float64)

    # Parallelize by source deme so each lane processes one source row at a time.
    for src in prange(n_demes):
        thread_id = get_thread_id()
        out_ind = out_ind_by_thread[thread_id]
        out_sperm = out_sperm_by_thread[thread_id]
        distributed = distributed_by_thread[thread_id]

        src_nnz = int(row_nnz[src])

        # Handle female virgin + sperm-coupled buckets.
        for age in range(n_ages):
            for female_genotype in range(n_genotypes):
                # Recompute stored sperm total so virgin females can be separated.
                stored_total = 0.0
                for male_genotype in range(n_genotypes):
                    stored_total += sperm_store_all[src, age, female_genotype, male_genotype]

                female_total = ind_count_all[src, 0, age, female_genotype]
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    virgin_count = 0.0

                if is_stochastic:
                    _migrate_scalar_bucket(
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
                else:
                    # Deterministic calculation
                    if src_nnz > 0:
                        outbound = virgin_count * rate
                        stay = virgin_count - outbound
                        out_ind[src, 0, age, female_genotype] += stay
                        for nnz_idx in range(src_nnz):
                            dst = int(row_dst_idx[src, nnz_idx])
                            prob = row_dst_prob[src, nnz_idx]
                            out_ind[dst, 0, age, female_genotype] += outbound * prob
                    else:
                        out_ind[src, 0, age, female_genotype] += virgin_count

                for male_genotype in range(n_genotypes):
                    sperm_value = sperm_store_all[src, age, female_genotype, male_genotype]

                    if is_stochastic:
                        _migrate_sperm_bucket(
                            value=sperm_value,
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
                    else:
                        # Deterministic calculation
                        if src_nnz > 0:
                            outbound_sperm = sperm_value * rate
                            stay_sperm = sperm_value - outbound_sperm
                            out_sperm[src, age, female_genotype, male_genotype] += stay_sperm
                            out_ind[src, 0, age, female_genotype] += stay_sperm
                            for nnz_idx in range(src_nnz):
                                dst = int(row_dst_idx[src, nnz_idx])
                                prob = row_dst_prob[src, nnz_idx]
                                moved_sperm = outbound_sperm * prob
                                out_sperm[dst, age, female_genotype, male_genotype] += moved_sperm
                                out_ind[dst, 0, age, female_genotype] += moved_sperm
                        else:
                            out_sperm[src, age, female_genotype, male_genotype] += sperm_value
                            out_ind[src, 0, age, female_genotype] += sperm_value

        # Handle remaining individual buckets (male and other sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for genotype in range(n_genotypes):
                    value = ind_count_all[src, sex, age, genotype]

                    if is_stochastic:
                        _migrate_scalar_bucket(
                            value=value,
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
                    else:
                        # Deterministic calculation
                        if src_nnz > 0:
                            outbound = value * rate
                            stay = value - outbound
                            out_ind[src, sex, age, genotype] += stay
                            for nnz_idx in range(src_nnz):
                                dst = int(row_dst_idx[src, nnz_idx])
                                prob = row_dst_prob[src, nnz_idx]
                                out_ind[dst, sex, age, genotype] += outbound * prob
                        else:
                            out_ind[src, sex, age, genotype] += value

    # Merge thread-local partial sums to final state.
    out_ind = np.zeros_like(ind_count_all)
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        out_ind += out_ind_by_thread[thread_id]
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm


@njit_switch(cache=True)
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
    """Apply one deterministic migration step using sparse per-source rows."""
    return _apply_spatial_adjacency_migration_internal(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        row_dst_idx=row_dst_idx,
        row_dst_prob=row_dst_prob,
        row_nnz=row_nnz,
        rate=rate,
        is_stochastic=False,
        use_continuous_sampling=False,
        n_threads=n_threads,
    )


@njit_switch(cache=True)
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
    """Apply one stochastic migration step using sparse routing rows."""
    return _apply_spatial_adjacency_migration_internal(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        row_dst_idx=row_dst_idx,
        row_dst_prob=row_dst_prob,
        row_nnz=row_nnz,
        rate=rate,
        is_stochastic=True,
        use_continuous_sampling=use_continuous_sampling,
        n_threads=n_threads,
    )

@njit_switch(cache=True)
def apply_spatial_adjacency_mode(
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
    """Apply one synchronized migration step in adjacency backend mode."""
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
        return float(nbc.continuous_binomial(float(value), float(rate)))
    # Discrete mode treats each scalar bucket as a Bernoulli family and keeps
    # the migrated amount integer-valued.
    return float(nbc.binomial(int(round(float(value))), float(rate)))


@njit_switch(cache=True)
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
        nbc.continuous_multinomial(float(outbound), probs, distributed)
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


# Public aliases for kernel backend reuse.
migrate_scalar_bucket = _migrate_scalar_bucket
migrate_sperm_bucket = _migrate_sperm_bucket
