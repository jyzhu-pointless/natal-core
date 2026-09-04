"""CSR-mode spatial migration engine (slice 5).

Runtime migration is a pure numeric loop: a per-deme/per-sex/per-age
rate column multiplied by the frozen CSR routing table folded at build
time (:mod:`natal.frontend.spatial.migration`).  There is no topology,
kernel, or adjacency interpretation left at runtime — changing any of
those rebuilds the model (Blueprint discipline).

The bucket helpers keep the pre-slice-5 arithmetic order entry for
entry, so deterministic trajectories are bit-identical.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import natal.backends.reference.sampling as sampling

prange = range


def get_thread_id() -> int:
    """Return a dummy thread ID — the reference backend is single-threaded."""
    return 0


MAX_THREADS = 1

__all__ = [
    "apply_csr_migration",
    "migrate_scalar_bucket",
    "migrate_sperm_bucket",
]
def _apply_csr_migration_internal(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    stochastic: bool,
    continuous_sampling: bool,
    stay_after_send: bool,
    n_threads: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared CSR migration body for the deterministic and stochastic paths.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        indptr: CSR row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR outbound weight per entry (already normalized at
            build time).
        rate: ``(n_demes, n_sexes, n_ages)`` migration-rate column.
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
            approximations.
        stay_after_send: Deterministic bookkeeping order — ``False``
            keeps the adjacency-mode "stay first" order, ``True`` the
            kernel-mode "distribute first, residual last" order.
        n_threads: Number of thread lanes reserved for thread-local
            buffers.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one migration step.
    """
    # Read the leading dimensions once to avoid repeated shape indexing.
    n_demes = ind_count_all.shape[0]
    n_sexes = ind_count_all.shape[1]
    n_ages = ind_count_all.shape[2]
    n_ztypes = ind_count_all.shape[3]
    max_row = 1
    for src in range(n_demes):
        row_len = int(indptr[src + 1] - indptr[src])
        if row_len > max_row:
            max_row = row_len

    # Thread-local accumulation avoids write conflicts across ``prange`` source lanes.
    out_ind_by_thread = np.zeros((n_threads,) + ind_count_all.shape, dtype=np.float64)
    out_sperm_by_thread = np.zeros((n_threads,) + sperm_store_all.shape, dtype=np.float64)
    distributed_by_thread = np.zeros((n_threads, max_row), dtype=np.float64)

    # Parallelize by source deme so each lane processes one source row at a time.
    for src in prange(n_demes):
        thread_id = get_thread_id()
        out_ind = out_ind_by_thread[thread_id]
        out_sperm = out_sperm_by_thread[thread_id]
        distributed = distributed_by_thread[thread_id]

        row_start = int(indptr[src])
        row_end = int(indptr[src + 1])
        src_nnz = row_end - row_start
        row_dst_idx = dest_idx[row_start:row_end]
        row_dst_prob = weights[row_start:row_end]

        # Handle female virgin + sperm-coupled buckets.
        for age in range(n_ages):
            for female_ztype in range(n_ztypes):
                # Recompute stored sperm total so virgin females can be separated.
                stored_total = 0.0
                for male_ztype in range(n_ztypes):
                    stored_total += sperm_store_all[src, age, female_ztype, male_ztype]

                female_total = ind_count_all[src, 0, age, female_ztype]
                virgin_count = female_total - stored_total
                if virgin_count < 0.0 and abs(virgin_count) < 1e-9:
                    virgin_count = 0.0

                female_rate = rate[src, 0, age]
                if stochastic:
                    _migrate_scalar_bucket(
                        value=virgin_count,
                        row_dst_idx=row_dst_idx,
                        row_dst_prob=row_dst_prob,
                        row_dst_count=src_nnz,
                        rate=female_rate,
                        stochastic=True,
                        continuous_sampling=continuous_sampling,
                        distributed=distributed,
                        out_ind=out_ind,
                        source_idx=src,
                        sex_idx=0,
                        age_idx=age,
                        genotype_idx=female_ztype,
                    )
                else:
                    # Deterministic calculation
                    if src_nnz > 0:
                        outbound = virgin_count * female_rate
                        if stay_after_send:
                            moved_total = 0.0
                            for nnz_idx in range(src_nnz):
                                dst = int(row_dst_idx[nnz_idx])
                                moved = outbound * row_dst_prob[nnz_idx]
                                out_ind[dst, 0, age, female_ztype] += moved
                                moved_total += moved
                            out_ind[src, 0, age, female_ztype] += (
                                virgin_count - moved_total
                            )
                        else:
                            stay = virgin_count - outbound
                            out_ind[src, 0, age, female_ztype] += stay
                            for nnz_idx in range(src_nnz):
                                dst = int(row_dst_idx[nnz_idx])
                                prob = row_dst_prob[nnz_idx]
                                out_ind[dst, 0, age, female_ztype] += outbound * prob
                    else:
                        out_ind[src, 0, age, female_ztype] += virgin_count

                # Stored sperm travels with its mated female: it must use
                # the female rate, otherwise the virgin/stored bookkeeping
                # (female_total >= stored_total) breaks after migration.
                male_rate = female_rate
                for male_ztype in range(n_ztypes):
                    sperm_value = sperm_store_all[src, age, female_ztype, male_ztype]

                    if stochastic:
                        _migrate_sperm_bucket(
                            value=sperm_value,
                            row_dst_idx=row_dst_idx,
                            row_dst_prob=row_dst_prob,
                            row_dst_count=src_nnz,
                            rate=male_rate,
                            stochastic=True,
                            continuous_sampling=continuous_sampling,
                            distributed=distributed,
                            out_ind=out_ind,
                            out_sperm=out_sperm,
                            source_idx=src,
                            age_idx=age,
                            female_genotype_idx=female_ztype,
                            male_genotype_idx=male_ztype,
                        )
                    else:
                        # Deterministic calculation
                        if src_nnz > 0:
                            outbound_sperm = sperm_value * male_rate
                            if stay_after_send:
                                moved_total = 0.0
                                for nnz_idx in range(src_nnz):
                                    dst = int(row_dst_idx[nnz_idx])
                                    moved_sperm = outbound_sperm * row_dst_prob[nnz_idx]
                                    out_sperm[dst, age, female_ztype, male_ztype] += moved_sperm
                                    out_ind[dst, 0, age, female_ztype] += moved_sperm
                                    moved_total += moved_sperm
                                out_sperm[src, age, female_ztype, male_ztype] += (
                                    sperm_value - moved_total
                                )
                                out_ind[src, 0, age, female_ztype] += (
                                    sperm_value - moved_total
                                )
                            else:
                                stay_sperm = sperm_value - outbound_sperm
                                out_sperm[src, age, female_ztype, male_ztype] += stay_sperm
                                out_ind[src, 0, age, female_ztype] += stay_sperm
                                for nnz_idx in range(src_nnz):
                                    dst = int(row_dst_idx[nnz_idx])
                                    prob = row_dst_prob[nnz_idx]
                                    moved_sperm = outbound_sperm * prob
                                    out_sperm[dst, age, female_ztype, male_ztype] += moved_sperm
                                    out_ind[dst, 0, age, female_ztype] += moved_sperm
                        else:
                            out_sperm[src, age, female_ztype, male_ztype] += sperm_value
                            out_ind[src, 0, age, female_ztype] += sperm_value

        # Handle remaining individual buckets (male and other sexes).
        for sex in range(1, n_sexes):
            for age in range(n_ages):
                for ztype in range(n_ztypes):
                    value = ind_count_all[src, sex, age, ztype]
                    bucket_rate = rate[src, sex, age]

                    if stochastic:
                        _migrate_scalar_bucket(
                            value=value,
                            row_dst_idx=row_dst_idx,
                            row_dst_prob=row_dst_prob,
                            row_dst_count=src_nnz,
                            rate=bucket_rate,
                            stochastic=True,
                            continuous_sampling=continuous_sampling,
                            distributed=distributed,
                            out_ind=out_ind,
                            source_idx=src,
                            sex_idx=sex,
                            age_idx=age,
                            genotype_idx=ztype,
                        )
                    else:
                        # Deterministic calculation
                        if src_nnz > 0:
                            outbound = value * bucket_rate
                            if stay_after_send:
                                moved_total = 0.0
                                for nnz_idx in range(src_nnz):
                                    dst = int(row_dst_idx[nnz_idx])
                                    moved = outbound * row_dst_prob[nnz_idx]
                                    out_ind[dst, sex, age, ztype] += moved
                                    moved_total += moved
                                out_ind[src, sex, age, ztype] += value - moved_total
                            else:
                                stay = value - outbound
                                out_ind[src, sex, age, ztype] += stay
                                for nnz_idx in range(src_nnz):
                                    dst = int(row_dst_idx[nnz_idx])
                                    prob = row_dst_prob[nnz_idx]
                                    out_ind[dst, sex, age, ztype] += outbound * prob
                        else:
                            out_ind[src, sex, age, ztype] += value

    # Merge thread-local partial sums to final state.
    out_ind = np.zeros_like(ind_count_all)
    out_sperm = np.zeros_like(sperm_store_all)
    for thread_id in range(n_threads):
        out_ind += out_ind_by_thread[thread_id]
        out_sperm += out_sperm_by_thread[thread_id]

    return out_ind, out_sperm
def apply_csr_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    stochastic: bool,
    continuous_sampling: bool,
    stay_after_send: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one synchronized migration step over the CSR routing table.

    Args:
        ind_count_all: Stacked individual-count tensor.
        sperm_store_all: Stacked sperm-storage tensor.
        indptr: CSR row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        rate: ``(n_demes, n_sexes, n_ages)`` migration-rate column.
        stochastic: Whether to use stochastic migration sampling.
        continuous_sampling: Whether to use continuous sampling.

    Returns:
        A tuple ``(ind_next, sperm_next)`` after one migration step.
    """
    if np.all(rate <= 0.0):
        return ind_count_all, sperm_store_all
    return _apply_csr_migration_internal(
        ind_count_all=ind_count_all,
        sperm_store_all=sperm_store_all,
        indptr=indptr,
        dest_idx=dest_idx,
        weights=weights,
        rate=rate,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
        stay_after_send=stay_after_send,
        n_threads=MAX_THREADS,
    )
def _sample_outbound_count(
    value: float,
    rate: float,
    stochastic: bool,
    continuous_sampling: bool,
) -> float:
    """Compute the outbound amount for one scalar bucket.

    Args:
        value: Source bucket mass before migration.
        rate: Migration probability applied to this bucket.
        stochastic: Whether to sample rather than use the expectation.
        continuous_sampling: Whether stochastic mode should use the
            continuous Beta approximation.

    Returns:
        The amount of mass that leaves the source bucket.
    """
    if value <= 0.0 or rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return float(value)
    if not stochastic:
        # Deterministic migration moves the expectation directly.
        return float(value) * rate
    if continuous_sampling:
        # Continuous mode keeps the state real-valued while still injecting
        # stochasticity into the outbound amount.
        return float(sampling.continuous_binomial(float(value), float(rate)))
    # Discrete mode treats each scalar bucket as a Bernoulli family and keeps
    # the migrated amount integer-valued.
    return float(sampling.binomial(int(round(float(value))), float(rate)))
def _distribute_outbound_count(
    outbound: float,
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    stochastic: bool,
    continuous_sampling: bool,
    distributed: NDArray[np.float64],
) -> None:
    """Distribute one outbound amount across sparse destinations.

    Args:
        outbound: Total mass already selected to leave the source bucket.
        row_dst_prob: Destination weights for one source deme.
        row_dst_count: Number of valid destination entries in ``row_dst_prob``.
        stochastic: Whether to sample rather than use expectations.
        continuous_sampling: Whether stochastic mode should use the
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
    if not stochastic:
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
    if continuous_sampling:
        # Continuous multinomial keeps real-valued buckets while conserving
        # the sampled outbound total.
        sampling.continuous_multinomial(float(outbound), probs, distributed)
        return

    # Discrete multinomial allocates an integer outbound count to
    # destination demes while preserving the total exactly.
    # Discrete stochastic split preserving integer outbound total.
    sampled = sampling.multinomial(int(round(float(outbound))), probs)
    for idx in range(row_dst_count):
        distributed[idx] = float(sampled[idx])
def _migrate_scalar_bucket(
    value: float,
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    rate: float,
    stochastic: bool,
    continuous_sampling: bool,
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
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
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
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
    )
    # Split outbound mass across sparse destinations.
    _distribute_outbound_count(
        outbound=outbound,
        row_dst_prob=row_dst_prob,
        row_dst_count=row_dst_count,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
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
def _migrate_sperm_bucket(
    value: float,
    row_dst_idx: NDArray[np.int64],
    row_dst_prob: NDArray[np.float64],
    row_dst_count: int,
    rate: float,
    stochastic: bool,
    continuous_sampling: bool,
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
        stochastic: Whether outbound mass is sampled.
        continuous_sampling: Whether stochastic mode uses continuous
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
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
    )
    # Split outbound sperm mass across sparse destinations.
    _distribute_outbound_count(
        outbound=outbound,
        row_dst_prob=row_dst_prob,
        row_dst_count=row_dst_count,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
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
