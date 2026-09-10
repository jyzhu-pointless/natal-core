"""Build-time migration normalization and CSR folding.

Everything spatial about migration — strategy resolution, adjacency
coercion, kernel-bank selection, kernel-center handling, and boundary
normalization — is resolved here once at build time and folded into the
three CSR arrays stored on the frozen
:class:`~natal.contracts.blueprint.Blueprint`.  The ``strategy`` string
is a *materialization* choice only: once the CSR exists, the runtime
migration stage is a pure numeric loop (rate column x fixed CSR) and
neither the topology nor the kernel is consulted again.  Changing the
topology or the kernel therefore means rebuilding the model, exactly
like any other frozen Blueprint field.

Bitwise-parity contract: the fold reproduces, entry for entry, the
arithmetic order of the legacy runtime row builders
(``_build_sparse_migration_rows`` for adjacency mode,
``_build_source_kernel_sparse_row`` for kernel mode), so deterministic
trajectories are identical before and after the refactor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.spatial.topology import GridTopology

__all__ = [
    "MigrationCSR",
    "csr_dense_row",
    "fold_migration_csr",
    "normalize_migration_rate",
    "resolve_migration_mode",
]

MigrationStrategy = Literal["auto", "adjacency", "kernel", "hybrid"]

# The user-facing migration-rate declaration shared by the builder sugar
# and the runtime writer.
RateDeclaration: TypeAlias = float | int | NDArray[np.floating] | Sequence[float] | dict[
    str, float | Sequence[float] | NDArray[np.floating]
]


class MigrationCSR(NamedTuple):
    """Frozen outbound migration routing in CSR form.

    Attributes:
        indptr: ``(n_demes + 1,)`` int64 row pointer;
            ``indptr[d]:indptr[d + 1]`` slices the outbound entries of
            deme *d*.
        dest_idx: ``(nnz,)`` int64 destination deme per entry.
        weights: ``(nnz,)`` float64 outbound weight per entry.
            Adjacency-mode rows carry the raw adjacency values;
            kernel-mode rows carry the historically composed final
            per-entry probability (scaled then row-normalized with the
            same float-operation order as the legacy pipeline).
        stay_after_send: Deterministic bookkeeping order.  ``False``
            (adjacency mode) keeps the historical "stay = value -
            outbound first, then distribute" order; ``True`` (kernel
            mode) keeps "distribute first, then residual = value -
            moved_total at source".  Both reproduce the legacy
            arithmetic bitwise.
    """

    indptr: NDArray[np.int64]
    dest_idx: NDArray[np.int64]
    weights: NDArray[np.float64]
    stay_after_send: bool


def normalize_migration_rate(
    rate: float | int | NDArray[np.floating] | Sequence[float] | dict[
        str, float | Sequence[float] | NDArray[np.floating]
    ],
    n_sexes: int,
    n_ages: int,
    adult_start_age: int,
) -> NDArray[np.float64]:
    """Normalize a migration-rate declaration to a ``(S, A)`` column.

    Sugar rules (mirroring the historical ``_normalize_migration_rate``):

    - **scalar**: applies to the adult ages (``>= adult_start_age``) of
      both sexes; juvenile ages default to ``0``.  Discrete models
      (``n_ages == 1``) give the single age the full rate regardless of
      ``adult_start_age``.
    - **per-sex mapping** ``{"F": 0.2, "M": 0.05}``: each value follows
      the scalar / ``(n_ages,)`` rules for that sex; a missing sex
      defaults to all-zero.
    - **(n_ages,) array**: used as-is for both sexes.
    - **(n_sexes, n_ages) array**: used as-is.

    Args:
        rate: User-declared migration rate (scalar, per-age vector,
            ``(S, A)`` array, or per-sex mapping).
        n_sexes: Number of sexes of the rate axis.
        n_ages: Number of age classes of the rate axis.
        adult_start_age: First adult age class for the scalar sugar.

    Returns:
        A fresh ``(n_sexes, n_ages)`` float64 array.

    Raises:
        ValueError: If an array shape does not match, or a mapping key
            is not a recognized sex label.
    """
    per_sex = _per_sex_rates(rate, n_ages)
    if per_sex is not None:
        result = np.zeros((n_sexes, n_ages), dtype=np.float64)
        sex_ids = {"f": 0, "female": 0, "m": 1, "male": 1}
        for key, value in per_sex.items():
            sex_id = sex_ids.get(str(key).lower())
            if sex_id is None or sex_id >= n_sexes:
                raise ValueError(
                    f"migration_rate mapping key {key!r} is not a valid sex "
                    f"label (use 'F'/'M' or 'female'/'male')"
                )
            result[sex_id, :] = _age_vector(value, n_ages, adult_start_age)
        return result

    arr = np.asarray(rate, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (n_sexes, n_ages):
            raise ValueError(
                f"migration_rate shape {arr.shape} does not match "
                f"(n_sexes={n_sexes}, n_ages={n_ages})"
            )
        return arr.astype(np.float64, copy=True)
    age_row = _age_vector(arr, n_ages, adult_start_age)
    return np.tile(age_row, (n_sexes, 1))


def _per_sex_rates(
    rate: object,
    n_ages: int,
) -> dict[str, object] | None:
    """Return the per-sex mapping when *rate* is a mapping declaration.

    Args:
        rate: Raw rate declaration.
        n_ages: Expected age count (unused for detection; kept for
            signature symmetry with future per-sex array validation).

    Returns:
        The mapping, or ``None`` when *rate* is not a mapping.
    """
    del n_ages
    if isinstance(rate, dict):
        entries: dict[str, object] = {}
        for raw_key, raw_value in cast("dict[object, object]", rate).items():
            entries[str(raw_key)] = raw_value
        return entries
    return None


def _age_vector(
    value: object,
    n_ages: int,
    adult_start_age: int,
) -> NDArray[np.float64]:
    """Normalize one sex's rate to an ``(n_ages,)`` vector.

    Args:
        value: Scalar or ``(n_ages,)`` declaration for one sex.
        n_ages: Number of age classes.
        adult_start_age: First adult age class (scalar sugar).

    Returns:
        A fresh ``(n_ages,)`` float64 vector.

    Raises:
        ValueError: If an explicit vector length mismatches ``n_ages``.
        TypeError: If the value is neither numeric nor a 1-D sequence.
    """
    if isinstance(value, bool) or isinstance(value, int | float):
        scalar = float(value)  # type: ignore[arg-type]  # bool/int/float narrowed above
        if n_ages > 1 and adult_start_age > 0:
            result = np.full(n_ages, scalar, dtype=np.float64)
            result[:adult_start_age] = 0.0
            return result
        return np.full(n_ages, scalar, dtype=np.float64)
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.ndim == 1 and arr.shape[0] == n_ages:
        return arr.astype(np.float64, copy=True)
    if arr.ndim == 1 and arr.shape[0] == 1:
        # Length-1 sequence behaves like the scalar sugar.
        return _age_vector(float(arr[0]), n_ages, adult_start_age)
    raise ValueError(
        f"migration_rate shape {arr.shape} does not match n_ages={n_ages}"
    )


def resolve_migration_mode(
    strategy: MigrationStrategy,
    migration_kernel: NDArray[np.float64] | None,
    kernel_bank: Sequence[NDArray[np.float64]] | None,
    deme_kernel_ids: NDArray[np.int64] | None,
) -> Literal["adjacency", "kernel"]:
    """Resolve the strategy-level policy into one concrete backend mode.

    The strategy string is a build-time materialization choice only; the
    returned mode is consumed by the CSR fold and then disappears.

    Args:
        strategy: ``"auto"`` / ``"adjacency"`` / ``"kernel"`` /
            ``"hybrid"`` (``"hybrid"`` remains a forward-compatible
            alias of ``"auto"``).
        migration_kernel: Single shared kernel, when provided.
        kernel_bank: Heterogeneous kernel bank, when provided.
        deme_kernel_ids: Per-deme kernel ids, when provided.

    Returns:
        ``"adjacency"`` or ``"kernel"``.

    Raises:
        ValueError: If the strategy is unknown or kernel mode was
            requested without any kernel.
    """
    has_heterogeneous = kernel_bank is not None and deme_kernel_ids is not None
    if strategy == "adjacency":
        return "adjacency"
    if strategy == "kernel":
        if migration_kernel is None and not has_heterogeneous:
            raise ValueError(
                "migration_kernel is required in kernel mode unless "
                "kernel_bank and deme_kernel_ids are both provided"
            )
        return "kernel"
    if strategy in ("auto", "hybrid"):
        if migration_kernel is not None or has_heterogeneous:
            return "kernel"
        return "adjacency"
    raise ValueError(
        "migration_strategy must be one of: auto, adjacency, kernel, hybrid"
    )


def fold_migration_csr(
    n_demes: int,
    topology: GridTopology | None,
    adjacency_dense: NDArray[np.float64],
    migration_kernel: NDArray[np.float64] | None,
    kernel_bank: Sequence[NDArray[np.float64]] | None,
    deme_kernel_ids: NDArray[np.int64] | None,
    kernel_include_center: bool,
    adjust_on_edge: bool,
    mode: Literal["adjacency", "kernel"],
) -> MigrationCSR:
    """Fold the migration configuration into CSR form (build time).

    Adjacency mode stores each source row in destination-ascending
    order with the raw adjacency values — the same entries, in the same
    order, the runtime sparse-row builder compacted on every call before
    the CSR fold.

    Kernel mode reproduces the per-source kernel row builder exactly:
    offsets are visited in kernel row-major order, invalid (out-of-grid)
    offsets are dropped (or wrapped), and each emitted entry is scaled
    by the reciprocal of the kernel total — or of the valid-row total
    when ``adjust_on_edge`` is set.  Entries are kept unmerged in visit
    order because a wrapping kernel narrower than the grid can emit the
    same destination twice, and deterministic migration must add the two
    contributions as separate multiplications to stay bit-identical.

    Args:
        n_demes: Number of demes.
        topology: Grid topology (required in kernel mode).
        adjacency_dense: Dense ``(n_demes, n_demes)`` adjacency matrix.
        migration_kernel: Single shared kernel (kernel mode, no bank).
        kernel_bank: Heterogeneous kernel bank, when used.
        deme_kernel_ids: Per-deme kernel ids into the bank, when used.
        kernel_include_center: Whether the kernel center is an outbound
            target of its own source deme.
        adjust_on_edge: Whether boundary demes renormalize to the full
            migration rate (row-total scaling) instead of keeping mass
            at the source (kernel-total scaling).
        mode: Resolved backend mode from :func:`resolve_migration_mode`.

    Returns:
        The folded :class:`MigrationCSR`.

    Raises:
        ValueError: If kernel mode is requested without a usable kernel.
    """
    indptr = np.zeros(n_demes + 1, dtype=np.int64)
    dest_parts: list[NDArray[np.int64]] = []
    weight_parts: list[NDArray[np.float64]] = []

    if mode == "adjacency":
        for src in range(n_demes):
            row = adjacency_dense[src]
            keep = np.nonzero(row > 0.0)[0]
            dest_parts.append(keep.astype(np.int64))
            weight_parts.append(row[keep].astype(np.float64))
            indptr[src + 1] = indptr[src] + keep.shape[0]
        if not dest_parts:
            dest_parts.append(np.zeros(0, dtype=np.int64))
            weight_parts.append(np.zeros(0, dtype=np.float64))
        return MigrationCSR(
            indptr=indptr,
            dest_idx=np.concatenate(dest_parts),
            weights=np.concatenate(weight_parts),
            stay_after_send=False,
        )

    if topology is None:
        # Historical behavior: kernel-bank routing without a topology
        # resolved per-deme offset tables against zero topology rows,
        # which emitted no outbound entries.
        indptr_out = np.zeros(n_demes + 1, dtype=np.int64)
        return MigrationCSR(
            indptr=indptr_out,
            dest_idx=np.zeros(0, dtype=np.int64),
            weights=np.zeros(0, dtype=np.float64),
            stay_after_send=True,
        )
    if kernel_bank is not None and deme_kernel_ids is not None:
        kernels = [np.asarray(k, dtype=np.float64) for k in kernel_bank]
        ids = [int(i) for i in deme_kernel_ids]
    elif migration_kernel is not None:
        shared = np.asarray(migration_kernel, dtype=np.float64)
        kernels = [shared]
        ids = [0] * n_demes
    else:
        raise ValueError(
            "kernel-mode migration requires migration_kernel or a kernel bank"
        )

    for src in range(n_demes):
        entries = _kernel_row_entries(
            kernel=kernels[ids[src]],
            source_idx=src,
            topology=topology,
            include_center=kernel_include_center,
            adjust_on_edge=adjust_on_edge,
        )
        dest_parts.append(np.array(entries[0], dtype=np.int64))
        weight_parts.append(np.array(entries[1], dtype=np.float64))
        indptr[src + 1] = indptr[src] + len(entries[0])

    dest_all = np.concatenate(dest_parts) if dest_parts else np.zeros(0, dtype=np.int64)
    weight_all = (
        np.concatenate(weight_parts) if weight_parts else np.zeros(0, dtype=np.float64)
    )
    return MigrationCSR(
        indptr=indptr,
        dest_idx=dest_all,
        weights=weight_all,
        stay_after_send=True,
    )


def _kernel_row_entries(
    kernel: NDArray[np.float64],
    source_idx: int,
    topology: GridTopology,
    include_center: bool,
    adjust_on_edge: bool,
) -> tuple[list[int], list[float]]:
    """Emit one source deme's kernel-mode CSR entries in visit order.

    Reproduces the legacy pipeline bitwise by composing the same
    float operations in the same order: the compact offset table sums
    positive weights in kernel row-major order; each emitted entry is
    scaled by a reciprocal multiply (kernel total, or the valid-row
    total when ``adjust_on_edge`` is set); and the runtime distributor
    finally divided every entry by the emitted-row sum.  The fold
    performs that final division here, so the runtime's plain
    ``outbound * weight`` multiply reproduces the historical result
    exactly.

    Args:
        kernel: Odd-sized 2-D kernel centered on the source cell.
        source_idx: Flattened source deme index.
        topology: Grid topology providing boundary handling.
        include_center: Whether the kernel center is emitted.
        adjust_on_edge: Row-total scaling instead of kernel-total.

    Returns:
        ``(destinations, weights)`` parallel lists.
    """
    kernel_rows = int(kernel.shape[0])
    kernel_cols = int(kernel.shape[1])
    center_row = kernel_rows // 2
    center_col = kernel_cols // 2

    kernel_total = 0.0
    offsets: list[tuple[int, int, float]] = []
    for kernel_row in range(kernel_rows):
        for kernel_col in range(kernel_cols):
            if not include_center and kernel_row == center_row and kernel_col == center_col:
                continue
            weight = float(kernel[kernel_row, kernel_col])
            if weight <= 0.0:
                continue
            offsets.append((kernel_row - center_row, kernel_col - center_col, weight))
            kernel_total += weight

    destinations: list[int] = []
    weights: list[float] = []
    if not offsets or kernel_total <= 0.0:
        return destinations, weights

    src_row, src_col = topology.from_index(source_idx)
    total = 0.0
    for d_row, d_col, weight in offsets:
        mapped = topology.normalize_coord(src_row + d_row, src_col + d_col)
        if mapped is None:
            continue
        destinations.append(topology.to_index(mapped))
        weights.append(weight)
        total += weight

    if total <= 0.0:
        return [], []

    if adjust_on_edge:
        inv = 1.0 / total
    else:
        inv = 1.0 / kernel_total
    weights = [w * inv for w in weights]
    # The historical runtime distributor renormalized the (already
    # scaled) row by its emitted sum, dividing each entry.  Fold that
    # division in so runtime multiplies stay bitwise identical.
    row_sum = 0.0
    for w in weights:
        row_sum += w
    weights = [w / row_sum for w in weights]
    return destinations, weights


def csr_dense_row(
    migration_csr: MigrationCSR,
    source_idx: int,
    n_demes: int,
) -> NDArray[np.float64]:
    """Scatter one source deme's CSR row into a dense weight vector.

    Readout helper for :meth:`SpatialPopulation.migration_row`; the
    caller decides whether to renormalize the result.

    Args:
        migration_csr: The folded CSR.
        source_idx: Source deme index.
        n_demes: Number of demes (dense output width).

    Returns:
        A fresh ``(n_demes,)`` float64 vector of outbound weights.
    """
    row = np.zeros(n_demes, dtype=np.float64)
    start = int(migration_csr.indptr[source_idx])
    end = int(migration_csr.indptr[source_idx + 1])
    for pos in range(start, end):
        row[int(migration_csr.dest_idx[pos])] += float(migration_csr.weights[pos])
    return row
