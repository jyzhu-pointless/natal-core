"""Slice-5 spatial adversarial tests: the migration CSR data plane under attack.

Every assertion here proves a numerical invariant against an independently
written reference, not against the implementation's own readout:

- **CSR fold equivalence** — the deterministic engines reproduce a manual
  application of the folded routing entries bit-for-bit, for both
  bookkeeping orders (``stay_after_send`` False = stay-first adjacency
  order, True = send-first kernel order), including the wrapping narrow
  grid where one CSR row emits the same destination twice.
- **Sugar rules** — ``normalize_migration_rate`` produces the exact
  documented column for every declaration shape, and rejects malformed
  ones.
- **Contract sentinels** — panmictic defaults, the 1-deme self-loop, the
  ``(0, 0, 0)`` not-declared rate sentinel, frozen Blueprint fields, and
  materialization copy isolation.
- **Two-backend consistency** — Python and Rust produce
  bit-identical deterministic migration from the same CSR + rate column
  (single-lane accumulation keeps the order sequential, so
  bitwise comparison is meaningful).
- **Per-deme/per-sex consumption** — an asymmetric ``tensor_write`` rate
  column is consumed at exactly the written entries by population-level
  runs on all three execution paths.
- **Boundary semantics unification** — a boundary deme of a kernel-mode
  grid emits its full ``rate * value`` outbound over the shared CSR
  (reference semantics), not the abandoned pre-slice-5 Rust
  kernel-total-scaled behavior.
- **Negative contracts** — the deleted migration surface is
  unconstructible, not merely unused.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray


from contextlib import contextmanager


@contextmanager
def python_reference():
    """Portable stand-in for the retired compiled-backend disable guard.

    The only non-Rust execution vehicle is the pure-Python reference;
    this context manager is a semantic no-op kept so test bodies that
    previously forced the Python path stay readable.
    """
    yield
from natal.backends.reference.migration.adjacency import (
    _apply_csr_migration_internal,
    apply_csr_migration,
)
from natal.contracts.materialize import SpatialMigration, materialize
from natal.frontend.genetics import Species
from natal.frontend.spatial import (
    SpatialPopulation,
    SquareGrid,
    batch_setting,
    build_adjacency_matrix,
)
from natal.frontend.spatial.migration import (
    MigrationCSR,
    RateDeclaration,
    csr_dense_row,
    fold_migration_csr,
    normalize_migration_rate,
    resolve_migration_mode,
)

def _rust_available() -> bool:
    try:
        from natal.backends.rust.rust_backend import rust_backend_available

        return rust_backend_available()
    except ImportError:
        return False


_RUST_OK = _rust_available()

#: The pure-Python migration body (single sequential lane).
_python_migration_body = _apply_csr_migration_internal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _species(prefix: str) -> Species:
    return Species.from_dict(prefix, {"chr1": {"loc": ["WT", "Dr"]}})


def _fold_adjacency(adjacency: NDArray[np.float64]) -> MigrationCSR:
    return fold_migration_csr(
        n_demes=adjacency.shape[0],
        topology=None,
        adjacency_dense=adjacency,
        migration_kernel=None,
        kernel_bank=None,
        deme_kernel_ids=None,
        kernel_include_center=False,
        adjust_on_edge=False,
        mode="adjacency",
    )


def _fold_kernel(
    topology: SquareGrid,
    kernel: NDArray[np.float64],
    *,
    adjust_on_edge: bool = False,
    kernel_bank: Sequence[NDArray[np.float64]] | None = None,
    deme_kernel_ids: NDArray[np.int64] | None = None,
) -> MigrationCSR:
    return fold_migration_csr(
        n_demes=topology.n_demes,
        topology=topology,
        adjacency_dense=np.zeros((topology.n_demes, topology.n_demes)),
        migration_kernel=None if kernel_bank is not None else kernel,
        kernel_bank=kernel_bank,
        deme_kernel_ids=deme_kernel_ids,
        kernel_include_center=False,
        adjust_on_edge=adjust_on_edge,
        mode="kernel",
    )


def _csr_rows(csr: MigrationCSR) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Slice the CSR into per-source ``(destinations, weights)`` pairs."""
    n_demes = csr.indptr.shape[0] - 1
    rows: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []
    for src in range(n_demes):
        lo = int(csr.indptr[src])
        hi = int(csr.indptr[src + 1])
        rows.append((csr.dest_idx[lo:hi], csr.weights[lo:hi]))
    return rows


def _valid_state(
    n_demes: int, n_ages: int, n_ztypes: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random stacked state satisfying ``female_total >= stored_sperm``."""
    rng = np.random.default_rng(seed)
    ind = rng.random((n_demes, 2, n_ages, n_ztypes)) * 90.0 + 10.0
    sperm = rng.random((n_demes, n_ages, n_ztypes, n_ztypes)) * 3.0
    sperm[:, : max(0, n_ages - 1), :, :] = 0.0
    for deme in range(n_demes):
        for age in range(n_ages):
            for female_z in range(n_ztypes):
                if sperm[deme, age, female_z, :].sum() > ind[deme, 0, age, female_z]:
                    sperm[deme, age, female_z, :] = 0.0
    return ind, sperm


def _apply_reference(
    ind: NDArray[np.float64],
    sperm: NDArray[np.float64],
    rows: Sequence[tuple[NDArray[np.int64], NDArray[np.float64]]],
    rate: NDArray[np.float64],
    stay_after_send: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent deterministic migration in the engine's entry order.

    Reproduces the two historical bookkeeping orders bit-for-bit:
    ``stay_after_send=False`` keeps ``value - outbound`` at the source
    before distributing, ``True`` distributes first and keeps the
    ``value - moved_total`` residual last.
    """
    out_ind = np.zeros_like(ind)
    out_sperm = np.zeros_like(sperm)
    n_demes = ind.shape[0]
    n_ages = ind.shape[2]
    n_ztypes = ind.shape[3]
    for src in range(n_demes):
        dests, weights = rows[src]
        nnz = int(dests.shape[0])
        for age in range(n_ages):
            for female_z in range(n_ztypes):
                stored_total = 0.0
                for male_z in range(n_ztypes):
                    stored_total += sperm[src, age, female_z, male_z]
                female_total = ind[src, 0, age, female_z]
                virgin = female_total - stored_total
                if virgin < 0.0 and abs(virgin) < 1e-9:
                    virgin = 0.0
                female_rate = rate[src, 0, age]
                if nnz > 0:
                    outbound = virgin * female_rate
                    if stay_after_send:
                        moved_total = 0.0
                        for pos in range(nnz):
                            moved = outbound * weights[pos]
                            out_ind[int(dests[pos]), 0, age, female_z] += moved
                            moved_total += moved
                        out_ind[src, 0, age, female_z] += virgin - moved_total
                    else:
                        stay = virgin - outbound
                        out_ind[src, 0, age, female_z] += stay
                        for pos in range(nnz):
                            out_ind[int(dests[pos]), 0, age, female_z] += (
                                outbound * weights[pos]
                            )
                else:
                    out_ind[src, 0, age, female_z] += virgin
                for male_z in range(n_ztypes):
                    value = sperm[src, age, female_z, male_z]
                    if nnz > 0:
                        outbound = value * female_rate
                        if stay_after_send:
                            moved_total = 0.0
                            for pos in range(nnz):
                                moved = outbound * weights[pos]
                                out_sperm[
                                    int(dests[pos]), age, female_z, male_z
                                ] += moved
                                out_ind[int(dests[pos]), 0, age, female_z] += moved
                                moved_total += moved
                            out_sperm[src, age, female_z, male_z] += value - moved_total
                            out_ind[src, 0, age, female_z] += value - moved_total
                        else:
                            stay = value - outbound
                            out_sperm[src, age, female_z, male_z] += stay
                            out_ind[src, 0, age, female_z] += stay
                            for pos in range(nnz):
                                moved = outbound * weights[pos]
                                out_sperm[
                                    int(dests[pos]), age, female_z, male_z
                                ] += moved
                                out_ind[int(dests[pos]), 0, age, female_z] += moved
                    else:
                        out_sperm[src, age, female_z, male_z] += value
                        out_ind[src, 0, age, female_z] += value
        for sex in range(1, ind.shape[1]):
            for age in range(n_ages):
                for ztype in range(n_ztypes):
                    value = ind[src, sex, age, ztype]
                    bucket_rate = rate[src, sex, age]
                    if nnz > 0:
                        outbound = value * bucket_rate
                        if stay_after_send:
                            moved_total = 0.0
                            for pos in range(nnz):
                                moved = outbound * weights[pos]
                                out_ind[int(dests[pos]), sex, age, ztype] += moved
                                moved_total += moved
                            out_ind[src, sex, age, ztype] += value - moved_total
                        else:
                            stay = value - outbound
                            out_ind[src, sex, age, ztype] += stay
                            for pos in range(nnz):
                                out_ind[int(dests[pos]), sex, age, ztype] += (
                                    outbound * weights[pos]
                                )
                    else:
                        out_ind[src, sex, age, ztype] += value
    return out_ind, out_sperm


def _run_engine(
    ind: NDArray[np.float64],
    sperm: NDArray[np.float64],
    csr: MigrationCSR,
    rate: NDArray[np.float64],
    backend: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one deterministic migration step on the named in-process backend."""
    if backend == "python":
        return _python_migration_body(
            ind.copy(),
            sperm.copy(),
            csr.indptr,
            csr.dest_idx,
            csr.weights,
            rate,
            False,
            False,
            csr.stay_after_send,
            1,
        )
    raise ValueError(f"unknown backend {backend!r}")


def _reconstruct_kernel_rows(
    topology: SquareGrid,
    kernel: NDArray[np.float64],
    include_center: bool,
    adjust_on_edge: bool,
) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Rebuild the pre-slice-5 kernel row pipeline independently.

    Composes the float operations in the historical order: offsets in
    kernel row-major order, reciprocal-multiply by the kernel total (or
    the valid-row total when adjusting edges), then the runtime
    distributor's division by the emitted-row sum.  Duplicate
    destinations stay unmerged, exactly like the historical row builder.
    """
    kernel_rows, kernel_cols = kernel.shape
    center_row, center_col = kernel_rows // 2, kernel_cols // 2
    kernel_total = 0.0
    offsets: list[tuple[int, int, float]] = []
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            if not include_center and kr == center_row and kc == center_col:
                continue
            weight = float(kernel[kr, kc])
            if weight <= 0.0:
                continue
            offsets.append((kr - center_row, kc - center_col, weight))
            kernel_total += weight
    rows: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []
    for src in range(topology.n_demes):
        src_row, src_col = topology.from_index(src)
        dests: list[int] = []
        scaled: list[float] = []
        total = 0.0
        for d_row, d_col, weight in offsets:
            mapped = topology.normalize_coord(src_row + d_row, src_col + d_col)
            if mapped is None:
                continue
            dests.append(topology.to_index(mapped))
            scaled.append(weight)
            total += weight
        if not dests or total <= 0.0:
            rows.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)))
            continue
        inv = 1.0 / total if adjust_on_edge else 1.0 / kernel_total
        scaled = [w * inv for w in scaled]
        row_sum = 0.0
        for w in scaled:
            row_sum += w
        dest_arr = np.array(dests, dtype=np.int64)
        weight_arr = np.array([w / row_sum for w in scaled], dtype=np.float64)
        rows.append((dest_arr, weight_arr))
    return rows


# ---------------------------------------------------------------------------
# 1. CSR fold equivalence (bitwise, both bookkeeping orders)
# ---------------------------------------------------------------------------


class TestAdjacencyFoldEquivalence:
    """Adjacency mode: stay-first order, raw weights, no renormalization."""

    def test_engine_bitwise_matches_manual_dense_application(self) -> None:
        """Topology adjacency + varied rate column == manual application, bit-for-bit."""
        topology = SquareGrid(rows=2, cols=3, neighborhood="von_neumann", wrap=False)
        adjacency = build_adjacency_matrix(topology, row_normalize=True)
        csr = _fold_adjacency(adjacency)
        assert csr.stay_after_send is False

        ind, sperm = _valid_state(n_demes=6, n_ages=2, n_ztypes=2, seed=101)
        rate = np.full((6, 2, 2), 0.1)
        rate[:, 1, 0] = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        rate[:, 1, 1] = 0.3
        rate[3, :, :] = 0.0  # one fully non-migrating deme

        expected_ind, expected_sperm = _apply_reference(
            ind, sperm, _csr_rows(csr), rate, stay_after_send=False
        )
        for backend in ("python",):
            got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, backend)
            assert np.array_equal(got_ind, expected_ind), backend
            assert np.array_equal(got_sperm, expected_sperm), backend
        # Row-normalized adjacency conserves the total mass.
        assert np.isclose(
            expected_ind.sum() + expected_sperm.sum(),
            ind.sum() + sperm.sum(),
            rtol=0.0,
            atol=1e-9,
        )

    def test_raw_weights_keep_residual_at_source(self) -> None:
        """A non-row-stochastic adjacency moves only ``rate * row_sum`` of mass.

        The deterministic adjacency order computes ``stay = value -
        outbound`` from the full outbound, so a row summing to 0.3 leaves
        the undelivered share at the source instead of renormalizing.
        """
        adjacency = np.array([[0.0, 0.3], [0.0, 0.0]])
        csr = _fold_adjacency(adjacency)
        ind = np.zeros((2, 2, 2, 1))
        ind[0, 1, :, 0] = [1000.0, 2000.0]
        sperm = np.zeros((2, 2, 1, 1))
        rate = np.full((2, 2, 2), 0.5)

        got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, "python")
        # Same expression order as the engine, written out per bucket.
        for age, value in ((0, 1000.0), (1, 2000.0)):
            assert got_ind[1, 1, age, 0] == (value * 0.5) * 0.3
            assert got_ind[0, 1, age, 0] == value - value * 0.5
            assert got_ind[1, 0, age, 0] == 0.0
        assert got_sperm.sum() == 0.0

    def test_tiny_negative_virgin_drift_is_clamped(self) -> None:
        """A ``|virgin| < 1e-9`` negative drift must not migrate negative mass."""
        csr = _fold_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))
        ind = np.zeros((2, 2, 2, 1))
        ind[0, 0, 1, 0] = 500.0 - 5e-10  # female total slightly below stored
        sperm = np.zeros((2, 2, 1, 1))
        sperm[0, 1, 0, 0] = 500.0
        rate = np.full((2, 2, 2), 0.5)

        got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, "python")
        assert (got_ind >= 0.0).all()
        assert (got_sperm >= 0.0).all()
        # The clamped virgin contributes zero outbound (the -5e-10 drift is
        # discarded); the mated mass moves at the female rate, intact.
        assert got_ind[1, 0, 1, 0] == 500.0 * 0.5
        assert got_sperm[1, 1, 0, 0] == 500.0 * 0.5
        assert got_ind[0, 0, 1, 0] == 500.0 - 500.0 * 0.5
        assert got_sperm[0, 1, 0, 0] == 500.0 - 500.0 * 0.5


    def test_all_zero_rate_short_circuits_to_the_input(self) -> None:
        """An all-zero rate column returns the state unchanged, bit-for-bit."""
        csr = _fold_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))
        ind, sperm = _valid_state(n_demes=2, n_ages=2, n_ztypes=2, seed=707)
        got = apply_csr_migration(
            ind.copy(), sperm.copy(), csr.indptr, csr.dest_idx, csr.weights,
            np.zeros((2, 2, 2)), False, False, csr.stay_after_send,
        )
        assert np.array_equal(got[0], ind)
        assert np.array_equal(got[1], sperm)


class TestKernelFoldEquivalence:
    """Kernel mode: send-first order, fold-baked normalization, duplicates."""

    def test_fold_and_engine_match_independent_reconstruction(self) -> None:
        """Folded rows equal an independent old-pipeline rebuild, then the engine."""
        topology = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=False)
        kernel = np.array([[0.05, 0.1, 0.2], [0.05, 0.0, 0.1], [0.02, 0.04, 0.08]])
        csr = _fold_kernel(topology, kernel)
        assert csr.stay_after_send is True

        expected_rows = _reconstruct_kernel_rows(
            topology, kernel, include_center=False, adjust_on_edge=False
        )
        for src, (dests, weights) in enumerate(expected_rows):
            lo, hi = int(csr.indptr[src]), int(csr.indptr[src + 1])
            assert np.array_equal(csr.dest_idx[lo:hi], dests)
            assert np.array_equal(csr.weights[lo:hi], weights)

        ind, sperm = _valid_state(n_demes=6, n_ages=2, n_ztypes=2, seed=202)
        rate = np.full((6, 2, 2), 0.2)
        rate[0, 1, :] = 0.35
        expected_ind, expected_sperm = _apply_reference(
            ind, sperm, expected_rows, rate, stay_after_send=True
        )
        for backend in ("python",):
            got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, backend)
            assert np.array_equal(got_ind, expected_ind), backend
            assert np.array_equal(got_sperm, expected_sperm), backend

    def test_wrap_narrow_grid_duplicate_destinations_bitwise(self) -> None:
        """A kernel wider than the grid emits duplicate destinations; order holds.

        On a 2-column wrapped grid the von-Neumann left/right offsets of
        the same source collapse onto one destination column, so the CSR
        row carries the same destination twice.  Deterministic migration
        must add the two contributions as separate multiplies in visit
        order — merging them would break bitwise parity.
        """
        topology = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=True)
        kernel = np.array([[2.0, 1.0, 4.0], [8.0, 0.0, 16.0], [32.0, 64.0, 128.0]])
        csr = _fold_kernel(topology, kernel)

        first_row = csr.dest_idx[int(csr.indptr[0]) : int(csr.indptr[1])]
        assert len(first_row.tolist()) != len(set(first_row.tolist()))

        expected_rows = _reconstruct_kernel_rows(
            topology, kernel, include_center=False, adjust_on_edge=False
        )
        for src, (dests, weights) in enumerate(expected_rows):
            lo, hi = int(csr.indptr[src]), int(csr.indptr[src + 1])
            assert np.array_equal(csr.dest_idx[lo:hi], dests)
            assert np.array_equal(csr.weights[lo:hi], weights)

        ind, sperm = _valid_state(n_demes=6, n_ages=2, n_ztypes=2, seed=303)
        rate = np.full((6, 2, 2), 0.25)
        expected_ind, expected_sperm = _apply_reference(
            ind, sperm, expected_rows, rate, stay_after_send=True
        )
        for backend in ("python",):
            got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, backend)
            assert np.array_equal(got_ind, expected_ind), backend
            assert np.array_equal(got_sperm, expected_sperm), backend
        assert np.isclose(
            got_ind.sum() + got_sperm.sum(),
            ind.sum() + sperm.sum(),
            rtol=0.0,
            atol=1e-9,
        )

    def test_kernel_fold_without_topology_emits_empty_csr(self) -> None:
        """Kernel mode without a topology folds to an empty CSR (historical)."""
        kernel = np.ones((3, 3))
        csr = fold_migration_csr(
            n_demes=2,
            topology=None,
            adjacency_dense=np.zeros((2, 2)),
            migration_kernel=kernel,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=False,
            mode="kernel",
        )
        assert csr.dest_idx.size == 0
        assert csr.weights.size == 0
        assert csr.stay_after_send is True
        # An empty CSR routes nothing: with unstored sperm (virgin ==
        # female_total exactly) the engine returns the state unchanged.
        ind, _ = _valid_state(n_demes=2, n_ages=2, n_ztypes=2, seed=404)
        sperm = np.zeros((2, 2, 2, 2))
        got_ind, got_sperm = _run_engine(
            ind, sperm, csr, np.full((2, 2, 2), 0.4), "python"
        )
        assert np.array_equal(got_ind, ind)
        assert np.array_equal(got_sperm, sperm)

    def test_kernel_bank_routes_per_deme(self) -> None:
        """``kernel_bank`` + ``deme_kernel_ids`` fold each deme's own kernel."""
        topology = SquareGrid(rows=3, cols=1, neighborhood="von_neumann", wrap=False)
        cross3 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        cross5 = np.zeros((5, 5))
        for offset in (-2, -1, 1, 2):
            cross5[2 + offset, 2] = 1.0
            cross5[2, 2 + offset] = 1.0
        csr = _fold_kernel(
            topology,
            cross3,
            kernel_bank=[cross3, cross5],
            deme_kernel_ids=np.array([0, 1, 0], dtype=np.int64),
        )
        # Deme 0/2 use cross3 (one valid neighbor on the linear grid):
        # 1 * (1/4) / (1/4) = 1.0.  Deme 1 uses cross5 (two valid axial
        # neighbors out of total 8): 1 * (1/8) / (2/8) = 0.5 each.
        assert np.array_equal(csr.indptr, np.array([0, 1, 3, 4], dtype=np.int64))
        assert np.array_equal(csr.dest_idx, np.array([1, 0, 2, 1], dtype=np.int64))
        assert np.array_equal(csr.weights, np.array([1.0, 0.5, 0.5, 1.0]))

    def test_bookkeeping_orders_are_distinguishable(self) -> None:
        """The two ``stay_after_send`` orders produce different, each-exact results.

        With one raw weight of 0.3, stay-first keeps ``value - outbound``
        at the source (row sum < 1, undelivered share withheld) while
        send-first keeps ``value - moved_total`` (conserved).  The engine
        must switch between exactly these two arithmetic orders.
        """
        ind = np.zeros((2, 2, 1, 1))
        ind[0, 1, 0, 0] = 1000.0
        sperm = np.zeros((2, 1, 1, 1))
        rate = np.full((2, 2, 1), 0.5)
        indptr = np.array([0, 1, 1], dtype=np.int64)
        dest_idx = np.array([1], dtype=np.int64)
        weights = np.array([0.3])

        stay_ind, stay_sperm = apply_csr_migration(
            ind.copy(), sperm.copy(), indptr, dest_idx, weights, rate,
            False, False, False,
        )
        send_ind, send_sperm = apply_csr_migration(
            ind.copy(), sperm.copy(), indptr, dest_idx, weights, rate,
            False, False, True,
        )
        # Stay-first: the full outbound leaves, only rate*weight arrives.
        assert stay_ind[0, 1, 0, 0] == 1000.0 - 1000.0 * 0.5
        assert stay_ind[1, 1, 0, 0] == (1000.0 * 0.5) * 0.3
        # Send-first: moved_total decides the source residual.
        assert send_ind[1, 1, 0, 0] == (1000.0 * 0.5) * 0.3
        assert send_ind[0, 1, 0, 0] == 1000.0 - (1000.0 * 0.5) * 0.3
        # The two orders genuinely differ at the source bucket.
        assert stay_ind[0, 1, 0, 0] != send_ind[0, 1, 0, 0]
        assert stay_sperm.sum() == 0.0 and send_sperm.sum() == 0.0

    def test_csr_dense_row_accumulates_duplicates(self) -> None:
        """``csr_dense_row`` sums duplicate entries into one dense slot."""
        topology = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=True)
        kernel = np.array([[2.0, 1.0, 4.0], [8.0, 0.0, 16.0], [32.0, 64.0, 128.0]])
        csr = _fold_kernel(topology, kernel)
        row = csr_dense_row(csr, source_idx=0, n_demes=6)
        lo, hi = int(csr.indptr[0]), int(csr.indptr[1])
        manual = float(sum(float(csr.weights[p]) for p in range(lo, hi)))
        assert row.shape == (6,)
        assert row.sum() == manual


# ---------------------------------------------------------------------------
# 2. Sugar rules matrix
# ---------------------------------------------------------------------------


class TestRateSugarMatrix:
    """``normalize_migration_rate`` exact values and rejection rules."""

    @pytest.mark.parametrize(
        ("n_sexes", "n_ages", "adult_start", "scalar", "expected_row"),
        [
            (2, 3, 1, 0.25, np.array([0.0, 0.25, 0.25])),
            (2, 2, 0, 0.4, np.array([0.4, 0.4])),  # adult_start 0: no zeroing
            (1, 4, 2, 0.1, np.array([0.0, 0.0, 0.1, 0.1])),
            (2, 2, 1, 0.0, np.array([0.0, 0.0])),
        ],
    )
    def test_scalar_sugar_exact_values(
        self,
        n_sexes: int,
        n_ages: int,
        adult_start: int,
        scalar: float,
        expected_row: NDArray[np.float64],
    ) -> None:
        result = normalize_migration_rate(scalar, n_sexes, n_ages, adult_start)
        assert result.shape == (n_sexes, n_ages)
        assert result.dtype == np.float64
        for sex in range(n_sexes):
            assert np.array_equal(result[sex, :], expected_row)

    def test_discrete_single_age_gets_full_rate(self) -> None:
        """Discrete models (``n_ages == 1``) ignore the adult-age sugar."""
        result = normalize_migration_rate(0.3, 2, 1, 1)
        assert np.array_equal(result, np.full((2, 1), 0.3))
        # A length-1 sequence behaves like the scalar sugar.
        vector = normalize_migration_rate([0.3], 2, 1, 1)
        assert np.array_equal(vector, np.full((2, 1), 0.3))
        # On an age-structured model the length-1 sequence re-enters the
        # scalar sugar, so juveniles are zeroed.
        tiled = normalize_migration_rate([0.5], 2, 3, 1)
        assert np.array_equal(tiled, np.tile(np.array([0.0, 0.5, 0.5]), (2, 1)))

    def test_per_sex_dict_exact_values(self) -> None:
        result = normalize_migration_rate(
            {"F": 0.2, "M": [0.01, 0.02, 0.03]}, 2, 3, 1
        )
        assert np.array_equal(result[0, :], np.array([0.0, 0.2, 0.2]))
        assert np.array_equal(result[1, :], np.array([0.01, 0.02, 0.03]))

    def test_per_sex_dict_missing_sex_defaults_to_zero(self) -> None:
        result = normalize_migration_rate({"F": 0.2}, 2, 2, 1)
        assert np.array_equal(result[0, :], np.array([0.0, 0.2]))
        assert np.array_equal(result[1, :], np.array([0.0, 0.0]))

    @pytest.mark.parametrize(("key", "sex_row"), [("f", 0), ("Male", 1)])
    def test_per_sex_dict_accepts_case_variants(self, key: str, sex_row: int) -> None:
        result = normalize_migration_rate({key: 0.7}, 2, 2, 1)
        expected = np.zeros((2, 2))
        expected[sex_row, :] = (0.0, 0.7)
        assert np.array_equal(result, expected)

    def test_age_vector_bypasses_adult_sugar(self) -> None:
        """An explicit ``(n_ages,)`` vector is used as-is, juveniles included."""
        declaration = np.array([0.05, 0.1, 0.2])
        result = normalize_migration_rate(declaration, 2, 3, 1)
        assert np.array_equal(result, np.tile(declaration, (2, 1)))
        # Fresh copy: mutating the declaration afterwards changes nothing.
        declaration[2] = 9.0
        assert result[1, 2] == 0.2

    def test_two_d_table_exact_and_copied(self) -> None:
        table = np.array([[0.0, 0.3], [0.01, 0.02]])
        result = normalize_migration_rate(table, 2, 2, 1)
        assert np.array_equal(result, table)
        assert result is not table
        table[0, 1] = 5.0
        assert result[0, 1] == 0.3

    def test_three_d_column_is_rejected_by_the_sugar(self) -> None:
        """Build-time declarations are per-deme uniform; the 3-D column is
        the exclusive ``tensor_write`` form."""
        with pytest.raises(ValueError, match="does not match"):
            normalize_migration_rate(np.zeros((2, 2, 2)), 2, 2, 1)

    @pytest.mark.parametrize(
        "declaration",
        [
            np.zeros((3, 2)),  # 2-D shape mismatch
            np.zeros(5),  # 1-D length mismatch
            {"F": [0.1, 0.2, 0.3]},  # per-sex vector length mismatch
        ],
    )
    def test_invalid_shapes_raise_value_error(self, declaration: RateDeclaration) -> None:
        with pytest.raises(ValueError, match="does not match|not a valid sex"):
            normalize_migration_rate(declaration, 2, 2, 1)

    def test_per_sex_key_out_of_range_raises(self) -> None:
        """A valid label beyond ``n_sexes`` is rejected, not silently dropped."""
        with pytest.raises(ValueError, match="not a valid sex"):
            normalize_migration_rate({"M": 0.1}, 1, 2, 1)


class TestFoldErrorPaths:
    """Degenerate folds and unknown strategies fail loudly, never silently."""

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="migration_strategy must be one of"):
            resolve_migration_mode("diagonal", None, None, None)

    def test_kernel_mode_without_any_kernel_raises(self) -> None:
        topology = SquareGrid(rows=1, cols=2, neighborhood="von_neumann", wrap=False)
        with pytest.raises(ValueError, match="requires migration_kernel"):
            fold_migration_csr(
                n_demes=2,
                topology=topology,
                adjacency_dense=np.zeros((2, 2)),
                migration_kernel=None,
                kernel_bank=None,
                deme_kernel_ids=None,
                kernel_include_center=False,
                adjust_on_edge=False,
                mode="kernel",
            )

    def test_all_zero_kernel_emits_empty_rows(self) -> None:
        """A kernel with no positive weight folds to empty rows, not NaNs."""
        topology = SquareGrid(rows=1, cols=2, neighborhood="von_neumann", wrap=False)
        csr = _fold_kernel(topology, np.zeros((3, 3)))
        assert np.array_equal(csr.indptr, np.zeros(3, dtype=np.int64))
        assert csr.dest_idx.size == 0
        assert csr.weights.size == 0
        assert csr.stay_after_send is True

    def test_zero_deme_adjacency_fold_is_empty(self) -> None:
        csr = fold_migration_csr(
            n_demes=0,
            topology=None,
            adjacency_dense=np.zeros((0, 0)),
            migration_kernel=None,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=False,
            mode="adjacency",
        )
        assert csr.indptr.shape == (1,)
        assert csr.dest_idx.size == 0
        assert csr.weights.size == 0
        assert csr.stay_after_send is False


# ---------------------------------------------------------------------------
# 3. Contract sentinels
# ---------------------------------------------------------------------------


class TestContractSentinels:
    """Panmictic defaults, the 1-deme self-loop, and frozen discipline."""

    def test_default_contract_fields_carry_panmictic_sentinels(self) -> None:
        """Bare constructions default to one deme, no edges, (0,0,0) rate."""
        from natal.contracts.blueprint import Blueprint
        from natal.contracts.params import Params

        defaults = Blueprint._field_defaults
        assert defaults["n_demes"] == 1
        assert defaults["migration_indptr"].shape == (0,)
        assert defaults["migration_dest_idx"].shape == (0,)
        assert defaults["migration_weights"].shape == (0,)
        rate_field = next(
            f for f in dataclasses.fields(Params) if f.name == "migration_rate"
        )
        factory = rate_field.default_factory
        assert factory is not None
        sentinel = factory()
        assert sentinel.shape == (0, 0, 0)
        assert sentinel.dtype == np.float64

    def test_one_deme_container_folds_identity_self_loop(self) -> None:
        """A 1-deme spatial container folds exactly one self-loop edge."""
        species = _species("adv_one_deme")
        pop = (
            SpatialPopulation.builder(species, n_demes=1)
            .setup(name="adv_one_deme", stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0]},
                    "male": {"WT|WT": [0.0, 100.0]},
                }
            )
            .build()
        )
        csr = pop.migration_csr
        assert np.array_equal(csr.indptr, np.array([0, 1], dtype=np.int64))
        assert np.array_equal(csr.dest_idx, np.array([0], dtype=np.int64))
        assert np.array_equal(csr.weights, np.array([1.0]))
        # The Blueprint mirrors the values but owns fresh copies.
        bp = pop.blueprint
        assert np.array_equal(bp.migration_weights, csr.weights)
        assert bp.migration_weights is not csr.weights
        assert bp.migration_dest_idx is not csr.dest_idx

    def test_blueprint_migration_fields_are_frozen(self) -> None:
        """Every migration field rejects attribute rebinding (NamedTuple)."""
        species = _species("adv_frozen_fields")
        pop = (
            SpatialPopulation.builder(species, n_demes=2)
            .setup(name="adv_frozen_fields", stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0]},
                    "male": {"WT|WT": [0.0, 100.0]},
                }
            )
            .build()
        )
        bp = pop.blueprint
        for field in ("migration_indptr", "migration_dest_idx", "migration_weights"):
            with pytest.raises(AttributeError):
                setattr(bp, field, np.zeros(3))  # type: ignore[misc]  # frozen-discipline probe

    def test_materialize_copies_the_migration_payload(self) -> None:
        """Mutating the caller's payload after materialization cannot leak."""
        species = _species("adv_payload_copy")
        pop = (
            SpatialPopulation.builder(species, n_demes=2)
            .setup(name="adv_payload_copy", stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0]},
                    "male": {"WT|WT": [0.0, 100.0]},
                }
            )
            .build()
        )
        csr = pop.migration_csr
        rate = np.full((2, 2, 2), 0.2)
        payload = SpatialMigration(
            indptr=csr.indptr,
            dest_idx=csr.dest_idx,
            weights=csr.weights,
            rate=rate,
        )
        mat = materialize(pop.deme(0).config, payload)
        payload.weights[:] = 99.0
        payload.rate[0, 0, 1] = 7.0
        assert not np.any(mat.blueprint.migration_weights == 99.0)
        assert mat.params.migration_rate[0, 0, 1] == 0.2


# ---------------------------------------------------------------------------
# 4. Three-backend consistency (bitwise)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RUST_OK, reason="natal._engine_rs not built")
class TestThreeBackendConsistency:
    """Python == Rust on identical CSR + rate column inputs."""

    @staticmethod
    def _assert_three_backends_bitwise(
        csr: MigrationCSR, rate: NDArray[np.float64], seed: int
    ) -> None:
        ind, sperm = _valid_state(
            n_demes=csr.indptr.shape[0] - 1,
            n_ages=rate.shape[2],
            n_ztypes=2,
            seed=seed,
        )
        from natal.backends.rust.rust_backend import rust_migrate_csr_deterministic

        py_ind, py_sperm = _run_engine(ind, sperm, csr, rate, "python")
        rs_ind, rs_sperm = rust_migrate_csr_deterministic(
            ind, sperm, csr.indptr, csr.dest_idx, csr.weights, rate,
            csr.stay_after_send,
        )
        assert np.array_equal(py_ind, rs_ind)
        assert np.array_equal(py_sperm, rs_sperm)

    def test_adjacency_csr_three_backends_bitwise(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.6, 0.4, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.2, 0.3, 0.0, 0.5],
                [0.0, 0.7, 0.3, 0.0],
            ]
        )
        rate = np.zeros((4, 2, 2))
        rate[:, 0, :] = 0.1
        rate[:, 1, 0] = np.array([0.0, 0.2, 0.4, 0.6])
        rate[:, 1, 1] = 0.15
        self._assert_three_backends_bitwise(_fold_adjacency(adjacency), rate, seed=505)

    def test_empty_row_isolated_deme_three_backends_bitwise(self) -> None:
        """An isolated deme (empty CSR row) keeps ALL its mass on every backend.

        Regression guard for the Rust deterministic kernel, which used to
        evaporate ``value * rate`` for empty rows in the stay_after=False
        bookkeeping order while the Python reference kept everything.
        An isolated deme makes no outbound moves; inbound moves are legal.
        """
        # 2 demes, explicit adjacency: deme 1 is isolated (empty row),
        # deme 0 sends everything to deme 1.
        adjacency = np.array([[0.0, 1.0], [0.0, 0.0]])
        csr = _fold_adjacency(adjacency)
        assert csr.indptr[1] - csr.indptr[0] == 1  # deme 0 has one entry
        assert csr.indptr[2] - csr.indptr[1] == 0  # deme 1 row is EMPTY
        rate = np.full((2, 2, 2), 0.3)

        # Males-only state keeps the mass ledger independent of the
        # virgin-female/sperm coupling bookkeeping.
        n_ages = rate.shape[2]
        ind = np.zeros((2, 2, n_ages, 2), dtype=np.float64)
        ind[:, 1, :, 0] = 100.0
        ind[:, 1, :, 1] = 40.0
        sperm = np.zeros((2, n_ages, 2, 2), dtype=np.float64)
        from natal.backends.rust.rust_backend import rust_migrate_csr_deterministic

        py_ind, py_sperm = _run_engine(ind, sperm, csr, rate, "python")
        rs_ind, rs_sperm = rust_migrate_csr_deterministic(
            ind, sperm, csr.indptr, csr.dest_idx, csr.weights, rate,
            csr.stay_after_send,
        )
        # Mass conservation across every backend.
        for name, (mi, ms) in {
            "python": (py_ind, py_sperm),
            "rust": (rs_ind, rs_sperm),
        }.items():
            assert mi.sum() == pytest.approx(ind.sum() + sperm.sum()), name
        # Isolated deme 1 sends nothing out but still receives deme 0's
        # outflow (value * 0.3): original [100, 40] per age + [30, 12].
        for name, mi in {"python": py_ind, "rust": rs_ind}.items():
            assert np.array_equal(mi[1, 1, :, 0], [130.0, 130.0]), name
            assert np.array_equal(mi[1, 1, :, 1], [52.0, 52.0]), name
        # And both backends agree bitwise.
        assert np.array_equal(py_ind, rs_ind)
        assert np.array_equal(py_sperm, rs_sperm)

    def test_kernel_wrap_csr_three_backends_bitwise(self) -> None:
        topology = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=True)
        kernel = np.array([[2.0, 1.0, 4.0], [8.0, 0.0, 16.0], [32.0, 64.0, 128.0]])
        csr = _fold_kernel(topology, kernel)
        rate = np.full((6, 2, 2), 0.2)
        rate[2, 1, :] = 0.45
        rate[4, :, 0] = 0.0
        self._assert_three_backends_bitwise(csr, rate, seed=606)


# ---------------------------------------------------------------------------
# 5. Per-deme / per-sex rate consumption at the population level
# ---------------------------------------------------------------------------


def _probe_population(
    name: str,
    rate3d: NDArray[np.float64] | None,
    rust: bool,
) -> SpatialPopulation:
    """Two-deme deterministic model: deme 0 seeded, one-way edge 0 -> 1."""
    species = _species(f"adv_{name}")
    adjacency = np.array([[0.0, 1.0], [0.0, 0.0]])
    pop = (
        SpatialPopulation.builder(species, n_demes=2)
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting(
                [
                    {"female": {"WT|WT": [50.0, 400.0, 0.0]},
                     "male": {"WT|WT": [40.0, 300.0, 0.0]}},
                    {"female": {"WT|WT": [0.0, 0.0, 0.0]},
                     "male": {"WT|WT": [0.0, 0.0, 0.0]}},
                ]
            )
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 1.0],
            male_age_based_survival=[1.0, 1.0, 1.0],
        )
        .reproduction(
            eggs_per_female=0.0,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0, 0.0],
            age_based_reproduction_rate=[0.0, 0.0, 0.0],
        )
        .competition(juvenile_growth_mode=0)
        .migration(adjacency=adjacency)
        .build()
    )
    if rate3d is not None:
        pop.params.tensor_write("migration_rate", rate3d)
    if rust:
        pop.enable_rust_backend(seed=5)
    return pop


@pytest.mark.skipif(not _RUST_OK, reason="natal._engine_rs not built")
class TestPerDemePerSexConsumption:
    """The rate column is consumed per (deme, sex, age) by live runs."""

    @pytest.mark.parametrize("backend", ["python_dispatch", "rust"])
    def test_asymmetric_rate_column_moves_exactly_its_entries(
        self, backend: str
    ) -> None:
        """Only deme-0 adult buckets migrate; every other bucket is untouched.

        Aging runs before migration, so the seeded age-1 cohort is subject
        to the age-2 rate entries — the write must be consumed at exactly
        the written (deme, sex, age) coordinates.
        """
        rate = np.zeros((2, 2, 3))
        rate[0, 0, 2] = 0.5  # deme 0, female, post-aging adult age
        rate[0, 1, 2] = 0.9  # deme 0, male, post-aging adult age
        use_rust = backend == "rust"

        def run(name: str, column: NDArray[np.float64] | None):
            pop = _probe_population(name, column, rust=use_rust)
            if backend == "python_dispatch":
                with python_reference():
                    pop.run_tick()
            else:
                pop.run_tick()
            return np.stack([d.state.individual_count for d in pop.demes])

        control = run(f"ctl_{backend}", None)
        treatment = run(f"trt_{backend}", rate)

        # Buckets whose rate entry is 0 stay bit-identical to the control,
        # including the aged juvenile cohorts (age < 2).
        assert np.array_equal(control[:, :, :2, :], treatment[:, :, :2, :])
        # Deme 0 female adults: outbound = 400 * 0.5, exactly.
        assert treatment[0, 0, 2, 0] == 400.0 - 400.0 * 0.5
        assert treatment[1, 0, 2, 0] == 400.0 * 0.5
        # Deme 0 male adults: outbound = 300 * 0.9, exactly.
        assert treatment[0, 1, 2, 0] == 300.0 - 300.0 * 0.9
        assert treatment[1, 1, 2, 0] == 300.0 * 0.9

    def test_rate_view_stays_stable_across_tensor_write(self) -> None:
        """``tensor_write`` copies contents in place; earlier views stay live."""
        rate = np.zeros((2, 2, 3))
        rate[0, 0, 2] = 0.5
        pop = _probe_population("view_stable", None, rust=False)
        view_before = pop.params.migration_rate
        pop.params.tensor_write("migration_rate", rate)
        assert np.array_equal(view_before, rate)
        view_after = pop.params.migration_rate
        assert view_before.base is view_after.base  # one live array underneath


# ---------------------------------------------------------------------------
# 6. Boundary semantics unification (kernel mode)
# ---------------------------------------------------------------------------


class TestBoundarySemanticsUnification:
    """Boundary demes follow the reference (row-renormalized) semantics."""

    @staticmethod
    def _linear_fold(adjust_on_edge: bool) -> MigrationCSR:
        topology = SquareGrid(rows=3, cols=1, neighborhood="von_neumann", wrap=False)
        cross = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        return _fold_kernel(topology, cross, adjust_on_edge=adjust_on_edge)

    @pytest.mark.parametrize("adjust_on_edge", [False, True])
    def test_boundary_deme_emits_full_rate_over_shared_csr(
        self, adjust_on_edge: bool
    ) -> None:
        """A 1-neighbor boundary deme sends its full ``value * rate`` outbound.

        The pre-slice-5 Rust kernel scaled boundary rows by the kernel
        total (sending only a quarter here); the unified CSR carries the
        reference semantics instead, and the engines consume it verbatim.
        """
        csr = self._linear_fold(adjust_on_edge)
        # Exact fold: boundary rows renormalize onto the single valid neighbor.
        assert np.array_equal(csr.indptr, np.array([0, 1, 3, 4], dtype=np.int64))
        assert np.array_equal(csr.dest_idx, np.array([1, 0, 2, 1], dtype=np.int64))
        assert np.array_equal(csr.weights, np.array([1.0, 0.5, 0.5, 1.0]))

        ind = np.zeros((3, 2, 1, 1))
        ind[:, 1, 0, 0] = [1000.0, 500.0, 250.0]
        sperm = np.zeros((3, 1, 1, 1))
        rate = np.full((3, 2, 1), 0.4)
        got_ind, got_sperm = _run_engine(ind, sperm, csr, rate, "python")
        # Deme 0 (one valid neighbor) moves its full outbound to deme 1
        # and still receives deme 1's cross-share.
        assert got_ind[0, 1, 0, 0] == 1000.0 - 1000.0 * 0.4 + 500.0 * 0.4 * 0.5
        # Deme 1: inbound from both boundary demes plus its own residual.
        assert got_ind[1, 1, 0, 0] == (
            1000.0 * 0.4  # from deme 0, weight 1.0
            + 500.0 - 500.0 * 0.4  # deme 1 residual (moved_total == outbound)
            + 250.0 * 0.4  # from deme 2, weight 1.0
        )
        assert got_ind[2, 1, 0, 0] == 250.0 - 250.0 * 0.4 + 500.0 * 0.4 * 0.5
        assert got_sperm.sum() == 0.0

    def test_adjust_flag_keeps_rows_normalized(self) -> None:
        """Both edge modes fold rows summing to one (flag order only)."""
        for adjust in (False, True):
            csr = self._linear_fold(adjust)
            for src in range(3):
                lo, hi = int(csr.indptr[src]), int(csr.indptr[src + 1])
                assert csr.weights[lo:hi].sum() == pytest.approx(1.0, abs=1e-15)
        # The folded weights agree between the two flags to well below 1 ulp.
        assert np.allclose(
            self._linear_fold(False).weights,
            self._linear_fold(True).weights,
            rtol=0.0,
            atol=1e-15,
        )


# ---------------------------------------------------------------------------
# 7. Negative contracts (hard)
# ---------------------------------------------------------------------------

_REMOVED_POPULATION_ATTRIBUTES = (
    "adjacency",
    "migration_mode",
    "migration_strategy",
    "migration_kernel",
    "kernel_bank",
    "deme_kernel_ids",
    "adjust_migration_on_edge",
    "migration_rate",
)


class TestRemovedSurfaceHard:
    """The deleted migration surface is unconstructible, not just unused."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "natal.backends.reference.migration.kernel",
            "natal.backends.reference.migration.kernel",
        ],
    )
    def test_kernel_module_is_unimportable(self, module_name: str) -> None:
        with pytest.raises(ImportError):
            importlib.import_module(module_name)

    @pytest.mark.parametrize("attribute", _REMOVED_POPULATION_ATTRIBUTES)
    def test_population_migration_surface_removed(self, attribute: str) -> None:
        species = _species("adv_negative_surface")
        pop = (
            SpatialPopulation.builder(species, n_demes=2)
            .setup(name="adv_negative_surface", stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0]},
                    "male": {"WT|WT": [0.0, 100.0]},
                }
            )
            .build()
        )
        assert not hasattr(pop, attribute)
        assert not hasattr(type(pop), attribute)
        # Positive control: the probe object is a real, populated container.
        assert hasattr(pop, "migration_csr")

    def test_topology_legacy_symbols_removed(self) -> None:
        topology_module = importlib.import_module("natal.frontend.spatial.topology")
        spatial_module = importlib.import_module("natal.frontend.spatial")
        for symbol in (
            "MigrationParams",
            "HeterogeneousKernelParams",
            "SpatialTopology",
        ):
            assert not hasattr(topology_module, symbol)
            assert symbol not in getattr(topology_module, "__all__", ())
            assert not hasattr(spatial_module, symbol)
        migration_package = importlib.import_module(
            "natal.backends.reference.migration"
        )
        assert "kernel" not in getattr(migration_package, "__all__", ())
        assert not hasattr(migration_package, "apply_spatial_kernel_migration")


# ---------------------------------------------------------------------------
# 8. Parameter snapshot
# ---------------------------------------------------------------------------


class TestRateSnapshot:
    """The written rate column is what the contract snapshot reports."""

    @staticmethod
    def _contract():
        species = _species("adv_snapshot")
        pop = (
            SpatialPopulation.builder(species, n_demes=2)
            .setup(name="adv_snapshot", stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0]},
                    "male": {"WT|WT": [0.0, 100.0]},
                }
            )
            .build()
        )
        csr = pop.migration_csr
        rate = np.zeros((2, 2, 2))
        rate[0, :, 1] = 0.3
        return materialize(
            pop.deme(0).config,
            SpatialMigration(
                indptr=csr.indptr,
                dest_idx=csr.dest_idx,
                weights=csr.weights,
                rate=rate,
            ),
        )

    def test_snapshot_ecology_carries_written_rate(self) -> None:
        mat = self._contract()
        snapshot = mat.params.snapshot_ecology()
        snapshot_rate = snapshot["migration_rate"]
        assert isinstance(snapshot_rate, np.ndarray)
        assert snapshot_rate.shape == (2, 2, 2)
        assert np.array_equal(snapshot_rate, mat.params.migration_rate)
        # An in-place contract write is reflected in the next snapshot.
        mat.params.migration_rate[1, 1, 1] = 0.9
        refreshed = mat.params.snapshot_ecology()
        refreshed_rate = refreshed["migration_rate"]
        assert isinstance(refreshed_rate, np.ndarray)
        assert refreshed_rate[1, 1, 1] == 0.9
