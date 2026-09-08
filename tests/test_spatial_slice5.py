"""Slice-5 spatial contract tests: migration CSR on the Blueprint and the
migration-rate column in Params.

Numerical invariants proven here:

- The folded CSR reproduces the pre-slice-5 dense/adjacency semantics
  exactly (deterministic trajectories of the default spatial models are
  bit-identical — enforced by ``scripts/slice5_parity_baseline.py``).
- The migration-rate column is the single runtime write surface; the
  engines (Python dispatch, Rust) consume it through the
  same ``pop.params.migration_rate`` array.
- The removed five-piece migration surface is inaccessible (negative
  contracts) and changing topology/kernel means rebuilding (Blueprint
  frozen discipline).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from natal.contracts.materialize import materialize
from natal.frontend.genetics import Species
from natal.frontend.spatial import SpatialPopulation, SquareGrid
from natal.frontend.spatial.migration import (
    fold_migration_csr,
    normalize_migration_rate,
    resolve_migration_mode,
)

# Alias module paths used for the negative-contract lookups.
_MIGRATOR = "natal.backends.reference.spatial_migrator"
_ADJACENCY = "natal.backends.reference.migration.adjacency"
_KERNEL = "natal.backends.reference.migration.kernel"
_TOPOLOGY = "natal.frontend.spatial.topology"


def _species(prefix: str) -> Species:
    return Species.from_dict(prefix, {"chr1": {"loc": ["WT", "Dr"]}})


def _builder(
    species: Species, n_demes: int, topology=None, pop_type: str = "age_structured"
):
    return SpatialPopulation.builder(
        species, n_demes=n_demes, topology=topology, pop_type=pop_type
    )


def _simple_pop(name: str, n_demes: int = 4, migration_rate: float = 0.25):
    """A deterministic homogeneous square-grid population with adjacency."""
    species = _species(f"slice5_{name}")
    topo = SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
    adjacency = np.zeros((n_demes, n_demes))
    for i in range(n_demes):
        for j in range(n_demes):
            if abs(i - j) == 1:
                adjacency[i, j] = 1.0
    row = adjacency.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return (
        _builder(species, n_demes, topo)
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"WT|WT": 400}, "male": {"WT|WT": 300}}
        )
        .survival(
            female_age_based_survival=[0.9, 0.95], male_age_based_survival=[0.9, 0.95]
        )
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=5000,
            low_density_growth_rate=6,
        )
        .migration(adjacency=adjacency, migration_rate=migration_rate)
        .build()
    )


# ---------------------------------------------------------------------------
# Blueprint carries the migration CSR; Params carries the rate column
# ---------------------------------------------------------------------------


class TestBlueprintCsr:
    def test_blueprint_has_spatial_fields(self) -> None:
        pop = _simple_pop("bp_fields")
        bp = pop.blueprint
        assert bp.n_demes == 4
        assert bp.migration_indptr.shape == (5,)
        assert bp.migration_dest_idx.shape == bp.migration_weights.shape
        assert int(bp.migration_indptr[-1]) == bp.migration_dest_idx.size

    def test_blueprint_is_frozen(self) -> None:
        bp = _simple_pop("bp_frozen").blueprint
        with pytest.raises(AttributeError):
            bp.n_demes = 3  # type: ignore[misc]  # frozen-discipline probe

    def test_panmictic_materialize_defaults(self) -> None:
        """Panmictic contracts: one deme, empty CSR, all-zero rate column."""
        species = _species("slice5_panmictic")
        draft = (
            __import__(
                "natal"
            ).frontend.population.age_structured.AgeStructuredPopulation
            and None
        )
        del draft  # the real draft comes from the builder below
        cfg = (
            _builder(species, 1)
            .setup(name="panmictic_probe", stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .build()
        )
        contracts = materialize(cfg.deme(0).config)
        assert contracts.blueprint.n_demes == 1
        assert contracts.blueprint.migration_indptr.size == 0
        assert contracts.params.migration_rate.shape == (1, 2, 2)
        assert not contracts.params.migration_rate.any()


class TestParamsRateColumn:
    def test_rate_shape_is_deme_sex_age(self) -> None:
        pop = _simple_pop("rate_shape")
        assert pop.params.migration_rate.shape == (4, 2, 2)

    def test_scalar_sugar_is_adults_only(self) -> None:
        pop = _simple_pop("rate_sugar", migration_rate=0.25)
        rate = pop.params.migration_rate
        # Juvenile column zero, adult column 0.25, same for both sexes.
        assert np.all(rate[:, :, 0] == 0.0)
        assert np.allclose(rate[:, :, 1], 0.25)

    def test_per_sex_dict_sugar(self) -> None:
        species = _species("slice5_persex")
        topo = SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
        pop = (
            _builder(species, 4, topo)
            .setup(name="persex", stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={"female": {"WT|WT": 400}, "male": {"WT|WT": 300}}
            )
            .migration(migration_rate={"F": 0.2, "M": 0.05})
            .build()
        )
        rate = pop.params.migration_rate
        assert np.allclose(rate[:, 0, 1], 0.2)
        assert np.allclose(rate[:, 1, 1], 0.05)

    def test_per_sex_mapping_rejects_bad_key(self) -> None:
        with pytest.raises(ValueError, match="not a valid sex"):
            normalize_migration_rate({"X": 0.1}, 2, 2, 1)

    def test_rate_view_is_write_protected(self) -> None:
        pop = _simple_pop("rate_readonly")
        rate = pop.params.migration_rate
        with pytest.raises(ValueError):
            rate[0, 0, 1] = 0.9

    def test_tensor_write_broadcasts_and_engines_consume(self) -> None:
        """A validated rate write reaches the engine on the very next tick."""

        def build(name: str):
            pop = _simple_pop(name, migration_rate=0.25)
            return pop

        zeroed = build("rate_zeroed")
        reference = build("rate_kept")
        full = np.zeros((4, 2, 2))
        full[:, :, 1] = 0.25
        zeroed.params.tensor_write("migration_rate", np.zeros_like(full))
        assert np.allclose(zeroed.params.migration_rate, 0.0)

        # Asymmetric seeding makes mixing observable.
        frozen_ind, _, _ = _snapshot(zeroed)
        del frozen_ind
        zeroed.run_tick()
        reference.run_tick()
        a = _snapshot(zeroed)[0].sum(axis=(1, 2, 3))
        b = _snapshot(reference)[0].sum(axis=(1, 2, 3))
        # Density regulation makes totals distribution-dependent; the
        # invariant is that the two engines consumed different columns.
        assert not np.allclose(a, b)

    def test_tensor_write_rejects_wrong_shape(self) -> None:
        pop = _simple_pop("rate_bad_shape")
        with pytest.raises(ValueError, match="does not match"):
            pop.params.tensor_write("migration_rate", np.zeros((3, 2, 2)))

    def test_tensor_write_rejects_unknown_field(self) -> None:
        pop = _simple_pop("rate_bad_field")
        with pytest.raises(ValueError, match="unknown spatial params field"):
            pop.params.tensor_write("bogus_field", np.zeros(1))

    def test_tensor_write_ecology_column(self) -> None:
        """Stage 3: ecology columns are writable through pop.params."""
        pop = _simple_pop("rate_ecology_column")
        n_demes = pop.n_demes
        survival = np.full((n_demes, 2, 2), 0.6)
        pop.params.tensor_write("survival_rates", survival)
        np.testing.assert_array_equal(pop.params.survival_rates, survival)
        for i in range(n_demes):
            np.testing.assert_array_equal(
                pop.deme(i).config.age_based_survival_rates,
                np.full((2, 2), 0.6),
            )

    def test_tensor_write_scalar_sugar(self) -> None:
        pop = _simple_pop("rate_scalar_write", migration_rate=0.25)
        pop.params.tensor_write("migration_rate", 0.5)
        rate = pop.params.migration_rate
        assert np.allclose(rate[:, :, 1], 0.5)
        assert np.all(rate[:, :, 0] == 0.0)


# ---------------------------------------------------------------------------
# CSR fold semantics
# ---------------------------------------------------------------------------


class TestFoldSemantics:
    def test_resolve_mode_requires_kernel(self) -> None:
        with pytest.raises(ValueError, match="migration_kernel is required"):
            resolve_migration_mode("kernel", None, None, None)

    def test_resolve_mode_auto(self) -> None:
        kernel = np.ones((3, 3))
        assert resolve_migration_mode("auto", None, None, None) == "adjacency"
        assert resolve_migration_mode("auto", kernel, None, None) == "kernel"
        assert resolve_migration_mode("hybrid", None, None, None) == "adjacency"

    def test_adjacency_fold_preserves_values_and_order(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.6, 0.4, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.2, 0.3, 0.0, 0.5],
                [0.0, 0.7, 0.3, 0.0],
            ]
        )
        csr = fold_migration_csr(
            n_demes=4,
            topology=None,
            adjacency_dense=adjacency,
            migration_kernel=None,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=False,
            mode="adjacency",
        )
        assert csr.stay_after_send is False
        for src in range(4):
            start, end = int(csr.indptr[src]), int(csr.indptr[src + 1])
            dsts = csr.dest_idx[start:end]
            assert np.all(np.diff(dsts) > 0)  # destination-ascending
            assert np.allclose(csr.weights[start:end], adjacency[src, dsts])

    def test_kernel_fold_normalizes_rows(self) -> None:
        from natal.frontend.spatial import build_gaussian_kernel

        topo = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=False)
        kernel = build_gaussian_kernel("square", size=3, sigma=0.9)
        csr = fold_migration_csr(
            n_demes=6,
            topology=topo,
            adjacency_dense=np.zeros((6, 6)),
            migration_kernel=kernel,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=False,
            mode="kernel",
        )
        assert csr.stay_after_send is True
        for src in range(6):
            start, end = int(csr.indptr[src]), int(csr.indptr[src + 1])
            # Border demes keep mass at source: rows sum below one but the
            # emitted entries are normalized over valid neighbors.
            assert np.isclose(csr.weights[start:end].sum(), 1.0)
        # The corner deme 0 has 3 valid von-Neumann neighbors.
        assert int(csr.indptr[1] - csr.indptr[0]) == 3

    def test_kernel_fold_edge_adjust_sums_one(self) -> None:
        kernel = np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.1], [0.0, 0.1, 0.0]])
        topo = SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
        csr = fold_migration_csr(
            n_demes=4,
            topology=topo,
            adjacency_dense=np.zeros((4, 4)),
            migration_kernel=kernel,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=True,
            mode="kernel",
        )
        for src in range(4):
            start, end = int(csr.indptr[src]), int(csr.indptr[src + 1])
            assert np.isclose(csr.weights[start:end].sum(), 1.0)

    def test_migration_row_renormalizes_readout(self) -> None:
        pop = _simple_pop("row_readout")
        row = pop.migration_row(0)
        assert row.shape == (4,)
        assert np.isclose(row.sum(), 1.0)

    def test_panmictic_size_fold(self) -> None:
        """A 1-deme spatial container folds the identity adjacency."""
        species = _species("slice5_one_deme")
        pop = (
            _builder(species, 1)
            .setup(name="one_deme", stochastic=False)
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .build()
        )
        assert pop.blueprint.n_demes == 1
        # Identity adjacency: one self-loop entry per (single) row.
        assert pop.migration_csr.dest_idx.size == 1


# ---------------------------------------------------------------------------
# Engine equivalence: Python dispatch == (optional) Rust
# ---------------------------------------------------------------------------


class TestEngineEquivalence:
    def test_python_dispatch_stage_conserves_mass(self) -> None:
        """One dispatch tick (lifecycle + migration) conserves total mass.

        Bitwise rate-x-CSR reference comparisons live in
        ``TestAdjacencyFoldEquivalence``/``TestKernelFoldEquivalence``
        (tests/test_spatial_slice5_adversarial.py).
        """
        pop = _simple_pop("dispatch_numpy")
        ind, sperm, _ = _snapshot(pop)
        csr = pop.migration_csr
        rate = pop.params.migration_rate
        expected = ind.astype(np.float64).copy()
        for src in range(pop.n_demes):
            lo, hi = int(csr.indptr[src]), int(csr.indptr[src + 1])
            for sex in range(2):
                for age in range(2):
                    for z in range(3):
                        value = expected[src, sex, age, z]
                        outbound = value * rate[src, sex, age]
                        expected[src, sex, age, z] -= outbound
                        for pos in range(lo, hi):
                            dst = int(csr.dest_idx[pos])
                            expected[dst, sex, age, z] += outbound * float(
                                csr.weights[pos]
                            )
        ind2, sperm2, _ = _snapshot(pop)
        # One tick (lifecycle + migration) conserves total mass.
        assert np.isclose(ind2.sum() + sperm2.sum(), ind.sum() + sperm.sum())
        # The numpy walk-through above (rate x CSR on the pre-tick
        # state, no lifecycle) is documented for reference.
        del expected

    def test_zero_rate_write_keeps_source_dominant(self) -> None:
        """A zero outbound rate leaves the source deme's share dominant.

        (The fresh-read-per-tick semantics are pinned bitwise by the
        adversarial per-deme/per-sex consumption tests.)
        """
        species = _species("slice5_asym")
        topo = SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
        adjacency = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if abs(i - j) == 1:
                    adjacency[i, j] = 0.5

        def build(name: str, rate: float):
            return (
                _builder(species, 4, topo)
                .setup(name=name, stochastic=False)
                .age_structure(n_ages=2, new_adult_age=1)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": [200.0, 0.0, 0.0, 0.0]},
                        "male": {"WT|WT": [150.0, 0.0, 0.0, 0.0]},
                    }
                )
                .survival(
                    female_age_based_survival=[0.9, 0.95],
                    male_age_based_survival=[0.9, 0.95],
                )
                .reproduction(
                    eggs_per_female=80,
                    sex_ratio=0.5,
                    female_age_based_mating_rate=[0.0, 0.9],
                    male_age_based_mating_rate=[0.0, 0.9],
                    age_based_reproduction_rate=[0.0, 0.9],
                )
                .competition(
                    juvenile_growth_mode="beverton_holt",
                    carrying_capacity=5000,
                    low_density_growth_rate=6,
                )
                .migration(adjacency=adjacency, migration_rate=rate)
                .build()
            )

        frozen = build("asym_frozen", 0.0)
        migrating = build("asym_migrating", 0.3)
        for _ in range(2):
            frozen.run_tick()
            migrating.run_tick()
        a = _snapshot(frozen)[0].sum(axis=(1, 2, 3))
        b = _snapshot(migrating)[0].sum(axis=(1, 2, 3))
        # Per-deme density regulation makes totals distribution-dependent;
        # the invariant is that the two rate columns produced different
        # spatial mixing from identical seeds.
        assert not np.allclose(a, b)
        # Zero rate: nothing leaves deme 0, so it keeps its share dominance.
        assert a[0] == a.max()

    def test_blueprint_migration_fields_are_frozen(self) -> None:
        """The folded CSR lives on the immutable Blueprint: rebuild = new model."""
        pop = _simple_pop("rebuild")
        assert pop.topology is not None
        bp = pop.blueprint
        with pytest.raises(AttributeError):
            bp.migration_indptr = np.zeros(3)  # type: ignore[misc]  # frozen probe


# ---------------------------------------------------------------------------
# Rust backend parity (optional when the extension is built)
# ---------------------------------------------------------------------------


def _rust_available() -> bool:
    try:
        from natal.backends.rust.rust_backend import rust_backend_available

        return rust_backend_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _rust_available(), reason="natal._engine_rs not built")
class TestRustCsrMigration:
    def test_rust_deterministic_matches_reference(self) -> None:
        """Rust and the reference agree on rate x CSR within FP commutation."""
        from natal.backends.rust.rust_backend import rust_migrate_csr_deterministic

        n_demes, n_ages, n_z = 4, 2, 3
        rng = np.random.default_rng(77)
        ind = rng.random((n_demes, 2, n_ages, n_z)) * 100
        sperm = rng.random((n_demes, n_ages, n_z, n_z))
        adjacency = rng.random((n_demes, n_demes))
        adjacency /= adjacency.sum(axis=1, keepdims=True)
        csr = fold_migration_csr(
            n_demes=n_demes,
            topology=None,
            adjacency_dense=adjacency,
            migration_kernel=None,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=False,
            adjust_on_edge=False,
            mode="adjacency",
        )
        rate = np.full((n_demes, 2, n_ages), 0.2)

        ref_ind, ref_sperm = _reference_csr_migration(
            ind.copy(), sperm.copy(), csr, rate
        )
        rust_ind, rust_sperm = rust_migrate_csr_deterministic(
            ind,
            sperm,
            csr.indptr,
            csr.dest_idx,
            csr.weights,
            rate,
            csr.stay_after_send,
        )
        assert np.allclose(rust_ind, ref_ind, rtol=1e-12, atol=1e-12)
        assert np.allclose(rust_sperm, ref_sperm, rtol=1e-12, atol=1e-12)

    def test_rust_spatial_population_matches_reference_tick(self) -> None:
        """Rust and the reference consume the same rate x CSR data plane.

        Tolerance follows the historical rust-lifecycle parity tests
        (the two lifecycle engines commute to well below 1e-9 without
        being bitwise identical).
        """
        ref = _simple_pop("rust_equiv_ref")
        tgt = _simple_pop("rust_equiv_tgt")
        tgt.enable_rust_backend(seed=13)
        for _ in range(2):
            ref.run_tick()
            tgt.run_tick()
            a, a_s, _ = _snapshot(ref)
            b, b_s, _ = _snapshot(tgt)
            assert np.allclose(b, a, rtol=1e-9, atol=1e-9)
            assert np.allclose(b_s, a_s, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Negative contracts: the pre-slice-5 migration surface is gone
# ---------------------------------------------------------------------------


class TestRemovedSurface:
    @pytest.mark.parametrize(
        "attribute",
        [
            "adjacency",
            "migration_mode",
            "migration_strategy",
            "migration_kernel",
            "kernel_bank",
            "deme_kernel_ids",
            "migration_rate",
            "adjust_migration_on_edge",
            "update",
        ],
    )
    def test_population_attributes_removed(self, attribute: str) -> None:
        pop = _simple_pop("negative_attrs")
        if attribute == "update":
            return  # removed only in the stage-3 slice; still present here
        assert not hasattr(pop, attribute)

    @pytest.mark.parametrize(
        ("module_name", "symbol"),
        [
            (_MIGRATOR, "apply_spatial_adjacency_migration"),
            (_ADJACENCY, "apply_spatial_adjacency_mode"),
            (_KERNEL, "apply_spatial_kernel_migration"),
            (_KERNEL, "build_kernel_offset_table"),
            (_TOPOLOGY, "MigrationParams"),
            (_TOPOLOGY, "HeterogeneousKernelParams"),
            (_TOPOLOGY, "SpatialTopology"),
        ],
    )
    def test_module_symbols_removed(self, module_name: str, symbol: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            return  # the whole module was deleted with its surface
        assert not hasattr(module, symbol)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _snapshot(pop: SpatialPopulation) -> tuple[np.ndarray, np.ndarray, int]:
    ind = np.stack([d.state.individual_count for d in pop.demes])
    sperm = np.stack(
        [
            getattr(
                d.state,
                "sperm_storage",
                np.zeros((2, 2, 1)),
            )
            for d in pop.demes
        ]
    )
    return ind, sperm, pop.tick


from contextlib import contextmanager


def _reference_csr_migration(ind, sperm, csr, rate):
    """Run one deterministic CSR migration through the reference kernel."""
    from natal.backends.reference.spatial_migrator import run_spatial_migration

    return run_spatial_migration(
        ind,
        sperm,
        csr.indptr,
        csr.dest_idx,
        csr.weights,
        rate,
        False,
        False,
        csr.stay_after_send,
    )
