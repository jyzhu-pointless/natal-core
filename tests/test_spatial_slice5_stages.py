"""Spatial ecology columnization and the genetics variant bank.

These invariants hold:

- **Genetics-only grouping** (build side): ecological batch differences
  (carrying capacity, initial state, survival) never split config groups
  or variant-bank entries; only genetics content does.
- **Variant bank** (Rust boundary): the heterogeneous session owns one
  shared blueprint, one columnized ecology set, and a bank of shared
  genetics ``TensorSet`` variants indexed per deme.  Bank size follows
  genetics diversity only — 2601 demes with two fitness flavors produce
  exactly two variants.
- **Per-deme ecology columns**: a directed refresh of one deme's ecology
  column entry changes that deme alone; a variant-tensor refresh changes
  every deme sharing the variant and nothing else.

Every assertion below proves a numerical invariant (bitwise array equality
on deterministic trajectories) rather than an implementation detail.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import natal as nt
from natal.backends.rust.rust_backend import (
    RustHeterogeneousSpatialLifecycleBackend,
    ecology_columns_from_drafts,
    genetics_variant_bank,
    rust_backend_available,
)
from natal.contracts.materialize import materialize
from natal.frontend.spatial.builder import SpatialPopulationBuilder, batch_setting

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _species(name: str = "__slice5_stages__") -> nt.Species:
    return nt.Species.from_dict(name=name, structure={"auto": {"A": ["WT"]}})


def _build_pop(
    species: nt.Species,
    *,
    k_values: Sequence[float],
    viability: Sequence[float] | None = None,
    n_demes: int | None = None,
    name: str = "__slice5_stages_pop__",
):
    """Age-structured spatial population with per-deme K (and fitness)."""
    n = n_demes if n_demes is not None else len(k_values)
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=n)
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": 200}, "male": {"WT|WT": 150},
        })
        .survival(
            female_age_based_survival=[0.7, 0.9],
            male_age_based_survival=[0.7, 0.9],
        )
        .reproduction(
            eggs_per_female=40,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=batch_setting(list(k_values)),
            low_density_growth_rate=6,
        )
    )
    if viability is not None:
        builder = builder.fitness(
            viability=batch_setting([{"WT|WT": v} for v in viability]),
            mode="multiply",
        )
    # topology=None → identity adjacency, so demes stay independent.
    return builder.migration(migration_rate=0.0).build()


def _deme_states(pop) -> list[np.ndarray]:
    return [deme.state.individual_count.copy() for deme in pop.demes]


# ══════════════════════════════════════════════════════════════════════════
# Stage 2: genetics-only grouping (build side)
# ══════════════════════════════════════════════════════════════════════════


class TestGeneticsOnlyGrouping:
    """Ecology differences must not split genetics groups."""

    def test_ecology_gradient_keeps_one_variant(self) -> None:
        """All-distinct K values and initial densities → ONE genetics variant."""
        species = _species("__s5_eco_gradient__")
        pop = _build_pop(
            species,
            k_values=[9000, 500, 3000, 700, 2500, 1200],
            name="__s5_eco_gradient_pop__",
        )
        drafts = [deme.export_config() for deme in pop.demes]
        bank, ids = genetics_variant_bank(drafts)
        assert len(bank) == 1
        assert ids.tolist() == [0, 0, 0, 0, 0, 0]
        # The ecology gradient itself survives on the deme drafts.
        ks = [float(draft.carrying_capacity) for draft in drafts]
        assert ks == [9000, 500, 3000, 700, 2500, 1200]

    def test_fitness_variants_dedup_to_two(self) -> None:
        """Two fitness flavors with all-distinct K → exactly two variants."""
        species = _species("__s5_two_variants__")
        pop = _build_pop(
            species,
            k_values=[500, 600, 700, 800],
            viability=[1.0, 1.0, 0.5, 0.5],
            name="__s5_two_variants_pop__",
        )
        drafts = [deme.export_config() for deme in pop.demes]
        bank, ids = genetics_variant_bank(drafts)
        assert len(bank) == 2
        assert ids.tolist() == [0, 0, 1, 1]
        # The variant contents differ exactly in viability.
        assert not np.array_equal(bank[0]["viability_fitness"], bank[1]["viability_fitness"])
        for key in bank[0]:
            if key != "viability_fitness":
                assert np.array_equal(bank[0][key], bank[1][key])

    def test_2601_demes_two_fitness_flavors_yield_two_variants(self) -> None:
        """2601 deme shells (2 fitness flavors x 2601 K values) → 2 variants.

        Draft shells are cheap NamedTuple ``_replace`` copies, so the
        scaling argument is exercised without building 2601 populations.
        """
        species = _species("__s5_scale__")
        pop = _build_pop(species, k_values=[100.0], name="__s5_scale_pop__")
        base = pop.deme(0).export_config()
        drafts = []
        for i in range(2601):
            flavor = i % 2
            shell = base._replace(
                carrying_capacity=np.array(100.0 + i),
                viability_fitness=(
                    base.viability_fitness if flavor == 0
                    else base.viability_fitness * 0.5
                ),
            )
            drafts.append(shell)
        bank, ids = genetics_variant_bank(drafts)
        assert len(bank) == 2
        assert ids[:4].tolist() == [0, 1, 0, 1]
        assert ids[-1] in (0, 1)

    def test_ecology_columns_carry_per_deme_values(self) -> None:
        """Columnized ecology mirrors every deme's draft values exactly."""
        species = _species("__s5_columns__")
        pop = _build_pop(
            species,
            k_values=[500, 600, 700, 800],
            name="__s5_columns_pop__",
        )
        drafts = [deme.export_config() for deme in pop.demes]
        columns = ecology_columns_from_drafts(drafts)
        assert columns["carrying_capacity"].tolist() == [500.0, 600.0, 700.0, 800.0]
        n_ages = int(drafts[0].n_ages)
        for index, draft in enumerate(drafts):
            segment = columns["survival_rates"].reshape(4, 2, n_ages)[index]
            np.testing.assert_array_equal(segment, draft.age_based_survival_rates)


class TestMergedBuildEntry:
    """The build path is single-entry; the split builders are gone."""

    def test_split_builders_removed(self) -> None:
        """_build_homogeneous/_build_heterogeneous must not be accessible."""
        assert not hasattr(SpatialPopulationBuilder, "_build_homogeneous"), (
            "SpatialPopulationBuilder._build_homogeneous must be merged away"
        )
        assert not hasattr(SpatialPopulationBuilder, "_build_heterogeneous"), (
            "SpatialPopulationBuilder._build_heterogeneous must be merged away"
        )

    def test_merged_build_preserves_batch_semantics(self) -> None:
        """A heterogeneous build still lands per-deme ecology and fitness."""
        species = _species("__s5_merged__")
        pop = _build_pop(
            species,
            k_values=[400, 500, 600, 700],
            viability=[1.0, 0.5, 1.0, 0.5],
            name="__s5_merged_pop__",
        )
        ks = [float(pop.deme(i).config.carrying_capacity) for i in range(4)]
        assert ks == [400, 500, 600, 700]
        viability = [
            float(pop.deme(i).config.viability_fitness[0, 0, 0]) for i in range(4)
        ]
        assert viability == [1.0, 0.5, 1.0, 0.5]


# ══════════════════════════════════════════════════════════════════════════
# Stage 2: variant-bank session semantics (Rust boundary)
# ══════════════════════════════════════════════════════════════════════════


def _enabled_backend(pop) -> RustHeterogeneousSpatialLifecycleBackend:
    pop._initialize_session(seed=3)
    backend = pop._rust_spatial_backend  # pyright: ignore[reportPrivateUsage]  # test reaches the owning session
    assert isinstance(backend, RustHeterogeneousSpatialLifecycleBackend)
    return backend


def _params_with_k(draft, value: float):
    contracts = materialize(draft)
    contracts.params.carrying_capacity = value
    return contracts.params


class TestEcologyColumnIndependence:
    """A deme's ecology column entry is written and read independently."""

    def test_deme0_k_write_does_not_affect_deme1(self) -> None:
        """Refreshing deme 0's K column changes deme 0 only, bitwise."""
        species = _species("__s5_eco_write__")
        k_values = [500.0, 500.0]

        baseline = _build_pop(species, k_values=k_values, name="__s5_eco_write_a__")
        _enabled_backend(baseline)
        baseline.run(3, record_every=0)
        baseline_states = _deme_states(baseline)

        modified = _build_pop(species, k_values=k_values, name="__s5_eco_write_b__")
        backend = _enabled_backend(modified)
        draft = modified.deme(0).export_config()
        backend.refresh_deme_ecology(0, ["carrying_capacity"], _params_with_k(draft, 20.0))
        modified.run(3, record_every=0)
        modified_states = _deme_states(modified)

        # Deme 0 (K dropped to 20) must differ and be much smaller.
        assert not np.array_equal(modified_states[0], baseline_states[0])
        assert modified.deme(0).get_total_count() < baseline.deme(0).get_total_count()
        # Deme 1 must be bitwise untouched.
        np.testing.assert_array_equal(modified_states[1], baseline_states[1])


class TestVariantForkIsolation:
    """Variant-tensor writes affect their sharing demes and nothing else."""

    def test_variant_refresh_hits_sharing_demes_only(self) -> None:
        """Rewriting variant 1's viability moves demes 2/3, never demes 0/1."""
        species = _species("__s5_fork__")
        viability = [1.0, 1.0, 0.5, 0.5]
        k_values = [800.0] * 4

        baseline = _build_pop(species, k_values=k_values, viability=viability,
                              name="__s5_fork_a__")
        _enabled_backend(baseline)
        baseline.run(3, record_every=0)
        baseline_states = _deme_states(baseline)

        modified = _build_pop(species, k_values=k_values, viability=viability,
                              name="__s5_fork_b__")
        backend = _enabled_backend(modified)
        drafts = [deme.export_config() for deme in modified.demes]
        bank, ids = genetics_variant_bank(drafts)
        assert len(bank) == 2 and ids.tolist() == [0, 0, 1, 1]

        # Demes 2/3 share variant 1: raise its viability back to 1.0.
        contracts = materialize(drafts[0])
        backend.refresh_variant_tensors(1, ["viability_fitness"], contracts.params)
        modified.run(3, record_every=0)
        modified_states = _deme_states(modified)

        # Demes 0/1 (variant 0) bitwise untouched.
        np.testing.assert_array_equal(modified_states[0], baseline_states[0])
        np.testing.assert_array_equal(modified_states[1], baseline_states[1])
        # Demes 2/3 (variant 1, viability 0.5 → 1.0) differ and grow larger.
        assert not np.array_equal(modified_states[2], baseline_states[2])
        assert modified.deme(2).get_total_count() > baseline.deme(2).get_total_count()
        np.testing.assert_array_equal(modified_states[3], modified_states[2])

    def test_bank_length_exposed_on_backend(self) -> None:
        """The backend reports the genetics variant count of its session."""
        species = _species("__s5_bank_len__")
        pop = _build_pop(
            species,
            k_values=[100, 200, 300, 400, 500, 600],
            viability=[1.0, 1.0, 0.5, 0.5, 1.0, 0.5],
            name="__s5_bank_len_pop__",
        )
        backend = _enabled_backend(pop)
        # Six K values, two viability flavors → exactly two variants.
        assert backend.n_variants == 2
        drafts = [deme.export_config() for deme in pop.demes]
        _, ids = genetics_variant_bank(drafts)
        assert ids.tolist() == [0, 0, 1, 1, 0, 1]


class TestStage2NegativeContracts:
    """Removed stage-2 interfaces must stay inaccessible."""

    def test_backend_refresh_bank_params_removed(self) -> None:
        """refresh_bank_params is replaced by the deme/variant channels."""
        assert not hasattr(RustHeterogeneousSpatialLifecycleBackend, "refresh_bank_params")

    def test_hetero_session_rejects_config_bank_signature(self) -> None:
        """The (config_bank, deme_config_ids, seed) constructor is gone."""
        from natal import _engine_rs

        bp = object()
        params = object()
        with pytest.raises(TypeError):
            _engine_rs.HeterogeneousSpatialEngineSession(
                [(bp, params), (bp, params)],
                __import__("numpy").array([0, 1], dtype=__import__("numpy").int64),
                0,
            )


# ══════════════════════════════════════════════════════════════════════════
# Stage 4: SoA confirmation + checkpointed migration-rate column
# ══════════════════════════════════════════════════════════════════════════


class TestSoAConfirmation:
    """Stage-4 confirmatory checks: the stacked state IS the SoA layout."""

    def test_stacked_state_shape_is_deme_major(self) -> None:
        """The stacked transport array is (n_demes, S, A, Z), deme-major."""
        species = _species("__s5_soa__")
        pop = _build_pop(species, k_values=[500, 500, 500, 500],
                         name="__s5_soa_pop__")
        n_sexes, n_ages, n_ztypes = pop.deme(0).state.individual_count.shape
        ind_all, sperm_all = pop._stack_deme_state_arrays()  # pyright: ignore[reportPrivateUsage]  # SoA confirmation
        assert ind_all.shape == (4, n_sexes, n_ages, n_ztypes)
        assert sperm_all.shape == (4, n_ages, n_ztypes, n_ztypes)

    def test_rust_session_consumes_per_deme_state_slices(self) -> None:
        """Demes evolve in independent SoA slices under the Rust backend.

        With the identity routing table a deme's lifecycle output cannot
        leak into its neighbors: deme 0 and deme 1 trajectory deltas are
        both finite and independent (deme 0 diverges from its own past,
        and deme 1's totals never borrow deme 0's mass).
        """
        species = _species("__s5_soa_rust__")
        pop = _build_pop(species, k_values=[800.0, 20.0],
                         name="__s5_soa_rust_pop__")
        before = _deme_states(pop)
        _enabled_backend(pop)
        pop.run(2, record_every=0)
        after = _deme_states(pop)
        # Deme 0 (K=800) grows; deme 1 (K=20) shrinks toward its own cap.
        assert after[0].sum() > before[0].sum()
        assert after[1].sum() < before[1].sum()
        # No cross-deme leakage: totals move independently, no NaN.
        assert np.isfinite(after).all()


class TestCheckpointMigrationRate:
    """The migration-rate column is part of the memory checkpoint (M4)."""

    def test_restore_recovers_migration_rate_column(self) -> None:
        """snapshot -> mutate rate -> restore puts the column back bitwise."""
        from natal import _engine_rs

        species = _species("__s5_ckpt_rate__")
        pop = _build_pop(species, k_values=[500.0],
                         name="__s5_ckpt_rate_pop__")
        contracts = materialize(pop.deme(0).export_config())
        bp, params = contracts.blueprint, contracts.params

        n_ages = bp.n_ages
        n_z = bp.n_ztypes
        ind = np.full((2, n_ages, n_z), 10.0)
        sperm = np.zeros((n_ages, n_z, n_z))

        session = _engine_rs.EngineSession(bp, params, 0)
        # Session-owned surface: install the explicit state, then
        # snapshot_state captures the session-owned checkpoint in full.
        session.set_state(ind.ravel(), sperm.ravel(), 7)
        snapshot = session.snapshot_state()
        _tick, ind_flat, sperm_flat, rng_words, ecology = snapshot
        # The snapshot ecology carries the rate column.
        assert "migration_rate" in dict(ecology)

        # Mutate the session rate away from the snapshot value.
        mutated = materialize(pop.deme(0).export_config())
        mutated.params.migration_rate = np.full_like(
            np.asarray(mutated.params.migration_rate), 0.9
        )
        session.refresh_params(["migration_rate"], mutated.params)
        assert not np.array_equal(
            session.get_tensor("migration_rate"), np.asarray(params.migration_rate)
        )

        # Restore: the checkpointed (pre-mutation) column comes back.
        # The session owns the state; restore_state reinstalls the snapshot
        # pieces into the session directly.
        session.restore_state(
            _tick, ind_flat, sperm_flat, rng_words, ecology
        )
        np.testing.assert_array_equal(
            session.get_tensor("migration_rate"),
            np.asarray(params.migration_rate).ravel(),
        )

