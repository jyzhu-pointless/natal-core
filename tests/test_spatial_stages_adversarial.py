"""Adversarial tests for slice-5 stages 2-4 (spatial domain closeout).

These tests attack the stage-2/3/4 data plane beyond the coverage of
``test_spatial_slice5_stages.py`` and ``test_spatial_update.py``:

- **Variant-bank fork semantics**: forking appends one bank entry per
  call (non-idempotent by implementation), a forked deme's tensor writes
  reach that deme only, and two demes forked off the same variant evolve
  independently.
- **Columnized read/write channels**: ``Params::from_columns`` adopts
  per-deme segments verbatim (no tiling), mid-deme refreshes land only in
  the targeted column segment (offset attack), and every error path
  leaves the session state untouched.
- **n_demes=1 flattening invariant**: length-1 columns are bit-identical
  to the pre-columnization scalars, and a single-deme heterogeneous
  session reproduces the panmictic Rust trajectory exactly.
- **DemeSlice write channels**: one ``write_ecology`` lands the same
  value in the deme draft, the container column, and the Rust session
  (trajectory parity against a freshly built column model); equilibrium
  metrics re-derive on read after a sensitive write; ``write_genetics``
  detaches the shared draft arrays and swaps the session variant.
- **2601-deme dedup numerics**: the K column carries 2601 independent
  values while the genetics bank holds exactly two variants.
- **Sentinel-column guards**: a multi-deme derive-mode sentinel rejects
  per-deme writes, and an empty-sentinel pull is a verifiable no-op.
- **Checkpoint columns**: a spatial contract (CSR + rate column)
  round-trips migration_rate and the ecology scalars through a memory
  checkpoint bitwise.

Every assertion proves a numerical or identity invariant (bitwise array
equality on deterministic trajectories, exact float equality against a
hand computation, or object-identity detachment); no assertion rests on
an incidental implementation detail.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.backends.rust.rust_backend import (
    RustHeterogeneousSpatialLifecycleBackend,
    ecology_columns_from_drafts,
    genetics_variant_bank,
    rust_backend_available,
)
from natal.contracts.materialize import SpatialMigration, materialize
from natal.contracts.params import Params
from natal.frontend.data.config import ModelDraft
from natal.frontend.spatial.configurator import batch_setting

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _species(name: str) -> nt.Species:
    """Build a one-locus species unique to *name* (no registry clashes)."""
    return nt.Species.from_dict(name=name, structure={"auto": {"A": ["WT"]}})


def _build_pop(
    species: nt.Species,
    *,
    k_values: Sequence[float],
    viability: Sequence[float] | None = None,
    survival_female: Sequence[NDArray[np.float64]] | None = None,
    eggs_values: Sequence[float] | None = None,
    n_demes: int | None = None,
    name: str = "__stages_adversarial_pop__",
) -> nt.SpatialPopulation:
    """Age-structured spatial population with per-deme batch settings.

    Args:
        species: Shared species object.
        k_values: Per-deme carrying capacities (batch_setting).
        viability: Optional per-deme viability fitness flavors.
        survival_female: Optional per-deme female survival vectors.
        eggs_values: Optional per-deme eggs-per-female values.
        n_demes: Deme count override (defaults to ``len(k_values)``).
        name: Population name.

    Returns:
        The built ``SpatialPopulation`` (identity adjacency, rate 0).
    """
    n = n_demes if n_demes is not None else len(k_values)
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=n)
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": 200}, "male": {"WT|WT": 150},
        })
    )
    if survival_female is not None:
        builder = builder.survival(
            female_age_based_survival=batch_setting(
                [np.array(v, dtype=np.float64) for v in survival_female]
            ),
            male_age_based_survival=batch_setting(
                [np.array(v, dtype=np.float64) for v in survival_female]
            ),
        )
    else:
        builder = builder.survival(
            female_age_based_survival=[0.7, 0.9],
            male_age_based_survival=[0.7, 0.9],
        )
    eggs: float | batch_setting[float] = (
        batch_setting(list(eggs_values)) if eggs_values is not None else 40
    )
    builder = builder.reproduction(
        eggs_per_female=eggs,
        sex_ratio=0.5,
        female_age_based_mating_rate=[0.0, 0.9],
        male_age_based_mating_rate=[0.0, 0.9],
        age_based_reproduction_rate=[0.0, 0.9],
    ).competition(
        juvenile_growth_mode="beverton_holt",
        carrying_capacity=batch_setting(list(k_values)),
        low_density_growth_rate=6,
    )
    if viability is not None:
        builder = builder.fitness(
            viability=batch_setting([{"WT|WT": v} for v in viability]),
            mode="multiply",
        )
    # topology=None → identity adjacency, so demes stay independent.
    return builder.migration(migration_rate=0.0).build()


def _states(pop: nt.SpatialPopulation) -> list[NDArray[np.float64]]:
    """Snapshot every deme's individual-count array (fresh copies)."""
    return [deme.state.individual_count.copy() for deme in pop.demes]


def _backend(
    pop: nt.SpatialPopulation, *, seed: int
) -> RustHeterogeneousSpatialLifecycleBackend:
    """Enable the Rust backend and return the owning heterogeneous session."""
    pop.enable_rust_backend(seed=seed)
    backend = pop._rust_spatial_backend  # pyright: ignore[reportPrivateUsage]  # test reaches the owning session
    assert isinstance(backend, RustHeterogeneousSpatialLifecycleBackend)
    return backend


def _params_with_k(draft: ModelDraft, value: float) -> Params:
    """Materialize a Params contract whose carrying capacity is *value*."""
    contracts = materialize(draft)
    contracts.params.carrying_capacity = value
    return contracts.params


def _params_with_viability(draft: ModelDraft, value: float) -> Params:
    """Materialize a Params contract with a uniform viability tensor."""
    contracts = materialize(draft)
    contracts.params.viability_fitness = np.full(
        np.asarray(contracts.params.viability_fitness).shape, value,
        dtype=np.float64,
    )
    return contracts.params


# ══════════════════════════════════════════════════════════════════════════
# 1. Variant-bank fork semantics (primary attack surface)
# ══════════════════════════════════════════════════════════════════════════


class TestVariantForkSemantics:
    """Fork clones the deme's variant, re-points the deme, and nothing else."""

    def test_fork_appends_one_bank_entry_per_call(self) -> None:
        """Each fork call appends exactly one entry (non-idempotent).

        Implementation lock-in: fork_variant has no private-variant check,
        so calling it twice on the same deme clones the clone again.  The
        bank grows by one per call and the returned ids are fresh.
        """
        species = _species("__adv_fork_bank__")
        pop = _build_pop(
            species,
            k_values=[800.0] * 4,
            viability=[1.0, 1.0, 0.5, 0.5],
            name="__adv_fork_bank_pop__",
        )
        backend = _backend(pop, seed=3)
        assert backend.n_variants == 2

        first = backend.fork_variant(1)
        assert first == 2
        assert backend.n_variants == 3

        second = backend.fork_variant(1)
        assert second == 3
        assert backend.n_variants == 4

        third = backend.fork_variant(0)
        assert third == 4
        assert backend.n_variants == 5

    def test_two_forked_demes_are_independent(self) -> None:
        """Two demes forked off one shared variant evolve independently.

        Fork deme 0 into a 0.5-viability variant and deme 1 into a 0.2
        variant; each forked deme must then reproduce, bitwise, the
        trajectory of an independently built model that carries that
        viability in its genetics from the start.
        """
        species = _species("__adv_fork_two__")
        baseline = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_fork_two_base__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_fork_two_mod__")
        ref_half = _build_pop(
            species, k_values=[500.0, 500.0], viability=[0.5, 1.0],
            name="__adv_fork_two_half__")
        ref_low = _build_pop(
            species, k_values=[500.0, 500.0], viability=[1.0, 0.2],
            name="__adv_fork_two_low__")

        backend = _backend(modified, seed=5)
        _backend(baseline, seed=5)
        _backend(ref_half, seed=5)
        _backend(ref_low, seed=5)
        assert backend.n_variants == 1

        variant_a = backend.fork_variant(0)
        variant_b = backend.fork_variant(1)
        assert variant_a != variant_b
        draft = modified.deme(0).export_config()
        backend.refresh_variant_tensors(
            variant_a, ["viability_fitness"], _params_with_viability(draft, 0.5))
        backend.refresh_variant_tensors(
            variant_b, ["viability_fitness"], _params_with_viability(draft, 0.2))

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        ref_half.run(3, record_every=0)
        ref_low.run(3, record_every=0)

        np.testing.assert_array_equal(
            modified.deme(0).state.individual_count,
            ref_half.deme(0).state.individual_count,
            err_msg="forked deme 0 must match an inherent 0.5-viability model",
        )
        np.testing.assert_array_equal(
            modified.deme(1).state.individual_count,
            ref_low.deme(1).state.individual_count,
            err_msg="forked deme 1 must match an inherent 0.2-viability model",
        )
        # Neither forked deme may keep the shared-variant trajectory.
        assert not np.array_equal(
            modified.deme(0).state.individual_count,
            baseline.deme(0).state.individual_count,
        )
        assert not np.array_equal(
            modified.deme(1).state.individual_count,
            baseline.deme(1).state.individual_count,
        )

    def test_forked_variant_write_leaves_sharing_deme_bitwise(self) -> None:
        """A tensor write on the forked variant never reaches the sharer.

        Deme 1 is forked and its new variant gets a 0.1 viability; deme 0
        keeps sharing the original variant and its trajectory must stay
        bitwise identical to the untouched baseline.
        """
        species = _species("__adv_fork_iso__")
        k_values = [800.0, 800.0]
        baseline = _build_pop(
            species, k_values=k_values, name="__adv_fork_iso_base__")
        modified = _build_pop(
            species, k_values=k_values, name="__adv_fork_iso_mod__")

        backend = _backend(modified, seed=2)
        _backend(baseline, seed=2)
        variant_id = backend.fork_variant(1)
        backend.refresh_variant_tensors(
            variant_id, ["viability_fitness"],
            _params_with_viability(modified.deme(1).export_config(), 0.1),
        )

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)

        np.testing.assert_array_equal(
            modified.deme(0).state.individual_count,
            baseline.deme(0).state.individual_count,
            err_msg="deme 0 shares the original variant and must not move",
        )
        assert not np.array_equal(
            modified.deme(1).state.individual_count,
            baseline.deme(1).state.individual_count,
        )
        assert (
            modified.deme(1).get_total_count()
            < baseline.deme(1).get_total_count()
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. Columnized read/write channels
# ══════════════════════════════════════════════════════════════════════════


class TestColumnizedRefreshChannels:
    """from_columns adopts segments verbatim; deme refreshes are isolated."""

    def test_from_columns_adopts_segments_without_tiling(self) -> None:
        """A manually assembled column session equals the built model.

        Feeding the per-deme K column [10, 500, 9000] through
        ``from_columns`` (via the backend constructor) must reproduce,
        deme by deme and bitwise, the trajectory of a population built
        with ``batch_setting`` carrying the same K gradient — proving the
        session consumed true per-deme segments rather than tiling
        deme 0's value.
        """
        species = _species("__adv_from_columns__")
        ref = _build_pop(
            species, k_values=[10.0, 500.0, 9000.0],
            name="__adv_from_columns_ref__")
        ref.enable_rust_backend(seed=6)

        drafts = [ref.deme(i).export_config() for i in range(ref.n_demes)]
        columns = ecology_columns_from_drafts(drafts)
        columns["migration_rate"] = np.asarray(
            ref.params.migration_rate, dtype=np.float64
        ).ravel()
        bank, ids = genetics_variant_bank(drafts)
        # Plan S3: the session owns the stacked state, so the manual
        # build hands the initial stacked arrays and tick over once.
        ind_all, sperm_all = ref._stack_deme_state_arrays()  # pyright: ignore[reportPrivateUsage]  # initial state before any run
        manual = RustHeterogeneousSpatialLifecycleBackend(
            ref.blueprint, columns, bank, ids, ind_all, sperm_all, 0, seed=6,
        )

        for _ in range(3):
            manual.run_tick()
        tick, ind_flat, _sperm_flat = manual.state_snapshot()
        n_ages = int(ref.blueprint.n_ages)
        n_ztypes = int(ref.blueprint.n_ztypes)
        ind = np.asarray(ind_flat, dtype=np.float64).reshape(
            ref.n_demes, 2, n_ages, n_ztypes
        )
        ref.run(3, record_every=0)

        assert tick == 3
        for i in range(3):
            np.testing.assert_array_equal(
                ind[i], ref.deme(i).state.individual_count,
                err_msg=f"deme {i} diverged between column session and build",
            )

    def test_mid_deme_scalar_refresh_hits_only_target(self) -> None:
        """Refreshing deme 1's K column entry leaves demes 0 and 2 alone.

        Offset attack: a mis-computed column offset would spill the write
        into deme 0 or deme 2; the bitwise parities below rule both out.
        """
        species = _species("__adv_mid_scalar__")
        baseline = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_mid_scalar_base__")
        modified = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_mid_scalar_mod__")
        ref = _build_pop(
            species, k_values=[500.0, 10.0, 500.0], name="__adv_mid_scalar_ref__")

        backend = _backend(modified, seed=4)
        _backend(baseline, seed=4)
        _backend(ref, seed=4)
        source = _params_with_k(modified.deme(1).export_config(), 10.0)
        backend.refresh_deme_ecology(1, ["carrying_capacity"], source)

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        ref.run(3, record_every=0)

        np.testing.assert_array_equal(
            modified.deme(1).state.individual_count,
            ref.deme(1).state.individual_count,
            err_msg="mid-deme refresh must match an inherent K=10 build",
        )
        np.testing.assert_array_equal(
            modified.deme(0).state.individual_count,
            baseline.deme(0).state.individual_count,
            err_msg="refresh spilled into deme 0",
        )
        np.testing.assert_array_equal(
            modified.deme(2).state.individual_count,
            baseline.deme(2).state.individual_count,
            err_msg="refresh spilled into deme 2",
        )

    def test_mid_deme_vector_refresh_hits_only_target(self) -> None:
        """Refreshing deme 1's survival segment leaves neighbors alone."""
        species = _species("__adv_mid_vector__")
        survivals = [
            np.array([0.7, 0.9]), np.array([0.3, 0.3]), np.array([0.7, 0.9]),
        ]
        baseline = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_mid_vector_base__")
        modified = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_mid_vector_mod__")
        ref = _build_pop(
            species, k_values=[500.0] * 3, survival_female=survivals,
            name="__adv_mid_vector_ref__")

        backend = _backend(modified, seed=4)
        _backend(baseline, seed=4)
        _backend(ref, seed=4)
        source = materialize(modified.deme(1).export_config()).params
        source.survival_rates = np.full((2, 2), 0.3, dtype=np.float64)
        backend.refresh_deme_ecology(1, ["survival_rates"], source)

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        ref.run(3, record_every=0)

        np.testing.assert_array_equal(
            modified.deme(1).state.individual_count,
            ref.deme(1).state.individual_count,
            err_msg="vector refresh must match an inherent 0.3-survival build",
        )
        np.testing.assert_array_equal(
            modified.deme(0).state.individual_count,
            baseline.deme(0).state.individual_count,
            err_msg="vector refresh spilled into deme 0",
        )
        np.testing.assert_array_equal(
            modified.deme(2).state.individual_count,
            baseline.deme(2).state.individual_count,
            err_msg="vector refresh spilled into deme 2",
        )

    def test_refresh_rejects_out_of_range_deme(self) -> None:
        """A deme index beyond the column count raises ValueError."""
        species = _species("__adv_refresh_oob__")
        pop = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_refresh_oob_pop__")
        backend = _backend(pop, seed=1)
        source = materialize(pop.deme(0).export_config()).params
        with pytest.raises(ValueError, match="out of range"):
            backend.refresh_deme_ecology(2, ["carrying_capacity"], source)

    def test_refresh_rejects_unknown_field(self) -> None:
        """An unknown field name raises KeyError."""
        species = _species("__adv_refresh_unknown__")
        pop = _build_pop(
            species, k_values=[500.0, 500.0],
            name="__adv_refresh_unknown_pop__")
        backend = _backend(pop, seed=1)
        source = materialize(pop.deme(0).export_config()).params
        with pytest.raises(KeyError, match="unknown params field"):
            backend.refresh_deme_ecology(0, ["no_such_field"], source)

    def test_refresh_rejects_genetics_name_through_ecology_channel(
        self,
    ) -> None:
        """Genetics tensors are refused by the ecology channel."""
        species = _species("__adv_refresh_cross__")
        pop = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_refresh_cross_pop__")
        backend = _backend(pop, seed=1)
        source = materialize(pop.deme(0).export_config()).params
        with pytest.raises(KeyError, match="genetics tensor"):
            backend.refresh_deme_ecology(0, ["viability_fitness"], source)

    def test_refresh_is_atomic_on_mixed_fields(self) -> None:
        """A batch with one legal and one unknown field writes nothing.

        The refresh must fail before any write: the legal carrying
        capacity in the same request stays at its old value, so the
        population's trajectory remains bitwise identical to the
        untouched baseline.
        """
        species = _species("__adv_refresh_atomic__")
        baseline = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_refresh_atomic_base__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_refresh_atomic_mod__")

        backend = _backend(modified, seed=8)
        _backend(baseline, seed=8)
        source = _params_with_k(modified.deme(0).export_config(), 10.0)
        with pytest.raises(KeyError):
            backend.refresh_deme_ecology(
                0, ["carrying_capacity", "no_such_field"], source)

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        for i in range(2):
            np.testing.assert_array_equal(
                modified.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
                err_msg=f"failed refresh leaked a partial write into deme {i}",
            )

    def test_from_columns_rejects_unknown_and_misaligned_columns(self) -> None:
        """from_columns guards its boundary: unknown keys, bad lengths."""
        species = _species("__adv_columns_guard__")
        pop = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_columns_guard_pop__")
        pop.enable_rust_backend(seed=1)
        drafts = [pop.deme(i).export_config() for i in range(3)]
        columns = ecology_columns_from_drafts(drafts)
        columns["migration_rate"] = np.asarray(
            pop.params.migration_rate, dtype=np.float64
        ).ravel()
        bank, ids = genetics_variant_bank(drafts)
        # Plan S3 constructor: the stacked state and tick travel with the
        # build handoff; the column guards below must be unchanged.
        ind_all, sperm_all = pop._stack_deme_state_arrays()  # pyright: ignore[reportPrivateUsage]

        unknown = dict(columns)
        unknown["bogus_column"] = np.zeros(3, dtype=np.float64)
        with pytest.raises(KeyError, match="unknown ecology column"):
            RustHeterogeneousSpatialLifecycleBackend(
                pop.blueprint, unknown, bank, ids, ind_all, sperm_all, 0, seed=0)

        short_scalar = dict(columns)
        short_scalar["carrying_capacity"] = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="expected 3 entries"):
            RustHeterogeneousSpatialLifecycleBackend(
                pop.blueprint, short_scalar, bank, ids, ind_all, sperm_all, 0,
                seed=0)

        ragged_vector = dict(columns)
        ragged_vector["survival_rates"] = np.zeros(5, dtype=np.float64)
        with pytest.raises(ValueError, match="not a multiple of 3"):
            RustHeterogeneousSpatialLifecycleBackend(
                pop.blueprint, ragged_vector, bank, ids, ind_all, sperm_all, 0,
                seed=0)


# ══════════════════════════════════════════════════════════════════════════
# 3. n_demes=1 flattening invariant
# ══════════════════════════════════════════════════════════════════════════


class TestSingleDemeFlatteningInvariant:
    """Length-1 columns are bit-identical to the flat pre-columnization."""

    def test_length_one_columns_match_draft_scalars_bitwise(self) -> None:
        """from_python(obj, 1) reproduces every scalar and vector exactly.

        Reading the session back through get_scalar / get_tensor must
        yield exactly the draft's scalar values and the draft vectors'
        flat contents — the numerical statement of "a length-1 column
        is the old scalar".
        """
        from natal import _engine_rs

        species = _species("__adv_flat_read__")
        pop = _build_pop(
            species, k_values=[432.5], name="__adv_flat_read_pop__")
        migration = SpatialMigration(
            indptr=pop.migration_csr.indptr,
            dest_idx=pop.migration_csr.dest_idx,
            weights=pop.migration_csr.weights,
            rate=pop.params.migration_rate,
        )
        contracts = materialize(
            pop._export_reference_draft(), migration,  # pyright: ignore[reportPrivateUsage]  # typed contract source
        )
        draft = pop.deme(0).export_config()
        session = _engine_rs.EngineSession(
            contracts.blueprint, contracts.params, 0,
        )

        assert session.get_scalar("carrying_capacity") == (
            float(draft.carrying_capacity)
        )
        assert session.get_scalar("eggs_per_female") == (
            float(draft.eggs_per_female)
        )
        assert session.get_scalar("growth_mode") == (
            float(draft.juvenile_growth_mode)
        )
        for name, draft_field in (
            ("survival_rates", "age_based_survival_rates"),
            ("mating_rates", "age_based_mating_rates"),
            ("reproduction_rates", "age_based_reproduction_rates"),
            ("fertility", "female_age_based_fertility"),
            ("competition_weights", "age_based_relative_competition_strength"),
        ):
            np.testing.assert_array_equal(
                session.get_tensor(name),
                np.asarray(getattr(draft, draft_field), dtype=np.float64).ravel(),
                err_msg=f"length-1 column {name} diverged from the flat draft",
            )
        np.testing.assert_array_equal(
            session.get_tensor("migration_rate"),
            np.asarray(pop.params.migration_rate, dtype=np.float64).ravel(),
        )

    def test_single_deme_hetero_session_matches_panmictic_rust(self) -> None:
        """One deterministic tick pipeline: hetero == panmictic Rust.

        A single-deme spatial population (from_columns path, per-deme
        assembly at deme 0) and a panmictic population (from_python
        path) built from the same draft with the same seed must produce
        bitwise-identical trajectories — the flattening invariant
        observed end-to-end.
        """
        from natal.frontend.configurator import Configurator

        species = _species("__adv_flat_parity__")

        def _panmictic() -> nt.AgeStructuredPopulation:
            return (
                Configurator.from_species(species)
                .setup(name="__adv_flat_parity_pan__", stochastic=False)
                .age_structure(n_ages=2, new_adult_age=1)
                .initial_state(individual_count={
                    "female": {"WT|WT": 200}, "male": {"WT|WT": 150},
                })
                .survival(
                    female_age_based_survival=[0.7, 0.9],
                    male_age_based_survival=[0.7, 0.9],
                )
                .reproduction(
                    eggs_per_female=40, sex_ratio=0.5,
                    female_age_based_mating_rate=[0.0, 0.9],
                    male_age_based_mating_rate=[0.0, 0.9],
                    age_based_reproduction_rate=[0.0, 0.9],
                )
                .competition(
                    juvenile_growth_mode="beverton_holt",
                    carrying_capacity=500.0,
                    low_density_growth_rate=6,
                )
                .build()
            )

        panmictic = _panmictic().enable_rust_backend(seed=9)
        spatial = _build_pop(
            species, k_values=[500.0], n_demes=1,
            name="__adv_flat_parity_sp__").enable_rust_backend(seed=9)

        panmictic.run(5)
        spatial.run(5, record_every=0)

        np.testing.assert_array_equal(
            spatial.deme(0).state.individual_count,
            panmictic.state.individual_count,
            err_msg=(
                "single-deme column session must reproduce the panmictic "
                "Rust trajectory bitwise"
            ),
        )


# ══════════════════════════════════════════════════════════════════════════
# 4. DemeSlice write-channel consistency
# ══════════════════════════════════════════════════════════════════════════


class TestDemeSliceWriteChannels:
    """One write lands in the draft, the column, and the Rust session."""

    def test_write_ecology_lands_in_draft_column_and_session(self) -> None:
        """write_ecology synchronizes all three storage sites.

        (a) the deme draft carries the new K, (b) the container's ecology
        column carries it at the deme's slot and only there, and (c) the
        Rust session consumes it — proven by bitwise trajectory parity
        with a model built from scratch with the new K column.
        """
        species = _species("__adv_write3__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_write3_mod__")
        ref = _build_pop(
            species, k_values=[100.0, 500.0], name="__adv_write3_ref__")

        _backend(modified, seed=7)
        _backend(ref, seed=7)
        modified.deme(0).write_ecology("carrying_capacity", 100.0)

        # (a) draft
        assert float(modified.deme(0).config.carrying_capacity) == 100.0
        # (b) container column
        assert modified.params.carrying_capacity.tolist() == [100.0, 500.0]
        # (c) session: trajectory parity against the freshly built model
        modified.run(4, record_every=0)
        ref.run(4, record_every=0)
        for i in range(2):
            np.testing.assert_array_equal(
                modified.deme(i).state.individual_count,
                ref.deme(i).state.individual_count,
                err_msg=f"session did not consume the written K at deme {i}",
            )

    def test_write_genetics_detaches_shared_arrays_and_forks_bank(self) -> None:
        """A genetics write forks the bank and detaches the draft arrays.

        Identity invariants: the previously shared array object keeps its
        contents bitwise (the write cannot penetrate the sharing demes),
        the writing deme's array is a different object, the other demes
        still reference the original, and the session bank grew by one.
        """
        species = _species("__adv_detach__")
        pop = _build_pop(
            species, k_values=[500.0] * 3, name="__adv_detach_pop__")
        backend = _backend(pop, seed=3)
        assert backend.n_variants == 1

        shared = pop.deme(0).config.viability_fitness
        assert pop.deme(1).config.viability_fitness is shared
        shared_copy = shared.copy()

        pop.deme(0).write_genetics(
            "viability_fitness", np.full_like(shared, 0.5))

        np.testing.assert_array_equal(
            shared, shared_copy,
            err_msg="the shared array's contents changed in place",
        )
        assert pop.deme(0).config.viability_fitness is not shared
        assert float(pop.deme(0).config.viability_fitness[0, 0, 0]) == 0.5
        for i in (1, 2):
            assert pop.deme(i).config.viability_fitness is shared
        assert backend.n_variants == 2

    def test_write_genetics_swaps_session_variant_used_by_run(self) -> None:
        """The forked variant is the one the session actually runs.

        After write_genetics, the deme's trajectory must match a model
        whose genetics carried the new viability from the start, and the
        untouched deme must keep the baseline trajectory.
        """
        species = _species("__adv_swap__")
        baseline = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_swap_base__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_swap_mod__")
        ref = _build_pop(
            species, k_values=[500.0, 500.0], viability=[0.5, 1.0],
            name="__adv_swap_ref__")

        _backend(modified, seed=6)
        _backend(baseline, seed=6)
        _backend(ref, seed=6)
        modified.deme(0).write_genetics(
            "viability_fitness",
            np.full_like(modified.deme(0).config.viability_fitness, 0.5),
        )

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        ref.run(3, record_every=0)

        np.testing.assert_array_equal(
            modified.deme(0).state.individual_count,
            ref.deme(0).state.individual_count,
            err_msg="the forked variant is not what the session consumed",
        )
        np.testing.assert_array_equal(
            modified.deme(1).state.individual_count,
            baseline.deme(1).state.individual_count,
            err_msg="the genetics write leaked into the sharing deme",
        )

    def test_write_ecology_then_run_matches_multi_field_build(self) -> None:
        """Consecutive writes of K and eggs match a fresh column build.

        Two staged writes (K for deme 0, eggs for deme 1) followed by a
        run must equal, bitwise, a model built directly with both column
        values — the dirty-bridge semantics: writes take effect on the
        next run without rebuilding the session.
        """
        species = _species("__adv_dirty_bridge__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], eggs_values=[40.0, 40.0],
            name="__adv_dirty_bridge_mod__")
        ref = _build_pop(
            species, k_values=[100.0, 500.0], eggs_values=[40.0, 15.0],
            name="__adv_dirty_bridge_ref__")

        _backend(modified, seed=11)
        _backend(ref, seed=11)
        modified.deme(0).write_ecology("carrying_capacity", 100.0)
        modified.deme(1).write_ecology("eggs_per_female", 15.0)

        modified.run(4, record_every=0)
        ref.run(4, record_every=0)
        for i in range(2):
            np.testing.assert_array_equal(
                modified.deme(i).state.individual_count,
                ref.deme(i).state.individual_count,
                err_msg=f"staged writes did not reach the session at deme {i}",
            )


# ══════════════════════════════════════════════════════════════════════════
# 5. 2601-scale dedup numerics
# ══════════════════════════════════════════════════════════════════════════


class TestLargeScaleColumnDedup:
    """Bank size follows genetics diversity; columns stay per-deme."""

    def test_2601_deme_columns_carry_independent_k_values(self) -> None:
        """2 fitness flavors x 2601 K values → bank 2, K column exact.

        The K column must equal ``100 + i`` for every one of the 2601
        demes (attacked at the extremes and the middle), sampled survival
        segments must stay at the shared base values, and the variant ids
        must alternate with the fitness flavor.
        """
        species = _species("__adv_scale2601__")
        pop = _build_pop(
            species, k_values=[100.0], name="__adv_scale2601_pop__")
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

        columns = ecology_columns_from_drafts(drafts)
        assert columns["carrying_capacity"].shape == (2601,)
        np.testing.assert_array_equal(
            columns["carrying_capacity"], np.arange(100.0, 2701.0),
            err_msg="the K column must carry 2601 distinct per-deme values",
        )
        assert columns["growth_mode"].dtype == np.int64
        assert columns["growth_mode"].shape == (2601,)

        n_ages = int(base.n_ages)
        survival = columns["survival_rates"].reshape(2601, 2, n_ages)
        for i in (0, 1, 1234, 2600):
            np.testing.assert_array_equal(
                survival[i], base.age_based_survival_rates,
                err_msg=f"deme {i}'s survival segment diverged from the base",
            )

        bank, ids = genetics_variant_bank(drafts)
        assert len(bank) == 2
        assert int(ids[1234]) == 1234 % 2
        assert int(ids[2600]) == 2600 % 2
        assert int(ids[1]) == 1


# ══════════════════════════════════════════════════════════════════════════
# 6. Sentinel-column guards (equilibrium_distribution)
# ══════════════════════════════════════════════════════════════════════════


class TestSentinelColumnGuards:
    """Multi-deme sentinel columns cannot be written one deme at a time."""

    def test_multi_deme_sentinel_write_rejected_without_mutation(self) -> None:
        """Declaring a distribution at one deme of a derive-mode model.

        A derive-mode (empty sentinel) multi-deme session must reject a
        per-deme equilibrium_distribution write with ValueError, and the
        rejection must leave the session untouched — the trajectory
        stays bitwise identical to the untouched baseline.
        """
        species = _species("__adv_sentinel_reject__")
        baseline = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_sentinel_base__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_sentinel_mod__")

        backend = _backend(modified, seed=2)
        _backend(baseline, seed=2)
        source = materialize(modified.deme(0).export_config()).params
        source.equilibrium_distribution = np.array(
            [[10.0, 20.0], [10.0, 20.0]], dtype=np.float64
        )
        with pytest.raises(ValueError, match="cannot write one deme"):
            backend.refresh_deme_ecology(
                0, ["equilibrium_distribution"], source)

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        for i in range(2):
            np.testing.assert_array_equal(
                modified.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
                err_msg=f"rejected sentinel write leaked into deme {i}",
            )

    def test_empty_sentinel_pull_is_noop(self) -> None:
        """Pulling the empty sentinel changes nothing and succeeds.

        A derive-mode source (empty equilibrium_distribution) pulled
        into a derive-mode session is a legal no-op: the call succeeds
        and the trajectory stays bitwise at the baseline.
        """
        species = _species("__adv_sentinel_noop__")
        baseline = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_noop_base__")
        modified = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_noop_mod__")

        backend = _backend(modified, seed=2)
        _backend(baseline, seed=2)
        source = materialize(modified.deme(0).export_config()).params
        assert np.asarray(source.equilibrium_distribution).size == 0
        backend.refresh_deme_ecology(0, ["equilibrium_distribution"], source)

        baseline.run(3, record_every=0)
        modified.run(3, record_every=0)
        for i in range(2):
            np.testing.assert_array_equal(
                modified.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
                err_msg=f"empty-sentinel pull changed deme {i}",
            )


# ══════════════════════════════════════════════════════════════════════════
# 7. Checkpoint columns (spatial contract)
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialCheckpointColumns:
    """A spatial contract round-trips its rate column through a checkpoint."""

    def test_spatial_contract_checkpoint_restores_rate_and_scalar(self) -> None:
        """snapshot → mutate → restore returns rate column and K bitwise.

        Unlike the panmictic materialization used by the stages suite,
        this contract carries a real spatial CSR and an (n_demes, S, A)
        rate column; both the full rate column and the mutated scalar
        must come back bit-identical from the memory checkpoint.
        """
        from natal import _engine_rs

        species = _species("__adv_ckpt_spatial__")
        pop = _build_pop(
            species, k_values=[500.0], name="__adv_ckpt_spatial_pop__")
        migration = SpatialMigration(
            indptr=pop.migration_csr.indptr,
            dest_idx=pop.migration_csr.dest_idx,
            weights=pop.migration_csr.weights,
            rate=pop.params.migration_rate,
        )
        contracts = materialize(
            pop._export_reference_draft(), migration,  # pyright: ignore[reportPrivateUsage]  # typed contract source
        )
        bp, params = contracts.blueprint, contracts.params
        rate_before = np.asarray(params.migration_rate, dtype=np.float64)

        n_ages, n_z = bp.n_ages, bp.n_ztypes
        ind = np.full((2, n_ages, n_z), 10.0)
        sperm = np.zeros((n_ages, n_z, n_z))
        session = _engine_rs.EngineSession(bp, params, 0)
        # Session-owned surface (plan S2): install the explicit state, then
        # snapshot_state captures the session-owned checkpoint in full.
        session.set_state(ind.ravel(), sperm.ravel(), 7)
        snapshot = session.snapshot_state()
        _tick, ind_flat, sperm_flat, rng_words, ecology = snapshot
        assert np.array_equal(
            np.asarray(dict(ecology)["migration_rate"]), rate_before.ravel()
        )

        mutated = materialize(
            pop._export_reference_draft(),  # pyright: ignore[reportPrivateUsage]  # typed contract source
            SpatialMigration(
                indptr=pop.migration_csr.indptr,
                dest_idx=pop.migration_csr.dest_idx,
                weights=pop.migration_csr.weights,
                rate=np.full_like(rate_before, 0.9),
            ),
        )
        mutated.params.carrying_capacity = 321.0
        session.refresh_params(
            ["migration_rate", "carrying_capacity"], mutated.params)
        assert session.get_scalar("carrying_capacity") == 321.0
        assert not np.array_equal(
            session.get_tensor("migration_rate"), rate_before.ravel()
        )

        # The session owns the state; restore_state reinstalls the snapshot
        # pieces into the session directly.
        session.restore_state(
            _tick, ind_flat, sperm_flat, rng_words, ecology)
        assert session.get_scalar("carrying_capacity") == 500.0
        np.testing.assert_array_equal(
            session.get_tensor("migration_rate"), rate_before.ravel(),
            err_msg="checkpoint must restore the full spatial rate column",
        )


# ══════════════════════════════════════════════════════════════════════════
# 8. Removed-surface spot check (negative contracts)
# ══════════════════════════════════════════════════════════════════════════


class TestRemovedSurfaceSpotCheck:
    """Spot-check of the deleted stage-2/3 interfaces (full suites exist)."""

    def test_removed_interfaces_stay_inaccessible(self) -> None:
        """Every interface deleted by stages 2-4 stays deleted.

        The per-file suites (test_spatial_update.py, slice5_stages) own
        the detailed contracts; this spot check pins them jointly in one
        place so no removal regresses silently.
        """
        import natal.frontend.spatial.population as spatial_population
        from natal.frontend.spatial.configurator import SpatialConfigurator

        species = _species("__adv_removed__")
        pop = _build_pop(
            species, k_values=[500.0, 500.0], name="__adv_removed_pop__")

        assert not hasattr(pop, "update")
        assert not hasattr(pop, "update_deme")
        assert not hasattr(spatial_population, "_SpatialUpdate")
        assert not hasattr(spatial_population, "_DETACH_FIELDS")
        assert not hasattr(SpatialConfigurator, "_build_homogeneous")
        assert not hasattr(SpatialConfigurator, "_build_heterogeneous")
        assert not hasattr(
            RustHeterogeneousSpatialLifecycleBackend, "refresh_bank_params"
        )
