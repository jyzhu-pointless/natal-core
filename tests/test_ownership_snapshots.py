"""Ownership snapshot contracts (plan 13.1 R4/R5, fixed in S2 batch 22).

Promoted from the S0 red-light repros ``repro_r4`` / ``repro_r5``:

- **R4**: every ndarray field of a materialized ``Blueprint`` is frozen
  read-only — the frozen-discipline promise in
  ``natal.contracts.blueprint`` becomes mechanically enforced, so no
  external holder can mutate the engine's frozen model arrays in place.
- **R5**: ``pop.state`` returns point-in-time snapshots; writing through
  the returned container can never reach the engine's live arrays.  The
  in-hook writable loan stays a separate controlled channel
  (``TickContext.state``).

Every assertion pins a numerical or structural invariant, not liveness.
"""

from __future__ import annotations

import gc
import tracemalloc
import weakref

import numpy as np
import pytest

import natal as nt
from natal.contracts.blueprint import Blueprint
from natal.contracts.materialize import SpatialMigration, materialize
from natal.frontend.spatial.migration import MigrationCSR
from natal.frontend.spatial.population import (
    _minimal_contract,  # pyright: ignore[reportPrivateUsage]  # frozen-discipline target under test
)

_BLUEPRINT_ARRAY_FIELDS = (
    "adult_ages",
    "female_only_by_sex_chrom",
    "male_only_by_sex_chrom",
    "initial_individual_count",
    "initial_sperm_storage",
    "migration_indptr",
    "migration_dest_idx",
    "migration_weights",
)

# Draft fields that feed the corresponding Blueprint arrays (the migration
# CSR comes from a SpatialMigration payload instead of the draft).
_DRAFT_BACKED_FIELDS = (
    "adult_ages",
    "female_only_by_sex_chrom",
    "male_only_by_sex_chrom",
    "initial_individual_count",
    "initial_sperm_storage",
)


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species for ownership samples."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_discrete(name: str) -> nt.DiscreteGenerationPopulation:
    """Return a deterministic discrete population with a known start."""
    return (
        nt.DiscreteGenerationPopulation.setup(species=_species(name), stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )


def _build_age(name: str) -> nt.AgeStructuredPopulation:
    """Return a deterministic age-structured population with sperm storage."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 0.0]},
                "male": {"WT|WT": [0.0, 20.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 0.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )


# ══════════════════════════════════════════════════════════════════════════
# R4: frozen Blueprint arrays
# ══════════════════════════════════════════════════════════════════════════


class TestBlueprintFrozenArrays:
    """Every ndarray a Blueprint exposes is read-only."""

    def test_all_eight_fields_readonly(self) -> None:
        """The eight blueprint ndarray fields reject in-place writes."""
        bp = materialize(_build_discrete("R4FrozenDiscrete").config).blueprint
        writable = [
            field
            for field in _BLUEPRINT_ARRAY_FIELDS
            if isinstance(getattr(bp, field), np.ndarray)
            and getattr(bp, field).flags.writeable
        ]
        assert writable == [], (
            f"Blueprint ndarray fields are writable: {writable} — external "
            "holders can mutate the engine's frozen model arrays in place"
        )

    def test_write_attempt_raises_and_leaves_draft_intact(self) -> None:
        """An in-place write raises ValueError and cannot corrupt the draft."""
        pop = _build_discrete("R4DraftGuard")
        cfg = pop.config
        draft_sentinel = float(cfg.initial_individual_count.sum())
        contract_arr = materialize(cfg).blueprint.initial_individual_count

        with pytest.raises(ValueError, match="read-only"):
            contract_arr[...] = -1.0
        assert float(cfg.initial_individual_count.sum()) == draft_sentinel

    def test_class_level_default_empties_are_frozen(self) -> None:
        """The shared panmictic default CSR arrays are read-only too."""
        bp = Blueprint(
            n_sexes=2,
            n_ages=2,
            n_ztypes=0,
            n_gtypes=0,
            n_glabs=0,
            new_adult_age=1,
            adult_ages=np.array([1], dtype=np.int64),
            stochastic=False,
            continuous_sampling=False,
            fixed_egg_count=False,
            has_sex_chromosomes=False,
            extreme_speed_mode=0,
            ztype_names=(),
            gtype_names=(),
            female_only_by_sex_chrom=np.zeros(0, dtype=np.bool_),
            male_only_by_sex_chrom=np.zeros(0, dtype=np.bool_),
            initial_individual_count=np.zeros((0,)),
            initial_sperm_storage=np.zeros((0,)),
        )
        for field in ("migration_indptr", "migration_dest_idx", "migration_weights"):
            assert not getattr(bp, field).flags.writeable, (
                f"shared class-level default {field} must be frozen: a write "
                "would leak across every default-constructed blueprint"
            )

    def test_frozen_blueprint_still_feeds_the_rust_session(self) -> None:
        """A frozen blueprint crosses the PyO3 boundary and runs a tick."""
        pop = _build_discrete("R4RustFeed")
        pop.enable_rust_backend(seed=7)
        assert pop.using_rust_backend
        pop.run(1)
        # Deterministic model: 10 females x 2 eggs x 0.5 sex ratio -> 10
        # juveniles of each sex replacing the adults; adults all survive.
        expected = float(pop.state.individual_count.sum())
        assert expected > 0.0
        assert float(pop.state.individual_count.sum()) == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════════════
# R5: pop.state returns snapshots
# ══════════════════════════════════════════════════════════════════════════


class TestStateSnapshotDiscipline:
    """Long-lived callers get copies; only the hook loan is writable."""

    def test_repro_r5_write_cannot_reach_engine(self) -> None:
        """Writing through the returned state cannot mutate the engine."""
        pop = _build_discrete("R5SnapshotDiscrete")
        before = pop.state.individual_count.copy()
        pop.state.individual_count[0, 1, 0] = 999.0

        assert np.array_equal(pop.state.individual_count, before), (
            "R5: writing through the returned state container mutated the "
            "engine's live arrays"
        )
        assert np.array_equal(pop._state.individual_count, before)  # pyright: ignore[reportPrivateUsage]  # engine truth

    def test_age_model_snapshot_includes_independent_sperm_copy(self) -> None:
        """Age snapshots copy both arrays; sperm writes stay detached."""
        pop = _build_age("R5SnapshotAge")
        snapshot = pop.state
        sperm_sentinel = float(snapshot.sperm_storage.sum())
        snapshot.individual_count[0, 1, 0] = 123.0
        snapshot.sperm_storage[1, 0, 0] = 456.0

        assert float(pop._state.individual_count.sum()) == pytest.approx(40.0)  # pyright: ignore[reportPrivateUsage]
        assert float(pop._state.sperm_storage.sum()) == sperm_sentinel  # pyright: ignore[reportPrivateUsage]

    def test_consecutive_snapshots_are_independent(self) -> None:
        """Two property reads yield independent containers."""
        pop = _build_discrete("R5Independent")
        first = pop.state
        second = pop.state
        assert first is not second
        first.individual_count[0, 1, 0] = 5.0
        assert float(second.individual_count[0, 1, 0]) == 10.0
        assert float(pop._state.individual_count[0, 1, 0]) == 10.0  # pyright: ignore[reportPrivateUsage]

    def test_snapshot_tick_tracks_the_engine(self) -> None:
        """The snapshot's tick equals the engine tick after each run."""
        pop = _build_discrete("R5Tick")
        assert pop.state.n_tick == 0
        pop.run(2)
        assert pop.state.n_tick == 2
        assert pop.state.n_tick == pop.tick

    def test_hook_state_loan_stays_writable(self) -> None:
        """The TickContext loan still writes the live arrays (R5's flip side)."""
        seen: dict[str, float] = {}

        @nt.hook(event="first")
        def boost(ctx: nt.TickContext) -> int:
            seen["before"] = float(ctx.state.individual_count[1, 1, 0])
            ctx.state.individual_count[1, 1, 0] += 7.0
            seen["after"] = float(ctx.state.individual_count[1, 1, 0])
            return 0

        pop = _build_discrete("R5Loan")
        pop.register_hooks(boost, event="first")
        pop.run(1)
        # The loan wrote the live arrays: the hook observed the change and
        # the engine carried the mutated adult count into age 1.
        assert seen["before"] == 10.0
        assert seen["after"] == 17.0

    def test_export_state_round_trip_still_works(self) -> None:
        """export/import operate on values, unaffected by the snapshot face."""
        pop = _build_discrete("R5RoundTrip")
        initial_counts = pop.state.individual_count.copy()
        flat = pop.export_state()
        pop.run(1)
        assert pop.tick == 1
        pop.import_state(flat)
        assert pop.tick == 0
        np.testing.assert_array_equal(pop.state.individual_count, initial_counts)


# ══════════════════════════════════════════════════════════════════════════
# Adversarial additions (batch 22b): freeze-bypass attacks, snapshot channel
# attacks, the _live_state contract, lifecycle transitions under the
# snapshot discipline, and frozen-write error paths.  Every test targets a
# concrete way R4/R5 could regress silently.
# ══════════════════════════════════════════════════════════════════════════


def _build_age_with_species(
    species: nt.Species, name: str, *, n_ages: int = 3
) -> nt.AgeStructuredPopulation:
    """Return a deterministic age population bound to a caller-owned species.

    ``_build_age`` mints its own species, which is unusable for spatial
    containers (all demes must share one Species object); this variant is
    the shared-species twin used by the spatial channel tests.

    Args:
        species: Species shared by every deme of the container.
        name: Population identifier.
        n_ages: Number of age classes (10 grows the snapshot payload for
            the memory-accumulation attack).

    Returns:
        A built deterministic age-structured population.
    """
    return (
        nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=n_ages, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0] + [0.0] * (n_ages - 2)},
                "male": {"WT|WT": [0.0, 20.0] + [0.0] * (n_ages - 2)},
            }
        )
        .survival(
            female_age_based_survival=[1.0] * (n_ages - 1) + [0.0],
            male_age_based_survival=[1.0] * (n_ages - 1) + [0.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )


def _build_discrete_raw(name: str) -> nt.DiscreteGenerationPopulation:
    """Return a deterministic discrete population with raw-mode history.

    Args:
        name: Population identifier.

    Returns:
        A built population whose History schema records full raw state
        rows, enabling ``restore_checkpoint`` round trips.
    """
    return (
        nt.DiscreteGenerationPopulation.setup(species=_species(name), stochastic=False)
        .initial_state(individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .record_history(mode="raw")
        .build()
    )


def _build_beverton(name: str, carrying_capacity: float) -> nt.DiscreteGenerationPopulation:
    """Return a discrete population whose capacity actually binds growth.

    ``eggs_per_female=6`` under Beverton-Holt dynamics pushes juveniles
    against the carrying capacity, so changing ``carrying_capacity``
    changes the deterministic trajectory — a change-detector assertion on
    the refresh path would otherwise pass vacuously.

    Args:
        name: Population identifier.
        carrying_capacity: Competition capacity written at build time.

    Returns:
        A built deterministic population with capacity-sensitive dynamics.
    """
    return (
        nt.DiscreteGenerationPopulation.setup(species=_species(name), stochastic=False)
        .initial_state(individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=2.0,
            juvenile_growth_mode="beverton_holt",
        )
        .build()
    )


def _defaulted_blueprint() -> Blueprint:
    """Return a Blueprint relying on the class-level CSR defaults.

    Builds with only the required fields so the three migration fields
    fall back to the shared class-level default empties.

    Returns:
        A minimal Blueprint whose CSR fields are the shared defaults.
    """
    return Blueprint(
        n_sexes=2,
        n_ages=2,
        n_ztypes=0,
        n_gtypes=0,
        n_glabs=0,
        new_adult_age=1,
        adult_ages=np.array([1], dtype=np.int64),
        stochastic=False,
        continuous_sampling=False,
        fixed_egg_count=False,
        has_sex_chromosomes=False,
        extreme_speed_mode=0,
        ztype_names=(),
        gtype_names=(),
        female_only_by_sex_chrom=np.zeros(0, dtype=np.bool_),
        male_only_by_sex_chrom=np.zeros(0, dtype=np.bool_),
        initial_individual_count=np.zeros((0,)),
        initial_sperm_storage=np.zeros((0,)),
    )


def _minimal_migration() -> MigrationCSR:
    """Return a two-deme symmetric migration CSR for spatial contracts.

    Returns:
        A CSR with one outbound edge per deme (0 -> 1 and 1 -> 0).
    """
    return MigrationCSR(
        indptr=np.array([0, 1, 2], dtype=np.int64),
        dest_idx=np.array([1, 0], dtype=np.int64),
        weights=np.array([0.1, 0.1]),
        stay_after_send=False,
    )


class TestR4FreezeBypassAttacks:
    """Attack the frozen-discipline enforcement through non-obvious channels.

    ``setflags(write=False)`` blocks ``arr[...] = x`` directly, but numpy
    offers other in-place channels (``resize``, ufunc ``out=``) and the
    freeze could also have been implemented by freezing the *draft* in
    place instead of copying — each test here targets one of those
    specific wrong implementations.
    """

    def test_resize_and_ufunc_out_channels_fail_and_leave_everything_intact(
        self,
    ) -> None:
        """resize/ufunc-out writes fail and corrupt neither draft nor contract.

        Attack vector: numpy 2.x does not consult ``writeable`` for
        ``resize`` on an unreferenced array, and ``np.add(..., out=arr)``
        is a separate code path from item assignment.  A freeze helper
        that only guards ``__setitem__`` would let both through.
        """
        pop = _build_discrete("R4BypassChannels")
        cfg = pop.config
        contract = materialize(cfg).blueprint
        draft_before = cfg.initial_individual_count.copy()
        contract_before = contract.initial_individual_count.copy()

        for arr in (contract.initial_individual_count, contract.adult_ages):
            # resize raises ValueError while the Blueprint holds a
            # reference (refcheck): the shape cannot be mutated either.
            with pytest.raises(ValueError):
                arr.resize(arr.size * 2)
        # float64 and int64 fields both reject ufunc out= writes.
        with pytest.raises(ValueError, match="read-only"):
            np.add(contract.initial_individual_count, 1.0, out=contract.initial_individual_count)
        with pytest.raises(ValueError, match="read-only"):
            np.add(contract.adult_ages, 1, out=contract.adult_ages)
        # fill() is yet another in-place channel behind the same guard.
        with pytest.raises(ValueError, match="read-only"):
            contract.initial_individual_count.fill(-1.0)

        np.testing.assert_array_equal(cfg.initial_individual_count, draft_before)
        np.testing.assert_array_equal(contract.initial_individual_count, contract_before)
        assert contract.adult_ages.tolist() == [1]

    def test_two_materializations_independent_and_draft_stays_writable(
        self,
    ) -> None:
        """Freezing the contract never freezes the source draft.

        Attack vector: a lazy freeze that calls ``frozen(draft.arr)``
        without copying would leave the draft read-only and both
        materializations sharing one buffer — breaking later rebuilds and
        cross-contaminating two populations materialized from one draft.
        """
        pop = _build_age("R4DraftFreedom")
        cfg = pop.config
        first = materialize(cfg).blueprint
        second = materialize(cfg).blueprint

        for field in _BLUEPRINT_ARRAY_FIELDS:
            assert not np.may_share_memory(
                getattr(first, field), getattr(second, field)
            ), f"two materializations share the {field} buffer"

        for field in _DRAFT_BACKED_FIELDS:
            assert getattr(cfg, field).flags.writeable, (
                f"draft field {field} became read-only after materialize; "
                "the freeze must apply to copies, never to the source"
            )

        # A draft edit after materialize must not leak into either
        # contract (freshly owned buffers), then the draft is restored.
        saved = cfg.initial_individual_count.copy()
        try:
            cfg.initial_individual_count[...] += 100.0
            np.testing.assert_array_equal(
                first.initial_individual_count, saved
            )
            np.testing.assert_array_equal(
                second.initial_individual_count, saved
            )
        finally:
            cfg.initial_individual_count[...] = saved

    def test_contract_arrays_are_not_views_of_the_draft_buffers(self) -> None:
        """No blueprint array shares memory with its draft source.

        Attack vector: ``frozen(np.asarray(x))`` on an already-cast array
        is a no-op view — the "frozen" array would alias the writable
        draft and the freeze would be load-bearing nothing.
        """
        pop = _build_age("R4NoViews")
        cfg = pop.config
        bp = materialize(cfg).blueprint
        for field in _DRAFT_BACKED_FIELDS:
            assert not np.may_share_memory(
                getattr(bp, field), getattr(cfg, field)
            ), f"blueprint {field} aliases the draft buffer"

    def test_spatial_csr_contract_is_independent_of_caller_arrays(self) -> None:
        """The folded CSR copies the caller's arrays; later edits cannot leak.

        Attack vector: folding that stores the caller's indptr/weights by
        reference would let the caller rewrite migration routing of a
        live contract after the fact.
        """
        pop = _build_discrete("R4CsrIndependence")
        indptr = np.array([0, 1, 2], dtype=np.int64)
        dest_idx = np.array([1, 0], dtype=np.int64)
        weights = np.array([0.1, 0.1])
        migration = SpatialMigration(
            indptr=indptr,
            dest_idx=dest_idx,
            weights=weights,
            rate=np.full((2, 2, 2), 0.05),
        )
        bp = materialize(pop.config, migration).blueprint

        assert not np.may_share_memory(bp.migration_indptr, indptr)
        assert not np.may_share_memory(bp.migration_dest_idx, dest_idx)
        assert not np.may_share_memory(bp.migration_weights, weights)

        indptr[0] = 42
        weights[0] = 0.9
        assert bp.migration_indptr.tolist() == [0, 1, 2]
        assert bp.migration_weights.tolist() == pytest.approx([0.1, 0.1])

    def test_spatial_test_double_blueprint_is_frozen(self) -> None:
        """The lightweight spatial double's blueprint arrays reject writes.

        Attack vector: ``_minimal_contract`` builds a Blueprint outside
        ``materialize``; a regression could skip ``frozen`` there and the
        test-double demes would hand writable engine arrays to fixtures.
        """
        bp, _params = _minimal_contract(
            n_demes=2,
            n_sexes=2,
            n_ages=2,
            migration_csr=_minimal_migration(),
            rate3d=np.zeros((2, 2, 2)),
        )
        for field in _BLUEPRINT_ARRAY_FIELDS:
            arr = getattr(bp, field)
            assert not arr.flags.writeable, f"double blueprint {field} is writable"
            with pytest.raises(ValueError, match="read-only"):
                arr[0] = 1

    def test_class_level_default_write_raises_and_shared_object_stays_zero(
        self,
    ) -> None:
        """A rejected write to the shared defaults corrupts no other blueprint.

        Attack vector: the class-level default empties are shared by
        identity across every default-constructed Blueprint; if the write
        guard failed silently, one write would poison all of them.
        """
        first = _defaulted_blueprint()
        second = _defaulted_blueprint()
        # Identity proves the blast radius of a successful write.
        assert first.migration_indptr is second.migration_indptr
        assert first.migration_weights is second.migration_weights

        with pytest.raises(ValueError, match="read-only"):
            first.migration_indptr[0] = 9
        with pytest.raises(ValueError, match="read-only"):
            first.migration_weights[0] = 9.0

        assert second.migration_indptr.sum() == 0
        assert second.migration_weights.sum() == 0.0


class TestDirectedRefreshUnderFrozenBlueprint:
    """Reconfiguration must survive the frozen-contract materialize path.

    ``pop.update()`` marks contract fields dirty and the next ``run``
    re-materializes the draft (Python dispatch) or refreshes the live
    Rust session in place (directed refresh).  A freeze that crossed the
    PyO3 boundary incorrectly would break exactly this path.
    """

    def test_update_competition_then_run_matches_fresh_build_python(
        self,
    ) -> None:
        """Directed refresh reproduces a from-scratch build bitwise.

        Twin invariant: (built with K=1e5, updated to K=8, run 4) must
        equal (built with K=8, run 4) exactly, and must differ from
        (built with K=1e5, run 4) — the third twin proves the capacity
        genuinely binds, so the equality is not vacuous.
        """
        refreshed = _build_beverton("R4RefreshPy", 100000.0)
        refreshed.update().competition(carrying_capacity=8.0)
        refreshed.run(4)

        twin = _build_beverton("R4RefreshPyTwin", 8.0)
        twin.run(4)
        unrefreshed = _build_beverton("R4RefreshPyOld", 100000.0)
        unrefreshed.run(4)

        np.testing.assert_array_equal(
            refreshed.state.individual_count, twin.state.individual_count
        )
        assert not np.array_equal(
            refreshed.state.individual_count,
            unrefreshed.state.individual_count,
        )
        assert float(refreshed.state.individual_count.sum()) == pytest.approx(
            8.311688311688313
        )

    def test_update_competition_then_run_matches_fresh_build_rust(self) -> None:
        """The Rust dirty-set bridge refreshes through materialize safely.

        Same twin design as the Python channel, but with the Rust session
        enabled: ``_sync_rust_backend`` calls ``materialize`` on the dirty
        path and pushes only the dirty params into the live session.
        """
        refreshed = _build_beverton("R4RefreshRs", 100000.0)
        refreshed.enable_rust_backend(seed=3)
        refreshed.update().competition(carrying_capacity=8.0)
        refreshed.run(4)

        twin = _build_beverton("R4RefreshRsTwin", 8.0)
        twin.enable_rust_backend(seed=3)
        twin.run(4)
        unrefreshed = _build_beverton("R4RefreshRsOld", 100000.0)
        unrefreshed.enable_rust_backend(seed=3)
        unrefreshed.run(4)

        np.testing.assert_array_equal(
            refreshed.state.individual_count, twin.state.individual_count
        )
        assert not np.array_equal(
            refreshed.state.individual_count,
            unrefreshed.state.individual_count,
        )
        assert refreshed._rust_dirty == set()  # pyright: ignore[reportPrivateUsage]  # drain contract of the dirty bridge
        assert refreshed.using_rust_backend

    def test_in_hook_update_deferred_through_dirty_bridge_matches_direct_push(
        self,
    ) -> None:
        """A mid-run capacity retune reaches the session via materialize only.

        Attack vector: while a Rust run is active the direct-push channel
        is closed (PyO3 borrow), so an in-hook ``update()`` lands in the
        draft and the dirty set alone — the next run must pull it through
        ``materialize`` + ``refresh_params``.  Dropping that sync (or a
        freeze that breaks the contract re-materialization) leaves the
        session running with the stale capacity.

        Twin invariant: (run 1 under K=1e5 with a retuning hook, run 3
        more) must bitwise equal (run 1 under K=1e5, update outside the
        run, run 3 more) — deferred and immediate channels converge.
        """
        @nt.hook(event="first")
        def retune(ctx: nt.TickContext) -> int:
            ctx.update().competition(carrying_capacity=8.0)
            return 0

        deferred = _build_beverton("R4RefreshDefer", 100000.0)
        deferred.enable_rust_backend(seed=5)
        deferred.register_hooks(retune, event="first")
        deferred.run(1)
        # The mid-run write deferred: the value is in the draft but the
        # session has not accepted it yet (no push during an active run).
        assert float(deferred.config.carrying_capacity) == 8.0
        assert deferred._rust_dirty == {"carrying_capacity"}  # pyright: ignore[reportPrivateUsage]  # deferral is the point of this channel
        deferred.run(3)

        immediate = _build_beverton("R4RefreshDirect", 100000.0)
        immediate.enable_rust_backend(seed=5)
        immediate.run(1)
        immediate.update().competition(carrying_capacity=8.0)
        immediate.run(3)

        never_retuned = _build_beverton("R4RefreshNever", 100000.0)
        never_retuned.enable_rust_backend(seed=5)
        never_retuned.run(4)

        np.testing.assert_array_equal(
            deferred.state.individual_count, immediate.state.individual_count
        )
        assert not np.array_equal(
            deferred.state.individual_count,
            never_retuned.state.individual_count,
        )


class TestR5SnapshotChannelAttacks:
    """Attack the engine through every object reachable from pop.state."""

    def test_engine_immutable_under_total_snapshot_mutation(self) -> None:
        """Filling every cell of a snapshot leaves the trajectory untouched.

        Attack vector: a snapshot that shares even one buffer with the
        engine would be corrupted by the full-array fill; the next tick
        would then diverge from the deterministic steady state.
        """
        pop = _build_discrete("R5TotalPoison")
        expected = pop._live_state().individual_count.copy()  # pyright: ignore[reportPrivateUsage]  # engine truth baseline
        pop.state.individual_count.fill(-999.0)
        pop.run(1)

        np.testing.assert_array_equal(
            pop._live_state().individual_count, expected  # pyright: ignore[reportPrivateUsage]  # engine truth after the run
        )
        # Exact steady state of the fixture: 10 adults of each sex, all WT|WT.
        assert float(pop.state.individual_count[0, 1, 0]) == 10.0
        assert float(pop.state.individual_count[1, 1, 0]) == 10.0
        assert float(pop.state.individual_count.sum()) == 20.0

    def test_snapshot_reads_do_not_accumulate_or_retain(self) -> None:
        """200 property reads leave no retained snapshots and no growth.

        Attack vector: a snapshot discipline that logs or caches every
        returned container would grow memory monotonically with property
        access; the weakref pin proves each snapshot dies with its
        caller, and the tracemalloc bound (observed ~2 KiB) catches any
        per-access retention (a leaked snapshot payload is ~1.2 KiB x
        200 = 240 KiB, far above the bound).
        """
        pop = _build_age_with_species(_species("R5Loop"), "R5Loop", n_ages=10)
        reference = pop.state.individual_count.copy()
        ticks_before = pop.history.ticks

        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        for _ in range(200):
            snapshot = pop.state
            assert float(snapshot.individual_count.sum()) == 40.0
        current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert current - base < 131072, (
            f"200 snapshot reads grew traced memory by {current - base} "
            "bytes; snapshots are being retained somewhere"
        )
        assert pop.history.ticks == ticks_before

        # The engine does not keep the snapshot (or its buffers) alive
        # after the caller drops the last reference.
        arr = pop.state.individual_count
        ref = weakref.ref(arr)
        np.testing.assert_array_equal(arr, reference)
        del arr, snapshot
        gc.collect()
        assert ref() is None

    def test_tick_context_metrics_reflect_engine_truth_after_poisoning(
        self,
    ) -> None:
        """In-hook metrics ignore a poisoned external snapshot entirely.

        Attack vector: if TickMetrics ever computed from ``pop.state``
        (the snapshot face) instead of the live loan, a caller-mutated
        snapshot would inject fake numbers into hooks.
        """
        seen: dict[str, float] = {}

        @nt.hook(event="first")
        def probe(ctx: nt.TickContext) -> int:
            seen["total"] = float(ctx.metrics.total)
            seen["female"] = float(ctx.metrics.by_sex[0])
            seen["male"] = float(ctx.metrics.by_sex[1])
            seen["wildtype"] = ctx.metrics.genotype_counts["WT|WT:default"]
            return 0

        pop = _build_age("R5MetricsTruth")
        pop.register_hooks(probe, event="first")
        # Poison every cell of an external snapshot before the run.
        poisoned = pop.state
        poisoned.individual_count.fill(-999.0)
        del poisoned

        twin = _build_age("R5MetricsTwin")
        twin.register_hooks(probe, event="first")

        pop.run(1)
        twin.run(1)

        # Engine truth at the first event: 40 adults, 20 per sex, all WT|WT.
        assert seen["total"] == 40.0
        assert seen["female"] == 20.0
        assert seen["male"] == 20.0
        assert seen["wildtype"] == 40.0
        # The poisoned snapshot left the trajectory bitwise unchanged.
        np.testing.assert_array_equal(
            pop.state.individual_count, twin.state.individual_count
        )
        np.testing.assert_array_equal(
            pop.state.sperm_storage, twin.state.sperm_storage
        )

    def test_spatial_slice_state_writes_are_live_and_scoped(self) -> None:
        """DemeSlice.state is the deme's live container (S3-pinned semantics).

        Current contract: writing through the slice reaches the deme's
        live arrays, is invisible to the other demes, and a sperm-only
        write leaves the container's individual-count aggregation (and
        every deme's counts) untouched.
        """
        species = _species("R5SpatialLive")
        d0 = _build_age_with_species(species, "R5SpatialD0")
        d1 = _build_age_with_species(species, "R5SpatialD1")
        spatial = nt.SpatialPopulation(
            [d0, d1], migration_rate=0.0, name="R5SpatialLive"
        )
        counts_before = spatial.demes[0].state.individual_count.copy()
        total_before = spatial.get_total_count()
        assert total_before == 80

        # Pinned S3 semantics: the slice hands out the live container.
        assert spatial.demes[0].state is d0._state  # pyright: ignore[reportPrivateUsage]  # the pinned live delegation itself

        spatial.demes[0].state.sperm_storage[1, 0, 0] = 777.0

        assert float(d0._state.sperm_storage.sum()) == 777.0  # pyright: ignore[reportPrivateUsage]  # live write landed
        assert float(d1._state.sperm_storage.sum()) == 0.0  # pyright: ignore[reportPrivateUsage]  # sibling deme isolated
        np.testing.assert_array_equal(
            spatial.demes[0].state.individual_count, counts_before
        )
        assert spatial.get_total_count() == total_before


class TestLiveStateContract:
    """Pin the narrowed accessor pair behind the snapshot face."""

    def test_uninitialized_state_raises_and_restore_leaves_state_intact(
        self,
    ) -> None:
        """None state: _live_state raises RuntimeError, state AttributeError.

        The two faces intentionally raise different exception types
        (public snapshot face: AttributeError miricking a missing
        attribute; internal accessor: RuntimeError).  Restoring the
        container afterwards must yield the exact pre-attack state.
        """
        pop = _build_discrete("LiveNone")
        original = pop._state  # pyright: ignore[reportPrivateUsage]  # set-then-restore attack
        pop._state = None  # pyright: ignore[reportPrivateUsage]  # set-then-restore attack
        try:
            with pytest.raises(RuntimeError, match="not been initialized"):
                pop._live_state()  # pyright: ignore[reportPrivateUsage]  # the accessor under test
            with pytest.raises(AttributeError, match="not been initialized"):
                _ = pop.state
        finally:
            pop._state = original  # pyright: ignore[reportPrivateUsage]  # set-then-restore attack

        assert pop.state.n_tick == original.n_tick
        assert float(pop.state.individual_count.sum()) == 20.0

    def test_live_state_returns_the_same_object_every_call(self) -> None:
        """_live_state is the stable engine channel; it never copies.

        Attack vector: an accessor that defensively copied would break
        every internal write path (state install, migration write-backs)
        — writes would land in a discarded copy.
        """
        for pop in (
            _build_discrete("LiveIdentityDiscrete"),
            _build_age("LiveIdentityAge"),
        ):
            assert pop._live_state() is pop._live_state()  # pyright: ignore[reportPrivateUsage]  # identity is the contract
            assert pop._live_state() is pop._state  # pyright: ignore[reportPrivateUsage]  # identity is the contract

    def test_snapshot_state_never_aliases_the_live_container(self) -> None:
        """_snapshot_state returns a fresh container sharing no buffers.

        Attack vector: a snapshot built with ``_replace`` (no copies)
        would pass an ``is not`` identity check while still exposing the
        engine's arrays; only the buffer-level check catches it.
        """
        for pop in (
            _build_discrete("LiveAliasDiscrete"),
            _build_age("LiveAliasAge"),
        ):
            live = pop._live_state()  # pyright: ignore[reportPrivateUsage]  # buffer-level ownership attack
            snapshot = pop._snapshot_state()  # pyright: ignore[reportPrivateUsage]  # hook under test
            assert snapshot is not live
            assert not np.may_share_memory(snapshot.individual_count, live.individual_count)
            if hasattr(live, "sperm_storage"):
                assert not np.may_share_memory(snapshot.sperm_storage, live.sperm_storage)
            second = pop._snapshot_state()  # pyright: ignore[reportPrivateUsage]  # hook under test
            assert not np.may_share_memory(
                snapshot.individual_count, second.individual_count
            )


class TestSnapshotDisciplineTransitions:
    """Lifecycle sequences through the _output mixin under snapshot discipline."""

    def test_restore_checkpoint_restores_counts_and_truncates_future(self) -> None:
        """restore -> run resumes from the restored tick with raw history.

        Exercises the _output.py path that writes through the live state
        (``self._state.individual_count[:] = ...``): counts must equal
        the recorded tick-1 row, both tick sources must sync, and the
        truncated timeline must continue without duplicate ticks.
        """
        pop = _build_discrete_raw("TransRestore")
        pop.run(3, record_every=1)
        assert pop.history.ticks == (0, 1, 2, 3)
        _, counts_at_1, _ = pop.history.restore_state(1)

        pop.restore_checkpoint(1)

        assert pop.tick == 1
        assert pop._live_state().n_tick == 1  # pyright: ignore[reportPrivateUsage]  # both tick sources synced
        assert pop.history.ticks == (0, 1)
        np.testing.assert_array_equal(pop.state.individual_count, counts_at_1)

        pop.run(1, record_every=1)
        assert pop.tick == 2
        assert pop.history.ticks == (0, 1, 2)
        assert float(pop.state.individual_count.sum()) == 20.0

    def test_reset_restores_initial_state_and_supports_rerun(self) -> None:
        """reset clears the timeline and returns the exact initial counts."""
        pop = _build_discrete_raw("TransReset")
        initial = pop.state.individual_count.copy()
        pop.run(2, record_every=1)
        assert pop.tick == 2

        pop.reset()

        assert pop.tick == 0
        assert pop.history.ticks == ()
        np.testing.assert_array_equal(pop.state.individual_count, initial)

        # The reset population reproduces a fresh population's first tick.
        twin = _build_discrete_raw("TransResetTwin")
        twin.run(1, record_every=1)
        pop.run(1, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, twin.state.individual_count
        )
        assert pop.history.ticks == twin.history.ticks

    def test_import_state_then_run_matches_fresh_run(self) -> None:
        """import -> run continues from the imported tick on a clean timeline."""
        pop = _build_discrete_raw("TransImport")
        flat = pop.export_state()
        pop.run(2, record_every=1)
        assert pop.tick == 2

        pop.import_state(flat)

        assert pop.tick == 0
        assert pop.history.ticks == ()

        twin = _build_discrete_raw("TransImportTwin")
        twin.run(1, record_every=1)
        pop.run(1, record_every=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, twin.state.individual_count
        )
        assert pop.history.ticks == twin.history.ticks


def _array_fingerprint(arr: np.ndarray) -> tuple[tuple[int, ...], str, float]:
    """Return a structural fingerprint (shape, dtype, value sum) of *arr*.

    The value sum is float64-cast so bool masks and int indices compare
    on one numeric axis.

    Args:
        arr: Array to fingerprint.

    Returns:
        ``(shape, dtype string, float64 value sum)``.
    """
    numeric = arr.astype(np.float64)
    return arr.shape, str(arr.dtype), float(numeric.sum())


class TestFrozenWriteErrorPaths:
    """Invalid writes raise loudly and atomically."""

    def test_all_eight_fields_reject_writes_with_read_only_message(self) -> None:
        """Every field raises ValueError('read-only') and nothing changes.

        The message must name the cause (a generic failure or a silent
        no-op write would hide the freeze regression), and the failed
        writes must leave the draft, the contract, and dtype/shape
        fingerprints exactly as they were.
        """
        pop = _build_age("ErrPathAllFields")
        cfg = pop.config
        contract = materialize(cfg).blueprint
        fingerprint_before = {
            field: _array_fingerprint(np.asarray(getattr(contract, field)))
            for field in _BLUEPRINT_ARRAY_FIELDS
        }
        draft_sums_before = {
            field: float(np.asarray(getattr(cfg, field), dtype=np.float64).sum())
            for field in _DRAFT_BACKED_FIELDS
        }

        for field in _BLUEPRINT_ARRAY_FIELDS:
            with pytest.raises(ValueError, match="read-only"):
                getattr(contract, field)[0] = 1

        for field in _BLUEPRINT_ARRAY_FIELDS:
            assert _array_fingerprint(np.asarray(getattr(contract, field))) == (
                fingerprint_before[field]
            )
        for field in _DRAFT_BACKED_FIELDS:
            assert float(np.asarray(getattr(cfg, field), dtype=np.float64).sum()) == (
                draft_sums_before[field]
            )
