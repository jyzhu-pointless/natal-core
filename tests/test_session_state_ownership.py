"""Session-owned state contracts (plan S2, batch 23).

The plain-model Rust sessions own the counts, sperm storage, and tick:
``run`` accepts control parameters only, Python reads state back through
snapshots, and the old per-run full-state handoff is gone.  These tests
pin the ownership surface and the negative contracts over the retired
handoff signatures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal import _engine_rs
from natal.contracts.params import Params
from natal.frontend.data import DiscretePopulationState, PopulationState
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.population.base import RUNTIME_FLUSH_FIELDS


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_discrete(name: str, *, stochastic: bool = False) -> nt.DiscreteGenerationPopulation:
    """Return a deterministic discrete population."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), stochastic=stochastic
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 30}, "male": {"WT|WT": 30}}
        )
        .survival(female_age0_survival=0.9, male_age0_survival=0.9)
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(carrying_capacity=100.0, low_density_growth_rate=2.0)
        .build()
    )


def _build_age(name: str, *, stochastic: bool = False) -> nt.AgeStructuredPopulation:
    """Return a deterministic age-structured population.

    Args:
        name: Unique population label.
        stochastic: Whether stochastic sampling is enabled (the only axis
            where the session RNG stream matters).
    """
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(name), stochastic=stochastic
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 0.0]},
                "male": {"WT|WT": [0.0, 20.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.8, 0.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0)
        .build()
    )


class TestNegativeContracts:
    """The retired handoff signatures are unreachable."""

    def test_backend_run_rejects_the_state_argument(self) -> None:
        """``backend.run(state=...)`` was the handoff API; it is gone."""
        from natal.backends.rust.rust_backend import (
            RustDiscreteLifecycleBackend,
        )

        pop = _build_discrete("NegRunDisc")
        backend = RustDiscreteLifecycleBackend(pop.config, None, seed=0)
        with pytest.raises(TypeError, match="unexpected keyword"):
            backend.run(state=pop._state, n_steps=1)  # pyright: ignore[reportCallIssue]  # negative contract probe passes the retired kwarg

    def test_raw_session_run_rejects_the_state_arrays(self) -> None:
        """The old ``session.run(ind, sperm, tick, ...)`` positional shape is gone."""
        pop = _build_discrete("NegSession")
        # Discrete session: run(n_ticks, record_interval, wf, ...) — an
        # ndarray in the first slot cannot coerce to int.
        session = _engine_rs.DiscreteEngineSession(
            _materialize_blueprint(pop), _materialize_params(pop), 0
        )
        with pytest.raises(TypeError):
            session.run(np.zeros((2, 2, 2)), 3, False)  # pyright: ignore[reportCallIssue]  # negative contract probe

    def test_raw_session_tick_rejects_the_state_arrays(self) -> None:
        """``session.tick(ind, tick, wf)`` is now ``tick(wf)``."""
        pop = _build_discrete("NegTick")
        session = _engine_rs.DiscreteEngineSession(
            _materialize_blueprint(pop), _materialize_params(pop), 0
        )
        with pytest.raises(TypeError):
            session.tick(np.zeros((2, 2, 2)), 0, False)  # pyright: ignore[reportCallIssue]  # negative contract probe


def _materialize_blueprint(pop: object) -> object:
    """Return the blueprint contract for a built population."""
    from natal.contracts.materialize import materialize

    return materialize(pop.config).blueprint  # type: ignore[attr-defined]  # duck-typed probe helper


def _materialize_params(pop: object) -> object:
    """Return the params contract for a built population."""
    from natal.contracts.materialize import materialize

    return materialize(pop.config).params  # type: ignore[attr-defined]  # duck-typed probe helper


class TestSessionOwnership:
    """The session is the sole owner of counts and tick while enabled."""

    def test_run_advances_session_state_not_caller_arrays(self) -> None:
        """Control-only run: caller arrays are never handed in or out."""
        pop = _build_discrete("OwnDisc")
        pop._initialize_session(seed=3)
        caller_copy = pop.state.individual_count.copy()
        pop.run(2)

        state = pop.state
        assert pop.tick == 2
        assert state.n_tick == 2
        # The post-run state differs from the pre-run copy (the engine
        # moved), but the caller's old copy was never mutated as a side
        # effect of the new ownership model.
        assert not np.array_equal(state.individual_count, caller_copy)

    def test_state_cache_refreshes_lazily_after_run(self) -> None:
        """Pop.state reflects the session immediately after a run."""
        pop = _build_discrete("OwnLazy")
        pop._initialize_session(seed=3)
        pop.run(2)
        snapshot = pop.state.individual_count

        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # ownership surface probe
        tick, ind_flat = backend.state_snapshot()  # type: ignore[union-attr]  # enabled above
        np.testing.assert_array_equal(snapshot.reshape(-1), ind_flat)
        assert pop.tick == int(tick)

    def test_mutating_the_cache_never_reaches_the_session(self) -> None:
        """A tampered Python cache is invisible to the session-run path."""
        pop = _build_discrete("OwnTamper")
        pop._initialize_session(seed=3)
        pop.run(2)
        pop._state.individual_count[...] = 999.0  # pyright: ignore[reportPrivateUsage]  # tamper the cache
        snapshot = pop.state.individual_count
        assert float(snapshot.sum()) != 999.0 * snapshot.size

    def test_import_state_installs_into_the_session(self) -> None:
        """import_state becomes the session state; reruns continue from it."""
        pop = _build_discrete("OwnImport")
        pop._initialize_session(seed=3)
        pop.run(2)
        pop.import_state(pop.export_state())
        # export_state carries the tick, so the import rebases to tick 2
        # and the session runs three more from there.
        assert pop.tick == 2
        pop.run(3)
        assert pop.tick == 5

    def test_hetero_spatial_uses_the_explicit_state_round_trip(self) -> None:
        """Spatial's per-tick data plane runs backends through run_tick(state)."""
        sp = _species("OwnSpatial")
        spatial = (
            nt.SpatialPopulation.builder(sp, n_demes=2, pop_type="discrete_generation")
            .setup(name="own_sp", stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": 20},
                    "male": {"WT|WT": 20},
                }
            )
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=500.0)
            .build()
        )
        spatial.run(2)
        assert float(spatial.get_total_count()) > 0.0
        for deme in spatial.demes:
            assert deme.tick == 2


def _tick0_flat_state(pop: nt.AgeStructuredPopulation) -> NDArray[np.float64]:
    """Return the blueprint initial population in ``import_state`` flat layout.

    ``export_state``/``import_state`` wire format: ``[n_tick, ind.ravel(),
    sperm.ravel()]``.  The blueprint holds the *declared* initial counts and
    storage, which is exactly the tick-0 state a fresh session starts from.

    Args:
        pop: A built population whose blueprint carries the initial state.

    Returns:
        The flat tick-0 state array.
    """
    bp = _materialize_blueprint(pop)  # type: ignore[attr-defined]  # duck-typed probe helper
    ind = np.asarray(bp.initial_individual_count, dtype=np.float64)  # type: ignore[attr-defined]  # duck-typed probe helper
    sperm = np.asarray(bp.initial_sperm_storage, dtype=np.float64)  # type: ignore[attr-defined]  # duck-typed probe helper
    return np.concatenate([[0.0], ind.ravel(), sperm.ravel()])


class TestSessionRngOwnership:
    """The session RNG streams across runs and is rebuilt, not saved.

    Stochastic age-structured trajectories are deterministic functions of
    one seeded session RNG: ``_initialize_session(seed=S)`` starts the
    stream at *S*, every run call keeps consuming it, and only a backend
    rebuild (a same-seed re-``enable``) resets it.  State
    pushes (``set_state``/``import_state``) never touch the stream.  All
    comparisons below are bitwise over the full state (counts + sperm).
    """

    def test_run_continuation_matches_a_single_batch_run(self) -> None:
        """run(1)+run(1) equals one seeded run(2): runs never reseed.

        A repeated ``run`` call on the same session must continue the same
        RNG stream; if ``run`` reseeded per call, the split trajectory
        would draw the seed-*S* stream twice from different states and
        diverge from the atomic two-tick run.
        """
        split = _build_age("RngCont", stochastic=True)
        split._initialize_session(seed=13)
        split.run(1)
        split.run(1)

        atomic = _build_age("RngBatch", stochastic=True)
        atomic._initialize_session(seed=13)
        atomic.run(2)

        np.testing.assert_array_equal(
            split.state.individual_count, atomic.state.individual_count
        )
        np.testing.assert_array_equal(
            split.state.sperm_storage, atomic.state.sperm_storage
        )
        assert split.tick == 2
        assert atomic.tick == 2

    def test_reenable_after_run_diverges_from_the_continuation(self) -> None:
        """A same-seed re-enable rebuilds the session: the RNG restarts
        while the state continues.

        After ``run(1)`` (state at tick 1) a same-seed re-enable hands the
        tick-1 state to a brand-new stream that starts at seed 13 again,
        whereas a fresh ``run(2)`` plays the same stream against the tick-0
        state.  The two end states are bitwise different: the rebuild did
        not replay the first tick's draws.
        """
        pop = _build_age("RngRefresh", stochastic=True)
        pop._initialize_session(seed=13)
        pop.run(1)
        pop._initialize_session(seed=13)
        pop.run(1)

        fresh = _build_age("RngRefreshRef", stochastic=True)
        fresh._initialize_session(seed=13)
        fresh.run(2)

        assert pop.tick == 2
        assert fresh.tick == 2
        assert not np.array_equal(
            pop.state.individual_count, fresh.state.individual_count
        )

    def test_re_enable_rebuilds_rng_and_preserves_state(self) -> None:
        """A second same-seed ``enable`` is a rebuild, not a no-op.

        ``enable(seed=13)`` after ``run(1)`` preserves the state while
        restarting the RNG at the original seed: the end state differs
        bitwise from the pure continuation run(2).
        """
        pop = _build_age("RngReenable", stochastic=True)
        pop._initialize_session(seed=13)
        pop.run(1)
        pop._initialize_session(seed=13)
        pop.run(1)

        continuation = _build_age("RngReenableCont", stochastic=True)
        continuation._initialize_session(seed=13)
        continuation.run(2)

        continuation = _build_age("RngReenableCont", stochastic=True)
        continuation._initialize_session(seed=13)
        continuation.run(2)

        assert not np.array_equal(
            pop.state.individual_count, continuation.state.individual_count
        )
        assert pop.tick == 2

    def test_state_rollback_keeps_stream_but_checkpoint_rollback_rewinds(self) -> None:
        """set_state moves state alone; only a checkpoint moves both.

        After ``run(2)``, pushing the blueprint tick-0 state back in (a
        ``set_state``-equivalent state rollback) and running two more ticks
        does NOT reproduce the original trajectory: the RNG kept streaming.
        The control — a full record-aligned checkpoint rollback to tick 0
        (state + RNG words) — replays the atomic ``run(2)`` bitwise.
        """
        pop = _build_age("RngState", stochastic=True)
        pop._initialize_session(seed=17)
        pop.run(2)
        first_ind = pop.state.individual_count.copy()
        first_sperm = pop.state.sperm_storage.copy()
        assert pop.tick == 2

        pop.import_state(_tick0_flat_state(pop))
        assert pop.tick == 0
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind_flat, sperm_flat = backend.state_snapshot()
        assert tick == 0
        bp = _materialize_blueprint(pop)  # type: ignore[attr-defined]  # duck-typed probe helper
        np.testing.assert_array_equal(
            ind_flat, np.asarray(bp.initial_individual_count, dtype=np.float64).ravel()
        )
        np.testing.assert_array_equal(
            sperm_flat, np.asarray(bp.initial_sperm_storage, dtype=np.float64).ravel()
        )

        pop.run(2)
        assert pop.tick == 2
        # State-only rollback: the stream moved on, so the replay diverges.
        assert not np.array_equal(pop.state.individual_count, first_ind)
        assert not np.array_equal(pop.state.sperm_storage, first_sperm)

        # Control: a checkpoint rollback rewinds state AND the RNG words,
        # so the replay reproduces the identical two-tick trajectory.
        control = _build_age("RngStateCtl", stochastic=True)
        control._initialize_session(seed=17)
        control.run(2, record_every=1)
        assert control.tick == 2
        np.testing.assert_array_equal(control.state.individual_count, first_ind)
        np.testing.assert_array_equal(control.state.sperm_storage, first_sperm)
        control.restore_checkpoint(0)
        assert control.tick == 0
        control.run(2, record_every=1)
        np.testing.assert_array_equal(control.state.individual_count, first_ind)
        np.testing.assert_array_equal(control.state.sperm_storage, first_sperm)


class TestManualHookFlush:
    """Hook writes outside a Rust run are flushed into the session."""

    def test_manual_trigger_event_write_reaches_the_engine_state(self) -> None:
        """A manual ``trigger_event`` write is visible to the next run.

        Returning a marker from a manual ``finish`` event must land in the
        session-owned state (``_flush_state_to_session`` after the
        callback), so the following ``run`` starts from the marked counts
        and matches bitwise a control whose markers were pushed through
        ``import_state``.  Rust never fires the ``finish`` callback inside
        a non-stopped batch, so an in-run re-apply cannot mask a lost
        flush.
        """

        @nt.hook(event="finish")
        def mark(pop: TickContext) -> int:
            pop.state.individual_count[0, 1, 0] = 23.0
            return 0

        test = _build_age("FlushTest")
        test.update().hooks(mark)
        test._initialize_session(seed=0)
        # The manual event runs the callback and must flush its write.
        assert test.trigger_event("finish") == 0
        backend = test._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind_flat, _sperm_flat = backend.state_snapshot()
        assert tick == 0
        assert float(ind_flat.reshape((2, 3, 3))[0, 1, 0]) == 23.0
        test.run(1)

        control = _build_age("FlushCtl")
        control._initialize_session(seed=0)
        marker = control._live_state().individual_count.copy()  # pyright: ignore[reportPrivateUsage]  # session probe
        marker[0, 1, 0] = 23.0
        control.import_state(
            PopulationState(
                n_tick=0,
                individual_count=marker,
                sperm_storage=control._live_state().sperm_storage,  # pyright: ignore[reportPrivateUsage]  # session probe
            )
        )
        control.run(1)

        np.testing.assert_array_equal(
            test.state.individual_count, control.state.individual_count
        )
        np.testing.assert_array_equal(
            test.state.sperm_storage, control.state.sperm_storage
        )
        assert test.tick == 1
        assert control.tick == 1


class TestSnapshotContainerIndependence:
    """Backend snapshots are fresh copies; writes never reach the session."""

    def test_snapshot_arrays_are_independent_copies(self) -> None:
        """Mutating one snapshot leaves the other and the session intact."""
        pop = _build_age("SnapIndep")
        pop._initialize_session(seed=2)
        pop.run(2)
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None

        tick_a, ind_a, sperm_a = backend.state_snapshot()
        tick_b, ind_b, sperm_b = backend.state_snapshot()
        assert tick_a == 2
        assert tick_b == 2

        ind_a[0] = 123.0
        sperm_a[0] = 999.0
        assert float(ind_a[0]) == 123.0
        assert float(sperm_a[0]) == 999.0

        tick_c, ind_c, sperm_c = backend.state_snapshot()
        assert tick_c == 2
        # The third snapshot (fresh session read) is bitwise equal to the
        # untouched second snapshot: the mutation stayed in the caller's copy.
        np.testing.assert_array_equal(ind_b, ind_c)
        np.testing.assert_array_equal(sperm_b, sperm_c)
        # The mutated copy did diverge, proving the mutation happened on a
        # caller-owned array rather than being silently dropped.
        assert not np.array_equal(ind_a, ind_c)
        assert not np.array_equal(sperm_a, sperm_c)


class TestEnableSeedsFromCurrentState:
    """Enabling installs the population's live state, not the blueprint's."""

    def test_mutation_before_enable_seeds_the_session(self) -> None:
        """A pre-enable state edit is the session's starting point."""
        pop = _build_age("SeedMut")
        pop._live_state().individual_count[0, 1, 0] = 23.0  # pyright: ignore[reportPrivateUsage]  # session probe
        pop._initialize_session(seed=0)
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind_flat, _sperm_flat = backend.state_snapshot()
        assert tick == 0
        assert float(ind_flat.reshape((2, 3, 3))[0, 1, 0]) == 23.0
        # The unmutated blueprint value at the same cell is 20.0 — a session
        # seeded from the blueprint initial population would report 20.0.
        plain = _build_age("SeedRef")
        assert float(plain.state.individual_count[0, 1, 0]) == 20.0

        # The engine really runs from the mutated state: one tick from the
        # mutated session diverges from one tick from the blueprint session.
        pop.run(1)
        plain._initialize_session(seed=0)
        plain.run(1)
        assert not np.array_equal(
            pop.state.individual_count, plain.state.individual_count
        )
        assert pop.tick == 1


class TestSessionStateErrorPaths:
    """Invalid session state pushes raise and leave the session unchanged."""

    def test_wrong_length_set_state_raises_and_leaves_state_unchanged(self) -> None:
        """Rejected age set_state calls are atomic: no partial state swap."""
        pop = _build_age("ErrAge")
        pop._initialize_session(seed=2)
        pop.run(2)
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind, sperm = backend.state_snapshot()

        wrong = PopulationState(
            n_tick=9,
            individual_count=np.zeros((2, 2, 2), dtype=np.float64),
            sperm_storage=np.zeros((2, 2, 2), dtype=np.float64),
        )
        with pytest.raises(ValueError, match="ind_flat"):
            backend.set_state(wrong)
        tick2, ind2, sperm2 = backend.state_snapshot()
        assert tick2 == 2
        np.testing.assert_array_equal(ind, ind2)
        np.testing.assert_array_equal(sperm, sperm2)

        wrong_sperm = PopulationState(
            n_tick=9,
            individual_count=ind.reshape((2, 3, 3)),
            sperm_storage=np.zeros((5,), dtype=np.float64),
        )
        with pytest.raises(ValueError, match="sperm_flat"):
            backend.set_state(wrong_sperm)
        tick3, ind3, sperm3 = backend.state_snapshot()
        assert tick3 == 2
        np.testing.assert_array_equal(ind, ind3)
        np.testing.assert_array_equal(sperm, sperm3)

    def test_discrete_wrong_length_set_state_raises_and_leaves_state_unchanged(self) -> None:
        """The discrete session rejects wrong-length counts atomically."""
        pop = _build_discrete("ErrDisc")
        pop._initialize_session(seed=2)
        pop.run(2)
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind = backend.state_snapshot()

        wrong = DiscretePopulationState(
            n_tick=9, individual_count=np.zeros((2, 2, 2), dtype=np.float64)
        )
        with pytest.raises(ValueError, match="ind_flat"):
            backend.set_state(wrong)
        tick2, ind2 = backend.state_snapshot()
        assert tick2 == 2
        np.testing.assert_array_equal(ind, ind2)

    def test_restore_from_checkpoint_after_clear_history_is_none(self) -> None:
        """Clearing history also drops the paired session checkpoints."""
        pop = _build_age("ErrCheck")
        pop._initialize_session(seed=2)
        pop.run(2, record_every=1)
        assert pop.tick == 2
        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        pop.clear_history()
        # Both recorded-aligned ticks lost their checkpoint: no rollback is
        # possible, and the population keeps running at tick 2.
        assert backend.restore_from_checkpoint(1) is None
        assert backend.restore_from_checkpoint(2) is None
        assert pop.tick == 2
        pop.run(1)
        assert pop.tick == 3


class TestZeroTickRun:
    """A zero-step run is a pure no-op on state and tick."""

    def test_run_zero_ticks_is_a_no_op(self) -> None:
        """run(0) leaves tick and full state bitwise unchanged."""
        pop = _build_age("Zero0", stochastic=True)
        pop._initialize_session(seed=5)
        before_ind = pop.state.individual_count.copy()
        before_sperm = pop.state.sperm_storage.copy()
        pop.run(0)
        assert pop.tick == 0
        np.testing.assert_array_equal(pop.state.individual_count, before_ind)
        np.testing.assert_array_equal(pop.state.sperm_storage, before_sperm)

        pop.run(2)
        after_ind = pop.state.individual_count.copy()
        after_sperm = pop.state.sperm_storage.copy()
        pop.run(0)
        assert pop.tick == 2
        np.testing.assert_array_equal(pop.state.individual_count, after_ind)
        np.testing.assert_array_equal(pop.state.sperm_storage, after_sperm)


class TestDirtyBridgeNegativeContracts:
    """The retired dirty bridge is unreachable (S2 batch 24)."""

    def test_dirty_set_and_mirror_are_gone(self) -> None:
        """``_rust_dirty``, ``_contract_params``, and ``_sync_rust_backend`` are gone."""
        pop = _build_discrete("NegDirty")
        pop._initialize_session(seed=3)
        for attr in ("_rust_dirty", "_contract_params", "_sync_rust_backend"):
            assert not hasattr(pop, attr), (
                f"{attr} is back — the retired dirty-bridge compartment returned"
            )

    def test_dirty_sink_parameter_is_gone_from_writers(self) -> None:
        """No writer accepts the dirty_sink argument anymore."""
        from natal.frontend.configurator._writers import (
            CoreConfigWriter,
            DraftWriter,
        )

        pop = _build_discrete("NegDirtySink")
        pop._initialize_session(seed=3)
        with pytest.raises(TypeError, match="dirty_sink"):
            CoreConfigWriter(pop.config, pop._rust_lifecycle_backend, dirty_sink=set())  # pyright: ignore[reportCallIssue]  # negative contract probe
        with pytest.raises(TypeError, match="dirty_sink"):
            DraftWriter(pop.config, dirty_sink=set())  # pyright: ignore[reportCallIssue]  # negative contract probe

    def test_needs_rebuild_flag_round_trip(self) -> None:
        """Program writes mark dispatch replacement; the session is preserved."""
        pop = _build_discrete("NegRebuild")
        pop._initialize_session(seed=3)
        backend_before = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]

        @nt.hook(event="first")
        def noop(ctx: nt.TickContext) -> int:
            return 0

        pop.register_hooks(noop, event="first")
        assert pop._rust_needs_rebuild is True  # pyright: ignore[reportPrivateUsage]

        pop.run(1)
        assert pop._rust_needs_rebuild is False  # pyright: ignore[reportPrivateUsage]
        assert pop._rust_lifecycle_backend is backend_before  # pyright: ignore[reportPrivateUsage]  # program replacement preserves session ownership

    def test_value_writes_do_not_flag_rebuild(self) -> None:
        """Value writes push straight to the session without a rebuild."""
        pop = _build_discrete("NegValue")
        pop._initialize_session(seed=3)
        backend_before = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]
        pop.update().competition(carrying_capacity=25.0)
        assert pop._rust_needs_rebuild is False  # pyright: ignore[reportPrivateUsage]
        assert pop._rust_lifecycle_backend is backend_before  # pyright: ignore[reportPrivateUsage]
        pop.run(1)
        assert pop._rust_lifecycle_backend is backend_before  # pyright: ignore[reportPrivateUsage]


# ============================================================================
# S2 batch 24: the run-boundary ecology flush and the rebuild-flag rollback
# ============================================================================


@nt.hook(event="early")
def _retune_capacity(ctx: TickContext) -> int:
    """Retune carrying capacity mid-tick (deferred to the draft under Rust)."""
    ctx.update().competition(carrying_capacity=25.0)
    return 0


@nt.hook(event="early")
def _noop_hook(ctx: TickContext) -> int:
    """Do-nothing callback (keeps the callback firing path identical)."""
    return 0


@nt.hook(event="early")
def _retune_eggs(ctx: TickContext) -> int:
    """Retune eggs per female mid-tick (deferred to the draft under Rust)."""
    ctx.update().reproduction(eggs_per_female=3.0)
    return 0


@nt.hook(event="early")
def _write_custom_slot(ctx: TickContext) -> int:
    """Write a custom slot from inside a run."""
    ctx.update().custom(probe=7.5)
    return 0


@nt.hook(event="early")
def _write_out_of_bounds(ctx: TickContext) -> int:
    """Write an out-of-bounds carrying capacity from inside a run."""
    ctx.update().competition(carrying_capacity=-5.0)
    return 0


def _age_with_hook(
    name: str,
    hook: Callable[[TickContext], int],
    *,
    stochastic: bool,
    custom: dict[str, float] | None = None,
) -> nt.AgeStructuredPopulation:
    """Return an age-structured population with one pre-enabled hook.

    Args:
        name: Unique population label.
        hook: Callback hook registered on the ``early`` event.
        stochastic: Whether the session RNG stream drives the trajectory.
        custom: Optional custom slots declared at build time.

    Returns:
        A built population ready for ``_initialize_session``.
    """
    chain = (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=stochastic)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 20.0, 0.0]},
                "male": {"WT|WT": [0.0, 20.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.8, 0.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0)
    )
    if custom is not None:
        chain = chain.custom(**custom)
    return chain.hooks(hook).build()


def _discrete_with_hook(
    name: str,
    hook: Callable[[TickContext], int],
    *,
    stochastic: bool,
) -> nt.DiscreteGenerationPopulation:
    """Return a discrete-generation population with one pre-enabled hook.

    Args:
        name: Unique population label.
        hook: Callback hook registered on the ``early`` event.
        stochastic: Whether the session RNG stream drives the trajectory.

    Returns:
        A built population ready for ``_initialize_session``.
    """
    chain = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), stochastic=stochastic
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 30}, "male": {"WT|WT": 30}}
        )
        .survival(female_age0_survival=0.9, male_age0_survival=0.9)
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(carrying_capacity=100.0, low_density_growth_rate=2.0)
    )
    return chain.hooks(hook).build()


def _drive_pop(
    name: str, drive: nt.HomingDrive, *, stochastic: bool
) -> nt.AgeStructuredPopulation:
    """Return an age-structured population carrying *drive* (preset).

    Args:
        name: Unique population label.
        drive: The homing-drive preset registered at build time.
        stochastic: Whether the session RNG stream drives the trajectory.

    Returns:
        A built population ready for ``_initialize_session``.
    """
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(name), stochastic=stochastic
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|Dr": [0.0, 100.0]},
                "male": {"WT|Dr": [0.0, 100.0]},
            }
        )
        .competition(carrying_capacity=500.0)
        .presets(drive)
        .build()
    )


class TestEventParameterCommit:
    """Callback candidates commit before subsequent lifecycle stages."""

    def test_in_run_capacity_retune_affects_current_age_tick(self) -> None:
        """Early-event K changes must affect the current density regulation.

        Reproduction precedes the early event, but it does not read K in
        this model. Retuning K before the run is therefore a valid exact
        control for both counts and sperm under the same session seed.
        """
        in_hook = _age_with_hook("DefEvent", _retune_capacity, stochastic=True)
        in_hook._initialize_session(seed=5)
        in_hook.run(1)
        first = in_hook.export_state().copy()
        backend = in_hook._rust_lifecycle_backend
        assert backend is not None
        assert backend._session.get_scalar("carrying_capacity") == 25.0
        in_hook.run(1)

        immediate = _age_with_hook("DefImmediate", _noop_hook, stochastic=True)
        immediate._initialize_session(seed=5)
        immediate.update().competition(carrying_capacity=25.0)
        immediate.run(1)
        np.testing.assert_array_equal(first, immediate.export_state())
        immediate.run(1)
        np.testing.assert_array_equal(in_hook.export_state(), immediate.export_state())
        assert in_hook.tick == immediate.tick == 2

        never = _age_with_hook("DefNever", _noop_hook, stochastic=True)
        never._initialize_session(seed=5)
        never.run(2)
        assert not np.array_equal(in_hook.state.individual_count, never.state.individual_count)

    def test_in_run_capacity_retune_drives_the_next_discrete_run(self) -> None:
        """The same boundary flush converges bitwise on the discrete model.

        The discrete fixture's default growth mode is ``no_competition``
        (K is inert), so the retune targets eggs per female — a value the
        discrete engine actually consumes.
        """
        deferred = _discrete_with_hook(
            "DefDiscrete", _retune_eggs, stochastic=True
        )
        deferred._initialize_session(seed=3)
        deferred.run(1)
        backend = deferred._rust_lifecycle_backend  # noqa: SLF001
        assert backend is not None
        assert backend._session.get_scalar("eggs_per_female") == 3.0  # noqa: SLF001
        # The boundary consumed the flag (flush ran) and the finally
        # cleared it; the session readback above is the effect proof.
        assert deferred._rust_deferred_writes is False  # noqa: SLF001
        deferred_tick1 = deferred.state.individual_count.copy()
        deferred.run(1)

        immediate = _discrete_with_hook(
            "DefDiscImmediate", _noop_hook, stochastic=True
        )
        immediate._initialize_session(seed=3)
        immediate.run(1)
        immediate.update().reproduction(eggs_per_female=3.0)
        immediate.run(1)

        # Tick 1 ran under the old eggs-per-female (single-tick reference).
        old_reference = _discrete_with_hook(
            "DefDiscOldRef", _noop_hook, stochastic=True
        )
        old_reference._initialize_session(seed=3)
        old_reference.run(1)
        np.testing.assert_array_equal(
            deferred_tick1, old_reference.state.individual_count
        )
        np.testing.assert_array_equal(
            deferred.state.individual_count, immediate.state.individual_count
        )
        assert deferred.tick == 2
        assert immediate.tick == 2
        # The retune was effective: 3 eggs per female halves reproduction.
        never = _discrete_with_hook("DefDiscNever", _noop_hook, stochastic=True)
        never._initialize_session(seed=3)
        never.run(2)
        assert not np.array_equal(
            deferred.state.individual_count, never.state.individual_count
        )


class TestInRunCustomSlotWrite:
    """A custom-slot write from inside a run must reach the session."""

    def test_in_run_custom_write_commits_without_boundary_replay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ctx.update().custom(...)`` inside a run reaches the session.

        The callback transaction writes custom values into the session.
        Replaying the complete Python draft after the run would overwrite
        other native changes, so that old bridge must remain unused.
        """
        pop = _age_with_hook("SlotHook", _write_custom_slot, stochastic=False)
        pop._initialize_session(seed=1)
        backend = pop._rust_lifecycle_backend  # noqa: SLF001
        assert backend is not None

        flushed: list[tuple[list[str], dict[str, object]]] = []
        original = backend.refresh_params

        def spy(fields: list[str], params_obj: Params) -> None:
            flushed.append((list(fields), dict(params_obj.custom_slots)))
            original(fields, params_obj)

        monkeypatch.setattr(backend, "refresh_params", spy)
        pop.run(1)

        assert dict(pop.config.custom) == {"probe": 7.5}
        # The native candidate committed the value without a full-draft replay.
        assert backend._session.get_custom_slots() == {"probe": 7.5}
        assert flushed == []

    def test_runtime_custom_write_accumulates_with_build_slots(self) -> None:
        """A runtime slot write must not wipe slots declared at build time.

        ``custom()`` documents that multiple calls accumulate; the runtime
        configurator re-validates the whole slot set, so a write through
        ``pop.update().custom(...)`` must preserve the slots the
        population was built with.
        """
        pop = _age_with_hook(
            "SlotAccum", _write_custom_slot, stochastic=False, custom={"mark": 3.0}
        )
        pop._initialize_session(seed=1)
        pop.run(1)
        assert dict(pop.config.custom) == {"mark": 3.0, "probe": 7.5}


class TestStructuralProgramUpdate:
    """Program updates preserve the session and its random stream."""

    def test_hook_registration_preserves_the_rng_stream(self) -> None:
        """A no-op hook changes dispatch without changing future draws.

        The Rust-only plan requires program replacement on the existing
        session: neither counts, sperm, tick, nor RNG may be reinitialized.
        """
        pop = _build_age("RebuildChain", stochastic=True)
        pop._initialize_session(seed=13)
        pop.run(1)
        backend_before = pop._rust_lifecycle_backend  # noqa: SLF001
        pop.register_hooks(_noop_hook, event="early")
        pop.run(1)
        assert pop._rust_lifecycle_backend is backend_before  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001
        assert pop.tick == 2

        atomic = _build_age("RebuildAtomic", stochastic=True)
        atomic._initialize_session(seed=13)
        atomic.run(2)
        np.testing.assert_array_equal(pop.export_state(), atomic.export_state())


class TestReconfigureRollbackFlagIntegrity:
    """Failed preset reconfigures must not schedule a session rebuild."""

    def _drive(self, name: str) -> nt.HomingDrive:
        """Return a homing drive preset registered on the test population."""
        return nt.HomingDrive(
            name=name,
            drive_allele="Dr",
            target_allele="WT",
            drive_conversion_rate=0.9,
        )

    def test_failed_reconfigure_keeps_stream_and_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed reconfigure leaves the RNG stream untouched (bitwise).

        ``refresh_modifier_maps`` marks ``_rust_needs_rebuild`` while the
        recipe runs; when the fitness patch then explodes, that mark must
        be rolled back together with the config.  A surviving mark routes
        the next run through a session rebuild, which reseeds the RNG:
        then split == fused fails bitwise (the exact regression that made
        a stub trajectory diverge 15.7%).
        """
        drive = self._drive("__rollback_reproc_flag__")
        split = _drive_pop("RecRollSplit", drive, stochastic=True)
        split._initialize_session(seed=29)
        split.run(1)
        backend_before = split._rust_lifecycle_backend  # noqa: SLF001

        def exploding_patch() -> None:
            raise RuntimeError("boom: fitness patch failure")

        monkeypatch.setattr(drive, "fitness_patch", exploding_patch)
        with pytest.raises(RuntimeError, match="boom"):
            split.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        monkeypatch.undo()

        # The attempted attribute change never landed on the preset: the
        # constructor-normalized (female, male) tuple is untouched, and the
        # rebuild flag sits at its pre-call (False) value.
        assert drive.drive_conversion_rate == (0.9, 0.9)
        assert split._rust_needs_rebuild is False  # noqa: SLF001

        # The next run must NOT rebuild: same backend object and a
        # bitwise match against the atomic two-tick run.  A separate
        # equivalent drive instance serves the control (presets bind to
        # their first species, so one object cannot serve two pops; the
        # maps depend on the parameters only).
        split.run(1)
        assert split._rust_lifecycle_backend is backend_before  # noqa: SLF001
        fused = _drive_pop(
            "RecRollFused", self._drive("__rollback_reproc_fused__"), stochastic=True
        )
        fused._initialize_session(seed=29)
        fused.run(2)
        np.testing.assert_array_equal(
            split.state.individual_count, fused.state.individual_count
        )
        np.testing.assert_array_equal(
            split.state.sperm_storage, fused.state.sperm_storage
        )
        assert split.tick == 2
        assert fused.tick == 2

    def test_failed_reconfigure_preserves_pending_structural_marks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing rebuild request survives a failed transaction.

        The rollback restores the flag instead of clearing it: a user's
        own pending structural update (here hook registration) must still
        be honored by the next run.
        """
        drive = self._drive("__rollback_pending_flag__")
        pop = _drive_pop("RecRollPending", drive, stochastic=True)
        pop._initialize_session(seed=29)
        pop.run(1)
        pop.register_hooks(_noop_hook, event="early")
        assert pop._rust_needs_rebuild is True  # noqa: SLF001
        backend_before = pop._rust_lifecycle_backend  # noqa: SLF001

        def exploding_patch() -> None:
            raise RuntimeError("boom: fitness patch failure")

        monkeypatch.setattr(drive, "fitness_patch", exploding_patch)
        with pytest.raises(RuntimeError, match="boom"):
            pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        monkeypatch.undo()

        # The pending user mark survived the rollback ...
        assert pop._rust_needs_rebuild is True  # noqa: SLF001
        # ... and the next run installs the program into the same session.
        pop.run(1)
        assert pop._rust_lifecycle_backend is backend_before  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001

    def test_successful_reconfigure_preserves_the_session(self) -> None:
        """A committed genetic recipe update cannot replace the session."""
        drive = self._drive("__recommit_flag__")
        pop = _drive_pop("RecCommit", drive, stochastic=False)
        pop._initialize_session(seed=29)
        pop.run(1)
        backend_before = pop._rust_lifecycle_backend  # noqa: SLF001

        pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
        assert drive.drive_conversion_rate == 0.3
        # The changed tables are submitted in place, without a rebuild.
        assert pop._rust_needs_rebuild is False  # noqa: SLF001

        pop.run(1)
        assert pop._rust_lifecycle_backend is backend_before  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001


class TestEnumAndSlotValueWrites:
    """Enum and slot-cell writes push values, never rebuild the session."""

    def test_mode_enum_write_pushes_session_without_rebuild(self) -> None:
        """``juvenile_growth_mode`` (enum) reaches the session as a value."""
        pop = _build_age("EnumWrite", stochastic=False)
        pop._initialize_session(seed=1)
        backend = pop._rust_lifecycle_backend  # noqa: SLF001
        assert backend is not None

        pop.params.juvenile_growth_mode = 4
        assert backend._session.get_scalar("growth_mode") == 4.0  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001
        assert pop._rust_lifecycle_backend is backend  # noqa: SLF001
        pop.run(1)
        assert pop._rust_lifecycle_backend is backend  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001

    def test_slot_cell_write_pushes_whole_vector_without_rebuild(self) -> None:
        """A slot cell pushes a whole vector into the session, no rebuild."""
        pop = _build_age("SlotWrite", stochastic=False)
        pop._initialize_session(seed=1)
        backend = pop._rust_lifecycle_backend  # noqa: SLF001
        assert backend is not None

        pop.params.competition_strength = 2.0
        np.testing.assert_array_equal(
            backend._session.get_tensor("competition_weights"),  # noqa: SLF001
            np.array([1.0, 2.0, 1.0], dtype=np.float64),
        )
        assert pop._rust_needs_rebuild is False  # noqa: SLF001
        assert pop._rust_lifecycle_backend is backend  # noqa: SLF001
        pop.run(1)
        assert pop._rust_lifecycle_backend is backend  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001


class TestDeferredFlushAtomicity:
    """A rejected in-run write must not schedule a boundary flush."""

    def test_in_run_writer_failure_leaves_no_deferral(self) -> None:
        """An out-of-bounds in-run write surfaces and leaves no deferral.

        The hook fires inside the batch; its write raises at validation.
        The failure must surface from ``run()`` and the deferred flag
        must NOT survive: the boundary flushes only when a write was
        deferred, so a failed write must not cause a later spurious flush
        that could re-push draft values over direct session writes (the
        inverted contract pinned by ``test_direct_write_survives_into_the_next_run``).
        """
        pop = _age_with_hook("BadWrite", _write_out_of_bounds, stochastic=False)
        pop._initialize_session(seed=1)
        d0 = float(pop.config.carrying_capacity)
        with pytest.raises(ValueError, match="carrying_capacity"):
            pop.run(1)
        # The failing write is atomic: neither the draft ...
        assert float(pop.config.carrying_capacity) == d0
        # ... nor the deferral bookkeeping was committed.
        assert pop._rust_deferred_writes is False  # noqa: SLF001
        assert pop._rust_needs_rebuild is False  # noqa: SLF001


class TestRuntimeEcoFieldList:
    """RUNTIME_FLUSH_FIELDS is the exact run-boundary flush set."""

    def test_constant_covers_exactly_the_contract_set(self) -> None:
        """The flush list is exactly the 7 scalars + 7 vectors + slots."""
        scalars = {
            "carrying_capacity",
            "eggs_per_female",
            "sex_ratio",
            "sperm_displacement_rate",
            "low_density_growth_rate",
            "growth_mode",
            "external_expected_eggs",
        }
        vectors = {
            "survival_rates",
            "mating_rates",
            "reproduction_rates",
            "fertility",
            "competition_weights",
            "equilibrium_distribution",
            "migration_rate",
        }
        genetics = {
            "viability_fitness",
            "fecundity_fitness",
            "sexual_selection_fitness",
            "zygote_viability_fitness",
            "offspring_tensor",
            "meiosis_map",
            "female_ztype_compatibility",
            "male_ztype_compatibility",
        }
        assert len(RUNTIME_FLUSH_FIELDS) == 23
        assert set(RUNTIME_FLUSH_FIELDS[:7]) == scalars
        assert set(RUNTIME_FLUSH_FIELDS[7:14]) == vectors
        assert RUNTIME_FLUSH_FIELDS[14] == "custom_slots"
        assert set(RUNTIME_FLUSH_FIELDS[15:]) == genetics
        assert len(set(RUNTIME_FLUSH_FIELDS)) == 23

    def test_event_write_does_not_replay_the_runtime_field_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The event commits K without replaying unrelated fields at run end."""
        pop = _age_with_hook("FieldList", _retune_capacity, stochastic=False)
        pop._initialize_session(seed=1)
        backend = pop._rust_lifecycle_backend  # noqa: SLF001
        assert backend is not None

        flushed: list[list[str]] = []
        original = backend.refresh_params

        def spy(fields: list[str], params_obj: Params) -> None:
            flushed.append(list(fields))
            original(fields, params_obj)

        monkeypatch.setattr(backend, "refresh_params", spy)
        pop.run(1)

        assert flushed == []
        # The refresh really landed in the session: the retuned K reads back.
        assert backend._session.get_scalar("carrying_capacity") == 25.0  # noqa: SLF001


class TestInRunGeneticsWrites:
    """In-run genetics/fitness writes reach the session at the boundary."""

    def test_in_run_viability_write_flushes_to_session(self) -> None:
        """A hook's fitness write lands in the session after the run (HB1).

        The run-boundary flush pulls every runtime contract field — the
        genetics tensors included — so a drift between the draft and the
        session can never survive a run boundary.
        """
        import natal as _nt

        sp = _species("InRunGenetics")

        @_nt.hook(event="early")
        def apply_viability(ctx: _nt.TickContext) -> int:
            ctx.update().fitness(viability={"WT|WT": 0.1})
            return 0

        pop = (
            _nt.DiscreteGenerationPopulation.setup(species=sp, name="irg", stochastic=False)
            .initial_state(
                individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=6, sex_ratio=0.5)
            .competition(carrying_capacity=100.0, low_density_growth_rate=2.0)
            .build()
        )
        pop._initialize_session(seed=11)
        pop.register_hooks(apply_viability, event="early")
        pop.run(1)

        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]
        session_viability = np.asarray(  # pyright: ignore[union-attr]
            backend._session.get_tensor("viability_fitness")  # noqa: SLF001
        )
        # Flat (2, n_ztypes) row-major: female x ztype 0 is index 0.
        assert float(session_viability[0]) == 0.1, (
            "in-run fitness write never reached the Rust session — the "
            "boundary flush must pull the genetics tensors"
        )

    def test_in_run_fitness_write_zeroes_survival_via_boundary(self) -> None:
        """A second in-run fitness field flushes through the same boundary."""
        import natal as _nt

        sp = _species("InRunFitness")

        @_nt.hook(event="early")
        def apply_zygote(ctx: _nt.TickContext) -> int:
            ctx.update().fitness(zygote_viability={"WT|Dr": 0.25})
            return 0

        pop = (
            _nt.DiscreteGenerationPopulation.setup(species=sp, name="irf", stochastic=False)
            .initial_state(
                individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=6, sex_ratio=0.5)
            .competition(carrying_capacity=100.0, low_density_growth_rate=2.0)
            .build()
        )
        pop._initialize_session(seed=11)
        pop.register_hooks(apply_zygote, event="early")
        pop.run(1)

        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]
        session_fitness = np.asarray(  # pyright: ignore[union-attr]
            backend._session.get_tensor("zygote_viability_fitness")  # noqa: SLF001
        )
        # Flat (2, n_ztypes) row-major: female x ztype 1 is index 1.
        assert float(session_fitness[1]) == 0.25
