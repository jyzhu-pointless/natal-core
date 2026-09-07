"""Session-owned state contracts (plan S2, batch 23).

The plain-model Rust sessions own the counts, sperm storage, and tick:
``run`` accepts control parameters only, Python reads state back through
snapshots, and the old per-run full-state handoff is gone.  These tests
pin the ownership surface and the negative contracts over the retired
handoff signatures.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal import _engine_rs
from natal.frontend.data import DiscretePopulationState, PopulationState
from natal.frontend.hooks.tick_context import TickContext


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
        pop.enable_rust_backend(seed=3)
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
        pop.enable_rust_backend(seed=3)
        pop.run(2)
        snapshot = pop.state.individual_count

        backend = pop._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # ownership surface probe
        tick, ind_flat = backend.state_snapshot()  # type: ignore[union-attr]  # enabled above
        np.testing.assert_array_equal(snapshot.reshape(-1), ind_flat)
        assert pop.tick == int(tick)

    def test_mutating_the_cache_never_reaches_the_session(self) -> None:
        """A tampered Python cache is invisible to the session-run path."""
        pop = _build_discrete("OwnTamper")
        pop.enable_rust_backend(seed=3)
        pop.run(2)
        pop._state.individual_count[...] = 999.0  # pyright: ignore[reportPrivateUsage]  # tamper the cache
        snapshot = pop.state.individual_count
        assert float(snapshot.sum()) != 999.0 * snapshot.size

    def test_import_state_installs_into_the_session(self) -> None:
        """import_state becomes the session state; reruns continue from it."""
        pop = _build_discrete("OwnImport")
        pop.enable_rust_backend(seed=3)
        pop.run(2)
        pop.import_state(pop.export_state())
        # export_state carries the tick, so the import rebases to tick 2
        # and the session runs three more from there.
        assert pop.tick == 2
        pop.run(3)
        assert pop.tick == 5

    def test_disable_pulls_the_session_state_back(self) -> None:
        """Disabling keeps every count and the tick on the reference path."""
        pop = _build_discrete("OwnDisable")
        pop.enable_rust_backend(seed=3)
        pop.run(2)
        expected = pop.state.individual_count.copy()
        pop.disable_rust_backend()
        np.testing.assert_array_equal(pop.state.individual_count, expected)
        assert pop.tick == 2

    def test_refresh_rebuild_keeps_state_round_trip(self) -> None:
        """refresh_rust_backend captures the old session state first."""
        pop = _build_discrete("OwnRefresh")
        pop.enable_rust_backend(seed=3)
        pop.run(3)
        before = pop.state.individual_count.copy()
        pop.refresh_rust_backend()
        # The rebuilt session starts from the pre-refresh state (only the
        # RNG reseeded to the same original seed).
        pop.run(1)
        assert pop.tick == 4
        assert not np.array_equal(pop.state.individual_count, before)

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
    one seeded session RNG: ``enable_rust_backend(seed=S)`` starts the
    stream at *S*, every run call keeps consuming it, and only a backend
    rebuild (``refresh_rust_backend`` / re-``enable``) resets it.  State
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
        split.enable_rust_backend(seed=13)
        split.run(1)
        split.run(1)

        atomic = _build_age("RngBatch", stochastic=True)
        atomic.enable_rust_backend(seed=13)
        atomic.run(2)

        np.testing.assert_array_equal(
            split.state.individual_count, atomic.state.individual_count
        )
        np.testing.assert_array_equal(
            split.state.sperm_storage, atomic.state.sperm_storage
        )
        assert split.tick == 2
        assert atomic.tick == 2

    def test_refresh_after_run_diverges_from_the_continuation(self) -> None:
        """refresh rebuilds the session: the RNG restarts while the state continues.

        After ``run(1)`` (state at tick 1) a same-seed refresh hands the
        tick-1 state to a brand-new stream that starts at seed 13 again,
        whereas a fresh ``run(2)`` plays the same stream against the tick-0
        state.  The two end states are bitwise different: the rebuild did
        not replay the first tick's draws.
        """
        pop = _build_age("RngRefresh", stochastic=True)
        pop.enable_rust_backend(seed=13)
        pop.run(1)
        pop.refresh_rust_backend()
        pop.run(1)

        fresh = _build_age("RngRefreshRef", stochastic=True)
        fresh.enable_rust_backend(seed=13)
        fresh.run(2)

        assert pop.tick == 2
        assert fresh.tick == 2
        assert not np.array_equal(
            pop.state.individual_count, fresh.state.individual_count
        )

    def test_re_enable_matches_refresh_and_differs_from_continuation(self) -> None:
        """A second same-seed ``enable`` is a rebuild, not a no-op.

        ``enable(seed=13)`` after ``run(1)`` must behave exactly like
        ``refresh_rust_backend()`` (state preserved, RNG restarted): the
        two end states are bitwise equal and both differ from the pure
        continuation.
        """
        pop = _build_age("RngReenable", stochastic=True)
        pop.enable_rust_backend(seed=13)
        pop.run(1)
        pop.enable_rust_backend(seed=13)
        pop.run(1)

        refreshed = _build_age("RngReenableRef", stochastic=True)
        refreshed.enable_rust_backend(seed=13)
        refreshed.run(1)
        refreshed.refresh_rust_backend()
        refreshed.run(1)

        continuation = _build_age("RngReenableCont", stochastic=True)
        continuation.enable_rust_backend(seed=13)
        continuation.run(2)

        np.testing.assert_array_equal(
            pop.state.individual_count, refreshed.state.individual_count
        )
        np.testing.assert_array_equal(
            pop.state.sperm_storage, refreshed.state.sperm_storage
        )
        assert not np.array_equal(
            pop.state.individual_count, continuation.state.individual_count
        )
        assert pop.tick == 2

    def test_disable_re_enable_loses_the_rng_stream(self) -> None:
        """The stream is session-owned: disabling discards it.

        ``enable(seed=13); run(1); disable; enable(seed=13); run(1)`` ends
        at tick 2 but draws the seed-13 stream's first segment against the
        tick-1 state, so it cannot reproduce the atomic ``run(2)``.  A
        cached/carry-over RNG would reproduce it exactly.
        """
        pop = _build_age("RngDisable", stochastic=True)
        pop.enable_rust_backend(seed=13)
        pop.run(1)
        pop.disable_rust_backend()
        pop.enable_rust_backend(seed=13)
        pop.run(1)

        fresh = _build_age("RngDisableRef", stochastic=True)
        fresh.enable_rust_backend(seed=13)
        fresh.run(2)

        assert pop.tick == 2
        assert fresh.tick == 2
        assert not np.array_equal(
            pop.state.individual_count, fresh.state.individual_count
        )

    def test_state_rollback_keeps_stream_but_checkpoint_rollback_rewinds(self) -> None:
        """set_state moves state alone; only a checkpoint moves both.

        After ``run(2)``, pushing the blueprint tick-0 state back in (a
        ``set_state``-equivalent state rollback) and running two more ticks
        does NOT reproduce the original trajectory: the RNG kept streaming.
        The control — a full record-aligned checkpoint rollback to tick 0
        (state + RNG words) — replays the atomic ``run(2)`` bitwise.
        """
        pop = _build_age("RngState", stochastic=True)
        pop.enable_rust_backend(seed=17)
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
        control.enable_rust_backend(seed=17)
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
        test.enable_rust_backend(seed=0)
        # The manual event runs the callback and must flush its write.
        assert test.trigger_event("finish") == 0
        backend = test._rust_lifecycle_backend  # pyright: ignore[reportPrivateUsage]  # session probe
        assert backend is not None
        tick, ind_flat, _sperm_flat = backend.state_snapshot()
        assert tick == 0
        assert float(ind_flat.reshape((2, 3, 3))[0, 1, 0]) == 23.0
        test.run(1)

        control = _build_age("FlushCtl")
        control.enable_rust_backend(seed=0)
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
        pop.enable_rust_backend(seed=2)
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
        pop.enable_rust_backend(seed=0)
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
        plain.enable_rust_backend(seed=0)
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
        pop.enable_rust_backend(seed=2)
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
        pop.enable_rust_backend(seed=2)
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
        pop.enable_rust_backend(seed=2)
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
        pop.enable_rust_backend(seed=5)
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
