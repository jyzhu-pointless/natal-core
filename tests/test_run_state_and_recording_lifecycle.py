"""Run-state authority and recording lifecycle contracts (plan phase P7).

The pinned behaviors:

- Run state (tick, finished, failed) has a single authority: the native
  session.  Every STOP, finish, failure, and restore path is readable
  through ``pop.tick`` / ``pop.is_finished`` / ``pop.is_failed`` with no
  per-run Python flag copies; the retired ``_finished`` / ``_failed``
  attributes are gone.
- The native recording surfaces (observation selector, history
  ownership, checkpoint pruner) bind once per (population, History);
  re-run bindings are no-ops, runtime ``max_rows`` adjustment still
  prunes rows and checkpoints against the stably-bound pruner, and
  observation-mode recording keeps working on every run.
- ``get_adult_count`` reads a native aggregation that is exactly equal
  to the retired NumPy reductions, including fractional-count states
  and the recursive pairwise-split regime.

Each test names the concrete regression it would catch: a resurrected
Python flag copy drifting from the session, a re-entrant native call
from inside a hook, a lost recording binding after a re-run, or a
native adult reduction that diverges from the retired sums.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.tick_context import TickContext

if TYPE_CHECKING:
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )

AnyPopulation: TypeAlias = "AgeStructuredPopulation | DiscreteGenerationPopulation"
AnyBuilder: TypeAlias = "Callable[[str], AnyPopulation]"


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species (unique name per call site)."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _builder(name: str) -> nt.PopulationBuilder:
    """Return a deterministic discrete-generation builder (fixed point 10+10)."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), name=name, stochastic=False
        )
        .initial_state(individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )


def _age_builder(name: str) -> nt.PopulationBuilder:
    """Return a deterministic age-structured builder with fractional counts."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [7.5, 40.5, 2.25], "WT|Dr": [3.5, 10.25, 0.0]},
                "male": {"WT|WT": [1.5, 30.125, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=800.0)
    )


# ============================================================================
# Run-state authority: STOP / finish / failure / restore paths
# ============================================================================


def test_retired_flag_attributes_are_gone() -> None:
    """The retired private flags no longer exist on a built population.

    Catches a resurrection of the per-run flag copies: any code path that
    starts writing ``pop._finished`` / ``pop._failed`` again without the
    session authority would reintroduce a second status source that this
    negative contract rejects.
    """
    pop = _builder("RS7NoFlags").build()
    assert not hasattr(pop, "_finished")
    assert not hasattr(pop, "_failed")
    assert pop.is_finished is False
    assert pop.is_failed is False


def test_stop_path_derives_finished_and_freezes_native_tick() -> None:
    """A hook stop leaves the session Stopped with the tick frozen.

    Catches a derivation that reads a stale Python mirror: the tick and
    the finished marker must both come from the session right after the
    run, while the state cache is still stale (``pop.state`` is read
    only afterwards).
    """
    holder: dict[str, object] = {}

    def stop_first(ctx: TickContext) -> int:
        return ctx.stop()

    pop = _builder("RS7Stop").hooks(stop_first, event="early").build()
    holder["pop"] = pop
    pop.run(5)

    assert pop.tick == 0  # read before any state refresh: session authority
    assert pop.is_finished
    assert not pop.is_failed
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)


def test_nonzero_callback_return_marks_finished() -> None:
    """Any nonzero callback return stops the run and finishes the population.

    Catches a STOP path that failed to translate the native status into
    the derived finished surface.
    """

    def return_stop(ctx: TickContext) -> int:
        _ = ctx
        return 7

    pop = _builder("RS7Nonzero").hooks(return_stop, event="late").build()
    pop.run(4)

    assert pop.tick == 0
    assert pop.is_finished


def test_finish_simulation_marks_session_stopped_and_finish_hooks_see_it() -> None:
    """finish_simulation stops the session before the finish event fires.

    Catches an ordering regression where finish hooks would observe
    ``is_finished == False`` (stop applied only after the event), and a
    finish_simulation that left the session Ready (runs would not be
    refused).
    """
    observed: dict[str, bool] = {}
    holder: dict[str, object] = {}

    def finish_probe(ctx: TickContext) -> int:
        pop: AnyPopulation = holder["pop"]  # type: ignore[assignment]  # holder stores the built population
        observed["finished"] = pop.is_finished
        return 0

    pop2 = (
        _builder("RS7FinishSimHooked")
        .hooks(finish_probe, event="finish")
        .build()
    )
    holder["pop"] = pop2
    pop2.run(1, finish=True)

    assert observed["finished"] is True
    assert pop2.is_finished
    with pytest.raises(RuntimeError, match="has finished"):
        pop2.run(1)
    # A direct finish_simulation on an already-finished population raises.
    with pytest.raises(RuntimeError, match="already finished"):
        pop2.finish_simulation()


def test_finish_simulation_without_prior_run_locks_the_population() -> None:
    """finish_simulation before any run creates the session and stops it.

    Catches a lazy-session population whose finish marker stayed
    Python-side (``is_finished`` would read False and allow runs).
    """
    pop = _builder("RS7FinishCold").build()
    pop.finish_simulation()

    assert pop.is_finished
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)


def test_failed_hook_marks_session_failed_and_restore_clears_it() -> None:
    """A failing run locks runs via the session Failed status.

    Catches a failure path where the Python handler no longer mirrors a
    flag but the session never marks Failed (runs would silently retry),
    and a restore that does not carry the recorded status back.
    """

    def boom_from_tick_one(ctx: TickContext) -> int:
        if int(ctx.tick) >= 1:
            raise ValueError("hook boom")
        return 0

    pop = (
        _builder("RS7Fail")
        .record_history(mode="raw")
        .hooks(boom_from_tick_one, event="early")
        .build()
    )
    with pytest.raises(ValueError, match="hook boom"):
        pop.run(3, record_every=1)

    assert pop.is_failed
    assert not pop.is_finished
    with pytest.raises(RuntimeError, match="has failed"):
        pop.run(1)

    # The recorded boundary carries the Ready status recorded before the
    # failure; restoring it returns to a runnable state.
    pop.reset()
    assert not pop.is_failed
    pop.run(1)
    assert pop.tick == 1


def test_restore_carries_recorded_execution_status() -> None:
    """Restoring a stopped boundary restores the finished marker natively.

    Catches a restore path that reconciles flags in Python instead of
    reading the checkpoint's recorded status (a partial snapshot taken
    after a stop must restore as finished).
    """

    def stop_when_warm(ctx: TickContext) -> int:
        if int(ctx.tick) >= 1:
            return ctx.stop()
        return 0

    pop = (
        _builder("RS7RestoreStop")
        .record_history(mode="raw")
        .hooks(stop_when_warm, event="late")
        .build()
    )
    # record_every=2 leaves the frozen tick 1 unrecorded, so the manual
    # snapshot below adds the Stopped boundary instead of duplicating.
    pop.run(5, record_every=2)
    assert pop.is_finished
    assert pop.history.ticks == (0,)

    pop.record_snapshot()
    assert pop.history.ticks == (0, 1)
    assert pop.history.boundary_metadata[-1][2] == "Stopped"

    # Restore the partial (Stopped) boundary: the finished marker
    # returns with the checkpoint's recorded status.
    pop.restore_checkpoint(1)
    assert pop.is_finished
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)

    # Rolling back further restores the Ready boundary recorded before
    # the stop, and runs resume (a restore erases its future, so the
    # Ready boundary is reached from the Stopped one).
    pop.restore_checkpoint(0)
    assert not pop.is_finished
    pop.run(1, record_every=0)
    assert pop.tick == 1


def test_import_state_resets_the_derived_lifecycle() -> None:
    """import_state installs the tick natively and clears stopped/failed.

    Catches a session push that skipped the execution-status reset (the
    imported population would stay locked for runs).
    """

    pop = _builder("RS7Import").build()
    pop.run(3, finish=True)
    assert pop.is_finished

    pop.import_state(pop.export_state())
    assert pop.tick == 3
    assert not pop.is_finished
    pop.run(2)
    assert pop.tick == 5


def test_reset_returns_session_to_ready() -> None:
    """reset() clears both derived markers through the native boundary.

    Catches a reset that only cleared Python-side state while the
    session stayed Stopped/Failed (``is_finished`` would stay True).
    """

    def stop_second(ctx: TickContext) -> int:
        if int(ctx.tick) >= 1:
            return ctx.stop()
        return 0

    pop = _builder("RS7ResetStop").hooks(stop_second, event="late").build()
    pop.run(5)
    assert pop.is_finished

    pop.reset()
    assert pop.tick == 0
    assert not pop.is_finished
    assert not pop.is_failed
    pop.run(1)
    assert pop.tick == 1


def test_manual_stop_marks_session_stopped() -> None:
    """A manual STOP event (outside a run) locks the population.

    Catches a trigger_event path that left the native status Stopped but
    a Python gate still reading a never-set mirror flag.
    """
    pop = _builder("RS7ManualStop").build()
    result = pop.trigger_event("first", deme_id=0)
    native = pop._rust_lifecycle_backend
    assert native is not None
    native.stop()

    assert pop.is_finished
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)
    assert result == 0


def test_tick_inside_hook_is_the_native_event_tick() -> None:
    """``pop.tick`` inside a callback reads the event tick, not the mirror.

    Catches a derived tick that would re-enter the borrowed session from
    a callback (native borrow error) or read a stale fallback value.
    """
    holder: dict[str, Any] = {}
    observed: list[tuple[int, int]] = []

    def probe(ctx: TickContext) -> int:
        pop: AnyPopulation = holder["pop"]
        observed.append((int(ctx.tick), int(pop.tick)))
        return 0

    for build in (_builder("RS7HookTick"), _age_builder("RS7HookTickAge")):
        pop = build.hooks(probe, event="early").build()
        holder["pop"] = pop
        observed.clear()
        pop.run(3)
        assert observed == [(0, 0), (1, 1), (2, 2)]


def test_borrowed_session_guard_falls_back_without_native_calls() -> None:
    """While a run holds the borrow, lifecycle reads skip the session.

    Catches a lifecycle read that would re-enter the borrowed session
    (native borrow panic) instead of degrading to the fallback clock.
    """
    pop = _builder("RS7BorrowGuard").build()
    native_tick = pop.tick
    object.__setattr__(pop, "_rust_run_active", True)
    try:
        # No active event: the derived surface must not touch the session.
        assert pop.tick == native_tick
        assert pop.is_finished is False
        assert pop.is_failed is False
    finally:
        object.__setattr__(pop, "_rust_run_active", False)


def test_log_param_value_stamps_the_session_tick() -> None:
    """Parameter audit rows carry the session-owned tick after runs.

    Catches an audit sink still reading the retired mirror clock (rows
    would be stamped with a stale tick after the first run).
    """
    pop = _builder("RS7AuditTick").build()
    pop.run(3)
    pop.update().competition(carrying_capacity=12345.0)
    rows = pop.params_log
    assert rows
    assert rows[-1][0] == 3


# ============================================================================
# Managed spatial demes: shared-session projection
# ============================================================================


def _spatial(name: str, *, stop_deme0: bool = True) -> tuple[Any, Any, Any]:
    """Return a 2-deme discrete spatial population, optionally stopping on deme 0."""
    del name

    def stop_on_deme_zero(ctx: TickContext) -> int:
        if int(ctx.deme_id) == 0:
            return ctx.stop()
        return 0

    species = _species("RS7SpatialShared")
    demes = []
    for i in range(2):
        chain = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name=f"rs7_d{i}", stochastic=False
            )
            .initial_state(individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        )
        if i == 0 and stop_deme0:
            chain = chain.hooks(stop_on_deme_zero, event="early")
        demes.append(chain.build())
    spatial = nt.SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)  # noqa: SLF001 — test wires the owning session explicitly
    return spatial, spatial.demes[0], spatial.demes[1]


def test_spatial_demes_project_the_shared_session_status() -> None:
    """Every deme derives finished/failed/tick from the owning session.

    Catches a reintroduced per-deme flag broadcast: after a stopped run
    each deme must report the shared status through its read channel,
    and reset must clear it natively.
    """
    spatial, d0, d1 = _spatial("RS7SpatialSharedStatus")
    assert not d0.is_finished and not d1.is_finished

    spatial.run(5, record_every=1)

    assert spatial.tick == 0
    assert d0.is_finished and d1.is_finished
    assert d0.tick == d1.tick == 0

    spatial.reset()
    assert not d0.is_finished and not d1.is_finished
    assert d0.tick == d1.tick == 0

    spatial.run(2, finish=True)
    assert d0.is_finished and d1.is_finished
    backend = spatial._rust_spatial_backend
    assert backend is not None
    assert backend.execution_state()[0] == "Stopped"
    with pytest.raises(RuntimeError, match="has finished"):
        spatial.run(1)


def test_spatial_deme_tick_reads_the_shared_clock_while_dirty() -> None:
    """``deme.tick`` answers from the session before any state refresh.

    Catches a deme clock that still depended on the container publishing
    mirrored tick metadata (the read would return a stale value while
    the deme state cache is dirty).
    """
    spatial, d0, d1 = _spatial("RS7SpatialDirtyTick", stop_deme0=False)
    spatial.run(3, record_every=1)

    # The run marked every deme cache dirty; the tick must still answer.
    assert spatial.tick == 3
    assert d0.tick == 3
    assert d1.tick == 3


# ============================================================================
# One-time recording binding (history retention lifecycle)
# ============================================================================


def test_recording_surfaces_bind_once_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-runs skip the observation/history/pruner binding entirely.

    Catches a per-run re-binding regression (the phase P6 behavior):
    after the first recording boundary, later runs must not re-install
    the frozen observation selector or re-hand history ownership.
    """
    from natal.frontend.output.history import History

    pop = _builder("RS7BindOnce").record_history(mode="raw").build()
    calls = {"configure": 0, "bind": 0, "pruner": 0}

    original_configure = History._configure_observation
    original_pruner = History._bind_checkpoint_pruner

    def spy_configure(self: History, observation: Any) -> None:
        calls["configure"] += 1
        original_configure(self, observation)

    def spy_pruner(self: History, prune: Callable[[int], None]) -> None:
        calls["pruner"] += 1
        original_pruner(self, prune)

    backend = pop._rust_lifecycle_backend
    assert backend is not None
    original_bind = type(backend).bind_history

    def spy_bind(session: Any, store: Any, log: Any) -> None:
        calls["bind"] += 1
        original_bind(session, store, log)

    monkeypatch.setattr(History, "_configure_observation", spy_configure)
    monkeypatch.setattr(History, "_bind_checkpoint_pruner", spy_pruner)
    monkeypatch.setattr(type(backend), "bind_history", spy_bind)

    pop.record_snapshot()
    assert calls == {"configure": 0, "bind": 1, "pruner": 1}

    pop.run(1, record_every=1)
    pop.run(2, record_every=1)
    pop.trigger_event("first", deme_id=0)
    # Raw mode never configures an observation; the store/pruner binding
    # happened exactly once and later boundaries are no-ops.
    assert calls == {"configure": 0, "bind": 1, "pruner": 1}
    assert pop.history.ticks == (0, 1, 2, 3)


def test_runs_record_continuously_with_updates_between() -> None:
    """run → update → run keeps recording without boundary errors.

    Catches a stale binding after a pause-phase update (the second run
    would drop rows or raise a duplicate-boundary error).
    """
    pop = _builder("RS7RunRunRun").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    pop.update().competition(carrying_capacity=99999.0)
    pop.run(2, record_every=1)
    pop.update().reproduction(eggs_per_female=3, sex_ratio=0.5)
    pop.run(2, record_every=1)

    assert pop.history.ticks == (0, 1, 2, 3, 4, 5, 6)
    assert len(pop.history) == 7


def test_max_rows_adjustment_prunes_rows_and_checkpoints() -> None:
    """A mid-life max_rows change prunes rows and their checkpoints.

    Catches a broken pruner pairing: the checkpoint release must fire
    through the stably-bound pruner, so pruned ticks stop being
    restorable while surviving ones keep restoring.
    """
    pop = _builder("RS7MaxRows").record_history(mode="raw").build()
    pop.run(5, record_every=1)
    assert pop.history.ticks == (0, 1, 2, 3, 4, 5)

    pop.history.max_rows = 2
    assert pop.history.ticks == (4, 5)

    with pytest.raises(ValueError, match="not found in history"):
        pop.restore_checkpoint(2)
    pop.restore_checkpoint(4)
    assert pop.tick == 4


def test_clear_history_pairs_rows_with_checkpoints() -> None:
    """clear_history drops rows and checkpoints together.

    Catches a clear path that left session checkpoints restorable after
    the rows were dropped (a cleared tick would resurrect).
    """
    pop = _builder("RS7Clear").record_history(mode="raw").build()
    pop.run(3, record_every=1)
    pop.clear_history()
    assert pop.history.is_empty

    with pytest.raises(ValueError, match="No history available"):
        pop.restore_checkpoint(2)

    pop.run(2, record_every=1)
    assert pop.history.ticks == (3, 4, 5)


def test_reinstalled_history_rebinds_on_next_boundary() -> None:
    """A fresh History object is bound again; the identity tracks it.

    Catches an identity cache that survives a History reinstall (rows
    would silently stop recording into the new container).
    """
    pop = _builder("RS7Rebind").record_history(mode="raw").build()
    pop.run(2, record_every=1)
    assert pop.history.ticks == (0, 1, 2)

    from natal.frontend.output.history import History, HistorySchema

    old = pop.history
    schema: HistorySchema = old.schema
    fresh = History(schema, max_rows=old.max_rows)
    pop._history_obj = fresh  # noqa: SLF001 — reinstalling the container is the scenario
    binding = pop._history_binding  # noqa: SLF001
    assert binding is not None and binding[0] is old

    pop.run(1, record_every=1)
    assert fresh.ticks == (2, 3)
    assert pop.history is fresh


# ============================================================================
# Observation-mode recording across runs (run-boundary reinstall dedup)
# ============================================================================


def test_observation_mode_records_on_every_run() -> None:
    """Second-run recording works with the observation selector bound once.

    Catches both the retired per-run reinstall and a wrong skip that
    would leave later runs recording empty or raw rows.
    """
    from natal.frontend.patterns import IndividualSelector

    groups: dict[str, IndividualSelector] = {
        "wild": IndividualSelector(ztype="WT|WT"),
    }
    pop = (
        _age_builder("RS7ObsRuns")
        .with_observation(groups=groups, collapse_age=True)
        .record_history(mode="observation")
        .build()
    )
    assert pop.history.schema.mode == "observation"

    pop.run(2, record_every=1)
    first = pop.history.values.copy()
    pop.run(2, record_every=1)
    second = pop.history.values.copy()

    assert len(pop.history) == 5  # ticks 0..4 recorded
    assert pop.history.ticks == (0, 1, 2, 3, 4)
    assert first.shape[0] == 3 and second.shape[0] == 5
    # Observation rows carry group sums, not raw state widths.
    assert second.shape[1:] == first.shape[1:]
    # The recorded projection matches a live observation of the same state.
    live = pop.observe()
    np.testing.assert_allclose(second[-1], live.values, rtol=0, atol=1e-12)


# ============================================================================
# Native adult counts
# ============================================================================


def _retired_adult_sum(pop: AnyPopulation, sex: str) -> int:
    """Recompute the retired Python reduction for *sex* from the snapshot."""
    ic = pop.state.individual_count
    adult_start = int(pop.config.new_adult_age)
    total = 0
    if sex in ("female", "F", "both"):
        total += float(ic[0, adult_start:, :].sum())
    if sex in ("male", "M", "both"):
        total += float(ic[1, adult_start:, :].sum())
    return int(total)


@pytest.mark.parametrize("sex", ["female", "male", "both", "F", "M"])
def test_native_adult_count_equals_retired_numpy_sums(sex: str) -> None:
    """The native adult reduction equals the retired sums exactly.

    Fractional counts make any summation-order or truncation divergence
    fail this equality; checked on the fresh build, after runs, and
    after an update that changed survival.
    """
    pop = _age_builder("RS7AdultNative").build()
    assert pop.get_adult_count(sex) == _retired_adult_sum(pop, sex)

    pop.run(1, record_every=0)
    assert pop.get_adult_count(sex) == _retired_adult_sum(pop, sex)

    pop.run(3, record_every=0)
    assert pop.get_adult_count(sex) == _retired_adult_sum(pop, sex)

    pop.update().survival(female_age_based_survival=[1.0, 0.5, 0.5])
    pop.run(1, record_every=0)
    assert pop.get_adult_count(sex) == _retired_adult_sum(pop, sex)


def test_native_adult_count_falls_back_to_local_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a session the adult count still comes from the container.

    Catches a wiring that unconditionally requires a backend and crashes
    for populations that have not initialized a session.
    """
    pop = _age_builder("RS7AdultFallback").build()
    expected = _retired_adult_sum(pop, "both")
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", None)
    assert pop.get_adult_count("both") == expected


def test_native_adult_count_avoids_full_state_pulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adult query never pulls the full state snapshot.

    Catches a retired-path regression where the adult count stale-read
    would export the whole state array for one scalar.
    """
    pop = _age_builder("RS7AdultNoPull").build()
    pop.run(1, record_every=0)
    backend = pop._rust_lifecycle_backend
    assert backend is not None
    calls = {"state": 0}
    backend_cls = type(backend)
    original_state = backend_cls.state_snapshot

    def spy_state(session: Any) -> Any:
        calls["state"] += 1
        return original_state(session)

    monkeypatch.setattr(backend_cls, "state_snapshot", spy_state)
    _ = pop.get_adult_count("both")
    _ = pop.get_adult_count("female")
    assert calls == {"state": 0}

    _ = pop.state  # the stale-cache path still pulls, proving the spy works
    assert calls == {"state": 1}


def test_native_adult_count_bit_exact_through_recursive_split() -> None:
    """Adult sums stay bit-identical when the adult slice exceeds 128.

    The pairwise reduction only recurses above NumPy's 128-wide block;
    this fixture crafts magnitude-mixed adult payloads so a summation
    order change is observable, then pins the native equality.
    """
    n_ages = 80
    rng = np.random.default_rng(778)
    flat = np.empty(2 * n_ages * 3, dtype=np.float64)
    flat[0::2] = 1e15 * (1.0 + 1e-7 * rng.random(flat[0::2].size))
    flat[1::2] = 1e4 * (1.0 + 1e-3 * rng.random(flat[1::2].size))
    ic = flat.reshape(2, n_ages, 3)

    pop = (
        nt.AgeStructuredPopulation.setup(
            species=_species("RS7AdultBigPlane"), stochastic=False
        )
        .age_structure(n_ages=n_ages, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [10.0] * n_ages},
                "male": {"WT|WT": [10.0] * n_ages},
            }
        )
        .survival(female_age_based_survival=[1.0] * n_ages, male_age_based_survival=[1.0] * n_ages)
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=0, carrying_capacity=8000.0)
        .build()
    )
    pop.run(1, record_every=0)
    imported = pop.state._replace(individual_count=ic)  # pyright: ignore[reportAttributeAccessIssue]  # NamedTuple state container
    pop.import_state(imported)

    adult_slice = ic[:, 1:, :]
    expected = int(float(adult_slice[0].sum()) + float(adult_slice[1].sum()))
    # Data precheck: the payload is summation-order sensitive, so an
    # accumulator-layout change could not pass vacuously (NumPy pairwise
    # vs plain left-to-right sequential summation disagree here).
    flat_adult = adult_slice.reshape(-1)
    assert float(adult_slice.sum()) != float(builtins.sum(iter(flat_adult)))
    assert pop.get_adult_count("both") == expected
    assert pop.get_adult_count("female") == int(float(ic[0, 1:, :].sum()))
    assert pop.get_adult_count("male") == int(float(ic[1, 1:, :].sum()))


def test_degenerate_backend_degrades_to_fallback_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session adapter without lifecycle reads degrades instead of crashing.

    Catches an unconditional native call on a duck-typed backend double
    (the runtime writer channel in degraded test hosts carries neither
    ``execution_state`` nor ``current_tick``).
    """
    pop = _builder("RS7DegenerateBackend").build()
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", object())
    assert pop.tick == 0  # fallback clock: the field installed at build
    assert pop.is_finished is False
    assert pop.is_failed is False


def test_observe_fallback_stamps_the_derived_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session-less observe() path stamps the derived tick.

    Catches a projection fallback that still read the retired mirror
    clock (the tick on projected results would drift after runs).
    """
    pop = _builder("RS7ObserveFallback").build()
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", None)
    result = pop.observe()
    assert result.tick == pop.tick == 0
    # Identity observation: the projected values sum to the seeded census.
    assert float(result.values.sum()) == 20.0


def test_finish_simulation_creates_the_session_when_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session-less population creates its session and stops it natively.

    Catches a cold finish that left the finished marker Python-side (the
    derived surface would keep answering False and allow runs).
    """
    pop = _builder("RS7FinishColdSession").build()
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", None)
    pop.finish_simulation()

    backend = pop._rust_lifecycle_backend
    assert backend is not None
    assert backend.execution_state()[0] == "Stopped"
    assert pop.is_finished
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)


def test_discrete_repr_reads_the_derived_status() -> None:
    """The repr's status word follows the session, not a retired flag.

    Catches a repr that would report ``Active`` for a session-stopped
    population (or crash on the removed attribute).
    """
    pop = _builder("RS7Repr").build()
    assert "status=Active" in repr(pop)
    pop.run(2, finish=True)
    assert "status=Finished" in repr(pop)


def test_adult_count_rejects_unknown_sex() -> None:
    """The sex identifier contract is unchanged on the native path."""
    pop = _age_builder("RS7AdultSex").build()
    with pytest.raises(ValueError, match="sex must be"):
        pop.get_adult_count("drones")
