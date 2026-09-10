"""Independent restore-boundary and storage regressions from plan review."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from tests.test_review_runtime_regressions import _population

HistoryModel = Literal["discrete", "age", "spatial"]
HistoryPopulation = nt.DiscreteGenerationPopulation | nt.AgeStructuredPopulation | nt.SpatialPopulation


def _history_population(
    name: str, model: HistoryModel, hook_calls: list | None = None
) -> HistoryPopulation:
    """Build raw-history populations with deterministic, neutral genetics."""
    if model != "spatial":
        return _population(name, model, stochastic=False, hook_calls=hook_calls)
    species = nt.Species.from_dict(name=name, structure={"chr1": {"loc": ["WT", "Dr"]}})
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(stochastic=False)
        .initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .record_history(mode="raw")
    )
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


def _state(pop: HistoryPopulation) -> NDArray[np.float64]:
    """Copy complete exposed biological state for atomicity comparisons."""
    if isinstance(pop, nt.SpatialPopulation):
        return np.concatenate([deme.export_state() for deme in pop.demes])
    return pop.export_state().copy()


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_manual_snapshot_creates_a_restorable_boundary(model: HistoryModel) -> None:
    """A snapshot taken without automatic recording must include a checkpoint."""
    pop = _history_population(f"ReviewManualBoundary_{model}", model)
    pop.run(1, record_every=0)
    expected = _state(pop)
    pop.record_snapshot()
    pop.run(1, record_every=0)
    pop.restore_checkpoint(1)
    assert pop.tick == 1
    np.testing.assert_array_equal(_state(pop), expected)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_unrecorded_tick_is_rejected_atomically(model: HistoryModel) -> None:
    """Sparse recording cannot silently round a requested tick downward."""
    pop = _history_population(f"ReviewExactBoundary_{model}", model)
    pop.run(4, record_every=2)
    before = _state(pop)
    ticks = pop.history.ticks
    with pytest.raises(ValueError):
        pop.restore_checkpoint(3)
    assert pop.tick == 4
    assert pop.history.ticks == ticks
    np.testing.assert_array_equal(_state(pop), before)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_next_run_does_not_overwrite_an_existing_boundary(model: Literal["discrete", "age"]) -> None:
    """A run boundary already recorded retains its original ecology snapshot."""
    pop = _history_population(f"ReviewDuplicateBoundary_{model}", model)
    pop.run(1, record_every=1)
    pop.update().competition(carrying_capacity=999.0)
    pop.run(1, record_every=1)
    pop.restore_checkpoint(1)
    assert pop.params.carrying_capacity == 100000.0


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_restore_discards_parameter_log_entries_from_future(model: Literal["discrete", "age"]) -> None:
    """Parameter history after restore describes only the surviving timeline."""
    pop = _population(f"ReviewLogTimeline_{model}", model, stochastic=False)
    pop.run(2, record_every=1)
    pop.update().competition(carrying_capacity=999.0)
    assert any(entry[0] == 2 for entry in pop.params_log)
    pop.restore_checkpoint(0)
    assert pop.params_log == ()
    assert pop.params.carrying_capacity == 100000.0
    pop.run(1)
    pop.update().competition(carrying_capacity=777.0)
    assert all(entry[0] <= pop.tick for entry in pop.params_log)
    assert all(entry[3] != 999.0 for entry in pop.params_log)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_bounded_history_eviction_rejects_restore_without_mutating_state(model: HistoryModel) -> None:
    """Every evicted row loses its checkpoint while retained rows still restore."""
    pop = _history_population(f"ReviewBoundedHistory_{model}", model)
    pop.history.max_rows = 2
    pop.run(10, record_every=1)
    assert pop.history.ticks == (9, 10)
    before = _state(pop)
    with pytest.raises(ValueError):
        pop.restore_checkpoint(8)
    assert pop.tick == 10
    np.testing.assert_array_equal(_state(pop), before)
    pop.restore_checkpoint(9)
    assert pop.tick == 9
    pop.run(1, record_every=1)
    np.testing.assert_array_equal(_state(pop), before)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_old_history_result_survives_clear_and_storage_reuse(model: HistoryModel) -> None:
    """An exported history array cannot alias storage reused after clear."""
    pop = _history_population(f"ReviewHistoryReuse_{model}", model)
    pop.history.max_rows = 2
    pop.run(2, record_every=1)
    exported = pop.history.individual_count
    expected = exported.copy()
    pop.clear_history()
    pop.run(3, record_every=1)
    np.testing.assert_array_equal(exported, expected)
    assert pop.history.ticks == (4, 5)
    with pytest.raises(ValueError):
        pop.restore_checkpoint(2)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_native_run_keeps_bounded_rows_without_returning_a_history_batch(
    model: Literal["discrete", "age"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long runs retain only the budgeted native rows without exporting all rows."""
    from natal import _engine_rs
    from natal.backends.rust.rust_backend import (
        RustDiscreteLifecycleBackend,
        RustLifecycleBackend,
    )

    pop = _population(f"ReviewNativeBound_{model}", model, stochastic=False)
    pop.history.max_rows = 2
    backend = pop._rust_lifecycle_backend
    assert backend is not None
    original = type(backend).run
    exported_sizes: list[int] = []

    def track_run(
        self: RustLifecycleBackend | RustDiscreteLifecycleBackend,
        n_steps: int,
        record_every: int = 0,
        observation_mask: NDArray[np.float64] | None = None,
        checkpoint_every: int = 0,
    ) -> tuple[int, NDArray[np.float64], bool]:
        """Measure the actual production return value without replacing execution."""
        result = original(self, n_steps, record_every, observation_mask, checkpoint_every)
        exported_sizes.append(result[1].size)
        return result

    monkeypatch.setattr(type(backend), "run", track_run)
    pop.run(100, record_every=1)
    assert exported_sizes and all(size == 0 for size in exported_sizes)
    assert isinstance(pop.history._store, _engine_rs.HistoryStore)
    assert len(pop.history._store) == 2
    assert pop.history.ticks == (99, 100)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_history_projection_runs_natively_with_exact_identity_values(
    model: HistoryModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """History projection never loops through the Python observation executor."""
    from natal.frontend.output.observation import Observation

    pop = _history_population(f"ReviewNativeProjection_{model}", model)
    pop.run(3, record_every=1)
    expected = np.moveaxis(pop.history.individual_count, -1, 1)

    def reject_python_projection(
        self: Observation, individual_count: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Reject the retired Python per-row projection path."""
        raise AssertionError("History projection called Python Observation.apply")

    monkeypatch.setattr(Observation, "apply", reject_python_projection)
    observed = pop.history.observe(pop.observation)
    assert observed.ticks == (0, 1, 2, 3)
    np.testing.assert_array_equal(observed.values, expected)


def test_spatial_batch_without_callbacks_does_not_materialize_python_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control-only spatial runs must not copy the complete state on each tick."""
    pop = _history_population("ReviewSpatialNoStateHandoff", "spatial")
    assert isinstance(pop, nt.SpatialPopulation)
    pop.history.max_rows = 2
    backend = pop._rust_spatial_backend
    assert backend is not None

    def reject_state_transfer() -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        """Detect a full state query made internally by the simulation loop."""
        raise AssertionError("Spatial run copied complete state into Python")

    monkeypatch.setattr(backend, "state_snapshot", reject_state_transfer)
    pop.run(100, record_every=1)
    assert pop.tick == 100
    assert pop.history.ticks == (99, 100)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_current_observation_does_not_export_raw_state(
    model: HistoryModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial identity observations retain all 200 individuals per deme natively."""
    pop = _history_population(f"ReviewCurrentObservation_{model}", model)
    backend = pop._rust_spatial_backend if isinstance(pop, nt.SpatialPopulation) else pop._rust_lifecycle_backend
    assert backend is not None

    def reject_state_transfer() -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        """Catch a hidden state export followed by an observation import."""
        raise AssertionError("Current observation exported complete state")

    monkeypatch.setattr(backend, "state_snapshot", reject_state_transfer)
    result = pop.observe()
    assert result.tick == 0
    # Identity selectors partition the initial 100 females and 100 males.
    assert result.values.sum() == (400 if model == "spatial" else 200)


def test_typed_native_parameter_log_owns_values_and_keeps_legacy_projection() -> None:
    """The complete log preserves types, additions, removals, and buffer isolation."""
    from natal._engine_rs import ParameterLog

    log = ParameterLog()
    original = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2)
    changed = original + 100
    log.append_value(2, "tensor", original, changed, "early", 1)
    log.append_value(2, "custom.flag", True, False, "early", 1)
    log.append_value(2, "custom.count", 7, 9, "early", 1)
    log.append_value(2, "custom.added", None, 3, "early", 1)
    log.append_value(2, "custom.deleted", 3, None, "early", 1)
    log.append_value(2, "unchanged", 4.0, 4.0, "early", 1)
    original.fill(-1)
    changed.fill(-2)
    details = log.details()
    assert len(details) == 5
    assert details[0][:4] == (2, "early", 1, "tensor")
    old, new = details[0][4:]
    assert isinstance(old, np.ndarray) and isinstance(new, np.ndarray)
    np.testing.assert_array_equal(old, np.arange(24).reshape(2, 3, 2, 2))
    np.testing.assert_array_equal(new, np.arange(24).reshape(2, 3, 2, 2) + 100)
    old.fill(0)
    np.testing.assert_array_equal(log.details()[0][4], np.arange(24).reshape(2, 3, 2, 2))
    assert type(details[1][4]) is bool and type(details[2][4]) is int
    assert details[3][4:] == (None, 3)
    assert details[4][4:] == (3, None)
    assert log.snapshot() == [(2, "custom.flag", 1.0, 0.0), (2, "custom.count", 7.0, 9.0)]
    mark = log.mark()
    with pytest.raises(TypeError):
        log.append_value(2, "invalid", None, "unsupported", "update", 0)  # type: ignore[arg-type]
    assert log.mark() == mark
    log.append_value(3, "temporary", 1.0, 2.0, "update", 0)
    log.rollback(mark)
    assert log.mark() == mark
    with pytest.raises(ValueError):
        log.rollback(mark + 1)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_stopped_snapshot_retains_phase_and_does_not_unlock_execution(model: HistoryModel) -> None:
    """A partial early-phase boundary must not masquerade as a Ready tick."""
    from natal.frontend.hooks.tick_context import TickContext

    def stop(ctx: TickContext) -> int:
        """Stop before any lifecycle phase changes the biological state."""
        ctx.stop()
        return 0

    pop = _history_population(
        f"ReviewStoppedBoundary_{model}",
        model,
        hook_calls=[((stop,), {"event": "early"})],
    )
    pop.run(1, record_every=0)
    pop.clear_history()
    pop.record_snapshot()
    assert pop.history.boundary_metadata == ((0, 2, "Stopped"),)
    pop.restore_checkpoint(0)
    assert pop.history.boundary_metadata == ((0, 2, "Stopped"),)
    with pytest.raises(RuntimeError):
        pop.run(1)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_public_typed_parameter_commits_are_logged_and_restore_exact_cursor(
    model: Literal["discrete", "age"],
) -> None:
    """Tensor/custom commits after a same-tick snapshot belong to its future."""
    pop = _population(f"ReviewPublicTypedAudit_{model}", model)
    pop.record_snapshot()
    original = pop.params.viability_fitness.array.copy()
    pop.params.tensor_write("viability_fitness", original * 0.5)
    pop.update().custom(grid=np.full((2, 2, 3), 19.0), added=13, flag=False)
    details = pop.params_log_details
    by_name = {entry[3]: entry for entry in details}
    assert by_name["viability_fitness"][:4] == (0, "update", 0, "viability_fitness")
    np.testing.assert_array_equal(by_name["viability_fitness"][4], original)
    np.testing.assert_array_equal(by_name["viability_fitness"][5], original * 0.5)
    np.testing.assert_array_equal(by_name["custom.grid"][5], np.full((2, 2, 3), 19.0))
    assert by_name["custom.added"][4:] == (None, 13)
    assert by_name["custom.flag"][4:] == (True, False)
    assert all(entry[1] not in ("viability_fitness", "custom.grid", "custom.added") for entry in pop.params_log)
    # A rejected routed tensor must not enter the audit, even at the same tick.
    size = len(details)
    with pytest.raises(ValueError):
        pop.params.tensor_write("viability_fitness", original * -1)
    assert len(pop.params_log_details) == size
    pop.restore_checkpoint(0)
    assert pop.params_log_details == ()
    assert "added" not in pop.config.custom
    # Genetics are deliberately outside checkpoint rollback; the audit still
    # describes the surviving timeline rather than pretending the write is new.
    np.testing.assert_array_equal(pop.params.viability_fitness.array, original * 0.5)
    assert by_name["custom.added"][4:] == (None, 13)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_hook_typed_log_is_committed_with_event_or_discarded_atomically(
    model: Literal["discrete", "age"],
) -> None:
    """Successful callbacks preserve typed provenance; failed callbacks log nothing."""
    from natal.frontend.hooks.tick_context import TickContext

    should_fail = [False]

    def update(ctx: TickContext) -> int:
        """Stage an owned custom tensor followed by an optional failure."""
        ctx.update().custom(grid=np.full((2, 2, 3), 21.0), added=5)
        if should_fail[0]:
            raise ValueError("typed audit failure")
        return 0

    pop = _population(f"ReviewCallbackTypedAudit_{model}", model, callback=update)
    pop.run(1)
    entries = {row[3]: row for row in pop.params_log_details}
    assert entries["custom.grid"][:4] == (0, "first", 0, "custom.grid")
    np.testing.assert_array_equal(entries["custom.grid"][5], np.full((2, 2, 3), 21.0))
    assert entries["custom.added"][4:] == (None, 5)
    should_fail[0] = True
    failed = _population(f"ReviewCallbackTypedAuditFailure_{model}", model, callback=update)
    with pytest.raises(ValueError, match="typed audit failure"):
        failed.run(1)
    assert failed.params_log_details == ()
    assert "added" not in failed.config.custom


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_compiled_genetic_commit_logs_native_products(model: Literal["discrete", "age"]) -> None:
    """Compiled presets use the same typed audit as routed tensor updates."""
    from natal.frontend.presets import HomingDrive

    pop = _population(f"ReviewCompiledAudit_{model}", model)
    before = pop.config.zygotes_to_gametes_map.copy()
    pop.update().presets(HomingDrive(
        name="AuditDrive", drive_allele="Dr", target_allele="WT",
        drive_conversion_rate=0.75,
    ))
    entry = next(row for row in pop.params_log_details if row[3] == "meiosis_map")
    assert entry[:4] == (0, "genetics", 0, "meiosis_map")
    np.testing.assert_array_equal(entry[4], before)
    np.testing.assert_array_equal(entry[5], pop.config.zygotes_to_gametes_map)


@pytest.mark.parametrize("model", ["discrete", "age", "spatial"])
def test_reducing_history_capacity_immediately_discards_native_checkpoints(model: HistoryModel) -> None:
    """A smaller retention budget releases checkpoints without another run tick."""
    pop = _history_population(f"ReviewImmediateCheckpointEviction_{model}", model)
    pop.run(10, record_every=1)
    pop.history.max_rows = 2
    assert pop.history.ticks == (9, 10)
    backend = pop._rust_spatial_backend if isinstance(pop, nt.SpatialPopulation) else pop._rust_lifecycle_backend
    assert backend is not None
    # Native lookup returning None proves the checkpoint itself was removed.
    # Keeping it and relying on HistoryStore to reject the tick would raise.
    assert backend.restore_from_checkpoint(8) is None
    assert pop.tick == 10
    pop.restore_checkpoint(9)
    assert pop.tick == 9


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_direct_population_snapshot_initializes_native_checkpoint_owner(model: Literal["discrete", "age"]) -> None:
    """Directly constructed populations use the same full snapshot contract."""
    built = _population(f"ReviewDirectSnapshotSource_{model}", model)
        # Internal materialization path (shared by build/clone/restore),
        # deliberately exercised; the public construction entry is the
        # builder chain.
    direct = type(built)(species=built.species, population_config=built.config, index_registry=built.index_registry)
    direct.record_snapshot()
    expected = direct.export_state().copy()
    direct.run(1)
    direct.restore_checkpoint(0)
    np.testing.assert_array_equal(direct.export_state(), expected)


def test_routed_writer_rejects_missing_atomic_channel_and_logs_only_commits() -> None:
    """Legacy scalar log adapters still honor the native transaction boundary."""
    from natal.frontend.builder._writers import CoreConfigWriter

    pop = _population("ReviewLegacyScalarAudit")
    rows: list[tuple[str, float, float]] = []
    with pytest.raises(TypeError, match="atomic session channel"):
        CoreConfigWriter(pop.config, object())
    writer = CoreConfigWriter(pop.config, pop._rust_lifecycle_backend, param_log=lambda name, old, new: rows.append((name, old, new)))
    with pytest.raises(ValueError):
        writer.apply({"carrying_capacity": 777., "sex_ratio": -1.})
    assert rows == []
    assert pop.params.carrying_capacity == 100000.
    writer.apply({"carrying_capacity": 777.})
    assert rows == [("carrying_capacity", 100000., 777.)]
    assert pop.params.carrying_capacity == 777.
