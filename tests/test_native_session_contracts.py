"""Validate the native session boundary independently of population adapters."""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from natal._engine_rs import DiscreteEngineSession, EngineSession, HistoryStore
from natal.contracts.materialize import materialize
from natal.frontend.hooks import Op
from tests.test_review_runtime_regressions import _population

Model = Literal["age", "discrete"]
Session = EngineSession | DiscreteEngineSession


def _native(model: Model) -> Session:
    """Construct an unbound native session with a nonzero neutral population."""
    pop = _population(f"NativeSession_{model}", model)
    contracts = materialize(pop.config)
    session_type = EngineSession if model == "age" else DiscreteEngineSession
    return session_type(contracts.blueprint, contracts.params, 123)


def _run(session: Session, ticks: int = 1) -> tuple[int, NDArray[np.float64], bool]:
    """Run the appropriate native model with record-aligned checkpoints."""
    if isinstance(session, EngineSession):
        return session.run(ticks, 1, checkpoint_every=1)
    return session.run(ticks, 1, False, checkpoint_every=1)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_session_invalid_event_flags_and_projection_preserve_state(model: Model) -> None:
    """Rejected metadata must preserve state, ecology, and the RNG stream."""
    session = _native(model)
    before = session.snapshot_state()
    with pytest.raises(ValueError, match="unknown hook event"):
        session.trigger_event(4)
    for invalid_speed in (-1, 4):
        with pytest.raises(ValueError, match="between 0 and 3"):
            session.set_execution_flags(False, True, True, invalid_speed)
    with pytest.raises(ValueError, match="dimensions or deme selection"):
        session.observe_current(np.ones(1), [], False, False)
    np.testing.assert_equal(session.snapshot_state(), before)
    assert session.execution_state() == ("Ready", 0)
    control = _native(model)
    _run(session)
    _run(control)
    np.testing.assert_equal(session.snapshot_state(), control.snapshot_state())


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("event", ["first", "finish"])
def test_explicit_native_event_records_parameter_provenance(model: Model, event: str) -> None:
    """Explicit events commit ecology and identify the responsible event in Rust."""
    pop = _population(
        f"NativeEventLog_{model}_{event}",
        model,
        hook_calls=[(([Op.set_param("carrying_capacity", 321.0)],), {"event": event})],
    )
    before = pop.export_state().copy()
    assert pop.trigger_event(event) == 0
    assert pop.params.carrying_capacity == 321.0
    assert pop._params_log.details() == [(0, event, 0, "carrying_capacity", 100000.0, 321.0)]
    np.testing.assert_array_equal(pop.export_state(), before)
    # A repeated assignment is a no-op, not a second ecological change.
    pop.trigger_event(event)
    assert len(pop._params_log.details()) == 1


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("stop", [False, True])
def test_explicit_native_callback_failure_or_stop_has_defined_state(model: Model, stop: bool) -> None:
    """Callback errors roll back candidate arrays; a stop commits its final marker."""
    session = _native(model)
    initial = session.snapshot_state()

    def callback(ind: NDArray[np.float64], sperm: NDArray[np.float64], tick: int, deme: int) -> int:
        """Mark a borrowed array, then choose the stop or exception path."""
        assert tick == 0 and deme == 0
        ind[0] += 7.0
        if not stop:
            raise ValueError("native callback sentinel")
        return 1

    session.set_python_callbacks([callback], [], [])
    if stop:
        assert session.trigger_event(0) == 1
        expected = initial[1].copy()
        expected[0] += 7.0
        np.testing.assert_array_equal(session.state_snapshot()[1], expected)
        assert session.execution_state()[0] == "Stopped"
    else:
        with pytest.raises(ValueError, match="native callback sentinel"):
            session.trigger_event(0)
        np.testing.assert_equal(session.snapshot_state(), initial)
        assert session.execution_state()[0] == "Failed"
    with pytest.raises(RuntimeError):
        _run(session)
    session.clear_python_callbacks()
    if isinstance(session, EngineSession):
        tick, ind, sperm, words, ecology = session.snapshot_state()
        session.restore_state(tick, ind, sperm, words, ecology)
    else:
        tick, ind, words, ecology = session.snapshot_state()
        session.restore_state(tick, ind, words, ecology)
    assert session.execution_state() == ("Ready", 0)
    assert _run(session)[0] == 1


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_missing_history_and_checkpoint_do_not_mutate_session(model: Model) -> None:
    """Recording requires a bound store; a nonexistent checkpoint is a no-op."""
    session = _native(model)
    before = session.snapshot_state()
    with pytest.raises(ValueError, match="History is not initialized"):
        session.record_history(False)
    assert session.restore_from_checkpoint(10) is None
    np.testing.assert_equal(session.snapshot_state(), before)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_restore_rejects_missing_history_boundary_before_mutation(model: Model) -> None:
    """A retained checkpoint cannot restore into an unrelated bound history."""
    session = _native(model)
    _run(session, 2)
    before = session.snapshot_state()
    # Checkpoint zero exists in the unbound run, but this new store lacks its log cursor.
    session.bind_history(HistoryStore(2, (1, 1, 1, 1), True))
    with pytest.raises(ValueError, match="not found in history"):
        session.restore_from_checkpoint(0)
    np.testing.assert_equal(session.snapshot_state(), before)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_checkpoint_pruning_preserves_retained_replay(model: Model) -> None:
    """Pruning changes retention only; a surviving checkpoint replays its RNG."""
    session = _native(model)
    _run(session, 3)
    final = session.snapshot_state()
    session.retain_checkpoints_from(1)
    assert session.restore_from_checkpoint(0) is None
    session.truncate_checkpoints(1)
    assert session.restore_from_checkpoint(2) is None
    result = session.restore_from_checkpoint(1)
    assert result is not None and result[0] == 1
    _run(session, 2)
    np.testing.assert_equal(session.snapshot_state(), final)
    session.clear_checkpoints()
    assert session.restore_from_checkpoint(1) is None


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_record_schema_error_leaves_state_and_store_usable(model: Model) -> None:
    """An incompatible history width raises without writing partial state or rows."""
    session = _native(model)
    store = HistoryStore(1, (1, 1, 1, 1), True)
    session.bind_history(store)
    before = session.snapshot_state()
    with pytest.raises(ValueError, match="invalid width"):
        session.record_history(False)
    np.testing.assert_equal(session.snapshot_state(), before)
    assert len(store) == 0 and store.ticks() == []
