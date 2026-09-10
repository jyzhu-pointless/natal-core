"""Contracts for the Rust-only hook execution surface."""

from __future__ import annotations

import importlib
from typing import Literal

from numpy.typing import NDArray

from natal.frontend.hooks.types import DemeSelector

import numpy as np
import pytest

import natal as nt
from tests.test_ops_setparam_convert import _build_age_structured, _fresh_species
from tests.test_review_runtime_regressions import _population


def test_python_hook_executor_and_csr_runtime_are_removed() -> None:
    """The retired Python execution modules cannot be imported."""
    for module_name in (
        "natal.frontend.hooks.compile.container",
        "natal.frontend.hooks.runtime.fallback",
        "natal.frontend.hooks.runtime.sampling",
        "natal.frontend.hooks.runtime.csr_kernel",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"retired hook runtime still importable: {module_name}")


def test_retired_hook_execution_exports_are_absent() -> None:
    """Public hooks expose compilation and callback APIs only."""
    hooks = importlib.import_module("natal.frontend.hooks")
    natal = importlib.import_module("natal")
    for name in (
        "HookExecutor",
        "CompiledEventHooks",
        "execute_csr_event_arrays",
        "execute_csr_event_program",
        "execute_csr_event_program_with_state",
        "execute_single_csr_hook",
        "eval_csr_condition_program",
    ):
        assert not hasattr(hooks, name)
        assert not hasattr(natal, name)


def test_native_public_event_path_preserves_priority_condition_and_stop() -> None:
    """Native dispatch retains the useful interpreter behavior contracts."""
    population = _population(
        "NativeHookSurface",
        "discrete",
        hook_calls=[
            (([nt.Op.add(genotypes="WT|WT", ages=1, sex="female", delta=2.0)],),
             {"event": "early", "priority": 1}),
            ((nt.Op.stop_if_above(
                genotypes="WT|WT",
                ages=1,
                sex="female",
                threshold=0.0,
                when="tick >= 0",
            ),), {"event": "early", "priority": 0}),
        ],
    )

    before = population.state.individual_count.copy()
    assert population.trigger_event("early") == nt.RESULT_STOP
    assert (population.state.individual_count == before).all()


def test_native_scale_preserves_sperm_and_virgin_ratio() -> None:
    """Native scale halves mothers and their mated sperm buckets together."""
    pop = _build_age_structured(
        _fresh_species(),
        "NativeSpermScale",
        hook_calls=[(([nt.Op.scale(genotypes="A|A", ages=[1], sex="female", factor=0.5)],), {"event": "early"})],
    )
    state = pop._live_state()
    individual_count = np.zeros_like(state.individual_count)
    sperm_storage = np.zeros_like(state.sperm_storage)
    individual_count[0, 1, 0] = 10.0
    sperm_storage[1, 0, :] = [2.0, 4.0, 0.0]
    pop.import_state(state._replace(individual_count=individual_count, sperm_storage=sperm_storage))
    pop.trigger_event("early")
    result = pop.state
    assert result.individual_count[0, 1, 0] == 5.0
    np.testing.assert_array_equal(result.sperm_storage[1, 0], [1.0, 2.0, 0.0])
    assert result.individual_count[0, 1, 0] - result.sperm_storage[1, 0].sum() == 2.0


def test_native_program_rejects_unknown_opcode_before_state_change() -> None:
    """Malformed native programs fail before mutating session state."""
    pop = _population(
        "NativeOpcodeBoundary",
        "discrete",
        stochastic=False,
        hook_calls=[([nt.Op.scale(genotypes="WT|WT", factor=0.5)], {"event": "early"})],
    )
    before = pop._rust_lifecycle_backend.state_snapshot()
    malformed = pop._hook_program._replace(
        op_types_data=np.array([-1], dtype=np.int32)
    )
    pop._rust_lifecycle_backend.configure_program(malformed, pop.config)
    with pytest.raises(RuntimeError, match="opcode"):
        pop._rust_lifecycle_backend._session.trigger_event(1, 0)
    after = pop._rust_lifecycle_backend.state_snapshot()
    assert after[0] == before[0]
    for after_array, before_array in zip(after[1:], before[1:]):
        np.testing.assert_array_equal(after_array, before_array)


@pytest.mark.parametrize("selector,deme_id,expected", [
    ("*", 9, True), (2, 2, True), (2, 3, False),
    (range(1, 5, 2), 3, True), (range(1, 5, 2), 2, False),
    ([1, 3], 3, True), ([1, 3], 2, False), ((1, 3), 1, True),
])
def test_public_deme_selector_matches(selector: DemeSelector, deme_id: int, expected: bool) -> None:
    """The retained selector helper obeys wildcard and container membership."""
    from natal.frontend.hooks.types import deme_selector_matches
    assert deme_selector_matches(selector, deme_id) is expected


def test_wright_fisher_batch_journal_uses_each_event_tick() -> None:
    """Each first-event commit is attributed to its actual simulation tick."""
    from natal.backends.rust.rust_backend import RustDiscreteLifecycleBackend
    pop = _population(
        "EvaluatorWFJournal",
        "discrete",
        stochastic=False,
        hook_calls=[((nt.Op.set_param("carrying_capacity", "carrying_capacity + 1"),), {"event": "first"})],
    )
    config = pop.config._replace(extreme_speed_mode=1)
    backend = RustDiscreteLifecycleBackend(config, seed=1)
    backend.configure_program(pop._hook_program, config)
    backend.set_state(pop.state)
    assert backend.run(3, 1)[0] == 3
    assert backend.drain_eco_journal() == [
        (tick, "carrying_capacity", 100000.0 + tick, 100001.0 + tick)
        for tick in range(3)
    ]


def test_wright_fisher_later_callback_failure_keeps_completed_tick() -> None:
    """A first-event failure at tick one preserves the completed tick zero."""
    from tests.test_native_session_contracts import _native
    session = _native("discrete")
    session.set_execution_flags(True, False, False, 1)
    seen: list[int] = []
    def fail_later(ind: NDArray[np.float64], sperm: NDArray[np.float64], tick: int, deme: int) -> int:
        seen.append(tick)
        if tick == 1:
            raise ValueError("WF later callback failure")
        return 0
    session.set_python_callbacks([fail_later], [], [])
    with pytest.raises(ValueError, match="WF later callback failure"):
        session.run(3, 1, True)
    assert seen == [0, 1]
    assert session.state_snapshot()[0] == 1
    assert session.execution_state() == ("Failed", 0)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_direct_population_event_initializes_native_session(model: Literal["age", "discrete"]) -> None:
    """Direct constructors lazily start a native session for explicit events."""
    source = _population(
        f"DirectEvent_{model}",
        model,
        stochastic=False,
        hook_calls=[((nt.Op.set_param("carrying_capacity", 321.0),), {"event": "first"})],
    )
    # The raw constructor is the internal mechanism build/clone/restore
    # share; the compiled plan travels with it exactly as with clones.
        # Internal materialization path (shared by build/clone/restore),
        # deliberately exercised; the public construction entry is the
        # builder chain.
    pop = type(source)(
        species=source.species,
        population_config=source.config,
        hook_descriptors=source._hook_descriptors,  # noqa: SLF001
    )
    assert pop._rust_lifecycle_backend is None
    assert pop.trigger_event("first") == 0
    assert pop._rust_lifecycle_backend is not None
    assert pop.params.carrying_capacity == 321.0


def test_explicit_event_without_available_session_reports_native_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lazy initialization cannot silently skip registered hooks."""
    source = _population("UnavailableNativeEvent", "discrete", stochastic=False)
        # Internal materialization path (shared by build/clone/restore),
        # deliberately exercised; the public construction entry is the
        # builder chain.
    pop = type(source)(species=source.species, population_config=source.config)
    monkeypatch.setattr(pop, "_initialize_session", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="Native hook execution is unavailable"):
        pop.trigger_event("first")


@pytest.mark.parametrize("model,wf", [("age", False), ("discrete", False), ("discrete", True)])
@pytest.mark.parametrize("ticks", [0, 2])
def test_unbound_native_observation_rows_match_raw_group_sums(model: Literal["age", "discrete"], wf: bool, ticks: int) -> None:
    """Group projections preserve native row order and sum only the ztype axis.

    Integer stochastic counts and unit masks give exactly representable sums.
    Identical seeds make the raw and projected runs follow the same trajectory.
    """
    from tests.test_native_session_contracts import _native
    raw, observed = _native(model), _native(model)
    source = _population(f"ProjectionDimensions_{model}", model)
    shape = source.state.individual_count.shape
    mask = np.ones((2, *shape), dtype=np.float64)
    mask[1, ..., 1:] = 0.0
    if wf:
        raw.set_execution_flags(True, False, False, 1)
        observed.set_execution_flags(True, False, False, 1)
    if model == "age":
        raw_result = raw.run(ticks, 1, checkpoint_every=1)
        result = observed.run(ticks, 1, mask, checkpoint_every=1)
    else:
        raw_result = raw.run(ticks, 1, wf, checkpoint_every=1)
        result = observed.run(ticks, 1, wf, mask, checkpoint_every=1)
    counts = raw_result[1][:, 1:1 + int(np.prod(shape))].reshape((-1, *shape))
    expected = np.stack([counts.sum(axis=-1), counts[..., 0]], axis=1).reshape((ticks + 1, -1))
    assert result[0] == ticks and not result[2]
    np.testing.assert_array_equal(result[1][:, 0], np.arange(ticks + 1))
    np.testing.assert_array_equal(result[1][:, 1:], expected)
    assert observed.restore_from_checkpoint(ticks)[0] == ticks


@pytest.mark.parametrize("model,wf", [("age", False), ("discrete", False), ("discrete", True)])
def test_unbound_native_checkpoints_require_both_intervals(model: Literal["age", "discrete"], wf: bool) -> None:
    """Record-every-two and checkpoint-every-three overlap at zero and six."""
    from tests.test_native_session_contracts import _native
    session = _native(model)
    if wf:
        session.set_execution_flags(True, False, False, 1)
    if model == "age":
        result = session.run(6, 2, checkpoint_every=3)
    else:
        result = session.run(6, 2, wf, checkpoint_every=3)
    np.testing.assert_array_equal(result[1][:, 0], [0, 2, 4, 6])
    # The low-level session returns None when no exact checkpoint exists.
    assert session.restore_from_checkpoint(3) is None
    assert session.state_snapshot()[0] == 6
    assert session.restore_from_checkpoint(6)[0] == 6
    assert session.restore_from_checkpoint(0)[0] == 0


def test_unbound_wright_fisher_stop_keeps_tick_and_pre_event_row() -> None:
    """STOP from the first event commits its state without advancing the clock."""
    from tests.test_native_session_contracts import _native
    session = _native("discrete")
    session.set_execution_flags(True, False, False, 1)
    initial = session.state_snapshot()[1].copy()
    def stop(ind: NDArray[np.float64], sperm: NDArray[np.float64], tick: int, deme: int) -> int:
        ind[0] += 7.0
        return 1
    session.set_python_callbacks([stop], [], [])
    tick, rows, stopped = session.run(3, 1, True)
    assert tick == 0 and stopped
    np.testing.assert_array_equal(rows[0, 1:], initial)
    expected = initial.copy()
    expected[0] += 7.0
    np.testing.assert_array_equal(session.state_snapshot()[1], expected)
    assert session.execution_state() == ("Stopped", 0)


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("condition,firing_ticks", [
    ("tick == 1 or tick == 2", {1, 2}),
    ("not (tick >= 2 and tick < 4)", {0, 1, 4, 5}),
    ("tick == 1 or tick >= 3 and tick < 5", {1, 3, 4}),
])
def test_native_condition_logic_uses_boolean_truth_table(
    model: Literal["age", "discrete"], condition: str, firing_ticks: set[int]
) -> None:
    """Native AND/OR/NOT and precedence preserve the removed interpreter's contract."""
    pop = _population(
        f"NativeTruthTable_{model}",
        model,
        stochastic=False,
        hook_calls=[((nt.Op.add(genotypes="WT|WT", ages=1, sex="female", delta=1.0, when=condition),), {"event": "early"})],
    )
    initial = pop.state
    for tick in range(6):
        pop.import_state(initial._replace(n_tick=tick))
        pop.trigger_event("early")
        expected = initial.individual_count.copy()
        if tick in firing_ticks:
            expected[0, 1, 0] += 1.0
        np.testing.assert_array_equal(pop.state.individual_count, expected)
