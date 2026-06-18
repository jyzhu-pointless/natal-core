"""Tests for the HookExecutor class in its own module (hooks/hook_executor.py).

HookExecutor is the Python fallback hook dispatcher used only when Numba is
disabled.  Tests here verify priority ordering, deme selector filtering,
and proper dispatch of CSR / njit / py_wrapper descriptors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hooks import (
    EVENT_EARLY,
    EVENT_FINISH,
    EVENT_FIRST,
    EVENT_LATE,
    RESULT_CONTINUE,
    RESULT_STOP,
    CompiledHookDescriptor,
    HookExecutor,
    HookProgram,
    Op,
)
from natal.hooks.declarative import compile_declarative_hook
from natal.hooks.types import CompiledHookPlan
from natal.numba_utils import numba_disabled


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DummyState:
    """Minimal state with individual_count and sperm_storage for HookExecutor."""

    def __init__(self, n_genotypes: int = 2, n_ages: int = 2) -> None:
        self.individual_count = np.zeros((2, n_ages, n_genotypes), dtype=np.float64)
        self.sperm_storage = np.zeros((0, 0, 0), dtype=np.float64)


class _DummyConfig:
    """Minimal config exposing stochastic / continuous_sampling flags."""

    def __init__(self, stochastic: bool = True) -> None:
        self.stochastic = stochastic
        self.continuous_sampling = False


class _DummyPop:
    """Minimal population exposing state and config for HookExecutor."""

    def __init__(self, n_genotypes: int = 2, n_ages: int = 2) -> None:
        self.state = _DummyState(n_genotypes, n_ages)
        self._config = _DummyConfig()

    @property
    def config(self) -> _DummyConfig:  # type: ignore[override]
        return self._config


def _empty_program() -> HookProgram:
    """Build a minimal empty HookProgram."""
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(5, dtype=np.int32),
        n_ops_list=np.zeros(0, dtype=np.int32),
        op_offsets=np.zeros(1, dtype=np.int32),
        op_types_data=np.zeros(0, dtype=np.int32),
        gidx_offsets_data=np.zeros(0, dtype=np.int32),
        gidx_data=np.zeros(0, dtype=np.int32),
        age_offsets_data=np.zeros(0, dtype=np.int32),
        age_data=np.zeros(0, dtype=np.int32),
        sex_masks_data=np.zeros(0, dtype=np.bool_),
        params_data=np.zeros(0, dtype=np.float64),
        condition_offsets_data=np.zeros(0, dtype=np.int32),
        condition_types_data=np.zeros(0, dtype=np.int32),
        condition_params_data=np.zeros(0, dtype=np.int32),
        deme_selector_types=np.zeros(0, dtype=np.int32),
        deme_selector_offsets=np.zeros(0, dtype=np.int32),
        deme_selector_data=np.zeros(0, dtype=np.int32),
    )


def _dummy_index_registry(n_genotypes: int = 2):
    """Create a minimal IndexRegistry for declarative hook compilation."""

    class _DummyRegistry:
        def num_genotypes(self) -> int:
            return n_genotypes

        def num_ages(self) -> int:
            return 2

        def get_gidx_range(self, selector: object) -> tuple[int, int]:
            return (0, n_genotypes)

        def get_age_range(self, selector: object) -> tuple[int, int]:
            return (0, 2)

    return _DummyRegistry()


class _DummyPopForDeclarative(_DummyPop):
    """Dummy population that supports declarative hook compilation."""

    def __init__(self, n_genotypes: int = 2, n_ages: int = 2) -> None:
        super().__init__(n_genotypes, n_ages)
        self.index_registry = _dummy_index_registry(n_genotypes)
        self.config = _DummyConfig()  # Expose config directly for declarative compile
        self._config = self.config


# ---------------------------------------------------------------------------
# Construction and from_compiled_hooks
# ---------------------------------------------------------------------------


def test_hook_executor_empty_construction() -> None:
    """HookExecutor constructed with empty lists yields no hooks per event."""
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [])
    assert executor.get_hooks_for_event(0) == []
    assert executor.get_hooks_for_event(1) == []
    assert executor.get_hooks_for_event(2) == []
    assert executor.get_hooks_for_event(3) == []


def test_hook_executor_skips_null_descriptors() -> None:
    """Descriptors without any execution payload are silently skipped."""
    desc = CompiledHookDescriptor(
        name="empty", event="early", priority=0, njit_fn=None, py_wrapper=None, plan=None
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    assert executor.get_hooks_for_event(EVENT_EARLY) == []


def test_hook_executor_groups_by_event() -> None:
    """Descriptors are grouped by event_id."""
    calls: list[str] = []

    def make_njit(name: str) -> object:
        def fn(state, config, deme_id=-1):
            calls.append(name)
            return RESULT_CONTINUE

        return fn

    desc_first = CompiledHookDescriptor(
        name="first_hook", event="first", priority=0, njit_fn=make_njit("first")
    )
    desc_early = CompiledHookDescriptor(
        name="early_hook", event="early", priority=0, njit_fn=make_njit("early")
    )

    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc_first, desc_early])
    assert len(executor.get_hooks_for_event(EVENT_FIRST)) == 1
    assert len(executor.get_hooks_for_event(EVENT_EARLY)) == 1
    assert len(executor.get_hooks_for_event(EVENT_LATE)) == 0


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_hook_executor_priority_ordering() -> None:
    """Hooks execute in priority order (lower values first)."""
    calls: list[str] = []

    def make_njit(name: str) -> object:
        def fn(state, config, deme_id=-1):
            calls.append(name)
            return RESULT_CONTINUE

        return fn

    desc_a = CompiledHookDescriptor(
        name="a", event="early", priority=10, njit_fn=make_njit("a")
    )
    desc_b = CompiledHookDescriptor(
        name="b", event="early", priority=0, njit_fn=make_njit("b")
    )
    desc_c = CompiledHookDescriptor(
        name="c", event="early", priority=5, njit_fn=make_njit("c")
    )

    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc_a, desc_b, desc_c])
    pop = _DummyPop()

    with numba_disabled():
        result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_CONTINUE
    assert calls == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# Deme selector filtering
# ---------------------------------------------------------------------------


def test_hook_executor_deme_selector_wildcard() -> None:
    """Wildcard deme selector '*' matches any deme_id."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append(f"run@{deme_id}")
        return RESULT_CONTINUE

    desc = CompiledHookDescriptor(
        name="wild", event="early", priority=0, njit_fn=fn, deme_selector="*"
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    with numba_disabled():
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=5)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=99)

    assert calls == ["run@0", "run@5", "run@99"]


def test_hook_executor_deme_selector_int() -> None:
    """Integer deme selector only matches that exact deme_id."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append(f"run@{deme_id}")
        return RESULT_CONTINUE

    desc = CompiledHookDescriptor(
        name="deme3", event="early", priority=0, njit_fn=fn, deme_selector=3
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    with numba_disabled():
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=2)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=3)

    assert calls == ["run@3"]


def test_hook_executor_deme_selector_range() -> None:
    """Range deme selector matches deme_id in [start, stop)."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append(f"run@{deme_id}")
        return RESULT_CONTINUE

    desc = CompiledHookDescriptor(
        name="rng", event="early", priority=0, njit_fn=fn, deme_selector=range(2, 5)
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    with numba_disabled():
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=1)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=2)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=4)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=5)

    assert calls == ["run@2", "run@4"]


def test_hook_executor_deme_selector_list() -> None:
    """List deme selector matches deme_id in the list."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append(f"run@{deme_id}")
        return RESULT_CONTINUE

    desc = CompiledHookDescriptor(
        name="lst", event="early", priority=0, njit_fn=fn, deme_selector=[1, 3, 7]
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    with numba_disabled():
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=1)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=3)
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=8)

    assert calls == ["run@1", "run@3"]


# ---------------------------------------------------------------------------
# RESULT_STOP propagation
# ---------------------------------------------------------------------------


def test_hook_executor_stop_propagation() -> None:
    """RESULT_STOP from any hook aborts the event immediately."""
    calls: list[str] = []

    def make_njit(name: str, result: int) -> object:
        def fn(state, config, deme_id=-1):
            calls.append(name)
            return result

        return fn

    desc_a = CompiledHookDescriptor(
        name="a", event="early", priority=0, njit_fn=make_njit("a", RESULT_CONTINUE)
    )
    desc_b = CompiledHookDescriptor(
        name="b", event="early", priority=1, njit_fn=make_njit("b", RESULT_STOP)
    )
    desc_c = CompiledHookDescriptor(
        name="c", event="early", priority=2, njit_fn=make_njit("c", RESULT_CONTINUE)
    )

    executor = HookExecutor.from_compiled_hooks(
        _empty_program(), [desc_a, desc_b, desc_c]
    )
    pop = _DummyPop()

    with numba_disabled():
        result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_STOP
    assert calls == ["a", "b"]  # c never runs


# ---------------------------------------------------------------------------
# Invalid event_id
# ---------------------------------------------------------------------------


def test_hook_executor_invalid_event_id() -> None:
    """Out-of-range event_id returns RESULT_CONTINUE silently."""
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [])
    pop = _DummyPop()
    result = executor.execute_event(999, pop, tick=0, deme_id=0)
    assert result == RESULT_CONTINUE

    result = executor.execute_event(-1, pop, tick=0, deme_id=0)
    assert result == RESULT_CONTINUE


# ---------------------------------------------------------------------------
# get_hooks_for_event
# ---------------------------------------------------------------------------


def test_get_hooks_for_event_returns_sorted() -> None:
    """get_hooks_for_event returns descriptors sorted by priority."""
    def _noop(state, config, deme_id=-1):
        return 0

    desc_a = CompiledHookDescriptor(
        name="a", event="early", priority=5, njit_fn=_noop
    )
    desc_b = CompiledHookDescriptor(
        name="b", event="early", priority=0, njit_fn=_noop
    )
    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc_a, desc_b])
    hooks = executor.get_hooks_for_event(EVENT_EARLY)
    assert len(hooks) == 2
    assert hooks[0].priority <= hooks[1].priority
