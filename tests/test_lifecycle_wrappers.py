"""Tests for the LifecycleWrappers class and compile_lifecycle_wrappers pipeline.

LifecycleWrappers lives in the engine layer (engine/lifecycle_wrappers.py)
and is the central integration point between the hook system and population
simulation loops.  Tests here verify construction, hook bundling, mixed
event detection, and (basic) lifecycle wrapper compilation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.engine.lifecycle_wrappers import (  # noqa: E402
    LifecycleWrappers,
    compile_lifecycle_wrappers,
)
from natal.hooks import (  # noqa: E402
    CompiledHookDescriptor,
    HookProgram,
)
from natal.numba.utils import numba_disabled  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _empty_program() -> HookProgram:
    """Build a minimal empty HookProgram."""
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(5, dtype=np.int32),
        n_ops_list=np.zeros(0, dtype=np.int32),
        op_offsets=np.zeros(1, dtype=np.int32),
        op_types_data=np.zeros(0, dtype=np.int32),
        zidx_offsets_data=np.zeros(0, dtype=np.int32),
        zidx_data=np.zeros(0, dtype=np.int32),
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


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_lifecycle_wrappers_default_construction() -> None:
    """Default LifecycleWrappers has noop hooks and None lifecycle wrappers."""
    lw = LifecycleWrappers()
    assert lw.hooks.first is not None
    assert lw.hooks.early is not None
    assert lw.hooks.late is not None
    assert lw.hooks.finish is not None
    assert lw.hooks.registry is None
    assert lw.run_tick_fn is None
    assert lw.run_fn is None
    assert lw.run_discrete_tick_fn is None
    assert lw.run_discrete_fn is None
    assert lw.spatial_tick_fn is None
    assert lw.spatial_run_fn is None
    assert lw.spatial_discrete_tick_fn is None
    assert lw.spatial_discrete_run_fn is None


def test_lifecycle_wrappers_registry_property() -> None:
    """registry property delegates to hooks.registry."""
    lw = LifecycleWrappers()
    assert lw.registry is None
    prog = _empty_program()
    lw.hooks.registry = prog
    assert lw.registry is prog


# ---------------------------------------------------------------------------
# compile_lifecycle_wrappers — basic scenarios
# ---------------------------------------------------------------------------


def test_compile_empty_hooks_returns_defaults() -> None:
    """compile_lifecycle_wrappers with an empty descriptor list returns defaults."""
    wrappers = compile_lifecycle_wrappers([], registry=None)
    assert isinstance(wrappers, LifecycleWrappers)
    assert wrappers.hooks.first is not None
    assert wrappers.hooks.early is not None
    assert wrappers.hooks.late is not None
    assert wrappers.hooks.finish is not None


def test_compile_with_njit_only_hooks() -> None:
    """Non-mixed njit-only hooks produce combined chains (Python path)."""
    calls: list[str] = []

    def fn_a(state, config, deme_id=-1):
        calls.append("a")
        return 0

    def fn_b(state, config, deme_id=-1):
        calls.append("b")
        return 0

    desc_a = CompiledHookDescriptor(
        name="a", event="early", priority=0, njit_fn=fn_a
    )
    desc_b = CompiledHookDescriptor(
        name="b", event="early", priority=1, njit_fn=fn_b
    )
    class S:
        individual_count = np.zeros((2, 1, 1))
        n_tick = 0
    class C:
        stochastic = False
        continuous_sampling = False
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([desc_a, desc_b], registry=None)
        assert wrappers.hooks.early is not None
        wrappers.hooks.early(S(), C(), 0)
    assert calls == ["a", "b"]


def test_compile_with_deme_guards() -> None:
    """Deme selectors on njit hooks produce guarded calls (Python path)."""
    calls: list[str] = []

    def fn_a(state, config, deme_id=-1):
        calls.append(f"a@{deme_id}")
        return 0

    desc_a = CompiledHookDescriptor(
        name="a", event="early", priority=0, njit_fn=fn_a, deme_selector=1
    )
    class S:
        individual_count = np.zeros((2, 1, 1))
        n_tick = 0
    class C:
        stochastic = False
        continuous_sampling = False
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([desc_a], registry=None)
        wrappers.hooks.early(S(), C(), 0)  # deme 0 — should skip
        wrappers.hooks.early(S(), C(), 1)  # deme 1 — should run
    assert calls == ["a@1"]


def test_compile_preserves_noop_on_empty_event() -> None:
    """Events without hooks default to noop callables."""
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([], registry=None)
        # Verify that noop hooks are present and callable.
        assert wrappers.hooks.first is not None
        assert wrappers.hooks.late is not None
        assert wrappers.hooks.finish is not None
        # Verify noop is the canonical noop (same identity when no hooks configured).
        assert wrappers.hooks.first is wrappers.hooks.late
        assert wrappers.hooks.first is wrappers.hooks.finish


# ---------------------------------------------------------------------------
# PyWrapper guard
# ---------------------------------------------------------------------------


@pytest.mark.numba_on
def test_compile_rejects_py_wrapper_with_numba_enabled() -> None:
    """compile_lifecycle_wrappers raises TypeError when Numba is on and
    a py_wrapper descriptor is present."""
    desc = CompiledHookDescriptor(
        name="py_wrapper_hook",
        event="early",
        priority=0,
        py_wrapper=lambda pop: None,
    )
    with pytest.raises(TypeError, match="py_wrapper"):
        compile_lifecycle_wrappers([desc], registry=None)


# ---------------------------------------------------------------------------
# compile_lifecycle_wrappers — disabled Numba path
# ---------------------------------------------------------------------------


def test_compile_with_numba_disabled() -> None:
    """When Numba is disabled, lifecycle wrappers are None but hooks work."""
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([], registry=None)
        assert wrappers.run_tick_fn is None
        assert wrappers.run_fn is None
        assert wrappers.run_discrete_tick_fn is None
        assert wrappers.run_discrete_fn is None
        assert wrappers.hooks.first is not None


# ---------------------------------------------------------------------------
# compile_lifecycle_wrappers — single hook optimisation
# ---------------------------------------------------------------------------


def test_compile_single_njit_hook_no_guard_returns_directly() -> None:
    """A single njit hook without deme guard returns the function directly."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append("single")
        return 0

    desc = CompiledHookDescriptor(
        name="single", event="early", priority=0, njit_fn=fn
    )
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([desc], registry=None)
        # The hook should be the original function (direct return optimisation).
        assert wrappers.hooks.early is fn


def test_compile_single_njit_with_guard_wraps_it() -> None:
    """A single njit hook WITH deme guard requires a wrapper (Python path)."""
    calls: list[str] = []

    def fn(state, config, deme_id=-1):
        calls.append(f"run@{deme_id}")
        return 0

    desc = CompiledHookDescriptor(
        name="guarded", event="early", priority=0, njit_fn=fn, deme_selector=2
    )
    class S:
        individual_count = np.zeros((2, 1, 1))
        n_tick = 0
    class C:
        stochastic = False
        continuous_sampling = False
    with numba_disabled():
        wrappers = compile_lifecycle_wrappers([desc], registry=None)
        # With a deme guard, the hook is wrapped (not the original function).
        assert wrappers.hooks.early is not fn
        wrappers.hooks.early(S(), C(), 0)  # deme 0 — skipped
        wrappers.hooks.early(S(), C(), 2)  # deme 2 — runs
    assert calls == ["run@2"]


# ---------------------------------------------------------------------------
# compile_lifecycle_wrappers — spatial flag
# ---------------------------------------------------------------------------


@pytest.mark.numba_on
def test_compile_with_spatial_wrappers() -> None:
    """include_spatial_wrappers=True compiles spatial lifecycle wrappers."""
    wrappers = compile_lifecycle_wrappers(
        [], registry=None, include_spatial_wrappers=True
    )
    assert wrappers.spatial_tick_fn is not None
    assert wrappers.spatial_run_fn is not None
    assert wrappers.spatial_discrete_tick_fn is not None
    assert wrappers.spatial_discrete_run_fn is not None
