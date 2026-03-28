#!/usr/bin/env python3
"""Unit tests for deme selector support in hook descriptors and executor."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hook_dsl import (  # noqa: E402
    CompiledHookDescriptor,
    HookExecutor,
    HookProgram,
    EVENT_EARLY,
    RESULT_CONTINUE,
)
from natal.numba_utils import numba_disabled  # noqa: E402


class _DummyState:
    def __init__(self):
        self.individual_count = np.zeros((2, 1, 1), dtype=np.float64)
        self.sperm_storage = np.zeros((0, 0, 0), dtype=np.float64)


class _DummyPop:
    def __init__(self):
        self.state = _DummyState()
        self._config = type("Cfg", (), {"is_stochastic": False, "use_dirichlet_sampling": False})()


def _empty_program() -> HookProgram:
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        n_ops_list=np.array([], dtype=np.int32),
        op_offsets=np.array([0], dtype=np.int32),
        op_types_data=np.array([], dtype=np.int32),
        gidx_offsets_data=np.array([0], dtype=np.int32),
        gidx_data=np.array([], dtype=np.int32),
        age_offsets_data=np.array([0], dtype=np.int32),
        age_data=np.array([], dtype=np.int32),
        sex_masks_data=np.array([], dtype=np.bool_),
        params_data=np.array([], dtype=np.float64),
        condition_offsets_data=np.array([0], dtype=np.int32),
        condition_types_data=np.array([], dtype=np.int32),
        condition_params_data=np.array([], dtype=np.int32),
        deme_selector_types=np.array([], dtype=np.int32),
        deme_selector_offsets=np.array([0], dtype=np.int32),
        deme_selector_data=np.array([], dtype=np.int32),
    )


def test_executor_filters_py_wrapper_by_deme_selector():
    calls = []

    def only_deme_two(pop):
        _ = pop
        calls.append("d2")

    desc = CompiledHookDescriptor(
        name="only_deme_two",
        event="early",
        priority=0,
        deme_selector=2,
        py_wrapper=only_deme_two,
    )

    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    with numba_disabled():
        result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=1)
        assert result == RESULT_CONTINUE
        assert calls == []

        result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=2)
        assert result == RESULT_CONTINUE
        assert calls == ["d2"]


def test_executor_filters_njit_like_hook_by_deme_selector():
    calls = []

    def fake_njit(ind_count, tick, deme_id):
        _ = (ind_count, tick, deme_id)
        calls.append("run")
        return RESULT_CONTINUE

    desc = CompiledHookDescriptor(
        name="set_selector_list",
        event="early",
        priority=0,
        deme_selector=[0, 3],
        njit_fn=fake_njit,
    )

    executor = HookExecutor.from_compiled_hooks(_empty_program(), [desc])
    pop = _DummyPop()

    executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=2)
    assert calls == []

    executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=3)
    assert calls == ["run"]
