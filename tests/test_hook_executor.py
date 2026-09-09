"""Tests for the HookExecutor class in its own module (hooks/runtime/fallback.py).

HookExecutor is the Python dispatch layer shared by the reference paths:
for one event it runs CSR declarative plans and Python callbacks
interleaved in a single stable ascending-priority order.  Tests here
verify event grouping, cross-type priority ordering, deme selector
filtering, RESULT_STOP propagation, and invalid event ids.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.entry.declarative import compile_declarative_hook
from natal.frontend.hooks.runtime.fallback import HookExecutor
from natal.frontend.hooks.tick_context import HookRunner
from natal.frontend.hooks.types import (
    EVENT_EARLY,
    EVENT_FINISH,
    EVENT_FIRST,
    EVENT_LATE,
    RESULT_CONTINUE,
    RESULT_STOP,
    CompiledHookDescriptor,
    empty_hook_program,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_pop(name: str) -> nt.DiscreteGenerationPopulation:
    """Build a quiescent discrete population (state changes only via hooks)."""
    species = nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [10.0, 0.0]},
                "male": {"WT|WT": [10.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


def _compiled(
    ops: list[object],
    pop: object,
    event: str,
    priority: int = 0,
    deme_selector: object = "*",
) -> CompiledHookDescriptor:
    """Compile declarative ops into a descriptor against *pop*."""
    return compile_declarative_hook(
        ops, pop, event, priority=priority, deme_selector=deme_selector
    )


def _executor(
    descriptors: list[CompiledHookDescriptor],
) -> HookExecutor:
    """Build an executor bound to a fresh quiescent population."""
    pop = _build_pop(f"executor_{len(descriptors)}_hook")
    return HookExecutor.from_compiled_hooks(
        empty_hook_program(), descriptors, HookRunner(pop)
    )


def _executor_with_pop(
    descriptors: list[CompiledHookDescriptor],
) -> tuple[HookExecutor, nt.DiscreteGenerationPopulation]:
    """Register *descriptors* on a population and build its dispatch pair."""
    pop = _build_pop("executor_registered")
    for desc in descriptors:
        pop.register_compiled_hook(desc)
    pop.ensure_hook_executor()
    assert pop.hook_executor is not None
    return pop.hook_executor, pop


# ---------------------------------------------------------------------------
# Construction and from_compiled_hooks
# ---------------------------------------------------------------------------


def test_hook_executor_empty_construction() -> None:
    """HookExecutor constructed with empty lists yields no hooks per event."""
    executor = _executor([])
    assert executor.get_hooks_for_event(0) == []
    assert executor.get_hooks_for_event(1) == []
    assert executor.get_hooks_for_event(2) == []
    assert executor.get_hooks_for_event(3) == []


def test_hook_executor_skips_null_descriptors() -> None:
    """Descriptors without any execution payload are silently skipped."""
    desc = CompiledHookDescriptor(name="empty", event="early", priority=0)
    executor = _executor([desc])
    assert executor.get_hooks_for_event(EVENT_EARLY) == []


def test_hook_executor_groups_by_event() -> None:
    """Descriptors are grouped by event_id."""
    pop = _build_pop("executor_grouping")
    ops = [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)]
    desc_first = _compiled(ops, pop, "first")
    desc_early = _compiled(ops, pop, "early")

    executor = _executor([desc_first, desc_early])
    assert len(executor.get_hooks_for_event(EVENT_FIRST)) == 1
    assert len(executor.get_hooks_for_event(EVENT_EARLY)) == 1
    assert len(executor.get_hooks_for_event(EVENT_LATE)) == 0
    assert len(executor.get_hooks_for_event(EVENT_FINISH)) == 0


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_hook_executor_priority_ordering() -> None:
    """Hooks execute in priority order (lower values first).

    Scale and add are non-commutative on the same cell, so the exact
    execution order is observable in the resulting count.
    """
    pop = _build_pop("executor_priority")
    add_one = _compiled([nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)], pop, "early", priority=10)
    scale_two = _compiled([nt.Op.scale(genotypes="WT|WT", ages=0, sex="female", factor=2.0)], pop, "early", priority=0)
    add_two = _compiled([nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=2.0)], pop, "early", priority=5)

    executor, pop = _executor_with_pop([scale_two, add_two, add_one])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_CONTINUE
    # order is scale(10→20), add2(→22), add1(→23); reversed order gives 32.
    assert float(pop.state.individual_count[0, 0, 0]) == 23.0


def test_hook_executor_interleaves_plan_and_callback_by_priority() -> None:
    """A callback with smaller priority runs before a CSR plan.

    The callback (priority 0) adds 100 first (10 → 110); the plan
    (priority 1) then sets the slot to 20.  Final 20 proves the
    callback executed first — the pre-interleaving executor always ran
    plans before callbacks and would finish at 120.
    """
    pop = _build_pop("executor_interleaved")
    set_twenty = _compiled(
        [nt.Op.set_count(genotypes="WT|WT", ages=0, sex="female", value=20.0)],
        pop,
        "early",
        priority=1,
    )

    def add_hundred_fn(ctx) -> int:
        ctx.state.individual_count[0, 0, 0] += 100.0
        return 0

    add_hundred = CompiledHookDescriptor(
        name="add_hundred", event="early", priority=0, callback=add_hundred_fn
    )

    executor, pop = _executor_with_pop([add_hundred, set_twenty])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_CONTINUE
    assert float(pop.state.individual_count[0, 0, 0]) == 20.0


def test_hook_executor_interleave_callback_reads_earlier_plan() -> None:
    """A callback observes the write of a smaller-priority CSR plan."""
    pop = _build_pop("executor_interleave_read")
    set_seven = _compiled(
        [nt.Op.set_count(genotypes="WT|WT", ages=0, sex="female", value=7.0)],
        pop,
        "early",
        priority=0,
    )
    observed: dict[str, float] = {}

    def reader(ctx) -> int:
        observed["female_age0"] = float(ctx.state.individual_count[0, 0, 0])
        return 0

    reader_desc = CompiledHookDescriptor(
        name="reader", event="early", priority=1, callback=reader
    )

    executor, pop = _executor_with_pop([set_seven, reader_desc])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_CONTINUE
    assert observed["female_age0"] == 7.0


# ---------------------------------------------------------------------------
# Deme selector filtering
# ---------------------------------------------------------------------------


def test_hook_executor_deme_selector_wildcard() -> None:
    """Wildcard deme selector '*' matches any deme_id."""
    pop = _build_pop("executor_sel_wild")
    desc = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
    )
    executor, pop = _executor_with_pop([desc])

    for deme_id in (0, 5, 99):
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=deme_id)

    assert float(pop.state.individual_count[0, 0, 0]) == 13.0


def test_hook_executor_deme_selector_int() -> None:
    """Integer deme selector only matches that exact deme_id."""
    pop = _build_pop("executor_sel_int")
    desc = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
        deme_selector=3,
    )
    executor, pop = _executor_with_pop([desc])

    executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=2)
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0

    executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=3)
    assert float(pop.state.individual_count[0, 0, 0]) == 11.0


def test_hook_executor_deme_selector_range() -> None:
    """Range deme selector matches deme_id in [start, stop)."""
    pop = _build_pop("executor_sel_range")
    desc = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
        deme_selector=range(2, 5),
    )
    executor, pop = _executor_with_pop([desc])

    for deme_id in (1, 5):
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=deme_id)
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0

    for deme_id in (2, 4):
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=deme_id)
    assert float(pop.state.individual_count[0, 0, 0]) == 12.0


def test_hook_executor_deme_selector_list() -> None:
    """List deme selector matches deme_id in the list."""
    pop = _build_pop("executor_sel_list")
    desc = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
        deme_selector=[1, 3, 7],
    )
    executor, pop = _executor_with_pop([desc])

    for deme_id in (0, 8):
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=deme_id)
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0

    for deme_id in (1, 3, 7):
        executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=deme_id)
    assert float(pop.state.individual_count[0, 0, 0]) == 13.0


# ---------------------------------------------------------------------------
# RESULT_STOP propagation
# ---------------------------------------------------------------------------


def test_hook_executor_stop_propagation() -> None:
    """RESULT_STOP from a CSR plan aborts the event immediately."""
    pop = _build_pop("executor_stop")
    add_a = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)],
        pop,
        "early",
        priority=0,
    )
    stop_b = _compiled(
        [nt.Op.stop_if_above(genotypes="WT|WT", ages=0, sex="female", threshold=0.0, when="tick >= 0")],
        pop,
        "early",
        priority=1,
    )
    add_c = _compiled(
        [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=100.0)],
        pop,
        "early",
        priority=2,
    )

    executor, pop = _executor_with_pop([add_a, stop_b, add_c])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_STOP
    # a runs (10+1); c never runs.
    assert float(pop.state.individual_count[0, 0, 0]) == 11.0


def test_hook_executor_callback_stop_propagation() -> None:
    """A Python callback returning nonzero stops the event."""
    calls: list[str] = []

    def stopping_callback(pop: object) -> int:
        """Request termination from inside an event."""
        _ = pop
        calls.append("stop")
        return RESULT_STOP

    pop = _build_pop("executor_callback_stop")
    desc = CompiledHookDescriptor(
        name="stopping_callback",
        event="early",
        priority=0,
        callback=stopping_callback,
        source=stopping_callback,
    )
    executor, pop = _executor_with_pop([desc])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_STOP
    assert calls == ["stop"]


def test_hook_executor_callback_stop_skips_later_plan() -> None:
    """A stopping callback (priority 0) skips a later-priority CSR plan."""

    def stopping_callback(pop: object) -> int:
        """Request termination before the plan's priority slot."""
        _ = pop
        return RESULT_STOP

    pop = _build_pop("executor_cb_stop_skips_plan")
    stopper = CompiledHookDescriptor(
        name="stopping_callback",
        event="early",
        priority=0,
        callback=stopping_callback,
        source=stopping_callback,
    )
    set_999 = _compiled(
        [nt.Op.set_count(genotypes="WT|WT", ages=0, sex="female", value=999.0)],
        pop,
        "early",
        priority=1,
    )

    executor, pop = _executor_with_pop([stopper, set_999])

    result = executor.execute_event(EVENT_EARLY, pop, tick=0, deme_id=0)

    assert result == RESULT_STOP
    # The plan never ran: the initial count is untouched.
    assert float(pop.state.individual_count[0, 0, 0]) == 10.0


# ---------------------------------------------------------------------------
# Invalid event_id
# ---------------------------------------------------------------------------


def test_hook_executor_invalid_event_id() -> None:
    """Out-of-range event_id returns RESULT_CONTINUE silently."""
    executor = _executor([])
    pop = _build_pop("executor_invalid_event")
    result = executor.execute_event(999, pop, tick=0, deme_id=0)
    assert result == RESULT_CONTINUE

    result = executor.execute_event(-1, pop, tick=0, deme_id=0)
    assert result == RESULT_CONTINUE


# ---------------------------------------------------------------------------
# get_hooks_for_event
# ---------------------------------------------------------------------------


def test_get_hooks_for_event_returns_sorted() -> None:
    """get_hooks_for_event returns priority-ordered (descriptor, index) pairs."""
    pop = _build_pop("executor_sorted")
    ops = [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)]
    desc_a = _compiled(ops, pop, "early", priority=5)
    desc_b = _compiled(ops, pop, "early", priority=0)
    executor = _executor([desc_a, desc_b])
    hooks = executor.get_hooks_for_event(EVENT_EARLY)
    assert len(hooks) == 2
    assert hooks[0][0].priority <= hooks[1][0].priority
    assert hooks[0][1] is None and hooks[1][1] is None  # CSR plans carry no callback index


def test_get_hooks_for_event_indexes_callbacks_in_priority_order() -> None:
    """Callback descriptors carry their index inside the callback subsequence."""
    pop = _build_pop("executor_sorted_cb")
    ops = [nt.Op.add(genotypes="WT|WT", ages=0, sex="female", delta=1.0)]
    csr_a = _compiled(ops, pop, "early", priority=0)

    def make_cb(name: str):
        def cb(ctx) -> int:
            return 0

        cb.__name__ = name
        return cb

    cb_low = CompiledHookDescriptor(
        name="cb_low", event="early", priority=2, callback=make_cb("cb_low")
    )
    cb_high = CompiledHookDescriptor(
        name="cb_high", event="early", priority=1, callback=make_cb("cb_high")
    )

    executor = _executor([csr_a, cb_low, cb_high])
    hooks = executor.get_hooks_for_event(EVENT_EARLY)
    # Priority order: csr(0), cb_high(1), cb_low(2); callback indexes
    # renumber the callbacks within their own subsequence.
    assert [h[0].name for h in hooks][1:] == ["cb_high", "cb_low"]
    assert hooks[0][0].plan is not None  # the priority-0 slot is the CSR plan
    assert [h[1] for h in hooks] == [None, 0, 1]
