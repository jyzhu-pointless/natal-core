"""Python hook executor — the orchestration-layer dispatch.

``HookExecutor`` is the single Python-side execution path for events: it
runs one event's descriptors — CSR declarative plans and
single-parameter Python callbacks alike — interleaved in a single stable
ascending-priority order.  Plans run through the CSR interpreter
(:mod:`natal.frontend.hooks.runtime.csr_kernel`); callbacks fire through
a :class:`~natal.frontend.hooks.tick_context.TickContext`.

Wiring:

- **finish events**: fired Python-side after a run stops or finishes —
  the executor runs the finish-event descriptors directly.
- **rust in-tick events**: the whole interleaved order runs inside the
  Rust engine; only out-of-band triggers use this executor.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np

from natal.frontend.hooks.tick_context import HookRunner
from natal.frontend.hooks.types import (
    ECO_PARAM_NAMES,
    EVENT_ID_MAP,
    NUM_EVENTS,
    RESULT_CONTINUE,
    RESULT_STOP,
    CompiledHookDescriptor,
    HookProgram,
    OpType,
    empty_hook_program,
)

from .csr_kernel import (
    deme_selector_matches,
    eval_csr_condition_program,
    execute_csr_event_arrays,
)

if TYPE_CHECKING:
    from natal.frontend.hooks.types import CompiledHookPlan
    from natal.frontend.population.base import BasePopulation


def _read_eco_values(population: BasePopulation[Any]) -> np.ndarray:
    """Snapshot the five runtime-mutable ecology scalars by ECO param id.

    Reads go through :class:`~natal.frontend.population._params_view.
    ParamsView` so alias resolution and draft reads match ``pop.params``
    exactly.

    Args:
        population: The population whose draft provides current values.

    Returns:
        A fresh float64 array of length ``len(ECO_PARAM_NAMES)``.
    """
    from natal.frontend.population._params_view import ParamsView

    view = ParamsView(population)
    values = np.zeros(len(ECO_PARAM_NAMES), dtype=np.float64)
    for pid, name in enumerate(ECO_PARAM_NAMES):
        raw = getattr(view, name)
        values[pid] = float(cast("float", raw))
    return values


def _fired_set_param_names(
    plan: CompiledHookPlan,
    tick: int,
) -> list[str]:
    """List the ECO param names a plan's set_param ops fire at *tick*.

    Re-evaluates the same schedule + ``when`` condition the kernel just
    applied, so the flush mirrors the handwritten ``pop.params`` channel
    exactly (same-value commits produce no journal row).

    Args:
        plan: ``CompiledHookPlan`` carrying the CSR columns.
        tick: Current simulation tick.

    Returns:
        Canonical ECO param names fired at this tick (may repeat).
    """
    op_types = plan.op_types
    fired: list[str] = []
    for i in range(int(plan.n_ops)):
        if int(op_types[i]) != int(OpType.SET_PARAM):
            continue
        start = int(plan.sp_start[i])
        every = int(plan.sp_every[i])
        if tick < start or (tick - start) % every != 0:
            continue
        ok = eval_csr_condition_program(
            plan.condition_types,
            plan.condition_params,
            int(plan.condition_offsets[i]),
            int(plan.condition_offsets[i + 1]),
            tick,
        )
        if ok:
            pid = int(plan.sp_param_ids[i])
            if 0 <= pid < len(ECO_PARAM_NAMES):
                fired.append(ECO_PARAM_NAMES[pid])
    return fired


def _flush_eco_writes(
    population: BasePopulation[Any],
    eco_values: np.ndarray,
    fired_names: list[str],
) -> None:
    """Commit fired set_param results through the ``pop.params`` channel.

    Each fired scalar is written with a ``ParamsView`` attribute
    assignment — the same writer stack as ``pop.params.<name> = value``:
    route dispatch with bounds validation, the Rust dirty bridge, and a
    ``(tick, name, old, new)`` parameter snapshot log row (change-only:
    same-value commits produce no log row, matching the handwritten channel).

    Args:
        population: The population whose parameters are written.
        eco_values: Post-CSR value array written by the kernel.
        fired_names: Canonical names fired at this tick (from
            ``_fired_set_param_names``).
    """
    from natal.frontend.population._params_view import ParamsView

    view = ParamsView(population)
    for name in fired_names:
        pid = ECO_PARAM_NAMES.index(name)
        setattr(view, name, float(eco_values[pid]))


class HookExecutor:
    """Python-layer coordinator for one event's cross-type priority order.

    The event's descriptors — CSR plans and Python callbacks alike —
    execute interleaved in one stable ascending-priority order.  CSR
    slots run through the CSR interpreter; callback slots dispatch
    through the :class:`~natal.frontend.hooks.tick_context.HookRunner`
    with a fresh :class:`TickContext`.
    """

    def __init__(
        self,
        registry: HookProgram,
        hooks_by_event: Dict[int, List[Tuple[CompiledHookDescriptor, Optional[int]]]],
        runner: HookRunner,
    ) -> None:
        """Initialize with a CSR registry, descriptor map, and callback runner.

        Args:
            registry: HookProgram for CSR operations (never ``None``).
            hooks_by_event: Descriptors grouped by event_id, priority
                ordered; callback descriptors carry their index into the
                runner's callback list (``None`` for CSR plans).  Built
                by ``from_compiled_hooks``.
            runner: The callback dispatcher for Python callbacks.
        """
        self.registry = registry
        self.hooks_by_event = hooks_by_event
        self._runner = runner

    @staticmethod
    def from_compiled_hooks(
        registry: HookProgram | None,
        compiled_hooks: List[CompiledHookDescriptor],
        runner: HookRunner,
    ) -> HookExecutor:
        """Group descriptors by event_id in one cross-type priority order.

        Both CSR plans and Python callbacks enter the same stable
        priority sort, so ``execute_event`` interleaves them.  Each
        callback descriptor is annotated with its position inside the
        event's callback subsequence — the same subsequence order the
        runner holds, because both sort the same registration-ordered
        list by priority.

        Descriptors without a recognized event_id or without any
        execution payload are silently skipped.
        """
        hooks_by_event: Dict[int, List[Tuple[CompiledHookDescriptor, Optional[int]]]] = defaultdict(list)
        callback_counter: Dict[int, int] = defaultdict(int)
        for desc in sorted(compiled_hooks, key=lambda x: x.priority):
            event_id = EVENT_ID_MAP.get(desc.event)
            if event_id is None:
                continue
            is_callback = desc.callback is not None
            if not is_callback and desc.plan is None:
                continue
            cb_index = callback_counter[event_id] if is_callback else None
            if is_callback:
                callback_counter[event_id] += 1
            hooks_by_event[event_id].append((desc, cb_index))

        return HookExecutor(
            registry if registry is not None else empty_hook_program(),
            dict(hooks_by_event),
            runner,
        )

    def execute_event(
        self,
        event_id: int,
        population: BasePopulation[Any],
        tick: int,
        deme_id: int = 0,
    ) -> int:
        """Run all hooks for *event_id* in one cross-type priority order.

        CSR plan descriptors and Python callback descriptors interleave
        by ascending priority (ties keep registration order).  A CSR stop
        op or a nonzero callback return aborts the event, skipping every
        later hook of either kind.

        Args:
            event_id: Numeric event id.
            population: The owning population.
            tick: Current tick.
            deme_id: Deme index for selector filtering.  ``0`` for
                panmictic populations; the live deme index under a
                SpatialPopulation.

        Returns:
            ``RESULT_CONTINUE`` or ``RESULT_STOP``.
        """
        if event_id < 0 or event_id >= NUM_EVENTS:
            return RESULT_CONTINUE

        # Live state on purpose: hooks borrow the writable arrays for one
        # callback (short-term loan).  The public ``population.state``
        # returns snapshots, so the executor borrows the live container
        # (a fresh session snapshot under Rust) and flushes any callback
        # writes back into the session after the event.
        state = population._live_state()  # pyright: ignore[reportPrivateUsage]  # sanctioned live loan channel into callbacks
        ind_count = state.individual_count

        # Resolve runtime state flags.  No dummy sperm array is created:
        # the CSR kernel accepts ``None`` when no sperm storage exists.
        sperm_store = getattr(state, "sperm_storage", None)
        has_sperm_storage = sperm_store is not None and sperm_store.size > 0
        stochastic = bool(getattr(getattr(population, "_config", None), "stochastic", False))
        continuous_sampling = bool(
            getattr(getattr(population, "_config", None), "continuous_sampling", False)
        )

        # OP_SET_PARAM scratch: initialized lazily by the first plan that
        # actually carries set_param ops (dummy configs in unit tests never
        # pay the draft-read cost), then kept live across descriptors so
        # later hooks evaluate expressions against values written by
        # earlier hooks.  Writes flush per descriptor so the params log
        # preserves hook priority order.
        eco_values: np.ndarray | None = None

        ran_callbacks = False
        for desc, cb_index in self.hooks_by_event.get(event_id, []):
            if not deme_selector_matches(desc.deme_selector, deme_id):
                continue
            if cb_index is not None:
                # Callback slot: dispatch exactly this callback so the
                # runner's deme selector stays authoritative too.
                result = self._runner.run_event(
                    event_id,
                    tick=tick,
                    deme_id=deme_id,
                    state=state,
                    only_index=cb_index,
                )
                if result != RESULT_CONTINUE:
                    if ran_callbacks:
                        population._flush_state_to_session(state)  # pyright: ignore[reportPrivateUsage]  # hook writes reach the session-owned state
                    return result
                ran_callbacks = True
                continue
            plan = desc.plan
            if plan is None:
                continue
            plan_has_set_param = bool(
                (plan.op_types == int(OpType.SET_PARAM)).any()
            )
            if plan_has_set_param and eco_values is None:
                eco_values = _read_eco_values(population)
            # The set_param/convert columns use getattr fallbacks so
            # duck-typed plans built before these fields existed (unit
            # tests) keep executing through this path unchanged.
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            result = execute_csr_event_arrays(
                n_events=np.int32(1),
                n_hooks=np.int32(1),
                hook_offsets=np.array([0, 1], dtype=np.int32),
                n_ops_list=np.array([plan.n_ops], dtype=np.int32),
                op_offsets=np.array([0, plan.n_ops], dtype=np.int32),
                op_types_data=plan.op_types,
                zidx_offsets_data=plan.zidx_offsets,
                zidx_data=plan.zidx_data,
                age_offsets_data=plan.age_offsets,
                age_data=plan.age_data,
                sex_masks_data=plan.sex_masks.ravel(),
                params_data=plan.params,
                condition_offsets_data=plan.condition_offsets,
                condition_types_data=plan.condition_types,
                condition_params_data=plan.condition_params,
                sp_param_ids_data=getattr(plan, "sp_param_ids", empty_i32),
                sp_every_data=getattr(plan, "sp_every", empty_i32),
                sp_start_data=getattr(plan, "sp_start", empty_i32),
                rpn_offsets_data=getattr(
                    plan, "rpn_offsets", np.array([0], dtype=np.int32)
                ),
                rpn_kinds_data=getattr(plan, "rpn_kinds", empty_i32),
                rpn_payload_data=getattr(plan, "rpn_payload", empty_i32),
                sp_literals_data=getattr(plan, "sp_literals", empty_f64),
                convert_source_z_data=getattr(plan, "convert_source_z", empty_i32),
                convert_target_z_data=getattr(plan, "convert_target_z", empty_i32),
                deme_selector_types=np.array([0], dtype=np.int32),
                deme_selector_offsets=np.array([0, 0], dtype=np.int32),
                deme_selector_data=np.array([], dtype=np.int32),
                event_id=0,
                individual_count=ind_count,
                sperm_storage=sperm_store,
                has_sperm_storage=has_sperm_storage,
                tick=tick,
                stochastic=stochastic,
                continuous_sampling=continuous_sampling,
                deme_id=deme_id,
                eco_values=eco_values,
            )
            if result == RESULT_STOP:
                if ran_callbacks:
                    population._flush_state_to_session(state)  # pyright: ignore[reportPrivateUsage]  # hook writes reach the session-owned state
                return RESULT_STOP
            if eco_values is not None and plan_has_set_param:
                _flush_eco_writes(
                    population, eco_values, _fired_set_param_names(plan, tick)
                )

        # Flush the borrowed container (a session snapshot under Rust)
        # once per event; callbacks already observed each other's writes
        # through the same live container.
        if ran_callbacks:
            population._flush_state_to_session(state)  # pyright: ignore[reportPrivateUsage]  # hook writes reach the session-owned state
        return RESULT_CONTINUE

    def get_hooks_for_event(
        self, event_id: int
    ) -> List[Tuple[CompiledHookDescriptor, Optional[int]]]:
        """Return the event's descriptors with callback indexes, priority ordered."""
        return self.hooks_by_event.get(event_id, [])
