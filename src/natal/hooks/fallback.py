"""Python fallback hook executor (used only when Numba is disabled).

When Numba is disabled, all hook types — CSR plans, njit functions,
Python wrappers — must run through a single sequential path.  This
module provides ``HookExecutor``, which holds descriptors sorted by
priority and dispatches each one in order.

When Numba IS enabled, ``natal.engine.lifecycle_wrappers.compile_lifecycle_wrappers()``
generates compiled lifecycle wrappers that call CSR and njit hooks
directly (including unified dispatch for mixed types).  HookExecutor
is never constructed in that case.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from .csr_kernel import deme_selector_matches, execute_csr_event_arrays
from .types import (
    EVENT_ID_MAP,
    NUM_EVENTS,
    RESULT_CONTINUE,
    RESULT_STOP,
    CompiledHookDescriptor,
    HookProgram,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation


class HookExecutor:
    """Python-layer coordinator for hook execution (Numba-disabled only).

    When Numba is disabled, all hook types — CSR plans, njit functions,
    Python wrappers — must run through a single sequential path.  This
    class holds descriptors sorted by priority and dispatches each one
    in order.
    """

    def __init__(
        self,
        registry: HookProgram,
        hooks_by_event: Dict[int, List[CompiledHookDescriptor]],
    ) -> None:
        """Initialise with a pre-built registry and descriptor map.

        Args:
            registry: HookProgram for CSR operations (may be empty).
            hooks_by_event: Descriptors grouped by event_id and sorted
                by priority.  Built by ``from_compiled_hooks``.
        """
        self.registry = registry
        self.hooks_by_event = hooks_by_event

    @staticmethod
    def from_compiled_hooks(
        registry: HookProgram,
        compiled_hooks: List[CompiledHookDescriptor],
    ) -> HookExecutor:
        """Group descriptors by event_id and sort by priority.

        Descriptors without a recognised event_id or without any
        execution payload are silently skipped.
        """
        hooks_by_event: Dict[int, List[CompiledHookDescriptor]] = defaultdict(list)
        for desc in compiled_hooks:
            event_id = EVENT_ID_MAP.get(desc.event)
            if event_id is not None:
                if desc.plan is not None or desc.njit_fn is not None or desc.py_wrapper is not None:
                    hooks_by_event[event_id].append(desc)

        for event_id in hooks_by_event:
            hooks_by_event[event_id].sort(key=lambda x: x.priority)

        return HookExecutor(registry, dict(hooks_by_event))

    def execute_event(
        self,
        event_id: int,
        population: BasePopulation[Any],
        tick: int,
        deme_id: int = 0,
    ) -> int:
        """Run all hooks for *event_id* in priority order.

        For each descriptor, in priority order:

        1. CSR plan — unpacks arrays and calls ``execute_csr_event_arrays``
           with a single-hook wrapper.  Aborts on ``RESULT_STOP``.
        2. njit function — calls ``desc.njit_fn(state, config, deme_id)``.
        3. Python wrapper — calls with population (1-param) or
           ``(state, config, deme_id)`` (3-param).  Only allowed when
           Numba is disabled.

        Returns:
            ``RESULT_CONTINUE`` or ``RESULT_STOP``.
        """
        if event_id < 0 or event_id >= NUM_EVENTS:
            return RESULT_CONTINUE

        ind_count = population.state.individual_count

        # Resolve runtime state flags.
        sperm_store = getattr(population.state, "sperm_storage", None)
        has_sperm_storage = sperm_store is not None and sperm_store.size > 0
        if not has_sperm_storage:
            sperm_store = np.zeros((0, 0, 0), dtype=np.float64)
        assert sperm_store is not None
        stochastic = bool(getattr(getattr(population, "_config", None), "stochastic", False))
        continuous_sampling = bool(
            getattr(getattr(population, "_config", None), "continuous_sampling", False)
        )

        from natal.numba_utils import NUMBA_ENABLED

        for desc in self.hooks_by_event.get(event_id, []):
            if not deme_selector_matches(desc.deme_selector, deme_id):
                continue

            # CSR: declarative Op.*
            if desc.plan is not None:
                result = execute_csr_event_arrays(
                    n_events=np.int32(1),
                    n_hooks=np.int32(1),
                    hook_offsets=np.array([0, 1], dtype=np.int32),
                    n_ops_list=np.array([desc.plan.n_ops], dtype=np.int32),
                    op_offsets=np.array([0, desc.plan.n_ops], dtype=np.int32),
                    op_types_data=desc.plan.op_types,
                    gidx_offsets_data=desc.plan.gidx_offsets,
                    gidx_data=desc.plan.gidx_data,
                    age_offsets_data=desc.plan.age_offsets,
                    age_data=desc.plan.age_data,
                    sex_masks_data=desc.plan.sex_masks.ravel(),
                    params_data=desc.plan.params,
                    condition_offsets_data=desc.plan.condition_offsets,
                    condition_types_data=desc.plan.condition_types,
                    condition_params_data=desc.plan.condition_params,
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
                )
                if result == RESULT_STOP:
                    return RESULT_STOP

            # njit: compiled custom/selector hook.
            if desc.njit_fn is not None:
                try:
                    result = desc.njit_fn(population.state, population.config, deme_id)
                    if result == RESULT_STOP:
                        return RESULT_STOP
                except Exception as e:
                    raise RuntimeError(f"Error in njit hook '{desc.name}': {e}") from e

            # Python wrapper (Numba off only).
            if desc.py_wrapper is not None and desc.njit_fn is None:
                if NUMBA_ENABLED:
                    raise RuntimeError(
                        f"Python py_wrapper hook '{desc.name}' is not allowed "
                        "when Numba is enabled."
                    )
                try:
                    import inspect

                    sig = inspect.signature(desc.py_wrapper)
                    params = list(sig.parameters.values())
                    if len(params) == 1:
                        desc.py_wrapper(population)
                    elif len(params) == 3:
                        desc.py_wrapper(population.state, population.config, deme_id)
                    else:
                        raise TypeError(
                            f"py_wrapper hook '{desc.name}' has {len(params)} "
                            "parameters; must have 1 (population) or 3 "
                            "(state, config, deme_id)."
                        )
                except Exception as e:
                    raise RuntimeError(f"Error in py_wrapper hook '{desc.name}': {e}") from e

        return RESULT_CONTINUE

    def get_hooks_for_event(self, event_id: int) -> List[CompiledHookDescriptor]:
        """Return descriptors for *event_id*, sorted by priority."""
        return self.hooks_by_event.get(event_id, [])
