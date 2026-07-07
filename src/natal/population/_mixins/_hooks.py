"""Hook management mixin for BasePopulation.

Extracted from :mod:`natal.population.base` to reduce the
BasePopulation ABC to its core lifecycle contract.
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import numpy as np

from natal.numba.utils import is_numba_enabled

if TYPE_CHECKING:
    from natal.engine.lifecycle_wrappers import LifecycleWrappers
    from natal.hooks.types import (
        CompiledHookDescriptor,
        DemeSelector,
        HookProgram,
    )
    from natal.population.base import BasePopulation

HookCallback = Callable[..., object]
HookEntry = Tuple[int, Optional[str], HookCallback]


class HookManagerMixin:
    """Mixin providing hook registration, compilation, and dispatch.

    Expects the host class (BasePopulation) to define these attributes:
    ``_hooks: Dict[str, List[HookEntry]]``,
    ``_pending_hooks: List[PendingHook]``,
    ``_compiled_hooks: List[CompiledHookDescriptor]``,
    ``_hook_executor: Optional[HookExecutor]``.
    """

    # Declared here so pyright knows these come from the host class.
    ALLOWED_EVENTS: list[str]  # type: ignore[assignment]
    tick: int  # type: ignore[assignment]
    _hooks: dict[str, list[tuple[int, Optional[str], HookCallback]]]  # type: ignore[assignment]
    _compiled_hooks: list[Any]  # type: ignore[assignment]
    _hook_executor: Any  # type: ignore[assignment]

    # ── Hook registration ────────────────────────────────────────────

    def set_hook(
        self,
        event_name: str,
        func: HookCallback,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None,
        compile: bool = True,
        deme_selector: Optional[DemeSelector] = None,
    ) -> None:
        """
        Register an event hook with optional automatic compilation.

        When ``compile=True`` and the function carries ``@hook`` metadata,
        it enters the DSL compilation pipeline:
        - declarative hook -> CSR plan in HookProgram (kernel executable)
        - selector hook -> ``py_wrapper`` or ``njit_fn`` (mode dependent)
        - numba hook -> ``njit_fn``

        Plain Python functions are still registered in traditional ``_hooks``
        for backward-compatible execution.

        Args:
            event_name: Event name (must exist in ``ALLOWED_EVENTS``).
            func: Callback function, supported forms include:
                  - plain function: ``func(population)``
                  - declarative ``@hook`` function: returns ``[Op.scale(...), ...]``
                  - selector ``@hook(selectors={...})`` function
            hook_id: Numeric execution priority (optional, auto-assigned if omitted).
                     Lower IDs execute first.
            hook_name: Optional human-readable name for debugging.
            compile: Whether to try compiling ``@hook``-decorated functions (default ``True``).
            deme_selector: Optional deme selector.
                - ``None``: keep panmictic default behavior (no explicit selector override)
                - non-``None``: passed into hook registration for spatial filtering

        Raises:
            ValueError: If event does not exist or hook_id is already in use.

        Examples:
            >>> # Plain function (backward compatible)
            >>> pop.set_hook('first', lambda p: print(f'Step {p.tick}'))
            >>>
            >>> # Declarative @hook function (auto-compiled)
            >>> @hook()
            >>> def reduce_juveniles():
            ...     return [Op.scale(genotypes='AA', ages=[0, 1], factor=0.9)]
            >>> pop.set_hook('early', reduce_juveniles)
            >>>
            >>> # Selector @hook function (auto-compiled)
            >>> @hook(selectors={'target': 'AA'})
            >>> def release(pop, target):
            ...     pop.state.individual_count[1, 2, target] += 100
            >>> pop.set_hook('first', release)
        """
        if event_name not in self.ALLOWED_EVENTS:
            raise ValueError(f"Event '{event_name}' not in {self.ALLOWED_EVENTS}")

        # BasePopulation itself is panmictic. Non-wildcard deme selectors are
        # interpreted by SpatialPopulation orchestration and should not be
        # consumed here.
        if deme_selector is not None and deme_selector != "*":
            warnings.warn(
                "BasePopulation ignores non-'*' deme_selector. "
                "Apply deme selection through SpatialPopulation-level logic instead.",
                UserWarning,
                stacklevel=2,
            )
            deme_selector = None

        # Check if function has @hook metadata and should be compiled
        hook_meta = getattr(func, 'meta', None)
        if is_numba_enabled() and hook_meta is None:
            raise TypeError(
                "Python-layer hooks are not allowed when Numba is enabled. "
                "Use @hook(...) with a compilable body or disable Numba."
            )

        if compile and hook_meta is not None:
            # Use the hook's register method with event override
            register_fn = getattr(func, 'register', None)
            if register_fn is not None:
                # Panmictic path: do not force any selector override.
                if deme_selector is None:
                    register_fn(self, event_override=event_name)
                else:
                    register_fn(self, event_override=event_name, deme_selector_override=deme_selector)
                # Compiled hooks are stored in _compiled_hooks.
                # Only selector-mode hooks with py_wrapper are mirrored to _hooks.
                self._hook_executor = None
                return

        # Traditional registration (no compilation)
        actual_name = hook_name or getattr(func, '__name__', None)

        current_ids = [hid for hid, _, _ in self._hooks[event_name]]

        if hook_id is None:
            hook_id = (max(current_ids) + 1) if current_ids else 0

        if hook_id in current_ids:
            raise ValueError(f"hook_id {hook_id} already exists in event '{event_name}'")

        self._hooks[event_name].append((hook_id, actual_name, func))
        # Sort by hook ID to preserve execution order.
        self._hooks[event_name].sort(key=lambda x: x[0])
        self._hook_executor = None

    def trigger_event(self, event_name: str, deme_id: int = -1) -> int:
        """
                Trigger an event and execute all registered hooks.

                Execution order:
                1. CSR operations (Numba fast path)
                2. ``njit_fn`` hooks (user-defined Numba functions)
                3. ``py_wrapper`` hooks (Python wrapper functions)

        Args:
                        event_name: Event name to trigger.
                        deme_id: Deme index. Default -1 for non-spatial populations.

        Returns:
                        int: ``RESULT_CONTINUE`` (0) to continue, ``RESULT_STOP`` (1) to stop.

        Note:
                        - Prefer HookExecutor (unified three-layer coordination).
                        - If executor is not built, fall back to traditional ``_hooks``
                            (Python callbacks only).
                        - In accelerated ``run()``, core events are mostly executed by engine;
                            ``trigger_event`` is used mainly for explicit events (for example ``finish``)
                            and compatibility paths.

        Examples:
                        >>> result = pop.trigger_event('first')  # Executes all 'first' hooks
            >>> if result == RESULT_STOP:
            ...     print("Simulation stopped by hook")
        """
        from natal.hooks import RESULT_CONTINUE

        # Prefer HookExecutor when available.
        if self._hook_executor is not None:
            from natal.hooks import EVENT_ID_MAP
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is not None:
                result = self._hook_executor.execute_event(event_id, self, self.tick, deme_id=deme_id)
                return result

        # Fallback to traditional _hooks for compatibility.
        for _, _, hook in self._hooks.get(event_name, []):
            hook(self)

        return RESULT_CONTINUE

    def get_hooks(self, event_name: str) -> List[HookEntry]:
        """
        Get all registered hooks for a specific event.

        Args:
            event_name: Event name.

        Returns:
            List of tuples ``[(hook_id, hook_name, hook_func), ...]``.
        """
        return list(self._hooks.get(event_name, []))

    def remove_hook(self, event_name: str, hook_id: int) -> bool:
        """
        Remove a specific hook from an event.

        Args:
            event_name: Event name.
            hook_id: Hook ID.

        Returns:
            True if removed successfully, otherwise False.
        """
        if event_name not in self._hooks:
            return False

        original_len = len(self._hooks[event_name])
        self._hooks[event_name] = [(hid, name, func) for hid, name, func in self._hooks[event_name]
                                    if hid != hook_id]
        self._hook_executor = None
        return len(self._hooks[event_name]) < original_len

    # ── Compiled Hooks (DSL / Numba-friendly) ────────────────────────

    def _register_compiled_hook(self, desc: CompiledHookDescriptor) -> None:
        """Register a compiled hook descriptor.

        Args:
            desc: CompiledHookDescriptor from hooks module.

        Note:
            To avoid maintaining two divergent hook sources, this method only
            mirrors compiled hooks into traditional ``_hooks`` when a real
            Python wrapper exists (selector-mode hooks). Pure declarative and
            njit hooks stay in ``_compiled_hooks`` and are executed by engine
            (or by HookExecutor when trigger_event is used).
        """
        self._compiled_hooks.append(desc)
        self._hook_executor = None

        from natal.numba.utils import NUMBA_ENABLED
        if NUMBA_ENABLED and desc.py_wrapper is not None and desc.njit_fn is None:
            raise TypeError(
                f"Python py_wrapper hook '{desc.name}' is not allowed when Numba is enabled. "
                "Please convert it to @njit or use declarative Op hooks."
            )

        # Mirror only real Python wrappers for trigger_event compatibility.
        # Do not inject no-op placeholders for declarative/njit hooks.
        if desc.py_wrapper is None:
            return
        hook_func = desc.py_wrapper

        # Register with traditional system
        event_name = desc.event
        if event_name in self._hooks:
            current_ids = [hid for hid, _, _ in self._hooks[event_name]]
            hook_id = desc.priority
            # Avoid duplicate IDs
            while hook_id in current_ids:
                hook_id += 1
            self._hooks[event_name].append((hook_id, desc.name, hook_func))
            self._hooks[event_name].sort(key=lambda x: x[0])

    def has_python_hooks(self) -> bool:
        """Return whether any Python-layer hooks are currently registered."""
        hooks_map = cast(Dict[str, List[HookEntry]], getattr(self, "_hooks", {}))
        return any(len(entries) > 0 for entries in hooks_map.values())

    def has_mixed_hook_types(self) -> bool:
        """Return whether any event mixes declarative/njit/python hook types."""
        for event_name in self.ALLOWED_EVENTS:
            kinds: set[str] = set()
            for desc in self.get_compiled_hooks(event_name):
                if getattr(desc, "plan", None) is not None:
                    kinds.add("declarative")
                if getattr(desc, "njit_fn", None) is not None:
                    kinds.add("njit")
                if getattr(desc, "py_wrapper", None) is not None and getattr(desc, "njit_fn", None) is None:
                    kinds.add("python")
            if len(kinds) > 1:
                return True
        return False

    def should_use_python_dispatch(self) -> bool:
                """Return whether this population should run with Python event dispatch.

                Policy:
                        - When Numba is disabled, any registered hook type uses Python
                            dispatch so py/declarative/njit hooks share one sequential path.
                        - When Numba is enabled, mixed hook-type timelines are handled by
                            unified njit functions generated in ``CompiledEventHooks.from_compiled_hooks``,
                            so no Python fallback is needed.
                """
                if not is_numba_enabled():
                        return self.has_python_hooks() or len(self.get_compiled_hooks()) > 0
                return False

    def ensure_hook_executor(self) -> None:
        """Build HookExecutor lazily for Python event-dispatch paths."""
        if self._hook_executor is None:
            self._hook_executor = self._build_hook_executor()

    def register_compiled_hook(self, desc: CompiledHookDescriptor) -> None:
        """Public wrapper for registering compiled hooks."""
        self._register_compiled_hook(desc)

    def get_compiled_hooks(self, event: Optional[str] = None) -> List[Any]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of CompiledHookDescriptor sorted by priority.
        """
        hooks = cast(List[Any], getattr(self, "_compiled_hooks", []))
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def register_declarative_hook(
        self: BasePopulation[Any],  # type: ignore[reportGeneralTypeIssues, reportMissingTypeArgument]  # mixin, host is BasePopulation subclass
        event: str,
        ops: List[Any],
        priority: int = 0,
        name: str = "declarative_hook"
    ) -> Any:
        """Register a declarative hook from a list of operations.

        This is an alternative to using the @hook decorator.

        Args:
            event: Event name ('first', 'early', 'late', 'finish')
            ops: List of HookOp operations (from Op.scale, Op.add, etc.)
            priority: Execution priority (lower = earlier)
            name: Hook name for debugging

        Returns:
            CompiledHookDescriptor: The compiled descriptor

        Examples:
            >>> from natal.hooks import Op
            >>> pop.register_declarative_hook(
            ...     event='early',
            ...     ops=[
            ...         Op.scale(genotypes='AA', ages=[0, 1], factor=0.9),
            ...         Op.add(genotypes='*', ages=0, delta=50, when='tick % 10 == 0'),
            ...     ],
            ...     name='juvenile_control'
            ... )
        """
        from natal.hooks import compile_declarative_hook
        desc = compile_declarative_hook(
            ops,
            self,
            event,
            priority=priority,
            name=name,
        )
        self._register_compiled_hook(desc)
        return desc

    def _build_hook_program(self) -> HookProgram:
        """Build HookProgram from compiled hooks.

        This packs all compiled hooks into a Numba-compatible jitclass
        for efficient execution during simulation.

        Returns:
            HookProgram: Compiled hook program data
        """
        from natal.hooks import EVENT_NAMES, HookProgram

        events = EVENT_NAMES
        n_events = len(events)

        # 1. Collect all hooks per event
        hook_offsets: List[int] = [0]
        hook_list_by_event: List[List[CompiledHookDescriptor]] = []

        for event_name in events:
            hooks = self.get_compiled_hooks(event_name)
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))

        n_hooks = hook_offsets[-1]

        # 2. Pack all operation data
        all_op_types: List[int] = []
        all_zidx_offsets: List[int] = [0]
        all_zidx_data: List[int] = []
        all_age_offsets: List[int] = [0]
        all_age_data: List[int] = []
        all_sex_masks: List[bool] = []
        all_params: List[float] = []
        all_cond_offsets: List[int] = [0]
        all_cond_types: List[int] = []
        all_cond_params: List[int] = []

        all_deme_sel_types: List[int] = []
        all_deme_sel_offsets: List[int] = [0]
        all_deme_sel_data: List[int] = []

        n_ops_list: List[int] = []
        op_offsets: List[int] = [0]

        for hooks in hook_list_by_event:
            for hook in hooks:
                plan = hook.plan
                if plan is None or plan.n_ops == 0:
                    n_ops_list.append(0)
                    op_offsets.append(op_offsets[-1])
                    continue

                n_ops_list.append(plan.n_ops)

                # Pack operation data
                all_op_types.extend(plan.op_types.tolist())

                # Handle zidx (adjust offsets for concatenation)
                zidx_offset_base = len(all_zidx_data)
                for i in range(plan.n_ops):
                    all_zidx_offsets.append(
                        zidx_offset_base + plan.zidx_offsets[i + 1] - plan.zidx_offsets[0]
                    )
                all_zidx_data.extend(plan.zidx_data.tolist())

                # Handle age
                age_offset_base = len(all_age_data)
                for i in range(plan.n_ops):
                    all_age_offsets.append(
                        age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0]
                    )
                all_age_data.extend(plan.age_data.tolist())

                # Handle sex masks (flatten 2D -> 1D)
                all_sex_masks.extend(plan.sex_masks.flatten().tolist())

                # Handle params, conditions
                all_params.extend(plan.params.tolist())
                cond_offset_base = len(all_cond_types)
                for i in range(plan.n_ops):
                    all_cond_offsets.append(
                        cond_offset_base + plan.condition_offsets[i + 1] - plan.condition_offsets[0]
                    )
                all_cond_types.extend(plan.condition_types.tolist())
                all_cond_params.extend(plan.condition_params.tolist())

                op_offsets.append(len(all_op_types))

                # Pack deme selector from CompiledHookDescriptor
                sel = hook.deme_selector
                if sel == "*":
                    all_deme_sel_types.append(0)
                elif isinstance(sel, int):
                    all_deme_sel_types.append(1)
                    all_deme_sel_data.append(int(sel))
                elif isinstance(sel, range):
                    all_deme_sel_types.append(2)
                    all_deme_sel_data.append(int(sel.start))
                    all_deme_sel_data.append(int(sel.stop))
                else:
                    all_deme_sel_types.append(3)
                    all_deme_sel_data.extend([int(x) for x in sel])
                all_deme_sel_offsets.append(len(all_deme_sel_data))

        # 3. Create HookProgram
        return HookProgram(
            n_events=np.int32(n_events),
            n_hooks=np.int32(n_hooks),
            hook_offsets=np.array(hook_offsets, dtype=np.int32),
            n_ops_list=np.array(n_ops_list, dtype=np.int32),
            op_offsets=np.array(op_offsets, dtype=np.int32),
            op_types_data=np.array(all_op_types, dtype=np.int32),
            zidx_offsets_data=np.array(all_zidx_offsets, dtype=np.int32),
            zidx_data=np.array(all_zidx_data, dtype=np.int32),
            age_offsets_data=np.array(all_age_offsets, dtype=np.int32),
            age_data=np.array(all_age_data, dtype=np.int32),
            sex_masks_data=np.array(all_sex_masks, dtype=np.bool_),
            params_data=np.array(all_params, dtype=np.float64),
            condition_offsets_data=np.array(all_cond_offsets, dtype=np.int32),
            condition_types_data=np.array(all_cond_types, dtype=np.int32),
            condition_params_data=np.array(all_cond_params, dtype=np.int32),
            deme_selector_types=np.array(all_deme_sel_types, dtype=np.int32),
            deme_selector_offsets=np.array(all_deme_sel_offsets, dtype=np.int32),
            deme_selector_data=np.array(all_deme_sel_data, dtype=np.int32),
        )

    def _build_hook_executor(self):
        """Build HookExecutor from compiled hooks and HookProgram.

        HookExecutor is a Python-layer coordinator that manages:
        1. CSR operations via execute_csr_event_program()
        2. njit_fn hooks (user Numba functions)
        3. py_wrapper hooks (Python wrappers for selector mode)

        Returns:
            HookExecutor: Executor instance, or None if no hooks compiled
        """
        from natal.hooks import HookExecutor

        # Get or build HookProgram for CSR operations
        program = self._build_hook_program()
        program_available = True

        # Get all compiled hooks
        compiled_hooks = self._compiled_hooks
        if not compiled_hooks:
            return None

        # If no program (no CSR operations), create an empty one
        # so HookExecutor can still manage njit_fn and py_wrapper hooks
        if not program_available:
            program = self._create_empty_hook_program()

        # Create executor
        executor = HookExecutor.from_compiled_hooks(program, compiled_hooks)
        return executor

    def _create_empty_hook_program(self):
        """Create an empty HookProgram for non-CSR operations.

        Used when there are no declarative Op.* operations,
        but there are njit_fn or py_wrapper hooks.
        """
        from natal.hooks import NUM_EVENTS, HookProgram

        n_events = NUM_EVENTS

        # Create empty CSR arrays
        hook_offsets = np.zeros(n_events + 1, dtype=np.int32)
        op_offsets = np.array([0], dtype=np.int32)

        return HookProgram(
            n_events=np.int32(n_events),
            n_hooks=np.int32(0),
            hook_offsets=hook_offsets,
            n_ops_list=np.array([], dtype=np.int32),
            op_offsets=op_offsets,
            op_types_data=np.array([], dtype=np.int32),
            zidx_offsets_data=np.array([0], dtype=np.int32),
            zidx_data=np.array([], dtype=np.int32),
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

    def get_compiled_event_hooks(self) -> LifecycleWrappers:
        """Get compiled hooks and lifecycle wrappers for kernel-based simulation.

        This method collects all registered hooks, compiles them into
        Numba-friendly combined functions, and wraps them in pre-compiled
        lifecycle loop functions (tick / run).

        Returns:
            LifecycleWrappers: Container with compiled event hooks
                (``.hooks.first`` etc.) plus pre-compiled lifecycle loop
                functions (``.run_fn``, ``.run_discrete_fn``, etc.).

        Examples:
            >>> wrappers = pop.get_compiled_event_hooks()
            >>> wrappers.run_fn is not None
            True
        """
        from natal.engine.lifecycle_wrappers import compile_lifecycle_wrappers
        registry = self._build_hook_program()
        return compile_lifecycle_wrappers(
            self._compiled_hooks,
            registry=registry,
            include_spatial_wrappers=False,
        )
