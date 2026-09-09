"""Hook management mixin for BasePopulation.

The registration entry stores one descriptor payload pair (CSR plan or
Python callback).  Native Rust sessions execute declarative plans and bridge
Python callbacks at event boundaries in one stable priority order.

The njit-era registration surface (``set_hook`` / ``get_hooks`` /
``remove_hook``, the plain ``(state, config, deme_id)`` hook map, and the
``hook_entries`` bookkeeping) was removed — the 26 decisions give njit
hooks no migration channel.
"""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from natal.frontend.hooks.types import (
    EVENT_ID_MAP,
    RESULT_CONTINUE,
    RPN_LITERAL,
    CompiledHookDescriptor,
    DemeSelector,
    HookOp,
    OpType,
    RunProgram,
)

if TYPE_CHECKING:
    from typing import Any as _Any
    from typing import Protocol as _Protocol

    from natal.frontend.hooks.tick_context import HookRunner
    from natal.frontend.hooks.types import HookProgram
    from natal.frontend.population.base import BasePopulation

    # Host-contract alias: the mixin is only mixed into BasePopulation
    # subclasses; the ``Any`` type argument stands for the host's
    # ``T_State`` (a documented host-contract ``Any``).
    _Population = BasePopulation[_Any]

    class _CallbackBridge(_Protocol):
        """Structural type of the Rust backend adapter's callback channel."""

        def set_python_callbacks(
            self,
            first: List[Callable[..., int]],
            early: List[Callable[..., int]],
            late: List[Callable[..., int]],
            finish: List[Callable[..., int]] | None = None,
        ) -> None:
            """Register per-event callback lists."""
            ...


class HookManagerMixin:
    """Mixin providing hook registration, compilation, and dispatch.

    Expects the host class (BasePopulation) to define these attributes:
    ``ALLOWED_EVENTS``, ``tick``, ``state``, ``config``,
    ``compiled_hook_descriptors``, ``_hook_runner``,
    ``_run_program``, and the ``_rust_needs_rebuild`` flag.
    """

    # Declared here so pyright knows these come from the host class.
    ALLOWED_EVENTS: list[str]  # type: ignore[assignment]
    tick: int  # type: ignore[assignment]
    # ``object`` is a host-contract declaration only: BasePopulation owns the
    # typed ``state``/``config`` properties; this mixin just narrows through
    # ``cast`` at the call site.
    state: object
    config: object
    compiled_hook_descriptors: list[CompiledHookDescriptor]
    _hook_runner: Optional[HookRunner]
    _run_program: RunProgram
    # Rust dirty-set bridge owned by BasePopulation (contract field names).
    _rust_needs_rebuild: bool

    # ── Hook registration (the single entry) ─────────────────────────

    def register_hooks(
        self,
        *items: object,
        event: Optional[str] = None,
        priority: int = 0,
        deme: DemeSelector = "*",
        name: Optional[str] = None,
    ) -> None:
        """Register hooks — the single entry behind ``.hooks()``.

        Accepted items:

        - a :class:`~natal.frontend.hooks.types.HookOp` (or a list of
          them) → one declarative CSR descriptor; the event rides on the
          op, the call, or the decorator default;
        - a ``@hook``-decorated function → compiled according to its
          shape (declarative / callback / selector callback);
        - a plain single-parameter callable → a Python callback.

        Registration is idempotent by object identity: registering the
        same object (or the same op group) twice is a no-op.

        Args:
            *items: Hook registrations (ops, op lists, decorated or plain
                single-parameter callables).
            event: Default event for items that do not carry one.
            priority: Default priority for items that do not carry one.
            deme: Default deme selector (``"*"`` = all demes).
            name: Optional override name for grouped op registrations.

        Raises:
            ValueError: If an event name is unknown or no event can be
                resolved for an item.
            TypeError: If an item has an unsupported shape (including the
                removed ``(state, config, deme_id)`` signature).
        """
        if event is not None and event not in self.ALLOWED_EVENTS:
            raise ValueError(f"Event '{event}' not in {self.ALLOWED_EVENTS}")

        # BasePopulation itself is panmictic.  Non-wildcard deme selectors
        # are interpreted by SpatialPopulation orchestration and should not
        # reach the per-deme descriptors.
        if deme != "*":
            import warnings

            warnings.warn(
                "BasePopulation ignores non-'*' deme selectors. "
                "Apply deme selection through SpatialPopulation-level "
                "logic instead.",
                UserWarning,
                stacklevel=2,
            )
            deme = "*"

        for item in items:
            if isinstance(item, HookOp):
                self._register_op_group([item], event, item.priority, deme, name)
            elif isinstance(item, (list, tuple)):
                raw: List[object] = list(cast("Sequence[object]", item))
                if not all(isinstance(op, HookOp) for op in raw):
                    raise TypeError(
                        "Op-list hook items must contain only HookOp "
                        "objects (build them with Op.scale / Op.add / ...)."
                    )
                self._register_op_group(
                    [op for op in raw if isinstance(op, HookOp)],
                    event,
                    priority,
                    deme,
                    name,
                )
            elif callable(item):
                self._register_callable_item(item, event, priority, deme)
            else:
                raise TypeError(
                    f"Unsupported hook item of type {type(item).__name__!r}. "
                    "Use HookOp objects (Op.*), @hook-decorated functions, "
                    "or single-parameter callables."
                )

    def _register_op_group(
        self,
        ops: List[HookOp],
        event: Optional[str],
        priority: int,
        deme: DemeSelector,
        name: Optional[str],
    ) -> None:
        """Register one declarative descriptor from a group of ops."""
        from natal.frontend.hooks.entry.declarative import compile_declarative_hook

        resolved_event = next((op.event for op in ops if op.event is not None), event)
        if resolved_event is None:
            # Documented default (2_hooks.md, plan §〇): declarative ops
            # registered without an event fire at "early".  An explicit
            # registration-level event still overrides, and an op-level
            # event still wins over the registration call.
            resolved_event = "early"
        descriptor_name = name or f"declarative_{resolved_event}_{len(ops)}op"
        desc = compile_declarative_hook(
            ops,
            cast("_Population", self),
            resolved_event,
            priority,
            deme_selector=deme,
            name=descriptor_name,
        )
        # Identity: a single op maps to itself; a group to its op tuple.
        desc.source = ops[0] if len(ops) == 1 else tuple(ops)
        self._register_compiled_hook(desc)

    def _register_callable_item(
        self,
        func: object,
        event: Optional[str],
        priority: int,
        deme: DemeSelector,
    ) -> None:
        """Register one callable, honoring ``@hook`` metadata when present."""
        meta = getattr(func, "meta", None)
        if meta is not None:
            register_fn = getattr(func, "register", None)
            if register_fn is None:
                raise TypeError(
                    "Objects carrying hook meta must expose register(); "
                    "decorate plain functions with @nt.hook."
                )
            register_fn(
                cast("_Population", self),
                event_override=event,
                deme_selector_override=deme if deme != "*" else None,
            )
            return

        # Plain callable: must be the single-parameter callback form.  The
        # signature check happens here (not only in the decorator) so that
        # bare functions get the same guidance as decorated ones.
        callable_fn = cast("Callable[..., object]", func)
        params = [
            param
            for param in inspect.signature(callable_fn).parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and param.default is inspect.Signature.empty
        ]
        if len(params) != 1:
            raise TypeError(
                "Plain hook callables must take exactly one parameter: "
                "def hook(pop) -> int (a TickContext; return 0 to "
                "continue, nonzero to stop). Decorate declarative Op "
                "hooks with @nt.hook(event='...')."
            )
        if event is None:
            raise ValueError(
                "No event specified for hook "
                f"'{getattr(func, '__name__', '<anonymous>')}'. Pass "
                ".hooks(..., event='early') or decorate with @nt.hook."
            )
        desc = CompiledHookDescriptor(
            name=getattr(func, "__name__", "hook"),
            event=event,
            priority=priority,
            deme_selector=deme,
            callback=cast("CallbackOfPopulation", func),
        )
        desc.source = func
        self._register_compiled_hook(desc)

    def _register_compiled_hook(self, desc: CompiledHookDescriptor) -> None:
        """Register a compiled hook descriptor (identity-idempotent)."""
        for existing in self.compiled_hook_descriptors:
            if self._same_hook_identity(existing, desc):
                return
        self.compiled_hook_descriptors.append(desc)
        self._hook_runner = None

        # The Rust session snapshots the CSR hook program and callbacks at
        # enable time; any post-enable registration must rebuild it before
        # the next run.  The sentinel routes _sync_rust_backend to a full
        # backend rebuild.
        # Structural mutation: the live session must be rebuilt before
        # the next run (the hook program moves into the session at build).
        self._rust_needs_rebuild = True
        self._refresh_run_program()

    @staticmethod
    def _same_hook_identity(
        left: CompiledHookDescriptor, right: CompiledHookDescriptor
    ) -> bool:
        """Return whether two descriptors are the same (source, event) hook.

        The same object registered for a *different* event is a distinct
        hook instance and registers again.
        """
        if left.event != right.event:
            return False
        left_source: object = left.source
        right_source: object = right.source
        if left_source is None or right_source is None:
            return False
        if isinstance(left_source, tuple) and isinstance(right_source, tuple):
            lefts = cast("Tuple[object, ...]", left_source)
            rights = cast("Tuple[object, ...]", right_source)
            return len(lefts) == len(rights) and all(
                a is b for a, b in zip(lefts, rights)
            )
        return left_source is right_source

    # ── Dispatch ──────────────────────────────────────────────────────

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Trigger an event and execute all registered hooks for it.

        Execution order per event: CSR declarative plans and Python
        callbacks interleaved by one stable ascending-priority order
        (each callback receives a fresh
        :class:`~natal.frontend.hooks.tick_context.TickContext`).

        Args:
            event_name: Event name to trigger.
            deme_id: Deme index the hooks execute as.  0 for panmictic
                populations; the live deme index when the population is
                managed by a SpatialPopulation.

        Returns:
            int: ``RESULT_CONTINUE`` (0) to continue, ``RESULT_STOP`` (1)
            to stop.
        """
        native = getattr(self, "_runtime_parameter_writer", None)
        if native is None:
            native = getattr(self, "_rust_lifecycle_backend", None)
        if native is None:
            # Directly constructed standalone populations do not create a
            # session until the first operation that needs native execution.
            # Managed spatial demes have a runtime writer and therefore stay
            # owned by their SpatialPopulation container.
            initialize = getattr(self, "_initialize_session", None)
            if callable(initialize):
                initialize(seed=int(getattr(self, "_rust_backend_seed", 0) or 0))
                native = getattr(self, "_rust_lifecycle_backend", None)
        if native is not None and hasattr(native, "trigger_event"):
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is None:
                return RESULT_CONTINUE
            # Refresh changed hook programs without replacing session state/RNG.
            if getattr(self, "_runtime_parameter_writer", None) is None:
                pop = cast("_Population", self)
                native.bind_history(pop.history._store, pop._params_log)  # pyright: ignore[reportPrivateUsage]  # manual events use the same native log as run checkpoints.
                pop.history._bind_checkpoint_pruner(native.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.
                native.configure_program(self._run_program.hooks, self.config)
                self._register_rust_callbacks(native)
            if getattr(self, "_runtime_parameter_writer", None) is None:
                result = int(native.trigger_event(event_id, deme_id))
            else:
                result = int(native.trigger_event(event_id))
            cast("_Population", self)._mark_state_cache_stale()  # pyright: ignore[reportPrivateUsage]  # host owns its snapshot cache
            return result
        raise RuntimeError(
            "Native hook execution is unavailable; initialize a Rust session "
            "before triggering events."
        )

    # ── Introspection ─────────────────────────────────────────────────

    def get_compiled_hooks(
        self, event: Optional[str] = None
    ) -> List[CompiledHookDescriptor]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of ``CompiledHookDescriptor`` sorted by priority.
        """
        hooks = self.compiled_hook_descriptors
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def has_python_callbacks(self) -> bool:
        """Return whether any registered hook is a Python callback."""
        return any(desc.callback is not None for desc in self.compiled_hook_descriptors)

    def has_python_hooks(self) -> bool:
        """Back-compatible alias for :meth:`has_python_callbacks`."""
        return self.has_python_callbacks()

    def invalidate_hook_dispatch(self) -> None:
        """Invalidate callback wrappers after hook storage changes.

        Declarative execution belongs to the native session.  The method is
        retained as a small lifecycle seam for SpatialPopulation, which
        invalidates all demes after copy-on-write hook registration.
        """
        self._hook_runner = None

    def _ensure_hook_runner(self) -> HookRunner:
        """Return the callback runner, building it on first use."""
        if self._hook_runner is None:
            from natal.frontend.hooks.tick_context import HookRunner

            self._hook_runner = HookRunner(cast("_Population", self))
        runner: HookRunner = self._hook_runner
        return runner

    def _register_rust_callbacks(self, backend: _CallbackBridge) -> None:
        """Bridge Python callbacks into a Rust session.

        One adapter per in-tick event (first/early/late); each adapter has
        the Rust ``(ind, sperm, tick, deme_id) -> int`` signature and runs
        every callback of its event in priority order.  Events without
        callbacks register an empty list so Rust kernels skip the GIL
        boundary entirely.
        """
        from natal.frontend.hooks.types import (
            EVENT_EARLY,
            EVENT_FINISH,
            EVENT_FIRST,
            EVENT_LATE,
        )

        runner = self._ensure_hook_runner()
        backend.set_python_callbacks(
            runner.rust_callbacks(EVENT_FIRST),
            runner.rust_callbacks(EVENT_EARLY),
            runner.rust_callbacks(EVENT_LATE),
            runner.rust_callbacks(EVENT_FINISH),
        )

    def register_compiled_hook(self, desc: CompiledHookDescriptor) -> None:
        """Public wrapper for registering compiled hooks."""
        self._register_compiled_hook(desc)

    # ── Program assembly ──────────────────────────────────────────────

    def _refresh_run_program(self) -> None:
        """Rebuild the CSR hook plan inside the run program."""
        self._run_program = self._run_program._replace(hooks=self._build_hook_program())

    def _build_hook_program(self) -> HookProgram:
        """Pack all declarative descriptors into a CSR ``HookProgram``.

        Callback-only descriptors contribute a (zero-op) hook slot so
        deme-selector arrays stay aligned with ``n_hooks``.  The slot
        column ``python_callback_slots`` marks those slots with the
        callback's index inside the event's priority-ordered callback
        list, so the executors interleave CSR plans and Python callbacks
        by one stable priority order.
        """
        import numpy as np

        from natal.frontend.hooks.types import EVENT_NAMES, HookProgram

        events = EVENT_NAMES
        n_events = len(events)

        hook_offsets: List[int] = [0]
        hook_list_by_event: List[List[CompiledHookDescriptor]] = []

        for event_name in events:
            hooks = self.get_compiled_hooks(event_name)
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))

        n_hooks = hook_offsets[-1]

        # Callback slot column: per event, the running index of each
        # callback-only descriptor matches the HookRunner's stable
        # priority sort of the same descriptors (both sort the same
        # registration-ordered list by priority).
        callback_slots: List[int] = []
        for hooks in hook_list_by_event:
            next_callback_index = 0
            for hook in hooks:
                if hook.callback is not None:
                    callback_slots.append(next_callback_index)
                    next_callback_index += 1
                else:
                    callback_slots.append(-1)

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

        # OP_SET_PARAM / OP_CONVERT flattened data area (rebasing per
        # plan so offsets stay global).
        all_sp_param_ids: List[int] = []
        all_sp_every: List[int] = []
        all_sp_start: List[int] = []
        all_rpn_offsets: List[int] = [0]
        all_rpn_kinds: List[int] = []
        all_rpn_payload: List[int] = []
        all_sp_literals: List[float] = []
        all_convert_source_z: List[int] = []
        all_convert_target_z: List[int] = []
        has_set_param = False

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
                    self._append_deme_selector(
                        hook.deme_selector,
                        all_deme_sel_types,
                        all_deme_sel_offsets,
                        all_deme_sel_data,
                    )
                    continue

                n_ops_list.append(plan.n_ops)

                all_op_types.extend(plan.op_types.tolist())
                has_set_param = has_set_param or bool(
                    (plan.op_types == int(OpType.SET_PARAM)).any()
                )

                zidx_offset_base = len(all_zidx_data)
                for i in range(plan.n_ops):
                    all_zidx_offsets.append(
                        zidx_offset_base
                        + plan.zidx_offsets[i + 1]
                        - plan.zidx_offsets[0]
                    )
                all_zidx_data.extend(plan.zidx_data.tolist())

                age_offset_base = len(all_age_data)
                for i in range(plan.n_ops):
                    all_age_offsets.append(
                        age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0]
                    )
                all_age_data.extend(plan.age_data.tolist())

                all_sex_masks.extend(plan.sex_masks.flatten().tolist())

                all_params.extend(plan.params.tolist())
                cond_offset_base = len(all_cond_types)
                for i in range(plan.n_ops):
                    all_cond_offsets.append(
                        cond_offset_base
                        + plan.condition_offsets[i + 1]
                        - plan.condition_offsets[0]
                    )
                all_cond_types.extend(plan.condition_types.tolist())
                all_cond_params.extend(plan.condition_params.tolist())

                # set_param / convert payload columns are per-op lists of
                # the same length; rpn token streams are rebased like the
                # condition streams.
                all_sp_param_ids.extend(plan.sp_param_ids.tolist())
                all_sp_every.extend(plan.sp_every.tolist())
                all_sp_start.extend(plan.sp_start.tolist())
                rpn_offset_base = len(all_rpn_kinds)
                for i in range(plan.n_ops):
                    all_rpn_offsets.append(
                        rpn_offset_base + plan.rpn_offsets[i + 1] - plan.rpn_offsets[0]
                    )
                all_rpn_kinds.extend(plan.rpn_kinds.tolist())
                # Literal payloads index the program-wide shared pool:
                # rebase each plan's local indices onto the current pool
                # length, or later plans read earlier plans' literals.
                literal_base = len(all_sp_literals)
                kinds = plan.rpn_kinds.tolist()
                payload = plan.rpn_payload.tolist()
                all_rpn_payload.extend(
                    p + literal_base if k == RPN_LITERAL else p
                    for k, p in zip(kinds, payload)
                )
                all_sp_literals.extend(plan.sp_literals.tolist())
                all_convert_source_z.extend(plan.convert_source_z.tolist())
                all_convert_target_z.extend(plan.convert_target_z.tolist())

                op_offsets.append(len(all_op_types))
                self._append_deme_selector(
                    hook.deme_selector,
                    all_deme_sel_types,
                    all_deme_sel_offsets,
                    all_deme_sel_data,
                )

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
            sp_param_ids=np.array(all_sp_param_ids, dtype=np.int32),
            sp_every=np.array(all_sp_every, dtype=np.int32),
            sp_start=np.array(all_sp_start, dtype=np.int32),
            rpn_offsets=np.array(all_rpn_offsets, dtype=np.int32),
            rpn_kinds=np.array(all_rpn_kinds, dtype=np.int32),
            rpn_payload=np.array(all_rpn_payload, dtype=np.int32),
            sp_literals=np.array(all_sp_literals, dtype=np.float64),
            convert_source_z=np.array(all_convert_source_z, dtype=np.int32),
            convert_target_z=np.array(all_convert_target_z, dtype=np.int32),
            has_set_param=has_set_param,
            deme_selector_types=np.array(all_deme_sel_types, dtype=np.int32),
            deme_selector_offsets=np.array(all_deme_sel_offsets, dtype=np.int32),
            deme_selector_data=np.array(all_deme_sel_data, dtype=np.int32),
            python_callback_slots=np.array(callback_slots, dtype=np.int32),
        )

    @staticmethod
    def _append_deme_selector(
        sel: DemeSelector,
        types_out: List[int],
        offsets_out: List[int],
        data_out: List[int],
    ) -> None:
        """Append one serialized deme selector entry (keeps arrays aligned)."""
        if sel == "*":
            types_out.append(0)
        elif isinstance(sel, int):
            types_out.append(1)
            data_out.append(int(sel))
        elif isinstance(sel, range):
            types_out.append(2)
            data_out.append(int(sel.start))
            data_out.append(int(sel.stop))
        else:
            types_out.append(3)
            data_out.extend(int(x) for x in sel)
        offsets_out.append(len(data_out))


# TYPE_CHECKING-only callback alias matching the descriptor payload type.
if TYPE_CHECKING:
    from typing import Callable as _Callable

    CallbackOfPopulation = _Callable[[object], Optional[int]]  # noqa: F401
