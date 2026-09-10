"""Build-time hook compilation and program packing.

The builder (PopulationBuilder) resolves every declared hook item once against
the final registry and injects the resulting plan into the population at
construction.  Populations expose no registration entry, so this module is
the single compile path shared by the panmictic and the spatial build
flows; the packed :class:`HookProgram` is fixed for the lifetime of a
population.
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
    EVENT_NAMES,
    RPN_LITERAL,
    CompiledHookDescriptor,
    DemeSelector,
    HookLayout,
    HookOp,
    HookProgram,
    OpType,
)

if TYPE_CHECKING:
    from natal.frontend.data import ModelDraft
    from natal.frontend.genetics import Species
    from natal.frontend.registry.index import IndexRegistry


class HookLayoutContext:
    """Build-time layout provider for hook compilation.

    Carries the final (already compressed) registry, the species, and the
    final draft so selectors resolve to the exact indices the executed
    model uses.  Populations expose the same read-only attribute surface;
    this class exists so the builder can compile before the population
    exists.

    Attributes:
        index_registry: Final active-type registry (post compression).
        species: Genetic architecture used for pattern resolution.
        config: Final draft (read for layout facts such as ``n_ages``).
    """

    __slots__ = ("_index_registry", "_species", "_config")

    def __init__(
        self,
        index_registry: IndexRegistry,
        species: Species,
        config: ModelDraft,
    ) -> None:
        """Bind the final layout facts (see class attributes)."""
        self._index_registry = index_registry
        self._species = species
        self._config = config

    @property
    def index_registry(self) -> IndexRegistry:
        """Final active-type registry (post compression)."""
        return self._index_registry

    @property
    def species(self) -> Species:
        """Genetic architecture used for pattern resolution."""
        return self._species

    @property
    def config(self) -> ModelDraft:
        """Final draft (read for layout facts such as ``n_ages``)."""
        return self._config


def compile_hook_call(
    layout: HookLayout,
    *items: object,
    event: Optional[str] = None,
    priority: int = 0,
    deme: DemeSelector = "*",
    name: Optional[str] = None,
    allowed_events: Sequence[str] = ("first", "early", "late", "finish"),
) -> List[CompiledHookDescriptor]:
    """Resolve one ``.hooks()`` call into compiled descriptors (pure).

    Accepted items mirror the historical registration surface:

    - a :class:`~natal.frontend.hooks.types.HookOp` (or a list of them)
      → one declarative CSR descriptor; the event rides on the op, the
      call, or the decorator default;
    - a ``@hook``-decorated function → compiled according to its shape
      (declarative / callback / selector callback);
    - a plain single-parameter callable → a Python callback.

    Args:
        layout: Final layout the selectors resolve against.
        *items: Hook declarations (ops, op lists, decorated or plain
            single-parameter callables).
        event: Default event for items that do not carry one.
        priority: Default priority for items that do not carry one.
        deme: Default deme selector (``"*"`` = all demes).  Carried
            verbatim onto the compiled descriptors; panmictic callers
            normalize non-wildcard selectors at declaration time.
        name: Optional override name for grouped op declarations.
        allowed_events: Event names the target model accepts.

    Returns:
        New descriptors in declaration order (identity dedupe is applied
        by the caller when accumulating across calls).

    Raises:
        ValueError: If an event name is unknown or no event can be
            resolved for an item.
        TypeError: If an item has an unsupported shape (including the
            removed ``(state, config, deme_id)`` signature).
    """
    if event is not None and event not in allowed_events:
        raise ValueError(f"Event '{event}' not in {list(allowed_events)}")

    descriptors: List[CompiledHookDescriptor] = []
    for item in items:
        if isinstance(item, HookOp):
            descriptors.append(
                _compile_op_group([item], layout, event, item.priority, deme, name)
            )
        elif isinstance(item, (list, tuple)):
            raw: List[object] = list(cast("Sequence[object]", item))
            if not all(isinstance(op, HookOp) for op in raw):
                raise TypeError(
                    "Op-list hook items must contain only HookOp "
                    "objects (build them with Op.scale / Op.add / ...)."
                )
            descriptors.append(
                _compile_op_group(
                    [op for op in raw if isinstance(op, HookOp)],
                    layout,
                    event,
                    priority,
                    deme,
                    name,
                )
            )
        elif callable(item):
            descriptors.append(
                _compile_callable_item(item, layout, event, priority, deme)
            )
        else:
            raise TypeError(
                f"Unsupported hook item of type {type(item).__name__!r}. "
                "Use HookOp objects (Op.*), @hook-decorated functions, "
                "or single-parameter callables."
            )
    return descriptors


def _compile_op_group(
    ops: List[HookOp],
    layout: HookLayout,
    event: Optional[str],
    priority: int,
    deme: DemeSelector,
    name: Optional[str],
) -> CompiledHookDescriptor:
    """Compile one declarative descriptor from a group of ops."""
    from natal.frontend.hooks.entry.declarative import compile_declarative_hook

    resolved_event = next((op.event for op in ops if op.event is not None), event)
    if resolved_event is None:
        # Documented default (2_hooks.md): declarative ops declared without
        # an event fire at "early".  An explicit declaration-level event
        # still overrides, and an op-level event still wins over the call.
        resolved_event = "early"
    descriptor_name = name or f"declarative_{resolved_event}_{len(ops)}op"
    desc = compile_declarative_hook(
        ops,
        layout,
        resolved_event,
        priority,
        deme_selector=deme,
        name=descriptor_name,
    )
    # Identity: a single op maps to itself; a group to its op tuple.
    desc.source = ops[0] if len(ops) == 1 else tuple(ops)
    return desc


def _compile_callable_item(
    func: object,
    layout: HookLayout,
    event: Optional[str],
    priority: int,
    deme: DemeSelector,
) -> CompiledHookDescriptor:
    """Compile one callable, honoring ``@hook`` metadata when present."""
    meta = getattr(func, "meta", None)
    if meta is not None:
        register_fn = getattr(func, "register", None)
        if register_fn is None:
            raise TypeError(
                "Objects carrying hook meta must expose register(); "
                "decorate plain functions with @nt.hook."
            )
        return cast(
            CompiledHookDescriptor,
            register_fn(
                layout,
                event_override=event,
                deme_selector_override=deme if deme != "*" else None,
            ),
        )

    # Plain callable: must be the single-parameter callback form.  The
    # signature check happens here (not only in the decorator) so that
    # bare functions get the same guidance as decorated ones.
    from natal.frontend.hooks.types import CompiledHookDescriptor as _Desc

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
            "hooks with @nt.hook."
        )
    if event is None:
        raise ValueError(
            "No event specified for hook "
            f"'{getattr(func, '__name__', '<anonymous>')}'. Pass "
            ".hooks(..., event='early') or decorate with @nt.hook."
        )
    return _Desc(
        name=getattr(func, "__name__", "hook"),
        event=event,
        priority=priority,
        deme_selector=deme,
        callback=cast("Callable[[object], Optional[int]]", func),
        source=func,
    )


def same_hook_identity(
    left: CompiledHookDescriptor, right: CompiledHookDescriptor
) -> bool:
    """Return whether two descriptors are the same (source, event) hook.

    The same object declared for a *different* event is a distinct hook
    instance and compiles again.
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


def append_deme_selector(
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


def build_hook_program(
    descriptors: Sequence[CompiledHookDescriptor],
    *,
    order_by_priority: bool,
) -> HookProgram:
    """Pack compiled descriptors into one CSR ``HookProgram``.

    Callback-only descriptors contribute a (zero-op) hook slot so
    deme-selector arrays stay aligned with ``n_hooks``.  The slot column
    ``python_callback_slots`` marks those slots with the callback's index
    inside the event's callback list, so the executors interleave CSR
    plans and Python callbacks by one stable order.

    Args:
        descriptors: Compiled descriptors to pack.  With
            ``order_by_priority=True`` each event's slots are stably
            sorted by priority (panmictic semantics: ties keep
            declaration order).  With ``False`` the given order is kept
            verbatim (spatial compact lists encode their own grouping).
        order_by_priority: Whether to sort each event's descriptors.

    Returns:
        The plain-data CSR program consumed by the native executors and
        the callback bridge.
    """
    n_events = len(EVENT_NAMES)

    hook_offsets: List[int] = [0]
    hook_list_by_event: List[List[CompiledHookDescriptor]] = []
    for event_name in EVENT_NAMES:
        hooks = [h for h in descriptors if h.event == event_name]
        if order_by_priority:
            # Stable sort: equal priorities keep declaration order.
            hooks = sorted(hooks, key=lambda h: h.priority)
        hook_list_by_event.append(hooks)
        hook_offsets.append(hook_offsets[-1] + len(hooks))

    n_hooks = hook_offsets[-1]

    # Callback slot column: per event, the running index of each
    # callback-only descriptor matches the HookRunner's stable priority
    # sort of the same descriptors (both consume the same order).
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
                # Keep offset arrays aligned even for hooks without
                # declarative operations (e.g. pure python descriptors):
                # the deme selector must still be packed, or the native
                # matcher indexes an empty array for this hook slot.
                n_ops_list.append(0)
                op_offsets.append(op_offsets[-1])
                append_deme_selector(
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
            append_deme_selector(
                hook.deme_selector,
                all_deme_sel_types,
                all_deme_sel_offsets,
                all_deme_sel_data,
            )

    import numpy as np

    from natal.frontend.hooks.types import HookProgram as _Program

    return _Program(
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
