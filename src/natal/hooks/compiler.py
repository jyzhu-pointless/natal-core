"""Unified hook entrypoints and event-wise compiler.
This module connects three authoring styles into one runtime contract:
1) declarative hooks (Op list -> CompiledHookPlan)
2) selector hooks (symbolic selectors -> wrapper/compiled callable)
3) custom hooks (user-provided njit or Python callback)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    cast,
)

from natal.hooks.declarative import compile_declarative_hook
from natal.hooks.selector import compile_selector_hook
from natal.numba_utils import njit_switch

from .declarative import HookOp
from .types import (
    EVENT_NAMES,
    DemeSelector,
    HookProgram,
    hash_key,
    is_njit_function,
    load_codegen_module,
    stable_callable_identity,
    write_codegen_module,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "engine" / "templates"


def _read_template(name: str) -> str:
    """Read a lifecycle codegen template from ``engine/templates/``."""
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


if TYPE_CHECKING:
    from natal.base_population import BasePopulation

    from .types import CompiledHookDescriptor

# Plain callable type — used everywhere that only needs "something you can call".
# noop hooks, njit functions, combined hooks all satisfy this.
HookCallable = Callable[..., Any]
DeclarativeCompiler = Callable[..., "CompiledHookDescriptor"]
SelectorCompiler = Callable[..., "CompiledHookDescriptor"]

class DecoratedHookFn(Protocol):
    """Protocol for functions that have been decorated with @hook().
    Only the @hook() decorator produces objects satisfying this protocol.
    All other hook callables (noop, njit, combined, kernel wrappers) are
    plain ``HookCallable``.
    """
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    __name__: str
    meta: Dict[str, Any]
    compiled: Optional[Any]
    event: Any
    selectors: Dict[str, Any]
    priority: int
    custom: bool
    deme_selector: Any
    register: Callable[..., Any]

@njit_switch(cache=True)
def _noop_hook(state: Any, config: Any = None, deme_id: int = -1) -> int:
    """Default hook: (state, config, deme_id) -> 0."""
    return 0

noop_hook = _noop_hook

def _normalize_njit_fn(fn: HookCallable) -> HookCallable:
    """Ensure an njit hook matches ``(state, config, deme_id)``.

    Hooks with 3+ args are passed through unchanged.
    Hooks with exactly 2 args are assumed to omit ``deme_id`` and are
    wrapped to ``(state, config)`` automatically — useful for panmictic
    models where ``deme_id`` is always ``-1``.
    """
    py_fn = getattr(fn, "py_func", fn)
    sig = inspect.signature(py_fn)
    params = list(sig.parameters.values())
    if len(params) >= 3:
        return fn
    # Wrap 2-arg (state, config) — omit deme_id for panmictic.
    @njit_switch(cache=True)
    def wrapped2(state: Any, config: Any = None, deme_id: int = -1) -> object:
        return fn(state, config)
    return wrapped2

def _normalize_py_hook(fn: HookCallable) -> HookCallable:
    """Ensure a Python hook matches ``(state, config, deme_id)``.

    Hooks with 3+ args are passed through unchanged.
    Hooks with exactly 2 args are assumed to omit ``deme_id`` and are
    wrapped to ``(state, config)`` automatically.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) >= 3:
        return fn
    def wrapped2(state: Any, config: Any = None, deme_id: int = -1) -> object:
        return fn(state, config)
    return wrapped2

def compile_combined_hook(
    njit_fns: List[HookCallable],
    deme_selectors: Optional[List[DemeSelector]] = None,
) -> HookCallable:
    """Combine multiple njit hooks into one generated njit function.

    We generate source code instead of composing Python closures so the result
    remains callable from njit engine.

    When ``deme_selectors`` is provided and contains non-wildcard values,
    each hook call is wrapped with an ``if deme_id == X`` guard so that
    per-deme hooks only execute for their target deme(s) — critical for
    spatial simulations where all hooks share one combined function.

    Args:
        njit_fns: List of njit-compiled hook functions.
        deme_selectors: Optional per-function deme target.  When ``None``
            or all ``"*"``, no guards are generated (panmictic-safe).
    """
    if len(njit_fns) == 0:
        return _noop_hook

    # Normalize to list so pyright can track the type (not Optional).
    ds_list: List[DemeSelector] = deme_selectors if deme_selectors is not None else []
    needs_guard = any(ds != "*" for ds in ds_list)

    # Without guards, single-hook combos can return the function directly.
    if not needs_guard and len(njit_fns) == 1:
        return njit_fns[0]

    # Stable key ensures deterministic module names and cache reuse.
    if needs_guard:
        combined_parts = ["combined_guarded"]
        for fn, ds in zip(njit_fns, ds_list):
            combined_parts.append(stable_callable_identity(fn))
            combined_parts.append(str(ds))
    else:
        combined_parts = ["combined"] + [stable_callable_identity(fn) for fn in njit_fns]
    key = hash_key(combined_parts)
    fn_name = f"_combined_hook_{key}"
    module_stem = f"combined_hook_{key}"
    placeholder_names = [f"_FN_{i}" for i in range(len(njit_fns))]

    # Generated module imports the same switch helper as the rest of hook DSL.
    lines = ["from natal.numba_utils import njit_switch"]
    lines.extend([f"{placeholder} = None" for placeholder in placeholder_names])
    lines.extend(
        [
            "",
            "@njit_switch(cache=True)",
            f"def {fn_name}(state, config=None, deme_id=-1):",
        ]
    )

    if needs_guard:
        for placeholder, ds in zip(placeholder_names, ds_list):
            if ds == "*":
                lines.append(f"    _result = {placeholder}(state, config, deme_id)")
                lines.append("    if _result != 0:")
                lines.append("        return _result")
            elif isinstance(ds, int):
                lines.append(f"    if deme_id == {int(ds)}:")
                lines.append(f"        _result = {placeholder}(state, config, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
            elif isinstance(ds, range):
                lines.append(f"    if {ds.start} <= deme_id < {ds.stop}:")
                lines.append(f"        _result = {placeholder}(state, config, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
            else:
                # List or tuple — generate a tuple literal for Numba's ``in``.
                items = ", ".join(str(int(x)) for x in ds)
                lines.append(f"    if deme_id in ({items}):")
                lines.append(f"        _result = {placeholder}(state, config, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
    else:
        for placeholder in placeholder_names:
            lines.append(f"    _result = {placeholder}(state, config, deme_id)")
            lines.append("    if _result != 0:")
            lines.append("        return _result")
    lines.append("    return 0")
    lines.append("")

    module_path = write_codegen_module(module_stem, "\n".join(lines))
    module = load_codegen_module(module_stem, module_path)

    for placeholder, fn in zip(placeholder_names, njit_fns):
        setattr(module, placeholder, fn)

    return getattr(module, fn_name)


# HookProgram field names needed by _execute_single_csr_hook (excludes n_events,
# n_ops_list, and hook_offsets — those are for event-level dispatch, not per-hook).
_CSR_HOOK_ARRAY_NAMES: Tuple[str, ...] = (
    "op_offsets",
    "op_types_data",
    "gidx_offsets_data",
    "gidx_data",
    "age_offsets_data",
    "age_data",
    "sex_masks_data",
    "params_data",
    "condition_offsets_data",
    "condition_types_data",
    "condition_params_data",
    "deme_selector_types",
    "deme_selector_offsets",
    "deme_selector_data",
)

# Module-global placeholder name for each HookProgram array.
_CSR_HOOK_GLOBAL_PREFIX = "_HP_"


def compile_unified_event_hook(
    schedule: List[Tuple[str, int]],
    njit_fns: List[HookCallable],
    deme_selectors: Optional[List[DemeSelector]],
    hook_program: HookProgram,
    has_sperm_storage: bool,
) -> HookCallable:
    """Generate a single njit function that interleaves CSR and njit hooks by priority.

    Follows the same codegen pattern as :func:`compile_combined_hook` but
    alternates between ``_execute_single_csr_hook`` calls (for declarative
    CSR hooks) and njit function calls (for custom/selector hooks).

    Args:
        schedule: Priority-ordered execution plan.  Each entry is
            ``("csr", hook_idx)`` or ``("njit", fn_idx)`` where
            ``hook_idx`` is the global position in the *full* ``HookProgram``
            and ``fn_idx`` indexes into ``njit_fns``.
        njit_fns: Njit-compiled hook functions (indexed by ``fn_idx``).
        deme_selectors: Per-njit-function deme targets.  ``None`` or ``"*"``
            means no guard.  CSR hooks handle deme filtering internally.
        hook_program: Full ``HookProgram`` whose arrays are set as module
            globals so ``_execute_single_csr_hook`` can index into them.
        has_sperm_storage: If True, the generated function reads
            ``state.sperm_storage``.  If False, it creates a dummy
            ``(0,0,0)`` array (discrete-generation case).

    Returns:
        An ``@njit_switch(cache=True)`` function with signature
        ``(state, config=None, deme_id=-1) -> int``.
    """
    if len(schedule) == 0:
        return _noop_hook

    # Single-njit, no guard → return directly (optimisation).
    ds_list: List[DemeSelector] = deme_selectors if deme_selectors is not None else []
    if len(schedule) == 1 and schedule[0][0] == "njit" and len(njit_fns) == 1:
        if not ds_list or ds_list[0] == "*":
            return njit_fns[0]

    # Build stable key for deterministic module naming and cache reuse.
    key_parts: List[str] = ["unified"]
    if has_sperm_storage:
        key_parts.append("sperm")
    else:
        key_parts.append("nosperm")
    for entry_type, idx in schedule:
        if entry_type == "csr":
            key_parts.append(f"csr{idx}")
        else:
            key_parts.append(stable_callable_identity(njit_fns[idx]))
            if ds_list and idx < len(ds_list):
                key_parts.append(str(ds_list[idx]))
    key = hash_key(key_parts)
    fn_name = f"_unified_hook_{key}"
    module_stem = f"unified_hook_{key}"
    n_njit = len(njit_fns)
    placeholder_names = [f"_FN_{i}" for i in range(n_njit)]
    hp_global_names = [f"{_CSR_HOOK_GLOBAL_PREFIX}{name.upper()}" for name in _CSR_HOOK_ARRAY_NAMES]

    # ---- Generate module source ----
    lines: List[str] = []
    lines.append("import numpy as np")
    lines.append("from natal.hooks.executor import _execute_single_csr_hook")
    lines.append("from natal.numba_utils import njit_switch")
    lines.append("")

    # HookProgram array placeholders (set via setattr after loading).
    lines.append("_HP_N_HOOKS = None  # np.int32")
    for gname in hp_global_names:
        lines.append(f"{gname} = None")

    # Njit function placeholders.
    for pname in placeholder_names:
        lines.append(f"{pname} = None")

    lines.append("")
    lines.append("@njit_switch(cache=True)")
    lines.append(f"def {fn_name}(state, config=None, deme_id=-1):")
    lines.append("    ind_count = state.individual_count")
    lines.append("    tick = state.n_tick")
    lines.append("    stochastic = config.stochastic")
    lines.append("    continuous_sampling = config.continuous_sampling")
    if has_sperm_storage:
        lines.append("    sperm_store = state.sperm_storage")
    else:
        lines.append("    sperm_store = np.zeros((0, 0, 0), dtype=np.float64)")
    lines.append("")

    # Generate each schedule entry.
    for entry_type, idx in schedule:
        if entry_type == "csr":
            lines.append(f"    # CSR hook_idx={idx}")
            lines.append("    _r = _execute_single_csr_hook(")
            lines.append(f"        {idx}, _HP_N_HOOKS,")
            # Unpack HookProgram array globals.
            for gname in hp_global_names:
                lines.append(f"        {gname},")
            lines.append(f"        ind_count, sperm_store, {has_sperm_storage},")
            lines.append("        tick, stochastic, continuous_sampling, deme_id,")
            lines.append("    )")
            lines.append("    if _r != 0:")
            lines.append("        return _r")
        else:  # "njit"
            fn_idx = idx
            ds: DemeSelector = ds_list[fn_idx] if fn_idx < len(ds_list) else "*"
            pname = placeholder_names[fn_idx]
            if ds == "*":
                lines.append(f"    _r = {pname}(state, config, deme_id)")
                lines.append("    if _r != 0:")
                lines.append("        return _r")
            elif isinstance(ds, int):
                lines.append(f"    if deme_id == {int(ds)}:")
                lines.append(f"        _r = {pname}(state, config, deme_id)")
                lines.append("        if _r != 0:")
                lines.append("            return _r")
            elif isinstance(ds, range):
                lines.append(f"    if {ds.start} <= deme_id < {ds.stop}:")
                lines.append(f"        _r = {pname}(state, config, deme_id)")
                lines.append("        if _r != 0:")
                lines.append("            return _r")
            else:
                items = ", ".join(str(int(x)) for x in ds)
                lines.append(f"    if deme_id in ({items}):")
                lines.append(f"        _r = {pname}(state, config, deme_id)")
                lines.append("        if _r != 0:")
                lines.append("            return _r")

    lines.append("    return 0")
    lines.append("")

    # ---- Write, load, and wire up globals ----
    module_path = write_codegen_module(module_stem, "\n".join(lines))
    module = load_codegen_module(module_stem, module_path)

    # Inject HookProgram arrays.
    setattr(module, "_HP_N_HOOKS", hook_program.n_hooks)  # noqa: B010
    hp = hook_program
    for name, gname in zip(_CSR_HOOK_ARRAY_NAMES, hp_global_names):
        setattr(module, gname, getattr(hp, name))  # noqa: B010

    # Inject njit functions.
    for i, pname in enumerate(placeholder_names):
        setattr(module, pname, njit_fns[i])  # noqa: B010

    return cast(HookCallable, getattr(module, fn_name))


def _build_filtered_hook_program(
    compiled_hooks: List[CompiledHookDescriptor],
    mixed_events: set[str],
) -> HookProgram:
    """Build a ``HookProgram`` that excludes CSR hooks for ``mixed_events``.

    CSR hooks for mixed events are handled by the unified njit function
    instead, so they must not appear in the HookProgram passed to the
    lifecycle template — otherwise they would be executed twice (once by
    the unified function, once by the template's CSR dispatch).

    The packing logic mirrors ``BasePopulation._build_hook_program``
    (base_population.py:1440-1564) but skips descriptors whose
    ``plan.n_ops > 0`` and ``event`` is in ``mixed_events``.
    """
    import numpy as np

    from .types import EVENT_NAMES, HookProgram

    events = EVENT_NAMES
    n_events = len(events)

    # 1. Collect per-event hooks (all types).
    hook_offsets: List[int] = [0]
    hook_list_by_event: List[List[CompiledHookDescriptor]] = []

    for event_name in events:
        event_hooks = sorted(
            [d for d in compiled_hooks if d.event == event_name],
            key=lambda d: d.priority,
        )
        hook_list_by_event.append(event_hooks)
        hook_offsets.append(hook_offsets[-1] + len(event_hooks))

    n_hooks = hook_offsets[-1]

    # 2. Pack operation data, skipping CSR hooks for mixed events.
    all_op_types: List[int] = []
    all_gidx_offsets: List[int] = [0]
    all_gidx_data: List[int] = []
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

            # Skip CSR hooks for mixed events — they're handled by the unified function.
            if hook.event in mixed_events and plan is not None and plan.n_ops > 0:
                n_ops_list.append(0)
                op_offsets.append(op_offsets[-1])
                continue

            if plan is None or plan.n_ops == 0:
                n_ops_list.append(0)
                op_offsets.append(op_offsets[-1])
                continue

            n_ops_list.append(plan.n_ops)

            # Pack operation data.
            all_op_types.extend(plan.op_types.tolist())

            # Handle gidx (adjust offsets for concatenation).
            gidx_offset_base = len(all_gidx_data)
            for i in range(plan.n_ops):
                all_gidx_offsets.append(
                    gidx_offset_base + plan.gidx_offsets[i + 1] - plan.gidx_offsets[0]
                )
            all_gidx_data.extend(plan.gidx_data.tolist())

            # Handle age.
            age_offset_base = len(all_age_data)
            for i in range(plan.n_ops):
                all_age_offsets.append(
                    age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0]
                )
            all_age_data.extend(plan.age_data.tolist())

            # Handle sex masks (flatten 2D -> 1D).
            all_sex_masks.extend(plan.sex_masks.flatten().tolist())

            # Handle params, conditions.
            all_params.extend(plan.params.tolist())
            cond_offset_base = len(all_cond_types)
            for i in range(plan.n_ops):
                all_cond_offsets.append(
                    cond_offset_base + plan.condition_offsets[i + 1] - plan.condition_offsets[0]
                )
            all_cond_types.extend(plan.condition_types.tolist())
            all_cond_params.extend(plan.condition_params.tolist())

            op_offsets.append(len(all_op_types))

            # Pack deme selector.
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

    # 3. Create filtered HookProgram.
    return HookProgram(
        n_events=np.int32(n_events),
        n_hooks=np.int32(n_hooks),
        hook_offsets=np.array(hook_offsets, dtype=np.int32),
        n_ops_list=np.array(n_ops_list, dtype=np.int32),
        op_offsets=np.array(op_offsets, dtype=np.int32),
        op_types_data=np.array(all_op_types, dtype=np.int32),
        gidx_offsets_data=np.array(all_gidx_offsets, dtype=np.int32),
        gidx_data=np.array(all_gidx_data, dtype=np.int32),
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

def _gen_lifecycle_source(
    is_discrete: bool,
    tick_fn_name: str,
    run_fn_name: str,
) -> str:
    """Generate the source code for a lifecycle wrapper module.

    Reads the template from ``engine/templates/`` and substitutes
    ``TICK_FN_NAME`` and ``RUN_FN_NAME`` placeholders.
    """
    name = "lifecycle_discrete_v2.tmpl.py" if is_discrete else "lifecycle_structured.tmpl.py"
    return (_read_template(name)
        .replace("TICK_FN_NAME", tick_fn_name)
        .replace("RUN_FN_NAME", run_fn_name))


def compile_lifecycle_wrapper(
    is_discrete: bool,
    first_hook: HookCallable,
    early_hook: HookCallable,
    late_hook: HookCallable,
) -> tuple[HookCallable, HookCallable]:
    """Generate a lifecycle wrapper module with hooks as module-level globals.

    This ensures each unique hook combination gets its own Numba
    ``@njit(cache=True)`` function keyed by source-code hash, so compilation
    is cached across process restarts — something Numba cannot do for
    function-valued parameters.

    Args:
        is_discrete: If True, generate discrete-generation (no sperm storage)
            wrappers using ``DiscretePopulationConfig`` and dedicated engine.
            Otherwise generate age-structured wrappers.
        first_hook: Combined njit function for the ``first`` event.
        early_hook: Combined njit function for the ``early`` event.
        late_hook: Combined njit function for the ``late`` event.

    Returns:
        A tuple ``(tick_fn, run_fn)`` where ``tick_fn`` executes one tick
        and ``run_fn`` executes multiple ticks with history recording.
    """
    mode = "discrete" if is_discrete else "structured"
    parts = [f"lifecycle_{mode}"] + [
        stable_callable_identity(fn) for fn in [first_hook, early_hook, late_hook]
    ]
    key = hash_key(parts)
    module_stem = f"lifecycle_{mode}_{key}"
    tick_fn_name = f"_lifecycle_tick_{key}"
    run_fn_name = f"_lifecycle_run_{key}"

    source = _gen_lifecycle_source(is_discrete, tick_fn_name, run_fn_name)
    module_path = write_codegen_module(module_stem, source)
    module = load_codegen_module(module_stem, module_path)

    setattr(module, "_FIRST_HOOK", first_hook)  # noqa: B010
    setattr(module, "_EARLY_HOOK", early_hook)  # noqa: B010
    setattr(module, "_LATE_HOOK", late_hook)  # noqa: B010

    return getattr(module, tick_fn_name), getattr(module, run_fn_name)


def _gen_spatial_lifecycle_source(
    is_discrete: bool,
    tick_fn_name: str,
    run_fn_name: str,
    panmictic_stem: str,
    panmictic_tick_fn_name: str,
) -> str:
    """Generate source for a spatial lifecycle wrapper module.

    Reads the template from ``engine/templates/`` and substitutes
    ``TICK_FN_NAME``, ``RUN_FN_NAME``, ``PANMICTIC_STEM``,
    ``PANMICTIC_TICK_FN_NAME`` placeholders.
    """
    name = "spatial_lifecycle_discrete.tmpl.py" if is_discrete else "spatial_lifecycle_structured.tmpl.py"
    return (_read_template(name)
        .replace("PANMICTIC_TICK_FN_NAME", panmictic_tick_fn_name)
        .replace("PANMICTIC_STEM", panmictic_stem)
        .replace("TICK_FN_NAME", tick_fn_name)
        .replace("RUN_FN_NAME", run_fn_name))


def compile_spatial_lifecycle_wrapper(
    is_discrete: bool,
    first_hook: HookCallable,
    early_hook: HookCallable,
    late_hook: HookCallable,
) -> tuple[HookCallable, HookCallable]:
    """Generate a spatial lifecycle wrapper that delegates per-deme work to the
    panmictic lifecycle tick inside ``prange``.

    The generated module provides two njit-compiled functions:
    - A ``tick`` function (parallel=True) that runs per-deme lifecycle by
      calling the panmictic lifecycle tick inside ``prange``, then migration.
    - A ``run`` function that calls the tick function in a loop with optional
      history recording.

    Hook globals (``_FIRST_HOOK``/``_EARLY_HOOK``/``_LATE_HOOK``) live on the
    panmictic module, not the spatial module.  The spatial module imports and
    calls the panmictic tick, which resolves hooks via its own module globals.

    Args:
        is_discrete: If True, generate discrete-generation per-deme lifecycle.
        first_hook: Combined njit function for the ``first`` event.
        early_hook: Combined njit function for the ``early`` event.
        late_hook: Combined njit function for the ``late`` event.

    Returns:
        A tuple ``(tick_fn, run_fn)`` where tick_fn executes one spatial tick
        and run_fn executes multiple ticks with history recording.
    """
    mode = "discrete" if is_discrete else "structured"
    # Compute the panmictic wrapper identity (same key as compile_lifecycle_wrapper)
    panmictic_parts = [f"lifecycle_{mode}"] + [
        stable_callable_identity(fn) for fn in [first_hook, early_hook, late_hook]
    ]
    panmictic_key = hash_key(panmictic_parts)
    panmictic_stem = f"lifecycle_{mode}_{panmictic_key}"
    panmictic_tick_fn_name = f"_lifecycle_tick_{panmictic_key}"

    # Compute the spatial wrapper identity
    spatial_parts = [f"spatial_lifecycle_{mode}"] + [
        stable_callable_identity(fn) for fn in [first_hook, early_hook, late_hook]
    ]
    spatial_key = hash_key(spatial_parts)
    module_stem = f"spatial_lifecycle_{mode}_{spatial_key}"
    tick_fn_name = f"_spatial_tick_{spatial_key}"
    run_fn_name = f"_spatial_run_{spatial_key}"

    source = _gen_spatial_lifecycle_source(
        is_discrete, tick_fn_name, run_fn_name,
        panmictic_stem, panmictic_tick_fn_name,
    )
    module_path = write_codegen_module(module_stem, source)
    module = load_codegen_module(module_stem, module_path)

    # No need to set _FIRST_HOOK/ _EARLY_HOOK/ _LATE_HOOK — those live on
    # the panmictic module and are already set by compile_lifecycle_wrapper.

    return getattr(module, tick_fn_name), getattr(module, run_fn_name)


class CompiledEventHooks:
    """Container for event-wise combined hook callables.

    Kernel code expects one callable per event name. This class stores those
    callables and optionally the declarative ``HookProgram`` registry.
    When hooks are present and Numba is enabled, lifecycle wrappers are
    pre-compiled with hooks as globals so Numba caching survives restarts.
    """
    __slots__ = (
        "first",
        "early",
        "late",
        "finish",
        "first_discrete",
        "early_discrete",
        "late_discrete",
        "finish_discrete",
        "registry",
        "run_tick_fn",
        "run_fn",
        "run_discrete_tick_fn",
        "run_discrete_fn",
        "spatial_tick_fn",
        "spatial_run_fn",
        "spatial_discrete_tick_fn",
        "spatial_discrete_run_fn",
        "_event_hooks",
    )
    # Type annotations for attributes
    first: HookCallable
    early: HookCallable
    late: HookCallable
    finish: HookCallable
    first_discrete: HookCallable
    early_discrete: HookCallable
    late_discrete: HookCallable
    finish_discrete: HookCallable
    registry: Optional[HookProgram]
    run_tick_fn: Optional[HookCallable]
    run_fn: Optional[HookCallable]
    run_discrete_tick_fn: Optional[HookCallable]
    run_discrete_fn: Optional[HookCallable]
    spatial_tick_fn: Optional[HookCallable]
    spatial_run_fn: Optional[HookCallable]
    spatial_discrete_tick_fn: Optional[HookCallable]
    spatial_discrete_run_fn: Optional[HookCallable]
    _event_hooks: Dict[str, HookCallable]

    def __init__(self) -> None:
        self.first = _noop_hook
        self.early = _noop_hook
        self.late = _noop_hook
        self.finish = _noop_hook
        self.first_discrete = _noop_hook
        self.early_discrete = _noop_hook
        self.late_discrete = _noop_hook
        self.finish_discrete = _noop_hook
        self.registry = None
        self.run_tick_fn = None
        self.run_fn = None
        self.run_discrete_tick_fn = None
        self.run_discrete_fn = None
        self.spatial_tick_fn = None
        self.spatial_run_fn = None
        self.spatial_discrete_tick_fn = None
        self.spatial_discrete_run_fn = None
        self._event_hooks = dict.fromkeys(EVENT_NAMES, _noop_hook)

    def get_hook(self, event_name: str) -> HookCallable:
        return self._event_hooks.get(event_name, _noop_hook)

    def set_hook(self, event_name: str, hook_fn: HookCallable) -> None:
        self._event_hooks[event_name] = hook_fn
        setattr(self, event_name, hook_fn)

    def set_discrete_hook(self, event_name: str, hook_fn: HookCallable) -> None:
        """Set the without-sperm-storage variant for ``event_name``."""
        setattr(self, event_name + "_discrete", hook_fn)

    @staticmethod
    def from_compiled_hooks(
        compiled_hooks: List[CompiledHookDescriptor],
        registry: Optional[HookProgram] = None,
        include_spatial_wrappers: bool = False,
    ) -> CompiledEventHooks:
        """Build event-wise combined callables and lifecycle wrappers.

        When an event mixes CSR (declarative) and njit (custom) hooks,
        this method generates a *unified* njit function that interleaves
        both types by priority instead of falling back to the Python
        ``HookExecutor``.  The ``registry`` is filtered so the template's
        CSR dispatch becomes a no-op for mixed events.

        Unlike the previous Jinja2-codegen approach, this method generates
        only the necessary lifecycle wrapper per hook combination using
        ``compile_lifecycle_wrapper``, which produces a uniquely-named njit
        function with hooks as globals. This ensures Numba ``cache=True``
        works across process restarts.
        """
        from ..numba_utils import NUMBA_ENABLED

        if NUMBA_ENABLED:
            for desc in compiled_hooks:
                if desc.py_wrapper is not None:
                    raise TypeError(
                        f"Hook '{desc.name}' uses py_wrapper, which is not allowed when Numba is enabled."
                    )

        result = CompiledEventHooks()
        result.registry = registry

        # ---- Detect mixed events ----
        mixed_events: set[str] = set()
        for event_name in EVENT_NAMES:
            event_descs = [d for d in compiled_hooks if d.event == event_name]
            has_csr = any(d.plan is not None and d.plan.n_ops > 0 for d in event_descs)
            has_njit = any(d.njit_fn is not None for d in event_descs)
            if has_csr and has_njit:
                mixed_events.add(event_name)

        # ---- Build filtered registry for mixed events ----
        if registry is not None and mixed_events:
            result.registry = _build_filtered_hook_program(compiled_hooks, mixed_events)

        # ---- Compute HookProgram global indices for CSR hooks ----
        # The HookProgram packing order is: EVENT_NAMES, then priority.
        # We need to know each CSR descriptor's hook_idx so the unified
        # function can pass it to _execute_single_csr_hook.
        hook_idx_map: Dict[int, int] = {}  # id(desc) -> global_hook_idx
        if mixed_events:
            idx = 0
            for event_name in EVENT_NAMES:
                event_hooks = sorted(
                    [d for d in compiled_hooks if d.event == event_name],
                    key=lambda d: d.priority,
                )
                for desc in event_hooks:
                    hook_idx_map[id(desc)] = idx
                    idx += 1

        # ---- Collect njit hooks per event (same as before) ----
        hooks_by_event: Dict[str, List[Tuple[int, HookCallable, DemeSelector]]] = {
            name: [] for name in EVENT_NAMES
        }
        for desc in compiled_hooks:
            if desc.njit_fn is not None and desc.event in hooks_by_event:
                hooks_by_event[desc.event].append(
                    (desc.priority, desc.njit_fn, desc.deme_selector)
                )

        for event_name, hook_list in hooks_by_event.items():
            if not hook_list:
                # No njit hooks for this event.
                # If mixed, we still need a unified function for the CSR hooks.
                if event_name in mixed_events:
                    event_descs = sorted(
                        [d for d in compiled_hooks if d.event == event_name],
                        key=lambda d: d.priority,
                    )
                    csr_schedule: List[Tuple[str, int]] = [
                        ("csr", hook_idx_map[id(d)])
                        for d in event_descs
                        if d.plan is not None and d.plan.n_ops > 0
                    ]
                    if csr_schedule and registry is not None:
                        unified_with = compile_unified_event_hook(
                            csr_schedule, [], None, registry, has_sperm_storage=True,
                        )
                        unified_without = compile_unified_event_hook(
                            csr_schedule, [], None, registry, has_sperm_storage=False,
                        )
                        result.set_hook(event_name, unified_with)
                        result.set_discrete_hook(event_name, unified_without)
                continue

            hook_list.sort(key=lambda x: x[0])
            njit_fns = [fn for _, fn, _ in hook_list]
            deme_selectors = cast("List[DemeSelector]", [ds for _, _, ds in hook_list])

            if event_name in mixed_events and registry is not None:
                # ---- Build unified function for mixed event ----
                event_descs = sorted(
                    [d for d in compiled_hooks if d.event == event_name],
                    key=lambda d: d.priority,
                )
                schedule: List[Tuple[str, int]] = []
                # Collect njit entries from the already-sorted hook_list.
                # hook_list is sorted by priority; we merge CSR entries into the
                # same priority order.
                njit_idx = 0
                for desc in event_descs:
                    if desc.njit_fn is not None:
                        schedule.append(("njit", njit_idx))
                        njit_idx += 1
                    elif desc.plan is not None and desc.plan.n_ops > 0:
                        schedule.append(("csr", hook_idx_map[id(desc)]))

                unified_with = compile_unified_event_hook(
                    schedule, njit_fns, deme_selectors, registry,
                    has_sperm_storage=True,
                )
                unified_without = compile_unified_event_hook(
                    schedule, njit_fns, deme_selectors, registry,
                    has_sperm_storage=False,
                )
                result.set_hook(event_name, unified_with)
                result.set_discrete_hook(event_name, unified_without)
            else:
                # ---- Non-mixed event: existing behaviour ----
                combined = compile_combined_hook(njit_fns, deme_selectors)
                result.set_hook(event_name, combined)

        # ---- Pre-compile lifecycle wrappers ----
        first_hook = result.first
        early_hook = result.early
        late_hook = result.late

        # For discrete lifecycle, use without-sperm hooks for mixed events,
        # otherwise fall back to the standard with-sperm hooks.
        # (_noop_hook is a module-level singleton, so ``is`` works here.)
        first_d = (
            result.first_discrete
            if result.first_discrete is not _noop_hook
            else first_hook
        )
        early_d = (
            result.early_discrete
            if result.early_discrete is not _noop_hook
            else early_hook
        )
        late_d = (
            result.late_discrete
            if result.late_discrete is not _noop_hook
            else late_hook
        )

        # Always compile lifecycle wrappers when Numba is enabled so the
        # population model can use them unconditionally.  Even with zero
        # user hooks the wrapper compiles with _noop_hook globals, and its
        # source hash stays stable across runs.
        if NUMBA_ENABLED:
            result.run_tick_fn, result.run_fn = compile_lifecycle_wrapper(
                False, first_hook, early_hook, late_hook,
            )
            result.run_discrete_tick_fn, result.run_discrete_fn = compile_lifecycle_wrapper(
                True, first_d, early_d, late_d,
            )

            if include_spatial_wrappers:
                result.spatial_tick_fn, result.spatial_run_fn = compile_spatial_lifecycle_wrapper(
                    False, first_hook, early_hook, late_hook,
                )
                result.spatial_discrete_tick_fn, result.spatial_discrete_run_fn = compile_spatial_lifecycle_wrapper(
                    True, first_d, early_d, late_d,
                )

        return result


def _has_required_parameters(func: HookCallable) -> bool:
    """Return whether calling ``func()`` would require positional/keyword args."""
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if param.default is inspect.Signature.empty:
                return True
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Signature.empty:
                return True
    return False


def _is_declarative_population_hook(func: HookCallable) -> bool:
    """Return whether func accepts a single population parameter (declarative Python hook)."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) == 1:
        param = params[0]
        if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and
                param.default is inspect.Signature.empty):
            # Single required parameter - likely a population hook, not a custom ind_count hook
            return True
    return False


def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    custom: bool = False,
    deme: DemeSelector = "*",
    mode: str = "auto",
) -> Callable[[Callable[..., Any]], DecoratedHookFn]:
    """Decorator entrypoint for all supported hook authoring styles.

    The decorated function gets a ``register(pop, event_override=None)``
    helper that compiles and registers a ``CompiledHookDescriptor``.

    Hook type is determined by:
    - selectors specified -> Selector hook
    - custom=True or has required params -> Custom hook
    - otherwise -> Declarative hook (function returns List[HookOp])

    For custom/selector hooks, Numba compilation is automatic — you do
    **not** need to stack ``@njit``.  If Numba is enabled, the function is
    wrapped with ``njit_switch`` automatically.  If Numba is disabled, a
    pure-Python wrapper is used.

    When a custom hook is called inside a spatial ``prange`` region, the
    ``deme_id`` parameter receives the current deme index, enabling one
    hook function to handle all demes with per-deme branching logic.

    Args:
        event: Hook event name.
        selectors: Optional symbolic selectors for selector-mode hooks.
        priority: Execution priority (lower values run earlier).
        custom: If True, treat as custom hook (function is called directly).
        deme: Target deme(s) for spatial populations.  ``"*"`` (default)
            means all demes.  Accepts a single int, list, tuple, or range.
        mode: Selector passing style.  ``"auto"`` (default) detects from
            function signature.  ``"expand"`` passes each selector as a
            separate keyword argument.  ``"aggregate"`` packs all selectors
            into a single namedtuple argument.
    """
    if mode not in ("auto", "expand", "aggregate"):
        raise ValueError(f"mode must be 'auto', 'expand', or 'aggregate', got {mode!r}")

    def decorator(func: Callable[..., Any]) -> DecoratedHookFn:
        hook_func = cast(DecoratedHookFn, func)
        hook_func.meta = {
            "event": event,
            "selectors": selectors or {},
            "priority": priority,
            "custom": custom,
            "deme_selector": deme,
            "mode": mode,
        }
        hook_func.compiled = None
        hook_func.event = event
        hook_func.selectors = selectors or {}
        hook_func.priority = priority
        hook_func.custom = custom
        hook_func.deme_selector = deme

        def register(
            pop: BasePopulation[Any],
            event_override: Optional[str] = None,
            deme_selector_override: Optional[DemeSelector] = None,
        ) -> CompiledHookDescriptor:
            """Compile this hook against one population instance."""
            from ..numba_utils import NUMBA_ENABLED
            from .types import CompiledHookDescriptor

            actual_event = event_override or event
            actual_deme_selector: DemeSelector = deme if deme_selector_override is None else deme_selector_override
            if actual_event is None:
                raise ValueError(
                    f"Event not specified for hook '{func.__name__}'. "
                    "Specify in decorator @hook(event='...') or call pop.set_hook('event', hook)"
                )

            has_required_params = _has_required_parameters(func)
            is_declarative_pop_hook = _is_declarative_population_hook(func)
            is_custom_or_selector = custom or selectors is not None or (has_required_params and not is_declarative_pop_hook)

            if is_custom_or_selector:
                if selectors is not None:
                    desc = compile_selector_hook(
                        func,
                        pop,
                        actual_event,
                        selectors,
                        priority,
                        deme_selector=actual_deme_selector,
                        mode=mode,
                    )
                else:
                    if is_njit_function(func):
                        # Already njit-decorated
                        norm_fn = func
                        desc = CompiledHookDescriptor(
                            name=func.__name__,
                            event=actual_event,
                            priority=priority,
                            deme_selector=actual_deme_selector,
                            njit_fn=norm_fn,
                            meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                        )
                    else:
                        # Try to use njit_switch
                        try:
                            decorated_func = njit_switch(cache=False)(func)
                            # Check if it's a valid compiled function
                            if NUMBA_ENABLED and is_njit_function(decorated_func):
                                norm_fn = _normalize_njit_fn(decorated_func)
                                desc = CompiledHookDescriptor(
                                    name=func.__name__,
                                    event=actual_event,
                                    priority=priority,
                                    deme_selector=actual_deme_selector,
                                    njit_fn=norm_fn,
                                    meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                                )
                            else:
                                # NUMBA_ENABLED is False, use py wrapper
                                wrapped_func = _normalize_py_hook(func)
                                desc = CompiledHookDescriptor(
                                    name=func.__name__,
                                    event=actual_event,
                                    priority=priority,
                                    deme_selector=actual_deme_selector,
                                    njit_fn=None,
                                    py_wrapper=wrapped_func,
                                    meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                                )
                        except Exception:
                            # Fall back to py wrapper
                            wrapped_func = _normalize_py_hook(func)
                            desc = CompiledHookDescriptor(
                                name=func.__name__,
                                event=actual_event,
                                priority=priority,
                                deme_selector=actual_deme_selector,
                                njit_fn=None,
                                py_wrapper=wrapped_func,
                                meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                            )
            elif is_declarative_pop_hook:
                # Single population parameter - use as py_wrapper, but check numba enabled
                if NUMBA_ENABLED:
                    raise TypeError(
                        f"Python hook '{func.__name__}' is not allowed when Numba is enabled. "
                        "Please convert it to @njit or use declarative Op hooks."
                    )
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    deme_selector=actual_deme_selector,
                    py_wrapper=func,
                    meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                )
            else:
                result = func()
                if isinstance(result, list):
                    result_ops = cast(List[object], result)
                    if not all(isinstance(op, HookOp) for op in result_ops):
                        raise TypeError(
                            f"Declarative hook '{func.__name__}' must return List[HookOp], "
                            "or use custom=True for custom mode."
                        )
                    ops = cast(List[HookOp], result_ops)
                    desc = compile_declarative_hook(
                        ops,
                        pop,
                        actual_event,
                        priority,
                        deme_selector=actual_deme_selector,
                        name=func.__name__,
                    )
                else:
                    raise TypeError(
                        f"Hook '{func.__name__}' must return List[HookOp] for declarative mode, "
                        "or use custom=True for custom mode."
                    )

            hook_func.compiled = desc  # type: ignore
            pop.register_compiled_hook(desc)
            return desc

        hook_func.register = register  # type: ignore
        return hook_func

    return decorator
