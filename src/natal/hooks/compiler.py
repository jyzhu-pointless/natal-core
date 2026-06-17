"""Hook compilation and codegen entrypoints.

This module is the **front door** of the hook system.  It connects three
authoring styles into one runtime contract:

1. **Declarative hooks** — functions returning ``List[HookOp]``, compiled
   into CSR plans and stored in the ``HookProgram``.
2. **Selector hooks** — functions with symbolic ``selectors=``, compiled via
   ``compile_selector_hook`` into njit wrappers or Python fallbacks.
3. **Custom hooks** — user-provided ``@njit`` or Python callbacks, wrapped
   with ``_normalize_njit_fn`` / ``_normalize_py_hook``.

The ``hook()`` decorator auto-detects which style a function uses.
``CompiledEventHooks.from_compiled_hooks`` is the central integration point:
it groups compiled descriptors by event, detects mixed CSR+njit scenarios,
and generates lifecycle wrappers (or unified dispatch functions for mixed
cases) that the population models use directly.

----
Codegen approach
----
Rather than composing Python closures (which Numba cannot cache across
process restarts), this module **generates Python source code** for each
unique hook combination.  The source is written to ``.numba_cache/``,
loaded as a module, and the resulting ``@njit_switch(cache=True)``
functions are stable and cacheable.  Module globals (hook functions,
HookProgram arrays) are injected via ``setattr`` after loading.

Template files live in two directories:

- ``engine/templates/`` — lifecycle wrapper templates
- ``hooks/templates/`` — unified mixed-type dispatch templates
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

_HOOK_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_hook_template(name: str) -> str:
    """Read a hook codegen template from ``hooks/templates/``."""
    return (_HOOK_TEMPLATE_DIR / name).read_text(encoding="utf-8")


if TYPE_CHECKING:
    from natal.base_population import BasePopulation

    from .types import CompiledHookDescriptor

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Any callable that can serve as a hook body (noop, njit, combined, kernel).
HookCallable = Callable[..., Any]
DeclarativeCompiler = Callable[..., "CompiledHookDescriptor"]
SelectorCompiler = Callable[..., "CompiledHookDescriptor"]


class DecoratedHookFn(Protocol):
    """Protocol for functions that have been decorated with ``@hook()``.

    Only the ``@hook()`` decorator produces objects satisfying this
    protocol.  All other hook callables (noop, njit, combined, kernel
    wrappers) are plain ``HookCallable``.
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


# ---------------------------------------------------------------------------
# No-op and signature normalisation
# ---------------------------------------------------------------------------


@njit_switch(cache=True)
def _noop_hook(state: Any, config: Any = None, deme_id: int = -1) -> int:
    """Default no-op hook: ``(state, config, deme_id) -> 0``.

    Used as the fallback when no hooks are registered for an event.
    """
    return 0


noop_hook = _noop_hook


def _normalize_njit_fn(fn: HookCallable) -> HookCallable:
    """Ensure an njit hook callable accepts ``(state, config, deme_id)``.

    The unified calling convention for all compiled hooks is three
    positional arguments.  This adapter handles two cases:

    * **3+ args** — passed through unchanged.
    * **2 args** — assumed to omit ``deme_id`` (panmictic-only hook).
      Wrapped with a thunk that drops the third argument.

    Returns:
        A callable with signature ``(state, config, deme_id) -> int``.
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
    """Ensure a Python hook callable accepts ``(state, config, deme_id)``.

    Python equivalent of ``_normalize_njit_fn``.  Used only when Numba
    is disabled.

    Returns:
        A callable with signature ``(state, config, deme_id) -> int``.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) >= 3:
        return fn

    def wrapped2(state: Any, config: Any = None, deme_id: int = -1) -> object:
        return fn(state, config)

    return wrapped2


# ===================================================================
# Combined hook codegen (njit-only, no CSR interleaving)
# ===================================================================


def compile_combined_hook(
    njit_fns: List[HookCallable],
    deme_selectors: Optional[List[DemeSelector]] = None,
) -> HookCallable:
    """Combine multiple njit hooks into a single generated njit function.

    Generates Python source for a new module containing a function that
    calls each hook in sequence.  Each call is wrapped with an optional
    ``if deme_id == X`` guard for spatial simulations.

    The source is written to ``.numba_cache/`` so Numba's ``cache=True``
    survives process restarts — this is not possible with runtime-composed
    closures.

    Args:
        njit_fns: Njit-compiled hook functions to call in order.
        deme_selectors: Per-function deme target.  ``None`` or ``"*"``
            (default) means no guard.  ``int``, ``list``, ``range``
            produce a guard for that specific deme or set.

    Returns:
        An ``@njit_switch(cache=True)`` function with signature
        ``(state, config=None, deme_id=-1) -> int``.  Returns the
        first non-zero result (``RESULT_STOP``), or 0.
    """
    if len(njit_fns) == 0:
        return _noop_hook

    # Normalize to list so pyright can track the type (not Optional).
    ds_list: List[DemeSelector] = deme_selectors if deme_selectors is not None else []
    needs_guard = any(ds != "*" for ds in ds_list)

    # Single-hook, no guard — return the function directly (optimisation).
    if not needs_guard and len(njit_fns) == 1:
        return njit_fns[0]

    # ---- Build stable cache key ----
    if needs_guard:
        combined_parts = ["combined_guarded"]
        for fn, ds in zip(njit_fns, ds_list):
            combined_parts.append(stable_callable_identity(fn))
            combined_parts.append(str(ds))
    else:
        combined_parts = ["combined"] + [
            stable_callable_identity(fn) for fn in njit_fns
        ]
    key = hash_key(combined_parts)
    fn_name = f"_combined_hook_{key}"
    module_stem = f"combined_hook_{key}"
    placeholder_names = [f"_FN_{i}" for i in range(len(njit_fns))]

    # ---- Generate module source ----
    # Each placeholder ``_FN_i = None`` is overridden via setattr after load.
    # The generated function calls each in sequence; the first non-zero
    # return propagates upward as RESULT_STOP.
    lines: list[str] = ["from natal.numba_utils import njit_switch"]
    lines.extend(f"{p} = None" for p in placeholder_names)
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

    # ---- Write, load, wire up globals ----
    # Module is written to .numba_cache/ so Numba's cache=True survives
    # restarts.  setattr overrides each ``_FN_i = None`` with the real
    # njit function before returning the generated callable.
    module_path = write_codegen_module(module_stem, "\n".join(lines))
    module = load_codegen_module(module_stem, module_path)

    for placeholder, fn in zip(placeholder_names, njit_fns):
        setattr(module, placeholder, fn)

    return getattr(module, fn_name)


# ===================================================================
# Unified mixed-type dispatch codegen (CSR + njit interleaved)
# ===================================================================

# HookProgram field names required by _execute_single_csr_hook.
# Excludes n_events, n_ops_list, and hook_offsets — those are only
# needed for event-level dispatch, not per-hook execution.
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

# Prefix for HookProgram array globals in generated modules.
_CSR_HOOK_GLOBAL_PREFIX = "_HP_"


def compile_unified_event_hook(
    schedule: List[Tuple[str, int]],
    njit_fns: List[HookCallable],
    deme_selectors: Optional[List[DemeSelector]],
    hook_program: HookProgram,
    has_sperm_storage: bool,
) -> HookCallable:
    """Generate a single njit function that interleaves CSR and njit hooks by priority.

    This is the codegen counterpart of ``compile_combined_hook`` for the
    case where a single event mixes CSR (declarative ``Op.*``) and njit
    (custom ``@hook``) hooks.  Instead of generating a simple sequential
    njit-call chain, it generates a function that alternates between
    ``_execute_single_csr_hook`` calls and njit function calls according
    to a priority-ordered *schedule*.

    The generated module imports ``_execute_single_csr_hook`` from
    ``executor`` and stores the ``HookProgram`` arrays as module globals
    so they are available at call sites without bloating the argument list.

    Template files used:
    - ``hooks/templates/unified_hook.tmpl.py`` — main module skeleton
    - ``hooks/templates/unified_hook_csr_entry.tmpl.py`` — per-CSR-entry fragment
    - ``hooks/templates/unified_hook_njit_entry.tmpl.py`` — per-njit-entry fragment

    Args:
        schedule: Priority-ordered execution plan.  Each entry is
            ``("csr", hook_idx)`` or ``("njit", fn_idx)`` where
            ``hook_idx`` is the global position in the *full*
            ``HookProgram`` and ``fn_idx`` indexes into ``njit_fns``.
        njit_fns: Njit-compiled hook functions, indexed by ``fn_idx``.
        deme_selectors: Per-njit-function deme targets.  ``None`` or
            ``"*"`` means no guard.  CSR hooks handle deme filtering
            inside ``_execute_single_csr_hook``.
        hook_program: The original ``HookProgram`` whose arrays are set
            as module globals so ``_execute_single_csr_hook`` can index
            into them by ``hook_idx``.
        has_sperm_storage: If ``True``, the generated function reads
            ``state.sperm_storage``.  If ``False``, it creates a dummy
            ``(0,0,0)`` array (discrete-generation case).

    Returns:
        A tuple ``(with_sperm_fn, without_sperm_fn)`` where each is an
        ``@njit_switch(cache=True)`` function with signature
        ``(state, config=None, deme_id=-1) -> int``.  The *with-sperm*
        variant accesses ``state.sperm_storage``; the *without-sperm*
        variant uses a dummy array.
    """
    if len(schedule) == 0:
        return _noop_hook

    # Single-njit, no guard → return directly (optimisation).
    ds_list: List[DemeSelector] = deme_selectors if deme_selectors is not None else []
    if len(schedule) == 1 and schedule[0][0] == "njit" and len(njit_fns) == 1:
        if not ds_list or ds_list[0] == "*":
            return njit_fns[0]

    # ---- Build stable cache key ----
    # The key must uniquely identify this hook combination.  It includes:
    #   - "unified" prefix (distinguishes from combined_hook_* modules)
    #   - sperm flag (with/without — different type signatures)
    #   - every schedule entry (CSR idx or njit identity + deme selector)
    # A SHA-256 of these parts produces the hex suffix in the module name.
    # If any hook changes, the key changes → new module → no cache collision.
    key_parts: List[str] = ["unified"]
    key_parts.append("sperm" if has_sperm_storage else "nosperm")
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
    hp_global_names = [
        f"{_CSR_HOOK_GLOBAL_PREFIX}{name.upper()}" for name in _CSR_HOOK_ARRAY_NAMES
    ]

    # Sperm setup string — injected into the template's
    # ``"PLACEHOLDER_SPERM_SETUP"`` string literal.
    sperm_setup = (
        "state.sperm_storage"
        if has_sperm_storage
        else "np.zeros((0, 0, 0), dtype=np.float64)"
    )

    # ---- Build schedule body from entry templates ----
    # Each schedule entry is assembled from a fragment template (csr_entry
    # or njit_entry).  The fragment already carries its own 4-space indent;
    # the main template's ``# PLACEHOLDER_SCHEDULE_BODY`` line is at column 0
    # so no double-indent occurs.
    csr_tmpl = _read_hook_template("unified_hook_csr_entry.tmpl.py")
    njit_tmpl = _read_hook_template("unified_hook_njit_entry.tmpl.py")
    has_sperm_str = str(has_sperm_storage)

    schedule_entries: List[str] = []
    for entry_type, idx in schedule:
        if entry_type == "csr":
            schedule_entries.append(
                csr_tmpl.replace("PLACEHOLDER_SCHEDULE_IDX", str(idx)).replace(
                    "PLACEHOLDER_HAS_SPERM", has_sperm_str
                )
            )
        else:  # "njit"
            fn_idx = idx
            ds: DemeSelector = ds_list[fn_idx] if fn_idx < len(ds_list) else "*"
            if ds == "*":
                guard_cond = "True"
            elif isinstance(ds, int):
                guard_cond = f"deme_id == {int(ds)}"
            elif isinstance(ds, range):
                guard_cond = f"{ds.start} <= deme_id < {ds.stop}"
            else:
                items = ", ".join(str(int(x)) for x in ds)
                guard_cond = f"deme_id in ({items})"
            schedule_entries.append(
                njit_tmpl.replace(
                    "PLACEHOLDER_NJIT_FN_NAME", placeholder_names[fn_idx]
                ).replace("PLACEHOLDER_NJIT_GUARD_CONDITION", guard_cond)
            )

    schedule_body_str = "\n".join(schedule_entries)

    # ---- Assemble module from template ----
    # Three string replacements turn the template skeleton into a complete
    # module: function name, sperm setup line, and the schedule body.
    # ``"PLACEHOLDER_SPERM_SETUP"`` includes the quotes so the replacement
    # text is injected as an expression (not a string literal).
    template = _read_hook_template("unified_hook.tmpl.py")
    source = (
        template.replace("_unified_hook_TEMPLATE", fn_name)
        .replace('"PLACEHOLDER_SPERM_SETUP"', sperm_setup)
        .replace("# PLACEHOLDER_SCHEDULE_BODY", schedule_body_str)
    )

    # ---- Write, load, wire up globals ----
    module_path = write_codegen_module(module_stem, source)
    module = load_codegen_module(module_stem, module_path)

    # Inject HookProgram arrays so _execute_single_csr_hook can index them.
    setattr(module, "_HP_N_HOOKS", hook_program.n_hooks)  # noqa: B010
    hp = hook_program
    for name, gname in zip(_CSR_HOOK_ARRAY_NAMES, hp_global_names):
        setattr(module, gname, getattr(hp, name))  # noqa: B010

    # Inject njit functions.
    for i, pname in enumerate(placeholder_names):
        setattr(module, pname, njit_fns[i])  # noqa: B010

    return cast(HookCallable, getattr(module, fn_name))


def build_filtered_hook_program(
    compiled_hooks: List[CompiledHookDescriptor],
    mixed_events: set[str],
) -> HookProgram:
    """Build a ``HookProgram`` that excludes CSR hooks for ``mixed_events``.

    CSR hooks that belong to a mixed event are handled by the unified
    njit function instead.  If they remained in the registry passed to
    the lifecycle template, they would be executed **twice** — once by
    the unified function and once by the template's ``execute_csr_*``
    call.

    This function rebuilds the entire ``HookProgram`` from scratch,
    skipping CSR descriptors whose event is in ``mixed_events``.
    The packing logic mirrors ``BasePopulation._build_hook_program``
    (base_population.py) but with the filter applied.

    **Important**: every hook (including no-ops and skipped CSR hooks)
    must append a deme selector entry so that ``deme_selector_types``
    stays aligned with ``n_hooks``.  Without this, the template's CSR
    dispatch would index out of bounds for non-mixed events whose
    hooks appear after the removed entries.

    Args:
        compiled_hooks: All compiled descriptors (CSR + njit + py).
        mixed_events: Events that have both CSR and njit hooks.

    Returns:
        A new ``HookProgram`` with CSR ops removed for mixed events.
    """
    import numpy as np

    from .types import EVENT_NAMES, HookProgram

    events = EVENT_NAMES
    n_events = len(events)

    # 1. Collect per-event hooks, maintaining EVENT_NAMES order and
    #    priority sort — the same layout as the original HookProgram.
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

    # 2. Pack operation data, skipping CSR hooks that belong to mixed events.
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
            sel = hook.deme_selector

            # Inline helper: append a deme selector entry for *s*.
            # Called for every hook to keep arrays aligned with n_hooks.
            def _append_deme_sel(s: DemeSelector) -> None:
                if s == "*":
                    all_deme_sel_types.append(0)
                elif isinstance(s, int):
                    all_deme_sel_types.append(1)
                    all_deme_sel_data.append(int(s))
                elif isinstance(s, range):
                    all_deme_sel_types.append(2)
                    all_deme_sel_data.append(int(s.start))
                    all_deme_sel_data.append(int(s.stop))
                else:
                    all_deme_sel_types.append(3)
                    all_deme_sel_data.extend([int(x) for x in s])
                all_deme_sel_offsets.append(len(all_deme_sel_data))

            # Skip CSR hooks for mixed events — the unified function
            # handles them.
            if hook.event in mixed_events and plan is not None and plan.n_ops > 0:
                n_ops_list.append(0)
                op_offsets.append(op_offsets[-1])
                _append_deme_sel(sel)
                continue

            # No-op hooks (njit/py without a CSR plan).
            if plan is None or plan.n_ops == 0:
                n_ops_list.append(0)
                op_offsets.append(op_offsets[-1])
                _append_deme_sel(sel)
                continue

            # ---- Pack a real CSR plan ----
            n_ops_list.append(plan.n_ops)

            all_op_types.extend(plan.op_types.tolist())

            # Genotype indices (adjust offsets for concatenation).
            gidx_offset_base = len(all_gidx_data)
            for i in range(plan.n_ops):
                all_gidx_offsets.append(
                    gidx_offset_base
                    + plan.gidx_offsets[i + 1]
                    - plan.gidx_offsets[0]
                )
            all_gidx_data.extend(plan.gidx_data.tolist())

            # Age indices.
            age_offset_base = len(all_age_data)
            for i in range(plan.n_ops):
                all_age_offsets.append(
                    age_offset_base
                    + plan.age_offsets[i + 1]
                    - plan.age_offsets[0]
                )
            all_age_data.extend(plan.age_data.tolist())

            # Sex masks (flatten 2D → 1D).
            all_sex_masks.extend(plan.sex_masks.flatten().tolist())

            # Params and condition tokens.
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

            op_offsets.append(len(all_op_types))
            _append_deme_sel(sel)

    # 3. Build the filtered HookProgram.
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


# ===================================================================
# CompiledEventHooks — pure hook container
# ===================================================================


class CompiledEventHooks:
    """Per-event hook container used by lifecycle wrappers.

    Holds one combined callable per event (``first`` / ``early`` /
    ``late`` / ``finish``) plus a ``registry`` (``HookProgram``) for
    CSR dispatch.  This is a **pure container** — code generation and
    lifecycle wrapper compilation live in
    ``natal.engine.lifecycle_wrappers``.

    Slots
    -----
    ``first`` / ``early`` / ``late`` / ``finish``
        Combined hook callables for each event.
    ``registry``
        ``HookProgram`` — may be filtered for mixed events.
    """

    __slots__ = (
        "first",
        "early",
        "late",
        "finish",
        "registry",
        "_event_hooks",
    )

    first: HookCallable
    early: HookCallable
    late: HookCallable
    finish: HookCallable
    registry: Optional[HookProgram]
    _event_hooks: Dict[str, HookCallable]

    def __init__(self) -> None:
        self.first = _noop_hook
        self.early = _noop_hook
        self.late = _noop_hook
        self.finish = _noop_hook
        self.registry = None
        self._event_hooks = dict.fromkeys(EVENT_NAMES, _noop_hook)

    def get_hook(self, event_name: str) -> HookCallable:
        """Return the combined callable for *event_name*."""
        return self._event_hooks.get(event_name, _noop_hook)

    def set_hook(self, event_name: str, hook_fn: HookCallable) -> None:
        """Set the combined callable for *event_name* (both dict and attr)."""
        self._event_hooks[event_name] = hook_fn
        setattr(self, event_name, hook_fn)


# ===================================================================
# Hook type detection helpers
# ===================================================================


def _has_required_parameters(func: HookCallable) -> bool:
    """Return ``True`` if *func* requires positional or keyword arguments."""
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Signature.empty:
                return True
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Signature.empty:
                return True
    return False


def _is_declarative_population_hook(func: HookCallable) -> bool:
    """Return ``True`` if *func* accepts a single required parameter.

    This is a heuristic: single-parameter functions are treated as
    "declarative population hooks" (legacy style) rather than custom
    hooks.  Functions with zero or multiple params are not.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) == 1:
        param = params[0]
        if (
            param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and param.default is inspect.Signature.empty
        ):
            return True
    return False


# ===================================================================
# @hook decorator
# ===================================================================


def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    custom: bool = False,
    deme: DemeSelector = "*",
    mode: str = "auto",
) -> Callable[[Callable[..., Any]], DecoratedHookFn]:
    """Decorator for all supported hook authoring styles.

    The decorated function gains a ``.register(pop)`` method that
    compiles and registers a ``CompiledHookDescriptor`` against a
    population instance.

    **Hook type auto-detection** (evaluated at ``.register()`` time):

    * ``selectors=`` is set → **Selector hook**
      (``compile_selector_hook``)
    * ``custom=True`` or function has required parameters →
      **Custom hook** (njit or Python wrapper)
    * Otherwise → **Declarative hook** (function returns
      ``List[HookOp]``, compiled via ``compile_declarative_hook``)

    For custom and selector hooks, Numba compilation is automatic — you
    do **not** need to stack ``@njit``.  When Numba is enabled the
    function is wrapped with ``njit_switch`` automatically; when Numba
    is disabled a pure-Python wrapper is used.

    In spatial simulations the ``deme_id`` parameter receives the current
    deme index, enabling one hook function to serve all demes with
    per-deme branching.

    Args:
        event: Hook event name (``"first"``, ``"early"``, ``"late"``,
            ``"finish"``).
        selectors: Symbolic selectors for selector-mode hooks.
        priority: Execution priority — lower values run first.
        custom: If ``True``, treat as custom hook regardless of signature.
        deme: Target deme(s).  ``"*"`` (default) means all demes.
            Accepts ``int``, ``list``, ``tuple``, or ``range``.
        mode: Selector passing style.  ``"auto"`` (default) auto-detects
            from the function signature.  ``"expand"`` passes each
            selector as a separate keyword argument.  ``"aggregate"``
            packs all selectors into a single namedtuple argument.

    Returns:
        A decorator that transforms a function into a ``DecoratedHookFn``
        with ``.register(pop)`` capability.

    Raises:
        ValueError: If *mode* is not one of ``"auto"``, ``"expand"``,
            ``"aggregate"``.

    Examples:

        Declarative hook (returns ops):

            @hook(event="early", priority=0)
            def cull_juveniles():
                return [Op.scale(ages=[0,1], factor=0.9)]

        Custom njit hook:

            @hook(event="first", priority=1)
            def release_males(state, config, deme_id=-1):
                state.individual_count[1, 2, 0] += 100
                return 0

        Selector hook:

            @hook(event="late", selectors={"target": "AA"})
            def count_homozygotes(state, config, target, deme_id=-1):
                ...
    """
    if mode not in ("auto", "expand", "aggregate"):
        raise ValueError(
            f"mode must be 'auto', 'expand', or 'aggregate', got {mode!r}"
        )

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
            """Compile this hook against *pop* and return a descriptor.

            Called by ``pop.set_hook()``.  Detects the hook type from
            the decorator metadata and function signature, then routes
            to the appropriate compiler.

            Args:
                pop: The population to compile against.
                event_override: Override the event name (used when
                    ``set_hook(event_name, ...)`` is called with a
                    different event than the decorator specifies).
                deme_selector_override: Override the deme selector
                    (used by SpatialPopulation to pin hooks to demes).

            Returns:
                A ``CompiledHookDescriptor`` registered on *pop*.
            """
            from ..numba_utils import NUMBA_ENABLED
            from .types import CompiledHookDescriptor

            actual_event = event_override or event
            actual_deme_selector: DemeSelector = (
                deme if deme_selector_override is None else deme_selector_override
            )
            if actual_event is None:
                raise ValueError(
                    f"Event not specified for hook '{func.__name__}'. "
                    "Specify in decorator @hook(event='...') or call "
                    "pop.set_hook('event', hook)"
                )

            # Detect hook type from decorator metadata + function signature.
            # Priority: explicit selectors > explicit custom > has params
            # (and not a single-param pop hook) > declarative (returns ops).
            has_required_params = _has_required_parameters(func)
            is_decl_pop_hook = _is_declarative_population_hook(func)
            is_custom_or_selector = (
                custom
                or selectors is not None
                or (has_required_params and not is_decl_pop_hook)
            )

            if is_custom_or_selector:
                # ---- Selector mode ----
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
                    # ---- Custom hook (njit or Python fallback) ----
                    if is_njit_function(func):
                        # Already decorated with @njit — use directly.
                        desc = CompiledHookDescriptor(
                            name=func.__name__,
                            event=actual_event,
                            priority=priority,
                            deme_selector=actual_deme_selector,
                            njit_fn=func,
                            meta={
                                "n_genotypes": pop.index_registry.num_genotypes(),
                                "n_ages": pop.config.n_ages,
                            },
                        )
                    else:
                        # Not @njit-decorated yet.  Wrap with njit_switch so
                        # the function can run in Numba's nopython mode.
                        # If Numba is disabled, njit_switch returns a Python
                        # callable — we detect this and use py_wrapper instead.
                        try:
                            decorated_func = njit_switch(cache=False)(func)
                            if NUMBA_ENABLED and is_njit_function(decorated_func):
                                norm_fn = _normalize_njit_fn(decorated_func)
                                desc = CompiledHookDescriptor(
                                    name=func.__name__,
                                    event=actual_event,
                                    priority=priority,
                                    deme_selector=actual_deme_selector,
                                    njit_fn=norm_fn,
                                    meta={
                                        "n_genotypes": pop.index_registry.num_genotypes(),
                                        "n_ages": pop.config.n_ages,
                                    },
                                )
                            else:
                                # Numba disabled — use Python wrapper.
                                wrapped_func = _normalize_py_hook(func)
                                desc = CompiledHookDescriptor(
                                    name=func.__name__,
                                    event=actual_event,
                                    priority=priority,
                                    deme_selector=actual_deme_selector,
                                    njit_fn=None,
                                    py_wrapper=wrapped_func,
                                    meta={
                                        "n_genotypes": pop.index_registry.num_genotypes(),
                                        "n_ages": pop.config.n_ages,
                                    },
                                )
                        except Exception:
                            # Fall back to Python wrapper.
                            wrapped_func = _normalize_py_hook(func)
                            desc = CompiledHookDescriptor(
                                name=func.__name__,
                                event=actual_event,
                                priority=priority,
                                deme_selector=actual_deme_selector,
                                njit_fn=None,
                                py_wrapper=wrapped_func,
                                meta={
                                    "n_genotypes": pop.index_registry.num_genotypes(),
                                    "n_ages": pop.config.n_ages,
                                },
                            )
            elif is_decl_pop_hook:
                # Legacy single-parameter population hook (Python only).
                if NUMBA_ENABLED:
                    raise TypeError(
                        f"Python hook '{func.__name__}' is not allowed "
                        "when Numba is enabled.  Please convert it to "
                        "@njit or use declarative Op hooks."
                    )
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    deme_selector=actual_deme_selector,
                    py_wrapper=func,
                    meta={
                        "n_genotypes": pop.index_registry.num_genotypes(),
                        "n_ages": pop.config.n_ages,
                    },
                )
            else:
                # ---- Declarative hook (returns List[HookOp]) ----
                # The function is called ONCE at registration time.  Its
                # return value (a list of HookOp objects) is compiled into
                # a CSR plan.  The function itself is NOT stored or called
                # at runtime — only the compiled plan is.
                result = func()
                if isinstance(result, list):
                    result_ops = cast(List[object], result)
                    if not all(isinstance(op, HookOp) for op in result_ops):
                        raise TypeError(
                            f"Declarative hook '{func.__name__}' must "
                            "return List[HookOp], or use custom=True "
                            "for custom mode."
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
                        f"Hook '{func.__name__}' must return List[HookOp] "
                        "for declarative mode, or use custom=True for "
                        "custom mode."
                    )

            hook_func.compiled = desc  # type: ignore
            pop.register_compiled_hook(desc)
            return desc

        hook_func.register = register  # type: ignore
        return hook_func

    return decorator
