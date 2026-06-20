"""Code generation for hook dispatch functions.

Rather than composing Python closures at runtime (which Numba cannot cache
across process restarts), this module **generates Python source code** for
each unique hook combination.  The source is written to ``.numba_cache/``,
loaded as a module, and the resulting ``@njit_switch(cache=True)``
functions are stable and cacheable.  Module globals (hook functions,
HookProgram arrays) are injected via ``setattr`` after loading.

Two codegen strategies:

1. ``compile_combined_hook`` — pure njit chain (no CSR interleaving).
   Generates a sequential-call function with per-call deme guards.

2. ``compile_unified_event_hook`` — mixed CSR + njit dispatch.
   Generates a function that interleaves ``_execute_single_csr_hook``
   calls with njit function calls, ordered by priority.

Template files live in ``hooks/templates/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np

from ..types import (
    EVENT_NAMES,
    CompiledHookDescriptor,
    DemeSelector,
    HookCallable,
    HookProgram,
    hash_key,
    load_codegen_module,
    stable_callable_identity,
    write_codegen_module,
)
from .container import noop_hook

_HOOK_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _read_hook_template(name: str) -> str:
    """Read a hook codegen template from ``hooks/templates/``."""
    return (_HOOK_TEMPLATE_DIR / name).read_text(encoding="utf-8")


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
        return noop_hook

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

    # ---- Build module-level placeholder declarations ----
    # Each _FN_i = lambda … is a type-stub that gets overridden via setattr.
    fn_decl_lines: List[str] = []
    for pname in placeholder_names:
        fn_decl_lines.append(
            f"{pname}: Callable[..., int] = "
            "lambda _s, _c=None, _d=-1: 0  # type: ignore[assignment]"
        )

    # ---- Build schedule body from njit entry template ----
    # Reuse the same njit entry fragment that unified dispatch uses.
    # Guard conditions computed identically to compile_unified_event_hook.
    njit_tmpl = _read_hook_template("unified_hook_njit_entry.tmpl.py")
    schedule_entries: List[str] = []

    for pname, ds in zip(placeholder_names, ds_list):
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
            njit_tmpl.replace("PLACEHOLDER_NJIT_FN_NAME", pname).replace(
                "PLACEHOLDER_NJIT_GUARD_CONDITION", guard_cond
            )
        )

    # ---- Assemble module from template ----
    template = _read_hook_template("combined_hook.tmpl.py")
    source = (
        template.replace("_combined_hook_TEMPLATE", fn_name)
        .replace("# PLACEHOLDER_FN_DECLARATIONS", "\n".join(fn_decl_lines))
        .replace("# PLACEHOLDER_SCHEDULE_BODY", "\n".join(schedule_entries))
    )

    # ---- Write, load, wire up globals ----
    # Module is written to .numba_cache/ so Numba's cache=True survives
    # restarts.  setattr overrides each placeholder with the real njit
    # function before returning the generated callable.
    module_path = write_codegen_module(module_stem, source)
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
    ``csr_kernel`` and stores the ``HookProgram`` arrays as module globals
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
        An ``@njit_switch(cache=True)`` function with signature
        ``(state, config=None, deme_id=-1) -> int``.  Whether it reads
        ``state.sperm_storage`` or uses a dummy ``(0,0,0)`` array is
        determined by the *has_sperm_storage* parameter.  The caller
        (``compile_lifecycle_wrappers``) invokes this function twice to
        obtain separate with/without-sperm variants for structured and
        discrete lifecycle wrappers respectively.
    """
    if len(schedule) == 0:
        return noop_hook

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
    # ``\"PLACEHOLDER_SPERM_SETUP\"`` includes the quotes so the replacement
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

            def _append_deme_sel(s: DemeSelector) -> None:
                """Append a serialised deme selector entry for *s*."""
                # Called for every hook to keep arrays aligned with n_hooks.
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
