"""Lifecycle wrappers and the hook-codegen integration pipeline.

This module lives in the **engine layer** and is the central integration
point between the hook system (``natal.hooks.*``) and the population
simulation loop.  It:

1. Defines ``LifecycleWrappers`` — a container that bundles pre-compiled
   lifecycle loop functions (``run_tick_fn``, ``run_fn``, etc.) with the
   compiled event hooks.
2. Provides ``compile_lifecycle_wrappers()`` — the main factory that takes
   compiled hook descriptors and a ``HookProgram`` registry, runs the full
   code generation pipeline, and returns a fully initialised
   ``LifecycleWrappers`` ready for use by population models.

----
Codegen approach
----
Rather than composing Python closures (which Numba cannot cache across
process restarts), this module **generates Python source code** for each
unique hook combination.  The source is written to ``.numba_cache/``,
loaded as a module, and the resulting ``@njit_switch(cache=True)``
functions are stable and cacheable.  Module globals (hook functions,
HookProgram arrays) are injected via ``setattr`` after loading.

Template files live in:
- ``engine/templates/`` — lifecycle wrapper templates
- ``hooks/templates/`` — unified mixed-type dispatch templates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from natal.hooks.compiler import (
    CompiledEventHooks,
    HookCallable,
    build_filtered_hook_program,
    compile_combined_hook,
    compile_unified_event_hook,
)
from natal.hooks.types import (
    EVENT_NAMES,
    CompiledHookDescriptor,
    DemeSelector,
    HookProgram,
    hash_key,
    load_codegen_module,
    stable_callable_identity,
    write_codegen_module,
)
import natal.numba_utils as _numba_utils

# Template directory for lifecycle wrapper templates.
_ENGINE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_engine_template(name: str) -> str:
    """Read a lifecycle codegen template from ``engine/templates/``."""
    return (_ENGINE_TEMPLATE_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LifecycleWrappers — container for compiled lifecycle loop functions
# ---------------------------------------------------------------------------


@dataclass
class LifecycleWrappers:
    """Container bundling compiled event hooks with pre-compiled lifecycle loops.

    A population model calls ``compile_lifecycle_wrappers()`` to obtain one of
    these.  It holds:

    * ``hooks`` — ``CompiledEventHooks`` with per-event combined callables
      (``first`` / ``early`` / ``late`` / ``finish``) and a CSR registry.
    * 8 lifecycle wrapper slots — pre-compiled ``@njit_switch`` functions
      for running one tick or multiple ticks, panmictic or spatial, with
      or without sperm storage (discrete vs structured lifecycle).

    Slots
    -----
    ``hooks``
        ``CompiledEventHooks`` — per-event hook container.
    ``run_tick_fn`` / ``run_fn``
        Structured lifecycle wrapper (single-tick / multi-tick).
    ``run_discrete_tick_fn`` / ``run_discrete_fn``
        Discrete lifecycle wrapper (single-tick / multi-tick).
    ``spatial_tick_fn`` / ``spatial_run_fn``
        Spatial structured lifecycle wrappers.
    ``spatial_discrete_tick_fn`` / ``spatial_discrete_run_fn``
        Spatial discrete lifecycle wrappers.
    """

    hooks: CompiledEventHooks = field(default_factory=CompiledEventHooks)
    run_tick_fn: Optional[HookCallable] = None
    run_fn: Optional[HookCallable] = None
    run_discrete_tick_fn: Optional[HookCallable] = None
    run_discrete_fn: Optional[HookCallable] = None
    spatial_tick_fn: Optional[HookCallable] = None
    spatial_run_fn: Optional[HookCallable] = None
    spatial_discrete_tick_fn: Optional[HookCallable] = None
    spatial_discrete_run_fn: Optional[HookCallable] = None

    @property
    def registry(self) -> Optional[HookProgram]:
        """Shorthand for ``self.hooks.registry`` (backward-compat)."""
        return self.hooks.registry


# ---------------------------------------------------------------------------
# Lifecycle wrapper codegen
# ---------------------------------------------------------------------------


def _gen_lifecycle_source(
    is_discrete: bool,
    tick_fn_name: str,
    run_fn_name: str,
) -> str:
    """Read a lifecycle template and substitute function name placeholders.

    Args:
        is_discrete: ``True`` for discrete-generation, ``False`` for
            age-structured.
        tick_fn_name: Name to substitute for ``TICK_FN_NAME``.
        run_fn_name: Name to substitute for ``RUN_FN_NAME``.

    Returns:
        The template source with placeholders replaced.
    """
    name = (
        "lifecycle_discrete_v2.tmpl.py"
        if is_discrete
        else "lifecycle_structured.tmpl.py"
    )
    return _read_engine_template(name).replace("TICK_FN_NAME", tick_fn_name).replace(
        "RUN_FN_NAME", run_fn_name
    )


def compile_lifecycle_wrapper(
    is_discrete: bool,
    first_hook: HookCallable,
    early_hook: HookCallable,
    late_hook: HookCallable,
) -> tuple[HookCallable, HookCallable]:
    """Generate a lifecycle wrapper module with hooks as module globals.

    Each unique ``(first, early, late)`` hook combination gets its own
    Numba ``@njit(cache=True)`` function keyed by source-code hash.
    This is the mechanism that makes hook compilation survive process
    restarts — Numba cannot do this with function-valued parameters,
    so we generate source code and inject hooks as module globals.

    Args:
        is_discrete: If ``True``, generate discrete-generation wrappers
            using ``DiscretePopulationConfig``.  If ``False``, generate
            age-structured wrappers with sperm storage.
        first_hook: Combined njit function for the ``first`` event.
        early_hook: Combined njit function for the ``early`` event.
        late_hook: Combined njit function for the ``late`` event.

    Returns:
        ``(tick_fn, run_fn)`` — the tick function executes one lifecycle
        tick; the run function loops for ``n_ticks`` with history recording.
    """
    mode = "discrete" if is_discrete else "structured"
    parts = ["lifecycle_" + mode] + [
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
    """Read a spatial lifecycle template and substitute placeholders.

    The spatial template wraps the panmictic tick inside ``prange``,
    running one tick per deme in parallel.

    Args:
        is_discrete: ``True`` for discrete-generation.
        tick_fn_name: Spatial tick function name.
        run_fn_name: Spatial run function name.
        panmictic_stem: Module stem of the corresponding panmictic wrapper.
        panmictic_tick_fn_name: Tick function name in the panmictic module.

    Returns:
        The template source with placeholders replaced.
    """
    name = (
        "spatial_lifecycle_discrete.tmpl.py"
        if is_discrete
        else "spatial_lifecycle_structured.tmpl.py"
    )
    return (
        _read_engine_template(name)
        .replace("PANMICTIC_TICK_FN_NAME", panmictic_tick_fn_name)
        .replace("PANMICTIC_STEM", panmictic_stem)
        .replace("TICK_FN_NAME", tick_fn_name)
        .replace("RUN_FN_NAME", run_fn_name)
    )


def compile_spatial_lifecycle_wrapper(
    is_discrete: bool,
    first_hook: HookCallable,
    early_hook: HookCallable,
    late_hook: HookCallable,
) -> tuple[HookCallable, HookCallable]:
    """Generate a spatial lifecycle wrapper that delegates to the panmictic
    tick inside ``prange``.

    The spatial module does **not** hold its own hook globals.  It imports
    and calls the panmictic tick function, which resolves ``_FIRST_HOOK``
    / ``_EARLY_HOOK`` / ``_LATE_HOOK`` via its own module globals.

    Args:
        is_discrete: If ``True``, generate discrete-generation per-deme
            lifecycle.
        first_hook: Combined njit function for the ``first`` event.
        early_hook: Combined njit function for the ``early`` event.
        late_hook: Combined njit function for the ``late`` event.

    Returns:
        ``(tick_fn, run_fn)`` — the spatial tick and run functions.
    """
    mode = "discrete" if is_discrete else "structured"

    # Panmictic wrapper identity (same key as compile_lifecycle_wrapper).
    panmictic_parts = ["lifecycle_" + mode] + [
        stable_callable_identity(fn) for fn in [first_hook, early_hook, late_hook]
    ]
    panmictic_key = hash_key(panmictic_parts)
    panmictic_stem = f"lifecycle_{mode}_{panmictic_key}"
    panmictic_tick_fn_name = f"_lifecycle_tick_{panmictic_key}"

    # Spatial wrapper identity.
    spatial_parts = ["spatial_lifecycle_" + mode] + [
        stable_callable_identity(fn) for fn in [first_hook, early_hook, late_hook]
    ]
    spatial_key = hash_key(spatial_parts)
    module_stem = f"spatial_lifecycle_{mode}_{spatial_key}"
    tick_fn_name = f"_spatial_tick_{spatial_key}"
    run_fn_name = f"_spatial_run_{spatial_key}"

    source = _gen_spatial_lifecycle_source(
        is_discrete,
        tick_fn_name,
        run_fn_name,
        panmictic_stem,
        panmictic_tick_fn_name,
    )
    module_path = write_codegen_module(module_stem, source)
    module = load_codegen_module(module_stem, module_path)

    return getattr(module, tick_fn_name), getattr(module, run_fn_name)


# ---------------------------------------------------------------------------
# Main integration pipeline
# ---------------------------------------------------------------------------


def compile_lifecycle_wrappers(
    compiled_hooks: List[CompiledHookDescriptor],
    registry: Optional[HookProgram] = None,
    include_spatial_wrappers: bool = False,
) -> LifecycleWrappers:
    """Build event-wise combined callables and lifecycle wrappers.

    This is the **main integration point** between the compilation
    pipeline and the population models.  It performs four steps:

    1. **Detect mixed events** — events that have both CSR (plan) and
       njit descriptors.
    2. **Build filtered registry** — CSR hooks for mixed events are
       removed from the ``HookProgram`` so the template's dispatch
       becomes a no-op (the unified function handles them instead).
    3. **Generate per-event functions** — for mixed events, generate
       unified dispatch (CSR + njit interleaved).  For non-mixed,
       use ``compile_combined_hook`` (njit-only chain).
    4. **Compile lifecycle wrappers** — structured and discrete,
       panmictic and spatial, using the generated hook functions.

    Args:
        compiled_hooks: All compiled descriptors for this population.
        registry: The original ``HookProgram`` (before filtering).
        include_spatial_wrappers: If ``True``, also compile spatial
            lifecycle wrappers.

    Returns:
        A fully initialised ``LifecycleWrappers`` ready for use by
        the population model.
    """
    if _numba_utils.NUMBA_ENABLED:
        for desc in compiled_hooks:
            if desc.py_wrapper is not None:
                raise TypeError(
                    f"Hook '{desc.name}' uses py_wrapper, which is "
                    "not allowed when Numba is enabled."
                )

    hooks = CompiledEventHooks()
    hooks.registry = registry

    # ---- Step 1: detect mixed events ----
    # An event is "mixed" when at least one CSR descriptor (plan.n_ops>0)
    # and at least one njit descriptor coexist.  CSR-only or njit-only
    # events are NOT mixed and use the standard (faster) paths.
    mixed_events: set[str] = set()
    for event_name in EVENT_NAMES:
        event_descs = [d for d in compiled_hooks if d.event == event_name]
        has_csr = any(
            d.plan is not None and d.plan.n_ops > 0 for d in event_descs
        )
        has_njit = any(d.njit_fn is not None for d in event_descs)
        if has_csr and has_njit:
            mixed_events.add(event_name)

    # ---- Step 2: filtered registry ----
    # Rebuild the HookProgram without CSR hooks for mixed events.
    # The lifecycle template still calls execute_csr_event_* for every
    # event; with CSR ops removed, those calls become no-ops and the
    # unified function handles everything.
    if registry is not None and mixed_events:
        hooks.registry = build_filtered_hook_program(
            compiled_hooks, mixed_events
        )

    # ---- Step 3: HookProgram index mapping ----
    # The HookProgram packs hooks in (EVENT_NAMES, priority) order.
    # We need each CSR descriptor's global hook_idx so the unified
    # function can pass it to _execute_single_csr_hook.
    hook_idx_map: Dict[int, int] = {}  # id(desc) → hook_idx
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

    # ---- Collect njit hooks per event ----
    hooks_by_event: Dict[
        str, List[Tuple[int, HookCallable, DemeSelector]]
    ] = {name: [] for name in EVENT_NAMES}
    for desc in compiled_hooks:
        if desc.njit_fn is not None and desc.event in hooks_by_event:
            hooks_by_event[desc.event].append(
                (desc.priority, desc.njit_fn, desc.deme_selector)
            )

    # ---- Step 4: generate per-event functions ----
    # Two codegen paths diverge here:
    #   Mixed event  → compile_unified_event_hook (CSR + njit interleaved)
    #   Non-mixed    → compile_combined_hook (njit-only chain)
    # Mixed events generate two variants (with/without sperm) — the
    # without-sperm version is stored in a local dict and fed to the
    # discrete lifecycle wrapper below.
    discrete_hooks: Dict[str, HookCallable] = {}

    for event_name, hook_list in hooks_by_event.items():
        if not hook_list:
            # No njit hooks.  If mixed (CSR-only), still build a
            # unified function so the CSR hooks execute.
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
                    hooks.set_hook(
                        event_name,
                        compile_unified_event_hook(
                            csr_schedule, [], None, registry,
                            has_sperm_storage=True,
                        ),
                    )
                    discrete_hooks[event_name] = (
                        compile_unified_event_hook(
                            csr_schedule, [], None, registry,
                            has_sperm_storage=False,
                        )
                    )
            continue

        hook_list.sort(key=lambda x: x[0])
        njit_fns = [fn for _, fn, _ in hook_list]
        deme_selectors = cast(
            "List[DemeSelector]", [ds for _, _, ds in hook_list]
        )

        if event_name in mixed_events and registry is not None:
            # Mixed: build priority-ordered schedule.
            event_descs = sorted(
                [d for d in compiled_hooks if d.event == event_name],
                key=lambda d: d.priority,
            )
            schedule: List[Tuple[str, int]] = []
            njit_idx = 0
            for desc in event_descs:
                if desc.njit_fn is not None:
                    schedule.append(("njit", njit_idx))
                    njit_idx += 1
                elif desc.plan is not None and desc.plan.n_ops > 0:
                    schedule.append(("csr", hook_idx_map[id(desc)]))

            hooks.set_hook(
                event_name,
                compile_unified_event_hook(
                    schedule, njit_fns, deme_selectors, registry,
                    has_sperm_storage=True,
                ),
            )
            discrete_hooks[event_name] = compile_unified_event_hook(
                schedule, njit_fns, deme_selectors, registry,
                has_sperm_storage=False,
            )
        else:
            # Non-mixed: standard njit-only chain.
            combined = compile_combined_hook(njit_fns, deme_selectors)
            hooks.set_hook(event_name, combined)

    # ---- Step 5: compile lifecycle wrappers ----
    # Structured wrappers use the standard hooks (with sperm).
    # Discrete wrappers use the without-sperm variants for mixed
    # events; for non-mixed events they fall back to the standard hook.
    first_hook = hooks.first
    early_hook = hooks.early
    late_hook = hooks.late

    first_d = discrete_hooks.get("first", first_hook)
    early_d = discrete_hooks.get("early", early_hook)
    late_d = discrete_hooks.get("late", late_hook)

    result = LifecycleWrappers(hooks=hooks)

    if _numba_utils.NUMBA_ENABLED:
        result.run_tick_fn, result.run_fn = compile_lifecycle_wrapper(
            False, first_hook, early_hook, late_hook
        )
        result.run_discrete_tick_fn, result.run_discrete_fn = (
            compile_lifecycle_wrapper(True, first_d, early_d, late_d)
        )

        if include_spatial_wrappers:
            result.spatial_tick_fn, result.spatial_run_fn = (
                compile_spatial_lifecycle_wrapper(
                    False, first_hook, early_hook, late_hook
                )
            )
            result.spatial_discrete_tick_fn, result.spatial_discrete_run_fn = (
                compile_spatial_lifecycle_wrapper(
                    True, first_d, early_d, late_d
                )
            )

    return result
