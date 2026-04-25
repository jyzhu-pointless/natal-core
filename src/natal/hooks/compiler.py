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

import numpy as np

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

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "kernels" / "templates"


def _read_template(name: str) -> str:
    """Read a lifecycle codegen template from ``kernels/templates/``."""
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
def _noop_hook(ind_count: np.ndarray, tick: int, deme_id: int = 0) -> int:
    """Default hook implementation used for missing event handlers."""
    return 0

noop_hook = _noop_hook

def _normalize_njit_fn(fn: HookCallable) -> HookCallable:
    """Ensure an njit hook matches the internal (ind_count, tick, deme_id) signature.

    If the user provided a 2-arg function, wrap it.
    """
    py_fn = getattr(fn, "py_func", fn)
    sig = inspect.signature(py_fn)
    params = list(sig.parameters.values())
    # If it already matches or has varargs, assume it handles 3 args.
    if len(params) >= 3:
        return fn
    # Wrap 2-arg function: (ind_count, tick) -> (ind_count, tick, deme_id)
    @njit_switch(cache=True)
    def wrapped(ind_count: np.ndarray, tick: int, deme_id: int = 0) -> object:
        return fn(ind_count, tick)
    return wrapped


def _normalize_py_hook(fn: HookCallable) -> HookCallable:
    """Ensure a Python custom hook matches the (ind_count, tick, deme_id) signature.

    If user provided a 2-arg function, wrap it.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) >= 3:
        return fn
    def wrapped(ind_count: np.ndarray, tick: int, deme_id: int = 0) -> object:
        return fn(ind_count, tick)
    return wrapped

def compile_combined_hook(
    njit_fns: List[HookCallable],
    deme_selectors: Optional[List[DemeSelector]] = None,
) -> HookCallable:
    """Combine multiple njit hooks into one generated njit function.

    We generate source code instead of composing Python closures so the result
    remains callable from njit kernels.

    When ``deme_selectors`` is provided and contains non-wildcard values,
    each hook call is wrapped with an ``if deme_id == X`` guard so that
    per-deme hooks only execute for their target deme(s) — critical for
    spatial simulations where all hooks share one combined function.

    Args:
        njit_fns: List of njit-compiled hook functions.
        name: Human-readable name for the generated function.
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
    lines = ["from natal.hook_dsl import njit_switch"]
    lines.extend([f"{placeholder} = None" for placeholder in placeholder_names])
    lines.extend(
        [
            "",
            "@njit_switch(cache=True)",
            f"def {fn_name}(ind_count, tick, deme_id=-1):",
        ]
    )

    if needs_guard:
        for placeholder, ds in zip(placeholder_names, ds_list):
            if ds == "*":
                lines.append(f"    _result = {placeholder}(ind_count, tick, deme_id)")
                lines.append("    if _result != 0:")
                lines.append("        return _result")
            elif isinstance(ds, int):
                lines.append(f"    if deme_id == {int(ds)}:")
                lines.append(f"        _result = {placeholder}(ind_count, tick, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
            elif isinstance(ds, range):
                lines.append(f"    if {ds.start} <= deme_id < {ds.stop}:")
                lines.append(f"        _result = {placeholder}(ind_count, tick, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
            else:
                # List or tuple — generate a tuple literal for Numba's ``in``.
                items = ", ".join(str(int(x)) for x in ds)
                lines.append(f"    if deme_id in ({items}):")
                lines.append(f"        _result = {placeholder}(ind_count, tick, deme_id)")
                lines.append("        if _result != 0:")
                lines.append("            return _result")
    else:
        for placeholder in placeholder_names:
            lines.append(f"    _result = {placeholder}(ind_count, tick, deme_id)")
            lines.append("    if _result != 0:")
            lines.append("        return _result")
    lines.append("    return 0")
    lines.append("")

    module_path = write_codegen_module(module_stem, "\n".join(lines))
    module = load_codegen_module(module_stem, module_path)

    for placeholder, fn in zip(placeholder_names, njit_fns):
        setattr(module, placeholder, fn)

    return getattr(module, fn_name)

def _gen_lifecycle_source(
    is_discrete: bool,
    tick_fn_name: str,
    run_fn_name: str,
) -> str:
    """Generate the source code for a lifecycle wrapper module.

    Reads the template from ``kernels/templates/`` and substitutes
    ``TICK_FN_NAME`` and ``RUN_FN_NAME`` placeholders.
    """
    name = "lifecycle_discrete.tmpl.py" if is_discrete else "lifecycle_structured.tmpl.py"
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
            wrappers.  Otherwise generate age-structured wrappers.
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

    Reads the template from ``kernels/templates/`` and substitutes
    ``TICK_FN_NAME``, ``RUN_FN_NAME``, ``PANMICTIC_STEM``,
    ``PANMICTIC_TICK_FN_NAME`` placeholders.
    """
    name = "spatial_lifecycle_discrete.tmpl.py" if is_discrete else "spatial_lifecycle_structured.tmpl.py"
    return (_read_template(name)
        .replace("TICK_FN_NAME", tick_fn_name)
        .replace("RUN_FN_NAME", run_fn_name)
        .replace("PANMICTIC_STEM", panmictic_stem)
        .replace("PANMICTIC_TICK_FN_NAME", panmictic_tick_fn_name))


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

    @staticmethod
    def from_compiled_hooks(compiled_hooks: List[CompiledHookDescriptor], registry: Optional[HookProgram] = None) -> CompiledEventHooks:
        """Build event-wise combined callables and lifecycle wrappers.

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
        hooks_by_event: Dict[str, List[Tuple[int, HookCallable, DemeSelector]]] = {name: [] for name in EVENT_NAMES}

        for desc in compiled_hooks:
            if desc.njit_fn is not None and desc.event in hooks_by_event:
                hooks_by_event[desc.event].append((desc.priority, desc.njit_fn, desc.deme_selector))

        for event_name, hook_list in hooks_by_event.items():
            if hook_list:
                hook_list.sort(key=lambda x: x[0])
                njit_fns = [fn for _, fn, _ in hook_list]
                deme_selectors = cast("List[DemeSelector]", [ds for _, _, ds in hook_list])
                combined = compile_combined_hook(njit_fns, deme_selectors)
                result.set_hook(event_name, combined)

        # Pre-compile lifecycle wrappers per hook combination so Numba
        # caches the compilation across process restarts. The wrappers use
        # module-level globals for the combined hooks instead of function
        # parameters, giving each combination a unique source-code hash.
        first_hook = result.first
        early_hook = result.early
        late_hook = result.late

        # Always compile lifecycle wrappers when Numba is enabled so the
        # population model can use them unconditionally.  Even with zero
        # user hooks the wrapper compiles with _noop_hook globals, and its
        # source hash stays stable across runs.
        if NUMBA_ENABLED:
            result.run_tick_fn, result.run_fn = compile_lifecycle_wrapper(
                False, first_hook, early_hook, late_hook,
            )
            result.run_discrete_tick_fn, result.run_discrete_fn = compile_lifecycle_wrapper(
                True, first_hook, early_hook, late_hook,
            )

            result.spatial_tick_fn, result.spatial_run_fn = compile_spatial_lifecycle_wrapper(
                False, first_hook, early_hook, late_hook,
            )
            result.spatial_discrete_tick_fn, result.spatial_discrete_run_fn = compile_spatial_lifecycle_wrapper(
                True, first_hook, early_hook, late_hook,
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
    """
    def decorator(func: Callable[..., Any]) -> DecoratedHookFn:
        hook_func = cast(DecoratedHookFn, func)
        hook_func.meta = {
            "event": event,
            "selectors": selectors or {},
            "priority": priority,
            "custom": custom,
            "deme_selector": deme,
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
