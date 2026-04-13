"""Unified hook entrypoints and event-wise compiler.
This module connects three authoring styles into one runtime contract:
1) declarative hooks (Op list -> CompiledHookPlan)
2) selector hooks (symbolic selectors -> wrapper/compiled callable)
3) custom hooks (user-provided njit or Python callback)
"""

from __future__ import annotations

import inspect
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
from natal.kernels.codegen import (
    compile_kernel_bound_wrappers,
    compile_spatial_kernel_bound_wrappers,
)
from natal.numba_utils import njit_switch

from .declarative import HookOp
from .types import (
    EVENT_NAMES,
    DemeSelector,
    HookProgram,
    hash_key,
    load_codegen_module,
    stable_callable_identity,
    validate_numba_hook_required,
    write_codegen_module,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation

    from .types import CompiledHookDescriptor

# Plain callable type — used everywhere that only needs "something you can call".
# noop hooks, njit functions, combined hooks, kernel wrappers all satisfy this.
HookCallable = Callable[..., Any]
KernelWrapperCompiler = Callable[[HookCallable, HookCallable, HookCallable], Tuple[HookCallable, HookCallable, HookCallable, HookCallable]]
SpatialKernelWrapperCompiler = Callable[[HookCallable, HookCallable, HookCallable], Tuple[HookCallable, HookCallable]]
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
    numba_mode: Any
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

def compile_combined_hook(njit_fns: List[HookCallable], name: str = "combined_hook") -> HookCallable:
    """Combine multiple njit hooks into one generated njit function.

    We generate source code instead of composing Python closures so the result
    remains callable from njit kernels.
    """
    if len(njit_fns) == 0:
        return _noop_hook

    if len(njit_fns) == 1:
        return njit_fns[0]

    # Stable key ensures deterministic module names and cache reuse.
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
            f"def {fn_name}(ind_count, tick, deme_id=0):",
        ]
    )

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

class CompiledEventHooks:
    """Container for event-wise combined hook callables.

    Kernel code expects one callable per event name. This class stores those
    callables and optionally the declarative ``HookProgram`` registry.
    """
    __slots__ = (
        "first",
        "early",
        "late",
        "finish",
        "registry",
        "_event_hooks",
        "run_tick_fn",
        "run_fn",
        "run_discrete_tick_fn",
        "run_discrete_fn",
        "run_spatial_tick_fn",
        "run_spatial_fn",
    )
    # Type annotations for attributes
    first: HookCallable
    early: HookCallable
    late: HookCallable
    finish: HookCallable
    registry: Optional[Any]
    _event_hooks: Dict[str, HookCallable]
    run_tick_fn: Optional[HookCallable]
    run_fn: Optional[HookCallable]
    run_discrete_tick_fn: Optional[HookCallable]
    run_discrete_fn: Optional[HookCallable]
    run_spatial_tick_fn: Optional[HookCallable]
    run_spatial_fn: Optional[HookCallable]

    def __init__(self) -> None:
        self.first = _noop_hook
        self.early = _noop_hook
        self.late = _noop_hook
        self.finish = _noop_hook
        self.registry = None
        self._event_hooks = dict.fromkeys(EVENT_NAMES, _noop_hook)
        self.run_tick_fn = None
        self.run_fn = None
        self.run_discrete_tick_fn = None
        self.run_discrete_fn = None
        self.run_spatial_tick_fn = None
        self.run_spatial_fn = None

    def get_hook(self, event_name: str) -> HookCallable:
        return self._event_hooks.get(event_name, _noop_hook)

    def set_hook(self, event_name: str, hook_fn: HookCallable) -> None:
        self._event_hooks[event_name] = hook_fn
        setattr(self, event_name, hook_fn)

    @staticmethod
    def from_compiled_hooks(compiled_hooks: List[CompiledHookDescriptor], registry: Optional[HookProgram] = None) -> CompiledEventHooks:
        """Build event-wise combined callables from descriptors."""
        from ..numba_utils import NUMBA_ENABLED

        if NUMBA_ENABLED:
            for desc in compiled_hooks:
                if desc.py_wrapper is not None:
                    raise TypeError(
                        f"Hook '{desc.name}' uses py_wrapper, which is not allowed when Numba is enabled."
                    )

        result = CompiledEventHooks()
        result.registry = registry
        hooks_by_event: Dict[str, List[Tuple[int, HookCallable]]] = {name: [] for name in EVENT_NAMES}

        for desc in compiled_hooks:
            if desc.njit_fn is not None and desc.event in hooks_by_event:
                hooks_by_event[desc.event].append((desc.priority, desc.njit_fn))

        for event_name, hook_list in hooks_by_event.items():
            if hook_list:
                hook_list.sort(key=lambda x: x[0])
                njit_fns = [fn for _, fn in hook_list]
                combined = compile_combined_hook(njit_fns, f"combined_{event_name}_hooks")
                result.set_hook(event_name, combined)

        (
            result.run_tick_fn,
            result.run_fn,
            result.run_discrete_tick_fn,
            result.run_discrete_fn,
        ) = compile_kernel_bound_wrappers(result.first, result.early, result.late)

        (
            result.run_spatial_tick_fn,
            result.run_spatial_fn,
        ) = compile_spatial_kernel_bound_wrappers(result.first, result.early, result.late)

        return result

def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    numba: bool = False,
    deme_selector: DemeSelector = "*",
) -> Callable[[Callable[..., Any]], DecoratedHookFn]:
    """Decorator entrypoint for all supported hook authoring styles.

    The decorated function gets a ``register(pop, event_override=None)``
    helper that compiles and registers a ``CompiledHookDescriptor``.

    Args:
        event: Hook event name.
        selectors: Optional symbolic selectors for selector-mode hooks.
        priority: Execution priority (lower values run earlier).
        numba: Whether this hook is an explicit custom njit hook.
        deme_selector: Transport-only selector metadata. The compiler stores
            and forwards this value into ``CompiledHookDescriptor`` but does
            not interpret spatial routing semantics itself.
    """
    def decorator(func: Callable[..., Any]) -> DecoratedHookFn:
        hook_func = cast(DecoratedHookFn, func)
        # Store metadata for debugging / introspection / future recompilation.
        hook_func.meta = {
            "event": event,
            "selectors": selectors or {},
            "priority": priority,
            "numba_mode": numba,
            "deme_selector": deme_selector,
        }
        hook_func.compiled = None
        hook_func.event = event
        hook_func.selectors = selectors or {}
        hook_func.priority = priority
        hook_func.numba_mode = numba
        hook_func.deme_selector = deme_selector

        def register(
            pop: BasePopulation[Any],
            event_override: Optional[str] = None,
            deme_selector_override: Optional[DemeSelector] = None,
        ) -> CompiledHookDescriptor:
            """Compile this hook against one population instance."""
            from ..numba_utils import NUMBA_ENABLED
            from .types import CompiledHookDescriptor

            actual_event = event_override or event
            actual_deme_selector = deme_selector if deme_selector_override is None else deme_selector_override
            if actual_event is None:
                raise ValueError(
                    f"Event not specified for hook '{func.__name__}'. "
                    "Specify in decorator @hook(event='...') or call pop.set_hook('event', hook)"
                )

            if numba:
                # Mode 1: explicit custom njit hook.
                validate_numba_hook_required(func, func.__name__, "@hook(numba=True)")
                norm_fn = _normalize_njit_fn(hook_func)
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    deme_selector=actual_deme_selector,
                    njit_fn=norm_fn,
                    meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                )

            elif selectors:
                # Mode 2: selector-based hook (python or njit wrapper path).
                desc = compile_selector_hook(
                    func,
                    pop,
                    actual_event,
                    selectors,
                    priority,
                    numba_mode=numba,
                    deme_selector=actual_deme_selector,
                )

            else:
                # Mode 3: declarative hook if function returns HookOp list,
                # otherwise plain Python callback fallback.
                if _has_required_parameters(func):
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
                                "or use function arguments for python hook mode."
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
                        if NUMBA_ENABLED:
                            raise TypeError(
                                f"Python hook '{func.__name__}' is not allowed when Numba is enabled. "
                                "Please convert it to @njit or use declarative Op hooks."
                            )
                        def _py_wrapper(p: object, f: HookCallable = func) -> object:
                            return f(p)

                        desc = CompiledHookDescriptor(
                            name=func.__name__,
                            event=actual_event,
                            priority=priority,
                            deme_selector=actual_deme_selector,
                            py_wrapper=_py_wrapper,
                            meta={"n_genotypes": pop.index_registry.num_genotypes(), "n_ages": pop.config.n_ages},
                        )

            hook_func.compiled = desc  # type: ignore
            pop.register_compiled_hook(desc)
            return desc

        hook_func.register = register  # type: ignore
        return hook_func

    return decorator

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
