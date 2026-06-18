"""``@hook()`` decorator — the front door of the hook system.

Detects hook type from decorator metadata and function signature, routes
to the appropriate compiler (declarative / selector / custom njit), and
returns a ``DecoratedHookFn`` with a ``.register(pop)`` method.

The decorator supports three authoring styles:

1. **Declarative** — function returns ``List[HookOp]``.
2. **Selector** — ``selectors={}`` specified, compiled via ``compile_selector_hook``.
3. **Custom njit / Python** — explicit ``custom=True`` or auto-detected via
   required parameters (2+ positional args).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, cast

from natal.hooks.declarative import compile_declarative_hook
from natal.hooks.selector import compile_selector_hook
from natal.numba_utils import njit_switch

from .declarative import HookOp
from .types import (
    CompiledHookDescriptor,
    DemeSelector,
    HookCallable,
    is_njit_function,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation


# ---------------------------------------------------------------------------
# Protocol for decorated functions
# ---------------------------------------------------------------------------


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
# Signature normalisation helpers
# ---------------------------------------------------------------------------


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
        """Thunk adapting 2-arg fn to 3-arg ``(state, config, deme_id)``."""
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
        """Python thunk adapting 2-arg fn to 3-arg ``(state, config, deme_id)``."""
        return fn(state, config)

    return wrapped2


# ---------------------------------------------------------------------------
# Hook type auto-detection
# ---------------------------------------------------------------------------


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
        """Transform *func* into a ``DecoratedHookFn`` with ``.register(pop)``."""
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
