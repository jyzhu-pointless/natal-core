"""``@hook()`` decorator — the front door of the hook system.

The hook contract recognizes three authoring shapes:

1. **Declarative** — function takes no parameters and returns
   ``List[HookOp]``; compiled to a CSR plan at build time.
2. **Callback** — function takes exactly one parameter (the
   :class:`~natal.frontend.hooks.tick_context.TickContext`); a Python
   callable fired at event boundaries on every backend.
3. **Selector callback** — ``selectors={}`` specified; resolved selector
   values are injected as keyword arguments after the context.

The legacy njit-era ``(state, config, deme_id)`` signature is rejected
with a :class:`TypeError` that guides authors to the new form.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Protocol, cast

from natal.frontend.hooks.types import (
    CompiledHookDescriptor,
    DemeSelector,
    HookLayout,
)

from .declarative import HookOp, compile_declarative_hook
from .selector import compile_selector_callback

# ---------------------------------------------------------------------------
# Protocol for decorated functions
# ---------------------------------------------------------------------------


class DecoratedHookFn(Protocol):
    """Protocol for functions that have been decorated with ``@hook()``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the decorated hook function with arbitrary arguments."""

    __name__: str
    meta: Dict[str, Any]
    event: Any
    selectors: Dict[str, Any]
    priority: int
    deme_selector: Any
    register: Callable[..., Any]


# ---------------------------------------------------------------------------
# Signature inspection
# ---------------------------------------------------------------------------


def _count_required_parameters(func: Callable[..., Any]) -> int:
    """Count required positional parameters of *func*.

    Args:
        func: The callable to inspect.

    Returns:
        The number of positional parameters without defaults.
    """
    sig = inspect.signature(func)
    return sum(
        1
        for param in sig.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and param.default is inspect.Signature.empty
    )


# ===================================================================
# @hook decorator
# ===================================================================


def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    deme: DemeSelector = "*",
) -> Callable[[Callable[..., Any]], DecoratedHookFn]:
    """Decorator for all supported hook authoring shapes.

    **Shape detection** (evaluated at compile time, when the declaring
    builder or population provides the layout):

    * ``selectors=`` is set → **Selector callback** (selector values
      injected as keyword arguments after the context).
    * function takes no required parameters → **Declarative hook**
      (called once; must return ``List[HookOp]``).
    * function takes exactly one required parameter → **Callback**
      (``def hook(pop) -> int``; ``0``/``None`` continues, nonzero stops).
    * anything else → :class:`TypeError` (the legacy
      ``(state, config, deme_id)`` njit-era signature has no migration
      channel; there are no existing users to support).

    Args:
        event: Hook event name (``"first"``, ``"early"``, ``"late"``,
            ``"finish"``).  May also be supplied by the registration call.
        selectors: Symbolic selectors resolved once at registration and
            injected as keyword arguments.
        priority: Execution priority — lower values run first.
        deme: Target deme(s).  ``"*"`` (default) means all demes.
            Accepts ``int``, ``list``, ``tuple``, or ``range``.

    Returns:
        A decorator that transforms a function into a ``DecoratedHookFn``
        with ``.register(pop)`` capability.

    Examples:

        Declarative hook (returns ops):

            @hook(event="early", priority=0)
            def cull_juveniles():
                return [Op.scale(ages=[0, 1], factor=0.9)]

        Callback hook (single parameter):

            @hook(event="first", priority=1)
            def release_males(pop):
                pop.state.individual_count[1, 0, 0] += 100
                return 0

        Selector callback:

            @hook(event="late", selectors={"target": "AA"})
            def count_homozygotes(pop, target):
                ...
    """
    def decorator(func: Callable[..., Any]) -> DecoratedHookFn:
        """Transform *func* into a ``DecoratedHookFn`` with ``.register(pop)``."""
        hook_func = cast(DecoratedHookFn, func)
        hook_func.meta = {
            "event": event,
            "selectors": selectors,
            "priority": priority,
            "deme_selector": deme,
        }
        hook_func.event = event
        hook_func.selectors = selectors or {}
        hook_func.priority = priority
        hook_func.deme_selector = deme

        def register(
            layout: HookLayout,
            event_override: Optional[str] = None,
            deme_selector_override: Optional[DemeSelector] = None,
        ) -> CompiledHookDescriptor:
            """Compile this hook against *layout* and return a descriptor.

            Compilation is pure: it resolves selectors against the given
            layout (a built population or the builder's build-time
            context) without installing anything.  The builder injects
            the returned descriptor into the population at ``build()``.

            Args:
                layout: Layout provider the selectors resolve against.
                event_override: Override the event name (used when the
                    declaration call supplies a different event than the
                    decorator).
                deme_selector_override: Override the deme selector (used
                    by the spatial build flow to pin hooks to demes).

            Returns:
                A ``CompiledHookDescriptor`` for the builder to inject.
            """
            actual_event = event_override or event
            actual_deme_selector: DemeSelector = (
                deme if deme_selector_override is None else deme_selector_override
            )
            if actual_event is None:
                raise ValueError(
                    f"Event not specified for hook '{func.__name__}'. "
                    "Specify in decorator @hook(event='...') or in the "
                    "declaration call .hooks(..., event='...')."
                )

            required = _count_required_parameters(func)
            if selectors is not None:
                desc = compile_selector_callback(
                    func,
                    layout,
                    actual_event,
                    selectors,
                    priority,
                    deme_selector=actual_deme_selector,
                )
            elif required == 0:
                # Declarative: called ONCE at compile time; its return value
                # (list of HookOp) is compiled into a CSR plan.
                result: object = func()
                items = list(cast("List[object]", result)) if isinstance(result, list) else []
                if not all(isinstance(op, HookOp) for op in items):
                    raise TypeError(
                        f"Declarative hook '{func.__name__}' must return "
                        "List[HookOp], or take one 'pop' parameter for the "
                        "callback style."
                    )
                ops = [op for op in items if isinstance(op, HookOp)]
                desc = compile_declarative_hook(
                    ops,
                    layout,
                    actual_event,
                    priority,
                    deme_selector=actual_deme_selector,
                    name=func.__name__,
                )
            elif required == 1:
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    deme_selector=actual_deme_selector,
                    callback=cast(Callable[[Any], int], func),
                    source=func,
                )
            else:
                raise TypeError(
                    f"Hook '{func.__name__}' has {required} required "
                    "parameters. The supported custom signature is "
                    "def hook(pop) -> int (a TickContext; return 0 to "
                    "continue, nonzero to stop). The legacy "
                    "(state, config, deme_id) form is no longer supported."
                )

            return desc

        hook_func.register = register  # type: ignore[assignment]  # set on DecoratedHookFn proxy
        return hook_func

    return decorator
