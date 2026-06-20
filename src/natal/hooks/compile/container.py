"""Compiled event-hook container — holds per-event callables and CSR registry.

``CompiledEventHooks`` is the **product** of the hook compilation pipeline.
It stores one combined callable per event (``first`` / ``early`` / ``late`` /
``finish``) plus a ``registry`` (``HookProgram``) for CSR dispatch.  This is
a pure container with no compilation or code-generation logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from natal.numba_utils import njit_switch

from ..types import EVENT_NAMES, HookCallable


@njit_switch(cache=True)
def _noop_hook(state: Any, config: Any = None, deme_id: int = -1) -> int:
    """Default no-op hook: ``(state, config, deme_id) -> 0``.

    Used as the fallback when no hooks are registered for an event.
    """
    return 0


noop_hook = _noop_hook


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
    registry: Optional[Any]  # HookProgram
    _event_hooks: Dict[str, HookCallable]

    def __init__(self) -> None:
        """Initialise all event hooks to no-op and registry to None."""
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
