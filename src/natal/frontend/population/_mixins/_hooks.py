"""Hook dispatch mixin for BasePopulation.

The hook plan is compiled once by the builder and injected at
construction: the descriptor tuple and the packed CSR program are fixed
for the population's lifetime.  This mixin owns only the execution and
introspection side — native Rust sessions execute declarative plans and
bridge Python callbacks at event boundaries in one stable priority order.

The post-construction registration surface (``register_hooks`` /
``register_compiled_hook``, identity-idempotent re-registration, and the
``_run_program`` rebind chain) was removed with the build-time hook
injection; there is no runtime registration channel left.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    cast,
)

from natal.frontend.hooks.types import (
    EVENT_ID_MAP,
    RESULT_CONTINUE,
    CompiledHookDescriptor,
)

if TYPE_CHECKING:
    from typing import Any as _Any
    from typing import Protocol as _Protocol

    from natal.frontend.hooks.tick_context import HookRunner
    from natal.frontend.hooks.types import HookProgram
    from natal.frontend.population.base import BasePopulation

    # Host-contract alias: the mixin is only mixed into BasePopulation
    # subclasses; the ``Any`` type argument stands for the host's
    # ``T_State`` (a documented host-contract ``Any``).
    _Population = BasePopulation[_Any]

    class _CallbackBridge(_Protocol):
        """Structural type of the Rust backend adapter's callback channel."""

        def set_python_callbacks(
            self,
            first: List[Callable[..., int]],
            early: List[Callable[..., int]],
            late: List[Callable[..., int]],
            finish: List[Callable[..., int]] | None = None,
        ) -> None:
            """Register per-event callback lists."""
            ...


class HookManagerMixin:
    """Mixin providing hook dispatch and introspection.

    Expects the host class (BasePopulation) to define these attributes:
    ``ALLOWED_EVENTS``, ``tick``, ``state``, ``config``,
    ``compiled_hook_descriptors``, ``_hook_program``, ``_hook_runner``,
    and the ``_rust_needs_rebuild`` flag.
    """

    # Declared here so pyright knows these come from the host class.
    ALLOWED_EVENTS: list[str]  # type: ignore[assignment]
    tick: int  # type: ignore[assignment]
    # ``object`` is a host-contract declaration only: BasePopulation owns the
    # typed ``state``/``config`` properties; this mixin just narrows through
    # ``cast`` at the call site.
    state: object
    config: object
    _hook_runner: Optional[HookRunner]
    _hook_program: HookProgram
    # Rust dirty-set bridge owned by BasePopulation (contract field names).
    _rust_needs_rebuild: bool

    @property
    def compiled_hook_descriptors(self) -> tuple[CompiledHookDescriptor, ...]:
        """Read-only descriptor query; the host class provides the body."""
        raise NotImplementedError

    # ── Dispatch ──────────────────────────────────────────────────────

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Trigger an event and execute the hooks declared for it.

        Execution order per event: CSR declarative plans and Python
        callbacks interleaved by one stable ascending-priority order
        (each callback receives a fresh
        :class:`~natal.frontend.hooks.tick_context.TickContext`).

        Args:
            event_name: Event name to trigger.
            deme_id: Deme index the hooks execute as.  0 for panmictic
                populations; the live deme index when the population is
                managed by a SpatialPopulation.

        Returns:
            int: ``RESULT_CONTINUE`` (0) to continue, ``RESULT_STOP`` (1)
            to stop.
        """
        native = getattr(self, "_runtime_parameter_writer", None)
        if native is None:
            native = getattr(self, "_rust_lifecycle_backend", None)
        if native is None:
            # Directly constructed standalone populations do not create a
            # session until the first operation that needs native execution.
            # Managed spatial demes have a runtime writer and therefore stay
            # owned by their SpatialPopulation container.
            initialize = getattr(self, "_initialize_session", None)
            if callable(initialize):
                initialize(seed=int(getattr(self, "_rust_backend_seed", 0) or 0))
                native = getattr(self, "_rust_lifecycle_backend", None)
        if native is not None and hasattr(native, "trigger_event"):
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is None:
                return RESULT_CONTINUE
            # Refresh the installed program without replacing session
            # state/RNG (manual events use the same native log as run
            # checkpoints).
            if getattr(self, "_runtime_parameter_writer", None) is None:
                pop = cast("_Population", self)
                native.bind_history(pop.history._store, pop._params_log)  # pyright: ignore[reportPrivateUsage]  # manual events use the same native log as run checkpoints.
                pop.history._bind_checkpoint_pruner(native.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.
                native.configure_program(self._hook_program, self.config)
                self._register_rust_callbacks(native)
            if getattr(self, "_runtime_parameter_writer", None) is None:
                result = int(native.trigger_event(event_id, deme_id))
            else:
                result = int(native.trigger_event(event_id))
            cast("_Population", self)._mark_state_cache_stale()  # pyright: ignore[reportPrivateUsage]  # host owns its snapshot cache
            return result
        raise RuntimeError(
            "Native hook execution is unavailable; initialize a Rust session "
            "before triggering events."
        )

    # ── Introspection ─────────────────────────────────────────────────

    def get_compiled_hooks(
        self, event: Optional[str] = None
    ) -> List[CompiledHookDescriptor]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of ``CompiledHookDescriptor`` sorted by priority.
        """
        hooks = list(self.compiled_hook_descriptors)
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def has_python_callbacks(self) -> bool:
        """Return whether the injected plan contains a Python callback."""
        return any(desc.callback is not None for desc in self.compiled_hook_descriptors)

    def has_python_hooks(self) -> bool:
        """Back-compatible alias for :meth:`has_python_callbacks`."""
        return self.has_python_callbacks()

    def _ensure_hook_runner(self) -> HookRunner:
        """Return the callback runner, building it on first use."""
        if self._hook_runner is None:
            from natal.frontend.hooks.tick_context import HookRunner

            self._hook_runner = HookRunner(cast("_Population", self))
        runner: HookRunner = self._hook_runner
        return runner

    def _register_rust_callbacks(self, backend: _CallbackBridge) -> None:
        """Bridge Python callbacks into a Rust session.

        One adapter per in-tick event (first/early/late); each adapter has
        the Rust ``(ind, sperm, tick, deme_id) -> int`` signature and runs
        every callback of its event in priority order.  Events without
        callbacks register an empty list so Rust kernels skip the GIL
        boundary entirely.
        """
        from natal.frontend.hooks.types import (
            EVENT_EARLY,
            EVENT_FINISH,
            EVENT_FIRST,
            EVENT_LATE,
        )

        runner = self._ensure_hook_runner()
        backend.set_python_callbacks(
            runner.rust_callbacks(EVENT_FIRST),
            runner.rust_callbacks(EVENT_EARLY),
            runner.rust_callbacks(EVENT_LATE),
            runner.rust_callbacks(EVENT_FINISH),
        )
