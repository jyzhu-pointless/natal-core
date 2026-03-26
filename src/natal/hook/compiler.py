"""Unified hook entrypoints and event-wise compiler.

This module connects three authoring styles into one runtime contract:

1) declarative hooks (Op list -> CompiledHookPlan)
2) selector hooks (symbolic selectors -> wrapper/compiled callable)
3) custom hooks (user-provided njit or Python callback)
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from natal.numba_utils import njit_switch

from .declarative import HookOp, compile_declarative_hook
from .selector import compile_selector_hook
from .types import (
    DemeSelector,
    EVENT_NAMES,
    HookProgram,
    _hash_key,
    _stable_callable_identity,
    _validate_numba_hook_required,
    _write_codegen_module,
    _load_codegen_module,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation
    from .types import CompiledHookDescriptor


# Internal alias used by generated wrapper modules.
_njit_switch = njit_switch


@_njit_switch(cache=True)
def _noop_hook(ind_count, tick):
    """Default hook implementation used for missing event handlers."""
    return 0


noop_hook = _noop_hook


def compile_combined_hook(njit_fns: List[Callable], name: str = "combined_hook") -> Callable:
    """Combine multiple njit hooks into one generated njit function.

    We generate source code instead of composing Python closures so the result
    remains callable from njit kernels.
    """
    if len(njit_fns) == 0:
        return _noop_hook
    if len(njit_fns) == 1:
        return njit_fns[0]

    # Stable key ensures deterministic module names and cache reuse.
    combined_parts = ["combined"] + [_stable_callable_identity(fn) for fn in njit_fns]
    key = _hash_key(combined_parts)
    fn_name = f"_combined_hook_{key}"
    module_stem = f"combined_hook_{key}"

    placeholder_names = [f"_FN_{i}" for i in range(len(njit_fns))]
    # Generated module imports the same switch alias as the rest of hook DSL.
    lines = ["from natal.hook_dsl import _njit_switch"]
    lines.extend([f"{placeholder} = None" for placeholder in placeholder_names])
    lines.extend(
        [
            "",
            "@_njit_switch(cache=True)",
            f"def {fn_name}(ind_count, tick):",
        ]
    )
    for placeholder in placeholder_names:
        lines.append(f"    _result = {placeholder}(ind_count, tick)")
        lines.append("    if _result != 0:")
        lines.append("        return _result")
    lines.append("    return 0")
    lines.append("")

    module_path = _write_codegen_module(module_stem, "\n".join(lines))
    module = _load_codegen_module(module_stem, module_path)
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
    )

    # Type annotations for attributes
    first: Callable
    early: Callable
    late: Callable
    finish: Callable
    registry: Optional[Any]
    _event_hooks: Dict[str, Callable]
    run_tick_fn: Optional[Callable]
    run_fn: Optional[Callable]
    run_discrete_tick_fn: Optional[Callable]
    run_discrete_fn: Optional[Callable]

    def __init__(self) -> None:
        self.first = _noop_hook
        self.early = _noop_hook
        self.late = _noop_hook
        self.finish = _noop_hook
        self.registry = None
        self._event_hooks = {name: _noop_hook for name in EVENT_NAMES}
        self.run_tick_fn = None
        self.run_fn = None
        self.run_discrete_tick_fn = None
        self.run_discrete_fn = None

    def get_hook(self, event_name: str) -> Callable:
        return self._event_hooks.get(event_name, _noop_hook)

    def set_hook(self, event_name: str, hook_fn: Callable) -> None:
        self._event_hooks[event_name] = hook_fn
        setattr(self, event_name, hook_fn)

    @staticmethod
    def from_compiled_hooks(compiled_hooks: List["CompiledHookDescriptor"], registry: Optional[HookProgram] = None):
        """Build event-wise combined callables from descriptors."""
        from .types import CompiledHookDescriptor
        from ..numba_utils import NUMBA_ENABLED

        if NUMBA_ENABLED:
            for desc in compiled_hooks:
                if desc.py_wrapper is not None:
                    raise TypeError(
                        f"Hook '{desc.name}' uses py_wrapper, which is not allowed when Numba is enabled."
                    )

        result = CompiledEventHooks()
        result.registry = registry

        hooks_by_event: Dict[str, List[Tuple[int, Callable]]] = {name: [] for name in EVENT_NAMES}
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
        ) = _compile_kernel_bound_wrappers(result.first, result.early, result.late)
        return result


def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    numba: bool = False,
    deme_selector: DemeSelector = "*",
):
    """Decorator entrypoint for all supported hook authoring styles.

    The decorated function gets a ``register(pop, event_override=None)``
    helper that compiles and registers a ``CompiledHookDescriptor``.
    """

    def decorator(func: Callable) -> Callable:
        # Store metadata for debugging / introspection / future recompilation.
        func._hook_meta = {  # type: ignore
            "event": event,
            "selectors": selectors or {},
            "priority": priority,
            "numba_mode": numba,
            "deme_selector": deme_selector,
        }
        func._hook_compiled = None  # type: ignore

        func._hook_event = event  # type: ignore
        func._hook_selectors_spec = selectors or {}  # type: ignore
        func._hook_priority = priority  # type: ignore
        func._hook_numba_mode = numba  # type: ignore
        func._hook_deme_selector = deme_selector  # type: ignore

        def register(
            pop: "BasePopulation",
            event_override: Optional[str] = None,
            deme_selector_override: Optional[DemeSelector] = None,
        ):
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
                _validate_numba_hook_required(func, func.__name__, "@hook(numba=True)")
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    deme_selector=actual_deme_selector,
                    njit_fn=func,
                    meta={"n_genotypes": pop._index_registry.num_genotypes(), "n_ages": pop._config.n_ages},
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
                        meta={"n_genotypes": pop._index_registry.num_genotypes(), "n_ages": pop._config.n_ages},
                    )
                else:
                    result = func()
                    if isinstance(result, list):
                        if not all(isinstance(op, HookOp) for op in result):
                            raise TypeError(
                                f"Declarative hook '{func.__name__}' must return List[HookOp], "
                                "or use function arguments for python hook mode."
                            )
                        desc = compile_declarative_hook(
                            result,
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
                        desc = CompiledHookDescriptor(
                            name=func.__name__,
                            event=actual_event,
                            priority=priority,
                            deme_selector=actual_deme_selector,
                            py_wrapper=lambda p, f=func: f(p),
                            meta={"n_genotypes": pop._index_registry.num_genotypes(), "n_ages": pop._config.n_ages},
                        )

            func._hook_compiled = desc  # type: ignore
            pop._register_compiled_hook(desc)
            return desc

        func.register = register  # type: ignore
        return func

    return decorator


def _has_required_parameters(func: Callable) -> bool:
    """Return whether calling ``func()`` would require positional/keyword args."""
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if param.default is inspect._empty:
                return True
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect._empty:
                return True
    return False


def _compile_kernel_bound_wrappers(first_fn: Callable, early_fn: Callable, late_fn: Callable):
    """Compile per-hook-set fixed-signature kernel wrappers.

    The generated wrappers bind hook chains as module globals and keep kernel
    signatures data-only, so no callable parameters are passed into njit ABI.
    """
    key = _hash_key(
        [
            "kernel_wrappers_v4",
            _stable_callable_identity(first_fn),
            _stable_callable_identity(early_fn),
            _stable_callable_identity(late_fn),
        ]
    )
    module_stem = f"kernel_wrappers_{key}"
    run_tick_name = f"_run_tick_bound_{key}"
    run_name = f"_run_bound_{key}"
    run_discrete_tick_name = f"_run_discrete_tick_bound_{key}"
    run_discrete_name = f"_run_discrete_bound_{key}"

    lines = [
        "import numpy as np",
        "from natal.hook_dsl import (_njit_switch, execute_csr_event_program_with_state, EVENT_FIRST, EVENT_EARLY, EVENT_LATE)",
        "from natal.population_state import PopulationState, DiscretePopulationState",
        "from natal.simulation_kernels import (run_reproduction, run_survival, run_aging, run_discrete_reproduction, run_discrete_survival, run_discrete_aging)",
        "",
        "_FIRST_HOOK = None",
        "_EARLY_HOOK = None",
        "_LATE_HOOK = None",
        "",
        "RESULT_CONTINUE = 0",
        "RESULT_STOP = 1",
        "",
        "@_njit_switch(cache=True)",
        "def _first_event(registry, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling):",
        "    result = execute_csr_event_program_with_state(registry, EVENT_FIRST, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return RESULT_STOP",
        "    return _FIRST_HOOK(ind_count, tick)",
        "",
        "@_njit_switch(cache=True)",
        "def _early_event(registry, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling):",
        "    result = execute_csr_event_program_with_state(registry, EVENT_EARLY, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return RESULT_STOP",
        "    return _EARLY_HOOK(ind_count, tick)",
        "",
        "@_njit_switch(cache=True)",
        "def _late_event(registry, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling):",
        "    result = execute_csr_event_program_with_state(registry, EVENT_LATE, ind_count, sperm_store, tick, is_stochastic, has_sperm_storage, use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return RESULT_STOP",
        "    return _LATE_HOOK(ind_count, tick)",
        "",
        "@_njit_switch(cache=True)",
        f"def {run_tick_name}(state, config, registry):",
        "    ind_count = state.individual_count.copy()",
        "    sperm_store = state.sperm_storage.copy()",
        "    tick = state.n_tick",
        "    result = _first_event(registry, ind_count, sperm_store, tick, config.is_stochastic, True, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, sperm_store, tick), RESULT_STOP",
        "    ind_count, sperm_store = run_reproduction(ind_count, sperm_store, config)",
        "    result = _early_event(registry, ind_count, sperm_store, tick, config.is_stochastic, True, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, sperm_store, tick), RESULT_STOP",
        "    ind_count, sperm_store = run_survival(ind_count, sperm_store, config)",
        "    result = _late_event(registry, ind_count, sperm_store, tick, config.is_stochastic, True, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, sperm_store, tick), RESULT_STOP",
        "    ind_count, sperm_store = run_aging(ind_count, sperm_store, config)",
        "    return (ind_count, sperm_store, np.int32(tick + 1)), RESULT_CONTINUE",
        "",
        "@_njit_switch(cache=True)",
        f"def {run_name}(state, config, registry, n_ticks, record_interval=0):",
        "    was_stopped = False",
        "    ind_count = state.individual_count.copy()",
        "    sperm_store = state.sperm_storage.copy()",
        "    tick = np.int32(state.n_tick)",
        "    ind_size = ind_count.size",
        "    sperm_size = sperm_store.size",
        "    flatten_size = 1 + ind_size + sperm_size",
        "    if record_interval > 0:",
        "        estimated_size = (n_ticks // record_interval) + 2",
        "        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)",
        "    else:",
        "        history_array = np.zeros((0, flatten_size), dtype=np.float64)",
        "    history_count = 0",
        "    if record_interval > 0 and (tick % record_interval == 0):",
        "        flat_state = np.zeros(flatten_size, dtype=np.float64)",
        "        flat_state[0] = tick",
        "        flat_state[1:1 + ind_size] = ind_count.flatten()",
        "        flat_state[1 + ind_size:] = sperm_store.flatten()",
        "        history_array[history_count, :] = flat_state",
        "        history_count += 1",
        "    for _ in range(n_ticks):",
        "        temp_state = PopulationState(n_tick=tick, individual_count=ind_count, sperm_storage=sperm_store)",
        f"        current_state, result = {run_tick_name}(temp_state, config, registry)",
        "        ind_count, sperm_store, tick = current_state",
        "        if record_interval > 0 and (tick % record_interval == 0):",
        "            flat_state = np.zeros(flatten_size, dtype=np.float64)",
        "            flat_state[0] = tick",
        "            flat_state[1:1 + ind_size] = ind_count.flatten()",
        "            flat_state[1 + ind_size:] = sperm_store.flatten()",
        "            history_array[history_count, :] = flat_state",
        "            history_count += 1",
        "        if result != RESULT_CONTINUE:",
        "            was_stopped = True",
        "            break",
        "    if record_interval > 0:",
        "        history_result = history_array[:history_count, :]",
        "    else:",
        "        history_result = None",
        "    return (ind_count, sperm_store, tick), history_result, was_stopped",
        "",
        "@_njit_switch(cache=True)",
        f"def {run_discrete_tick_name}(state, config, registry):",
        "    ind_count = state.individual_count.copy()",
        "    tick = state.n_tick",
        "    dummy_sperm_store = np.zeros((0, 0, 0), dtype=np.float64)",
        "    result = _first_event(registry, ind_count, dummy_sperm_store, tick, config.is_stochastic, False, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, tick), RESULT_STOP",
        "    ind_count = run_discrete_reproduction(ind_count, config)",
        "    result = _early_event(registry, ind_count, dummy_sperm_store, tick, config.is_stochastic, False, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, tick), RESULT_STOP",
        "    ind_count = run_discrete_survival(ind_count, config)",
        "    result = _late_event(registry, ind_count, dummy_sperm_store, tick, config.is_stochastic, False, config.use_dirichlet_sampling)",
        "    if result != RESULT_CONTINUE:",
        "        return (ind_count, tick), RESULT_STOP",
        "    ind_count = run_discrete_aging(ind_count)",
        "    return (ind_count, np.int32(tick + 1)), RESULT_CONTINUE",
        "",
        "@_njit_switch(cache=True)",
        f"def {run_discrete_name}(state, config, registry, n_ticks, record_interval=0):",
        "    was_stopped = False",
        "    ind_count = state.individual_count.copy()",
        "    tick = np.int32(state.n_tick)",
        "    ind_size = ind_count.size",
        "    flatten_size = 1 + ind_size",
        "    if record_interval > 0:",
        "        estimated_size = (n_ticks // record_interval) + 2",
        "        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)",
        "    else:",
        "        history_array = np.zeros((0, flatten_size), dtype=np.float64)",
        "    history_count = 0",
        "    if record_interval > 0 and (tick % record_interval == 0):",
        "        flat_state = np.zeros(flatten_size, dtype=np.float64)",
        "        flat_state[0] = tick",
        "        flat_state[1:1 + ind_size] = ind_count.flatten()",
        "        history_array[history_count, :] = flat_state",
        "        history_count += 1",
        "    for _ in range(n_ticks):",
        "        temp_state = DiscretePopulationState(n_tick=tick, individual_count=ind_count)",
        f"        current_state, result = {run_discrete_tick_name}(temp_state, config, registry)",
        "        ind_count, tick = current_state",
        "        if record_interval > 0 and (tick % record_interval == 0):",
        "            flat_state = np.zeros(flatten_size, dtype=np.float64)",
        "            flat_state[0] = tick",
        "            flat_state[1:1 + ind_size] = ind_count.flatten()",
        "            history_array[history_count, :] = flat_state",
        "            history_count += 1",
        "        if result != RESULT_CONTINUE:",
        "            was_stopped = True",
        "            break",
        "    if record_interval > 0:",
        "        history_result = history_array[:history_count, :]",
        "    else:",
        "        history_result = None",
        "    return (ind_count, tick), history_result, was_stopped",
        "",
    ]

    module_path = _write_codegen_module(module_stem, "\n".join(lines))
    module = _load_codegen_module(module_stem, module_path)
    setattr(module, "_FIRST_HOOK", first_fn)
    setattr(module, "_EARLY_HOOK", early_fn)
    setattr(module, "_LATE_HOOK", late_fn)
    return (
        getattr(module, run_tick_name),
        getattr(module, run_name),
        getattr(module, run_discrete_tick_name),
        getattr(module, run_discrete_name),
    )
