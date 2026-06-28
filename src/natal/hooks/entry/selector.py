"""Selector-based hook compilation.

Selector mode allows users to write hooks with symbolic selector arguments,
for example ``selectors={"target_gt": "AA"}``. This module resolves those
symbols once at registration time and then provides two execution paths:

1) Python wrapper path (``py_wrapper(state, config, deme_id, **resolved_selectors)``)
2) Numba path (generated ``njit_fn(state, config, deme_id)`` with baked literals)
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from ..types import (
    CompiledHookDescriptor,
    DemeSelector,
    hash_key,
    is_numba_dispatcher,
    load_codegen_module,
    stable_callable_identity,
    write_codegen_module,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _read_template(name: str) -> str:
    """Read a hook codegen template from ``hooks/templates/``."""
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")

if TYPE_CHECKING:
    from natal.base_population import BasePopulation
    from natal.genetic_entities import Genotype
    from natal.index_registry import IndexRegistry
    from natal.population_config import PopulationConfig
    from natal.population_state import PopulationState



SelectorItem: TypeAlias = Union[int, str, "Genotype"]
SelectorSpec: TypeAlias = Union[SelectorItem, range, List[SelectorItem], tuple[SelectorItem, ...]]


def _resolve_selector_to_array(
    spec: SelectorSpec,
    index_registry: IndexRegistry,
    diploid_genotypes: Sequence[Genotype],
) -> NDArray[np.int32]:
    """Resolve one selector spec into an int32 index array.

    We normalize all accepted selector forms to a single representation so the
    rest of the compiler does not need type-dependent branches.
    """
    if isinstance(spec, int):
        return np.array([spec], dtype=np.int32)

    if isinstance(spec, range):
        return np.array(list(spec), dtype=np.int32)

    if isinstance(spec, str):
        if spec == "*":
            return np.arange(index_registry.n_ztypes, dtype=np.int32)
        idx = index_registry.resolve_genotype_index(diploid_genotypes, spec, strict=True)
        if idx is None:
            raise ValueError(f"Cannot resolve genotype: {spec}")
        return np.array(index_registry.genotype_to_ztype_indices(idx), dtype=np.int32)

    if isinstance(spec, (list, tuple)):
        indices: List[int] = []
        for item in spec:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                idx = index_registry.resolve_genotype_index(diploid_genotypes, item, strict=True)
                if idx is None:
                    raise ValueError(f"Cannot resolve genotype: {item}")
                indices.extend(index_registry.genotype_to_ztype_indices(idx))
            else:
                # Genotype or other object
                idx = index_registry.genotype_to_index.get(item)
                if idx is None:
                    raise ValueError(f"Cannot resolve selector item: {item}")
                indices.extend(index_registry.genotype_to_ztype_indices(idx))
        return np.array(indices, dtype=np.int32)

    idx = index_registry.genotype_to_index.get(spec)
    if idx is not None:
        return np.array(index_registry.genotype_to_ztype_indices(idx), dtype=np.int32)

    raise ValueError(f"Cannot resolve selector spec: {spec}")


def compile_selector_hook(
    func: Callable[..., Any],
    pop: BasePopulation[Any],
    event: str,
    selectors_spec: Dict[str, SelectorSpec],
    priority: int = 0,
    deme_selector: DemeSelector = "*",
    mode: str = "auto",
) -> CompiledHookDescriptor:
    """Compile selector hook into njit or python descriptor.

    ``resolved`` stores canonical selector arrays and is reused by both
    execution paths.

    For selector hooks, Numba compilation depends on:
    - If function is @njit decorated, use it directly
    - Otherwise, use global NUMBA_ENABLED setting (auto-wrap if enabled)
    """
    index_registry = pop.registry
    diploid_genotypes = index_registry.index_to_genotype

    resolved = {
        name: _resolve_selector_to_array(spec, index_registry, diploid_genotypes)
        for name, spec in selectors_spec.items()
    }

    meta = {
        "n_ztypes": index_registry.n_ztypes,
        "n_ages": pop.config.n_ages,
    }

    from ...numba_utils import NUMBA_ENABLED

    is_njit_fn = is_numba_dispatcher(func)

    # Detect parameter layout based on *mode*.
    py_func = getattr(func, "py_func", func)
    sig = inspect.signature(py_func)
    pos_params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    selector_key_set = set(resolved.keys())

    extra_params = [p for p in pos_params[2:] if p.name != "deme_id"]
    has_deme_id = any(p.name == "deme_id" for p in pos_params[2:])

    if mode == "aggregate":
        use_namedtuple = True
        nt_param_name = extra_params[0].name if extra_params else "sel"
    elif mode == "expand":
        use_namedtuple = False
        nt_param_name = "sel"
    else:  # "auto"
        use_namedtuple = (
            len(extra_params) == 1
            and extra_params[0].name not in selector_key_set
        )
        nt_param_name = extra_params[0].name if use_namedtuple else "sel"

    # Determine state type for the generated wrapper's type annotation.
    from natal.discrete_generation_population import DiscreteGenerationPopulation
    if isinstance(pop, DiscreteGenerationPopulation):
        state_type_name = "DiscretePopulationState"
    else:
        state_type_name = "PopulationState"

    if is_njit_fn or NUMBA_ENABLED:
        njit_fn = _compile_selector_njit_wrapper(
            func, resolved, has_deme_id, state_type_name, use_namedtuple, nt_param_name,
        )
        return CompiledHookDescriptor(
            name=func.__name__,
            event=event,
            priority=priority,
            deme_selector=deme_selector,
            selectors=resolved,
            meta=meta,
            njit_fn=njit_fn,
        )

    # Python path — generate a 3-param wrapper (state, config, deme_id)
    # matching the current hook calling convention.  HookExecutor.execute_event
    # detects the parameter count and dispatches accordingly.
    if use_namedtuple:
        _Sel = _build_namedtuple_class(list(resolved.keys()))
        runtime_values = _build_selector_njit_runtime_values(resolved)

        def py_wrapper(state: PopulationState, config: PopulationConfig | None = None, deme_id: int = -1) -> None:
            nt = _Sel(**runtime_values)
            # Pass namedtuple as keyword arg so user's parameter name
            # (nt_param_name) matches the kwarg.
            if has_deme_id:
                func(state, config, deme_id, **{nt_param_name: nt})
            else:
                func(state, config, **{nt_param_name: nt})
    else:
        def py_wrapper(state: PopulationState, config: PopulationConfig | None = None, deme_id: int = -1) -> None:
            kwargs = _build_selector_python_kwargs(resolved)
            if has_deme_id:
                func(state, config, deme_id, **kwargs)
            else:
                func(state, config, **kwargs)

    return CompiledHookDescriptor(
        name=func.__name__,
        event=event,
        priority=priority,
        deme_selector=deme_selector,
        selectors=resolved,
        meta=meta,
        py_wrapper=py_wrapper,
    )


def _compile_selector_njit_wrapper(
    user_fn: Callable[..., Any],
    resolved_selectors: Dict[str, NDArray[np.int32]],
    has_deme_id: bool,
    state_type_name: str,
    use_namedtuple: bool = False,
    nt_param_name: str = "sel",
) -> Callable[..., Any]:
    """Generate a Numba wrapper with selector constants baked in.

    The wrapper accepts ``(state, config, deme_id)`` — the current hook
    calling convention — and forwards ``state``, ``config``, ``deme_id``
    plus the baked-in selector values to *user_fn*.

    When ``use_namedtuple`` is True, packs all selectors into a single
    ``namedtuple`` keyword argument::

        sel = _Sel(_SEL_target, _SEL_drive)
        return _USER_FN(state, config, deme_id, sel=sel)

    Otherwise the original per-selector keyword-arg style is used.
    """
    selector_keys = list(resolved_selectors.keys())

    # Deterministic identity key for caching — must include state type
    # because different state types produce different import lines.
    selector_parts = ["selector", stable_callable_identity(user_fn), state_type_name]
    if use_namedtuple:
        selector_parts.append("namedtuple")
    for key in sorted(selector_keys):
        values = ",".join(str(int(v)) for v in resolved_selectors[key].tolist())
        selector_parts.append(f"{key}={values}")

    key = hash_key(selector_parts)
    fn_name = f"_selector_wrapper_{key}"
    module_stem = f"selector_wrapper_{key}"

    # ---- Build replacement strings for template ----
    sel_globals_lines = [f"_SEL_{name}: object = None  # type: ignore[assignment]" for name in selector_keys]
    sel_globals_str = "\n".join(sel_globals_lines) if sel_globals_lines else ""

    if has_deme_id:
        pos_args = "state, config, deme_id"
    else:
        pos_args = "state, config"

    if use_namedtuple:
        nt_fields = ", ".join(f"'{k}'" for k in selector_keys)
        namedtuple_import = "from collections import namedtuple"
        namedtuple_def = f"_Sel = namedtuple('_Sel', [{nt_fields}])"
        nt_args = ", ".join(f"_SEL_{k}" for k in selector_keys)
        wrapper_body = (
            f"    {nt_param_name} = _Sel({nt_args})\n"
            f"    return _USER_FN({pos_args}, {nt_param_name}={nt_param_name})"
        )
    else:
        namedtuple_import = ""
        namedtuple_def = ""
        args_str = _build_selector_njit_literal_args(resolved_selectors)
        wrapper_body = f"    return _USER_FN({pos_args}, {args_str})"

    # ---- Assemble module from template ----
    template = _read_template("selector_wrapper.tmpl.py")
    source = (
        template.replace("# PLACEHOLDER_NAMEDTUPLE_IMPORT", namedtuple_import)
        .replace("# PLACEHOLDER_NAMEDTUPLE_DEF", namedtuple_def)
        .replace("# PLACEHOLDER_SEL_GLOBALS", sel_globals_str)
        .replace("PLACEHOLDER_FN_NAME", fn_name)
        .replace("# PLACEHOLDER_WRAPPER_BODY", wrapper_body)
    )

    module_path = write_codegen_module(module_stem, source)
    module = load_codegen_module(module_stem, module_path)
    module._USER_FN = user_fn  # type: ignore[assignment]
    for name, value in _build_selector_njit_runtime_values(resolved_selectors).items():
        setattr(module, f"_SEL_{name}", value)  # type: ignore[assignment]
    return getattr(module, fn_name)


def _build_namedtuple_class(field_names: list[str]) -> type:  # type: ignore[return]
    """Build a namedtuple class from selector field names (Python path only)."""
    from collections import namedtuple
    return namedtuple("_Sel", field_names)  # type: ignore[return-value]


def _build_selector_python_kwargs(resolved_selectors: Dict[str, NDArray[np.int32]]) -> Dict[str, int | NDArray[np.int32]]:
    """Convert internal selector arrays into user-facing kwargs."""
    kwargs: Dict[str, int | NDArray[np.int32]] = {}
    for key, values in resolved_selectors.items():
        kwargs[key] = int(values[0]) if len(values) == 1 else values
    return kwargs


def _build_selector_njit_literal_args(resolved_selectors: Dict[str, NDArray[np.int32]]) -> str:
    """Build keyword argument list for generated njit wrapper source code.

    Args are bound to generated module-level globals (``_SEL_<name>``) so we
    can pass either:
    - ``int`` for single-value selectors
    - ``np.ndarray[int32]`` for multi-value selectors
    """
    arg_lines: List[str] = []
    for name in resolved_selectors.keys():
        arg_lines.append(f"{name}=_SEL_{name}")
    return ", ".join(arg_lines)


def _build_selector_njit_runtime_values(resolved_selectors: Dict[str, NDArray[np.int32]]) -> Dict[str, int | NDArray[np.int32]]:
    """Build runtime values injected into generated selector wrapper module."""
    values: Dict[str, int | NDArray[np.int32]] = {}
    for name, indices in resolved_selectors.items():
        if len(indices) == 1:
            values[name] = int(indices[0])
        else:
            values[name] = np.array(indices, dtype=np.int32)
    return values
