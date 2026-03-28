"""Selector-based hook compilation.

Selector mode allows users to write hooks with symbolic selector arguments,
for example ``selectors={"target_gt": "AA"}``. This module resolves those
symbols once at registration time and then provides two execution paths:

1) Python wrapper path (``py_wrapper(pop, **resolved_selectors)``)
2) Numba path (generated ``njit_fn(ind_count, tick)`` with baked literals)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

import numpy as np
import inspect

from .types import (
    DemeSelector,
    _hash_key,
    _is_numba_dispatcher,
    _load_codegen_module,
    _stable_callable_identity,
    _validate_numba_hook_required,
    _write_codegen_module,
    CompiledHookDescriptor,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation


def _resolve_selector_to_array(spec: Any, index_registry, diploid_genotypes: List[Any]) -> np.ndarray:
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
            return np.arange(len(diploid_genotypes), dtype=np.int32)
        idx = index_registry.resolve_genotype_index(diploid_genotypes, spec, strict=True)
        if idx is None:
            raise ValueError(f"Cannot resolve genotype: {spec}")
        return np.array([idx], dtype=np.int32)

    if isinstance(spec, (list, tuple)):
        indices = []
        for item in spec:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                idx = index_registry.resolve_genotype_index(diploid_genotypes, item, strict=True)
                if idx is None:
                    raise ValueError(f"Cannot resolve genotype: {item}")
                indices.append(idx)
            else:
                idx = index_registry.genotype_to_index.get(item)
                if idx is None:
                    raise ValueError(f"Cannot resolve selector item: {item}")
                indices.append(idx)
        return np.array(indices, dtype=np.int32)

    idx = index_registry.genotype_to_index.get(spec)
    if idx is not None:
        return np.array([idx], dtype=np.int32)

    raise ValueError(f"Cannot resolve selector spec: {spec}")


def compile_selector_hook(
    func: Callable[..., Any],
    pop: "BasePopulation",
    event: str,
    selectors_spec: Dict[str, Any],
    priority: int = 0,
    numba_mode: bool = False,
    deme_selector: DemeSelector = "*",
) -> CompiledHookDescriptor:
    """Compile selector hook into njit or python descriptor.

    ``resolved`` stores canonical selector arrays and is reused by both
    execution paths.
    """
    index_registry = getattr(pop, "_index_registry", None)
    if index_registry is None:
        index_registry = getattr(pop, "_index_core", None)
    if index_registry is None:
        raise AttributeError("Population must expose _index_registry (or legacy _index_core)")
    diploid_genotypes = index_registry.index_to_genotype

    # Resolve selectors exactly once during registration.
    resolved = {
        name: _resolve_selector_to_array(spec, index_registry, diploid_genotypes)
        for name, spec in selectors_spec.items()
    }

    meta = {
        "n_genotypes": index_registry.num_genotypes(),
        "n_ages": pop._config.n_ages,
    }

    from ..numba_utils import NUMBA_ENABLED

    is_njit_fn = _is_numba_dispatcher(func)

    if NUMBA_ENABLED and not (numba_mode or is_njit_fn):
        raise TypeError(
            f"Selector hook '{func.__name__}' must be an @njit function when Numba is enabled. "
            "Either decorate with @njit or disable Numba explicitly."
        )

    if numba_mode or is_njit_fn:
        # Numba path: generate a thin wrapper with literal selector args.
        if numba_mode:
            _validate_numba_hook_required(func, func.__name__, "selector numba_mode=True")

        # Handle signature normalization for user function (2 or 3 args before selectors)
        py_func = getattr(func, "py_func", func)
        sig = inspect.signature(py_func)
        # Check if user fn expects deme_id (3 positional args before kwargs)
        has_deme_id = len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 3
        
        njit_fn = _compile_selector_njit_wrapper(func, resolved, has_deme_id)
        return CompiledHookDescriptor(
            name=func.__name__,
            event=event,
            priority=priority,
            deme_selector=deme_selector,
            selectors=resolved,
            meta=meta,
            njit_fn=njit_fn,
        )

    # Python path: pass scalar for length-1 selectors, full array otherwise.
    def py_wrapper(population):
        kwargs = _build_selector_python_kwargs(resolved)
        func(population, **kwargs)

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
    resolved_selectors: Dict[str, np.ndarray],
    has_deme_id: bool,
) -> Callable[..., Any]:
    """Generate a Numba wrapper with selector constants baked in.

    The generated module imports ``_njit_switch`` from ``hook_dsl`` so wrapper
    compilation respects the same global Numba switch and cache configuration.
    """
    args_str = _build_selector_njit_literal_args(resolved_selectors)

    # Build deterministic identity key so repeated registrations reuse the same
    # generated module file and compiled cache entry.
    selector_parts = ["selector", _stable_callable_identity(user_fn)]
    for key in sorted(resolved_selectors.keys()):
        values = ",".join(str(int(v)) for v in resolved_selectors[key].tolist())
        selector_parts.append(f"{key}={values}")

    key = _hash_key(selector_parts)
    fn_name = f"_selector_wrapper_{key}"
    module_stem = f"selector_wrapper_{key}"

    selector_placeholders = [f"_SEL_{name}" for name in resolved_selectors.keys()]
    code_lines = [
        "from natal.hook_dsl import _njit_switch",
        "_USER_FN = None",
    ]
    code_lines.extend([f"{placeholder} = None" for placeholder in selector_placeholders])
    
    call_args = "ind_count, tick, deme_id" if has_deme_id else "ind_count, tick"
    
    code_lines.extend(
        [
            "",
            "@_njit_switch(cache=True)",
            f"def {fn_name}(ind_count, tick, deme_id=0):",
            f"    return _USER_FN({call_args}, {args_str})",
            "",
        ]
    )
    code = "\n".join(code_lines)

    module_path = _write_codegen_module(module_stem, code)
    module = _load_codegen_module(module_stem, module_path)
    setattr(module, "_USER_FN", user_fn)
    for name, value in _build_selector_njit_runtime_values(resolved_selectors).items():
        setattr(module, f"_SEL_{name}", value)
    return getattr(module, fn_name)


def _build_selector_python_kwargs(resolved_selectors: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert internal selector arrays into user-facing kwargs."""
    kwargs: Dict[str, Any] = {}
    for key, values in resolved_selectors.items():
        kwargs[key] = int(values[0]) if len(values) == 1 else values
    return kwargs


def _build_selector_njit_literal_args(resolved_selectors: Dict[str, np.ndarray]) -> str:
    """Build keyword argument list for generated njit wrapper source code.

    Args are bound to generated module-level globals (``_SEL_<name>``) so we
    can pass either:
    - ``int`` for single-value selectors
    - ``np.ndarray[int32]`` for multi-value selectors
    """
    arg_lines = []
    for name in resolved_selectors.keys():
        arg_lines.append(f"{name}=_SEL_{name}")
    return ", ".join(arg_lines)


def _build_selector_njit_runtime_values(resolved_selectors: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Build runtime values injected into generated selector wrapper module."""
    values: Dict[str, Any] = {}
    for name, indices in resolved_selectors.items():
        if len(indices) == 1:
            values[name] = int(indices[0])
        else:
            values[name] = np.array(indices, dtype=np.int32)
    return values
