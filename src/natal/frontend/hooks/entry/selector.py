"""Selector-based hook compilation.

Selector mode lets hooks receive symbolic selector values as keyword
arguments, for example ``@hook(selectors={"target": "AA"})``.  Symbols are
resolved once at registration time into int32 index arrays; the compiled
descriptor carries a Python callback wrapper that injects the resolved
values after the :class:`TickContext` parameter.

Selector values follow the same resolution rules as declarative ops:
``"*"`` expands to all zygote types, genotype strings resolve through the
pattern parser, ``Genotype`` objects resolve through the registry, and
ints pass through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, Species
from natal.frontend.hooks.types import CompiledHookDescriptor, DemeSelector
from natal.frontend.patterns import resolve_zygote_type as _resolve_zygote_type
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.population.base import BasePopulation

SelectorItem: TypeAlias = Union[int, str, "Genotype"]
SelectorSpec: TypeAlias = Union[
    SelectorItem, range, List[SelectorItem], tuple[SelectorItem, ...]
]


def _resolve_selector_to_array(
    spec: SelectorSpec,
    index_registry: IndexRegistry,
    species: Species,
) -> NDArray[np.int32]:
    """Resolve one selector spec into an int32 index array.

    We normalize all accepted selector forms to a single representation so
    the rest of the compiler does not need type-dependent branches.
    """
    if isinstance(spec, bool):
        raise TypeError("selector specs must be genotype references or ints")
    if isinstance(spec, int):
        return np.array([spec], dtype=np.int32)

    if isinstance(spec, range):
        return np.array(list(spec), dtype=np.int32)

    if isinstance(spec, str):
        if spec == "*":
            return np.arange(index_registry.n_ztypes, dtype=np.int32)
        result = _resolve_zygote_type(spec, species, index_registry)
        if not result:
            raise ValueError(f"Cannot resolve genotype: {spec}")
        return np.array(result, dtype=np.int32)

    if isinstance(spec, (list, tuple)):
        indices: List[int] = []
        for item in spec:
            if isinstance(item, bool):
                raise TypeError("selector specs must be genotype references or ints")
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                result = _resolve_zygote_type(item, species, index_registry)
                if not result:
                    raise ValueError(f"Cannot resolve genotype: {item}")
                indices.extend(result)
            else:
                # Genotype object
                result = index_registry.ztype_indices_for(item)
                if not result:
                    raise ValueError(f"Cannot resolve selector item: {item}")
                indices.extend(result)
        return np.array(indices, dtype=np.int32)

    # spec is a Genotype object
    result = index_registry.ztype_indices_for(spec)
    if not result:
        raise ValueError(f"Cannot resolve selector spec: {spec}")
    return np.array(result, dtype=np.int32)


def _runtime_selector_value(
    indices: NDArray[np.int32],
) -> int | NDArray[np.int32]:
    """Collapse single-element selector arrays to plain ints."""
    if len(indices) == 1:
        return int(indices[0])
    return np.array(indices, dtype=np.int32)


def compile_selector_callback(
    func: Callable[..., Any],
    pop: BasePopulation[Any],
    event: str,
    selectors_spec: Dict[str, SelectorSpec],
    priority: int = 0,
    deme_selector: DemeSelector = "*",
) -> CompiledHookDescriptor:
    """Compile a selector hook into a Python-callback descriptor.

    The user function has the shape ``def hook(pop, **selectors)``; the
    wrapper injects the resolved selector values as keyword arguments
    (single-index selectors collapse to ints, multi-index selectors pass
    as int32 arrays).

    Args:
        func: The decorated user function.
        pop: The population to compile against.
        event: Resolved event name.
        selectors_spec: Symbolic selector specs.
        priority: Execution priority.
        deme_selector: Deme selector for spatial filtering.

    Returns:
        A ``CompiledHookDescriptor`` carrying the injected callback.
    """
    index_registry = pop.index_registry
    species = pop.species

    resolved = {
        name: _resolve_selector_to_array(spec, index_registry, species)
        for name, spec in selectors_spec.items()
    }
    runtime_values = {
        name: _runtime_selector_value(indices)
        for name, indices in resolved.items()
    }

    def injected(context: Any) -> int:
        """Call the user function with resolved selector kwargs."""
        result = func(context, **runtime_values)
        if result is None:
            return 0
        return int(result)

    return CompiledHookDescriptor(
        name=func.__name__,
        event=event,
        priority=priority,
        deme_selector=deme_selector,
        callback=injected,
        selectors=resolved,
        meta={
            "n_ztypes": index_registry.n_ztypes,
            "n_ages": pop.config.n_ages,
        },
        source=func,
    )
