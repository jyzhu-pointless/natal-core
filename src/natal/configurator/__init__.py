"""Configurator subpackage — chainable PopulationConfig builders.

Provides two API layers for constructing and modifying
``PopulationConfig`` / ``DiscretePopulationConfig``:

1. **New Configurator API** (:class:`Configurator`, ``AgeStructuredConfigurator``,
   ``DiscreteConfigurator``) — chainable domain methods
   (``.competition()``, ``.reproduction()``) that mutate config arrays
   in-place.  Created via ``Configurator.from_species()`` or bound to a
   running simulation via ``for_population()`` for runtime changes.

2. **Legacy Builder API** (``AgeStructuredPopulationBuilder``,
   ``DiscreteGenerationPopulationBuilder``) — deprecated class-based
   builders that construct a ``PopulationConfig`` from scratch and pass
   it to the Population constructor.

Utility symbols:
  - ``set_param`` / ``hook_set_param`` — write a scalar parameter by
    name, usable from pure Python or Numba hooks.
  - ``merge_hooks`` — combine @hook-decorated items into a single map.

Re-exports all public symbols from both API layers.
"""

from natal.configurator._base import (
    Configurator,
    hook_set_param,
    merge_hooks,
    set_param,
)
from natal.configurator._builder_age import AgeStructuredPopulationBuilder
from natal.configurator._builder_discrete import DiscreteGenerationPopulationBuilder
from natal.configurator._factory import (
    PopulationConfigBuilder,
)
from natal.configurator.age_structured import AgeStructuredConfigurator
from natal.configurator.discrete import DiscreteConfigurator

__all__ = [
    "AgeStructuredConfigurator",
    "AgeStructuredPopulationBuilder",
    "Configurator",
    "DiscreteConfigurator",
    "DiscreteGenerationPopulationBuilder",
    "PopulationConfigBuilder",
    "hook_set_param",
    "merge_hooks",
    "set_param",
]
