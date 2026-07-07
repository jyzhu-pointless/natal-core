"""Configurator subpackage — chainable PopulationConfig builders.

Provides the Configurator API for constructing and modifying
``PopulationConfig`` / ``DiscretePopulationConfig``:

- :class:`Configurator`, ``AgeStructuredConfigurator``,
  ``DiscreteConfigurator`` — chainable domain methods
  (``.competition()``, ``.reproduction()``) that mutate config arrays
  in-place.  Created via ``Configurator.from_species()`` or bound to a
  running simulation via ``for_population()`` for runtime changes.

Utility symbols:
  - ``set_param`` / ``hook_set_param`` — write a scalar parameter by
    name, usable from pure Python or Numba hooks.
  - ``merge_hooks`` — combine @hook-decorated items into a single map.
"""

from natal.configurator._base import (
    Configurator,
    hook_set_param,
    merge_hooks,
    set_param,
)
from natal.configurator._factory import (
    PopulationConfigBuilder,
)
from natal.configurator.age_structured import AgeStructuredConfigurator
from natal.configurator.discrete import DiscreteConfigurator

__all__ = [
    "AgeStructuredConfigurator",
    "Configurator",
    "DiscreteConfigurator",
    "PopulationConfigBuilder",
    "hook_set_param",
    "merge_hooks",
    "set_param",
]
