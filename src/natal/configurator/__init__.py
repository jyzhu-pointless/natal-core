"""Configurator subpackage — chainable PopulationConfig builders.

Re-exports all public symbols from both the new Configurator API
and the legacy population_builder classes.
"""

from natal.configurator._base import (
    Configurator,
    hook_set_param,
    merge_hooks,
    set_param,
)
from natal.configurator._factory import (
    AgeStructuredPopulationBuilder,
    DiscreteGenerationPopulationBuilder,
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
