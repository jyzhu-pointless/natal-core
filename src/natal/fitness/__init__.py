"""Fitness system — fitness patch construction and application.

This subpackage hosts fitness logic extracted from presets and
configurator modules, centered around ``FitnessPopulationView``
(a protocol satisfied by both ``BasePopulation`` and ``ConfigContext``).
"""

from natal.fitness._patch import apply_preset_fitness_patch

__all__ = [
    "apply_preset_fitness_patch",
]
