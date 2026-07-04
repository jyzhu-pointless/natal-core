"""Fitness system — fitness patch construction, application, and DSL writing.

This subpackage hosts fitness logic extracted from presets and
configurator modules:

- ``fitness/_patch.py``: core fitness patch application (allele scaling, slab
  scaling, selector-based writes).  Uses ``FitnessPopulationView`` protocol.
- ``fitness/_writer.py``: Configurator DSL writer — resolves genotype-pattern
  selectors to ztype indices and writes to config arrays.
"""

from typing import TYPE_CHECKING

__all__ = [
    "apply_preset_fitness_patch",
    "write_fitness_field",
]

if TYPE_CHECKING:
    from natal.fitness._patch import apply_preset_fitness_patch  # noqa: F401
    from natal.fitness._writer import write_fitness_field  # noqa: F401


def __getattr__(name: str) -> object:
    """Lazy-load public symbols to avoid circular imports at package-init time."""
    if name == "apply_preset_fitness_patch":
        from natal.fitness._patch import apply_preset_fitness_patch as _fn
        return _fn
    if name == "write_fitness_field":
        from natal.fitness._writer import write_fitness_field as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
