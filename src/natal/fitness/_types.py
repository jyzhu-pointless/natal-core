"""Shared types and protocols for the fitness system.

Private module — not part of the public API.
"""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from natal.data.config import DiscretePopulationConfig, PopulationConfig
    from natal.genetics.structures.species import Species
    from natal.registry.index import IndexRegistry


class FitnessPopulationView(Protocol):
    """Minimal view of a population needed for fitness operations.

    Both ``BasePopulation`` and ``ConfigContext`` (configurator adapter)
    satisfy this protocol — no actual population object required.
    """

    @property
    def config(self) -> 'PopulationConfig | DiscretePopulationConfig': ...

    @property
    def species(self) -> 'Species': ...

    @property
    def index_registry(self) -> 'IndexRegistry': ...
