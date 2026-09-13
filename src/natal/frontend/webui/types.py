"""Shared types for the Vue web UI server layer.

The web UI server holds a population object in-process.  ``SpatialPopulation`` intentionally does
not share a base class with the panmictic populations, so the server layer
works on the ``DashboardPopulation`` union and narrows via ``isinstance``.
"""

from __future__ import annotations

from typing import Union

from natal.frontend.data.state import DiscretePopulationState, PopulationState
from natal.frontend.population.base import BasePopulation
from natal.frontend.spatial.population import SpatialPopulation

#: Any population object the dashboard can visualize.
DashboardPopulation = Union[
    BasePopulation[PopulationState],
    BasePopulation[DiscretePopulationState],
    SpatialPopulation,
]
