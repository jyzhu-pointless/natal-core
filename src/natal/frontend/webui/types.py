"""Shared types for the Vue web UI server layer.

The web UI server holds a population object in-process (same ownership model
as the legacy NiceGUI dashboards).  ``SpatialPopulation`` intentionally does
not share a base class with the panmictic populations, so the server layer
works on the ``DashboardPopulation`` union and narrows via ``isinstance``.
"""

from __future__ import annotations

from typing import Protocol, Union, runtime_checkable

from natal.frontend.data.state import DiscretePopulationState, PopulationState
from natal.frontend.population.base import BasePopulation
from natal.frontend.spatial.population import SpatialPopulation

#: Any population object the dashboard can visualize.
DashboardPopulation = Union[
    BasePopulation[PopulationState],
    BasePopulation[DiscretePopulationState],
    SpatialPopulation,
]


@runtime_checkable
class RustBackendProbe(Protocol):
    """Structural view of the optional Rust lifecycle backend flag.

    ``using_rust_backend`` is declared per concrete population rather than on
    ``BasePopulation``, so the server probes it structurally instead of
    reaching into private attributes.
    """

    @property
    def using_rust_backend(self) -> bool: ...


def uses_rust_backend(population: DashboardPopulation) -> bool:
    """Return whether *population* currently runs its lifecycle in Rust.

    Args:
        population: Population object held by the dashboard.

    Returns:
        ``True`` when the Rust engine session drives the simulation ticks.
    """
    return isinstance(population, RustBackendProbe) and population.using_rust_backend
