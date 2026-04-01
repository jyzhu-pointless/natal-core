"""Dashboard entry points.

This module keeps the legacy ``natal.ui.dashboard`` import path stable while
dispatching between the panmictic and spatial dashboard implementations.
"""

from __future__ import annotations

from typing import Any

from natal.spatial_population import SpatialPopulation

from .dashboard_population import Dashboard as PopulationDashboard
from .dashboard_population import launch as launch_population
from .spatial_dashboard import SpatialDashboard, launch_spatial

Dashboard = PopulationDashboard

__all__ = [
    "Dashboard",
    "PopulationDashboard",
    "SpatialDashboard",
    "launch",
    "launch_population",
    "launch_spatial",
]


def launch(population: Any, port: int = 8080, title: str = "NATAL Dashboard") -> None:
    """Launch either the population or spatial dashboard."""
    if isinstance(population, SpatialPopulation):
        launch_spatial(population, port=port, title=title)
        return
    launch_population(population, port=port, title=title)
