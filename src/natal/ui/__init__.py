"""
NATAL UI Module
===============

Provides user interface components for visualizing and controlling simulations.
Requires `nicegui` to be installed (`pip install nicegui`).
"""

from .dashboard import (
    Dashboard,
    PopulationDashboard,
    SpatialDashboard,
    launch,
    launch_population,
    launch_spatial,
)

__all__ = [
    "Dashboard",
    "PopulationDashboard",
    "SpatialDashboard",
    "launch",
    "launch_population",
    "launch_spatial",
]
