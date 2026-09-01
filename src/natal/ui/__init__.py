"""Forwarding shim: the ``ui`` package now lives at
``natal.frontend.ui``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.ui.dashboard as _m0
import natal.frontend.ui.dashboard_helpers as _m1
import natal.frontend.ui.dashboard_population as _m2
import natal.frontend.ui.spatial_dashboard as _m3
import natal.frontend.ui.visualization as _m4
from natal.frontend.ui import (
    Dashboard,
    PopulationDashboard,
    SpatialDashboard,
    get_allele_color,
    launch,
    launch_population,
    launch_spatial,
    render_cell_svg,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.ui.dashboard"] = _m0
_sys.modules["natal.ui.dashboard_helpers"] = _m1
_sys.modules["natal.ui.dashboard_population"] = _m2
_sys.modules["natal.ui.spatial_dashboard"] = _m3
_sys.modules["natal.ui.visualization"] = _m4

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "Dashboard",
    "PopulationDashboard",
    "SpatialDashboard",
    "get_allele_color",
    "launch",
    "launch_population",
    "launch_spatial",
    "render_cell_svg",
]
