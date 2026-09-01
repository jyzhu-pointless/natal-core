"""Forwarding shim: the ``spatial`` package now lives at
``natal.frontend.spatial``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.spatial.configurator as _m0
import natal.frontend.spatial.population as _m1
import natal.frontend.spatial.topology as _m2
from natal.frontend.spatial import (
    BatchSetting,
    GridTopology,
    HeterogeneousKernelParams,
    HexGrid,
    MigrationParams,
    SpatialConfigurator,
    SpatialPopulation,
    SpatialTopology,
    SquareGrid,
    batch_setting,
    build_adjacency_matrix,
    build_gaussian_kernel,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.spatial.configurator"] = _m0
_sys.modules["natal.spatial.population"] = _m1
_sys.modules["natal.spatial.topology"] = _m2

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "BatchSetting",
    "GridTopology",
    "HeterogeneousKernelParams",
    "HexGrid",
    "MigrationParams",
    "SpatialConfigurator",
    "SpatialPopulation",
    "SpatialTopology",
    "SquareGrid",
    "batch_setting",
    "build_adjacency_matrix",
    "build_gaussian_kernel",
]
