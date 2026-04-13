"""Backward-compatible shim for spatial simulation kernels.

Core implementations now live in ``natal.kernels.spatial_simulation_kernels``.
This module keeps legacy imports working by re-exporting kernel primitives.
"""

from __future__ import annotations

from natal.kernels.spatial_migration_kernels import (
    apply_spatial_adjacency_migration,
    run_spatial_migration,
)
from natal.kernels.spatial_simulation_kernels import (
    run_spatial_tick,
    run_spatial_tick_heterogeneous,
    run_spatial_tick_with_adjacency_migration,
    run_spatial_tick_with_migration,
)

__all__ = [
    "run_spatial_tick",
    "run_spatial_tick_heterogeneous",
    "run_spatial_tick_with_migration",
    "run_spatial_tick_with_adjacency_migration",
    "apply_spatial_adjacency_migration",
    "run_spatial_migration",
]
