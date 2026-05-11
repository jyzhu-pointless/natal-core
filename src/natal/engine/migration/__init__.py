"""Spatial migration backend modules."""

from __future__ import annotations

from natal.engine.migration.adjacency import apply_spatial_adjacency_mode
from natal.engine.migration.kernel import (
    apply_spatial_kernel_migration,
    apply_spatial_kernel_migration_heterogeneous,
)

__all__ = [
    "apply_spatial_adjacency_mode",
    "apply_spatial_kernel_migration",
    "apply_spatial_kernel_migration_heterogeneous",
]
