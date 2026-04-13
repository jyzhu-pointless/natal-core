"""Spatial migration backend modules."""

from __future__ import annotations

from natal.kernels.migration.adjacency import apply_spatial_adjacency_mode
from natal.kernels.migration.kernel import apply_spatial_kernel_migration

__all__ = [
    "apply_spatial_adjacency_mode",
    "apply_spatial_kernel_migration",
]
