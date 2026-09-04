"""Spatial migration backend modules (slice 5 CSR engine)."""

from __future__ import annotations

from natal.backends.reference.migration.adjacency import (
    apply_csr_migration,
    migrate_scalar_bucket,
    migrate_sperm_bucket,
)

__all__ = [
    "apply_csr_migration",
    "migrate_scalar_bucket",
    "migrate_sperm_bucket",
]
