"""Registry subpackage for stable integer indexing of population entities.

Provides :class:`~natal.registry.index.IndexRegistry` which assigns and
maintains stable integer indices for genotypes, haploid genotypes, and
gamete/somatic labels used by the simulation engine.
"""

from .index import IndexRegistry

__all__ = ["IndexRegistry"]
