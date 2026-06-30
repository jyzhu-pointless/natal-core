"""Registry for stable integer indexing of population entities.

This module provides an :class:`IndexRegistry` that assigns and maintains
stable integer indices for genotypes, haploid genotypes, and gamete labels.
It uses flat dict-based lookups (ZType/GType spaces) instead of formula-based
indexing, allowing independent pruning of individual (genotype, slab) and
(haplogenotype, glab) pairs.

The registry is used throughout the simulation to translate domain objects
(e.g., ``Genotype`` instances, strings) into compact integer indices that
are suitable for NumPy arrays and Numba‑accelerated engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from natal.genetic_patterns import ZygoteTypePattern

from natal.genetic_entities import Genotype, HaploidGenotype


class IndexRegistry:
    """Registry providing stable integer indices for population entities.

    The IndexRegistry assigns and stores stable integer indices for entities
    that occur in the population.  Internally it uses flat dict-based lookups:
    ZType = (genotype, slab_label) for the diploid layer, and
    GType = (haplogenotype, glab_label) for the gamete layer.

    Examples:
        ic = IndexRegistry()
        gid = ic.register_genotype('g1')
        hid = ic.register_haplogenotype('h1')
        glid = ic.register_gamete_label('gl1')

    Attributes:
        slab_labels: List of registered somatic (slab) label strings.
        glab_labels: List of registered gamete (glab) label strings.
    """

    def __init__(self) -> None:
        # ---- ZType space (diploid layer primary index) ----
        self._ztype_to_index: Dict[Tuple[Genotype, str], int] = {}
        self._index_to_ztype: List[Tuple[Genotype, str]] = []

        # ---- GType space (gamete layer primary index) ----
        self._gtype_to_index: Dict[Tuple[HaploidGenotype, str], int] = {}
        self._index_to_gtype: List[Tuple[HaploidGenotype, str]] = []

        # ---- Label metadata (ordered lists, replaces old label dicts) ----
        self.slab_labels: List[str] = []
        self.glab_labels: List[str] = []

    # ==================================================================
    # Computed properties — derived from the flat index lists above
    # ==================================================================

    @property
    def n_ztypes(self) -> int:
        """Number of active ZType entries (computed from flat index list)."""
        return len(self._index_to_ztype)

    @property
    def n_gtypes(self) -> int:
        """Number of active GType entries (computed from flat index list)."""
        return len(self._index_to_gtype)

    @property
    def index_to_genotype(self) -> List[Genotype]:
        """Computed list of unique genotypes (in registration order) from ZType space."""
        seen: set[Genotype] = set()
        result: list[Genotype] = []
        for gt, _slab in self._index_to_ztype:
            if gt not in seen:
                result.append(gt)
                seen.add(gt)
        return result

    @property
    def index_to_ztype(self) -> List[Tuple[Genotype, str]]:
        """Computed list of (genotype, slab_label) pairs from ZType space.

        The index in this list is the ZType index — the same index consumed
        by the engine's ``individual_count`` arrays on the last axis.
        """
        return self._index_to_ztype.copy()

    @property
    def haplo_to_index(self) -> Dict[HaploidGenotype, int]:
        """Computed dict of unique haplotypes from the GType space.

        Derived from ``_index_to_gtype`` so that after compression only
        surviving haplotypes appear.
        """
        result: dict[HaploidGenotype, int] = {}
        seen: set[HaploidGenotype] = set()
        for hg, _glab in self._index_to_gtype:
            if hg not in seen:
                result[hg] = len(seen)
                seen.add(hg)
        return result

    @property
    def index_to_haplo(self) -> List[HaploidGenotype]:
        """Computed list of unique haplotypes (in registration order) from GType space."""
        seen: set[HaploidGenotype] = set()
        result: list[HaploidGenotype] = []
        for hg, _glab in self._index_to_gtype:
            if hg not in seen:
                result.append(hg)
                seen.add(hg)
        return result

    @property
    def index_to_gtype(self) -> List[Tuple[HaploidGenotype, str]]:
        """Computed list of (haplogenotype, glab_label) pairs from GType space.

        The index in this list is the GType index — the same index returned
        by ``gtype_index()`` and consumed by the engine's compressed arrays.
        """
        return self._index_to_gtype.copy()

    @property
    def glab_to_index(self) -> Dict[str, int]:
        """Computed dict mapping gamete label strings to their index in ``glab_labels``."""
        return {label: i for i, label in enumerate(self.glab_labels)}

    @property
    def index_to_glab(self) -> List[str]:
        """Alias for ``glab_labels`` (backward compat)."""
        return self.glab_labels

    @property
    def slab_to_index(self) -> Dict[str, int]:
        """Computed dict mapping somatic label strings to their index in ``slab_labels``."""
        return {label: i for i, label in enumerate(self.slab_labels)}

    @property
    def index_to_slab(self) -> List[str]:
        """Alias for ``slab_labels`` (backward compat)."""
        return self.slab_labels

    # ==================================================================
    # Flat dict-based registration API
    # ==================================================================

    def register_ztype(self, genotype: Genotype, slab_label: str) -> int:
        """Register a ZType (genotype, slab) pair and return its index.

        O(1) dict lookup for duplicates; appends to flat list for new entries.
        Automatically tracks ``slab_labels``.

        Args:
            genotype: A ``Genotype`` instance.
            slab_label: Somatic label string for this ZType variant.

        Returns:
            int: The assigned ZType index.  Stable until compression.
        """
        key = (genotype, slab_label)
        if key in self._ztype_to_index:
            return self._ztype_to_index[key]
        idx = len(self._index_to_ztype)
        self._index_to_ztype.append(key)
        self._ztype_to_index[key] = idx
        if slab_label not in self.slab_labels:
            self.slab_labels.append(slab_label)
        return idx

    def register_gtype(self, haplo: HaploidGenotype, glab_label: str) -> int:
        """Register a GType (haplogenotype, glab) pair and return its index.

        O(1) dict lookup for duplicates; appends to flat list for new entries.
        Automatically tracks ``glab_labels``.

        Args:
            haplo: A ``HaploidGenotype`` instance.
            glab_label: Gamete label string for this GType variant.

        Returns:
            int: The assigned GType index.  Stable until compression.
        """
        key = (haplo, glab_label)
        if key in self._gtype_to_index:
            return self._gtype_to_index[key]
        idx = len(self._index_to_gtype)
        self._index_to_gtype.append(key)
        self._gtype_to_index[key] = idx
        if glab_label not in self.glab_labels:
            self.glab_labels.append(glab_label)
        return idx

    def register_genotype(self, genotype_id: Genotype) -> List[int]:
        """Register a genotype and auto-cross-product with all slab labels.

        If ``slab_labels`` is empty it is auto-initialised to ``["default"]``.
        Each (genotype, slab) pair becomes a ZType entry.

        Args:
            genotype_id: A ``Genotype`` instance to register.

        Returns:
            list[int]: ZType indices for this genotype (one per slab label).
        """
        if not self.slab_labels:
            self.slab_labels = ["default"]
        indices: list[int] = []
        for slab in self.slab_labels:
            idx = self.register_ztype(genotype_id, slab)
            indices.append(idx)
        return indices

    def register_haplogenotype(self, haplo_id: Any) -> List[int]:
        """Register a haplogenotype and auto-cross-product with all glab labels.

        If ``glab_labels`` is empty it is auto-initialised to ``["default"]``.
        Each (haplogenotype, glab) pair becomes a GType entry.

        Args:
            haplo_id: A ``HaploidGenotype`` instance or opaque key.

        Returns:
            list[int]: GType indices for this haplotype (one per glab label).
        """
        if not self.glab_labels:
            self.glab_labels = ["default"]
        indices: list[int] = []
        for glab in self.glab_labels:
            idx = self.register_gtype(haplo_id, glab)
            indices.append(idx)
        return indices

    def register_gamete_label(self, gamete_label: str) -> int:
        """Register a gamete label and return its index.

        Adapted to use the new ``glab_labels`` list.

        Args:
            gamete_label: String label for gamete origin.

        Returns:
            int: Assigned integer index for the gamete label.
        """
        if gamete_label not in self.glab_labels:
            self.glab_labels.append(gamete_label)
        return self.glab_labels.index(gamete_label)

    def register_somatic_label(self, somatic_label: str) -> int:
        """Register a somatic label (slab) and return its index.

        Adapted to use the new ``slab_labels`` list.

        Args:
            somatic_label: String label for somatic state.

        Returns:
            int: Assigned integer index for the somatic label.
        """
        if somatic_label not in self.slab_labels:
            self.slab_labels.append(somatic_label)
        return self.slab_labels.index(somatic_label)

    # ==================================================================
    # Query API
    # ==================================================================

    def num_genotypes(self) -> int:
        """Return the number of unique diploid genotypes (derived from ZTypes).

        Returns:
            int: Count of unique diploid genotypes in the ZType space.
        """
        return len(self.index_to_genotype)

    def num_haplogenotypes(self) -> int:
        """Return the number of unique haploid genotypes (derived from GTypes).

        Returns:
            int: Count of unique haploid genotypes in the GType space.
        """
        return len(self.index_to_haplo)

    def ztype_index(self, genotype: Genotype, slab_label: str) -> int:
        """O(1) dict lookup for a ZType index.

        Replaces the formula ``g * n_slabs + slab``.

        Args:
            genotype: A ``Genotype`` instance.
            slab_label: Somatic label string.

        Returns:
            int: The ZType index.

        Raises:
            KeyError: If the (genotype, slab_label) pair is not registered.
        """
        return self._ztype_to_index[(genotype, slab_label)]

    def resolve_ztype_indices(self, pattern: ZygoteTypePattern) -> list[int]:
        """Return all ZType indices matching a ZygoteTypePattern.

        Iterates ``_index_to_ztype`` and tests each (genotype, slab_label)
        pair against *pattern*.  When *pattern* has no slab constraint
        (``slab is None``), all slab variants of matching genotypes match.
        """
        indices: list[int] = []
        for i, (gt, slab) in enumerate(self._index_to_ztype):
            if pattern.matches(gt, slab):
                indices.append(i)
        return indices

    def resolve_default_ztype_index(self, pattern: ZygoteTypePattern) -> int:
        """Return the first ZType index matching a ZygoteTypePattern.

        Used by ``initial_state`` which places individuals in the first
        (default) slab when no ``@slab`` is specified.
        """
        for i, (gt, slab) in enumerate(self._index_to_ztype):
            if pattern.matches(gt, slab):
                return i
        raise KeyError(f"No ZType matches pattern {pattern}")

    def ztype_indices_for(self, genotype: Genotype) -> list[int]:
        """Return all ZType indices for a given Genotype object.

        Scans ``_index_to_ztype`` — does NOT require ``species.unordered_genotype()``
        because both the stored and input genotypes are already canonicalized
        via ``Genotype.__new__`` cache key normalization.
        """
        return [i for i, (gt, _) in enumerate(self._index_to_ztype) if gt == genotype]

    def gtype_indices_for(self, haplo: Any) -> list[int]:
        """Return all GType indices for a given HaploidGenotype object."""
        return [i for i, (hg, _) in enumerate(self._index_to_gtype) if hg == haplo]

    def gtype_index(self, haplo: HaploidGenotype, glab_label: str) -> int:
        """O(1) dict lookup for a GType index.

        Replaces the formula ``compress_hg_glab(hg, glab, n_glabs)``.

        Args:
            haplo: A ``HaploidGenotype`` instance.
            glab_label: Gamete label string.

        Returns:
            int: The GType index.

        Raises:
            KeyError: If the (haplo, glab_label) pair is not registered.
        """
        return self._gtype_to_index[(haplo, glab_label)]

    # ==================================================================
    # Compression — permanently remove pruned entries
    # ==================================================================

    def compress(
        self,
        ztype_mask: NDArray[np.int32],
        gtype_mask: NDArray[np.int32],
    ) -> None:
        """Permanently remove pruned ZType and GType entries from the registry.

        Both masks use -1 for pruned entries and >=0 for surviving entries
        (the value is the new compressed index).

        Unlike the old formula-based compress, this operates directly on the
        flat ZType/GType spaces — individual (genotype, slab) ZTypes can be
        pruned independently.

        Args:
            ztype_mask: ``(old_n_ztypes,)`` int32 array — ZType-level
                compression mask (-1 = pruned).
            gtype_mask: ``(old_n_gtypes,)`` int32 array — GType-level
                compression mask (-1 = pruned).
        """
        self._compress_ztypes(ztype_mask)
        self._compress_gtypes(gtype_mask)

    def _compress_ztypes(self, ztype_mask: NDArray[np.int32]) -> None:
        """Rebuild ZType flat lists/dicts from the active mask entries."""
        active = ztype_mask >= 0

        new_index_to_ztype: list[Tuple[Genotype, str]] = [
            zt for i, zt in enumerate(self._index_to_ztype) if active[i]
        ]
        self._index_to_ztype = new_index_to_ztype
        self._ztype_to_index = {zt: i for i, zt in enumerate(new_index_to_ztype)}

    def _compress_gtypes(self, gtype_mask: NDArray[np.int32]) -> None:
        """Rebuild GType flat lists/dicts from the active mask entries."""
        active = gtype_mask >= 0

        new_index_to_gtype: list[Tuple[HaploidGenotype, str]] = [
            gt for i, gt in enumerate(self._index_to_gtype) if active[i]
        ]
        self._index_to_gtype = new_index_to_gtype
        self._gtype_to_index = {gt: i for i, gt in enumerate(new_index_to_gtype)}

# compress_hg_glab / decompress_hg_glab have been moved to
# natal.population_config as _compress_hl / _decompress_hl.
# They are only needed during species blueprint construction
# (before IndexRegistry exists).  For runtime use gtype_index().
