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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from natal.genetic_patterns import ZygoteTypePattern

from natal.genetic_entities import Genotype, HaploidGenotype
from natal.numba_utils import njit_switch


class _UnorderedGenotypeDict(dict[Genotype, int]):
    """Dict that auto-canonicalizes Genotype keys on lookup.

    Both ``A|a`` and ``a|A`` resolve to the same value because the key
    is passed through ``Species.unordered_genotype()`` before dict access.

    This is an internal helper so that callers can use
    ``registry.genotype_to_index[any_form]`` without manual canonicalization.
    """

    def _canonicalize(self, key: Genotype) -> Genotype:
        return key.species.unordered_genotype(key.maternal, key.paternal)

    def __getitem__(self, key: Genotype) -> int:
        return super().__getitem__(self._canonicalize(key))

    def __contains__(self, key: object) -> bool:
        if isinstance(key, Genotype):
            key = self._canonicalize(key)
        return super().__contains__(key)

    def __setitem__(self, key: Genotype, value: int) -> None:
        super().__setitem__(self._canonicalize(key), value)


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

    @n_ztypes.setter
    def n_ztypes(self, value: int) -> None:
        """Backward-compat no-op setter — n_ztypes is always computed."""
        pass

    @property
    def n_gtypes(self) -> int:
        """Number of active GType entries (computed from flat index list)."""
        return len(self._index_to_gtype)

    @n_gtypes.setter
    def n_gtypes(self, value: int) -> None:
        """Backward-compat no-op setter — n_gtypes is always computed."""
        pass

    @property
    def genotype_to_index(self) -> Dict[Genotype, int]:
        """Computed dict of unique genotypes from the ZType space.

        Derived from ``_index_to_ztype`` so that after compression only
        surviving genotypes appear.
        """
        result: dict[Genotype, int] = _UnorderedGenotypeDict()
        seen: set[Genotype] = set()
        for gt, _slab in self._index_to_ztype:
            if gt not in seen:
                result[gt] = len(seen)
                seen.add(gt)
        return result

    @genotype_to_index.setter
    def genotype_to_index(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

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

    @index_to_genotype.setter
    def index_to_genotype(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

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

    @haplo_to_index.setter
    def haplo_to_index(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

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

    @index_to_haplo.setter
    def index_to_haplo(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    @property
    def index_to_gtype(self) -> List[Tuple[HaploidGenotype, str]]:
        """Computed list of (haplogenotype, glab_label) pairs from GType space.

        The index in this list is the GType index — the same index returned
        by ``gtype_index()`` and consumed by the engine's compressed arrays.
        """
        return self._index_to_gtype.copy()

    @index_to_gtype.setter
    def index_to_gtype(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    @property
    def glab_to_index(self) -> Dict[str, int]:
        """Computed dict mapping gamete label strings to their index in ``glab_labels``."""
        return {label: i for i, label in enumerate(self.glab_labels)}

    @glab_to_index.setter
    def glab_to_index(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    @property
    def index_to_glab(self) -> List[str]:
        """Alias for ``glab_labels`` (backward compat)."""
        return self.glab_labels

    @index_to_glab.setter
    def index_to_glab(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    @property
    def slab_to_index(self) -> Dict[str, int]:
        """Computed dict mapping somatic label strings to their index in ``slab_labels``."""
        return {label: i for i, label in enumerate(self.slab_labels)}

    @slab_to_index.setter
    def slab_to_index(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    @property
    def index_to_slab(self) -> List[str]:
        """Alias for ``slab_labels`` (backward compat)."""
        return self.slab_labels

    @index_to_slab.setter
    def index_to_slab(self, value: object) -> None:
        """Backward-compat no-op setter."""
        pass

    # ==================================================================
    # Internal helpers — ensure a genotype/haplo is tracked in flat lists
    # ==================================================================

    def _ensure_genotype_registered(self, genotype: Genotype) -> int:
        """Register a genotype in the computed dicts if not already present.

        This is a no-op if the genotype already appears in ``_index_to_ztype``
        (via a prior ``register_ztype`` call).  Otherwise it registers the
        genotype with the first slab label (or ``"default"``).

        Returns:
            The genotype's index in the computed ``genotype_to_index`` dict.
        """
        # Check if already present via ZType space
        for gt, _slab in self._index_to_ztype:
            if gt == genotype:
                return self.genotype_to_index[genotype]
        # Register with first available slab (or "default")
        slab = self.slab_labels[0] if self.slab_labels else "default"
        self.register_ztype(genotype, slab)
        return self.genotype_to_index[genotype]

    def _ensure_haplo_registered(self, haplo: HaploidGenotype) -> int:
        """Register a haplotype in the computed dicts if not already present.

        Returns:
            The haplotype's index in the computed ``haplo_to_index`` dict.
        """
        for hg, _glab in self._index_to_gtype:
            if hg == haplo:
                return self.haplo_to_index[haplo]
        glab = self.glab_labels[0] if self.glab_labels else "default"
        self.register_gtype(haplo, glab)
        return self.haplo_to_index[haplo]

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

    def num_gamete_labels(self) -> int:
        """Return the number of registered gamete labels.

        Returns:
            int: Count of registered gamete labels.
        """
        return len(self.glab_labels)

    def num_somatic_labels(self) -> int:
        """Return the number of registered somatic labels."""
        return len(self.slab_labels)

    def genotype_to_ztype_indices(self, g_idx: int) -> list[int]:
        """Return all ZType indices for a genotype index.

        For a population with n_slabs somatic labels, genotype index g
        maps to ZType indices [g * n_slabs, ..., g * n_slabs + n_slabs - 1]
        before compression. After compression, indices are remapped but
        each surviving genotype still maps to its surviving slab variants.
        This method scans _index_to_ztype to find the actual indices,
        making it robust against compression reordering.

        Args:
            g_idx: Genotype index (0..num_genotypes()-1).

        Returns:
            list[int]: ZType indices for this genotype (one per surviving slab).
        """
        target = self.index_to_genotype[g_idx]
        indices: list[int] = []
        for zt_idx, (gt, _slab) in enumerate(self._index_to_ztype):
            if gt == target:
                indices.append(zt_idx)
        return indices

    def genotype_index(self, genotype_id: Any) -> int:
        """Return the index for a registered genotype key.

        Genotype instances are canonicalized before lookup so that ``A|a``
        and ``a|A`` resolve to the same index.

        Args:
            genotype_id: Registered genotype instance or identifier.

        Returns:
            int: The genotype index.

        Raises:
            KeyError: If the genotype_id is not registered.
        """
        return self.genotype_to_index[genotype_id]

    def haplo_index(self, haplo_id: Any) -> int:
        """Return the index for a registered haplogenotype key.

        Args:
            haplo_id: Registered haplogenotype instance or identifier.

        Returns:
            int: The haplogenotype index.

        Raises:
            KeyError: If the haplogenotype is not registered.
        """
        return self.haplo_to_index[haplo_id]

    def gamete_label_index(self, gamete_label: str) -> int:
        """Return the index for a registered gamete label key.

        Args:
            gamete_label: Registered gamete label string.

        Returns:
            int: The gamete label index.

        Raises:
            KeyError: If the gamete label is not registered.
        """
        return self.glab_to_index[gamete_label]

    def somatic_label_index(self, somatic_label: str) -> int:
        """Return the index for a registered somatic label key.

        Raises:
            KeyError: If the somatic label is not registered.
        """
        return self.slab_to_index[somatic_label]

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

    # ==================================================================
    # Internal helpers — convert selectors to valid indices
    # ==================================================================

    def _ensure_genotype_index(self, genotype_or_index: Union[Genotype, int]) -> int:
        """Convert a genotype or integer index to a valid registry index."""
        if isinstance(genotype_or_index, int):
            num_g = self.num_genotypes()
            if 0 <= genotype_or_index < num_g:
                return int(genotype_or_index)
            raise IndexError(f"Genotype index {genotype_or_index} out of range")
        assert not isinstance(genotype_or_index, int)
        slab = self.slab_labels[0] if self.slab_labels else "default"
        return self.register_ztype(genotype_or_index, slab)

    def _ensure_haplo_index(self, haplo_or_index: Union[Any, int]) -> int:
        """Convert a haplogenotype selector to an integer index.

        Behaves similarly to :meth:`_ensure_genotype_index`.

        Args:
            haplo_or_index: Either an int index or a haplogenotype key.

        Returns:
            int: A valid haplogenotype index.
        """
        num_h = self.num_haplogenotypes()
        if isinstance(haplo_or_index, int) and 0 <= haplo_or_index < num_h:
            return int(haplo_or_index)
        glab = self.glab_labels[0] if self.glab_labels else "default"
        # At this point haplo_or_index is not int (caught above), so cast to
        # HaploidGenotype for the type checker.  Runtime behaviour matches
        # the old register_haplogenotype which accepted Any.
        return self.register_gtype(cast(HaploidGenotype, haplo_or_index), glab)

    def _ensure_glab_index(self, glab_or_index: Union[str, int]) -> int:
        """Convert a gamete-label selector to an integer index.

        Behaves similarly to :meth:`_ensure_genotype_index`.

        Args:
            glab_or_index: Either an int index or a gamete-label key.

        Returns:
            int: A valid gamete-label index.

        Raises:
            AssertionError: If the input is not a string nor a valid integer index.
        """
        if isinstance(glab_or_index, int) and 0 <= glab_or_index < len(self.glab_labels):
            return int(glab_or_index)
        assert isinstance(glab_or_index, str), (
            f"Gamete label must be a string or int index, got {type(glab_or_index)}"
        )
        return self.register_gamete_label(glab_or_index)

    # ==================================================================
    # Resolver helpers (centralized key parsing)
    # ==================================================================

    def resolve_genotype_index(
        self, diploid_genotypes: Sequence[Any], gk: Any, strict: bool = True
    ) -> Optional[int]:
        """Resolve a flexible genotype selector to a diploid genotype index.

        Accepted selector types:
            - int: returned if within range
            - genotype object: matched by identity/equality in ``diploid_genotypes``
            - str: compared against ``genotype.to_string()`` where available

        Args:
            diploid_genotypes: Sequence of diploid genotype objects.
            gk: Selector (int, object or str) to resolve.
            strict: If True raise KeyError on failure, otherwise return None.

        Returns:
            Optional[int]: Resolved genotype index, or None if not found and
            ``strict`` is False.

        Raises:
            KeyError: If resolution fails and ``strict`` is True.
        """
        if isinstance(gk, int):
            if 0 <= gk < len(diploid_genotypes):
                return int(gk)
            if strict:
                raise KeyError(f"genotype index out of range: {gk}")
            return None

        # direct object match
        try:
            if gk in diploid_genotypes:
                return int(diploid_genotypes.index(gk))
        except Exception:
            pass

            # string match via to_string() — try unordered form first,
        # then the reversed maternal/paternal form (since genotypes
        # are canonicalized, a user writing "a|A" should still match
                # the unordered "A|a").
        if isinstance(gk, str):
            for i, g in enumerate(diploid_genotypes):
                try:
                    if hasattr(g, "to_string") and g.to_string() == gk:
                        return i
                except Exception:
                    continue
            # Try reversed form: swap maternal/paternal in each stored
            # genotype's string representation.
            for i, g in enumerate(diploid_genotypes):
                try:
                    if hasattr(g, "to_string") and hasattr(g, "maternal") and hasattr(g, "paternal"):
                        rev = f"{g.paternal.to_string()}|{g.maternal.to_string()}"
                        if rev == gk:
                            return i
                except Exception:
                    continue

        if strict:
            raise KeyError(f"Cannot resolve genotype key: {gk} in {diploid_genotypes}")
        return None

    def resolve_hg_glab_part(
        self,
        haploid_genotypes: Sequence[HaploidGenotype],
        part: Union[Tuple[int, int], Tuple[HaploidGenotype, Union[int, str]], int, str],
        n_glabs: int,
    ) -> Tuple[int, int]:
        """Resolve a haploid/genetic part into an (hg_idx, glab_idx) pair.

        Accepted input formats for ``part``:
            - (int, int): already (hg_idx, glab_idx)
            - (HaploidGenotype, Union[int, str]): where ``Union[int, str]`` is int or string gamete label
            - HaploidGenotype object: maps to (idx, 0)
            - int: treated as compressed index and decompressed
            - str: matched against ``haploid.to_string()`` and returns (idx, 0)

        Args:
            haploid_genotypes: Sequence of haploid genotype objects.
            part: The flexible selector to resolve.
            n_glabs: Number of gamete labels (used for decompression).

        Returns:
            Tuple[int, int]: The resolved (hg_idx, glab_idx) pair.

        Raises:
            KeyError: If resolution fails.
        """
        pair_part = _as_pair(part)

        # tuple of ints (already decompressed)
        if pair_part is not None and isinstance(pair_part[0], int) and isinstance(pair_part[1], int):
            return (int(pair_part[0]), int(pair_part[1]))

        # (str, glab) where first element is haploid string representation
        if pair_part is not None and isinstance(pair_part[0], str):
            name, lab = pair_part
            found_idx = None
            for i, hg in enumerate(haploid_genotypes):
                try:
                    if hasattr(hg, "to_string") and hg.to_string() == name:
                        found_idx = i
                        break
                except Exception:
                    pass
                try:
                    if str(hg) == name:
                        found_idx = i
                        break
                except Exception:
                    pass

            if found_idx is None:
                raise KeyError(f"Unknown haploid string: {name}")

            if isinstance(lab, int):
                glab_idx = int(lab)
            else:
                glab_idx = self.glab_to_index.get(str(lab))
                if glab_idx is None:
                    raise KeyError(f"Unknown glab label: {lab}")
            return (found_idx, glab_idx)

        # (HaploidGenotype, glab)
        if pair_part is not None and isinstance(pair_part[0], HaploidGenotype):
            hg_obj, lab = pair_part
            try:
                idx_hg = int(haploid_genotypes.index(hg_obj))
            except ValueError:
                raise KeyError(f"Unknown haploid object: {hg_obj}") from ValueError

            if isinstance(lab, int):
                glab_idx = int(lab)
            else:
                glab_idx = self.glab_to_index.get(str(lab))
                if glab_idx is None:
                    raise KeyError(f"Unknown glab label: {lab}")
            return (idx_hg, glab_idx)

        # HaploidGenotype object -> default glab 0
        if isinstance(part, HaploidGenotype):
            try:
                return (int(haploid_genotypes.index(part)), 0)
            except ValueError:
                raise KeyError(f"Unknown haploid object: {part}") from ValueError

        # compressed integer
        if isinstance(part, int):
            try:
                return decompress_hg_glab(part, n_glabs)
            except Exception:
                raise KeyError(f"Unknown compressed index: {part}") from Exception

        # string matching to_string()
        if isinstance(part, str):
            for i, hg in enumerate(haploid_genotypes):
                try:
                    if hasattr(hg, "to_string") and hg.to_string() == part:
                        return (i, 0)
                except Exception:
                    continue

        raise KeyError(f"Cannot resolve hg+glab part: {part}")

    def resolve_comp_idx(
        self,
        haploid_genotypes: Sequence[Any],
        n_glabs: int,
        comp_key: Any,
        strict: bool = False,
    ) -> Optional[int]:
        """Resolve a comp-map key into a compressed hg+glab integer index.

        Supported key formats:
            - int: returned directly
            - (hg_part, glab_part): each part may be int, HaploidGenotype or str
            - HaploidGenotype: maps to (hg_idx, 0)
            - str: matched against haploid.to_string()

        Args:
            haploid_genotypes: Sequence of haploid genotype objects.
            n_glabs: Number of gamete labels used for compression.
            comp_key: The flexible key to resolve.
            strict: If True raise KeyError on failure, otherwise return None.

        Returns:
            Optional[int]: Compressed hg+glab index or None when unresolved and
            ``strict`` is False.

        Raises:
            KeyError: If resolution fails and ``strict`` is True.
        """
        # direct int
        if isinstance(comp_key, int):
            return int(comp_key)

        # tuple (hg_part, glab_part)
        pair_comp = _as_pair(comp_key)
        if pair_comp is not None:
            part_hg, part_glab = pair_comp
            # resolve hg part
            if isinstance(part_hg, int):
                idx_hg = int(part_hg)
            elif isinstance(part_hg, HaploidGenotype):
                try:
                    idx_hg = int(haploid_genotypes.index(part_hg))
                except ValueError:
                    if strict:
                        raise KeyError(f"Cannot resolve haploid object: {part_hg}") from ValueError
                    return None
            elif isinstance(part_hg, str):
                found = False
                idx_hg = None
                for i, hg in enumerate(haploid_genotypes):
                    try:
                        if hasattr(hg, "to_string") and hg.to_string() == part_hg:
                            idx_hg = i
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    if strict:
                        raise KeyError(f"Cannot resolve haploid string key: {part_hg}")
                    return None
            else:
                if strict:
                    raise KeyError(f"Unsupported haploid key type: {type(part_hg)}")
                return None

            # resolve glab part
            if isinstance(part_glab, int):
                glab_idx = int(part_glab)
            else:
                glab_idx = self.glab_to_index.get(str(part_glab))
                if glab_idx is None:
                    if strict:
                        raise KeyError(f"Unknown glab label: {part_glab}")
                    return None
            assert isinstance(idx_hg, int) and isinstance(glab_idx, int), (
                "Resolved indices must be integers"
            )
            return compress_hg_glab(idx_hg, glab_idx, n_glabs)

        # HaploidGenotype -> default glab 0
        if isinstance(comp_key, HaploidGenotype):
            try:
                idx_hg = int(haploid_genotypes.index(comp_key))
            except ValueError:
                if strict:
                    raise KeyError(f"Unknown haploid object: {comp_key}") from ValueError
                return None
            return compress_hg_glab(idx_hg, 0, n_glabs)

        # string -> match to_string
        if isinstance(comp_key, str):
            for i, hg in enumerate(haploid_genotypes):
                try:
                    if hasattr(hg, "to_string") and hg.to_string() == comp_key:
                        return compress_hg_glab(i, 0, n_glabs)
                except Exception:
                    continue
            if strict:
                raise KeyError(f"Cannot resolve haploid string key: {comp_key}")
            return None

        if strict:
            raise KeyError(f"Unsupported comp_map key type: {type(cast(object, comp_key))}")
        return None


def _as_pair(value: object) -> Optional[Tuple[object, object]]:
    if not isinstance(value, tuple):
        return None
    tuple_value = cast(Tuple[object, ...], value)
    if len(tuple_value) != 2:
        return None
    return tuple_value[0], tuple_value[1]


# ==================================================================
# Module-level formula helpers — kept for BFS reachability computation
# These are private (_ prefix) as they're an implementation detail.
# ==================================================================


@njit_switch(cache=True)
def compress_hg_glab(hg_idx: int, glab_idx: int, n_glabs: int) -> int:
    """Compress a (haplogenotype, glab) pair into a single integer.

    The compressed representation is ``hg_idx * n_glabs + glab_idx`` and is
    commonly used to index flattened tensors that combine haplogenotype and
    gamete-label axes.

    Args:
        hg_idx: Haplogenotype index.
        glab_idx: Gamete-label index.
        n_glabs: Number of distinct gamete labels.

    Returns:
        int: The compressed combined index.
    """
    return int(hg_idx) * int(n_glabs) + int(glab_idx)


@njit_switch(cache=True)
def decompress_hg_glab(compressed_idx: int, n_glabs: int) -> Tuple[int, int]:
    """Decompress a combined hg+glab index back into its components.

    Args:
        compressed_idx: The compressed integer index.
        n_glabs: Number of distinct gamete labels used during compression.

    Returns:
        Tuple[int, int]: ``(hg_idx, glab_idx)`` unpacked from ``compressed_idx``.
    """
    hg_idx = int(compressed_idx) // int(n_glabs)
    glab_idx = int(compressed_idx) % int(n_glabs)
    return hg_idx, glab_idx
