"""Registry for stable integer indexing of population entities.

This module provides an :class:`IndexRegistry` that assigns and maintains
stable integer indices for genotypes, haploid genotypes, and gamete labels.
It also offers helper functions to compress/decompress combined
(haplogenotype, gamete‑label) indices.

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
        if key.species.unordered:
            key = key.species.unordered_genotype(key.maternal, key.paternal)
        return key

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

    The IndexRegistry assigns and stores stable integer indices for entities that
    occur in the population: diploid genotypes, haploid genotypes (haplogenotypes),
    and gamete labels. It exposes small helper methods to export index arrays and
    to resolve flexible selector types (objects, ints, or string keys) into
    numeric indices suitable for numeric backends.

    Examples:
        ic = IndexRegistry()
        gid = ic.register_genotype('g1')
        hid = ic.register_haplogenotype('h1')
        glid = ic.register_gamete_label('gl1')

    Attributes:
        genotype_to_index: Mapping from genotype identifier to assigned index.
        index_to_genotype: List mapping index back to genotype identifier.
        haplo_to_index: Mapping from haplogenotype identifier to assigned index.
        index_to_haplo: List mapping index back to haplogenotype identifier.
        glab_to_index: Mapping from gamete label identifier to assigned index.
        index_to_glab: List mapping index back to gamete label identifier.
    """

    def __init__(self) -> None:
        # entity mappings
        # genotype_to_index auto-canonicalizes Genotype keys on lookup
        # so that A|a and a|A both resolve to the same index.
        self.genotype_to_index: Dict[Genotype, int] = _UnorderedGenotypeDict()
        self.index_to_genotype: List[Genotype] = []

        self.haplo_to_index: Dict[HaploidGenotype, int] = {}
        self.index_to_haplo: List[HaploidGenotype] = []

        self.glab_to_index: Dict[str, int] = {}

        # n_ztypes tracks the engine-visible G-axis count.  It starts at
        # num_genotypes() (the registered count before compression) and is updated
        # by compression.  Hooks and pattern resolvers read this, not
        # num_genotypes().
        self.n_ztypes: int = 0
        self.index_to_glab: List[str] = []

        self.slab_to_index: Dict[str, int] = {}
        self.index_to_slab: List[str] = []

        # axis sizes for compatibility (not used for numeric flattening)
        self.axis_sizes: Dict[str, int] = {}

    # ---------- registration API ----------
    def register_genotype(self, genotype_id: Genotype) -> int:
        """Register a genotype and return its stable integer index.

        The genotype is canonicalized via ``Species.unordered_genotype()``
        so that ``A|a`` and ``a|A`` share the same index.

        Args:
            genotype_id: A ``Genotype`` instance to register.

        Returns:
            int: The assigned integer index.  Indices remain stable
            until the registry is compacted.
        """
        # Only canonicalize unordered species — sex chromosomes
        # require maternal/paternal ordering (X|Y ≠ Y|X).
        if genotype_id.species.unordered:
            genotype_id = genotype_id.species.unordered_genotype(
                genotype_id.maternal, genotype_id.paternal,
            )
        if genotype_id in self.genotype_to_index:
            return self.genotype_to_index[genotype_id]
        idx = len(self.index_to_genotype)
        self.genotype_to_index[genotype_id] = idx
        self.index_to_genotype.append(genotype_id)
        return idx

    def register_haplogenotype(self, haplo_id: Any) -> int:
        """Register a haploid genotype (haplogenotype) and return its index.

        Args:
            haplo_id: Haploid genotype instance or opaque identifier used as
                the canonical key.

        Returns:
            int: Assigned integer index for the haplogenotype.
        """
        if haplo_id in self.haplo_to_index:
            return self.haplo_to_index[haplo_id]
        idx = len(self.index_to_haplo)
        self.haplo_to_index[haplo_id] = idx
        self.index_to_haplo.append(haplo_id)
        return idx

    def register_gamete_label(self, gamete_label: str) -> int:
        """Register a gamete label and return its index.

        Args:
            gamete_label: String label for gamete origin.

        Returns:
            int: Assigned integer index for the gamete label.
        """
        if gamete_label in self.glab_to_index:
            return self.glab_to_index[gamete_label]
        idx = len(self.index_to_glab)
        self.glab_to_index[gamete_label] = idx
        self.index_to_glab.append(gamete_label)
        return idx

    # ---------- query API ----------
    def num_genotypes(self) -> int:
        """Return the number of registered diploid genotypes.

        Returns:
            int: Count of registered diploid genotypes.
        """
        return len(self.index_to_genotype)

    def compress(
        self,
        ztype_mask: NDArray[np.int32],
        gtype_mask: NDArray[np.int32],
        n_slabs: int = 1,
    ) -> None:
        """Permanently remove pruned genotypes and haplotypes from the registry.

        After compression, lookups for pruned entries raise ``KeyError``
        naturally — no special guard code needed.  Both masks use -1 for
        pruned entries.

        Args:
            ztype_mask: ``(G_orig * n_slabs,)`` int32 array — ZType-level
                compression mask (-1 = pruned).  When *n_slabs* > 1 each
                genotype has *n_slabs* entries; all slab variants of a
                genotype share the same fate so we check slab 0.
            gtype_mask: ``(HL,)`` int32 array — haplotype-level
                compression mask (-1 = pruned).
            n_slabs: Number of somatic labels (default 1).
        """
        n_z = self._compress_genotypes(ztype_mask, n_slabs)
        self._compress_haplotypes(gtype_mask)
        self.n_ztypes = n_z

    def _compress_genotypes(
        self, ztype_mask: NDArray[np.int32], n_slabs: int
    ) -> int:
        _z_full = ztype_mask >= 0
        _z_active = _z_full[::n_slabs] if n_slabs > 1 else _z_full
        n_z = int(_z_active.sum())

        old_to_new: dict[int, int] = {
            int(old): new
            for new, old in enumerate(np.where(_z_active)[0])
        }

        self.index_to_genotype = [
            gt
            for i, gt in enumerate(self.index_to_genotype)
            if i < len(_z_active) and _z_active[i]
        ]

        new_dict: dict[Genotype, int] = _UnorderedGenotypeDict()
        for genotype, old_idx in list(self.genotype_to_index.items()):
            new_idx = old_to_new.get(old_idx)
            if new_idx is not None:
                new_dict[genotype] = new_idx
        self.genotype_to_index = new_dict
        return n_z

    def _compress_haplotypes(self, gtype_mask: NDArray[np.int32]) -> None:
        _g_active = gtype_mask >= 0

        old_to_new: dict[int, int] = {
            int(old): new
            for new, old in enumerate(np.where(_g_active)[0])
        }

        self.index_to_haplo = [
            hg
            for i, hg in enumerate(self.index_to_haplo)
            if i < len(_g_active) and _g_active[i]
        ]

        new_dict: dict[HaploidGenotype, int] = {}
        for haplo, old_idx in list(self.haplo_to_index.items()):
            new_idx = old_to_new.get(old_idx)
            if new_idx is not None:
                new_dict[haplo] = new_idx
        self.haplo_to_index = new_dict

    def update_n_ztypes(self, n: int) -> None:  # pragma: no cover
        """Deprecated: use :meth:`compress` instead.

        Only kept for backward compatibility during migration.
        """
        import warnings
        warnings.warn(
            "update_n_ztypes is deprecated — use registry.compress(ztype_mask, gtype_mask)",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_ztypes = n

    def num_haplogenotypes(self) -> int:
        """Return the number of registered haploid genotypes.

        Returns:
            int: Count of registered haploid genotypes.
        """
        return len(self.index_to_haplo)

    def num_gamete_labels(self) -> int:
        """Return the number of registered gamete labels.

        Returns:
            int: Count of registered gamete labels.
        """
        return len(self.index_to_glab)

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
        if isinstance(genotype_id, Genotype):
            # Only canonicalize unordered species — sex chromosomes
            # require maternal/paternal ordering (X|Y ≠ Y|X).
            if genotype_id.species.unordered:
                genotype_id = genotype_id.species.unordered_genotype(
                    genotype_id.maternal, genotype_id.paternal,
                )
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

    def register_somatic_label(self, somatic_label: str) -> int:
        """Register a somatic label (slab) and return its index.

        Symmetric with :meth:`register_gamete_label`.

        Args:
            somatic_label: String label for somatic state.

        Returns:
            int: Assigned integer index for the somatic label.
        """
        if somatic_label in self.slab_to_index:
            return self.slab_to_index[somatic_label]
        idx = len(self.index_to_slab)
        self.slab_to_index[somatic_label] = idx
        self.index_to_slab.append(somatic_label)
        return idx

    def num_somatic_labels(self) -> int:
        """Return the number of registered somatic labels."""
        return len(self.index_to_slab)

    def somatic_label_index(self, somatic_label: str) -> int:
        """Return the index for a registered somatic label key.

        Raises:
            KeyError: If the somatic label is not registered.
        """
        return self.slab_to_index[somatic_label]

    # ---------- helpers ----------
    def _ensure_genotype_index(self, genotype_or_index: Genotype | int) -> int:
        """Convert a genotype or integer index to a valid registry index."""
        if isinstance(genotype_or_index, int):
            if 0 <= genotype_or_index < len(self.index_to_genotype):
                return int(genotype_or_index)
            raise IndexError(f"Genotype index {genotype_or_index} out of range")
        assert not isinstance(genotype_or_index, int)
        return self.register_genotype(genotype_or_index)

    def _ensure_haplo_index(self, haplo_or_index: Union[Any, int]) -> int:
        """Convert a haplogenotype selector to an integer index.

        Behaves similarly to :meth:`_ensure_genotype_index`.

        Args:
            haplo_or_index: Either an int index or a haplogenotype key.

        Returns:
            int: A valid haplogenotype index.
        """
        if isinstance(haplo_or_index, int) and 0 <= haplo_or_index < len(self.index_to_haplo):
            return int(haplo_or_index)
        return self.register_haplogenotype(haplo_or_index)

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
        if isinstance(glab_or_index, int) and 0 <= glab_or_index < len(self.index_to_glab):
            return int(glab_or_index)
        assert isinstance(glab_or_index, str), f"Gamete label must be a string or int index, got {type(glab_or_index)}"
        return self.register_gamete_label(glab_or_index)

    # ---------- helpers for compressed hg+glab indexing ----------
    @staticmethod
    def compress_hg_glab(hg_idx: int, glab_idx: int, n_glabs: int) -> int:
        """Compress a (haplogenotype, glab) pair into a single integer.

        See :func:`compress_hg_glab` for details.

        Args:
            hg_idx: Haplogenotype index.
            glab_idx: Gamete-label index.
            n_glabs: Number of distinct gamete labels.

        Returns:
            int: The compressed combined index.
        """
        return compress_hg_glab(hg_idx, glab_idx, n_glabs)

    @staticmethod
    def decompress_hg_glab(compressed_idx: int, n_glabs: int) -> Tuple[int, int]:
        """Decompress a combined hg+glab index back into its components.

        See :func:`decompress_hg_glab` for details.

        Args:
            compressed_idx: The compressed integer index.
            n_glabs: Number of distinct gamete labels used during compression.

        Returns:
            Tuple[int, int]: ``(hg_idx, glab_idx)`` unpacked from ``compressed_idx``.
        """
        return decompress_hg_glab(compressed_idx, n_glabs)

    def num_hg_glabs(self, n_glabs: int, n_hg: Optional[int] = None) -> int:
        """Return the product of haplogenotype count and gamete-label count.

        Args:
            n_glabs: Number of gamete labels.
            n_hg: Optional number of haplogenotypes. If None the currently
                registered haplogenotype count is used.

        Returns:
            int: The product ``n_hg * n_glabs``.
        """
        if n_hg is None:
            n_hg = self.num_haplogenotypes()
        return int(n_hg) * int(n_glabs)

    # ---------- resolver helpers (centralized key parsing) ----------
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
                return self.decompress_hg_glab(part, n_glabs)
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
            assert isinstance(idx_hg, int) and isinstance(glab_idx, int), "Resolved indices must be integers"
            return self.compress_hg_glab(idx_hg, glab_idx, n_glabs)

        # HaploidGenotype -> default glab 0
        if isinstance(comp_key, HaploidGenotype):
            try:
                idx_hg = int(haploid_genotypes.index(comp_key))
            except ValueError:
                if strict:
                    raise KeyError(f"Unknown haploid object: {comp_key}") from ValueError
                return None
            return self.compress_hg_glab(idx_hg, 0, n_glabs)

        # string -> match to_string
        if isinstance(comp_key, str):
            for i, hg in enumerate(haploid_genotypes):
                try:
                    if hasattr(hg, "to_string") and hg.to_string() == comp_key:
                        return self.compress_hg_glab(i, 0, n_glabs)
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


# ---------------------------------------------------------------------------
# Genotype index helpers — symmetric with compress_hg_glab / decompress_hg_glab
# ---------------------------------------------------------------------------
# Both axes use pure stride arithmetic.
#
#   gamete:   hg × glab       →  hg * n_glabs + glab          (pure stride)
#   genotype: g_orig × slab   →  g_orig * n_slabs + slab      (pure stride)
#
# Compression (BFS reachability pruning) produces masks that map original
# indices → compressed indices (or -1 for pruned).  These masks are cached
# on IndexRegistry (per-config, not per-species) rather than on any config
# type, because they are index metadata — not population parameters.  See
# cache_compression_masks / get_cached_compression_masks below.


@njit_switch(cache=True)
def compress_genotype_index(
    g_orig: int,
    slab: int,
    n_slabs: int,
) -> int:
    """Flatten ``(g_orig, slab)`` → integer index via pure stride arithmetic.

    Symmetric to ``compress_hg_glab(hg, glab, n_glabs)``.
    Compression (maternal/paternal symmetry, unreachable pairs) is applied
    separately via ``genotype_compression_mask`` on PopulationConfig.
    """
    return g_orig * n_slabs + slab


@njit_switch(cache=True)
def decompress_genotype_index(
    flat_idx: int,
    n_slabs: int,
) -> tuple[int, int]:
    """Decompose a flat index → ``(g_comp, slab)``.

    Symmetric to ``decompress_hg_glab(idx, n_glabs)``.
    """
    return flat_idx // n_slabs, flat_idx % n_slabs


# ---------------------------------------------------------------------------
# Compression mask cache
# ---------------------------------------------------------------------------
# Compression masks map original indices → compressed indices (-1 = pruned).
# They are computed by build_gamete_compression_mask() (BFS reachability)
# and cached here — NOT on PopulationConfig nor DiscretePopulationConfig —
# because they are index metadata, not population parameters.
#
# Why store them at all?  BFS is O(G² × HL²) worst-case.  Re-running on
# every modifier change is wasteful when the modifier only changes
# probability values (not reachability).  However, a modifier that drives
# a probability to zero can make previously reachable genotypes
# unreachable, invalidating cached masks.
#
# Default policy: refresh_modifier_maps() re-runs BFS unconditionally
# (guaranteed correctness).  Pass skip_compression_bfs=True to reuse
# cached masks when reachability is known to be stable.
#
# Stored per-Species (keyed by id(species)) because masks vary by initial
# state, not just genotype space — different Population objects for the
# same Species can produce different reachable sets.  Use
# clear_compression_masks(species) if building a new config for the same
# Species with a different initial state.

_compression_cache: dict[int, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}


def cache_compression_masks(
    species: object,
    gtype_mask: NDArray[np.int32],
    ztype_mask: NDArray[np.int32],
) -> None:
    """Cache BFS-derived compression masks keyed by Species identity.

    Called by ``_rebuild_config_maps`` and ``refresh_modifier_maps``
    after computing fresh masks.  The caller is responsible for ensuring
    the masks are still valid before reusing them.

    Args:
        species: The Species whose genotype space was compressed.
        gtype_mask: (HL,) int32 array — -1 = pruned, else compressed index.
        ztype_mask: (G_orig × n_slabs,) int32 — -1 = pruned.
    """
    _compression_cache[id(species)] = (gtype_mask, ztype_mask)


def get_cached_compression_masks(
    species: object,
) -> Optional[tuple[NDArray[np.int32], NDArray[np.int32]]]:
    """Return cached masks for *species*, or None if never computed.

    The caller should validate that the masks are still current for the
    maps being compressed.  By default ``refresh_modifier_maps`` re-runs
    BFS and refreshes the cache rather than trusting stale masks.
    """
    return _compression_cache.get(id(species))


def clear_compression_masks(species: object) -> None:
    """Discard cached masks — call when reachability may have changed.

    For example, after building a new config with a different initial
    state for the same Species, or after a modifier change that is known
    to alter reachability.
    """
    _compression_cache.pop(id(species), None)
