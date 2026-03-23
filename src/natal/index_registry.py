from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from natal.genetic_entities import Genotype, HaploidGenotype

import numpy as np

from natal.numba_utils import njit_switch

# TODO: 性染色体相关基因的支持。如果设置了性染色体，genotype to index 在雌雄中不同
# TODO: 当然，可以有空缺

class IndexRegistry:
    """Registry providing stable integer indices for population entities.

    The IndexRegistry assigns and stores stable integer indices for entities that
    occur in the population: diploid genotypes, haploid genotypes (haplogenotypes),
    and gamete labels. It exposes small helper methods to export index arrays and
    to resolve flexible selector types (objects, ints, or string keys) into
    numeric indices suitable for numeric backends.

    Example:


        ic = IndexRegistry()
        gid = ic.register_genotype('g1')
        hid = ic.register_haplogenotype('h1')
        glid = ic.register_gamete_label('gl1')

    Args:
        (no arguments)

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
        self.genotype_to_index: Dict[Any, int] = {}
        self.index_to_genotype: List[Any] = []

        self.haplo_to_index: Dict[Any, int] = {}
        self.index_to_haplo: List[Any] = []

        self.glab_to_index: Dict[Any, int] = {}
        self.index_to_glab: List[Any] = []

        # axis sizes for compatibility (not used for numeric flattening)
        self.axis_sizes: Dict[str, int] = {}

    # ---------- registration API ----------
    def register_genotype(self, genotype_id: Any) -> int:
        """Register a genotype and return its stable integer index.

        If the genotype key is already present the existing index is returned.

        Args:
            genotype_id: A genotype instance or an opaque identifier. The
                provided object is used as the canonical registry key.

        Returns:
            int: The assigned integer index for the genotype. Indices remain
            stable until :meth:`compact` is called.
        """
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
        return len(self.index_to_genotype)

    def num_haplogenotypes(self) -> int:
        return len(self.index_to_haplo)

    def num_gamete_labels(self) -> int:
        return len(self.index_to_glab)

    def genotype_index(self, genotype_id: Any) -> int:
        """Return the index for a registered genotype key.

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

    # ---------- helpers ----------
    def _ensure_genotype_index(self, genotype_or_index: Union[Any, int]) -> int:
        """Convert a genotype selector to an integer index.

        If the input is an integer within the current valid range it is
        returned. Otherwise the input is registered as a new genotype key and
        its newly assigned index is returned.

        Args:
            genotype_or_index: Either an int index or a genotype key.

        Returns:
            int: A valid genotype index.
        """
        if isinstance(genotype_or_index, int) and 0 <= genotype_or_index < len(self.index_to_genotype):
            return int(genotype_or_index)
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

    def _ensure_glab_index(self, glab_or_index: Union[Any, int]) -> int:
        """Convert a gamete-label selector to an integer index.

        Behaves similarly to :meth:`_ensure_genotype_index`.

        Args:
            glab_or_index: Either an int index or a gamete-label key.

        Returns:
            int: A valid gamete-label index.
        """
        if isinstance(glab_or_index, int) and 0 <= glab_or_index < len(self.index_to_glab):
            return int(glab_or_index)
        return self.register_gamete_label(glab_or_index)

    # ---------- helpers for compressed hg+glab indexing ----------
    @staticmethod
    def compress_hg_glab(hg_idx: int, glab_idx: int, n_glabs: int) -> int:
        """Compress a (haplogenotype, glab) pair into a single integer.
        """
        return compress_hg_glab(hg_idx, glab_idx, n_glabs)
    
    @staticmethod
    def decompress_hg_glab(compressed_idx: int, n_glabs: int) -> Tuple[int, int]:
        """Decompress a combined hg+glab index back into its components.
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
    def resolve_genotype_index(self, diploid_genotypes: Sequence[Any], gk: Any, strict: bool = False) -> Optional[int]:
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

        # string match via to_string()
        if isinstance(gk, str):
            for i, g in enumerate(diploid_genotypes):
                try:
                    if hasattr(g, "to_string") and g.to_string() == gk:
                        return i
                except Exception:
                    continue

        if strict:
            raise KeyError(f"Cannot resolve genotype key: {gk}")
        return None

    def resolve_hg_glab_part(self, haploid_genotypes: Sequence[Any], part: Any, n_glabs: int, strict: bool = False) -> Optional[Tuple[int, int]]:
        """Resolve a haploid/genetic part into an (hg_idx, glab_idx) pair.

        Accepted input formats for ``part``:
            - (int, int): already (hg_idx, glab_idx)
            - (HaploidGenotype, glab): where ``glab`` is int or string label
            - HaploidGenotype object: maps to (idx, 0)
            - int: treated as compressed index and decompressed
            - str: matched against ``haploid.to_string()`` and returns (idx, 0)

        Args:
            haploid_genotypes: Sequence of haploid genotype objects.
            part: The flexible selector to resolve.
            n_glabs: Number of gamete labels (used for decompression).
            strict: If True raise KeyError on failure, otherwise return None.

        Returns:
            Optional[Tuple[int, int]]: The resolved (hg_idx, glab_idx) pair or
            None when unresolved and ``strict`` is False.

        Raises:
            KeyError: If resolution fails and ``strict`` is True.
        """
        # tuple of ints (already decompressed)
        if isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], int) and isinstance(part[1], int):
            return (int(part[0]), int(part[1]))

        # (HaploidGenotype, glab)
        # (str, glab) where first element is haploid string representation
        if isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], str):
            name, lab = part
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
                if strict:
                    raise KeyError(f"Unknown haploid string: {name}")
                return None

            if isinstance(lab, int):
                glab_idx = int(lab)
            else:
                glab_idx = self.glab_to_index.get(str(lab))
                if glab_idx is None:
                    if strict:
                        raise KeyError(f"Unknown glab label: {lab}")
                    return None
            return (found_idx, glab_idx)

        # (HaploidGenotype, glab)
        if isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], HaploidGenotype):
            hg_obj, lab = part
            try:
                idx_hg = int(haploid_genotypes.index(hg_obj))
            except ValueError:
                if strict:
                    raise KeyError(f"Unknown haploid object: {hg_obj}")
                return None

            if isinstance(lab, int):
                glab_idx = int(lab)
            else:
                glab_idx = self.glab_to_index.get(str(lab))
                if glab_idx is None:
                    if strict:
                        raise KeyError(f"Unknown glab label: {lab}")
                    return None
            return (idx_hg, glab_idx)

        # HaploidGenotype object -> default glab 0
        if isinstance(part, HaploidGenotype):
            try:
                return (int(haploid_genotypes.index(part)), 0)
            except ValueError:
                if strict:
                    raise KeyError(f"Unknown haploid object: {part}")
                return None

        # compressed integer
        if isinstance(part, int):
            try:
                return self.decompress_hg_glab(part, n_glabs)
            except Exception:
                if strict:
                    raise
                return None

        # string matching to_string()
        if isinstance(part, str):
            for i, hg in enumerate(haploid_genotypes):
                try:
                    if hasattr(hg, "to_string") and hg.to_string() == part:
                        return (i, 0)
                except Exception:
                    continue

        if strict:
            raise KeyError(f"Cannot resolve hg+glab part: {part}")
        return None

    def resolve_comp_idx(self, haploid_genotypes: Sequence[Any], n_glabs: int, comp_key: Any, strict: bool = False) -> Optional[int]:
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
        if isinstance(comp_key, tuple) and len(comp_key) == 2:
            part_hg, part_glab = comp_key
            # resolve hg part
            if isinstance(part_hg, int):
                idx_hg = int(part_hg)
            elif isinstance(part_hg, HaploidGenotype):
                try:
                    idx_hg = int(haploid_genotypes.index(part_hg))
                except ValueError:
                    if strict:
                        raise KeyError(f"Cannot resolve haploid object: {part_hg}")
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

            return self.compress_hg_glab(idx_hg, glab_idx, n_glabs)

        # HaploidGenotype -> default glab 0
        if isinstance(comp_key, HaploidGenotype):
            try:
                idx_hg = int(haploid_genotypes.index(comp_key))
            except ValueError:
                if strict:
                    raise KeyError(f"Unknown haploid object: {comp_key}")
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
            raise KeyError(f"Unsupported comp_map key type: {type(comp_key)}")
        return None

    # ---------- maintenance ----------
    def compact(self) -> Dict[int, int]:
        """Reassign genotype indices densely and return an old->new map.

        This operation may change previously-assigned stable indices. Callers
        that keep external references to genotype indices must apply the
        returned mapping to update those references.

        Returns:
            Dict[int, int]: Mapping from old genotype index to new index.
        """
        # compact genotypes
        old_to_new_g = {old: new for new, old in enumerate(range(self.num_genotypes()))}
        # currently genotypes are dense so mapping is identity; placeholder
        return old_to_new_g

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