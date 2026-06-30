"""Modifier system for population simulations.

This module defines protocols and helper functions for constructing and
wrapping modifiers that alter gamete or zygote production in the simulation.
Modifiers are callable objects that return frequency distributions, and are
converted into tensor‑level functions that directly update NumPy arrays.

Two modifier types are supported:
- Gamete modifiers: alter the mapping from (sex, diploid genotype) to
  haploid gamete frequencies.
- Zygote modifiers: alter the mapping from a pair of haploid gametes
  (with gamete labels) to a diploid zygote genotype.

The wrapper factories (`wrap_gamete_modifier`, `wrap_zygote_modifier`) take
high‑level modifiers that return domain‑object dictionaries and produce
callables that operate on NumPy tensors.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeGuard,
    Union,
    cast,
)

import numpy as np

from natal.genetic_entities import Genotype, HaploidGenotype
from natal.helpers import resolve_sex_label

GenotypeFilter = Optional[Union[Callable[[Genotype], bool], str]]
GlabSelector = Optional[Union[str, int]]

# Bulk-only modifier interface expectations (strict form):
# - gamete modifier: callable() -> Dict[(sex_idx:int, genotype_idx:int) -> Dict[compressed_hg_glab_idx:int -> freq:float]]
# - zygote modifier: callable() -> Dict[(c1:int, c2:int) -> replacement]
#
# The modifiers use compressed integer indices as keys so that outputs can be
# written back directly into underlying numeric tensors. This avoids expensive
# object-to-index lookups inside wrappers and prevents passing large object
# graphs at runtime.

class GameteModifier(Protocol):
    """Protocol for a bulk gamete modifier.

    Implementations should provide a callable that accepts either zero or one
    argument (an optional `population` object) and returns a nested mapping of
    gamete frequency updates. The canonical return type is::

        Dict[Tuple[int, int], Dict[int, float]]

    where the outer key is ``(sex_idx, genotype_idx)`` and the inner mapping is
    ``{ compressed_hg_glab_idx: frequency, ... }``. Keys may be flexible types
    in wrappers (for convenience) but should ultimately resolve to integers.

    ``sex_idx`` is an ``int``. ``genotype_idx`` may be an ``int``, a
    ``Genotype`` object, or a string produced by ``Genotype.to_string()``.

    Examples:
        return {(0, 5): {3: 0.2, 4: 0.8}, (1, 5): {3: 1.0}}

    The result writes frequency distributions for compressed indices directly
    back into numeric tensors.
    """
    def __call__(self, *args: object, **kwargs: object) -> Mapping[Any, Mapping[Any, float]]: ...


class ZygoteModifier(Protocol):
    """Protocol for a bulk zygote modifier.

    Implementations should provide a callable that accepts zero or one argument
    (an optional `population`) and returns a mapping from a flexible key to a
    replacement. The key identifies the zygote pairing and may take one of
    several forms that wrappers can resolve into compressed coordinate pairs
    ``(c1, c2)``.

    Supported key representations include:
        - compressed index pair ``(c1, c2)``
        - nested tuples ``((hg_obj|hg_str|idx_hg, glab_label?), (hg_obj|hg_str|idx_hg, glab_label?))``
        - other wrapper-resolvable representations

    Replacement values may be one of:
        - an integer index ``idx_modified`` (index into diploid genotype list)
        - a ``Genotype`` instance (wrappers will convert to an index)
        - a dict ``{ idx_modified: probability, ... }`` specifying a distribution

    The protocol returns::

        Dict[Any, Union[int, Genotype, Dict[int, float]]]
    """
    def __call__(self, *args: object, **kwargs: object) -> Mapping[Any, Union[int, Genotype, Mapping[Any, float]]]: ...


# ============================================================================
# HELPER FUNCTIONS FOR MODIFIER CONSTRUCTION
# ============================================================================

def _invoke_modifier(mod: Callable[..., Any], population: Any = None) -> object:
    """Invoke a modifier callable, supporting both 0-arg and 1-arg signatures.

    Args:
        mod: The modifier callable.
        population: Optional population object to pass if the modifier accepts one.

    Returns:
        The dict returned by the modifier.
    """
    sig = inspect.signature(mod)
    if len(sig.parameters) == 0:
        return mod()
    else:
        return mod(population)


def _resolve_sex_name(key: str) -> Optional[int]:
    """Normalize string sex names to sex index.

    Returns None for unknown keys.
    """
    try:
        return resolve_sex_label(key)
    except (TypeError, ValueError):
        return None


def evaluate_genotype_filter(
    genotype_filter: GenotypeFilter,
    genotype: Genotype,
    compiled_filter: Optional[Callable[[Genotype], bool]],
) -> Tuple[bool, Optional[Callable[[Genotype], bool]]]:
    """Evaluate genotype_filter and lazily compile pattern-string filters.

    The function supports three filter forms:
    - ``None``: always pass
    - callable: evaluate directly
    - string pattern: compile once via ``GenotypePatternParser`` then reuse
    """
    if genotype_filter is None:
        return True, compiled_filter

    if callable(genotype_filter):
        return genotype_filter(genotype), compiled_filter

    if compiled_filter is None:
        from natal.genetic_patterns import GenotypePatternParser
        try:
            pattern = GenotypePatternParser(genotype.species).parse(genotype_filter)
        except Exception as exc:
            raise ValueError(
                f"Invalid genotype_filter pattern: {genotype_filter}"
            ) from exc
        compiled_filter = pattern.to_filter()
    return compiled_filter(genotype), compiled_filter

    raise TypeError("genotype_filter must be a callable, pattern string, or None")


# ============================================================================
# TENSOR-LEVEL WRAPPER FACTORIES
# ============================================================================
# These functions wrap high-level modifiers (returning dicts of domain objects)
# into tensor-level callables that accept/return NumPy arrays. They encapsulate
# the key-parsing and index-resolution logic so that both base_population and
# external modifier systems (e.g. gamete_allele_conversion) can reuse them.


def _resolve_gidx(
    gk: object,
    diploid_genotypes: List[Genotype],
    index_registry: Any,
) -> List[int]:
    """Resolve a flexible genotype key directly to ZType indices.

    Accepted key forms:
        - int: genotype index (0-based)
        - Genotype object: matched by identity in ``diploid_genotypes``
        - str: compared against ``genotype.to_string()``

    Returns all ZType (genotype × slab) indices for the resolved genotype.

    Raises:
        IndexError: integer key is out of range.
        ValueError: object key is not found in the list.
        KeyError: string key cannot be resolved.
    """
    if isinstance(gk, int):
        gidx: int = int(gk)
    elif not isinstance(gk, (str,)):
        gidx = list(diploid_genotypes).index(cast(Genotype, gk))
    else:
        for i, g in enumerate(diploid_genotypes):
            if hasattr(g, "to_string") and g.to_string() == gk:
                gidx = i
                break
        else:
            raise KeyError(f"Cannot resolve genotype key: {gk}")
    gt = diploid_genotypes[gidx]
    return index_registry.ztype_indices_for(gt)


def wrap_gamete_modifier(
    mod: GameteModifier,
    population: Any,
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int,
    expand_to_ztypes: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level GameteModifier into a tensor-level callable.

    The returned callable accepts a tensor of shape (n_sexes, n_ztypes, n_hg_glabs)
    and returns a modified copy.

    Args:
        mod: A GameteModifier callable (returns dict mapping keys to freq dicts).
        population: The population object (passed to mod if it takes an argument).
        index_registry: IndexRegistry instance for key resolution.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.
        expand_to_ztypes: If True, expand resolved genotype indices to all
            ZType (genotype×slab) indices before writing.  Required when the
            tensor's genotype axis is pre-expanded (G×S).

    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()
        n_sexes, n_ztypes, n_hg_glabs = modified.shape

        bulk_obj = _invoke_modifier(mod, population)

        if not isinstance(bulk_obj, Mapping):
            raise TypeError("Gamete modifier must return a mapping from keys to compressed-index->freq mappings")
        bulk = cast(Mapping[object, object], bulk_obj)

        for key, val in bulk.items():
            # Case A: top-level sex-name ('male'/'female')
            sex_idx = _resolve_sex_name(key) if isinstance(key, str) else None
            if sex_idx is not None and isinstance(val, Mapping):
                sex_val = cast(Mapping[object, object], val)
                for gk, comp_map in sex_val.items():
                    try:
                        ztype_indices = _resolve_gidx(gk, diploid_genotypes, index_registry)
                    except (KeyError, IndexError, ValueError):
                        continue
                    if not (0 <= sex_idx < n_sexes):
                        continue
                    if isinstance(comp_map, Mapping):
                        for zidx in ztype_indices:
                            if 0 <= zidx < n_ztypes:
                                _apply_comp_map(
                                    modified,
                                    sex_idx,
                                    zidx,
                                    cast(Mapping[object, object], comp_map),
                                    index_registry,
                                    haploid_genotypes,
                                    n_glabs,
                                    n_hg_glabs,
                                )
                continue

            # Case B: explicit (sex_idx, genotype_key) tuple
            key_tuple = _as_pair(key)
            if key_tuple is not None:
                sex_obj, gk = key_tuple
                if not isinstance(sex_obj, int):
                    continue
                sex_idx = sex_obj
                if not (0 <= sex_idx < n_sexes):
                    continue
                try:
                    for zidx in _resolve_gidx(gk, diploid_genotypes, index_registry):
                        if 0 <= zidx < n_ztypes:
                            _apply_comp_map(
                                modified,
                                sex_idx,
                                zidx,
                                cast(Mapping[object, object], val),
                                index_registry,
                                haploid_genotypes,
                                n_glabs,
                                n_hg_glabs,
                            )
                except (KeyError, IndexError, ValueError):
                    continue
                continue

            # Case C: key is genotype_key applied to all sexes
            try:
                ztype_indices = _resolve_gidx(key, diploid_genotypes, index_registry)
            except (KeyError, IndexError, ValueError):
                continue
            if not isinstance(val, Mapping):
                continue
            for sex_idx in range(n_sexes):
                for zidx in ztype_indices:
                    if 0 <= zidx < n_ztypes:
                        _apply_comp_map(
                            modified,
                            sex_idx,
                            zidx,
                            cast(Mapping[object, object], val),
                            index_registry,
                            haploid_genotypes,
                            n_glabs,
                            n_hg_glabs,
                        )

        return modified
    return tensor_modifier


def wrap_zygote_modifier(
    mod: ZygoteModifier,
    population: Any,
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int,
    expand_to_ztypes: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level ZygoteModifier into a tensor-level callable.

    The returned callable accepts a tensor of shape (n_hg_glabs, n_hg_glabs, n_ztypes)
    and returns a modified copy.

    Args:
        mod: A ZygoteModifier callable.
        population: The population object (passed to mod if it takes an argument).
        index_registry: IndexRegistry instance for key resolution.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.
        expand_to_ztypes: If True, expand resolved genotype indices to all
            ZType (genotype×slab) indices before writing.  Required when the
            tensor's genotype axis is pre-expanded (G×S).

    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()

        bulk_obj = _invoke_modifier(mod, population)

        if not isinstance(bulk_obj, Mapping):
            raise TypeError("Zygote modifier must return a mapping from keys to replacements")
        bulk = cast(Mapping[object, object], bulk_obj)

        for key, val in bulk.items():
            c1, c2 = _parse_zygote_key(key, index_registry, haploid_genotypes, n_glabs)
            mapping = _normalize_zygote_val(val, index_registry, diploid_genotypes)
            _write_zygote_mapping(modified, c1, c2, mapping)

        return modified
    return tensor_modifier


def build_modifier_wrappers(
    gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]],
    zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]],
    population: Any,
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    expand_to_ztypes: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[Callable[[np.ndarray], np.ndarray]]]:
    """Wrap high-level gamete/zygote modifiers into tensor-level callables.

    This is the shared implementation used by BasePopulation and any external
    modifier systems that need to convert high-level modifiers to tensor ops.

    Args:
        gamete_modifiers: List of (modifier_id, name, modifier) tuples for gamete modifiers.
        zygote_modifiers: List of (modifier_id, name, modifier) tuples for zygote modifiers.
        population: The population object.
        index_registry: IndexRegistry instance.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.
        expand_to_ztypes: If True, expand genotype indices to ZType indices
            in both gamete and zygote wrappers.  Use when the tensor's
            genotype axis is pre-expanded (G×S).

    Returns:
        Tuple of (gamete_modifier_funcs, zygote_modifier_funcs), each a list
        of callables that accept and return NumPy tensors.
    """
    gamete_modifier_funcs: List[Callable[[np.ndarray], np.ndarray]] = []
    zygote_modifier_funcs: List[Callable[[np.ndarray], np.ndarray]] = []

    for _, _, mod in zygote_modifiers:
        zygote_modifier_funcs.append(
            wrap_zygote_modifier(mod, population, index_registry, haploid_genotypes, diploid_genotypes, n_glabs, expand_to_ztypes=expand_to_ztypes)
        )

    for _, _, mod in gamete_modifiers:
        gamete_modifier_funcs.append(
            wrap_gamete_modifier(mod, population, index_registry, haploid_genotypes, diploid_genotypes, n_glabs, expand_to_ztypes=expand_to_ztypes)
        )

    return gamete_modifier_funcs, zygote_modifier_funcs


# ============================================================================
# INTERNAL HELPERS (used by the wrapper factories above)
# ============================================================================

def _apply_comp_map(
    modified: np.ndarray,
    sex_idx: int,
    zidx: int,
    comp_map: Mapping[object, object],
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int,
    n_hg_glabs: int,
) -> None:
    """Apply a comp_map (comp_key->freq) into the tensor slice [sex_idx, zidx].

    Args:
        modified: The target tensor (n_sexes, n_ztypes, n_hg_glabs).
        sex_idx: Sex index.
        zidx: ZType index (genotype × slab position).
        comp_map: Mapping from compressed‑key to frequency.
        index_registry: IndexRegistry for resolving keys.
        haploid_genotypes: List of all haploid genotypes.
        n_glabs: Number of gamete‑label variants.
        n_hg_glabs: Total number of compressed haploid entries.
    """
    modified[sex_idx, zidx, :] = 0.0
    for comp_key, freq in comp_map.items():
        if not isinstance(freq, (int, float)):
            continue
        # Resolve comp_key to a compressed GType index using gtype_index.
        if isinstance(comp_key, int):
            comp_idx = int(comp_key)
        else:
            pair = _as_pair(comp_key)
            if pair is not None:
                hg_part: object = pair[0]
                glab_part: object = pair[1]
                # resolve hg_part → HaploidGenotype object
                if isinstance(hg_part, int):
                    hg_obj: HaploidGenotype = haploid_genotypes[hg_part]
                elif isinstance(hg_part, HaploidGenotype):
                    hg_obj = hg_part
                elif isinstance(hg_part, str):
                    found: Optional[HaploidGenotype] = None
                    for hg in haploid_genotypes:
                        if hasattr(hg, "to_string") and hg.to_string() == hg_part:
                            found = hg
                            break
                        try:
                            if str(hg) == hg_part:
                                found = hg
                                break
                        except Exception:
                            continue
                    if found is None:
                        continue
                    hg_obj = found
                else:
                    continue
                # resolve glab_part → string label
                if isinstance(glab_part, int):
                    glab_str = index_registry.glab_labels[glab_part]
                else:
                    glab_str = str(glab_part)
                try:
                    comp_idx = index_registry.gtype_index(hg_obj, glab_str)
                except KeyError:
                    continue
            elif isinstance(comp_key, HaploidGenotype):
                try:
                    comp_idx = index_registry.gtype_index(comp_key, "default")
                except KeyError:
                    continue
            elif isinstance(comp_key, str):
                found = None
                for hg in haploid_genotypes:
                    if hasattr(hg, "to_string") and hg.to_string() == comp_key:
                        found = hg
                        break
                    try:
                        if str(hg) == comp_key:
                            found = hg
                            break
                    except Exception:
                        continue
                if found is None:
                    continue
                try:
                    comp_idx = index_registry.gtype_index(found, "default")
                except KeyError:
                    continue
            else:
                continue
        if not (0 <= comp_idx < n_hg_glabs):
            continue
        modified[sex_idx, zidx, comp_idx] = float(freq)


def _find_haploid_by_name(
    name: str,
    haploid_genotypes: List[HaploidGenotype],
) -> HaploidGenotype:
    """Find a HaploidGenotype by to_string() or str() match.

    Raises:
        KeyError: If no haploid genotype matches the name.
    """
    for hg in haploid_genotypes:
        if hasattr(hg, "to_string") and hg.to_string() == name:
            return hg
        try:
            if str(hg) == name:
                return hg
        except Exception:
            continue
    raise KeyError(f"Unknown haploid string: {name}")


def _resolve_part_to_compressed(
    part: object,
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int,
) -> int:
    """Resolve a single zygote-key part to a compressed GType index.

    Accepted part forms:
        - int: returned as-is (already a compressed index)
        - (int, int): (hg_idx, glab_idx) pair, compressed via formula
        - (HaploidGenotype, int/str): resolved via gtype_index
        - (str, int/str): hg found by to_string(), then gtype_index
        - HaploidGenotype: gtype_index(hg, "default")
        - str: hg found by to_string(), then gtype_index(hg, "default")

    Raises:
        KeyError: If the part cannot be resolved.
    """
    if isinstance(part, int):
        return int(part)

    pair = _as_pair(part)
    if pair is not None:
        hg_part, glab_part = pair
        if isinstance(hg_part, int) and isinstance(glab_part, int):
            return int(hg_part) * n_glabs + int(glab_part)
        if isinstance(hg_part, int):
            hg_obj: HaploidGenotype = haploid_genotypes[hg_part]
        elif isinstance(hg_part, HaploidGenotype):
            hg_obj = hg_part
        elif isinstance(hg_part, str):
            hg_obj = _find_haploid_by_name(hg_part, haploid_genotypes)
        else:
            raise KeyError(f"Cannot resolve haploid part: {hg_part!r}")
        if isinstance(glab_part, int):
            glab_str = index_registry.glab_labels[glab_part]
        else:
            glab_str = str(glab_part)
        return index_registry.gtype_index(hg_obj, glab_str)

    if isinstance(part, HaploidGenotype):
        return index_registry.gtype_index(part, "default")

    if isinstance(part, str):
        hg_obj = _find_haploid_by_name(part, haploid_genotypes)
        return index_registry.gtype_index(hg_obj, "default")

    raise KeyError(f"Cannot resolve part: {part!r}")


def _parse_zygote_key(
    key: Any,
    index_registry: Any,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int,
) -> Tuple[int, int]:
    """Parse modifier key for zygote wrappers into compressed coords (c1, c2).

    Args:
        key: The key from the modifier mapping (e.g. a tuple or pair of keys).
        index_registry: IndexRegistry for resolving haploid/gamete‑label parts.
        haploid_genotypes: List of all haploid genotypes.
        n_glabs: Number of gamete‑label variants.

    Returns:
        A tuple of two compressed indices (c1, c2).
    """
    if _is_int_pair(key):
        return key[0], key[1]
    key_tuple = _as_pair(key)
    if key_tuple is None:
        raise TypeError("Zygote modifier key must be a 2-tuple")
    part1, part2 = key_tuple
    c1 = _resolve_part_to_compressed(part1, index_registry, haploid_genotypes, n_glabs)
    c2 = _resolve_part_to_compressed(part2, index_registry, haploid_genotypes, n_glabs)
    return c1, c2


def _resolve_genotype(candidate: object, diploid_genotypes: List[Genotype]) -> Genotype:
    """Resolve a flexible genotype selector to a Genotype object.

    Accepted selector types:
        - int: genotype index (0-based)
        - Genotype object: matched by identity/equality
        - str: compared against genotype.to_string()

    Raises:
        IndexError: integer index out of range
        ValueError: object not found in list
        KeyError: string key cannot be resolved
    """
    if isinstance(candidate, int):
        return diploid_genotypes[int(candidate)]
    if not isinstance(candidate, str):
        idx = list(diploid_genotypes).index(cast(Genotype, candidate))
        return diploid_genotypes[idx]
    # String match via to_string()
    for g in diploid_genotypes:
        if hasattr(g, "to_string") and g.to_string() == candidate:
            return g
    raise KeyError(f"Cannot resolve genotype key: {candidate}")


def _normalize_zygote_val(
    val: Any,
    index_registry: Any,
    diploid_genotypes: List[Genotype],
) -> Dict[int, float]:
    """Normalize zygote replacement value into a mapping ztype_idx->prob.

    Integer selectors are used directly as ZType indices (the caller is
    expected to already know the correct index).  Non‑integer selectors
    (object / string) are resolved to a Genotype and then expanded to all
    its ZType (genotype × slab) indices.

    Args:
        val: The value from the modifier mapping.
        index_registry: IndexRegistry for resolving genotype indices.
        diploid_genotypes: List of all diploid genotypes.

    Returns:
        Dictionary mapping ZType index to probability.
    """
    mapping: Dict[int, float] = {}

    # single tuple (idx_or_genotype, prob)
    pair_val = _as_idx_prob_pair(val)
    if pair_val is not None:
        idx_candidate, prob = pair_val
        if isinstance(idx_candidate, int):
            mapping[int(idx_candidate)] = float(prob)
        else:
            gt = _resolve_genotype(idx_candidate, diploid_genotypes)
            for zidx in index_registry.ztype_indices_for(gt):
                mapping[int(zidx)] = float(prob)
        return mapping

    # distribution dict
    if isinstance(val, Mapping):
        val_map = cast(Mapping[object, object], val)
        for idx_candidate, prob in val_map.items():
            if not isinstance(prob, (int, float)):
                raise TypeError("Zygote replacement probabilities must be numeric")
            if isinstance(idx_candidate, int):
                mapping[int(idx_candidate)] = float(prob)
            else:
                gt = _resolve_genotype(idx_candidate, diploid_genotypes)
                for zidx in index_registry.ztype_indices_for(gt):
                    mapping[int(zidx)] = float(prob)
        return mapping

    # single genotype replacement
    if isinstance(val, int):
        mapping[int(val)] = 1.0
    else:
        gt = _resolve_genotype(val, diploid_genotypes)
        for zidx in index_registry.ztype_indices_for(gt):
            mapping[int(zidx)] = 1.0
    return mapping


def _write_zygote_mapping(
    modified: np.ndarray,
    c1: int,
    c2: int,
    mapping: Dict[int, float],
) -> None:
    """Apply mapping (idx->prob) to the compressed zygote slice.

    Args:
        modified: The target tensor (n_hg_glabs, n_hg_glabs, n_ztypes).
        c1: Compressed index of first gamete.
        c2: Compressed index of second gamete.
        mapping: Dictionary mapping genotype index to probability.
    """
    modified[c1, c2, :] = 0.0
    for idx_mod, prob in mapping.items():
        modified[c1, c2, int(idx_mod)] = float(prob)


def _as_pair(value: object) -> Optional[Tuple[object, object]]:
    if not isinstance(value, tuple):
        return None
    items = cast(Tuple[object, ...], value)
    if len(items) != 2:
        return None
    return items[0], items[1]


def _is_int_pair(value: object) -> TypeGuard[Tuple[int, int]]:
    pair = _as_pair(value)
    return pair is not None and isinstance(pair[0], int) and isinstance(pair[1], int)


def _as_idx_prob_pair(value: object) -> Optional[Tuple[object, float]]:
    pair = _as_pair(value)
    if pair is None or not isinstance(pair[1], (int, float)):
        return None
    return pair[0], float(pair[1])


# Public aliases for cross-module helper reuse.
apply_comp_map = _apply_comp_map
parse_zygote_key = _parse_zygote_key
normalize_zygote_val = _normalize_zygote_val
write_zygote_mapping = _write_zygote_mapping
