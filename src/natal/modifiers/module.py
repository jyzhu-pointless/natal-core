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
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
)

if TYPE_CHECKING:
    from natal.registry.index import IndexRegistry

import numpy as np

from natal.genetics import Genotype, HaploidGenotype
from natal.utils.helpers import resolve_sex_label

GenotypeFilter = Optional[Union[Callable[[Genotype], bool], str]]
GlabSelector = Optional[Union[str, int]]

# Key types accepted by the unified resolvers.  Ints are pre-resolved
# compressed indices; strings and domain objects are runtime-dispatched.
ZtypeKey: TypeAlias = Union[int, str, Genotype]
GtypeKey: TypeAlias = Union[int, str, HaploidGenotype, Tuple[int, int], Tuple[str, int]]

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
    def __call__(self, *args: object, **kwargs: object) -> Mapping[tuple[int, ZtypeKey], Mapping[GtypeKey, float]]:
        """Call the gamete modifier, returning frequency distributions per sex/ztype."""
        ...


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

        Dict[object, Union[int, Genotype, Dict[int, float]]]
    """
    def __call__(self, *args: object, **kwargs: object) -> Mapping[tuple[int, int], Union[int, Genotype, Mapping[ZtypeKey, float]]]:
        """Call the zygote modifier, returning replacement mappings per gamete pair."""
        ...


# ============================================================================
# HELPER FUNCTIONS FOR MODIFIER CONSTRUCTION
# ============================================================================

def _invoke_modifier(
    mod: Callable[..., object],
    population: object | None = None,
) -> object:
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
        from natal.patterns import GenotypePatternParser
        try:
            pattern = GenotypePatternParser(genotype.species).parse(genotype_filter)
        except Exception as exc:
            raise ValueError(
                f"Invalid genotype_filter pattern: {genotype_filter}"
            ) from exc
        compiled_filter = pattern.to_filter()
    return compiled_filter(genotype), compiled_filter


# ============================================================================
# Unified key resolution — ztype / gtype selectors → numeric indices
# ============================================================================


def _resolve_ztype_key(key: ZtypeKey, registry: IndexRegistry) -> list[int]:
    """Resolve a ztype selector to a list of ztype indices.

    Accepted forms:
        - ``int``: returned as-is (pre-resolved ztype index)
        - ``str``: matched via ``genotype.to_string()``, then expanded
          to all slab ztype indices via ``registry.ztype_indices_for()``
        - ``Genotype``: expanded to all slab ztype indices

    Returns all matching ztype indices.  Callers typically iterate the
    result and write to each index.
    """
    if isinstance(key, int):
        return [key]
    if isinstance(key, str):
        for g in registry.index_to_genotype:
            if hasattr(g, "to_string") and g.to_string() == key:
                return registry.ztype_indices_for(g)
        raise KeyError(f"Cannot resolve zygote type key: {key!r}")
    return registry.ztype_indices_for(key)


def _resolve_gtype_key(key: GtypeKey, registry: IndexRegistry) -> int:
    """Resolve a gtype selector to a compressed gtype index.

    Accepted forms:
        - ``int``: returned as-is (pre-resolved compressed index)
        - ``(int, int)``: ``(hg_idx, glab_idx)`` pair
        - ``(HaploidGenotype, int|str)``: resolved via ``gtype_index()``
        - ``(str, int|str)``: haploid found by name, then ``gtype_index()``
        - ``HaploidGenotype``: ``gtype_index(hg, "default")``
        - ``str``: haploid found by name, then ``gtype_index(hg, "default")``
    """
    if isinstance(key, int):
        return key
    pair = _as_pair(key)
    if pair is not None:
        hg_part, glab_part = pair
        if isinstance(hg_part, int):
            hg = registry.index_to_haplo[hg_part]
        elif isinstance(hg_part, HaploidGenotype):
            hg = hg_part
        elif isinstance(hg_part, str):
            hg = _resolve_haplo_str(hg_part, registry)
        else:
            raise KeyError(f"Cannot resolve haploid part: {hg_part!r}")
        glab = registry.glab_labels[glab_part] if isinstance(glab_part, int) else str(glab_part)
        return registry.gtype_index(hg, glab)
    if isinstance(key, HaploidGenotype):
        return registry.gtype_index(key, "default")
    if isinstance(key, str):
        hg = _resolve_haplo_str(key, registry)
        return registry.gtype_index(hg, "default")
    raise KeyError(f"Cannot resolve gamete type key: {key!r}")


def _resolve_haplo_str(name: str, registry: IndexRegistry) -> HaploidGenotype:
    """Find a HaploidGenotype by ``to_string()`` match via the registry."""
    for hg in registry.index_to_haplo:
        if hasattr(hg, "to_string") and hg.to_string() == name:
            return hg
    raise KeyError(f"Unknown haploid: {name!r}")


# ============================================================================
# Unified tensor writers — frequency / probability distribution → tensor
# ============================================================================


def _write_gamete_distribution(
    tensor: np.ndarray,
    sex_idx: int,
    zidx: int,
    distribution: Mapping[GtypeKey, float],
    registry: IndexRegistry,
    n_gtypes: int,
) -> None:
    """Write ``{gtype_key: freq}`` into ``tensor[sex_idx, zidx, :]``."""
    tensor[sex_idx, zidx, :] = 0.0
    for key, freq in distribution.items():
        try:
            gt = _resolve_gtype_key(key, registry)
        except (KeyError, IndexError, ValueError):
            continue
        if 0 <= gt < n_gtypes:
            tensor[sex_idx, zidx, gt] = float(freq)


def _write_zygote_distribution(
    tensor: np.ndarray,
    c1: int,
    c2: int,
    distribution: Mapping[int, float],
) -> None:
    """Write ``{ztype_idx: prob}`` into ``tensor[c1, c2, :]``."""
    tensor[c1, c2, :] = 0.0
    for zidx, prob in distribution.items():
        tensor[c1, c2, int(zidx)] = float(prob)


def _normalize_zygote_val_to_distribution(
    val: int | tuple[ZtypeKey, float] | Mapping[ZtypeKey, float] | ZtypeKey,
    registry: IndexRegistry,
) -> dict[int, float]:
    """Normalize a zygote replacement value into ``{ztype_idx: prob}``.

    Handles:
        - ``(int|Genotype|str, float)``: single target + weight
        - ``{int|Genotype|str: float}``: multi-target distribution
        - ``int``: single ztype target (prob=1.0)
        - ``Genotype|str``: expanded to all slab ztypes (prob split equally)
    """
    result: dict[int, float] = {}
    pair = _as_idx_prob_pair(val)
    if pair is not None:
        candidate, prob = pair
        if isinstance(candidate, int):
            result[int(candidate)] = float(prob)
        else:
            gt_obj = _resolve_genotype_from_registry(cast(ZtypeKey, candidate), registry)
            z_indices = registry.ztype_indices_for(gt_obj)
            each = float(prob) / len(z_indices)
            for zi in z_indices:
                result[int(zi)] = each
        return result

    if isinstance(val, Mapping):
        for cand, prob in val.items():
            assert isinstance(prob, (int, float)), "Zygote replacement probabilities must be numeric"
            if isinstance(cand, int):
                result[int(cand)] = float(cast(float, prob))
            else:
                gt_obj = _resolve_genotype_from_registry(cast(ZtypeKey, cand), registry)
                z_indices = registry.ztype_indices_for(gt_obj)
                each = float(prob) / len(z_indices)
                for zi in z_indices:
                    result[int(zi)] = each
        return result

    if isinstance(val, int):
        result[int(val)] = 1.0
    elif isinstance(val, (str, Genotype)):
        gt_obj = _resolve_genotype_from_registry(val, registry)
        z_indices = registry.ztype_indices_for(gt_obj)
        each = 1.0 / len(z_indices)
        for zi in z_indices:
            result[int(zi)] = each
    else:
        raise TypeError(f"Unsupported zygote replacement value: {type(val)}")
    return result


def _resolve_genotype_from_registry(key: ZtypeKey, registry: IndexRegistry) -> Genotype:
    """Resolve a genotype selector (int, str, or Genotype) via registry."""
    if isinstance(key, int):
        return registry.index_to_genotype[key]
    if isinstance(key, str):
        for g in registry.index_to_genotype:
            if hasattr(g, "to_string") and g.to_string() == key:
                return g
        raise KeyError(f"Cannot resolve genotype: {key!r}")
    return key


# ============================================================================
# TENSOR-LEVEL WRAPPER FACTORIES
# ============================================================================
# These functions wrap high-level modifiers (returning dicts of domain objects)
# into tensor-level callables that accept/return NumPy arrays. They encapsulate
# the key-parsing and index-resolution logic so that both base_population and
# external modifier systems (e.g. gamete_allele_conversion) can reuse them.


def wrap_gamete_modifier(
    mod: GameteModifier,
    population: object | None,
    registry: IndexRegistry,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level GameteModifier into a tensor-level callable.

    The returned callable accepts a tensor of shape ``(n_sexes, n_ztypes, n_gtypes)``
    and returns a modified copy.  All key resolution is done via *registry*.

    Args:
        mod: A GameteModifier callable.
        population: The population object (passed to mod if it takes an argument).
        registry: IndexRegistry for key resolution.

    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()
        n_sexes, n_ztypes, n_gtypes = modified.shape

        bulk_obj = _invoke_modifier(mod, population)
        if not isinstance(bulk_obj, Mapping):
            raise TypeError(
                "Gamete modifier must return a mapping from keys to "
                "compressed-index->freq mappings"
            )
        # User-provided modifier returns heterogeneous dicts — cast at boundary.
        bulk = cast(Mapping[Union[str, tuple[object, object]], Mapping[object, object]], bulk_obj)

        for key, val in bulk.items():
            sex_idx = _resolve_sex_name(key) if isinstance(key, str) else None

            if sex_idx is not None:
                for ztype_key, distribution in val.items():
                    try:
                        indices = _resolve_ztype_key(cast(ZtypeKey, ztype_key), registry)
                    except (KeyError, IndexError, ValueError):
                        continue
                    if not (0 <= sex_idx < n_sexes):
                        continue
                    if isinstance(distribution, Mapping):
                        for zi in indices:
                            if 0 <= zi < n_ztypes:
                                _write_gamete_distribution(
                                    modified, sex_idx, zi,
                                    cast(Mapping[GtypeKey, float], distribution),
                                    registry, n_gtypes,
                                )
                continue

            key_tuple = _as_pair(key)
            if key_tuple is not None and isinstance(key_tuple[0], int):
                sex_idx = key_tuple[0]
                ztype_key = key_tuple[1]
                if not (0 <= sex_idx < n_sexes):
                    continue
                try:
                    for zi in _resolve_ztype_key(cast(ZtypeKey, ztype_key), registry):
                        if 0 <= zi < n_ztypes:
                            _write_gamete_distribution(
                                modified, sex_idx, zi,
                                cast(Mapping[GtypeKey, float], val),
                                registry, n_gtypes,
                            )
                except (KeyError, IndexError, ValueError):
                    continue
                continue

            # Case C: key is ztype_key applied to all sexes
            try:
                indices = _resolve_ztype_key(cast(ZtypeKey, key), registry)
            except (KeyError, IndexError, ValueError):
                continue
            for sex_idx in range(n_sexes):
                for zi in indices:
                    if 0 <= zi < n_ztypes:
                        _write_gamete_distribution(
                            modified, sex_idx, zi,
                            cast(Mapping[GtypeKey, float], val),
                            registry, n_gtypes,
                        )

        return modified
    return tensor_modifier


def wrap_zygote_modifier(
    mod: ZygoteModifier,
    population: object | None,
    registry: IndexRegistry,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level ZygoteModifier into a tensor-level callable.

    The returned callable accepts a tensor of shape ``(n_gtypes, n_gtypes, n_ztypes)``
    and returns a modified copy.  All key resolution is done via *registry*.

    Args:
        mod: A ZygoteModifier callable.
        population: The population object (passed to mod if it takes an argument).
        registry: IndexRegistry for key resolution.

    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()

        bulk_obj = _invoke_modifier(mod, population)
        if not isinstance(bulk_obj, Mapping):
            raise TypeError(
                "Zygote modifier must return a mapping from keys to replacements"
            )
        # User-provided modifier returns heterogeneous dicts — cast at boundary.
        bulk = cast(Mapping[tuple[object, object], object], bulk_obj)

        for key, val in bulk.items():
            if _is_int_pair(key):
                c1, c2 = key
            else:
                pair = _as_pair(key)
                if pair is None:
                    raise TypeError("Zygote modifier key must be a 2-tuple")
                c1 = _resolve_gtype_key(cast(GtypeKey, pair[0]), registry)
                c2 = _resolve_gtype_key(cast(GtypeKey, pair[1]), registry)

            distribution = _normalize_zygote_val_to_distribution(
                cast(Union[int, tuple[ZtypeKey, float], Mapping[ZtypeKey, float], ZtypeKey], val),
                registry,
            )
            _write_zygote_distribution(modified, c1, c2, distribution)

        return modified
    return tensor_modifier


def build_modifier_wrappers(
    gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]],
    zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]],
    population: object | None,
    registry: IndexRegistry,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[Callable[[np.ndarray], np.ndarray]]]:
    """Wrap high-level gamete/zygote modifiers into tensor-level callables.

    Args:
        gamete_modifiers: List of (modifier_id, name, modifier) tuples.
        zygote_modifiers: List of (modifier_id, name, modifier) tuples.
        population: The population object.
        registry: IndexRegistry for all key resolution.

    Returns:
        Tuple of (gamete_modifier_funcs, zygote_modifier_funcs).
    """
    gamete_modifier_funcs: List[Callable[[np.ndarray], np.ndarray]] = []
    zygote_modifier_funcs: List[Callable[[np.ndarray], np.ndarray]] = []

    for _, _, mod in zygote_modifiers:
        zygote_modifier_funcs.append(
            wrap_zygote_modifier(mod, population, registry)
        )

    for _, _, mod in gamete_modifiers:
        gamete_modifier_funcs.append(
            wrap_gamete_modifier(mod, population, registry)
        )

    return gamete_modifier_funcs, zygote_modifier_funcs


# ============================================================================
# Generic tuple utilities (used by key resolvers above)
# ============================================================================


def _as_pair(value: object) -> Optional[Tuple[object, object]]:
    """Safely extract a 2-tuple from an unknown value.

    Args:
        value: The value to convert.

    Returns:
        A 2-tuple if *value* is a tuple of length 2, else None.
    """
    if not isinstance(value, tuple):
        return None
    items = cast(Tuple[object, ...], value)
    if len(items) != 2:
        return None
    return items[0], items[1]


def _is_int_pair(value: object) -> TypeGuard[Tuple[int, int]]:
    """Type guard: check if *value* is a 2-tuple of ints.

    Args:
        value: The value to check.

    Returns:
        True if *value* is a 2-tuple where both elements are ints.
    """
    pair = _as_pair(value)
    return pair is not None and isinstance(pair[0], int) and isinstance(pair[1], int)


def _as_idx_prob_pair(value: object) -> Optional[Tuple[object, float]]:
    """Extract an ``(index, probability)`` pair from a value.

    Args:
        value: The value to convert.

    Returns:
        A tuple of ``(index, probability)`` if *value* is a 2-tuple
        with a numeric second element, else None.
    """
    pair = _as_pair(value)
    if pair is None or not isinstance(pair[1], (int, float)):
        return None
    return pair[0], float(pair[1])

