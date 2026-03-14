from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Optional, Dict, Any, Callable, Union, List, Sequence
from typing import TypeVar, cast
import inspect
import numpy as np
from natal.type_def import Sex
from natal.genetic_entities import Genotype, HaploidGenotype

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

    Notes:
        - ``sex_idx`` is an ``int``.
        - ``genotype_idx`` may be an ``int``, a ``Genotype`` object, or a
          string produced by ``Genotype.to_string()``.

    Example:

        return {(0, 5): {3: 0.2, 4: 0.8}, (1, 5): {3: 1.0}}

    The result writes frequency distributions for compressed indices directly
    back into numeric tensors.
    """
    def __call__(self, *args: Any) -> Dict[Any, Dict[int, float]]: ...

# 合子修饰器接口保持不变
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
    def __call__(self, *args: Any) -> Dict[Any, Union[int, Genotype, Dict[int, float]]]: ...


# ============================================================================
# HELPER FUNCTIONS FOR MODIFIER CONSTRUCTION
# ============================================================================

def _invoke_modifier(mod: Callable, population: Any = None) -> dict:
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
    if not isinstance(key, str):
        return None
    k = key.lower()
    if k in ("male", "m"):
        return int(Sex.MALE)
    if k in ("female", "f"):
        return int(Sex.FEMALE)
    return None


# ============================================================================
# TENSOR-LEVEL WRAPPER FACTORIES
# ============================================================================
# These functions wrap high-level modifiers (returning dicts of domain objects)
# into tensor-level callables that accept/return NumPy arrays. They encapsulate
# the key-parsing and index-resolution logic so that both base_population and
# external modifier systems (e.g. gamete_allele_conversion) can reuse them.


def wrap_gamete_modifier(
    mod: GameteModifier,
    population: Any,
    index_core: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level GameteModifier into a tensor-level callable.
    
    The returned callable accepts a tensor of shape (n_sexes, n_genotypes, n_hg_glabs)
    and returns a modified copy.
    
    Args:
        mod: A GameteModifier callable (returns dict mapping keys to freq dicts).
        population: The population object (passed to mod if it takes an argument).
        index_core: IndexCore instance for key resolution.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.
    
    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()
        n_sexes, n_genotypes, n_hg_glabs = modified.shape

        bulk = _invoke_modifier(mod, population)

        if not isinstance(bulk, dict):
            raise TypeError("Gamete modifier must return a dict mapping keys to compressed-index->freq dicts")

        for key, val in bulk.items():
            # Case A: top-level sex-name ('male'/'female')
            sex_idx = _resolve_sex_name(key) if isinstance(key, str) else None
            if sex_idx is not None and isinstance(val, dict):
                for gk, comp_map in val.items():
                    try:
                        gidx = gk if isinstance(gk, int) else index_core.resolve_genotype_index(diploid_genotypes, gk, strict=True)
                    except KeyError:
                        continue
                    if not (0 <= sex_idx < n_sexes and 0 <= gidx < n_genotypes):
                        continue
                    _apply_comp_map(modified, sex_idx, gidx, comp_map, index_core, haploid_genotypes, n_glabs, n_hg_glabs)
                continue

            # Case B: explicit (sex_idx, genotype_key) tuple
            if isinstance(key, tuple) and len(key) == 2:
                sex_idx, gk = key
                gidx = gk if isinstance(gk, int) else index_core.resolve_genotype_index(diploid_genotypes, gk, strict=True)
                if not (0 <= sex_idx < n_sexes and 0 <= gidx < n_genotypes):
                    continue
                _apply_comp_map(modified, sex_idx, gidx, val, index_core, haploid_genotypes, n_glabs, n_hg_glabs)
                continue

            # Case C: key is genotype_key applied to all sexes
            try:
                gidx = key if isinstance(key, int) else index_core.resolve_genotype_index(diploid_genotypes, key, strict=True)
            except KeyError:
                continue
            if not isinstance(val, dict):
                continue
            for sex_idx in range(n_sexes):
                _apply_comp_map(modified, sex_idx, gidx, val, index_core, haploid_genotypes, n_glabs, n_hg_glabs)

        return modified
    return tensor_modifier


def wrap_zygote_modifier(
    mod: ZygoteModifier,
    population: Any,
    index_core: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level ZygoteModifier into a tensor-level callable.
    
    The returned callable accepts a tensor of shape (n_hg_glabs, n_hg_glabs, n_genotypes)
    and returns a modified copy.
    
    Args:
        mod: A ZygoteModifier callable.
        population: The population object (passed to mod if it takes an argument).
        index_core: IndexCore instance for key resolution.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.
    
    Returns:
        A callable (np.ndarray) -> np.ndarray.
    """
    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()

        bulk = _invoke_modifier(mod, population)

        if not isinstance(bulk, dict):
            raise TypeError("Zygote modifier must return a dict mapping keys to replacements")

        for key, val in bulk.items():
            c1, c2 = _parse_zygote_key(key, index_core, haploid_genotypes, n_glabs)
            mapping = _normalize_zygote_val(val, index_core, diploid_genotypes)
            _write_zygote_mapping(modified, c1, c2, mapping)

        return modified
    return tensor_modifier


def build_modifier_wrappers(
    gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]],
    zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]],
    population: Any,
    index_core: Any,
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
) -> Tuple[List[Callable], List[Callable]]:
    """Wrap high-level gamete/zygote modifiers into tensor-level callables.

    This is the shared implementation used by BasePopulation and any external
    modifier systems that need to convert high-level modifiers to tensor ops.

    Args:
        gamete_modifiers: List of (hook_id, name, modifier) tuples for gamete modifiers.
        zygote_modifiers: List of (hook_id, name, modifier) tuples for zygote modifiers.
        population: The population object.
        index_core: IndexCore instance.
        haploid_genotypes: List of all HaploidGenotype objects.
        diploid_genotypes: List of all Genotype objects.
        n_glabs: Number of gamete-label variants.

    Returns:
        Tuple of (gamete_modifier_funcs, zygote_modifier_funcs), each a list
        of callables that accept and return NumPy tensors.
    """
    gamete_modifier_funcs = []
    zygote_modifier_funcs = []

    for _, _, mod in zygote_modifiers:
        zygote_modifier_funcs.append(
            wrap_zygote_modifier(mod, population, index_core, haploid_genotypes, diploid_genotypes, n_glabs)
        )

    for _, _, mod in gamete_modifiers:
        gamete_modifier_funcs.append(
            wrap_gamete_modifier(mod, population, index_core, haploid_genotypes, diploid_genotypes, n_glabs)
        )

    return gamete_modifier_funcs, zygote_modifier_funcs


# ============================================================================
# INTERNAL HELPERS (used by the wrapper factories above)
# ============================================================================

def _apply_comp_map(
    modified: np.ndarray,
    sex_idx: int,
    gidx: int,
    comp_map: Any,
    index_core: Any,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int,
    n_hg_glabs: int,
) -> None:
    """Apply a comp_map (comp_key->freq) into the tensor slice [sex_idx, gidx]."""
    modified[sex_idx, gidx, :] = 0.0
    if not isinstance(comp_map, dict):
        return
    for comp_key, freq in comp_map.items():
        comp_idx = index_core.resolve_comp_idx(haploid_genotypes, n_glabs, comp_key, strict=False)
        if comp_idx is None:
            continue
        if not (0 <= comp_idx < n_hg_glabs):
            continue
        modified[sex_idx, gidx, comp_idx] = float(freq)


def _parse_zygote_key(
    key: Any,
    index_core: Any,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int,
) -> Tuple[int, int]:
    """Parse modifier key for zygote wrappers into compressed coords (c1, c2)."""
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(x, int) for x in key):
        return key[0], key[1]
    part1, part2 = key
    idx_hg1, glab1 = index_core.resolve_hg_glab_part(haploid_genotypes, part1, n_glabs, strict=True)
    idx_hg2, glab2 = index_core.resolve_hg_glab_part(haploid_genotypes, part2, n_glabs, strict=True)
    from natal.index_core import compress_hg_glab
    c1 = compress_hg_glab(idx_hg1, glab1, n_glabs)
    c2 = compress_hg_glab(idx_hg2, glab2, n_glabs)
    return c1, c2


def _normalize_zygote_val(
    val: Any,
    index_core: Any,
    diploid_genotypes: List[Genotype],
) -> Dict[int, float]:
    """Normalize zygote replacement value into a mapping idx->prob."""
    mapping: Dict[int, float] = {}

    # single tuple (idx_or_genotype, prob)
    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], (int, float)):
        idx_candidate, prob = val
        if isinstance(idx_candidate, int):
            idx = int(idx_candidate)
        else:
            idx = index_core.resolve_genotype_index(diploid_genotypes, idx_candidate, strict=True)
        mapping[int(idx)] = float(prob)
        return mapping

    # distribution dict
    if isinstance(val, dict):
        for idx_candidate, prob in val.items():
            if not isinstance(idx_candidate, int):
                idx_candidate = index_core.resolve_genotype_index(diploid_genotypes, idx_candidate, strict=True)
            mapping[int(idx_candidate)] = float(prob)
        return mapping

    # single genotype replacement
    idx = index_core.resolve_genotype_index(diploid_genotypes, val, strict=True)
    mapping[int(idx)] = 1.0
    return mapping


def _write_zygote_mapping(
    modified: np.ndarray,
    c1: int,
    c2: int,
    mapping: Dict[int, float],
) -> None:
    """Apply mapping (idx->prob) to the compressed zygote slice."""
    modified[c1, c2, :] = 0.0
    for idx_mod, prob in mapping.items():
        modified[c1, c2, int(idx_mod)] = float(prob)
