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
    from natal.frontend.registry.index import IndexRegistry

import numpy as np

from natal.frontend.genetics import Genotype, HaploidGenotype
from natal.frontend.utils.helpers import resolve_sex_label

GenotypeFilter = Optional[Union[Callable[[Genotype], bool], str]]
GlabSelector = Optional[Union[str, int]]

# Key types accepted by the unified resolvers.  Ints are pre-resolved
# compressed indices; strings and domain objects are runtime-dispatched.
ZtypeKey: TypeAlias = Union[int, str, Genotype]
GtypeKey: TypeAlias = Union[int, str, HaploidGenotype, Tuple[int, int], Tuple[str, int]]

# Bulk-only modifier interface expectations (strict form):
# - gamete modifier: callable() -> Dict[(sex_idx:int, ztype_idx:int) -> Dict[compressed_hg_glab_idx:int -> freq:float]]
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

    where the outer key is ``(sex_idx, ztype_key)`` and the inner mapping is
    ``{ compressed_hg_glab_idx: frequency, ... }``. Keys may be flexible types
    in wrappers (for convenience) but should ultimately resolve to integers.

    ``sex_idx`` is an ``int``. ``ztype_key`` may be an ``int``, a
    ``Genotype`` object, or a string produced by ``Genotype.to_string()``.
    An integer identifies one ZType; a genotype object or string selects
    every slab belonging to that genotype.

    Key resolution is strict: an unresolvable source or target key, or a
    resolved index outside the population's active (possibly compressed)
    axis, raises ``ValueError`` at apply time.  An empty distribution is
    legal and clears the row (an all-zero gamete output is a valid model).

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

class CompiledRuleModifier:
    """Apply a compiled cascade to the entering construction tensor.

    Direct calls evaluate the species baseline for inspection. During a
    build, wrappers supply the current tensor so separate rule sets compose
    in declaration order. Rebuilds still start from the species baseline.
    """

    def __init__(
        self,
        baseline: np.ndarray,
        cascade: Callable[[np.ndarray, int, int], dict[int, float]],
    ) -> None:
        """Capture the projected baseline and compiled row transformation."""
        self._baseline = baseline
        self._cascade = cascade

    def rows_for(self, tensor: np.ndarray) -> dict[tuple[int, int], dict[int, float]]:
        """Transform each entering row, retaining the tensor's probability mass."""
        if tensor.shape != self._baseline.shape:
            raise ValueError("Conversion tensor shape differs from the compiled registry")
        result: dict[tuple[int, int], dict[int, float]] = {}
        for first in range(tensor.shape[0]):
            for second in range(tensor.shape[1]):
                branches = self._cascade(tensor[first, second], first, second)
                if branches:
                    result[first, second] = branches
        return result

    def __call__(self, *_args: object, **_kwargs: object) -> dict[tuple[int, int], dict[int, float]]:
        """Return the standalone cascade evaluated from the Mendelian baseline."""
        return self.rows_for(self._baseline)


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
        from natal.frontend.patterns.entries import parse_selector
        try:
            # The selector entry owns the unordered | → :: promotion, so a
            # modifier filter matches what the same spelling matches in
            # fitness and the rules.
            pattern = parse_selector(
                genotype_filter, species=genotype.species,
                kind="genotype", context="genotype_filter",
            )
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

    An ``int`` is returned as-is (pre-resolved ztype index); a ``str`` is
    matched via ``genotype.to_string()`` and then expanded to all slab
    ztype indices via ``registry.ztype_indices_for()``; a ``Genotype`` is
    expanded to all slab ztype indices.

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

    An ``int`` is returned as-is (pre-resolved compressed index); an
    ``(int, int)`` pair is an ``(hg_idx, glab_idx)`` pair; a
    ``(HaploidGenotype, int|str)`` or ``(str, int|str)`` pair resolves via
    ``gtype_index()`` (a string haploid is looked up by name); a bare
    ``HaploidGenotype`` resolves as ``gtype_index(hg, "default")``; and a
    bare ``str`` is looked up by name, then resolved as
    ``gtype_index(hg, "default")``.
    """
    if isinstance(key, int):
        return key
    pair = _as_pair(key)
    if pair is not None:
        hg_part, glab_part = pair
        if isinstance(hg_part, int):
            if not 0 <= hg_part < len(registry.index_to_haplo):
                raise IndexError(f"haploid index {hg_part} outside active axis")
            hg = registry.index_to_haplo[hg_part]
        elif isinstance(hg_part, HaploidGenotype):
            hg = hg_part
        elif isinstance(hg_part, str):
            hg = _resolve_haplo_str(hg_part, registry)
        else:
            raise KeyError(f"Cannot resolve haploid part: {hg_part!r}")
        if isinstance(glab_part, int) and not 0 <= glab_part < len(registry.glab_labels):
            raise IndexError(f"gamete label index {glab_part} outside active axis")
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
    context: str,
) -> None:
    """Write ``{gtype_key: freq}`` into ``tensor[sex_idx, zidx, :]``.

    Raises:
        ValueError: If a target key cannot be resolved against the
            registry, or the resolved index falls outside the active
            compressed gtype axis.  Invalid declarations must fail the
            apply instead of silently dropping targets.
    """
    tensor[sex_idx, zidx, :] = 0.0
    for key, freq in distribution.items():
        gt = _resolve_gtype_key(key, registry)
        if not 0 <= gt < n_gtypes:
            raise ValueError(
                f"{context}: target gamete index {gt} (key {key!r}) is "
                f"outside the compressed gtype axis [0, {n_gtypes})"
            )
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

    Supported values: an ``(int, float)`` pair targets a single ztype
    with the given weight; a ``(Genotype|str, float)`` pair expands to
    every slab ztype of that genotype and splits the weight evenly among
    them; a ``{int|Genotype|str: float}`` mapping combines those per-key
    rules into one multi-target distribution; a bare ``int`` targets a
    single ztype with probability 1.0; and a bare ``Genotype|str``
    expands to all its slab ztypes with probability split equally.
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
    name: str | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a high-level GameteModifier into a tensor-level callable.

    The returned callable accepts a tensor of shape ``(n_sexes, n_ztypes, n_gtypes)``
    and returns a modified copy.  All key resolution is done via *registry*.

    Key resolution is strict: a source or target key that cannot be
    resolved against *registry*, or a resolved index outside the active
    axis, raises ``ValueError`` instead of being silently dropped.  Legal
    but empty distributions are kept (an all-zero gamete row is a valid
    model), and writes land on a copy, so a raising modifier never leaves
    a half-applied result behind.

    Args:
        mod: A GameteModifier callable.
        population: The population object (passed to mod if it takes an argument).
        registry: IndexRegistry for key resolution.
        name: Optional modifier declaration name used in error messages.

    Returns:
        A callable (np.ndarray) -> np.ndarray.

    Raises:
        ValueError: When the modifier declares an invalid source or
            target key, or an out-of-range index.
        TypeError: When a replacement value is not a mapping (raised
            directly, not wrapped into ``ValueError``).
    """
    context = f"Gamete modifier {name!r}" if name else "Gamete modifier"

    def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
        modified = tensor.copy()
        n_sexes, n_ztypes, n_gtypes = modified.shape

        bulk_obj = (
            mod.rows_for(tensor) if isinstance(mod, CompiledRuleModifier)
            else _invoke_modifier(mod, population)
        )
        if not isinstance(bulk_obj, Mapping):
            raise TypeError(
                "Gamete modifier must return a mapping from keys to "
                "compressed-index->freq mappings"
            )
        # User-provided modifier returns heterogeneous dicts — cast at boundary.
        bulk = cast(Mapping[Union[str, tuple[object, object]], Mapping[object, object]], bulk_obj)

        for key, val in bulk.items():
            try:
                _apply_gamete_replacement(
                    modified, key, val, registry, n_sexes, n_ztypes, n_gtypes, context
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(f"{context}: invalid source key {key!r}: {exc}") from exc

        return modified
    return tensor_modifier


def _apply_gamete_replacement(
    modified: np.ndarray,
    key: Union[str, tuple[object, object]],
    val: object,
    registry: IndexRegistry,
    n_sexes: int,
    n_ztypes: int,
    n_gtypes: int,
    context: str,
) -> None:
    """Apply one gamete replacement entry onto *modified* (strict).

    Raises:
        KeyError: If a source key cannot be resolved against the registry.
        IndexError: If an integer index is out of range.
        ValueError: If a resolved index or distribution shape is invalid.
        TypeError: If a replacement value is not a mapping.
    """
    sex_idx = _resolve_sex_name(key) if isinstance(key, str) else None

    if sex_idx is not None:
        if not 0 <= sex_idx < n_sexes:
            raise IndexError(f"sex index {sex_idx} outside [0, {n_sexes})")
        if not isinstance(val, Mapping):
            raise TypeError(
                f"replacement for sex key {key!r} must be a mapping from "
                f"ztype keys to distributions, got {type(val).__name__}"
            )
        for ztype_key, distribution in cast(Mapping[object, object], val).items():
            indices = _resolve_ztype_key(cast(ZtypeKey, ztype_key), registry)
            _require_indices(indices, ztype_key, n_ztypes, context)
            for zi in indices:
                _write_gamete_distribution(
                    modified, sex_idx, zi,
                    cast(Mapping[GtypeKey, float], distribution),
                    registry, n_gtypes, context,
                )
        return

    key_tuple = _as_pair(key)
    if key_tuple is not None and isinstance(key_tuple[0], int):
        sex_idx = key_tuple[0]
        ztype_key = key_tuple[1]
        if not 0 <= sex_idx < n_sexes:
            raise IndexError(f"sex index {sex_idx} outside [0, {n_sexes})")
        indices = _resolve_ztype_key(cast(ZtypeKey, ztype_key), registry)
        _require_indices(indices, ztype_key, n_ztypes, context)
        for zi in indices:
            _write_gamete_distribution(
                modified, sex_idx, zi,
                cast(Mapping[GtypeKey, float], val),
                registry, n_gtypes, context,
            )
        return

    # Case C: key is ztype_key applied to all sexes
    indices = _resolve_ztype_key(cast(ZtypeKey, key), registry)
    _require_indices(indices, key, n_ztypes, context)
    for sex_idx in range(n_sexes):
        for zi in indices:
            _write_gamete_distribution(
                modified, sex_idx, zi,
                cast(Mapping[GtypeKey, float], val),
                registry, n_gtypes, context,
            )


def _require_indices(
    indices: list[int],
    key: object,
    n_ztypes: int,
    context: str,
) -> None:
    """Reject a source key that resolves outside the active ztype axis.

    A source genotype that exists in the species but not in this
    population's (possibly compressed) axis can never match; declaring it
    is a mismatch, not a silent no-op.
    """
    for zi in indices:
        if not 0 <= zi < n_ztypes:
            raise IndexError(
                f"{context}: source ztype index {zi} (key {key!r}) is outside "
                f"the compressed ztype axis [0, {n_ztypes})"
            )
    if not indices:
        raise KeyError(
            f"source ztype key {key!r} matches no ztype in this population"
        )


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

        bulk_obj = (
            mod.rows_for(tensor) if isinstance(mod, CompiledRuleModifier)
            else _invoke_modifier(mod, population)
        )
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

    for _, mod_name, mod in gamete_modifiers:
        gamete_modifier_funcs.append(
            wrap_gamete_modifier(mod, population, registry, name=mod_name)
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
