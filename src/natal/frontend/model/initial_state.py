"""Initial-state parsing: user declaration dicts to engine arrays.

Extracted from the builder parameter helpers so the model assembly side
owns the initial-input resolution; the builder chain and the spatial
builder consume these as plain functions.

The builder stores the user's initial distribution as an
:class:`InitialDistributionDeclaration` — the authoritative input — and
resolves it into engine arrays through these functions whenever the final
dimensions are known (at declaration time, on ``age_structure()`` rebuilds,
and at ``build()``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, Species
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.helpers import resolve_sex_label
from natal.frontend.utils.types import Sex

__all__ = [
    "InitialDistributionDeclaration",
    "resolve_age_structured_initial_individual_count",
    "resolve_age_structured_initial_sperm_storage",
    "resolve_discrete_initial_individual_count",
]



ArrayF64 = NDArray[np.float64]
InitialAgeCountValue: TypeAlias = (
    Sequence[float | int] | Mapping[int, float | int] | ArrayF64 | int | float
)
InitialIndividualCountInput: TypeAlias = Mapping[str, Mapping[Any, InitialAgeCountValue]]
InitialSpermStorageInput: TypeAlias = Mapping[Any, Mapping[Any, InitialAgeCountValue]]


class _FrozenList(tuple[Any, ...]):
    """Tuple-backed marker retaining that the source value was a list."""


def _immutable_array(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return an array whose backing storage cannot be made writable."""
    contiguous: np.ndarray[Any, Any] = np.ascontiguousarray(value)
    return cast(
        "np.ndarray[Any, Any]",
        np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
            contiguous.shape
        ),
    )


def _freeze_declaration_value(value: Any) -> Any:
    """Recursively own declaration containers while preserving opaque objects."""
    if isinstance(value, np.ndarray):
        return _immutable_array(cast("np.ndarray[Any, Any]", value))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        frozen: dict[Any, Any] = {
            key: _freeze_declaration_value(item) for key, item in mapping.items()
        }
        return MappingProxyType(frozen)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return _FrozenList(_freeze_declaration_value(item) for item in items)
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(_freeze_declaration_value(item) for item in items)
    return value


def _copy_declaration_value(value: Any) -> Any:
    """Return an isolated mutable copy for the public declaration surface."""
    if isinstance(value, np.ndarray):
        return cast("np.ndarray[Any, Any]", value.copy())
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {
            key: _copy_declaration_value(item) for key, item in mapping.items()
        }
    if isinstance(value, _FrozenList):
        return [_copy_declaration_value(item) for item in value]
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(_copy_declaration_value(item) for item in items)
    return value


class InitialDistributionDeclaration:
    """A user's raw initial-distribution declaration.

    The authoritative input the builder keeps: nested containers and numeric
    arrays are recursively owned and frozen on construction, while opaque
    references such as ``Genotype`` objects keep their identity.  Resolution
    into engine arrays happens against whatever dimensions are current; a
    resolution is a pure function of (species identity, declaration,
    dimensions), so resolved arrays are memoized and safely shared as
    immutable views.  Drafts that may be written get fresh copies.
    """

    __slots__ = ("_individual_count", "_sperm_storage", "_resolved")

    def __init__(
        self,
        individual_count: InitialIndividualCountInput,
        sperm_storage: InitialSpermStorageInput | None,
    ) -> None:
        """Store recursively owned, immutable declaration containers."""
        self._individual_count = _freeze_declaration_value(individual_count)
        self._sperm_storage = (
            None if sperm_storage is None else _freeze_declaration_value(sperm_storage)
        )
        self._resolved: Dict[
            Tuple[Species, int, bool, int, int], Tuple[ArrayF64, Optional[ArrayF64]]
        ] = {}

    @classmethod
    def capture(
        cls,
        individual_count: InitialIndividualCountInput,
        sperm_storage: InitialSpermStorageInput | None,
    ) -> InitialDistributionDeclaration:
        """Capture the declaration, keeping opaque references.

        Args:
            individual_count: The ``{sex: {genotype: count}}`` mapping as
                the user passed it.
            sperm_storage: The optional sperm-storage mapping, ``None``
                when the user declared none.

        Returns:
            The frozen declaration.
        """
        return cls(individual_count, sperm_storage)

    @property
    def individual_count(self) -> InitialIndividualCountInput:
        """Return an isolated copy of the raw individual-count declaration."""
        return cast(InitialIndividualCountInput, _copy_declaration_value(self._individual_count))

    @property
    def sperm_storage(self) -> InitialSpermStorageInput | None:
        """Return an isolated copy of the raw sperm-storage declaration."""
        if self._sperm_storage is None:
            return None
        return cast(InitialSpermStorageInput, _copy_declaration_value(self._sperm_storage))

    def resolve(
        self,
        species: Species,
        *,
        discrete_generation: bool,
        n_ages: int,
        new_adult_age: int,
    ) -> Tuple[ArrayF64, Optional[ArrayF64]]:
        """Resolve the declared arrays against the given dimensions.

        Memoized per Species and dimension key: the same declaration resolved
        for the same inputs always yields the same arrays, so builders,
        definition captures, and group projections share one resolution
        instead of re-enumerating the species catalog each time.

        Args:
            species: Species whose catalog resolves genotype selectors.
            discrete_generation: Whether the draft uses the flat discrete
                layout.
            n_ages: Age-class count of the current draft.
            new_adult_age: First adult age of the current draft.

        Returns:
            ``(counts, sperm)`` engine arrays; *sperm* is ``None`` when
            none was declared.

        Raises:
            ValueError: If a selector or an age key is invalid for the
                dimensions (surfacing the resolvers' own messages).
        """
        # Keep the Species reference in the key as well as its id.  The id
        # distinguishes equal-but-distinct Species objects; retaining the
        # reference prevents id reuse while this declaration is alive.
        key = (
            species, id(species), bool(discrete_generation),
            int(n_ages), int(new_adult_age),
        )
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        counts = (
            resolve_discrete_initial_individual_count(
                species=species, distribution=self._individual_count,
            )
            if discrete_generation
            else resolve_age_structured_initial_individual_count(
                species=species, distribution=self._individual_count,
                n_ages=n_ages, new_adult_age=new_adult_age,
            )
        )
        sperm: Optional[ArrayF64] = (
            None
            if self._sperm_storage is None or discrete_generation
            else resolve_age_structured_initial_sperm_storage(
                species=species,
                sperm_storage=cast(InitialSpermStorageInput, self._sperm_storage),
                n_ages=n_ages, new_adult_age=new_adult_age,
            )
        )
        frozen_counts = cast(ArrayF64, _immutable_array(counts))
        frozen_sperm = None if sperm is None else cast(ArrayF64, _immutable_array(sperm))
        self._resolved[key] = (frozen_counts, frozen_sperm)
        return frozen_counts, frozen_sperm



def _resolve_sex_index(sex_key: Union[str, Sex]) -> int:
    """Resolve a sex key into an integer index (0 or 1).

    Args:
        sex_key: The sex label or enum.

    Returns:
        0 for female, 1 for male.

    Raises:
        AssertionError: If sex_key is neither str, int, nor Sex (the type
            check is the assertion inside :func:`resolve_sex_label`).
        ValueError: If sex_key is an integer other than 0/1 or an
            unrecognized sex label.
    """
    if isinstance(sex_key, Sex):
        return int(sex_key.value)
    return resolve_sex_label(sex_key)


def _resolve_age_counts_age_structured(
    age_data: InitialAgeCountValue,
    n_ages: int,
    new_adult_age: int,
) -> Dict[int, float]:
    """Normalize age-based distribution data into a sparse dictionary.

    Args:
        age_data: Raw age distribution data.
        n_ages: Total number of age classes.
        new_adult_age: Minimum age for adults.

    Returns:
        Mapping of age to individual count.

    Raises:
        ValueError: If counts are negative or ages are out of range.
        TypeError: If data type is unsupported.
    """
    # Mapping form: explicit {age: count}; ages must lie in [0, n_ages) and counts
    # must be non-negative.  Zero counts are omitted because callers accumulate.
    if isinstance(age_data, Mapping):
        age_map = age_data
        out: Dict[int, float] = {}
        for age, count in age_map.items():
            if age < 0 or age >= n_ages:
                raise ValueError(f"Age {age} out of range [0, {n_ages})")
            fcount = float(count)
            if fcount < 0:
                raise ValueError(f"Count must be non-negative, got {fcount}")
            if fcount > 0:
                out[age] = fcount
        return out

    # Sequence form: positional by age index, so trailing elements beyond n_ages
    # are truncated rather than rejected (no lower bound needed for an index).
    if isinstance(age_data, (Sequence, np.ndarray)) and not isinstance(
        age_data, (str, bytes, bytearray)
    ):
        arr = np.asarray(age_data, dtype=np.float64)
        out = {}
        for age, count in enumerate(arr):
            if age >= n_ages:
                break
            if count < 0:
                raise ValueError(f"Count must be non-negative, got {count}")
            if count > 0:
                out[age] = float(count)
        return out

    # Scalar form: one count replicated over every adult age [new_adult_age, n_ages);
    # zero places no individuals at all.
    fcount = float(age_data)
    if fcount < 0:
        raise ValueError(f"Count must be non-negative, got {fcount}")
    if fcount <= 0:
        return {}
    return dict.fromkeys(range(new_adult_age, n_ages), fcount)


def resolve_genotype_key_ztype_index(
    genotype_key: Any,
    species: Species,
    registry: IndexRegistry,
) -> int:
    """Resolve one exact initial-state genotype key to a ztype index.

    Genotype-bearing keys (a :class:`Genotype` instance or a
    ``(genotype, slab)`` tuple) resolve through the registry's identity
    maps — no string round trip, so a sex-chromosome genotype can never
    land on a same-autosome opposite-sex ztype.  Exact strings are
    canonicalized via :meth:`Species.get_genotype_from_str` and then
    resolved the same way.  A bare Genotype keeps the historical
    first-slab placement.

    Args:
        genotype_key: Genotype, str, or ``(genotype, slab)`` tuple.
        species: The bound Species object.
        registry: The registry whose ztype axis the index refers to.

    Returns:
        The ztype index in *registry*.

    Raises:
        TypeError: If the key is not one of the accepted forms.
        KeyError: If the resolved genotype is not in the registry.
        ValueError: If an exact string cannot be parsed.
    """
    # (genotype, slab) tuple: the slab is explicit, so the registry's identity map
    # resolves the pair directly, with no string round trip to re-canonicalize.
    if isinstance(genotype_key, tuple):
        _key, _slab = cast("tuple[object, str]", genotype_key)
        if isinstance(_key, Genotype):
            return registry.ztype_index(_key, _slab)
        if isinstance(_key, str):
            gt = species.get_genotype_from_str(_key)
            return registry.ztype_index(gt, _slab)
        raise TypeError(
            f"Tuple first element must be Genotype or str, got {type(_key)}"
        )
    if isinstance(genotype_key, Genotype):
        gt = genotype_key
    elif isinstance(genotype_key, str):
        # An optional "@slab" suffix pins the somatic label.  The split is
        # the pattern grammar's own "@" analysis, so a malformed suffix
        # (empty or doubled) fails here exactly as it does in the pattern
        # entries, and a pinned label must be one exact name — sets,
        # negations and "*" cannot identify a single ztype.
        from natal.frontend.patterns.parser import GenotypePatternParser

        gt_str, slab = GenotypePatternParser.split_label_suffix(genotype_key)
        gt = species.get_genotype_from_str(gt_str)
        indices = registry.ztype_indices_for(gt)
        if not indices:
            raise KeyError(
                f"Genotype {gt.to_string()!r} is not in the active ztype catalog"
            )
        if slab is not None:
            if slab.negate or slab.lab_set is not None or slab.lab is None:
                raise ValueError(
                    f"initial_state key {genotype_key!r} must pin one exact "
                    "slab name after '@'; sets, negations and '*' cannot "
                    "identify a single ztype"
                )
            return registry.ztype_index(gt, slab.lab)
        return indices[0]
    else:
        raise TypeError(
            f"genotype_key must be Genotype, str, or tuple, got {type(genotype_key)}"
        )
    indices = registry.ztype_indices_for(gt)
    if not indices:
        raise KeyError(
            f"Genotype {gt.to_string()!r} is not in the active ztype catalog"
        )
    return indices[0]


def _fresh_species_registry(species: Species) -> IndexRegistry:
    """Build a registry holding exactly the species' active genotype set."""
    # Mirror the species product order used by build_registry(), so key resolution
    # agrees with the population's full-catalog ztype axis.
    registry = IndexRegistry()
    slabs = species.somatic_labels or ["default"]
    for slab in slabs:
        registry.register_somatic_label(slab)
    # Enumerate with the species' own unordered flag so the key canonicalization
    # here matches the population's catalog (parental order may collapse).
    genotypes = species.get_all_genotypes(unordered=species.unordered)
    for gt in genotypes:
        registry.register_genotype(gt)
    return registry


def resolve_age_structured_initial_individual_count(
    species: Species,
    distribution: InitialIndividualCountInput,
    n_ages: int,
    new_adult_age: int,
) -> NDArray[np.float64]:
    """Resolve initial individual counts for age-structured models.

    Args:
        species: The bound Species object.
        distribution: User-provided distribution mapping.
        n_ages: Total number of age classes.
        new_adult_age: Minimum age for adults.

    Returns:
        A 3D array ``[sex, age, genotype]``.
    """
    registry = _fresh_species_registry(species)
    # Plane layout [sex, age, genotype]: Rust stacks one such plane per deme into
    # its (n_demes, 2, n_ages, n_ztypes) row-major state tensor.
    out = np.zeros((2, n_ages, registry.n_ztypes), dtype=np.float64)
    for sex_key, genotype_dist in distribution.items():
        sex_idx = _resolve_sex_index(sex_key)
        for genotype_key, age_data in genotype_dist.items():
            z_idx = resolve_genotype_key_ztype_index(genotype_key, species, registry)
            age_counts = _resolve_age_counts_age_structured(
                age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
            )
            for age, count in age_counts.items():
                # Accumulate: distinct key spellings (bare genotype, @slab, tuple,
                # reversed parental order) can canonicalize to the same cell.
                out[sex_idx, age, z_idx] += float(count)
    return out


def resolve_age_structured_initial_sperm_storage(
    species: Species,
    sperm_storage: InitialSpermStorageInput,
    n_ages: int,
    new_adult_age: int,
) -> NDArray[np.float64]:
    """Resolve initial sperm storage for age-structured models.

    Args:
        species: The bound Species object.
        sperm_storage: User-provided sperm storage mapping.
        n_ages: Total number of age classes.
        new_adult_age: Minimum age for adults.

    Returns:
        A 3D array ``[age, female_genotype, male_genotype]``.

    Raises:
        AttributeError: If sperm_storage (or a per-female value) does not
            provide ``.items()``. The mapping type is not validated: a
            non-mapping input fails at the ``.items()`` access rather
            than with a ``TypeError``.
    """
    registry = _fresh_species_registry(species)
    # Layout [age, female_genotype, male_genotype]: Rust stacks this into its
    # (n_demes, n_ages, n_ztypes, n_ztypes) sperm-storage tensor.
    out = np.zeros((n_ages, registry.n_ztypes, registry.n_ztypes), dtype=np.float64)

    for female_key, male_dict in sperm_storage.items():
        f_z = resolve_genotype_key_ztype_index(female_key, species, registry)

        for male_key, age_data in male_dict.items():
            m_z = resolve_genotype_key_ztype_index(male_key, species, registry)

            age_counts = _resolve_age_counts_age_structured(
                age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
            )
            for age, count in age_counts.items():
                # Accumulate over canonicalized key spellings, as for individual_count.
                out[age, f_z, m_z] += float(count)
    return out


def _resolve_discrete_age_distribution(
    age_data: InitialAgeCountValue,
) -> Tuple[float, float]:
    """Normalize discrete distribution data into (age0, age1) counts.

    Args:
        age_data: Raw distribution data.

    Returns:
        Count for age 0 and age 1.

    Raises:
        ValueError: If negative counts or invalid lengths are provided.
    """
    # Discrete models expose two age classes; a plain number means age 1 (the
    # adult entry age), matching the age-structured scalar rule.
    if isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
        value = float(age_data)
        if value < 0:
            raise ValueError(f"Count must be non-negative, got {value}")
        return 0.0, value

    # List/array form: positional (age 0, age 1); a single element is again
    # treated as age 1, and more than two elements is rejected.
    if isinstance(age_data, (Sequence, np.ndarray)) and not isinstance(
        age_data, (str, bytes, bytearray)
    ):
        arr = np.asarray(age_data, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        if arr.size == 1:
            if arr[0] < 0:
                raise ValueError(f"Count must be non-negative, got {arr[0]}")
            return 0.0, float(arr[0])
        if arr.size == 2:
            if np.any(arr < 0):
                raise ValueError(f"Count must be non-negative, got {arr}")
            return float(arr[0]), float(arr[1])
        raise ValueError(f"Discrete initial list/array must have length <= 2, got {arr.size}")

    # Dict form: only explicit age keys 0 and 1 are accepted; missing keys are 0.
    if isinstance(age_data, Mapping):
        age_map = age_data
        unsupported_keys = [k for k in age_map.keys() if k not in (0, 1)]
        if unsupported_keys:
            raise ValueError(
                f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}"
            )
        age0 = float(age_map.get(0, 0.0))
        age1 = float(age_map.get(1, 0.0))
        if age0 < 0 or age1 < 0:
            raise ValueError("Count must be non-negative")
        return age0, age1

    raise TypeError(f"Unsupported age_data type: {type(age_data)}")


def resolve_discrete_initial_individual_count(
    species: Species,
    distribution: InitialIndividualCountInput,
) -> NDArray[np.float64]:
    """Resolve initial individual counts for discrete generation models.

    Args:
        species: The bound Species object.
        distribution: User-provided distribution mapping.

    Returns:
        A 3D array ``[sex, age, genotype]`` with age max 2.
    """
    registry = _fresh_species_registry(species)
    # Fixed two-age plane [sex, age, genotype] (age 0 newborns, age 1 adults);
    # its per-deme shape matches the engine's discrete state tensor.
    out = np.zeros((2, 2, registry.n_ztypes), dtype=np.float64)

    for sex_key, genotype_dist in distribution.items():
        sex_idx = _resolve_sex_index(sex_key)
        for genotype_key, age_data in genotype_dist.items():
            z_idx = resolve_genotype_key_ztype_index(genotype_key, species, registry)
            age0, age1 = _resolve_discrete_age_distribution(age_data)
            # Accumulate: several genotype keys can share one canonical ztype cell.
            out[sex_idx, 0, z_idx] += age0
            out[sex_idx, 1, z_idx] += age1
    return out
