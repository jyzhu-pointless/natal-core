"""Initial-state parsing: user declaration dicts to engine arrays.

Extracted from the builder parameter helpers so the model assembly side
owns the initial-input resolution; the builder chain and the spatial
builder consume these as plain functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, Species
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.helpers import resolve_sex_label
from natal.frontend.utils.types import Sex

__all__ = [
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
        # An optional "@slab" suffix pins the somatic label.  This is the
        # remaining ad-hoc "@" split: it scans the string itself instead of
        # the pattern grammar's shared analysis, so an empty suffix means
        # "default slab" here where the pattern entries reject it
        # (FRONTEND_REFACTOR_PLAN.md §5.3, initial-state row).
        slab_name: Optional[str] = None
        if "@" in genotype_key:
            base, suffix = genotype_key.rsplit("@", 1)
            gt_str = base
            if suffix:
                slab_name = suffix
        else:
            gt_str = genotype_key
        gt = species.get_genotype_from_str(gt_str)
        indices = registry.ztype_indices_for(gt)
        if not indices:
            raise KeyError(
                f"Genotype {gt.to_string()!r} is not in the active ztype catalog"
            )
        if slab_name is not None:
            return registry.ztype_index(gt, slab_name)
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
