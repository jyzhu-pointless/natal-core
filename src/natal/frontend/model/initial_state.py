"""Initial-state parsing: user declaration dicts to engine arrays.

Extracted from the builder parameter helpers so the model assembly side
owns the initial-input resolution; the builder chain and the spatial
builder consume these as plain functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple, TypeAlias, Union, cast

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
        TypeError: If sex_key is neither str nor Sex.
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

    fcount = float(age_data)
    if fcount < 0:
        raise ValueError(f"Count must be non-negative, got {fcount}")
    if fcount <= 0:
        return {}
    return dict.fromkeys(range(new_adult_age, n_ages), fcount)


def _parse_genotype_key(
    genotype_key: Any,
    species: Species,
) -> Any:
    """Parse a user genotype key into a :class:`ZygoteTypePattern`.

    Accepts a ``(genotype, slab)`` tuple, a slab-qualified string such as
    ``"WT|WT@default"``, a plain string genotype name, or a
    :class:`Genotype` instance.

    Raises:
        TypeError: If the key is not one of the accepted forms.
    """
    from natal.frontend.patterns import GenotypePatternParser, ZygoteTypePattern

    if isinstance(genotype_key, tuple):
        _key, _slab = cast("tuple[object, str]", genotype_key)
        if isinstance(_key, Genotype):
            return ZygoteTypePattern.from_pair(_key, _slab, species)
        if isinstance(_key, str):
            _gt = species.get_genotype_from_str(_key)
            return ZygoteTypePattern.parse(f"{str(_gt)}@{_slab}", species)
        raise TypeError(
            f"Tuple first element must be Genotype or str, got {type(_key)}"
        )
    if isinstance(genotype_key, str):
        return ZygoteTypePattern.from_slab_key(genotype_key, species)
    if isinstance(genotype_key, Genotype):
        parser = GenotypePatternParser(species)
        return ZygoteTypePattern(parser.parse(str(genotype_key)), slab=None)
    raise TypeError(
        f"genotype_key must be Genotype, str, or tuple, got {type(genotype_key)}"
    )


def _fresh_species_registry(species: Species) -> IndexRegistry:
    """Build a registry holding exactly the species' active genotype set."""
    registry = IndexRegistry()
    slabs = species.somatic_labels or ["default"]
    for slab in slabs:
        registry.register_somatic_label(slab)
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
    out = np.zeros((2, n_ages, registry.n_ztypes), dtype=np.float64)
    for sex_key, genotype_dist in distribution.items():
        sex_idx = _resolve_sex_index(sex_key)
        for genotype_key, age_data in genotype_dist.items():
            pattern = _parse_genotype_key(genotype_key, species)
            z_idx = registry.resolve_default_ztype_index(pattern)
            age_counts = _resolve_age_counts_age_structured(
                age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
            )
            for age, count in age_counts.items():
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
        TypeError: If storage value is not a dictionary.
    """
    registry = _fresh_species_registry(species)
    out = np.zeros((n_ages, registry.n_ztypes, registry.n_ztypes), dtype=np.float64)

    for female_key, male_dict in sperm_storage.items():
        female_pattern = _parse_genotype_key(female_key, species)
        f_z = registry.resolve_default_ztype_index(female_pattern)

        for male_key, age_data in male_dict.items():
            male_pattern = _parse_genotype_key(male_key, species)
            m_z = registry.resolve_default_ztype_index(male_pattern)

            age_counts = _resolve_age_counts_age_structured(
                age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
            )
            for age, count in age_counts.items():
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
    if isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
        value = float(age_data)
        if value < 0:
            raise ValueError(f"Count must be non-negative, got {value}")
        return 0.0, value

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
    out = np.zeros((2, 2, registry.n_ztypes), dtype=np.float64)

    for sex_key, genotype_dist in distribution.items():
        sex_idx = _resolve_sex_index(sex_key)
        for genotype_key, age_data in genotype_dist.items():
            pattern = _parse_genotype_key(genotype_key, species)
            z_idx = registry.resolve_default_ztype_index(pattern)
            age0, age1 = _resolve_discrete_age_distribution(age_data)
            out[sex_idx, 0, z_idx] += age0
            out[sex_idx, 1, z_idx] += age1
    return out
