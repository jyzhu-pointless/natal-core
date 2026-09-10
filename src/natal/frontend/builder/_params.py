"""Parameter and initial-state resolution helpers shared by the PopulationBuilder.

Standalone functions used by the PopulationBuilder build/update paths, the
spatial builder dispatch, and the population modules.  Extracting them
into a shared module avoids circular imports between ``_base.py``,
the spatial builder, ``discrete_generation.py``, and
``age_structured.py``.

Function overview:
  - ``resolve_age_param()`` — convert flexible survival specs
    (scalar, list, dict, callable) into a 1-D float array.
  - ``resolve_growth_mode()`` — normalize juvenile growth mode
    string or int to the internal integer constant.
  - ``resolve_carrying_capacity()`` — determine K from the three
    available sources (explicit, legacy alias, or initial state).
  - ``build_equilibrium_distribution()`` — propagate K through
    survival rates to produce a (2, n_ages) equilibrium array.
  - ``compute_expected_eggs_from_females()`` — forward-propagate
    a target adult female count to compute total egg production.
  - ``resolve_age_structured_initial_individual_count()`` — parse
    JSON-style ``{sex: {genotype: count}}`` into a ``(2, n_ages,
    n_ztypes)`` ndarray.
  - ``resolve_age_structured_initial_sperm_storage()`` — same for the
    sperm-storage table (``(n_ages, n_ztypes, n_ztypes)``).
  - ``resolve_discrete_initial_individual_count()`` — same as the first
    resolver for the two-age discrete model.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data import (
    BEVERTON_HOLT,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
)
from natal.frontend.genetics import Genotype, Species
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.helpers import resolve_sex_label
from natal.frontend.utils.types import Sex

__all__: list[str] = []  # internal helpers, not re-exported


# ─────────────────────────────────────────────────────────────────────────────
# From population_builder.py: resolve_age_param
# ─────────────────────────────────────────────────────────────────────────────


def resolve_age_param(
    param: Optional[Any],
    expected_length: int,
    default: list[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Resolve flexible survival spec into a 1D float array.

    Note:
        Supports various input types:
        - None: uses default.
        - numeric scalar: fills all ages with this value.
        - sequence/ndarray: truncated or padded with 0.
        - dict[int, float]: sparse age map, unspecified ages default to 1.0.
        - callable(age): returns float for each age.

    Args:
        param (Optional[Any]): The flexible survival parameter to resolve.
        expected_length (int): Required length of the output array.
        default (List[float]): Default values to fallback to.

    Returns:
        NDArray[np.float64]: A 1D array of resolved survival rates.

    Raises:
        ValueError: If rates are negative or out of range.
        TypeError: If input type is unsupported.
    """
    if param is None:
        out = np.array(default[:expected_length], dtype=np.float64)
        if out.size < expected_length:
            out = np.pad(out, (0, expected_length - out.size), constant_values=0.0)
        return out

    if isinstance(param, (int, float)) and not isinstance(param, bool):
        val = float(param)
        if val < 0:
            raise ValueError("Survival rates must be non-negative")
        return np.full(expected_length, val, dtype=np.float64)

    if isinstance(param, dict):
        param_map = cast(Dict[int, float], param)
        out = np.ones(expected_length, dtype=np.float64)
        for age, value in param_map.items():
            if age < 0 or age >= expected_length:
                raise ValueError(f"Age {age} out of range [0, {expected_length})")
            fval = float(value)
            if fval < 0:
                raise ValueError("Survival rates must be non-negative")
            out[age] = fval
        return out

    if callable(param):
        sig = inspect.signature(param)
        required_positional = 0
        accepts_var_positional = False
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if p.default is inspect.Signature.empty:
                    required_positional += 1
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                accepts_var_positional = True
        if required_positional > 1 or (required_positional == 0 and not accepts_var_positional):
            raise TypeError("Survival callable must accept one int age argument")

        vals = np.empty(expected_length, dtype=np.float64)
        for age in range(expected_length):
            try:
                value = param(age)
                if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
                    raise TypeError(
                        f"Survival callable must return a float-compatible number, got {type(value)}"
                    )
                numeric_value = cast(int | float | np.integer[Any] | np.floating[Any], value)
                vals[age] = float(numeric_value)
            except Exception as exc:
                raise ValueError(f"Error calling survival rate function at age {age}: {exc}") from exc
        if np.any(vals < 0):
            raise ValueError("Survival rates must be non-negative")
        return vals

    if isinstance(param, (list, tuple, np.ndarray)):
        obj_arr = np.array(param, dtype=object)
        if obj_arr.size == 0:
            return np.zeros(expected_length, dtype=np.float64)

        if obj_arr[-1] is None:
            non_none = None
            for value in obj_arr[::-1]:
                if value is not None:
                    non_none = float(value)
                    break
            if non_none is None:
                out = np.array(default[:expected_length], dtype=np.float64)
                if out.size < expected_length:
                    out = np.pad(out, (0, expected_length - out.size), constant_values=0.0)
                return out
            prefix_vals: List[float] = []
            for value in obj_arr[:-1]:
                if value is None:
                    raise TypeError("None only allowed as final sentinel in survival list")
                prefix_vals.append(float(value))
            out = np.empty(expected_length, dtype=np.float64)
            prefix = min(len(prefix_vals), expected_length)
            if prefix > 0:
                out[:prefix] = np.asarray(prefix_vals[:prefix], dtype=np.float64)
            if prefix < expected_length:
                out[prefix:] = float(non_none)
            if np.any(out < 0):
                raise ValueError("Survival rates must be non-negative")
            return out

        arr = np.asarray(obj_arr, dtype=np.float64)
        out = np.zeros(expected_length, dtype=np.float64)
        prefix = min(arr.size, expected_length)
        if prefix > 0:
            out[:prefix] = arr[:prefix]
        if np.any(out < 0):
            raise ValueError("Survival rates must be non-negative")
        return out

    raise TypeError(
        "survival rates must be None, sequence, dict, callable or numeric constant"
    )


# ─────────────────────────────────────────────────────────────────────────────
# From population_builder.py: _resolve_growth_mode, _resolve_carrying_capacity,
# _build_equilibrium_distribution, compute_expected_eggs_from_females
# ─────────────────────────────────────────────────────────────────────────────


def resolve_growth_mode(mode: Union[int, str]) -> int:
    """Normalize a juvenile growth mode specification to the internal integer constant.

    Accepts either the string name (case-insensitive: ``"logistic"``,
    ``"beverton_holt"``, ``"fixed"``, ``"linear"``,
    ``"no_competition"``) or the corresponding integer constant from
    :mod:`natal.frontend.data`.  Strings are mapped via a lookup table; integers
    are validated against the set of known constants.

    Args:
        mode: Growth mode as a string or integer constant.

    Returns:
        The canonical integer constant.

    Raises:
        ValueError: If the string is not recognized or the integer is
            not a valid constant.
    """
    if isinstance(mode, int):
        if mode not in [NO_COMPETITION, FIXED, LOGISTIC, BEVERTON_HOLT, LINEAR]:
            raise ValueError(f"Invalid growth mode constant: {mode}")
        return mode
    mode_map = {
        'NO_COMPETITION': NO_COMPETITION, 'FIXED': FIXED,
        'LOGISTIC': LOGISTIC,
        'BEVERTON_HOLT': BEVERTON_HOLT, 'LINEAR': LINEAR,
    }
    upper_mode = mode.upper()
    if upper_mode not in mode_map:
        raise ValueError(f"Unknown growth mode string: {mode}")
    return mode_map[upper_mode]


def resolve_carrying_capacity(
    age_1_carrying_capacity: Optional[float],
    old_juvenile_carrying_capacity: Optional[float],
    initial_individual_count: Optional[NDArray[np.float64]] = None,
) -> float:
    """Resolve the carrying capacity K from available sources.

    Priority order:
    1. ``age_1_carrying_capacity`` (explicit value).
    2. ``old_juvenile_carrying_capacity`` (deprecated alias).
    3. Sum of age-1 counts from ``initial_individual_count`` (auto-detect).
    4. Sum of all counts from ``initial_individual_count`` (fallback).

    Args:
        age_1_carrying_capacity: Explicit K value (age-1 total at equilibrium).
        old_juvenile_carrying_capacity: Deprecated alias for *age_1_carrying_capacity*.
        initial_individual_count: Optional initial count array of shape
            ``(2, n_ages, n_ztypes)`` used for auto-detection when K is not
            explicitly provided.

    Returns:
        The resolved carrying capacity.

    Raises:
        ValueError: If none of the three sources yields a valid K >= 0.5.
    """
    # Priority 1: age_1_carrying_capacity
    if age_1_carrying_capacity is not None:
        return float(age_1_carrying_capacity)

    # Priority 2: old_juvenile_carrying_capacity (legacy alias)
    if old_juvenile_carrying_capacity is not None:
        return float(old_juvenile_carrying_capacity)

    # Priority 3: initial_individual_count (fallback)
    # K is age-1 total, so extract age-1 count specifically.
    if initial_individual_count is not None:
        n_ages = initial_individual_count.shape[1]
        if n_ages >= 2:
            age_1_count = float(initial_individual_count[:, 1, :].sum())
            if age_1_count >= 0.5:
                return age_1_count
        # Fallback for edge cases (n_ages=1 or zero age-1)
        total_both = float(initial_individual_count.sum())
        if total_both >= 0.5:
            return total_both

    raise ValueError(
        "No valid carrying capacity source. Provide age_1_carrying_capacity "
        "or initial_individual_count."
    )


def build_equilibrium_distribution(
    K: float,
    sex_ratio: float,
    age_based_survival_rates: NDArray[np.float64],
    n_ages: int,
) -> NDArray[np.float64]:
    """Build equilibrium individual distribution by forward propagation from K.

    Age-1 is allocated as ``(K * sex_ratio, K * (1-sex_ratio))`` for females
    and males. Subsequent ages are propagated forward via survival rates.

    Args:
        K: Carrying capacity (total individuals at age=1).
        sex_ratio: Female proportion.
        age_based_survival_rates: (2, n_ages) survival array.
        n_ages: Number of age classes.

    Returns:
        NDArray of shape (2, n_ages) with the equilibrium distribution.
    """
    dist = np.zeros((2, n_ages), dtype=np.float64)
    dist[0, 1] = K * sex_ratio
    dist[1, 1] = K * (1.0 - sex_ratio)
    for age in range(2, n_ages):
        dist[0, age] = dist[0, age - 1] * age_based_survival_rates[0, age - 1]
        dist[1, age] = dist[1, age - 1] * age_based_survival_rates[1, age - 1]
    return dist


def compute_expected_eggs_from_females(
    expected_num_new_adult_females: float,
    eggs_per_female: float,
    age_based_survival_rates: NDArray[np.float64],
    age_based_reproduction_rates: Optional[NDArray[np.float64]],
    female_age_based_fertility: NDArray[np.float64],
    sex_ratio: float,
    new_adult_age: int,
    n_ages: int,
) -> float:
    """Compute total expected egg production from a target adult female count.

    Forward-propagates ``expected_num_new_adult_females`` across adult ages via
    survival rates (same direction as ``compute_equilibrium_metrics``), then
    computes total egg production from the resulting age-specific female counts.

    Args:
        expected_num_new_adult_females: Number of adult females at new_adult_age.
        eggs_per_female: Base eggs per female.
        age_based_survival_rates: (2, n_ages) survival array.
        age_based_reproduction_rates: Female reproduction participation by age.
            If None, falls back to female mating rates.
        female_age_based_fertility: Relative fertility by age.
        sex_ratio: Sex ratio (not directly used in forward propagation).
        new_adult_age: First adult age class.
        n_ages: Total age classes.

    Returns:
        float: Total expected egg production.
    """
    if age_based_reproduction_rates is None:
        reproduction_rates = np.ones(n_ages, dtype=np.float64)
        reproduction_rates[:new_adult_age] = 0.0
    else:
        reproduction_rates = age_based_reproduction_rates

    # Build female-only adult distribution (forward propagation)
    female_dist = np.zeros(n_ages, dtype=np.float64)
    female_dist[new_adult_age] = expected_num_new_adult_females
    for age in range(new_adult_age + 1, n_ages):
        female_dist[age] = female_dist[age - 1] * age_based_survival_rates[0, age - 1]

    # Compute total expected eggs
    eggs = 0.0
    for age in range(new_adult_age, n_ages):
        p_reproducing = min(1.0, max(0.0, float(reproduction_rates[age])))
        eggs += female_dist[age] * p_reproducing * female_age_based_fertility[age] * eggs_per_female

    return eggs


# ─────────────────────────────────────────────────────────────────────────────
# From _base.py: _resolve_param
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: the former ``resolve_param`` name lookup moved to
# ``_routes.lookup()`` / ``_routes.lookup_or_none()`` — the route table
# pre-indexes full keys, short names, and aliases at import time.


def iter_sexual_selection_entries(
    sexual_selection: dict[Any, float | dict[Any, float]]
) -> list[tuple[Any, Any, float]]:
    """Parse sexual selection map into (female_selector, male_selector, value) triples.

    Supports flat (male->value, female wildcard ``*``) and nested
    (female->male->value) forms.

    Args:
        sexual_selection: The raw user-provided sexual selection map.

    Returns:
        List of ``(female_selectors, male_selectors, value)`` entries.
    """
    if not sexual_selection:
        return []
    has_nested = any(isinstance(v, dict) for v in sexual_selection.values())
    entries: list[tuple[Any, Any, float]] = []
    if has_nested:
        for female_selector, male_map in sexual_selection.items():
            if not isinstance(male_map, dict):
                raise TypeError(
                    "When using nested sexual_selection, each female key must map to a dict of male->value"
                )
            for male_selector, value in male_map.items():
                entries.append((female_selector, male_selector, float(value)))
        return entries
    for male_selector, value in sexual_selection.items():
        assert isinstance(value, float), "In flat sexual_selection form, values must be floats"
        entries.append(("*", male_selector, value))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Initial-state resolution: user dicts → engine arrays
# ─────────────────────────────────────────────────────────────────────────────

GenotypeSelectorAtom = Union[Genotype, str]
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


def compute_expected_eggs_from_distribution(
    equilibrium_distribution: NDArray[np.float64],
    eggs_per_female: float,
    age_based_reproduction_rates: NDArray[np.float64],
    female_age_based_fertility: NDArray[np.float64],
    new_adult_age: int,
    n_ages: int,
) -> float:
    """Compute total expected egg production from an equilibrium distribution.

    The distribution-side twin of
    :func:`compute_expected_eggs_from_females`: instead of forward
    propagating a target adult-female count, it integrates the egg
    production over the adult female column of an explicit equilibrium
    distribution.

    Args:
        equilibrium_distribution: (2, n_ages) equilibrium distribution.
        eggs_per_female: Base eggs per female.
        age_based_reproduction_rates: Female reproduction participation by age.
        female_age_based_fertility: Relative fertility by age.
        new_adult_age: First adult age class.
        n_ages: Total age classes.

    Returns:
        Total expected egg production.
    """
    eggs = 0.0
    for age in range(new_adult_age, n_ages):
        n_f = float(equilibrium_distribution[0, age])
        p_reproducing = min(1.0, max(0.0, float(age_based_reproduction_rates[age])))
        eggs += n_f * p_reproducing * female_age_based_fertility[age] * eggs_per_female
    return eggs
