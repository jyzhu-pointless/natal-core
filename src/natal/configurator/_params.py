"""Parameter resolution helpers extracted from population_builder and configurator.

These are standalone functions used by both the Configurator and the legacy
PopulationConfigBuilder.  Extracting them into a shared module avoids circular
imports between _base.py, _factory.py, discrete.py, and age_structured.py.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    BEVERTON_HOLT,
    CONCAVE,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
)
from natal.utils.parameters import ALL_PARAMETERS, ParamDescriptor

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
    """Resolve juvenile growth mode string or int to internal constant."""
    if isinstance(mode, int):
        if mode not in [NO_COMPETITION, FIXED, LOGISTIC, CONCAVE, BEVERTON_HOLT, LINEAR]:
            raise ValueError(f"Invalid growth mode constant: {mode}")
        return mode
    mode_map = {
        'NO_COMPETITION': NO_COMPETITION, 'FIXED': FIXED,
        'LOGISTIC': LOGISTIC, 'CONCAVE': CONCAVE,
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
    expected_num_adult_females: float,
    eggs_per_female: float,
    age_based_survival_rates: NDArray[np.float64],
    age_based_reproduction_rates: Optional[NDArray[np.float64]],
    female_age_based_fertility: NDArray[np.float64],
    sex_ratio: float,
    new_adult_age: int,
    n_ages: int,
) -> float:
    """Compute total expected egg production from a target adult female count.

    Forward-propagates ``expected_num_adult_females`` across adult ages via
    survival rates (same direction as ``compute_equilibrium_metrics``), then
    computes total egg production from the resulting age-specific female counts.

    Args:
        expected_num_adult_females: Number of adult females at new_adult_age.
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
    female_dist[new_adult_age] = expected_num_adult_females
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


def resolve_param(name: str) -> ParamDescriptor | None:
    """Look up a parameter name in ``ALL_PARAMETERS`` with three fallback tiers.

    Tier 1: exact match — ``"competition.carrying_capacity"``.
    Tier 2: short-name match — ``"carrying_capacity"`` matches via
            ``key.endswith(".carrying_capacity")``.
    Tier 3: alias match — user-friendly names defined in each
            ``ParamDescriptor.aliases``.

    Returns the ``ParamDescriptor`` or ``None``.
    """
    # Tier 1: O(1) exact key lookup.
    if name in ALL_PARAMETERS:
        return ALL_PARAMETERS[name]

    # Tier 2-3: linear scan for short-name / alias match.
    for key, desc in ALL_PARAMETERS.items():
        if key.endswith(f".{name}"):
            return desc
        if name in desc.aliases:
            return desc

    return None
