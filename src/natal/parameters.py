"""Parameter descriptor registry for the natal simulation model.

Each ``ParamDescriptor`` maps a user-facing Builder parameter to its
``PopulationConfig`` field and array path.  This is the single source
of truth shared by the Builder API and the inference layer
(``natal-inferencer``).

Usage::

    from natal.parameters import ALL_PARAMETERS

    desc = ALL_PARAMETERS["competition.carrying_capacity"]
    assert desc.config_field == "carrying_capacity"
    assert desc.config_path == ()
    assert desc.dtype is float
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = [
    "ParameterDomain",
    "ParamDescriptor",
    "ALL_PARAMETERS",
    "PARAMETERS_BY_DOMAIN",
]


# ---------------------------------------------------------------------------
# ParameterDomain — mirrors the Builder method chain
# ---------------------------------------------------------------------------


class ParameterDomain(enum.Enum):
    """Builder method that owns the parameter."""

    SETUP = "setup"
    AGE_STRUCTURE = "age_structure"
    INITIAL_STATE = "initial_state"
    SURVIVAL = "survival"
    REPRODUCTION = "reproduction"
    COMPETITION = "competition"
    FITNESS = "fitness"
    HOOK = "hook"
    MIGRATION = "migration"


# ---------------------------------------------------------------------------
# ParamDescriptor — single estimable parameter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamDescriptor:
    """Describes one estimable parameter of the population model.

    Attributes:
        domain: Builder method that sets this parameter.
        name: User-facing name (e.g. ``"carrying_capacity"``).
        config_field: ``PopulationConfig`` field name.
        config_path: Index path into the config array.  Scalars use ``()``.
        dtype: Python type (``float``, ``int``, or ``bool``).
        bounds: Plausible range ``(lo, hi)`` for prior construction.
        doc: One-line description.
    """

    domain: ParameterDomain
    name: str
    config_field: str
    config_path: tuple[int, ...]
    dtype: type
    bounds: tuple[float, float]
    doc: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARAMS: list[ParamDescriptor] = []


def _reg(
    domain: ParameterDomain,
    name: str,
    config_field: str,
    config_path: tuple[int, ...],
    dtype: type,
    bounds: tuple[float, float],
    doc: str = "",
) -> ParamDescriptor:
    d = ParamDescriptor(domain, name, config_field, config_path, dtype, bounds, doc)
    _PARAMS.append(d)
    return d


# =============================================================================
# SETUP
# =============================================================================

D = ParameterDomain

_reg(D.SETUP, "stochastic", "is_stochastic", (), bool, (0, 1))
_reg(D.SETUP, "continuous_sampling", "use_continuous_sampling", (), bool, (0, 1))
_reg(D.SETUP, "fixed_egg_count", "use_fixed_egg_count", (), bool, (0, 1))

# =============================================================================
# AGE_STRUCTURE
# =============================================================================

_reg(D.AGE_STRUCTURE, "n_ages", "n_ages", (), int, (1, 200))
_reg(D.AGE_STRUCTURE, "new_adult_age", "new_adult_age", (), int, (1, 100))
_reg(D.AGE_STRUCTURE, "generation_time", "generation_time", (), float, (0.01, 100))

# =============================================================================
# SURVIVAL
# =============================================================================

_reg(D.SURVIVAL, "female_survival", "age_based_survival_rates", (0,), float, (0, 1))
_reg(D.SURVIVAL, "male_survival", "age_based_survival_rates", (1,), float, (0, 1))
_reg(D.SURVIVAL, "female_age0_survival", "age_based_survival_rates", (0, 0), float, (0, 1))
_reg(D.SURVIVAL, "male_age0_survival", "age_based_survival_rates", (1, 0), float, (0, 1))
_reg(D.SURVIVAL, "adult_survival", "age_based_survival_rates", (0, 1), float, (0, 1))

# =============================================================================
# REPRODUCTION
# =============================================================================

_reg(D.REPRODUCTION, "eggs_per_female", "expected_eggs_per_female", (), float, (0, 1e6))
_reg(D.REPRODUCTION, "sex_ratio", "sex_ratio", (), float, (0, 1))
_reg(D.REPRODUCTION, "female_mating_rate", "age_based_mating_rates", (0,), float, (0, 1))
_reg(D.REPRODUCTION, "male_mating_rate", "age_based_mating_rates", (1,), float, (0, 1))
_reg(D.REPRODUCTION, "female_adult_mating_rate", "age_based_mating_rates", (0, 1), float, (0, 1))
_reg(D.REPRODUCTION, "male_adult_mating_rate", "age_based_mating_rates", (1, 1), float, (0, 1))
_reg(D.REPRODUCTION, "reproduction_rate", "age_based_reproduction_rates", (), float, (0, 1))
_reg(D.REPRODUCTION, "sperm_displacement_rate", "sperm_displacement_rate", (), float, (0, 1))

# =============================================================================
# COMPETITION
# =============================================================================

_reg(D.COMPETITION, "competition_strength", "age_based_relative_competition_strength", (1,), float, (0, 1e6))
_reg(D.COMPETITION, "juvenile_growth_mode", "juvenile_growth_mode", (), int, (0, 2))
_reg(D.COMPETITION, "low_density_growth_rate", "low_density_growth_rate", (), float, (0, 1e6))
_reg(D.COMPETITION, "carrying_capacity", "carrying_capacity", (), float, (0, 1e12))

# =============================================================================
# FITNESS
# =============================================================================

_reg(D.FITNESS, "viability", "viability_fitness", (), float, (0, 100))
_reg(D.FITNESS, "fecundity", "fecundity_fitness", (), float, (0, 100))
_reg(D.FITNESS, "sexual_selection", "sexual_selection_fitness", (), float, (0, 100))
_reg(D.FITNESS, "zygote_viability", "zygote_viability_fitness", (), float, (0, 100))

# =============================================================================
# MIGRATION (spatial only)
# =============================================================================

_reg(D.MIGRATION, "migration_rate", "migration_rate", (), float, (0, 1))

# =============================================================================
# Build the registry dicts
# =============================================================================

ALL_PARAMETERS: dict[str, ParamDescriptor] = {
    f"{d.domain.value}.{d.name}": d for d in _PARAMS
}

PARAMETERS_BY_DOMAIN: dict[ParameterDomain, dict[str, ParamDescriptor]] = {}
for d in _PARAMS:
    PARAMETERS_BY_DOMAIN.setdefault(d.domain, {})[d.name] = d
