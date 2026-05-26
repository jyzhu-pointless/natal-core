"""Parameter descriptor registry for the natal simulation model.

Each ``ParamDescriptor`` maps a user-facing Builder parameter to its
``PopulationConfig`` field and array path.  This is the single source
of truth shared by the Builder API, ``Configurator``, the spatial
builder dispatch, and the inference layer (``natal-inferencer``).

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
    """Describes one parameter of the population model.

    Attributes:
        domain: Builder method that sets this parameter.
        name: User-facing name (e.g. ``"carrying_capacity"``).
        config_field: ``PopulationConfig`` field name, or ``None`` if the
            parameter lives outside PopulationConfig (e.g. spatial-only).
        config_path: Index path into the config array.  Scalars use ``()``.
        dtype: Python type (``float``, ``int``, or ``bool``).
        bounds: Plausible range ``(lo, hi)`` for prior construction.
        doc: One-line description.
        aliases: Historical names that map to this parameter.
        is_tensor: If True, the config field is a multi-dimensional array
            (e.g. fitness tensors) rather than a scalar or 1-D array element.
        is_0d: If True, the config field is a 0-d ndarray (writable via
            ``field[()] = v``).  Python scalars should leave this as False.
        is_array: If True, the parameter writes to a 1-D or 2-D array slice
            (e.g. ``field[0, :] = values``).  Not handled by ``set_param``
            or ``set_config_param`` — Configurator writes directly.
        target: Where the parameter lives — ``"config"`` (PopulationConfig),
            ``"spatial"`` (SpatialPopulation), or ``"hook"`` (hook runtime).
    """

    domain: ParameterDomain
    name: str
    config_field: str | None
    config_path: tuple[int, ...]
    dtype: type
    bounds: tuple[float, float]
    doc: str = ""
    aliases: tuple[str, ...] = ()
    is_tensor: bool = False
    is_0d: bool = False
    is_array: bool = False
    target: str = "config"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARAMS: list[ParamDescriptor] = []


def _register(
    domain: ParameterDomain,
    name: str,
    config_field: str | None,
    config_path: tuple[int, ...],
    dtype: type,
    bounds: tuple[float, float],
    doc: str = "",
    *,
    aliases: tuple[str, ...] = (),
    is_tensor: bool = False,
    is_0d: bool = False,
    is_array: bool = False,
    target: str = "config",
) -> ParamDescriptor:
    d = ParamDescriptor(
        domain, name, config_field, config_path, dtype, bounds,
        doc=doc, aliases=aliases, is_tensor=is_tensor, is_0d=is_0d,
        is_array=is_array, target=target,
    )
    _PARAMS.append(d)
    return d


D = ParameterDomain

# =============================================================================
# SETUP
# =============================================================================

_register(
    domain=D.SETUP, name="stochastic",
    config_field="is_stochastic", config_path=(),
    dtype=bool, bounds=(0, 1)
)

_register(
    domain=D.SETUP, name="continuous_sampling",
    config_field="use_continuous_sampling", config_path=(),
    dtype=bool, bounds=(0, 1)
)

_register(
    domain=D.SETUP, name="fixed_egg_count",
    config_field="use_fixed_egg_count", config_path=(),
    dtype=bool, bounds=(0, 1)
)

_register(
    domain=D.SETUP, name="has_sex_chromosomes",
    config_field="has_sex_chromosomes", config_path=(),
    dtype=bool, bounds=(0, 1)
)

# =============================================================================
# AGE_STRUCTURE
# =============================================================================

_register(
    domain=D.AGE_STRUCTURE, name="n_ages",
    config_field="n_ages", config_path=(),
    dtype=int, bounds=(1, 200)
)

_register(
    domain=D.AGE_STRUCTURE, name="n_sexes",
    config_field="n_sexes", config_path=(),
    dtype=int, bounds=(1, 10)
)

_register(
    domain=D.AGE_STRUCTURE, name="n_genotypes",
    config_field="n_genotypes", config_path=(),
    dtype=int, bounds=(1, 100_000)
)

_register(
    domain=D.AGE_STRUCTURE, name="n_haploid_genotypes",
    config_field="n_haploid_genotypes", config_path=(),
    dtype=int, bounds=(1, 100_000)
)

_register(
    domain=D.AGE_STRUCTURE, name="n_glabs",
    config_field="n_glabs", config_path=(),
    dtype=int, bounds=(1, 100)
)

_register(
    domain=D.AGE_STRUCTURE, name="new_adult_age",
    config_field="new_adult_age", config_path=(),
    dtype=int, bounds=(1, 100)
)

_register(
    domain=D.AGE_STRUCTURE, name="generation_time",
    config_field="generation_time", config_path=(),
    dtype=float, bounds=(0.01, 100),
    is_0d=True,
)

# =============================================================================
# INITIAL_STATE
# =============================================================================

_register(
    domain=D.INITIAL_STATE, name="initial_individual_count",
    config_field="initial_individual_count", config_path=(),
    dtype=float, bounds=(0, 1e12),
    is_tensor=True,
    doc="(n_sexes, n_ages, n_genotypes) initial population distribution"
)

_register(
    domain=D.INITIAL_STATE, name="initial_sperm_storage",
    config_field="initial_sperm_storage", config_path=(),
    dtype=float, bounds=(0, 1e12),
    is_tensor=True,
    doc="(n_ages, n_genotypes, n_genotypes) initial sperm storage"
)

# =============================================================================
# SURVIVAL
# =============================================================================

_register(
    domain=D.SURVIVAL, name="female_survival",
    config_field="age_based_survival_rates", config_path=(0,),
    dtype=float, bounds=(0, 1),
    doc="(n_ages,) female survival rates — use per-age sub-params for scalar estimation"
)

_register(
    domain=D.SURVIVAL, name="male_survival",
    config_field="age_based_survival_rates", config_path=(1,),
    dtype=float, bounds=(0, 1),
    doc="(n_ages,) male survival rates"
)

_register(
    domain=D.SURVIVAL, name="female_age0_survival",
    config_field="age_based_survival_rates", config_path=(0, 0),
    dtype=float, bounds=(0, 1)
)

_register(
    domain=D.SURVIVAL, name="male_age0_survival",
    config_field="age_based_survival_rates", config_path=(1, 0),
    dtype=float, bounds=(0, 1)
)

_register(
    domain=D.SURVIVAL, name="female_survival_rates",
    config_field="age_based_survival_rates", config_path=(0,),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age female survival — array write via resolve_age_param"
)

_register(
    domain=D.SURVIVAL, name="male_survival_rates",
    config_field="age_based_survival_rates", config_path=(1,),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age male survival — array write via resolve_age_param"
)

_register(
    domain=D.SURVIVAL, name="adult_survival",
    config_field="age_based_survival_rates", config_path=(),
    dtype=float, bounds=(0, 1),
    doc="scalar applied to all adult ages (≥ new_adult_age)"
)

# =============================================================================
# REPRODUCTION
# =============================================================================

_register(
    domain=D.REPRODUCTION, name="eggs_per_female",
    config_field="expected_eggs_per_female", config_path=(),
    dtype=float, bounds=(0, 1e6),
    aliases=("expected_eggs_per_female",),
    is_0d=True,
)

_register(
    domain=D.REPRODUCTION, name="sex_ratio",
    config_field="sex_ratio", config_path=(),
    dtype=float, bounds=(0, 1),
    is_0d=True,
)

_register(
    domain=D.REPRODUCTION, name="female_mating_rate",
    config_field="age_based_mating_rates", config_path=(0,),
    dtype=float, bounds=(0, 1),
    doc="(n_ages,) female mating rates — use per-age sub-params for scalar estimation"
)

_register(
    domain=D.REPRODUCTION, name="male_mating_rate",
    config_field="age_based_mating_rates", config_path=(1,),
    dtype=float, bounds=(0, 1),
    doc="(n_ages,) male mating rates"
)

_register(
    domain=D.REPRODUCTION, name="female_adult_mating_rate",
    config_field="age_based_mating_rates", config_path=(0, 1),
    dtype=float, bounds=(0, 1)
)

_register(
    domain=D.REPRODUCTION, name="male_adult_mating_rate",
    config_field="age_based_mating_rates", config_path=(1, 1),
    dtype=float, bounds=(0, 1)
)

_register(
    domain=D.REPRODUCTION, name="reproduction_rate",
    config_field="age_based_reproduction_rates", config_path=(1,),
    dtype=float, bounds=(0, 1)
)

_register(
    domain=D.REPRODUCTION, name="sperm_displacement_rate",
    config_field="sperm_displacement_rate", config_path=(),
    dtype=float, bounds=(0, 1),
    is_0d=True,
)

_register(
    domain=D.REPRODUCTION, name="female_fertility",
    config_field="female_age_based_relative_fertility", config_path=(0,),
    dtype=float, bounds=(0, 1),
    doc="(n_ages,) female relative fertility"
)

_register(
    domain=D.REPRODUCTION, name="female_mating_rates",
    config_field="age_based_mating_rates", config_path=(0,),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age female mating rates — array write via resolve_age_param"
)

_register(
    domain=D.REPRODUCTION, name="male_mating_rates",
    config_field="age_based_mating_rates", config_path=(1,),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age male mating rates — array write via resolve_age_param"
)

_register(
    domain=D.REPRODUCTION, name="reproduction_rates",
    config_field="age_based_reproduction_rates", config_path=(),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age female reproduction participation"
)

_register(
    domain=D.REPRODUCTION, name="female_fertility_rates",
    config_field="female_age_based_relative_fertility", config_path=(),
    dtype=float, bounds=(0, 1),
    is_array=True,
    doc="(n_ages,) per-age female relative fertility — array write via resolve_age_param"
)

# =============================================================================
# COMPETITION
# =============================================================================

_register(
    domain=D.COMPETITION, name="competition_strength",
    config_field="age_based_relative_competition_strength", config_path=(1,),
    dtype=float, bounds=(0, 1e6),
    aliases=("relative_competition_factor",)
)

_register(
    domain=D.COMPETITION, name="juvenile_growth_mode",
    config_field="juvenile_growth_mode", config_path=(),
    dtype=int, bounds=(0, 3),
    is_0d=True,
)

_register(
    domain=D.COMPETITION, name="low_density_growth_rate",
    config_field="low_density_growth_rate", config_path=(),
    dtype=float, bounds=(0, 1e6),
    is_0d=True,
)

_register(
    domain=D.COMPETITION, name="carrying_capacity",
    config_field="carrying_capacity", config_path=(),
    dtype=float, bounds=(0, 1e12),
    aliases=("age_1_carrying_capacity", "old_juvenile_carrying_capacity"),
    is_0d=True,
)

_register(
    domain=D.COMPETITION, name="expected_competition_strength",
    config_field="expected_competition_strength", config_path=(),
    dtype=float, bounds=(0, 1e6),
    is_0d=True,
)

_register(
    domain=D.COMPETITION, name="expected_survival_rate",
    config_field="expected_survival_rate", config_path=(),
    dtype=float, bounds=(0, 1),
    is_0d=True,
)

# =============================================================================
# FITNESS
# =============================================================================

_register(
    domain=D.FITNESS, name="viability",
    config_field="viability_fitness", config_path=(),
    dtype=float, bounds=(0, 100),
    is_tensor=True,
    doc="(n_sexes, n_ages, n_genotypes) viability fitness tensor"
)

_register(
    domain=D.FITNESS, name="fecundity",
    config_field="fecundity_fitness", config_path=(),
    dtype=float, bounds=(0, 100),
    is_tensor=True,
    doc="(n_sexes, n_genotypes) fecundity fitness"
)

_register(
    domain=D.FITNESS, name="sexual_selection",
    config_field="sexual_selection_fitness", config_path=(),
    dtype=float, bounds=(0, 100),
    is_tensor=True,
    doc="(n_genotypes, n_genotypes) sexual selection matrix"
)

_register(
    domain=D.FITNESS, name="zygote_viability",
    config_field="zygote_viability_fitness", config_path=(),
    dtype=float, bounds=(0, 100),
    is_tensor=True,
    doc="(n_sexes, n_genotypes) zygote viability fitness"
)

# =============================================================================
# HOOK
# =============================================================================

_register(
    domain=D.HOOK, name="hook_slot",
    config_field="hook_slot", config_path=(),
    dtype=int, bounds=(0, 100)
)

# =============================================================================
# MIGRATION (spatial only)
# =============================================================================

_register(
    domain=D.MIGRATION, name="migration_rate",
    config_field=None, config_path=(),
    dtype=float, bounds=(0, 1),
    target="spatial",
    doc="Migration rate stored on SpatialPopulation, not PopulationConfig"
)

# =============================================================================
# Sex-chromosome arrays (tensors, for introspection)
# =============================================================================

_register(
    domain=D.FITNESS, name="female_genotype_compatibility",
    config_field="female_genotype_compatibility", config_path=(),
    dtype=float, bounds=(0, 1),
    is_tensor=True,
    doc="(n_genotypes,) female-side compatibility weights"
)

_register(
    domain=D.FITNESS, name="male_genotype_compatibility",
    config_field="male_genotype_compatibility", config_path=(),
    dtype=float, bounds=(0, 1),
    is_tensor=True,
    doc="(n_genotypes,) male-side compatibility weights"
)

# =============================================================================
# Build the registry dicts
# =============================================================================

ALL_PARAMETERS: dict[str, ParamDescriptor] = {
    f"{d.domain.value}.{d.name}": d for d in _PARAMS
}

PARAMETERS_BY_DOMAIN: dict[ParameterDomain, dict[str, ParamDescriptor]] = {}
for d in _PARAMS:
    PARAMETERS_BY_DOMAIN.setdefault(d.domain, {})[d.name] = d

PARAM_IDS: dict[str, int] = {
    f"{d.domain.value}.{d.name}": i for i, d in enumerate(_PARAMS)
}
