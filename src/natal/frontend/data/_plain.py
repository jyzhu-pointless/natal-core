"""Plain config serialization helpers.

This private module contains functions for converting PopulationConfig to/from
a plain (copied) representation.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from .config import PopulationConfig


def _maybe_copy_array(arr: NDArray[Any], copy: bool) -> NDArray[Any]:
    """Helper to conditionally copy a NumPy array."""
    return arr.copy() if copy else arr


def to_plain_population_config(config: PopulationConfig, copy: bool = True) -> PopulationConfig:
    """Convert config object to a plain (copied) PopulationConfig.

    If `copy` is True, all arrays are deep‑copied; otherwise they are referenced
    directly.

    Args:
        config: Input PopulationConfig instance.
        copy: Whether to copy the arrays.

    Returns:
        A new PopulationConfig instance (with the same scalar values).
    """
    return PopulationConfig(
        stochastic=bool(config.stochastic),
        continuous_sampling=bool(config.continuous_sampling),
        n_sexes=int(config.n_sexes),
        n_ages=int(config.n_ages),
        n_ztypes=int(config.n_ztypes),
        n_gtypes=int(config.n_gtypes),
        n_glabs=int(config.n_glabs),
        n_slabs=int(config.n_slabs),
        age_based_mating_rates=_maybe_copy_array(config.age_based_mating_rates, copy),
        age_based_reproduction_rates=_maybe_copy_array(config.age_based_reproduction_rates, copy),
        age_based_survival_rates=_maybe_copy_array(config.age_based_survival_rates, copy),
        female_age_based_fertility=_maybe_copy_array(config.female_age_based_fertility, copy),
        viability_fitness=_maybe_copy_array(config.viability_fitness, copy),
        fecundity_fitness=_maybe_copy_array(config.fecundity_fitness, copy),
        sexual_selection_fitness=_maybe_copy_array(config.sexual_selection_fitness, copy),
        zygote_viability_fitness=_maybe_copy_array(config.zygote_viability_fitness, copy),
        age_based_relative_competition_strength=_maybe_copy_array(config.age_based_relative_competition_strength, copy),
        sperm_displacement_rate=config.sperm_displacement_rate.copy() if copy else config.sperm_displacement_rate,
        eggs_per_female=config.eggs_per_female.copy() if copy else config.eggs_per_female,
        fixed_egg_count=bool(config.fixed_egg_count),
        carrying_capacity=config.carrying_capacity.copy() if copy else config.carrying_capacity,
        sex_ratio=config.sex_ratio.copy() if copy else config.sex_ratio,
        low_density_growth_rate=config.low_density_growth_rate.copy() if copy else config.low_density_growth_rate,
        juvenile_growth_mode=config.juvenile_growth_mode.copy() if copy else config.juvenile_growth_mode,
        expected_competition_strength=config.expected_competition_strength.copy() if copy else config.expected_competition_strength,
        expected_survival_rate=config.expected_survival_rate.copy() if copy else config.expected_survival_rate,
        generation_time=config.generation_time.copy() if copy else config.generation_time,
        new_adult_age=int(config.new_adult_age),
        hook_slot=int(config.hook_slot),
        has_sex_chromosomes=bool(config.has_sex_chromosomes),
        female_ztype_compatibility=_maybe_copy_array(config.female_ztype_compatibility, copy),
        male_ztype_compatibility=_maybe_copy_array(config.male_ztype_compatibility, copy),
        female_only_by_sex_chrom=_maybe_copy_array(config.female_only_by_sex_chrom, copy),
        male_only_by_sex_chrom=_maybe_copy_array(config.male_only_by_sex_chrom, copy),
        adult_ages=config.adult_ages.copy() if copy else config.adult_ages,
        zygotes_to_gametes_map=_maybe_copy_array(config.zygotes_to_gametes_map, copy),
        gametes_to_zygotes_map=_maybe_copy_array(config.gametes_to_zygotes_map, copy),
        offspring_tensor=_maybe_copy_array(config.offspring_tensor, copy),
        initial_individual_count=_maybe_copy_array(config.initial_individual_count, copy),
        initial_sperm_storage=_maybe_copy_array(config.initial_sperm_storage, copy),
        equilibrium_individual_distribution=config.equilibrium_individual_distribution,
        custom=config.custom.copy() if copy else config.custom,
    )


def from_plain_population_config(plain: PopulationConfig) -> PopulationConfig:
    """Compatibility adapter: returns a copied PopulationConfig.

    Args:
        plain: Input PopulationConfig.

    Returns:
        A copied PopulationConfig (arrays are deep‑copied).
    """
    return to_plain_population_config(plain, copy=True)
