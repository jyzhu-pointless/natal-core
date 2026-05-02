"""Discrete-generation population configuration.

A slimmed-down config for the non-overlapping generation model. All
age-dimension arrays are collapsed to scalars (n_ages is always 2), and
sperm-storage fields are omitted.  The offspring probability tensor is
precomputed at construction time.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from natal.population_config import PopulationConfig

__all__ = [
    "DiscretePopulationConfig",
    "from_population_config",
]


class DiscretePopulationConfig(NamedTuple):
    """Immutable configuration for discrete-generation simulations."""

    # Sampling
    is_stochastic: bool
    use_continuous_sampling: bool

    # Dimensions (n_ages=2, n_sexes=2 are implicit)
    n_genotypes: int
    n_haploid_genotypes: int
    n_glabs: int

    # Reproduction — scalars
    expected_eggs_per_female: float
    sex_ratio: float
    male_mating_rate: float
    female_mating_rate: float
    reproduction_rate: float

    # Reproduction — arrays
    sexual_selection_fitness: NDArray[np.float64]        # (g, g)
    fecundity_f: NDArray[np.float64]                      # (g,)
    fecundity_m: NDArray[np.float64]                      # (g,)
    meiosis_f: NDArray[np.float64]                         # (g, hl)
    meiosis_m: NDArray[np.float64]                         # (g, hl)
    gametes_to_zygote_map: NDArray[np.float64]             # (hl, hl, g)
    offspring_tensor: NDArray[np.float64]                  # (g, g, g) — precomputed

    # Sex chromosomes
    has_sex_chromosomes: bool
    female_genotype_compatibility: NDArray[np.float64]    # (g,)
    male_genotype_compatibility: NDArray[np.float64]      # (g,)
    female_only_by_sex_chrom: NDArray[np.bool_]           # (g,)
    male_only_by_sex_chrom: NDArray[np.bool_]             # (g,)

    # Survival
    base_survival_f: float
    base_survival_m: float
    viability_f: NDArray[np.float64]                      # (g,)
    viability_m: NDArray[np.float64]                      # (g,)

    # Competition
    juvenile_growth_mode: int
    carrying_capacity: float
    expected_competition_strength: float
    expected_survival_rate: float
    low_density_growth_rate: float

    # Init
    initial_individual_count: NDArray[np.float64]         # (2, 2, g)
    population_scale: float
    hook_slot: int


def from_population_config(cfg: PopulationConfig) -> DiscretePopulationConfig:
    """Extract discrete-only fields from a full ``PopulationConfig``."""
    return DiscretePopulationConfig(
        is_stochastic=cfg.is_stochastic,
        use_continuous_sampling=cfg.use_continuous_sampling,
        n_genotypes=cfg.n_genotypes,
        n_haploid_genotypes=cfg.n_haploid_genotypes,
        n_glabs=cfg.n_glabs,
        expected_eggs_per_female=cfg.expected_eggs_per_female,
        sex_ratio=cfg.sex_ratio,
        male_mating_rate=float(cfg.age_based_mating_rates[1, 1]),
        female_mating_rate=float(cfg.age_based_mating_rates[0, 1]),
        reproduction_rate=float(cfg.age_based_reproduction_rates[1]),
        sexual_selection_fitness=cfg.sexual_selection_fitness,
        fecundity_f=cfg.fecundity_fitness[0],
        fecundity_m=cfg.fecundity_fitness[1],
        meiosis_f=cfg.genotype_to_gametes_map[0],
        meiosis_m=cfg.genotype_to_gametes_map[1],
        gametes_to_zygote_map=cfg.gametes_to_zygote_map,
        offspring_tensor=cfg.offspring_tensor,
        has_sex_chromosomes=cfg.has_sex_chromosomes,
        female_genotype_compatibility=cfg.female_genotype_compatibility,
        male_genotype_compatibility=cfg.male_genotype_compatibility,
        female_only_by_sex_chrom=cfg.female_only_by_sex_chrom,
        male_only_by_sex_chrom=cfg.male_only_by_sex_chrom,
        base_survival_f=float(cfg.age_based_survival_rates[0, 0]),
        base_survival_m=float(cfg.age_based_survival_rates[1, 0]),
        viability_f=cfg.viability_fitness[0, 0, :],
        viability_m=cfg.viability_fitness[1, 0, :],
        juvenile_growth_mode=cfg.juvenile_growth_mode,
        carrying_capacity=cfg.carrying_capacity,
        expected_competition_strength=cfg.expected_competition_strength,
        expected_survival_rate=cfg.expected_survival_rate,
        low_density_growth_rate=cfg.low_density_growth_rate,
        initial_individual_count=cfg.initial_individual_count,
        population_scale=cfg.population_scale,
        hook_slot=int(cfg.hook_slot),
    )
