"""Discrete-generation population configuration.

A dedicated config for the non-overlapping generation model.  All age-
dimensioned arrays are at most length 2.  Scalar fields accessible to
compiled kernels eliminate runtime array-indexing overhead.
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

    # -- Sampling --
    is_stochastic: bool
    use_continuous_sampling: bool

    # -- Dimensions --
    n_sexes: int                    # always 2
    n_ages: int                     # always 2
    n_genotypes: int
    n_haploid_genotypes: int
    n_glabs: int

    # -- Age-structured arrays (n_ages ≤ 2, kept for spatial builder) --
    age_based_mating_rates: NDArray[np.float64]        # (2, 2)
    age_based_reproduction_rates: NDArray[np.float64]   # (2,)
    age_based_survival_rates: NDArray[np.float64]       # (2, 2)
    female_age_based_relative_fertility: NDArray[np.float64]  # (2,)

    # -- Viability / fecundity / fitness (full arrays for preset compatibility) --
    viability_fitness: NDArray[np.float64]              # (2, 2, g)
    fecundity_fitness: NDArray[np.float64]              # (2, g)
    zygote_viability_fitness: NDArray[np.float64]       # (2, g)
    sexual_selection_fitness: NDArray[np.float64]        # (g, g)

    # -- Competition --
    age_based_relative_competition_strength: NDArray[np.float64]  # (2,)

    # -- Reproduction scalars --
    expected_eggs_per_female: float
    use_fixed_egg_count: bool
    sex_ratio: float
    sperm_displacement_rate: float

    # -- Per-sex scalars (pre-extracted for kernel performance) --
    male_mating_rate: float
    female_mating_rate: float
    reproduction_rate: float
    base_survival_f: float
    base_survival_m: float

    # -- Reproduction arrays --
    genotype_to_gametes_map: NDArray[np.float64]         # (2, g, hl)
    gametes_to_zygote_map: NDArray[np.float64]           # (hl, hl, g)
    offspring_tensor: NDArray[np.float64]                # (g, g, g)

    # -- Per-sex array views (pre-extracted, share memory with full arrays) --
    meiosis_f: NDArray[np.float64]                      # (g, hl) — genotype_to_gametes_map[0]
    meiosis_m: NDArray[np.float64]                      # (g, hl) — genotype_to_gametes_map[1]
    fecundity_f: NDArray[np.float64]                    # (g,) — fecundity_fitness[0]
    fecundity_m: NDArray[np.float64]                    # (g,) — fecundity_fitness[1]
    viability_f: NDArray[np.float64]                    # (g,) — viability_fitness[0, 0, :]
    viability_m: NDArray[np.float64]                    # (g,) — viability_fitness[1, 0, :]

    # -- Sex chromosomes --
    has_sex_chromosomes: bool
    female_genotype_compatibility: NDArray[np.float64]    # (g,)
    male_genotype_compatibility: NDArray[np.float64]      # (g,)
    female_only_by_sex_chrom: NDArray[np.bool_]           # (g,)
    male_only_by_sex_chrom: NDArray[np.bool_]             # (g,)

    # -- Competition scalars --
    juvenile_growth_mode: int
    carrying_capacity: float
    base_carrying_capacity: float
    base_expected_num_adult_females: float
    expected_competition_strength: float
    expected_survival_rate: float
    low_density_growth_rate: float
    generation_time: float

    # -- Age structure --
    new_adult_age: int
    adult_ages: NDArray[np.int64]                         # [1]

    # -- Init --
    initial_individual_count: NDArray[np.float64]         # (2, 2, g)
    initial_sperm_storage: NDArray[np.float64]            # (2, g, g) — empty for discrete
    population_scale: float
    hook_slot: int

    # ── methods ──────────────────────────────────────────────────────────────

    def get_scaled_initial_individual_count(self) -> NDArray[np.float64]:
        return self.initial_individual_count * float(self.population_scale)

    def get_scaled_initial_sperm_storage(self) -> NDArray[np.float64]:
        return self.initial_sperm_storage * float(self.population_scale)

    # -- setters compatible with PopulationConfig API, used by presets --

    def set_viability_fitness(
        self, sex: int, genotype_idx: int, value: float, age: int = -1
    ) -> None:
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, genotype_idx] = value

    def set_fecundity_fitness(
        self, sex: int, genotype_idx: int, value: float
    ) -> None:
        self.fecundity_fitness[sex, genotype_idx] = value

    def set_sexual_selection_fitness(
        self, female_geno_idx: int, male_geno_idx: int, value: float
    ) -> None:
        self.sexual_selection_fitness[female_geno_idx, male_geno_idx] = value

    def set_zygote_viability_fitness(
        self, sex: int, genotype_idx: int, value: float
    ) -> None:
        self.zygote_viability_fitness[sex, genotype_idx] = value

    def get_effective_carrying_capacity(self) -> float:
        return self.carrying_capacity * float(self.population_scale)

    def get_effective_expected_adult_females(self) -> float:
        return self.base_expected_num_adult_females * float(self.population_scale)

    def set_population_scale(self, scale: float) -> DiscretePopulationConfig:
        return self._replace(
            population_scale=float(scale),
            carrying_capacity=self.base_carrying_capacity * float(scale),
        )


# ── bridge: PopulationConfig → DiscretePopulationConfig ────────────────────


def from_population_config(cfg: PopulationConfig) -> DiscretePopulationConfig:
    """Build a ``DiscretePopulationConfig`` from a full ``PopulationConfig``."""
    return DiscretePopulationConfig(
        is_stochastic=cfg.is_stochastic,
        use_continuous_sampling=cfg.use_continuous_sampling,
        n_sexes=int(cfg.n_sexes),
        n_ages=int(cfg.n_ages),
        n_genotypes=cfg.n_genotypes,
        n_haploid_genotypes=cfg.n_haploid_genotypes,
        n_glabs=cfg.n_glabs,
        age_based_mating_rates=cfg.age_based_mating_rates,
        age_based_reproduction_rates=cfg.age_based_reproduction_rates,
        age_based_survival_rates=cfg.age_based_survival_rates,
        female_age_based_relative_fertility=cfg.female_age_based_relative_fertility,
        viability_fitness=cfg.viability_fitness,
        fecundity_fitness=cfg.fecundity_fitness,
        zygote_viability_fitness=cfg.zygote_viability_fitness,
        sexual_selection_fitness=cfg.sexual_selection_fitness,
        age_based_relative_competition_strength=cfg.age_based_relative_competition_strength,
        expected_eggs_per_female=cfg.expected_eggs_per_female,
        use_fixed_egg_count=cfg.use_fixed_egg_count,
        sex_ratio=cfg.sex_ratio,
        sperm_displacement_rate=cfg.sperm_displacement_rate,
        male_mating_rate=float(cfg.age_based_mating_rates[1, 1]),
        female_mating_rate=float(cfg.age_based_mating_rates[0, 1]),
        reproduction_rate=float(cfg.age_based_reproduction_rates[1]),
        base_survival_f=float(cfg.age_based_survival_rates[0, 0]),
        base_survival_m=float(cfg.age_based_survival_rates[1, 0]),
        genotype_to_gametes_map=cfg.genotype_to_gametes_map,
        gametes_to_zygote_map=cfg.gametes_to_zygote_map,
        offspring_tensor=cfg.offspring_tensor,
        meiosis_f=cfg.genotype_to_gametes_map[0],
        meiosis_m=cfg.genotype_to_gametes_map[1],
        fecundity_f=cfg.fecundity_fitness[0],
        fecundity_m=cfg.fecundity_fitness[1],
        viability_f=cfg.viability_fitness[0, 0, :],
        viability_m=cfg.viability_fitness[1, 0, :],
        has_sex_chromosomes=cfg.has_sex_chromosomes,
        female_genotype_compatibility=cfg.female_genotype_compatibility,
        male_genotype_compatibility=cfg.male_genotype_compatibility,
        female_only_by_sex_chrom=cfg.female_only_by_sex_chrom,
        male_only_by_sex_chrom=cfg.male_only_by_sex_chrom,
        juvenile_growth_mode=cfg.juvenile_growth_mode,
        carrying_capacity=cfg.carrying_capacity,
        base_carrying_capacity=cfg.base_carrying_capacity,
        base_expected_num_adult_females=cfg.base_expected_num_adult_females,
        expected_competition_strength=cfg.expected_competition_strength,
        expected_survival_rate=cfg.expected_survival_rate,
        low_density_growth_rate=cfg.low_density_growth_rate,
        generation_time=cfg.generation_time,
        new_adult_age=int(cfg.new_adult_age),
        adult_ages=cfg.adult_ages.copy(),
        initial_individual_count=cfg.initial_individual_count,
        initial_sperm_storage=cfg.initial_sperm_storage,
        population_scale=cfg.population_scale,
        hook_slot=int(cfg.hook_slot),
    )
