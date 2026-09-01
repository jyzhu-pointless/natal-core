"""Immutable population configuration containers.

This module defines ``PopulationConfig`` and ``DiscretePopulationConfig`` —
two NamedTuple-based configuration containers used to parameterise simulation
engines.  Scalar fields are immutable (rebuild with ``_replace``), while
NumPy arrays can be mutated in place.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from natal.utils.types import Sex

__all__ = [
    'PopulationConfig',
    'DiscretePopulationConfig',
]


class PopulationConfig(NamedTuple):
    """Primary immutable configuration container.

    Scalar fields are immutable (rebuild with ``_replace``). NumPy arrays are
    mutable in-place.

    Attributes:
        stochastic: Whether demographic events are stochastic.
        continuous_sampling: If True, use Dirichlet sampling for gamete
            proportions; otherwise use multinomial sampling.
        n_sexes: Number of sexes (usually 2).
        n_ages: Number of age classes.
        n_ztypes: Number of zygote types (diploid genotype types after slab expansion).
        n_gtypes: Total number of gamete types (haploid genotype count × gamete label count).
        n_glabs: Number of gamete‑label variants per haplotype.
        age_based_mating_rates: Shape (n_sexes, n_ages) – mating rates per sex/age.
        age_based_reproduction_rates: Shape (n_ages,) – female reproduction
            participation rates per age.
        age_based_survival_rates: Shape (n_sexes, n_ages) – survival probabilities.
        female_age_based_fertility: Shape (n_ages,) – relative fertility
            of females at each age.
        viability_fitness: Shape (n_sexes, n_ages, n_ztypes) – viability
            fitness coefficients.
        fecundity_fitness: Shape (n_sexes, n_ztypes) – fecundity fitness
            coefficients.
        sexual_selection_fitness: Shape (n_ztypes, n_ztypes) – sexual
            selection coefficients (female genotype × male genotype).
        zygote_viability_fitness: Shape (n_sexes, n_ztypes) – zygote fitness
            coefficients applied during reproduction stage before survival.
            Represents the probability that a zygote survives to become an
            individual, applied before competition and viability selection.
        age_based_relative_competition_strength: Shape (n_ages,) – relative
            contribution to competition for each age.
        sperm_displacement_rate: Probability that a new mating displaces stored
            sperm.
        eggs_per_female: Expected number of eggs per female per tick.
        fixed_egg_count: If True, use the deterministic expected egg count;
            otherwise sample from a Poisson distribution.
        carrying_capacity: Current carrying capacity (0-d ndarray, mutable).
        sex_ratio: Proportion of newborns that are female.
        low_density_growth_rate: Intrinsic growth rate at low density.
        juvenile_growth_mode: Growth mode for juveniles (see constants).
        expected_competition_strength: Pre‑computed equilibrium competition
            strength.
        expected_survival_rate: Pre‑computed equilibrium survival rate.
        generation_time: Pre‑computed mean generation time.
        new_adult_age: Age at which individuals become adults.
        hook_slot: Slot index for hook functions (reserved).
        has_sex_chromosomes: Whether the species has sex-chromosome constraints
            (e.g., XY or ZW systems). Used to determine if offspring sex is
            genotype-determined (True) or ratio-determined (False). This flag
            is independent of gamete modifier effects or temporary lethality.
        female_ztype_compatibility: Shape (n_ztypes,) – female-side
            compatibility weight per genotype.
        male_ztype_compatibility: Shape (n_ztypes,) – male-side
            compatibility weight per genotype.
        female_only_by_sex_chrom: Shape (n_ztypes,) – True where genotype
            is female-only under sex-chromosome constraints.
        male_only_by_sex_chrom: Shape (n_ztypes,) – True where genotype is
            male-only under sex-chromosome constraints.
        adult_ages: 1D array of age indices that are considered adult.
        zygotes_to_gametes_map: Shape (n_sexes, n_ztypes, n_hg*n_glabs) –
            probability of producing each (haplotype, glab) combination.
        gametes_to_zygotes_map: Shape (n_hg*n_glabs, n_hg*n_glabs, n_ztypes) –
            probability of forming a given diploid genotype from two gametes.
        initial_individual_count: Shape (n_sexes, n_ages, n_ztypes) – initial
            population distribution.
        initial_sperm_storage: Shape (n_ages, n_ztypes, n_ztypes) – initial
            stored sperm counts (female genotype × male genotype).
    """

    # Scalars are immutable; rebuild this NamedTuple for scalar updates.
    stochastic: bool
    continuous_sampling: bool
    n_sexes: int
    n_ages: int
    n_ztypes: int
    n_gtypes: int
    n_glabs: int
    n_slabs: int
    age_based_mating_rates: NDArray[np.float64]
    age_based_reproduction_rates: NDArray[np.float64]
    age_based_survival_rates: NDArray[np.float64]
    female_age_based_fertility: NDArray[np.float64]
    viability_fitness: NDArray[np.float64]
    fecundity_fitness: NDArray[np.float64]
    sexual_selection_fitness: NDArray[np.float64]
    zygote_viability_fitness: NDArray[np.float64]
    age_based_relative_competition_strength: NDArray[np.float64]
    sperm_displacement_rate: NDArray[np.float64]           # 0-d, mutable
    eggs_per_female: NDArray[np.float64]          # 0-d, mutable
    fixed_egg_count: bool
    carrying_capacity: NDArray[np.float64]                 # 0-d, mutable
    sex_ratio: NDArray[np.float64]                         # 0-d, mutable
    low_density_growth_rate: NDArray[np.float64]           # 0-d, mutable
    juvenile_growth_mode: NDArray[np.int64]                # 0-d, mutable
    expected_competition_strength: NDArray[np.float64]     # 0-d, mutable
    expected_survival_rate: NDArray[np.float64]            # 0-d, mutable
    generation_time: NDArray[np.float64]                   # 0-d, mutable
    new_adult_age: int
    hook_slot: int
    has_sex_chromosomes: bool
    female_ztype_compatibility: NDArray[np.float64]
    male_ztype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]
    # NumPy arrays are still mutable in-place.
    adult_ages: NDArray[np.int64]
    zygotes_to_gametes_map: NDArray[np.float64]
    gametes_to_zygotes_map: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]    # (g, g, g) — precomputed from meiosis × zygote maps
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]
    equilibrium_individual_distribution: Optional[NDArray[np.float64]]  # pre-computed equilibrium age distribution

    # -- slab inheritance (set by presets) --

    # -- custom fields (structured numpy scalar, set via Configurator.custom()) --
    custom: NDArray[Any]  # typed structured array when custom fields registered; float64 placeholder otherwise

    def set_viability_fitness(self, sex: int, ztype_idx: int, value: float, age: int = -1) -> None:
        """Set viability fitness for a specific (sex, genotype, age) combination.

        Args:
            sex: Sex index.
            ztype_idx: ZType (genotype) index.
            value: Fitness value.
            age: Age class; if negative, defaults to new_adult_age - 1.
        """
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, ztype_idx] = value

    def set_fecundity_fitness(self, sex: int, ztype_idx: int, value: float) -> None:
        """Set fecundity fitness for a specific (sex, genotype).

        Args:
            sex: Sex index.
            ztype_idx: ZType (genotype) index.
            value: Fitness value.
        """
        self.fecundity_fitness[sex, ztype_idx] = value

    def set_sexual_selection_fitness(self, female_ztype_idx: int, male_ztype_idx: int, value: float) -> None:
        """Set sexual selection fitness for a female‑male genotype pair.

        Args:
            female_ztype_idx: Female ZType (genotype) index.
            male_ztype_idx: Male ZType (genotype) index.
            value: Fitness value.
        """
        self.sexual_selection_fitness[female_ztype_idx, male_ztype_idx] = value

    def set_zygote_viability_fitness(self, sex: int, ztype_idx: int, value: float) -> None:
        """Set zygote fitness for a specific (sex, genotype) combination.

        Zygote fitness represents the probability that a zygote survives to become
        an individual, applied during reproduction stage before survival and
        competition.

        Args:
            sex: Sex index.
            ztype_idx: ZType (genotype) index.
            value: Fitness value (0.0 to 1.0).
        """
        self.zygote_viability_fitness[sex, ztype_idx] = value

    def compute_generation_time(self) -> float:
        """Compute the mean generation time from the current configuration.

        Uses the age‑based survival and mating rates to calculate the average
        age of reproduction.

        Returns:
            Mean generation time (float).
        """
        gen_times = np.zeros(self.n_sexes, dtype=np.float64)
        for sex in range(self.n_sexes):
            cumulative_survival = np.ones(self.n_ages, dtype=np.float64)
            for age in range(1, self.n_ages):
                cumulative_survival[age] = cumulative_survival[age - 1] * self.age_based_survival_rates[sex, age - 1]

            numerator = 0.0
            denominator = 0.0
            for age in range(self.n_ages):
                cumulative_mating_value = self.age_based_mating_rates[sex, age]
                if sex == Sex.FEMALE:
                    cumulative_mating_value *= self.female_age_based_fertility[age]
                if cumulative_mating_value > 0:
                    numerator += age * cumulative_survival[age] * cumulative_mating_value
                    denominator += cumulative_survival[age] * cumulative_mating_value

            if denominator > 0:
                gen_times[sex] = numerator / denominator

        return float(np.mean(gen_times))


PlainPopulationConfig = PopulationConfig


class DiscretePopulationConfig(NamedTuple):
    """Immutable configuration for discrete-generation simulations."""

    # -- Sampling --
    stochastic: bool
    continuous_sampling: bool

    # -- Dimensions --
    n_sexes: int                    # always 2
    n_ages: int                     # always 2
    n_ztypes: int
    n_gtypes: int
    n_glabs: int
    n_slabs: int

    # -- Age-structured arrays (kept for spatial builder compat; inactive in discrete)
    female_age_based_fertility: NDArray[np.float64]  # (2,)
    viability_fitness: NDArray[np.float64]              # (2, 2, g) — kept for compat with presets/fitness code
    fecundity_fitness: NDArray[np.float64]              # (2, g)
    zygote_viability_fitness: NDArray[np.float64]       # (2, g)
    sexual_selection_fitness: NDArray[np.float64]        # (g, g)

    # -- Competition --
    age_based_relative_competition_strength: NDArray[np.float64]  # (2,)

    # -- Reproduction scalars --
    eggs_per_female: NDArray[np.float64]          # 0-d, mutable
    fixed_egg_count: bool
    sex_ratio: NDArray[np.float64]                         # 0-d, mutable
    sperm_displacement_rate: NDArray[np.float64]           # 0-d, mutable

    # -- Per-demographic scalars (plain Python float, sole source of truth) --
    female_adult_mating_rate: float
    male_adult_mating_rate: float
    reproduction_rate: float
    female_age0_survival: float
    male_age0_survival: float
    female_fertility: float

    # -- Reproduction arrays --
    zygotes_to_gametes_map: NDArray[np.float64]         # (2, g, hl)
    gametes_to_zygotes_map: NDArray[np.float64]           # (hl, hl, g)
    offspring_tensor: NDArray[np.float64]                # (g, g, g)

    # -- Per-sex array views (pre-extracted from full arrays) --
    meiosis_f: NDArray[np.float64]                      # zygotes_to_gametes_map[0]
    meiosis_m: NDArray[np.float64]                      # zygotes_to_gametes_map[1]
    fecundity_f: NDArray[np.float64]                    # fecundity_fitness[0]
    fecundity_m: NDArray[np.float64]                    # fecundity_fitness[1]
    viability_f: NDArray[np.float64]                    # viability_fitness[0, 0, :]
    viability_m: NDArray[np.float64]                    # viability_fitness[1, 0, :]

    # -- Sex chromosomes --
    has_sex_chromosomes: bool
    female_ztype_compatibility: NDArray[np.float64]    # (g,)
    male_ztype_compatibility: NDArray[np.float64]      # (g,)
    female_only_by_sex_chrom: NDArray[np.bool_]           # (g,)
    male_only_by_sex_chrom: NDArray[np.bool_]             # (g,)

    # -- Competition scalars --
    juvenile_growth_mode: NDArray[np.int64]                # 0-d, mutable
    carrying_capacity: NDArray[np.float64]                 # 0-d, mutable
    expected_competition_strength: NDArray[np.float64]     # 0-d, mutable
    expected_survival_rate: NDArray[np.float64]            # 0-d, mutable
    low_density_growth_rate: NDArray[np.float64]           # 0-d, mutable
    generation_time: NDArray[np.float64]                   # 0-d, mutable

    # -- Age structure --
    new_adult_age: int
    adult_ages: NDArray[np.int64]                         # [1]

    # -- Init --
    initial_individual_count: NDArray[np.float64]         # (2, 2, g)
    initial_sperm_storage: NDArray[np.float64]            # (2, g, g) — empty for discrete
    hook_slot: int

    # -- Extreme speed (Wright-Fisher) --
    extreme_speed_mode: int            # 0=off, 1=multinomial, 2=poisson, 3=deterministic

    # -- custom fields --
    custom: NDArray[Any]  # placeholder float64; replaced by build_custom_array when registered

    def set_viability_fitness(
        self, sex: int, ztype_idx: int, value: float, age: int = -1
    ) -> None:
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, ztype_idx] = value

    def set_fecundity_fitness(
        self, sex: int, ztype_idx: int, value: float
    ) -> None:
        self.fecundity_fitness[sex, ztype_idx] = value

    def set_sexual_selection_fitness(
        self, female_ztype_idx: int, male_ztype_idx: int, value: float
    ) -> None:
        self.sexual_selection_fitness[female_ztype_idx, male_ztype_idx] = value

    def set_zygote_viability_fitness(
        self, sex: int, ztype_idx: int, value: float
    ) -> None:
        self.zygote_viability_fitness[sex, ztype_idx] = value
