"""Population configuration container and related utilities.

This module defines the immutable configuration structure ``PopulationConfig``,
functions to build, convert, and inspect configuration objects, as well as
helpers to initialise genotype/gamete mapping arrays.

The configuration is designed to be passed into simulation engine and remains
compatible with Numba.  Scalar fields are immutable (rebuild with ``_replace``),
while NumPy arrays can be mutated in place.
"""

from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

import natal.engine.simulation.age_structured as alg
from natal.genetic_entities import Genotype, HaploidGenotype
from natal.index_registry import compress_hg_glab, decompress_hg_glab
from natal.type_def import Sex

__all__ = [
    'NO_COMPETITION', 'FIXED', 'LOGISTIC', 'LINEAR', 'CONCAVE', 'BEVERTON_HOLT',
    'PopulationConfig',
    'build_population_config',
    'PlainPopulationConfig',
    'to_plain_population_config',
    'from_plain_population_config',
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
    'DiscretePopulationConfig', 'from_population_config',
]

# Growth mode constants (keep in sync with algorithms.py)
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3


class PopulationConfig(NamedTuple):
    """Primary immutable configuration container.

    Scalar fields are immutable (rebuild with ``_replace``). NumPy arrays are
    mutable in-place.

    Attributes:
        is_stochastic: Whether demographic events are stochastic.
        use_continuous_sampling: If True, use Dirichlet sampling for gamete
            proportions; otherwise use multinomial sampling.
        n_sexes: Number of sexes (usually 2).
        n_ages: Number of age classes.
        n_genotypes: Number of diploid genotype types.
        n_haploid_genotypes: Number of haploid genotype types.
        n_glabs: Number of gamete‑label variants per haplotype.
        age_based_mating_rates: Shape (n_sexes, n_ages) – mating rates per sex/age.
        age_based_reproduction_rates: Shape (n_ages,) – female reproduction
            participation rates per age.
        age_based_survival_rates: Shape (n_sexes, n_ages) – survival probabilities.
        female_age_based_relative_fertility: Shape (n_ages,) – relative fertility
            of females at each age.
        viability_fitness: Shape (n_sexes, n_ages, n_genotypes) – viability
            fitness coefficients.
        fecundity_fitness: Shape (n_sexes, n_genotypes) – fecundity fitness
            coefficients.
        sexual_selection_fitness: Shape (n_genotypes, n_genotypes) – sexual
            selection coefficients (female genotype × male genotype).
        zygote_viability_fitness: Shape (n_sexes, n_genotypes) – zygote fitness
            coefficients applied during reproduction stage before survival.
            Represents the probability that a zygote survives to become an
            individual, applied before competition and viability selection.
        age_based_relative_competition_strength: Shape (n_ages,) – relative
            contribution to competition for each age.
        sperm_displacement_rate: Probability that a new mating displaces stored
            sperm.
        expected_eggs_per_female: Expected number of eggs per female per tick.
        use_fixed_egg_count: If True, use the deterministic expected egg count;
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
        female_genotype_compatibility: Shape (n_genotypes,) – female-side
            compatibility weight per genotype.
        male_genotype_compatibility: Shape (n_genotypes,) – male-side
            compatibility weight per genotype.
        female_only_by_sex_chrom: Shape (n_genotypes,) – True where genotype
            is female-only under sex-chromosome constraints.
        male_only_by_sex_chrom: Shape (n_genotypes,) – True where genotype is
            male-only under sex-chromosome constraints.
        adult_ages: 1D array of age indices that are considered adult.
        genotype_to_gametes_map: Shape (n_sexes, n_genotypes, n_hg*n_glabs) –
            probability of producing each (haplotype, glab) combination.
        gametes_to_zygote_map: Shape (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) –
            probability of forming a given diploid genotype from two gametes.
        initial_individual_count: Shape (n_sexes, n_ages, n_genotypes) – initial
            population distribution.
        initial_sperm_storage: Shape (n_ages, n_genotypes, n_hg * n_glabs) – initial
            stored sperm counts.
    """

    # Scalars are immutable; rebuild this NamedTuple for scalar updates.
    is_stochastic: bool
    use_continuous_sampling: bool
    n_sexes: int
    n_ages: int
    n_genotypes: int
    n_haploid_genotypes: int
    n_glabs: int
    age_based_mating_rates: NDArray[np.float64]
    age_based_reproduction_rates: NDArray[np.float64]
    age_based_survival_rates: NDArray[np.float64]
    female_age_based_relative_fertility: NDArray[np.float64]
    viability_fitness: NDArray[np.float64]
    fecundity_fitness: NDArray[np.float64]
    sexual_selection_fitness: NDArray[np.float64]
    zygote_viability_fitness: NDArray[np.float64]
    age_based_relative_competition_strength: NDArray[np.float64]
    sperm_displacement_rate: NDArray[np.float64]           # 0-d, mutable
    expected_eggs_per_female: NDArray[np.float64]          # 0-d, mutable
    use_fixed_egg_count: bool
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
    female_genotype_compatibility: NDArray[np.float64]
    male_genotype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]
    # NumPy arrays are still mutable in-place.
    adult_ages: NDArray[np.int64]
    genotype_to_gametes_map: NDArray[np.float64]
    gametes_to_zygote_map: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]    # (g, g, g) — precomputed from meiosis × zygote maps
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]

    # -- custom fields (structured numpy scalar, set via Configurator.custom()) --
    custom: NDArray[Any]  # typed structured array when custom fields registered; float64 placeholder otherwise

    def set_viability_fitness(self, sex: int, genotype_idx: int, value: float, age: int = -1) -> None:
        """Set viability fitness for a specific (sex, genotype, age) combination.

        Args:
            sex: Sex index.
            genotype_idx: Diploid genotype index.
            value: Fitness value.
            age: Age class; if negative, defaults to new_adult_age - 1.
        """
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, genotype_idx] = value

    def set_fecundity_fitness(self, sex: int, genotype_idx: int, value: float) -> None:
        """Set fecundity fitness for a specific (sex, genotype).

        Args:
            sex: Sex index.
            genotype_idx: Diploid genotype index.
            value: Fitness value.
        """
        self.fecundity_fitness[sex, genotype_idx] = value

    def set_sexual_selection_fitness(self, female_geno_idx: int, male_geno_idx: int, value: float) -> None:
        """Set sexual selection fitness for a female‑male genotype pair.

        Args:
            female_geno_idx: Female genotype index.
            male_geno_idx: Male genotype index.
            value: Fitness value.
        """
        self.sexual_selection_fitness[female_geno_idx, male_geno_idx] = value

    def set_zygote_viability_fitness(self, sex: int, genotype_idx: int, value: float) -> None:
        """Set zygote fitness for a specific (sex, genotype) combination.

        Zygote fitness represents the probability that a zygote survives to become
        an individual, applied during reproduction stage before survival and
        competition.

        Args:
            sex: Sex index.
            genotype_idx: Diploid genotype index.
            value: Fitness value (0.0 to 1.0).
        """
        self.zygote_viability_fitness[sex, genotype_idx] = value

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
                    cumulative_mating_value *= self.female_age_based_relative_fertility[age]
                if cumulative_mating_value > 0:
                    numerator += age * cumulative_survival[age] * cumulative_mating_value
                    denominator += cumulative_survival[age] * cumulative_mating_value

            if denominator > 0:
                gen_times[sex] = numerator / denominator

        return float(np.mean(gen_times))


PlainPopulationConfig = PopulationConfig


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
        is_stochastic=bool(config.is_stochastic),
        use_continuous_sampling=bool(config.use_continuous_sampling),
        n_sexes=int(config.n_sexes),
        n_ages=int(config.n_ages),
        n_genotypes=int(config.n_genotypes),
        n_haploid_genotypes=int(config.n_haploid_genotypes),
        n_glabs=int(config.n_glabs),
        age_based_mating_rates=_maybe_copy_array(config.age_based_mating_rates, copy),
        age_based_reproduction_rates=_maybe_copy_array(config.age_based_reproduction_rates, copy),
        age_based_survival_rates=_maybe_copy_array(config.age_based_survival_rates, copy),
        female_age_based_relative_fertility=_maybe_copy_array(config.female_age_based_relative_fertility, copy),
        viability_fitness=_maybe_copy_array(config.viability_fitness, copy),
        fecundity_fitness=_maybe_copy_array(config.fecundity_fitness, copy),
        sexual_selection_fitness=_maybe_copy_array(config.sexual_selection_fitness, copy),
        zygote_viability_fitness=_maybe_copy_array(config.zygote_viability_fitness, copy),
        age_based_relative_competition_strength=_maybe_copy_array(config.age_based_relative_competition_strength, copy),
        sperm_displacement_rate=config.sperm_displacement_rate.copy() if copy else config.sperm_displacement_rate,
        expected_eggs_per_female=config.expected_eggs_per_female.copy() if copy else config.expected_eggs_per_female,
        use_fixed_egg_count=bool(config.use_fixed_egg_count),
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
        female_genotype_compatibility=_maybe_copy_array(config.female_genotype_compatibility, copy),
        male_genotype_compatibility=_maybe_copy_array(config.male_genotype_compatibility, copy),
        female_only_by_sex_chrom=_maybe_copy_array(config.female_only_by_sex_chrom, copy),
        male_only_by_sex_chrom=_maybe_copy_array(config.male_only_by_sex_chrom, copy),
        adult_ages=config.adult_ages.copy() if copy else config.adult_ages,
        genotype_to_gametes_map=_maybe_copy_array(config.genotype_to_gametes_map, copy),
        gametes_to_zygote_map=_maybe_copy_array(config.gametes_to_zygote_map, copy),
        offspring_tensor=_maybe_copy_array(config.offspring_tensor, copy),
        initial_individual_count=_maybe_copy_array(config.initial_individual_count, copy),
        initial_sperm_storage=_maybe_copy_array(config.initial_sperm_storage, copy),
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


def build_population_config(
    n_genotypes: int = 0,
    n_haploid_genotypes: int = 0,
    n_sexes: Optional[int] = None,
    n_ages: int = 2,
    n_glabs: int = 1,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
    age_based_mating_rates: Optional[NDArray[np.float64]] = None,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,
    age_based_survival_rates: Optional[NDArray[np.float64]] = None,
    female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,
    viability_fitness: Optional[NDArray[np.float64]] = None,
    fecundity_fitness: Optional[NDArray[np.float64]] = None,
    sexual_selection_fitness: Optional[NDArray[np.float64]] = None,
    zygote_viability_fitness: Optional[NDArray[np.float64]] = None,
    age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
    new_adult_age: int = 2,
    sperm_displacement_rate: float = 0.05,
    expected_eggs_per_female: float = 100.0,
    use_fixed_egg_count: bool = False,
    carrying_capacity: Optional[float] = None,
    sex_ratio: float = 0.5,
    low_density_growth_rate: float = 6.0,
    juvenile_growth_mode: int = LOGISTIC,
    generation_time: Optional[float] = None,
    hook_slot: int = 0,
    has_sex_chromosomes: bool = False,
    genotype_to_gametes_map: Optional[NDArray[np.float64]] = None,
    gametes_to_zygote_map: Optional[NDArray[np.float64]] = None,
    initial_individual_count: Optional[NDArray[np.float64]] = None,
    initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    age_1_carrying_capacity: Optional[float] = None,
    old_juvenile_carrying_capacity: Optional[float] = None,
    expected_num_adult_females: Optional[float] = None,
    infer_capacity_from_initial_state: bool = True,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]] = None,
    external_expected_eggs: Optional[float] = None,
) -> PopulationConfig:
    """Build an immutable PopulationConfig directly (legacy‑free path).

    This function constructs a complete configuration, filling missing arrays
    with sensible defaults and computing derived values such as equilibrium
    metrics and generation time.

    Args:
        n_genotypes: Number of diploid genotype types.
        n_haploid_genotypes: Number of haploid genotype types.
        n_sexes: Number of sexes (default 2).
        n_ages: Number of age classes (default 2).
        n_glabs: Number of gamete‑label variants per haplotype (default 1).
        is_stochastic: Whether to use stochastic demography.
        use_continuous_sampling: Use Dirichlet sampling for gamete proportions.
        age_based_mating_rates: Array (n_sexes, n_ages) – mating rates.
        age_based_reproduction_rates: Array (n_ages,) – female reproduction
            participation rates.
        age_based_survival_rates: Array (n_sexes, n_ages) – survival probabilities.
        female_age_based_relative_fertility: Array (n_ages,) – relative female
            fertility per age.
        viability_fitness: Array (n_sexes, n_ages, n_genotypes) – viability fitness.
        fecundity_fitness: Array (n_sexes, n_genotypes) – fecundity fitness.
        sexual_selection_fitness: Array (n_genotypes, n_genotypes) – sexual
            selection coefficients.
        age_based_relative_competition_strength: Array (n_ages,) – competition
            weight per age.
        new_adult_age: Age at which individuals become adults (default 2).
        sperm_displacement_rate: Probability of sperm displacement (default 0.05).
        expected_eggs_per_female: Expected number of eggs per female per tick.
        use_fixed_egg_count: If True, use deterministic egg count.
        carrying_capacity: Optional explicit carrying capacity (scaled later).
        sex_ratio: Proportion of newborns that are female.
        low_density_growth_rate: Intrinsic growth rate at low density.
        juvenile_growth_mode: Growth mode (see constants).
        generation_time: Optional pre‑computed generation time; if None, computed.
        hook_slot: Slot index for hooks (default 0).
        has_sex_chromosomes: Whether the species has sex‑chromosome constraints.
            If True, offspring sex is determined by genotype compatibility;
            if False, only sex_ratio is used (default False).
        genotype_to_gametes_map: Pre‑built mapping from genotype to gametes.
        gametes_to_zygote_map: Pre‑built mapping from gamete pair to zygote.
        initial_individual_count: Initial population counts (n_sexes, n_ages,
            n_genotypes). If None, filled with zeros.
        initial_sperm_storage: Initial sperm storage counts (n_ages, n_genotypes,
            n_genotypes). If None, filled with zeros.
        age_1_carrying_capacity: Population carrying capacity at age=1.
        old_juvenile_carrying_capacity: Alias for age_1_carrying_capacity (deprecated, use age_1_carrying_capacity).
        expected_num_adult_females: Expected number of adult females (unscaled).
        infer_capacity_from_initial_state: If True and carrying_capacity is None,
            compute base capacity from initial_individual_count.
        equilibrium_individual_distribution: Optional distribution used to compute
            equilibrium metrics.
        external_expected_eggs: Optional override for ``produced_age_0`` in the
            survival rate calculation. When provided, the expected survival rate is
            computed as ``total_age_1 / (external_expected_eggs * s_0_avg)`` instead
            of using the distribution-computed egg count.

    Returns:
        A fully populated PopulationConfig instance.

    Raises:
        AssertionError: If required dimensions are invalid or shape mismatches occur.
    """
    if n_sexes is None:
        n_sexes = 2

    assert n_genotypes > 0 and n_haploid_genotypes > 0 and n_glabs > 0, "invalid dimensions for PopulationConfig"
    assert n_ages > 0, "n_ages must be positive"

    n_hg_glabs = n_haploid_genotypes * n_glabs
    n_sexes_i = int(n_sexes)
    n_ages_i = int(n_ages)
    n_genotypes_i = int(n_genotypes)
    n_haploid_genotypes_i = int(n_haploid_genotypes)
    n_glabs_i = int(n_glabs)
    new_adult_age_i = int(new_adult_age)
    adult_ages = np.arange(new_adult_age_i, n_ages_i, dtype=np.int64)

    if initial_individual_count is not None:
        init_ind = initial_individual_count.copy()
    else:
        init_ind = np.zeros((n_sexes_i, n_ages_i, n_genotypes_i), dtype=np.float64)

    if initial_sperm_storage is not None:
        init_sperm = initial_sperm_storage.copy()
    else:
        init_sperm = np.zeros((n_ages_i, n_genotypes_i, n_hg_glabs), dtype=np.float64)

    # Resolve carrying_capacity directly (no base/scale separation).
    resolved_age_1 = age_1_carrying_capacity or old_juvenile_carrying_capacity
    if resolved_age_1 is not None:
        carrying_capacity_f = np.array(float(resolved_age_1))
    elif carrying_capacity is not None:
        carrying_capacity_f = np.array(float(carrying_capacity))
    elif infer_capacity_from_initial_state and initial_individual_count is not None:
        k_val = float(initial_individual_count[:, 1, :].sum())
        if k_val <= 0:
            k_val = 1000.0
        carrying_capacity_f = np.array(k_val)
    else:
        carrying_capacity_f = np.array(1000.0)

    def _validate_or_default_array(
        arr: Optional[NDArray[np.float64]],
        expected_shape: tuple[int, ...],
        name: str,
        default_value: Callable[[tuple[int, ...], type], NDArray[np.float64]] = np.ones,
        has_sex_dim: Optional[bool] = None,
        set_juvenile_values_to_zero: bool = False,
    ) -> NDArray[np.float64]:
        if arr is not None:
            assert arr.shape == expected_shape, f"invalid shape for {name}: expected {expected_shape}, got {arr.shape}"
            return arr
        arr2 = default_value(expected_shape, np.float64)
        if set_juvenile_values_to_zero:
            if has_sex_dim:
                arr2[:, :new_adult_age_i] = 0.0
            else:
                arr2[:new_adult_age_i] = 0.0
        return arr2

    mating = _validate_or_default_array(
        age_based_mating_rates,
        (n_sexes_i, n_ages_i),
        "age_based_mating_rates",
        has_sex_dim=True,
        set_juvenile_values_to_zero=True,
    )
    reproduction = _validate_or_default_array(
        age_based_reproduction_rates,
        (n_ages_i,),
        "age_based_reproduction_rates",
        has_sex_dim=False,
        set_juvenile_values_to_zero=True,
    )
    survival = _validate_or_default_array(
        age_based_survival_rates,
        (n_sexes_i, n_ages_i),
        "age_based_survival_rates",
        has_sex_dim=True,
        set_juvenile_values_to_zero=True,
    )
    female_fertility = _validate_or_default_array(
        female_age_based_relative_fertility,
        (n_ages_i,),
        "female_age_based_relative_fertility",
        has_sex_dim=False,
        set_juvenile_values_to_zero=True,
    )
    viability = _validate_or_default_array(viability_fitness, (n_sexes_i, n_ages_i, n_genotypes_i), "viability_fitness")
    fecundity = _validate_or_default_array(fecundity_fitness, (n_sexes_i, n_genotypes_i), "fecundity_fitness")
    sexual = _validate_or_default_array(sexual_selection_fitness, (n_genotypes_i, n_genotypes_i), "sexual_selection_fitness")
    zygote = _validate_or_default_array(zygote_viability_fitness, (n_sexes_i, n_genotypes_i), "zygote_viability_fitness")
    competition = _validate_or_default_array(
        age_based_relative_competition_strength,
        (n_ages_i,),
        "age_based_relative_competition_strength",
    )
    g2g = _validate_or_default_array(
        genotype_to_gametes_map,
        (n_sexes_i, n_genotypes_i, n_hg_glabs),
        "genotype_to_gametes_map",
        default_value=np.zeros,
    )
    g2z = _validate_or_default_array(
        gametes_to_zygote_map,
        (n_hg_glabs, n_hg_glabs, n_genotypes_i),
        "gametes_to_zygote_map",
        default_value=np.zeros,
    )
    female_genotype_compatibility = g2g[0].sum(axis=1)
    male_genotype_compatibility = g2g[1].sum(axis=1)
    female_only_by_sex_chrom = np.zeros(n_genotypes_i, dtype=np.bool_)
    male_only_by_sex_chrom = np.zeros(n_genotypes_i, dtype=np.bool_)
    if has_sex_chromosomes:
        for g_off in range(n_genotypes_i):
            f_ok = female_genotype_compatibility[g_off] > alg.EPS
            m_ok = male_genotype_compatibility[g_off] > alg.EPS
            female_only_by_sex_chrom[g_off] = f_ok and not m_ok
            male_only_by_sex_chrom[g_off] = m_ok and not f_ok

    expected_competition_strength, expected_survival_rate = alg.compute_equilibrium_metrics(
        carrying_capacity=float(carrying_capacity_f),
        expected_eggs_per_female=float(expected_eggs_per_female),
        age_based_survival_rates=survival,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        female_age_based_relative_fertility=female_fertility,
        relative_competition_strength=competition,
        sex_ratio=float(sex_ratio),
        new_adult_age=new_adult_age_i,
        n_ages=n_ages_i,
        equilibrium_individual_count=equilibrium_individual_distribution,
        external_expected_eggs=external_expected_eggs,
    )

    # Precompute offspring probability tensor from meiosis and zygote maps.
    offspring_tensor = alg.compute_offspring_probability_tensor(
        meiosis_f=g2g[0], meiosis_m=g2g[1],
        haplo_to_genotype_map=g2z,
        n_genotypes=n_genotypes_i,
        n_haplogenotypes=n_haploid_genotypes_i,
        n_glabs=n_glabs_i,
    )

    if generation_time is None:
        temp_cfg = PopulationConfig(
            is_stochastic=bool(is_stochastic),
            use_continuous_sampling=bool(use_continuous_sampling),
            n_sexes=n_sexes_i,
            n_ages=n_ages_i,
            n_genotypes=n_genotypes_i,
            n_haploid_genotypes=n_haploid_genotypes_i,
            n_glabs=n_glabs_i,
            age_based_mating_rates=mating,
            age_based_reproduction_rates=reproduction,
            age_based_survival_rates=survival,
            female_age_based_relative_fertility=female_fertility,
            viability_fitness=viability,
            fecundity_fitness=fecundity,
            sexual_selection_fitness=sexual,
            zygote_viability_fitness=zygote,
            age_based_relative_competition_strength=competition,
            sperm_displacement_rate=np.array(float(sperm_displacement_rate)),
            expected_eggs_per_female=np.array(float(expected_eggs_per_female)),
            use_fixed_egg_count=bool(use_fixed_egg_count),
            carrying_capacity=carrying_capacity_f,
            sex_ratio=np.array(float(sex_ratio)),
            low_density_growth_rate=np.array(float(low_density_growth_rate)),
            juvenile_growth_mode=np.array(int(juvenile_growth_mode), dtype=np.int64),
            expected_competition_strength=np.array(float(expected_competition_strength)),
            expected_survival_rate=np.array(float(expected_survival_rate)),
            generation_time=np.array(0.0),
            new_adult_age=new_adult_age_i,
            hook_slot=int(hook_slot),
            has_sex_chromosomes=bool(has_sex_chromosomes),
            female_genotype_compatibility=female_genotype_compatibility,
            male_genotype_compatibility=male_genotype_compatibility,
            female_only_by_sex_chrom=female_only_by_sex_chrom,
            male_only_by_sex_chrom=male_only_by_sex_chrom,
            adult_ages=adult_ages,
            genotype_to_gametes_map=g2g,
            gametes_to_zygote_map=g2z,
            offspring_tensor=offspring_tensor,
            initial_individual_count=init_ind,
            initial_sperm_storage=init_sperm,
            custom=np.zeros(0, dtype=np.float64),
        )
        generation_time_f = np.array(float(temp_cfg.compute_generation_time()))
    else:
        generation_time_f = np.array(float(generation_time))

    return PopulationConfig(
        is_stochastic=bool(is_stochastic),
        use_continuous_sampling=bool(use_continuous_sampling),
        n_sexes=n_sexes_i,
        n_ages=n_ages_i,
        n_genotypes=n_genotypes_i,
        n_haploid_genotypes=n_haploid_genotypes_i,
        n_glabs=n_glabs_i,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        age_based_survival_rates=survival,
        female_age_based_relative_fertility=female_fertility,
        viability_fitness=viability,
        fecundity_fitness=fecundity,
        sexual_selection_fitness=sexual,
        zygote_viability_fitness=zygote,
        age_based_relative_competition_strength=competition,
        sperm_displacement_rate=np.array(float(sperm_displacement_rate)),
        expected_eggs_per_female=np.array(float(expected_eggs_per_female)),
        use_fixed_egg_count=bool(use_fixed_egg_count),
        carrying_capacity=carrying_capacity_f,
        sex_ratio=np.array(float(sex_ratio)),
        low_density_growth_rate=np.array(float(low_density_growth_rate)),
        juvenile_growth_mode=np.array(int(juvenile_growth_mode), dtype=np.int64),
        expected_competition_strength=np.array(float(expected_competition_strength)),
        expected_survival_rate=np.array(float(expected_survival_rate)),
        generation_time=generation_time_f,
        new_adult_age=new_adult_age_i,
        hook_slot=int(hook_slot),
        has_sex_chromosomes=bool(has_sex_chromosomes),
        female_genotype_compatibility=female_genotype_compatibility,
        male_genotype_compatibility=male_genotype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        adult_ages=adult_ages,
        genotype_to_gametes_map=g2g,
        gametes_to_zygote_map=g2z,
        offspring_tensor=offspring_tensor,
        initial_individual_count=init_ind,
        initial_sperm_storage=init_sperm,
        custom=np.zeros(0, dtype=np.float64),
    )


# -------------------------------------------
# Helper functions for initializing maps
# -------------------------------------------
def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    zygote_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygote_map`` tensor.

    The function first populates a baseline mapping following Mendelian
    inheritance for all haplotype pairs and gamete-label combinations, and
    then applies optional zygote modifiers to transform the tensor.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygote_map`` tensor.

    Returns:
        Array of shape (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) representing
        the probability of each zygote genotype given a pair of gametes.

    Raises:
        ValueError: If any of the input lists is empty or n_glabs is not positive.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_hg_glabs = n_hg * n_glabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    # 1. Build baseline one-hot tensor according to Mendelian inheritance
    gametes_to_zygote_map: NDArray[np.float64] = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)

    for idx_hg1, hg1 in enumerate(haploid_genotypes):
        for idx_hg2, hg2 in enumerate(haploid_genotypes):
            zygote_gt = Genotype(
                species=hg1.species,
                maternal=hg1,
                paternal=hg2
            )

            if zygote_gt in diploid_genotypes:
                idx_gt = diploid_genotypes.index(zygote_gt)
                # Baseline: labels are equivalent — populate all (glab1, glab2)
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = compress_hg_glab(idx_hg1, glab1, n_glabs)
                        compressed_idx2 = compress_hg_glab(idx_hg2, glab2, n_glabs)
                        gametes_to_zygote_map[compressed_idx1, compressed_idx2, idx_gt] = 1.0

    # 2. Apply optional zygote modifiers
    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygote_map = modifier(gametes_to_zygote_map)
    return gametes_to_zygote_map


def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    gamete_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None
) -> NDArray[np.float64]:
    """Create and return a ``genotype_to_gametes_map`` tensor.

    This mirrors the style of :func:`initialize_zygote_map`: build a baseline
    mapping from each diploid genotype's gamete production and then apply
    optional modifier callables.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``genotype_to_gametes_map`` tensor.

    Returns:
        NDArray[np.float64]: Array shaped ``(n_sexes, n_genotypes, n_hg*n_glabs)``.

    Raises:
        ValueError: If any of the input lists is empty or n_glabs is not positive.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    # Infer number of sexes from Sex enum
    n_sexes = max(int(s.value) for s in Sex) + 1
    n_hg_glabs = n_hg * n_glabs

    genotype_to_gametes_map: NDArray[np.float64] = np.zeros((n_sexes, n_genotypes, n_hg_glabs), dtype=np.float64)
    haplo_to_idx = {hg: idx for idx, hg in enumerate(haploid_genotypes)}

    # Build optional sex-specific haploid availability constraints from species.
    # This keeps backward compatibility for autosome-only species (no filtering),
    # while making XY/ZW systems sex-aware by default.
    allowed_haplotypes_by_sex: dict[int, set[HaploidGenotype]] = {}
    if haploid_genotypes:
        species = haploid_genotypes[0].species
        try:
            female_allowed = set(species.get_maternal_haploid_genotypes())
            male_allowed = set(species.get_paternal_haploid_genotypes())
            if female_allowed:
                allowed_haplotypes_by_sex[int(Sex.FEMALE)] = female_allowed
            if male_allowed:
                allowed_haplotypes_by_sex[int(Sex.MALE)] = male_allowed
        except Exception:
            # If species does not provide parent-role iterators, fall back to
            # legacy behavior (same gamete distribution for all sexes).
            allowed_haplotypes_by_sex = {}

    # Populate baseline mapping using genotype.produce_gametes()
    for idx_genotype, genotype in enumerate(diploid_genotypes):
        base_gametes = genotype.produce_gametes()
        for sex_idx in range(n_sexes):
            allowed = allowed_haplotypes_by_sex.get(sex_idx)
            if allowed is None:
                filtered_gametes = base_gametes
            else:
                filtered_gametes = {
                    gamete: freq for gamete, freq in base_gametes.items() if gamete in allowed
                }

            total_freq = float(sum(filtered_gametes.values()))
            if total_freq <= 0.0:
                continue

            inv_total = 1.0 / total_freq
            for gamete, freq in filtered_gametes.items():
                idx_hg = haplo_to_idx.get(gamete)
                if idx_hg is None:
                    continue
                # By default, only map frequency for the default glab (0)
                compressed_idx = compress_hg_glab(idx_hg, 0, n_glabs)
                genotype_to_gametes_map[sex_idx, idx_genotype, compressed_idx] = float(freq) * inv_total

    # Apply optional modifier callables
    if gamete_modifiers:
        for modifier in gamete_modifiers:
            genotype_to_gametes_map = modifier(genotype_to_gametes_map)

    return genotype_to_gametes_map


def extract_gamete_frequencies(
    genotype_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[HaploidGenotype, float]:
    """Extract gamete frequencies for a specific (sex, genotype) pair.

    This convenience function converts a row of genotype_to_gametes_map
    from compressed haploid-glab indices back to HaploidGenotype objects with
    their aggregated frequencies across all glab variants.

    Args:
        genotype_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping HaploidGenotype -> aggregated frequency across all glabs.
        Only includes haplotype types with non-zero frequency.

    Examples:
        >>> config = population._config
        >>> hg_list = population._get_all_possible_haploid_genotypes()
        >>> freqs = extract_gamete_frequencies(
        ...     config.genotype_to_gametes_map,
        ...     sex_idx=0,
        ...     genotype_idx=5,
        ...     haploid_genotypes=hg_list,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # freqs = {haplotype_obj: 0.5, another_haplotype_obj: 0.5}
    """
    gamete_freqs_array = genotype_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[HaploidGenotype, float] = {}

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            hg_idx, _glab_idx = decompress_hg_glab(compressed_idx, n_glabs)
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                # Aggregate frequencies across all glab variants
                result[hg] = result.get(hg, 0.0) + freq

    return result


def extract_gamete_frequencies_by_glab(
    genotype_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[tuple[HaploidGenotype, int], float]:
    """Extract gamete frequencies at (HaploidGenotype, glab_idx) granularity.

    Unlike ``extract_gamete_frequencies`` which aggregates across all glab
    variants, this function preserves the glab dimension, returning separate
    entries for each (haplotype, glab) combination.

    Args:
        genotype_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping (HaploidGenotype, glab_idx) -> frequency.
        Only includes entries with non-zero frequency.

    Examples:
        >>> freqs = extract_gamete_frequencies_by_glab(
        ...     config.genotype_to_gametes_map, 0, 5, hg_list, n_glabs=2
        ... )
        >>> # freqs = {(hg_A, 0): 0.3, (hg_A, 1): 0.2, (hg_B, 0): 0.5}
    """
    gamete_freqs_array = genotype_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[tuple[HaploidGenotype, int], float] = {}

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:
            hg_idx, glab_idx = decompress_hg_glab(compressed_idx, n_glabs)
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                result[(hg, glab_idx)] = freq

    return result


def extract_zygote_frequencies(
    gametes_to_zygote_map: NDArray[np.float64],
    gamete1_compressed_idx: int,
    gamete2_compressed_idx: int,
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
) -> dict[Genotype, float]:
    """Extract zygote frequencies for a specific pair of gametes.

    This convenience function converts a slice of gametes_to_zygote_map
    from compressed gamete indices to Genotype objects with their frequencies.

    Args:
        gametes_to_zygote_map: The (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) array.
        gamete1_compressed_idx: Compressed index of first gamete (maternal).
        gamete2_compressed_idx: Compressed index of second gamete (paternal).
        diploid_genotypes: List of all Genotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping Genotype -> frequency. Only includes genotypes with
        non-zero frequency.

    Examples:
        >>> config = population._config
        >>> genotypes = list(population._genotypes)
        >>> zygote_freqs = extract_zygote_frequencies(
        ...     config.gametes_to_zygote_map,
        ...     gamete1_compressed_idx=0,
        ...     gamete2_compressed_idx=1,
        ...     diploid_genotypes=genotypes,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # zygote_freqs = {genotype1: 1.0 or {genotype2: 0.5, genotype3: 0.5}, etc}
    """
    zygote_freqs_array = gametes_to_zygote_map[gamete1_compressed_idx, gamete2_compressed_idx, :]
    result: dict[Genotype, float] = {}

    for genotype_idx, freq in enumerate(zygote_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            if genotype_idx < len(diploid_genotypes):
                genotype = diploid_genotypes[genotype_idx]
                result[genotype] = result.get(genotype, 0.0) + freq

    return result


# ── Discrete-generation variant ──────────────────────────────────────────────


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

    # -- Viability / fecundity / fitness --
    viability_fitness: NDArray[np.float64]              # (2, 2, g)
    fecundity_fitness: NDArray[np.float64]              # (2, g)
    zygote_viability_fitness: NDArray[np.float64]       # (2, g)
    sexual_selection_fitness: NDArray[np.float64]        # (g, g)

    # -- Competition --
    age_based_relative_competition_strength: NDArray[np.float64]  # (2,)

    # -- Reproduction scalars --
    expected_eggs_per_female: NDArray[np.float64]          # 0-d, mutable
    use_fixed_egg_count: bool
    sex_ratio: NDArray[np.float64]                         # 0-d, mutable
    sperm_displacement_rate: NDArray[np.float64]           # 0-d, mutable

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

    # -- Per-sex array views (pre-extracted from full arrays) --
    meiosis_f: NDArray[np.float64]                      # genotype_to_gametes_map[0]
    meiosis_m: NDArray[np.float64]                      # genotype_to_gametes_map[1]
    fecundity_f: NDArray[np.float64]                    # fecundity_fitness[0]
    fecundity_m: NDArray[np.float64]                    # fecundity_fitness[1]
    viability_f: NDArray[np.float64]                    # viability_fitness[0, 0, :]
    viability_m: NDArray[np.float64]                    # viability_fitness[1, 0, :]

    # -- Sex chromosomes --
    has_sex_chromosomes: bool
    female_genotype_compatibility: NDArray[np.float64]    # (g,)
    male_genotype_compatibility: NDArray[np.float64]      # (g,)
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

    # -- custom fields --
    custom: NDArray[np.void]

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
        male_mating_rate=cfg.age_based_mating_rates[1, 1],
        female_mating_rate=cfg.age_based_mating_rates[0, 1],
        reproduction_rate=cfg.age_based_reproduction_rates[1],
        base_survival_f=cfg.age_based_survival_rates[0, 0],
        base_survival_m=cfg.age_based_survival_rates[1, 0],
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
        expected_competition_strength=cfg.expected_competition_strength,
        expected_survival_rate=cfg.expected_survival_rate,
        low_density_growth_rate=cfg.low_density_growth_rate,
        generation_time=cfg.generation_time,
        new_adult_age=int(cfg.new_adult_age),
        adult_ages=cfg.adult_ages.copy(),
        initial_individual_count=cfg.initial_individual_count,
        initial_sperm_storage=cfg.initial_sperm_storage,
        hook_slot=int(cfg.hook_slot),
        custom=cfg.custom,
    )
