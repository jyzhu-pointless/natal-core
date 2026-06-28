"""Population configuration container and related utilities.

This module defines the immutable configuration structure ``PopulationConfig``,
functions to build, convert, and inspect configuration objects, as well as
helpers to initialise genotype/gamete mapping arrays.

The configuration is designed to be passed into simulation engine and remains
compatible with Numba.  Scalar fields are immutable (rebuild with ``_replace``),
while NumPy arrays can be mutated in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, List, NamedTuple, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from natal.genetic_entities import Genotype, HaploidGenotype
from natal.type_def import Sex

__all__ = [
    'NO_COMPETITION', 'FIXED', 'LOGISTIC', 'LINEAR', 'CONCAVE', 'BEVERTON_HOLT',
    'PopulationConfig',
    'extract_gamete_frequencies',
    'extract_gamete_frequencies_by_glab',
    'extract_zygote_frequencies',
    'DiscretePopulationConfig',
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
        stochastic: Whether demographic events are stochastic.
        continuous_sampling: If True, use Dirichlet sampling for gamete
            proportions; otherwise use multinomial sampling.
        n_sexes: Number of sexes (usually 2).
        n_ages: Number of age classes.
        n_ztypes: Number of zygote types (diploid genotype types after slab expansion).
        n_haploid_genotypes: Number of haploid genotype types.
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
        female_genotype_compatibility: Shape (n_ztypes,) – female-side
            compatibility weight per genotype.
        male_genotype_compatibility: Shape (n_ztypes,) – male-side
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
    n_haploid_genotypes: int
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
    female_genotype_compatibility: NDArray[np.float64]
    male_genotype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]
    # NumPy arrays are still mutable in-place.
    adult_ages: NDArray[np.int64]
    zygotes_to_gametes_map: NDArray[np.float64]
    gametes_to_zygotes_map: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]    # (g, g, g) — precomputed from meiosis × zygote maps
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]

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
        n_haploid_genotypes=int(config.n_haploid_genotypes),
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
        female_genotype_compatibility=_maybe_copy_array(config.female_genotype_compatibility, copy),
        male_genotype_compatibility=_maybe_copy_array(config.male_genotype_compatibility, copy),
        female_only_by_sex_chrom=_maybe_copy_array(config.female_only_by_sex_chrom, copy),
        male_only_by_sex_chrom=_maybe_copy_array(config.male_only_by_sex_chrom, copy),
        adult_ages=config.adult_ages.copy() if copy else config.adult_ages,
        zygotes_to_gametes_map=_maybe_copy_array(config.zygotes_to_gametes_map, copy),
        gametes_to_zygotes_map=_maybe_copy_array(config.gametes_to_zygotes_map, copy),
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


class _ComputedMaps(NamedTuple):
    """Intermediate result of shared config computation.

    Contains all arrays derived from raw inputs — before packaging into
    either ``PopulationConfig`` or ``DiscretePopulationConfig``.  Not part
    of the public API.
    """

    # -- Dimensions --
    n_sexes: int
    n_ages: int
    n_genotypes_orig: int   # G_orig (pre-expansion)
    n_haploid_genotypes: int
    n_glabs: int
    n_slabs: int
    n_ztypes: int           # engine-visible G = G_orig × n_slabs
    n_g_compressed: int     # after slab expansion (may differ from n_ztypes if compressed later)
    n_hg_effective: int
    n_glabs_effective: int
    new_adult_age: int
    adult_ages: NDArray[np.int64]

    # -- Demographic arrays --
    mating: NDArray[np.float64]          # (2, A)
    reproduction: NDArray[np.float64]    # (A,)
    survival: NDArray[np.float64]        # (2, A)
    female_fertility: NDArray[np.float64]  # (A,)

    # -- Fitness arrays --
    viability: NDArray[np.float64]       # (2, A, G×S)
    fecundity: NDArray[np.float64]       # (2, G×S)
    sexual: NDArray[np.float64]          # (G×S, G×S)
    zygote: NDArray[np.float64]          # (2, G×S)
    competition: NDArray[np.float64]     # (A,)

    # -- Expanded maps (pre-compression) --
    meiosis_f: NDArray[np.float64]       # (G×S, HL)
    meiosis_m: NDArray[np.float64]       # (G×S, HL)
    zygote_map: NDArray[np.float64]      # (HL, HL, G×S)

    # -- Compatibility --
    female_genotype_compatibility: NDArray[np.float64]
    male_genotype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]

    # -- Offspring tensor --
    offspring_tensor: NDArray[np.float64]

    # -- Initial state --
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]

    # -- Equilibrium & competition --
    carrying_capacity: NDArray[np.float64]          # 0-d
    expected_competition_strength: float
    expected_survival_rate: float
    eggs_per_female: float
    sex_ratio: float
    sperm_displacement_rate: float
    fixed_egg_count: bool
    low_density_growth_rate: float
    juvenile_growth_mode: int
    has_sex_chromosomes: bool


def _build_config_maps(
    n_genotypes: int,
    n_haploid_genotypes: int,
    n_sexes: int,
    n_ages: int,
    n_glabs: int,
    n_slabs: int,
    gamete_labels: Optional[list[str]],
    somatic_labels: Optional[list[str]],
    new_adult_age: int,
    stochastic: bool,
    continuous_sampling: bool,
    age_based_mating_rates: Optional[NDArray[np.float64]],
    age_based_reproduction_rates: Optional[NDArray[np.float64]],
    age_based_survival_rates: Optional[NDArray[np.float64]],
    female_age_based_fertility: Optional[NDArray[np.float64]],
    viability_fitness: Optional[NDArray[np.float64]],
    fecundity_fitness: Optional[NDArray[np.float64]],
    sexual_selection_fitness: Optional[NDArray[np.float64]],
    zygote_viability_fitness: Optional[NDArray[np.float64]],
    age_based_relative_competition_strength: Optional[NDArray[np.float64]],
    sperm_displacement_rate: float,
    eggs_per_female: float,
    fixed_egg_count: bool,
    carrying_capacity: Optional[float],
    sex_ratio: float,
    low_density_growth_rate: float,
    juvenile_growth_mode: int,
    has_sex_chromosomes: bool,
    zygotes_to_gametes_map: Optional[NDArray[np.float64]],
    gametes_to_zygotes_map: Optional[NDArray[np.float64]],
    initial_individual_count: Optional[NDArray[np.float64]],
    initial_sperm_storage: Optional[NDArray[np.float64]],
    age_1_carrying_capacity: Optional[float],
    old_juvenile_carrying_capacity: Optional[float],
    infer_capacity_from_initial_state: bool,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]],
    external_expected_eggs: Optional[float],
) -> _ComputedMaps:
    """Shared computation engine for config building.

    Validates inputs, fills defaults, expands slabs, and computes the
    offspring probability tensor.  Returns all derived arrays in a single
    structure that both ``build_population_config`` and
    ``build_discrete_engine_config`` consume independently.

    Not part of the public API.
    """
    import natal.engine.simulation.age_structured as alg

    assert n_genotypes > 0 and n_haploid_genotypes > 0 and n_glabs > 0, "invalid dimensions"
    assert n_ages > 0, "n_ages must be positive"

    n_hg_glabs = n_haploid_genotypes * n_glabs
    n_sexes_i = int(n_sexes)
    n_ages_i = int(n_ages)
    n_genotypes_i = int(n_genotypes)
    n_haploid_genotypes_i = int(n_haploid_genotypes)
    n_glabs_i = int(n_glabs)
    n_slabs_i = int(n_slabs)
    n_ztypes_i = n_genotypes_i * n_slabs_i
    new_adult_age_i = int(new_adult_age)
    adult_ages = np.arange(new_adult_age_i, n_ages_i, dtype=np.int64)

    if initial_individual_count is not None:
        init_ind = initial_individual_count.copy()
    else:
        init_ind = np.zeros((n_sexes_i, n_ages_i, n_ztypes_i), dtype=np.float64)

    if initial_sperm_storage is not None:
        init_sperm = initial_sperm_storage.copy()
    else:
        init_sperm = np.zeros((n_ages_i, n_ztypes_i, n_ztypes_i), dtype=np.float64)

    # Resolve carrying_capacity.
    if age_1_carrying_capacity is not None:
        resolved_age_1 = age_1_carrying_capacity
    elif old_juvenile_carrying_capacity is not None:
        resolved_age_1 = old_juvenile_carrying_capacity
    else:
        resolved_age_1 = None
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
        age_based_mating_rates, (n_sexes_i, n_ages_i), "age_based_mating_rates",
        has_sex_dim=True, set_juvenile_values_to_zero=True,
    )
    reproduction = _validate_or_default_array(
        age_based_reproduction_rates, (n_ages_i,), "age_based_reproduction_rates",
        has_sex_dim=False, set_juvenile_values_to_zero=True,
    )
    survival = _validate_or_default_array(
        age_based_survival_rates, (n_sexes_i, n_ages_i), "age_based_survival_rates",
        has_sex_dim=True, set_juvenile_values_to_zero=True,
    )
    female_fertility = _validate_or_default_array(
        female_age_based_fertility, (n_ages_i,), "female_age_based_fertility",
        has_sex_dim=False, set_juvenile_values_to_zero=True,
    )
    viability = _validate_or_default_array(viability_fitness, (n_sexes_i, n_ages_i, n_ztypes_i), "viability_fitness")
    fecundity = _validate_or_default_array(fecundity_fitness, (n_sexes_i, n_ztypes_i), "fecundity_fitness")
    sexual = _validate_or_default_array(sexual_selection_fitness, (n_ztypes_i, n_ztypes_i), "sexual_selection_fitness")
    zygote = _validate_or_default_array(zygote_viability_fitness, (n_sexes_i, n_ztypes_i), "zygote_viability_fitness")
    competition = _validate_or_default_array(
        age_based_relative_competition_strength, (n_ages_i,), "age_based_relative_competition_strength",
    )
    z2g = _validate_or_default_array(
        zygotes_to_gametes_map, (n_sexes_i, n_genotypes_i, n_hg_glabs), "zygotes_to_gametes_map",
        default_value=np.zeros,
    )
    g2z = _validate_or_default_array(
        gametes_to_zygotes_map, (n_hg_glabs, n_hg_glabs, n_genotypes_i), "gametes_to_zygotes_map",
        default_value=np.zeros,
    )
    expected_competition_strength, expected_survival_rate = alg.compute_equilibrium_metrics(
        carrying_capacity=float(carrying_capacity_f),
        eggs_per_female=float(eggs_per_female),
        age_based_survival_rates=survival,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        female_age_based_fertility=female_fertility,
        relative_competition_strength=competition,
        sex_ratio=float(sex_ratio),
        new_adult_age=new_adult_age_i,
        n_ages=n_ages_i,
        equilibrium_individual_count=equilibrium_individual_distribution,
        external_expected_eggs=external_expected_eggs,
    )

    # Index compression mask placeholders (compression is applied externally).
    n_g_compressed = n_genotypes_i
    n_hg_effective = n_haploid_genotypes_i
    n_glabs_effective = n_glabs_i

    # Slab expansion (skip if maps are already expanded by the caller).
    if z2g.shape[1] == n_genotypes_i:
        z2g_expanded, _z2g, n_g_compressed = _expand_slab_maps(
            z2g=z2g, g2z=g2z,
            n_slabs=n_slabs_i, n_genotypes=n_genotypes_i,
            gamete_labels=gamete_labels, somatic_labels=somatic_labels,
            n_haploid_genotypes=n_haploid_genotypes_i, n_glabs=n_glabs_i,
        )
    else:
        # Maps already slab-expanded — use as-is.
        z2g_expanded = z2g
        _z2g = g2z
        n_g_compressed = z2g.shape[1]
    _m_f = z2g_expanded[0]
    _m_m = z2g_expanded[1]

    # Genotype compatibility (computed from expanded maps).
    female_genotype_compatibility = _m_f.sum(axis=1)
    male_genotype_compatibility = _m_m.sum(axis=1)
    female_only_by_sex_chrom = np.zeros(n_g_compressed, dtype=np.bool_)
    male_only_by_sex_chrom = np.zeros(n_g_compressed, dtype=np.bool_)
    if has_sex_chromosomes:
        _ztype_index: dict[tuple[int, int], int] = {
            (g, s): g * n_slabs_i + s
            for g in range(n_genotypes_i) for s in range(n_slabs_i)
        }
        for g_off in range(n_genotypes_i):
            f_ok = female_genotype_compatibility[g_off] > alg.EPS
            m_ok = male_genotype_compatibility[g_off] > alg.EPS
            if n_slabs_i > 1:
                for s in range(n_slabs_i):
                    z = _ztype_index[(g_off, s)]
                    female_only_by_sex_chrom[z] = f_ok and not m_ok
                    male_only_by_sex_chrom[z] = m_ok and not f_ok
            else:
                female_only_by_sex_chrom[g_off] = f_ok and not m_ok
                male_only_by_sex_chrom[g_off] = m_ok and not f_ok

    # Offspring probability tensor.
    offspring_tensor = alg.compute_offspring_probability_tensor(
        meiosis_f=_m_f, meiosis_m=_m_m,
        haplo_to_genotype_map=_z2g,
        n_ztypes=n_g_compressed,
        n_haplogenotypes=n_hg_effective,
        n_glabs=n_glabs_effective,
    )

    return _ComputedMaps(
        n_sexes=n_sexes_i,
        n_ages=n_ages_i,
        n_genotypes_orig=n_genotypes_i,
        n_haploid_genotypes=n_haploid_genotypes_i,
        n_glabs=n_glabs_i,
        n_slabs=n_slabs_i,
        n_ztypes=n_ztypes_i,
        n_g_compressed=n_g_compressed,
        n_hg_effective=n_hg_effective,
        n_glabs_effective=n_glabs_effective,
        new_adult_age=new_adult_age_i,
        adult_ages=adult_ages,
        mating=mating,
        reproduction=reproduction,
        survival=survival,
        female_fertility=female_fertility,
        viability=viability,
        fecundity=fecundity,
        sexual=sexual,
        zygote=zygote,
        competition=competition,
        meiosis_f=_m_f,
        meiosis_m=_m_m,
        zygote_map=_z2g,
        female_genotype_compatibility=female_genotype_compatibility,
        male_genotype_compatibility=male_genotype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        offspring_tensor=offspring_tensor,
        initial_individual_count=init_ind,
        initial_sperm_storage=init_sperm,
        carrying_capacity=carrying_capacity_f,
        expected_competition_strength=expected_competition_strength,
        expected_survival_rate=expected_survival_rate,
        eggs_per_female=float(eggs_per_female),
        sex_ratio=float(sex_ratio),
        sperm_displacement_rate=float(sperm_displacement_rate),
        fixed_egg_count=bool(fixed_egg_count),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=int(juvenile_growth_mode),
        has_sex_chromosomes=bool(has_sex_chromosomes),
    )


def build_population_config(
    n_genotypes: int = 0,
    n_haploid_genotypes: int = 0,
    n_sexes: Optional[int] = None,
    n_ages: int = 2,
    n_glabs: int = 1,
    n_slabs: int = 1,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    stochastic: bool = True,
    continuous_sampling: bool = False,
    age_based_mating_rates: Optional[NDArray[np.float64]] = None,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,
    age_based_survival_rates: Optional[NDArray[np.float64]] = None,
    female_age_based_fertility: Optional[NDArray[np.float64]] = None,
    viability_fitness: Optional[NDArray[np.float64]] = None,
    fecundity_fitness: Optional[NDArray[np.float64]] = None,
    sexual_selection_fitness: Optional[NDArray[np.float64]] = None,
    zygote_viability_fitness: Optional[NDArray[np.float64]] = None,
    age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
    new_adult_age: int = 2,
    sperm_displacement_rate: float = 0.05,
    eggs_per_female: float = 100.0,
    fixed_egg_count: bool = False,
    carrying_capacity: Optional[float] = None,
    sex_ratio: float = 0.5,
    low_density_growth_rate: float = 6.0,
    juvenile_growth_mode: int = LOGISTIC,
    generation_time: Optional[float] = None,
    hook_slot: int = 0,
    has_sex_chromosomes: bool = False,
    zygotes_to_gametes_map: Optional[NDArray[np.float64]] = None,
    gametes_to_zygotes_map: Optional[NDArray[np.float64]] = None,
    initial_individual_count: Optional[NDArray[np.float64]] = None,
    initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    age_1_carrying_capacity: Optional[float] = None,
    old_juvenile_carrying_capacity: Optional[float] = None,
    infer_capacity_from_initial_state: bool = True,
    equilibrium_individual_distribution: Optional[NDArray[np.float64]] = None,
    external_expected_eggs: Optional[float] = None,
) -> PopulationConfig:
    """Build an immutable PopulationConfig directly (legacy‑free path).

    This function constructs a complete configuration, filling missing arrays
    with sensible defaults and computing derived values such as equilibrium
    metrics and generation time.

    Args:
        n_genotypes: Number of diploid genotype types BEFORE slab expansion
            (G_orig).  The engine-visible axis size is ``n_ztypes = n_genotypes *
            n_slabs``, so fitness and initial-state arrays must use the expanded
            shape.
        n_haploid_genotypes: Number of haploid genotype types.
        n_sexes: Number of sexes (default 2).
        n_ages: Number of age classes (default 2).
        n_glabs: Number of gamete‑label variants per haplotype (default 1).
        stochastic: Whether to use stochastic demography.
        continuous_sampling: Use Dirichlet sampling for gamete proportions.
        age_based_mating_rates: Array (n_sexes, n_ages) – mating rates.
        age_based_reproduction_rates: Array (n_ages,) – female reproduction
            participation rates.
        age_based_survival_rates: Array (n_sexes, n_ages) – survival probabilities.
        female_age_based_fertility: Array (n_ages,) – relative female
            fertility per age.
        viability_fitness: Array (n_sexes, n_ages, n_ztypes) – viability fitness.
        fecundity_fitness: Array (n_sexes, n_ztypes) – fecundity fitness.
        sexual_selection_fitness: Array (n_ztypes, n_ztypes) – sexual
            selection coefficients.
        age_based_relative_competition_strength: Array (n_ages,) – competition
            weight per age.
        new_adult_age: Age at which individuals become adults (default 2).
        sperm_displacement_rate: Probability of sperm displacement (default 0.05).
        eggs_per_female: Expected number of eggs per female per tick.
        fixed_egg_count: If True, use deterministic egg count.
        carrying_capacity: Optional explicit carrying capacity (scaled later).
        sex_ratio: Proportion of newborns that are female.
        low_density_growth_rate: Intrinsic growth rate at low density.
        juvenile_growth_mode: Growth mode (see constants).
        generation_time: Optional pre‑computed generation time; if None, computed.
        hook_slot: Slot index for hooks (default 0).
        has_sex_chromosomes: Whether the species has sex‑chromosome constraints.
            If True, offspring sex is determined by genotype compatibility;
            if False, only sex_ratio is used (default False).
        zygotes_to_gametes_map: Pre‑built mapping from genotype to gametes.
        gametes_to_zygotes_map: Pre‑built mapping from gamete pair to zygote.
        initial_individual_count: Initial population counts (n_sexes, n_ages,
            n_ztypes). If None, filled with zeros.
        initial_sperm_storage: Initial sperm storage counts (n_ages, n_ztypes,
            n_ztypes). If None, filled with zeros.
        age_1_carrying_capacity: Population carrying capacity at age=1.
        old_juvenile_carrying_capacity: Alias for age_1_carrying_capacity (deprecated, use age_1_carrying_capacity).
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
    m = _build_config_maps(
        n_genotypes=n_genotypes,
        n_haploid_genotypes=n_haploid_genotypes,
        n_sexes=2 if n_sexes is None else int(n_sexes),
        n_ages=int(n_ages),
        n_glabs=int(n_glabs),
        n_slabs=int(n_slabs),
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=int(new_adult_age),
        stochastic=bool(stochastic),
        continuous_sampling=bool(continuous_sampling),
        age_based_mating_rates=age_based_mating_rates,
        age_based_reproduction_rates=age_based_reproduction_rates,
        age_based_survival_rates=age_based_survival_rates,
        female_age_based_fertility=female_age_based_fertility,
        viability_fitness=viability_fitness,
        fecundity_fitness=fecundity_fitness,
        sexual_selection_fitness=sexual_selection_fitness,
        zygote_viability_fitness=zygote_viability_fitness,
        age_based_relative_competition_strength=age_based_relative_competition_strength,
        sperm_displacement_rate=float(sperm_displacement_rate),
        eggs_per_female=float(eggs_per_female),
        fixed_egg_count=bool(fixed_egg_count),
        carrying_capacity=carrying_capacity,
        sex_ratio=float(sex_ratio),
        low_density_growth_rate=float(low_density_growth_rate),
        juvenile_growth_mode=int(juvenile_growth_mode),
        has_sex_chromosomes=bool(has_sex_chromosomes),
        zygotes_to_gametes_map=zygotes_to_gametes_map,
        gametes_to_zygotes_map=gametes_to_zygotes_map,
        initial_individual_count=initial_individual_count,
        initial_sperm_storage=initial_sperm_storage,
        age_1_carrying_capacity=age_1_carrying_capacity,
        old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
        infer_capacity_from_initial_state=infer_capacity_from_initial_state,
        equilibrium_individual_distribution=equilibrium_individual_distribution,
        external_expected_eggs=external_expected_eggs,
    )

    if generation_time is None:
        temp_cfg = PopulationConfig(
            stochastic=bool(stochastic),
            continuous_sampling=bool(continuous_sampling),
            n_sexes=m.n_sexes,
            n_ages=m.n_ages,
            n_ztypes=m.n_ztypes,
            n_haploid_genotypes=m.n_haploid_genotypes,
            n_glabs=m.n_glabs,
            n_slabs=m.n_slabs,
            age_based_mating_rates=m.mating,
            age_based_reproduction_rates=m.reproduction,
            age_based_survival_rates=m.survival,
            female_age_based_fertility=m.female_fertility,
            viability_fitness=m.viability,
            fecundity_fitness=m.fecundity,
            sexual_selection_fitness=m.sexual,
            zygote_viability_fitness=m.zygote,
            age_based_relative_competition_strength=m.competition,
            sperm_displacement_rate=np.array(m.sperm_displacement_rate),
            eggs_per_female=np.array(m.eggs_per_female),
            fixed_egg_count=m.fixed_egg_count,
            carrying_capacity=m.carrying_capacity,
            sex_ratio=np.array(m.sex_ratio),
            low_density_growth_rate=np.array(m.low_density_growth_rate),
            juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
            expected_competition_strength=np.array(m.expected_competition_strength),
            expected_survival_rate=np.array(m.expected_survival_rate),
            generation_time=np.array(0.0),
            new_adult_age=m.new_adult_age,
            hook_slot=int(hook_slot),
            has_sex_chromosomes=m.has_sex_chromosomes,
            female_genotype_compatibility=m.female_genotype_compatibility,
            male_genotype_compatibility=m.male_genotype_compatibility,
            female_only_by_sex_chrom=m.female_only_by_sex_chrom,
            male_only_by_sex_chrom=m.male_only_by_sex_chrom,
            adult_ages=m.adult_ages,
            zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
            gametes_to_zygotes_map=m.zygote_map,
            offspring_tensor=m.offspring_tensor,
            initial_individual_count=m.initial_individual_count,
            initial_sperm_storage=m.initial_sperm_storage,
            custom=np.zeros(0, dtype=np.float64),
        )
        generation_time_f = np.array(float(temp_cfg.compute_generation_time()))
    else:
        generation_time_f = np.array(float(generation_time))

    return PopulationConfig(
        stochastic=bool(stochastic),
        continuous_sampling=bool(continuous_sampling),
        n_sexes=m.n_sexes,
        n_ages=m.n_ages,
        n_ztypes=m.n_ztypes,
        n_haploid_genotypes=m.n_haploid_genotypes,
        n_glabs=m.n_glabs,
        n_slabs=m.n_slabs,
        age_based_mating_rates=m.mating,
        age_based_reproduction_rates=m.reproduction,
        age_based_survival_rates=m.survival,
        female_age_based_fertility=m.female_fertility,
        viability_fitness=m.viability,
        fecundity_fitness=m.fecundity,
        sexual_selection_fitness=m.sexual,
        zygote_viability_fitness=m.zygote,
        age_based_relative_competition_strength=m.competition,
        sperm_displacement_rate=np.array(m.sperm_displacement_rate),
        eggs_per_female=np.array(m.eggs_per_female),
        fixed_egg_count=m.fixed_egg_count,
        carrying_capacity=m.carrying_capacity,
        sex_ratio=np.array(m.sex_ratio),
        low_density_growth_rate=np.array(m.low_density_growth_rate),
        juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
        expected_competition_strength=np.array(m.expected_competition_strength),
        expected_survival_rate=np.array(m.expected_survival_rate),
        generation_time=generation_time_f,
        new_adult_age=m.new_adult_age,
        hook_slot=int(hook_slot),
        has_sex_chromosomes=m.has_sex_chromosomes,
        female_genotype_compatibility=m.female_genotype_compatibility,
        male_genotype_compatibility=m.male_genotype_compatibility,
        female_only_by_sex_chrom=m.female_only_by_sex_chrom,
        male_only_by_sex_chrom=m.male_only_by_sex_chrom,
        adult_ages=m.adult_ages,
        zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
        gametes_to_zygotes_map=m.zygote_map,
        offspring_tensor=m.offspring_tensor,
        initial_individual_count=m.initial_individual_count,
        initial_sperm_storage=m.initial_sperm_storage,
        custom=np.zeros(0, dtype=np.float64),
    )


# -------------------------------------------
# Helper functions for initializing maps
# -------------------------------------------
def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    zygote_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
        unordered: bool = False,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygotes_map`` tensor.

    The function first populates a baseline mapping following Mendelian
    inheritance for all haplotype pairs and gamete-label combinations, and
    then applies optional zygote modifiers to transform the tensor.

    When *unordered* is True, uses ``unordered_genotype()`` so that
    ``(hg_a, hg_b)`` and ``(hg_b, hg_a)`` map to the same unordered
    genotype index, collapsing symmetric pairs.  Default ``False``
    preserves maternal/paternal ordering.

    When *n_slabs* > 1 the genotype axis is expanded so that each base
    genotype has *n_slabs* slab variants.  Zygote modifiers are applied
    BEFORE expansion — they operate in the unexpanded G_orig space.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygotes_map`` tensor.
        unordered: If True, use unordered genotype canonicalization.
        n_slabs: Number of somatic slabs (≥ 1).  When > 1 each genotype is
            replicated across slabs.

    Returns:
        Array of shape (HL, HL, G_orig * n_slabs) representing the
        probability of each zygote genotype given a pair of gametes.

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
    gametes_to_zygotes_map: NDArray[np.float64] = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=np.float64)

    # Local dict-based lookup replacing formula: compressed = hg_idx * n_glabs + glab_idx
    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
    }

    for idx_hg1, hg1 in enumerate(haploid_genotypes):
        for idx_hg2, hg2 in enumerate(haploid_genotypes):
            if unordered:
                zygote_gt = hg1.species.unordered_genotype(hg1, hg2)
            else:
                zygote_gt = Genotype(
                    species=hg1.species,
                    maternal=hg1,
                    paternal=hg2,
                )

            if zygote_gt in diploid_genotypes:
                idx_gt = diploid_genotypes.index(zygote_gt)
                # Baseline: labels are equivalent — populate all (glab1, glab2)
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = _gtype_index[(idx_hg1, glab1)]
                        compressed_idx2 = _gtype_index[(idx_hg2, glab2)]
                        gametes_to_zygotes_map[compressed_idx1, compressed_idx2, idx_gt] = 1.0

    # 2. Apply optional zygote modifiers (before slab expansion).
    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygotes_map = modifier(gametes_to_zygotes_map)

    # 3. Slab expansion: expand from G_orig → G_orig * n_slabs.
    if n_slabs > 1:
        n_ztypes = n_genotypes * n_slabs
        expanded = np.zeros((n_hg_glabs, n_hg_glabs, n_ztypes), dtype=np.float64)
        _ztype_index: dict[tuple[int, int], int] = {
            (g, s): g * n_slabs + s
            for g in range(n_genotypes) for s in range(n_slabs)
        }
        for g_raw in range(n_genotypes):
            expanded[:, :, _ztype_index[(g_raw, 0)]] = gametes_to_zygotes_map[:, :, g_raw]
        gametes_to_zygotes_map = expanded

    return gametes_to_zygotes_map


def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    gamete_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
    n_slabs: int = 1,
) -> NDArray[np.float64]:
    """Create and return a ``zygotes_to_gametes_map`` tensor.

    This mirrors the style of :func:`initialize_zygote_map`: build a baseline
    mapping from each diploid genotype's gamete production and then apply
    optional modifier callables.

    When *n_slabs* > 1 the genotype axis is tiled so that each base genotype
    is replicated *n_slabs* times (one per somatic label).  Modifier callables
    are applied BEFORE tiling — they operate in the unexpanded G_orig space.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects.
        n_glabs: Number of gamete labels (default: 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``zygotes_to_gametes_map`` tensor.
        n_slabs: Number of somatic slabs (≥ 1).  When > 1 each genotype is
            replicated across slabs with identical gamete production.

    Returns:
        NDArray[np.float64]: Array shaped ``(n_sexes, G_orig * n_slabs, n_hg*n_glabs)``.

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

    zygotes_to_gametes_map: NDArray[np.float64] = np.zeros((n_sexes, n_genotypes, n_hg_glabs), dtype=np.float64)
    haplo_to_idx = {hg: idx for idx, hg in enumerate(haploid_genotypes)}

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
    }

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
                compressed_idx = _gtype_index[(idx_hg, 0)]
                zygotes_to_gametes_map[sex_idx, idx_genotype, compressed_idx] = float(freq) * inv_total

    # Apply optional modifier callables (before slab expansion).
    if gamete_modifiers:
        for modifier in gamete_modifiers:
            zygotes_to_gametes_map = modifier(zygotes_to_gametes_map)

    # Slab expansion: tile each genotype row S times.
    if n_slabs > 1:
        zygotes_to_gametes_map = np.repeat(zygotes_to_gametes_map, n_slabs, axis=1)

    return zygotes_to_gametes_map


def _expand_slab_maps(
    z2g: NDArray[np.float64],
    g2z: NDArray[np.float64],
    n_slabs: int,
    n_genotypes: int,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    n_haploid_genotypes: int = 0,
    n_glabs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Expand genotype-to-gamete and gametes-to-zygote maps for n_slabs > 1.

    When ``n_slabs <= 1``, returns the originals unchanged (identity).

    When ``n_slabs > 1``, the genotype axis of both maps is expanded from
    ``G_orig`` to ``G_orig * n_slabs`` via tiling (meiosis maps) and slab-0
    default expansion (zygote map).  Cytoplasmic gamete tagging (Wolbachia)
    and zygote redirect are applied if the relevant label sets are provided.

    Args:
        z2g: Genotype-to-gamete map, shape ``(2, G, HL)``.
        g2z: Gametes-to-zygote map, shape ``(HL, HL, G)``.
        n_slabs: Number of somatic labels (>= 1).
        n_genotypes: Original G_orig (pre-expansion).
        gamete_labels: Ordered gamete label names (for Wolbachia tagging).
        somatic_labels: Ordered somatic label names (for Wolbachia tagging).
        n_haploid_genotypes: Number of haploid genotype types.
        n_glabs: Number of gamete label types.

    Returns:
        ``(z2g_expanded, g2z_expanded, n_ztypes)`` where ``z2g_expanded`` has
        shape ``(2, G_orig * n_slabs, HL)`` and ``g2z_expanded`` has shape
        ``(HL, HL, G_orig * n_slabs)``.  ``n_ztypes == G_orig * n_slabs``.
    """
    if n_slabs <= 1:
        return z2g, g2z, n_genotypes

    n_ztypes = n_genotypes * n_slabs

    _ztype_index: dict[tuple[int, int], int] = {
        (g, s): g * n_slabs + s
        for g in range(n_genotypes) for s in range(n_slabs)
    }
    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_haploid_genotypes) for gi in range(n_glabs)
    }

    # Meiosis: tile each genotype row S times (identity — slab does not
    # affect gamete production in the baseline).
    _m_f = np.repeat(z2g[0].copy(), n_slabs, axis=0)
    _m_m = np.repeat(z2g[1].copy(), n_slabs, axis=0)

    # Cytoplasmic gamete tagging: non-default glabs tag maternal gametes
    # from the corresponding non-default slab.  For example, glab "wolbachia"
    # tags gametes from "infected" mothers, enabling maternal inheritance.
    # The mapping is convention-based: glab[i] ↔ slab[i] for i >= 1.
    # We iterate matching index pairs — glab 1 ↔ slab 1, glab 2 ↔ slab 2,
    # etc. — NOT the Cartesian product, because each glab belongs to exactly
    # one slab.
    if gamete_labels and somatic_labels and n_glabs > 1 and n_slabs > 1:
        for idx in range(1, min(n_slabs, n_glabs)):
            if idx >= len(gamete_labels) or idx >= len(somatic_labels):
                continue
            for g_raw in range(n_genotypes):
                z_target = _ztype_index[(g_raw, idx)]
                for hg_idx in range(n_haploid_genotypes):
                    src = _gtype_index[(hg_idx, 0)]   # default glab
                    dst = _gtype_index[(hg_idx, idx)]
                    _m_f[z_target, dst] = _m_f[z_target, src]
                    _m_f[z_target, src] = 0.0

    # Zygote map: expand from G_orig → G_orig * n_slabs, filling only
    # slab 0 (default) — symmetric with glab default behaviour.
    _z2g = np.zeros((g2z.shape[0], g2z.shape[1], n_ztypes), dtype=np.float64)
    for g_raw in range(n_genotypes):
        _z2g[:, :, _ztype_index[(g_raw, 0)]] = g2z[:, :, g_raw]

    # Cytoplasmic zygote redirect: for each non-default glab/slab pair,
    # redirect tagged gamete pairs from slab-0 to the matching child slab.
    # Convention: glab[i] ↔ slab[i] for i >= 1 — same matching-index
    # iteration as the gamete tagging block above.
    if gamete_labels and somatic_labels and n_glabs > 1 and n_slabs > 1:
        from natal.genetic_presets import CytoplasmicPreset
        for idx in range(1, min(n_slabs, n_glabs)):
            if idx >= len(gamete_labels) or idx >= len(somatic_labels):
                continue
            CytoplasmicPreset.apply_zygote_redirect(
                _z2g, gamete_labels[idx], somatic_labels[idx],
                gamete_labels, somatic_labels,
                n_slabs, n_genotypes,
                n_haploid_genotypes, n_glabs,
            )

    z2g_expanded = np.stack([_m_f, _m_m], axis=0)
    return z2g_expanded, _z2g, n_ztypes


def extract_gamete_frequencies(
    zygotes_to_gametes_map: NDArray[np.float64],
    sex_idx: int,
    genotype_idx: int,
    haploid_genotypes: List[HaploidGenotype],
    n_glabs: int = 1,
) -> dict[HaploidGenotype, float]:
    """Extract gamete frequencies for a specific (sex, genotype) pair.

    This convenience function converts a row of zygotes_to_gametes_map
    from compressed haploid-glab indices back to HaploidGenotype objects with
    their aggregated frequencies across all glab variants.

    Args:
        zygotes_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
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
        ...     config.zygotes_to_gametes_map,
        ...     sex_idx=0,
        ...     genotype_idx=5,
        ...     haploid_genotypes=hg_list,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # freqs = {haplotype_obj: 0.5, another_haplotype_obj: 0.5}
    """
    gamete_freqs_array = zygotes_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[HaploidGenotype, float] = {}

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:  # Only include non-zero frequencies
            hg_idx = compressed_idx // n_glabs
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                # Aggregate frequencies across all glab variants
                result[hg] = result.get(hg, 0.0) + freq

    return result


def extract_gamete_frequencies_by_glab(
    zygotes_to_gametes_map: NDArray[np.float64],
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
        zygotes_to_gametes_map: The (n_sexes, n_genotypes, n_hg*n_glabs) array.
        sex_idx: Sex index (0, 1, ...).
        genotype_idx: Diploid genotype index.
        haploid_genotypes: List of all HaploidGenotype objects (aligned with indices).
        n_glabs: Number of gamete-label variants per haplotype (default: 1).

    Returns:
        Dictionary mapping (HaploidGenotype, glab_idx) -> frequency.
        Only includes entries with non-zero frequency.

    Examples:
        >>> freqs = extract_gamete_frequencies_by_glab(
        ...     config.zygotes_to_gametes_map, 0, 5, hg_list, n_glabs=2
        ... )
        >>> # freqs = {(hg_A, 0): 0.3, (hg_A, 1): 0.2, (hg_B, 0): 0.5}
    """
    gamete_freqs_array = zygotes_to_gametes_map[sex_idx, genotype_idx, :]
    result: dict[tuple[HaploidGenotype, int], float] = {}

    for compressed_idx, freq in enumerate(gamete_freqs_array):
        if freq > 0:
            hg_idx = compressed_idx // n_glabs
            glab_idx = compressed_idx % n_glabs
            if hg_idx < len(haploid_genotypes):
                hg = haploid_genotypes[hg_idx]
                result[(hg, glab_idx)] = freq

    return result


def extract_zygote_frequencies(
    gametes_to_zygotes_map: NDArray[np.float64],
    gamete1_compressed_idx: int,
    gamete2_compressed_idx: int,
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
) -> dict[Genotype, float]:
    """Extract zygote frequencies for a specific pair of gametes.

    This convenience function converts a slice of gametes_to_zygotes_map
    from compressed gamete indices to Genotype objects with their frequencies.

    Args:
        gametes_to_zygotes_map: The (n_hg*n_glabs, n_hg*n_glabs, n_genotypes) array.
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
        ...     config.gametes_to_zygotes_map,
        ...     gamete1_compressed_idx=0,
        ...     gamete2_compressed_idx=1,
        ...     diploid_genotypes=genotypes,
        ...     n_glabs=config.n_glabs
        ... )
        >>> # zygote_freqs = {genotype1: 1.0 or {genotype2: 0.5, genotype3: 0.5}, etc}
    """
    zygote_freqs_array = gametes_to_zygotes_map[gamete1_compressed_idx, gamete2_compressed_idx, :]
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
    stochastic: bool
    continuous_sampling: bool

    # -- Dimensions --
    n_sexes: int                    # always 2
    n_ages: int                     # always 2
    n_ztypes: int
    n_haploid_genotypes: int
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



def build_discrete_engine_config(
    *,
    n_genotypes: int,
    n_haploid_genotypes: int,
    n_glabs: int,
    n_slabs: int = 1,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    zygotes_to_gametes_map: NDArray[np.float64],
    gametes_to_zygotes_map: NDArray[np.float64],
    carrying_capacity: float | None = None,
    has_sex_chromosomes: bool = False,
    **kwargs: Any,
) -> DiscretePopulationConfig:
    """Build a ``DiscretePopulationConfig`` independently of ``PopulationConfig``.

    Unlike the old ``from_population_config`` / ``build_discrete_population_config``
    path, this function computes everything from the shared blueprint without
    creating an intermediate ``PopulationConfig``.  The two config types are
    built from the same computation engine (``_build_config_maps``) but assembled
    independently — no conversion, no field-by-field copy.

    Discrete-specific defaults (juvenile survival=1.0, adult survival=0.0,
    new_adult_age=1) are applied before the shared computation runs.
    """
    # Discrete-generation defaults.
    n_ages = int(kwargs.pop("n_ages", 2))
    new_adult_age = int(kwargs.pop("new_adult_age", 1))

    # Survival: juveniles survive to adult (1.0), adults replaced every tick (0.0).
    survival = np.ones((2, n_ages), dtype=np.float64)
    survival[:, 0] = 1.0
    survival[:, 1] = 0.0

    # Mating / reproduction: only adults (age 1) participate.
    mating = np.ones((2, n_ages), dtype=np.float64)
    mating[:, 0] = 0.0
    reproduction = np.ones(n_ages, dtype=np.float64)
    reproduction[0] = 0.0
    fertility = np.ones(n_ages, dtype=np.float64)
    fertility[0] = 0.0

    m = _build_config_maps(
        n_genotypes=n_genotypes,
        n_haploid_genotypes=n_haploid_genotypes,
        n_sexes=2,
        n_ages=n_ages,
        n_glabs=n_glabs,
        n_slabs=n_slabs,
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=new_adult_age,
        stochastic=bool(kwargs.pop("stochastic", True)),
        continuous_sampling=bool(kwargs.pop("continuous_sampling", False)),
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        age_based_survival_rates=survival,
        female_age_based_fertility=fertility,
        viability_fitness=kwargs.pop("viability_fitness", None),
        fecundity_fitness=kwargs.pop("fecundity_fitness", None),
        sexual_selection_fitness=kwargs.pop("sexual_selection_fitness", None),
        zygote_viability_fitness=kwargs.pop("zygote_viability_fitness", None),
        age_based_relative_competition_strength=kwargs.pop("age_based_relative_competition_strength", None),
        sperm_displacement_rate=float(kwargs.pop("sperm_displacement_rate", 0.05)),
        eggs_per_female=float(kwargs.pop("eggs_per_female", 100.0)),
        fixed_egg_count=bool(kwargs.pop("fixed_egg_count", False)),
        carrying_capacity=carrying_capacity or 1000.0,
        sex_ratio=float(kwargs.pop("sex_ratio", 0.5)),
        low_density_growth_rate=float(kwargs.pop("low_density_growth_rate", 6.0)),
        juvenile_growth_mode=int(kwargs.pop("juvenile_growth_mode", 0)),  # LOGISTIC
        has_sex_chromosomes=has_sex_chromosomes,
        zygotes_to_gametes_map=zygotes_to_gametes_map,
        gametes_to_zygotes_map=gametes_to_zygotes_map,
        initial_individual_count=kwargs.pop("initial_individual_count", None),
        initial_sperm_storage=kwargs.pop("initial_sperm_storage", None),
        age_1_carrying_capacity=kwargs.pop("age_1_carrying_capacity", None),
        old_juvenile_carrying_capacity=kwargs.pop("old_juvenile_carrying_capacity", None),
        infer_capacity_from_initial_state=bool(kwargs.pop("infer_capacity_from_initial_state", True)),
        equilibrium_individual_distribution=kwargs.pop("equilibrium_individual_distribution", None),
        external_expected_eggs=kwargs.pop("external_expected_eggs", None),
    )

    return DiscretePopulationConfig(
        stochastic=bool(kwargs.pop("stochastic", True)),
        continuous_sampling=bool(kwargs.pop("continuous_sampling", False)),
        n_sexes=m.n_sexes,
        n_ages=m.n_ages,
        n_ztypes=m.n_g_compressed,
        n_haploid_genotypes=m.n_haploid_genotypes,
        n_glabs=m.n_glabs,
        n_slabs=m.n_slabs,
        female_age_based_fertility=m.female_fertility,
        viability_fitness=m.viability,
        fecundity_fitness=m.fecundity,
        zygote_viability_fitness=m.zygote,
        sexual_selection_fitness=m.sexual,
        age_based_relative_competition_strength=m.competition,
        eggs_per_female=np.array(m.eggs_per_female),
        fixed_egg_count=m.fixed_egg_count,
        sex_ratio=np.array(m.sex_ratio),
        sperm_displacement_rate=np.array(m.sperm_displacement_rate),
        female_adult_mating_rate=float(m.mating[0, 1]),
        male_adult_mating_rate=float(m.mating[1, 1]),
        reproduction_rate=float(m.reproduction[1]),
        female_age0_survival=float(m.survival[0, 0]),
        male_age0_survival=float(m.survival[1, 0]),
        female_fertility=float(m.female_fertility[0]),
        zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
        gametes_to_zygotes_map=m.zygote_map,
        offspring_tensor=m.offspring_tensor,
        meiosis_f=m.meiosis_f,
        meiosis_m=m.meiosis_m,
        fecundity_f=m.fecundity[0],
        fecundity_m=m.fecundity[1],
        viability_f=m.viability[0, 0, :],
        viability_m=m.viability[1, 0, :],
        has_sex_chromosomes=m.has_sex_chromosomes,
        female_genotype_compatibility=m.female_genotype_compatibility,
        male_genotype_compatibility=m.male_genotype_compatibility,
        female_only_by_sex_chrom=m.female_only_by_sex_chrom,
        male_only_by_sex_chrom=m.male_only_by_sex_chrom,
        juvenile_growth_mode=np.array(m.juvenile_growth_mode, dtype=np.int64),
        carrying_capacity=m.carrying_capacity,
        expected_competition_strength=np.array(m.expected_competition_strength),
        expected_survival_rate=np.array(m.expected_survival_rate),
        low_density_growth_rate=np.array(m.low_density_growth_rate),
        generation_time=np.array(0.0),
        new_adult_age=m.new_adult_age,
        adult_ages=m.adult_ages.copy(),
        initial_individual_count=m.initial_individual_count,
        initial_sperm_storage=m.initial_sperm_storage,
        hook_slot=int(kwargs.pop("hook_slot", 0)),
        extreme_speed_mode=int(kwargs.pop("extreme_speed_mode", 0)),
        custom=np.zeros(0, dtype=np.float64),
    )



@overload
def build_custom_array(
    specs: Mapping[
        str,
        bool | np.bool_ | int | np.integer[Any] | float | np.floating[Any] | NDArray[np.float64],
    ],
) -> NDArray[np.void]: ...


@overload
def build_custom_array(specs: Mapping[str, object]) -> NDArray[np.void]: ...


def build_custom_array(specs: Mapping[str, object]) -> NDArray[np.void]:
    """Build a 0-d structured numpy array from custom field specs.

    Called by :meth:`Configurator.custom` and the legacy
    ``PopulationBuilderBase.custom``.

    Each entry in *specs* becomes a named field in the output array:

    - ``bool`` / ``np.bool_`` values produce a ``np.bool_`` field.
    - ``int`` / ``np.integer`` values produce a ``np.int64`` field.
    - ``float`` / ``np.floating`` values produce a ``np.float64`` field.
    - 3-D ``np.ndarray`` values produce a fixed-shape sub-array field
      ``np.float64`` with the array's shape, accessed as
      ``config.custom['name'][sex, age, genotype]``.

    All scalar fields are accessed via ``config.custom['name'][()]``.
    Fields are sorted alphabetically to produce a stable dtype.  The
    returned array is 0-d (scalar structured array), compatible with
    Numba's ``njit`` functions via bracket or attribute access.

    Args:
        specs: ``{name: value}`` mapping of custom field names to values.

    Returns:
        A 0-d structured ``np.ndarray`` for ``PopulationConfig.custom``.

    Raises:
        TypeError: If a value has an unsupported type.
    """
    # Empty specs → 0-d array with empty structured dtype.
    if not specs:
        return np.zeros((), dtype=np.dtype([]))

    # Stage 1: determine dtype fields from each value's Python type.
    # Fields are sorted alphabetically for deterministic byte-identical dtypes.
    fields: list[tuple[str, Any] | tuple[str, Any, tuple[int, ...]]] = []
    for name in sorted(specs):
        val = specs[name]

        if isinstance(val, np.ndarray):
            array_val = cast(np.ndarray[Any, np.dtype[Any]], val)
            shape = array_val.shape
            if len(shape) == 3:
                fields.append((name, np.float64, shape))
            else:
                raise TypeError(
                    f"custom field '{name}' is a {len(shape)}-D ndarray. "
                    f"Only 3-D (sex, age, genotype) arrays are supported."
                )

        # bool checked before int — bool is a subclass of int in Python
        elif isinstance(val, (bool, np.bool_)):
            fields.append((name, np.bool_))

        elif isinstance(val, (int, np.integer)):
            fields.append((name, np.int64))

        elif isinstance(val, (float, np.floating)):
            fields.append((name, np.float64))

        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}. "
                f"Supported types: bool, int, float (including NumPy scalars), "
                f"or 3-D np.ndarray."
            )

    # Stage 2: build the structured dtype and allocate the 0-d array.
    dtype = np.dtype(fields)
    custom = np.zeros((), dtype=dtype)

    # Stage 3: write initial values.
    for name in sorted(specs):
        val = specs[name]

        if isinstance(val, np.ndarray):
            custom[name][...] = val         # sub-array field: copy block
        elif isinstance(val, (bool, np.bool_, int, np.integer, float, np.floating)):
            custom[name][()] = val          # scalar field: 0-d element access
        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}."
            )

    return custom


# ---------------------------------------------------------------------------
# Gamete-axis compression helpers
# ---------------------------------------------------------------------------


def compress_gamete_map(
    z2g: NDArray[np.float64],
    mask: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compress the last axis of a genotype-to-gametes map.

    Args:
        z2g: ``(n_sexes, n_genotypes, HL)`` — full gamete map.
        mask: ``(HL,) int32`` — compression mask (-1 = prune).

    Returns:
        Compressed map, shape ``(n_sexes, n_genotypes, HL')``.
    """
    active = mask >= 0
    return z2g[:, :, active]


def compress_zygote_map(
    z2g: NDArray[np.float64],
    mask: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compress both gamete axes of a zygote map.

    Args:
        z2g: ``(HL, HL, n_genotypes)`` — full zygote map.
        mask: ``(HL,) int32`` — compression mask.

    Returns:
        Compressed map, shape ``(HL', HL', n_genotypes)``.
    """
    active = mask >= 0
    return z2g[active, :, :][:, active, :]


def apply_ztype_mask(
    z2g: NDArray[np.float64],
    mask: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compress the genotype (ZType) axis of a genotype-to-gametes map.

    Args:
        z2g: ``(n_sexes, G_orig, HL)`` — full map.
        mask: ``(G_orig,)`` int32 — ZType compression mask (-1 = prune).

    Returns:
        Compressed map, shape ``(n_sexes, G_total, HL)``.
    """
    active = mask >= 0
    return z2g[:, active, :]


def apply_ztype_mask_zygote(
    z2g: NDArray[np.float64],
    mask: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compress the genotype (ZType) axis of a zygote map.

    Args:
        z2g: ``(HL, HL, G_orig)`` — full zygote map.
        mask: ``(G_orig,)`` int32 — ZType compression mask.

    Returns:
        Compressed map, shape ``(HL, HL, G_total)``.
    """
    active = mask >= 0
    return z2g[:, :, active]


def compress_config(
    config: Union[PopulationConfig, DiscretePopulationConfig],
    ztype_mask: NDArray[np.int32],
) -> Union[PopulationConfig, DiscretePopulationConfig]:
    """Subslice all G-axis-dependent fields to match a ZType compression mask.

    Pure function — returns a new config via ``_replace`` without mutating
    the original.  Handles both ``PopulationConfig`` and
    ``DiscretePopulationConfig``.

    Args:
        config: Config to compress.  All arrays indexed by ``n_ztypes``
            will be subsliced.
        ztype_mask: ``(n_ztypes,)`` int32 array — -1 = pruned.

    Returns:
        A new config with all G-dependent arrays compressed.
    """
    _z_active = ztype_mask >= 0
    n_g = int(_z_active.sum())

    overrides: dict[str, Any] = {
        "n_ztypes": n_g,
        "initial_individual_count": config.initial_individual_count[:, :, _z_active],
        "viability_fitness": config.viability_fitness[:, :, _z_active],
        "fecundity_fitness": config.fecundity_fitness[:, _z_active],
        "sexual_selection_fitness": config.sexual_selection_fitness[_z_active, :][:, _z_active],
        "zygote_viability_fitness": config.zygote_viability_fitness[:, _z_active],
        "female_genotype_compatibility": config.female_genotype_compatibility[_z_active],
        "male_genotype_compatibility": config.male_genotype_compatibility[_z_active],
        "female_only_by_sex_chrom": config.female_only_by_sex_chrom[_z_active],
        "male_only_by_sex_chrom": config.male_only_by_sex_chrom[_z_active],
        "initial_sperm_storage": config.initial_sperm_storage[:, _z_active, :][:, :, _z_active],
    }

    return config._replace(**overrides)
