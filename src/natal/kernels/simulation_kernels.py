"""Pure-function simulation kernels run outside Population with Numba support."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import natal.algorithms as alg
from natal.numba_compat import binomial
from natal.numba_utils import njit_switch
from natal.population_config import FIXED, LOGISTIC, NO_COMPETITION, PopulationConfig
from natal.population_state import PopulationState

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation

__all__ = [
    # No user-facing API for now
]

# ============================================================================
# Export/import helpers (lightweight wrappers; call population methods directly)
# ============================================================================

def export_config(pop: 'AgeStructuredPopulation') -> 'PopulationConfig':
    """Export population configuration. Prefer ``pop.export_config()`` directly."""
    return pop.export_config()


def import_config(pop: 'AgeStructuredPopulation', config: 'PopulationConfig') -> None:
    """Import configuration into population. Prefer ``pop.import_config()`` directly."""
    pop.import_config(config)


def export_state(pop: 'AgeStructuredPopulation') -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Export population state. Prefer ``pop.export_state()`` directly."""
    return pop.export_state()


def import_state(pop: 'AgeStructuredPopulation', state: 'PopulationState') -> None:
    """Import state into population. Prefer ``pop.import_state()`` directly."""
    pop.import_state(state)

# ============================================================================
# Core: separated stage functions (reproduction, survival, aging)
# ============================================================================
@njit_switch(cache=True)
def run_reproduction_with_precomputed_offspring_probability(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
    offspring_probability: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run reproduction stage: mating, sperm-store update, and offspring generation.

    Args:
        ind_count: Individual-count array ``(n_sexes, n_ages, n_genotypes)``.
        sperm_store: Sperm-store array ``(n_ages, n_genotypes, n_genotypes)``.
        config: PopulationConfig object.
        offspring_probability: Precomputed offspring tensor
            ``P_offspring[gf, gm, g_off]`` reused across demes/ticks.

    Returns:
        Tuple[ind_count, sperm_store]: Updated arrays.
    """
    # Modify ind_count in-place; callers do not expect original to be preserved.

    n_ages = config.n_ages
    n_gen = config.n_genotypes
    adult_ages = config.adult_ages
    adult_start_age = adult_ages[0] if len(adult_ages) > 0 else 0
    is_stochastic = config.is_stochastic
    use_continuous_sampling = config.use_continuous_sampling

    # 1. Extract effective adult male counts (weighted by age-specific mating rates).
    # effective_male_counts = Σ (male_counts[age] * male_mating_rate[age])
    effective_male_counts = np.zeros(n_gen, dtype=np.float64)
    for age in adult_ages:
        if age < n_ages:
            male_mating_rate_at_age = config.age_based_mating_rates[1, age]  # sex=1 is MALE
            effective_male_counts += ind_count[1, age, :] * male_mating_rate_at_age

    if effective_male_counts.sum() == 0:
        # No males or no mating males, no new matings, no offspring
        return ind_count, sperm_store

    # 2. Compute mating probability matrix (g, g) from effective male counts.
    mating_prob = alg.compute_mating_probability_matrix(
        config.sexual_selection_fitness,
        effective_male_counts,
        n_gen
    )

    # 3. Update sperm-store state (the mating process).
    # alg.sample_mating updates sperm storage based on mating rates
    female_counts = ind_count[0, :, :] # (n_ages, n_genotypes)

    sperm_store = alg.sample_mating(
        female_counts,
        sperm_store,
        mating_prob,
        config.age_based_mating_rates[0, :],  # female age-specific mating rates
        config.sperm_displacement_rate,
        adult_start_age,
        n_ages,
        n_gen,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )

    # 4. Generate offspring (fertilization).
    female_genotype_compatibility = config.female_genotype_compatibility
    male_genotype_compatibility = config.male_genotype_compatibility
    female_only_by_sex_chrom = config.female_only_by_sex_chrom
    male_only_by_sex_chrom = config.male_only_by_sex_chrom
    has_sex_chromosomes = config.has_sex_chromosomes

    n_0_female, n_0_male = alg.fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
        female_counts,
        sperm_store,
        config.fecundity_fitness[0], # sex=0 is FEMALE
        config.fecundity_fitness[1], # sex=1 is MALE
        offspring_probability,
        config.expected_eggs_per_female,
        adult_start_age,
        n_ages,
        n_gen,
        config.n_haploid_genotypes,
        female_genotype_compatibility,
        male_genotype_compatibility,
        female_only_by_sex_chrom,
        male_only_by_sex_chrom,
        config.n_glabs,
        config.age_based_reproduction_rates,  # 直接传递年龄特定的繁殖率
        config.female_age_based_relative_fertility,  # 传递年龄特定的相对生育率
        config.use_fixed_egg_count, # fixed_eggs
        config.sex_ratio,
        has_sex_chromosomes=has_sex_chromosomes,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )

    # Note: Sex.FEMALE = 0, Sex.MALE = 1.
    ind_count[0, 0, :] = n_0_female  # sex=0 is FEMALE
    ind_count[1, 0, :] = n_0_male    # sex=1 is MALE

    # 5. Apply zygote fitness to newly formed offspring (age-0 individuals)
    if hasattr(config, 'zygote_fitness'):
        # Apply zygote fitness to age-0 individuals with proper stochastic sampling
        if is_stochastic:
            # Use stochastic sampling for zygote survival
            female_offspring = ind_count[0, 0, :].copy()
            male_offspring = ind_count[1, 0, :].copy()

            # Apply zygote fitness using binomial sampling
            for g in range(n_gen):
                if use_continuous_sampling:
                    # Continuous sampling: use continuous_binomial function
                    if female_offspring[g] > 0:
                        female_offspring[g] = alg.continuous_binomial(
                            female_offspring[g], config.zygote_fitness[0, g]
                        )
                    if male_offspring[g] > 0:
                        male_offspring[g] = alg.continuous_binomial(
                            male_offspring[g], config.zygote_fitness[1, g]
                        )
                else:
                    # Discrete sampling: use standard binomial distribution
                    if female_offspring[g] > 0:
                        n_female = int(round(female_offspring[g]))
                        if n_female > 0:
                            female_offspring[g] = binomial(n_female, config.zygote_fitness[0, g])
                    if male_offspring[g] > 0:
                        n_male = int(round(male_offspring[g]))
                        if n_male > 0:
                            male_offspring[g] = binomial(n_male, config.zygote_fitness[1, g])

            ind_count[0, 0, :] = female_offspring
            ind_count[1, 0, :] = male_offspring
        else:
            # Deterministic mode: simple multiplication
            ind_count[0, 0, :] *= config.zygote_fitness[0, :]  # Female offspring
            ind_count[1, 0, :] *= config.zygote_fitness[1, :]  # Male offspring

    return ind_count, sperm_store


@njit_switch(cache=True)
def run_reproduction(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run reproduction stage: mating, sperm-store update, and offspring generation.

    Args:
        ind_count: Individual-count array ``(n_sexes, n_ages, n_genotypes)``.
        sperm_store: Sperm-store array ``(n_ages, n_genotypes, n_genotypes)``.
        config: PopulationConfig object.

    Returns:
        Tuple[ind_count, sperm_store]: Updated arrays.
    """
    offspring_probability = alg.compute_offspring_probability_tensor(
        meiosis_f=config.genotype_to_gametes_map[0],
        meiosis_m=config.genotype_to_gametes_map[1],
        haplo_to_genotype_map=config.gametes_to_zygote_map,
        n_genotypes=config.n_genotypes,
        n_haplogenotypes=config.n_haploid_genotypes,
        n_glabs=config.n_glabs,
    )
    return run_reproduction_with_precomputed_offspring_probability(
        ind_count=ind_count,
        sperm_store=sperm_store,
        config=config,
        offspring_probability=offspring_probability,
    )

@njit_switch(cache=True)
def run_survival(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run survival stage: apply survival/viability and juvenile recruitment.

    New flow:
    1. Compute survival components (as survival-rate arrays)
    2. Apply all survival rates in one pass (stochastic or deterministic)
    3. Perform density-dependent juvenile recruitment

    Args:
        ind_count: Individual-count array ``(n_sexes, n_ages, n_genotypes)``.
        sperm_store: Sperm-store array ``(n_ages, n_genotypes, n_genotypes)``.
        config: PopulationConfig instance.

    Returns:
        Tuple[ind_count, sperm_store]: Updated individual counts and sperm store.
    """
    ind_count = ind_count.copy()
    sperm_store = sperm_store.copy()
    n_ages = config.n_ages
    n_gen = config.n_genotypes
    is_stochastic = config.is_stochastic
    use_continuous_sampling = config.use_continuous_sampling

    # =========================================================================
    # Firstly, apply density-dependent survival to age 0 individuals (juveniles) based on the configured growth mode.
    # =========================================================================
    # Use the unified recruit_juveniles_given_scaling_factor_sampling API.
    # Mode constants: 0=NO_COMPETITION, 1=FIXED, 2=LOGISTIC/LINEAR, 3=BEVERTON_HOLT/CONCAVE
    juvenile_growth_mode = config.juvenile_growth_mode
    new_adult_age = config.new_adult_age

    # Compute scaling_factor.
    age_0_counts = (ind_count[0, 0, :], ind_count[1, 0, :])
    total_age_0 = float(ind_count[0, 0, :].sum() + ind_count[1, 0, :].sum())

    if juvenile_growth_mode == NO_COMPETITION:
        # Mode 0: NO_COMPETITION - no density dependence.
        scaling_factor = 1.0
    elif juvenile_growth_mode == FIXED:
        # Mode 1: FIXED - scale down proportionally when above K.
        scaling_factor = alg.compute_scaling_factor_fixed(
            total_age_0=total_age_0,
            carrying_capacity=config.carrying_capacity,
        )
    else:
        # Mode 2 (LOGISTIC/LINEAR) or Mode 3 (BEVERTON_HOLT/CONCAVE).
        # Aggregate juvenile counts by age and compute actual competition strength.
        juvenile_counts = np.zeros(new_adult_age, dtype=np.float64)
        for age in range(new_adult_age):
            juvenile_counts[age] = float(ind_count[0, age, :].sum() + ind_count[1, age, :].sum())

        actual_comp = alg.compute_actual_competition_strength(
            juvenile_counts_by_age=juvenile_counts,
            relative_competition_strength=config.age_based_relative_competition_strength,
            new_adult_age=new_adult_age
        )

        if juvenile_growth_mode == LOGISTIC:
            scaling_factor = alg.compute_scaling_factor_logistic(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
        else: # Mode 3: BEVERTON_HOLT / CONCAVE
            scaling_factor = alg.compute_scaling_factor_beverton_holt(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )

    # Unified call to recruit_juveniles_given_scaling_factor_sampling.
    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        age_0_counts,
        scaling_factor,
        n_gen,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )
    ind_count[0, 0, :] = f_rec
    ind_count[1, 0, :] = m_rec

    # =========================================================================
    # Then, apply age-specific survival and viability selection to all individuals.
    # =========================================================================

    # 1 Compute age-specific survival rates
    # 1.1 Age-specific survival rates -> shape (n_ages,).
    s_age_f, s_age_m = alg.compute_age_based_survival_rates(
        config.age_based_survival_rates[0],
        config.age_based_survival_rates[1],
        n_ages
    )

    # 1.2 Viability survival rates -> shape (n_ages, n_genotypes).
    target_viability_age = config.new_adult_age - 1
    s_via_f, s_via_m = alg.compute_viability_survival_rates(
        config.viability_fitness[0, target_viability_age, :],
        config.viability_fitness[1, target_viability_age, :],
        n_gen,
        target_viability_age,
        n_ages
    )

    # 2 Combine survival rates (age-specific × viability) → shape (n_ages, n_genotypes)
    # Total survival rate = age-based survival x viability survival.
    # Broadcasting needed: s_age_f shape (n_ages,) and s_via_f shape (n_ages, n_genotypes).
    s_combined_f = s_age_f[:, None] * s_via_f  # (n_ages, n_genotypes)
    s_combined_m = s_age_m[:, None] * s_via_m  # (n_ages, n_genotypes)

    # 3 Apply combined survival rates to individuals
    if is_stochastic:
        # Stochastic sampling: keep sperm_store and individual counts synchronized.
        f_surv, m_surv, sperm_store = alg.sample_survival_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,  # shape (n_ages, n_genotypes)
            s_combined_m,
            n_gen,
            n_ages,
            use_continuous_sampling=use_continuous_sampling
        )
        ind_count[0], ind_count[1] = f_surv, m_surv
    else:
        # Deterministic scaling: update individual counts and sperm store together.
        ind_count[0], ind_count[1], sperm_store = alg.apply_survival_rates_deterministic_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,
            s_combined_m,
            n_gen,
            n_ages
        )

    return ind_count, sperm_store

@njit_switch(cache=True)
def run_aging(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run aging stage: advance age classes.

    Args:
        ind_count: Individual-count array ``(n_sexes, n_ages, n_genotypes)``.
        sperm_store: Sperm-store array.
        config: PopulationConfig instance.

    Returns:
        Tuple[ind_count, sperm_store]: Updated arrays.
    """
    ind_count = ind_count.copy()
    sperm_store = sperm_store.copy()

    n_ages = config.n_ages

    # Age advancement.
    for age in range(n_ages - 1, 0, -1):
        ind_count[:, age, :] = ind_count[:, age - 1, :]
        sperm_store[age, :, :] = sperm_store[age - 1, :, :]

    ind_count[:, 0, :] = 0.0
    sperm_store[0, :, :] = 0.0

    return ind_count, sperm_store


# ============================================================================
# Discrete Generation Kernels
# ============================================================================

@njit_switch(cache=True)
def run_discrete_reproduction(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]:
    """Run reproduction stage (discrete generation): direct fertilization without long-term sperm storage."""
    ind_count = ind_count.copy()
    n_gen = config.n_genotypes
    is_stochastic = config.is_stochastic
    use_continuous_sampling = config.use_continuous_sampling

    adult_age = 1
    female_adults = ind_count[0, adult_age, :]
    male_adults = ind_count[1, adult_age, :]

    male_mating_rate = config.age_based_mating_rates[1, adult_age]
    effective_male_counts = male_adults * male_mating_rate

    if effective_male_counts.sum() == 0 or female_adults.sum() == 0:
        return ind_count

    mating_prob = alg.compute_mating_probability_matrix(
        config.sexual_selection_fitness,
        effective_male_counts,
        n_gen
    )

    temp_sperm_store = np.zeros((2, n_gen, n_gen), dtype=np.float64)
    temp_female_counts = np.zeros((2, n_gen), dtype=np.float64)
    temp_female_counts[adult_age, :] = female_adults

    temp_sperm_store = alg.sample_mating(
        temp_female_counts,
        temp_sperm_store,
        mating_prob,
        config.age_based_mating_rates[0, :],  # female age-specific mating rates
        1.0,
        adult_age,
        2,
        n_gen,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )

    female_genotype_compatibility = config.female_genotype_compatibility
    male_genotype_compatibility = config.male_genotype_compatibility
    female_only_by_sex_chrom = config.female_only_by_sex_chrom
    male_only_by_sex_chrom = config.male_only_by_sex_chrom

    n_0_female, n_0_male = alg.fertilize_with_mating_genotype(
        temp_female_counts,
        temp_sperm_store,
        config.fecundity_fitness[0],
        config.fecundity_fitness[1],
        config.genotype_to_gametes_map[0],
        config.genotype_to_gametes_map[1],
        config.gametes_to_zygote_map,
        config.expected_eggs_per_female,
        adult_age,
        2,
        n_gen,
        config.n_haploid_genotypes,
        female_genotype_compatibility,
        male_genotype_compatibility,
        female_only_by_sex_chrom,
        male_only_by_sex_chrom,
        config.n_glabs,
        config.age_based_reproduction_rates,
        config.female_age_based_relative_fertility,
        config.use_fixed_egg_count,
        config.sex_ratio,
        has_sex_chromosomes=config.has_sex_chromosomes,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )

    ind_count[0, 0, :] = n_0_female
    ind_count[1, 0, :] = n_0_male

    return ind_count

@njit_switch(cache=True)
def run_discrete_survival(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]:
    """Run survival stage (discrete generation): juvenile competition and survival filtering."""
    ind_count = ind_count.copy()
    n_gen = config.n_genotypes
    # Read booleans through explicit casts to avoid ambiguous truthy handling
    # and to force recompilation after PopulationConfig schema updates.
    is_stochastic = bool(config.is_stochastic)
    use_continuous_sampling = bool(config.use_continuous_sampling)

    juvenile_growth_mode = config.juvenile_growth_mode
    total_age_0 = float(ind_count[0, 0, :].sum() + ind_count[1, 0, :].sum())

    if juvenile_growth_mode == NO_COMPETITION:
        scaling_factor = 1.0
    elif juvenile_growth_mode == FIXED:
        scaling_factor = alg.compute_scaling_factor_fixed(
            total_age_0=total_age_0,
            carrying_capacity=config.carrying_capacity,
        )
    else:
        # Discrete generation has exactly one juvenile age (age 0),
        # so competition strength reduces to the age-0 total count.
        actual_comp = total_age_0
        if juvenile_growth_mode == LOGISTIC:
            scaling_factor = alg.compute_scaling_factor_logistic(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )
        else:
            scaling_factor = alg.compute_scaling_factor_beverton_holt(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength,
                expected_survival_rate=config.expected_survival_rate,
                low_density_growth_rate=config.low_density_growth_rate,
            )

    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        (ind_count[0, 0, :], ind_count[1, 0, :]),
        scaling_factor,
        n_gen,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling
    )

    s_age_f, s_age_m = alg.compute_age_based_survival_rates(
        config.age_based_survival_rates[0],
        config.age_based_survival_rates[1],
        n_ages=2
    )
    s_via_f, s_via_m = alg.compute_viability_survival_rates(
        config.viability_fitness[0, 0, :],
        config.viability_fitness[1, 0, :],
        n_gen,
        target_age=0,
        n_ages=2
    )

    s_combined_0_f = s_age_f[0] * s_via_f[0, :]
    s_combined_0_m = s_age_m[0] * s_via_m[0, :]

    if is_stochastic:
        if use_continuous_sampling:
            # Continuous approximation: use Beta-based binomial emulation.
            f_surv = np.empty(n_gen, dtype=np.float64)
            m_surv = np.empty(n_gen, dtype=np.float64)
            for g in range(n_gen):
                f_surv[g] = alg.continuous_binomial(f_rec[g], s_combined_0_f[g])
                m_surv[g] = alg.continuous_binomial(m_rec[g], s_combined_0_m[g])
        else:
            f_surv = np.zeros(n_gen, dtype=np.float64)
            m_surv = np.zeros(n_gen, dtype=np.float64)
            for g in range(n_gen):
                nf = int(round(f_rec[g]))
                nm = int(round(m_rec[g]))
                f_surv[g] = float(binomial(nf, s_combined_0_f[g]))  # pyright: ignore
                m_surv[g] = float(binomial(nm, s_combined_0_m[g]))  # pyright: ignore
    else:
        f_surv = f_rec * s_combined_0_f
        m_surv = m_rec * s_combined_0_m

    ind_count[0, 0, :] = f_surv
    ind_count[1, 0, :] = m_surv

    return ind_count

@njit_switch(cache=True)
def run_discrete_aging(
    ind_count: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run generation turnover (discrete generation): juveniles become adults and old adults are discarded."""
    ind_count = ind_count.copy()

    ind_count[0, 1, :] = ind_count[0, 0, :]
    ind_count[0, 0, :] = 0.0

    ind_count[1, 1, :] = ind_count[1, 0, :]
    ind_count[1, 0, :] = 0.0

    return ind_count
