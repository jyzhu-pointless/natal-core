"""Pure-function simulation engine run outside Population with Numba support."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import natal.engine.simulation.age_structured as alg
from natal.data import (
    FIXED,
    LOGISTIC,
    NO_COMPETITION,
    PopulationConfig,
    PopulationState,
)
from natal.numba import compat as nbc
from natal.numba.compat import binomial
from natal.numba.utils import njit_switch

__all__ = [
    # No user-facing API for now
]

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
        ind_count: Individual-count array ``(n_sexes, n_ages, n_ztypes)``.
        sperm_store: Sperm-store array ``(n_ages, n_ztypes, n_ztypes)``.
        config: PopulationConfig object.
        offspring_probability: Precomputed offspring tensor
            ``P_offspring[gf, gm, g_off]`` reused across demes/ticks.

    Returns:
        Tuple[ind_count, sperm_store]: Updated arrays.
    """
    # Modify ind_count in-place; callers do not expect original to be preserved.

    n_ages = config.n_ages
    n_ztypes = config.n_ztypes
    adult_ages = config.adult_ages
    adult_start_age = adult_ages[0] if len(adult_ages) > 0 else 0
    stochastic = config.stochastic
    continuous_sampling = config.continuous_sampling

    # 1. Extract effective adult male counts (weighted by age-specific mating rates).
    # effective_male_counts = Σ (male_counts[age] * male_mating_rate[age])
    effective_male_counts = np.zeros(n_ztypes, dtype=np.float64)
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
        n_ztypes
    )

    # 3. Update sperm-store state (the mating process).
    # alg.sample_mating updates sperm storage based on mating rates
    female_counts = ind_count[0, :, :] # (n_ages, n_ztypes)

    sperm_store = alg.sample_mating(
        female_counts,
        sperm_store,
        mating_prob,
        config.age_based_mating_rates[0, :],  # female age-specific mating rates
        config.sperm_displacement_rate[()],  # pyright: ignore[reportArgumentType]
        adult_start_age,
        n_ages,
        n_ztypes,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling
    )

    # 4. Generate offspring (fertilization).
    female_ztype_compatibility = config.female_ztype_compatibility
    male_ztype_compatibility = config.male_ztype_compatibility
    female_only_by_sex_chrom = config.female_only_by_sex_chrom
    male_only_by_sex_chrom = config.male_only_by_sex_chrom
    has_sex_chromosomes = config.has_sex_chromosomes

    n_0_female, n_0_male = alg.fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
        female_counts,
        sperm_store,
        config.fecundity_fitness[0], # sex=0 is FEMALE
        config.fecundity_fitness[1], # sex=1 is MALE
        offspring_probability,
        config.eggs_per_female[()],  # pyright: ignore[reportArgumentType]
        adult_start_age,
        n_ages,
        n_ztypes,
        config.n_gtypes,
        female_ztype_compatibility,
        male_ztype_compatibility,
        female_only_by_sex_chrom,
        male_only_by_sex_chrom,
        config.n_glabs,
        config.age_based_reproduction_rates,  # 直接传递年龄特定的繁殖率
        config.female_age_based_fertility,  # 传递年龄特定的相对生育率
        config.fixed_egg_count, # fixed_eggs
        config.sex_ratio[()],  # pyright: ignore[reportArgumentType]
        has_sex_chromosomes=has_sex_chromosomes,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling
    )

    # Note: Sex.FEMALE = 0, Sex.MALE = 1.
    ind_count[0, 0, :] = n_0_female  # sex=0 is FEMALE
    ind_count[1, 0, :] = n_0_male    # sex=1 is MALE

    # 5. Apply zygote fitness to newly formed offspring (age-0 individuals)
    if hasattr(config, 'zygote_viability_fitness'):
        # Apply zygote fitness to age-0 individuals with proper stochastic sampling
        if stochastic:
            # Use stochastic sampling for zygote survival
            female_offspring = ind_count[0, 0, :].copy()
            male_offspring = ind_count[1, 0, :].copy()

            # Apply zygote fitness using binomial sampling
            for g in range(n_ztypes):
                if continuous_sampling:
                    # Continuous sampling: use continuous_binomial function
                    if female_offspring[g] > 0:
                        female_offspring[g] = nbc.continuous_binomial(
                            female_offspring[g], config.zygote_viability_fitness[0, g]
                        )
                    if male_offspring[g] > 0:
                        male_offspring[g] = nbc.continuous_binomial(
                            male_offspring[g], config.zygote_viability_fitness[1, g]
                        )
                else:
                    # Discrete sampling: use standard binomial distribution
                    if female_offspring[g] > 0:
                        n_female = int(round(female_offspring[g]))
                        if n_female > 0:
                            female_offspring[g] = nbc.binomial(n_female, config.zygote_viability_fitness[0, g])
                    if male_offspring[g] > 0:
                        n_male = int(round(male_offspring[g]))
                        if n_male > 0:
                            male_offspring[g] = binomial(n_male, config.zygote_viability_fitness[1, g])

            ind_count[0, 0, :] = female_offspring
            ind_count[1, 0, :] = male_offspring
        else:
            # Deterministic mode: simple multiplication
            ind_count[0, 0, :] *= config.zygote_viability_fitness[0, :]  # Female offspring
            ind_count[1, 0, :] *= config.zygote_viability_fitness[1, :]  # Male offspring

    return ind_count, sperm_store


@njit_switch(cache=True)
def run_reproduction(
    state: PopulationState,
    config: PopulationConfig,
) -> PopulationState:
    """Run reproduction stage: mating, sperm-store update, and offspring generation.

    Args:
        state: Current population state.
        config: PopulationConfig object.

    Returns:
        New population state with updated individual counts and sperm store.
        ``n_tick`` is preserved; the lifecycle orchestrator advances it after
        aging.
    """
    ind_count, sperm_store = run_reproduction_with_precomputed_offspring_probability(
        ind_count=state.individual_count,
        sperm_store=state.sperm_storage,
        config=config,
        offspring_probability=config.offspring_tensor,
    )
    return PopulationState(
        n_tick=state.n_tick,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )

@njit_switch(cache=True)
def run_survival(
    state: PopulationState,
    config: PopulationConfig,
) -> PopulationState:
    """Run survival stage: apply survival/viability and juvenile recruitment.

    New flow:
    1. Compute survival components (as survival-rate arrays)
    2. Apply all survival rates in one pass (stochastic or deterministic)
    3. Perform density-dependent juvenile recruitment

    Args:
        state: Current population state.
        config: PopulationConfig instance.

    Returns:
        New population state with updated individual counts and sperm store.
    """
    ind_count = state.individual_count.copy()
    sperm_store = state.sperm_storage.copy()
    n_ages = config.n_ages
    n_ztypes = config.n_ztypes
    stochastic = config.stochastic
    continuous_sampling = config.continuous_sampling

    # =========================================================================
    # Firstly, apply density-dependent survival to age 0 individuals (juveniles) based on the configured growth mode.
    # =========================================================================
    # Use the unified recruit_juveniles_given_scaling_factor_sampling API.
    # Mode constants: 0=NO_COMPETITION, 1=FIXED, 2=LOGISTIC/LINEAR, 3=BEVERTON_HOLT/CONCAVE
    juvenile_growth_mode = config.juvenile_growth_mode[()]  # pyright: ignore[reportArgumentType]
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
            carrying_capacity=config.carrying_capacity[()],  # pyright: ignore[reportArgumentType]
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
                expected_competition_strength=config.expected_competition_strength[()],  # pyright: ignore[reportArgumentType]
                expected_survival_rate=config.expected_survival_rate[()],  # pyright: ignore[reportArgumentType]
                low_density_growth_rate=config.low_density_growth_rate[()],  # pyright: ignore[reportArgumentType]
            )
        else: # Mode 3: BEVERTON_HOLT / CONCAVE
            scaling_factor = alg.compute_scaling_factor_beverton_holt(
                actual_competition_strength=actual_comp,
                expected_competition_strength=config.expected_competition_strength[()],  # pyright: ignore[reportArgumentType]
                expected_survival_rate=config.expected_survival_rate[()],  # pyright: ignore[reportArgumentType]
                low_density_growth_rate=config.low_density_growth_rate[()],  # pyright: ignore[reportArgumentType]
            )

    # Unified call to recruit_juveniles_given_scaling_factor_sampling.
    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        age_0_counts,
        scaling_factor,
        n_ztypes,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling
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

    # 1.2 Viability survival rates -> shape (n_ages, n_ztypes).
    target_viability_age = config.new_adult_age - 1
    s_via_f, s_via_m = alg.compute_viability_survival_rates(
        config.viability_fitness[0, target_viability_age, :],
        config.viability_fitness[1, target_viability_age, :],
        n_ztypes,
        target_viability_age,
        n_ages
    )

    # 2 Combine survival rates (age-specific × viability) → shape (n_ages, n_ztypes)
    # Total survival rate = age-based survival x viability survival.
    # Broadcasting needed: s_age_f shape (n_ages,) and s_via_f shape (n_ages, n_ztypes).
    s_combined_f = s_age_f[:, None] * s_via_f  # (n_ages, n_ztypes)
    s_combined_m = s_age_m[:, None] * s_via_m  # (n_ages, n_ztypes)

    # 3 Apply combined survival rates to individuals
    if stochastic:
        # Stochastic sampling: keep sperm_store and individual counts synchronized.
        f_surv, m_surv, sperm_store = alg.sample_survival_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,  # shape (n_ages, n_ztypes)
        s_combined_m,
        n_ztypes,
        n_ages,
        continuous_sampling=continuous_sampling
        )
        ind_count[0], ind_count[1] = f_surv, m_surv
    else:
        # Deterministic scaling: update individual counts and sperm store together.
        ind_count[0], ind_count[1], sperm_store = alg.apply_survival_rates_deterministic_with_sperm_storage(
            (ind_count[0], ind_count[1]),
            sperm_store,
            s_combined_f,
        s_combined_m,
        n_ztypes,
        n_ages
    )

    return PopulationState(
        n_tick=state.n_tick,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )

@njit_switch(cache=True)
def run_aging(
    state: PopulationState,
    config: PopulationConfig,
) -> PopulationState:
    """Run aging stage: advance age classes.

    Args:
        state: Current population state.
        config: PopulationConfig instance.

    Returns:
        New population state with advanced age classes.
    """
    ind_count = state.individual_count.copy()
    sperm_store = state.sperm_storage.copy()

    n_ages = config.n_ages

    # Age advancement.
    for age in range(n_ages - 1, 0, -1):
        ind_count[:, age, :] = ind_count[:, age - 1, :]
        sperm_store[age, :, :] = sperm_store[age - 1, :, :]

    ind_count[:, 0, :] = 0.0
    sperm_store[0, :, :] = 0.0

    return PopulationState(
        n_tick=state.n_tick,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )
