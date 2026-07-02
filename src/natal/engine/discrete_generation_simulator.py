"""Discrete-generation lifecycle engine.

Orchestrate the three lifecycle stages using dedicated discrete algorithms.
Each function takes ``DiscretePopulationConfig``.
"""

import numpy as np
from numpy.typing import NDArray

import natal.engine.simulation.age_structured as alg
from natal import numba_compat as nbc
from natal.engine.simulation.discrete_generation import (
    fertilize_discrete,
    mate_discrete,
)
from natal.numba_utils import njit_switch
from natal.population_config import (
    FIXED,
    LOGISTIC,
    NO_COMPETITION,
    DiscretePopulationConfig,
)

__all__ = [
    "run_discrete_reproduction",
    "run_discrete_survival",
    "run_discrete_aging",
]

# ── Stage A: reproduction ────────────────────────────────────────────────────


@njit_switch(cache=True)
def run_discrete_reproduction(
    ind_count: NDArray[np.float64],
    cfg: DiscretePopulationConfig,
) -> NDArray[np.float64]:
    """One tick of discrete reproduction: mate → fertilize → offspring in age 0."""
    ind_count = ind_count.copy()
    n_ztypes = cfg.n_ztypes
    stochastic = cfg.stochastic
    continuous = cfg.continuous_sampling

    females = ind_count[0, 1, :]
    males = ind_count[1, 1, :]
    effective_males = males * cfg.male_adult_mating_rate
    if effective_males.sum() == 0.0 or females.sum() == 0.0:
        return ind_count

    mating_prob = alg.compute_mating_probability_matrix(
        cfg.sexual_selection_fitness, effective_males, n_ztypes,
    )

    sperm = mate_discrete(
        females,
        mating_prob,
        cfg.female_adult_mating_rate,
        stochastic,
        continuous
    )

    n_f, n_m = fertilize_discrete(
        sperm, cfg.offspring_tensor,
        cfg.fecundity_f, cfg.fecundity_m,
        cfg.eggs_per_female[()],  # pyright: ignore[reportArgumentType]
        cfg.reproduction_rate,
        cfg.sex_ratio[()],  # pyright: ignore[reportArgumentType]
        cfg.has_sex_chromosomes,
        cfg.female_ztype_compatibility, cfg.male_ztype_compatibility,
        cfg.female_only_by_sex_chrom, cfg.male_only_by_sex_chrom,
        stochastic, continuous,
    )

    ind_count[0, 0, :] = n_f
    ind_count[1, 0, :] = n_m
    return ind_count


# ── Stage B: survival ────────────────────────────────────────────────────────


@njit_switch(cache=True)
def run_discrete_survival(
    ind_count: NDArray[np.float64],
    cfg: DiscretePopulationConfig,
) -> NDArray[np.float64]:
    """Juvenile density regulation then genotype viability selection."""
    ind_count = ind_count.copy()
    n_ztypes = cfg.n_ztypes
    stochastic = cfg.stochastic
    continuous = cfg.continuous_sampling
    mode = cfg.juvenile_growth_mode[()]  # pyright: ignore[reportArgumentType]

    total_age_0 = float(ind_count[0, 0, :].sum() + ind_count[1, 0, :].sum())

    if mode == NO_COMPETITION:
        scaling = 1.0
    elif mode == FIXED:
        scaling = alg.compute_scaling_factor_fixed(total_age_0, cfg.carrying_capacity[()])  # pyright: ignore[reportArgumentType]
    else:
        if mode == LOGISTIC:
            scaling = alg.compute_scaling_factor_logistic(
                total_age_0,
                cfg.expected_competition_strength[()],  # pyright: ignore[reportArgumentType]
                cfg.expected_survival_rate[()],  # pyright: ignore[reportArgumentType]
                cfg.low_density_growth_rate[()],  # pyright: ignore[reportArgumentType]
            )
        else:
            scaling = alg.compute_scaling_factor_beverton_holt(
                total_age_0,
                cfg.expected_competition_strength[()],  # pyright: ignore[reportArgumentType]
                cfg.expected_survival_rate[()],  # pyright: ignore[reportArgumentType]
                cfg.low_density_growth_rate[()],  # pyright: ignore[reportArgumentType]
            )

    f_rec, m_rec = alg.recruit_juveniles_given_scaling_factor_sampling(
        (ind_count[0, 0, :], ind_count[1, 0, :]),
        scaling, n_ztypes,
        stochastic=stochastic, continuous_sampling=continuous,
    )

    s_f = cfg.female_age0_survival * cfg.viability_f
    s_m = cfg.male_age0_survival * cfg.viability_m

    if stochastic:
        if continuous:
            for k in range(n_ztypes):
                ind_count[0, 0, k] = nbc.continuous_binomial(f_rec[k], s_f[k])
                ind_count[1, 0, k] = nbc.continuous_binomial(m_rec[k], s_m[k])
        else:
            for k in range(n_ztypes):
                nf = int(round(f_rec[k]))
                nm = int(round(m_rec[k]))
                ind_count[0, 0, k] = float(nbc.binomial(nf, s_f[k])) if nf > 0 else 0.0  # pyright: ignore[reportUnknownArgumentType]
                ind_count[1, 0, k] = float(nbc.binomial(nm, s_m[k])) if nm > 0 else 0.0  # pyright: ignore[reportUnknownArgumentType]
    else:
        ind_count[0, 0, :] = f_rec * s_f
        ind_count[1, 0, :] = m_rec * s_m

    return ind_count


# ── Stage C: aging ───────────────────────────────────────────────────────────


@njit_switch(cache=True)
def run_discrete_aging(
    ind_count: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Shift age-0 juveniles to age-1 adults.  Old adults are discarded."""
    ind_count = ind_count.copy()
    ind_count[0, 1, :] = ind_count[0, 0, :]
    ind_count[0, 0, :] = 0.0
    ind_count[1, 1, :] = ind_count[1, 0, :]
    ind_count[1, 0, :] = 0.0
    return ind_count
