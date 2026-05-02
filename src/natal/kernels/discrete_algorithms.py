"""Discrete-generation algorithms: mating allocation and fertilization.

These are pure algorithm functions specific to the non-overlapping generation
model — no sperm storage, no age iteration, no remating displacement.

Each function is decorated with ``@njit_switch`` so it compiles to native code
when Numba is available and falls back to pure Python otherwise.
"""

from typing import Annotated, Tuple

import numpy as np
from numpy.typing import NDArray

from natal import numba_compat as nbc
from natal.numba_utils import njit_switch

__all__ = [
    "mate_discrete",
    "fertilize_discrete",
]

EPS = 1e-10


@njit_switch(cache=True)
def mate_discrete(
    females: Annotated[NDArray[np.float64], "shape=(g,)"],
    mating_prob: Annotated[NDArray[np.float64], "shape=(g,g)"],
    p_mating: float,
    stochastic: bool,
    continuous: bool,
) -> Annotated[NDArray[np.float64], "shape=(g,g)"]:
    """Allocate mating pairs for one tick in a discrete-generation population.

    Discrete populations have no persistent sperm storage — each tick starts
    fresh.  For every female genotype *gf*:

    1. Sample how many females mate (Binomial if *stochastic*, else
       deterministic fraction).
    2. Distribute the mating females across male genotypes via *mating_prob*
       (Multinomial if *stochastic*, else proportional allocation).

    Args:
        females: Female counts per genotype, shape ``(g,)``.
        mating_prob: Row-normalized mating probability matrix, shape ``(g, g)``,
            where ``mating_prob[gf, gm]`` is the probability a mating female
            of genotype *gf* pairs with a male of genotype *gm*.
        p_mating: Probability a female mates in this tick (clamped to [0, 1]).
        stochastic: If True, use Binomial/Multinomial sampling.
        continuous: If True (with *stochastic*), use Beta/Dirichlet instead.

    Returns:
        Pair-count matrix with shape ``(g, g)``, where ``result[gf, gm]`` is
        the number of mated females of genotype *gf* paired with males of
        genotype *gm*.
    """
    g = int(females.shape[0])
    pm = nbc.clamp01(p_mating)
    pair_counts = np.zeros((g, g), dtype=np.float64)

    for gf in range(g):
        n_female = float(females[gf])
        if n_female <= 0.0:
            continue

        if stochastic:
            if continuous:
                n_mating = nbc.continuous_binomial(n_female, pm)
            else:
                n_int = max(0, int(round(n_female)))
                n_mating = float(nbc.binomial(n_int, pm)) if n_int > 0 else 0.0
        else:
            n_mating = n_female * pm

        if n_mating <= EPS:
            continue

        if stochastic:
            if continuous:
                tmp = np.zeros(g, dtype=np.float64)
                nbc.continuous_multinomial(n_mating, mating_prob[gf, :], tmp)
                for gm in range(g):
                    pair_counts[gf, gm] += tmp[gm]
            else:
                n_int = max(0, int(round(n_mating)))
                if n_int > 0:
                    draws = nbc.multinomial(n_int, mating_prob[gf, :])
                    for gm in range(g):
                        pair_counts[gf, gm] += float(draws[gm])
        else:
            for gm in range(g):
                pair_counts[gf, gm] += n_mating * mating_prob[gf, gm]

    return pair_counts


@njit_switch(cache=True)
def fertilize_discrete(
    pair_counts: Annotated[NDArray[np.float64], "shape=(g,g)"],
    offspring_tensor: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    fert_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fert_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    eggs_per_female: float,
    reproduction_rate: float,
    sex_ratio: float,
    has_sex_chromosomes: bool,
    female_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    stochastic: bool,
    continuous: bool,
) -> Tuple[
    Annotated[NDArray[np.float64], "shape=(g,)"],
    Annotated[NDArray[np.float64], "shape=(g,)"],
]:
    """Produce offspring from mating pairs for a discrete-generation population.

    Iterates over every (gf, gm) pair with non-zero *pair_counts*, computes
    the expected number of eggs, samples offspring genotypes from the
    precomputed *offspring_tensor*, and assigns sex.

    In the discrete model there is no age-specific variation in reproduction
    or fertility — every mating female has the same base reproduction rate
    and relative fertility (implicitly 1.0).

    Args:
        pair_counts: Mated female counts per genotype pair, shape ``(g, g)``.
        offspring_tensor: Precomputed probability tensor ``P[gf, gm, go]``,
            shape ``(g, g, g)``.
        fert_f: Female fecundity per genotype, shape ``(g,)``.
        fert_m: Male fecundity per genotype, shape ``(g,)``.
        eggs_per_female: Expected number of eggs per reproducing female.
        reproduction_rate: Fraction of mated females that actually reproduce
            (clamped to [0, 1]).
        sex_ratio: Global female offspring fraction (used when
            *has_sex_chromosomes* is False).
        has_sex_chromosomes: If True, sex is genotype-constrained.
        female_compat: Female compatibility weight per offspring genotype,
            shape ``(g,)``.
        male_compat: Male compatibility weight per offspring genotype,
            shape ``(g,)``.
        female_only: Mask of genotypes that can only be female, shape ``(g,)``.
        male_only: Mask of genotypes that can only be male, shape ``(g,)``.
        stochastic: If True, use Poisson/Binomial/Multinomial sampling.
        continuous: If True (with *stochastic*), use continuous distributions.

    Returns:
        Tuple ``(n_female, n_male)``, each with shape ``(g,)``, giving
        offspring counts per genotype.
    """
    g = int(fert_f.shape[0])
    sex_ratio_c = nbc.clamp01(sex_ratio)
    p_reproduce = nbc.clamp01(reproduction_rate)

    n_offspring = np.zeros(g, dtype=np.float64)
    p_norm = np.zeros(g, dtype=np.float64)
    tmp = np.zeros(g, dtype=np.float64)
    has_any = False

    for gf in range(g):
        ff = float(fert_f[gf])
        for gm in range(g):
            n_pairs = float(pair_counts[gf, gm])
            if n_pairs <= 0.0:
                continue
            has_any = True

            # Eggs per reproducing pair.
            # fertility_factor = 1.0 in discrete (no age-dependent fertility).
            eggs_per_pair = eggs_per_female * ff * fert_m[gm]

            if stochastic:
                n_pairs_eff = n_pairs if continuous else np.round(n_pairs)
                if n_pairs_eff <= 0.0:
                    continue

                n_reproducing = float(n_pairs_eff)
                if p_reproduce < 1.0 - EPS:
                    n_reproducing = (
                        nbc.continuous_binomial(n_pairs_eff, p_reproduce)
                        if continuous
                        else float(nbc.binomial(int(n_pairs_eff), p_reproduce))
                    )

                total_lambda = max(0.0, n_reproducing * eggs_per_pair)
                if continuous:
                    n_total = nbc.continuous_poisson(total_lambda)
                else:
                    n_total = float(np.random.poisson(total_lambda))
            else:
                n_reproducing = n_pairs * p_reproduce
                n_total = n_reproducing * eggs_per_pair

            if n_total <= EPS:
                continue

            # Fraction of zygotes that are viable (sum of tensor slice).
            p_surv = 0.0
            for go in range(g):
                p_surv += offspring_tensor[gf, gm, go]

            if stochastic:
                if p_surv <= EPS:
                    continue
                n_viable = (
                    n_total
                    if p_surv >= 1.0 - EPS
                    else (
                        nbc.continuous_binomial(n_total, p_surv)
                        if continuous
                        else float(nbc.binomial(int(round(n_total)), p_surv))
                    )
                )
                if n_viable <= EPS:
                    continue

                inv = 1.0 / p_surv
                for go in range(g):
                    p_norm[go] = offspring_tensor[gf, gm, go] * inv

                if continuous:
                    nbc.continuous_multinomial(n_viable, p_norm, tmp)
                    for go in range(g):
                        n_offspring[go] += tmp[go]
                else:
                    draws = nbc.multinomial(int(round(n_viable)), p_norm)
                    for go in range(g):
                        n_offspring[go] += float(draws[go])
            else:
                for go in range(g):
                    n_offspring[go] += n_total * offspring_tensor[gf, gm, go]

    if not has_any:
        return np.zeros(g, dtype=np.float64), np.zeros(g, dtype=np.float64)

    total = n_offspring.sum()
    if total <= EPS:
        return np.zeros(g, dtype=np.float64), np.zeros(g, dtype=np.float64)

    # Assign sex to each offspring genotype.
    n_f = np.zeros(g, dtype=np.float64)
    n_m = np.zeros(g, dtype=np.float64)

    for go in range(g):
        n_g = n_offspring[go]
        if n_g <= EPS:
            continue

        if has_sex_chromosomes and female_only[go]:
            n_f[go] = n_g
        elif has_sex_chromosomes and male_only[go]:
            n_m[go] = n_g
        else:
            p_f = sex_ratio_c
            if has_sex_chromosomes:
                denom = female_compat[go] + male_compat[go]
                p_f = (
                    nbc.clamp01(female_compat[go] / denom)
                    if denom > EPS
                    else 0.5
                )
            if stochastic:
                n_fem = (
                    nbc.continuous_binomial(n_g, p_f)
                    if continuous
                    else float(nbc.binomial(int(round(n_g)), p_f))
                )
            else:
                n_fem = n_g * p_f
            n_f[go] = n_fem
            n_m[go] = n_g - n_fem

    return n_f, n_m
