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
from natal.engine.simulation.age_structured import (
    BEVERTON_HOLT,
    FIXED,
    LOGISTIC,
    compute_scaling_factor_beverton_holt,
    compute_scaling_factor_fixed,
    compute_scaling_factor_logistic,
)
from natal.numba_utils import njit_switch

__all__ = [
    "mate_discrete",
    "fertilize_discrete",
    "run_wf_tick",
    "run_wf_loop",
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


# ---------------------------------------------------------------------------
# Wright-Fisher extreme-speed mode
# ---------------------------------------------------------------------------
# Replaces the mate→fertilize→survive pipeline with a single expected-
# offspring computation followed by one multinomial draw per tick.  Models
# effective population size dynamics rather than census dynamics.
#
# Mode constants (stored in DiscretePopulationConfig.extreme_speed_mode):
#   0 — off (standard sequential pipeline)
#   1 — MULTINOMIAL  (classic Wright-Fisher single draw)
#   2 — POISSON      (independent Poisson per genotype, large-N approx)
#   3 — DETERMINISTIC (infinite-population limit, no randomness)

_WF_MULTINOMIAL = 1
_WF_POISSON = 2
_WF_DETERMINISTIC = 3


@njit_switch(cache=True)
def run_wf_tick(
    ind_count: Annotated[NDArray[np.float64], "shape=(2,2,g)"],
    offspring_tensor: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    fecundity_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fecundity_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    sexual_selection: Annotated[NDArray[np.float64], "shape=(g,g)"],
    viability_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    viability_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    eggs_per_female: float,
    sex_ratio: float,
    female_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    has_sex_chromosomes: bool,
    mode: int,
    stochastic: bool,
    mating_rate_f: float = 1.0,
    mating_rate_m: float = 1.0,
    reproduction_rate: float = 1.0,
    carrying_capacity: float = 1e18,
    juvenile_growth_mode: int = 0,
    low_density_growth_rate: float = 1.0,
    expected_competition_strength: float = 1.0,
    expected_survival_rate: float = 1.0,
) -> Annotated[NDArray[np.float64], "shape=(2,2,g)"]:
    """Single Wright-Fisher tick for discrete-generation populations.

    Computes the expected offspring genotype distribution from adult
    frequencies and the precomputed *offspring_tensor*, then produces
    the next generation via a single sampling step.

    Args:
        ind_count: ``(2, 2, g)`` — [sex, age, genotype].  Age 1 = adults,
            age 0 = offspring (filled by this function).
        offspring_tensor: Precomputed ``P[gf, gm, go]``, shape ``(g, g, g)``.
        fecundity_f, fecundity_m: Per-genotype fecundity, shape ``(g,)``.
        sexual_selection: Row-normalised mating preference, shape ``(g, g)``.
        viability_f, viability_m: Age-0 survival fitness, shape ``(g,)``.
        eggs_per_female: Expected eggs per reproducing female.
        sex_ratio: Global female fraction at birth (unused when
            *has_sex_chromosomes* is True).
        female_compat, male_compat: Per-genotype sex compatibility,
            shape ``(g,)``.
        female_only, male_only: Genotypes restricted to one sex,
            shape ``(g,)``.
        has_sex_chromosomes: If True, sex is genotype-constrained.
        mode: 1=MULTINOMIAL, 2=POISSON, 3=DETERMINISTIC.
        stochastic: If False, skip all randomness (deterministic path).

    Returns:
        Updated *ind_count* with new offspring at age 0 and adults zeroed.

        .. note::
            The *continuous_sampling* config flag is ignored in WF mode —
            offspring are always drawn via multinomial / Poisson / deterministic
            integer counts.
    """
    # Validate mode early — before the O(g³) expected-offspring computation.
    if mode not in (_WF_DETERMINISTIC, _WF_MULTINOMIAL, _WF_POISSON):
        raise ValueError(
            f"Unrecognised extreme_speed_mode={mode}. "
            f"Expected 1 (MULTINOMIAL), 2 (POISSON), or 3 (DETERMINISTIC)."
        )

    g = int(offspring_tensor.shape[0])
    adult_f = ind_count[0, 1, :]
    adult_m = ind_count[1, 1, :]

    # Effective male count per genotype: raw count × mating rate.
    # Fecundity is applied separately (after pair allocation) to match
    # the standard path.
    effective_m = adult_m * mating_rate_m

    # ---- expected offspring distributions ----
    expected_f = np.zeros(g, dtype=np.float64)
    expected_m = np.zeros(g, dtype=np.float64)

    for gf in range(g):
        nf = adult_f[gf] * fecundity_f[gf] * mating_rate_f
        if nf <= 0.0:
            continue

        # Row-normalise by sexual-selection-weighted effective male count.
        row_sum = np.dot(sexual_selection[gf, :], effective_m)
        if row_sum <= 0.0:
            continue

        for gm in range(g):
            nm_eff = effective_m[gm]
            if nm_eff <= 0.0:
                continue
            pair_weight = nf * (nm_eff / row_sum) * sexual_selection[gf, gm]
            if pair_weight <= 0.0:
                continue
            for go in range(g):
                prob = offspring_tensor[gf, gm, go]
                if prob <= 0.0:
                    continue
                offspring = pair_weight * prob * eggs_per_female * reproduction_rate * fecundity_m[gm]
                if has_sex_chromosomes:
                    if female_only[go]:
                        expected_f[go] += offspring
                    elif male_only[go]:
                        expected_m[go] += offspring
                    else:
                        expected_f[go] += offspring * female_compat[go]
                        expected_m[go] += offspring * male_compat[go]
                else:
                    expected_f[go] += offspring * sex_ratio
                    expected_m[go] += offspring * (1.0 - sex_ratio)

    # ---- viability scaling ----
    for go in range(g):
        expected_f[go] *= viability_f[go]
        expected_m[go] *= viability_m[go]

    # ---- density-dependent competition ----
    # Compute scaling factor from total expected offspring.  Uses the same
    # growth-mode constants and scaling formulas as the standard path
    # (discrete_generation_simulator.run_discrete_survival).
    if juvenile_growth_mode > 0:  # 0 = NO_COMPETITION
        total_expected = expected_f.sum() + expected_m.sum()
        # In discrete-generation models only age-0 juveniles compete.
        # actual_competition_strength is just the total juvenile count.
        actual_competition = total_expected

        if juvenile_growth_mode == FIXED:
            sf = compute_scaling_factor_fixed(
                total_expected, carrying_capacity,
            )
        elif juvenile_growth_mode == LOGISTIC:
            sf = compute_scaling_factor_logistic(
                actual_competition, expected_competition_strength,
                expected_survival_rate, low_density_growth_rate,
            )
        elif juvenile_growth_mode == BEVERTON_HOLT:
            sf = compute_scaling_factor_beverton_holt(
                actual_competition, expected_competition_strength,
                expected_survival_rate, low_density_growth_rate,
            )
        else:
            sf = 1.0  # unknown mode — no scaling

        for go in range(g):
            expected_f[go] *= sf
            expected_m[go] *= sf

    # ---- sampling ----
    new_f = np.zeros(g, dtype=np.float64)
    new_m = np.zeros(g, dtype=np.float64)

    if mode == _WF_DETERMINISTIC or not stochastic:
        new_f[:] = expected_f
        new_m[:] = expected_m
    elif mode == _WF_MULTINOMIAL:
        total_expected = expected_f.sum() + expected_m.sum()
        if total_expected > 0:
            probs = np.empty(2 * g, dtype=np.float64)
            probs[:g] = expected_f
            probs[g:] = expected_m
            probs /= total_expected
            n_total = int(round(total_expected))
            if n_total > 0:
                draws = nbc.multinomial(n_total, probs)
                for go in range(g):
                    new_f[go] = float(draws[go])
                    new_m[go] = float(draws[g + go])
    elif mode == _WF_POISSON:
        for go in range(g):
            ef = max(0.0, expected_f[go])
            em = max(0.0, expected_m[go])
            new_f[go] = float(np.random.poisson(ef))
            new_m[go] = float(np.random.poisson(em))

    # ---- aging: offspring become next generation's adults ----
    # Discrete-generation: new offspring replace the previous adults
    # directly.  Age 1 holds the reproducing adults; age 0 is cleared
    # after each tick (non-overlapping generations).
    ind_count[:, 0, :] = 0.0       # clear old offspring slot
    ind_count[0, 1, :] = new_f     # new females → adult
    ind_count[1, 1, :] = new_m     # new males → adult
    return ind_count


# ---------------------------------------------------------------------------
# Convenience: single-call WF loop (competition included here in engine)
# ---------------------------------------------------------------------------


@njit_switch(cache=True)
def run_wf_loop(
    ind_count: Annotated[NDArray[np.float64], "shape=(2,2,g)"],
    n_ticks: int,
    offspring_tensor: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    fecundity_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fecundity_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    sexual_selection: Annotated[NDArray[np.float64], "shape=(g,g)"],
    viability_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    viability_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    eggs_per_female: float,
    sex_ratio: float,
    female_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_compat: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only: Annotated[NDArray[np.bool_], "shape=(g,)"],
    has_sex_chromosomes: bool,
    mode: int,
    stochastic: bool,
    mating_rate_f: float,
    mating_rate_m: float,
    reproduction_rate: float,
    carrying_capacity: float,
    juvenile_growth_mode: int,
    low_density_growth_rate: float,
    expected_competition_strength: float,
    expected_survival_rate: float,
) -> Annotated[NDArray[np.float64], "shape=(2,2,g)"]:
    """Run *n_ticks* WF ticks and return the final individual count."""
    for _ in range(n_ticks):
        ind_count = run_wf_tick(
            ind_count=ind_count,
            offspring_tensor=offspring_tensor,
            fecundity_f=fecundity_f,
            fecundity_m=fecundity_m,
            sexual_selection=sexual_selection,
            viability_f=viability_f,
            viability_m=viability_m,
            eggs_per_female=eggs_per_female,
            sex_ratio=sex_ratio,
            female_compat=female_compat,
            male_compat=male_compat,
            female_only=female_only,
            male_only=male_only,
            has_sex_chromosomes=has_sex_chromosomes,
            mode=mode,
            stochastic=stochastic,
            mating_rate_f=mating_rate_f,
            mating_rate_m=mating_rate_m,
            reproduction_rate=reproduction_rate,
            carrying_capacity=carrying_capacity,
            juvenile_growth_mode=juvenile_growth_mode,
            low_density_growth_rate=low_density_growth_rate,
            expected_competition_strength=expected_competition_strength,
            expected_survival_rate=expected_survival_rate,
        )
    return ind_count
