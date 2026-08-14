"""MGDrivE1-compatible patch lifecycle used by matched benchmarks.

This internal NATAL Core kernel follows the daily event order from MGDrivE1
1.6.2. It operates over a leading deme axis so the same implementation serves
single-patch and spatial benchmark scenarios.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    # Numba does not publish complete type stubs for its public API.
    from numba import (  # pyright: ignore[reportMissingTypeStubs] -- Numba has no complete type stubs.
        prange,
    )
except ImportError:
    # Python fallback preserves prange's integer-iterator contract.
    prange = range  # type: ignore[assignment] -- range is the non-Numba prange fallback.

from natal.numba import compat as nbc
from natal.numba.utils import njit_switch

FloatArray = NDArray[np.float64]

__all__ = [
    "advance_mgdrive1_lifecycle",
]


@njit_switch(cache=True)
def _sample_multinomial(
    n: int,
    probabilities: FloatArray,
) -> NDArray[np.int64]:
    """Sample one multinomial vector using conditional binomials.

    Args:
        n: Total number of trials.
        probabilities: Category probabilities summing to one.

    Returns:
        Integer category counts summing to ``n``.
    """
    result = np.zeros(probabilities.size, dtype=np.int64)
    remaining = n
    probability_left = 1.0
    for category in range(probabilities.size - 1):
        if remaining <= 0:
            break
        conditional = (
            probabilities[category] / probability_left
            if probability_left > 0.0
            else 0.0
        )
        conditional = min(1.0, max(0.0, conditional))
        drawn = nbc.fast_binomial(remaining, conditional)
        result[category] = drawn
        remaining -= drawn
        probability_left -= probabilities[category]
    result[-1] = remaining
    return result


@njit_switch(cache=True, parallel=True)
def advance_mgdrive1_lifecycle(
    aquatic_input: FloatArray,
    adult_male_input: FloatArray,
    adult_female_input: FloatArray,
    unmated_female_input: FloatArray,
    time_egg: int,
    time_larva: int,
    time_pupa: int,
    beta: float,
    adult_mortality: float,
    aquatic_mortality: float,
    alpha: float,
    inheritance: FloatArray,
    mating_fitness: FloatArray,
    female_fraction: FloatArray,
    adult_survival_modifier: FloatArray,
    female_emergence: FloatArray,
    male_emergence: FloatArray,
    fertility_modifier: FloatArray,
    stochastic: bool,
    release_deme: int,
    release_adult_male: FloatArray,
    release_unmated_female: FloatArray,
    release_adult_female: FloatArray,
    release_eggs: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Advance every deme through one MGDrivE1-compatible day.

    Adult releases are applied after emergence and before mating. Egg releases
    are applied after oviposition. A negative ``release_deme`` disables all
    release arrays for the current day.

    Args:
        aquatic_input: Deme-by-genotype-by-aquatic-day counts.
        adult_male_input: Deme-by-genotype adult male counts.
        adult_female_input: Deme-by-female-by-mate mated counts.
        unmated_female_input: Deme-by-genotype unmated female counts.
        time_egg: Number of daily egg cohorts.
        time_larva: Number of daily larval cohorts.
        time_pupa: Number of daily pupal cohorts.
        beta: Wild-type eggs laid per mated female per day.
        adult_mortality: Daily adult mortality probability.
        aquatic_mortality: Daily density-independent aquatic mortality.
        alpha: Larval density-dependence centering parameter.
        inheritance: Female-by-male-by-offspring inheritance probabilities.
        mating_fitness: Female-by-male mating weights.
        female_fraction: Genotype-specific female fraction at emergence.
        adult_survival_modifier: Genotype-specific adult survival multiplier.
        female_emergence: Genotype-specific female emergence success.
        male_emergence: Genotype-specific male emergence success.
        fertility_modifier: Female-genotype fertility multiplier.
        stochastic: Whether to sample demographic events.
        release_deme: Target deme, or a negative value for no release.
        release_adult_male: Released adult males by genotype.
        release_unmated_female: Released unmated females by genotype.
        release_adult_female: Released mated females by parental genotype.
        release_eggs: Released eggs by genotype.

    Returns:
        Aquatic, adult male, mated female, and unmated female arrays after one
        local lifecycle day.
    """
    aquatic = aquatic_input.copy()
    adult_male = adult_male_input.copy()
    adult_female = adult_female_input.copy()
    unmated_female = unmated_female_input.copy()
    n_demes = aquatic.shape[0]
    n_genotypes = aquatic.shape[1]
    aquatic_survival = 1.0 - aquatic_mortality
    adult_survival = (
        (1.0 - adult_mortality) * adult_survival_modifier
    )

    larva_start = time_egg
    larva_end = time_egg + time_larva
    pupa_start = larva_end
    pupa_end = time_egg + time_larva + time_pupa

    # Each deme is independent until the caller runs migration. Keeping one
    # complete daily lifecycle in each prange lane avoids shared scratch state.
    for deme in prange(n_demes):
        for genotype in range(n_genotypes):
            if stochastic:
                adult_male[deme, genotype] = nbc.fast_binomial(
                    int(round(adult_male_input[deme, genotype])),
                    adult_survival[genotype],
                )
            else:
                adult_male[deme, genotype] *= adult_survival[genotype]
            for mate_genotype in range(n_genotypes):
                if stochastic:
                    adult_female[deme, genotype, mate_genotype] = (
                        nbc.fast_binomial(
                            int(
                                round(
                                    adult_female_input[
                                        deme,
                                        genotype,
                                        mate_genotype,
                                    ]
                                )
                            ),
                            adult_survival[genotype],
                        )
                    )
                else:
                    adult_female[deme, genotype, mate_genotype] *= (
                        adult_survival[genotype]
                    )

        emerging = np.zeros(n_genotypes, dtype=np.float64)
        for genotype in range(n_genotypes):
            final_pupa = aquatic_input[deme, genotype, pupa_end - 1]
            if stochastic:
                emerging[genotype] = nbc.fast_binomial(
                    int(round(final_pupa)),
                    aquatic_survival,
                )
            else:
                emerging[genotype] = final_pupa * aquatic_survival
        for age in range(pupa_end - 2, pupa_start - 1, -1):
            for genotype in range(n_genotypes):
                count = aquatic_input[deme, genotype, age]
                if stochastic:
                    aquatic[deme, genotype, age + 1] = nbc.fast_binomial(
                        int(round(count)),
                        aquatic_survival,
                    )
                else:
                    aquatic[deme, genotype, age + 1] = (
                        count * aquatic_survival
                    )

        larval_total = 0.0
        for genotype in range(n_genotypes):
            for age in range(larva_start, larva_end):
                larval_total += aquatic_input[deme, genotype, age]
        density_survival = (
            alpha / (alpha + larval_total)
        ) ** (1.0 / time_larva)
        larval_survival = density_survival * aquatic_survival
        for age in range(larva_end - 1, larva_start - 1, -1):
            for genotype in range(n_genotypes):
                count = aquatic_input[deme, genotype, age]
                if stochastic:
                    aquatic[deme, genotype, age + 1] = nbc.fast_binomial(
                        int(round(count)),
                        larval_survival,
                    )
                else:
                    aquatic[deme, genotype, age + 1] = (
                        count * larval_survival
                    )

        for age in range(time_egg - 1, -1, -1):
            for genotype in range(n_genotypes):
                count = aquatic_input[deme, genotype, age]
                if stochastic:
                    aquatic[deme, genotype, age + 1] = nbc.fast_binomial(
                        int(round(count)),
                        aquatic_survival,
                    )
                else:
                    aquatic[deme, genotype, age + 1] = (
                        count * aquatic_survival
                    )

        for genotype in range(n_genotypes):
            if stochastic:
                surviving = nbc.fast_binomial(
                    int(round(emerging[genotype])),
                    1.0 - adult_mortality,
                )
                female_pupae = nbc.fast_binomial(
                    surviving,
                    female_fraction[genotype],
                )
                adult_male[deme, genotype] += nbc.fast_binomial(
                    surviving - female_pupae,
                    male_emergence[genotype],
                )
                unmated_female[deme, genotype] += nbc.fast_binomial(
                    female_pupae,
                    female_emergence[genotype],
                )
            else:
                surviving_float = (
                    emerging[genotype]
                    * (1.0 - adult_mortality)
                )
                adult_male[deme, genotype] += (
                    surviving_float
                    * (1.0 - female_fraction[genotype])
                    * male_emergence[genotype]
                )
                unmated_female[deme, genotype] += (
                    surviving_float
                    * female_fraction[genotype]
                    * female_emergence[genotype]
                )

        if release_deme == deme:
            for genotype in range(n_genotypes):
                adult_male[deme, genotype] += release_adult_male[genotype]
                unmated_female[deme, genotype] += (
                    release_unmated_female[genotype]
                )
                for mate_genotype in range(n_genotypes):
                    adult_female[
                        deme,
                        genotype,
                        mate_genotype,
                    ] += release_adult_female[genotype, mate_genotype]

        probabilities = np.zeros(n_genotypes, dtype=np.float64)
        for female_genotype in range(n_genotypes):
            unmated = unmated_female[deme, female_genotype]
            if unmated <= 0.0:
                continue
            weight_sum = 0.0
            for male_genotype in range(n_genotypes):
                weight = (
                    adult_male[deme, male_genotype]
                    * mating_fitness[female_genotype, male_genotype]
                )
                probabilities[male_genotype] = weight
                weight_sum += weight
            if weight_sum > 0.0:
                for male_genotype in range(n_genotypes):
                    probabilities[male_genotype] /= weight_sum
                if stochastic:
                    draws = _sample_multinomial(
                        int(round(unmated)),
                        probabilities,
                    )
                    for male_genotype in range(n_genotypes):
                        adult_female[
                            deme,
                            female_genotype,
                            male_genotype,
                        ] += draws[male_genotype]
                else:
                    for male_genotype in range(n_genotypes):
                        adult_female[
                            deme,
                            female_genotype,
                            male_genotype,
                        ] += (
                            unmated * probabilities[male_genotype]
                        )
                unmated_female[deme, female_genotype] = 0.0
            elif stochastic:
                unmated_female[deme, female_genotype] = nbc.fast_binomial(
                    int(round(unmated)),
                    adult_survival[female_genotype],
                )
            else:
                unmated_female[deme, female_genotype] *= adult_survival[
                    female_genotype
                ]

        for offspring_genotype in range(n_genotypes):
            expected_eggs = 0.0
            for female_genotype in range(n_genotypes):
                for male_genotype in range(n_genotypes):
                    expected_eggs += (
                        adult_female[
                            deme,
                            female_genotype,
                            male_genotype,
                        ]
                        * beta
                        * fertility_modifier[female_genotype]
                        * inheritance[
                            female_genotype,
                            male_genotype,
                            offspring_genotype,
                        ]
                    )
            if stochastic:
                aquatic[deme, offspring_genotype, 0] = np.random.poisson(
                    expected_eggs
                )
            else:
                aquatic[deme, offspring_genotype, 0] = expected_eggs
            if release_deme == deme:
                aquatic[deme, offspring_genotype, 0] += release_eggs[
                    offspring_genotype
                ]

    return aquatic, adult_male, adult_female, unmated_female
