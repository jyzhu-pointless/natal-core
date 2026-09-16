"""Behavioral checks for fixed egg counts in discrete generations."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _population(
    name: str,
    *,
    eggs: float,
    fixed: bool,
    continuous: bool,
    reproduction_rate: float = 1.0,
    stochastic: bool = True,
) -> nt.DiscreteGenerationPopulation:
    """Build a one-genotype population with no downstream count changes."""
    species = nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["W"]}},
        gamete_labels=["default"],
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous,
        )
        .initial_state(
            individual_count={"female": {"W|W": 100.0}, "male": {"W|W": 100.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(
            eggs_per_female=eggs,
            fixed_egg_count=fixed,
            female_adult_mating_rate=1.0,
            male_adult_mating_rate=1.0,
            sex_ratio=0.5,
        )
        .competition(
            juvenile_growth_mode="no_competition",
            carrying_capacity=1e12,
            low_density_growth_rate=1.0,
        )
        .build()
    )
    if reproduction_rate != 1.0:
        # Discrete populations expose the adult reproduction slot through the
        # route-table alias ``reproduction_rate`` at runtime.
        population.params.reproduction_rate = reproduction_rate
    return population


def _offspring_total(population: nt.DiscreteGenerationPopulation, seed: int) -> float:
    population._initialize_session(seed=seed)
    population._rust_backend_seed = seed
    population.run(1)
    return float(np.asarray(population.state.individual_count[:, 1, :]).sum())


@pytest.mark.parametrize("continuous", [False, True])
def test_fixed_egg_count_removes_clutch_noise_in_both_sampling_modes(
    continuous: bool,
) -> None:
    """With p(reproduce)=1, fixed eggs conserve the independently derived total."""
    totals = [
        _offspring_total(
            _population(
                f"fixed_{continuous}_{seed}",
                eggs=1.0,
                fixed=True,
                continuous=continuous,
            ),
            seed,
        )
        for seed in (1, 2, 3, 4, 5)
    ]
    # 100 females x 1 egg, with no viability or survival loss.
    np.testing.assert_allclose(totals, [100.0] * len(totals), rtol=0.0, atol=1e-10)


def test_fixed_egg_count_rounds_discrete_expected_clutch() -> None:
    """Discrete fixed egg counts use integer rounding after pair thinning."""
    population = _population(
        "fixed_rounding",
        eggs=1.605,
        fixed=True,
        continuous=False,
    )

    assert _offspring_total(population, 11) == 161.0


def test_fixed_egg_count_keeps_reproduction_rate_thinning() -> None:
    """Fixed eggs retain binomial reproduction thinning before clutch sizing."""
    totals = np.array(
        [
            _offspring_total(
                _population(
                    f"fixed_reproduction_thinning_{seed}",
                    eggs=2.0,
                    fixed=True,
                    continuous=False,
                    reproduction_rate=0.5,
                ),
                seed,
            )
            for seed in range(1, 257)
        ]
    )
    # X = 2 * Binomial(100, .5): E[X]=100 and SD(sample mean)=sqrt(100/256).
    six_sigma = 6.0 * np.sqrt(100.0 / len(totals))
    assert abs(float(totals.mean()) - 100.0) < six_sigma
    # Var(2*Binomial(100,.5))=100. The fourth central moment is
    # 16*(3*25**2 + 25*(1-6*.25))=29800, giving the exact variance
    # of the unbiased sample variance below. This rejects deterministic
    # thinning, which has the right mean but zero variance.
    n = len(totals)
    variance_sd = np.sqrt((29800 - (n - 3) / (n - 1) * 100**2) / n)
    assert abs(float(totals.var(ddof=1)) - 100) < 6 * variance_sd


def test_default_discrete_egg_count_has_poisson_mean() -> None:
    """Across independent seeds, default clutch totals have Poisson mean 2000."""
    totals = np.array(
        [
            _offspring_total(
                _population(
                    f"poisson_{seed}",
                    eggs=20.0,
                    fixed=False,
                    continuous=False,
                ),
                seed,
            )
            for seed in range(1, 65)
        ]
    )
    # The reference is Poisson(lambda=100*20).  For N independent runs,
    # SD(sample mean)=sqrt(lambda/N), and SD(sample variance) is approximated
    # by lambda*sqrt(2/(N-1)); use a six-sigma bound for each statistic.
    sample_mean_sd = np.sqrt(2000.0 / len(totals))
    sample_variance_sd = 2000.0 * np.sqrt(2.0 / (len(totals) - 1))
    assert abs(float(totals.mean()) - 2000.0) < 6.0 * sample_mean_sd
    assert abs(float(totals.var(ddof=1)) - 2000.0) < 6.0 * sample_variance_sd


def test_fixed_continuous_clutch_preserves_fractional_count() -> None:
    """Continuous sampling must not round 100 mothers times 1.605 eggs."""
    pop = _population("continuous_fractional_fixed", eggs=1.605, fixed=True, continuous=True)
    assert _offspring_total(pop, 7) == pytest.approx(160.5, abs=1e-10)
