"""WF regulation uses the same pre-survival juvenile pool as staged ticks."""

import numpy as np
import pytest

import natal as nt


def _population(mode: str, speed: int, survival: tuple[float, float], fitness: bool = False) -> nt.DiscreteGenerationPopulation:
    species = nt.Species.from_dict(
        name=f"wf_density_{mode}_{speed}_{survival}_{fitness}",
        structure={"chr1": {"A": ["W", "D"]}},
    )
    builder = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False, extreme_speed_mode=speed)
        .initial_state({"female": {"W|D": 3000}, "male": {"W|W": 3000}})
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .survival(female_age0_survival=survival[0], male_age0_survival=survival[1])
        .competition(juvenile_growth_mode=mode, carrying_capacity=5000, low_density_growth_rate=2)
    )
    if fitness:
        builder = builder.fitness(
            zygote_viability={"W|D": {"female": 0.6, "male": 0.8}},
            viability={"W|D": {"female": 0.4, "male": 0.7}},
        )
    return builder.build()


@pytest.mark.parametrize("mode", ["no_competition", "fixed", "logistic", "beverton_holt", "ricker"])
@pytest.mark.parametrize("survival", [(1.0, 1.0), (0.8, 0.8), (0.6, 0.9)])
@pytest.mark.parametrize("fitness", [False, True])
def test_wf_matches_staged_regulation_before_survival(
    mode: str, survival: tuple[float, float], fitness: bool,
) -> None:
    """Sex/genotype viability follows density regulation; zygote viability precedes it."""
    staged = _population(mode, 0, survival, fitness)
    wf = _population(mode, 3, survival, fitness)
    for _ in range(4):
        staged.run(1)
        wf.run(1)
        # Different aggregation orders can round differently; this tolerance
        # admits float64 roundoff over four ticks, not the former 25% bias.
        np.testing.assert_allclose(
            wf.state.individual_count, staged.state.individual_count, rtol=1e-12, atol=1e-9,
        )


def test_wf_beverton_holt_matches_independent_recurrence() -> None:
    """Neutral equal-survival calibration gives N'=N*r/(1+(r-1)*N/K)."""
    pop = _population("beverton_holt", 3, (0.8, 0.8))
    expected = 6000.0
    for _ in range(30):
        expected = expected * 2 / (1 + expected / 5000)
        pop.run(1)
        assert pop.state.individual_count.sum() == pytest.approx(expected, rel=1e-12)
    assert pop.state.individual_count.sum() == pytest.approx(5000, abs=1e-5)
