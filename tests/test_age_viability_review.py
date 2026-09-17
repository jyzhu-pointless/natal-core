"""Regression contracts for explicitly age-indexed survival fitness."""

import numpy as np
import pytest

import natal as nt


def _builder(stochastic: bool = False) -> nt.PopulationBuilder:
    """Isolate existing cohorts from births, mating, and density regulation."""
    species = nt.Species.from_dict("age_viability_review", {"c": {"l": ["A"]}})
    return (
        nt.AgeStructuredPopulation.setup(species, stochastic=stochastic)
        .age_structure(n_ages=5, new_adult_age=3)
        .initial_state(
            individual_count={sex: {"A|A": {1: 100, 2: 100, 3: 100}} for sex in ["female", "male"]},
            sperm_storage={"A|A": {"A|A": {3: 100}}},
        )
        .survival(female_age_based_survival=[1, 1, 1, .5, 0], male_age_based_survival=[1, 1, 1, .5, 0])
        .reproduction(eggs_per_female=0, female_age_based_mating_rate=[0]*5, male_age_based_mating_rate=[0]*5)
        .competition(juvenile_growth_mode="no_competition")
    )


def test_explicit_juvenile_and_adult_viability_combines_with_survival_and_sperm() -> None:
    """Each cohort uses its own sex/age fitness; stored sperm follows females."""
    pop = _builder().fitness(viability={"A|A": {
        "female": {1: .25, 2: .5, 3: .75}, "male": {1: .75, 2: .25, 3: .5},
    }}).build()
    pop.run(1)
    # Initial 100 times fitness, with the adult cohort also multiplied by .5.
    np.testing.assert_array_equal(pop.state.individual_count[:, 2:5, 0], [[25, 50, 37.5], [75, 25, 25]])
    assert pop.state.sperm_storage[4, 0, 0] == 37.5


@pytest.mark.parametrize("stochastic", [False, True])
def test_zero_viability_at_nondefault_ages_kills_both_sexes(stochastic: bool) -> None:
    """Probability zero gives exact extinction, including the stochastic path."""
    pop = _builder(stochastic).fitness(viability={"A|A": {1: 0, 3: 0}}).build()
    pop.run(1)
    np.testing.assert_array_equal(pop.state.individual_count[:, 2:5, 0], [[0, 100, 0], [0, 100, 0]])
    assert pop.state.sperm_storage.sum() == 0


def test_scalar_viability_keeps_single_default_age_application() -> None:
    """An omitted age remains the last juvenile age and does not recur in adults."""
    pop = _builder().fitness(viability={"A|A": .5}).build()
    pop.run(1)
    np.testing.assert_array_equal(pop.state.individual_count[:, 2:5, 0], [[100, 50, 50], [100, 50, 50]])
    pop.run(1)
    np.testing.assert_array_equal(pop.state.individual_count[:, 3:5, 0], [[50, 25], [50, 25]])


def test_runtime_explicit_adult_viability_reaches_backend() -> None:
    """A live fitness update has the same nondefault-age semantics as build."""
    pop = _builder().build()
    pop.update().fitness(viability={"A|A": {3: .5}})
    pop.run(1)
    np.testing.assert_array_equal(pop.state.individual_count[:, 4, 0], [25, 25])
