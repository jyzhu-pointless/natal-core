"""Competition weights default to all ones, on every construction path.

``age_based_relative_competition_strength`` weights the juvenile age classes
when density regulation measures competition.  Age 0 is fixed at 1.0, and the
second juvenile age class is user-settable through ``competition_strength``
(``parameters.jsonc`` writes ``config_path=[1]``).  The spatial builder used
to inject 5.0 at that index by default while every other entry point left it
at 1.0, so the same configuration carried different competition weights
depending on which builder produced it.  The default is now uniform: an
unset ``competition_strength`` means "no special weighting" everywhere.

The argument also used to be a silent no-op when the model has no second
juvenile age (age 0 is the only juvenile age, and its weight is fixed at
1.0).  That combination is now rejected, so the parameter's documented scope
matches its effect.

The positive case — an explicit weight landing on age 1 — is covered by
``test_population_builder_carrying_capacity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting


def _species(name: str) -> nt.Species:
    """Return a one-locus species with a single default gamete label."""
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _plain_age(name: str, *, n_ages: int, new_adult_age: int) -> nt.AgeStructuredPopulation:
    """Build a non-spatial age-structured population with default competition."""
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(name + "_sp"), name=name, stochastic=False
        )
        .age_structure(n_ages=n_ages, new_adult_age=new_adult_age)
        .initial_state(
            individual_count={
                "female": {"W|W": {new_adult_age: 100}},
                "male": {"W|W": {new_adult_age: 100}},
            }
        )
        .survival(
            female_age_based_survival=[1.0] * n_ages,
            male_age_based_survival=[1.0] * n_ages,
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0)
        .build()
    )


def _spatial_age(name: str, *, n_ages: int, new_adult_age: int) -> nt.SpatialPopulation:
    """Build a two-deme spatial age-structured population."""
    counts = batch_setting(
        [
            {"female": {"W|W": {new_adult_age: 100}}, "male": {"W|W": {new_adult_age: 100}}}
            for _ in range(2)
        ]
    )
    return (
        nt.SpatialPopulation.builder(
            _species(name + "_sp"),
            n_demes=2,
            topology=nt.SquareGrid(1, 2),
            pop_type="age_structured",
        )
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=n_ages, new_adult_age=new_adult_age)
        .initial_state(individual_count=counts)
        .survival(
            female_age_based_survival=[1.0] * n_ages,
            male_age_based_survival=[1.0] * n_ages,
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0)
        .build()
    )


def _weight_vector(pop: object) -> np.ndarray:
    """Read the competition weights from a plain or spatial population."""
    if hasattr(pop, "config"):
        return np.asarray(pop.config.age_based_relative_competition_strength)  # type: ignore[attr-defined]
    draft = pop._export_deme_drafts()[0]  # type: ignore[attr-defined]  # spatial: per-deme draft
    return np.asarray(draft.age_based_relative_competition_strength)


@pytest.mark.parametrize("n_ages,new_adult_age", [(2, 1), (4, 2)])
def test_plain_age_structured_default_weights_are_ones(n_ages: int, new_adult_age: int) -> None:
    """An unset competition_strength leaves every juvenile weight at 1.0."""
    pop = _plain_age(f"cw_plain_{new_adult_age}", n_ages=n_ages, new_adult_age=new_adult_age)
    assert _weight_vector(pop) == pytest.approx(np.ones(n_ages))


@pytest.mark.parametrize("n_ages,new_adult_age", [(2, 1), (4, 2)])
def test_spatial_age_structured_default_weights_are_ones(n_ages: int, new_adult_age: int) -> None:
    """The spatial builder no longer injects a 5.0 weight for age 1."""
    pop = _spatial_age(f"cw_spatial_{new_adult_age}", n_ages=n_ages, new_adult_age=new_adult_age)
    assert _weight_vector(pop) == pytest.approx(np.ones(n_ages))


def test_weight_is_rejected_when_there_is_no_second_juvenile_age() -> None:
    """Passing the weight where it cannot take effect is an error, not a no-op.

    ``competition_strength`` writes ``age_based_relative_competition_strength[1]``,
    and with ``new_adult_age == 1`` age 0 is the only juvenile age, so the
    value would never reach the kernels.
    """
    species = _species("cw_reject_sp")
    builder = (
        nt.DiscreteGenerationPopulation.setup(species=species, name="cw_reject", stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 100}, "male": {"W|W": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
    )
    with pytest.raises(ValueError, match="no effect when new_adult_age is 1"):
        builder.competition(carrying_capacity=1000.0, competition_strength=5.0).build()

    # The same guard covers a two-age age-structured model, whose only
    # juvenile age is also age 0.
    age_builder = (
        nt.AgeStructuredPopulation.setup(
            species=_species("cw_reject_age_sp"), name="cw_reject_age", stochastic=False
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"W|W": {1: 100}}, "male": {"W|W": {1: 100}}}
        )
        .survival(
            female_age_based_survival=[1.0, 1.0],
            male_age_based_survival=[1.0, 1.0],
        )
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
    )
    with pytest.raises(ValueError, match="no effect when new_adult_age is 1"):
        age_builder.competition(
            carrying_capacity=1000.0, competition_strength=5.0
        ).build()
