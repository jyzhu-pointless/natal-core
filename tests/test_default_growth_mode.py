"""Default growth mode: Beverton-Holt for every engine.

Omitting ``growth_mode`` used to mean NO_COMPETITION for discrete populations
(unbounded growth) and LOGISTIC for age-structured ones (a hard ``max(0, .)``
clamp that oscillates at high r).  Identical builder chains therefore behaved
qualitatively differently, and the discrete default let any model that forgot
the knob grow without bound.

All three entry points — age-structured, discrete generation and spatial
discrete — now default to BEVERTON_HOLT, the monotone compensatory curve
``g(x) = r / (1 + (r - 1) x)``.  It is positive for every finite x (so it
never clamps the cohort to zero) and ``|g'(1)| < 1`` for ``r > 1``, so a
population above the carrying capacity converges instead of swinging.

These tests pin that default at each entry point and the behavior it buys.
The curve identities themselves are covered by the density-regulation unit
tests; here the point is that the *default* is regulated and stable.
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting


def _species(name: str) -> nt.Species:
    """Return a one-locus species with a single default gamete label."""
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _discrete(name: str, **competition: object) -> nt.DiscreteGenerationPopulation:
    """Build a discrete-generation population, omitting the mode by default."""
    species = _species(name + "_sp")
    return (
        nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
        .initial_state(individual_count={"female": {"W|W": 500}, "male": {"W|W": 500}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .competition(
            carrying_capacity=1000.0, low_density_growth_rate=6.0, **competition
        )
        .build()
    )


def _age_structured(name: str, **competition: object) -> nt.AgeStructuredPopulation:
    """Build a three-age population, omitting the mode by default."""
    species = _species(name + "_sp")
    return (
        nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"W|W": 900}, "male": {"W|W": 900}}
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 1.0],
            male_age_based_survival=[1.0, 1.0, 1.0],
        )
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .competition(
            carrying_capacity=1000.0, low_density_growth_rate=6.0, **competition
        )
        .build()
    )


def _spatial_discrete(name: str) -> nt.SpatialPopulation:
    """Build a three-deme discrete spatial population with default settings."""
    species = _species(name + "_sp")
    counts = batch_setting(
        [{"female": {"W|W": 100}, "male": {"W|W": 100}} for _ in range(3)]
    )
    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=3,
            topology=nt.SquareGrid(1, 3),
            pop_type="discrete_generation",
        )
        .setup(name=name, stochastic=False)
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0, low_density_growth_rate=2.0)
        .build()
    )


def test_every_engine_defaults_to_beverton_holt() -> None:
    """Omitting growth_mode selects BEVERTON_HOLT, never NO_COMPETITION."""
    assert _discrete("default_discrete").params.growth_mode == nt.BEVERTON_HOLT
    assert _age_structured("default_age").params.growth_mode == nt.BEVERTON_HOLT

    spatial = _spatial_discrete("default_spatial")
    modes = spatial.params.growth_mode
    assert all(mode == nt.BEVERTON_HOLT for mode in modes), modes


def test_discrete_default_regulates_instead_of_growing_without_bound() -> None:
    """The default discrete engine caps growth at K; NO_COMPETITION does not."""
    regulated = _discrete("discrete_regulated")
    unregulated = _discrete("discrete_unregulated", growth_mode="no_competition")

    for _ in range(10):
        regulated.run(1)
        unregulated.run(1)

    assert float(regulated.state.individual_count.sum()) == pytest.approx(
        1000.0, rel=0.02
    )
    assert float(unregulated.state.individual_count.sum()) > 1.0e6


def test_age_structured_default_converges_instead_of_oscillating() -> None:
    """The default age-structured engine settles; explicit LOGISTIC cycles."""
    default = _age_structured("age_default")
    logistic = _age_structured("age_logistic", growth_mode="logistic")

    for _ in range(25):
        default.run(1)
        logistic.run(1)

    # Beverton-Holt is unconditionally stable: the last steps are numerically
    # flat and the census sits at the 2*K equilibrium of this 3-age layout.
    stable_tail = []
    for _ in range(3):
        default.run(1)
        stable_tail.append(float(default.state.individual_count.sum()))
    assert stable_tail == pytest.approx([2000.0, 2000.0, 2000.0], rel=1e-9)

    # The logistic curve keeps a large period-3 swing at r = 6.
    logistic_tail = []
    for _ in range(3):
        logistic.run(1)
        logistic_tail.append(float(logistic.state.individual_count.sum()))
    relative_swing = max(logistic_tail) / min(logistic_tail)
    assert relative_swing > 1.5, logistic_tail
