"""A zero equilibrium competition strength must extinguish, not deregulate.

``C*`` is the reference competition strength the compensatory curves are
evaluated against; it comes from the carrying capacity, or from a declared
equilibrium distribution when one is given.  When ``C*`` was zero, the ratio
guard substituted 1.0 — the neutral point of every compensatory curve — and
the survival guard substituted 1.0 as well, so their product was exactly 1.0:
regulation silently switched off and a population with no habitat grew without
bound (1000 adults became 3,125,000 in five ticks).  Modes 2..=4 now return a
zero scaling there, which is what FIXED mode already did.

FIXED and NO_COMPETITION are checked alongside as the two reference points:
one caps at zero habitat, the other is the only mode that is *supposed* to
leave recruitment unregulated.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt

_COMPENSATORY = ("logistic", "beverton_holt", "ricker")


def _species(name: str) -> nt.Species:
    """Return a one-locus species with a single default gamete label."""
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _discrete(name: str, *, mode: str, capacity: float, equilibrium=None):
    """Build a discrete population that would explode without regulation."""
    competition = dict(
        carrying_capacity=capacity,
        low_density_growth_rate=6.0,
        growth_mode=mode,
    )
    if equilibrium is not None:
        competition["equilibrium_distribution"] = equilibrium
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name + "_sp"), name=name, stochastic=False
        )
        .initial_state(individual_count={"female": {"W|W": 500}, "male": {"W|W": 500}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10, sex_ratio=0.5)
        .competition(**competition)
        .build()
    )


@pytest.mark.parametrize("mode", _COMPENSATORY)
def test_zero_capacity_extinguishes_for_every_compensatory_mode(mode: str) -> None:
    """No habitat under a compensatory curve must drive extinction."""
    pop = _discrete(f"zero_k_{mode}", mode=mode, capacity=0.0)
    for _ in range(5):
        pop.run(1)
    assert float(pop.state.individual_count.sum()) == 0.0


def test_zero_capacity_extinguishes_for_fixed_mode() -> None:
    """FIXED already capped at ``min(1, K / N)`` and must keep doing so."""
    pop = _discrete("zero_k_fixed", mode="fixed", capacity=0.0)
    for _ in range(5):
        pop.run(1)
    assert float(pop.state.individual_count.sum()) == 0.0


def test_no_competition_still_grows_without_regulation() -> None:
    """NO_COMPETITION remains the one mode that intentionally regulates nothing."""
    pop = _discrete("zero_k_none", mode="no_competition", capacity=0.0)
    for _ in range(5):
        pop.run(1)
    assert float(pop.state.individual_count.sum()) > 1.0e6


def test_tiny_capacity_does_not_explode() -> None:
    """The cliff used to sit exactly at K = 0; a tiny K must stay extinct."""
    pop = _discrete("tiny_k", mode="beverton_holt", capacity=1e-9)
    for _ in range(5):
        pop.run(1)
    assert float(pop.state.individual_count.sum()) == pytest.approx(0.0, abs=1e-6)


def test_declared_equilibrium_without_reproducing_females_does_not_explode() -> None:
    """A positive K cannot save a declared equilibrium that produces no eggs.

    The declared distribution replaces the derived one, so an all-zero table
    makes ``C*`` zero even though K is positive — the same guard must apply.
    """
    pop = _discrete(
        "declared_empty",
        mode="beverton_holt",
        capacity=1000.0,
        equilibrium=np.zeros((2, 2), dtype=np.float64),
    )
    for _ in range(5):
        pop.run(1)
    assert float(pop.state.individual_count.sum()) == 0.0
