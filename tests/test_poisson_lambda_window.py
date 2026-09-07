"""Regression for the discrete Poisson lambda ceiling (S2 residual).

rand_distr's ``Poisson::new`` rejects lambdas above ``Poisson::MAX_LAMBDA``
(1.844e19) with ``ShapeTooLarge``; the Rust helper's old guard only kicked
in at the 2^104 resolution limit, leaving that window to panic a discrete
stochastic run outright (``PanicException``).  The helper now returns the
mean at the library ceiling, and these tests pin the run-level contract: a
census whose per-pair egg count falls inside the window must complete and
stay finite.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.backends.rust.rust_backend import rust_backend_available

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_window_population(name: str, seed: int) -> nt.DiscreteGenerationPopulation:
    """Return a population whose tick-0 egg lambda sits in the bad window.

    Counts are float64 state, so an astronomically large initial census
    needs no per-individual allocation; multiply it by the per-pair egg
    count and the tick's ``total_lambda`` lands above rand_distr's Poisson
    ceiling while staying below the 2^104 resolution limit.
    """
    return (
        nt.DiscreteGenerationPopulation.setup(species=_species(name), stochastic=True)
        .initial_state(
            individual_count={"female": {"WT|WT": 2e18}, "male": {"WT|WT": 2e18}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
        .build()
    ).enable_rust_backend(seed=seed)


@pytest.mark.parametrize("seed", [0, 7])
def test_discrete_stochastic_window_lambda_completes(seed: int) -> None:
    """A census inside the Poisson ceiling window must not panic the run."""
    pop = _build_window_population("window", seed=seed)
    pop.run_tick()
    state = pop.state
    total = float(np.asarray(state.individual_count).sum())
    assert state.n_tick == 1
    assert np.isfinite(total)
    # The poisoned poisson used to panic; with the guard the tick completes
    # and the census carries the i64-saturated floor (~2^63) of the average
    # egg total, far beyond any plausible census.
    assert 1e18 < total < 1e20
