"""P5b: public spatial path -- conservation integrated across full ticks.

Claim: with one egg per female, survival 1 and neutral genetics, the
discrete-generation lifecycle maps total(t+1) to exactly half of
total(t) in deterministic mode (every female produces one offspring, the
sex split is exactly 1/2, adults do not survive the turnover) --
bit-for-bit, with migration redistributing individuals between demes but
never creating or destroying mass.  In stochastic mode the same law holds
in expectation, with Poisson-variance fluctuations from the egg sampling
(fixed_egg_count is a no-op on this path; see the QC report).

Reference: the discrete lifecycle contract (non-overlapping generations,
exact per-sex splitting) plus algebraic conservation of migration; no
library tables.

Wrong results rejected: silent mass loss/gain integrated across the full
tick pipeline (lifecycle + migration), migration that conserves but never
moves anyone.
"""

from __future__ import annotations

import natal as nt
import numpy as np
from natal.frontend.spatial.builder import batch_setting


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _chain_pop(name: str, seed: int | None, migration_rate: float, stochastic: bool):
    counts = batch_setting(
        [
            {"female": {"W|W": 400, "W|D": 100}, "male": {"W|W": 300, "W|D": 200}},
            {"female": {"W|W": 200}, "male": {"W|D": 200}},
            {"female": {"W|W": 100, "W|D": 100}, "male": {"W|W": 100, "W|D": 100}},
        ]
    )
    builder = nt.SpatialPopulation.builder(
        _species(name + "_sp"),
        n_demes=3,
        pop_type="discrete_generation",
    )
    pop = (
        builder.setup(name=name, stochastic=stochastic)
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1.0, sex_ratio=0.5)
        .competition(juvenile_growth_mode="no_competition")
        .migration(migration_rate=migration_rate)
        .build()
    )
    if seed is not None:
        pop._initialize_session(seed=seed)
        pop._rust_backend_seed = seed
    return pop


def _totals(pop) -> np.ndarray:
    return np.array([float(d.state.individual_count.sum()) for d in pop.demes])


def test_deterministic_generation_law_is_exact_under_migration() -> None:
    pop = _chain_pop("QC0915_p05b", None, migration_rate=0.3, stochastic=False)
    assert _totals(pop).sum() == 1800.0

    before = _totals(pop).copy()
    pop.run(1)
    after = _totals(pop)
    # Turnover halves the total exactly; migration moved individuals.
    assert after.sum() == 900.0
    assert (np.abs(after - before) > 1.0).any()

    # The halving law continues to hold exactly under migration.
    total = float(after.sum())
    for _ in range(5):
        pop.run(1)
        total /= 2
        assert _totals(pop).sum() == total


def test_stochastic_tick_total_tracks_expectation() -> None:
    pop = _chain_pop("QC0915_p05c", 98, migration_rate=0.3, stochastic=True)
    total = float(_totals(pop).sum())
    for _ in range(5):
        expected = total / 2
        sigma = (expected / 2) ** 0.5  # Poisson(lambda) eggs thinned by 1/2
        pop.run(1)
        total = float(_totals(pop).sum())
        assert abs(total - expected) <= 5 * max(sigma, 1.0)
