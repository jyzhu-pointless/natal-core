"""R4-08: spatial migration coupled to per-deme density regulation.

Contracts under attack (rust ``spatial.rs`` stage order — the per-deme
lifecycle runs first, then migration consumes the same per-deme streams):

1. Equal carrying capacities with a symmetric mixing matrix: the uniform
   vector is a fixed point of both the migration operator and every deme's
   Beverton-Holt regulation, so each deme must sit *exactly* at its own K.
2. No migration: every deme sits exactly at its own K (unequal K's).
3. Source-sink with a one-way edge 0 -> 1 at rate m and ``K_1 = 0``
   (a zero-habitat deme, the D2 contract).  Reading the state after
   migration, the source solves
       N_0 = (1 - m) * (N_0 / 2) * E * g(N_0 / K) * s*  = (1 - m) * N_0 * g(x)
   with ``g`` the Beverton-Holt curve and ``s* E / 2 = 1``, i.e.
       N_0 = K (r (1 - m) - 1) / (r - 1),   N_1 = m N_0 / (1 - m),
   and the sink holds exactly the freshly arrived migrants, which the next
   tick's zero-equilibrium regulation kills (no accumulation, no growth).
4. Unequal K's with migration: every deme stays finite and within a sane
   band of the largest K (a D3-style silent mass creation would blow past
   it within a few hundred ticks).
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting
from natal.frontend.spatial.population import SpatialPopulation

TICKS = 300


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W"]}}, gamete_labels=["default"]
    )


def _build(
    name: str,
    *,
    n_demes: int,
    capacities: list[float],
    counts: list[dict[str, dict[str, float]]],
    adjacency: np.ndarray,
    migration_rate: float,
    growth_rate: float = 3.0,
) -> SpatialPopulation:
    builder = (
        SpatialPopulation.builder(
            _species(f"{name}_sp"), n_demes=n_demes, pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=False)
        .initial_state(individual_count=batch_setting(counts))
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(
            carrying_capacity=batch_setting(capacities),
            low_density_growth_rate=growth_rate,
            juvenile_growth_mode="beverton_holt",
        )
    )
    if migration_rate > 0.0:
        builder = builder.migration(
            adjacency=adjacency, migration_rate=migration_rate, strategy="adjacency"
        )
    return builder.build()


def _deme_totals(pop: SpatialPopulation) -> list[float]:
    return [float(d.state.individual_count.sum()) for d in pop.demes]


CHAIN3 = np.array(
    [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=float
)
ALL_TO_ALL3 = np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=float
)


class TestEqualCapacitiesWithMixing:
    def test_uniform_vector_is_the_fixed_point(self) -> None:
        counts = [{"female": {"W|W": 500.0}, "male": {"W|W": 500.0}}] * 3
        pop = _build(
            "equal_k",
            n_demes=3,
            capacities=[2000.0, 2000.0, 2000.0],
            counts=counts,
            adjacency=ALL_TO_ALL3,
            migration_rate=0.3,
        )
        pop.run(TICKS)
        for deme, total in enumerate(_deme_totals(pop)):
            assert total == pytest.approx(2000.0, rel=1e-9), (deme, total)


class TestNoMigration:
    def test_each_deme_sits_at_its_own_k(self) -> None:
        counts = [{"female": {"W|W": 500.0}, "male": {"W|W": 500.0}}] * 3
        pop = _build(
            "no_migration",
            n_demes=3,
            capacities=[1000.0, 2000.0, 3000.0],
            counts=counts,
            adjacency=ALL_TO_ALL3,
            migration_rate=0.0,
        )
        pop.run(TICKS)
        assert _deme_totals(pop) == pytest.approx([1000.0, 2000.0, 3000.0], rel=1e-9)


class TestSourceSink:
    ONE_WAY = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)

    @pytest.mark.parametrize(("migration_rate", "growth_rate"), [(0.2, 3.0), (0.1, 6.0)])
    def test_source_and_sink_sizes_follow_the_closed_form(
        self, migration_rate: float, growth_rate: float
    ) -> None:
        k = 2000.0
        counts = [
            {"female": {"W|W": 500.0}, "male": {"W|W": 500.0}},
            {"female": {"W|W": 0.0}, "male": {"W|W": 0.0}},
        ]
        pop = _build(
            f"sink_{migration_rate}_{growth_rate}".replace(".", "p"),
            n_demes=2,
            capacities=[k, 0.0],
            counts=counts,
            adjacency=self.ONE_WAY,
            migration_rate=migration_rate,
            growth_rate=growth_rate,
        )
        pop.run(TICKS)
        source, sink = _deme_totals(pop)
        r, m = growth_rate, migration_rate
        expected_source = k * (r * (1.0 - m) - 1.0) / (r - 1.0)
        expected_sink = m * expected_source / (1.0 - m)
        assert source == pytest.approx(expected_source, rel=1e-9)
        assert sink == pytest.approx(expected_sink, rel=1e-9)

    def test_sink_does_not_accumulate(self) -> None:
        """The zero-habitat deme never exceeds the one-tick migrant intake."""
        k = 2000.0
        counts = [
            {"female": {"W|W": 500.0}, "male": {"W|W": 500.0}},
            {"female": {"W|W": 0.0}, "male": {"W|W": 0.0}},
        ]
        pop = _build(
            "sink_accum",
            n_demes=2,
            capacities=[k, 0.0],
            counts=counts,
            adjacency=self.ONE_WAY,
            migration_rate=0.2,
        )
        samples = []
        for _ in range(40):
            pop.run(10)
            samples.append(_deme_totals(pop)[1])
        assert all(value <= 400.0 + 1e-9 for value in samples), samples
        assert samples[-1] == pytest.approx(samples[-2], rel=1e-9)


class TestUnequalCapacitiesStayBounded:
    def test_no_deme_explodes(self) -> None:
        """Migration across unequal K's must not create mass (D3 regression)."""
        counts = [{"female": {"W|W": 500.0}, "male": {"W|W": 500.0}}] * 3
        pop = _build(
            "unequal_k",
            n_demes=3,
            capacities=[1000.0, 2000.0, 3000.0],
            counts=counts,
            adjacency=ALL_TO_ALL3,
            migration_rate=0.3,
        )
        pop.run(TICKS)
        totals = _deme_totals(pop)
        assert all(np.isfinite(totals))
        assert all(total <= 3000.0 + 1e-6 for total in totals), totals
        assert sum(totals) <= 6000.0 * 1.01, totals

    def test_chain_migration_also_stays_bounded(self) -> None:
        counts = [{"female": {"W|W": 500.0}, "male": {"W|W": 500.0}}] * 3
        pop = _build(
            "chain_k",
            n_demes=3,
            capacities=[1000.0, 2000.0, 3000.0],
            counts=counts,
            adjacency=CHAIN3,
            migration_rate=0.4,
        )
        pop.run(TICKS)
        totals = _deme_totals(pop)
        assert all(np.isfinite(totals))
        assert all(total <= 3000.0 + 1e-6 for total in totals), totals
