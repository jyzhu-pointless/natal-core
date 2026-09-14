"""Builder-level migration conservation regressions.

Adjacency rows are *relative outbound weights*: the builder normalizes every
non-empty row to a probability vector, so row-stochastic, sub-stochastic and
super-stochastic inputs all conserve mass.  "Wants less migration" is
expressed through ``migration_rate``.  These tests pin that contract on the
default topology path (which used to reach the engine with raw degree
weights) and on explicit hand-written CSRs, and cover the build-time
per-deme ``migration_rate`` forms.

Every conservation check compares against a no-migration twin built from the
identical lifecycle, so the assertion expresses conservation without
encoding any post-fix absolute value.  Deterministic lifecycle bookkeeping
carries ~1e-10 relative dust per tick, so the ratio bound is 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting
from natal.frontend.spatial.population import SpatialPopulation

# Row sums 0.5: the old adjacency bookkeeping dropped the unrouted 0.5.
ROW_SUBSTOCHASTIC = np.array(
    [[0.0, 0.5, 0.0], [0.25, 0.0, 0.25], [0.0, 0.5, 0.0]], dtype=np.float64
)
# Row sums 1: the consistent reference row-stochastic form.
ROW_STOCHASTIC = np.array(
    [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float64
)

RATE = 0.4


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _build(
    name: str,
    *,
    adjacency: np.ndarray | None = None,
    topology: nt.SquareGrid | None = None,
    migration_rate: object = RATE,
    stochastic: bool = False,
    seed: int | None = None,
) -> SpatialPopulation:
    """Build a 3-deme discrete chain with equal 500/500 seeds per deme."""
    counts = batch_setting(
        [{"female": {"W|W": 500}, "male": {"W|W": 500}}] * 3
    )
    builder = (
        nt.SpatialPopulation.builder(
            _species(name + "_sp"),
            n_demes=3,
            pop_type="discrete_generation",
            topology=topology,
        )
        .setup(name=name, stochastic=stochastic)
        .initial_state(individual_count=counts)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(
            carrying_capacity=1e12,
            low_density_growth_rate=2.0,
            juvenile_growth_mode="logistic",
        )
    )
    if adjacency is None:
        builder = builder.migration(migration_rate=migration_rate)
    else:
        builder = builder.migration(
            adjacency=adjacency, migration_rate=migration_rate, strategy="adjacency"
        )
    pop = builder.build()
    if seed is not None:
        pop._initialize_session(seed=seed)  # noqa: SLF001 - explicit seeding for stochastic twins
    return pop


def _totals(pop: SpatialPopulation) -> list[float]:
    return [float(deme.state.individual_count.sum()) for deme in pop.demes]


def _total(pop: SpatialPopulation) -> float:
    return float(sum(_totals(pop)))


def _adult_index(pop: SpatialPopulation) -> int:
    return int(pop.blueprint.new_adult_age)


class TestAdjacencyConservation:
    def test_default_topology_adjacency_conserves(self) -> None:
        """The builder's default topology adjacency is row-normalized.

        ``SquareGrid(1, 3)`` has row sums ``[1, 2, 1]``; used raw, those
        degree weights create mass at the middle deme.  Normalization makes
        the migrated run track its no-migration twin.
        """
        topology = nt.SquareGrid(1, 3)
        migrated = _build("cons_default", topology=topology, stochastic=False)
        twin = _build(
            "cons_default_twin", topology=topology, migration_rate=0.0,
            stochastic=False,
        )
        # Structure: every non-empty adjacency row is a probability vector.
        rows = migrated.migration_csr.indptr
        for src in range(migrated.n_demes):
            start, end = int(rows[src]), int(rows[src + 1])
            assert np.isclose(
                migrated.migration_csr.weights[start:end].sum(), 1.0
            )
        for _ in range(4):
            migrated.run(1)
            twin.run(1)
            assert _total(migrated) / _total(twin) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("adjacency", [ROW_SUBSTOCHASTIC, ROW_STOCHASTIC])
    def test_explicit_csr_conserves(self, adjacency: np.ndarray) -> None:
        """Sub- and row-stochastic adjacency inputs conserve identically.

        A row summing to 0.5 must not lose the unrouted half; the old
        adjacency bookkeeping produced a 0.8^t decay against the twin.
        """
        migrated = _build("cons_csr", adjacency=adjacency, stochastic=False)
        twin = _build("cons_csr_twin", migration_rate=0.0, stochastic=False)
        for _ in range(4):
            migrated.run(1)
            twin.run(1)
            assert _total(migrated) / _total(twin) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("stochastic", [False, True])
    def test_deterministic_and_stochastic_agree_on_conservation(
        self, stochastic: bool
    ) -> None:
        """The same CSR conserves in both engines under one seed.

        Tick 1 lifecycle draws precede any migration draw, so the
        migrated/twin comparison is exact for both branches; a CSR that
        conserves stochastically must not lose mass deterministically.
        """
        seed = 7
        migrated = _build(
            f"cons_branch_{stochastic}",
            adjacency=ROW_SUBSTOCHASTIC,
            stochastic=stochastic,
            seed=seed,
        )
        twin = _build(
            f"cons_branch_twin_{stochastic}",
            migration_rate=0.0,
            stochastic=stochastic,
            seed=seed,
        )
        migrated.run(1)
        twin.run(1)
        assert _total(migrated) == pytest.approx(_total(twin), rel=0.0, abs=1e-9)


class TestBuildTimePerDemeRate:
    def test_batch_setting_sets_each_deme(self) -> None:
        """``batch_setting`` gives one rate declaration per deme."""
        pop = _build(
            "per_deme_batch",
            topology=nt.SquareGrid(1, 3),
            migration_rate=batch_setting([0.1, 0.4, 0.1]),
        )
        adult = _adult_index(pop)
        rate = pop.params.migration_rate
        assert rate.shape == (3, 2, 2)
        for sex in range(2):
            assert np.allclose(
                rate[:, sex, adult], np.array([0.1, 0.4, 0.1])
            )
        # Structure: the middle deme's outbound quota is the largest.
        assert rate[1, :, adult].min() > rate[0, :, adult].max()

    def test_batch_elements_keep_their_sugar(self) -> None:
        """Every batch element is normalized with the homogeneous sugar rules."""
        per_sex = _build(
            "per_deme_sugar_dict",
            topology=nt.SquareGrid(1, 3),
            migration_rate=batch_setting(
                [{"F": 0.1, "M": 0.05}, {"F": 0.4, "M": 0.2}, {"F": 0.1, "M": 0.05}]
            ),
        )
        adult = _adult_index(per_sex)
        rate = per_sex.params.migration_rate
        assert np.allclose(rate[:, 0, adult], [0.1, 0.4, 0.1])
        assert np.allclose(rate[:, 1, adult], [0.05, 0.2, 0.05])

        # An explicit (n_ages,) element is used as-is for both sexes.
        per_age = _build(
            "per_deme_sugar_age",
            topology=nt.SquareGrid(1, 3),
            migration_rate=batch_setting(
                [[0.01, 0.1], [0.02, 0.4], [0.01, 0.1]]
            ),
        )
        age_rate = per_age.params.migration_rate
        assert np.allclose(age_rate[:, 0, :], [[0.01, 0.1], [0.02, 0.4], [0.01, 0.1]])
        assert np.allclose(age_rate[:, 0, :], age_rate[:, 1, :])

    def test_batch_setting_middle_deme_out_migrates_more(self) -> None:
        """The middle deme's larger quota shows up as a smaller resident share."""
        batched = _build(
            "per_deme_batch_dyn",
            topology=nt.SquareGrid(1, 3),
            migration_rate=batch_setting([0.1, 0.4, 0.1]),
        )
        uniform = _build(
            "per_deme_uniform_dyn",
            topology=nt.SquareGrid(1, 3),
            migration_rate=batch_setting([0.1, 0.1, 0.1]),
        )
        batched.run(1)
        uniform.run(1)
        assert _totals(batched)[1] < _totals(uniform)[1]

    def test_full_and_per_deme_age_columns(self) -> None:
        """The canonical ``(D, S, A)`` column and the ``(D, A)`` shortcut agree."""
        topology = nt.SquareGrid(1, 3)
        # Discrete chain here has n_ages=2 with the adult at index 1.
        adult = 1
        full = np.zeros((3, 2, 2), dtype=np.float64)
        full[:, :, adult] = np.array([[0.1], [0.4], [0.1]])
        per_deme_age = np.zeros((3, 2), dtype=np.float64)
        per_deme_age[:, adult] = np.array([0.1, 0.4, 0.1])

        from_full = _build(
            "per_deme_full", topology=topology, migration_rate=full
        )
        from_age = _build(
            "per_deme_age", topology=topology, migration_rate=per_deme_age
        )
        assert np.allclose(from_full.params.migration_rate, full)
        # The (D, A) shortcut broadcasts the age vector across both sexes.
        assert np.allclose(from_age.params.migration_rate, full)
