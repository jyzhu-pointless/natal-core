"""Tests for SpatialPopulation.update() — runtime per-deme modification."""

import pytest

import natal as nt
from natal.spatial_configurator import batch_setting


@pytest.fixture(scope="module")
def species():
    return nt.Species.from_dict(
        name="__test_spatial_update__",
        structure={"auto": {"A": ["WT"]}},
    )


@pytest.fixture
def homogeneous_pop(species):
    """2×2 homogeneous spatial population — all demes share the same config."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="discrete_generation")
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build()
    )


@pytest.fixture
def homogeneous_age_pop(species):
    """Age-structured variant for tests that need per-age parameters."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="age_structured")
        .setup(name="test", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": {1: 100}}, "male": {"WT|WT": {1: 100}},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                     juvenile_growth_mode="concave")
        .build()
    )


@pytest.fixture
def heterogeneous_pop(species):
    """2×2 spatial population with per-deme K via batch_setting."""
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=topo, pop_type="discrete_generation")
        .setup(name="test", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(
            carrying_capacity=batch_setting([500, 600, 700, 800]),
            low_density_growth_rate=6.0,
            juvenile_growth_mode="concave",
        )
        .build()
    )


class TestSpatialUpdateAllDemes:
    """pop.update() without deme argument — modifies all demes."""

    def test_homogeneous_scalar_applied_to_all(self, homogeneous_pop):
        """Scalar update on homogeneous pop changes all demes' shared K."""
        pop = homogeneous_pop
        pop.update().competition(carrying_capacity=300)
        for i in range(4):
            assert pop.deme(i).config.carrying_capacity[()] == 300

    def test_heterogeneous_scalar_applied_to_all_unique_configs(self, heterogeneous_pop):
        """Scalar update on heterogeneous pop modifies every unique config."""
        pop = heterogeneous_pop
        pop.update().competition(low_density_growth_rate=3.0)
        for i in range(4):
            assert pop.deme(i).config.low_density_growth_rate[()] == 3.0
        # K should be unchanged
        expected_k = [500, 600, 700, 800]
        for i, ek in enumerate(expected_k):
            assert pop.deme(i).config.carrying_capacity[()] == ek


class TestSpatialUpdateSingleDeme:
    """pop.update(deme=N) — per-deme modification with clone-on-write."""

    def test_single_deme_change_does_not_affect_others(self, homogeneous_pop):
        """Modifying deme 2 should not affect deme 0."""
        pop = homogeneous_pop
        pop.update(deme=2).competition(carrying_capacity=100)
        assert pop.deme(2).config.carrying_capacity[()] == 100
        assert pop.deme(0).config.carrying_capacity[()] == 500

    def test_clone_on_write_isolates_array(self, homogeneous_pop):
        """After clone-on-write, the modified deme has a private array."""
        pop = homogeneous_pop
        k_before = pop.deme(0).config.carrying_capacity
        pop.update(deme=1).competition(carrying_capacity=999)
        # deme 0 array unchanged (same object)
        assert pop.deme(0).config.carrying_capacity is k_before
        # deme 1 has a different array object
        assert pop.deme(1).config.carrying_capacity is not k_before
        assert pop.deme(1).config.carrying_capacity[()] == 999

    def test_multiple_deme_updates_isolated(self, homogeneous_pop):
        """Each singled-out deme gets its own private config."""
        pop = homogeneous_pop
        pop.update(deme=0).competition(carrying_capacity=100)
        pop.update(deme=2).competition(carrying_capacity=300)
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(1).config.carrying_capacity[()] == 500  # untouched
        assert pop.deme(2).config.carrying_capacity[()] == 300
        assert pop.deme(3).config.carrying_capacity[()] == 500  # untouched

    def test_single_deme_heterogeneous(self, heterogeneous_pop):
        """Modifying one deme in a heterogeneous population."""
        pop = heterogeneous_pop
        # deme 1 originally K=600
        pop.update(deme=1).competition(carrying_capacity=999)
        assert pop.deme(1).config.carrying_capacity[()] == 999
        assert pop.deme(0).config.carrying_capacity[()] == 500
        assert pop.deme(2).config.carrying_capacity[()] == 700


class TestSpatialUpdateBatch:
    """pop.update() with batch_setting."""

    def test_batch_across_demes(self, homogeneous_pop):
        """Batch K values applied per-deme."""
        pop = homogeneous_pop
        pop.update().competition(
            carrying_capacity=batch_setting([100, 200, 300, 400])
        )
        expected = [100, 200, 300, 400]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.carrying_capacity[()] == ek

    def test_batch_with_none_skips_deme(self, homogeneous_pop):
        """None values skip the corresponding deme."""
        pop = homogeneous_pop
        original_k = pop.deme(1).config.carrying_capacity[()].copy()
        pop.update().competition(
            carrying_capacity=batch_setting([100, None, 300, None])
        )
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(1).config.carrying_capacity[()] == original_k  # skipped
        assert pop.deme(2).config.carrying_capacity[()] == 300
        assert pop.deme(3).config.carrying_capacity[()] == original_k  # skipped

    def test_batch_mixed_with_scalar(self, homogeneous_pop):
        """Batch K with scalar r applied everywhere."""
        pop = homogeneous_pop
        pop.update().competition(
            carrying_capacity=batch_setting([100, 200, 300, 400]),
            low_density_growth_rate=2.0,
        )
        for i in range(4):
            assert pop.deme(i).config.low_density_growth_rate[()] == 2.0
        assert pop.deme(0).config.carrying_capacity[()] == 100
        assert pop.deme(3).config.carrying_capacity[()] == 400


class TestSpatialUpdateCustom:
    """pop.update() with .custom() fields."""

    def test_custom_field_on_all_demes(self, homogeneous_pop):
        """Custom field update applies to all demes."""
        pop = homogeneous_pop
        pop.update().custom(temperature=35.0)
        for i in range(4):
            assert float(pop.deme(i).config.custom['temperature'][()]) == 35.0

    def test_custom_field_single_deme(self, homogeneous_pop):
        """Custom field on a single deme after clone-on-write."""
        pop = homogeneous_pop
        pop.update(deme=3).custom(temperature=99.0)
        assert float(pop.deme(3).config.custom['temperature'][()]) == 99.0


# ══════════════════════════════════════════════════════════════════════════
# _SpatialUpdate: survival / reproduction / setup / fitness on all demes
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialUpdateSurvivalReproduction:
    """Verify that .update().survival() and .reproduction() work on spatial
    populations, not just .competition() and .custom()."""

    def test_survival_scalar_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().survival(female_age0_survival=...) on all demes."""
        pop = homogeneous_pop
        pop.update().survival(female_age0_survival=0.6, male_age0_survival=0.4)
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.female_age0_survival == pytest.approx(0.6)
            assert cfg.male_age0_survival == pytest.approx(0.4)

    def test_survival_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).survival(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=1).survival(female_age0_survival=0.3)
        assert pop.deme(1).config.female_age0_survival == pytest.approx(0.3)

    def test_reproduction_scalar_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().reproduction(eggs_per_female=..., sex_ratio=...)."""
        pop = homogeneous_pop
        pop.update().reproduction(eggs_per_female=100, sex_ratio=0.7)
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.eggs_per_female[()] == 100.0
            assert cfg.sex_ratio[()] == 0.7

    def test_reproduction_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).reproduction(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=2).reproduction(eggs_per_female=200)
        assert pop.deme(2).config.eggs_per_female[()] == 200.0
        assert pop.deme(0).config.eggs_per_female[()] == 10.0  # unchanged

    def test_setup_applies_to_all_demes(self, homogeneous_pop):
        """pop.update().setup(stochastic=...) on all demes."""
        pop = homogeneous_pop
        pop.update().setup(stochastic=True)
        for i in range(4):
            assert pop.deme(i).config.stochastic is True

    def test_setup_single_deme(self, homogeneous_pop):
        """pop.update(deme=N).setup(...) on a single deme."""
        pop = homogeneous_pop
        pop.update(deme=3).setup(stochastic=True)
        assert pop.deme(3).config.stochastic is True
        assert pop.deme(0).config.stochastic is False  # unchanged

    def test_combined_reproduction_and_survival(self, homogeneous_pop):
        """Chaining .reproduction() and .survival() in one update call."""
        pop = homogeneous_pop
        pop.update().reproduction(eggs_per_female=50).survival(
            female_age0_survival=0.5
        )
        for i in range(4):
            cfg = pop.deme(i).config
            assert cfg.eggs_per_female[()] == 50.0
            assert cfg.female_age0_survival == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════
# _dispatch_scalar: non-scalar kwarg handling
# ══════════════════════════════════════════════════════════════════════════


class TestDispatchScalar:
    """Verify _dispatch_scalar delegates to Configurator methods correctly.

    After the refactor, _dispatch_scalar calls the full Configurator method
    (e.g. cfg.survival(...)).  For discrete models, this means age-structured
    params (list/dict for per-age rates) correctly raise TypeError instead of
    being silently dropped.
    """

    def test_survival_rejects_per_age_list_on_discrete(self, homogeneous_pop):
        """Discrete model rejects per-age survival list — use age-structured."""
        pop = homogeneous_pop
        with pytest.raises(TypeError):
            pop.update().survival(female=[0.5, 0.6])

    def test_reproduction_rejects_per_age_dict_on_discrete(self, homogeneous_pop):
        """Discrete model rejects per-age mating dict — use age-structured."""
        pop = homogeneous_pop
        with pytest.raises(TypeError):
            pop.update().reproduction(
                female_age_based_mating_rate={0: 0.5, 1: 0.8}
            )

    def test_batch_reproduction_on_all_demes(self, homogeneous_pop):
        """Batch reproduction params via batch_setting on all demes."""
        pop = homogeneous_pop
        pop.update().reproduction(
            eggs_per_female=batch_setting([50, 60, 70, 80])
        )
        expected = [50, 60, 70, 80]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.eggs_per_female[()] == ek

    def test_batch_survival_on_all_demes(self, homogeneous_pop):
        """Batch survival params via batch_setting on all demes.

        Each batch write goes through update_deme() + clone-on-write +
        _replace(), creating per-deme private configs with correct scalars.
        """
        pop = homogeneous_pop
        pop.update().survival(
            female_age0_survival=batch_setting([0.5, 0.6, 0.7, 0.8])
        )
        expected = [0.5, 0.6, 0.7, 0.8]
        for i, ek in enumerate(expected):
            assert pop.deme(i).config.female_age0_survival == pytest.approx(ek)


# ══════════════════════════════════════════════════════════════════════════
# Spatial update → run cycle
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialUpdateRunCycle:
    """Verify that per-deme config changes affect simulation output."""

    def test_update_k_changes_population_size(self, homogeneous_pop):
        """Changing K via update() should affect the population after a run."""
        pop = homogeneous_pop
        # Reduce K drastically
        pop.update().competition(carrying_capacity=50, low_density_growth_rate=2.0)
        pop.run(5)
        # With K=50, the population should be significantly reduced
        total = pop.get_total_count()
        # Population should be below original K=2000 (4 demes × 500)
        assert total < 2000, \
            f"Population {total} should be reduced after K dropped to 50"

    def test_single_deme_update_affects_only_that_deme(self, homogeneous_pop):
        """After updating a single deme's K, that deme should shrink."""
        pop = homogeneous_pop
        # Drastically reduce K for deme 1 only
        pop.update(deme=1).competition(carrying_capacity=10, low_density_growth_rate=2.0)
        pop.run(5)
        # Deme 0 (unchanged, K=500) should be larger than deme 1 (K=10)
        assert pop.deme(0).get_total_count() > pop.deme(1).get_total_count(), \
            "Deme 0 (K=500) should have more individuals than Deme 1 (K=10)"
