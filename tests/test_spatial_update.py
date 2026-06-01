"""Tests for SpatialPopulation.update() — runtime per-deme modification."""

import pytest

import natal as nt
from natal.spatial_builder import batch_setting


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
