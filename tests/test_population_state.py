"""Tests for PopulationState and DiscretePopulationState containers."""

from __future__ import annotations

import numpy as np
import pytest

from natal.population_state import (
    DiscretePopulationState,
    PopulationState,
    parse_flattened_discrete_state,
    parse_flattened_state,
)

# ═══════════════════════════════════════════════════════════════════════════
# PopulationState.create
# ═══════════════════════════════════════════════════════════════════════════


class TestPopulationStateCreate:
    def test_default_creation(self):
        state = PopulationState.create(n_genotypes=3)
        assert state.n_tick == 0
        assert state.individual_count.shape == (2, 2, 3)
        assert state.individual_count.dtype == np.float64
        assert (state.individual_count == 0).all()
        assert state.sperm_storage.shape == (2, 3, 3)
        assert (state.sperm_storage == 0).all()

    def test_custom_dimensions(self):
        state = PopulationState.create(n_genotypes=5, n_sexes=3, n_ages=4, n_tick=10)
        assert state.n_tick == 10
        assert state.individual_count.shape == (3, 4, 5)
        assert state.sperm_storage.shape == (4, 5, 5)
        assert (state.individual_count == 0).all()
        assert (state.sperm_storage == 0).all()

    def test_provided_individual_count(self):
        arr = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        state = PopulationState.create(n_genotypes=3, n_sexes=1, n_ages=2, individual_count=arr)
        assert state.individual_count.shape == (1, 2, 3)
        assert state.individual_count[0, 0, 0] == 1.0
        assert state.individual_count[0, 1, 2] == 6.0
        assert state.individual_count.dtype == np.float64
        # verify it's a copy (not the same object)
        assert state.individual_count is not arr

    def test_provided_individual_count_int_array(self):
        """Integer input array is cast to float64."""
        arr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
        state = PopulationState.create(n_genotypes=3, n_sexes=1, n_ages=2, individual_count=arr)
        assert state.individual_count.dtype == np.float64
        assert state.individual_count[0, 0, 0] == 1.0

    def test_provided_sperm_storage(self):
        sperm = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        state = PopulationState.create(n_genotypes=2, n_ages=1, sperm_storage=sperm)
        assert state.sperm_storage.shape == (1, 2, 2)
        assert state.sperm_storage[0, 0, 0] == 1.0
        assert state.sperm_storage[0, 1, 1] == 4.0
        assert state.sperm_storage.dtype == np.float64
        assert state.sperm_storage is not sperm  # it's a copy

    def test_n_genotypes_zero_raises(self):
        with pytest.raises(AssertionError, match="n_genotypes must be positive"):
            PopulationState.create(n_genotypes=0)

    def test_n_ages_zero_raises(self):
        with pytest.raises(AssertionError, match="n_ages must be positive"):
            PopulationState.create(n_genotypes=2, n_ages=0)

    def test_individual_count_shape_mismatch_raises(self):
        arr = np.ones((2, 2, 3))  # expected (2, 2, 5) for n_genotypes=5
        with pytest.raises(AssertionError, match="Invalid shape for individual_count"):
            PopulationState.create(n_genotypes=5, individual_count=arr)

    def test_sperm_storage_shape_mismatch_raises(self):
        sperm = np.ones((2, 3, 3))  # expected (2, 5, 5) for n_genotypes=5
        with pytest.raises(AssertionError, match="Invalid shape for sperm_storage"):
            PopulationState.create(n_genotypes=5, n_ages=2, sperm_storage=sperm)


# ═══════════════════════════════════════════════════════════════════════════
# PopulationState accessors
# ═══════════════════════════════════════════════════════════════════════════


class TestPopulationStateAccessors:
    @pytest.fixture
    def state(self):
        return PopulationState.create(n_genotypes=3, n_sexes=2, n_ages=2)

    def test_get_count_returns_correct_value(self, state):
        state.individual_count[1, 0, 2] = 7.5
        assert state.get_count(1, 0, 2) == 7.5

    def test_add_count_increments(self, state):
        assert state.get_count(0, 0, 0) == 0.0
        state.add_count(0, 0, 0, 5.0)
        assert state.get_count(0, 0, 0) == 5.0

    def test_add_count_negative(self, state):
        state.set_count(0, 0, 0, 10.0)
        state.add_count(0, 0, 0, -3.0)
        assert state.get_count(0, 0, 0) == 7.0

    def test_set_count_overwrites(self, state):
        state.set_count(0, 0, 0, 4.0)
        assert state.get_count(0, 0, 0) == 4.0
        state.set_count(0, 0, 0, 9.0)
        assert state.get_count(0, 0, 0) == 9.0

    def test_get_stored_sperm_returns_value(self, state):
        state.sperm_storage[1, 2, 0] = 3.0
        assert state.get_stored_sperm(1, 2, 0) == 3.0

    def test_set_stored_sperm_adds_inplace(self, state):
        state.set_stored_sperm(0, 0, 0, 2.0)
        assert state.get_stored_sperm(0, 0, 0) == 2.0
        state.set_stored_sperm(0, 0, 0, 3.0)
        assert state.get_stored_sperm(0, 0, 0) == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# DiscretePopulationState
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscretePopulationState:
    def test_create_default(self):
        state = DiscretePopulationState.create(n_sexes=2, n_ages=2, n_genotypes=4)
        assert state.n_tick == 0
        assert state.individual_count.shape == (2, 2, 4)
        assert state.individual_count.dtype == np.float64
        assert (state.individual_count == 0).all()

    def test_create_custom_tick(self):
        state = DiscretePopulationState.create(n_sexes=2, n_ages=2, n_genotypes=4, n_tick=42)
        assert state.n_tick == 42

    def test_create_provided_array(self):
        arr = np.array([[[1.5, 2.5]]])
        state = DiscretePopulationState.create(
            n_sexes=1, n_ages=1, n_genotypes=2, individual_count=arr
        )
        assert state.individual_count[0, 0, 0] == 1.5
        assert state.individual_count[0, 0, 1] == 2.5
        assert state.individual_count.dtype == np.float64

    def test_flatten_all_roundtrip(self):
        n_sexes, n_ages, n_genotypes = 2, 3, 4
        state = DiscretePopulationState.create(
            n_sexes=n_sexes, n_ages=n_ages, n_genotypes=n_genotypes, n_tick=7
        )
        state.individual_count[1, 2, 3] = 99.0
        flat = state.flatten_all()
        restored = parse_flattened_discrete_state(
            flat, n_sexes=n_sexes, n_ages=n_ages, n_genotypes=n_genotypes
        )
        assert restored.n_tick == 7
        assert (restored.individual_count == state.individual_count).all()
        assert restored.individual_count[1, 2, 3] == 99.0
        assert restored.individual_count.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════
# PopulationState flatten roundtrip
# ═══════════════════════════════════════════════════════════════════════════


class TestPopulationStateFlatten:
    def test_flatten_all_roundtrip(self):
        n_sexes, n_ages, n_genotypes = 2, 3, 4
        state = PopulationState.create(
            n_genotypes=n_genotypes, n_sexes=n_sexes, n_ages=n_ages, n_tick=5
        )
        state.individual_count[0, 1, 2] = 10.0
        state.sperm_storage[1, 2, 3] = 20.0
        flat = state.flatten_all()
        restored = parse_flattened_state(
            flat, n_sexes=n_sexes, n_ages=n_ages, n_genotypes=n_genotypes
        )
        assert restored.n_tick == 5
        assert (restored.individual_count == state.individual_count).all()
        assert (restored.sperm_storage == state.sperm_storage).all()
        assert restored.individual_count[0, 1, 2] == 10.0
        assert restored.sperm_storage[1, 2, 3] == 20.0
        assert restored.individual_count.dtype == np.float64
        assert restored.sperm_storage.dtype == np.float64
