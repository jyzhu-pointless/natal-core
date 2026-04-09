#!/usr/bin/env python3
"""Unit tests for population builder carrying capacity resolution logic."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.population_builder import PopulationConfigBuilder


def _make_species(name: str = "TestSp") -> nt.Species:
    """Create a simple test species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


class TestCarryingCapacityResolution:
    """Test the carrying capacity resolution logic."""

    def test_old_juvenile_carrying_capacity_has_priority(self) -> None:
        """Test that old_juvenile_carrying_capacity is used if provided."""
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=500.0,
            expected_num_adult_females=None,
            expected_eggs_per_female=10.0,
        )
        assert result == 500.0

    def test_explicit_carrying_capacity_used_when_no_old_juvenile(self) -> None:
        """Test that explicit age_1_carrying_capacity is used when old_juvenile is None."""
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,
            age_1_carrying_capacity=1000.0,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=None,
            expected_eggs_per_female=10.0,
        )
        assert result == 1000.0

    def test_expected_num_adult_females_without_rates(self) -> None:
        """Test fallback scaling when age rates not provided."""
        result = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=100.0,
            expected_eggs_per_female=10.0,
        )
        # Should use simple scaling: 100 * 10 = 1000
        assert result == 1000.0

    def test_error_when_no_source_provided(self) -> None:
        """Test that ValueError is raised when no source is available."""
        with pytest.raises(ValueError, match="No valid carrying capacity source"):
            PopulationConfigBuilder._resolve_carrying_capacity(
                carrying_capacity=None,
                age_1_carrying_capacity=None,
                old_juvenile_carrying_capacity=None,
                expected_num_adult_females=None,
                expected_eggs_per_female=10.0,
            )

    def test_equilibrium_carrying_capacity_calculation(self) -> None:
        """Test carrying capacity computed from expected_num_adult_females using equilibrium."""
        # Create test arrays
        n_ages = 8
        age_based_survival_rates = np.array([
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0],  # female
            [1.0, 0.85, 0.75, 0.65, 0.5, 0.35, 0.15, 0.0],  # male
        ], dtype=np.float64)

        age_based_mating_rates = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # female
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # male
        ], dtype=np.float64)

        female_age_based_relative_fertility = np.array(
            [0.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.2, 0.0],
            dtype=np.float64
        )

        expected_eggs_per_female = 50.0
        sex_ratio = 0.5
        new_adult_age = 1

        result = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=500.0,  # target adult female count
            expected_eggs_per_female=expected_eggs_per_female,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            female_age_based_relative_fertility=female_age_based_relative_fertility,
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
        )

        # Result should be positive and reasonable
        assert result > 0
        # Inferred K from 500 females with 50 eggs per female, considering survival
        # Should be on the order of 20000-100000 depending on fertility and survival distributions
        assert result < 200000  # Should not be unreasonably large

    def test_builder_integration_with_expected_num_adult_females(self) -> None:
        """Test that builder correctly uses equilibrium carrying capacity."""
        sp = _make_species("TestBuilderCapacity")

        # Build with expected_num_adult_females (no explicit carrying_capacity)
        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="BuilderTest", stochastic=False)
            .age_structure(n_ages=6, new_adult_age=2)
            .survival(
                female_age_based_survival_rates=[1.0, 0.95, 0.9, 0.8, 0.6, 0.0],
                male_age_based_survival_rates=[1.0, 0.9, 0.85, 0.75, 0.5, 0.0],
            )
            .reproduction(
                female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                eggs_per_female=40.0,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 0.0, 100.0, 100.0, 50.0, 0.0]},
                    "male": {"WT|WT": [0.0, 0.0, 100.0, 100.0, 50.0, 0.0]},
                }
            )
            .competition(
                juvenile_growth_mode="logistic",
                expected_num_adult_females=300.0,  # Use this to infer K
            )
            .build()
        )

        # Should have a valid config
        assert pop is not None
        cfg = pop.export_config()
        assert cfg.carrying_capacity > 0

    def test_builder_prefers_explicit_carrying_capacity(self) -> None:
        """Test that old_juvenile_carrying_capacity takes precedence over expected_num_adult_females."""
        sp = _make_species("TestPrecedence")

        pop = (
            nt.AgeStructuredPopulation
            .setup(species=sp, name="PrecedenceTest", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .reproduction(eggs_per_female=20.0)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0, 50.0, 25.0]},
                    "male": {"WT|WT": [0.0, 100.0, 50.0, 25.0]},
                }
            )
            .competition(
                juvenile_growth_mode="fixed",
                old_juvenile_carrying_capacity=5000,  # Explicit K (legacy)
                expected_num_adult_females=100,  # Should be ignored
            )
            .build()
        )

        cfg = pop.export_config()
        # Should use explicit old_juvenile_carrying_capacity
        assert cfg.carrying_capacity == 5000.0

    def test_equilibrium_distribution_consistency(self) -> None:
        """Test that inferred carrying capacity is used correctly in config."""
        n_ages = 6
        age_based_survival_rates = np.array([
            [1.0, 1.0, 0.9, 0.8, 0.6, 0.0],  # female
            [1.0, 1.0, 0.85, 0.75, 0.5, 0.0],  # male
        ], dtype=np.float64)

        age_based_mating_rates = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # female
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # male
        ], dtype=np.float64)

        female_age_based_relative_fertility = np.array(
            [0.0, 1.0, 1.0, 1.0, 0.8, 0.0],
            dtype=np.float64
        )

        expected_eggs_per_female = 100.0
        sex_ratio = 0.5
        new_adult_age = 1
        expected_num_adult_females = 200.0

        # Compute inferred K
        inferred_k = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,
            age_1_carrying_capacity=None,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=expected_num_adult_females,
            expected_eggs_per_female=expected_eggs_per_female,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            female_age_based_relative_fertility=female_age_based_relative_fertility,
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
        )

        # Should have inferred a positive K
        assert inferred_k > 0
        assert inferred_k < 100000  # Reasonable upper bound

        # Verify that use of inferred K in equilibrium metrics is self-consistent
        from natal.algorithms import compute_equilibrium_metrics

        age_based_relative_competition_strength = np.ones(n_ages, dtype=np.float64)

        comp_strength, exp_survival_rate = compute_equilibrium_metrics(
            carrying_capacity=inferred_k,
            expected_eggs_per_female=expected_eggs_per_female,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            female_age_based_relative_fertility=female_age_based_relative_fertility,
            relative_competition_strength=age_based_relative_competition_strength,
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
            equilibrium_individual_count=None,
        )

        # Expected survival rate should be non-negative and reasonable
        assert exp_survival_rate >= 0.0
        assert exp_survival_rate < 10.0  # Should not be unreasonably large
