"""
Test random sampling consistency across different modes.

This test systematically checks that all random sampling paths behave consistently
across deterministic, discrete stochastic, and continuous stochastic modes.
"""

import numpy as np
import pytest

from natal.algorithms import (
    _fertilize_with_precomputed_offspring_probability,
    _fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction,
    recruit_juveniles_sampling,
    recruit_juveniles_given_scaling_factor_sampling
)


def test_fertilization_sampling_consistency():
    """Test that fertilization functions behave consistently across sampling modes."""

    # Setup test data
    n_genotypes = 2
    n_ages = 3
    adult_start_idx = 1

    sperm_store = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
    sperm_store[1, 0, 0] = 10.0  # Adult age 1, genotype 0-0 pair
    sperm_store[2, 0, 0] = 5.0   # Adult age 2, genotype 0-0 pair

    fertility_f = np.ones(n_genotypes, dtype=np.float64)
    fertility_m = np.ones(n_genotypes, dtype=np.float64)

    offspring_probability = np.zeros((n_genotypes, n_genotypes, n_genotypes), dtype=np.float64)
    offspring_probability[0, 0, 0] = 0.5
    offspring_probability[0, 0, 1] = 0.5

    # Test different sampling modes
    modes = [
        ("deterministic", False, False),
        ("discrete_stochastic", True, False),
        ("continuous_stochastic", True, True)
    ]

    results = {}

    for mode_name, is_stochastic, use_continuous_sampling in modes:
        # Test with age-specific reproduction rates
        n_female, n_male = _fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
            sperm_storage_by_male_genotype=sperm_store,
            fertility_f=fertility_f,
            fertility_m=fertility_m,
            offspring_probability=offspring_probability,
            average_eggs_per_wt_female=10.0,
            adult_start_idx=adult_start_idx,
            n_ages=n_ages,
            n_genotypes=n_genotypes,
            female_genotype_compatibility=np.ones(n_genotypes),
            male_genotype_compatibility=np.ones(n_genotypes),
            female_only_by_sex_chrom=np.zeros(n_genotypes, dtype=np.bool_),
            male_only_by_sex_chrom=np.zeros(n_genotypes, dtype=np.bool_),
            n_glabs=1,
            age_based_reproduction_rates=np.array([0.0, 0.8, 0.5]),  # Age-specific rates
            female_age_based_relative_fertility=np.array([0.0, 1.0, 0.8]),  # Age-specific fertility
            fixed_eggs=False,
            sex_ratio=0.5,
            has_sex_chromosomes=False,
            is_stochastic=is_stochastic,
            use_continuous_sampling=use_continuous_sampling
        )

        results[mode_name] = (n_female.sum(), n_male.sum())

        # Basic consistency checks
        assert n_female.shape == (n_genotypes,)
        assert n_male.shape == (n_genotypes,)

        # In deterministic mode, values should be exact
        if not is_stochastic:
            assert np.all(n_female >= 0)
            assert np.all(n_male >= 0)

    # Verify that results are reasonable across modes
    deterministic_total = results["deterministic"][0] + results["deterministic"][1]

    # Stochastic modes should have similar expected values
    for mode_name in ["discrete_stochastic", "continuous_stochastic"]:
        stochastic_total = results[mode_name][0] + results[mode_name][1]
        # Allow some tolerance for stochastic variation
        assert abs(stochastic_total - deterministic_total) / deterministic_total < 0.5


def test_competition_sampling_consistency():
    """Test that competition functions behave consistently across sampling modes."""

    n_genotypes = 2
    juvenile_counts = (np.array([50.0, 30.0]), np.array([40.0, 20.0]))  # Female, male
    carrying_capacity = 100.0

    modes = [
        ("deterministic", False, False),
        ("discrete_stochastic", True, False),
        ("continuous_stochastic", True, True)
    ]

    results = {}

    for mode_name, is_stochastic, use_continuous_sampling in modes:
        female_new, male_new = recruit_juveniles_sampling(
            age_0_juvenile_counts=juvenile_counts,
            carrying_capacity=carrying_capacity,
            n_genotypes=n_genotypes,
            is_stochastic=is_stochastic,
            use_continuous_sampling=use_continuous_sampling
        )

        results[mode_name] = (female_new.sum(), male_new.sum())

        # Basic checks
        assert female_new.shape == (n_genotypes,)
        assert male_new.shape == (n_genotypes,)

        # Total should not exceed carrying capacity (allow small numerical error)
        total = female_new.sum() + male_new.sum()
        assert total <= carrying_capacity + 1e-5  # Allow small numerical error

    # Verify consistency
    deterministic_total = results["deterministic"][0] + results["deterministic"][1]

    for mode_name in ["discrete_stochastic", "continuous_stochastic"]:
        stochastic_total = results[mode_name][0] + results[mode_name][1]
        # Stochastic results should be reasonable
        assert stochastic_total >= 0
        assert stochastic_total <= carrying_capacity


def test_scaling_sampling_consistency():
    """Test that scaling functions behave consistently across sampling modes."""

    n_genotypes = 2
    juvenile_counts = (np.array([20.0, 10.0]), np.array([15.0, 5.0]))  # Female, male
    scaling_factor = 0.8

    modes = [
        ("deterministic", False, False),
        ("discrete_stochastic", True, False),
        ("continuous_stochastic", True, True)
    ]

    results = {}

    for mode_name, is_stochastic, use_continuous_sampling in modes:
        female_new, male_new = recruit_juveniles_given_scaling_factor_sampling(
            age_0_juvenile_counts=juvenile_counts,
            scaling_factor=scaling_factor,
            n_genotypes=n_genotypes,
            is_stochastic=is_stochastic,
            use_continuous_sampling=use_continuous_sampling
        )

        results[mode_name] = (female_new.sum(), male_new.sum())

        # Basic checks
        assert female_new.shape == (n_genotypes,)
        assert male_new.shape == (n_genotypes,)

    # Verify scaling is consistent
    original_total = sum(juvenile_counts[0]) + sum(juvenile_counts[1])
    expected_total = original_total * scaling_factor

    deterministic_total = results["deterministic"][0] + results["deterministic"][1]
    assert abs(deterministic_total - expected_total) < 1e-6

    for mode_name in ["discrete_stochastic", "continuous_stochastic"]:
        stochastic_total = results[mode_name][0] + results[mode_name][1]
        # Allow tolerance for stochastic rounding
        assert abs(stochastic_total - expected_total) / expected_total < 0.1


if __name__ == "__main__":
    test_fertilization_sampling_consistency()
    test_competition_sampling_consistency()
    test_scaling_sampling_consistency()
    print("All random sampling consistency tests passed!")
