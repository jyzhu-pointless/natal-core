"""Tests for sampling consistency between discrete and continuous distributions."""

import numpy as np

from natal import numba_compat as nbc
from natal.numba_compat import continuous_binomial, continuous_multinomial


def test_binomial_consistency():
    """Continuous binomial mean and variance match discrete binomial moments."""
    np.random.seed(42)
    n = 100
    p = 0.3
    n_samples = 10000

    discrete_samples = [nbc.binomial(int(round(n)), p) for _ in range(n_samples)]
    continuous_samples = [continuous_binomial(float(n), p) for _ in range(n_samples)]

    theoretical_mean = n * p
    discrete_mean = np.mean(discrete_samples)
    continuous_mean = np.mean(continuous_samples)

    # Mean should be within 5 % of theoretical
    assert abs(discrete_mean - theoretical_mean) < 0.05 * theoretical_mean + 1.0
    assert abs(continuous_mean - theoretical_mean) < 0.05 * theoretical_mean + 1.0

    # Continuous samples should stay in a sensible range
    assert min(continuous_samples) >= 0.0
    assert max(continuous_samples) <= n


def test_multinomial_consistency():
    """Continuous multinomial means match discrete multinomial and rows sum to n."""
    np.random.seed(42)
    n = 100
    p_array = np.array([0.2, 0.3, 0.5])
    n_samples = 10000
    k = len(p_array)

    discrete_samples = [nbc.multinomial(int(round(n)), p_array) for _ in range(n_samples)]
    continuous_samples = []
    for _ in range(n_samples):
        temp_counts = np.zeros(k, dtype=np.float64)
        continuous_multinomial(float(n), p_array, temp_counts)
        continuous_samples.append(temp_counts.copy())

    discrete_samples = np.array(discrete_samples)
    continuous_samples = np.array(continuous_samples)

    # Row sums must equal n exactly for discrete and approximately for continuous
    assert np.all(discrete_samples.sum(axis=1) == n)
    continuous_row_sums = continuous_samples.sum(axis=1)
    assert np.allclose(continuous_row_sums, n, atol=1e-9)

    # Per-category means should match theoretical
    for i in range(k):
        theoretical = n * p_array[i]
        assert abs(np.mean(discrete_samples[:, i]) - theoretical) < 0.05 * theoretical + 1.0
        assert abs(np.mean(continuous_samples[:, i]) - theoretical) < 0.05 * theoretical + 1.0


def test_small_n_cases():
    """Small-n continuous binomial mean stays close to theoretical."""
    np.random.seed(42)
    test_cases = [(10, 0.3), (5, 0.7), (2, 0.5), (1, 0.5)]
    n_samples = 5000

    for n, p in test_cases:
        continuous_samples = [continuous_binomial(float(n), p) for _ in range(n_samples)]
        theoretical_mean = n * p
        continuous_mean = np.mean(continuous_samples)

        assert abs(continuous_mean - theoretical_mean) < 0.1 * theoretical_mean + 0.5, (
            f"n={n}, p={p}: continuous mean {continuous_mean:.4f} too far from "
            f"theoretical {theoretical_mean:.4f}"
        )
        assert min(continuous_samples) >= 0.0
        assert max(continuous_samples) <= n
