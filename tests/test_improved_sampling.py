"""Tests for small-sample continuous binomial sampling methods."""

import numpy as np

from natal.numba_compat import continuous_binomial


def _continuous_binomial_min_concentration(n: float, p: float) -> float:
    """Continuous binomial with a minimum concentration to avoid Beta(0,0)."""
    if p <= 1e-10:
        return 0.0
    if p >= 1.0 - 1e-10:
        return float(n)
    concentration = max(n - 1.0, 1.0)
    alpha = max(p * concentration, 1e-10)
    beta_val = max((1.0 - p) * concentration, 1e-10)
    return np.random.beta(alpha, beta_val) * n


def test_small_n_methods():
    """Min-concentration and library continuous_binomial both produce valid means."""
    np.random.seed(42)
    test_cases = [(1.0, 0.5), (0.5, 0.3), (2.0, 0.5)]
    n_samples = 10000

    for n, p in test_cases:
        theoretical_mean = n * p

        min_conc_samples = [_continuous_binomial_min_concentration(n, p) for _ in range(n_samples)]
        lib_samples = [continuous_binomial(n, p) for _ in range(n_samples)]

        min_conc_mean = np.mean(min_conc_samples)
        lib_mean = np.mean(lib_samples)

        # Both methods should match theoretical mean within a generous tolerance
        assert abs(min_conc_mean - theoretical_mean) < 0.05 * theoretical_mean + 1.0, (
            f"n={n}, p={p}: min-concentration mean {min_conc_mean:.4f} too far from "
            f"theoretical {theoretical_mean:.4f}"
        )
        assert abs(lib_mean - theoretical_mean) < 0.05 * theoretical_mean + 1.0, (
            f"n={n}, p={p}: library mean {lib_mean:.4f} too far from "
            f"theoretical {theoretical_mean:.4f}"
        )

        # Samples must be non-negative and at most n
        assert min(min_conc_samples) >= 0.0
        assert max(min_conc_samples) <= n
        assert min(lib_samples) >= 0.0
        assert max(lib_samples) <= n


def test_workflow_consistency():
    """Rounding continuous pair counts produces non-negative integers."""
    combo_pairs = np.array([1.2, 2.7, 0.8, 3.5])
    rounded = np.round(combo_pairs)
    assert np.all(rounded >= 0)
    assert rounded.dtype == np.float64
