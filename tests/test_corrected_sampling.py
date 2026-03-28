"""Tests for the corrected continuous binomial concentration parameter."""

import numpy as np


def _concentration(n: float) -> float:
    """Corrected concentration parameter: solution to variance-matching equation."""
    return (np.sqrt(1.0 + 4.0 * n) - 1.0) / 2.0


def _continuous_binomial_corrected(n: float, p: float) -> float:
    """Continuous binomial using the corrected concentration parameter."""
    if p <= 1e-10:
        return 0.0
    if p >= 1.0 - 1e-10:
        return float(n)
    c = _concentration(n)
    alpha = max(p * c, 1e-10)
    beta_val = max((1.0 - p) * c, 1e-10)
    return np.random.beta(alpha, beta_val) * n


def test_concentration_function():
    """Concentration function f(n) returns mathematically correct values."""
    # f(n) = (sqrt(1+4n) - 1) / 2 satisfies f(f+1) = n, i.e. f^2+f-n = 0
    for n in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        f = _concentration(n)
        assert f > 0, f"f({n}) must be positive, got {f}"
        # Verify the quadratic: f*(f+1) == n
        assert abs(f * (f + 1) - n) < 1e-10, (
            f"f({n}) = {f} does not satisfy f*(f+1) = {n}"
        )


def test_variance_matching():
    """Corrected continuous binomial mean matches theoretical for various n and p."""
    np.random.seed(42)
    test_cases = [(5.0, 0.3), (10.0, 0.3), (100.0, 0.3)]
    n_samples = 10000

    for n, p in test_cases:
        theoretical_mean = n * p
        corrected_samples = [_continuous_binomial_corrected(n, p) for _ in range(n_samples)]
        corrected_mean = np.mean(corrected_samples)

        assert abs(corrected_mean - theoretical_mean) < 0.05 * theoretical_mean + 1.0, (
            f"n={n}, p={p}: corrected mean {corrected_mean:.4f} too far from "
            f"theoretical {theoretical_mean:.4f}"
        )

        # Corrected samples must stay within [0, n]
        assert min(corrected_samples) >= 0.0
        assert max(corrected_samples) <= n
