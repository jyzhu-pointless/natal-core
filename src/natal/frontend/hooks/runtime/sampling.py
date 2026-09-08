"""Sampling helpers for the finish-event CSR interpreter.

The declarative-hook interpreter needs a small, self-contained sampler
set (discrete binomial plus the NATAL-specific continuous binomial
surrogate) so the hook runtime stays independent of the engine session.

The discrete sampler delegates directly to NumPy's native generator.
The continuous sampler implements the NATAL-specific moment-matching
surrogate as plain Python math.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "binomial",
    "continuous_binomial",
]

EPS = 1e-10

_CONTINUOUS_SAMPLING_RESOLUTION_LIMIT: float = float(2**104)
_GAMMA_NORMAL_APPROXIMATION_THRESHOLD: float = 1e8
_MAX_GAMMA_ATTEMPTS: int = 1024


def binomial(n: int, p: float) -> int:
    """Draw from the Binomial distribution.

    Args:
        n: Number of independent trials.
        p: Success probability.

    Returns:
        A sampled count in ``[0, n]``.
    """
    return int(np.random.binomial(n, p))


def _bounded_gamma(shape: float) -> float:
    """Sample a unit-scale Gamma variate with bounded rejection attempts.

    Args:
        shape: Positive finite Gamma shape parameter.

    Returns:
        A Gamma sample, or the distribution mean if numerical degeneration
        prevents acceptance within the fixed attempt budget.
    """
    if shape >= _CONTINUOUS_SAMPLING_RESOLUTION_LIMIT:
        return shape
    # For large shape, Gamma skewness is negligible while the rejection
    # formula below loses variance through cancellation of O(shape) terms.
    if shape >= _GAMMA_NORMAL_APPROXIMATION_THRESHOLD:
        sample = shape + np.sqrt(shape) * np.random.normal()
        return max(sample, 0.0)

    magic = 1.0 + np.log(4.5)
    if shape > 1.0:
        inverse_scale = np.sqrt(2.0 * shape - 1.0)
        shifted_shape = shape - np.log(4.0)
        proposal_scale = shape + inverse_scale
        for _ in range(_MAX_GAMMA_ATTEMPTS):
            uniform_1 = np.random.random()
            if not 1e-7 < uniform_1 < 0.9999999:
                continue
            uniform_2 = 1.0 - np.random.random()
            logit = np.log(uniform_1 / (1.0 - uniform_1)) / inverse_scale
            proposal = shape * np.exp(logit)
            product = uniform_1 * uniform_1 * uniform_2
            log_acceptance = shifted_shape + proposal_scale * logit - proposal
            if (
                log_acceptance + magic - 4.5 * product >= 0.0
                or log_acceptance >= np.log(product)
            ):
                return proposal
        return shape

    if shape == 1.0:
        return -np.log(1.0 - np.random.random())

    coefficient = (np.e + shape) / np.e
    for _ in range(_MAX_GAMMA_ATTEMPTS):
        scaled_uniform = coefficient * np.random.random()
        if scaled_uniform <= 1.0:
            proposal = scaled_uniform ** (1.0 / shape)
        else:
            proposal = -np.log((coefficient - scaled_uniform) / shape)
        acceptance_uniform = np.random.random()
        if scaled_uniform > 1.0:
            if acceptance_uniform <= proposal ** (shape - 1.0):
                return proposal
        elif acceptance_uniform <= np.exp(-proposal):
            return proposal
    return shape


def continuous_binomial(n: float, p: float) -> float:
    """Continuousized Binomial surrogate.

    Args:
        n: Binomial sample size.
        p: Binomial success probability (0 < p < 1).

    Returns:
        Continuous count value (float between 0 and n).

    Raises:
        ValueError: If ``n`` or ``p`` is not finite.
    """
    if not np.isfinite(n) or not np.isfinite(p):
        raise ValueError("continuous_binomial(): n and p must be finite")
    if p <= EPS:
        return 0.0
    if p >= 1.0 - EPS:
        return float(n)

    # When n <= 1, the concentration (n-1) is non-positive, making it impossible to perform effective moment matching via Beta distribution.
    # In this case, forced sampling would cause severe numerical bias (tending towards 0.5*n), so we fall back to deterministic expected value.
    if n <= 1.0 + EPS:
        return n * p

    # Moment matching: map Binomial(n, p) to proportion variable r~Beta(alpha,beta), then return n*r.
    # The larger the concentration, the smaller the fluctuation in r (closer to deterministic p).
    concentration = n - 1.0
    # alpha / (alpha + beta) = p, ensuring the proportion mean is p
    alpha = p * concentration
    beta_val = (1.0 - p) * concentration

    # Numerical protection
    alpha = max(alpha, EPS)
    beta_val = max(beta_val, EPS)

    numerator = _bounded_gamma(alpha)
    denominator_component = _bounded_gamma(beta_val)
    proportion = (
        0.0 if numerator == 0.0 else numerator / (numerator + denominator_component)
    )
    # Return "continuous count" rather than proportion: count = n * proportion
    return proportion * n
