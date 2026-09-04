"""Pure-Python sampling helpers for the reference backend.

The discrete samplers (binomial / multinomial / poisson) delegate directly to
NumPy's native generators.  The continuous samplers (``continuous_binomial``,
``continuous_multinomial``, ``continuous_poisson``) implement NATAL-specific
moment-matching surrogates as plain Python math; their formulas are
bit-for-bit identical to the previous compat implementations.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EPS",
    "binomial",
    "clamp01",
    "continuous_binomial",
    "continuous_multinomial",
    "continuous_poisson",
    "fast_binomial",
    "multinomial",
]

EPS = 1e-10

_CONTINUOUS_SAMPLING_RESOLUTION_LIMIT: float = float(2**104)
_GAMMA_NORMAL_APPROXIMATION_THRESHOLD: float = 1e8
_MAX_GAMMA_ATTEMPTS: int = 1024


def clamp01(x: float) -> float:
    """Clamp a probability-like value into [0, 1].

    Args:
        x: Value to clamp.

    Returns:
        ``0.0`` when *x* is below the unit interval, ``1.0`` when above,
        otherwise *x* unchanged.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def binomial(n: int, p: float) -> int:
    """Draw from the Binomial distribution.

    Args:
        n: Number of independent trials.
        p: Success probability.

    Returns:
        A sampled count in ``[0, n]``.
    """
    return int(np.random.binomial(n, p))


def fast_binomial(n: int, p: float) -> int:
    """Alias of :func:`binomial` kept for call sites that used the JIT flavor.

    Args:
        n: Number of independent trials.
        p: Success probability.

    Returns:
        A sampled count in ``[0, n]``.
    """
    return int(np.random.binomial(n, p))


def multinomial(
    n: int,
    pvals: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Draw from the Multinomial distribution.

    Args:
        n: Total number of trials.
        pvals: Probability vector with shape ``(k,)`` that sums to 1.0.

    Returns:
        Sampled counts with shape ``(k,)`` that sum to *n*.
    """
    return np.random.multinomial(n, pvals)


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
            logit = (
                np.log(uniform_1 / (1.0 - uniform_1)) / inverse_scale
            )
            proposal = shape * np.exp(logit)
            product = uniform_1 * uniform_1 * uniform_2
            log_acceptance = (
                shifted_shape + proposal_scale * logit - proposal
            )
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
            proposal = -np.log(
                (coefficient - scaled_uniform) / shape
            )
        acceptance_uniform = np.random.random()
        if scaled_uniform > 1.0:
            if acceptance_uniform <= proposal ** (shape - 1.0):
                return proposal
        elif acceptance_uniform <= np.exp(-proposal):
            return proposal
    return shape


def _continuous_poisson(lam: float) -> float:
    """Use Gamma distribution to continuousize Poisson distribution.

    Moments matching: Poisson(λ) -> Gamma(λ, 1)
    Mean and variance are both λ.

    Args:
        lam: Poisson parameter λ

    Returns:
        Value sampled from Gamma(λ, 1)

    Raises:
        ValueError: If ``lam`` is not finite.
    """
    if not np.isfinite(lam):
        raise ValueError("continuous_poisson(): lam must be finite")
    if lam >= _CONTINUOUS_SAMPLING_RESOLUTION_LIMIT:
        return lam
    if lam <= EPS:
        return 0.0
    return _bounded_gamma(lam)


def _continuous_binomial(n: float, p: float) -> float:
    """Use Beta distribution to continuousize Binomial distribution.

    Moments matching: Binomial(n, p) -> Beta((n-1)*p, (n-1)*(1-p))
    Multiply the sampled proportion by n to get "continuous count".

    Args:
        n: Binomial sample size
        p: Binomial success probability (0 < p < 1)

    Returns:
        Continuous count value (float between 0 and n)

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
        0.0
        if numerator == 0.0
        else numerator / (numerator + denominator_component)
    )
    # Return "continuous count" rather than proportion: count = n * proportion
    return proportion * n


def _continuous_multinomial(n: float, p_array: NDArray[np.float64], out_counts: NDArray[np.float64]) -> None:
    """Use Dirichlet distribution to continuousize Multinomial distribution.

    Moments matching: Multinomial(n, p) → Dirichlet((n-1) × p).
    Generates continuous samples by scaling independent Gamma variates.
    Results are stored in out_counts in-place (no return value).

    This function avoids memory allocation issues by using component-wise Gamma
    sampling and normalizing, rather than calling a high-level Dirichlet routine.
    For very small total counts (n ≤ 1), falls back to deterministic expected values.

    Args:
        n: Total count (continuous multinomial sample size).
        p_array: Probability vector with shape (k,), must sum to 1.
        out_counts: Pre-allocated output array with shape (k,). Will be filled with
            continuous category counts that sum to approximately n (modified in-place).

    Raises:
        ValueError: If ``n`` or any probability is not finite.
    """
    k = len(p_array)
    if not np.isfinite(n):
        raise ValueError("continuous_multinomial(): n must be finite")
    for i in range(k):
        if not np.isfinite(p_array[i]):
            raise ValueError(
                "continuous_multinomial(): probabilities must be finite"
            )
    # Performance optimization and numerical protection: for extremely small sample sizes, use deterministic allocation directly.
    if n <= 1.0 + EPS:
        for i in range(k):
            out_counts[i] = n * p_array[i]
        return

    # Similar to continuous_binomial, Dirichlet total concentration is set to (n-1).
    # Each category concentration alpha_i = p_i * (n-1), so the mean is p_i.
    concentration = n - 1.0
    sum_gamma = 0.0

    # Generate k Gamma(α_i, 1) variables
    for i in range(k):
        alpha = p_array[i] * concentration

        if alpha <= EPS:
            # If probability is extremely low, set to 0 directly
            val = 0.0
        else:
            val = _bounded_gamma(alpha)

        out_counts[i] = val
        sum_gamma += val

    # Normalize and multiply by total n:
    # If g_i ~ Gamma(alpha_i,1), then g_i/sum(g) ~ Dirichlet(alpha)
    # Finally out_i = n * g_i/sum(g) is the continuous "category count".
    if sum_gamma > EPS:
        factor = n / sum_gamma
        for i in range(k):
            out_counts[i] *= factor
    else:
        # Extreme case (all alpha close to 0）
        # Use original probability vector for fallback to maintain total approximately n
        for i in range(k):
            out_counts[i] = n * p_array[i]

    # Final numerical validation: ensure output sum is approximately equal to n, avoiding cumulative numerical errors
    total = 0.0
    for i in range(k):
        total += out_counts[i]

    # If total is very small or already within reasonable error range, no additional processing needed
    tol = 1e-6 * max(1.0, n)
    if total > EPS and abs(total - n) > tol:
        # Lightweight rescaling to correct deviations caused by floating point errors
        correction = n / total
        for i in range(k):
            out_counts[i] *= correction


def continuous_poisson(lam: float) -> float:
    """Continuousized Poisson surrogate.

    Args:
        lam: Poisson parameter λ.

    Returns:
        Value sampled from Gamma(λ, 1).

    Raises:
        ValueError: If ``lam`` is not finite.
    """
    return _continuous_poisson(lam)


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
    return _continuous_binomial(n, p)


def continuous_multinomial(
    n: float,
    p_array: NDArray[np.float64],
    out_counts: NDArray[np.float64],
) -> None:
    """Continuousized Multinomial surrogate (writes into *out_counts*).

    Args:
        n: Total count (continuous multinomial sample size).
        p_array: Probability vector with shape (k,), must sum to 1.
        out_counts: Output array with shape (k,) filled in place.

    Raises:
        ValueError: If ``n`` or any probability is not finite.
    """
    _continuous_multinomial(n, p_array, out_counts)
