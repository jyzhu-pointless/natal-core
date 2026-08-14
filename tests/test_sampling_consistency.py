"""Tests for sampling consistency between discrete and continuous distributions."""

import subprocess
import sys

import numpy as np
import pytest

from natal.numba import compat as nbc
from natal.numba.compat import continuous_binomial, continuous_multinomial


@pytest.mark.parametrize(
    "expression",
    [
        "nbc.continuous_poisson(float('nan'))",
        "nbc.continuous_poisson(float('inf'))",
        "nbc.continuous_poisson(float('-inf'))",
        "nbc.continuous_binomial(float('nan'), 0.5)",
        "nbc.continuous_binomial(float('inf'), 0.5)",
        "nbc.continuous_binomial(float('-inf'), 0.5)",
        "nbc.continuous_binomial(10.0, float('nan'))",
        "nbc.continuous_binomial(10.0, float('inf'))",
        "nbc.continuous_binomial(10.0, float('-inf'))",
        (
            "nbc.continuous_multinomial("
            "float('inf'), np.array([0.5, 0.5]), np.zeros(2))"
        ),
        (
            "nbc.continuous_multinomial("
            "float('-inf'), np.array([0.5, 0.5]), np.zeros(2))"
        ),
        (
            "nbc.continuous_multinomial("
            "float('nan'), np.array([0.5, 0.5]), np.zeros(2))"
        ),
        (
            "nbc.continuous_multinomial("
            "10.0, np.array([float('nan'), 1.0]), np.zeros(2))"
        ),
        (
            "nbc.continuous_multinomial("
            "10.0, np.array([float('inf'), 0.0]), np.zeros(2))"
        ),
        (
            "nbc.continuous_multinomial("
            "10.0, np.array([float('-inf'), 1.0]), np.zeros(2))"
        ),
    ],
)
def test_continuous_samplers_reject_non_finite_inputs(
    expression: str,
) -> None:
    """Ensure non-finite parameters fail instead of looping forever.

    Args:
        expression: Python expression that invokes a sampler with invalid input.
    """
    code = (
        "import numpy as np\n"
        "from natal.numba import compat as nbc\n"
        "try:\n"
        f"    {expression}\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ValueError')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("n", "probabilities"),
    [
        (float("nan"), np.array([0.25, 0.75])),
        (10.0, np.array([0.25, float("inf")])),
    ],
)
def test_continuous_multinomial_invalid_input_preserves_output(
    n: float,
    probabilities: np.ndarray,
) -> None:
    """Ensure validation failures preserve the caller's output buffer.

    Args:
        n: Invalid or finite continuous sample size.
        probabilities: Probability vector containing the invalid test input.
    """
    sentinel = np.array([-7.0, -11.0])
    out_counts = sentinel.copy()

    with pytest.raises(ValueError):
        continuous_multinomial(n, probabilities, out_counts)

    assert np.array_equal(out_counts, sentinel)


def test_continuous_samplers_finish_at_float64_resolution_limits() -> None:
    """Extreme finite inputs remain close to their expectations and finish promptly."""
    code = """
import time
import numpy as np
from natal.numba import compat as nbc

probabilities = np.array([0.25, 0.75])
scratch = np.empty(2)

# Compile before timing so the deadline measures sampling rather than JIT work.
nbc.continuous_poisson(10.0)
nbc.continuous_binomial(10.0, 0.25)
nbc.continuous_multinomial(10.0, probabilities, scratch)

resolution_limit = float(2**104)
largest_finite = np.finfo(np.float64).max
started = time.monotonic()
for total in (resolution_limit, largest_finite):
    poisson_sample = nbc.continuous_poisson(total)
    assert np.isfinite(poisson_sample)
    assert poisson_sample == total

    binomial_sample = nbc.continuous_binomial(total, 0.25)
    assert np.isfinite(binomial_sample)
    assert 0.0 <= binomial_sample <= total
    assert np.isclose(binomial_sample, total * 0.25, rtol=1e-12)

    out_counts = np.full(2, np.nan)
    nbc.continuous_multinomial(total, probabilities, out_counts)
    expected_counts = total * probabilities
    assert np.all(np.isfinite(out_counts))
    assert np.all(out_counts >= 0.0)
    assert np.allclose(out_counts, expected_counts, rtol=1e-12)

assert time.monotonic() - started < 2.0
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_rare_categories_remain_stochastic_at_large_total_count() -> None:
    """Large totals preserve resolvable rare-category mean and variance."""
    code = """
import time
import numpy as np
from natal.numba import compat as nbc

total = float(2**104)
rare_probability = np.nextafter(nbc.EPS, np.inf)
rare_probabilities = np.array(
    [rare_probability, 1.0 - rare_probability]
)
common_probabilities = np.array([0.25, 0.75])
out_counts = np.empty(2)

# Compile before timing so the deadline measures sampling rather than JIT work.
nbc.continuous_binomial(10.0, 0.25)
nbc.continuous_multinomial(10.0, common_probabilities, out_counts)
nbc.set_numba_seed(194903)

n_samples = 20_000
binomial_samples = np.empty(n_samples)
multinomial_rare_samples = np.empty(n_samples)
started = time.monotonic()
for index in range(n_samples):
    binomial_samples[index] = nbc.continuous_binomial(
        total,
        rare_probability,
    )
    nbc.continuous_multinomial(
        total,
        rare_probabilities,
        out_counts,
    )
    multinomial_rare_samples[index] = out_counts[0]

# Common high-count categories must also remain prompt and finite.
for _ in range(2_000):
    common_binomial = nbc.continuous_binomial(total, 0.25)
    nbc.continuous_multinomial(
        total,
        common_probabilities,
        out_counts,
    )
    assert np.isfinite(common_binomial)
    assert np.all(np.isfinite(out_counts))

assert time.monotonic() - started < 2.0

expected_mean = total * rare_probability
expected_variance = total * rare_probability * (1.0 - rare_probability)
for samples in (binomial_samples, multinomial_rare_samples):
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= 0.0)
    assert np.unique(samples).size > 100
    assert abs(np.mean(samples) - expected_mean) <= 0.01 * expected_mean
    assert (
        abs(np.var(samples, ddof=1) - expected_variance)
        <= 0.10 * expected_variance
    )
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_continuous_samplers_finish_around_stochastic_boundaries() -> None:
    """Inputs on both sides of stochastic cutoffs remain finite and bounded."""
    code = """
import time
import numpy as np
from natal.numba import compat as nbc

probabilities = np.array([0.25, 0.75])
scratch = np.empty(2)
nbc.continuous_poisson(10.0)
nbc.continuous_binomial(10.0, 0.25)
nbc.continuous_multinomial(10.0, probabilities, scratch)

resolution_limit = float(2**104)
just_below_limit = np.nextafter(resolution_limit, 0.0)
just_above_eps = np.nextafter(nbc.EPS, np.inf)
started = time.monotonic()

for total in (just_below_limit, resolution_limit):
    poisson_sample = nbc.continuous_poisson(total)
    assert np.isfinite(poisson_sample)
    assert poisson_sample >= 0.0

    binomial_sample = nbc.continuous_binomial(total, 0.25)
    assert np.isfinite(binomial_sample)
    assert 0.0 <= binomial_sample <= total

    out_counts = np.empty(2)
    nbc.continuous_multinomial(total, probabilities, out_counts)
    assert np.all(np.isfinite(out_counts))
    assert np.all(out_counts >= 0.0)
    assert np.allclose(out_counts, total * probabilities, rtol=1e-14)

poisson_sample = nbc.continuous_poisson(just_above_eps)
assert np.isfinite(poisson_sample)
assert poisson_sample >= 0.0

binomial_sample = nbc.continuous_binomial(2.0, just_above_eps)
assert np.isfinite(binomial_sample)
assert 0.0 <= binomial_sample <= 2.0

out_counts = np.empty(2)
nbc.continuous_multinomial(
    2.0,
    np.array([just_above_eps, 1.0 - just_above_eps]),
    out_counts,
)
assert np.all(np.isfinite(out_counts))
assert np.all(out_counts >= 0.0)
np.testing.assert_allclose(np.sum(out_counts), 2.0, rtol=1e-12)

assert time.monotonic() - started < 2.0
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_bounded_gamma_uses_mean_after_rejection_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adversarial RNG cannot keep the Python Gamma sampler in its loop.

    Args:
        monkeypatch: Pytest fixture used to force every proposal to be rejected.
    """
    python_sampler = getattr(
        nbc._bounded_gamma,
        "py_func",
        nbc._bounded_gamma,
    )
    call_count = 0

    def always_rejected() -> float:
        """Return an adversarial value and record each rejection attempt."""
        nonlocal call_count
        call_count += 1
        return 0.0

    monkeypatch.setattr(np.random, "random", always_rejected)

    sample = python_sampler(2.0)

    assert sample == 2.0
    assert call_count == nbc._MAX_GAMMA_ATTEMPTS


def test_python_bounded_gamma_covers_each_numerical_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify exact fallback values across every Gamma shape regime.

    Args:
        monkeypatch: Pytest fixture supplying deterministic random variates.
    """
    python_sampler = getattr(
        nbc._bounded_gamma,
        "py_func",
        nbc._bounded_gamma,
    )

    assert python_sampler(float(2**104)) == float(2**104)
    monkeypatch.setattr(np.random, "normal", lambda: 0.0)
    assert python_sampler(1e8) == 1e8
    monkeypatch.setattr(np.random, "normal", lambda: -1e9)
    assert python_sampler(1e8) == 0.0

    values = iter((0.5, 0.5))
    monkeypatch.setattr(np.random, "random", lambda: next(values))
    assert python_sampler(2.0) == pytest.approx(2.0)

    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    assert python_sampler(1.0) == pytest.approx(np.log(2.0))

    values = iter((0.25, 0.0))
    monkeypatch.setattr(np.random, "random", lambda: next(values))
    assert 0.0 < python_sampler(0.5) < 1.0

    values = iter((0.99, 0.0))
    monkeypatch.setattr(np.random, "random", lambda: next(values))
    assert python_sampler(0.5) > 0.0

    call_count = 0

    def reject_small_shape() -> float:
        """Alternate a valid proposal with an acceptance rejection."""
        nonlocal call_count
        call_count += 1
        return 0.5 if call_count % 2 == 1 else 0.99

    monkeypatch.setattr(np.random, "random", reject_small_shape)
    assert python_sampler(0.5) == 0.5
    assert call_count == 2 * nbc._MAX_GAMMA_ATTEMPTS


def test_python_continuous_samplers_preserve_exact_moments_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise coverage-visible sampler bodies with exact Gamma means.

    Args:
        monkeypatch: Pytest fixture replacing Gamma draws by their means.
    """
    poisson = getattr(
        nbc._continuous_poisson,
        "py_func",
        nbc._continuous_poisson,
    )
    binomial = getattr(
        nbc._continuous_binomial,
        "py_func",
        nbc._continuous_binomial,
    )
    multinomial = getattr(
        nbc._continuous_multinomial,
        "py_func",
        nbc._continuous_multinomial,
    )
    monkeypatch.setattr(nbc, "_bounded_gamma", lambda shape: shape)

    assert poisson(0.0) == 0.0
    assert poisson(4.0) == 4.0
    assert poisson(float(2**104)) == float(2**104)
    assert binomial(1.0, 0.25) == 0.25
    assert binomial(8.0, 0.0) == 0.0
    assert binomial(8.0, 1.0) == 8.0
    assert binomial(8.0, 0.25) == pytest.approx(2.0)

    probabilities = np.array([0.25, 0.75], dtype=np.float64)
    output = np.full(2, -1.0, dtype=np.float64)
    multinomial(8.0, probabilities, output)
    np.testing.assert_allclose(output, [2.0, 6.0], rtol=0.0, atol=1e-15)

    small_output = np.full(2, -1.0, dtype=np.float64)
    multinomial(1.0, probabilities, small_output)
    np.testing.assert_allclose(
        small_output,
        [0.25, 0.75],
        rtol=0.0,
        atol=1e-15,
    )

    monkeypatch.setattr(nbc, "_bounded_gamma", lambda _shape: 0.0)
    fallback_output = np.full(2, -1.0, dtype=np.float64)
    multinomial(8.0, probabilities, fallback_output)
    np.testing.assert_allclose(
        fallback_output,
        [2.0, 6.0],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("sampler_name", "args"),
    [
        ("_continuous_poisson", (float("nan"),)),
        ("_continuous_binomial", (10.0, float("inf"))),
        (
            "_continuous_multinomial",
            (10.0, np.array([0.5, float("-inf")]), np.array([-3.0, -5.0])),
        ),
        (
            "_continuous_multinomial",
            (float("nan"), np.array([0.5, 0.5]), np.array([-3.0, -5.0])),
        ),
    ],
)
def test_python_continuous_samplers_reject_nonfinite_before_mutation(
    sampler_name: str,
    args: tuple[object, ...],
) -> None:
    """Reject non-finite Python-fallback inputs before output mutation.

    Args:
        sampler_name: Private sampler dispatcher name.
        args: Invalid runtime arguments.
    """
    sampler = getattr(nbc, sampler_name)
    python_sampler = getattr(sampler, "py_func", sampler)
    output_before = (
        args[-1].copy()
        if sampler_name == "_continuous_multinomial"
        else None
    )

    with pytest.raises(ValueError):
        python_sampler(*args)

    if output_before is not None:
        np.testing.assert_array_equal(args[-1], output_before)


@pytest.mark.parametrize(
    ("shape", "seed"),
    [(1e8, 69101), (2e21, 69102)],
)
def test_bounded_gamma_large_shapes_match_target_moments(
    shape: float,
    seed: int,
) -> None:
    """Normal-approximation shapes retain Gamma mean and variance.

    Args:
        shape: Large Gamma shape in the normal-approximation interval.
        seed: Deterministic unit-test seed for the sampler.
    """
    nbc.set_numba_seed(seed)
    n_samples = 30_000

    samples = np.array(
        [nbc._bounded_gamma(shape) for _ in range(n_samples)]
    )

    assert np.all(np.isfinite(samples))
    assert np.all(samples >= 0.0)
    assert np.mean(samples) == pytest.approx(shape, rel=0.01)
    assert np.var(samples, ddof=1) == pytest.approx(shape, rel=0.04)


@pytest.mark.parametrize(
    ("lam", "seed"),
    [(0.5, 90210), (1.0, 90211), (40.0, 90212)],
)
def test_continuous_poisson_matches_target_moments(
    lam: float,
    seed: int,
) -> None:
    """Gamma surrogate preserves the Poisson mean and variance.

    Args:
        lam: Gamma shape and target Poisson mean/variance.
        seed: Deterministic unit-test seed for the sampler.
    """
    nbc.set_numba_seed(seed)
    n_samples = 30_000

    samples = np.array(
        [nbc.continuous_poisson(lam) for _ in range(n_samples)]
    )

    assert np.mean(samples) == pytest.approx(lam, rel=0.03)
    assert np.var(samples, ddof=1) == pytest.approx(lam, rel=0.08)


def test_binomial_consistency() -> None:
    """Continuous binomial mean and variance match discrete binomial moments."""
    nbc.set_numba_seed(42)
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
    assert np.var(continuous_samples, ddof=1) == pytest.approx(
        n * p * (1.0 - p),
        rel=0.08,
    )


def test_multinomial_consistency() -> None:
    """Continuous multinomial means match discrete multinomial and rows sum to n."""
    nbc.set_numba_seed(42)
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

    theoretical_covariance = n * (
        np.diag(p_array) - np.outer(p_array, p_array)
    )
    observed_covariance = np.cov(continuous_samples, rowvar=False)
    assert np.allclose(
        observed_covariance,
        theoretical_covariance,
        rtol=0.12,
        atol=0.5,
    )


def test_small_n_cases() -> None:
    """Small-n continuous binomial moments match the defined surrogate."""
    nbc.set_numba_seed(42)
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
        expected_variance = 0.0 if n == 1 else n * p * (1.0 - p)
        assert np.var(continuous_samples, ddof=1) == pytest.approx(
            expected_variance,
            abs=1e-15 if n == 1 else 0.0,
            rel=0.12 if n > 1 else 0.0,
        )
