"""R3-01: Beverton-Holt and Ricker recursions, fixed points, stability.

Requirement under attack (density regulation, `growth_mode`):
the discrete-generation engine applies the juvenile scaling factor to the
newborn cohort *before* ordinary age-0 survival, and the equilibrium
calibration (C*, s*) is defined so that the per-generation map reduces to
the textbook stock-recruitment recursions.

Independent reference (literature, not this codebase):

- Beverton-Holt (Beverton & Holt 1957):
      N_{t+1} = N_t * r / (1 + (r - 1) * N_t / K)
  with the analytic fixed point N* = K, globally stable for r > 1, and a
  *monotone* approach (the map is increasing and concave, so it never
  overshoots for N_0 > 0).
- Ricker (Ricker 1954; May 1976, Nature 261:459):
      N_{t+1} = N_t * exp(a * (1 - N_t / K))
  natal's curve ``g(x) = r ** (1 - x)`` is exactly this map with
  a = ln r, so the classical stability boundary a = 2 becomes r = e^2
  and the attractor above it is the stable 2-cycle, i.e. the non-trivial
  roots of f(f(N)) = N.

What a wrong result would look like: overshoot in Beverton-Holt
(mis-ordered scaling/survival), a fixed point different from K
(C* or s* mis-derived), convergence instead of a 2-cycle at r = 10
(curve or growth-rate mis-wired), or a cycle that does not satisfy
f(f(N)) = N for a = ln r.
"""

from __future__ import annotations

import math

import pytest
from scipy.optimize import brentq

from _helpers_r3 import adult_total, discrete_pop, species_locus

CONVERGE_TICKS = 250


def _neutral(name: str, *, growth: str, r: float, K: float, adults: float):
    return discrete_pop(
        name,
        species=species_locus(f"R3_01_{name}", ["W", "D"]),
        female={"W|W": adults / 2.0},
        male={"W|W": adults / 2.0},
        eggs_per_female=4.0,
        growth_mode=growth,
        carrying_capacity=K,
        low_density_growth_rate=r,
    )


def _bh_map(n: float, r: float, k: float) -> float:
    return n * r / (1.0 + (r - 1.0) * n / k)


def _ricker_map(n: float, r: float, k: float) -> float:
    return n * r ** (1.0 - n / k)


def _two_cycle_roots(r: float, K: float) -> tuple[float, float]:
    """Non-trivial roots of f(f(N)) = N for the Ricker map, low then high."""
    lo_guess, hi_guess = 0.02 * K, 4.0 * K

    def g(n: float) -> float:
        return _ricker_map(_ricker_map(n, r, K), r, K) - n

    roots = []
    grid = [lo_guess * (hi_guess / lo_guess) ** (i / 40000.0) for i in range(40001)]
    for a, b in zip(grid, grid[1:]):
        fa, fb = g(a), g(b)
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            roots.append(brentq(g, a, b, xtol=1e-15, rtol=1e-15))
    # Drop the fixed point N = K and keep the two cycle members.
    cycle = sorted(root for root in roots if abs(root - K) > 1e-6 * K)
    assert len(cycle) == 2, cycle
    return cycle[0], cycle[1]


@pytest.mark.parametrize("r", [1.5, 3.0, 8.0])
@pytest.mark.parametrize("start", [0.05, 0.5, 3.0])
def test_beverton_holt_matches_closed_form_and_converges_to_k(r: float, start: float) -> None:
    """Trajectory equals the BH recursion and converges monotonically to K."""
    K = 1000.0
    n0 = start * K
    pop = _neutral(f"bh_{r}_{start}", growth="beverton_holt", r=r, K=K, adults=n0)

    expected = [n0]
    observed = [adult_total(pop)]
    assert observed[0] == pytest.approx(n0, rel=1e-12)
    for _ in range(CONVERGE_TICKS):
        pop.run(1)
        expected.append(_bh_map(expected[-1], r, K))
        observed.append(adult_total(pop))

    for got, want in zip(observed, expected):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9)

    # Monotone approach: the sign of (N - K) never changes.
    tail = expected[1:]
    if n0 < K:
        assert all(b >= a - 1e-12 for a, b in zip(tail, tail[1:]))
    else:
        assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:]))
    assert abs(expected[-1] - K) < 1e-6, expected[-1]


def test_ricker_below_boundary_is_stable_above_is_two_cycle() -> None:
    """Ricker map with a = ln r: stable fixed point for a < 2, 2-cycle above.

    a = ln(5) = 1.609 -> damped convergence to K.
    a = ln(10) = 2.303 -> the fixed point is unstable; the attractor is the
    period-2 orbit, the non-trivial roots of f(f(N)) = N.
    """
    K = 1000.0

    # --- below the boundary: converge to K ---
    pop = _neutral("ricker_low", growth="ricker", r=5.0, K=K, adults=100.0)
    pop.run(CONVERGE_TICKS)
    assert adult_total(pop) == pytest.approx(K, rel=1e-6)

    # --- above the boundary: a two-cycle, located independently ---
    r_high = 10.0
    cycle_low, cycle_high = _two_cycle_roots(r_high, K)
    assert cycle_low < K < cycle_high

    pop2 = _neutral("ricker_high", growth="ricker", r=r_high, K=K, adults=100.0)
    pop2.run(600)
    samples = []
    for _ in range(6):
        pop2.run(1)
        samples.append(adult_total(pop2))
    even, odd = samples[0::2], samples[1::2]
    assert max(even) - min(even) < 1e-6, even
    assert max(odd) - min(odd) < 1e-6, odd
    assert {round(even[0], 6), round(odd[0], 6)} == {
        round(cycle_low, 6),
        round(cycle_high, 6),
    }


def test_ricker_two_cycle_is_reached_from_both_sides() -> None:
    """The period-2 attractor does not depend on the starting population."""
    K = 1000.0
    r = 12.0  # a = ln 12 = 2.48, inside the stable two-cycle band
    cycle_low, cycle_high = _two_cycle_roots(r, K)
    for label, start in (("low", 10.0), ("high", 5000.0)):
        pop = _neutral(f"ricker_both_{label}", growth="ricker", r=r, K=K, adults=start)
        pop.run(600)
        values = []
        for _ in range(4):
            pop.run(1)
            values.append(adult_total(pop))
        # Phase depends on the starting side; the attractor set must not.
        assert {
            round(values[0::2][0], 6),
            round(values[1::2][0], 6),
        } == {round(cycle_low, 6), round(cycle_high, 6)}
    assert math.isfinite(cycle_low) and cycle_low > 0.0
