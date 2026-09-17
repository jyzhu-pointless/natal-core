"""R4-10: the Ricker stability boundary.

The engine's Ricker curve is ``g(x) = r ** (1 - x)`` on the competition
ratio, so the per-generation map is Ricker's ``N' = N exp(a (1 - N/K))``
with ``a = ln r`` (Ricker 1954; May 1976).  The classical stability
boundary is ``a = 2``, i.e. ``r = e^2 = 7.389056...``: below it the fixed
point N = K is attracting, above it the attractor is the stable 2-cycle.

The boundary is bracketed from both sides at r = 7.2 / 7.6, and the
r = 7.6 attractor is checked against the roots of ``f(f(N)) = N``.

The ``g_ricker`` comment's "oscillates around equilibrium for ``r > e``"
describes the earlier (``a = 1``) onset of damped oscillation, which the
engine does exhibit for ``e < r < e^2``; the wording was reviewed and kept
in the 2026-09-17 round, so no documentation assertion lives here.
"""

from __future__ import annotations

import pytest
from scipy.optimize import brentq

from _helpers_r4 import adult_total, discrete_pop, species_locus

K = 1000.0


def _pop(name: str, *, r: float, adults: float = 100.0):
    return discrete_pop(
        name,
        species=species_locus(f"R4_10_{name}", ["W"]),
        female={"W|W": adults / 2.0},
        male={"W|W": adults / 2.0},
        eggs_per_female=4.0,
        growth_mode="ricker",
        carrying_capacity=K,
        low_density_growth_rate=r,
    )


def _ricker_map(n: float, r: float) -> float:
    return n * r ** (1.0 - n / K)


def _two_cycle_roots(r: float) -> tuple[float, float]:
    def g(n: float) -> float:
        return _ricker_map(_ricker_map(n, r), r) - n

    # Linear grid over [1, 3K]: fine enough (spacing ~0.015) to bracket the
    # two extra roots of f(f(N)) = N even just above the bifurcation.
    grid = [1.0 + i * (3.0 * K - 1.0) / 200000.0 for i in range(200001)]
    roots = []
    for a, b in zip(grid, grid[1:]):
        if g(a) * g(b) < 0.0:
            roots.append(brentq(g, a, b, xtol=1e-14, rtol=1e-14))
    cycle = sorted(root for root in roots if abs(root - K) > 1e-6 * K)
    assert len(cycle) == 2, cycle
    return cycle[0], cycle[1]


class TestBoundaryIsAtE2:
    def test_just_below_the_boundary_is_stable(self) -> None:
        """r = 7.2 < e^2: the fixed point is still attracting."""
        r = 7.2
        assert r < 7.3890560989306504  # e^2
        pop = _pop("below", r=r)
        pop.run(800)
        samples = []
        for _ in range(6):
            pop.run(1)
            samples.append(adult_total(pop))
        assert max(samples) - min(samples) < 1e-6, samples
        assert samples[0] == pytest.approx(K, rel=1e-6)

    def test_just_above_the_boundary_is_a_two_cycle(self) -> None:
        """r = 7.6 > e^2: a stable 2-cycle, the roots of f(f(N)) = N."""
        r = 7.6
        assert r > 7.3890560989306504  # e^2
        low, high = _two_cycle_roots(r)
        assert low < K < high
        pop = _pop("above", r=r)
        pop.run(800)
        samples = []
        for _ in range(6):
            pop.run(1)
            samples.append(adult_total(pop))
        even, odd = samples[0::2], samples[1::2]
        assert max(even) - min(even) < 1e-6, even
        assert max(odd) - min(odd) < 1e-6, odd
        assert {round(even[0], 6), round(odd[0], 6)} == {
            round(low, 6),
            round(high, 6),
        }
