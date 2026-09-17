"""P8: density regulation -- Beverton-Holt closed form and guards.

Claim (compensatory modes, derived equilibrium, discrete generations,
equal sex survival s, neutral genetics): the deterministic adult total N
follows the classic discrete Beverton-Holt map::

    N_{t+1} = N_t * r / (1 + (r - 1) * N_t / K)

because the calibrated equilibrium survival factor s* = K / (C*_eggs * s)
exactly cancels the ordinary survival thinning at recruitment.  N = K is
an exact fixed point.  The ``fixed`` mode instead caps juveniles at K
*before* survival: N_{t+1} = min(N_f * b, K) * s.  A zero equilibrium
reference (K = 0) must extinguish compensatory recruitment, not explode.

Reference: the Beverton-Holt stock-recruitment map, derived here from the
documented equilibrium calibration (C* = K * sex_ratio * eggs, s* =
K / (produced_eggs * s0_avg)), not from library tables.

Wrong results rejected: an off-by-factor recruitment curve, a K-excursion
at the fixed point, population growth without habitat (the pre-2026-09-14
D2 defect), ricker/logistic parameter misuse.
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population

TOL = 1e-8


def _adult_total(pop):
    return float(np.asarray(pop.state.individual_count[:, 1, :]).sum())


def _beverton_holt_step(n, r, k):
    return n * r / (1.0 + (r - 1.0) * n / k)


def test_beverton_holt_closed_form_on_randomized_grid() -> None:
    rng = np.random.RandomState(20260915)
    cases = []
    for _ in range(8):
        r = float(rng.uniform(1.2, 15.0))
        k = float(rng.uniform(100.0, 4000.0))
        n0 = float(rng.uniform(10.0, 2.5 * k))
        cases.append((r, k, n0))
    for i, (r, k, n0) in enumerate(cases):
        pop = neutral_population(
            qc_species_2(f"QC0915_p08a_{i}"),
            f"QC0915_p08a_{i}",
            female={"W|W": n0 / 2},
            male={"W|W": n0 / 2},
            eggs_per_female=10.0,
            survival=0.8,
            growth_mode="beverton_holt",
            carrying_capacity=k,
            low_density_growth_rate=r,
        )
        expected = n0
        for _ in range(3):
            pop.run(1)
            expected = _beverton_holt_step(expected, r, k)
            assert abs(_adult_total(pop) - expected) <= TOL * max(expected, 1.0), (
                f"r={r} K={k} N0={n0}: expected {expected}, got {_adult_total(pop)}"
            )


def test_beverton_holt_fixed_point_at_k_is_exact() -> None:
    k = 2000.0
    pop = neutral_population(
        qc_species_2("QC0915_p08b"),
        "QC0915_p08b",
        female={"W|W": k / 2},
        male={"W|W": k / 2},
        eggs_per_female=10.0,
        survival=0.8,
        growth_mode="beverton_holt",
        carrying_capacity=k,
        low_density_growth_rate=6.0,
    )
    for _ in range(3):
        pop.run(1)
        assert abs(_adult_total(pop) - k) < 1e-6


def test_fixed_mode_caps_juveniles_before_survival() -> None:
    s, b = 0.8, 10.0
    # Below cap: E = 500 * 10 = 5000 < K -> N1 = 5000 * 0.8 = 4000.
    pop = neutral_population(
        qc_species_2("QC0915_p08c1"),
        "QC0915_p08c1",
        female={"W|W": 500},
        male={"W|W": 500},
        eggs_per_female=b,
        survival=s,
        growth_mode="fixed",
        carrying_capacity=10000.0,
    )
    pop.run(1)
    assert abs(_adult_total(pop) - 4000.0) < TOL
    # Above cap: E = 5000 > K = 2000 -> N1 = 2000 * 0.8 = 1600.
    pop2 = neutral_population(
        qc_species_2("QC0915_p08c2"),
        "QC0915_p08c2",
        female={"W|W": 500},
        male={"W|W": 500},
        eggs_per_female=b,
        survival=s,
        growth_mode="fixed",
        carrying_capacity=2000.0,
    )
    pop2.run(1)
    assert abs(_adult_total(pop2) - 1600.0) < TOL


def test_zero_capacity_extinguishes_compensatory_recruitment() -> None:
    # Regression guard for the 2026-09-14 D2 defect: with no habitat the
    # compensatory modes must collapse recruitment to zero.
    for mode in ("beverton_holt", "logistic", "ricker"):
        pop = neutral_population(
            qc_species_2(f"QC0915_p08d_{mode}"),
            f"QC0915_p08d_{mode}",
            female={"W|W": 1000},
            male={"W|W": 1000},
            eggs_per_female=10.0,
            growth_mode=mode,
            carrying_capacity=0.0,
            low_density_growth_rate=10.0,
        )
        pop.run(1)
        assert _adult_total(pop) == 0.0, mode


def test_tiny_capacity_does_not_explode() -> None:
    pop = neutral_population(
        qc_species_2("QC0915_p08e"),
        "QC0915_p08e",
        female={"W|W": 1000},
        male={"W|W": 1000},
        eggs_per_female=10.0,
        growth_mode="beverton_holt",
        carrying_capacity=1e-9,
        low_density_growth_rate=10.0,
    )
    for _ in range(3):
        pop.run(1)
        assert _adult_total(pop) < 10.0
