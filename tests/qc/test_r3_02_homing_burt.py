"""R3-02: homing-drive spread recursion and its cost structure.

Literature model under attack (Burt 2003, Proc R Soc B 270:921;
Deredec, Burt & Godfray 2008, Genetics 179:2013;
Unckless, Messer, Connallon & Clark 2015, Genetics 201:425):

- A dominant drive allele ``D`` converts the wild-type ``W`` allele in
  heterozygotes during gametogenesis with efficiency ``e``, so a ``W|D``
  parent transmits ``D`` with probability ``(1 + e) / 2``.
- With genotype viabilities ``w_WW = 1``, ``w_WD = 1 - h s``,
  ``w_DD = 1 - s`` and random mating, the gamete-pool drive frequency one
  generation after gamete frequency ``q`` is
      F(q) = [ q^2 (1 - s) + q (1 - q) (1 - h s) (1 + e) ] / Wbar,
      Wbar = (1 - q)^2 + 2 q (1 - q) (1 - h s) + q^2 (1 - s).
- Cost-free (h = s = 0): F(q) = q + e q (1 - q)  -> supra-Mendelian and
  threshold-free, the drive fixes from any positive frequency.
- Homozygous cost only (h = 0, 0 < e < s): F(q) - q = q s (q - e/s)
  (q - 1) / (1 - s q^2), so the *gamete* frequency converges to
  q* = e / s exactly from either side and the drive never fixes.  The
  observed adult genotype frequencies at that equilibrium are the
  selection-weighted Hardy-Weinberg values, i.e. exact rationals:
      W|W = p^2 / Wbar,   W|D = 2 p (1 - p) / Wbar,   D|D = p^2 (1 - s) / Wbar
  with p = e/s and Wbar = 1 - s p^2.
- Dominant cost (h = 1, e < s): there *is* an unstable interior
  equilibrium (the classic threshold).  For s = 0.5, e = 0.3 it solves
      0.5 q^2 - 0.85 q + 0.35 = 0   ->   q* = 0.7 (gamete frequency).

Wrong results rejected: homing applied to homozygotes as well as
heterozygotes, conversion applied per individual rather than per target
copy, the cost applied before rather than after gamete formation, a drive
with no heterozygous cost mysteriously acquiring a fixation threshold,
or a heterozygous-cost drive crossing its threshold unnoticed.
"""

from __future__ import annotations

import pytest
from scipy.optimize import brentq

from _helpers_r3 import discrete_pop, genotype_masses, species_locus
from natal.frontend.presets.homing import HomingDrive


def _drive_pop(
    name: str,
    *,
    e: float,
    s: float = 0.0,
    h: float = 0.0,
    female: dict[str, float],
    male: dict[str, float],
    eggs: float = 2.0,
):
    drive = HomingDrive(
        name=f"{name}_drive",
        drive_allele="D",
        target_allele="W",
        drive_conversion_rate=e,
    )
    weights = {"W|W": 1.0, "W|D": 1.0 - h * s, "D|D": 1.0 - s}

    def extra(builder):
        builder = builder.presets(drive)
        if s > 0.0:
            builder = builder.fitness(
                viability={
                    genotype: {"female": value, "male": value}
                    for genotype, value in weights.items()
                    if value != 1.0
                }
            )
        return builder

    return discrete_pop(
        name,
        species=species_locus(f"R3_02_{name}", ["W", "D"]),
        female=female,
        male=male,
        eggs_per_female=eggs,
        growth_mode="no_competition",
        extra=extra,
    )


def _freqs(pop) -> dict[str, float]:
    masses = genotype_masses(pop)
    total = sum(masses.values())
    return {key: value / total for key, value in masses.items()}


def _gamete_q(freq: dict[str, float], e: float) -> float:
    """Drive frequency in the gamete pool (exact for any genotype mix)."""
    return freq["D|D"] + freq["W|D"] * (1.0 + e) / 2.0


def _adult_q(freq: dict[str, float]) -> float:
    return freq["D|D"] + 0.5 * freq["W|D"]


def _hw(q: float) -> dict[str, float]:
    return {"W|W": (1.0 - q) ** 2, "W|D": 2.0 * q * (1.0 - q), "D|D": q**2}


def test_cost_free_drive_follows_logistic_closed_form() -> None:
    """e = 0.8, no cost: q' = q + e q (1 - q) and fixation follows."""
    e = 0.8
    q0 = 0.01
    pop = _drive_pop(
        "free",
        e=e,
        female={"W|W": 1000.0 * (1 - q0) ** 2, "W|D": 1000.0 * 2 * q0 * (1 - q0),
                "D|D": 1000.0 * q0**2},
        male={"W|W": 1000.0 * (1 - q0) ** 2, "W|D": 1000.0 * 2 * q0 * (1 - q0),
              "D|D": 1000.0 * q0**2},
    )
    q = _adult_q(_freqs(pop))
    assert abs(q - q0) < 1e-12

    for generation in range(20):
        pop.run(1)
        freq = _freqs(pop)
        q_next = _adult_q(freq)
        # The HW adult population transmits the supra-Mendelian gamete pool,
        # so the next adult frequency is q + e q (1 - q).
        assert q_next == pytest.approx(q + e * q * (1.0 - q), abs=1e-9), generation
        # With no selection the adults stay Hardy-Weinberg in that frequency.
        for genotype, expected in _hw(q_next).items():
            assert freq[genotype] == pytest.approx(expected, abs=1e-9), (generation, genotype)
        q = q_next

    assert q > 0.9999  # fixation, no interior equilibrium


@pytest.mark.parametrize("side", ["below", "above"])
def test_homozygous_cost_equilibrium_is_exact_rational(side: str) -> None:
    """h = 0, e = 0.3, s = 0.5: gamete frequency -> e/s = 0.6 exactly.

    p = e/s = 0.6, Wbar = 1 - 0.5 * 0.36 = 0.82, so the observed adult
    frequencies are 0.16/0.82 = 8/41, 0.48/0.82 = 24/41 and
    0.18/0.82 = 9/41 -- from either side of the equilibrium.
    """
    e, s = 0.3, 0.5
    composition = {
        "below": {"W|W": 0.5, "W|D": 0.5, "D|D": 0.0},  # gamete q = 0.325
        "above": {"W|W": 0.0, "W|D": 0.6, "D|D": 0.4},  # gamete q = 0.79
    }[side]
    counts = {key: 1000.0 * value for key, value in composition.items()}
    pop = _drive_pop(
        f"h0_{side}", e=e, s=s, h=0.0, female=dict(counts), male=dict(counts), eggs=4.0
    )

    p = e / s
    w_bar = 1.0 - s * p**2
    expected = {
        "W|W": (1.0 - p) ** 2 / w_bar,
        "W|D": 2.0 * p * (1.0 - p) / w_bar,
        "D|D": p**2 * (1.0 - s) / w_bar,
    }
    assert expected["W|W"] == pytest.approx(8.0 / 41.0, rel=1e-12)

    pop.run(400)
    freq = _freqs(pop)
    for genotype, value in expected.items():
        assert freq[genotype] == pytest.approx(value, abs=1e-10), genotype
    # The gamete frequency at the fixed point is exactly e/s.
    assert _gamete_q(freq, e) == pytest.approx(e / s, abs=1e-10)


@pytest.mark.parametrize("side", ["below", "above"])
def test_dominant_cost_drive_has_a_threshold(side: str) -> None:
    """h = 1, e = 0.3, s = 0.5: unstable equilibrium at gamete q* = 0.7.

    The threshold solves the literature fixed-point equation
    (1 - s)(1 + e - e q) = 1 - s (2 q - q^2), i.e. for these values
    0.5 q^2 - 0.85 q + 0.35 = 0, whose in-range root is q* = 0.7.
    Below it the drive is lost; above it the drive fixes.
    """
    e, s, h = 0.3, 0.5, 1.0

    def residual(q: float) -> float:
        w_bar = (1 - q) ** 2 + 2 * q * (1 - q) * (1 - h * s) + q**2 * (1 - s)
        return w_bar - (1 - s) * (1 + e - e * q)

    q_star = brentq(residual, 0.2, 0.95, xtol=1e-15)
    assert q_star == pytest.approx(0.7, abs=1e-12)
    # Cross-check against the explicit quadratic for these parameters.
    roots = sorted(
        (
            (0.85 + sign * (0.85**2 - 4 * 0.5 * 0.35) ** 0.5) / (2 * 0.5)
            for sign in (1.0, -1.0)
        )
    )
    # The two roots are the threshold q* = 0.7 and the trivial q = 1.
    assert roots[0] == pytest.approx(q_star, abs=1e-12)
    assert roots[1] == pytest.approx(1.0, abs=1e-12)

    composition = {
        "below": {"W|W": 0.0, "W|D": 1.0, "D|D": 0.0},  # gamete q = 0.65
        "above": {"W|W": 0.0, "W|D": 0.6, "D|D": 0.4},  # gamete q = 0.79
    }[side]
    counts = {key: 1000.0 * value for key, value in composition.items()}
    pop = _drive_pop(
        f"h1_{side}", e=e, s=s, h=h, female=dict(counts), male=dict(counts), eggs=4.0
    )
    q0 = _gamete_q(_freqs(pop), e)
    assert (q0 < q_star) if side == "below" else (q0 > q_star)

    pop.run(400)
    q_final = _gamete_q(_freqs(pop), e)
    if side == "below":
        assert q_final < 1e-6, q_final
    else:
        assert q_final > 1.0 - 1e-6, q_final


def test_full_drive_above_cost_always_fixes() -> None:
    """e = 0.8 > s = 0.4 with h = 0: no interior equilibrium, drive fixes."""
    e, s = 0.8, 0.4
    counts = {"W|W": 400.0, "W|D": 600.0, "D|D": 0.0}
    pop = _drive_pop("above_cost", e=e, s=s, h=0.0, female=dict(counts), male=dict(counts), eggs=4.0)
    pop.run(300)
    freq = _freqs(pop)
    assert _adult_q(freq) > 1.0 - 1e-6
    assert freq["D|D"] > 1.0 - 1e-6
