"""R3-07: equilibrium calibration vs the declared carrying capacity K.

Contract (fixed 2026-09-16): the derived reference distribution splits the
age-1 total by the *surviving* sex ratio, so the deterministic equilibrium is
exactly the declared ``carrying_capacity`` for every compensatory curve even
when the sexes differ in age-0 survival.  Before the fix the split used the raw
offspring ``sex_ratio``, and the realized equilibrium missed K by up to 37.5%
(+2.8% at s_f=0.9/s_m=0.8/r=3, +15.6% at s_f=0.8/s_m=0.3/r=3, +37.5% at
sex_ratio=0.4/s_f=0.8/s_m=0.3/r=2, and -0.6% for ricker at the same parameters)
in both engines.  The permanent Python regressions for this contract live in
tests/test_density_growth_validation.py; these probes keep the same coverage
next to the rest of the round-3 evidence.

Wrong results rejected: a reference composition that ignores per-sex age-0
survival, a curve whose calibrated fixed point is not its own g(1) = 1 point,
and asymmetry between the discrete and age-structured calibration cores.
"""

from __future__ import annotations

import pytest

from _helpers_r3 import adult_total, age_pop, discrete_pop, female_age_counts, male_age_counts, species_locus


def _build_discrete(name: str, *, sex_ratio: float, s_f: float, s_m: float, r: float,
                    growth: str, K: float = 2000.0):
    population = discrete_pop(
        name,
        species=species_locus(f"{name}_sp", ["W", "D"]),
        female={"W|W": K / 2.0},
        male={"W|W": K / 2.0},
        eggs_per_female=6.0,
        sex_ratio=sex_ratio,
        survival=0.5,
        growth_mode=growth,
        carrying_capacity=K,
        low_density_growth_rate=r,
    )
    population.update().survival(female_age0_survival=s_f, male_age0_survival=s_m)
    return population


def test_equal_sex_survival_reaches_k_exactly() -> None:
    """Control: with s_f == s_m the fixed point is exactly K for both curves."""
    K = 2000.0
    for growth in ("beverton_holt", "ricker"):
        pop = _build_discrete(f"R3_07_equal_{growth}", sex_ratio=0.4, s_f=0.6, s_m=0.6,
                              r=3.0, growth=growth, K=K)
        pop.run(400)
        assert adult_total(pop) == pytest.approx(K, rel=1e-9), growth


@pytest.mark.parametrize(
    "sex_ratio,s_f,s_m,r,growth,pre_fix_equilibrium",
    [
        (0.5, 0.9, 0.8, 3.0, "beverton_holt", 2055.5556),
        (0.5, 0.8, 0.3, 3.0, "beverton_holt", 2312.5),
        (0.4, 0.8, 0.3, 2.0, "beverton_holt", 2750.0),
        (0.5, 0.9, 0.8, 3.0, "ricker", 1987.1637),
    ],
)
def test_equilibrium_is_declared_k_under_sex_specific_juvenile_survival(
    sex_ratio: float, s_f: float, s_m: float, r: float, growth: str,
    pre_fix_equilibrium: float,
) -> None:
    """The deterministic equilibrium equals K for any per-sex survival pair.

    ``pre_fix_equilibrium`` records what the raw-sex-ratio split produced for
    this parameter set (K = 2000), so a regression that reintroduces the
    composition-blind reference is named by the number it lands on.
    """
    K = 2000.0
    pop = _build_discrete(
        f"R3_07_asym_{growth}_{sex_ratio}_{s_f}_{s_m}_{r}",
        sex_ratio=sex_ratio,
        s_f=s_f,
        s_m=s_m,
        r=r,
        growth=growth,
        K=K,
    )
    pop.run(400)
    realized = adult_total(pop)
    assert realized == pytest.approx(K, rel=1e-9), (
        f"{growth} with sex_ratio={sex_ratio}, s_f={s_f}, s_m={s_m}, r={r}: "
        f"realized {realized:.6f} (K={K}; the pre-fix split gave "
        f"{pre_fix_equilibrium:.4f})"
    )


def test_age_structured_equilibrium_is_declared_k_under_sex_specific_survival() -> None:
    """Age-structured engine: the age-1 total is exactly K.

    Three-age life table, adults at ages 1 and 2, Beverton-Holt with K = 2000.
    With s_f = 0.8, s_m = 0.3 the pre-fix calibration settled at 2312.5.
    """
    K = 2000.0
    sex_ratio, s_f, s_m, r = 0.5, 0.8, 0.3, 3.0
    pop = age_pop(
        "R3_07_age_asym",
        species=species_locus("R3_07_age_asym_sp", ["W", "D"]),
        n_ages=3,
        new_adult_age=1,
        initial={"female": {"W|W": {1: 500.0}}, "male": {"W|W": {1: 500.0}}},
        survival_f=[s_f, s_f, 0.5],
        survival_m=[s_m, s_m, 0.5],
        mating_f=[0.0, 1.0, 1.0],
        mating_m=[0.0, 1.0, 1.0],
        eggs_per_female=20.0,
        sex_ratio=sex_ratio,
        growth_mode="beverton_holt",
        carrying_capacity=K,
        low_density_growth_rate=r,
    )
    pop.run(600)
    age1 = float(female_age_counts(pop)[1] + male_age_counts(pop)[1])
    assert age1 == pytest.approx(K, rel=1e-9), (
        f"declared K={K} but the age-1 total equilibrates at {age1:.4f} "
        f"({age1 / K:.4f} x K)"
    )
