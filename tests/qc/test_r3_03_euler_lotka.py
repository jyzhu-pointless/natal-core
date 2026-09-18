"""R3-03: age-structured life table -> Leslie/Euler-Lotka reproduction.

Literature reference (Euler 1760; Leslie 1945; standard in the
age-structured mosquito models of Beaghton/Godfray and North/Burt):

For a pre-breeding census with ages 0..A-1, adult from age 1, and female
life table ``l_x = prod_{i<x} s_i`` with per-capita female offspring
``F_x`` at age x, the female projection matrix

    N_1(t+1) = s_0 * sum_x F_x N_x(t),   N_{x+1}(t+1) = s_x N_x(t)

has dominant eigenvalue lambda solving the Euler-Lotka equation
``sum_x l_x F_x lambda**(-x) = 1``, and the stable age distribution
``N_{x+1} / N_x = s_x / lambda``.

With A = 3, ``s_0 = 0.9``, ``s_1 = 0.7``, ``F_1 = F_2 = eggs * sex_ratio``
and ``eggs = 8, sex_ratio = 0.5`` (so F = 4) the equation
``lambda^2 = 3.6 lambda + 2.52`` has the exact root lambda = 4.2 and the
exact stable ratio N_2/N_1 = s_1/lambda = 1/6.

Wrong results rejected: a reproduction or survival stage read at the
wrong age, aging in the wrong direction or dropping the wrong cohort,
the sex ratio applied on the wrong side of the life cycle, density
regulation leaking into the exponential phase, or a growth rate that
does not satisfy Euler-Lotka.
"""

from __future__ import annotations

import math

import pytest

from _helpers_r3 import age_pop, female_age_counts, male_age_counts, species_locus

TOL = 1e-9


def _life_table_pop(name: str, *, s_f, s_m, eggs, mating_on: bool = True):
    """Three-age female life table; adults at ages 1 and 2."""
    return age_pop(
        name,
        species=species_locus(f"R3_03_{name}", ["W", "D"]),
        n_ages=3,
        new_adult_age=1,
        initial={"female": {"W|W": {1: 1000.0}}, "male": {"W|W": {1: 1000.0}}},
        survival_f=list(s_f),
        survival_m=list(s_m),
        mating_f=[0.0, 1.0, 1.0] if mating_on else [0.0, 0.0, 0.0],
        mating_m=[0.0, 1.0, 1.0] if mating_on else [0.0, 0.0, 0.0],
        eggs_per_female=eggs,
        sex_ratio=0.5,
        extra=None,
    )


def test_lambda_is_the_euler_lotka_root_and_matches_simulation() -> None:
    """s = (0.9, 0.7, .), F = 4: lambda = 4.2 exactly, N_2/N_1 = 1/6."""
    s0, s1 = 0.9, 0.7
    f = 4.0
    # Euler-Lotka closed form for the two-adult-age life table:
    # lambda^2 - s0 F lambda - s0 s1 F = 0.
    lam = (s0 * f + math.sqrt(s0**2 * f**2 + 4.0 * s0 * s1 * f)) / 2.0
    assert lam == pytest.approx(4.2, rel=1e-15)
    # The defining equation holds to machine precision.
    assert abs(s0 * f / lam + s0 * s1 * f / lam**2 - 1.0) < 1e-15

    pop = _life_table_pop("lotka", s_f=(s0, s1, 0.0), s_m=(s0, s1, 0.0), eggs=8.0)

    totals: list[float] = []
    for _ in range(40):
        pop.run(1)
        totals.append(float(female_age_counts(pop).sum()))

    # Asymptotic growth ratio equals the dominant eigenvalue.
    for earlier, later in zip(totals[-6:-1], totals[-5:]):
        assert later / earlier == pytest.approx(lam, rel=1e-9)

    # Stable age distribution: N_2/N_1 = s_1 / lambda = 1/6 exactly.
    ages = female_age_counts(pop)
    assert ages[0] == 0.0  # newborns are cleared by aging every tick
    assert ages[2] / ages[1] == pytest.approx(s1 / lam, rel=1e-9)
    assert ages[2] / ages[1] == pytest.approx(1.0 / 6.0, rel=1e-9)

    # And the measured growth rate satisfies Euler-Lotka itself.
    measured = totals[-1] / totals[-2]
    assert abs(s0 * f / measured + s0 * s1 * f / measured**2 - 1.0) < 1e-9


def test_life_table_is_recomputed_for_a_second_survival_set() -> None:
    """A different life table moves lambda to its own Euler-Lotka root."""
    s0, s1 = 0.5, 0.4
    f = 3.0
    lam_expected = (s0 * f + math.sqrt(s0**2 * f**2 + 4.0 * s0 * s1 * f)) / 2.0
    # lambda = (1.5 + sqrt(2.25 + 1.8)) / 2 = (1.5 + 2.01246...) / 2
    pop = _life_table_pop("lotka2", s_f=(s0, s1, 0.0), s_m=(s0, s1, 0.0), eggs=6.0)
    female = []
    for _ in range(60):
        pop.run(1)
        female.append(float(female_age_counts(pop).sum()))
    measured = female[-1] / female[-2]
    assert measured == pytest.approx(lam_expected, rel=1e-9)
    assert abs(s0 * f / measured + s0 * s1 * f / measured**2 - 1.0) < 1e-9


def test_cohort_survival_ladder_and_oldest_age_drop() -> None:
    """With reproduction off, a cohort walks the survival ladder and drops out.

    n_ages = 4, cohort of 1000 age-1 females: after t ticks the survivors
    sit at age 1+t with mass 1000 * prod_{i=1..t} s_i, and past the oldest
    class they are gone.  Male counts follow the male survival row.
    """
    s_f = (0.8, 0.9, 0.7, 0.5)
    s_m = (0.6, 0.5, 0.4, 0.3)
    pop = age_pop(
        "cohort",
        species=species_locus("R3_03_cohort", ["W", "D"]),
        n_ages=4,
        new_adult_age=1,
        initial={"female": {"W|W": {1: 1000.0}}, "male": {"W|W": {1: 1000.0}}},
        survival_f=list(s_f),
        survival_m=list(s_m),
        mating_f=[0.0, 0.0, 0.0, 0.0],
        mating_m=[0.0, 0.0, 0.0, 0.0],
        eggs_per_female=10.0,
    )

    n_ages = 4
    alive_f = alive_m = 1000.0
    age = 1  # cohort position at the start of the next tick
    for tick in range(1, 5):
        if age < n_ages:
            alive_f *= s_f[age]
            alive_m *= s_m[age]
            age += 1
        else:  # passed the oldest class in the previous tick
            alive_f = alive_m = 0.0
        pop.run(1)
        expected_f = [0.0] * n_ages
        expected_m = [0.0] * n_ages
        if alive_f > 0.0 and age < n_ages:
            expected_f[age] = alive_f
            expected_m[age] = alive_m
        females = female_age_counts(pop)
        males = male_age_counts(pop)
        # Age 0 stays empty: newborns only appear through reproduction.
        assert females[0] == 0.0 and males[0] == 0.0
        for index in range(n_ages):
            assert females[index] == pytest.approx(expected_f[index], abs=1e-9), (tick, index)
            assert males[index] == pytest.approx(expected_m[index], abs=1e-9), (tick, index)
        assert females.sum() == pytest.approx(sum(expected_f), abs=1e-9)

    assert female_age_counts(pop).sum() == 0.0
    assert male_age_counts(pop).sum() == 0.0


def test_sex_ratio_scales_the_female_offspring_side_only() -> None:
    """Halving sex_ratio halves female recruitment and leaves males untouched.

    The male age-1 count is fed by the *male* share of eggs, so with no
    sex-chromosome system it must be (1 - sex_ratio) / sex_ratio times the
    female count of the same age cohort.
    """
    s0 = 0.9
    for sex_ratio in (0.5, 0.25):
        name = f"sr_{sex_ratio}"
        pop = age_pop(
            name,
            species=species_locus(f"R3_03_{name}", ["W", "D"]),
            n_ages=3,
            new_adult_age=1,
            initial={"female": {"W|W": {1: 1000.0}}, "male": {"W|W": {1: 1000.0}}},
            survival_f=[s0, 0.7, 0.0],
            survival_m=[s0, 0.7, 0.0],
            mating_f=[0.0, 1.0, 1.0],
            mating_m=[0.0, 1.0, 1.0],
            eggs_per_female=8.0,
            sex_ratio=sex_ratio,
        )
        pop.run(1)
        females = female_age_counts(pop)
        males = male_age_counts(pop)
        expected_f = 1000.0 * 8.0 * sex_ratio * s0
        expected_m = 1000.0 * 8.0 * (1.0 - sex_ratio) * s0
        assert females[1] == pytest.approx(expected_f, rel=1e-12)
        assert males[1] == pytest.approx(expected_m, rel=1e-12)
        assert males[1] / females[1] == pytest.approx(
            (1.0 - sex_ratio) / sex_ratio, rel=1e-12
        )
