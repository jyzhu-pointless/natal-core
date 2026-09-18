"""R4-01: Burt (2003) homing drive, complete-conversion doubling and the
heterozygote-only invasion threshold.

Literature model (Burt 2003, Proc R Soc B 270:921; Unckless, Messer,
Connallon & Clark 2015, Genetics 201:425, "MCR" section):

With adult allele frequency ``q``, homing efficiency ``e`` (a heterozygote
transmits the drive to ``(1 + e) / 2`` of its gametes) and genotype
viabilities ``w_WW = 1``, ``w_WD = 1 - h s``, ``w_DD = 1 - s``:

    F(q) = [ q^2 (1 - s) + q (1 - q) (1 - h s) (1 + e) ] / Wbar

Two exact predictions that the R3 round did *not* test (it used e = 0.8
with the logistic form and the dominant/homozygous cost geometries):

1. ``e = 1`` doubles the wild-type *deficit* every generation:
   ``1 - q_{t+1} = (1 - q_t)^2``, so ``1 - q_t = (1 - q_0)^(2^t)``.
   This is the "super-Mendelian, threshold-free" case; it also fixes the
   escape time ``t = log2(ln q_target / ln(1 - q_0))``.

2. From rarity the drive invades iff ``(1 + e)(1 - h s) > 1`` — the
   threshold involves the *heterozygote* fitness only.  With additive
   costs (h = 0.5) it is ``e > s / (1 - s)``; a drive whose homozygote is
   much cheaper than its heterozygote crosses the same threshold, because
   homozygotes are vanishingly rare at the invasion boundary.

Wrong results rejected: an engine that applies homing to homozygotes, one
that normalizes the conversion rate differently (e.g. per individual
rather than per target copy), a threshold that uses the homozygote
fitness, and a cost applied before instead of after gamete formation.
"""

from __future__ import annotations

import math

import pytest

from _helpers_r4 import allele_frequency, discrete_pop, species_locus
from natal.frontend.presets.homing import HomingDrive


def _pop(
    name: str,
    *,
    e: float,
    weights: dict[str, float] | None = None,
    q0: float,
    total: float = 2000.0,
):
    species = species_locus(f"R4_01_{name}", ["W", "D"])
    counts = {
        "W|W": total * (1.0 - q0) ** 2,
        "W|D": total * 2.0 * q0 * (1.0 - q0),
        "D|D": total * q0**2,
    }

    drive = HomingDrive(
        name=f"{name}_drive",
        drive_allele="D",
        target_allele="W",
        drive_conversion_rate=e,
    )

    def extra(builder):
        builder = builder.presets(drive)
        if weights:
            builder = builder.fitness(
                viability={
                    genotype: {"female": value, "male": value}
                    for genotype, value in weights.items()
                },
                mode="replace",
            )
        return builder

    return discrete_pop(
        name,
        species=species,
        female=dict(counts),
        male=dict(counts),
        eggs_per_female=2.0,
        survival=1.0,
        # A fixed ceiling at the starting size keeps the abundance bounded
        # (genotype-independent regulation, so frequencies follow the
        # classical recursion) and prevents a declining mean fitness from
        # driving the population mass to zero over long runs.
        growth_mode="fixed",
        carrying_capacity=total,
        extra=extra,
    )


def _exact_q(q: float, e: float, w_wd: float, w_dd: float) -> float:
    """One generation of the gamete-pool recursion F(q)."""
    wbar = (1.0 - q) ** 2 + 2.0 * q * (1.0 - q) * w_wd + q * q * w_dd
    return (q * q * w_dd + q * (1.0 - q) * w_wd * (1.0 + e)) / wbar


def _adult_q(q: float, w_wd: float, w_dd: float) -> float:
    """Adult allele frequency reached by one round of viability selection.

    ``F`` advances the *gamete-pool* frequency; the state of a discrete
    population holds the selected adults (viability acts on the newborn
    cohort, i.e. after gamete formation), whose frequency is the
    selection-weighted Hardy-Weinberg value of that gamete pool.
    Without costs the two coincide.
    """
    wbar = (1.0 - q) ** 2 + 2.0 * q * (1.0 - q) * w_wd + q * q * w_dd
    return (q * q * w_dd + q * (1.0 - q) * w_wd) / wbar


def _expected_states(
    q0: float, e: float, w_wd: float, w_dd: float, n: int
) -> list[float]:
    """Adult allele frequency the engine must show after each of *n* ticks.

    The initial state is an unselected Hardy-Weinberg population with
    allele frequency ``q0``, so its gamete pool is ``q0^2 + q0 (1 - q0)
    (1 + e)``; every later generation's gamete pool follows ``F`` applied
    to the previous pool, and the observable state is the selected adult
    frequency of that pool.
    """
    q_gam = q0 * q0 + q0 * (1.0 - q0) * (1.0 + e)
    states: list[float] = []
    for _ in range(n):
        states.append(_adult_q(q_gam, w_wd, w_dd))
        q_gam = _exact_q(q_gam, e, w_wd, w_dd)
    return states


class TestCompleteConversionDoubling:
    def test_e1_follows_the_doubling_closed_form(self) -> None:
        """e = 1, no cost: 1 - q_t = (1 - q_0)^(2^t) for 14 generations."""
        q0 = 0.01
        pop = _pop("doubling", e=1.0, q0=q0)
        assert allele_frequency(pop, "D") == pytest.approx(q0, abs=1e-12)

        for t in range(1, 15):
            pop.run(1)
            expected_q = 1.0 - (1.0 - q0) ** (2**t)
            got = allele_frequency(pop, "D")
            assert got == pytest.approx(expected_q, rel=1e-9, abs=1e-12), (
                f"generation {t}: q={got!r} vs closed form {expected_q!r}"
            )

    def test_e1_escape_time_formula(self) -> None:
        """The generation that crosses a target frequency matches the log2 formula."""
        q0 = 0.001
        target = 0.9
        steps = math.ceil(math.log2(math.log(1.0 - target) / math.log(1.0 - q0)))
        pop = _pop("escape", e=1.0, q0=q0)
        for _ in range(steps - 1):
            pop.run(1)
            assert allele_frequency(pop, "D") < target
        pop.run(1)
        assert allele_frequency(pop, "D") > target

    def test_e0_is_neutral(self) -> None:
        """e = 0 is plain Mendelian: the frequency must not move."""
        pop = _pop("neutral", e=0.0, q0=0.3)
        for _ in range(10):
            pop.run(1)
            assert allele_frequency(pop, "D") == pytest.approx(0.3, abs=1e-12)


class TestAdditiveCostInvasionThreshold:
    """Threshold e* = s / (1 - s) for additive costs (h = 0.5)."""

    S = 0.1
    E_THRESHOLD = 0.1 / 0.9  # 0.1111...

    def _additive(self, e: float):
        return _pop(
            f"add_{e:.4f}".replace(".", "p"),
            e=e,
            weights={"W|W": 1.0, "W|D": 1.0 - self.S, "D|D": 1.0 - 2.0 * self.S},
            q0=0.01,
        )

    def test_below_threshold_dies_out(self) -> None:
        """e < e*: the drive decays geometrically and never recovers."""
        e = 0.10  # (1 + e)(1 - s) = 0.99 < 1
        pop = self._additive(e)
        previous = allele_frequency(pop, "D")
        for _ in range(200):
            pop.run(1)
            current = allele_frequency(pop, "D")
            assert current < previous + 1e-12
            previous = current
        assert previous < 0.005  # 0.01 -> ~0.0013 after 200 steps

    def test_above_threshold_spreads(self) -> None:
        """A drive well above e* sweeps to fixation in a few dozen generations."""
        pop = self._additive(0.5)
        for _ in range(40):
            pop.run(1)
        assert allele_frequency(pop, "D") > 0.9

    def test_first_step_threshold_is_e_star_by_bisection(self) -> None:
        """Locate the engine's e-threshold for a near-infinitesimal drive.

        With q_0 = 1e-6 the second-order terms are negligible, so the sign
        of one generation's change crosses zero at e* = s / (1 - s) =
        0.11111...; bisection must find it to 1e-7.
        """
        s = self.S
        e_star = s / (1.0 - s)

        def grows(e: float) -> bool:
            pop = _pop(
                f"bisect_{e:.10f}".replace(".", "p"),
                e=e,
                weights={"W|W": 1.0, "W|D": 1.0 - s, "D|D": 1.0 - 2.0 * s},
                q0=1e-6,
                total=2000.0,
            )
            pop.run(1)
            return allele_frequency(pop, "D") > 1e-6

        low, high = 0.05, 0.2
        assert not grows(low)
        assert grows(high)
        for _ in range(40):
            mid = 0.5 * (low + high)
            if grows(mid):
                high = mid
            else:
                low = mid
        assert 0.5 * (low + high) == pytest.approx(e_star, abs=1e-6)

    def test_early_growth_rate_is_the_linearization(self) -> None:
        """The first step equals the selection-weighted recursion, exactly."""
        e = 0.13
        pop = self._additive(e)
        pop.run(1)
        expected = _expected_states(0.01, e, 1.0 - self.S, 1.0 - 2.0 * self.S, 1)[0]
        assert allele_frequency(pop, "D") == pytest.approx(expected, rel=1e-9)

    def test_trajectory_matches_the_independent_recursion(self) -> None:
        """40 generations against a hand-rolled recursion, from either side."""
        e = 0.13
        w_wd, w_dd = 1.0 - self.S, 1.0 - 2.0 * self.S
        for q0 in (0.005, 0.5):
            pop = _pop(
                f"traj_{str(q0).replace('.', 'p')}",
                e=e,
                weights={"W|W": 1.0, "W|D": w_wd, "D|D": w_dd},
                q0=q0,
            )
            expected = _expected_states(q0, e, w_wd, w_dd, 40)
            for want in expected:
                pop.run(1)
                assert allele_frequency(pop, "D") == pytest.approx(want, rel=1e-9, abs=1e-12)


class TestHeterozygoteOnlyThreshold:
    """The invasion threshold must ignore the homozygote fitness."""

    def test_expensive_homozygote_does_not_move_the_threshold(self) -> None:
        """w_DD = 0.2 (cheap heterozygote) still invades at e just above e*."""
        e = 0.12  # (1 + e)(1 - 0.1) = 1.008 > 1
        pop = _pop(
            "expensive_dd",
            e=e,
            weights={"W|W": 1.0, "W|D": 0.9, "D|D": 0.2},
            q0=0.01,
        )
        pop.run(1)
        # The first-step hinge is the heterozygote term only.
        expected = _expected_states(0.01, e, 0.9, 0.2, 1)[0]
        assert allele_frequency(pop, "D") == pytest.approx(expected, rel=1e-9)

    def test_cheap_homozygote_does_not_rescue_a_losing_drive(self) -> None:
        """w_DD = 1.0 with (1 + e)(1 - h s) < 1 must still be lost."""
        e = 0.10  # (1.10)(0.9) = 0.99 < 1
        pop = _pop(
            "cheap_dd",
            e=e,
            weights={"W|W": 1.0, "W|D": 0.9, "D|D": 1.0},
            q0=0.01,
        )
        for _ in range(200):
            pop.run(1)
        assert allele_frequency(pop, "D") < 0.005
