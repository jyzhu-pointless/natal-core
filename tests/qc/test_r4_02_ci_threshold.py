"""R4-02: Cytoplasmic incompatibility (Wolbachia) threshold, built from
public primitives.

Literature model (Caspari & Watson 1959; Turelli & Hoffmann 1995,
Genetics 140:1319; Hoffmann & Turelli 1997): with infected frequency
``p``, a fecundity cost ``s_f`` on infected females, embryonic lethality
``s_h`` in the incompatible (uninfected female x infected male) cross and
perfect maternal transmission,

    p' = p (1 - s_f) / [ p (1 - s_f) + (1 - p)^2 + (1 - p) p (1 - s_h) ]

so infection spreads to fixation only above the interior unstable
equilibrium ``p* = s_f / s_h`` (zero when the symbiont is cost-free).

The engine has no CI preset (the docs say so since the 0916 round), so the
mechanism is assembled from public primitives: gamete labels tag infected
mothers/fathers, a maternal tag redirects offspring to the infected slab,
a paternal tag redirects *remaining* default-slab offspring to a dead slab
at rate ``s_h``, the dead slab carries viability 0 and the infected slab a
female-only fecundity factor.

Wrong results rejected: a CI mechanism that kills the infected-mother
cross too (maternal tag leaking into the paternal rule), one that applies
the fecundity cost through the father as well, one whose threshold is at
1 - s_f/s_h, and "CI kills a fixed fraction of all embryos" instead of the
incompatible cross only.
"""

from __future__ import annotations

import pytest

from _helpers_r4 import ad_slab_frequency, build_ci_population, ci_map, ci_threshold


def _run(name: str, *, p0: float, s_f: float, s_h: float, ticks: int, total: float = 4000.0):
    pop = build_ci_population(name, p0=p0, total=total, s_f=s_f, s_h=s_h, eggs=2.0)
    pop.run(ticks)
    return pop


class TestCiRecursion:
    def test_trajectory_matches_the_recursion(self) -> None:
        """12 generations of the infected frequency against the closed map."""
        s_f, s_h = 0.1, 0.5
        pop = build_ci_population("traj", p0=0.5, total=4000.0, s_f=s_f, s_h=s_h, eggs=2.0)
        expected = 0.5
        for t in range(1, 13):
            pop.run(1)
            expected = ci_map(expected, s_f, s_h)
            got = ad_slab_frequency(pop, "infected")
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), (
                f"generation {t}: infected={got!r} vs recursion {expected!r}"
            )

    def test_first_step_counts_are_exact(self) -> None:
        """One tick from a half-infected population: 0.45 / 0.825 exactly.

        Adults: 2000 infected + 2000 uninfected, two offspring each.
        Infected mothers pay the fecundity cost (0.9); the incompatible
        cross keeps 1 - s_h = 0.5 of its offspring; compatible crosses
        keep everything.  Offspring = 2000*0.9*2 + 2000*2*(0.5*0.5 + 0.5)
        = 3600 + 3000 = 6600, infected share 3600/6600 = 0.5454...
        """
        pop = build_ci_population(
            "exact", p0=0.5, total=4000.0, s_f=0.1, s_h=0.5, eggs=2.0
        )
        pop.run(1)
        assert pop.state.individual_count.sum() == pytest.approx(6600.0, rel=1e-9)
        assert ad_slab_frequency(pop, "infected") == pytest.approx(3600.0 / 6600.0, rel=1e-9)


class TestCiThreshold:
    @pytest.mark.parametrize(
        ("s_f", "s_h", "p_star"),
        [(0.1, 0.5, 0.2), (0.2, 0.25, 0.8)],
    )
    def test_bisection_locates_p_star(self, s_f: float, s_h: float, p_star: float) -> None:
        """The first-generation sign change sits exactly at p* = s_f / s_h."""

        def rises(p0: float) -> bool:
            pop = build_ci_population(
                f"bis_{s_f}_{s_h}_{p0:.9f}".replace(".", "p"),
                p0=p0,
                total=4000.0,
                s_f=s_f,
                s_h=s_h,
                eggs=2.0,
            )
            pop.run(1)
            return ad_slab_frequency(pop, "infected") > p0

        low, high = 0.0, 1.0
        # Establish the bracket: p* is interior for these parameter pairs.
        assert not rises(p_star * 0.5)
        assert rises(min(1.0, p_star + 0.05))
        for _ in range(60):
            mid = 0.5 * (low + high)
            if rises(mid):
                high = mid
            else:
                low = mid
        assert 0.5 * (low + high) == pytest.approx(ci_threshold(s_f, s_h), abs=1e-9)

    def test_below_threshold_loses_the_infection(self) -> None:
        """p_0 = 0.02 < p* = 0.2: monotone decay with per-step factor -> 1 - s_f."""
        pop = build_ci_population("below", p0=0.02, total=4000.0, s_f=0.1, s_h=0.5, eggs=2.0)
        previous = ad_slab_frequency(pop, "infected")
        for _ in range(150):
            pop.run(1)
            current = ad_slab_frequency(pop, "infected")
            assert current < previous
            # Away from p* the decay factor approaches the maternal-cost-only
            # value 1 - s_f = 0.9 (CI only bites through the frequency-
            # dependent term).
            if previous < 0.01:
                assert current / previous == pytest.approx(0.9, rel=0.02)
            previous = current
        assert previous < 1e-8

    def test_above_threshold_sweeps_to_fixation(self) -> None:
        pop = build_ci_population("above", p0=0.5, total=4000.0, s_f=0.1, s_h=0.5, eggs=2.0)
        for _ in range(80):
            pop.run(1)
        assert ad_slab_frequency(pop, "infected") > 0.999

    def test_cost_free_ci_has_no_threshold(self) -> None:
        """s_f = 0: infection is pushed up from any positive frequency (p* = 0).

        The invasion rate is frequency dependent (the map is
        ``p' = p / (1 - p (1 - p) s_h)``, so ``p' - p`` is O(p^2)), hence a
        moderate starting frequency is used; the closed map is the
        independent expectation.
        """
        s_h = 0.5
        pop = build_ci_population("free", p0=0.05, total=4000.0, s_f=0.0, s_h=s_h, eggs=2.0)
        expected = 0.05
        for _ in range(200):
            pop.run(1)
            expected = ci_map(expected, 0.0, s_h)
            assert ad_slab_frequency(pop, "infected") == pytest.approx(expected, rel=1e-9)
        assert expected > 0.999
        # And the O(p^2) rule: from 1e-6 the map moves by ~5e-7 per step.
        small = build_ci_population("small", p0=1e-6, total=1e6, s_f=0.0, s_h=s_h, eggs=2.0)
        small.run(200)
        assert ad_slab_frequency(small, "infected") == pytest.approx(ci_map(1e-6, 0.0, s_h), abs=1e-9)


class TestCostIsMaternalOnly:
    def test_infected_father_pays_no_fecundity_cost(self) -> None:
        """1000 uninfected females x 1000 infected males, 2 eggs, s_h = 0.5.

        Only CI acts: 2000 zygotes, half killed -> 1000 survivors.  If the
        fecundity cost were charged through the father as well the count
        would be 800.
        """
        pop = build_ci_population(
            "paternal", p0=0.0, male_p0=1.0, total=1000.0, s_f=0.2, s_h=0.5, eggs=2.0
        )
        pop.run(1)
        survivors = float(pop.state.individual_count.sum())
        assert survivors == pytest.approx(1000.0, rel=1e-9)


class TestCiRecursionUnchangedByFathersCost:
    def test_first_step_follows_the_cross_weighted_map(self) -> None:
        """Males all infected, females half: the first step weights the cross.

        Offspring mass = p_f (1 - s_f) [infected mothers, compatible] +
        (1 - p_f)(1 - p_m s_h) [uninfected mothers, fraction p_m of their
        crosses are incompatible]; the infected share is the first term over
        the total, i.e. 0.45 / 0.7 = 0.642857...
        """
        pop = build_ci_population(
            "male_side", p0=0.5, male_p0=1.0, total=4000.0, s_f=0.1, s_h=0.5, eggs=2.0
        )
        pop.run(1)
        expected = 0.5 * 0.9 / (0.5 * 0.9 + 0.5 * (1.0 - 1.0 * 0.5))
        assert ad_slab_frequency(pop, "infected") == pytest.approx(expected, rel=1e-9)

    def test_male_side_converges_to_the_standard_map(self) -> None:
        """From tick 2 on, p_m = p_f and the classical map takes over."""
        s_f, s_h = 0.1, 0.5
        pop = build_ci_population(
            "male_side2", p0=0.5, male_p0=1.0, total=4000.0, s_f=s_f, s_h=s_h, eggs=2.0
        )
        pop.run(1)
        expected = 0.5 * 0.9 / (0.5 * 0.9 + 0.5 * (1.0 - 1.0 * 0.5))
        for _ in range(8):
            pop.run(1)
            expected = ci_map(expected, s_f, s_h)
            assert ad_slab_frequency(pop, "infected") == pytest.approx(expected, rel=1e-9)
