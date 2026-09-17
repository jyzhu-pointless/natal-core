"""QC spot-check 24: age-structured production semantics (repaired verdicts).

The first draft of this file claimed an "eggs_per_female ignored" engine
defect.  Independent adversarial review REFUTED that: the tests had left
`juvenile_growth_mode` at the age-structured draft default (2 = logistic,
src/natal/frontend/model/assembly.py:361) while the discrete engine
defaults to 0 (NO_COMPETITION, assembly.py:598).  Under the compensatory
curves the equilibrium survival is s* = K/(K*E*s0) = 1/E, so the newborn
total becomes pairs*E*(r/E) = pairs*r — eggs cancel mathematically.  That
is documented curve semantics, not a bug, but the DEFAULT asymmetry
between the two engines makes identical configs behave qualitatively
differently (kept below as a pinned smell).

Also corrected: pre-initialized sperm_storage IS consumed by the engine;
zero output in the first draft came from the reproduction() early-return
(zero mating-weighted adult males => fertilize never runs).  With an
available male the stored sperm produces exactly n_pairs * eggs.

All tests in this file PASS on feat/post-p10-residuals@cd36f9b; they pin
the corrected semantics.
"""

from __future__ import annotations

import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l": ["W", "D"]}}, gamete_labels=["default"]
    )


def _build(name: str, eggs: float, storage: dict | None = None,
           female_mating: float = 1.0, male_mating: float = 1.0):
    pop = (
        nt.AgeStructuredPopulation.setup(species=_species(name + "_sp"), name=name,
                                         stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"W|W": {1: 600}},
                "male": {"W|W": {1: 600}},
            },
            sperm_storage=storage,
        )
        .survival(female_age_based_survival=[1.0] * 3, male_age_based_survival=[1.0] * 3)
        .reproduction(
            female_age_based_mating_rate=[0.0, female_mating, 0.0],
            male_age_based_mating_rate=[0.0, male_mating, 0.0],
            eggs_per_female=eggs,
        )
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                     growth_mode="fixed")
        .build()
    )
    return pop


def _newborns(pop: nt.AgeStructuredPopulation) -> float:
    # new_adult_age=1: the newborn cohort sits at age 1 after one tick,
    # the parental cohort at age 2.
    return float(pop.state.individual_count[:, 1, :].sum())


class TestEggsScaleOutput:
    def test_eggs_per_female_scales_output(self) -> None:
        """Claim: with a fixed (no-cancellation) growth mode, 600 mated
        females x eggs=E produce exactly 600*E newborns.

        Rejects an eggs-to-session wiring bug; guards the discrete/age
        parity of the fertilize formula.
        """
        for eggs in (0.5, 1.0, 3.0):
            pop = _build(f"qc_prod_{eggs}", eggs)
            pop.run(1)
            assert _newborns(pop) == pytest.approx(600.0 * eggs, abs=1e-6)

    def test_zero_eggs_zero_output(self) -> None:
        """Claim: eggs=0 yields no newborns."""
        pop = _build("qc_prod_zero", 0.0)
        pop.run(1)
        assert _newborns(pop) == 0.0

    def test_preinitialized_sperm_storage_produces(self) -> None:
        """Claim: 600 stored sperm pairs + eggs=1 produce 600 newborns
        without any new mating, provided a mating-capable male exists.

        The reproduction stage early-returns only when the mating-rate
        weighted ADULT MALE count is zero — that gate, not a dropped
        sperm table, explains a zero output when male mating rates are 0.
        """
        pop = _build(
            "qc_prod_store",
            1.0,
            storage={"W|W": {"W|W": {1: 600}}},
            female_mating=0.0,
            male_mating=1.0,
        )
        pop.run(1)
        assert _newborns(pop) == pytest.approx(600.0, abs=1e-6)

    def test_storage_without_mating_capable_males_gates_to_zero(self) -> None:
        """Claim (documented semantics, pinned): zero male mating rate
        gates fertilization entirely — even with stored sperm — because
        reproduction early-returns on zero effective males."""
        pop = _build(
            "qc_prod_gate",
            1.0,
            storage={"W|W": {"W|W": {1: 600}}},
            female_mating=0.0,
            male_mating=0.0,
        )
        pop.run(1)
        assert _newborns(pop) == 0.0


class TestDefaultGrowthMode:
    def test_age_default_is_beverton_holt(self) -> None:
        """Claim (repaired): omitting the mode selects 3 = BEVERTON_HOLT.

        The engines used to disagree — age-structured defaulted to 2
        (logistic) and discrete to 0 (NO_COMPETITION, unbounded growth) — so
        identical builder chains behaved qualitatively differently.  Every
        entry point now defaults to the monotone compensatory curve.

        600 pairs, r = low_density_growth_rate = 2: the newborn total is
        pairs*r for ANY eggs > 0, because the equilibrium survival s* = 1/E
        cancels the clutch size.  That identity holds for the default curve.
        """
        for eggs in (1.0, 3.0):
            pop = (
                nt.AgeStructuredPopulation.setup(
                    species=_species(f"qc_asym_{eggs}_sp"),
                    name=f"qc_asym_{eggs}", stochastic=False
                )
                .age_structure(n_ages=3, new_adult_age=1)
                .initial_state(individual_count={
                    "female": {"W|W": {1: 600}},
                    "male": {"W|W": {1: 600}},
                })
                .survival(female_age_based_survival=[1.0] * 3,
                          male_age_based_survival=[1.0] * 3)
                .reproduction(
                    female_age_based_mating_rate=[0.0, 1.0, 0.0],
                    male_age_based_mating_rate=[0.0, 1.0, 0.0],
                    eggs_per_female=eggs,
                )
                .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
                .build()
            )
            assert pop.params.growth_mode == 3
            pop.run(1)
            assert _newborns(pop) == pytest.approx(1200.0, abs=1e-6), (
                f"eggs={eggs}: the default curve cancels the clutch size"
            )

    def test_discrete_default_matches_the_age_structured_default(self) -> None:
        """Claim (repaired): the discrete engine shares the same default mode.

        It used to default to 0 (NO_COMPETITION), which carried every egg
        through (1800 offspring here) while the age-structured engine used a
        compensatory curve.  Both now default to 3 = BEVERTON_HOLT; at this
        enormous K the compensated one-tick total is 2400.
        """
        sp = _species("qc_asym_disc_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, name="qc_asym_disc",
                                                  stochastic=False)
            .initial_state(individual_count={"female": {"W|W": 600},
                                             "male": {"W|W": 600}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=3, sex_ratio=0.5)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
            .build()
        )
        assert pop.params.growth_mode == 3
        pop.run(1)
        assert float(pop.state.individual_count.sum()) == pytest.approx(2400.0, abs=1e-4)
