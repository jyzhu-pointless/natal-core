"""R4-09: multi-age density equilibrium, the derived age profile, and the
once-per-lifetime viability contract.

Contracts:

- ``carrying_capacity``/K is the *age-1* equilibrium head count; for a model
  with more than one juvenile age the derived competition strength C* is a
  weighted sum over the juvenile ages and the calibration must still land
  the age-1 count exactly on K (the QC-023 fix generalised beyond two ages),
- the derived equilibrium age profile decays by the declared survival rates
  (age a+1 = age a x s(a)),
- genotype viability is applied once, at the last juvenile age
  (``new_adult_age - 1``) — not at birth and not at every age.

Wrong results rejected: a K that applies to the total population instead of
the age-1 cohort, an equilibrium profile that ignores the survival ladder,
viability applied at age 0 (or repeatedly per age), and a cohort that loses
mass at an age where no rate is declared.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers_r4 import age_pop, female_age_counts, species_locus

K = 2000.0
N_AGES = 4
NEW_ADULT_AGE = 2
SURVIVAL = [0.7, 0.8, 0.9, 0.0]


def _pop(name: str, *, viability: dict[str, float] | None = None, initial=None):
    species = species_locus(f"R4_09_{name}", ["A", "a"])
    if initial is None:
        initial = {"female": {"A|A": {1: K / 2.0}}, "male": {"A|A": {1: K / 2.0}}}

    def extra(builder):
        if viability:
            builder = builder.fitness(
                viability={
                    genotype: {"female": value, "male": value}
                    for genotype, value in viability.items()
                },
                mode="replace",
            )
        return builder

    return age_pop(
        name,
        species=species,
        n_ages=N_AGES,
        new_adult_age=NEW_ADULT_AGE,
        initial=initial,
        survival_f=SURVIVAL,
        survival_m=SURVIVAL,
        mating_f=[0.0, 0.0, 1.0, 1.0],
        mating_m=[0.0, 0.0, 1.0, 1.0],
        eggs_per_female=10.0,
        sex_ratio=0.5,
        growth_mode="beverton_holt",
        carrying_capacity=K,
        low_density_growth_rate=3.0,
        extra=extra,
    )


def _age_totals(pop) -> np.ndarray:
    counts = np.asarray(pop.state.individual_count)
    return counts.sum(axis=(0, 2))


class TestMultiAgeEquilibrium:
    def test_age_1_count_is_exactly_k(self) -> None:
        pop = _pop("equilibrium")
        pop.run(600)
        totals = _age_totals(pop)
        assert totals[1] == pytest.approx(K, rel=1e-6), totals

    def test_age_profile_follows_the_survival_ladder(self) -> None:
        pop = _pop("profile")
        pop.run(600)
        totals = _age_totals(pop)
        assert totals[2] == pytest.approx(totals[1] * SURVIVAL[1], rel=1e-6)
        assert totals[3] == pytest.approx(totals[2] * SURVIVAL[2], rel=1e-6)

    def test_equilibrium_total_is_the_ladder_sum(self) -> None:
        pop = _pop("total")
        pop.run(600)
        totals = _age_totals(pop)
        expected = K * (1.0 + SURVIVAL[1] + SURVIVAL[1] * SURVIVAL[2])
        assert totals.sum() == pytest.approx(expected, rel=1e-6)


class TestViabilityAppliedOnceAtTheLastJuvenileAge:
    def test_cohort_ladder_places_the_viability_factor_once(self) -> None:
        """A tagged cohort of 1000 age-0s shrinks only at age new_adult_age-1.

        Survival is 1 everywhere except the terminal age, so the cohort
        must read 1000 / 1000 / 500 / 500 and then vanish — a factor-0.5
        viability applied at the wrong age (or twice) changes this ladder.
        """
        pop = age_pop(
            "viability_ladder",
            species=species_locus("R4_09_ladder", ["A"]),
            n_ages=N_AGES,
            new_adult_age=NEW_ADULT_AGE,
            initial={"female": {"A|A": {0: 1000.0}}, "male": {"A|A": {0: 0.0}}},
            # Reproduction switched off (mating rates 0) so the cohort ages
            # without any new births.
            survival_f=[1.0, 1.0, 1.0, 0.0],
            survival_m=[1.0, 1.0, 1.0, 0.0],
            mating_f=[0.0, 0.0, 0.0, 0.0],
            mating_m=[0.0, 0.0, 0.0, 0.0],
            eggs_per_female=0.0,
            sex_ratio=0.5,
            growth_mode="fixed",
            carrying_capacity=1e12,
            extra=lambda b: b.fitness(
                viability={"A|A": {"female": 0.5, "male": 0.5}}, mode="replace"
            ),
        )
        expected = [1000.0, 1000.0, 500.0, 500.0]
        observed = []
        for tick in range(N_AGES):
            observed.append(float(np.asarray(pop.state.individual_count)[0].sum()))
            pop.run(1)
        assert observed == pytest.approx(expected, rel=1e-9), observed
        # The oldest class drops off after one adult tick.
        assert float(np.asarray(pop.state.individual_count)[0].sum()) == pytest.approx(0.0, abs=1e-12)

    def test_female_only_cohort_keeps_its_sex_plane(self) -> None:
        """The ladder above must not leak mass into the male plane."""
        pop = age_pop(
            "viability_plane",
            species=species_locus("R4_09_plane", ["A"]),
            n_ages=N_AGES,
            new_adult_age=NEW_ADULT_AGE,
            initial={"female": {"A|A": {0: 1000.0}}, "male": {"A|A": {0: 0.0}}},
            survival_f=[1.0, 1.0, 1.0, 0.0],
            survival_m=[1.0, 1.0, 1.0, 0.0],
            mating_f=[0.0, 0.0, 0.0, 0.0],
            mating_m=[0.0, 0.0, 0.0, 0.0],
            eggs_per_female=0.0,
            sex_ratio=0.5,
            growth_mode="fixed",
            carrying_capacity=1e12,
            extra=lambda b: b.fitness(
                viability={"A|A": {"female": 0.5, "male": 0.5}}, mode="replace"
            ),
        )
        pop.run(N_AGES)
        counts = np.asarray(pop.state.individual_count)
        assert counts[1].sum() == pytest.approx(0.0, abs=1e-12)
        assert female_age_counts(pop).sum() == pytest.approx(0.0, abs=1e-9)
