"""R4-05: sex-chromosome species use ``sex_ratio`` in the K calibration
although the engine documents and implements it as ignored.

Documented contract (docs/*/2_population_initialization.md): ``sex_ratio``
is "Ignored when sex chromosomes can determine offspring sex".  The engine
honours that — an XY species with ``sex_ratio = 0.3`` still produces a
1:1 genetic sex split — but ``equilibrium_metrics`` keeps consuming the
same parameter:

- the derived age-1 reference split uses
  ``sex_ratio * s_f0 / (sex_ratio * s_f0 + (1 - sex_ratio) * s_m0)``,
- the survival-rate denominator uses
  ``s_0_avg = sex_ratio * s_f0 + (1 - sex_ratio) * s_m0``,

both of which should use the *genetic* 1:1 split (0.5) for a
sex-chromosome species.  Consequence: the simulation settles away from the
declared ``carrying_capacity`` K whenever ``sex_ratio != 0.5`` — measured
below at +270 %o ... +57 % (both engines; the *same* 2709.677 for the
discrete and the age-structured path at K = 2000, s_f0 = 0.9, s_m0 = 0.5).

The equal-survival case isolates the mechanism: with ``s_f0 = s_m0`` the
s_0_avg term is unchanged by ``sex_ratio``, yet the equilibrium still moves
as ``K * (1.5 - sex_ratio)`` — the age-1 reference split is the culprit
there, and the survival denominator is the extra term when the sexes
survive differently.

Wrong results rejected: an equilibrium that is K only when the parameter is
at its default (0.5), a K that moves with a parameter the engine says is
ignored, and a discrete/age-structured disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers_r4 import age_pop, discrete_pop, species_locus, species_xy

K = 2000.0
S_F0 = 0.9
S_M0 = 0.5
SEX_RATIOS = (0.5, 0.4, 0.3, 0.2)


def _xy_discrete(name: str, *, sex_ratio: float) -> object:
    sp = species_xy(f"R4_05_{name}", ("A", "a"))
    return discrete_pop(
        name,
        species=sp,
        female={"A|A;X1|X1": K / 2.0},
        male={"A|A;X1|Y1": K / 2.0},
        eggs_per_female=10.0,
        sex_ratio=sex_ratio,
        survival=1.0,
        growth_mode="beverton_holt",
        carrying_capacity=K,
        low_density_growth_rate=3.0,
        extra=lambda b: b.survival(
            female_age0_survival=S_F0, male_age0_survival=S_M0
        ),
    )


def _autosomal_discrete(name: str, *, sex_ratio: float) -> object:
    sp = species_locus(f"R4_05_{name}", ["A"])
    return discrete_pop(
        name,
        species=sp,
        female={"A|A": K / 2.0},
        male={"A|A": K / 2.0},
        eggs_per_female=10.0,
        sex_ratio=sex_ratio,
        survival=1.0,
        growth_mode="beverton_holt",
        carrying_capacity=K,
        low_density_growth_rate=3.0,
        extra=lambda b: b.survival(
            female_age0_survival=S_F0, male_age0_survival=S_M0
        ),
    )


def _xy_age(name: str, *, sex_ratio: float) -> object:
    sp = species_xy(f"R4_05_{name}", ("A", "a"))
    return age_pop(
        name,
        species=sp,
        n_ages=2,
        new_adult_age=1,
        initial={
            "female": {"A|A;X1|X1": {1: K / 2.0}},
            "male": {"A|A;X1|Y1": {1: K / 2.0}},
        },
        survival_f=[S_F0, 0.9],
        survival_m=[S_M0, 0.9],
        mating_f=[0.0, 1.0],
        mating_m=[0.0, 1.0],
        eggs_per_female=10.0,
        sex_ratio=sex_ratio,
        growth_mode="beverton_holt",
        carrying_capacity=K,
        low_density_growth_rate=3.0,
    )


def _total(pop) -> float:
    return float(np.asarray(pop.state.individual_count).sum())


def _sex_totals(pop) -> tuple[float, float]:
    counts = np.asarray(pop.state.individual_count)
    return float(counts[0].sum()), float(counts[1].sum())


class TestEngineIgnoresSexRatioForGeneticSex:
    def test_realized_sex_ratio_is_genetic(self) -> None:
        """Positive control for the inconsistency: the engine's split is 1:1."""
        for sex_ratio in (0.5, 0.3):
            pop = _xy_discrete(f"sexratio_{str(sex_ratio).replace('.', 'p')}", sex_ratio=sex_ratio)
            pop.run(200)
            female, male = _sex_totals(pop)
            # Surviving-sex ratio of a 1:1 genetic split: 0.9 / 0.5.
            expected = S_F0 / (S_F0 + S_M0)
            assert female / (female + male) == pytest.approx(expected, rel=1e-9)


class TestAutosomalControl:
    @pytest.mark.parametrize("sex_ratio", SEX_RATIOS)
    def test_autosomal_equilibrium_is_exactly_k(self, sex_ratio: float) -> None:
        pop = _autosomal_discrete(f"auto_{str(sex_ratio).replace('.', 'p')}", sex_ratio=sex_ratio)
        pop.run(400)
        assert _total(pop) == pytest.approx(K, rel=1e-6)


class TestSexChromosomeCalibration:
    @pytest.mark.parametrize("sex_ratio", SEX_RATIOS)
    def test_xy_discrete_equilibrium_is_k(self, sex_ratio: float) -> None:
        """The declared K must be reached for every sex_ratio (ignored parameter)."""
        pop = _xy_discrete(f"xy_{str(sex_ratio).replace('.', 'p')}", sex_ratio=sex_ratio)
        pop.run(400)
        observed = _total(pop)
        assert observed == pytest.approx(K, rel=1e-6), (
            f"sex_ratio={sex_ratio}: equilibrium {observed:.3f} != declared K={K} "
            f"({100.0 * (observed / K - 1.0):+.1f}%)"
        )

    @pytest.mark.parametrize("sex_ratio", (0.5, 0.3))
    def test_xy_age_structured_equilibrium_is_k(self, sex_ratio: float) -> None:
        pop = _xy_age(f"xyage_{str(sex_ratio).replace('.', 'p')}", sex_ratio=sex_ratio)
        pop.run(600)
        observed = _total(pop)
        assert observed == pytest.approx(K, rel=1e-6), (
            f"sex_ratio={sex_ratio}: age-structured equilibrium {observed:.3f} != K={K}"
        )
