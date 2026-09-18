"""The equilibrium calibration must consume engine facts, not raw cells.

Two stored inputs used to be read verbatim while the owning tick read them
differently, which silently moved the declared carrying capacity ``K``:

- ``sex_ratio`` on a species whose sex is determined by sex chromosomes: the
  tick ignores the parameter (the documented contract), while the calibration
  split both the age-1 reference composition and ``s_0_avg`` by it, so a
  non-0.5 value moved the equilibrium by tens of percent;
- per-age ``fertility``: the discrete-generation tick reads no age-dependent
  fertility at all (implicit 1.0) and the age-structured tick clamps the
  weight to ``[0, 1]``, while the calibration read the stored value verbatim,
  so a raw tensor write moved the equilibrium by exactly the written factor.

Both paths funnel through the same Rust kernel core, so each test asserts the
restored contract end-to-end: the simulated equilibrium is ``K`` for every
stored value the tick would ignore or clamp, and the ``pop.params`` query
surface reports the calibration the tick actually uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt

K = 2000.0
_XY_FEMALE = "A|A;X1|X1"
_XY_MALE = "A|A;X1|Y1"


def _autosomal_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["W"]}},
        gamete_labels=["default"],
    )


def _xy_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )


def _xy_discrete(name: str, *, sex_ratio: float) -> nt.DiscreteGenerationPopulation:
    """XY species, unequal juvenile survival, Beverton-Holt at K."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_xy_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {_XY_FEMALE: K / 2},
                "male": {_XY_MALE: K / 2},
            }
        )
        .survival(female_age0_survival=0.9, male_age0_survival=0.5)
        .reproduction(eggs_per_female=10.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )


def _xy_age_structured(name: str, *, sex_ratio: float) -> nt.AgeStructuredPopulation:
    """Age-structured twin of :func:`_xy_discrete`."""
    return (
        nt.AgeStructuredPopulation.setup(
            species=_xy_species(name), name=name, stochastic=False
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {_XY_FEMALE: {1: K / 2}},
                "male": {_XY_MALE: {1: K / 2}},
            }
        )
        .survival(
            female_age_based_survival=[0.9, 0.9],
            male_age_based_survival=[0.5, 0.9],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0],
            eggs_per_female=10.0,
            sex_ratio=sex_ratio,
        )
        .competition(
            carrying_capacity=K,
            low_density_growth_rate=3.0,
            growth_mode="beverton_holt",
        )
        .build()
    )


def _total(population: object) -> float:
    return float(np.asarray(population.state.individual_count).sum())


@pytest.mark.parametrize("sex_ratio", [0.2, 0.3, 0.4, 0.6, 0.7])
def test_sex_chromosome_discrete_equilibrium_ignores_sex_ratio(sex_ratio: float) -> None:
    """An ignored parameter must not move K (both directions of the bias).

    The genetic split decides offspring sex here, so the calibration uses the
    balanced 1:1 split.  Reading ``sex_ratio`` instead settled at
    0.7*K*(1.5-sr)/(0.5+0.4*sr): 3137.9 at sr = 0.2 (+56.9 %) and 2333.3 at
    sr = 0.4 (+16.7 %); a value above 0.5 moves the other way (sr = 0.7 was
    below K).
    """
    population = _xy_discrete(f"xy_discrete_{sex_ratio}", sex_ratio=sex_ratio)
    population.run(400)
    assert _total(population) == pytest.approx(K, rel=1e-9)


@pytest.mark.parametrize("sex_ratio", [0.3, 0.7])
def test_sex_chromosome_age_structured_equilibrium_ignores_sex_ratio(
    sex_ratio: float,
) -> None:
    """The age-structured twin shares the kernel, so it must agree (2709.677
    at sex_ratio = 0.3 before the fix)."""
    population = _xy_age_structured(f"xy_age_{sex_ratio}", sex_ratio=sex_ratio)
    population.run(600)
    assert _total(population) == pytest.approx(K, rel=1e-9)


def test_sex_chromosome_query_surface_ignores_sex_ratio() -> None:
    """``pop.params.expected_*`` is the tick's calibration, not a second rule.

    Both queries come from the same flat kernel entry the engines use; the
    draft's stored ``sex_ratio`` differs only in the parameter the
    sex-chromosome engine ignores.
    """
    biased = _xy_discrete("xy_query_biased", sex_ratio=0.3)
    balanced = _xy_discrete("xy_query_balanced", sex_ratio=0.5)
    assert biased.params.expected_competition_strength == (
        balanced.params.expected_competition_strength
    )
    assert biased.params.expected_survival_rate == balanced.params.expected_survival_rate


def test_sex_chromosome_wrapper_and_flat_entries_agree() -> None:
    """One rule, two calibration entries, bit-identical results.

    Both engines call the bp-aware wrapper while ``pop.params.expected_*``
    rides the flat query entry; a rule applied to only one of them would
    report a calibration the simulation never uses.
    """
    import struct

    from natal._engine_rs import equilibrium_metrics
    from natal.contracts.materialize import materialize

    population = _xy_discrete("xy_entry_parity", sex_ratio=0.3)
    contracts = materialize(population.config)
    wrapper_comp, wrapper_surv = equilibrium_metrics(
        contracts.blueprint, contracts.params
    )
    assert struct.pack(">d", wrapper_comp) == struct.pack(
        ">d", population.params.expected_competition_strength
    )
    assert struct.pack(">d", wrapper_surv) == struct.pack(
        ">d", population.params.expected_survival_rate
    )


@pytest.mark.parametrize("fertility", [0.5, 2.0])
def test_discrete_fertility_write_cannot_move_k(fertility: float) -> None:
    """The discrete tick has no age-dependent fertility; nor may the calibration.

    The builder already rejects the per-age parameter on discrete drafts, but
    the raw tensor channel accepts any non-negative value.  Both the equilibrium
    (2500 at 0.5, 1000 at 2.0 before the fix) and the query surface must stay
    put.
    """
    name = f"discrete_fertility_{fertility}"
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_autosomal_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"W|W": K / 2},
                "male": {"W|W": K / 2},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=K,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    before = (
        population.params.expected_competition_strength,
        population.params.expected_survival_rate,
    )
    population.params.tensor_write("fertility", np.array([0.0, fertility]))
    after = (
        population.params.expected_competition_strength,
        population.params.expected_survival_rate,
    )
    assert after == before
    population.run(400)
    assert _total(population) == pytest.approx(K, rel=1e-9)


@pytest.mark.parametrize("fertility,clamped", [(0.5, False), (2.0, True)])
def test_age_structured_fertility_write_matches_the_tick(
    fertility: float, clamped: bool
) -> None:
    """The age-structured tick consumes ``clamp01(fertility)``; so must the
    calibration.

    An in-domain weight is effective in both consumers (C* halves and s*
    doubles at 0.5, so K still holds), while an out-of-domain weight clamps to
    1 in both (the query surface must not move, and the pre-fix equilibrium of
    1000 at 2.0 becomes K).
    """
    name = f"age_fertility_{fertility}"
    population = (
        nt.AgeStructuredPopulation.setup(
            species=_autosomal_species(name), name=name, stochastic=False
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"W|W": {1: K / 2}},
                "male": {"W|W": {1: K / 2}},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9],
            male_age_based_survival=[1.0, 0.9],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0],
            eggs_per_female=10.0,
            sex_ratio=0.5,
        )
        .competition(
            carrying_capacity=K,
            low_density_growth_rate=3.0,
            growth_mode="beverton_holt",
        )
        .build()
    )
    before = (
        population.params.expected_competition_strength,
        population.params.expected_survival_rate,
    )
    population.params.tensor_write("fertility", np.array([0.0, fertility]))
    after = (
        population.params.expected_competition_strength,
        population.params.expected_survival_rate,
    )
    if clamped:
        assert after == before
    else:
        assert after != before
    population.run(800)
    assert _total(population) == pytest.approx(K, rel=1e-9)
