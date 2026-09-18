"""Cross-specific Wolbachia costs at three distinct lifecycle stages."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import natal as nt

Effect = Literal["zygote_viability", "viability", "fecundity"]


def _species(name: str, *, extra_labels: bool = True, unordered: bool = False) -> nt.Species:
    return nt.Species.from_dict(
        name, {"c": {"l": ["A", "a"]}}, unordered=unordered,
        somatic_labels=["normal", "infected", "incompatible"] if extra_labels else ["normal", "infected"],
        gamete_labels=["default", "wolbachia", "wolbachia_ci"] if extra_labels else ["default", "wolbachia"],
    )


def _population(
    name: str, effect: Effect, cost: float, *, mother: str = "normal",
    father: str = "infected", stochastic: bool = False, fixed_egg_count: bool = True,
    wf: int = 0, model: str = "discrete", extra_labels: bool = True,
) -> nt.AgeStructuredPopulation | nt.DiscreteGenerationPopulation:
    species = _species(name, extra_labels=extra_labels)
    preset = nt.Wolbachia(
        "w", incompatibility_cost=cost, incompatibility_effect=effect,
    )
    if model == "age":
        builder = nt.AgeStructuredPopulation.setup(
            species, stochastic=stochastic, fixed_egg_count=fixed_egg_count,
        ).age_structure(n_ages=4, new_adult_age=1)
        builder = builder.survival(female_age_based_survival=[1, 1, 1, 0], male_age_based_survival=[1, 1, 1, 0])
    else:
        builder = nt.DiscreteGenerationPopulation.setup(
            species, stochastic=stochastic, fixed_egg_count=fixed_egg_count, extreme_speed_mode=wf,
        ).survival(female_age0_survival=1, male_age0_survival=1)
    return (
        builder.initial_state(individual_count={
            "female": {f"A|a@{mother}": {1: 100}},
            "male": {f"A|a@{father}": {1: 100}},
        })
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(juvenile_growth_mode="no_competition")
        .presets(preset).build()
    )


def _newborn_cohort(pop: nt.AgeStructuredPopulation | nt.DiscreteGenerationPopulation) -> np.ndarray:
    """After one full tick, the new cohort occupies age 1."""
    return np.asarray(pop.state.individual_count)[:, 1, :].sum(axis=0)


@pytest.mark.parametrize("effect", ["zygote_viability", "viability", "fecundity"])
@pytest.mark.parametrize("cost", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("mother,father", [("normal", "normal"), ("normal", "infected"), ("infected", "normal"), ("infected", "infected")])
@pytest.mark.parametrize("model", ["age", "discrete"])
def test_cross_table(effect: Effect, cost: float, mother: str, father: str, model: str) -> None:
    pop = _population(f"ci_{effect}_{cost}_{mother}_{father}_{model}", effect, cost, mother=mother, father=father, model=model)
    pop.run(1)
    cohort = _newborn_cohort(pop)
    expected = 200 * (1 - cost if effect != "fecundity" and mother == "normal" and father == "infected" else 1)
    assert cohort.sum() == pytest.approx(expected, abs=1e-11)
    infected = [i for i, (_, slab) in enumerate(pop.registry.index_to_ztype) if slab == "infected"]
    assert cohort[infected].sum() == pytest.approx(expected if mother == "infected" else 0, abs=1e-11)


@pytest.mark.parametrize("effect", ["zygote_viability", "viability", "fecundity"])
def test_wf_deterministic_respects_cost(effect: Effect) -> None:
    pop = _population(f"ci_wf_{effect}", effect, 0.4, wf=3)
    pop.run(1)
    assert _newborn_cohort(pop).sum() == pytest.approx(200 if effect == "fecundity" else 120)


@pytest.mark.parametrize("effect", ["zygote_viability", "viability", "fecundity"])
def test_incompatible_survivor_is_uninfected_and_susceptible(effect: Effect) -> None:
    pop = _population(f"ci_survivor_{effect}", effect, 0.4, mother="incompatible")
    pop.run(1)
    assert _newborn_cohort(pop).sum() == pytest.approx(120)
    pop = _population(f"ci_clear_{effect}", effect, 0.4, mother="incompatible", father="normal")
    pop.run(1)
    counts = _newborn_cohort(pop)
    normal = [i for i, (_, slab) in enumerate(pop.registry.index_to_ztype) if slab == "normal"]
    assert counts[normal].sum() == pytest.approx(120 if effect == "fecundity" else 200)


def test_zero_cost_preserves_legacy_labels() -> None:
    pop = _population("ci_disabled", "zygote_viability", 0, extra_labels=False)
    pop.run(1)
    assert _newborn_cohort(pop).sum() == pytest.approx(200)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_cost(bad: float) -> None:
    with pytest.raises(ValueError, match="incompatibility_cost"):
        nt.Wolbachia("bad", incompatibility_cost=bad)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_fecundity_cost_does_not_reduce_birth_of_ci_individuals(model: str) -> None:
    pop = _population(f"ci_fec_birth_{model}", "fecundity", 1.0, stochastic=True, model=model)
    pop.run(1)
    assert _newborn_cohort(pop).sum() == 200


@pytest.mark.parametrize("mother,father,factor", [
    ("incompatible", "normal", 0.6),
    ("normal", "incompatible", 0.6),
    ("incompatible", "incompatible", 0.36),
])
def test_ci_fecundity_belongs_to_each_reproducing_individual(mother: str, father: str, factor: float) -> None:
    pop = _population(f"ci_parent_cost_{mother}_{father}", "fecundity", 0.4, mother=mother, father=father)
    pop.run(1)
    assert _newborn_cohort(pop).sum() == pytest.approx(200 * factor)


def test_ci_fecundity_is_expressed_in_the_following_generation() -> None:
    pop = _population("ci_two_generations", "fecundity", 0.4)
    pop.run(1)
    assert _newborn_cohort(pop).sum() == 200
    # 100 CI females x CI males, both have fecundity .6. They are
    # uninfected, so their 72 children return to the normal label.
    pop.run(1)
    counts = _newborn_cohort(pop)
    normal = [i for i, (_, slab) in enumerate(pop.registry.index_to_ztype) if slab == "normal"]
    assert counts.sum() == pytest.approx(72)
    assert counts[normal].sum() == pytest.approx(72)


@pytest.mark.parametrize("effect,expected", [("zygote_viability", 100), ("viability", 50), ("fecundity", 100)])
def test_density_regulation_distinguishes_cost_stages(effect: Effect, expected: float) -> None:
    pop = _population(f"ci_density_{effect}", effect, 0.5)
    pop.update().competition(growth_mode="fixed", carrying_capacity=100)
    pop.run(1)
    # 200 eggs: embryo cost precedes the K=100 cap, juvenile cost follows
    # the cap, and fecundity cost does not affect the newborn generation.
    assert _newborn_cohort(pop).sum() == pytest.approx(expected)


@pytest.mark.parametrize("effect", ["zygote_viability", "viability"])
def test_stochastic_survival_cost_matches_binomial_moments(effect: Effect) -> None:
    pop = _population(f"ci_binomial_{effect}", effect, 0.4, stochastic=True)
    replicates = 400
    values = []
    for seed in range(replicates):
        pop._rust_backend_seed = seed  # independent reproducible streams for this distribution check
        pop.reset()
        pop.run(1)
        values.append(_newborn_cohort(pop).sum())
    samples = np.asarray(values)
    # Fixed 200 eggs and survival .6: Binomial(200,.6), mean 120,
    # variance 48. Six SE bounds control false failures across both modes.
    assert samples.mean() == pytest.approx(120, abs=6 * np.sqrt(48 / replicates))
    assert samples.var(ddof=1) == pytest.approx(48, abs=6 * 48 * np.sqrt(2 / (replicates - 1)))


@pytest.mark.parametrize("name", ["viability_scaling", "fecundity_scaling"])
@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_infection_costs_reject_nonfinite_or_negative_values(name: str, bad: float) -> None:
    with pytest.raises(ValueError, match=name):
        nt.Wolbachia("bad", **{name: bad})
