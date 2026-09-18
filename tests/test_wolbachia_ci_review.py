"""Independent CI contracts for lifecycle, composition, and reconfiguration."""
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.modifiers import ZygoteConversionRuleSet
from natal.frontend.modifiers.module import wrap_zygote_modifier


def _species(name: str, *, unordered: bool = True, labels: bool = True) -> nt.Species:
    """Declare infection, CI-origin, and protected labels for isolated tests."""
    return nt.Species.from_dict(
        name, {"c": {"l": ["A", "a"]}}, unordered=unordered,
        somatic_labels=["normal", "infected", "incompatible", "protected"] if labels else ["normal", "infected"],
        gamete_labels=["default", "wolbachia", "wolbachia_ci"] if labels else ["default", "wolbachia"],
    )


def _builder(species: nt.Species, *, compress: bool = False) -> nt.PopulationBuilder:
    """Use neutral ecology and a 20% infected population of both sexes."""
    return (nt.DiscreteGenerationPopulation.setup(species, stochastic=False, compress=compress)
            .initial_state(individual_count={"female": {"A|A@normal": 80, "A|A@infected": 20},
                                             "male": {"A|A@normal": 80, "A|A@infected": 20}})
            .reproduction(eggs_per_female=2)
            .survival(female_age0_survival=1, male_age0_survival=1)
            .competition(juvenile_growth_mode="no_competition"))


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("unordered", [False, True])
def test_classic_ci_recursion_includes_surviving_uninfected_ci_cohort(compress: bool, unordered: bool) -> None:
    """Compare six generations with the independently derived cross sum."""
    sp = _species(f"ci_review_recursion_{compress}_{unordered}", unordered=unordered)
    pop = _builder(sp, compress=compress).presets(nt.Wolbachia(
        "w", incompatibility_cost=.4, viability_scaling=.9, fecundity_scaling=.8,
    )).build()
    p = .2
    for _ in range(6):
        # Male fecundity .8 also changes cross output: uninfected mothers
        # produce (1-p) + p*.8*(1-.4), infected mothers .8*((1-p)+p*.8).
        male_output = (1-p) + p*.8
        infected_output = p*.8*.9*male_output
        normal_output = (1-p)*((1-p)+p*.8*.6)
        p = infected_output / (infected_output + normal_output)
        pop.run(1)
        counts = pop.state.individual_count.sum(axis=(0, 1))
        infected = [i for i, (_, slab) in enumerate(pop.registry.index_to_ztype) if slab == "infected"]
        assert counts[infected].sum()/counts.sum() == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("options", [
    {"incompatibility_effect": "reproductive_output"},
    {"infected_slab": "normal"},
    {"incompatibility_slab": "infected"},
    {"incompatibility_slab": "normal"},
    {"default_glab": "wolbachia"},
    {"paternal_glab": "default"},
    {"paternal_glab": "wolbachia"},
])
def test_invalid_ci_roles_rejected(options: dict[str, str]) -> None:
    """Reject overlapping tag roles or an unsupported fitness effect."""
    with pytest.raises(ValueError):
        nt.Wolbachia("bad", incompatibility_cost=.4, **options)


@pytest.mark.parametrize("options", [{"paternal_glab": "missing"}, {"incompatibility_slab": "missing"}])
def test_missing_ci_labels_rejected_at_build(options: dict[str, str]) -> None:
    """Reject required tags absent from the species registry."""
    sp = _species("ci_review_missing_" + next(iter(options)))
    with pytest.raises(ValueError, match="Unknown"):
        _builder(sp).presets(nt.Wolbachia("bad", incompatibility_cost=.4, **options)).build()


def test_custom_ci_labels_and_prior_modifier_keep_genotype_and_other_slabs() -> None:
    """CI and rescue preserve incoming genotypes and non-source somatic mass."""
    sp = nt.Species.from_dict("ci_review_composition", {"c": {"l": ["A", "B"]}},
                             somatic_labels=["clear", "carrying", "damaged", "protected"],
                             gamete_labels=["untagged", "wolbachia", "paternal"])
    reg = build_registry(sp)
    host = SimpleNamespace(species=sp, registry=reg, index_registry=reg)
    _, incoming = project_mendelian_maps(sp, reg)
    prior = ZygoteConversionRuleSet()
    prior.add_ztype_convert(to="B|B@protected", rate=.25, filters={"current": "*@clear"})
    prior.add_ztype_convert(to="B|B@*", rate=1)
    previous = wrap_zygote_modifier(prior.to_zygote_modifier(host), None, reg)(incoming)
    preset = nt.Wolbachia("custom", infected_slab="carrying", normal_slab="clear",
                         default_glab="untagged", paternal_glab="paternal",
                         incompatibility_slab="damaged", incompatibility_cost=.4)
    mod = preset.zygote_modifier(host)
    actual = wrap_zygote_modifier(mod, None, reg)(previous)
    aa = sp.get_haploid_genotype_from_str("A")
    father = reg.gtype_index(aa, "paternal")
    genotype = sp.get_genotype_from_str("B|B")
    for mother_tag, expected_slab in [("untagged", "damaged"), ("wolbachia", "carrying")]:
        row = actual[reg.gtype_index(aa, mother_tag), father]
        assert row[reg.ztype_index(genotype, expected_slab)] == .75
        assert row[reg.ztype_index(genotype, "protected")] == .25
        assert row.sum() == 1
    np.testing.assert_array_equal(previous.sum(axis=-1), actual.sum(axis=-1))


@pytest.mark.parametrize("effect,first", [("zygote_viability", 120), ("viability", 200)])
def test_survival_stage_respects_last_juvenile_age(effect: Literal["zygote_viability", "viability"], first: float) -> None:
    """Ordinary viability acts at the last juvenile age, after embryo survival."""
    sp = _species("ci_review_stage_" + effect)
    pop = (nt.AgeStructuredPopulation.setup(sp, stochastic=False)
           .age_structure(n_ages=4, new_adult_age=2)
           .initial_state(individual_count={"female": {"A|A@normal": {2: 100}}, "male": {"A|A@infected": {2: 100}}})
           .survival(female_age_based_survival=[1, 1, 1, 0], male_age_based_survival=[1, 1, 1, 0])
           .reproduction(eggs_per_female=2)
           .competition(juvenile_growth_mode="no_competition")
           .presets(nt.Wolbachia("w", incompatibility_cost=.4, incompatibility_effect=effect)).build())
    pop.run(1)
    assert pop.state.individual_count[:, 1, :].sum() == pytest.approx(first)
    pop.run(1)
    assert pop.state.individual_count[:, 2, :].sum() == pytest.approx(120)


@pytest.mark.parametrize("compress", [False, True])
def test_reconfigure_ci_cost_and_effect_refresh_without_accumulation(compress: bool) -> None:
    """Refresh matches fresh fitness and an invalid reconfiguration is atomic."""
    sp = _species(f"ci_review_refresh_{compress}")
    preset = nt.Wolbachia("w", incompatibility_cost=.4)
    pop = _builder(sp, compress=compress).presets(preset).build()
    for effect, cost in [("fecundity", .8), ("viability", .2), ("zygote_viability", 0)]:
        pop.update().reconfigure_preset(preset, incompatibility_cost=cost, incompatibility_effect=effect)
        pop.refresh_modifiers()
        fresh = _builder(sp, compress=compress).presets(nt.Wolbachia(
            "fresh", incompatibility_cost=cost, incompatibility_effect=effect)).build()
        # A compressed zero-cost fresh build may prune the now-unused CI slab.
        for field in ["viability_fitness", "fecundity_fitness", "zygote_viability_fitness"]:
            for i, ztype in enumerate(fresh.registry.index_to_ztype):
                j = pop.registry.ztype_index(*ztype)
                np.testing.assert_array_equal(getattr(pop.config, field)[..., j], getattr(fresh.config, field)[..., i])
    before = pop.config.offspring_tensor.copy()
    with pytest.raises(ValueError, match="incompatibility_cost"):
        pop.update().reconfigure_preset(preset, incompatibility_cost=float("nan"))
    assert preset.incompatibility_cost == 0
    np.testing.assert_array_equal(pop.config.offspring_tensor, before)


@pytest.mark.parametrize("effect,expected", [("zygote_viability", 120), ("viability", 120), ("fecundity", 200)])
def test_spatial_ci_preset_and_one_deme_reconfiguration_are_isolated(effect: Literal["zygote_viability", "viability", "fecundity"], expected: float) -> None:
    """Changing a shared preset in one deme preserves its neighbor's effect."""
    sp = _species("ci_review_spatial_" + effect)
    preset = nt.Wolbachia("w", incompatibility_cost=.4, incompatibility_effect=effect)
    pop = (nt.SpatialPopulation.builder(sp, n_demes=2, pop_type="discrete_generation")
           .setup(stochastic=False, compress=True)
           .initial_state(individual_count={"female": {"A|A@normal": 100}, "male": {"A|A@infected": 100}})
           .reproduction(eggs_per_female=2)
           .survival(female_age0_survival=1, male_age0_survival=1)
           .competition(juvenile_growth_mode="no_competition")
           .presets(preset).build())
    pop.deme(0).update().reconfigure_preset(preset, incompatibility_cost=0)
    pop.run(1)
    assert pop.deme(0).get_total_count() == pytest.approx(200)
    assert pop.deme(1).get_total_count() == pytest.approx(expected)
