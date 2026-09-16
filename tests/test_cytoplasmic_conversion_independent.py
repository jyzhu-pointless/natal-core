"""Independent contracts for cytoplasmic conversion on incoming distributions."""

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.modifiers import ZygoteConversionRuleSet
from natal.frontend.modifiers.module import wrap_gamete_modifier, wrap_zygote_modifier
from natal.frontend.presets.cytoplasmic import CytoplasmicPreset


class TwoTags(CytoplasmicPreset):
    _maternal_map = {"infected_a": "tag_a", "infected_b": "tag_b"}


def species():
    return nt.Species.from_dict(
        "independent_cytoplasm", {"c": {"l": ["A", "B"]}},
        gamete_labels=["baseline", "tag_a", "tag_b", "default"],
        somatic_labels=["normal", "infected_a", "infected_b", "default"],
    )


def fixture():
    sp = species()
    reg = build_registry(sp)
    host = SimpleNamespace(species=sp, registry=reg, index_registry=reg)
    meiosis, fertilization = project_mendelian_maps(sp, reg)
    return sp, reg, host, meiosis, fertilization


def test_gamete_explicit_source_label_and_multiple_maternal_tags():
    _, reg, host, meiosis, _ = fixture()
    incoming = meiosis * 0.75
    # A preceding modifier has tagged one quarter of every gamete row.
    for haplo in reg.index_to_haplo:
        base = reg.gtype_index(haplo, "baseline")
        incoming[:, :, reg.gtype_index(haplo, "default")] += meiosis[:, :, base] * 0.25
    before = incoming.copy()
    mod = TwoTags(default_glab="baseline", default_slab="normal").gamete_modifier(host)
    assert mod is not None
    actual = wrap_gamete_modifier(mod, None, reg)(incoming)
    expected = incoming.copy()
    for genotype in reg.index_to_genotype:
        for slab, tag in TwoTags._maternal_map.items():
            zi = reg.ztype_index(genotype, slab)
            for haplo in reg.index_to_haplo:
                base = reg.gtype_index(haplo, "baseline")
                expected[0, zi, reg.gtype_index(haplo, tag)] += expected[0, zi, base]
                expected[0, zi, base] = 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(incoming, before)
    np.testing.assert_allclose(actual.sum(axis=-1), meiosis.sum(axis=-1))


def test_zygote_composition_preserves_prior_genotype_and_nondefault_label():
    _, reg, host, _, fertilization = fixture()
    prior = ZygoteConversionRuleSet()
    prior.add_ztype_convert(to="B|B@default", rate=0.25, filters={"current": "*@normal"})
    prior.add_ztype_convert(to="B|B@*", rate=1.0)
    before_cyto = wrap_zygote_modifier(prior.to_zygote_modifier(host), None, reg)(fertilization)
    mod = TwoTags(default_glab="baseline", default_slab="normal").zygote_modifier(host)
    assert mod is not None
    cyto = wrap_zygote_modifier(mod, None, reg)
    actual = cyto(before_cyto)
    expected = before_cyto.copy()
    for haplo in reg.index_to_haplo:
        for slab, tag in TwoTags._maternal_map.items():
            maternal = reg.gtype_index(haplo, tag)
            for genotype in reg.index_to_genotype:
                source = reg.ztype_index(genotype, "normal")
                dest = reg.ztype_index(genotype, slab)
                expected[maternal, :, dest] += expected[maternal, :, source]
                expected[maternal, :, source] = 0
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, before_cyto)
    np.testing.assert_allclose(actual.sum(axis=-1), fertilization.sum(axis=-1))
    # After inheritance the first rule cannot see tagged mothers' offspring.
    reversed_result = wrap_zygote_modifier(prior.to_zygote_modifier(host), None, reg)(cyto(fertilization))
    assert not np.array_equal(reversed_result, actual)


def test_nonfirst_source_labels_select_only_the_requested_probability_mass():
    _, reg, host, meiosis, fertilization = fixture()
    preset = TwoTags(default_glab="default", default_slab="default")
    gametes = meiosis * 0.75
    for haplo in reg.index_to_haplo:
        gametes[:, :, reg.gtype_index(haplo, "default")] = (
            meiosis[:, :, reg.gtype_index(haplo, "baseline")] * 0.25
        )
    gm = preset.gamete_modifier(host)
    zm = preset.zygote_modifier(host)
    assert gm is not None and zm is not None
    actual_g = wrap_gamete_modifier(gm, None, reg)(gametes)
    expected_g = gametes.copy()
    for genotype in reg.index_to_genotype:
        for slab, tag in preset._maternal_map.items():
            zi = reg.ztype_index(genotype, slab)
            for haplo in reg.index_to_haplo:
                src = reg.gtype_index(haplo, "default")
                expected_g[0, zi, reg.gtype_index(haplo, tag)] += expected_g[0, zi, src]
                expected_g[0, zi, src] = 0
    np.testing.assert_array_equal(actual_g, expected_g)

    zygotes = fertilization * 0.75
    for genotype in reg.index_to_genotype:
        zygotes[:, :, reg.ztype_index(genotype, "default")] = (
            fertilization[:, :, reg.ztype_index(genotype, "normal")] * 0.25
        )
    expected_z = zygotes.copy()
    for haplo in reg.index_to_haplo:
        for slab, tag in preset._maternal_map.items():
            maternal = reg.gtype_index(haplo, tag)
            for genotype in reg.index_to_genotype:
                src = reg.ztype_index(genotype, "default")
                expected_z[maternal, :, reg.ztype_index(genotype, slab)] += expected_z[maternal, :, src]
                expected_z[maternal, :, src] = 0
    np.testing.assert_array_equal(wrap_zygote_modifier(zm, None, reg)(zygotes), expected_z)
    # Selecting an absent baseline mass never changes Species' default maps.
    np.testing.assert_array_equal(wrap_gamete_modifier(gm, None, reg)(meiosis), meiosis)
    np.testing.assert_array_equal(wrap_zygote_modifier(zm, None, reg)(fertilization), fertilization)


@pytest.mark.parametrize("label", ["missing", "*", ""])
@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_invalid_explicit_source_labels_fail_compilation(label, stage):
    _, _, host, _, _ = fixture()
    preset = TwoTags(**{"default_glab" if stage == "gamete" else "default_slab": label})
    with pytest.raises(ValueError, match="unknown default_"):
        getattr(preset, stage + "_modifier")(host)


@pytest.mark.parametrize("compress", [False, True])
def test_two_maternal_tags_survive_population_build_and_reproduction(compress):
    sp = species()
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=sp, stochastic=False, compress=compress)
        .initial_state(individual_count={
            "female": {"A|A@normal": 20, "A|A@infected_a": 30, "A|A@infected_b": 50},
            "male": {"A|A@infected_b": 100},
        })
        .competition(juvenile_growth_mode=0)
        .reproduction(eggs_per_female=2)
        .survival(female_age0_survival=1, male_age0_survival=1)
        .presets(TwoTags(default_glab="baseline", default_slab="normal"))
        .build()
    )
    pop.run(1)
    counts = pop.state.individual_count.sum(axis=(0, 1))
    reg = pop.index_registry
    for slab, expected in {"normal": 40, "infected_a": 60, "infected_b": 100, "default": 0}.items():
        indices = [i for i, (_, label) in enumerate(reg.index_to_ztype) if label == slab]
        assert counts[indices].sum() == pytest.approx(expected)
    if compress:
        assert len(reg.index_to_genotype) == 1
