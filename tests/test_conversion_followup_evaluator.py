"""Independent CR-1 follow-up contracts for composition and pattern scopes."""

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet
from natal.frontend.modifiers.module import wrap_gamete_modifier, wrap_zygote_modifier
from natal.frontend.patterns import ZygoteTypePattern
from natal.frontend.presets._types import carrier_pattern


def _host():
    species = nt.Species.from_dict(
        name="followup_composition", structure={"chr": {"loc": ["A", "B", "C"]}},
        gamete_labels=["default", "I"], somatic_labels=["default", "I"],
    )
    return SimpleNamespace(species=species, registry=build_registry(species))


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_multiple_rulesets_cascade_current_and_keep_source_fixed(stage):
    """Two independent 1/2 events produce 1/2 A, 1/4 B and 1/4 C joint states."""
    host = _host()
    species, reg = host.species, host.registry
    meiosis, fertilization = project_mendelian_maps(species, reg)
    if stage == "gamete":
        first = GameteConversionRuleSet().add_gtype_convert(
            to="B@I", rate=0.5, filters={"current": "A@default"},
        )
        second = GameteConversionRuleSet().add_gtype_convert(
            to="C@*", rate=0.5,
            filters={"current": "B@I", "parent": "A|A@default", "parent_sex": "female"},
        )
        modifiers = [wrap_gamete_modifier(r.to_gamete_modifier(host), None, reg) for r in (first, second)]
        baseline = meiosis
        row_key = (0, reg.ztype_index(species.get_genotype_from_str("A|A"), "default"))
        expected = {
            reg.gtype_index(species.get_haploid_genotype_from_str(g), label): p
            for g, label, p in [("A", "default", 0.5), ("B", "I", 0.25), ("C", "I", 0.25)]
        }
    else:
        first = ZygoteConversionRuleSet().add_ztype_convert(
            to="B|B@I", rate=0.5, filters={"current": "A|A@default"},
        )
        second = ZygoteConversionRuleSet().add_ztype_convert(
            to="C|C@*", rate=0.5,
            filters={"current": "B|B@I", "maternal": "A@default", "paternal": "A@default"},
        )
        modifiers = [wrap_zygote_modifier(r.to_zygote_modifier(host), None, reg) for r in (first, second)]
        baseline = fertilization
        a = reg.gtype_index(species.get_haploid_genotype_from_str("A"), "default")
        row_key = (a, a)
        expected = {
            reg.ztype_index(species.get_genotype_from_str(g), label): p
            for g, label, p in [("A|A", "default", 0.5), ("B|B", "I", 0.25), ("C|C", "I", 0.25)]
        }
    reference = np.zeros(baseline.shape[-1])
    for i, p in expected.items():
        reference[i] = p
    # Recompilation starts from the baseline every time; never stack the last result.
    for _ in range(2):
        actual = baseline.copy()
        for modifier in modifiers:
            actual = modifier(actual)
        np.testing.assert_allclose(actual[row_key], reference, rtol=0, atol=1e-14)
    fresh = project_mendelian_maps(species, reg)[0 if stage == "gamete" else 1]
    np.testing.assert_array_equal(baseline, fresh)


@pytest.mark.parametrize(
    ("required", "genotype", "expected"),
    [
        (("A", "a"), "A/B|a/b", True),
        (("A", "a"), "A/B|A/b", False),
        (("A", "B"), "A/B|a/b", True),
        (("A", "B"), "A/b|a/B", True),
        (("A", "B"), "a/B|A/b", True),
        (("A", "B"), "a/B|a/b", False),
    ],
)
def test_carrier_requirements_are_per_locus_not_same_haplotype(required, genotype, expected):
    """Carrier membership is conjunction of allele presence, independent of phase."""
    species = nt.Species.from_dict(
        name="followup_carrier_phase", structure={"chr": {"l1": ["A", "a"], "l2": ["B", "b"]}}, unordered=False,
    )
    pattern = ZygoteTypePattern.parse(carrier_pattern(species, *required), species)
    assert pattern.matches(species.get_genotype_from_str(genotype), "default") is expected


@pytest.mark.parametrize(
    ("genotype", "expected"),
    [("A/B|a/b", True), ("a/B|A/b", True), ("A/b|a/B", False), ("a/b|A/B", False)],
)
def test_mixed_locus_ordering_does_not_swap_ordered_locus(genotype, expected):
    """First locus is unordered; second locus must retain maternal B/paternal b."""
    species = nt.Species.from_dict(
        name="followup_mixed_ordering", structure={"chr": {"l1": ["A", "a"], "l2": ["B", "b"]}}, unordered=False,
    )
    pattern = ZygoteTypePattern.parse("(A::a; B|b)", species)
    assert pattern.matches(species.get_genotype_from_str(genotype), "default") is expected


@pytest.mark.parametrize("stage,key", [("gamete", "current"), ("gamete", "parent"), ("zygote", "current"), ("zygote", "maternal"), ("zygote", "paternal")])
@pytest.mark.parametrize("suffix", ["{default,typo}", "!{typo}", "", "{default", "default}"])
def test_filter_label_validation_rejects_invalid_sets_negation_and_syntax(stage, key, suffix):
    """Every named label must exist even in a negated set, and syntax must be complete."""
    host = _host()
    filters = {key: f"*@{suffix}"}
    with pytest.raises(ValueError):
        if stage == "gamete":
            GameteConversionRuleSet().add_gtype_convert(to="*@*", rate=1, filters=filters).to_gamete_modifier(host)()
        else:
            ZygoteConversionRuleSet().add_ztype_convert(to="*@*", rate=1, filters=filters).to_zygote_modifier(host)()


class _RulesPreset(nt.GeneticPreset):
    """Expose an ordered ruleset through the public preset compilation boundary."""

    def __init__(self, name, stage, rules):
        super().__init__(name=name)
        self.stage = stage
        self.rules = rules

    def gamete_modifier(self, host):
        return self.rules.to_gamete_modifier(host) if self.stage == "gamete" else None

    def zygote_modifier(self, host):
        return self.rules.to_zygote_modifier(host) if self.stage == "zygote" else None


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_public_multiple_presets_refresh_does_not_stack_conversion(stage):
    """Build and two refreshes preserve A=1/2, B=1/4, C=1/4 under two presets."""
    species = _host().species
    if stage == "gamete":
        first = GameteConversionRuleSet().add_gtype_convert(to="B@*", rate=0.5, filters={"current": "A@*"})
        second = GameteConversionRuleSet().add_gtype_convert(to="C@*", rate=0.5, filters={"current": "B@*"})
    else:
        first = ZygoteConversionRuleSet().add_ztype_convert(to="B|B@*", rate=0.5, filters={"current": "A|A@*"})
        second = ZygoteConversionRuleSet().add_ztype_convert(to="C|C@*", rate=0.5, filters={"current": "B|B@*"})
    pop = nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False).presets(
        _RulesPreset("first", stage, first), _RulesPreset("second", stage, second),
    ).build()
    reg = pop.registry
    for iteration in range(3):
        if iteration:
            pop.refresh_modifiers()
        if stage == "gamete":
            zidx = reg.ztype_index(species.get_genotype_from_str("A|A"), "default")
            row = pop.config.zygotes_to_gametes_map[0, zidx]
            indices = [reg.gtype_index(species.get_haploid_genotype_from_str(g), "default") for g in ("A", "B", "C")]
        else:
            gidx = reg.gtype_index(species.get_haploid_genotype_from_str("A"), "default")
            row = pop.config.gametes_to_zygotes_map[gidx, gidx]
            indices = [reg.ztype_index(species.get_genotype_from_str(g), "default") for g in ("A|A", "B|B", "C|C")]
        expected = np.zeros(len(row))
        expected[indices] = [0.5, 0.25, 0.25]
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)


def test_x_linked_carrier_with_different_y_locus_count():
    """A D/E X chromosome carries D in XY even though Y has only one locus."""
    species = nt.Species.from_dict(
        name="followup_xlink_unequal_loci",
        structure={
            "chrX": {"sex_type": "X", "loci": {"lx": ["D", "W"], "extra": ["E"]}},
            "chrY": {"sex_type": "Y", "loci": {"ly": ["Y"]}},
        },
        unordered=False,
    )
    pattern = ZygoteTypePattern.parse(carrier_pattern(species, "D"), species)
    assert pattern.matches(species.get_genotype_from_str("D/E|Y"), "default")
    assert not pattern.matches(species.get_genotype_from_str("W/E|Y"), "default")


@pytest.mark.parametrize("system", ["XY", "ZW"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("required", "primary_allele", "partner_allele", "expected"),
    [
        (("D",), "D", "V", True),
        (("D",), "d", "V", False),
        (("D", "d"), "D", "V", False),
        (("V",), "D", "V", True),
        (("V",), "D", "v", False),
        (("V", "v"), "D", "V", False),
    ],
)
def test_heteromorphic_carrier_uses_locus_identity(system, reverse, required, primary_allele, partner_allele, expected):
    """Hemizygous carrier checks cannot borrow a different chromosome's same-index gene."""
    primary, partner = system
    species = nt.Species.from_dict(
        name=f"followup_carrier_identity_{system}",
        structure={
            "primary": {"sex_type": primary, "loci": {"p0": ["E"], "p1": ["D", "d"]}},
            "partner": {"sex_type": partner, "loci": {"q0": ["U"], "q1": ["V", "v"], "q2": ["F"]}},
        },
        unordered=False,
    )
    genomes = [f"E/{primary_allele}", f"U/{partner_allele}/F"]
    if reverse:
        genomes.reverse()
    genotype = species.get_genotype_from_str("|".join(genomes))
    pattern = ZygoteTypePattern.parse(carrier_pattern(species, *required), species)
    assert pattern.matches(genotype, "default") is expected


@pytest.mark.parametrize("system", ["XY", "ZW"])
@pytest.mark.parametrize("heteromorphic", [False, True])
def test_carrier_can_require_genes_on_both_sex_chromosomes(system, heteromorphic):
    """Requiring both sex-chromosome genes is conjunction, not a homology claim."""
    primary, partner = system
    species = nt.Species.from_dict(
        name=f"followup_both_sex_members_{system}",
        structure={
            "primary": {"sex_type": primary, "loci": {"lp": ["D"]}},
            "partner": {"sex_type": partner, "loci": {"lq": ["V"]}},
        },
        unordered=False,
    )
    genotype = species.get_genotype_from_str("D|V" if heteromorphic else "D|D")
    pattern = ZygoteTypePattern.parse(carrier_pattern(species, "D", "V"), species)
    assert pattern.matches(genotype, "default") is heteromorphic
