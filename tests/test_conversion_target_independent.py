"""Independent contracts for partial conversion targets shared by modifiers and Ops."""

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet
from natal.frontend.modifiers.module import wrap_gamete_modifier, wrap_zygote_modifier
from natal.frontend.patterns.parser import GenotypePatternParser


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_omitted_chromosome_in_modifier_target_preserves_each_source(stage: str) -> None:
    """A first-chromosome replacement leaves the independent second locus unchanged."""
    species = nt.Species.from_dict(
        name=f"independent_partial_modifier_{stage}",
        structure={"first": {"one": ["A", "a"]}, "second": {"two": ["B", "b"]}},
        somatic_labels=["default", "infected"],
        gamete_labels=["default", "infected"],
    )
    registry = build_registry(species)
    host = SimpleNamespace(species=species, registry=registry)
    meiosis, fertilization = project_mendelian_maps(species, registry)
    if stage == "gamete":
        rules = GameteConversionRuleSet().add_gtype_convert(to="a@infected", rate=1.0)
        modifier = wrap_gamete_modifier(rules.to_gamete_modifier(host), None, registry)
        actual = modifier(meiosis)
        source = registry.ztype_index(species.get_genotype_from_str("A|A;B|b"), "default")
        expected = np.zeros(registry.n_gtypes)
        for second in ("B", "b"):
            target = species.get_haploid_genotype_from_str(f"a;{second}")
            expected[registry.gtype_index(target, "infected")] = 0.5
        np.testing.assert_array_equal(actual[0, source], expected)
    else:
        rules = ZygoteConversionRuleSet().add_ztype_convert(to="a|a@infected", rate=1.0)
        modifier = wrap_zygote_modifier(rules.to_zygote_modifier(host), None, registry)
        actual = modifier(fertilization)
        maternal = registry.gtype_index(species.get_haploid_genotype_from_str("A;B"), "default")
        paternal = registry.gtype_index(species.get_haploid_genotype_from_str("A;b"), "default")
        expected = np.zeros(registry.n_ztypes)
        target = species.get_genotype_from_str("a|a;B|b")
        expected[registry.ztype_index(target, "infected")] = 1.0
        np.testing.assert_array_equal(actual[maternal, paternal], expected)


def test_unordered_partial_target_cannot_choose_a_source_side() -> None:
    """Unordered matching must not silently become a left-side assignment."""
    species = nt.Species.from_dict(
        name="independent_unordered_partial_target",
        structure={"first": {"one": ["A", "B", "C"]}},
        unordered=False,
    )
    source = species.get_genotype_from_str("A|B")
    with pytest.raises(ValueError):
        nt.parse_target("C::*@*", species=species).apply_zygote(source, "default", species)


def test_complete_target_preserves_exact_parser_chromosome_identity_semantics() -> None:
    """A legacy complete genotype names chromosomes by their genes, not string order."""
    species = nt.Species.from_dict(
        name="independent_exact_target_order",
        structure={"first": {"one": ["A", "a"]}, "second": {"two": ["B", "b"]}},
        unordered=False,
    )
    source = species.get_genotype_from_str("A|A;B|B")
    expected = species.get_genotype_from_str("B|b;A|a")
    actual, label = nt.parse_target("B|b;A|a@*", species=species).apply_zygote(source, "default", species)
    assert actual == expected
    assert label == "default"


def test_complete_target_can_replace_sex_chromosome_identity() -> None:
    """A complete legal XY genotype can replace XX without a cross-locus partial edit."""
    species = nt.Species.from_dict(
        name="independent_exact_target_sex_chromosome",
        structure={
            "chrA": {"loci": {"A": ["A"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )
    source = species.get_genotype_from_str("A|A;X1|X1")
    expected = species.get_genotype_from_str("A|A;X1|Y1")
    actual, label = nt.parse_target("A|A;X1|Y1@*", species=species).apply_zygote(source, "default", species)
    assert actual == expected
    assert label == "default"


@pytest.mark.parametrize(
    ("target", "expected"),
    [("a/*|*", "a/B|a/b"), ("*|A/*", "A/B|A/b"), ("A/*|*", "A/B|a/b")],
)
def test_ordered_partial_locus_replacement_preserves_other_source_positions(target: str, expected: str) -> None:
    """An explicit allele changes one homolog and one locus only; omitted label stays."""
    species = nt.Species.from_dict(
        name="independent_partial_locus",
        structure={"chr": {"one": ["A", "a"], "two": ["B", "b"]}},
        unordered=False,
    )
    source = species.get_genotype_from_str("A/B|a/b")
    actual, label = nt.parse_target(target, species=species).apply_zygote(source, "existing", species)
    assert actual == species.get_genotype_from_str(expected)
    assert label == "existing"


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
@pytest.mark.parametrize("target", ["A", "@default", "A@", "A@@default"])
def test_legacy_rule_declaration_still_requires_valid_explicit_label_separator(stage: str, target: str) -> None:
    """Shared parsing must preserve legacy declaration-time malformed-target errors."""
    with pytest.raises(ValueError):
        if stage == "gamete":
            GameteConversionRuleSet().add_gtype_convert(to=target, rate=1)
        else:
            ZygoteConversionRuleSet().add_ztype_convert(to=target, rate=1)


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
@pytest.mark.parametrize("target", ["*@missing", "(@*", "*@{default}", "*@!default"])
def test_modifier_target_rejects_invalid_syntax_or_nonliteral_labels(stage: str, target: str) -> None:
    """Targets require a concrete registered label or preservation, even singleton sets."""
    species = nt.Species.from_dict(
        name=f"independent_target_errors_{stage}", structure={"chr": {"loc": ["A", "a"]}},
    )
    host = SimpleNamespace(species=species, registry=build_registry(species))
    with pytest.raises(ValueError):
        if stage == "gamete":
            GameteConversionRuleSet().add_gtype_convert(to=target, rate=1).to_gamete_modifier(host)()
        else:
            ZygoteConversionRuleSet().add_ztype_convert(to=target, rate=1).to_zygote_modifier(host)()


@pytest.mark.parametrize("target", ["*/B/*|*", "Unknown/*|*", "B/*|*", "{A}/*|*", "A|A;*"])
def test_partial_target_cannot_invent_locus_correspondence(target: str) -> None:
    """Partial replacements reject unknown alleles, wrong loci, and incomplete chromosomes."""
    species = nt.Species.from_dict(
        name="independent_partial_target_error",
        structure={"first": {"one": ["A", "a"], "two": ["B", "b"]}, "second": {"three": ["C"]}},
        unordered=False,
    )
    source = species.get_genotype_from_str("A/B|a/b;C|C")
    with pytest.raises(ValueError):
        parsed = nt.parse_target(target, species=species)
        parsed.apply_zygote(source, "default", species)


def test_target_stage_mismatch_is_rejected_explicitly() -> None:
    """A compiled haploid target must not be applied to a diploid source, or vice versa."""
    species = nt.Species.from_dict("independent_target_stage_error", {"chr": {"loc": ["A"]}})
    source = species.get_genotype_from_str("A|A")
    with pytest.raises(TypeError, match="diploid"):
        nt.parse_target("*@*", haploid=True, species=species).apply_zygote(source, "default", species)
    with pytest.raises(TypeError, match="haploid"):
        nt.parse_target("*@*", species=species).apply_gamete(source.maternal, "default", species)


def test_target_parser_rejects_nontext_inputs() -> None:
    """Both declaration and species-aware parsing reject accidental numeric target values."""
    species = nt.Species.from_dict("independent_target_type_error", {"chr": {"loc": ["A"]}})
    with pytest.raises(TypeError):
        GenotypePatternParser.split_conversion_target(42)
    with pytest.raises(TypeError):
        nt.parse_target(42, species=species)


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
@pytest.mark.parametrize("token", ["{A,a}", "Unknown"])
def test_ambiguous_target_is_rejected_even_when_filters_have_no_reachable_branch(stage: str, token: str) -> None:
    """Forbidden sets or unknown alleles remain invalid even when no branch reaches them."""
    species = nt.Species.from_dict(
        name=f"independent_unreachable_target_{stage}",
        structure={"chr": {"one": ["A", "a"], "two": ["B", "b"]}},
        unordered=False,
    )
    host = SimpleNamespace(species=species, registry=build_registry(species))
    with pytest.raises(ValueError):
        if stage == "gamete":
            rules = GameteConversionRuleSet().add_gtype_convert(
                to=f"{token}/*@*", rate=1,
                filters={"current": "a/*", "parent": "A/*|A/*"},
            )
            rules.to_gamete_modifier(host)()
        else:
            rules = ZygoteConversionRuleSet().add_ztype_convert(
                to=f"{token}/*|*@*", rate=1,
                filters={"current": "a/*|a/*", "maternal": "A/*"},
            )
            rules.to_zygote_modifier(host)()
