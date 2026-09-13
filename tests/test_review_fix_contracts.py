"""Additional error recovery and parser boundary contracts for review fixes."""

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet
from natal.frontend.presets._types import carrier_pattern


def test_deleted_baseline_catalog_recovers_from_structure():
    """Deleting returned cache fields must not poison subsequent builds."""
    species = nt.Species.from_dict(name="fix_deleted_cache", structure={"c": {"l": ["A", "B"]}})
    baseline = species.get_config_blueprint()
    expected = baseline["offspring_tensor"].copy()
    baseline.clear()
    fresh = species.get_config_blueprint()
    np.testing.assert_array_equal(fresh["offspring_tensor"], expected)
    assert species.get_config_blueprint() is fresh


@pytest.mark.parametrize("stage,pattern", [
    ("gamete", "A;"), ("gamete", ";A"), ("gamete", "A;;*"),
    ("gamete", "(A"), ("gamete", "A)"), ("gamete", "(A;)"),
    ("gamete", "A@default@default"), ("gamete", "!{}"),
    ("zygote", "A|A;"), ("zygote", ";A|A"), ("zygote", "A|A;;*"),
    ("zygote", "(A|A"), ("zygote", "A|A)"), ("zygote", "(A|A;)"),
    ("zygote", "A|A@default@default"),
])
def test_malformed_filter_cannot_silently_match(stage, pattern):
    """Empty segments and mismatched delimiters fail at conversion compilation."""
    species = nt.Species.from_dict(name="fix_filter_syntax", structure={"c": {"l": ["A", "B"]}})
    host = SimpleNamespace(species=species, registry=build_registry(species))
    if stage == "gamete":
        rules = GameteConversionRuleSet().add_gtype_convert(to="*@*", rate=1, filters={"current": pattern})
        with pytest.raises(ValueError):
            rules.to_gamete_modifier(host)
    else:
        rules = ZygoteConversionRuleSet().add_ztype_convert(to="*@*", rate=1, filters={"current": pattern})
        with pytest.raises(ValueError):
            rules.to_zygote_modifier(host)


def test_impossible_carrier_requirement_is_explicit():
    """A diploid cannot satisfy three distinct required alleles at one locus."""
    species = nt.Species.from_dict(name="fix_impossible_carrier", structure={"c": {"l": ["A", "B", "C"]}})
    with pytest.raises(ValueError, match="more than two"):
        carrier_pattern(species, "A", "B", "C")


def test_cached_carrier_pattern_survives_locus_replacement():
    """An existing pattern must resolve the current locus, not retain its old identity."""
    species = nt.Species.from_dict(
        name="fix_carrier_locus_replacement", unordered=False,
        structure={
            "X": {"sex_type": "X", "loci": {"lx": ["D", "d"]}},
            "Y": {"sex_type": "Y", "loci": {"ly": ["Y"]}},
        },
    )
    pattern = nt.GenotypePatternParser(species).parse(carrier_pattern(species, "D"))
    assert pattern.matches(species.get_genotype_from_str("D|Y"))
    chrom = species.get_chromosome("X")
    chrom.remove_locus("lx")
    chrom.add_locus("lx")
    locus = species.get_locus("lx")
    assert locus is not None
    locus.add_alleles([nt.Gene("D", locus=locus), nt.Gene("d", locus=locus)])
    assert pattern.matches(species.get_genotype_from_str("D|Y"))
    assert not pattern.matches(species.get_genotype_from_str("d|Y"))
