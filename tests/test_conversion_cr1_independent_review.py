"""Independent regression checks for the CR-1 conversion migration."""

from types import SimpleNamespace

import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet


@pytest.mark.parametrize(
    ("stage", "key"),
    [("gamete", "current"), ("gamete", "parent"),
     ("zygote", "current"), ("zygote", "maternal"), ("zygote", "paternal")],
)
def test_conversion_rejects_unknown_filter_label(stage, key):
    """A misspelled exact label must fail, rather than silently disable a rule."""
    species = nt.Species.from_dict(
        name=f"review_label_{stage}_{key}", structure={"chr": {"loc": ["A", "B"]}},
    )
    host = SimpleNamespace(species=species, registry=build_registry(species))
    if stage == "gamete":
        rules = GameteConversionRuleSet()
        rules.add_gtype_convert(to="*@*", rate=1, filters={key: "*@typo"})
        with pytest.raises(ValueError):
            rules.to_gamete_modifier(host)()
    else:
        rules = ZygoteConversionRuleSet()
        rules.add_ztype_convert(to="*@*", rate=1, filters={key: "*@typo"})
        with pytest.raises(ValueError):
            rules.to_zygote_modifier(host)()


def test_homing_requires_all_requested_carrier_alleles():
    """An A/C parent lacks Cas9 B, so its gametes must remain 1/2 A, 1/2 C."""
    species = nt.Species.from_dict(
        name="review_same_locus_cas9", structure={"chr": {"loc": ["A", "B", "C"]}},
    )
    registry = build_registry(species)
    host = SimpleNamespace(species=species, registry=registry)
    preset = nt.HomingDrive(
        name="split", drive_allele="A", target_allele="C", cas9_allele="B",
        drive_conversion_rate=1, species=species,
    )
    rows = preset.gamete_modifier(host)()
    zidx = registry.ztype_index(species.get_genotype_from_str("A|C"), "default")
    target = registry.gtype_index(species.get_haploid_genotype_from_str("C"), "default")
    assert rows[0, zidx].get(target, 0.0) == pytest.approx(0.5, abs=1e-14)


def test_autosomal_homing_works_in_xy_species():
    """An XX D/W carrier at rate 1 must emit only D gametes, even with XY axes."""
    species = nt.Species.from_dict(
        name="review_xy_autosomal_homing",
        structure={
            "auto": {"loc": ["D", "W"]},
            "chrX": {"sex_type": "X", "loci": {"lx": ["X"]}},
            "chrY": {"sex_type": "Y", "loci": {"ly": ["Y"]}},
        },
    )
    registry = build_registry(species)
    host = SimpleNamespace(species=species, registry=registry)
    preset = nt.HomingDrive(
        name="drive", drive_allele="D", target_allele="W",
        drive_conversion_rate=1, species=species,
    )
    rows = preset.gamete_modifier(host)()
    zidx = registry.ztype_index(species.get_genotype_from_str("D|W;X|X"), "default")
    target = registry.gtype_index(species.get_haploid_genotype_from_str("D;X"), "default")
    assert rows[0, zidx].get(target, 0.0) == pytest.approx(1.0, abs=1e-14)


def test_disjoint_rulesets_do_not_reset_preceding_sex_conversion():
    """A male-only modifier cannot restore the female row changed by an earlier modifier."""
    from natal.frontend.genetics.compile import project_mendelian_maps
    from natal.frontend.modifiers.module import wrap_gamete_modifier

    species = nt.Species.from_dict(
        name="review_disjoint_rulesets", structure={"chr": {"loc": ["A", "B", "C"]}},
    )
    registry = build_registry(species)
    host = SimpleNamespace(species=species, registry=registry)
    female_rules = GameteConversionRuleSet().add_allele_convert(
        from_allele="A", to_allele="B", rate=1, filters={"parent_sex": "female"},
    )
    male_rules = GameteConversionRuleSet().add_allele_convert(
        from_allele="A", to_allele="C", rate=1, filters={"parent_sex": "male"},
    )
    tensor, _ = project_mendelian_maps(species, registry)
    for rules in (female_rules, male_rules):
        tensor = wrap_gamete_modifier(rules.to_gamete_modifier(host), None, registry)(tensor)
    zidx = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    bidx = registry.gtype_index(species.get_haploid_genotype_from_str("B"), "default")
    cidx = registry.gtype_index(species.get_haploid_genotype_from_str("C"), "default")
    assert tensor[0, zidx, bidx] == pytest.approx(1.0, abs=1e-14)
    assert tensor[1, zidx, cidx] == pytest.approx(1.0, abs=1e-14)
