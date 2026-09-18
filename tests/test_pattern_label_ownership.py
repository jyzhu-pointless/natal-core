"""Contract tests for label ownership on the pattern types.

FRONTEND_REFACTOR_PLAN.md §5.2 fixes each pattern type's responsibility: a
``GenotypePattern`` matches a ``Genotype`` and carries no slab, a
``HaploidGenomePattern`` matches a ``HaploidGenome`` and carries no glab, while
``ZygoteTypePattern`` and ``GameteTypePattern`` compose those content patterns
with a label pattern.  §5.4 removes the low-level ``@``-stripping that could
drop a label written in a nested position, and §5.6 asks for inaccessibility
tests for the entries the refactor removed.

These tests pin the observable contract.  The same behaviour was cross-checked
against the parent revision with a differential probe (old and new agree on
every matching set; the only differences are the intended rejections).
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.patterns import PatternParseError, ZygoteTypePattern
from natal.frontend.patterns.elements.diploid import GenotypePattern
from natal.frontend.patterns.elements.haploid import (
    GameteTypePattern,
    HaploidGenomePattern,
)
from natal.frontend.patterns.parser import GenotypePatternParser

SOMATIC_LABELS = ["default", "infected", "cas9_high"]


@pytest.fixture(scope="module")
def species() -> nt.Species:
    """Two chromosome groups and three somatic labels."""
    return nt.Species.from_dict(
        "label_ownership_two_chr",
        {"chr1": {"loc1": ["A", "a"]}, "chr2": {"loc2": ["X", "Y"]}},
        gamete_labels=["default", "cas9_deposited", "cas9_high"],
        somatic_labels=list(SOMATIC_LABELS),
    )


@pytest.fixture(scope="module")
def registry(species: nt.Species) -> nt.IndexRegistry:
    """A registry holding every (genotype, slab) pair of the species."""
    registry = nt.IndexRegistry()
    for slab in SOMATIC_LABELS:
        registry.register_somatic_label(slab)
    for genotype in species.get_all_genotypes():
        for slab in SOMATIC_LABELS:
            registry.register_ztype(genotype, slab)
    return registry


def test_from_slab_key_is_removed(species: nt.Species) -> None:
    """§5.2/§5.6: the redundant constructor entry is gone for good."""
    assert not hasattr(ZygoteTypePattern, "from_slab_key")
    assert not hasattr(nt.parse_selector("A|a; X|X", species=species), "from_slab_key")


def test_content_patterns_carry_no_label(species: nt.Species) -> None:
    """§5.2: Genotype/HaploidGenome patterns match content only."""
    genotype_pattern = nt.parse_selector("A|a; X|X", species=species, kind="genotype")
    haploid_pattern = nt.parse_selector("A; X", species=species, kind="haploid")
    for pattern in (genotype_pattern, haploid_pattern):
        assert not hasattr(pattern, "lab")
        assert "lab" not in vars(pattern)


def test_content_pattern_constructors_no_longer_take_a_label() -> None:
    """§5.2: the ``lab=`` argument went with the field it assigned."""
    with pytest.raises(TypeError):
        GenotypePattern([], lab=None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        HaploidGenomePattern([], lab=None)  # type: ignore[call-arg]


def test_gamete_type_pattern_exposes_a_complete_genome(species: nt.Species) -> None:
    """§5.2: GameteTypePattern pairs a whole genome pattern with ``glab``."""
    gamete = nt.parse_selector(
        "A; X@cas9_deposited"
    , species=species, kind="gtype")
    assert isinstance(gamete, GameteTypePattern)
    assert not hasattr(gamete, "haplotype_path")
    assert not hasattr(gamete, "lab")
    assert gamete.glab is not None
    assert gamete.glab.matches("cas9_deposited")
    assert not gamete.glab.matches("default")
    assert len(gamete.genome.haplotype_patterns) == 2


def test_gamete_genome_keeps_each_chromosome_separate(species: nt.Species) -> None:
    """§5.2: a multi-chromosome gamete selector is not flattened."""
    gamete = nt.parse_selector("A; X", species=species, kind="gtype")
    assert gamete.genome.matches(species.get_haploid_genotype_from_str("A; X"))
    assert not gamete.genome.matches(species.get_haploid_genotype_from_str("A; Y"))
    assert not gamete.genome.matches(species.get_haploid_genotype_from_str("a; X"))


def test_reprs_show_a_label_only_on_the_type_patterns(species: nt.Species) -> None:
    """§5.2: the textual form carries the same label ownership as the fields."""
    assert "@" not in repr(nt.parse_selector("A|a; X|X", species=species, kind="genotype"))
    assert "@" not in repr(nt.parse_selector("A; X", species=species, kind="haploid"))
    gamete = repr(nt.parse_selector("A; X@cas9_deposited", species=species, kind="gtype"))
    assert "LabPattern(cas9_deposited)" in gamete
    assert "@" not in repr(nt.parse_selector("A; X", species=species, kind="gtype"))
    assert "LabPattern(infected)" in repr(nt.parse_selector("A|a; X|X@infected", species=species))


@pytest.mark.parametrize(
    "pattern",
    ["A; X", "A;*", "(A);X", "*", "!A;X", "{A,a};X", "A;X@cas9_deposited"],
)
def test_gamete_and_content_haploid_entries_agree(
    species: nt.Species, pattern: str
) -> None:
    """§5.2: both haploid entries describe the same content the same way.

    ``parse_haplotype_pattern`` differs from the content-only entry only by
    carrying the ``@glab`` suffix; the genetic part must select identically.
    """
    gamete = nt.parse_selector(pattern, species=species, kind="gtype")
    content, label = GenotypePatternParser.split_label_suffix(pattern.strip())
    plain = nt.parse_selector(content, species=species, kind="haploid")

    assert (gamete.glab is None) == (label is None)
    genomes = list(species.iter_haploid_genotypes())
    assert [g for g in genomes if gamete.genome.matches(g)] == [
        g for g in genomes if plain.matches(g)
    ]


def test_zygote_label_takes_part_in_matching(
    species: nt.Species, registry: nt.IndexRegistry
) -> None:
    """§5.6: GType/ZType labels match fully through the label-aware type."""
    genotype = species.get_genotype_from_str("A|a; X|X")
    infected = registry.ztype_index(genotype, "infected")

    assert registry.resolve_ztype_indices(
        nt.parse_selector("A|a; X|X@infected", species=species)
    ) == [infected]
    assert registry.resolve_ztype_indices(
        nt.parse_selector("A|a; X|X@{infected,cas9_high}", species=species)
    ) == [infected, registry.ztype_index(genotype, "cas9_high")]
    assert registry.resolve_ztype_indices(
        nt.parse_selector("A|a; X|X@!infected", species=species)
    ) == [
        registry.ztype_index(genotype, "default"),
        registry.ztype_index(genotype, "cas9_high"),
    ]
    # ``@*`` equals an omitted suffix: every slab of the named genotype.
    assert len(
        registry.resolve_ztype_indices(nt.parse_selector("A|a; X|X@*", species=species))
    ) == len(SOMATIC_LABELS)


@pytest.mark.parametrize("pattern", ["A@infected|a", "(A@infected|a)", "A;X@a@b"])
def test_a_label_in_a_nested_position_is_rejected(
    species: nt.Species, pattern: str
) -> None:
    """§5.4: a label the grammar does not accept as a suffix is not dropped."""
    with pytest.raises(PatternParseError):
        nt.parse_selector(pattern, species=species, kind="genotype")
