"""A content-only pattern entry must reject an ``@label`` suffix.

FRONTEND_REFACTOR_PLAN.md §5.2: "纯 Genotype/HaploidGenome 输入遇到 ``@label``
明确报错，不接受后再忽略".  The suffix used to be parsed into the pattern
object and then never consulted by ``matches()``, so a labelled query silently
matched every label of the genotypes it named.

The label-aware entries keep taking labels: ``ZygoteTypePattern.parse`` (which
composes a genotype pattern with a slab), ``IndividualSelector(ztype=...)``,
the conversion-rule filters, and
``GenotypePatternParser.parse_haplotype_pattern`` (a ``GameteTypePattern``).
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.patterns import PatternParseError, ZygoteTypePattern
from natal.frontend.patterns.parser import GenotypePatternParser


@pytest.fixture(scope="module")
def species() -> nt.Species:
    return nt.Species.from_dict(
        "unlabelled_pattern_entries",
        {"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default", "cas9"],
        somatic_labels=["default", "infected"],
    )


@pytest.mark.parametrize("pattern", ["WT|Dr@cas9", "WT|Dr@*", "WT|Dr@infected"])
def test_genotype_pattern_entry_rejects_a_label(species: nt.Species, pattern: str) -> None:
    with pytest.raises(PatternParseError, match="does not take an '@label' suffix"):
        species.parse_genotype_pattern(pattern)


@pytest.mark.parametrize("pattern", ["WT@cas9", "WT@*"])
def test_haploid_genome_pattern_entry_rejects_a_label(
    species: nt.Species, pattern: str
) -> None:
    with pytest.raises(PatternParseError, match="does not take an '@label' suffix"):
        species.parse_haploid_genome_pattern(pattern)


def test_filter_helper_rejects_a_label_too(species: nt.Species) -> None:
    """The collection filters parse through the same entry, so they inherit it."""
    genotypes = species.get_all_genotypes()
    with pytest.raises(PatternParseError, match="does not take an '@label' suffix"):
        species.filter_genotypes_by_pattern(genotypes, "WT|Dr@infected")


@pytest.mark.parametrize("pattern", ["WT|Dr", "WT::Dr", "*", "WT|*", "{WT,Dr}|WT"])
def test_unlabelled_genotype_forms_still_parse(species: nt.Species, pattern: str) -> None:
    assert callable(species.parse_genotype_pattern(pattern))


@pytest.mark.parametrize("pattern", ["WT", "*", "{WT,Dr}", "!Dr"])
def test_unlabelled_haploid_forms_still_parse(species: nt.Species, pattern: str) -> None:
    assert callable(species.parse_haploid_genome_pattern(pattern))


@pytest.mark.parametrize(
    "pattern,message", [("WT|Dr@", "Empty @lab suffix"), ("WT|Dr@a@b", "Only one @lab suffix")]
)
def test_malformed_suffix_keeps_its_own_message(
    species: nt.Species, pattern: str, message: str
) -> None:
    """The parser's own ``@`` diagnostics are unchanged, not replaced."""
    with pytest.raises(PatternParseError, match=message):
        species.parse_genotype_pattern(pattern)


def test_label_aware_entries_still_accept_labels(species: nt.Species) -> None:
    het = species.get_genotype_from_str("WT|Dr")
    labelled = ZygoteTypePattern.parse("WT|Dr@infected", species)
    assert labelled.slab is not None
    assert labelled.matches(het, "infected")
    assert not labelled.matches(het, "default")
    assert nt.IndividualSelector(ztype="WT|Dr@infected") is not None
    assert GenotypePatternParser(species).parse_haplotype_pattern("WT@cas9") is not None


def test_selector_resolution_rejects_a_labelled_mixed_input(species: nt.Species) -> None:
    """A label in a genotype-level selector fails instead of being dropped.

    This is what a tuple patch key such as ``("WT|Dr@infected", "Dr|Dr")`` hits:
    previously the label vanished and every slab of ``WT|Dr`` was written.
    """
    with pytest.raises(ValueError, match="does not take an '@label' suffix"):
        species.resolve_genotype_selectors(
            selector="WT|Dr@infected",
            all_genotypes=species.get_all_genotypes(),
            context="probe",
        )
