"""Frozen user-surface contract samples: species structure registration.

RUST_ONLY_REFACTOR_PLAN.md section 2.1 freezes the ``Species.from_dict``
declaration syntax: chromosomes / loci / alleles, sex chromosomes,
labels, and recombination rates must keep their current expression
forms.  These tests are executable samples of that syntax and pin the
resulting genotype space and recombination numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt

# ══════════════════════════════════════════════════════════════════════════════
# Structure dict forms
# ══════════════════════════════════════════════════════════════════════════════


def test_locus_name_list_form_builds_skeleton_loci() -> None:
    """``{"chr": ["locA"]}`` registers the locus with alleles deferred."""
    species = nt.Species.from_dict(name="FrozenSpListForm", structure={"chr1": ["locA"]})

    loci = species.get_chromosome("chr1").loci
    assert [locus.name for locus in loci] == ["locA"]
    # Alleles are inferred on first reference, so the skeleton is empty.
    assert [list(locus.alleles) for locus in loci] == [[]]
    assert list(species.get_all_genotypes()) == []


def test_locus_allele_map_form_expands_genotypes() -> None:
    """``{"chr": {"loc": ["WT", "Dr"]}}`` yields the unordered genotype set.

    The unordered fold collapses the two heterozygote orders into one
    entry; the ordered enumeration keeps both.
    """
    species = nt.Species.from_dict(
        name="FrozenSpMapForm",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )

    assert species.unordered
    folded = sorted(str(g) for g in species.iter_genotypes(unordered=True))
    assert folded == ["Dr|Dr", "WT|Dr", "WT|WT"]

    ordered = sorted(str(g) for g in species.get_all_genotypes())
    assert ordered == ["Dr|Dr", "WT|Dr", "WT|Dr", "WT|WT"]


def test_multi_chromosome_structure_joins_with_semicolons() -> None:
    """Genotype strings join per-chromosome segments with ``;``."""
    species = nt.Species.from_dict(
        name="FrozenSpMultiChr",
        structure={
            "chr1": {"a": ["A1", "A2"]},
            "chr2": {"b": ["B1"]},
        },
    )

    genotypes = {str(g) for g in species.get_all_genotypes()}
    # Three genotypes on chr1 crossed with the fixed B1|B1 on chr2.
    assert genotypes == {
        "A1|A1;B1|B1",
        "A1|A2;B1|B1",
        "A2|A2;B1|B1",
    }


@pytest.mark.parametrize("sex_type", ["X", "Y", "Z", "W"])
def test_extended_spec_sex_types_register(sex_type: str) -> None:
    """``{"sex_type": ..., "loci": {...}}`` marks the chromosome's role."""
    species = nt.Species.from_dict(
        name=f"FrozenSpSexType{sex_type}",
        structure={"chrS": {"sex_type": sex_type, "loci": {"s": ["S1"]}}},
    )

    chrom = species.get_chromosome("chrS")
    assert chrom.is_sex_chromosome
    assert not chrom.is_autosome
    assert str(chrom.sex_type).endswith(sex_type)


def test_xy_species_produce_sex_specific_genotypes() -> None:
    """An XY species exposes exactly the XX and XY diploid genotypes.

    With one locus on each sex chromosome the haploid set is ``{A1;X,
    A1;Y}``; the diploid set combines them into XX and XY.  The XY
    string compresses its empty per-parent sex segments to ``A1|A1``.
    """
    species = nt.Species.from_dict(
        name="FrozenSpXY",
        structure={
            "chrX": {"sex_type": "X", "loci": {"sx": ["X"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y"]}},
            "chrA": {"loci": {"a": ["A1"]}},
        },
        unordered=False,
    )

    genotypes = {str(g) for g in species.iter_genotypes(unordered=False)}
    assert genotypes == {"X|X;A1|A1", "A1|A1"}

    haploids = {str(h) for h in species.get_all_haploid_genotypes()}
    assert haploids == {"A1;X", "A1;Y"}


def test_empty_y_chromosome_is_a_valid_segment() -> None:
    """``{"sex_type": "Y", "loci": {}}`` builds without loci."""
    species = nt.Species.from_dict(
        name="FrozenSpEmptyY",
        structure={
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {}},
        },
    )

    assert species.get_chromosome("chrY").loci == []


# ══════════════════════════════════════════════════════════════════════════════
# Labels
# ══════════════════════════════════════════════════════════════════════════════


def test_gamete_and_somatic_labels_round_trip() -> None:
    """``gamete_labels`` / ``somatic_labels`` are stored verbatim."""
    species = nt.Species.from_dict(
        name="FrozenSpLabels",
        structure={"c1": {"l1": ["WT", "Dr"]}},
        gamete_labels=["default", "cas9_deposited"],
        somatic_labels=["normal", "infected"],
    )

    assert list(species.gamete_labels) == ["default", "cas9_deposited"]
    assert list(species.somatic_labels) == ["normal", "infected"]


def test_invalid_label_characters_raise_value_error() -> None:
    """Labels must match ``[A-Za-z0-9_]+``; violations raise ValueError."""
    with pytest.raises(ValueError, match=r"Labels must match \[A-Za-z0-9_\]\+"):
        nt.Species.from_dict(
            name="FrozenSpBadLabel",
            structure={"c1": {"l1": ["A"]}},
            gamete_labels=["bad label"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Recombination
# ══════════════════════════════════════════════════════════════════════════════


def test_set_recombination_records_adjacent_rate() -> None:
    """``chrom.set_recombination(a, b, r)`` stores the pairwise rate."""
    species = nt.Species.from_dict(
        name="FrozenSpRecomb",
        structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
    )

    chrom = species.get_chromosome("chr1")
    chrom.set_recombination("locA", "locB", 0.3)

    rates = np.asarray(chrom.recombination_matrix)
    np.testing.assert_allclose(rates, [0.3])


# ══════════════════════════════════════════════════════════════════════════════
# Genotype string syntax and identity
# ══════════════════════════════════════════════════════════════════════════════


def test_genotype_string_syntax_parses_and_round_trips() -> None:
    """``/`` separates genes, ``|`` homologs, ``;`` chromosomes."""
    species = nt.Species.from_dict(
        name="FrozenSpParse",
        structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
    )

    parsed = species.get_genotype_from_str("A1/B1|A2/B2")
    assert str(parsed) == "A1/B1|A2/B2"

    with pytest.raises(ValueError, match="Use '/' to separate"):
        species.get_genotype_from_str("A1B1|A2B2")


def test_same_name_from_dict_returns_cached_instance() -> None:
    """``Species.from_dict`` is singleton-scoped by name."""
    structure = {"chr1": {"loc": ["WT", "Dr"]}}
    first = nt.Species.from_dict(name="FrozenSpIdempotent", structure=structure)
    second = nt.Species.from_dict(name="FrozenSpIdempotent", structure=structure)

    assert first is second
