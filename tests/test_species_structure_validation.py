"""Species structure completeness validation (CR-9 regression).

Contract: every chromosome used for genetic computation carries at least
one locus, and every locus carries at least one allele — for autosomes
and sex chromosomes alike.  Construction and stepwise editing may be
temporarily incomplete, but the calculation entry points (genotype
enumeration, genotype string parsing, baseline acquisition) validate the
structure first and raise a ``ValueError`` naming the species,
chromosome, and locus.  Cache hits on the config blueprint must not
bypass the check.
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.genetics.entities.gene import Gene


def _single_locus_species(name: str) -> nt.Species:
    """Build a fresh single-locus species.

    Species instances are cached globally by name (a ``Species`` has no
    parent species), so every test must use a unique name to avoid
    inheriting mutations performed on an earlier test's instance.
    """
    return nt.Species.from_dict(
        name=name, structure={"chr1": {"loc1": ["A", "B"]}}
    )


def test_complete_species_validates_and_computes() -> None:
    """A complete species passes validate_structure and computes a baseline."""
    s = _single_locus_species("cr9_complete")
    s.validate_structure()  # no raise
    blueprint = s.get_config_blueprint()
    assert blueprint["n_ztypes"] == 3  # AA, AB, BB x 1 slab


def test_chromosome_without_loci_raises_on_all_entries() -> None:
    """An empty chromosome fails enumeration, parsing, and baseline."""
    s = _single_locus_species("cr9_empty_chrom")
    s.add_chromosome(nt.Chromosome(name="chr2"))

    with pytest.raises(ValueError, match="chr2.*no loci"):
        s.validate_structure()
    with pytest.raises(ValueError, match="chr2"):
        s.get_all_genotypes(unordered=True)
    with pytest.raises(ValueError, match="chr2"):
        s.get_all_haploid_genotypes()
    with pytest.raises(ValueError, match="chr2"):
        s.get_genotype_from_str("A|A")
    with pytest.raises(ValueError, match="chr2"):
        s.get_config_blueprint()


def test_locus_without_alleles_raises_with_location() -> None:
    """An allele-less locus fails validation naming species and locus."""
    s = _single_locus_species("cr9_empty_locus")
    s.get_chromosome("chr1").add_locus("loc2")

    with pytest.raises(ValueError, match="cr9_empty_locus.*chr1.*loc2.*no alleles"):
        s.validate_structure()


def test_stepwise_construction_allowed_until_computation() -> None:
    """Editing may be incomplete; completing the structure restores compute."""
    s = _single_locus_species("cr9_stepwise")
    s.add_chromosome(nt.Chromosome(name="chr2"))
    # No computation attempted yet: still constructible/editable.
    chr2 = s.get_chromosome("chr2")
    assert chr2 is not None
    chr2.add_locus("locX")
    locus = s.get_locus("locX")
    assert locus is not None
    locus.add_alleles([Gene(name="A", locus=locus), Gene(name="B", locus=locus)])
    s.get_config_blueprint()
    assert s.get_config_blueprint()["n_ztypes"] == 9  # 3x3 diploid x 1 slab


def test_blueprint_cache_hit_does_not_bypass_validation() -> None:
    """A cached baseline still validates on the next acquisition."""
    s = _single_locus_species("cr9_cache")
    s.get_config_blueprint()  # populates the cache from a complete structure
    s.get_chromosome("chr1").remove_locus("loc1")
    with pytest.raises(ValueError, match="no loci"):
        s.get_config_blueprint()


def test_sex_chromosomes_are_covered_too() -> None:
    """An X chromosome without loci fails validation like any autosome."""
    s = nt.Species.from_dict(
        name="cr9xy",
        structure={"auto": {"loc1": ["A", "B"]}, "X": {}, "Y": {}},
    )
    with pytest.raises(ValueError, match="cr9xy.*X.*no loci"):
        s.validate_structure()
