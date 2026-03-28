"""Unit tests for natal.genetic_structures."""

import pytest  # type: ignore
import natal as nt
from natal.genetic_structures import SexChromosomeType


class TestSexChromosomeType:
    def test_autosome_not_sex_chromosome(self):
        assert SexChromosomeType.AUTOSOME.is_sex_chromosome is False

    def test_x_is_sex_chromosome(self):
        assert SexChromosomeType.X.is_sex_chromosome is True

    def test_y_is_sex_chromosome(self):
        assert SexChromosomeType.Y.is_sex_chromosome is True

    def test_z_is_sex_chromosome(self):
        assert SexChromosomeType.Z.is_sex_chromosome is True

    def test_w_is_sex_chromosome(self):
        assert SexChromosomeType.W.is_sex_chromosome is True

    def test_autosome_sex_system_none(self):
        assert SexChromosomeType.AUTOSOME.sex_system is None

    def test_x_sex_system_xy(self):
        assert SexChromosomeType.X.sex_system == "XY"

    def test_y_sex_system_xy(self):
        assert SexChromosomeType.Y.sex_system == "XY"

    def test_z_sex_system_zw(self):
        assert SexChromosomeType.Z.sex_system == "ZW"

    def test_w_sex_system_zw(self):
        assert SexChromosomeType.W.sex_system == "ZW"

    def test_w_maternal_only(self):
        assert SexChromosomeType.W.maternal_only is True

    def test_y_paternal_only(self):
        assert SexChromosomeType.Y.paternal_only is True

    def test_x_not_maternal_only(self):
        assert SexChromosomeType.X.maternal_only is False

    def test_x_not_paternal_only(self):
        assert SexChromosomeType.X.paternal_only is False

    def test_autosome_not_maternal_only(self):
        assert SexChromosomeType.AUTOSOME.maternal_only is False

    def test_autosome_not_paternal_only(self):
        assert SexChromosomeType.AUTOSOME.paternal_only is False


class TestSpeciesFromDict:
    def test_simple_format_chromosomes(self):
        sp = nt.Species.from_dict(
            name="S_simple_chr",
            structure={"chr1": ["locA", "locB"], "chr2": ["locC"]},
        )
        chrom_names = [c.name for c in sp.chromosomes]
        assert "chr1" in chrom_names
        assert "chr2" in chrom_names

    def test_simple_format_loci_count(self):
        sp = nt.Species.from_dict(
            name="S_simple_loci",
            structure={"chr1": ["locA", "locB"]},
        )
        assert len(sp.chromosomes[0].loci) == 2

    def test_detailed_format_alleles(self):
        sp = nt.Species.from_dict(
            name="S_detail_alleles",
            structure={
                "chr1": {
                    "loc": ["WT", "Dr", "R2"],
                }
            },
            gamete_labels=["default"],
        )
        locus = sp.chromosomes[0].loci[0]
        allele_names = [a.name for a in locus.alleles]
        assert "WT" in allele_names
        assert "Dr" in allele_names
        assert "R2" in allele_names
        assert len(allele_names) == 3

    def test_gamete_labels_stored(self):
        sp = nt.Species.from_dict(
            name="S_glab",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default", "cas9"],
        )
        assert sp.gamete_labels == ["default", "cas9"]

    def test_no_gamete_labels_empty_list(self):
        sp = nt.Species.from_dict(
            name="S_no_glab",
            structure={"chr1": ["locA"]},
        )
        assert sp.gamete_labels == []

    def test_invalid_spec_type_raises(self):
        with pytest.raises(TypeError):
            nt.Species.from_dict(
                name="S_bad",
                structure={"chr1": 42},  # type: ignore[arg-type]
            )

    def test_chromosome_count(self):
        sp = nt.Species.from_dict(
            name="S_3chrom",
            structure={
                "chr1": ["locA"],
                "chr2": ["locB"],
                "chr3": ["locC"],
            },
        )
        assert len(sp.chromosomes) == 3

    def test_idempotent_same_name_returns_same_instance(self):
        """Calling from_dict twice with the same name must return the same object.

        Previously ChildStructureRegistry.add raised ValueError when an
        identically-named child was added a second time.  The API now silently
        returns the cached instance so repeated construction is safe.
        """
        sp1 = nt.Species.from_dict(
            name="S_idem",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        sp2 = nt.Species.from_dict(
            name="S_idem",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        assert sp1 is sp2

    def test_idempotent_chromosome_registry_unchanged(self):
        """Repeated from_dict calls must not duplicate chromosomes."""
        sp = nt.Species.from_dict(
            name="S_idem_chr",
            structure={"chr1": ["locA"]},
        )
        nt.Species.from_dict(
            name="S_idem_chr",
            structure={"chr1": ["locA"]},
        )
        assert len(sp.chromosomes) == 1


class TestSpeciesQueries:
    def test_get_locus_found(self, simple_species):
        locus = simple_species.get_locus("loc")
        assert locus is not None
        assert locus.name == "loc"

    def test_get_locus_not_found(self, simple_species):
        assert simple_species.get_locus("nonexistent") is None

    def test_get_chromosome_found(self, simple_species):
        chrom = simple_species.get_chromosome("chr1")
        assert chrom is not None
        assert chrom.name == "chr1"

    def test_get_chromosome_not_found(self, simple_species):
        assert simple_species.get_chromosome("chrX") is None

    def test_autosomes_list(self, simple_species):
        autosomes = simple_species.autosomes
        assert len(autosomes) == 1
        assert autosomes[0].name == "chr1"

    def test_sex_chromosomes_empty_for_autosome_only(self, simple_species):
        assert simple_species.sex_chromosomes == []


class TestLocus:
    def test_alleles_property(self, simple_species):
        locus = simple_species.get_locus("loc")
        names = [a.name for a in locus.alleles]
        assert names == ["WT", "Dr", "R2"]

    def test_locus_name(self, simple_species):
        locus = simple_species.get_locus("loc")
        assert locus.name == "loc"
