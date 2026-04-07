"""Unit tests for natal.genetic_structures."""

import pytest  # type: ignore

import natal as nt
from natal.genetic_structures import SexChromosomeType
from natal.population_config import extract_gamete_frequencies, initialize_gamete_map
from natal.type_def import Sex


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
        with pytest.raises(AssertionError):  # type: ignore
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

    def test_extended_format_supports_sex_type(self):
        sp = nt.Species.from_dict(
            name="S_with_sex_type",
            structure={
                "chrX": {
                    "sex_type": "X",
                    "loci": {"sx": ["X1"]},
                },
                "chrY": {
                    "sex_type": "Y",
                    "loci": {"sy": ["Y1"]},
                },
                "chrA": {
                    "loci": {"a": ["A"]},
                },
            },
        )

        assert sp.get_chromosome("chrX").sex_type == SexChromosomeType.X
        assert sp.get_chromosome("chrY").sex_type == SexChromosomeType.Y
        assert sp.sex_system == "XY"

    def test_extended_format_invalid_sex_type_raises(self):
        with pytest.raises(AssertionError):  # type: ignore
            nt.Species.from_dict(
                name="S_bad_sex_type",
                structure={
                    "chrX": {
                        "sex_type": 123,  # type: ignore[arg-type]
                        "loci": {"sx": ["X1"]},
                    }
                },
            )


class TestSexAwareGameteMap:
    def test_initialize_gamete_map_is_sex_aware_for_xy(self):
        species = nt.Species.from_dict(
            name="S_xy_gamete_map",
            structure={
                "chrA": {"loci": {"a": ["A"]}},
                "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
                "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
            },
        )

        haploid_genotypes = species.get_all_haploid_genotypes()
        diploid_genotypes = species.get_all_genotypes()
        gamete_map = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=1,
        )

        chrom_x = species.get_chromosome("chrX")
        chrom_y = species.get_chromosome("chrY")
        assert chrom_x is not None
        assert chrom_y is not None

        idx_xy = None
        for idx, gt in enumerate(diploid_genotypes):
            try:
                mat_x = gt.maternal.get_haplotype_for_chromosome(chrom_x)
                pat_y = gt.paternal.get_haplotype_for_chromosome(chrom_y)
            except ValueError:
                continue
            if mat_x is not None and pat_y is not None:
                idx_xy = idx
                break
        assert idx_xy is not None

        female_freqs = extract_gamete_frequencies(
            genotype_to_gametes_map=gamete_map,
            sex_idx=int(Sex.FEMALE),
            genotype_idx=idx_xy,
            haploid_genotypes=haploid_genotypes,
            n_glabs=1,
        )
        male_freqs = extract_gamete_frequencies(
            genotype_to_gametes_map=gamete_map,
            sex_idx=int(Sex.MALE),
            genotype_idx=idx_xy,
            haploid_genotypes=haploid_genotypes,
            n_glabs=1,
        )

        female_y_freq = 0.0
        male_y_freq = 0.0
        for hg, freq in female_freqs.items():
            try:
                hg.get_haplotype_for_chromosome(chrom_y)
            except ValueError:
                continue
            female_y_freq += freq
        for hg, freq in male_freqs.items():
            try:
                hg.get_haplotype_for_chromosome(chrom_y)
            except ValueError:
                continue
            male_y_freq += freq

        assert female_y_freq == pytest.approx(0.0)
        assert male_y_freq > 0.0
        assert sum(female_freqs.values()) == pytest.approx(1.0)
        assert sum(male_freqs.values()) == pytest.approx(1.0)

    def test_species_haploid_getters_support_parent_role(self):
        species = nt.Species.from_dict(
            name="S_xy_haploid_getters",
            structure={
                "chrA": {"loci": {"a": ["A"]}},
                "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
                "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
            },
        )

        all_hap = species.get_haploid_genotypes()
        mat_hap = species.get_haploid_genotypes("maternal")
        pat_hap = species.get_haploid_genotypes("paternal")

        assert all_hap == species.get_all_haploid_genotypes()
        assert mat_hap == species.get_maternal_haploid_genotypes()
        assert pat_hap == species.get_paternal_haploid_genotypes()

        chrom_y = species.get_chromosome("chrY")
        assert chrom_y is not None
        mat_has_y = any(
            hap.get_haplotype_for_chromosome(chrom_y) is not None
            for hap in mat_hap
            if any(h.chromosome is chrom_y for h in hap.haplotypes)
        )
        pat_has_y = any(
            hap.get_haplotype_for_chromosome(chrom_y) is not None
            for hap in pat_hap
            if any(h.chromosome is chrom_y for h in hap.haplotypes)
        )
        assert mat_has_y is False
        assert pat_has_y is True

    def test_species_haploid_getters_invalid_parent_role_raises(self):
        species = nt.Species.from_dict(
            name="S_xy_invalid_parent_role",
            structure={
                "chrA": {"loci": {"a": ["A"]}},
                "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
                "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
            },
        )

        with pytest.raises(ValueError):
            species.get_haploid_genotypes("unknown")  # type: ignore[arg-type]

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
