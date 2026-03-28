"""Unit tests for natal.genetic_entities."""

import pytest
from natal.genetic_entities import Gene, Haplotype, HaploidGenotype, Genotype
import natal as nt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_species(name: str):
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        gamete_labels=["default"],
    )


def _make_entities(sp):
    """Return (locus, chrom, wt_gene, dr_gene, r2_gene, wt_haplo, dr_haplo,
    wt_hg, dr_hg, wt_wt, wt_dr, dr_wt, dr_dr) for a simple species."""
    locus = sp.chromosomes[0].loci[0]
    chrom = sp.chromosomes[0]
    wt = Gene("WT", locus=locus)
    dr = Gene("Dr", locus=locus)
    r2 = Gene("R2", locus=locus)
    wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
    dr_haplo = Haplotype(chromosome=chrom, genes=[dr])
    r2_haplo = Haplotype(chromosome=chrom, genes=[r2])
    wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
    dr_hg = HaploidGenotype(species=sp, haplotypes=[dr_haplo])
    r2_hg = HaploidGenotype(species=sp, haplotypes=[r2_haplo])
    wt_wt = Genotype(species=sp, maternal=wt_hg, paternal=wt_hg)
    wt_dr = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
    dr_wt = Genotype(species=sp, maternal=dr_hg, paternal=wt_hg)
    dr_dr = Genotype(species=sp, maternal=dr_hg, paternal=dr_hg)
    return locus, chrom, wt, dr, r2, wt_haplo, dr_haplo, r2_haplo, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr


class TestGene:
    def test_creation(self):
        sp = _make_species("GeneTest_creation")
        locus = sp.chromosomes[0].loci[0]
        gene = Gene("WT", locus=locus)
        assert gene.name == "WT"

    def test_singleton_per_locus(self):
        """Same name + same locus returns the identical object."""
        sp = _make_species("GeneTest_singleton")
        locus = sp.chromosomes[0].loci[0]
        g1 = Gene("WT", locus=locus)
        g2 = Gene("WT", locus=locus)
        assert g1 is g2

    def test_different_names_are_distinct(self):
        sp = _make_species("GeneTest_distinct")
        locus = sp.chromosomes[0].loci[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        assert wt is not dr

    def test_locus_reference(self):
        sp = _make_species("GeneTest_locus_ref")
        locus = sp.chromosomes[0].loci[0]
        gene = Gene("WT", locus=locus)
        assert gene.locus is locus

    def test_no_locus_raises(self):
        with pytest.raises(TypeError):
            Gene("WT")  # type: ignore[call-arg]


class TestHaplotype:
    def test_creation(self):
        sp = _make_species("HapTest_creation")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        haplo = Haplotype(chromosome=chrom, genes=[wt])
        assert haplo.genes == [wt]

    def test_singleton_same_genes(self):
        sp = _make_species("HapTest_singleton")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        h1 = Haplotype(chromosome=chrom, genes=[wt])
        h2 = Haplotype(chromosome=chrom, genes=[wt])
        assert h1 is h2

    def test_different_genes_are_distinct(self):
        sp = _make_species("HapTest_distinct")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        h_wt = Haplotype(chromosome=chrom, genes=[wt])
        h_dr = Haplotype(chromosome=chrom, genes=[dr])
        assert h_wt is not h_dr

    def test_chromosome_reference(self):
        sp = _make_species("HapTest_chrom_ref")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        haplo = Haplotype(chromosome=chrom, genes=[wt])
        assert haplo.chromosome is chrom

    def test_incomplete_locus_coverage_raises(self):
        """A haplotype must cover all loci on the chromosome."""
        sp = nt.Species.from_dict(
            name="HapTest_incomplete",
            structure={"chr1": {"locA": ["A1"], "locB": ["B1"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = sp.get_locus("locA")
        a1 = Gene("A1", locus=loc_a)
        with pytest.raises(ValueError, match="Incomplete haplotype"):
            Haplotype(chromosome=chrom, genes=[a1])


class TestHaploidGenotype:
    def test_creation(self):
        sp = _make_species("HG_creation")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        haplo = Haplotype(chromosome=chrom, genes=[wt])
        hg = HaploidGenotype(species=sp, haplotypes=[haplo])
        assert hg.species is sp

    def test_singleton_same_haplotypes(self):
        sp = _make_species("HG_singleton")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        haplo = Haplotype(chromosome=chrom, genes=[wt])
        hg1 = HaploidGenotype(species=sp, haplotypes=[haplo])
        hg2 = HaploidGenotype(species=sp, haplotypes=[haplo])
        assert hg1 is hg2

    def test_str_representation(self):
        sp = _make_species("HG_str")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        haplo = Haplotype(chromosome=chrom, genes=[wt])
        hg = HaploidGenotype(species=sp, haplotypes=[haplo])
        assert str(hg) == "WT"

    def test_missing_chromosome_raises(self):
        sp = nt.Species.from_dict(
            name="HG_missing_chr",
            structure={"chr1": {"loc": ["WT"]}, "chr2": {"loc2": ["X"]}},
        )
        chrom1 = sp.get_chromosome("chr1")
        loc1 = sp.get_locus("loc")
        wt = Gene("WT", locus=loc1)
        haplo1 = Haplotype(chromosome=chrom1, genes=[wt])
        with pytest.raises(ValueError, match="Incomplete haploid genotype"):
            HaploidGenotype(species=sp, haplotypes=[haplo1])


class TestGenotype:
    def test_string_homozygous(self):
        sp = _make_species("GT_str_homo")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert str(wt_wt) == "WT|WT"

    def test_string_heterozygous(self):
        sp = _make_species("GT_str_het")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert str(wt_dr) == "WT|Dr"

    def test_maternal_paternal_order_preserved(self):
        sp = _make_species("GT_order")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert str(wt_dr) != str(dr_wt)

    def test_singleton(self):
        sp = _make_species("GT_singleton")
        locus = sp.chromosomes[0].loci[0]
        chrom = sp.chromosomes[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
        dr_haplo = Haplotype(chromosome=chrom, genes=[dr])
        wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
        dr_hg = HaploidGenotype(species=sp, haplotypes=[dr_haplo])
        gt1 = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
        gt2 = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
        assert gt1 is gt2

    def test_maternal_attribute(self):
        sp = _make_species("GT_maternal")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert wt_dr.maternal is wt_hg

    def test_paternal_attribute(self):
        sp = _make_species("GT_paternal")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert wt_dr.paternal is dr_hg
