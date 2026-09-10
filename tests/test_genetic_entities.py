"""Unit tests for natal.genetic_entities."""

import inspect
from math import comb

import numpy as np
import pytest  # type: ignore

import natal as nt
from natal.frontend.genetics import initialize_gamete_map
from natal.frontend.genetics import (
    Gene,
    Genotype,
    HaploidGenotype,
    Haplotype,
    compute_recombinant_haplotypes,
    compute_recombinant_haplotypes_with_alleles,
)
from natal.frontend.genetics.entities.genotype import _compute_both_homolog_patterns

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
        with pytest.raises(TypeError, match="must be bound to a Locus"):
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
        assert loc_a is not None
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
        assert chrom1 is not None
        loc1 = sp.get_locus("loc")
        assert loc1 is not None
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

    def test_maternal_paternal_order_normalized(self):
        """A|a and a|A normalize to the same canonical form."""
        sp = _make_species("GT_order")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        assert str(wt_dr) == str(dr_wt)
        assert wt_dr is dr_wt

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


class TestGenotypeAlleleQueries:
    """Tests for Genotype allele query methods."""

    def test_get_alleles_at_locus_homozygous(self):
        sp = _make_species("AlleleQ_homo")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        mat_allele, pat_allele = wt_wt.get_alleles_at_locus(locus)
        assert mat_allele is wt_hg.haplotypes[0].genes[0]
        assert pat_allele is wt_hg.haplotypes[0].genes[0]

    def test_get_alleles_at_locus_heterozygous(self):
        sp = _make_species("AlleleQ_het")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        mat_allele, pat_allele = wt_dr.get_alleles_at_locus(locus)
        assert mat_allele is wt_hg.haplotypes[0].genes[0]
        assert pat_allele is dr_hg.haplotypes[0].genes[0]

    def test_is_homozygous_at_true(self):
        sp = _make_species("AlleleQ_hom_true")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        assert wt_wt.is_homozygous_at(locus) is True

    def test_is_homozygous_at_false(self):
        sp = _make_species("AlleleQ_hom_false")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        assert wt_dr.is_homozygous_at(locus) is False

    def test_is_heterozygous_at_true(self):
        sp = _make_species("AlleleQ_het_true")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        assert wt_dr.is_heterozygous_at(locus) is True

    def test_is_heterozygous_at_false(self):
        sp = _make_species("AlleleQ_het_false")
        *_, wt_hg, dr_hg, r2_hg, wt_wt, wt_dr, dr_wt, dr_dr = _make_entities(sp)
        locus = sp.chromosomes[0].loci[0]
        assert wt_wt.is_heterozygous_at(locus) is False


class TestGenotypeProduceGametes:
    """Tests for Genotype.produce_gametes()."""

    def test_homozygous_single_locus(self):
        """WT|WT returns a single gamete with frequency 1.0."""
        sp = nt.Species.from_dict(
            name="PG_hom_single",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        chrom = sp.chromosomes[0]
        locus = chrom.loci[0]
        wt = Gene("WT", locus=locus)
        _ = Gene("Dr", locus=locus)
        wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
        wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
        gt = Genotype(species=sp, maternal=wt_hg, paternal=wt_hg)
        gametes = gt.produce_gametes()
        assert len(gametes) == 1
        hg = list(gametes.keys())[0]
        assert hg is wt_hg
        assert gametes[hg] == 1.0

    def test_heterozygous_single_locus(self):
        """WT|Dr returns two gametes each at 0.5 (Mendelian segregation)."""
        sp = nt.Species.from_dict(
            name="PG_het_single",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        chrom = sp.chromosomes[0]
        locus = chrom.loci[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
        dr_haplo = Haplotype(chromosome=chrom, genes=[dr])
        wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
        dr_hg = HaploidGenotype(species=sp, haplotypes=[dr_haplo])
        gt = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
        gametes = gt.produce_gametes()
        assert len(gametes) == 2
        assert gametes[wt_hg] == 0.5
        assert gametes[dr_hg] == 0.5

    def test_two_loci_no_recombination(self):
        """Two heterozygous loci with no recombination → two parental types at 0.5 each."""
        sp = nt.Species.from_dict(
            name="PG_two_no_recomb",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()
        assert len(gametes) == 2
        assert gametes[mat_hg] == 0.5
        assert gametes[pat_hg] == 0.5

    def test_two_loci_with_recombination(self):
        """Recombination rate=0.1 → four gametes: parental (1-r)/2, recombinant r/2."""
        sp = nt.Species.from_dict(
            name="PG_two_recomb",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        # Set recombination rate BEFORE creating the genotype
        chrom.set_recombination(loc_a, loc_b, 0.1)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        rec_1_hap = Haplotype(chromosome=chrom, genes=[a1, b2])
        rec_2_hap = Haplotype(chromosome=chrom, genes=[a2, b1])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        rec_1 = HaploidGenotype(species=sp, haplotypes=[rec_1_hap])
        rec_2 = HaploidGenotype(species=sp, haplotypes=[rec_2_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()

        # Issue #43 regression: gametes starting on either homolog must be
        # enumerated — parental haplotypes at (1-r)/2, recombinants at r/2.
        expected = {mat_hg: 0.45, pat_hg: 0.45, rec_1: 0.05, rec_2: 0.05}
        assert set(gametes.keys()) == set(expected.keys())
        for gamete, freq in expected.items():
            assert gametes[gamete] == pytest.approx(freq)
        assert sum(gametes.values()) == pytest.approx(1.0)

    def test_small_rate_regression_issue_43(self):
        """Recombination rate=0.01 → 0.495/0.495/0.005/0.005 (issue #43 values)."""
        sp = nt.Species.from_dict(
            name="PG_recomb_small_rate",
            structure={"chr1": {"DriveSite": ["WT_D", "Drive"], "RescueSite": ["WT_R", "Rescue"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        wt_d = Gene("WT_D", locus=loc_a)
        drive = Gene("Drive", locus=loc_a)
        wt_r = Gene("WT_R", locus=loc_b)
        rescue = Gene("Rescue", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.01)
        mat_hg = HaploidGenotype(
            species=sp, haplotypes=[Haplotype(chromosome=chrom, genes=[wt_d, wt_r])]
        )
        pat_hg = HaploidGenotype(
            species=sp, haplotypes=[Haplotype(chromosome=chrom, genes=[drive, rescue])]
        )
        rec_1 = HaploidGenotype(
            species=sp, haplotypes=[Haplotype(chromosome=chrom, genes=[wt_d, rescue])]
        )
        rec_2 = HaploidGenotype(
            species=sp, haplotypes=[Haplotype(chromosome=chrom, genes=[drive, wt_r])]
        )
        genotype = sp.get_genotype_from_str("Drive/Rescue|WT_D/WT_R")
        gametes = genotype.produce_gametes()

        expected = {mat_hg: 0.495, pat_hg: 0.495, rec_1: 0.005, rec_2: 0.005}
        assert set(gametes.keys()) == set(expected.keys())
        for gamete, freq in expected.items():
            assert gametes[gamete] == pytest.approx(freq)

    def test_two_loci_half_recombination_equal_quarters(self):
        """Recombination rate=0.5 → all four gametes at 0.25 each (free recombination)."""
        sp = nt.Species.from_dict(
            name="PG_recomb_half",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.5)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        rec_1_hap = Haplotype(chromosome=chrom, genes=[a1, b2])
        rec_2_hap = Haplotype(chromosome=chrom, genes=[a2, b1])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        rec_1 = HaploidGenotype(species=sp, haplotypes=[rec_1_hap])
        rec_2 = HaploidGenotype(species=sp, haplotypes=[rec_2_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()

        assert len(gametes) == 4
        assert gametes[mat_hg] == pytest.approx(0.25)
        assert gametes[pat_hg] == pytest.approx(0.25)
        assert gametes[rec_1] == pytest.approx(0.25)
        assert gametes[rec_2] == pytest.approx(0.25)

    def test_first_locus_marginal_is_mendelian(self):
        """First-locus marginal transmission stays 0.5 for any recombination rate."""
        for rate in (0.01, 0.1, 0.25, 0.5):
            sp = nt.Species.from_dict(
                name=f"PG_marginal_{rate}",
                structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
            )
            chrom = sp.chromosomes[0]
            loc_a = chrom.loci[0]
            loc_b = chrom.loci[1]
            a1 = Gene("A1", locus=loc_a)
            a2 = Gene("A2", locus=loc_a)
            b1 = Gene("B1", locus=loc_b)
            b2 = Gene("B2", locus=loc_b)
            chrom.set_recombination(loc_a, loc_b, rate)
            mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
            pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
            mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
            pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
            gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
            gametes = gt.produce_gametes()

            maternal_mass = sum(
                freq
                for gamete, freq in gametes.items()
                if gamete.get_gene_at_locus(loc_a) is a1
            )
            assert maternal_mass == pytest.approx(0.5), f"rate={rate}"

    def test_maternal_paternal_swap_symmetry(self):
        """Swapping maternal/paternal haplotypes permutes gamete keys only."""
        sp = nt.Species.from_dict(
            name="PG_swap_symmetry",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.1)
        hap_11 = Haplotype(chromosome=chrom, genes=[a1, b1])
        hap_22 = Haplotype(chromosome=chrom, genes=[a2, b2])
        hap_12 = Haplotype(chromosome=chrom, genes=[a1, b2])
        hap_21 = Haplotype(chromosome=chrom, genes=[a2, b1])
        hg_11 = HaploidGenotype(species=sp, haplotypes=[hap_11])
        hg_22 = HaploidGenotype(species=sp, haplotypes=[hap_22])
        hg_12 = HaploidGenotype(species=sp, haplotypes=[hap_12])
        hg_21 = HaploidGenotype(species=sp, haplotypes=[hap_21])

        gt = Genotype(species=sp, maternal=hg_11, paternal=hg_22)
        gt_swapped = Genotype(species=sp, maternal=hg_22, paternal=hg_11)
        gametes = gt.produce_gametes()
        gametes_swapped = gt_swapped.produce_gametes()

        assert set(gametes.keys()) == {hg_11, hg_22, hg_12, hg_21}
        assert set(gametes_swapped.keys()) == {hg_11, hg_22, hg_12, hg_21}
        # Recombination treats both homologs symmetrically: swapping which
        # homolog is "maternal" permutes the keys but not the frequencies.
        assert gametes[hg_11] == pytest.approx(gametes_swapped[hg_22])
        assert gametes[hg_22] == pytest.approx(gametes_swapped[hg_11])
        assert gametes[hg_12] == pytest.approx(gametes_swapped[hg_21])
        assert gametes[hg_21] == pytest.approx(gametes_swapped[hg_12])

    def test_duplicate_haplotypes_accumulated(self):
        """Homozygous first locus: both starts yield identical haplotypes; mass accumulates."""
        sp = nt.Species.from_dict(
            name="PG_dup_accumulate",
            structure={"chr1": {"locA": ["A1"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.1)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a1, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()

        # Homozygous first locus makes recombination invisible: the only
        # inheritance signal is the Mendelian 0.5/0.5 at the second locus.
        # Overwrite-style bookkeeping would leave sum(gametes.values()) == 0.5.
        assert len(gametes) == 2
        assert gametes[mat_hg] == pytest.approx(0.5)
        assert gametes[pat_hg] == pytest.approx(0.5)
        assert sum(gametes.values()) == pytest.approx(1.0)

    def test_gamete_cache(self):
        """Caching: same object on second call, new object after cache clear."""
        sp = nt.Species.from_dict(
            name="PG_cache",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        chrom = sp.chromosomes[0]
        locus = chrom.loci[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
        dr_haplo = Haplotype(chromosome=chrom, genes=[dr])
        wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
        dr_hg = HaploidGenotype(species=sp, haplotypes=[dr_haplo])
        gt = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
        first = gt.produce_gametes()
        second = gt.produce_gametes()
        assert first is second
        gt._gamete_cache = None
        third = gt.produce_gametes()
        assert first is not third

    def test_frequencies_sum_to_one(self):
        """All gamete frequency dicts sum to 1.0."""
        sp = nt.Species.from_dict(
            name="PG_sum_to_one",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        chrom = sp.chromosomes[0]
        locus = chrom.loci[0]
        wt = Gene("WT", locus=locus)
        dr = Gene("Dr", locus=locus)
        wt_haplo = Haplotype(chromosome=chrom, genes=[wt])
        dr_haplo = Haplotype(chromosome=chrom, genes=[dr])
        wt_hg = HaploidGenotype(species=sp, haplotypes=[wt_haplo])
        dr_hg = HaploidGenotype(species=sp, haplotypes=[dr_haplo])
        gt = Genotype(species=sp, maternal=wt_hg, paternal=dr_hg)
        gametes = gt.produce_gametes()
        assert sum(gametes.values()) == pytest.approx(1.0)

    def test_homozygous_multi_locus(self):
        """Three loci all homozygous → single gamete at frequency 1.0."""
        sp = nt.Species.from_dict(
            name="PG_hom_multi",
            structure={"chr1": {"locA": ["A1"], "locB": ["B1"], "locC": ["C1"]}},
        )
        chrom = sp.chromosomes[0]
        genes = [Gene(_loc.name, locus=_loc) for _loc in chrom.loci]
        hap = Haplotype(chromosome=chrom, genes=genes)
        hg = HaploidGenotype(species=sp, haplotypes=[hap])
        gt = Genotype(species=sp, maternal=hg, paternal=hg)
        gametes = gt.produce_gametes()
        assert len(gametes) == 1
        assert list(gametes.values())[0] == pytest.approx(1.0)

    def test_zero_rate_takes_mendelian_shortcut(self):
        """Error/edge path: explicitly zero rates take the 0.5/0.5 shortcut.

        Exactly the two parental gametes are emitted — no zero-frequency
        recombinant rows — and the split is exactly Mendelian.
        """
        sp = nt.Species.from_dict(
            name="PG_zero_rate_shortcut",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.0)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()

        assert set(gametes.keys()) == {mat_hg, pat_hg}
        assert gametes[mat_hg] == 0.5
        assert gametes[pat_hg] == 0.5
        assert sum(gametes.values()) == pytest.approx(1.0)

    def test_returned_mapping_is_cache_and_probability_complete(self):
        """Ownership: produce_gametes() returns the live cache dict itself
        (documented design, no copy), and the sum-to-1 invariant holds on
        that exact object, so every caller sees probability-complete data."""
        sp = nt.Species.from_dict(
            name="PG_ownership_cache",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.1)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)

        gametes = gt.produce_gametes()
        # The returned mapping IS the documented internal cache object.
        assert gt._gamete_cache is gametes
        # Probability completeness holds on that exact object.
        assert sum(gametes.values()) == pytest.approx(1.0)
        assert gt.produce_gametes() is gametes

    def test_rate_change_requires_manual_cache_clear(self):
        """State transition: recombination rates set AFTER the first
        produce_gametes() call are ignored until ``_gamete_cache = None``
        (documented Note), after which the new rate is honored exactly."""
        sp = nt.Species.from_dict(
            name="PG_cache_staleness",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.1)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        rec_1_hap = Haplotype(chromosome=chrom, genes=[a1, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        rec_1 = HaploidGenotype(species=sp, haplotypes=[rec_1_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)

        # Step 1: cache computed at r=0.1 → recombinant at r/2 = 0.05.
        first = gt.produce_gametes()
        assert first[rec_1] == pytest.approx(0.05)

        # Step 2: mutate the rate; documented behavior returns the stale cache.
        chrom.set_recombination(loc_a, loc_b, 0.5)
        stale = gt.produce_gametes()
        assert stale is first
        assert stale[rec_1] == pytest.approx(0.05)

        # Step 3: manual clear → fresh computation reflects r=0.5 (quarter split).
        gt._gamete_cache = None
        fresh = gt.produce_gametes()
        assert fresh is not stale
        assert fresh[rec_1] == pytest.approx(0.25)
        assert sum(fresh.values()) == pytest.approx(1.0)


class TestComputeRecombinantHaplotypes:
    """Tests for the pure function compute_recombinant_haplotypes()."""

    def test_single_locus(self):
        """n_loci=1 → one pattern [[0]], one frequency [1.0]."""
        patterns, freqs = compute_recombinant_haplotypes(1, np.array([]))
        np.testing.assert_array_equal(patterns, [[0]])
        np.testing.assert_array_equal(freqs, [1.0])

    def test_two_loci_no_recombination(self):
        """rates=[0.0] → patterns [[0,0],[0,1]], freqs [1.0, 0.0]."""
        patterns, freqs = compute_recombinant_haplotypes(2, np.array([0.0]))
        np.testing.assert_array_equal(patterns, [[0, 0], [0, 1]])
        np.testing.assert_array_equal(freqs, [1.0, 0.0])

    def test_two_loci_full_recombination(self):
        """rates=[0.5] → patterns [[0,0],[0,1]], freqs [0.5, 0.5]."""
        patterns, freqs = compute_recombinant_haplotypes(2, np.array([0.5]))
        np.testing.assert_array_equal(patterns, [[0, 0], [0, 1]])
        np.testing.assert_allclose(freqs, [0.5, 0.5])

    def test_three_loci_known_rates(self):
        """rates=[0.1, 0.2] → 4 patterns, verify frequencies."""
        patterns, freqs = compute_recombinant_haplotypes(
            3, np.array([0.1, 0.2])
        )
        assert patterns.shape == (4, 3)
        # All patterns start with 0 (maternal)
        for row in patterns:
            assert row[0] == 0
        # Verify frequencies: product of (1-r) and r for each boundary
        expected = [
            (1 - 0.1) * (1 - 0.2),  # [0,0,0]: no crossover
            0.1 * (1 - 0.2),          # [0,1,1]: crossover at boundary 0
            (1 - 0.1) * 0.2,          # [0,0,1]: crossover at boundary 1
            0.1 * 0.2,                # [0,1,0]: crossover at both boundaries
        ]
        np.testing.assert_allclose(freqs, expected)

    def test_all_frequencies_sum_to_one(self):
        """For any input, frequencies sum to 1.0."""
        patterns, freqs = compute_recombinant_haplotypes(
            3, np.array([0.1, 0.2])
        )
        assert freqs.sum() == pytest.approx(1.0)
        patterns, freqs = compute_recombinant_haplotypes(
            4, np.array([0.05, 0.15, 0.25])
        )
        assert freqs.sum() == pytest.approx(1.0)

    def test_n_loci_zero_raises(self):
        """n_loci=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_loci must be >= 1"):
            compute_recombinant_haplotypes(0, np.array([]))

    def test_start_maternal_false(self):
        """start_maternal=False → patterns start with 1 (paternal)."""
        patterns, freqs = compute_recombinant_haplotypes(
            2, np.array([0.0]), start_maternal=False
        )
        np.testing.assert_array_equal(patterns, [[1, 1], [1, 0]])
        np.testing.assert_array_equal(freqs, [1.0, 0.0])
        patterns2, freqs2 = compute_recombinant_haplotypes(
            3, np.array([0.1, 0.2]), start_maternal=False
        )
        for row in patterns2:
            assert row[0] == 1  # First locus always paternal

    def test_property_based(self):
        """Multiple random rate sets: pattern count = 2^(n_loci-1), sum(freqs)=1."""
        rng = np.random.default_rng(42)
        for n_loci in [2, 3, 4, 5]:
            rates = rng.uniform(0, 0.5, n_loci - 1)
            patterns, freqs = compute_recombinant_haplotypes(n_loci, rates)
            expected_count = 2 ** (n_loci - 1)
            assert len(patterns) == expected_count
            assert freqs.sum() == pytest.approx(1.0)


class TestComputeRecombinantHaplotypesWithAlleles:
    """Tests for compute_recombinant_haplotypes_with_alleles()."""

    def test_both_homolog_starts_enumerated(self):
        """rates=[0.1] → four haplotype strings at 0.45/0.45/0.05/0.05."""
        result = compute_recombinant_haplotypes_with_alleles(
            ["A1", "B1"], ["A2", "B2"], np.array([0.1])
        )
        assert result == pytest.approx(
            {"A1/B1": 0.45, "A2/B2": 0.45, "A1/B2": 0.05, "A2/B1": 0.05}
        )

    def test_duplicate_alleles_accumulated(self):
        """Homozygous first locus: identical strings from both starts accumulate to Mendelian 0.5/0.5."""
        result = compute_recombinant_haplotypes_with_alleles(
            ["A1", "B1"], ["A1", "B2"], np.array([0.1])
        )
        assert result == pytest.approx({"A1/B1": 0.5, "A1/B2": 0.5})

    def test_mismatched_lengths_raise(self):
        """Allele lists of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_recombinant_haplotypes_with_alleles(
                ["A1"], ["A2", "B2"], np.array([0.1])
            )

    def test_empty_allele_lists_raise(self):
        """Error path: zero loci (both lists empty) still raises ValueError
        from the pure function's ``n_loci must be >= 1`` guard."""
        with pytest.raises(ValueError, match="n_loci must be >= 1"):
            compute_recombinant_haplotypes_with_alleles([], [], np.array([]))

    def test_returns_fresh_dict_per_call(self):
        """Ownership: the pure API has no shared mutable cache — each call
        returns an equal but distinct dict, so caller mutation cannot leak."""
        r1 = compute_recombinant_haplotypes_with_alleles(
            ["A1", "B1"], ["A2", "B2"], np.array([0.1])
        )
        r2 = compute_recombinant_haplotypes_with_alleles(
            ["A1", "B1"], ["A2", "B2"], np.array([0.1])
        )
        assert r1 is not r2
        assert r1 == pytest.approx(r2)


class TestRecombinationStartMaternalContract:
    """Issue #43 negative contract: ``start_maternal`` is removed from the
    public with-alleles API (both-homolog enumeration is unconditional) but
    must remain preserved on the pure function."""

    def test_with_alleles_rejects_start_maternal_keyword(self):
        """Passing ``start_maternal`` as a keyword raises TypeError and the
        parameter is absent from the signature (no silent re-acceptance)."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            compute_recombinant_haplotypes_with_alleles(
                ["A1", "B1"], ["A2", "B2"], np.array([0.1]), start_maternal=True
            )
        sig = inspect.signature(compute_recombinant_haplotypes_with_alleles)
        assert "start_maternal" not in sig.parameters

    def test_with_alleles_rejects_start_maternal_positional(self):
        """A fourth positional argument (the old ``start_maternal`` slot)
        must also be rejected, not silently bound."""
        with pytest.raises(TypeError, match="positional argument"):
            compute_recombinant_haplotypes_with_alleles(
                ["A1", "B1"], ["A2", "B2"], np.array([0.1]), True
            )

    def test_pure_function_still_accepts_start_maternal(self):
        """Contract preserved: the pure function keeps ``start_maternal``
        (default True) and both keyword values give exact pattern blocks."""
        sig = inspect.signature(compute_recombinant_haplotypes)
        assert "start_maternal" in sig.parameters
        assert sig.parameters["start_maternal"].default is True

        patterns, freqs = compute_recombinant_haplotypes(
            2, np.array([0.0]), start_maternal=True
        )
        np.testing.assert_array_equal(patterns, [[0, 0], [0, 1]])
        np.testing.assert_array_equal(freqs, [1.0, 0.0])

        patterns_p, freqs_p = compute_recombinant_haplotypes(
            2, np.array([0.0]), start_maternal=False
        )
        np.testing.assert_array_equal(patterns_p, [[1, 1], [1, 0]])
        np.testing.assert_array_equal(freqs_p, [1.0, 0.0])


class TestBothHomologPatterns:
    """Structural contract of the module-level helper behind issue #43:
    both homolog starts enumerated with equal 0.5 prior weight."""

    def test_row_structure_and_complement_symmetry(self):
        """For rates=[0.1, 0.2], n_loci=3: 2*2^(n-1) rows, maternal-start
        block first, and each paternal row is the bitwise complement of the
        corresponding maternal row at equal frequency (the exact reason every
        per-locus marginal is 0.5)."""
        rates = np.array([0.1, 0.2])
        patterns, freqs = _compute_both_homolog_patterns(3, rates)
        half = 2 ** (3 - 1)
        assert patterns.shape == (2 * half, 3)
        np.testing.assert_array_equal(
            patterns[:, 0], np.array([0] * half + [1] * half)
        )
        np.testing.assert_array_equal(patterns[:half], 1 - patterns[half:])
        np.testing.assert_allclose(freqs[:half], freqs[half:])
        assert freqs.sum() == pytest.approx(1.0)

    def test_two_loci_exact_rows_and_frequencies(self):
        """rates=[0.1], n_loci=2 → exact row order [[0,0],[0,1],[1,1],[1,0]]
        with frequencies [0.45, 0.05, 0.45, 0.05]."""
        patterns, freqs = _compute_both_homolog_patterns(2, np.array([0.1]))
        np.testing.assert_array_equal(patterns, [[0, 0], [0, 1], [1, 1], [1, 0]])
        np.testing.assert_allclose(freqs, [0.45, 0.05, 0.45, 0.05])

    def test_single_locus_two_half_mass_patterns(self):
        """n_loci=1 → patterns [[0],[1]] at exactly 0.5 each."""
        patterns, freqs = _compute_both_homolog_patterns(1, np.array([]))
        np.testing.assert_array_equal(patterns, [[0], [1]])
        np.testing.assert_allclose(freqs, [0.5, 0.5])


class TestProduceGametesAxisCombinations:
    """Axis-combination attack: recombination rate x locus count, asserting
    probability conservation, exact frequency multisets, and the 0.5
    maternal marginal at EVERY locus (not just the first)."""

    @pytest.mark.parametrize("rate", [0.0, 0.01, 0.1, 0.5])
    @pytest.mark.parametrize("n_loci", [2, 3])
    def test_rate_by_locus_count_cartesian_product(self, rate: float, n_loci: int):
        """For each (rate, n_loci) pair: frequencies sum to 1, every locus has
        maternal marginal 0.5, and the frequency multiset matches the closed
        form 0.5 * r^k * (1-r)^(n-1-k) with multiplicity 2*C(n-1, k)."""
        locus_names = ["locA", "locB", "locC"][:n_loci]
        sp = nt.Species.from_dict(
            name=f"PG_axis_L{n_loci}_R{rate}",
            structure={"chr1": {ln: [f"{ln}_m", f"{ln}_p"] for ln in locus_names}},
        )
        chrom = sp.chromosomes[0]
        loci = list(chrom.loci)
        for i in range(n_loci - 1):
            chrom.set_recombination(loci[i], loci[i + 1], rate)
        mat_genes = [Gene(f"{ln}_m", locus=loci[i]) for i, ln in enumerate(locus_names)]
        pat_genes = [Gene(f"{ln}_p", locus=loci[i]) for i, ln in enumerate(locus_names)]
        mat_hap = Haplotype(chromosome=chrom, genes=mat_genes)
        pat_hap = Haplotype(chromosome=chrom, genes=pat_genes)
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()

        # Invariant 1: probability conservation.
        assert sum(gametes.values()) == pytest.approx(1.0)

        # Invariant 2: EVERY locus transmits the maternal allele with
        # marginal exactly 0.5 (both homolog starts, equal weight).
        for locus_idx, locus in enumerate(loci):
            maternal_mass = sum(
                freq
                for hg, freq in gametes.items()
                if hg.get_gene_at_locus(locus) is mat_genes[locus_idx]
            )
            assert maternal_mass == pytest.approx(0.5), (
                f"rate={rate}, locus={locus_idx}"
            )

        if rate == 0.0:
            # All-zero rates take the Mendelian shortcut: only the two
            # parental gametes at exactly 0.5 each.
            assert len(gametes) == 2
            assert sorted(gametes.values()) == pytest.approx([0.5, 0.5])
        else:
            # All 2**n_loci allele strings carry positive mass, and a gamete
            # whose inheritance chain crosses k boundaries has frequency
            # 0.5 * r^k * (1-r)^(n-1-k); each such value occurs once per
            # homolog start per crossover-bit arrangement: 2*C(n-1, k) times.
            assert len(gametes) == 2 ** n_loci
            expected_sorted = sorted(
                0.5 * rate**k * (1.0 - rate) ** (n_loci - 1 - k)
                for k in range(n_loci)
                for _ in range(2 * comb(n_loci - 1, k))
            )
            assert sorted(gametes.values()) == pytest.approx(expected_sorted), (
                f"rate={rate}, n_loci={n_loci}"
            )


class TestZygotesToGametesMapIntegration:
    """Downstream integration (issue #43): the engine's zygotes_to_gametes_map
    must inherit the corrected Mendelian distribution per sex slice."""

    def test_gamete_map_row_matches_mendelian_two_locus(self):
        """For a two-locus heterozygote at r=0.1 the gamete-map row equals
        produce_gametes() exactly (renormalization is the identity when the
        baseline sums to 1), and the first-locus maternal-allele marginal is
        0.5 in every sex slice — a 1.0/0.0 segregation distortion would fail."""
        sp = nt.Species.from_dict(
            name="GMap_recomb_two_locus",
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
        )
        chrom = sp.chromosomes[0]
        loc_a = chrom.loci[0]
        loc_b = chrom.loci[1]
        a1 = Gene("A1", locus=loc_a)
        a2 = Gene("A2", locus=loc_a)
        b1 = Gene("B1", locus=loc_b)
        b2 = Gene("B2", locus=loc_b)
        chrom.set_recombination(loc_a, loc_b, 0.1)
        mat_hap = Haplotype(chromosome=chrom, genes=[a1, b1])
        pat_hap = Haplotype(chromosome=chrom, genes=[a2, b2])
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        het_gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)

        haploid_genotypes = sp.get_all_haploid_genotypes()
        diploid_genotypes = sp.get_all_genotypes()
        # Species enumeration repeats the cached instance (unordered species);
        # every matching entry must be the identical genotype object, and the
        # repeated ztype rows must all carry the same distribution.
        rows = [i for i, g in enumerate(diploid_genotypes) if g is het_gt]
        assert len(rows) >= 1
        zrow = rows[0]  # n_slabs == 1 → ztype index equals genotype index

        gamete_map = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=1,
        )
        base = het_gt.produce_gametes()

        for sex_idx in range(gamete_map.shape[0]):
            row = gamete_map[sex_idx, zrow, :]
            for dup in rows:
                np.testing.assert_allclose(
                    gamete_map[sex_idx, dup, :], row, err_msg=f"sex={sex_idx}"
                )
            assert row.sum() == pytest.approx(1.0), f"sex={sex_idx}"
            # Exact equality with the fixed per-genotype distribution.
            for j, hg in enumerate(haploid_genotypes):
                assert row[j] == pytest.approx(base.get(hg, 0.0)), (
                    f"sex={sex_idx}, gtype={j}"
                )
            # First-locus maternal marginal is Mendelian in each sex slice.
            maternal_mass = sum(
                float(row[j])
                for j, hg in enumerate(haploid_genotypes)
                if hg.get_gene_at_locus(loc_a) is a1
            )
            assert maternal_mass == pytest.approx(0.5), f"sex={sex_idx}"
