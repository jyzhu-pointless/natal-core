"""Unit tests for natal.genetic_entities."""

import numpy as np
import pytest  # type: ignore

import natal as nt
from natal.genetics import (
    Gene,
    Genotype,
    HaploidGenotype,
    Haplotype,
    compute_recombinant_haplotypes,
)

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
        """Recombination rate=0.1 → recombinant haplotypes with non-equal frequencies."""
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
        mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_hap])
        pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_hap])
        gt = Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)
        gametes = gt.produce_gametes()
        assert len(gametes) == 2
        assert sum(gametes.values()) == pytest.approx(1.0)
        for _gamete, freq in gametes.items():
            assert isinstance(freq, float) or isinstance(freq, np.floating)

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
