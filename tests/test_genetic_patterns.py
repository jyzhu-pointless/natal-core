"""Unit tests for natal.genetic_patterns.GenotypePatternParser."""

import pytest
import natal as nt
from natal.genetic_entities import Gene, Haplotype, HaploidGenotype, Genotype


def _build_genotype(sp, mat_allele: str, pat_allele: str) -> Genotype:
    """Helper: create a Genotype with the given maternal/paternal allele names."""
    locus = sp.chromosomes[0].loci[0]
    chrom = sp.chromosomes[0]
    mat_gene = Gene(mat_allele, locus=locus)
    pat_gene = Gene(pat_allele, locus=locus)
    mat_haplo = Haplotype(chromosome=chrom, genes=[mat_gene])
    pat_haplo = Haplotype(chromosome=chrom, genes=[pat_gene])
    mat_hg = HaploidGenotype(species=sp, haplotypes=[mat_haplo])
    pat_hg = HaploidGenotype(species=sp, haplotypes=[pat_haplo])
    return Genotype(species=sp, maternal=mat_hg, paternal=pat_hg)


class TestExactMatch:
    def test_homozygous_wt_matches_wt_wt(self):
        sp = nt.Species.from_dict(
            name="Pat_homo_match",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("WT|WT")
        gt = _build_genotype(sp, "WT", "WT")
        assert pattern.matches(gt) is True

    def test_homozygous_wt_does_not_match_wt_dr(self):
        sp = nt.Species.from_dict(
            name="Pat_homo_no_match",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("WT|WT")
        gt = _build_genotype(sp, "WT", "Dr")
        assert pattern.matches(gt) is False

    def test_heterozygous_wt_dr_matches_wt_dr(self):
        sp = nt.Species.from_dict(
            name="Pat_het_match",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("WT|Dr")
        gt = _build_genotype(sp, "WT", "Dr")
        assert pattern.matches(gt) is True

    def test_maternal_paternal_order_matters(self):
        """WT|Dr must NOT match Dr|WT."""
        sp = nt.Species.from_dict(
            name="Pat_order",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("WT|Dr")
        gt_dr_wt = _build_genotype(sp, "Dr", "WT")
        assert pattern.matches(gt_dr_wt) is False

    def test_dr_wt_matches_dr_wt(self):
        sp = nt.Species.from_dict(
            name="Pat_dr_wt",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("Dr|WT")
        gt = _build_genotype(sp, "Dr", "WT")
        assert pattern.matches(gt) is True


class TestWildcard:
    def test_wildcard_matches_all(self):
        sp = nt.Species.from_dict(
            name="Pat_wild_all",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("*|*")
        for mat in ("WT", "Dr", "R2"):
            for pat in ("WT", "Dr", "R2"):
                gt = _build_genotype(sp, mat, pat)
                assert pattern.matches(gt) is True, f"Expected *|* to match {mat}|{pat}"

    def test_wildcard_pat_matches_any_paternal(self):
        sp = nt.Species.from_dict(
            name="Pat_wild_pat",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        pattern = parser.parse("WT|*")
        assert pattern.matches(_build_genotype(sp, "WT", "WT")) is True
        assert pattern.matches(_build_genotype(sp, "WT", "Dr")) is True
        assert pattern.matches(_build_genotype(sp, "Dr", "WT")) is False


class TestMultipleGenotypes:
    def test_three_alleles_all_homozygous_patterns(self):
        sp = nt.Species.from_dict(
            name="Pat_three_homo",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        parser = nt.GenotypePatternParser(sp)
        for allele in ("WT", "Dr", "R2"):
            pattern = parser.parse(f"{allele}|{allele}")
            gt_match = _build_genotype(sp, allele, allele)
            assert pattern.matches(gt_match) is True
            # must not match the other homozygous genotype
            for other in ("WT", "Dr", "R2"):
                if other != allele:
                    gt_other = _build_genotype(sp, other, other)
                    assert pattern.matches(gt_other) is False
