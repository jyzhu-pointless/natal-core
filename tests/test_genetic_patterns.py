"""Unit tests for natal.genetic_patterns.GenotypePatternParser."""

from typing import Iterable, List

import natal as nt
from natal.genetic_entities import Gene, Genotype, HaploidGenotype, Haplotype


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


class TestUnorderedChromosomes:
    @staticmethod
    def normalize_enum_output(species: nt.Species, enum: Iterable[Genotype]) -> List[str]:
        return sorted([gt.to_string() for gt in enum])

    def test_unordered_chromosomes(self):
        sp = nt.Species.from_dict(
            name="test2026041201",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )
        enums = [
            sp.enumerate_genotypes_matching_pattern("WT::Dr"),
            sp.enumerate_genotypes_matching_pattern("!Dr::R2"),
            sp.enumerate_genotypes_matching_pattern("{WT,R2}::Dr"),
            sp.enumerate_genotypes_matching_pattern("!{WT,R2}::R2"),
            sp.enumerate_genotypes_matching_pattern("!{WT,Dr,R2}::*"),
        ]
        expected = [
            ["WT|Dr", "Dr|WT"],
            ["WT|R2", "R2|R2", "R2|WT"],
            ["WT|Dr", "R2|Dr", "Dr|WT", "Dr|R2"],
            ["Dr|R2", "R2|Dr"],
            []
        ]
        for enum, exp in zip(enums, expected):
            assert self.normalize_enum_output(sp, enum) == sorted(exp)

    def test_unordered_chromosomes_without_braces(self):
        """Test that set patterns work without braces (WT,R2 should be equivalent to {WT,R2})."""
        sp = nt.Species.from_dict(
            name="test2026041202",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        )

        # Test that WT,R2::Dr produces the same result as {WT,R2}::Dr
        enum_without_braces = list(sp.enumerate_genotypes_matching_pattern("WT,R2::Dr"))
        enum_with_braces = list(sp.enumerate_genotypes_matching_pattern("{WT,R2}::Dr"))

        result_without_braces = self.normalize_enum_output(sp, enum_without_braces)
        result_with_braces = self.normalize_enum_output(sp, enum_with_braces)

        assert result_without_braces == result_with_braces
        assert result_without_braces == sorted(["WT|Dr", "Dr|WT", "Dr|R2", "R2|Dr"])

        # Test that !WT,R2::R2 produces the same result as !{WT,R2}::R2
        enum_neg_without_braces = list(sp.enumerate_genotypes_matching_pattern("!WT,R2::R2"))
        enum_neg_with_braces = list(sp.enumerate_genotypes_matching_pattern("!{WT,R2}::R2"))

        result_neg_without_braces = self.normalize_enum_output(sp, enum_neg_without_braces)
        result_neg_with_braces = self.normalize_enum_output(sp, enum_neg_with_braces)

        assert result_neg_without_braces == result_neg_with_braces
        assert result_neg_without_braces == sorted(["Dr|R2", "R2|Dr"])


class TestPatternOmissionSyntax:
    """Test pattern matching with omission and shorthand syntax."""

    def normalize_enum_output(self, species: nt.Species, enum: Iterable[Genotype]) -> List[str]:
        return sorted([gt.to_string() for gt in enum])

    def test_omit_full_chromosome(self):
        """Test: Pattern with omitted chromosome uses wildcard for omitted chromosome."""
        sp = nt.Species.from_dict(
            "Demo",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: A1/B1|A2/B2 (omits chr2, so matches any C)
        pattern = "A1/B1|A2/B2"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes with correct chr1 pattern and any chr2
        expected = [
            "A1/B1|A2/B2;C1|C1",
            "A1/B1|A2/B2;C1|C2",
            "A1/B1|A2/B2;C2|C1",
            "A1/B1|A2/B2;C2|C2"
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_wildcard_positions(self):
        """Test: Wildcard in specific allele positions."""
        sp = nt.Species.from_dict(
            "test_wildcard_positions",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: A1/*|A2/B2; C1|C2 (B can be any allele on maternal)
        pattern = "A1/*|A2/B2; C1|C2"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes with A1/* on maternal, A2/B2 on paternal, C1|C2 on chr2
        expected = [
            "A1/B1|A2/B2;C1|C2",  # B1 matches *
            "A1/B2|A2/B2;C1|C2",  # B2 matches *
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_set_pattern(self):
        """Test: Set pattern {A,B,C} matches multiple alleles."""
        sp = nt.Species.from_dict(
            "test_set_pattern",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: {A2,A3}/B1|A2/B2; C1|C2 (chr1 maternal A can be A2 or A3)
        pattern = "{A2,A3}/B1|A2/B2; C1|C2"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes with A2 or A3 on maternal, B1 on maternal, A2/B2 on paternal
        expected = [
            "A2/B1|A2/B2;C1|C2",  # A2 in set
            "A3/B1|A2/B2;C1|C2",  # A3 in set
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_negation_pattern(self):
        """Test: Negation pattern !A matches anything except A."""
        sp = nt.Species.from_dict(
            "test_negation",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: !A1/B1|A2/B2; C1|C2 (maternal A is NOT A1)
        pattern = "!A1/B1|A2/B2; C1|C2"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes where maternal A is not A1
        expected = [
            "A2/B1|A2/B2;C1|C2",  # A2 is not A1
            "A3/B1|A2/B2;C1|C2",  # A3 is not A1
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_unordered_pairs(self):
        """Test: Unordered pair pattern with :: (matches both A1|B1 and B1|A1)."""
        sp = nt.Species.from_dict(
            "test_unordered",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: (A1::A2; B1|B1); C1|C1
        pattern = "(A1::A2; B1|B1); C1|C1"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes where A1 and A2 are present (unordered) and B1 on both
        expected = [
            "A1/B1|A2/B1;C1|C1",  # A1 on maternal, A2 on paternal
            "A2/B1|A1/B1;C1|C1",  # A2 on maternal, A1 on paternal (reversed)
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_bracket_grouping(self):
        """Test: Brackets explicitly group locus pairs with controlled parent-of-origin."""
        sp = nt.Species.from_dict(
            "test_bracket_grouping",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: (A1|A2; B1|B2); C1|C1
        pattern = "(A1|A2; B1|B2); C1|C1"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes with exact parent-of-origin specification
        expected = [
            "A1/B1|A2/B2;C1|C1",  # Matches: A1,B1 on maternal; A2,B2 on paternal
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"

    def test_combined_features(self):
        """Test: Combining multiple features (sets, wildcards, omission)."""
        sp = nt.Species.from_dict(
            "test_combined",
            {
                "chr1": {
                    "A": ["A1", "A2", "A3"],
                    "B": ["B1", "B2"],
                },
                "chr2": {
                    "C": ["C1", "C2"],
                },
            }
        )

        # Pattern: {A1,A2}/*|A3/*; C1|*
        pattern = "{A1,A2}/*|A3/*; C1|*"
        enum = list(sp.enumerate_genotypes_matching_pattern(pattern))
        result = self.normalize_enum_output(sp, enum)

        # Expected: Should match genotypes with various combinations
        # Pattern: {A1,A2}/*|A3/*; C1|* means:
        # - Maternal A can be A1 or A2, any B
        # - Paternal A must be A3, any B
        # - Maternal C must be C1, paternal C can be anything
        expected = [
            "A1/B1|A3/B1;C1|C1",  # Matches all constraints
            "A1/B1|A3/B1;C1|C2",  # Different C on paternal
            "A1/B1|A3/B2;C1|C1",  # Different B on paternal
            "A1/B1|A3/B2;C1|C2",  # Different B and C on paternal
            "A1/B2|A3/B1;C1|C1",  # Different B on maternal
            "A1/B2|A3/B1;C1|C2",  # Different B on maternal and C on paternal
            "A1/B2|A3/B2;C1|C1",  # Different Bs
            "A1/B2|A3/B2;C1|C2",  # Different Bs and C on paternal
            "A2/B1|A3/B1;C1|C1",  # A2 in set
            "A2/B1|A3/B1;C1|C2",  # A2 in set, different C on paternal
            "A2/B1|A3/B2;C1|C1",  # A2 in set, different B on paternal
            "A2/B1|A3/B2;C1|C2",  # A2 in set, different B and C on paternal
            "A2/B2|A3/B1;C1|C1",  # A2 in set, different B on maternal
            "A2/B2|A3/B1;C1|C2",  # A2 in set, different B on maternal and C on paternal
            "A2/B2|A3/B2;C1|C1",  # A2 in set, different Bs
            "A2/B2|A3/B2;C1|C2",  # A2 in set, different Bs and C on paternal
        ]

        assert result == sorted(expected), f"Expected {sorted(expected)}, got {result}"


class TestComprehensivePatterns:
    """Test comprehensive pattern matching functionality."""

    def normalize_enum_output(self, species: nt.Species, enum: Iterable[Genotype]) -> List[str]:
        return sorted([gt.to_string() for gt in enum])

    def test_haploid_genome_pattern_basic(self):
        """Test basic HaploidGenome pattern parsing and matching."""
        sp = nt.Species.from_dict(
            "test_haploid_basic",
            {
                "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
                "Chr2": {"C": ["C1", "C2"]},
            }
        )

        # Test haploid genome pattern: A1/B1; C1
        pattern = "A1/B1; C1"
        filter_func = sp.parse_haploid_genome_pattern(pattern)

        # Test haploid genomes
        test_cases = [
            ("A1/B1; C1", True),   # Exact match
            ("A1/B2; C1", False),  # Wrong B
            ("A2/B1; C1", False),  # Wrong A
        ]

        for hap_str, expected in test_cases:
            hg = sp.get_haploid_genome_from_str(hap_str)
            matches = filter_func(hg)
            assert matches == expected, f"{hap_str}: expected {expected}, got {matches}"

    def test_haploid_wildcard_pattern(self):
        """Test wildcard patterns for haploid genomes."""
        sp = nt.Species.from_dict(
            "test_haploid_wildcard",
            {
                "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
                "Chr2": {"C": ["C1", "C2"]},
            }
        )

        # Pattern: */B1; C1 (any A allele with B1, and C1)
        pattern = "*/B1; C1"
        filter_func = sp.parse_haploid_genome_pattern(pattern)

        test_cases = [
            ("A1/B1; C1", True),   # A1 matches *
            ("A2/B1; C1", True),   # A2 matches *
            ("A1/B2; C1", False),  # Wrong B
            ("A2/B2; C1", False),  # Wrong B
        ]

        for hap_str, expected in test_cases:
            hg = sp.get_haploid_genome_from_str(hap_str)
            matches = filter_func(hg)
            assert matches == expected, f"{hap_str}: expected {expected}, got {matches}"

    def test_haploid_set_pattern(self):
        """Test set patterns for haploid genomes."""
        sp = nt.Species.from_dict(
            "test_haploid_set",
            {
                "Chr1": {"A": ["A1", "A2", "A3"]},
            }
        )

        # Pattern: {A1,A2}
        pattern = "{A1,A2}"
        filter_func = sp.parse_haploid_genome_pattern(pattern)

        test_cases = [
            ("A1", True),   # A1 in set
            ("A2", True),   # A2 in set
            ("A3", False),  # A3 not in set
        ]

        for allele_str, expected in test_cases:
            hg = sp.get_haploid_genome_from_str(allele_str)
            matches = filter_func(hg)
            assert matches == expected, f"{allele_str}: expected {expected}, got {matches}"

    def test_haploid_negation_pattern(self):
        """Test negation patterns for haploid genomes."""
        sp = nt.Species.from_dict(
            "test_haploid_negation",
            {
                "Chr1": {"A": ["A1", "A2", "A3"]},
            }
        )

        # Pattern: !A1
        pattern = "!A1"
        filter_func = sp.parse_haploid_genome_pattern(pattern)

        test_cases = [
            ("A1", False),  # A1 matches negation (excluded)
            ("A2", True),   # A2 is not A1
            ("A3", True),   # A3 is not A1
        ]

        for allele_str, expected in test_cases:
            hg = sp.get_haploid_genome_from_str(allele_str)
            matches = filter_func(hg)
            assert matches == expected, f"{allele_str}: expected {expected}, got {matches}"

    def test_genotype_pattern_filter(self):
        """Test Genotype pattern filtering."""
        sp = nt.Species.from_dict(
            "test_genotype_filter",
            {
                "Chr1": {"A": ["A1", "A2"]},
            }
        )

        # Create some genotypes
        genotypes = [
            sp.get_genotype_from_str("A1|A1"),
            sp.get_genotype_from_str("A1|A2"),
            sp.get_genotype_from_str("A2|A2"),
        ]

        # Filter by pattern: A1|*
        pattern = "A1|*"
        filtered = sp.filter_genotypes_by_pattern(genotypes, pattern)

        # Expected: genotypes that have A1 on maternal and anything on paternal
        expected = ["A1|A1", "A1|A2"]
        result = [gt.to_string() for gt in filtered]

        assert sorted(result) == sorted(expected), f"Expected {sorted(expected)}, got {sorted(result)}"
