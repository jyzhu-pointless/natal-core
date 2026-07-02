"""Test bracket pattern parsing — verifies proper handling of parentheses in patterns.

Uses a single global species to avoid caching issues.
"""

from natal.genetic_patterns import GenotypePatternParser, PatternParseError
from natal.genetic_structures import Species

TEST_SPECIES = Species.from_dict('test_bracket', {
    'chr1': {'A': ['A1', 'A2', 'A3'], 'B': ['B1', 'B2']},
    'chr2': {'C': ['C1', 'C2']}
})


def test_split_function():
    """Test the split_by_semicolon_respecting_parens function."""
    parser = GenotypePatternParser(TEST_SPECIES)

    test_cases = [
        ('(A1::A2; B1|B1); C1|C1', ['(A1::A2; B1|B1)', 'C1|C1']),
        ('A1|A2; B1|B2', ['A1|A2', 'B1|B2']),
        ('(A1; B1); (C1; D1)', ['(A1; B1)', '(C1; D1)']),
        ('A1; (B1; B2); C1', ['A1', '(B1; B2)', 'C1']),
    ]

    for input_str, expected in test_cases:
        result = parser._split_by_semicolon_respecting_parens(input_str)
        assert result == expected, f"'{input_str}': expected {expected}, got {result}"


def test_genotype_bracket_patterns():
    """Test genotype patterns with brackets.

    For genotypes, brackets represent a PAIR of haplotypes (two DNA strands).
    Inside brackets, ``;`` separates locus pairs using ``|`` (ordered) or
    ``::`` (unordered).
    """
    patterns = [
        ('(A1::A2; B1|B1); C1|C1', 'Bracket: unordered + ordered locus pairs'),
        ('A1/B1|A2/B2; C1|C2', 'Standard genotype (no brackets)'),
        ('(A1|B1; A2|B2); C1|C1', 'Bracket with locus pairs, both ordered'),
    ]

    for pattern_str, description in patterns:
        pattern = TEST_SPECIES.parse_genotype_pattern(pattern_str)
        assert pattern is not None, f"'{description}' failed to parse: {pattern_str}"


def test_haploid_bracket_patterns():
    """Test haploid genome patterns with brackets.

    For haploid genomes, brackets represent a SINGLE haplotype (one DNA strand).
    Inside brackets, ``;`` separates individual loci.
    """
    patterns = [
        ('(A1; B1); C1', 'Bracket with semicolon-separated loci at chr1'),
        ('A1/B1; C1', 'Standard haploid pattern without brackets'),
        ('(A1; B1)', 'Single chromosome with bracket'),
    ]

    for pattern_str, description in patterns:
        pattern = TEST_SPECIES.parse_haploid_genome_pattern(pattern_str)
        assert pattern is not None, f"'{description}' failed to parse: {pattern_str}"


def test_important_distinction():
    """Verify the key difference between genotype and haploid bracket semantics.

    Genotype brackets represent a PAIR of haplotypes (two DNA strands),
    using ``|`` or ``::`` to separate the two alleles within each locus pair.

    Haploid genome brackets represent a SINGLE haplotype (one DNA strand),
    using ``;`` to separate individual loci.
    """
    gt_pattern = TEST_SPECIES.parse_genotype_pattern('(A1::A2; B1|B1); C1|C1')
    assert gt_pattern is not None, "genotype bracket pattern should parse"

    hg_pattern = TEST_SPECIES.parse_haploid_genome_pattern('(A1; B1); C1')
    assert hg_pattern is not None, "haploid bracket pattern should parse"


def test_genotype_bracket_invalid_format():
    """Verify that ``/`` is NOT valid inside a genotype bracket — must use ``|`` or ``::``."""
    import pytest
    with pytest.raises(PatternParseError, match="Locus pair must contain"):
        TEST_SPECIES.parse_genotype_pattern('(A1::A2; B1/B1); C1|C1')
