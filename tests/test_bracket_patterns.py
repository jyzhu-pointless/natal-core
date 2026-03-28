#!/usr/bin/env python
"""
Test bracket pattern parsing - verifies proper handling of parentheses in patterns.
Uses a single global species to avoid caching issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from natal.genetic_structures import Species
from natal.genetic_patterns import GenotypePatternParser


# Global test species - created once to avoid re-instantiation issues
TEST_SPECIES = Species.from_dict('test', {
    'chr1': {'A': ['A1', 'A2', 'A3'], 'B': ['B1', 'B2']},
    'chr2': {'C': ['C1', 'C2']}
})


def test_split_function():
    """Test the split_by_semicolon_respecting_parens function."""
    print("\n=== Test: Split Function (Respecting Parentheses) ===")
    
    parser = GenotypePatternParser(TEST_SPECIES)
    
    test_cases = [
        ('(A1::A2; B1/B1); C1|C1', ['(A1::A2; B1/B1)', 'C1|C1']),
        ('A1|A2; B1|B2', ['A1|A2', 'B1|B2']),
        ('(A1; B1); (C1; D1)', ['(A1; B1)', '(C1; D1)']),
        ('A1; (B1; B2); C1', ['A1', '(B1; B2)', 'C1']),
    ]
    
    for input_str, expected in test_cases:
        result = parser._split_by_semicolon_respecting_parens(input_str)
        matches = result == expected
        status = "✓" if matches else "✗"
        print(f"{status} {input_str}")
        if not matches:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")


def test_genotype_bracket_patterns():
    """Test genotype patterns with brackets.
    
    For genotypes, brackets represent a PAIR of haplotypes (two DNA strands).
    Inside brackets, `;` separates locus pairs like A1::A2 or B1/B1.
    """
    print("\n=== Test: Genotype Bracket Patterns ===")
    
    patterns = [
        # (pattern_string, description)
        ('(A1::A2; B1/B1); C1|C1', 'Bracket with unordered locus pair + ordered locus pair'),
        ('A1/B1|A2/B2; C1|C2', 'Standard genotype (no brackets)'),
        ('(A1/B1; A2/B2); C1|C1', 'Bracket with locus pairs, both ordered'),
    ]
    
    for pattern_str, description in patterns:
        try:
            pattern = TEST_SPECIES.parse_genotype_pattern(pattern_str)
            print(f"✓ {description}")
            print(f"  Pattern: {pattern_str}")
        except Exception as e:
            print(f"✗ {description}")
            print(f"  Pattern: {pattern_str}")
            print(f"  Error: {e}")


def test_haploid_bracket_patterns():
    """Test haploid genome patterns with brackets.
    
    For haploid genomes, brackets represent a SINGLE haplotype (one DNA strand).
    Inside brackets, `;` separates individual loci.
    """
    print("\n=== Test: HaploidGenome Bracket Patterns ===")
    
    patterns = [
        # (pattern_string, description)
        ('(A1; B1); C1', 'Bracket with semicolon-separated loci at chr1'),
        ('A1/B1; C1', 'Standard haploid pattern without brackets'),
        ('(A1; B1)', 'Single chromosome with bracket'),
    ]
    
    for pattern_str, description in patterns:
        try:
            pattern = TEST_SPECIES.parse_haploid_genome_pattern(pattern_str)
            print(f"✓ {description}")
            print(f"  Pattern: {pattern_str}")
        except Exception as e:
            print(f"✗ {description}")
            print(f"  Pattern: {pattern_str}")
            print(f"  Error: {e}")


def test_important_distinction():
    """Verify the key difference between genotype and haploid bracket semantics."""
    print("\n=== Key Distinction: Genotype vs HaploidGenome Brackets ===")
    
    print("\nGenotype brackets represent a PAIR of haplotypes:")
    print("  (A1::A2; B1/B1) = Two DNA strands with locus pairs")
    print("  A1::A2 = first locus: strand1 has A1, strand2 has A2 (unordered)")
    print("  B1/B1 = second locus: both strands have B1 (maternal/paternal ordered)")
    
    try:
        gt_pattern = TEST_SPECIES.parse_genotype_pattern('(A1::A2; B1/B1); C1|C1')
        print(f"✓ Genotype pattern parsed: (A1::A2; B1/B1); C1|C1")
    except Exception as e:
        print(f"✗ Genotype pattern failed: {e}")
    
    print("\nHaploid genome brackets represent a SINGLE haplotype:")
    print("  (A1; B1) = One DNA strand with arbitrary loci")
    print("  A1 = first locus: this strand has A1")
    print("  B1 = second locus: this strand has B1")
    
    try:
        hg_pattern = TEST_SPECIES.parse_haploid_genome_pattern('(A1; B1); C1')
        print(f"✓ HaploidGenome pattern parsed: (A1; B1); C1")
    except Exception as e:
        print(f"✗ HaploidGenome pattern failed: {e}")


if __name__ == '__main__':
    test_split_function()
    test_genotype_bracket_patterns()
    test_haploid_bracket_patterns()
    test_important_distinction()
    print("\n=== All Tests Complete ===\n")
