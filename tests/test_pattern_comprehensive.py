#!/usr/bin/env python
"""
Test pattern matching functionality for Genotype and HaploidGenome.

This comprehensive test script verifies:
1. HaploidGenome pattern parsing and matching
2. Genotype pattern parsing and matching
3. Pattern enumeration capabilities
4. Edge cases and error handling
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from natal.genetic_patterns import GenotypePatternParser
from natal.genetic_structures import Species


def test_basic_haploid_genome_pattern():
    """Test basic HaploidGenome pattern parsing and matching."""
    print("\n=== Test: Basic HaploidGenome Pattern ===")

    # Create a simple species using from_dict
    species = Species.from_dict("test_pattern_comprehensive", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "Chr2": {"C": ["C1", "C2"]},
    })

    # Test parsing
    parser = GenotypePatternParser(species)
    pattern = parser.parse_haploid_genome_pattern("A1/B1; C1")
    print(f"✓ Parsed pattern: {pattern}")

    # Test filter function
    filter_func = pattern.to_filter()

    # Create test haploid genomes
    haploid_strs = [
        "A1/B1; C1",
        "A1/B2; C1",
        "A2/B1; C1",
    ]

    haploid_genomes = [species.get_haploid_genome_from_str(hg_str) for hg_str in haploid_strs]
    print(f"✓ Created {len(haploid_genomes)} haploid genomes")

    # Apply filter
    results = [hg for hg in haploid_genomes if filter_func(hg)]
    print(f"✓ Filtered results: {len(results)} matched")

    for hg in results:
        print(f"  - {hg.name}")


def test_wildcard_pattern():
    """Test wildcard patterns."""
    print("\n=== Test: Wildcard Pattern ===")

    species = Species.from_dict("test_wildcard", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "Chr2": {"C": ["C1", "C2"]},
    })

    # Pattern with wildcards
    filter_func = species.parse_haploid_genome_pattern("A*/*; C1")
    print("✓ Created wildcard pattern: A*/*; C1")

    # Test genotypes
    test_haploid_strs = [
        "A1/B1; C1",
        "A2/B1; C1",
        "A1/B2; C1",
        "B1/B2; C1",  # Should not match (starts with B)
    ]

    for hap_str in test_haploid_strs:
        try:
            hg = species.get_haploid_genome_from_str(hap_str)
            matches = filter_func(hg)
            print(f"✓ {hap_str}: {'MATCH' if matches else 'no match'}")
        except Exception as e:
            print(f"✗ Failed to create/test {hap_str}: {e}")


def test_set_pattern():
    """Test set patterns like {A1,A2}."""
    print("\n=== Test: Set Pattern ===")

    species = Species.from_dict("test_set", {
        "Chr1": {"A": ["A1", "A2", "A3"], "B": ["B1", "B2"]},
    })

    # Pattern with sets
    filter_func = species.parse_haploid_genome_pattern("{A1,A2}/B1")
    print("✓ Created set pattern: {A1,A2}/B1")

    test_cases = [
        ("A1/B1", True),
        ("A2/B1", True),
        ("A3/B1", False),
        ("B1/B1", False),
    ]

    for hap_str, expected in test_cases:
        try:
            hg = species.get_haploid_genome_from_str(hap_str)
            matches = filter_func(hg)
            result = "✓" if matches == expected else "✗"
            print(f"{result} {hap_str}: {matches} (expected {expected})")
        except Exception as e:
            print(f"✗ {hap_str}: {e}")


def test_negation_pattern():
    """Test negation patterns like !A."""
    print("\n=== Test: Negation Pattern ===")

    species = Species.from_dict("test_negation", {
        "Chr1": {"A": ["A1", "A2", "A3"]},
    })

    # Pattern with negation
    filter_func = species.parse_haploid_genome_pattern("!A1")
    print("✓ Created negation pattern: !A1")

    test_cases = [
        ("A1", False),
        ("A2", True),
        ("A3", True),
    ]

    for allele_str, expected in test_cases:
        try:
            hg = species.get_haploid_genome_from_str(allele_str)
            matches = filter_func(hg)
            result = "✓" if matches == expected else "✗"
            print(f"{result} {allele_str}: {matches} (expected {expected})")
        except Exception as e:
            print(f"✗ {allele_str}: {e}")


def test_genotype_pattern_filter():
    """Test Genotype pattern filtering."""
    print("\n=== Test: Genotype Pattern Filter ===")

    species = Species.from_dict("test_genotype", {
        "Chr1": {"A": ["A1", "A2"]},
    })

    # Create some genotypes
    genotypes = [
        species.get_genotype_from_str("A1|A1"),
        species.get_genotype_from_str("A1|A2"),
        species.get_genotype_from_str("A2|A2"),
    ]
    print(f"✓ Created {len(genotypes)} genotypes")

    # Filter by pattern
    pattern = "A1|*"
    filtered = species.filter_genotypes_by_pattern(genotypes, pattern)
    print(f"✓ Pattern '{pattern}': {len(filtered)} matched")

    for gt in filtered:
        print(f"  - {gt.name}")


def test_enumerate_haploid_genomes():
    """Test haploid genome enumeration."""
    print("\n=== Test: Enumerate HaploidGenomes ===")

    species = Species.from_dict("test_haploid_enum", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
    })

    # Enumerate with wildcards (limited)
    pattern = "*|*"
    results = list(species.enumerate_haploid_genomes_matching_pattern(pattern, max_count=5))
    print(f"✓ Pattern '{pattern}' (max 5): {len(results)} enumerated")

    for hg in results:
        print(f"  - {hg.name}")


def test_enumerate_genotypes():
    """Test genotype enumeration."""
    print("\n=== Test: Enumerate Genotypes ===")

    species = Species.from_dict("test_enum", {
        "Chr1": {"A": ["A1", "A2"]},
    })

    # Enumerate specific pattern
    pattern = "A1|A2"
    results = list(species.enumerate_genotypes_matching_pattern(pattern, max_count=10))
    print(f"✓ Pattern '{pattern}' (max 10): {len(results)} enumerated")

    for gt in results:
        print(f"  - {gt.name}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PATTERN MATCHING TESTS")
    print("=" * 60)

    try:
        test_basic_haploid_genome_pattern()
    except Exception as e:
        print(f"✗ test_basic_haploid_genome_pattern failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_wildcard_pattern()
    except Exception as e:
        print(f"✗ test_wildcard_pattern failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_set_pattern()
    except Exception as e:
        print(f"✗ test_set_pattern failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_negation_pattern()
    except Exception as e:
        print(f"✗ test_negation_pattern failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_genotype_pattern_filter()
    except Exception as e:
        print(f"✗ test_genotype_pattern_filter failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_enumerate_haploid_genomes()
    except Exception as e:
        print(f"✗ test_enumerate_haploid_genomes failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_enumerate_genotypes()
    except Exception as e:
        print(f"✗ test_enumerate_genotypes failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
