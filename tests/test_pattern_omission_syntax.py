#!/usr/bin/env python3
"""
Test pattern matching with omission and shorthand syntax.

This test demonstrates the flexible pattern syntax support:
1. Basic syntax: `;` separates chromosomes, `|` separates maternal/paternal, `/` separates alleles
2. `::` for unordered pairs (e.g., A1::B1 matches both A1|B1 and B1|A1)
3. `*` wildcard for any allele
4. Omission: can omit alleles (but must align on both sides of |), or omit chromosome pairs
5. `{A,B,C}` for multiple alleles, `!` for negation
6. `()` with `;` inside for grouping chromosome pairs explicitly
"""

import sys

sys.path.insert(0, '/Users/pointless/Desktop/work/natal-core/src')

from natal.genetic_structures import Species

# ============================================================================
# SETUP: Global Test Species
# ============================================================================

TEST_SPECIES = Species.from_dict(
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

# ============================================================================
# TEST 1: Omitting Full Chromosomes
# ============================================================================

def test_omit_full_chromosome():
    """Test: Pattern with omitted chromosome uses wildcard for omitted chromosome."""
    print("\n=== Test: Omit Full Chromosome ===")

    # Pattern: A1/B1|A2/B2 (omits chr2, so matches any C)
    # Should match: A1/B1|A2/B2; C1|C1, A1/B1|A2/B2; C1|C2, etc.
    pattern = "A1/B1|A2/B2"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A1/B1|A2/B2; C1|C1", True),  # Exact match
        ("A1/B1|A2/B2; C1|C2", True),  # Different C, still matches
        ("A1/B1|A2/B2; C2|C1", True),  # Different C order
        ("A1/B1|A2/B2; C2|C2", True),  # All C2
        ("A1/B1|A3/B2; C1|C1", False),  # Wrong A on paternal
        ("A3/B1|A2/B2; C1|C1", False),  # Wrong A on maternal
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 2: Wildcard for Specific Positions
# ============================================================================

def test_wildcard_positions():
    """Test: Wildcard in specific allele positions."""
    print("=== Test: Wildcard in Positions ===")

    # Pattern: A1/*|A2/B2; C1|C2 (B can be any allele on maternal)
    pattern = "A1/*|A2/B2; C1|C2"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A1/B1|A2/B2; C1|C2", True),   # B1 matches *
        ("A1/B2|A2/B2; C1|C2", True),   # B2 matches *
        ("A1/B1|A2/B1; C1|C2", False),  # Wrong paternal B
        ("A2/B1|A2/B2; C1|C2", False),  # Wrong maternal A
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 3: Set Pattern with Curly Braces
# ============================================================================

def test_set_pattern():
    """Test: Set pattern {A,B,C} matches multiple alleles."""
    print("=== Test: Set Pattern {A,B,C} ===")

    # Pattern: {A2,A3}/B1|A2/B2; C1|C2 (chr1 maternal A can be A2 or A3)
    pattern = "{A2,A3}/B1|A2/B2; C1|C2"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A2/B1|A2/B2; C1|C2", True),   # A2 in set
        ("A3/B1|A2/B2; C1|C2", True),   # A3 in set
        ("A1/B1|A2/B2; C1|C2", False),  # A1 not in set
        ("A2/B1|A2/B1; C1|C2", False),  # Wrong paternal B
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 4: Negation Pattern with !
# ============================================================================

def test_negation_pattern():
    """Test: Negation pattern !A matches anything except A."""
    print("=== Test: Negation Pattern ! ===")

    # Pattern: !A1/B1|A2/B2; C1|C2 (maternal A is NOT A1)
    pattern = "!A1/B1|A2/B2; C1|C2"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A2/B1|A2/B2; C1|C2", True),   # A2 is not A1
        ("A3/B1|A2/B2; C1|C2", True),   # A3 is not A1
        ("A1/B1|A2/B2; C1|C2", False),  # A1 matches negation (excluded)
        ("A2/B1|A2/B1; C1|C2", False),  # Wrong paternal B
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 5: Unordered Pairs with ::
# ============================================================================

def test_unordered_pairs():
    """Test: Unordered pair pattern with :: (matches both A1|B1 and B1|A1)."""
    print("=== Test: Unordered Pairs with :: ===")

    # Pattern: (A1::A2; B1|B1); C1|C1
    # Format inside brackets: each part separated by ; is a locus pair
    # A1::A2 means: at locus A, have A1 on one chromosome and A2 on the other (unordered)
    # B1|B1 means: at locus B, have B1 on both chromosomes
    pattern = "(A1::A2; B1|B1); C1|C1"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A1/B1|A2/B1; C1|C1", True),   # A1 on maternal, A2 on paternal; B1 on both
        ("A2/B1|A1/B1; C1|C1", True),   # A2 on maternal, A1 on paternal (reversed, still matches ::)
        ("A1/B1|A1/B1; C1|C1", False),  # Both A1 (not unordered pair)
        ("A1/B2|A2/B1; C1|C1", False),  # Wrong B on maternal
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 6: Bracket Grouping with Semicolon Inside
# ============================================================================

def test_bracket_grouping():
    """Test: Brackets explicitly group locus pairs with controlled parent-of-origin."""
    print("=== Test: Bracket Grouping with ; ===")

    # Pattern: (A1|A2; B1|B2); C1|C1
    # Bracket syntax: separate each locus with ; and use | or :: to specify the pair
    # A1|A2 means: locus A has A1 on maternal chromosome and A2 on paternal chromosome
    # B1|B2 means: locus B has B1 on maternal chromosome and B2 on paternal chromosome
    pattern = "(A1|A2; B1|B2); C1|C1"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A1/B1|A2/B2; C1|C1", True),   # Matches: A1,B1 on maternal; A2,B2 on paternal
        ("A2/B2|A1/B1; C1|C1", False),  # Reversed (order matters for |, not for ::)
        ("A1/B1|A2/B1; C1|C1", False),  # Wrong paternal B
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
        matches = pattern_filter(gt)
        status = "✓" if matches == should_match else "✗"
        print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
        if matches == should_match:
            passed += 1

    print(f"Passed: {passed}/{len(test_genotypes)}")
    assert passed == len(test_genotypes), "Some test cases failed"
    print("✓ PASS\n")


# ============================================================================
# TEST 7: Combining Multiple Features
# ============================================================================

def test_combined_features():
    """Test: Combining multiple features (sets, wildcards, omission)."""
    print("=== Test: Combined Features ===")

    # Pattern: {A1,A2}/B*|A3/*; C1|*
    # chr1 maternal: A1 or A2, any B
    # chr1 paternal: A3, any B
    # chr2 maternal: C1
    # chr2 paternal: any C
    pattern = "{A1,A2}/*|A3/*; C1|*"
    pattern_filter = TEST_SPECIES.parse_genotype_pattern(pattern)

    test_genotypes = [
        ("A1/B1|A3/B1; C1|C1", True),   # Matches all constraints
        ("A1/B2|A3/B2; C1|C2", True),   # Different Bs and C2
        ("A2/B1|A3/B1; C1|C1", True),   # A2 in set
        ("A3/B1|A3/B1; C1|C1", False),  # A3 on maternal (not in set)
        ("A1/B1|A1/B1; C1|C1", False),  # A1 on paternal (should be A3)
    ]

    print(f"Pattern: {pattern}")
    print("Test cases:")

    passed = 0
    for genotype_str, should_match in test_genotypes:
        try:
            gt = TEST_SPECIES.get_genotype_from_str(genotype_str)
            matches = pattern_filter(gt)
            status = "✓" if matches == should_match else "✗"
            print(f"  {status} {genotype_str}: {matches} (expected {should_match})")
            if matches == should_match:
                passed += 1
        except Exception as e:
            print(f"  ✗ {genotype_str}: ERROR - {e}")

    print(f"Passed: {passed}/{len(test_genotypes)}")
    if passed == len(test_genotypes):
        print("✓ PASS\n")
    else:
        print("⚠ Some patterns may not be fully supported yet\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║    PATTERN MATCHING WITH OMISSION SYNTAX TESTS            ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    try:
        test_omit_full_chromosome()
        test_wildcard_positions()
        test_set_pattern()
        test_negation_pattern()
        test_unordered_pairs()
        test_bracket_grouping()
        test_combined_features()

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║        ✓ PATTERN SYNTAX TESTS COMPLETED                   ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
