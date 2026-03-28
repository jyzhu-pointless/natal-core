#!/usr/bin/env python3
"""Test script for genotype pattern matching system."""

import sys
sys.path.insert(0, '/Users/pointless/Desktop/work/natal-core/src')

from natal.genetic_structures import Species
from natal.genetic_patterns import GenotypePatternParser, PatternParseError

# Create a test species
species = Species.from_dict("TestSpecies", {
    "Chr1": {
        "A": ["A1", "A2", "A3"],
        "B": ["B1", "B2"]
    },
    "Chr2": {
        "C": ["C1", "C2"],
        "D": ["D1", "D2", "D3"]
    }
})

print("=" * 60)
print("Testing Genotype Pattern Matching System")
print("=" * 60)


# Test 1: Basic pattern parsing and filtering
print("\n[Test 1] Basic pattern parsing")
try:
    pattern_filter = species.parse_genotype_pattern("A1/B1|A2/B2; C1/C1")
    print("✓ Pattern parsed successfully: 'A1/B1|A2/B2; C1/C1'")
    
    # Test a matching genotype
    gt1 = species.get_genotype_from_str("A1/B1|A2/B2; C1/C1")
    result = pattern_filter(gt1)
    print(f"✓ Filter applied to matching genotype: {result}")
    assert result == True, "Should match"
    
    # Test a non-matching genotype
    gt2 = species.get_genotype_from_str("A1/B2|A2/B1; C1/C1")
    result = pattern_filter(gt2)
    print(f"✓ Filter applied to non-matching genotype: {result}")
    assert result == False, "Should not match"
    
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 2: Wildcard patterns
print("\n[Test 2] Wildcard patterns (*)")
try:
    pattern_filter = species.parse_genotype_pattern("A1/*|A2/B2; */*")
    print("✓ Wildcard pattern parsed: 'A1/*|A2/B2; */*'")
    
    gt1 = species.get_genotype_from_str("A1/B1|A2/B2; C1/C2")
    result = pattern_filter(gt1)
    print(f"✓ Wildcard pattern matches: {result}")
    assert result == True, "Should match"
    
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 3: Set patterns
print("\n[Test 3] Set patterns {{A,B,C}}")
try:
    pattern_filter = species.parse_genotype_pattern("{A1,A2}/B1|A3/B2; C1/C1")
    print("✓ Set pattern parsed: '{{A1,A2}}/B1|A3/B2; C1/C1'")
    
    gt1 = species.get_genotype_from_str("A1/B1|A3/B2; C1/C1")
    result = pattern_filter(gt1)
    print(f"✓ Set pattern matches A1: {result}")
    assert result == True, "Should match A1"
    
    gt2 = species.get_genotype_from_str("A2/B1|A3/B2; C1/C1")
    result = pattern_filter(gt2)
    print(f"✓ Set pattern matches A2: {result}")
    assert result == True, "Should match A2"
    
    gt3 = species.get_genotype_from_str("A3/B1|A1/B2; C1/C1")
    result = pattern_filter(gt3)
    print(f"✓ Set pattern does not match A3 in first position: {result}")
    assert result == False, "Should not match A3 in first position"
    
except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 4: Negation patterns
print("\n[Test 4] Negation patterns (!A)")
try:
    pattern_filter = species.parse_genotype_pattern("!A1/B1|A2/B2; C1/C1")
    print("✓ Negation pattern parsed: '!A1/B1|A2/B2; C1/C1'")
    
    gt1 = species.get_genotype_from_str("A2/B1|A2/B2; C1/C1")
    result = pattern_filter(gt1)
    print(f"✓ Negation pattern matches A2 (not A1): {result}")
    assert result == True, "Should match A2"
    
    gt2 = species.get_genotype_from_str("A1/B1|A2/B2; C1/C1")
    result = pattern_filter(gt2)
    print(f"✓ Negation pattern does not match A1: {result}")
    assert result == False, "Should not match A1"
    
except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 5: Unordered patterns (::)
print("\n[Test 5] Unordered patterns (::)")
try:
    pattern_filter = species.parse_genotype_pattern("A1::A2|B1/B2; C1/C1")
    print("✓ Unordered pattern parsed: 'A1::A2|B1/B2; C1/C1'")
    
    gt1 = species.get_genotype_from_str("A1/B1|A2/B2; C1/C1")
    result = pattern_filter(gt1)
    print(f"✓ Unordered pattern matches A1|A2: {result}")
    assert result == True, "Should match A1|A2"
    
    gt2 = species.get_genotype_from_str("A2/B1|A1/B2; C1/C1")
    result = pattern_filter(gt2)
    print(f"✓ Unordered pattern matches A2|A1 (reversed): {result}")
    assert result == True, "Should match A2|A1"
    
except Exception as e:
    print(f"✗ Test 5 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 6: filter_genotypes_by_pattern
print("\n[Test 6] Filtering genotype collections")
try:
    genotypes = [
        species.get_genotype_from_str("A1/B1|A2/B2; C1/C1"),
        species.get_genotype_from_str("A1/B2|A2/B1; C1/C2"),
        species.get_genotype_from_str("A2/B1|A3/B2; C2/C2"),
    ]
    
    matched = species.filter_genotypes_by_pattern(genotypes, "A1/*|A2/B2; C1/*")
    print(f"✓ Filtered genotypes: {len(matched)} matches out of {len(genotypes)}")
    assert len(matched) == 1, f"Expected 1 match, got {len(matched)}"
    
except Exception as e:
    print(f"✗ Test 6 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 7: enumerate_genotypes_matching_pattern
print("\n[Test 7] Enumerating matching genotypes")
try:
    count = 0
    for gt in species.enumerate_genotypes_matching_pattern("A1/B1|A2/B2; C1/C1", max_count=5):
        count += 1
        if count <= 3:
            print(f"  Generated genotype {count}: {gt}")
    
    print(f"✓ Enumerated {count} genotypes")
    
except Exception as e:
    print(f"✗ Test 7 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 8: Pattern with omitted chromosomes
print("\n[Test 8] Omitted chromosomes")
try:
    pattern_filter = species.parse_genotype_pattern("A1/B1|A2/B2")
    print("✓ Pattern with omitted chromosome parsed: 'A1/B1|A2/B2'")
    
    gt1 = species.get_genotype_from_str("A1/B1|A2/B2; C1/C1")
    result = pattern_filter(gt1)
    print(f"✓ Omitted chromosome matches any value: {result}")
    assert result == True, "Should match (Chr2 is omitted)"
    
    gt2 = species.get_genotype_from_str("A1/B1|A2/B2; C2/C2")
    result = pattern_filter(gt2)
    print(f"✓ Omitted chromosome matches different values: {result}")
    assert result == True, "Should match (Chr2 is omitted)"
    
except Exception as e:
    print(f"✗ Test 8 failed: {e}")
    import traceback
    traceback.print_exc()


# Test 9: Integration with GameteConversionRuleSet
print("\n[Test 9] Integration with GameteConversionRuleSet")
try:
    from natal.gamete_allele_conversion import GameteConversionRuleSet, GameteAlleleConversionRule
    
    ruleset = GameteConversionRuleSet("TestRuleset")
    
    # Add a rule with genotype filter
    pattern_filter = species.parse_genotype_pattern("A1/B1|A2/B2")
    ruleset.add_rule(
        GameteAlleleConversionRule(
            from_allele="A1",
            to_allele="A2",
            rate=0.5,
            genotype_filter=pattern_filter
        )
    )
    
    print(f"✓ Rule with pattern filter added: {ruleset}")
    
except Exception as e:
    print(f"✗ Test 9 failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
