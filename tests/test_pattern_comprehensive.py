"""Test pattern matching functionality for Genotype and HaploidGenome.

Verifies:
1. HaploidGenome pattern parsing and matching
2. Genotype pattern parsing and matching
3. Pattern enumeration capabilities
"""

from natal.genetic_patterns import GenotypePatternParser
from natal.genetic_structures import Species


def test_basic_haploid_genome_pattern():
    species = Species.from_dict("tpc_basic", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "Chr2": {"C": ["C1", "C2"]},
    })

    parser = GenotypePatternParser(species)
    pattern = parser.parse_haploid_genome_pattern("A1/B1; C1")
    filter_func = pattern.to_filter()

    haploid_strs = ["A1/B1; C1", "A1/B2; C1", "A2/B1; C1"]
    haploid_genomes = [species.get_haploid_genome_from_str(s) for s in haploid_strs]
    results = [hg for hg in haploid_genomes if filter_func(hg)]

    assert len(results) == 1, f"expected 1 match, got {len(results)}"
    assert results[0].name == "A1/B1;C1"


def test_set_pattern():
    species = Species.from_dict("tpc_set", {
        "Chr1": {"A": ["A1", "A2", "A3"], "B": ["B1", "B2"]},
    })

    filter_func = species.parse_haploid_genome_pattern("{A1,A2}/B1")

    test_cases = [
        ("A1/B1", True),
        ("A2/B1", True),
        ("A3/B1", False),
        ("A1/B2", False),
    ]
    for hap_str, expected in test_cases:
        hg = species.get_haploid_genome_from_str(hap_str)
        assert filter_func(hg) == expected, f"{hap_str}: expected {expected}"


def test_negation_pattern():
    species = Species.from_dict("tpc_negation", {
        "Chr1": {"A": ["A1", "A2", "A3"]},
    })

    filter_func = species.parse_haploid_genome_pattern("!A1")

    test_cases = [("A1", False), ("A2", True), ("A3", True)]
    for allele_str, expected in test_cases:
        hg = species.get_haploid_genome_from_str(allele_str)
        assert filter_func(hg) == expected, f"{allele_str}: expected {expected}"


def test_wildcard_pattern_basic():
    """Wildcard * matches any allele at a locus. Verify filtering behavior."""
    species = Species.from_dict("tpc_wildcard", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "Chr2": {"C": ["C1", "C2"]},
    })

    filter_func = species.parse_haploid_genome_pattern("A1/B1; C1")

    hg_match = species.get_haploid_genome_from_str("A1/B1; C1")
    hg_nomatch = species.get_haploid_genome_from_str("A2/B1; C1")

    assert filter_func(hg_match) is True
    assert filter_func(hg_nomatch) is False


def test_genotype_pattern_filter():
    species = Species.from_dict("tpc_genotype", {
        "Chr1": {"A": ["A1", "A2"]},
    })

    genotypes = [
        species.get_genotype_from_str("A1|A1"),
        species.get_genotype_from_str("A1|A2"),
        species.get_genotype_from_str("A2|A2"),
    ]

    filtered = species.filter_genotypes_by_pattern(genotypes, "A1|*")
    assert len(filtered) == 2, f"pattern 'A1|*': expected 2 matches, got {len(filtered)}"


def test_enumerate_haploid_genomes():
    species = Species.from_dict("tpc_haploid_enum", {
        "Chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
    })

    results = list(species.enumerate_haploid_genomes_matching_pattern("A1/B1", max_count=5))
    assert len(results) == 1, f"expected 1 haploid genome matching A1/B1, got {len(results)}"
    assert results[0].name == "A1/B1"


def test_enumerate_genotypes():
    species = Species.from_dict("tpc_enum", {
        "Chr1": {"A": ["A1", "A2"]},
    })

    results = list(species.enumerate_genotypes_matching_pattern("A1|A2", max_count=10))
    assert len(results) == 1, f"expected 1 matching genotype, got {len(results)}"
    assert results[0].name == "A1|A2"
