# Genotype Pattern Matching

This chapter provides a detailed introduction to NATAL's pattern matching mechanism, which allows users to describe and batch-filter genotypes using formatted strings. Pattern matching is a natural extension of the precise genotype string format, supporting flexible pattern description for both diploid genotypes (`Genotype`) and haploid genotypes (`HaploidGenotype`).

## Overview

### Why Pattern Matching Is Needed

When the genetic model reaches the following levels of complexity, hardcoding genotype lists becomes difficult to maintain:

- Multiple chromosomes, multiple loci
- Large numbers of allele combinations
- Need to define rules or observation groups by "certain classes of genotypes" in batches

Pattern matching upgrades explicit enumeration of genotype lists to **semantic expressions**, avoiding verbose enumeration while providing a more intuitive and readable representation.

### Supported Match Types

NATAL supports two types of pattern matching:

1. **`GenotypePattern`**: Pattern matching for diploid genotypes
2. **`HaploidGenotypePattern`**: Pattern matching for haploid genotypes

Both pattern types share the same syntax fundamentals but differ in how they handle the chromosome layer.

## Syntax Fundamentals

### Basic Structure

Pattern strings are parsed in three layers, from outer to inner:

1. **Chromosome layer**: Multiple chromosome segments are separated by `;`
2. **Homologous chromosome layer**: Each segment must contain `|` or `::` (for `GenotypePattern` only)
3. **Locus layer**: Within each homologous chromosome, locus patterns are separated by `/`

### Delimiter Meanings

| Syntax Element | Meaning | Applicable Pattern | Example |
|----------------|---------|--------------------|---------|
| `;` | Separates different chromosome segments | Both | `A/B|C/D; E/F|G/H` |
| `|` | Ordered matching: `Maternal|Paternal` | GenotypePattern | `A/B|C/D` |
| `::` | Unordered matching: homologous chromosomes can be swapped | GenotypePattern | `A/B::C/D` |
| `/` | Separates loci within a single chromosome | Both | `A/B/C` |

### Locus Atomic Patterns

| Pattern | Meaning | Example |
|---------|---------|---------|
| `X` | Exact match for allele `X` | `A1` |
| `*` | Wildcard, matches any allele | `*` |
| `{A,B,C}` | Matches any element in the enumerated set | `{A1,A2}` |
| `!X` | Excludes `X`, matches any other allele | `!A1` |

## GenotypePattern: Diploid Genotype Matching

### Basic Syntax

`GenotypePattern` is used to match diploid genotypes. Its basic syntax is the same as the precise string format:

`<chr1_hap1>/<...>|<chr1_hap2>/<...>; <chr2_hap1>/<...>|<chr2_hap2>/<...>`

### Combination Examples

1. **Exact match**: `A1/B1|A2/B2; C1/D1|C2/D2`
2. **Mixed wildcards**: `A1/*|A2/B2; */D1|C2/*`
3. **Set matching**: `{A1,A2}/B1|A3/B2; C1/D1|C2/D2`
4. **Unordered matching**: `A1/B1::A2/B2; C1/D1::C2/D2`

### Ordered vs Unordered Matching

- **`|` (ordered)**: Strictly distinguishes maternal and paternal order
- **`::` (unordered)**: The two homologous chromosome copies can be swapped

```python
# Ordered matching: Maternal|Paternal strictly distinguished
pattern1 = "A1/B1|A2/B2"

# Unordered matching: homologous chromosomes can be swapped
pattern2 = "A1/B1::A2/B2"
```

## HaploidGenotypePattern: Haploid Genotype Matching

### Basic Syntax

`HaploidGenotypePattern` is used to match haploid genotypes, with a simpler syntax:

`<chr1_hap>/<...>; <chr2_hap>/<...>`

### Combination Examples

1. **Exact match**: `A1/B1; C1/D1`
2. **Mixed wildcards**: `A1/*; */D1`
3. **Set matching**: `{A1,A2}/B1; C1/D1`
4. **Exclusion matching**: `!A1/B1; C1/D1`

### Usage Examples

```python
# Haploid genotype pattern matching
pattern = sp.parse_haploid_genome_pattern("A1/*; C1")

# Filter matching haploid genotypes
matching_haploids = [hg for hg in all_haploids if pattern(hg)]

# Or use the enumeration method
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=10):
    print(f"Matching haploid genotype: {hg}")
```

## Advanced Syntax Features

### Parentheses: Internal Separation Within a Chromosome Pair

Parentheses `(...)` allow further separation within a chromosome pair using `;`, particularly useful in complex scenarios mixing ordered and unordered matching:

```python
# Internal separation within a chromosome pair: locus A ordered matching, locus B unordered matching
pattern1 = "(A1|A2);(B1::B2)"
# Equivalent to: locus A must strictly follow maternal|paternal order, locus B can be swapped

# Mixed ordered and unordered matching
pattern2 = "(A1|A2);(B1::B2);(C1|C2)"
# Loci A and C use ordered matching, locus B uses unordered matching

# Complex nested patterns
pattern3 = "(A1/{B1,B2}|A2/{B1,B2});(C1::C2)"
```

Parentheses syntax is applicable to both haploid and diploid patterns and can significantly improve the readability and maintainability of complex patterns.

## Common Errors and Corrections

### General Errors

1. **Error**: Mismatch in the number of chromosome segments
   - **Cause**: The number of `;`-separated segments does not match the species' chromosome count
   - **Correction**: Complete the chromosome segments one by one according to the species definition

2. **Error**: Mismatch in the number of loci
   - **Cause**: The number of locus patterns separated by `/` does not match the locus count on that chromosome
   - **Correction**: Complete locus by locus, or use `*` as a placeholder

### GenotypePattern-Specific Errors

1. **Error**: `Chromosome pattern must contain '|' or '::'`
   - **Cause**: A chromosome segment is missing the homologous chromosome dual-copy delimiter
   - **Correction**: Do not write `C1/C1`; change to the full `...|...` or `...::...` form

## Application Integration

### Integrating with Observations

The `groups["genotype"]` in the Observation section supports `GenotypePattern` parsing:

```python
groups = {
    "target_group": {
        # Ordered matching: Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_group_unordered": {
        # Unordered matching: two homologous chromosome copies can be swapped
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}
```

### Integrating with Presets

It is recommended to encapsulate the pattern parsing logic within the Preset:

```python
class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def _build_filter(self, species):
        return species.parse_genotype_pattern(self.target_pattern)

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("pattern_rules")
        pattern_filter = self._build_filter(population.species)

        ruleset.add_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            genotype_filter=pattern_filter,
        )
        return ruleset.to_gamete_modifier(population)
```

## Debugging and Validation

To debug the match set, you can use the following methods for offline expansion checking:

```python
# Check GenotypePattern match results
for gt in sp.enumerate_genotypes_matching_pattern("A1/*|A2/B2", max_count=5):
    print(f"Matching genotype: {gt}")

# Check HaploidGenotypePattern match results
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=5):
    print(f"Matching haploid genotype: {hg}")
```

---

## Related Sections

- [Population Observation Rules](2_data_output.md)
- [Design Your Own Presets](3_custom_presets.md)
- [Genetic Presets Usage Guide](2_genetic_presets.md)
