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
2. **`HaploidGenomePattern`**: Pattern matching for haploid genotypes

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

### Label Matching (@lab)

A `@` suffix attaches a gamete-label (`glab`) or somatic-label (`slab`)
constraint to a selector. Pattern parsing has two semantic entries and only
those two: `parse_selector` for matching and `parse_target` for
keep-or-replace conversion (`natal.parse_selector` / `natal.parse_target`).
Content patterns carry no label: `parse_selector(..., kind="genotype")` and
`kind="haploid"`, plus the `Species` helpers built on them
(`parse_genotype_pattern`, `enumerate_genotypes_matching_pattern`,
`parse_haploid_genome_pattern`, `enumerate_haploid_genomes_matching_pattern`),
reject a labelled pattern outright with a `PatternParseError` rather than
accepting one and ignoring it. Labels belong to the label-aware kinds —
`kind="ztype"` composes a `ZygoteTypePattern` (genotype + slab) and
`kind="gtype"` a `GameteTypePattern` (haploid genome + glab) — and to the
conversion-rule filters and `IndividualSelector(ztype=...)`.
On an unordered species a selector written with `|` promotes every `|` to
`::` before parsing, so one spelling matches through fitness, rules,
observation and hooks alike; the content helpers above keep `|` strictly
ordered, and targets are never promoted.
Label syntax mirrors allele syntax:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `@X` | Exact label match | `A\|a@Cas9_high` |
| `@!X` | Exclude label X | `A\|a@!wildtype` |
| `@{A,B}` | Any label in set | `A\|a@{high,low}` |
| `@!{A,B}` | Exclude labels in set | `*|*@!{wildtype,default}` |
| `@*` | Any label (same as omitting @) | `A\|a@*` |

A somatic label rides `kind="ztype"`; a gamete label rides `kind="gtype"`:

```python
nt.parse_selector("A|a@Cas9_high", species=species)                 # somatic label
nt.parse_selector("A@Cas9_deposited", species=species, kind="gtype")  # gamete label
```

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

- **`|` (single pipe)**: Strictly ordered — `Dr|WT` matches only the literal `Dr|WT` phase, regardless of the Species' `unordered` setting. With the default `unordered=True` species (canonical heterozygote `WT|Dr`), the pattern `Dr|WT` matches nothing.
- **`::` (double colon)**: Unordered matching — the two homologous chromosome copies can be swapped, regardless of the Species setting.

```python
# | syntax: strictly ordered — matches only this exact maternal/paternal phase
pattern1 = "A1/B1|A2/B2"

# :: syntax: unordered — homologous chromosomes can swap
pattern2 = "A1/B1::A2/B2"
```

The `unordered`-based tolerance does exist, but at the **selector layer**, not
in the pattern parser: when the species has `unordered=True`,
`IndividualSelector(ztype=...)` and `Species.resolve_single_genotype_selector()`
rewrite `|` to `::` before parsing, so selector strings match either phase.

## HaploidGenomePattern: Haploid Genotype Matching

### Basic Syntax

`HaploidGenomePattern` is used to match haploid genotypes, with a simpler syntax:

`<chr1_hap>/<...>; <chr2_hap>/<...>`

### Combination Examples

1. **Exact match**: `A1/B1; C1/D1`
2. **Mixed wildcards**: `A1/*; */D1`
3. **Set matching**: `{A1,A2}/B1; C1/D1`
4. **Exclusion matching**: `!A1/B1; C1/D1`

### Usage Examples

```python
# Haploid genome pattern matching.
# Species.parse_haploid_genome_pattern() returns a filter callable,
# not the HaploidGenomePattern object itself.
pattern = sp.parse_haploid_genome_pattern("A1/*; C1")

# Filter matching haploid genotypes
matching_haploids = [hg for hg in all_haploids if pattern(hg)]

# Or use the enumeration method
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=10):
    print(f"Matching haploid genotype: {hg}")
```

## Advanced Syntax Features

### Parentheses: Internal Separation Within a Chromosome Pair

Parentheses `(...)` group per-locus diploid conditions on **one chromosome pair**. Semicolons inside the parentheses separate loci; semicolons outside separate chromosome pairs. Within the group, each `|` checks maternal/paternal order at that locus, while each `::` allows either order at that locus independently. This does not mean reversing the whole chromosome haplotype.

```python
import natal as nt

single = nt.Species.from_dict(
    name="PatternSingleChromosome",
    structure={"chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]}},
    unordered=False,
)
pattern1 = single.parse_genotype_pattern("(A1|A2;B1::B2)")
assert pattern1(single.get_genotype_from_str("A1/B1|A2/B2"))
assert pattern1(single.get_genotype_from_str("A1/B2|A2/B1"))
assert not pattern1(single.get_genotype_from_str("A2/B1|A1/B2"))

multiple = nt.Species.from_dict(
    name="PatternTwoChromosomes",
    structure={
        "chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "chr2": {"C": ["C1", "C2"]},
    },
    unordered=False,
)
# A and B are on chr1; C is on chr2.
pattern2 = multiple.parse_genotype_pattern("(A1|A2;B1::B2);C1|C2")
assert pattern2(multiple.get_genotype_from_str("A1/B2|A2/B1;C1|C2"))
assert not pattern2(multiple.get_genotype_from_str("A1/B2|A2/B1;C2|C1"))
```

For three loci on the same chromosome, write `(A1|A2;B1::B2;C1|C2)` instead. These per-locus diploid conditions do not apply to haploid patterns, which use `/` between loci.


## Common Errors and Corrections

### General Errors

1. **Error**: Mismatch in the number of chromosome segments
   - **Cause**: The parser counts one segment per autosome plus one per sex-chromosome group (not per individual sex chromosome). More segments than the species' groups raises an error; see the per-sex-group string form in [Genetics](2_genetics.md#sex-chromosome-string-format)
   - **Correction**: Write one segment per autosome and one per sex-chromosome group, following the species definition

2. **Error**: Mismatch in the number of loci
   - **Cause**: The number of locus patterns separated by `/` does not match the locus count on that chromosome
   - **Correction**: Complete locus by locus, or use `*` as a placeholder

### GenotypePattern-Specific Errors

1. **Error**: `Chromosome pattern must contain '|' or '::'`
   - **Cause**: A chromosome segment is missing the homologous chromosome dual-copy delimiter
   - **Correction**: Do not write `C1/C1`; change to the full `...|...` or `...::...` form

## Application Integration

### Integrating with Observations

Each value of `with_observation(groups=...)` in the Observation section must be an `IndividualSelector`, whose `ztype` field supports `GenotypePattern` parsing:

```python
import natal as nt

groups = {
    "target_group": nt.IndividualSelector(
        # Ordered matching: Maternal|Paternal
        ztype="A1/B1|A2/B2; C1/D1|C2/D2",
        sex="female",
    ),
    "target_group_unordered": nt.IndividualSelector(
        # Unordered matching: two homologous chromosome copies can be swapped
        ztype="A1/B1::A2/B2; C1/D1::C2/D2",
        sex="female",
    ),
}

# Pass it at build time: .with_observation(groups)
```

### Integrating with Presets

Keep the pattern string in the preset and pass it through `filters`; the rule compiler performs parsing:

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet

class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def zygote_modifier(self, host):
        return None

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("pattern_rules")

        ruleset.add_allele_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            filters={"parent": self.target_pattern},
        )
        return ruleset.to_gamete_modifier(host)
```

## Debugging and Validation

To debug the match set, you can use the following methods for offline expansion checking:

```python
# Check GenotypePattern match results
for gt in sp.enumerate_genotypes_matching_pattern("A1/*|A2/B2", max_count=5):
    print(f"Matching genotype: {gt}")

# Check HaploidGenomePattern match results
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=5):
    print(f"Matching haploid genotype: {hg}")
```

---

## Related Sections

- [Population Observation Rules](2_data_output.md)
- [Design Your Own Presets](3_custom_presets.md)
- [Genetic Presets Usage Guide](2_genetic_presets.md)
