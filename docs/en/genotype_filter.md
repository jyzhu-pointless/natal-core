# Designing Your Own Preset (2): Using genotype_filter to Control Rule Scope

After defining conversion rules, `genotype_filter` solves a key problem: **the same conversion rule should generally not apply to all genotypes**.

`genotype_filter` applies pattern matching results to the rule's scope, enabling precise control over rules.

## Understanding genotype_filter

`genotype_filter` is a function that takes a `Genotype` as input and returns `True` or `False`:

- Returns `True`: the rule applies to this genotype
- Returns `False`: the rule does not apply to this genotype

```python
def my_filter(genotype):
    return True  # or False
```

## Core Example: W->D Only in W::D Heterozygotes

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet


def is_wd_heterozygote(genotype) -> bool:
    name = str(genotype)
    return name in {"W|D", "D|W"}


ruleset = GameteConversionRuleSet("homing_drive")
ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    genotype_filter=is_wd_heterozygote,
)
```

This clearly defines the scope of the mechanism.

## Common Filtering Patterns

### Carrying a Specific Allele
Suitable for scenarios like "trigger whenever the drive allele is present".

### Specifying Heterozygous/Homozygous
Suitable for scenarios like "cleavage only in heterozygotes" or "effective only in homozygotes".

### Combinatorial Logic
Multiple filters can be combined with AND/OR/NOT logic to keep rules readable.

## Integration with Pattern Matching Syntax

When rule conditions are complex, it is recommended to reuse the pattern syntax from Chapter 13 rather than writing fragile string containment checks.

```python
def build_filter_from_pattern(species, pattern: str):
    return species.parse_genotype_pattern(pattern)


ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    genotype_filter=build_filter_from_pattern(
        population.species,
        "A1/B1|A2/B2; C1/D1|C2/D2",
    ),
)
```

Benefits of this approach:

1. Unified semantics: consistent with the pattern expansion rules from the Observation chapter
2. Maintainable: patterns can be placed directly in experimental configuration files
3. Testable: pattern-matched sets can be independently verified

## Practical Advice for Designing Filters

1. Filters should have a "single responsibility"
2. Start with the simplest readable version, then optimize for performance
3. Write unit tests for complex filters to avoid mis-screening
4. Record filter names and semantics in experiment logs

## Combining with Pattern Matching (Recommended Practice)

When the rule scope is complex, it is recommended to use the species' pattern parsing capability to generate `genotype_filter`, avoiding fragile string comparisons.

```python
class PatternBasedPreset(GeneticPreset):
    def __init__(self, pattern: str, conversion_rate: float = 0.95):
        super().__init__(name="PatternBasedPreset")
        self.pattern = pattern
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("PatternBased")
        pattern_filter = population.species.parse_genotype_pattern(self.pattern)

        ruleset.add_convert(
            from_allele="WT",
            to_allele="Drive",
            rate=self.conversion_rate,
            genotype_filter=pattern_filter,
        )
        return ruleset.to_gamete_modifier(population)
```

Practical advice:

1. Maintain pattern strings in configuration files
2. The Preset is responsible for compiling the pattern internally
3. Observation grouping should also use the same pattern or be expanded from the same pattern, ensuring statistical scope aligns with rule scope

## Conditional Mutation (Genotype-Dependent)

```python
class ConditionalMutation(GeneticPreset):
    """Conditional Mutation - only occurs in specific genetic backgrounds"""

    def __init__(self, target_allele: str = "B", required_background: str = "A"):
        super().__init__(name="ConditionalMutation")
        self.target_allele = target_allele
        self.required_background = required_background

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("ConditionalMutation")

        # Mutation only occurs when the background allele is present
        ruleset.add_convert(
            from_allele=self.target_allele,
            to_allele=f"{self.target_allele}_mutant",
            rate=1e-4,
            genotype_filter=lambda gt: self.required_background in str(gt)
        )

        return ruleset.to_gamete_modifier(population)
```

## Maintaining Consistency with Observation Statistics

It is recommended to use the same pattern for both:

1. The Preset's `genotype_filter` (determining who is affected by the rule)
2. Observation's `groups["genotype"]` (determining who is counted)

If different definitions are used on both sides, common symptoms include "the rule appears to take effect, but the observation metric does not change" or "observed changes are inconsistent with mechanism expectations."

## Debugging Methods

When the filter does not behave as expected, you can:

1. Print the filter's hit results
2. Check whether the pattern compiles correctly
3. Verify the string representation of genotypes
4. Compare expected and actual genotype sets

## Chapter Summary

Through `genotype_filter`, you can precisely control the scope of conversion rules. The next chapter will teach you how to encapsulate these rules into reusable Presets.
