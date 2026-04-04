# Designing Your Own Preset (2): Using `genotype_filter` to Control Rule Scope

In the previous chapter you defined conversion rules. This chapter continues the main line and solves a key problem:

**A single conversion rule should usually not apply to all genotypes.**

This is the role of `genotype_filter` – applying the results of pattern matching to the rule’s scope.

## 1. What is `genotype_filter`

`genotype_filter` is a function that:

- Input: `Genotype`
- Output: `True` or `False`

When it returns `True`, the rule applies to that genotype; when it returns `False`, it does not.

```python
def my_filter(genotype):
    return True  # or False
```

## 2. Main Line Example: `W->D` Only in `W|D` Heterozygotes

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

This makes the “mechanism scope” explicit.

## 3. Common Filtering Patterns

### 3.1 Carries a certain allele

Suitable for scenarios where “any individual carrying the drive triggers the effect”.

### 3.2 Specify heterozygote / homozygote

Suitable for scenarios where “cutting only in heterozygotes” or “effect only in homozygotes”.

### 3.3 Combining logic

You can combine multiple filters with AND/OR/NOT logic to keep rules readable.

## 4. Integrating with Pattern Matching Syntax (Chapter 13)

When the rule condition is complex, it is recommended to reuse the pattern syntax from Chapter 13 instead of writing hand‑crafted string‑containment checks.

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

Advantages:

1. Semantic unification: consistent with the pattern expansion rules from the Observation chapter.
2. Maintainability: patterns can be placed directly into experiment configuration files.
3. Testability: the pattern’s matched set can be verified independently.

## 5. Practical Advice for Designing Filters

1. Filters should have a single responsibility.
2. Start with the simplest readable version, then optimise for performance later.
3. Write unit tests for complex filters to avoid incorrect filtering.
4. Record the filter name and semantics in experiment logs.

## 6. Keeping Statistical Consistency with Observation

It is recommended to use the same pattern for both:

1. The Preset’s `genotype_filter` (which determines who is affected by the rule).
2. The Observation’s `groups["genotype"]` (which determines who is counted).

If different definitions are used on both sides, common symptoms are “the rule seems to work, but the observed metrics do not move” or “observed changes do not match the expected mechanism”.

## 7. Debugging Methods

It is recommended to check the “hit rate” on a small sample:

1. Enumerate the genotypes of interest.
2. For each genotype, print the result of `genotype_filter`.
3. Verify that it matches the biological expectation.

This step often prevents many invalid simulations in advance.

## 8. Making Rules More Maintainable

When the number of rules grows, split filters by semantics:

- `is_drive_carrier`
- `is_target_heterozygote`
- `is_male_specific_target`

Then combine them to build the final filter, rather than writing one huge function.

## 9. Chapter Summary

You have completed the core two steps of Preset design:

1. Define the rules (Chapter 1).
2. Refine the rule scope (this chapter).

The next chapter will engineer these concepts and explain how to encapsulate rules and filters into reusable Preset classes.

---

## Next Chapters

- [Designing Your Own Preset (3): Encapsulation, Validation, and Pre‑release Checks](preset_encapsulation_and_validation.md)
- [Genotype Pattern Matching](genotype_pattern_matching_design.md)
- [Population Observation Rules](observation_rules.md)
