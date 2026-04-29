# Designing Your Own Preset

This section will guide you through designing, implementing, validating, and publishing custom Genetic Presets from scratch.

## 1. Start with Allele Conversion Rules

The design process of a `GeneticPreset` begins with a clear expression of the genetic mechanism. For most drive systems, this step is usually embodied in the formulation of **allele conversion rules**.

### Defining the Mechanism Goal

Before writing any code, you need to clearly answer three key questions:

1. Which allele will be converted (`from_allele`)?
2. Converted to what (`to_allele`)?
3. What is the conversion probability (`rate`)?

For example, a minimal drive hypothesis can be stated as:

- During gamete production, `W -> D`, with probability `0.5`.

### Rule Objects and Rule Sets

NATAL provides a two-layer structure for organizing conversion rules:

- `GameteAlleleConversionRule`: A single conversion rule
- `GameteConversionRuleSet`: A collection of rules

Think of it as:

- A Rule is "one sentence"
- A RuleSet is "one paragraph"

### Minimal Working Example

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

This example is sufficient to describe a minimal conversion mechanism.

### Zygote Conversion Rules (Fertilized Egg Stage)

Allele conversion can also occur at the zygote (fertilized egg) stage, typically used to simulate the following mechanisms:

- **Gene drive repair**: Repair systems expressed in the zygote (e.g., Cas9 cleavage repair)
- **Allele-specific mortality**: Reduced viability of certain zygotic genotypes
- **Post-meiotic conversion**: Allele conversion during development

#### Key Differences from Gamete to Zygote

| Stage | Input | Mechanism | Use Cases |
|-------|-------|-----------|-----------|
| **Gamete** | Gamete (haploid) | Conversion during gamete production | Gamete drive systems |
| **Zygote** | Zygote (diploid) | Conversion immediately after fertilization | Zygote drive, zygote repair |

#### Using ZygoteConversionRuleSet

```python
from natal.gamete_allele_conversion import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# In the zygote, if the A locus has the D allele, convert W->D
def has_d_at_a(genotype) -> bool:
    # Pseudo-code, depends on your Genotype structure
    return "D" in str(genotype)

ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    genotype_filter=has_d_at_a,
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

#### Combining Gamete + Zygote Usage

Drive systems typically use both types of rules simultaneously:

```python
# Gamete stage: W -> D (biased)
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_convert("W", "D", rate=0.99)

# Zygote stage: copy conversion (ensure homozygosity)
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_convert(
    "W", "D",
    rate=0.95,
    genotype_filter=lambda g: "D" in str(g)
)

pop.add_gamete_modifier(gamete_ruleset.to_gamete_modifier(pop))
pop.add_zygote_modifier(zygote_ruleset.to_zygote_modifier(pop))
```

### Notes on Designing Rules

1. Start with one rule, don't write a dozen at once
2. After adding each rule, run 20-50 steps to check if the direction matches expectations
3. Record the "biological hypothesis -> parameter value" mapping to avoid later confusion

### Basic Template

Before designing complex conversion rules, it is important to understand the basic template of `GeneticPreset`:

```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.modifiers import GameteModifier, ZygoteModifier
from typing import Optional

class MyCustomPreset(GeneticPreset):
    """Custom genetic modification preset"""

    def __init__(self, name: str = "MyCustom", species=None):
        super().__init__(name=name, species=species)
        # Custom parameters
        self.custom_param = 0.5

    def gamete_modifier(self, population) -> Optional[GameteModifier]:
        """Define modification logic at the gamete stage"""
        # Return GameteModifier or None
        return None

    def zygote_modifier(self, population) -> Optional[ZygoteModifier]:
        """Define modification logic at the zygote stage"""
        # Return ZygoteModifier or None
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """Define fitness effects"""
        # Return fitness configuration dictionary or None
        return None
```

Implementation notes:

1. **All methods are optional** - you can implement 1-3 methods
2. **At least one method must be implemented** - otherwise the preset will have no effect
3. **Can return None** - indicates no modification needed at that stage
4. **Supports deferred species binding** - can create without specifying `Species`

### Simple Examples

#### Simple Point Mutation

```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.gamete_allele_conversion import GameteConversionRuleSet

class PointMutation(GeneticPreset):
    """Simple point mutation: WT mutates to Mutant at a certain frequency"""

    def __init__(self, mutation_rate: float = 1e-5):
        super().__init__(name="PointMutation")
        self.mutation_rate = mutation_rate

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("PointMutation")
        ruleset.add_convert("WT", "Mutant", rate=self.mutation_rate)
        return ruleset.to_gamete_modifier(population)

    def fitness_patch(self):
        return {
            "viability_allele": {"Mutant": 0.98}  # Slightly deleterious
        }
```

#### Bidirectional Mutation Balance

```python
class BidirectionalMutation(GeneticPreset):
    """Bidirectional mutation balance"""

    def __init__(self, forward_rate: float = 1e-5, backward_rate: float = 1e-6):
        super().__init__(name="BidirectionalMutation")
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("BidirectionalMutation")

        # A -> B (forward mutation)
        ruleset.add_convert("A", "B", rate=self.forward_rate)
        # B -> A (back mutation)
        ruleset.add_convert("B", "A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(population)
```

## 2. Using genotype_filter to Control Rule Scope

After defining conversion rules, `genotype_filter` is used to solve a key problem: **the same conversion rule typically should not apply to all genotypes**.

`genotype_filter` applies the results of pattern matching to the rule's scope, enabling precise control of the rule.

### Understanding genotype_filter

`genotype_filter` is a function that takes a `Genotype` as input and returns `True` or `False`:

- Returns `True`: the rule applies to this genotype
- Returns `False`: the rule does not apply to this genotype

```python
def my_filter(genotype):
    return True  # or False
```

### Core Example: W->D Only in W::D Heterozygotes

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

### Common Filtering Patterns

- **Carries a certain allele**: Suitable for "trigger if carrying drive" scenarios.
- **Specific heterozygote/homozygote**: Suitable for "cleave only in heterozygotes" or "apply only in homozygotes" scenarios.
- **Combined logic**: Multiple filters can be combined with AND/OR/NOT logic while maintaining readability.

### Integration with Pattern Matching Syntax

When rule conditions are complex, it is recommended to reuse the pattern syntax from the documentation rather than writing manual string containment checks.

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

1. Semantic consistency: aligns with the pattern expansion rules in the Observation section
2. Maintainability: patterns can be placed directly in experiment configuration files
3. Testability: the pattern's matching set can be independently verified

### Practical Advice for Designing Filters

1. Filters should have "single responsibility"
2. First write the simplest readable version, then optimize for performance
3. Write unit tests for complex filters to avoid false matches
4. Record filter names and semantics in experiment logs

### Integration with Pattern Matching (Recommended Approach)

When the rule scope is complex, it is recommended to use the species' pattern resolution capability to generate `genotype_filter`, avoiding fragile string checks.

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
3. Observation groups should also use the same pattern or be expanded from the same pattern, ensuring statistical and rule scopes are aligned

### Conditional Mutation (Genotype-Dependent)

```python
class ConditionalMutation(GeneticPreset):
    """Conditional mutation - only occurs in specific genetic backgrounds"""

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

### Maintaining Statistical Consistency with Observations

It is recommended to use the same pattern for both:

1. The Preset's `genotype_filter` (determining who is affected by the rule)
2. The Observation's `groups["genotype"]` (determining who is counted)

If different definitions are used on both sides, common symptoms are "the rule appears to be working, but the observed metrics don't change" or "observation changes don't match mechanism expectations."

### Debugging Methods

When the filter's effect is not as expected, you can:

1. Print the filter's matching results
2. Check if the pattern was compiled correctly
3. Verify the genotype string representation
4. Compare expected and actual genotype sets

## 3. Encapsulation, Validation, and Pre-Release Checks

This chapter is the final part of the "Design Your Own Preset" main line. In the previous two chapters, you completed:

1. Rule definition (Gamete and Zygote conversion)
2. Fine-grained control of rule scope

This chapter teaches you how to encapsulate these into **reusable Presets**, perform thorough validation, and finally release them for use.

### Value of Encapsulation as a Preset

If you only write rules in scripts, you will encounter three problems later:

1. Hard to reuse: every experiment requires copying logic
2. Hard to trace: difficult to tell "which set of rules this version used"
3. Hard to maintain: rules, fitness, and hooks are scattered across multiple files

The value of a Preset is to consolidate these into a stable configuration unit.

### Recommended Preset Structure

A practical Preset should include:

1. Mechanism rules (conversion rules and filters)
2. Fitness patch (if needed)
3. Optional parameters (e.g., conversion rate, sex limitations)
4. Clear name and version marker

### Example: Encapsulating a Minimal DrivePreset

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet


class DrivePreset(GeneticPreset):
    def __init__(self, conversion_rate: float = 0.5):
        super().__init__(name="DrivePreset")
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("drive_rules")

        def is_wd_heterozygote(genotype) -> bool:
            name = str(genotype)
            return name in {"W|D", "D|W"}

        ruleset.add_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            genotype_filter=is_wd_heterozygote,
        )

        return ruleset.to_gamete_modifier(population)
```

### Applying Presets in the Builder

```python
pop = (
    AgeStructuredPopulationBuilder(species)
    .setup(name="DriveExperiment", stochastic=True)
    .age_structure(n_ages=8)
    .initial_state({...})
    .presets(DrivePreset(conversion_rate=0.55))
    .build()
)
```

This is the most recommended way to integrate "Preset as a configuration component."

### Validation Checklist (Highly Recommended)

Before conducting large-scale experiments, at least complete the following checks:

1. Mechanism check: Are the conversion direction and target allele correct?
2. Filter check: Does the `genotype_filter` hit range match expectations?
3. Conservation check: Is frequency normalization valid?
4. Control check: Is the trend reasonable compared to a baseline without Preset?
5. Stability check: Are conclusions robust when changing random seeds?

### Experiment Recording Advice

It is recommended to write Preset configuration into experiment metadata:

- Preset name
- Key parameters (e.g., `conversion_rate`)
- Code version or commit
- Random seed

This significantly reduces the risk of "results cannot be reproduced."

### Complex Gene Drive Example

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.zygote_allele_conversion import ZygoteConversionRuleSet

class ComplexDrive(GeneticPreset):
    """Complex gene drive with multi-stage conversion"""

    def __init__(self):
        super().__init__(name="ComplexDrive")

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("ComplexDrive")

        # Stage 1: Drive conversion (WT -> Drive)
        ruleset.add_convert("WT", "Drive", rate=0.95,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        # Stage 2: Resistance formation (remaining WT -> Resistance)
        ruleset.add_convert("WT", "Resistance", rate=0.05,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        return ruleset.to_gamete_modifier(population)

    def zygote_modifier(self, population):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # Additional modification at the embryo stage
        ruleset.add_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            maternal_glab="cas9"  # Requires maternal Cas9 deposition
        )

        return ruleset.to_zygote_modifier(population)

    def fitness_patch(self):
        return {
            "viability_per_allele": {
                "Drive": 0.9,      # Drive allele cost
                "Resistance": 1.0   # Resistance allele neutral
            },
            "fecundity_per_allele": {
                "Drive": 0.95
            },
            "zygote_per_allele": {
                "Drive": 0.8,     # Reduced zygote survival rate
                "Resistance": 1.0   # Resistance allele neutral
            }
        }
```

### Common Errors and Debugging

#### Parameter Validation Errors
- Verify conversion rate is in range [0, 1]

#### Species Binding Errors
- Ensure the preset and population use the same species
- Use deferred binding (create without specifying `Species`)

#### Performance Issues
- Avoid creating many temporary objects in modifiers
- Use rule set caching
- Consider simplifying complex rule chains

#### Debugging Techniques

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, population):
        print(f"Applying preset to species: {population.species.name}")
        print(f"Available alleles: {list(population.species.gene_index.keys())}")

        # Create modifier and return
        # ...
```

### Pre-Release Checklist

Before releasing a Preset, it is recommended to:

- [ ] Unit tests covering main functionality
- [ ] Clear and complete documentation
- [ ] Parameter range validation passed
- [ ] Compatibility testing with existing systems
- [ ] Performance benchmark testing

### Chapter Summary

Congratulations! You have completed the full "Design Your Own Preset" main line:

1. Rule definition (Gamete and Zygote conversion)
2. Fine-grained rule scope control (genotype_filter)
3. Preset engineering, validation, and release

You now have mastered the complete workflow of designing, implementing, validating, and publishing custom Presets from scratch.
