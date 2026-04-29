# Designing Your Own Preset (1): Starting with Allele Conversion Rules

The design process of a `GeneticPreset` begins with the clear expression of the genetic mechanism. For most drive systems, this step is typically embodied in the formulation of **allele conversion rules**.

## Defining the Mechanism Goal

Before writing any code, three key questions need to be clearly answered:

1. Which allele will be converted (`from_allele`)?
2. What will it be converted to (`to_allele`)?
3. What is the conversion probability (`rate`)?

For example, a minimal drive hypothesis can be stated as:

- During gamete production, `W -> D`, with probability `0.5`.

## Rule Objects and Rule Sets

NATAL provides two layers of structure to organize conversion rules:

- `GameteAlleleConversionRule`: A single conversion rule
- `GameteConversionRuleSet`: A collection of rules

This can be understood as:

- A Rule is "a sentence"
- A RuleSet is "a paragraph"

## Minimal Working Example

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

This example is already sufficient to describe a minimal conversion mechanism.

## Zygote Conversion Rules (Fertilized Egg Stage)

Allele conversion can also occur at the zygote (fertilized egg) stage, typically used to simulate the following mechanisms:

- **Gene drive repair**: Repair systems expressed in zygotes (e.g., Cas9 cleavage repair)
- **Allele-specific mortality**: Reduced viability of certain zygote genotypes
- **Post-meiotic conversion**: Allele conversion during development

### Key Differences from Gamete to Zygote

| Stage | Input | Mechanism | Applicable Scenario |
|-------|-------|-----------|---------------------|
| **Gamete** | Gamete (haploid) | Conversion during gametogenesis | Gamete drive systems |
| **Zygote** | Zygote (diploid) | Conversion immediately after fertilization | Zygote drive, zygote repair |

### Using ZygoteConversionRuleSet

```python
from natal.gamete_allele_conversion import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# In the zygote, if the A locus contains the D allele, convert W->D
def has_d_at_a(genotype) -> bool:
    # Pseudo-code, actual implementation depends on your Genotype structure
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

### Combined Use of Gamete + Zygote

Drive systems typically use both types of rules simultaneously:

```python
# Gamete stage: W -> D (biased)
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_convert("W", "D", rate=0.99)

# Zygote stage: achieve copying (ensure homozygosity)
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_convert(
    "W", "D",
    rate=0.95,
    genotype_filter=lambda g: "D" in str(g)
)

pop.add_gamete_modifier(gamete_ruleset.to_gamete_modifier(pop))
pop.add_zygote_modifier(zygote_ruleset.to_zygote_modifier(pop))
```

## Notes When Designing Rules

1. Start with one rule, do not write over a dozen at once
2. After adding each rule, run 20-50 steps to check if the direction matches expectations
3. Document the "biological hypothesis → parameter value" mapping to avoid later difficulty in explanation

## Basic Template

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
        """Define gamete-stage modification logic"""
        # Return GameteModifier or None
        return None

    def zygote_modifier(self, population) -> Optional[ZygoteModifier]:
        """Define zygote-stage modification logic"""
        # Return ZygoteModifier or None
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """Define fitness effects"""
        # Return fitness configuration dict or None
        return None
```

Implementation highlights:

1. **All methods are optional** - you can implement 1-3 methods
2. **At least implement one method** - otherwise the preset will have no effect
3. **Can return None** - indicating no modification is needed at that stage
4. **Supports deferred species binding** - `Species` can be unspecified at creation time

## Simple Examples

### Simple Point Mutation

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

### Bidirectional Mutation Balance

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

        # A → B (forward mutation)
        ruleset.add_convert("A", "B", rate=self.forward_rate)
        # B → A (back mutation)
        ruleset.add_convert("B", "A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(population)
```

## Chapter Summary

You have completed the first step of Preset design: defining allele conversion rules. The next chapter will cover how to control the scope of rule application.
