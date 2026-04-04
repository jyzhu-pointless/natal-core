# Genetic Presets Guide

This document describes how to use and create genetic presets, including gene drives, mutation systems, and other genetic modifications.

## Overview

**Genetic Presets** are a mechanism in the NATAL framework for defining reusable genetic modifications. Presets can:

- Modify gamete production rules (e.g., meiotic drive)
- Alter zygote development processes (e.g., embryonic resistance formation)
- Adjust fitness parameters (e.g., cost of a drive allele)

## Applying Presets in the Builder

```python
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="TestPop")
       .presets(preset1, preset2)  # multiple presets can be applied
       .build())
```

## Built‑in Presets

### HomingDrive – Homing‑Based Gene Drive

Implements a CRISPR/Cas9‑type homing gene drive:

```python
from natal.genetic_presets import HomingDrive

# Create a basic gene drive
drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,   # 95% conversion efficiency
    late_germline_resistance_formation_rate=0.03   # 3% resistance formation
)

# Apply to a population
population.apply_preset(drive)
```

#### Advanced Configuration

```python
# Sex‑specific parameters
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # sex‑specific
    late_germline_resistance_formation_rate=(0.02, 0.04),  # tuple (female, male)
    embryo_resistance_formation_rate=0.01,
    functional_resistance_ratio=0.2,   # 20% of resistance alleles are functional

    # Fitness costs
    viability_scaling=0.9,      # 10% viability cost
    fecundity_scaling=0.95,     # 5% fecundity cost
    sexual_selection_scaling=0.85   # 15% mating disadvantage
)
```

### ToxinAntidoteDrive – Toxin‑Antidote Drive (TARE/TADE)

`ToxinAntidoteDrive` models systems where “the drive allele triggers disruption at the target locus, the disrupted allele causes a fitness loss, and the drive allele provides rescue”.

```python
from natal.genetic_presets import ToxinAntidoteDrive

ta_drive = ToxinAntidoteDrive(
    name="TARE_Drive",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    conversion_rate=0.95,
    embryo_disruption_rate={"female": 0.30, "male": 0.0},
    viability_scaling=0.0,
    fecundity_scaling=1.0,
    viability_mode="recessive",
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9",
)

population.apply_preset(ta_drive)
```

Parameter explanations:

1. `conversion_rate`: probability of `target -> disrupted` conversion in the germline. Supports a `float`, a tuple `(female, male)`, or a sex‑specific dictionary.
2. `embryo_disruption_rate`: conversion probability at the embryonic stage. Can be combined with `cas9_deposition_glab` / `use_paternal_deposition` to model maternal/paternal deposition effects.
   - If `cas9_deposition_glab` is set, ensure that the species of the population has registered that label via `gamete_labels` when created; otherwise applying the preset will raise a `KeyError`.
3. `viability_scaling` and `viability_mode`: define the toxin effect of the `disrupted` allele. For TARE, typical values are `viability_scaling=0.0` and `viability_mode="recessive"`.
4. `fecundity_scaling` and `fecundity_mode`: define fecundity costs.
5. `sexual_selection_scaling` (optional): define sexual selection effects. Supports a scalar or a tuple `(default_male, carrier_male)`, used together with `sexual_selection_mode`.

Example: adding a mating cost

```python
ta_drive_with_mating_cost = ToxinAntidoteDrive(
    name="TA_WithMatingCost",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    sexual_selection_scaling=(1.0, 0.8),
    sexual_selection_mode="dominant",
)
```

## Creating Custom Presets

### Combine with Pattern Matching (strongly recommended)

When the rule scope is complex, it is advisable to avoid fragile string checks like `lambda gt: "X" in str(gt)`. Instead, use the pattern parsing capabilities provided by the species to generate a `genotype_filter`.

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

1. Maintain pattern strings in configuration files.
2. The preset is responsible for compiling the pattern.
3. Use the same pattern (or an expansion of it) for observation groups, ensuring that the statistical and rule scopes are consistent.

### Basic Template

All custom presets should inherit from `GeneticPreset`:

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
        """Define gamete‑stage modification logic"""
        # Return a GameteModifier or None
        return None

    def zygote_modifier(self, population) -> Optional[ZygoteModifier]:
        """Define zygote‑stage modification logic"""
        # Return a ZygoteModifier or None
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """Define fitness effects"""
        # Return a fitness configuration dictionary or None
        return None
```

### Implementation Points

1. **All methods are optional** – you can implement 1‑3 methods.
2. **Implement at least one method** – otherwise the preset will have no effect.
3. **Can return `None`** – indicates that no modification is needed at that stage.
4. **Supports delayed species binding** – you can create a preset without specifying a `Species`.

## Practical Examples

### Example 1: Simple Point Mutation

```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.population_builder import AgeStructuredPopulationBuilder

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
            "viability_allele": {"Mutant": 0.98}   # slightly deleterious
        }

# Usage example
species = Species.from_dict("TestSpecies", {
    "chr1": {"GeneA": ["WT", "Mutant"]}
})

mutation = PointMutation(mutation_rate=1e-5)

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="MutationTest", stochastic=False)
       .age_structure(n_ages=5)
       .initial_state({"female": {"WT|WT": [0, 100, 100, 100, 100]}})
       .presets(mutation)
       .build())
```

### Example 2: Bidirectional Mutation Balance

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
        # B → A (reverse mutation)
        ruleset.add_convert("B", "A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(population)
```

### Example 3: Conditional Mutation (Genotype‑dependent)

```python
class ConditionalMutation(GeneticPreset):
    """Conditional mutation – only occurs in a specific genetic background"""

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

### Example 4: Complex Gene Drive

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.zygote_allele_conversion import ZygoteConversionRuleSet
from natal.population_builder import AgeStructuredPopulationBuilder

class ComplexDrive(GeneticPreset):
    """Complex gene drive with multiple stages of conversion"""

    def __init__(self):
        super().__init__(name="ComplexDrive")

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("ComplexDrive")

        # Stage 1: drive conversion (WT → Drive)
        ruleset.add_convert("WT", "Drive", rate=0.95,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        # Stage 2: resistance formation (remaining WT → Resistance)
        ruleset.add_convert("WT", "Resistance", rate=0.05,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        return ruleset.to_gamete_modifier(population)

    def zygote_modifier(self, population):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # Additional embryonic‑stage modification
        ruleset.add_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            maternal_glab="cas9"   # requires maternal Cas9 deposition
        )

        return ruleset.to_zygote_modifier(population)

    def fitness_patch(self):
        return {
            "viability_allele": {
                "Drive": 0.9,      # cost of the drive allele
                "Resistance": 1.0   # resistance allele is neutral
            },
            "fecundity_allele": {
                "Drive": 0.95
            }
        }

# Usage example
species = Species.from_dict("DriveSpecies", {
    "chr1": {"drive_locus": ["WT", "Drive", "Resistance"]}
})

complex_drive = ComplexDrive()

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="ComplexDrivePop", stochastic=False)
       .age_structure(n_ages=8)
       .initial_state({
           "female": {"WT|WT": [0, 500, 500, 400, 300, 200, 100, 50]},
           "male": {"WT|WT": [0, 250, 250, 200, 150, 100, 50, 25]}
       })
       .presets(complex_drive)
       .build())
```

## Detailed Fitness Configuration

### Types of Fitness Effects

```python
def fitness_patch(self):
    return {
        # 1. Genotype‑specific fitness
        "viability": {
            "Drive|Drive": 0.8,      # specific genotype
            "Drive|WT": 0.9,
            "WT|WT": 1.0
        },

        # 2. Allele‑based fitness (recommended)
        "viability_allele": {
            "Drive": 0.9,            # multiplies by copy number
            "Resistance": 1.0
        },

        # 3. Fecundity effects
        "fecundity_allele": {
            "Drive": 0.95            # affects females only
        },

        # 4. Sexual selection effects
        "sexual_selection_allele": {
            "Drive": (1.0, 0.8)      # (default male, carrier male)
        }
    }
```

### Sex‑ and Age‑Specificity

```python
def fitness_patch(self):
    return {
        # Sex‑specific
        "viability_allele": {
            "Drive": {
                "female": 0.95,      # in females
                "male": 0.85         # more severe in males
            }
        },

        # Age‑specific
        "viability_allele": {
            "Drive": {
                0: 1.0,               # age 0 (juvenile)
                1: 0.95,              # age 1
                2: 0.90               # age 2+
            }
        },

        # Combined: sex + age
        "viability_allele": {
            "Drive": {
                "female": {0: 0.98, 1: 0.96, 2: 0.94},
                "male": {0: 0.92, 1: 0.88, 2: 0.84}
            }
        }
    }
```

## Advanced Topics

### Combining Multiple Presets

```python
# Create multiple presets
mutation = PointMutation(mutation_rate=1e-5)
drive = HomingDrive(name="GeneDrive", drive_allele="Drive", target_allele="WT")

class SelectionPreset(GeneticPreset):
    def __init__(self, target_allele: str = "Deleterious", cost: float = 0.1):
        super().__init__(name="SelectionPreset")
        self.target_allele = target_allele
        self.cost = cost

    def gamete_modifier(self, population):
        return None

    def zygote_modifier(self, population):
        return None

    def fitness_patch(self):
        return {
            "viability_allele": {
                self.target_allele: 1.0 - self.cost,
            }
        }

selection = SelectionPreset(target_allele="Deleterious", cost=0.1)

# Apply multiple presets at once
population.apply_preset(mutation, drive, selection)

# Or in the builder
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="ComplexModel")
       .presets(mutation, drive, selection)
       .build())
```

### Order and Interaction of Presets

Multiple presets are applied in the order they are given:

1. Gamete modifiers are applied in order.
2. Zygote modifiers are applied in order.
3. Fitness patches are merged (later patches override earlier ones).

### Performance Optimisation

```python
class OptimizedPreset(GeneticPreset):
    """Preset implementation with performance optimisation"""

    def __init__(self):
        super().__init__(name="OptimizedPreset")
        # Pre‑compute commonly used data
        self._precomputed_rules = None

    def gamete_modifier(self, population):
        # Cache the rule set to avoid recreating it
        if self._precomputed_rules is None:
            from natal.gamete_allele_conversion import GameteConversionRuleSet
            self._precomputed_rules = GameteConversionRuleSet("Optimized")
            # ... add rules

        return self._precomputed_rules.to_gamete_modifier(population)
```

## Troubleshooting

### Common Issues

1. **Preset has no effect**
   - Check that not all methods return `None`.
   - Verify that allele names are spelled correctly.
   - Confirm that conversion rates are within [0, 1].

2. **Species binding error**
   - Ensure that the preset and the population use the same species.
   - Use delayed binding (do not specify `Species` at creation time).

3. **Performance issues**
   - Avoid creating many temporary objects inside modifiers.
   - Use rule set caching.
   - Consider simplifying complex rule chains.

### Debugging Tips

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, population):
        print(f"Applying preset to species: {population.species.name}")
        print(f"Available alleles: {list(population.species.gene_index.keys())}")

        # Create and return the modifier
        # ...
```

## Related Chapters

- [Allele Conversion Rules](allele_conversion_rules.md) – detailed conversion rule system
- [Pattern Matching and Extensible Configuration](genotype_patterns.md) – syntax rules and pattern design
- [Observation Rules for Populations](observation_rules.md) – using patterns in observation groups
- [Modifier Mechanism](modifiers.md) – low‑level modifier principles
- [Quick Start](quickstart.md) – basic usage tutorial
