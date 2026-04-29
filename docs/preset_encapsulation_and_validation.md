# Designing Your Own Preset (3): Encapsulation, Validation, and Pre-release Checks

This chapter is the final chapter of the "Designing Your Own Preset" series. In the previous two chapters, you completed:

1. Rule definition (Gamete and Zygote conversion)
2. Fine-grained control of rule scope

This chapter will teach you how to encapsulate these components into **reusable Presets**, perform thorough validation, and finally publish them for use.

## Value of Encapsulation into a Preset

If you only write rules in scripts, you will encounter three problems later:

1. Hard to reuse: every experiment requires copying logic
2. Hard to trace: difficult to determine "which version used which set of rules"
3. Hard to maintain: rules, fitness, and hooks are scattered across multiple files

The value of a Preset is to converge all these elements into a stable configuration unit.

## Recommended Preset Structure

A practical Preset should include:

1. Mechanism rules (conversion rules and filters)
2. Fitness patches (if needed)
3. Optional parameters (e.g., conversion rate, sex restrictions)
4. Clear name and version markers

## Example: Encapsulating a Minimal DrivePreset

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

## Applying a Preset in the Builder

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

This is the most recommended way to integrate a Preset as a configuration component.

## Validation Checklist (Strongly Recommended)

Before conducting large-scale experiments, at least complete the following checks:

1. Mechanism check: are the conversion direction and target allele correct?
2. Filter check: does the `genotype_filter` hit the expected scope?
3. Conservation check: is frequency normalization valid?
4. Control check: is the trend reasonable compared to a baseline without the Preset?
5. Stability check: are conclusions robust when the random seed changes?

## Experiment Recording Recommendations

It is recommended to write Preset configuration into experiment metadata:

- Preset name
- Key parameters (e.g., `conversion_rate`)
- Code version or commit
- Random seed

This significantly reduces the risk of "results cannot be reproduced."

## Complex Gene Drive Example

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

        # Stage 1: Drive conversion (WT → Drive)
        ruleset.add_convert("WT", "Drive", rate=0.95,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        # Stage 2: Resistance formation (remaining WT → Resistance)
        ruleset.add_convert("WT", "Resistance", rate=0.05,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        return ruleset.to_gamete_modifier(population)

    def zygote_modifier(self, population):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # Additional embryonic stage modification
        ruleset.add_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            maternal_glab="cas9"  # requires maternal Cas9 deposition
        )

        return ruleset.to_zygote_modifier(population)

    def fitness_patch(self):
        return {
            "viability_per_allele": {
                "Drive": 0.9,      # drive allele cost
                "Resistance": 1.0   # resistance allele neutral
            },
            "fecundity_per_allele": {
                "Drive": 0.95
            },
            "zygote_per_allele": {
                "Drive": 0.8,     # reduced survival at zygote stage
                "Resistance": 1.0   # resistance allele neutral
            }
        }
```

## Common Errors and Debugging

### Parameter Validation Errors
- Verify the conversion rate is within the [0, 1] range

### Species Binding Errors
- Ensure the preset and population use the same species
- Use lazy binding (do not specify `Species` at creation time)

### Performance Issues
- Avoid creating many temporary objects in modifiers
- Use rule set caching
- Consider simplifying complex rule chains

### Debugging Tips

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, population):
        print(f"Applying preset to species: {population.species.name}")
        print(f"Available alleles: {list(population.species.gene_index.keys())}")

        # Create modifier and return
        # ...
```

## Pre-release Checklist

Before publishing a Preset, it is recommended to complete:

- [ ] Unit tests covering main functionality
- [ ] Clear and complete documentation
- [ ] Parameter range validation passed
- [ ] Compatibility testing with existing systems
- [ ] Performance benchmarking

## Chapter Summary

Congratulations! You have completed the full "Designing Your Own Preset" series:

1. Rule definition (Gamete and Zygote conversion)
2. Fine-grained rule scope control (genotype_filter)
3. Preset engineering, validation, and publishing

You now have a complete workflow for designing, implementing, validating, and publishing custom Presets from scratch.
