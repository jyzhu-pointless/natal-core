# Designing Your Own Preset (3): Encapsulation, Validation, and Pre‑release Checks

This is the final chapter of the “Designing Your Own Preset” main line. In the previous two chapters you completed:

1. Rule definition (Gamete and Zygote conversions).
2. Fine‑grained control over rule scope.

This chapter explains how to encapsulate these elements into a **reusable Preset**, validate it thoroughly, and finally release it.

## 1. Why Encapsulate into a Preset

If you write rules only in a single script, you will encounter three problems later:

1. **Hard to reuse**: each experiment requires copying the logic.
2. **Hard to trace**: it is difficult to say exactly which set of rules was used in a given version.
3. **Hard to maintain**: rules, fitness effects, and Hooks are scattered across multiple files.

The value of a Preset is to converge these elements into a stable configuration unit.

## 2. Recommended Preset Structure

A practical Preset should include:

1. Mechanism rules (conversion rules and filters).
2. Fitness patches (if needed).
3. Optional parameters (e.g., conversion rate, sex restrictions).
4. A clear name and version marker.

## 3. Example: Encapsulating a Minimal DrivePreset

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

## 4. Applying the Preset in the Builder

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

This is the recommended way to integrate a Preset as a configuration component.

## 5. Validation Checklist (Strongly Recommended)

Before running large‑scale experiments, at least complete the following checks:

1. **Mechanism check**: Is the direction of conversion and the target allele correct?
2. **Filter check**: Does the `genotype_filter` hit the expected set of genotypes?
3. **Mass conservation check**: Are the frequencies properly normalised?
4. **Control check**: Compared to a baseline without the Preset, are the trends reasonable?
5. **Stability check**: Are the conclusions robust when the random seed is changed?

## 6. Experiment Logging Recommendations

It is recommended to record Preset configuration in the experiment metadata:

- Preset name
- Key parameters (e.g., `conversion_rate`)
- Code version or commit hash
- Random seed

This significantly reduces the risk of irreproducible results.

## 7. Chapter Summary

🎉 Congratulations! You have completed the full main line of “Designing Your Own Preset”:

1. Rule definition (Gamete and Zygote conversions) – Chapter 14.
2. Fine‑grained rule scope control (`genotype_filter`) – Chapter 15.
3. Preset engineering, validation, and release – Chapter 16 (this chapter).

You now master the complete loop from “biological hypothesis in your mind” to “stable, reproducible Preset components”. These three steps are sufficient to simulate most genetic drive systems.

---

## Looking Back and Extending

If you want to revisit previous content or deepen your understanding:

- [Genotype Pattern Matching](genotype_pattern_matching_design.md) – Chapter 13, foundational concepts
- [Designing Your Own Preset (1): Starting from Allele Conversion Rules](allele_conversion_rules.md) – Chapter 14
- [Designing Your Own Preset (2): Using `genotype_filter` to Control Rule Scope](genotype_filter.md) – Chapter 15
- [Genetic Presets Usage Guide](genetic_presets.md) – learn about built‑in presets in NATAL
- [Observation Rules for Populations](observation_rules.md) – how to observe and analyse simulation results
