# Modifier Mechanism

Modifiers are used to inject "rule-level changes" into the genetic workflow.

If you want to express mechanisms such as gene drive, embryo rescue, or cytoplasmic incompatibility, Modifiers are one of the core tools.

> For common scenarios, prioritize the use of the [Genetic Preset System](2_genetic_presets.md).
> Presets are more concise, while Modifiers are more flexible.

## 1. Where Modifiers Act

In simulation, genetic outcomes are typically determined by two types of mappings:

1. The mapping from genotypes to gametes.
2. The mapping from gamete combination to zygotes.

Modifiers act to rewrite these two mappings in a controlled manner.

```text
Genotype
  → (Gamete Modifier) → Gamete distribution
  → (Zygote Modifier) → Zygote distribution
```

## 2. Two Types of Modifiers

### 2.1 Gamete Modifier

Used to change the probability of a parental genotype producing specific gametes.

Typical uses:

- Biased segregation of drive alleles
- Enhancement or suppression of specific gamete types
- Attaching tags to gametes (e.g., deposition markers)

### 2.2 Zygote Modifier

Used to change the offspring genotype distribution from the combination of "maternal gamete + paternal gamete."

Typical uses:

- Embryo death or rescue
- Cytoplasmic incompatibility
- Non-Mendelian offspring redistribution

## 3. Recommended Integration Method (Builder)

In practice, it is recommended to register Modifiers uniformly at the Builder stage:

```python
from natal.population_builder import AgeStructuredPopulationBuilder


def my_gamete_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        }
    }


pop = (
    AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .modifiers(gamete_modifiers=[(None, "drive", my_gamete_modifier)])
    .build()
)
```

This allows model configuration, initial state, and genetic rules to be organized in a single pass.

## 4. Gamete Modifier Examples

### 4.1 Gene Drive Bias

```python
def heg_drive_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.98,
            ("WT", "Cas9_deposited"): 0.02,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.98,
            ("WT", "Cas9_deposited"): 0.02,
        },
    }
```

### 4.2 Tagging Gametes

```python
def cas9_deposition_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        "WT|WT": {
            ("WT", "default"): 1.0,
        },
    }
```

## 5. Zygote Modifier Examples

### 5.1 Embryo Rescue

```python
def embryo_rescue_modifier(pop):
    return {
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Drive|WT": 0.4,
            "WT|WT": 0.0,
            "Drive|Drive": 0.6,
        },
    }
```

### 5.2 Cytoplasmic Incompatibility

```python
def ci_modifier(pop):
    return {
        (("Allele1", "uninfected"), ("Allele1", "Wolbachia")): {
            # This combination can be mapped to low survival or no offspring as needed
        },
    }
```

## 6. Gamete Labels

Gamete labels are used to represent "different biological states of the same allele," such as whether it carries a cytoplasmic factor.

For example:

- `(A1, default)`
- `(A1, Cas9_deposited)`

Both have the same allele but different labels, potentially triggering different rules in the zygote stage.

### 6.1 Configuring Labels

```python
pop = AgeStructuredPopulation(
    ...,
    gamete_labels=["default", "Cas9_deposited"],
)
```

## 7. Registration Methods and Priority

### 7.1 Dynamic Registration

```python
pop.set_gamete_modifier(my_gamete_modifier, hook_name="drive")
pop.set_zygote_modifier(embryo_rescue_modifier, hook_name="rescue")
```

### 7.2 Priority

When multiple Modifiers act simultaneously, they execute in order of priority.

```python
pop.set_gamete_modifier(base_mod, priority=1, hook_name="base")
pop.set_gamete_modifier(drive_mod, priority=2, hook_name="drive")
```

In practice, it is recommended to put "base rules" at a lower priority and "override/correction rules" at a higher priority.

## 8. Modeling Advice

1. Start with minimal rules, then gradually add complexity.
2. For every Modifier added, run small-scale interpretability tests.
3. For critical combinations, output intermediate results to verify probability normalization.
4. For complex systems, prioritize presets; only drop down to custom Modifiers when presets are insufficient.

## 9. Minimal Workflow

```python
# 1) Define genetic architecture
# 2) Write a minimal gamete modifier
# 3) Run a short simulation and check genotype frequency changes
# 4) Then add a zygote modifier
# 5) Finally expand to full parameter scanning
```

## 10. Chapter Summary

Modifiers are the core mechanism in NATAL for expressing "genetic rule rewriting."

- Gamete Modifiers focus on "which gametes are produced."
- Zygote Modifiers focus on "which offspring are produced after gamete combination."

Using both in coordination allows expressing most advanced genetic mechanisms.

---

## Related Chapters

- [Genetic Preset System](2_genetic_presets.md)
- [Hook System](2_hooks.md)
- [Simulation Kernels Deep Dive](4_simulation_kernels.md)
- [PopulationState and PopulationConfig](4_population_state_config.md)
