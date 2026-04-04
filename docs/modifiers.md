# Modifier Mechanism

Modifiers are used to inject “rule‑level changes” into the genetic flow.

If you want to model mechanisms such as gene drive, embryo rescue, or cytoplasmic incompatibility, modifiers are one of the core tools.

> For common cases, prefer the [Genetic Preset System](genetic_presets.md).
> Presets are more concise; modifiers are more flexible.

## 1. Where Modifiers Act

In the simulation, genetic outcomes are typically determined by two types of mappings:

1. Genotype‑to‑gamete mapping.
2. Gamete‑combination‑to‑zygote mapping.

Modifiers allow controlled modifications of these two mappings.

```text
Genotype
  → (Gamete Modifier) → gamete distribution
  → (Zygote Modifier) → zygote distribution
```

## 2. Two Types of Modifiers

### 2.1 Gamete Modifier

Used to change the probability of producing certain gametes from a given parental genotype.

Typical use cases:

- Biased segregation of a drive allele.
- Enhancement or suppression of specific gamete types.
- Attaching labels (e.g., deposition markers) to gametes.

### 2.2 Zygote Modifier

Used to change the offspring genotype distribution after combining a maternal gamete and a paternal gamete.

Typical use cases:

- Embryo death or rescue.
- Cytoplasmic incompatibility.
- Non‑Mendelian offspring redistribution.

## 3. Recommended Integration via Builder

In practice, it is recommended to register modifiers at the Builder stage:

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

This allows the model configuration, initial state, and genetic rules to be organised in one go.

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

### 4.2 Labelling Gametes

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
            # This combination could be mapped to low viability or no offspring, depending on the model
        },
    }
```

## 6. Gamete Labels

Gamete labels are used to represent “different biological states of the same allele”, for example whether a cytoplasmic factor is present.

Examples:

- `(A1, default)`
- `(A1, Cas9_deposited)`

Both have the same allele but different labels, which may trigger different rules at the zygote stage.

### 6.1 Configuring Labels

```python
pop = AgeStructuredPopulation(
    ...,
    gamete_labels=["default", "Cas9_deposited"],
)
```

## 7. Registration and Priority

### 7.1 Dynamic Registration

```python
pop.set_gamete_modifier(my_gamete_modifier, hook_name="drive")
pop.set_zygote_modifier(embryo_rescue_modifier, hook_name="rescue")
```

### 7.2 Priority

When multiple modifiers act together, they are executed in order of priority.

```python
pop.set_gamete_modifier(base_mod, priority=1, hook_name="base")
pop.set_gamete_modifier(drive_mod, priority=2, hook_name="drive")
```

In practice, place “basic rules” at lower priority and “overriding/corrective rules” at higher priority.

## 8. Modelling Advice

1. Start with minimal rules, then gradually add complexity.
2. For each added modifier, run small‑scale interpretability tests.
3. Output intermediate results for critical combinations to verify that probabilities are normalised.
4. Prefer presets for complex systems; only fall back to custom modifiers when presets are insufficient.

## 9. Minimal Workflow

```python
# 1) Define the genetic architecture
# 2) Write a minimal gamete modifier
# 3) Run a short simulation and check genotype frequency changes
# 4) Add a zygote modifier
# 5) Expand to full parameter sweeps
```

## 10. Chapter Summary

Modifiers are the core mechanism for expressing “rewritten genetic rules” in NATAL.

- Gamete modifiers focus on “which gametes are produced”.
- Zygote modifiers focus on “which offspring are produced after gamete fusion”.

Using them together allows you to model most advanced genetic mechanisms.

---

## Related Chapters

- [Genetic Preset System](genetic_presets.md)
- [Hook System](hooks.md)
- [Deep Dive into Simulation Kernels](simulation_kernels.md)
- [PopulationState and PopulationConfig](population_state_config.md)
