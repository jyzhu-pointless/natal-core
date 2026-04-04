# Designing Your Own Preset (1): Starting from Allele Conversion Rules

This chapter begins a complete main line: **how to design your own Preset**.

The first step is not to write the Preset class immediately, but to clearly express the “genetic mechanism”. For most drive systems, this step is usually reflected in **allele conversion rules**.

## 1. First Define the Mechanism Goal

Before writing any code, answer three questions:

1. Which allele will be converted (`from_allele`)?
2. What will it be converted to (`to_allele`)?
3. What is the conversion probability (`rate`)?

For example, a minimal drive assumption could be written as:

- During gamete production, `W -> D` with probability $0.5$.

## 2. Rule Object and Rule Set

NATAL provides two layers:

- `GameteAlleleConversionRule`: a single rule.
- `GameteConversionRuleSet`: a set of rules.

You can think of it as:

- A Rule is a “sentence”.
- A RuleSet is a “paragraph”.

## 3. Minimal Working Example

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

This example is already sufficient to describe a minimal conversion mechanism.

## 4. Zygote Conversion Rules (Fertilised Egg Stage)

Allele conversion can also happen at the zygote stage, typically used to simulate:

- **Gene drive repair**: a repair system (e.g., Cas9 cleavage repair) expressed in the zygote.
- **Allele‑specific death**: reduced viability of certain genotype zygotes.
- **Post‑zygotic conversion**: allele conversion during development.

### 4.1 From Gamete to Zygote

Key differences:

| Stage | Input | Mechanism | When to use |
|-------|-------|-----------|-------------|
| **Gamete** | Gamete (haploid) | Conversion during gamete production | Gamete drive systems |
| **Zygote** | Zygote (diploid) | Conversion immediately after fertilisation | Zygotic drive, zygotic repair |

### 4.2 Using `ZygoteConversionRuleSet`

```python
from natal.gamete_allele_conversion import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# In the zygote, if the A locus contains a D allele, convert W -> D
def has_d_at_a(genotype) -> bool:
    # pseudo‑code; actual depends on your Genotype structure
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

### 4.3 Combining Gamete + Zygote

Usually a drive system uses both types of rules:

```python
# Gamete stage: W -> D (bias)
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_convert("W", "D", rate=0.99)

# Zygote stage: copy to ensure homozygosity
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_convert(
    "W", "D",
    rate=0.95,
    genotype_filter=lambda g: "D" in str(g)
)

pop.add_gamete_modifier(gamete_ruleset.to_gamete_modifier(pop))
pop.add_zygote_modifier(zygote_ruleset.to_zygote_modifier(pop))
```

This allows more complex genetic drive mechanisms to be simulated.

## 5. Connecting to a Population

The rule set itself is just a definition; to take effect, it must be converted into a modifier and bound to a population. The workflow is the same for both Gamete and Zygote:

```python
# Gamete conversion
gamete_mod = gamete_ruleset.to_gamete_modifier(pop)
pop.add_gamete_modifier(gamete_mod, name="homing")

# Zygote conversion
zygote_mod = zygote_ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="repair")
```

In practice, it is recommended to first verify the direction of frequency changes in a short simulation before adding more complex rules.

## 6. Common Control Parameters

### 6.1 `sex_filter`

Used to specify which sex the rule applies to:

- `"both"`: applies to both sexes (default).
- `"male"`: males only.
- `"female"`: females only.

Example:

```python
ruleset.add_convert("W", "D", rate=0.8, sex_filter="male")
```

### 6.2 `name`

It is recommended to give each rule and rule set a human‑readable name to facilitate experiment reproducibility.

## 7. Rule Design Advice

1. Start with a single rule; do not write a dozen at once.
2. After adding each rule, run 20‑50 steps to check whether the direction matches expectations.
3. Keep a record of the “biological hypothesis → parameter value” mapping to avoid later interpretability issues.

## 8. Chapter Summary

You have completed the first step of Preset design:

- Express the mechanism as executable rules.
- Bind the rules to a population.

The next chapter continues the main line: **how to use `genotype_filter` to restrict the rules to your desired scope**.

---

## Next Chapter

- [Designing Your Own Preset (2): Using `genotype_filter` to Control Rule Scope](genotype_filter.md)
