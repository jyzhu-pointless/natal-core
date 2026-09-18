# Designing Your Own Preset

This section will guide you through designing, implementing, validating, and publishing custom Genetic Presets from scratch.

## Four conversion rules

All rule constructors use keyword arguments. `rate` is required, finite, and in `[0, 1]`; `filters=None` means unrestricted, and `name=None` is an optional display name.

| Rule | Required fields | Optional fields |
|---|---|---|
| `GameteGtypeConversionRule` | `to`, `rate` | `filters`, `name` |
| `ZygoteZtypeConversionRule` | `to`, `rate` | `filters`, `name` |
| `GameteAlleleConversionRule` | `from_allele`, `to_allele`, `rate` | `filters`, `name` |
| `ZygoteAlleleConversionRule` | `from_allele`, `to_allele`, `rate` | `filters`, `name`, `side="both"` |

Whole-state `to` strings have the form `[genotype or *]@[label or *]`; the gamete genotype is haploid. Both parts must be explicit. A whole-part `*` preserves that input component. Other target components must be exact: partial wildcards, sets, and unordered target alternatives are not supported.

| Zygote target | Action |
|---|---|
| `A\|B@I` | Jointly replace genotype and slab |
| `*@I` | Replace slab, preserve genotype |
| `A\|B@*` | Replace genotype, preserve slab |
| `*@*` | Identity conversion |

A whole-state rule is one probabilistic event: at rate 0.4, `A@S` to `B@I` produces 60% `A@S` and 40% `B@I`. It does not independently convert the two components. To model independent changes, declare two rules whose filters cover the relevant branches.

Allele rules accept source and target gene names as strings. Gene names are unique across the species, so no `locus` argument is needed; the target must belong to the source's locus. Allele rules preserve labels. For zygotes, `side` is `maternal`, `paternal`, or `both`; each eligible copy independently converts at `rate`. With `side="both"`, an ordered `A|A` input at rate 0.4 gives 36% `A|A`, 24% `B|A`, 24% `A|B`, and 16% `B|B`.

RuleSets apply rules in declaration order, including branches produced by earlier rules. There is no numeric priority or first-match stop. An eligible state without the source allele remains unchanged; unknown alleles, cross-locus targets, and illegal targets are errors. See [filters](genotype_filter.md) for the five supported scope keys.

When several RuleSets are registered in one gamete or zygote modifier pipeline, each consumes the preceding result in registration order. Rebuilding or refreshing the model starts a fresh pipeline from the unmodified species baseline; it does not reapply rules to the previous compiled result.

The following declarations are independently runnable; compiling them requires matching species alleles and labels.

```python
from natal import (
    GameteGtypeConversionRule, ZygoteZtypeConversionRule,
    GameteAlleleConversionRule, ZygoteAlleleConversionRule,
)

whole_gamete = GameteGtypeConversionRule(
    filters={"current": "A@default"}, to="B@I", rate=0.4,
)
whole_zygote = ZygoteZtypeConversionRule(
    filters={"maternal": "*@I"}, to="*@I", rate=0.9,
)
gamete_allele = GameteAlleleConversionRule(
    from_allele="A", to_allele="B", rate=0.4,
    filters={"parent_sex": "female"},
)
zygote_allele = ZygoteAlleleConversionRule(
    from_allele="A", to_allele="B", rate=0.4, side="both",
    filters={"current": "*@I"},
)
```

Use `add_rule(rule)` to append these declarations. `GameteConversionRuleSet.add_gtype_convert()`, `ZygoteConversionRuleSet.add_ztype_convert()`, and each stage's `add_allele_convert()` expose the corresponding constructor fields.

This API intentionally breaks compatibility. The former label-only classes and rule aliases are removed; label conversion uses the whole-state rules above. Old rule parameters, object/callable inputs, and colon-separated label names are not supported. Use `@` for labels; the existing unordered filter syntax `::` is unchanged. In particular, migrate an allele conversion that jointly changed a label to a whole-state joint conversion, not to two independent events.

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
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.5)
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

The following two fragments assume an existing `pop` whose species declares W and D at one locus.

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# In the zygote, if the A locus has the D allele, convert W->D
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

#### Combining Gamete + Zygote Usage

Drive systems typically use both types of rules simultaneously:

```python
# Gamete stage: W -> D (biased)
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.99)

# Zygote stage: allele conversion (ensure homozygosity)
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.95,
    filters={"current": "*::D"}
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
from natal.frontend.presets import GeneticPreset, PresetFitnessPatch
from natal.frontend.modifiers import GameteModifier, ZygoteModifier
from typing import Optional

class MyCustomPreset(GeneticPreset):
    """Custom genetic modification preset"""

    def __init__(self, name: str = "MyCustom", species=None):
        super().__init__(name=name, species=species)
        # Custom parameters
        self.custom_param = 0.5

    def gamete_modifier(self, host) -> Optional[GameteModifier]:
        """Define modification logic at the gamete stage"""
        # Return GameteModifier or None
        return None

    def zygote_modifier(self, host) -> Optional[ZygoteModifier]:
        """Define modification logic at the zygote stage"""
        # Return ZygoteModifier or None
        return None

    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """Define fitness effects"""
        # Return fitness configuration dictionary or None
        return None
```

Implementation notes:

1. **`gamete_modifier` and `zygote_modifier` are abstract methods** - both must be implemented (they may return `None`); otherwise the subclass cannot be instantiated (`TypeError`)
2. **`fitness_patch` is optional** - the default implementation returns `None`
3. **Can return None** - indicates no modification needed at that stage
4. **Supports deferred species binding** - can create without specifying `Species`
5. **The parameter of `gamete_modifier` / `zygote_modifier` is `host`** - one uniform entry point (interface contract `natal.frontend.genetics.compile.RecipeHost`): at runtime it points to the live Population, during compilation it points to the in-progress PopulationBuilder; both expose the same four read-only attributes — `species`, `config`, `registry`, `index_registry`

### Simple Examples

#### Simple Point Mutation

```python
from natal.frontend.presets import GeneticPreset, PresetFitnessPatch
from natal.frontend.modifiers import GameteConversionRuleSet

class PointMutation(GeneticPreset):
    """Simple point mutation: WT mutates to Mutant at a certain frequency"""

    def __init__(self, mutation_rate: float = 1e-5):
        super().__init__(name="PointMutation")
        self.mutation_rate = mutation_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("PointMutation")
        ruleset.add_allele_convert(from_allele="WT", to_allele="Mutant", rate=self.mutation_rate)
        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None

    def fitness_patch(self):
        return {
            "viability_per_allele": {"Mutant": 0.98}  # Slightly deleterious
        }
```

> **Note**: NATAL ships a built-in `PointMutation` preset covering this behavior
> (plus multi-target competition and sex-specific rates); see
> [Genetic Presets](2_genetic_presets.md). The class above is a minimal
> custom-preset exercise, not the built-in API.

#### Bidirectional Mutation Balance

```python
class BidirectionalMutation(GeneticPreset):
    """Bidirectional mutation balance"""

    def __init__(self, forward_rate: float = 1e-5, backward_rate: float = 1e-6):
        super().__init__(name="BidirectionalMutation")
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def gamete_modifier(self, host):
        from natal.frontend.modifiers import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("BidirectionalMutation")

        # A -> B (forward mutation)
        ruleset.add_allele_convert(from_allele="A", to_allele="B", rate=self.forward_rate)
        # B -> A (back mutation)
        ruleset.add_allele_convert(from_allele="B", to_allele="A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None
```

## 2. Using filters to Control Rule Scope

`filters` is a mapping from scope names to existing type-pattern strings. It does not accept functions or parsed Pattern objects. Pass the original pattern string; the rule compiler resolves it against the host species and registry.

### Supported keys

The two gamete rules share one set of keys; the two zygote rules share another.

| Key | Gamete rules | Zygote rules |
|---|---|---|
| `current` | Entering gamete gtype | Entering zygote branch ztype |
| `parent` | Producer parent ztype | Unsupported |
| `parent_sex` | `female`, `male`, or `both` | Unsupported |
| `maternal` | Unsupported | Fertilizing maternal gamete gtype |
| `paternal` | Unsupported | Fertilizing paternal gamete gtype |

A gtype is `haploid_genotype@glab`; a ztype is `genotype@slab`. Multiple keys are AND-ed. Omitted keys, `filters=None`, and an empty mapping impose no corresponding restriction. Unknown keys, keys from the wrong stage, invalid patterns, and unknown labels are errors; they do not silently match nothing.

`current` is checked when each rule receives a branch, after earlier rules have acted. Parent and fertilizing-gamete information remains fixed throughout that stage. There is no separate `when` parameter or new condition-expression language.

### Parent and current-state conditions

This declaration is independently runnable. Compiling it requires a species with W and D at the same locus.

```python
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet("homing_drive")
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    filters={"parent": "W::D", "parent_sex": "female"},
)
```

`W::D` selects heterozygous parents in either left/right order of the homologous chromosomes. Use `*::D` for any parent carrying D in this single-locus example. In a multilocus model, supply the full appropriate pattern rather than a substring test on the genotype name.

At the zygote stage, use `current` for a condition on the offspring itself:

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

zygote_rules = ZygoteConversionRuleSet("zygote_copy")
zygote_rules.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},
)
```

### Labels and fixed sources

A bare genetic pattern does not restrict the label. `*@infected` restricts only the label; `@default` explicitly names the default label. Both output catalogs and patterns use `@`; the old colon label format is not supported. The existing unordered-pair separator `::` retains its meaning.

The following declaration keeps the offspring genotype and gives an uninfected offspring the infected slab with probability 0.9 if its maternal gamete is labeled infected. The host must declare both labels in their appropriate gamete and somatic catalogs.

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

infection = ZygoteConversionRuleSet("maternal_transmission")
infection.add_ztype_convert(
    filters={"current": "*@default", "maternal": "*@infected"},
    to="*@infected",
    rate=0.9,
)
```

`maternal` and `paternal` refer to gametes, not to the parents' diploid genotypes or somatic labels.

### Reusing a pattern in a preset

Store strings in configuration and pass them directly. This class definition is runnable; using it requires WT and Drive at one locus and a valid parent pattern for that species.

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet

class PatternBasedPreset(GeneticPreset):
    def __init__(self, pattern: str, conversion_rate: float = 0.95):
        super().__init__(name="PatternBasedPreset")
        self.pattern = pattern
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("PatternBased")
        ruleset.add_allele_convert(
            from_allele="WT",
            to_allele="Drive",
            rate=self.conversion_rate,
            filters={"parent": self.pattern},
        )
        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None
```

For background-dependent mutation, put the background requirement in the parent pattern and the source allele in `from_allele`. Use the same type pattern for observation groups when the intended population scope is the same. A parent condition and a current-offspring condition describe different objects even if their strings look identical.

See [conversion rules](allele_conversion_rules.md) for targets and probability semantics, and [preset validation](preset_encapsulation_and_validation.md) for a complete builder example.

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
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet


class DrivePreset(GeneticPreset):
    def __init__(self, conversion_rate: float = 0.5):
        super().__init__(name="DrivePreset")
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("drive_rules")

        ruleset.add_allele_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            filters={"parent": "W::D"},
        )

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None
```

### Applying Presets in the PopulationBuilder Build Chain

```python
import natal as nt

# The species must declare the W and D alleles used by DrivePreset
species = nt.Species.from_dict(name="DriveExpSpecies", structure={"chr1": {"A": ["W", "D"]}})

pop = (
    nt.AgeStructuredPopulation
    .setup(species=species, name="DriveExperiment", stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({"female": {"W|W": 500}, "male": {"W|W": 500}})
    .presets(DrivePreset(conversion_rate=0.55))
    .build()
)
```

This is the most recommended way to integrate "Preset as a configuration component."

### Validation Checklist (Highly Recommended)

Before conducting large-scale experiments, at least complete the following checks:

1. Mechanism check: Are the conversion direction and target allele correct?
2. Filter check: Does the `filters` hit range match expectations?
3. Conservation check: Is frequency normalization valid?
4. Control check: Is the trend reasonable compared to a baseline without Preset?
5. Stability check: for stochastic models (`stochastic=True`), are conclusions robust across repeated runs? (No public random-seed API is exposed yet.)

### Experiment Recording Advice

It is recommended to write Preset configuration into experiment metadata:

- Preset name
- Key parameters (e.g., `conversion_rate`)
- Code version or commit
- Randomness settings (e.g. `stochastic`) and runtime environment

This significantly reduces the risk of "results cannot be reproduced."

### Complex Gene Drive Example

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet

class ComplexDrive(GeneticPreset):
    """Complex gene drive with multi-stage conversion"""

    def __init__(self):
        super().__init__(name="ComplexDrive")

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("ComplexDrive")

        # Stage 1: Drive conversion (WT -> Drive)
        ruleset.add_allele_convert(from_allele="WT", to_allele="Drive", rate=0.95,
                           filters={"parent": "*::Drive"})

        # Stage 2: Resistance formation (remaining WT -> Resistance)
        ruleset.add_allele_convert(from_allele="WT", to_allele="Resistance", rate=0.05,
                           filters={"parent": "*::Drive"})

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # Additional modification at the embryo stage
        ruleset.add_allele_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            filters={"maternal": "*@cas9"}  # Requires maternal Cas9 deposition
        )

        return ruleset.to_zygote_modifier(host)

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
- Check every top-level key of `fitness_patch`: an unsupported key raises a `ValueError` that names it and lists the supported keys (for example `viability_allele` instead of `viability_per_allele`), and nothing is applied to the model

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
    def gamete_modifier(self, host):
        print(f"Applying preset to species: {host.species.name}")
        print(f"Available alleles: {list(host.species.gene_index.keys())}")

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
2. Fine-grained rule scope control (filters)
3. Preset engineering, validation, and release

You now have mastered the complete workflow of designing, implementing, validating, and publishing custom Presets from scratch.
