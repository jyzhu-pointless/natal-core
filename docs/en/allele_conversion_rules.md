# Designing Your Own Preset (1): Starting with Allele Conversion Rules

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
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.5)
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

The following two fragments assume an existing `pop` whose species declares W and D at one locus.

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# In the zygote, convert W->D only for zygotes already carrying D
# ("carries D" expressed as an unordered-carrier pattern; the entering
# branch state is checked via filters["current"]).
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

### Combined Use of Gamete + Zygote

Drive systems typically use both types of rules simultaneously:

```python
# Gamete stage: W -> D (biased)
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.99)

# Zygote stage: achieve copying (ensure homozygosity)
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_allele_convert(
    from_allele="W", to_allele="D", rate=0.95,
    filters={"current": "*::D"},
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
        """Define gamete-stage modification logic"""
        # Return GameteModifier or None
        return None

    def zygote_modifier(self, host) -> Optional[ZygoteModifier]:
        """Define zygote-stage modification logic"""
        # Return ZygoteModifier or None
        return None

    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """Define fitness effects"""
        # Return fitness configuration dict or None
        return None
```

Implementation highlights:

1. **`gamete_modifier` and `zygote_modifier` are required** - `GeneticPreset` is an abstract base class, and a subclass missing either one cannot be instantiated (returning `None` is fine when that stage needs no modification)
2. **`fitness_patch` is optional** - omit it for no fitness effect; it may also return `None`
3. **Can return None** - indicating no modification is needed at that stage
4. **Supports deferred species binding** - `Species` can be unspecified at creation time
5. **The parameter of `gamete_modifier` / `zygote_modifier` is `host`** - one uniform entry point (interface contract `natal.frontend.genetics.compile.RecipeHost`): at runtime it points to the live Population, during compilation it points to the in-progress PopulationBuilder; both expose the same four read-only attributes — `species`, `config`, `registry`, `index_registry`

## Simple Examples

### Simple Point Mutation

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
        return None  # no zygote-stage modification

    def fitness_patch(self):
        return {
            "viability_per_allele": {"Mutant": 0.98}  # Slightly deleterious
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

    def gamete_modifier(self, host):
        from natal.frontend.modifiers import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("BidirectionalMutation")

        # A → B (forward mutation)
        ruleset.add_allele_convert(from_allele="A", to_allele="B", rate=self.forward_rate)
        # B → A (back mutation)
        ruleset.add_allele_convert(from_allele="B", to_allele="A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None  # no zygote-stage modification
```

## Chapter Summary

You have completed the first step of Preset design: defining allele conversion rules. The next chapter will cover how to control the scope of rule application.
