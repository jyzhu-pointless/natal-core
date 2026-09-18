# Designing Your Own Preset (2): Using filters to Control Rule Scope

`filters` is a mapping from scope names to existing type-pattern strings. It does not accept functions or parsed Pattern objects. Pass the original pattern string; the rule compiler resolves it against the host species and registry.

## Supported keys

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

## Parent and current-state conditions

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

## Labels and fixed sources

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

## Reusing a pattern in a preset

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
