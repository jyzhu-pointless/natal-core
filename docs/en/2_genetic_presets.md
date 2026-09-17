# Genetic Presets

`Genetic Presets` are a mechanism in the NATAL framework for defining reusable genetic modifications, supporting rapid configuration of gene drives, mutation systems, and other genetic modifications.

## Overview

**Genetic Presets** provide a standardized way to define genetic modification rules, including:

- Modifying gamete production rules (e.g., gene drive segregation distortion)
- Altering zygote development processes (e.g., embryonic resistance formation)
- Adjusting fitness parameters (e.g., cost of the drive allele)

## Applying Presets

```python
import natal as nt

pop = (nt.DiscreteGenerationPopulation.setup(species, name="TestPop")
       .presets(preset1, preset2)  # Multiple presets can be applied
       .build())
```

## Built-in Presets

### CytoplasmicPreset and Wolbachia — Maternal Label Inheritance

`CytoplasmicPreset` uses conversion rules for both gamete tagging and zygote
label inheritance. Its keyword-only parameters `default_glab` and `default_slab`
explicitly select the source gamete and somatic labels eligible for conversion;
both default to the name `default`. Names must exist in the corresponding species
label lists when the inheritance rules are compiled.

For females in a mapped somatic label, only gametes still carrying the default
gamete label receive the corresponding maternal tag. During fertilization, that
tag redirects offspring still carrying the default somatic label to the mapped
label, without changing their genotype. Other labels are preserved.

These rules act on the distribution left by preceding modifiers. In particular,
an earlier modifier's genotype changes remain, and offspring already assigned
a non-default somatic label are not relabeled. Modifier order therefore matters.
`Wolbachia` uses this mechanism for infection inheritance. Its `default_glab`
parameter selects the source gamete label (default `default`), and its existing
`normal_slab` parameter also selects the source somatic label (default `normal`).
For custom names, pass them explicitly; for example, a species whose uninfected
somatic label is `default` needs `normal_slab="default"`.

These preset parameters do not reorder labels or change the species' baseline
distribution, which still assigns probability to the first label in each list.
Selecting a different source label only processes branches already carrying
that label, for example after an earlier modifier has assigned it.

#### Optional incompatibility costs

`Wolbachia` assumes one strain, perfect maternal transmission, and complete
rescue by infected mothers. `incompatibility_cost` defaults to `0`, preserving
maternal inheritance without additional labels. A positive cost marks offspring
of an uninfected mother and an infected father with `incompatibility_slab`
(default `incompatible`), then multiplies their own selected fitness by
`1 - incompatibility_cost`. The cost must be finite and in `[0, 1]`.

| `incompatibility_effect` | Effect on the marked individual |
|---|---|
| `"zygote_viability"` (default) | Embryonic survival before juvenile competition |
| `"viability"` | Ordinary viability at the last juvenile age (`new_adult_age - 1`; age 0 in discrete generations) |
| `"fecundity"` | Its own reproductive output when it reproduces, without reducing its birth or survival |

The scalar cost applies to both sexes. With `fecundity`, one CI parent contributes
one multiplier; two CI parents contribute its square, following the framework's
existing parental fecundity rules. This is not a reduction of the original
incompatible parents' clutch. `viability_scaling` and `fecundity_scaling` remain
separate costs of infected carriers and must be finite and nonnegative.

For positive costs, declare the additional somatic label `incompatible` and
gamete label `wolbachia_ci`, or supply custom `incompatibility_slab` and
`paternal_glab` names. The latter labels gametes from infected males as CI-inducing;
it does not transmit infection. Source gamete and somatic labels must still
match `default_glab` and `normal_slab`, and other modifiers' labels are preserved.
The origin label persists on surviving individuals but is not inherited:
CI-marked mothers are uninfected, provide no rescue, and can produce new
CI-marked offspring when paired with infected fathers. Their compatible
crosses produce normal-label offspring. Explicit fitness selectors targeting
only `*@normal` do not include the distinct `*@incompatible` group; include
both or use a broader selector when both should share another fitness effect.

Setting cost to zero disables CI marking and its fitness patch; it does not
relabel existing individuals. Runtime reconfiguration follows the existing
fixed-layout rules and cannot introduce offspring types pruned at build time.
For a runnable example, see [Modifier Mechanism, section 5.2](3_modifiers.md#52-cytoplasmic-incompatibility).

### HomingDrive -- Homing-based Gene Drive

`HomingDrive` implements CRISPR/Cas9-type homing-based gene drive:

```python
from natal.frontend.presets import HomingDrive

# Create a basic gene drive
drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,  # 95% conversion efficiency
    late_germline_resistance_formation_rate=0.03  # 3% resistance formation
)

# Apply to population
population.apply_preset(drive)
```

#### Advanced Configuration

```python
import natal as nt
from natal.frontend.presets import HomingDrive

species = nt.Species.from_dict(
    name="DepositionExample",
    structure={"chr1": {"drive": ["WT", "Drive", "Resistance", "FunctionalResistance"]}},
    gamete_labels=["default", "Cas9_deposited"],
)

# Sex-specific parameters
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    functional_resistance_allele="FunctionalResistance",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # Sex-specific rates
    late_germline_resistance_formation_rate=(0.02, 0.04),  # Tuple format (female, male)
    embryo_resistance_formation_rate=0.01,
    cas9_deposition_glab="Cas9_deposited",
    functional_resistance_ratio=0.2,  # 20% of resistance alleles are functional

    # Fitness costs
    viability_scaling=0.9,      # 10% viability cost
    fecundity_scaling=0.95,     # 5% fecundity cost
    sexual_selection_scaling=0.85  # 15% sexual selection disadvantage
)
population = (
    nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
    .initial_state({"female": {"WT|Drive": 100}, "male": {"WT|WT": 100}})
    .presets(drive)
    .build()
)
population.run(1)
```

Embryo resistance is triggered only by parental Cas9 deposition. Register
`cas9_deposition_glab` in the species' `gamete_labels`; without a configured
label, embryo editing is inactive even when the embryo inherits drive or Cas9.
Carrier mothers label all their output gametes, so embryos that do not inherit
drive can still be edited. For split drives, the depositing parent must carry
both the drive and Cas9 alleles.

The embryo rate's `female` and `male` entries refer to maternal and paternal
sources, not offspring sex. A scalar sets both rates, but the paternal source
is active only with `use_paternal_deposition=True`. Thus the example applies
1% editing per remaining target copy from maternal deposition only. When both
enabled sources are present, their rates act sequentially on the remaining
target copies: the total conversion probability is `1 - (1 - e_m) * (1 - e_p)`.

### ToxinAntidoteDrive -- Toxin-Antidote Drive (TARE/TADE)

`ToxinAntidoteDrive` is used for modeling systems where "the drive allele triggers target site disruption, the disrupted allele causes fitness loss, and the drive allele provides rescue."

```python
from natal.frontend.presets import ToxinAntidoteDrive

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

Parameter descriptions:

1. `conversion_rate`: Probability of `target -> disrupted` conversion in the germline, supports `float`, `(female, male)`, or per-sex dictionary
2. `embryo_disruption_rate`: Embryonic conversion probability, can be combined with `cas9_deposition_glab` / `use_paternal_deposition` to model maternal/paternal deposition effects
   - If `cas9_deposition_glab` is set, ensure that the species to which the population belongs registered the same label via `gamete_labels` at creation time; otherwise, applying the preset will raise a `KeyError`
3. `viability_scaling` and `viability_mode`: Used to define the toxin effect of the `disrupted` allele; TARE commonly uses `viability_scaling=0.0` with `viability_mode="recessive"`
4. `fecundity_scaling` and `fecundity_mode`: Define fecundity costs
5. `sexual_selection_scaling` (optional): Defines sexual selection effects; supports scalar or tuple `(default_male, carrier_male)`, used in conjunction with `sexual_selection_mode`

Example with mating cost:

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

### PointMutation -- Spontaneous Point Mutation

`PointMutation` models spontaneous mutation of a source allele into one or more target alleles. The mutation happens in every gamete carrying the source allele, regardless of the parent's genotype, and each target keeps the rate you declare — the targets compete instead of consuming each other's share:

```python
from natal.frontend.presets import PointMutation

# Single target
mutation = PointMutation(
    "A2B",
    source_allele="A",
    target_allele="B",
    mutation_rate=1e-5,          # 1e-5 of the source gametes become B
    viability_scaling=0.98,      # optional: slightly deleterious target
)

# Several targets: the rates are effective rates, not cascade shares
multi = PointMutation(
    "MultiMut",
    source_allele="A",
    target_alleles=["B", "C", "D"],
    mutation_rates=[1e-7, 5e-6, 1e-5],
)

population.apply_preset(mutation)
```

Parameter descriptions:

1. `source_allele`: The allele that mutates. Every gamete carrying it converts, with no parent-genotype filter (point mutation is spontaneous).
2. `target_allele` / `mutation_rate` and `target_alleles` / `mutation_rates`: The single-target and multi-target declaration forms; each rate accepts a `float`, a `(female, male)` pair, or a per-sex dictionary. A missing sex key means no conversion for that sex.
3. `rate_mode`: `"strict"` (default) treats the rates as probabilities and rejects a sum above 1; `"proportional"` treats them as proportions and scales them to sum to 1, so `[2, 3, 5]` is the same model as `[0.2, 0.3, 0.5]`.
4. `viability_scaling` / `fecundity_scaling` / `sexual_selection_scaling` / `zygote_viability_scaling` (and their `*_mode`): Fitness effects applied to the whole target group; all default to neutral.

In the current implementation, the mutation happens in the germline only (while gametes are produced, before fertilization); the preset registers no zygote-stage modifier.

The conversion rules of one ruleset run as a cascade: each rule only sees the source mass the previous rule left. Declaring `[0.3, 0.5, 0.1]` would therefore hand the second target an effective share of `0.5 × 0.7 = 0.35` if the raw rates were passed through. `PointMutation` compensates internally with `r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)`, so the realized gamete distribution of an `A|A` parent is exactly:

| target | declared rate | rate passed to the cascade | realized share |
|---|---|---|---|
| B | 0.3 | 0.3 | 0.3 |
| C | 0.5 | 0.5 / 0.7 ≈ 0.714 | 0.5 |
| D | 0.1 | 0.1 / 0.2 = 0.5 | 0.1 |
| A (unchanged) | — | — | 0.1 |

This "simultaneous competition" semantics is what distinguishes one multi-target `PointMutation` from several stacked single-target presets, whose rules cascade in registration order (first declared, first served). The compensation is computed per sex, so sex-specific rates compete independently within each sex.

Stacking presets therefore builds a *sequential* model, not the simultaneous one. A forward rule `W -> D` at `mu` declared before a reverse rule `D -> W` at `nu` gives `q' = (1 - nu) (q + mu (1 - q))` with equilibrium `mu (1 - nu) / (nu + mu (1 - nu))`, whereas the textbook "each gamete mutates at most once" model gives `q' = q (1 - nu) + mu (1 - q)` with equilibrium `mu / (mu + nu)`. The two differ by the double-mutation term `mu nu (1 - q)`: a gamete converted to `D` and back to `W` within one meiosis stays `W` in a cascade but counts as `D` when both rules act simultaneously (0.2386 against 0.25 for `mu = 0.02`, `nu = 0.06`). To recover the textbook model exactly, scale the *first-declared* rule: declare `mu / (1 - nu)` first and `nu` second (reverse-first: `nu / (1 - mu)` then `mu`), which matches both the slope and the intercept of the recursion. The preset's internal `r'k = rk / (1 - sum(ri, i < k))` compensation is not the cross-preset recipe — it matches the slope only and lands further from the textbook equilibrium (0.2347) than leaving the rates alone. With realistic rates (at most `1e-3`) the uncorrected offset is around `1e-4` and can be ignored.

## Practical Examples

### Simple Point Mutation

```python
import natal as nt
from natal.frontend.presets import PointMutation

# A wild-type allele A mutates into the allele R at 1e-4
mutation = PointMutation(
    name="A2R",
    source_allele="A",
    target_allele="R",
    mutation_rate=1e-4,
)

# Build population and apply preset
species = nt.Species.from_dict("PointMutationSpecies", {
    "chr1": {"GeneA": ["A", "R"]}
})

pop = (nt.AgeStructuredPopulation.setup(species, name="MutationTest", stochastic=False)
       .age_structure(n_ages=5, new_adult_age=2)
       .initial_state({"female": {"A|A": [0, 0, 100, 0, 0]}})
       .presets(mutation)
       .build())

# Run simulation
pop.run(n_steps=100)
```

### Combining Multiple Presets

```python
import natal as nt
from natal.frontend.presets import HomingDrive, ToxinAntidoteDrive

# The species must declare every allele the presets use
species = nt.Species.from_dict("MultiDriveSpecies", {
    "chr1": {"A": ["WT", "Drive", "Toxin", "Target", "Disrupted"]}
})

# Create multiple presets
drive1 = HomingDrive("Drive1", "Drive", "WT", drive_conversion_rate=0.95)
drive2 = ToxinAntidoteDrive("Drive2", "Toxin", "Target", "Disrupted", conversion_rate=0.90)

# Apply multiple presets simultaneously
pop = (nt.DiscreteGenerationPopulation.setup(species, name="MultiDriveTest")
       .presets(drive1, drive2)  # Apply multiple presets
       .build())
```

## Further Learning

Creating custom presets is an advanced topic. For detailed content, please refer to the following dedicated documentation:

- [Design Your Own Presets](3_custom_presets.md)

## Related Sections

- [Design Your Own Presets](3_custom_presets.md) - Detailed conversion rule system and preset design
- [Genotype Pattern Matching](2_genotype_patterns.md) - Syntax rules and pattern design
- [Population Observation Rules](2_data_output.md) - Using patterns in observation groups
- [Modifier Mechanism](3_modifiers.md) - Underlying modifier principles
- [Quick Start](1_quickstart.md) - Basic usage tutorial
