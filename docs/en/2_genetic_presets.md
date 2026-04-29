# Genetic Presets

`Genetic Presets` are a mechanism in the NATAL framework for defining reusable genetic modifications, supporting rapid configuration of gene drives, mutation systems, and other genetic modifications.

## Overview

**Genetic Presets** provide a standardized way to define genetic modification rules, including:

- Modifying gamete production rules (e.g., gene drive segregation distortion)
- Altering zygote development processes (e.g., embryonic resistance formation)
- Adjusting fitness parameters (e.g., cost of the drive allele)

## Applying Presets in the Builder

```python
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="TestPop")
       .presets(preset1, preset2)  # Multiple presets can be applied
       .build())
```

## Built-in Presets

### HomingDrive -- Homing-based Gene Drive

`HomingDrive` implements CRISPR/Cas9-type homing-based gene drive:

```python
from natal.genetic_presets import HomingDrive

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
# Sex-specific parameters
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # Sex-specific rates
    late_germline_resistance_formation_rate=(0.02, 0.04),  # Tuple format (female, male)
    embryo_resistance_formation_rate=0.01,
    functional_resistance_ratio=0.2,  # 20% of resistance alleles are functional

    # Fitness costs
    viability_scaling=0.9,      # 10% viability cost
    fecundity_scaling=0.95,     # 5% fecundity cost
    sexual_selection_scaling=0.85  # 15% sexual selection disadvantage
)
```

### ToxinAntidoteDrive -- Toxin-Antidote Drive (TARE/TADE)

`ToxinAntidoteDrive` is used for modeling systems where "the drive allele triggers target site disruption, the disrupted allele causes fitness loss, and the drive allele provides rescue."

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

## Practical Examples

### Simple Point Mutation

```python
from natal.genetic_presets import HomingDrive
from natal.population_builder import AgeStructuredPopulationBuilder

# Create gene drive
drive = HomingDrive(
    name="DemoDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate=0.95
)

# Build population and apply preset
species = Species.from_dict("TestSpecies", {
    "chr1": {"GeneA": ["WT", "Drive"]}
})

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="DriveTest", stochastic=False)
       .age_structure(n_ages=5)
       .initial_state({"female": {"WT|WT": [0, 0, 100, 0, 0]}})
       .presets(drive)
       .build())

# Run simulation
pop.run(ticks=100)
```

### Combining Multiple Presets

```python
from natal.genetic_presets import HomingDrive, ToxinAntidoteDrive

# Create multiple presets
drive1 = HomingDrive("Drive1", "Drive", "WT", conversion_rate=0.95)
drive2 = ToxinAntidoteDrive("Drive2", "Toxin", "Target", conversion_rate=0.90)

# Apply multiple presets simultaneously
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="MultiDriveTest")
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
