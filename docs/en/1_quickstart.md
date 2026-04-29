# Quick Start: Get Started with NATAL in 15 Minutes

This chapter will walk you through the core modeling workflow and visualization tools of NATAL using a **discrete-generation population** and an **age-structured population** as examples.
If you haven't installed `natal-core` yet, please refer to the [homepage](index.md) to complete the installation.

---

## 1️⃣ Step 1: Define the Genetic Architecture

NATAL uses a **declarative** syntax to define genetic architecture. You can quickly describe complex loci, chromosomes, and gamete labels using `Species.from_dict()`:

```python
import natal as nt

# Define genetic architecture
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {    # Chromosome
            "A": ["WT", "Drive", "Resistance"]    # Locus: [list of alleles]
        }
    },
    gamete_labels=["default", "Cas9_deposited"]    # Optional: gamete labels for simulating cytoplasmic deposition effects
)
```

### Understanding Key Concepts in Genetic Architecture

This framework divides biological genetic information into two layers: **genetic structures** (static templates) and **genetic entities** (dynamic instances).

#### Genetic Structures -- The Species' "Blueprint"

Defines "what exists," describing only possibilities, not specific choices:

- **`Species`**: A diploid organism species, containing multiple homologous chromosome pairs, serving as the top-level container for genomic information.
- **`Chromosome`**: A chromosome (e.g., `chr1`), containing multiple loci.
- **`Locus`**: A genetic locus (e.g., `A`), defining the possible allele names at that position (e.g., `["WT", "Drive"]`).

> The structure layer is like a "floor plan" -- how many rooms and halls, what kind of furniture each room can hold.

#### Genetic Entities -- The Concrete "Decorated Instance"

Based on the blueprint, actually selecting an allele at each locus to form the genetic material that truly exists in the simulation:

- **`Gene` (or `Allele`)**: A specific allele (e.g., `WT`, `Drive`). It is a concrete variant instantiated at a locus.
- **`Haplotype`**: The combination of selected alleles across all loci on one chromosome.
- **`HaploidGenotype`**: One haplotype contributed from each homologous chromosome pair in the species, together forming a complete haploid genome.
- **`Genotype` (Diploid Genotype)**: A combination of two haploid genomes from the maternal and paternal parents, representing the individual's complete genetic information.
  - **Note:** Genotypes strictly distinguish between maternal and paternal haploid genomes. In string representation, the order follows `Maternal|Paternal`. `A|a` and `a|A` are considered different genotypes.

> The entity layer is like a "decorated house" -- each room has already selected specific furniture styles, and the windows have been determined as round or square.

#### Why This Distinction?

- **Structures** are model-level, immutable configurations (e.g., "this species has two loci, A and B"), which can be defined once before the simulation starts.
- **Entities** are population-level, dynamically occurring instances (e.g., "the current population has four genotypes: `WT|WT`, `WT|Drive`, `Drive|WT`, `Drive|Drive`"), generated and transmitted through genetic rules.

This separation allows the simulation to flexibly define complex genetic architectures while maintaining efficient computation at runtime.

> For complete concepts, object relationships, and more examples, please refer to [Genetic Structures and Entities](2_genetics.md).

### Verify the Architecture

```python
# View all possible genotypes
all_genotypes = sp.get_all_genotypes()
print(f"There are {len(all_genotypes)} genotypes in total")
# Output: There are 9 genotypes in total
# (WT|WT, WT|Drive, WT|Resistance, Drive|WT, Drive|Drive, Drive|Resistance, Resistance|WT, Resistance|Drive, Resistance|Resistance)

# Get specific genotypes
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
print(f"WT|WT: {wt_wt}")
print(f"WT|Drive: {wt_drive}")
```

> For more details on genetic architecture, see [Genetic Structures and Entities](2_genetics.md)

---

## 2️⃣ Step 2: Initialize the Population

Initialization is a crucial step in model construction. At this stage, NATAL performs a "compilation" process, converting high-level objects into efficient numerical mapping matrices.

### DiscreteGenerationPopulation -- The Simplest Starting Point

If your model doesn't need age structure (e.g., theoretical models, laboratory organisms like fruit flies), you can use the simpler discrete-generation model:

```python
# Discrete-generation model (no age structure)
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="FruitFlyPop",
        stochastic=True                # True: stochastic model; False: deterministic model
    )
    .initial_state({
        "female": {"WT|WT": 1000},
        "male":   {"WT|WT": 1000}
    })
    .reproduction(
        eggs_per_female=50,            # Eggs per female
        sex_ratio=0.5                  # Offspring sex ratio
    )
    .build()
)

print(f"Initial population: {pop.get_total_count()}")
```

### AgeStructuredPopulation -- Closer to Nature

For natural populations or mixed-cage populations that require age structure, use the age-structured model:

```python
# Age-structured model
pop = (nt.AgeStructuredPopulation
    .setup(
        species=sp,
        name="MosquitoPop",
        stochastic=False               # False: deterministic model; True: stochastic model
    )
    .age_structure(
        n_ages=8,                      # 8 age classes
        new_adult_age=2                # Age 2 is considered adult
    )
    .initial_state({
        "female": {
            "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
        },
        "male": {
            "WT|WT":    [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0],
        }
    })
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    .reproduction(
        eggs_per_female=100,
        sex_ratio=0.5,
        use_sperm_storage=True,        # Enable sperm storage mechanism
    )
    .competition(
        juvenile_growth_mode=1,        # 1: Fixed competition mode
        age_1_carrying_capacity=1200
    )
    .build()
)

print(f"Initialization complete!")
print(f"Total population size: {pop.get_total_count():.0f}")
print(f"Total females: {pop.get_female_count():.0f}")
print(f"Total males: {pop.get_male_count():.0f}")
```

### Comparison of Population Types

| Feature | DiscreteGenerationPopulation | AgeStructuredPopulation |
|---------|------------------------------|------------------------|
| Age structure | Not supported | Supported |
| Survival rates | Fixed probability | Age-specific |
| Sperm storage | Not supported | Supported |
| Use cases | Laboratory serial-passaged populations | Natural populations, mixed-cage populations |
| Complexity | Lower | Higher |

---

## 3️⃣ Step 3: Use the Genetic Presets System

For common genetic phenomena such as gene drives and point mutations, NATAL provides a **Genetic Presets** system. Compared to manually writing low-level mapping modifier functions, the preset system is more concise, reusable, and maintainable.

### Using a Gene Drive Preset

```python
# Create a gene drive preset
drive = nt.HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

# Add the preset to a discrete-generation population
pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="FruitFlyPop", stochastic=True)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .presets(drive)                 # Apply the gene drive preset
    .build()
)
```

### Using Other Presets

The current preset system includes [HomingDrive](api/genetic_presets.md#natal.genetic_presets.HomingDrive) and [ToxinAntidoteDrive](api/genetic_presets.md#natal.genetic_presets.ToxinAntidoteDrive), with more preset types being continuously expanded in the future.

You can also define custom presets; see [Design Your Own Presets](3_custom_presets.md) for details.

### Fitness Settings (Optional)

If you need to configure fitness effects, you can do so in the `fitness()` method:

```python
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .fitness(viability={
        "Resistance|Resistance": {"female": 0.7},   # Resistance homozygotes have reduced survival
        "Drive|Drive": {"female": 0.0}              # Drive homozygotes are sterile
    })
    .presets(drive)
    .build()
)
```

> **Tip**: For advanced users who need to customize complex genetic rules, you can refer to the [Modifier Mechanism](3_modifiers.md) to write modifier functions manually. However, for most common scenarios, the preset system is simpler and more reliable.

---

## 4️⃣ Step 4: Define Simulation Logic -- Hooks

The **Hook system** allows you to inject custom intervention or monitoring logic at key points in the simulation loop (e.g., at the start of each step, after survival screening). Using the declarative `Op` syntax is the most efficient and intuitive approach:

```python
from natal.hook_dsl import hook, Op

@hook(event='first')
def release_drive_males():
    """Release drive-carrying males at tick == 10"""
    return [
        Op.add(
            genotypes='WT|Drive',    # Select WT|Drive genotype
            ages=2,                  # Adult age (only effective for age-structured models)
            sex='male',              # Release only males
            delta=500,               # Add 500 individuals
            when='tick == 10'        # Condition
        )
    ]

# Register with the population
release_drive_males.register(pop)

# Or register during the build process
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    # ... (other initialization methods)
    .hooks(release_drive_males)
    .build()
)
```

> **Tip**: For advanced users requiring high performance or complex logic, native Numba Hooks are available. See [Hook System](2_hooks.md) for details.

---

## 5️⃣ Step 5: Run Simulation and Analyze Results

```python
# Run for 100 time steps, recording history every 10 steps
pop.run(n_steps=100, record_every=10)

# Or run until a specific condition (defined in a Hook)
pop.run(n_steps=200, record_every=5, finish=False)
```

### View Results

```python
# 1) View current state as a readable dictionary (suitable for logging/debugging/API responses)
state_view = nt.population_to_readable_dict(pop)
print(state_view["state_type"], state_view["tick"])
print(state_view["individual_count"]["female"].keys())

# 2) For JSON, export directly
state_json = nt.population_to_readable_json(pop, indent=2)
print(state_json[:240])

# 3) Define reusable observation rules (recommended via the population API)
observation = pop.create_observation(
    groups={
        "adult_drive_female": {
            "genotype": "Drive::*",
            "sex": "female",
            "age": [2, 3, 4, 5, 6, 7],
        },
        "all_adults": {
            "age": [2, 3, 4, 5, 6, 7],
        },
    },
    collapse_age=False,
)

# 4) Export the current snapshot (state translation + observation)
current_obs = pop.output_current_state(
    observation=observation,
    include_zero_counts=False,
)
print(current_obs["labels"])
print(current_obs["observed"]["adult_drive_female"])

# 5) Export historical observations (can be directly used for plotting/export)
history_obs = pop.output_history(
    observation=observation,
    include_zero_counts=False,
)
print(history_obs["n_snapshots"])
print(history_obs["snapshots"][0]["observed"]["all_adults"])
```

It is recommended to use `output_current_state()` together with `output_history()`:

- `observation` defines the observation targets (grouping and filtering rules)
- The state translation API defines the export format (readable dict/JSON)

If you prefer module-level functions, you can also use
`nt.output_current_state(...)` and `nt.output_history(...)`; the semantics are
equivalent to the population methods.

### Corresponding Runnable Examples

The following scripts correspond directly to the above workflow (run from the repository root):

```bash
python demos/observation_history_demo.py
python demos/discrete.py
python demos/mosquito.py
```

For more scenarios, refer to the `demos/` directory.

### Using the Built-in Visualization Dashboard (Optional)

NATAL provides a NiceGUI-based real-time visualization dashboard that allows you to observe population dynamics in a browser:

```python
import natal as nt
from natal.ui import launch

# ... define genetic architecture, build population ...

# Launch the dashboard
launch(pop, port=8080, title="My Simulation")
```

Once launched, open <http://localhost:8080> in your browser to view dynamic charts of population counts, genotype frequencies, etc.

---

## Deep Dive: The "Compilation" Process During Initialization

Although the high-level code is intuitive and readable, a series of complex operations occur under the hood during `build()`:

1. **Index Registration**: All genotypes are assigned integer indices, stored in `pop.registry` (IndexRegistry)
2. **Mapping Matrix Generation**: Based on genetic presets and the genetic mapping `modifiers`, two key matrices are generated:
   - `Genotype → Gamete`: Specifies which gametes each genotype produces
   - `Gamete → Zygote`: Specifies which genotypes gamete combinations produce
3. **Configuration Compilation**: All parameters are compiled into a `PopulationConfig` NamedTuple, preparing for Numba JIT optimization
4. **Hooks Compilation**: User-defined Hooks are compiled into execution plans, to be called at the appropriate times
5. **State Initialization**: A `PopulationState` NamedTuple (containing numpy arrays) is created based on the initial distribution

This process is transparent to the user, but understanding it is important. See:
- [Index Registry](4_index_registry.md)
- [PopulationState & PopulationConfig](4_population_state_config.md)
- [Modifiers System](3_modifiers.md) and [Genetic Presets System](2_genetic_presets.md)
- [Hooks System](2_hooks.md)
- [Numba Optimization Guide](4_numba_optimization.md)

---

## Complete Examples

### Example 1: Discrete-Generation Population + Gene Drive + Hook

```python
import natal as nt
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

sp = nt.Species.from_dict(
    name="FruitFly",
    structure={"chr1": {"A": ["WT", "Drive", "Resistance"]}}
)

drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

@hook(event='first')
def release_drive():
    return [Op.add(genotypes='Drive|WT', delta=50, when='tick == 10')]

pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="FruitFlyPop", stochastic=True)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .presets(drive)
    .hooks(release_drive)              # Register Hook
    .build()
)

pop.run(n_steps=100, record_every=10)
print(f"Final population: {pop.get_total_count():.0f}")
print(f"Allele frequencies: {pop.compute_allele_frequencies()}")
```

### Example 2: Age-Structured Population + Gene Drive + Fitness + Hook

```python
import natal as nt
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

sp = nt.Species.from_dict(
    name="AnophelesGambiae",
    structure={"chr1": {"A": ["WT", "Drive", "Resistance"]}},
    gamete_labels=["default", "Cas9_deposited"]
)

drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

@hook(event='first')
def release_drive():
    return [Op.add(genotypes='Drive|WT', ages=[2,3,4,5,6,7], delta=100, when='tick == 10')]

pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MosquitoPop", stochastic=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({
        "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
        "male": {
            "WT|WT": [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0]
        }
    })
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    .reproduction(eggs_per_female=100, sex_ratio=0.5, use_sperm_storage=True)
    .competition(juvenile_growth_mode=1, age_1_carrying_capacity=1200)
    .fitness(viability={"Drive|Drive": {"female": 0.0}})
    .presets(drive)
    .hooks(release_drive)
    .build()
)

pop.run(n_steps=100, record_every=10)
print(f"Final population: {pop.get_total_count():.0f}")
print(f"Allele frequencies: {pop.compute_allele_frequencies()}")
```

---

## Next Steps

Now that you have mastered the basics! Next, you can:

1. **Dive Deeper into the Genetic Presets System**: [Genetic Presets](2_genetic_presets.md) - Learn how to create custom presets
2. **Understand Genetic Architecture**: [Genetic Structures and Entities](2_genetics.md) - Gain in-depth knowledge of Species, Chromosome, and other concepts
3. **Master Advanced Features**: [Hook System](2_hooks.md) - Learn how to inject custom simulation logic
4. **Need Custom Genetic Rules**: [Modifier Mechanism](3_modifiers.md) - Write manual gamete/zygote modifiers
5. **Performance Optimization**: [Numba Optimization Guide](4_numba_optimization.md) - Improve simulation performance

---

## FAQ

### Q: What are "gamete_labels"?
**A**: Additional dimensions used to label gametes. For example, "default" and "Cas9_deposited" can distinguish between gametes with or without Cas9 protein deposition. When calculating zygotes, both the allele and the label of the gamete are considered.

### Q: Why is initialization slow?
**A**: During initialization, two mapping matrices need to be generated, with complexity related to the 3rd-4th power of the number of genotypes. Depending on the Numba cache status, varying degrees of compilation may also be required. For relatively simple genetic setups (only a few dozen genotypes), this is expected to take from a few seconds to tens of seconds. This only happens once. Each subsequent tick is very fast.

### Q: When should I use a discrete-generation population?
**A**: When your model does not require age structure, using `DiscreteGenerationPopulationBuilder` is simpler. It is suitable for:
- Laboratory models such as fruit flies
- Theoretical model studies
- Simulations that do not require age-related effects

### Q: What is the difference between "deterministic" and "stochastic"?
**A**:
- `is_stochastic=False`: Uses expectations from the multinomial distribution, results are completely deterministic (allowing non-integer values)
- `is_stochastic=True`: Uses random sampling, results fluctuate randomly (always integers)

---

**Ready to learn more?** [Go to the next chapter: Genetic Structures and Entities](2_genetics.md)
