# Quick Start: 15 Minutes to NATAL

This chapter will guide you through the core modeling workflow and visualization tools of NATAL using examples of a **discrete‑generation population** and an **age‑structured population**.
If you haven't installed `natal-core` yet, please refer to the [Home Page](./index.md) for installation instructions.

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

### Understanding Key Concepts in the Architecture

This section retains the minimal mental model needed for the quick start. For complete concepts, object relationships, and more examples, see [Genetic Structures and Entities](genetic_structures.md).

This framework separates biological genetic information into two layers: **genetic structures** (static templates) and **genetic entities** (dynamic instances).

#### Genetic Structures – The "Blueprint" of a Species

Defines “what exists”, describing possibilities without concrete choices:

- **`Species`**: A diploid biological species containing multiple homologous chromosomes; the top‑level container of genomic information.
- **`Chromosome`**: A chromosome (e.g., `chr1`), containing multiple loci.
- **`Locus`**: A genetic locus (e.g., `A`), defining the possible allele names at that position (e.g., `["WT", "Drive"]`).

> The structure layer is like a “floor plan” – how many rooms, what furniture can go where.

#### Genetic Entities – The “Furnished” Instances

Based on the blueprint, actual alleles are chosen at each locus to form the genetic material that really exists in the simulation:

- **`Gene` (or `Allele`)**: A specific allele (e.g., `WT`, `Drive`). It is the concrete variant instantiated at a locus.
- **`Haplotype`**: The combination of chosen alleles on one chromosome.
- **`HaploidGenotype`**: One haplotype contributed from each homologous chromosome, forming a complete haploid genome.
- **`Genotype` (diploid genotype)**: The combination of maternal and paternal haploid genotypes, representing the individual’s full genetic information.
  - **Note:** Genotypes strictly distinguish the maternal and paternal haploid origins. In string representation, the order is `Maternal|Paternal`. `A|a` and `a|A` are considered different genotypes.

> The entity layer is like a “furnished house” – each room has specific furniture, and windows are either round or square.

#### Why Separate These Layers?

- **Structures** are model‑level, immutable configurations (e.g., “this species has loci A and B”), defined once before the simulation starts.
- **Entities** are population‑level, dynamically appearing instances (e.g., “the current population contains `WT|WT`, `WT|Drive`, `Drive|WT`, and `Drive|Drive` genotypes”), generated and transmitted by genetic rules.

This separation allows the simulation to flexibly define complex genetic architectures while maintaining high runtime performance.

### Verifying the Architecture

```python
# List all possible genotypes
all_genotypes = sp.get_all_genotypes()
print(f"Total number of genotypes: {len(all_genotypes)}")
# Output: Total number of genotypes: 9
# (WT|WT, WT|Drive, WT|Resistance, Drive|WT, Drive|Drive, Drive|Resistance, Resistance|WT, Resistance|Drive, Resistance|Resistance)

# Obtain a specific genotype
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
print(f"WT|WT: {wt_wt}")
print(f"WT|Drive: {wt_drive}")
```

> For more details on genetic architecture, see [Genetic Structures and Entities](genetic_structures.md).

---

## 2️⃣ Step 2: Initialise the Population

Initialisation is a key step in model construction. At this stage, NATAL performs a “compilation” process that converts high‑level objects into efficient numerical mapping matrices.

### Discrete‑Generation Population – The Simplest Starting Point

If your model does not need age structure (e.g., theoretical models, laboratory organisms like fruit flies), you can use the simpler discrete‑generation model:

```python
# Discrete‑generation model (no age structure)
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
        eggs_per_female=50,            # Number of eggs per female
        sex_ratio=0.5                  # Offspring sex ratio
    )
    .build()
)

print(f"Initial population size: {pop.get_total_count()}")
```

### Age‑Structured Population – More Realistic for Natural Settings

For natural populations or mixed‑cage populations that require age structure, use the age‑structured model:

```python
# Age‑structured model
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
        juvenile_growth_mode=1,        # 1: fixed competition mode
        age_1_carrying_capacity=1200
    )
    .build()
)

print("Initialisation complete!")
print(f"Total population size: {pop.get_total_count():.0f}")
print(f"Total females: {pop.get_female_count():.0f}")
print(f"Total males: {pop.get_male_count():.0f}")
```

### Comparison of the Two Population Types

| Feature | DiscreteGenerationPopulation | AgeStructuredPopulation |
|---------|------------------------------|--------------------------|
| Age structure | ❌ Not supported | ✅ Supported |
| Survival | Fixed probability | Age‑specific rates |
| Sperm storage | ❌ Not supported | ✅ Supported |
| Use case | Laboratory artificially propagated populations | Natural populations, mixed‑cage populations |
| Complexity | Lower | Higher |

---

## 3️⃣ Step 3: Use the Genetic Preset System

For common genetic phenomena such as gene drives and point mutations, NATAL provides a **Genetic Presets** system. Compared to manually writing low‑level mapping modifier functions, presets are simpler, reusable, and easier to maintain.

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

# Add the preset to a discrete‑generation population
pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="FruitFlyPop", stochastic=True)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .presets(drive)                 # Apply the gene drive preset
    .build()
)
```

### Other Presets

The preset system supports various genetic modifications:

- **HomingDrive**: CRISPR/Cas9 gene drive
- **PointMutation**: Simple point mutation
- **CustomPresets**: User‑defined custom presets

### Fitness Configuration (Optional)

If you need to set fitness effects, you can configure them in the `fitness()` method:

```python
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .fitness(viability={
        "Resistance|Resistance": {"female": 0.7},  # Reduced survival for resistance homozygotes
        "Drive|Drive": {"female": 0.0}              # Drive homozygotes are sterile
    })
    .presets(drive)
    .build()
)
```

> **💡 Tip**: Advanced users who need custom complex genetic rules can refer to the [Modifier mechanism](modifiers.md) to manually write modifier functions. However, for most common scenarios, the preset system is simpler and more reliable.

---

## 4️⃣ Step 4: Define Simulation Logic – Hooks

The **Hook system** allows you to inject custom interventions or monitoring logic at key points in the simulation loop (e.g., at the beginning of each step, after survival selection, etc.). Using declarative `Op` syntax is the most efficient and intuitive way:

```python
from natal.hook_dsl import hook, Op

@hook(event='first')
def release_drive_males():
    """Release drive‑carrying males when tick == 10"""
    return [
        Op.add(
            genotypes='WT|Drive',    # Select WT|Drive genotype
            ages=2,                  # Adult age (only effective for age‑structured models)
            sex='male',              # Release only males
            delta=500,               # Add 500 individuals
            when='tick == 10'        # Condition
        )
    ]

# Register the hook with the population
release_drive_males.register(pop)

# You can also register the hook during the building process
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    # ... (other initialisation methods)
    .hooks(release_drive_males)
    .build()
)
```

> **💡 Tip**: For high‑performance or complex logic, advanced users can use native Numba hooks. See [Hook System](hooks.md) for details.

---

## 5️⃣ Step 5: Run the Simulation and Analyse Results

```python
# Run 100 time steps, record history every 10 steps
pop.run(n_steps=100, record_every=10)

# Or run until a specific condition (defined in a hook)
pop.run(n_steps=200, record_every=5, finish=False)
```

### Viewing Results

```python
# 1) Get the current state as a readable dictionary (good for logging/debugging/API responses)
state_view = nt.population_to_readable_dict(pop)
print(state_view["state_type"], state_view["tick"])
print(state_view["individual_count"]["female"].keys())

# 2) If JSON is needed, you can export directly
state_json = nt.population_to_readable_json(pop, indent=2)
print(state_json[:240])

# 3) Integrate observation rules to view grouped results by business logic
observed = nt.population_to_observation_dict(
    pop,
    groups={
        "adult_drive_female": {
            "genotype": ["WT|Drive", "Drive|Drive"],
            "sex": "female",
            "age": [2, 3, 4, 5, 6, 7],
        }
    },
    collapse_age=False,
)
print(observed["observed"]["adult_drive_female"])
```

### 🎛️ Use the Built‑in Visualisation Dashboard (Optional)

NATAL provides a real‑time visualisation dashboard based on NiceGUI, allowing you to observe population dynamics in your browser:

```python
import natal as nt
from natal.ui import launch

# ... define genetic architecture, build the population ...

# Launch the dashboard
launch(pop, port=8080, title="My Simulation")
```

Once launched, open <http://localhost:8080> in your browser to view dynamic charts of population size, genotype frequencies, etc.

---

## Deep Dive: The “Compilation” Process During Initialisation

Although the high‑level code is intuitive, the underlying `build()` call performs a series of complex operations:

1. **Index registration**: All genotypes are assigned integer indices, stored in `pop.registry` (IndexRegistry).
2. **Mapping matrix generation**: Based on genetic presets and genetic mapping modifiers, two key matrices are generated:
   - `genotype → gamete`: specifies which gametes each genotype produces.
   - `gamete → zygote`: specifies which genotypes result from gamete combinations.
3. **Configuration compilation**: All parameters are compiled into a `PopulationConfig` NamedTuple, ready for Numba JIT optimisation.
4. **Hook compilation**: User‑defined hooks are compiled into execution plans that will be invoked at the appropriate times.
5. **State initialisation**: Creates a `PopulationState` NamedTuple (containing numpy arrays) from the initial distribution.

This process is transparent to the user, but understanding it is important. See also:
- [Index registry](index_registry.md)
- [PopulationState & PopulationConfig](population_state_config.md)
- [Modifiers system](modifiers.md) and [Genetic Presets system](genetic_presets.md)
- [Hooks system](hooks.md)

---

## 📊 Complete Examples

### Example 1: Discrete‑Generation Population + Gene Drive + Hook

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
    .hooks(release_drive)              # Register the hook
    .build()
)

pop.run(n_steps=100, record_every=10)
print(f"Final population size: {pop.get_total_count():.0f}")
print(f"Allele frequencies: {pop.compute_allele_frequencies()}")
```

### Example 2: Age‑Structured Population + Gene Drive + Fitness + Hook

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
    return [Op.add(genotypes='Drive|*', ages=[2,3,4,5,6,7], delta=100, when='tick == 10')]

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
print(f"Final population size: {pop.get_total_count():.0f}")
print(f"Allele frequencies: {pop.compute_allele_frequencies()}")
```

---

## 🎯 Next Steps

Now that you have mastered the basics, you can:

1. **Dive deeper into the Genetic Preset system**: [Genetic Presets Usage Guide](genetic_presets.md) – learn how to create custom presets
2. **Understand Genetic Architecture**: [Genetic Structures and Entities](genetic_structures.md) – in‑depth look at Species, Chromosome, and other concepts
3. **Master advanced features**: [Hook System](hooks.md) – learn to inject custom simulation logic
4. **Customise genetic rules**: [Modifier Mechanism](modifiers.md) – manually write gamete/zygote modifiers
5. **Optimise performance**: [Numba Optimization Guide](numba_optimization.md) – improve simulation performance

---

## ❓ Frequently Asked Questions

### Q: What are “gamete_labels”?
**A**: They are an extra dimension to label gametes. For example, `"default"` and `"Cas9_deposited"` can distinguish whether a gamete carries Cas9 protein deposition. When calculating zygotes, both the alleles and the labels of the gametes are considered.

### Q: Why is initialisation relatively slow?
**A**: Initialisation involves generating two mapping matrices, with complexity roughly O((number of genotypes)^(3‑4)). Depending on Numba’s caching behaviour, some compilation may also be required. For relatively simple genetic architectures (only dozens of genotypes), expect several to tens of seconds. This happens only once; subsequent ticks are very fast.

### Q: When should I use the discrete‑generation population?
**A**: Use the discrete‑generation population when your model does not need age structure. It is suitable for:
- Laboratory models like fruit flies
- Theoretical model studies
- Simulations that do not require age‑dependent effects

### Q: What is the difference between “deterministic” and “stochastic”?
**A**:
- `is_stochastic=False`: uses the expectation of a multinomial distribution – results are completely deterministic (may be fractional)
- `is_stochastic=True`: uses random sampling – results are stochastic (always integers)

---

**Ready to go deeper?** [Proceed to the next chapter: Genetic Structures and Entities →](genetic_structures.md)
