# Population Model (Panmictic)

The `Population` class is the core component of NATAL Core, responsible for managing the genetic state and simulation process of the population.

> **Note**: `DiscreteGenerationPopulation` and `AgeStructuredPopulation` are **panmictic (single-deme, well-mixed)** models. For multi-deme spatial populations with migration topology and heterogeneous parameters, see [Spatial Simulation Guide](3_spatial_simulation.md).

## Population Types

NATAL Core provides two main population types:

### Discrete Generation Population
`DiscreteGenerationPopulation` is suitable for species with non-overlapping generations, where each generation completely replaces the previous one. The simulation process is simple and efficient.

### Age-Structured Population
`AgeStructuredPopulation` is suitable for species with overlapping generations, supporting age-dependent survival and fecundity, and configurable sperm storage mechanisms.

> Both population types are subclasses of `BasePopulation` and share most methods.

## Creating a Population

Use the fluent chain API. The default `Configurator` path writes parameters immediately.
See [Population Initialization](2_population_initialization.md) for details.

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

# Discrete-generation
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .build()
)

# Age-structured
pop = (
    nt.AgeStructuredPopulation.setup(sp, legacy_path=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({
        "female": {"WT|WT": [0, 0, 100, 100, 80, 60, 40, 20]},
        "male":   {"WT|WT": [0, 0, 100, 100, 80, 60, 40, 20]},
    })
    .survival(female=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0],
              male=[1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.0])
    .reproduction(eggs_per_female=100,
                  female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
    .competition(carrying_capacity=5000, low_density_growth_rate=6.0,
                 juvenile_growth_mode="logistic")
    .build()
)
```

### Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding:

```python
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)
```

See [Runtime Parameter Modification](3_runtime_modification.md) for details.

## Running Simulations

### Single-Step Simulation

```python
# Simulate one step (one time unit)
pop.run_tick()

# Simulate multiple steps, printing state after each step
for _ in range(100):
    pop.run_tick()
    print(pop.output_current_state())
```

### Batch Simulation

```python
# Simulate 100 steps
pop.run(100)
# or
pop.run(n_steps=100)
```

## Accessing Population State

### Current State Information

```python
# Population size
current_size = pop.total_population_size
print(f"Current population size: {current_size}")

# Female count
female_count = pop.total_females
print(f"Female count: {female_count}")

# Male count
male_count = pop.total_males
print(f"Male count: {male_count}")

# Sex ratio
ratio = pop.sex_ratio
print(f"Sex ratio (female/male): {ratio}")

# Current time step
current_tick = pop.tick
print(f"Current tick: {current_tick}")
```

### Allele Frequencies

```python
# Compute allele frequencies
allele_freqs = pop.compute_allele_frequencies()
print("Allele frequencies:", allele_freqs)

# Get specific allele frequency
drive_freq = allele_freqs.get("D", 0.0)
print(f"Drive allele frequency: {drive_freq}")
```

## History Recording System

### History Configuration

The population object has built-in history recording functionality, with configurable recording frequency and storage format:

```python
# Configure history recording
pop.record_every = 10  # Record every 10 steps
pop.max_history = 1000  # Maximum of 1000 snapshots

# Run simulation with history recording
results = pop.run(n_steps=500, record_every=5)
```

### History Data Access

```python
# Get complete history
full_history = pop.output_history()
print("Number of history records:", len(full_history["snapshots"]))
print("Last step data:", full_history["snapshots"][-1])

# Get history at a specific tick
history_at_tick_100 = pop.output_history(tick=100)
print("State at tick 100:", history_at_tick_100)

# Get list of recorded time steps
ticks = [snapshot["tick"] for snapshot in full_history["snapshots"]]
print("Recorded ticks:", ticks)
```

### History Management

```python
# Clear history to save memory
pop.clear_history()

# Restart recording
results = pop.run(n_steps=100, record_every=5)
```

## Output Functions

### Current State Output

```python
# Get detailed snapshot of current state
current_state = pop.output_current_state()
print("Current state:", current_state)

# Get readable dictionary format
readable_state = pop.output_current_state(as_dict=True)
print("Readable state:", readable_state)

# Get JSON format (for transport and storage)
json_state = pop.output_current_state(as_json=True)
print("JSON state:", json_state[:200])  # Show first 200 characters
```

### Integration with Observation Rules

Combined with observation rules, specific subpopulation data can be extracted from the population state. For detailed methods, see [Extracting Population Simulation Data](2_data_output.md).

```python
# Create observation rules
observation = pop.create_observation(
    groups={
        "adult_wt": {"genotype": ["WT|WT"], "age": [1]},
        "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    },
    collapse_age=False,
)

# Get current state with observation rules
current = pop.output_current_state(observation=observation)
print("Current observation data:", current["observed"])

# Get history data with observation rules
history = pop.output_history(observation=observation)
print("History observation data:", history["observed"])
```

## Reset and Restart

```python
# Reset to initial state
pop.reset()

# Re-simulate after reset
pop.reset()
results = pop.run(n_steps=50)
```

## Simulation Control

### Check Simulation Status

```python
# Check if simulation is finished
if pop.is_finished:
    print("Simulation complete")
else:
    print("Simulation still running")

# Manually finish simulation
pop.finish_simulation()
```
