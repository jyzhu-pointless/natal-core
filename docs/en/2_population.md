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
    nt.AgeStructuredPopulation.setup(sp)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({
        "female": {"WT|WT": [0, 0, 100, 100, 80, 60, 40, 20]},
        "male":   {"WT|WT": [0, 0, 100, 100, 80, 60, 40, 20]},
    })
    .survival(female_age_based_survival=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0],
              male_age_based_survival=[1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.0])
    .reproduction(eggs_per_female=100,
                  female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
    .competition(carrying_capacity=5000, low_density_growth_rate=6.0,
                 juvenile_growth_mode="logistic")
    .build()
)
```

### Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding:

```python
# Single parameter
pop.update().competition(carrying_capacity=5000)

# Chain multiple parameters
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# Custom fields — read/write in hooks via config.custom['name'][()]
pop.update().custom(temperature=35.0)
```

Changes are written in-place via `set_param(config, name, value)` to 0-d ndarrays, taking effect immediately.

#### Low-Level set_param

```python
from natal.configurator import set_param

# Full name, short name, or alias all work
set_param(config, "competition.carrying_capacity", 5000.0)
set_param(config, "carrying_capacity", 5000.0)      # short name
set_param(config, "eggs_per_female", 100.0)          # alias
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
    print(pop.observe().values)
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
var_freq = allele_freqs.get("Var", 0.0)
print(f"Var allele frequency: {var_freq}")
```

## History Recording System

### History Configuration

Choose the History mode and capacity before `build()`. Raw mode and unlimited
capacity are the defaults:

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500})
    .record_history(mode="raw", max_rows=1000)
    .build()
)

# Run simulation with history recording
pop.run(n_steps=500, record_every=5)
```

### History Data Access

```python
history = pop.history
print("Number of history records:", history.n_records)
print("Individual-count shape:", history.individual_count.shape)
ticks = history.ticks
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
# Get current state projection
result = pop.observe()
print("Observation axes:", result.axes)
print("Observation values:", result.values)

# Define a custom observation at build time with IndividualSelector
from natal.patterns import IndividualSelector

pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .with_observation(
        groups={"adult": IndividualSelector(age=[1])},
        collapse_age=True,
    )
    .initial_state(...)
    .competition(...)
    .build()
)

# pop.observe() automatically uses the configured observation
detailed = pop.observe()
print("Detailed state:", detailed)
```

### Integration with Observation Rules

Combined with observation rules, specific subpopulation data can be extracted from the population state. For detailed methods, see [Extracting Population Simulation Data](2_data_output.md).

```python
# Every Population has a canonical observation; the default is identity
current = pop.observe()
print(current.axes)  # ("group", "sex", "age")

# Raw History can be projected later through the same observation
observed_history = pop.history.observe(pop.observation)
print(observed_history.values.shape)
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

## Wright-Fisher Extreme Speed Mode

Discrete-generation populations support a Wright-Fisher extreme speed mode: a single multinomial draw per tick replaces the step-by-step mate→fertilize→survive pipeline. Designed for effective population size modeling, 10-100× faster.

### Sampling Modes

| Mode | Description |
|------|-------------|
| DETERMINISTIC (3) | Infinite population limit, no randomness |
| MULTINOMIAL (1) | Classic Wright-Fisher single multinomial draw |
| POISSON (2) | Independent Poisson draws (large-N approximation) |

### Enabling (preliminary API)

```python
object.__setattr__(pop, "_config", pop.config._replace(extreme_speed_mode=3))
pop.run(100)
```

### Competition and Hooks

All three competition modes (FIXED/LOGISTIC/BEVERTON_HOLT) are supported, sharing the same scaling functions as the standard path. Only FIRST hooks are supported (fired before the WF tick). Deterministic WF mode matches the standard deterministic path tick-by-tick.

## Index Compression

Index compression prunes unreachable gamete types (GType) and zygote/individual types (ZType) at build time, reducing array dimensions.

### Enabling

```python
pop = nt.DiscreteGenerationPopulation.setup(
    species=sp, stochastic=False, compress=True,
).initial_state(...).competition(...)
    .build()
```

### Effect

- GType: single-locus with only A|A initially → HL from 2 to 1
- ZType: only A|A reachable → G from 4 to 1, offspring_tensor from 64 to 1 elements
- Combined: >98% reduction possible
