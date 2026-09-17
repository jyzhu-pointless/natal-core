# Population Model (Panmictic)

Population objects are the core component of NATAL Core, responsible for managing the genetic state and simulation process of the population. There is no `Population` class — the panmictic models are `DiscreteGenerationPopulation` and `AgeStructuredPopulation`, both subclasses of `BasePopulation`.

> **Note**: `DiscreteGenerationPopulation` and `AgeStructuredPopulation` are **panmictic (single-deme, well-mixed)** models. For multi-deme spatial populations with migration topology and heterogeneous parameters, see [Spatial Simulation Guide](3_spatial_simulation.md).

## Population Types

NATAL Core provides two main population types:

### Discrete Generation Population
`DiscreteGenerationPopulation` is suitable for species with non-overlapping generations, where each generation completely replaces the previous one. The simulation process is simple and efficient. It fixes 2 age classes: age-0 (juvenile) and age-1 (adult).

### Age-Structured Population
`AgeStructuredPopulation` is suitable for species with overlapping generations, supporting age-dependent survival and fecundity, and configurable sperm storage mechanisms.

> Both population types are subclasses of `BasePopulation` and share most methods.

## Creating a Population

Use the fluent chain API. The default `PopulationBuilder` path writes parameters immediately.
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

# Custom fields — write in callbacks with ctx.update().custom(...); read pop.config.custom outside
pop.update().custom(temperature=35.0)
```

Each `pop.update()` call is validated against the route table and committed to the running population (draft and Rust session stay in sync), appending one parameter-log row when the value actually changes.

### Low-Level set_param (draft level)

`set_param()` writes only the draft you pass in and never commits to a running population; ecology scalars go through `NamedTuple._replace`, so the return value must be rebound. Use `pop.update()` or `pop.params` to modify a running population:

```python
from natal.frontend.builder import set_param

draft = pop.config                       # query snapshot
draft = set_param(draft, "competition.carrying_capacity", 5000.0)
draft = set_param(draft, "carrying_capacity", 5000.0)      # short name
draft = set_param(draft, "eggs_per_female", 100.0)         # alias
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

Choose the History mode and capacity before `build()`. Raw mode is the default;
the default capacity is the population's bounded `max_history` (5000 rows) —
once the limit is reached the oldest rows are evicted FIFO, and evicted rows
take their paired restore checkpoints with them:

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
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
from natal.frontend.patterns import IndividualSelector

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

# Check whether the last run failed (a failing hook or engine error
# marks the session Failed; restore_checkpoint() or reset() clears it)
if pop.is_failed:
    print("Last run failed")
```

## Wright-Fisher Extreme Speed Mode

The discrete-generation engine ships a Wright-Fisher extreme speed mode: a single multinomial draw per tick replaces the step-by-step mate→fertilize→survive pipeline, aimed at effective population size modeling.

The fused tick is its own model, not an optimized staged pipeline, and the two do not share a sampling variance. The staged stochastic tick draws the offspring genotypes and then applies density regulation by resampling the age-0 cohort, so its per-generation allele-frequency variance is about twice the classical single-draw value (`2 * 2N p (1 - p)` instead of `2N p (1 - p)`): the drift standard deviation is `sqrt(2)` times larger and the staged path's effective population size is half the fused mode's at equal census size. `fixed_egg_count=True` removes clutch noise but not that resampling, and `no_competition` does not skip it either. Compare runs against textbook Wright-Fisher expectations with `extreme_speed_mode=1`, or account for the extra draw when comparing staged runs.

### Sampling Modes

| Mode | Description |
|------|-------------|
| DETERMINISTIC (3) | Infinite population limit, no randomness |
| MULTINOMIAL (1) | Classic Wright-Fisher single multinomial draw |
| POISSON (2) | Independent Poisson draws (large-N approximation) |

`extreme_speed_mode` is selected on the build chain (the low-level factory
construction path has been retired):

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(
        species, stochastic=False, extreme_speed_mode=3
    )
    .initial_state({"female": {"WT|WT": 50}, "male": {"Drive|Drive": 50}})
    .build()
)
```

### Competition and Hooks

All built-in competition modes (FIXED, LOGISTIC/LINEAR, BEVERTON_HOLT, and RICKER) are supported, sharing the same scaling functions as the standard path. Only FIRST hooks are supported (fired before the WF tick); EARLY/LATE hooks have no natural insertion point in the fused WF tick. Deterministic WF mode matches the standard deterministic path tick-by-tick. Embryonic viability is applied before measuring juvenile competition; ordinary survival and genotype viability are applied after density regulation. WF stochastic modes retain their single final-generation sampling step, so their variance need not match the staged stochastic lifecycle.

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

- GType: single-locus with only A|A initially → haplotypes from 2 to 1
- ZType: only A|A reachable → genotypes from 3 to 1 (the default `unordered=True` registry keeps one canonical phase per heterozygote, so a two-allele single-locus species has 3 genotypes, not 4). The offspring tensor is derived during publication on the resulting runtime axes.
- Combined: >98% reduction possible
