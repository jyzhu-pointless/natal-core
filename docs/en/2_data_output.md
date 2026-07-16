# Extracting Population Simulation Data

This chapter introduces how to extract and analyze data from NATAL Core simulations, including observation rules, history records, and output formats. These features are key components for data analysis, visualization, and statistical inference.

## Data Extraction Overview

NATAL Core provides three main methods for data extraction:

### Observation Rules
Observation rules are used to extract specific subpopulations from the complete population state, supporting flexible grouping and aggregation, suitable for real-time monitoring and statistical analysis.

### History Records
The history recording feature saves state snapshots during the simulation, supporting time series analysis, with configurable recording frequency and storage format.

### Output Formats
The output format system provides multiple data export formats, supporting integration with external tools for subsequent analysis and visualization.

## Observation Rule System

### Core Objects

| Object | Purpose |
|--------|---------|
| **pop.observation** | Immutable rule created at build time; present on every Population |
| **pop.observe()** | Projects current state through the canonical observation |
| **pop.history** | Typed `History` container with an immutable schema |

### Recommended Workflow

Define groups at build time with `with_observation()`. Its argument must be a
non-empty ordered mapping whose keys are non-empty strings and whose values are
`IndividualSelector` instances. A regular `dict` preserves insertion order;
that order becomes the result's group-axis order.

```python
import natal as nt

pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .with_observation(
        {
            "adult_wt": nt.IndividualSelector(ztype="WT|WT", age=[1]),
            "drive_carriers": (
                nt.IndividualSelector(ztype="WT|Drive")
                | nt.IndividualSelector(ztype="Drive|Drive")
            ),
        },
        collapse_age=False,
    )
    .record_history(mode="raw")
    .build()
)

# Project current state; group is always the first axis
current = pop.observe()
print(current.axes)                 # ("group", "sex", "age")
print(current.labels["group"])
print(current.values)

# Raw History can be projected later through the same rule
observed_history = pop.history.observe(pop.observation)
print(observed_history.values.shape)  # (record, group, sex, age)
```

If `with_observation()` is omitted, the build installs an identity observation:
one lossless group per ZType. Thus `pop.observation` and `pop.observe()` are
always available on panmictic, age-structured, discrete-generation, and spatial
Populations; they do not return `None`.

## Observation-Based History Recording (Compression Mode)

In large simulations (many genotypes, many demes), the storage cost of full raw history records is extremely high -- each snapshot contains counts for all genotypes. The Observation system can project the genotype dimension onto user-defined groups, performing aggregation directly during the recording stage and only recording the aggregated results, significantly reducing memory usage.

### Comparison of Two Modes

| Mode | Typed `History` data | Typical panmictic shape | Can be re-observed later? |
|------|----------------------|---------------------------|---------------------------|
| Raw (default) | `individual_count`, optional `sperm_storage` | `(record, sex, age, ztype)` | Yes |
| Observation | `values` | `(record, group, sex, age)` | No; original ZTypes were discarded |

When `n_groups << n_genotypes` (common scenario), the compression ratio is approximately `n_genotypes / n_groups`.

### Configuration Methods

Observation rules and History mode are configured only at build time:

**Method 1: Build-time — `with_observation()` + `record_history(mode="observation")` (recommended)**

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(groups={
        "wt": nt.IndividualSelector(ztype="WT|WT"),
        "het": nt.IndividualSelector(ztype="WT|Dr"),
        "hom": nt.IndividualSelector(ztype="Dr|Dr"),
    }, collapse_age=True)
    .record_history(mode="observation", max_rows=5000)
    .build()
)
pop.run(n_steps=100, record_every=10)
```

`record_history()` and `with_observation()` are **independent** — chain order
does not matter. When `mode="observation"` is set without explicit
`with_observation()`, an identity observation is auto-generated.

**Method 2: Build-time — auto-identity observation (no explicit groups)**

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .record_history(mode="observation")  # auto-generates identity observation
    .build()
)
```

No `with_observation()` needed — one group per ZType is automatically created,
providing lossless projection.

Observation rules are frozen at Population build time via `with_observation()`
and cannot be modified at runtime. This guarantees that all records in the same
History have consistent semantics.

Use `pop.history` for direct typed history access.

### Panmictic Example

```python
import natal as nt

species = nt.Species.from_dict(
    name="demo",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="obs_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Dr|WT": 50},
        "male": {"WT|WT": 500, "Dr|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "wildtype": nt.IndividualSelector(ztype="WT|WT"),
            "drive": (
                nt.IndividualSelector(ztype="WT|Dr")
                | nt.IndividualSelector(ztype="Dr|Dr")
            ),
        },
        collapse_age=True,
    )
    .record_history(mode="observation")
    .build()
)

pop.run(n_steps=100, record_every=10)

# Current observation and History use the same canonical observation
print(pop.observe().axes)        # ("group", "sex")
print(pop.history.ticks)
print(pop.history.values.shape)  # (record, group, sex)
```

### Spatial Example

```python
from natal import SpatialPopulation, HexGrid
import numpy as np

species = nt.Species.from_dict(
    name="spatial_obs",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

kernel = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

spatial = (
    SpatialPopulation.builder(species, n_demes=9, topology=HexGrid(3, 3))
    .setup(name="spatial_obs_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500}, "male": {"WT|WT": 500},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=kernel, migration_rate=0.2)
    .with_observation(
        {
            "wt": nt.IndividualSelector(ztype="WT|WT"),
            "dr": (
                nt.IndividualSelector(ztype="WT|Dr")
                + nt.IndividualSelector(ztype="Dr|Dr")
            ),
        },
        collapse_age=True,
        demes=[2, 0],
        deme_mode="preserve",
    )
    .record_history(mode="observation")
    .build()
)

spatial.run(n_steps=50, record_every=5)

# preserve keeps the shared deme axis in demes=[2, 0] order
print(spatial.observe().axes)        # ("group", "deme", "sex")
print(spatial.history.values.shape)  # (record, group, 2, sex)
```

### Post-hoc Observation (Without Modifying Recording Mode)

Raw mode preserves complete state, so the canonical observation can be applied
after the simulation without changing the raw History:

```python
# Apply post-hoc observation to already-recorded raw history
observed_history = pop.history.observe(pop.observation)

# This returns a new observation-mode History; pop.history remains raw
print(observed_history.values.shape)
print(pop.history.individual_count.shape)
```

`History.observe()` compares the observation's Population-layout fingerprint
with the History layout. It raises `ValueError` for an observation from another
layout even when array shapes happen to match, preventing a grouping rule from
being silently applied to the wrong species/ZType layout. Observation-mode
History has already discarded original ZType data and cannot be projected again.

### Spatial History Axes

Spatial `with_observation()` uses `demes` to define one ordered deme set shared
by every group. `deme_mode="preserve"` retains that shared axis, while
`"aggregate"` sums and removes it. `demes=None` selects every deme by default.
Public array shapes are:

| Data/mode | `collapse_age=False` | `collapse_age=True` |
|-----------|----------------------|---------------------|
| `spatial.observe().values`, preserve | `(group, selected_deme, sex, age)` | `(group, selected_deme, sex)` |
| `spatial.observe().values`, aggregate | `(group, sex, age)` | `(group, sex)` |
| raw `spatial.history.individual_count` | `(record, deme, sex, age, ztype)` | Not applicable; raw mode never collapses age |
| observation `spatial.history.values`, preserve | `(record, group, selected_deme, sex, age)` | `(record, group, selected_deme, sex)` |
| observation `spatial.history.values`, aggregate | `(record, group, sex, age)` | `(record, group, sex)` |

Raw History always stores every deme, regardless of the Observation selection
or aggregation mode.
Even when a spatial Population has only one deme, raw arrays retain a length-one
deme axis; only non-spatial Populations omit that axis.

### When to Use Observation Mode vs Post-hoc

| Scenario | Recommended Approach |
|----------|---------------------|
| Detailed analysis requiring all genotype data | Raw history (default) |
| Only care about time series of a few groups | `record_history(mode="observation")` |
| Need complete state and post-hoc projection | Raw history + `history.observe(pop.observation)` |
| Large-scale spatial (thousands of demes) | `record_history(mode="observation")` |
| Memory-constrained environments | `record_history(mode="observation")` |

## History Recording System

### Recording Mode and Capacity

The Configurator provides `record_history()` to set the recording mode and
capacity during the build phase. This method is **independent** of
`with_observation()` — chain order does not matter.

```python
# Build-time: configure recording mode and capacity
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(individual_count={"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .record_history(mode="observation", max_rows=5000)  # observation mode, FIFO limit
    .build()
)
```

When `mode="observation"` is set without an explicit `with_observation()`, an
**identity observation** (one group per ZType) is automatically generated,
providing lossless projection without requiring manual group definitions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"raw"` | `"raw"` for full-state recording; `"observation"` for compressed aggregate recording |
| `max_rows` | `None` | Maximum snapshots to keep (FIFO eviction). `None` = unlimited |

### Runtime Recording Configuration

The population also exposes runtime recording controls for backward compatibility:

```python
pop.record_every = 10  # Record every 10 steps
pop.max_history = 1000  # Maximum of 1000 snapshots (legacy)
```

The recording schema (mode, row size, layout) is **frozen at build time** and
cannot change after the first row is recorded. Once configured via the
Configurator, `pop.record_every` and `pop.max_history` only control the
recording **frequency** and **legacy limit**, not the schema.

```python
# Run simulation and record history
results = pop.run(n_steps=500, record_every=5)

# Get typed History data
history = pop.history
print("Number of history records:", history.n_records)
print("Recording mode:", history.schema.mode)
```

### Accessing History Data

```python
# Every Population, including SpatialPopulation, owns one History container
history = pop.history
ticks = history.ticks
print("Recorded time steps:", ticks)

# Raw mode exposes complete individual counts
if history.schema.mode == "raw":
    print(history.individual_count.shape)
    print(history.sperm_storage)  # None for discrete-generation populations
else:
    print(history.values.shape)

# Clear history to save memory
pop.clear_history()
```

### pop.observation — Read-Only Observation Property

Every Population exposes its canonical build-time observation through a
read-only property:

```python
obs = pop.observation
print(f"Groups: {obs.labels}")
print(f"Collapse age: {obs.collapse_age}")
```

When no rule is configured explicitly, this is an identity observation with
one group per ZType. It is frozen at build time and cannot be changed later.

### pop.observe() — Project Current State

Project the current population state through the configured observation and
return a structured `ObservationResult`:

```python
result = pop.observe()
print(f"Tick: {result.tick}")
print(f"Axes: {result.axes}")       # ("group", "sex", "age")
print(f"Values shape: {result.values.shape}")
print(f"Group labels: {result.labels['group']}")
```

Group is always the first axis. With `collapse_age=True`, age is summed and the
age axis is removed. For spatial results, `deme_mode="preserve"` retains deme
immediately after group, while `"aggregate"` sums selected demes and removes
that axis. Raises `RuntimeError` if the population has no state yet.

### pop.record_snapshot() — Manual Recording

Manually record the current stable state into history outside of `run()`:

```python
pop.run_tick()
pop.record_snapshot()  # manually record after a single tick
```

Call it at a stable boundary between `run()` calls. If the current tick is
already recorded, it raises `ValueError`. It raises `RuntimeError` on a
finished population.

### pop.restore_checkpoint(tick) — State Restoration

Restore population state from a raw-mode history record at a specific tick.
All records after that tick are removed:

```python
# Record raw history during simulation
pop.run(n_steps=100, record_every=1)

# Restore to tick 50
pop.restore_checkpoint(tick=50)
# Population state is now identical to what it was at tick 50
# All history records after tick 50 are discarded
```

Only valid for raw-mode history (`mode="raw"`). Raises `ValueError` when
mode is `"observation"` — observation-mode history does not retain the
full-state data needed for restoration, so record raw history if you plan
to use checkpoints.

### History Data Analysis

```python
# Analyze allele frequency changes over time
allele_freq_history = []
for snapshot in full_history["snapshots"]:
    # Recompute allele frequency for each time step
    # This needs to be adjusted based on the actual data structure
    freq = calculate_allele_frequency(snapshot)
    allele_freq_history.append(freq)

# Plot time series
import matplotlib.pyplot as plt
plt.plot(ticks, allele_freq_history)
plt.xlabel("Time Step")
plt.ylabel("Allele Frequency")
plt.show()
```

## Output Format System

### Current State Output

```python
# Project current population state through the canonical observation
result = pop.observe()
print("Tick:", result.tick)
print("Axes:", result.axes)
print("Labels:", result.labels)
print("Values shape:", result.values.shape)
print("Values:", result.values)
```

### Data Export

```python
import json

# Get current projected state as a structured result
result = pop.observe()

# Convert to a JSON-serializable dictionary
data_dict = {
    "tick": result.tick,
    "axes": list(result.axes),
    "labels": {k: list(v) for k, v in result.labels.items()},
    "values": result.values.tolist(),
}

# Save to JSON file
with open("population_state.json", "w") as f:
    json.dump(data_dict, f, indent=2)
```

### Integration with External Tools

```python
import pandas as pd

# Convert observation-mode history to pandas DataFrame
def history_to_dataframe(observed_history):
    """Convert observed history records to DataFrame"""
    data = []
    group_labels = observed_history.labels["group"]
    for i, tick in enumerate(observed_history.ticks):
        row = {
            "tick": tick,
            "total_population": observed_history.values[i].sum(),
        }
        for j, group in enumerate(group_labels):
            row[group] = observed_history.values[i, j].sum()
        data.append(row)
    return pd.DataFrame(data)

# Usage example
observed = pop.history.observe(pop.observation)
history_df = history_to_dataframe(observed)
print(history_df.head())
```

## Observation Rules in Detail

### Group Format

Groups must be defined using `IndividualSelector` instances passed to
`.with_observation()` at build time. Each key in the groups mapping becomes
a group label, and its value is an `IndividualSelector` that selects the
individuals belonging to that group.

`IndividualSelector` accepts the following keyword-only arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `ztype` | `str` | Diploid genotype string (e.g. `"WT|Dr"`) |
| `gtype` | `str` | Haploid genotype string |
| `sex` | `str` or `int` | `"female"`, `"male"`, or `0` / `1` |
| `age` | `range`, `int`, or sequence of `int` | Age or age interval |

Selectors can be combined with `|` (union) and `+` (intersection) operators:

```python
# Union — individuals matching either selector
combined = nt.IndividualSelector(ztype="WT|Dr") | nt.IndividualSelector(ztype="Dr|Dr")

# Intersection — individuals matching both selectors
both = nt.IndividualSelector(sex="female") + nt.IndividualSelector(age=range(2, 5))
```

### Grouping Examples

```python
# Single genotype group
{"wt": nt.IndividualSelector(ztype="WT|WT")}

# Age range group
{"adults": nt.IndividualSelector(age=range(2, 8))}

# Combined criteria
{"juvenile_female": nt.IndividualSelector(sex="female", age=range(0, 2))}

# Union of genotypes
{"drive_carriers": (
    nt.IndividualSelector(ztype="WT|Drive")
    | nt.IndividualSelector(ztype="Drive|Drive")
)}

# Wildcard — all genotypes (identity group)
{"all": nt.IndividualSelector()}
```

## Practical Examples

### Monitoring Gene Drive Spread

```python
import natal as nt

species = nt.Species.from_dict(
    name="drive_monitor",
    structure={"chr1": {"loc": ["WT", "Drive"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="drive_monitor", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Drive|WT": 50},
        "male": {"WT|WT": 500, "Drive|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "wild_type": nt.IndividualSelector(ztype="WT|WT"),
            "heterozygotes": nt.IndividualSelector(ztype="WT|Drive"),
            "homozygotes": nt.IndividualSelector(ztype="Drive|Drive"),
            "total_drive": (
                nt.IndividualSelector(ztype="WT|Drive")
                | nt.IndividualSelector(ztype="Drive|Drive")
            ),
        },
        collapse_age=True,
    )
    .build()
)

for step in range(100):
    pop.run_tick()
    if step % 10 == 0:
        result = pop.observe()
        group_map = {
            name: i for i, name in enumerate(result.labels["group"])
        }
        values = result.values
        print(f"Step {step}: "
              f"WT={values[group_map['wild_type']].sum():.0f}, "
              f"Het={values[group_map['heterozygotes']].sum():.0f}, "
              f"Hom={values[group_map['homozygotes']].sum():.0f}")
```

### Age Structure Analysis

```python
import natal as nt

species = nt.Species.from_dict(
    name="age_analysis",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="age_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Dr|WT": 50},
        "male": {"WT|WT": 500, "Dr|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "juveniles": nt.IndividualSelector(age=range(0, 2)),
            "young_adults": nt.IndividualSelector(age=range(2, 4)),
            "mature_adults": nt.IndividualSelector(age=range(4, 6)),
            "old_adults": nt.IndividualSelector(age=range(6, 8)),
        },
        collapse_age=True,
    )
    .record_history(mode="raw")
    .build()
)

pop.run(n_steps=100, record_every=1)

# Project raw history through the canonical observation
observed = pop.history.observe(pop.observation)
for i, tick in enumerate(observed.ticks):
    values = observed.values[i]  # (group, sex)
    total = float(values.sum())
    if total > 0:
        group_labels = observed.labels["group"]
        juv_idx = group_labels.index("juveniles")
        juvenile_ratio = values[juv_idx].sum() / total
        print(f"Tick {tick}: juvenile ratio = {juvenile_ratio:.3f}")
```

## Best Practices

### Observation Rule Design
- Use meaningful group names for easier subsequent analysis
- Keep groups mutually exclusive to avoid double counting
- Prefer pattern matching over hardcoded genotype lists

### History Record Management
- Set an appropriate `record_every` parameter to balance precision and performance
- Use `clear_history()` to manage memory usage
- Regularly export history data to avoid data loss

### Data Export
- Use standard formats (JSON, dictionary) for easier tool integration
- Include sufficient metadata (time steps, parameter settings, etc.)
- Consider data compression and storage efficiency

## FAQ

### What is the difference between observation rules and history records?
Observation rules define how to extract data from the state; history records save the time series of states. Observation rules can be applied to the current state or to history records.

### How to optimize history records for large datasets?
Increase the `record_every` interval, use `clear_history()` for periodic cleanup, or export to external storage. Note that `clear_history()` preserves the recording schema — you can clear data and continue recording without reconfiguring.

### Do observation rules affect simulation performance?
Observation rules themselves do not affect simulation performance, but frequent data extraction and storage may impact overall performance.

### Can I change recording rules after building the Population?
No. The canonical observation and History schema are frozen by `build()`.
Both `pop.update().with_observation(...)` and
`pop.update().record_history(...)` raise `RuntimeError`. At runtime, read
`pop.observation`, call `pop.observe()`, or call
`pop.history.observe(pop.observation)` on raw History.

### What's the difference between `record_history()` and `with_observation()`?
`with_observation()` defines *which groups* to observe (the observation projection). `record_history()` sets *how to record* — raw full-state or compressed observation-mode. They are independent: you can have observation groups without compressed recording, or compressed recording without explicit groups (auto-identity).

### Can I restore my population to a previous state?
Yes, if you recorded raw history (`mode="raw"`), use `pop.restore_checkpoint(tick)`. It restores individual counts (and sperm storage when applicable) to the exact state at that tick. Observation-mode history cannot be used for checkpoint restoration because it does not retain per-genotype data.

---

This chapter introduced how to extract and analyze data from NATAL Core simulations. In real projects, it is recommended to first design appropriate observation rules, then choose the suitable data extraction method based on your needs.
