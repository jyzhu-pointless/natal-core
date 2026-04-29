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
| **Population.create_observation(...)** | Recommended public entry point for building reusable observation objects |
| **output_current_state / output_history** | Recommended public output API with built-in observation integration |
| **ObservationFilter** | Advanced compilation helper class (only needed for low-level customization workflows) |

### Recommended Workflow

Direct instantiation of `Observation` in business code is discouraged; prefer using the population-layer public methods.

```python
# Create observation rules
observation = pop.create_observation(
    groups={
        "adult_wt": {"genotype": ["WT|WT"], "age": [1]},
        "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    },
    collapse_age=False,
)

# Get current state
current = pop.output_current_state(observation=observation)
print("Current observation data:", current["observed"])

# Get historical data
history = pop.output_history(observation=observation)
print("Historical observation data:", history["observed"])
```

This approach allows reusable observation definitions and defers dimension validity checking to the application/output stage.

## Observation-Based History Recording (Compression Mode)

In large simulations (many genotypes, many demes), the storage cost of full raw history records is extremely high -- each snapshot contains counts for all genotypes. The Observation system can project the genotype dimension onto user-defined groups, performing aggregation directly during the recording stage and only recording the aggregated results, significantly reducing memory usage.

### Comparison of Two Modes

| Mode | Recorded Content | Size per Row | Re-parsing Needed on Export |
|------|------------------|-------------|-----------------------------|
| Raw | `[tick, ind.ravel(), sperm.ravel()]` | `1 + n_sexes x n_ages x n_geno + ...` | Yes, expand by genotype |
| Observation | `[tick, observed.ravel()]` | `1 + n_groups x n_sexes x n_ages` | No, expand directly by group name |

When `n_groups << n_genotypes` (common scenario), the compression ratio is approximately `n_genotypes / n_groups`.

### Configuration Methods

Both methods can activate observation mode:

**Method 1: Create Observation first, then set `record_observation`**

```python
obs = pop.create_observation(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "het": {"genotype": ["WT|Dr"]},
        "hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.record_observation = obs  # Activate observation mode
pop.run(n_steps=100, record_every=10)
```

**Method 2: Directly use `set_observations()`**

```python
pop.set_observations(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "het": {"genotype": ["WT|Dr"]},
        "hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.run(n_steps=100, record_every=10)
```

Once `record_observation` is set, the kernel automatically uses observation aggregation mode during recording. `output_history()` automatically detects and selects the correct export path:

```python
history = pop.output_history()
# Automatically exports in observation mode, each snapshot containing:
# - tick
# - labels: ["wt", "het", "hom"]
# - observed: { "wt": { "female": ..., "male": ... }, ... }
```

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
    .build()
)

# Activate observation mode
pop.set_observations(
    groups={
        "wildtype": {"genotype": ["WT|WT"]},
        "drive_het": {"genotype": ["WT|Dr"]},
        "drive_hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.run(n_steps=100, record_every=10)

# Export -- automatically uses observation mode
history = pop.output_history()
for snap in history["snapshots"]:
    print(f"tick {snap['tick']}: {snap['observed']}")

# You can switch back to raw mode at any time for inspection
pop.record_observation = None  # Disable observation mode
# Subsequent run() calls will use raw recording
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
    .build()
)

# Activate observation mode
spatial.set_observations(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "dr": {"genotype": ["WT|Dr", "Dr|Dr"]},
    },
    collapse_age=True,
)
spatial.run(n_steps=50, record_every=5)

# Export -- expanded per deme, with cross-deme aggregation
history = spatial.output_history()
for snap in history["snapshots"]:
    print(f"tick {snap['tick']}")
    for deme_key, deme_obs in snap["demes"].items():
        print(f"  {deme_key}: {deme_obs}")
    print(f"  aggregate: {snap['aggregate']}")
```

### Post-hoc Observation (Without Modifying Recording Mode)

If you don't want to change the recording mode but need to view history in observation format, you can pass the `observation` parameter:

```python
obs = pop.create_observation(groups={
    "females": {"sex": "female"},
    "males": {"sex": "male"},
})

# Post-hoc observation on already recorded raw history
history = pop.output_history(observation=obs)
# Note: If history is in raw mode (record_observation not set),
# observations are recomputed per snapshot on export (slower but does not require re-running the simulation).
# If history is in observation mode, the compressed data is read directly.
```

### Spatial Deme Selector

In Spatial mode, group specs support the `"deme"` key to control which demes are included in the group:

```python
spatial.set_observations(
    groups={
        "center_release": {
            "genotype": ["Dr|Dr"],
            "deme": [(1, 1)],          # Observe only the center deme
        },
        "all_wt": {
            "genotype": ["WT|WT"],
            "deme": "all",             # All demes (default)
        },
    },
)
```

`"deme"` supports:
- `"all"` or omitted: all demes
- Integer list: flat indices, e.g., `[0, 2, 4]`
- Coordinate list: `(row, col)` tuples, automatically resolved through topology

### When to Use Observation Mode vs Post-hoc

| Scenario | Recommended Approach |
|----------|---------------------|
| Detailed analysis requiring all genotype data | Raw history (default) |
| Only care about time series of a few groups | `record_observation` observation mode |
| Need to inspect history repeatedly with different groupings | Raw history + post-hoc `output_history(observation=obs)` |
| Large-scale spatial (thousands of demes) | `record_observation` observation mode |
| Memory-constrained environments | `record_observation` observation mode |

## History Recording System

### History Recording Configuration

The population object has built-in history recording functionality with configurable recording frequency and storage format:

```python
# Configure history recording
pop.record_every = 10  # Record every 10 steps
pop.max_history = 1000  # Maximum of 1000 snapshots

# Run simulation and record history
results = pop.run(n_steps=500, record_every=5)

# Get history data
history_data = pop.output_history()
print("Number of history records:", len(history_data["snapshots"]))
print("Last step data:", history_data["snapshots"][-1])
```

### Accessing History Data

```python
# Get full history
full_history = pop.output_history()

# Get history at a specific time step
history_at_tick_100 = pop.output_history(tick=100)

# Get list of time steps in history
ticks = [snapshot["tick"] for snapshot in full_history["snapshots"]]
print("Recorded time steps:", ticks)

# Clear history to save memory
pop.clear_history()
```

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
# Get detailed snapshot of current state
current_state = pop.output_current_state()
print("Current state:", current_state)

# Get readable dictionary format
readable_state = pop.output_current_state(as_dict=True)
print("Readable state:", readable_state)

# Get JSON format (suitable for transmission and storage)
json_state = pop.output_current_state(as_json=True)
print("JSON state:", json_state[:200])  # Show first 200 characters
```

### Data Export

```python
# Export as dictionary format
data_dict = pop.output_current_state(as_dict=True)

# Export as JSON format
json_data = pop.output_current_state(as_json=True, indent=2)

# Save to file
import json
with open("population_state.json", "w") as f:
    json.dump(data_dict, f, indent=2)

# Export observation data
observation_data = pop.output_current_state(
    observation=observation,
    as_dict=True
)
```

### Integration with External Tools

```python
import pandas as pd
import numpy as np

# Convert history data to pandas DataFrame
def history_to_dataframe(history_data):
    """Convert history records to DataFrame"""
    data = []
    for snapshot in history_data["snapshots"]:
        row = {
            "tick": snapshot["tick"],
            "total_population": snapshot["total_count"],
            "females": snapshot["female_count"],
            "males": snapshot["male_count"]
        }
        # Add observation data
        if "observed" in snapshot:
            for group_name, count in snapshot["observed"].items():
                row[f"observed_{group_name}"] = count
        data.append(row)

    return pd.DataFrame(data)

# Usage example
history_df = history_to_dataframe(full_history)
print(history_df.head())
```

## Observation Rules in Detail

### Group Format

Observation rules support multiple group formats:

#### 1. Dictionary Format (Named Groups)

```python
groups = {
    "all_females": {"sex": "female"},
    "adults": {"age": [2, 3, 4, 5, 6, 7]},
    "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]},
    "juvenile_drive": {
        "genotype": ["WT|Drive"],
        "age": [0, 1],
        "sex": "female"
    },
}
```

#### 2. Pattern Matching (Recommended)

```python
groups = {
    "target_female": {
        # Ordered matching Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_female_unordered": {
        # Unordered matching (two homologous chromosome copies can be swapped)
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}
```

### Selector Formats

#### Genotype Selector

```python
# String (comma-separated)
{"genotype": "WT|WT"}

# Pattern string (recommended)
{"genotype": "A1/B1|A2/B2; C1/D1::C2/D2"}

# String list
{"genotype": ["WT|WT", "WT|Drive", "Drive|Drive"]}

# Wildcard (all genotypes)
{"genotype": "*"}
```

#### Sex Selector

```python
# String
{"sex": "female"}  # or {"sex": "male"}

# Integer
{"sex": 0}  # Female, {"sex": 1} Male

# List
{"sex": ["female", "male"]}  # Both sexes
```

#### Age Selector

```python
# Explicit list
{"age": [2, 3, 4]}

# Closed interval tuple
{"age": [2, 7]}  # ages 2,3,4,5,6,7

# Interval list (union)
{"age": [[0, 1], [4, 6]]}  # ages 0,1,4,5,6

# Callable
{"age": lambda a: a >= 2}  # age greater than or equal to 2
```

## Practical Examples

### Monitoring Gene Drive Spread

```python
# Create observation rules specifically for monitoring gene drive
drive_observation = pop.create_observation(
    groups={
        "wild_type": {"genotype": ["WT|WT"]},
        "heterozygotes": {"genotype": ["WT|Drive"]},
        "homozygotes": {"genotype": ["Drive|Drive"]},
        "total_drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    }
)

# Run simulation and monitor in real time
for step in range(100):
    pop.run_tick()

    if step % 10 == 0:
        current = pop.output_current_state(observation=drive_observation)
        observed = current["observed"]
        print(f"Step {step}: WT={observed['wild_type']}, "
              f"Het={observed['heterozygotes']}, "
              f"Hom={observed['homozygotes']}")
```

### Age Structure Analysis

```python
# Analyze population dynamics across different age groups
age_observation = pop.create_observation(
    groups={
        "juveniles": {"age": [0, 1]},
        "young_adults": {"age": [2, 3]},
        "mature_adults": {"age": [4, 5]},
        "old_adults": {"age": [6, 7]}
    }
)

# Get history data and analyze
history = pop.output_history(observation=age_observation)

# Analyze age structure changes
for snapshot in history["snapshots"]:
    total = sum(snapshot["observed"].values())
    if total > 0:
        juvenile_ratio = snapshot["observed"]["juveniles"] / total
        print(f"Juvenile ratio: {juvenile_ratio:.3f}")
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
Increase the `record_every` interval, use `clear_history()` for periodic cleanup, or export to external storage.

### Do observation rules affect simulation performance?
Observation rules themselves do not affect simulation performance, but frequent data extraction and storage may impact overall performance.

---

This chapter introduced how to extract and analyze data from NATAL Core simulations. In real projects, it is recommended to first design appropriate observation rules, then choose the suitable data extraction method based on your needs.
