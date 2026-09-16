# Spatial Simulation Guide

This chapter introduces the practical usage of `SpatialPopulation`: using the `SpatialPopulationBuilder` to quickly build multi-deme populations, configure topology and migration kernels, and control inter-deme flow.

After reading this, you will be able to write code like this:

```python
spatial = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=my_kernel, migration_rate=0.15)
    .build()
)
```

> **Tip**: `SpatialPopulationBuilder` is the preferred construction method for homogeneous/heterogeneous spatial populations (build one template, clone the rest). See [SpatialPopulationBuilder Documentation](spatial_population_builder.md).

## Two Construction Paths

### Recommended: SpatialPopulationBuilder (Chainable API)

```python
from natal import Species, HexGrid, SquareGrid, SpatialPopulation
from natal.frontend.spatial import batch_setting

species = Species.from_dict(name="demo", structure={"chr1": {"loc": ["A", "B"]}})

# Homogeneous: all demes have the same parameters
pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    .setup(name="homo_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(migration_rate=0.1)
    .build()
)

# Heterogeneous: specify different parameters for different demes via batch_setting
pop_het = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="het_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000]))
    .migration(migration_rate=0.1)
    .build()
)
```

### Manual Construction (Compatibility Path)

If you already have an independently constructed list of demes, you can pass them directly to the `SpatialPopulation` constructor. All demes must share the same Species object:

```python
from natal.frontend.spatial import SpatialPopulation
from natal.frontend.spatial import SquareGrid

shared_config = demes[0].export_config()
for deme in demes[1:]:
    deme.import_config(shared_config)

spatial = SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=2, cols=2),
    migration_rate=0.15,
)
```

## Core Parameters of SpatialPopulation

The `SpatialPopulation` constructor supports these most commonly used parameters:

- `demes`: Pre-built list of demes.
- `topology`: Optional grid topology, commonly `SquareGrid` or `HexGrid`.
- `adjacency`: Explicit adjacency matrix; if not provided, it is typically derived from `topology`.
- `migration_kernel`: Migration kernel, used when following the kernel path.
- `kernel_bank`: Optional collection of kernels, used when different source demes use different kernels.
- `deme_kernel_ids`: Optional per-deme kernel ids, indexing into `kernel_bank`.
- `migration_rate`: Per-deme proportion of individuals migrating per step. A scalar applies only to adult ages (>= `new_adult_age` from config); juveniles default to 0. A `(n_ages,)` array sets explicit per-age rates, an `(n_sexes, n_ages)` table or a per-sex mapping sets rates per sex, and an `(n_demes, n_sexes, n_ages)` column (or `(n_demes, n_ages)`, broadcast across sexes) sets each deme directly. `batch_setting` gives one declaration per deme. Migration also needs outbound targets: with no `topology`, `kernel`, or `adjacency` the resolved adjacency is the identity matrix, so a non-zero rate moves nobody. The constructor warns when a multi-deme layout has no inter-deme edge at all; a layout where any deme does connect stays silent, so deliberately isolated demes are not flagged. A single-deme population has no inter-deme edges by construction and is never flagged.
- `migration_strategy`: `auto`, `adjacency`, `kernel`, or `hybrid`; default is `auto`.
- `kernel_include_center`: Whether to include the center cell as a migration target in the kernel path, default `False`.
- `adjust_migration_on_edge`: Legacy bit-parity flag, default `False`. It does not change the destination distribution (up to ~1 ulp of floating-point rounding): outbound rows are normalized to relative weights either way, so the denominator it selects cancels (see "migration_rate and Boundary Effects").

The most important rules:

1. Pass `adjacency` to use the adjacency matrix path.
2. Pass `migration_kernel` to use the kernel path, and topology must also be present.
3. Pass `kernel_bank` + `deme_kernel_ids` to also use the kernel path (heterogeneous kernel).
4. `hybrid` is reserved for a combined adjacency+kernel mixed strategy, and is not required for heterogeneous kernels.

## Chainable API

The `SpatialPopulationBuilder` chainable call flow is consistent with the panmictic builder. Below are the methods listed in recommended order. Methods marked with `->` are spatial-specific, and parameters marked with `[B]` accept `batch_setting` (cross-deme heterogeneous configuration).

```python
pop = (
    SpatialPopulation.builder(species, n_demes=9, topology=SquareGrid(3, 3))
    ->                   # Entry: specify deme count and topology
    .setup(name="demo", stochastic=False, continuous_sampling=False)
                        # Basic settings: name, stochasticity, sampling mode
    .age_structure(n_ages=8, new_adult_age=2)
                        # [age_structured only] Age group count, adult starting age
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
                        # [B] Initial genotype distribution
    .survival(female_age_based_survival=[...], ...)
                        # Survival rates (age_structured uses age vectors, discrete uses scalars)
    .reproduction(eggs_per_female=50.0, sex_ratio=0.5)
                        # [B] Reproduction parameters
    .competition(carrying_capacity=10000, juvenile_growth_mode="logistic")
                        # [B] Density dependence
    .presets(HomingDrive(name="Drive", ...))
                        # [B] Gene drive preset
    .fitness(viability={"R2|R2": 0.0}, mode="replace")
                        # [B] Fitness
    .hooks(my_hook)
                        # Lifecycle hooks (does not accept batch_setting)
    .migration(kernel=kernel, migration_rate=0.2)
    ->                   # [B] Spatial-specific: migration kernel, migration rate
    .build()            # -> SpatialPopulation
)
```

Detailed parameter descriptions for each method can be found in [Population Initialization](2_population_initialization.md) (setup, initial_state, survival, reproduction, competition), [Hook System](2_hooks.md), and [Gene Drive Presets](2_genetic_presets.md).

Both spatial engines apply an explicit `.reproduction(fixed_egg_count=True/False)`
setting. Omitting the parameter or passing `None` preserves the `.setup()` value
(initially `False`).

### Spatial-Specific: `.migration()`

```python
.migration(
    kernel=None,                     # [B] NDArray: odd-dimension migration kernel
    migration_rate=0.0,             # [B] float | NDArray | Sequence | dict: per-deme migration proportion
    strategy="auto",                # "auto" | "adjacency" | "kernel" | "hybrid"
    adjacency=None,                 # Relative outbound weights (rows are normalized)
    kernel_bank=None,               # Heterogeneous kernel collection
    deme_kernel_ids=None,           # Per-deme kernel index
    kernel_include_center=False,    # Whether to include the center cell
    adjust_migration_on_edge=False, # Legacy bit-parity no-op
)
```

`kernel` accepts `batch_setting`. Passing a per-deme kernel list automatically converts it to `kernel_bank` + `deme_kernel_ids`, equivalent to manually specifying heterogeneous kernels. `kernel_bank` / `deme_kernel_ids` are mutually exclusive with `batch_setting`.

See the "Migration Paths" and "migration_rate and Boundary Effects" sections for details.

### Parameters Supporting `[B]` Overview

| Method | Parameter | Type |
|--------|-----------|------|
| `initial_state` | `individual_count` | dict (genotype -> count) |
| `initial_state` | `sperm_storage` | dict |
| `reproduction` | `eggs_per_female` | float |
| `reproduction` | `sex_ratio` | float |
| `competition` | `carrying_capacity` / `age_1_carrying_capacity` | float |
| `competition` | `low_density_growth_rate` | float |
| `competition` | `juvenile_growth_mode` | str |
| `competition` | `expected_num_new_adult_females` | float |
| `age_structure` | `equilibrium_distribution` | list[float] |
| `presets` | positional arguments | preset object |
| `fitness` | `viability` / `fecundity` / `sexual_selection` / `zygote_viability` | dict |
| `migration` | `kernel` | NDArray |
| `migration` | `migration_rate` | float / NDArray / mapping |

The following parameters do **not** accept `batch_setting`:
- **hooks**: Per-deme selective execution is achieved via `.hooks(..., deme=...)`.
- **Spatial functions require topology**: `(row, col)` form requires the builder to have been given a `topology`. The `(flat_idx)` form does not depend on topology.

## batch_setting Heterogeneous Configuration

`batch_setting` is the core mechanism of `SpatialPopulationBuilder`, allowing different demes to specify different parameter values within the same chainable call. Internally, it automatically optimizes through config equivalence grouping -- demes with the same parameters share compiled artifacts, only the state arrays are independent.

### Four Input Forms

```python
from natal.frontend.spatial import batch_setting
import numpy as np

# 1. Scalar list (one-to-one correspondence with n_demes demes)
batch_setting([10000, 5000, 5000, 8000])

# 2. 1D NumPy array
batch_setting(np.array([10000, 5000, 5000, 8000]))

# 3. 2D NumPy array (shape = (rows, cols), flattened in row-major order)
batch_setting(np.array([[10000, 5000],
                         [5000, 8000]]))

# 4. Spatial function: (flat_idx) -> float or (row, col) -> float
batch_setting(lambda i: 10000 if i < 4 else 5000)
batch_setting(lambda r, c: 10000 if r == 0 else 5000)
```

Spatial functions auto-detect based on the number of parameters: 1 parameter receives `(flat_idx)`, 2 parameters receive `(row, col)`. Requires the builder to have been given a `topology` parameter; evaluation occurs at `build()` time.

### Pattern 1: Heterogeneous Carrying Capacity

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="het_K", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000]))
    .migration(migration_rate=0.1)
    .build()
)
# deme 0: K=10000, deme 1: K=5000, deme 2: K=5000, deme 3: K=8000
# builder auto-groups: {10000: [0], 5000: [1,2], 8000: [3]} -> 3 templates
```

### Pattern 2: Heterogeneous Initial State

Specify different initial genotype distributions for each deme, commonly used in spatial drive release scenarios:

```python
import numpy as np
from natal import Species, HexGrid, SpatialPopulation, HomingDrive
from natal.frontend.spatial import batch_setting

drive_species = Species.from_dict(
    name="spatial_drive_release",
    structure={"chr1": {"drive": ["WT", "Dr", "R1", "R2"]}},
)
drive_kernel = np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])

# Default: all demes have only WT
n_demes = 100
default_state = {"female": {"WT|WT": 500}, "male": {"WT|WT": 500}}

# Center deme releases drive heterozygotes
release_state = {"female": {"WT|WT": 450, "Dr|WT": 50},
                 "male":   {"WT|WT": 450, "Dr|WT": 50}}

states = [default_state] * n_demes
states[n_demes // 2] = release_state

pop = (
    SpatialPopulation.builder(drive_species, n_demes=n_demes, topology=HexGrid(10, 10))
    .setup(name="drive_release", stochastic=True, continuous_sampling=True)
    .initial_state(individual_count=batch_setting(states))
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6,
                 juvenile_growth_mode="beverton_holt")
    .presets(HomingDrive(name="Drive", drive_allele="Dr", target_allele="WT",
                         resistance_allele="R2", functional_resistance_allele="R1",
                         drive_conversion_rate=0.95))
    .fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}})
    .migration(kernel=drive_kernel, migration_rate=0.2)
    .build()
)
```

### Pattern 3: Multiple batch Parameters Combined

When multiple `batch_setting` parameters are used simultaneously, the builder computes signatures from parameter value tuples and groups accordingly:

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="multi_het", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=batch_setting([50, 50, 30, 30]))
    .competition(
        carrying_capacity=batch_setting([10000, 5000, 10000, 5000]),
        low_density_growth_rate=batch_setting([6, 6, 4, 4]),
    )
    .migration(migration_rate=0.1)
    .build()
)
# Signature grouping:
#   deme 0: (eggs=50, K=10000, r=6)
#   deme 1: (eggs=50, K=5000,  r=6)
#   deme 2: (eggs=30, K=10000, r=4)
#   deme 3: (eggs=30, K=5000,  r=4)
# -> 4 independent groups, each group builds one template
```

### Pattern 4: Spatial Gradient Function

Use `lambda` to create smooth spatial gradients (e.g., north-south gradient, center-edge gradient):

```python
# Center-high, edge-low carrying capacity gradient -- using (row, col) two-parameter signature
def capacity_gradient(r, c):
    center_r, center_c = 4.5, 4.5  # Center of 10x10 grid
    dist = ((r - center_r)**2 + (c - center_c)**2) ** 0.5
    max_dist = (center_r**2 + center_c**2) ** 0.5
    return 10000 * (1 - 0.8 * dist / max_dist)  # Drops to 2000 at edges

pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    .setup(name="gradient", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting(capacity_gradient))
    .migration(migration_rate=0.1)
    .build()
)
```

### Pattern 5: Heterogeneous Fitness

Different demes can have different fitness configurations, commonly used in spatially differentiated selection pressure scenarios:

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="het_fitness", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .fitness(viability=batch_setting([
        {"A|A": 1.0},   # deme 0: normal
        {"A|A": 0.5},   # deme 1: A|A semi-lethal
        {"A|A": 0.0},   # deme 2: A|A fully lethal
        {"A|A": 1.0},   # deme 3: normal
    ]))
    .migration(migration_rate=0.1)
    .build()
)
# demes 0 and 3 have the same signature -> share one config
# demes 1 and 2 each rebuild independently
```

### Pattern 6: Heterogeneous Migration Kernels

Specify different migration kernels for different demes via `batch_setting`, automatically converted to `kernel_bank` + `deme_kernel_ids`:

```python
import numpy as np

# Two asymmetric kernels: rightward and leftward
right_kernel = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
left_kernel  = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(1, 4))
    .setup(name="het_kernel", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=batch_setting([right_kernel, left_kernel, right_kernel, left_kernel]),
               migration_rate=0.5)
    .build()
)
# Equivalent to:
#   .migration(kernel_bank=(right_kernel, left_kernel),
#              deme_kernel_ids=np.array([0, 1, 0, 1]),
#              migration_rate=0.5)
```

## Migration Paths

### Kernel Path

```python
import numpy as np
from natal import Species, SpatialPopulation, HexGrid

species = Species.from_dict(name="hex_demo", structure={"chr1": {"loc": ["A", "B"]}})

spatial = (
    SpatialPopulation.builder(species, n_demes=10000, topology=HexGrid(100, 100))
    .setup(name="SpatialHexDemo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(
        kernel=np.array(
            [[0.00, 0.10, 0.05],
             [0.10, 0.00, 0.10],
             [0.05, 0.10, 0.00]],
            dtype=np.float64,
        ),
        kernel_include_center=False,
        migration_rate=0.2,
        adjust_migration_on_edge=False,
    )
    .build()
)
```

### Heterogeneous Kernels (Kernel Bank)

Different source demes can use different migration kernels. This is achieved via `kernel_bank` + `deme_kernel_ids`:

```python
right_only = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
left_only  = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

spatial = SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=1, cols=3),
    kernel_bank=(right_only, left_only),
    deme_kernel_ids=np.array([0, 1, 0], dtype=np.int64),
    migration_rate=1.0,
)
```

Each source deme selects its own kernel via `deme_kernel_ids[src]`. Internally, offset tables are built grouped by kernel. During migration, the lookup is done via `deme_kernel_ids[src]` inside a `prange` loop -- no dense adjacency matrix of size `O(n_demes^2)` is pre-built.

## Running Simulations

`SpatialPopulation` inherits all runtime interfaces from `BasePopulation`. The semantics are consistent with panmictic populations, but operations apply to all demes.

### Single-Step and Batch Running

```python
# Single-step advancement
pop.run_tick()

# Batch run 100 steps
pop.run(100)

# Batch run with recording
pop.run(500, record_every=5)
```

`SpatialPopulation.run()`'s `record_every` parameter controls the history sampling interval within the execution kernel. Setting it to 0 disables history recording.

### Accessing Aggregate State

```python
# Cross-deme aggregation
pop.get_total_count()       # Total individual count
pop.get_female_count()      # Total female count
pop.get_male_count()        # Total male count
pop.get_female_count() / pop.get_male_count()   # Sex ratio (female/male)
pop.tick                    # Current time step

# Allele frequencies (full spatial aggregation)
freqs = pop.compute_allele_frequencies()

# Aggregate individual count tensor (summed over all demes)
aggregate = pop.aggregate_individual_count()
```

### Accessing Individual Demes

```python
# Get deme by index
deme_0 = pop.deme(0)
print(deme_0.get_total_count())
print(pop.compute_allele_frequencies())   # allele frequencies are a container query

# Iterate over all demes
for i in range(pop.n_demes):
    d = pop.deme(i)
    print(f"deme {i}: {d.get_total_count()}")
```

Each deme is accessed through a `DemeSlice` view whose surface is aligned
with `Population`: reads (`name`, `species`, `config`, `state`, `params`,
`params_log`, `index_registry`, `presets`, `definition`), queries
(`get_total_count`, `get_female_count`, `get_male_count`, `export_config`,
`export_state`), `update()` (committing through the parent spatial
session), plus the deme-specific `index`, `write_ecology`, and
`write_genetics`. Unlisted attributes raise `AttributeError`. The spatial
container itself provides the canonical `observation`, `observe()`, and
typed `history` interfaces.

### Reset and Control

```python
# Reset all demes to their initial state (clears finished marks)
pop.reset()

# Finish the shared run without advancing its clock
pop.run(0, finish=True)
```

Neither the container nor the aligned deme surface exposes
`is_finished` / `finish_simulation()`: finished state is owned by the
shared native session, and once any deme has finished, `run()` /
`run_tick()` raise `RuntimeError`; when a hook requests a stop, the
session as a whole enters the Stopped state.

Managed deme handles expose only the aligned surface above: lifecycle and
container controls such as ``run``, ``reset``, ``restore_checkpoint``,
``finish``, ``clone``, ``trigger_event``, ``history``, and ``observe`` do
not exist on them (access raises ``AttributeError``). These operations must
use the spatial container; initial states belong to builder declarations,
and a mid-run state change uses the selected deme's ``TickContext.state``.

### Data Output

The default spatial Observation is identity (one group per ZType) and preserves
the complete deme axis. The default History records raw state:

```python
current = pop.observe()
print(current.axes)  # ("group", "deme", "sex", "age")

history = pop.history
print(history.individual_count.shape)
# (record, deme, sex, age, ztype)

# Raw History can be projected later through the canonical observation
observed_history = history.observe(pop.observation)
print(observed_history.values.shape)
# (record, group, deme, sex, age)
```

With `collapse_age=True` at build time, both `observe().values` and
observation-mode `history.values` remove the final age axis.
Spatial `with_observation()` can also use `demes` to select one ordered deme set
shared by every group, and `deme_mode="aggregate"` to sum and remove the deme
axis. Raw History still stores every deme. `with_observation()` and
`record_history()` are independent and build-time only.

For detailed usage, see [Extracting Population Simulation Data](2_data_output.md).

### Runtime Internal Flow

The internal execution order of each `run_tick()`:

1. Check the shared session's execution state (a finished run is rejected).
2. The session owns the stacked state and the config bank; the run stays inside one engine session.
3. Spatial lifecycle: each deme's lifecycle executes at the deme granularity -> unified migration.
4. The updated state remains session-owned; Python reads derive it on demand.

If a deme triggers a termination condition first (e.g., population extinction), the entire `SpatialPopulation` stops advancing. For detailed execution flow, see [Spatial Lifecycle Wrapper](spatial_lifecycle_wrapper.md).

## migration_rate and Boundary Effects

### migration_rate

`migration_rate` is the per-deme outbound quota: the proportion of a deme's mass involved in cross-deme flow per step. The following forms are accepted:

- **Scalar** (`float`): Only adult ages (>= population's `new_adult_age`) receive the rate; juveniles default to 0. In discrete-generation populations the single age class receives the full rate.
- **Age-specific array** (`NDArray[np.float64]` or `Sequence[float]`, shape `(n_ages,)`): Each age is set explicitly.
- **Per-sex table or mapping** (`(n_sexes, n_ages)` array, or `{"F": 0.2, "M": 0.05}`): The scalar/vector rules are applied per sex.
- **Per-deme column** (`(n_demes, n_sexes, n_ages)`, or `(n_demes, n_ages)` broadcast across sexes): Each deme receives its own rate.
- **`batch_setting([...])`** (builder only): One declaration per deme; each element accepts any of the forms above.

Every form lands on `pop.params.migration_rate` with shape `(n_demes, n_sexes, n_ages)` — the same column the runtime `params.tensor_write("migration_rate", ...)` channel writes.

A 2-D declaration whose shape is exactly `(n_sexes, n_ages)` is read as the shared per-sex table; only another 2-D shape means per-deme age vectors. When `n_demes == n_sexes` those shapes are indistinguishable, so pass the explicit 3-D column for per-deme rates.

The first two examples below assume `demes` contains age-structured populations
with four ages and `new_adult_age=2`, sharing the same Species.

```python
# Scalar — juveniles age < new_adult_age emigrate 0%, adults emigrate 10%
spatial = SpatialPopulation(demes, migration_rate=0.1)

# Age-specific — explicit per-age (default new_adult_age=2)
spatial = SpatialPopulation(demes, migration_rate=[0.0, 0.0, 0.3, 0.1])

# Per-deme — the middle deme of a 3-deme chain emigrates four times as much
spatial = (
    SpatialPopulation.builder(species, n_demes=3, topology=SquareGrid(1, 3))
    .age_structure(n_ages=4, new_adult_age=2)
    .migration(migration_rate=batch_setting([0.1, 0.4, 0.1]))
    .build()
)

# Runtime update
spatial.params.tensor_write("migration_rate", 0.2)  # adult ages only
spatial.params.tensor_write("migration_rate", [0.0, 0.0, 0.3, 0.1])  # four ages
```

- `0.0`: No migration (all ages).
- `0.1`: Adult ages (>= new_adult_age) emigrate 10% per step; discrete-generation populations emigrate 10% overall.

### Outbound Weights and Boundary Effects

The migration CSR stores **relative outbound weights**, not probabilities. The builder row-normalizes every non-empty row to a probability vector before folding it, so a row-stochastic input, a sub-stochastic input (row sum < 1) and a super-stochastic input (row sum > 1) all describe the same outbound distribution and migration conserves mass. "Migrate less" is expressed through `migration_rate`, never by shrinking a row; an all-zero row (an isolated deme) is kept as is, so that deme holds its mass.

When `topology.wrap=False`, boundary demes have fewer valid neighbors. They still send their full `migration_rate` quota — the offsets that fall outside the grid are dropped and their share is redistributed over the remaining neighbors. A boundary deme therefore sends a **larger share to each** neighbor, not a smaller total.

`adjust_migration_on_edge` is retained as a legacy bit-parity flag: both values produce the same destination distribution (up to ~1 ulp of rounding), because the denominator it selects is cancelled by the row normalization.

**Practical impact**:

```python
# 3x3 von Neumann kernel (4 neighbors), migration_rate = r
# Interior deme (4 neighbors): each neighbor receives r / 4, total migration = r
# Corner deme   (2 valid neighbors): each neighbor receives r / 2, total migration = r
#   -> Every deme emigrates the same total amount; a boundary deme simply
#      splits its quota among fewer destinations
```

**Special case**: When `topology.wrap=True`, every deme has the same number of valid neighbors, so all demes normalize over the same neighbor set and the boundary distinction disappears.

### Non-Uniform Weight Kernels

When kernel weights are not all 1 (e.g. a Gaussian kernel), the relative weight structure is preserved: each neighbor's share is its weight divided by the sum of the source's valid weights.

```python
# 5x5 Gaussian kernel: center weights high, edge weights low
#
# Interior deme (all 25 neighbors valid):
#   Each neighbor share = weight / valid_weight_sum
#   Total migration rate = rate * 1.0
#
# Boundary deme (e.g. 15 valid neighbors):
#   Each neighbor share = weight / valid_weight_sum  (relative weights unchanged)
#   Total migration rate = rate * 1.0 — the dropped offsets redistribute their
#   share over the valid neighbors instead of leaving mass behind
```

### Kernel Implementation

For details on the kernel offset table, computation of `kernel_total_sum`, and the implementation of `adjust_on_edge` in `prange`, see [Migration Kernel Implementation](migration_kernel_impl.md).

## Mathematical Form of the Migration Kernel

A migration kernel $K$ is an odd-dimension matrix, centered at $(\lfloor R/2 \rfloor, \lfloor C/2 \rfloor)$. For a source deme at coordinates $(r_s, c_s)$, each non-zero kernel weight $K_{i,j} > 0$ corresponds to a potential target coordinate:

$$(r_d, c_d) = (r_s + (i - i_c),\; c_s + (j - j_c))$$

where $(i_c, j_c)$ are the matrix coordinates of the kernel center. Coordinates that fall within the grid become valid neighbors; coordinates outside the grid are discarded when `wrap=False` or wrapped by modulo when `wrap=True`.

Each source deme's outbound distribution is proportional to the kernel weights and normalized over its valid neighbors:

$$p_n = \frac{w_n}{\sum_m w_m}$$

where $\sum_m w_m$ sums the weights of the source deme's valid neighbors (the coordinates kept above). Every deme therefore emigrates its full quota $r$; a boundary deme's dropped offsets redistribute their share over the valid neighbors, so each of them receives a larger share. The `adjust_migration_on_edge` denominator (kernel total vs. valid-row total) is cancelled by this normalization up to ~1 ulp of rounding, so the flag does not change the destination distribution.

### Constructing Common Kernels

NATAL provides the `build_gaussian_kernel()` factory function, automatically using the correct distance metric based on topology type:

```python
from natal.frontend.spatial import build_gaussian_kernel, HexGrid, SquareGrid

# Hexagonal grid Gaussian kernel -- automatically uses cosine law distance formula
hex_kernel = build_gaussian_kernel(HexGrid, size=11, sigma=1.5)

# Square grid Gaussian kernel -- uses Cartesian distance
square_kernel = build_gaussian_kernel(SquareGrid, size=7, sigma=2.0)

# String shorthand
hex_kernel = build_gaussian_kernel("hex", size=11, sigma=1.5)

# Specify mean dispersal distance for more intuitive control
# sigma = mean_dispersal / sqrt(pi/2)
hex_kernel = build_gaussian_kernel("hex", size=11, mean_dispersal=2.0)
```

`sigma` and `mean_dispersal` are mutually exclusive. In a 2D isotropic Gaussian distribution, the mean displacement follows a Rayleigh distribution: $\bar{d} = \sigma\sqrt{\pi/2}$.

Kernels can also be constructed manually (compatible with legacy code):

```python
import numpy as np

# von Neumann 3x3 (4 neighbors, excluding center)
von_neumann = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

# Moore 3x3 (8 neighbors, excluding center)
moore = np.ones((3, 3), dtype=np.float64)
moore[1, 1] = 0.0
```

## Topology Structures

NATAL provides two grid topologies: `SquareGrid` and `HexGrid`. Both share the same coordinate system -- demes are arranged in row-major order, and the conversion between flat index and grid coordinates is:

$$i_{\text{flat}} = r \cdot \text{cols} + c, \qquad (r, c) = (i_{\text{flat}} \mathbin{//} \text{cols},\; i_{\text{flat}} \bmod \text{cols})$$

Boundary behavior is controlled uniformly by the `wrap` parameter, applied to all neighbor offsets:

$$\text{normalize}(r, c) = \begin{cases} (r \bmod R,\; c \bmod C) & \text{wrap=True} \\ \text{None (discarded)} & \text{wrap=False and coordinates out of bounds} \end{cases}$$

### SquareGrid

```python
SquareGrid(rows=R, cols=C, neighborhood="moore", wrap=False)
```

**Von Neumann neighborhood** (`neighborhood="von_neumann"`): 4 directional offsets

$$\Delta = \{(-1,0),\;(1,0),\;(0,-1),\;(0,1)\}$$

**Moore neighborhood** (`neighborhood="moore"`, default): 8 directional offsets

$$\Delta = \{(-1,-1),(-1,0),(-1,1),\;(0,-1),(0,1),\;(1,-1),(1,0),(1,1)\}$$

### HexGrid

```python
HexGrid(rows=R, cols=C, wrap=False)
```

HexGrid uses parallelogram coordinates $(i, j)$, with 6 neighbor offsets fixed as:

$$\Delta = \{(1,0),\;(0,1),\;(-1,1),\;(-1,0),\;(0,-1),\;(1,-1)\}$$

The planar embedding uses pointy-top hexagons:

$$x = i + 0.5j, \qquad y = \frac{\sqrt{3}}{2}\,j$$

The six neighbors are equidistant from the source deme in the embedding space, giving better isotropic diffusion compared to SquareGrid.

### Neighbor Count Under Boundary Conditions

Let $N_{\text{max}}$ be the maximum neighbor count for interior demes in the grid (4 or 8 for SquareGrid, 6 for HexGrid), and $(r, c)$ be the grid coordinates.

With **wrap=False**, out-of-bounds neighbors are discarded, giving boundary demes $N_{\text{eff}}(r, c) < N_{\text{max}}$. Corner positions have the fewest neighbors:

| Topology | Neighborhood | Interior | Edge | Corner |
|----------|-------------|----------|------|--------|
| SquareGrid | von_neumann | 4 | 3 | 2 |
| SquareGrid | moore | 8 | 5 | 3 |
| HexGrid | -- | 6 | 4 or 5 | 3 or 4 |

With **wrap=True**, coordinates wrap by modulo, giving $N_{\text{eff}}(r, c) = N_{\text{max}}$ for all positions.

### Selection Guide

| Scenario | Recommended Topology |
|----------|---------------------|
| Rapid prototyping, mixing with adjacency matrix patterns | `SquareGrid` + `von_neumann` |
| Richer local connectivity | `SquareGrid` + `moore` |
| Isotropic diffusion, large-scale spatial simulation | `HexGrid` |
| Eliminating boundary artifacts | Any topology + `wrap=True` |
| Preserving natural boundary effects (fewer, larger shares at the edge) | Any topology + `wrap=False` |

### Complete Example: SquareGrid

```python
import numpy as np
from natal import Species, SpatialPopulation, SquareGrid

species = Species.from_dict(name="sq", structure={"chr1": {"loc": ["A", "B"]}})

kernel = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

pop = (
    SpatialPopulation.builder(species, n_demes=9, topology=SquareGrid(3, 3,
        neighborhood="von_neumann", wrap=False))
    .setup(name="square_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000)
    .migration(kernel=kernel, migration_rate=0.2, adjust_migration_on_edge=False)
    .build()
)

pop.run(10)
```

### Complete Example: HexGrid

```python
from natal import Species, SpatialPopulation, HexGrid
from natal.frontend.spatial import build_gaussian_kernel

species = Species.from_dict(name="hex", structure={"chr1": {"loc": ["WT", "Dr"]}})

# Use build_gaussian_kernel to automatically handle hex coordinate distance metric
kernel = build_gaussian_kernel(HexGrid, size=11, sigma=1.5)

pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10, wrap=False))
    .setup(name="hex_demo", stochastic=True, continuous_sampling=True)
    .initial_state(individual_count={"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6, juvenile_growth_mode="beverton_holt")
    .migration(kernel=kernel, migration_rate=0.5)
    .build()
)

pop.run(10)
```

## WebUI Debugging

Spatial models can be directly connected to the Vue dashboard (hexagonal landscape map, click-to-inspect demes, migration panel, Debug tab):

```python
from natal import launch_vue

launch_vue(spatial, port=8000, title="Spatial Debug Dashboard")
```

## Common Errors and Troubleshooting

### Error 1: Demes Not from the Same Species

If demes are not from the same Species, `SpatialPopulation` will raise an error immediately.

### Error 2: Inconsistent Migration Sampling Mode Across Demes

Heterogeneous deme configs are supported. However, when migration is enabled, all demes'
`stochastic` and `continuous_sampling` must be consistent;
otherwise `run_tick()` / `run(...)` will raise an error.

### Error 3: Incorrect Kernel Dimensions

If the passed `migration_kernel` is not an odd-dimension 2D array, an error will be raised during construction.

### Error 4: Incorrect Adjacency Matrix Size

`adjacency.shape` must equal `(n_demes, n_demes)`.

### Error 5: kernel_bank Mismatch with topology

Heterogeneous kernels (`kernel_bank` + `deme_kernel_ids`) follow the kernel path and require `topology` to be present. If only `kernel_bank` is passed without `topology`, an error will be raised during construction.

## Chapter Summary

The practical usage order of SpatialPopulation can be remembered in four steps:

1. Start chain construction with `SpatialPopulation.builder(...)`.
2. Heterogeneous deme configs (`batch_setting`) can be used, but migration sampling mode must be consistent across all demes.
3. Choose between adjacency or migration_kernel; set each deme's outbound quota with `migration_rate` (the legacy `adjust_migration_on_edge` flag changes nothing).
4. Debug with `run_tick()`, run batch experiments with `run(...)`.

---

## Related Chapters

- [SpatialPopulationBuilder: Batch Construction of Spatial Populations](spatial_population_builder.md)
- [Spatial Lifecycle Wrapper](spatial_lifecycle_wrapper.md)
- [Migration Kernel Implementation](migration_kernel_impl.md)
- [the Simulation Engine Deep Dive](4_simulation_engine.md)
- [Hook System](2_hooks.md)
