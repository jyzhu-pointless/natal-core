# Spatial Simulation Guide

This chapter introduces the practical usage of `SpatialPopulation`: using the `SpatialBuilder` to quickly build multi-deme populations, configure topology and migration kernels, and control inter-deme flow.

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

> **Tip**: `SpatialBuilder` is the preferred construction method for homogeneous/heterogeneous spatial populations. The construction time for 2601 homogeneous demes has been reduced from ~2.6s to ~16ms. See [SpatialBuilder Documentation](spatial_builder.md).

## Two Construction Paths

### Recommended: SpatialBuilder (Chainable API)

```python
from natal import Species, HexGrid, SpatialPopulation
from natal.spatial_builder import batch_setting

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
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid

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
- `migration_rate`: Proportion of individuals migrating per step.
- `migration_strategy`: `auto`, `adjacency`, `kernel`, or `hybrid`; default is `auto`.
- `kernel_include_center`: Whether to include the center cell as a migration target in the kernel path, default `False`.
- `adjust_migration_on_edge`: Whether to adjust migration volume at boundaries (see "migration_rate and Boundary Effects" section), default `False`.

The most important rules:

1. Pass `adjacency` to use the adjacency matrix path.
2. Pass `migration_kernel` to use the kernel path, and topology must also be present.
3. Pass `kernel_bank` + `deme_kernel_ids` to also use the kernel path (heterogeneous kernel).
4. `hybrid` is reserved for a combined adjacency+kernel mixed strategy, and is not required for heterogeneous kernels.

## Chainable API

The `SpatialBuilder` chainable call flow is consistent with the panmictic builder. Below are the methods listed in recommended order. Methods marked with `->` are spatial-specific, and parameters marked with `[B]` accept `batch_setting` (cross-deme heterogeneous configuration).

```python
pop = (
    SpatialPopulation.builder(species, n_demes=9, topology=SquareGrid(3, 3))
    ->                   # Entry: specify deme count and topology
    .setup(name="demo", stochastic=False, use_continuous_sampling=False)
                        # Basic settings: name, stochasticity, sampling mode
    .age_structure(n_ages=8, new_adult_age=2)
                        # [age_structured only] Age group count, adult starting age
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
                        # [B] Initial genotype distribution
    .survival(female_age_based_survival_rates=[...], ...)
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

### Spatial-Specific: `.migration()`

```python
.migration(
    kernel=None,                     # [B] NDArray: odd-dimension migration kernel
    migration_rate=0.0,             # float: migration proportion
    strategy="auto",                # "auto" | "adjacency" | "kernel" | "hybrid"
    adjacency=None,                 # Explicit adjacency matrix
    kernel_bank=None,               # Heterogeneous kernel collection
    deme_kernel_ids=None,           # Per-deme kernel index
    kernel_include_center=False,    # Whether to include the center cell
    adjust_migration_on_edge=False, # Whether to adjust migration at boundaries
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
| `competition` | `expected_num_adult_females` | float |
| `age_structure` | `equilibrium_distribution` | list[float] |
| `presets` | positional arguments | preset object |
| `fitness` | `viability` / `fecundity` / `sexual_selection` / `zygote_viability` | dict |
| `migration` | `kernel` | NDArray |

The following parameters do **not** accept `batch_setting`:
- **hooks**: Per-deme selective execution is achieved via `@hook(deme=...)`.
- **Spatial functions require topology**: `(row, col)` form requires the builder to have been given a `topology`. The `(flat_idx)` form does not depend on topology.

## batch_setting Heterogeneous Configuration

`batch_setting` is the core mechanism of `SpatialBuilder`, allowing different demes to specify different parameter values within the same chainable call. Internally, it automatically optimizes through config equivalence grouping -- demes with the same parameters share compiled artifacts, only the state arrays are independent.

### Four Input Forms

```python
from natal.spatial_builder import batch_setting
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
from natal.spatial_builder import batch_setting

# Default: all demes have only WT
n_demes = 100
default_state = {"female": {"WT|WT": 500}, "male": {"WT|WT": 500}}

# Center deme releases drive heterozygotes
release_state = {"female": {"WT|WT": 450, "Dr|WT": 50},
                 "male":   {"WT|WT": 450, "Dr|WT": 50}}

states = [default_state] * n_demes
states[n_demes // 2] = release_state

pop = (
    SpatialPopulation.builder(species, n_demes=n_demes, topology=HexGrid(10, 10))
    .setup(name="drive_release", stochastic=True, use_continuous_sampling=True)
    .initial_state(individual_count=batch_setting(states))
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6,
                 juvenile_growth_mode="concave")
    .presets(HomingDrive(name="Drive", drive_allele="Dr", target_allele="WT",
                         resistance_allele="R2", functional_resistance_allele="R1",
                         drive_conversion_rate=0.95))
    .fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}})
    .migration(kernel=kernel, migration_rate=0.2)
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

`SpatialPopulation.run()`'s `record_every` parameter controls the history sampling interval within the Numba-compiled kernel. Setting it to 0 disables history recording.

### Accessing Aggregate State

```python
# Cross-deme aggregation
pop.total_population_size   # Total individual count
pop.total_females           # Total female count
pop.total_males             # Total male count
pop.sex_ratio               # Sex ratio (female/male)
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
print(deme_0.total_population_size)
print(deme_0.compute_allele_frequencies())

# Iterate over all demes
for i in range(pop.n_demes):
    d = pop.deme(i)
    print(f"deme {i}: {d.total_population_size}")
```

Each deme is an instance of `AgeStructuredPopulation` or `DiscreteGenerationPopulation`, supporting all panmictic interfaces such as `output_current_state()` and `compute_allele_frequencies()`.

### Reset and Control

```python
# Reset all demes to initial state
pop.reset()

# Check if simulation is finished
if pop.is_finished:
    print("Simulation has terminated")

# Manually terminate
pop.finish_simulation()
```

### Data Output

`output_current_state()` and `output_history()` work the same as for panmictic populations, supporting observation rule filtering:

```python
# Current state snapshot
state = pop.output_current_state()

# History export with observation rules
observation = pop.create_observation(
    groups={"adult_wt": {"genotype": ["WT|WT"], "age": [2]}},
    collapse_age=True,
)
history = pop.output_history(observation=observation)
```

For detailed usage, see [Extracting Population Simulation Data](2_data_output.md).

### Runtime Internal Flow

The internal execution order of each `run_tick()`:

1. Check whether each deme has `is_finished`.
2. Concatenate all demes' state into a unified array, build a config bank.
3. Run the Numba-compiled spatial lifecycle wrapper: `prange` parallel execution of each deme's lifecycle -> unified migration.
4. Write the updated state back to each deme.

If a deme triggers a termination condition first (e.g., population extinction), the entire `SpatialPopulation` stops advancing. For detailed execution flow, see [Spatial Lifecycle Wrapper](spatial_lifecycle_wrapper.md).

## migration_rate and Boundary Effects

### migration_rate

`migration_rate` controls the proportion of mass involved in cross-deme flow per step:

- `0.0`: No migration.
- `0.1`: 10% of mass participates in migration per step.
- `1.0`: All mass is redistributed according to adjacency/migration kernel each step.

### Boundary Effect and adjust_migration_on_edge

When `topology.wrap=False`, boundary demes have fewer effective neighbors than interior demes. `adjust_migration_on_edge` controls how this difference is handled:

| `adjust_migration_on_edge` | Behavior |
|---|---|
| `False` (default) | Boundary demes naturally emigrate less. Each neighbor's migration probability = `weight / kernel_total_sum`, total migration is proportional to the number of effective neighbors |
| `True` | All demes emigrate the same total amount. Each neighbor's migration probability = `weight / effective_sum` (normalized to 1.0) |

Where `kernel_total_sum` is the sum of all positive weights in the kernel, serving as the unified scaling reference.

**Practical impact**:

```python
# 3x3 kernel, center weight 0, surrounding weights 1.0
# kernel_total_sum = 8.0

# Default behavior (adjust_migration_on_edge=False):
#   Interior deme (8 neighbors): each neighbor probability = 1.0/8.0 = 0.125, total migration = rate * 1.0
#   Corner deme (3 neighbors): each neighbor probability = 1.0/8.0 = 0.125, total migration = rate * 0.375
#   -> Boundary emigrates less, more biologically intuitive

# Adjusted behavior (adjust_migration_on_edge=True):
#   Interior deme (8 neighbors): each neighbor probability = 1.0/8.0 = 0.125, total migration = rate * 1.0
#   Corner deme (3 neighbors): each neighbor probability = 1.0/3.0 ≈ 0.333, total migration = rate * 1.0
#   -> All demes emigrate the same total amount, boundary effect is artificially smoothed
```

**Special case**: When `topology.wrap=True`, all demes have the same number of effective neighbors, and both modes behave identically.

### Non-Uniform Weight Kernels

When weights in the kernel are not all 1 (e.g., Gaussian kernel), `kernel_total_sum` preserves the relative weight structure of the kernel:

```python
# 5x5 Gaussian kernel: center weights high, edge weights low
# kernel_total_sum is the sum of all weights
#
# Interior deme (all 25 neighbors effective):
#   Each neighbor probability = weight / kernel_total_sum
#   Total migration rate = rate * (effective_sum / kernel_total_sum) = rate * 1.0
#
# Boundary deme (e.g., 15 effective neighbors):
#   Each neighbor probability = weight / kernel_total_sum  (relative weights unchanged)
#   Total migration rate = rate * (effective_sum / kernel_total_sum) ≈ rate * 0.6
```

### Kernel Implementation

For details on the kernel offset table, computation of `kernel_total_sum`, and the implementation of `adjust_on_edge` in `prange`, see [Migration Kernel Implementation](migration_kernel_impl.md).

## Mathematical Form of the Migration Kernel

A migration kernel $K$ is an odd-dimension matrix, centered at $(\lfloor R/2 \rfloor, \lfloor C/2 \rfloor)$. For a source deme at coordinates $(r_s, c_s)$, each non-zero kernel weight $K_{i,j} > 0$ corresponds to a potential target coordinate:

$$(r_d, c_d) = (r_s + (i - i_c),\; c_s + (j - j_c))$$

where $(i_c, j_c)$ are the matrix coordinates of the kernel center. Coordinates that fall within the grid become effective neighbors; coordinates outside the grid are discarded when `wrap=False` or wrapped by modulo when `wrap=True`.

The probability of a source deme migrating to neighbor $n$ is determined by `adjust_migration_on_edge`:

$$p_n = \frac{w_n}{S_{\text{ref}}}, \quad S_{\text{ref}} = \begin{cases} \sum_{m} w_m & \text{(adjust=True, normalized by effective neighbors)} \\ \sum_{i,j} K_{i,j} & \text{(adjust=False, scaled by kernel sum)} \end{cases}$$

where $\sum_{i,j} K_{i,j}$ is the sum of all kernel weights (denoted `kernel_total_sum`), and $\sum_m w_m$ is the sum of weights for the current deme's actually effective neighbors. Under `adjust=False`, the total emigration from a boundary deme is $r \cdot \frac{\sum_m w_m}{\sum_{i,j} K_{i,j}}$, naturally smaller than that of interior demes.

### Constructing Common Kernels

NATAL provides the `build_gaussian_kernel()` factory function, automatically using the correct distance metric based on topology type:

```python
from natal.spatial_topology import build_gaussian_kernel, HexGrid, SquareGrid

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
| Preserving natural boundary effects + boundary-aware migration | Any topology + `wrap=False` + `adjust_migration_on_edge=False` |

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
from natal.spatial_topology import build_gaussian_kernel

species = Species.from_dict(name="hex", structure={"chr1": {"loc": ["WT", "Dr"]}})

# Use build_gaussian_kernel to automatically handle hex coordinate distance metric
kernel = build_gaussian_kernel(HexGrid, size=11, sigma=1.5)

pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10, wrap=False))
    .setup(name="hex_demo", stochastic=True, use_continuous_sampling=True)
    .initial_state(individual_count={"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6, juvenile_growth_mode="concave")
    .migration(kernel=kernel, migration_rate=0.5)
    .build()
)

pop.run(10)
```

## WebUI Debugging

Spatial models can be directly connected to `natal.ui.launch(...)`.

```python
from natal.ui import launch

launch(spatial, port=8080, title="Spatial Debug Dashboard")
```

## Common Errors and Troubleshooting

### Error 1: Demes Not from the Same Species

If demes are not from the same Species, `SpatialPopulation` will raise an error immediately.

### Error 2: Inconsistent Migration Sampling Mode Across Demes

Heterogeneous deme configs are supported. However, when migration is enabled, all demes'
`is_stochastic` and `use_continuous_sampling` must be consistent;
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
3. Choose between adjacency or migration_kernel; use `adjust_migration_on_edge` for boundary-aware migration.
4. Debug with `run_tick()`, run batch experiments with `run(...)`.

---

## Related Chapters

- [SpatialBuilder: Batch Construction](spatial_builder.md)
- [Spatial Lifecycle Wrapper](spatial_lifecycle_wrapper.md)
- [Migration Kernel Implementation](migration_kernel_impl.md)
- [Simulation Kernels Deep Dive](4_simulation_kernels.md)
- [Hook System](2_hooks.md)
