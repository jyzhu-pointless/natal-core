# Spatial Simulation Guide

This chapter describes the practical usage of `SpatialPopulation`: first build a set of demes that share a common species and a common configuration, then wrap them into a spatial container, and finally control movement between demes using topology or migration kernels.

After reading this chapter, you should be able to directly write code like the following:

```python
spatial = SpatialPopulation(
  demes=demes,
  topology=SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
  adjacency=adjacency,
  migration_rate=0.15,
)
```

## 1. A Real Constraint to Remember

`SpatialPopulation` is not a factory that creates “new populations” from scratch; it organises already constructed demes.

There are two hard requirements in the implementation:

1. All demes must share the same `Species` object.
2. All demes must share the same `config` object.

Therefore, in the demo, each deme is built first, and then `share_config(demes)` is applied:

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)
```

## 2. Core Parameters of `SpatialPopulation`

The constructor of `SpatialPopulation` supports these most common parameters:

- `demes`: a list of already built demes.
- `topology`: optional grid topology, typically `SquareGrid` or `HexGrid`.
- `adjacency`: explicit adjacency matrix; if not provided, it is usually derived from `topology`.
- `migration_kernel`: a migration kernel used when following the kernel path.
- `migration_rate`: the proportion of individuals that participate in migration each step.
- `migration_strategy`: `auto`, `adjacency`, `kernel`, or `hybrid`; default is `auto`.
- `kernel_include_center`: under the kernel path, whether to include the centre cell as a migration target.

If you only want to remember the most important rules, keep these two in mind:

1. If you provide `adjacency`, the adjacency path is used.
2. If you provide `migration_kernel`, the kernel path is used, and `topology` must also exist.

## 3. Minimal Working Flow

### 3.1 Build the Demes First

Each deme is an ordinary population instance. The most common practice is to first build age‑structured or discrete‑generation populations, and then put them into the spatial container.

### 3.2 Share the Configuration

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)
```

### 3.3 Choose a Migration Path

You have two common paths:

1. **adjacency path** – suitable for scenarios where you first define the topology and then generate the adjacency matrix.
2. **kernel path** – suitable for scenarios where you want to describe local migration weights using a 3×3 or 5×5 kernel.

### 3.4 Assemble the Spatial Container

```python
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid, build_adjacency_matrix

demes = [deme0, deme1, deme2, deme3]
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)

adjacency = build_adjacency_matrix(
  SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
  row_normalize=True,
)

spatial = SpatialPopulation(
  demes=demes,
  adjacency=adjacency,
  migration_rate=0.15,
  name="SpatialDemo",
)

spatial.run_tick()
spatial.run(n_steps=5, record_every=1)
```

### 3.5 Kernel Path Example

```python
import numpy as np
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid

spatial = SpatialPopulation(
  demes=demes,
  topology=HexGrid(rows=100, cols=100, wrap=False),
  migration_kernel=np.array(
    [
      [0.00, 0.10, 0.05],
      [0.10, 0.00, 0.10],
      [0.05, 0.10, 0.00],
    ],
    dtype=np.float64,
  ),
  migration_rate=0.2,
  name="SpatialHexDemo",
)
```

## 4. What Happens at Runtime

The semantics of `run_tick()` and `run(...)` are the same as for ordinary populations, but they advance all demes together:

1. Check whether each deme has already finished.
2. Read the compiled hooks from the first deme.
3. Concatenate the states of all demes into a large array.
4. Run the spatial version of the kernels.
5. Split the updated state back to each deme.

Therefore, the debugging order for a spatial model is clear:

- First, use `run_tick()` to verify that a single step is correct.
- Then use `run(...)` to run multiple steps.
- If one deme finishes early, the whole `SpatialPopulation` will stop advancing.

## 5. `migration_rate` and Migration Modes

`migration_rate` is the proportion of individuals that participate in inter‑deme movement each step.

You can think of it simply as:

- `0.0`: no migration.
- `0.1`: 10% of the mass participates in migration each step.
- `1.0`: all mass is redistributed each step according to adjacency / migration kernel.

In the implementation, the kernel path requires:

- `topology` to exist.
- `migration_kernel` to be a 2D array.
- Both dimensions of `migration_kernel` must be odd.

This is why demos often use a `3×3` or `5×5` kernel.

## 6. How to Choose a Topology

### 6.1 `SquareGrid`

If you just want the easiest spatial model to understand, start with `SquareGrid`.

It supports two neighbourhood types:

1. `von_neumann`: connects only up, down, left, right.
2. `moore`: connects up, down, left, right and the four diagonals.

A common 2×2 example is:

```python
SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
```

This is the most direct spatial migration example in the demo.

### 6.2 `HexGrid`

If you want each deme to have six neighbours and more uniform diffusion directions, use `HexGrid`.

Its features are simple:

- When `wrap=False`, boundary demes have fewer than six neighbours.
- When `wrap=True`, boundaries wrap around to the opposite side.

Thus, in large hex grids, a common pattern is:

```python
HexGrid(rows=100, cols=100, wrap=False)
```

## 7. WebUI Debugging

A spatial model can be directly connected to `natal.ui.launch(...)`.

```python
from natal.ui import launch

launch(spatial, port=8080, title="Spatial Debug Dashboard")
```

This dashboard is suitable for three tasks:

1. Checking whether global totals are abnormal.
2. Seeing whether a particular deme experiences local outbreaks or depletion.
3. Verifying that spatial gradients after migration meet expectations.

## 8. Common Errors and Troubleshooting

### Error 1: Demes are not of the same species

If the demes do not share the same `Species`, `SpatialPopulation` will directly raise an error.

### Error 2: Configuration not shared

If each deme does not use the same `config` object, the shared configuration check will fail during `run_tick()` / `run(...)`.

### Error 3: Kernel dimensions are incorrect

If you provide `migration_kernel` but it is not an odd‑dimension 2D array, construction will raise an error.

### Error 4: Adjacency matrix size is wrong

`adjacency.shape` must equal `(n_demes, n_demes)`.

## 9. Practical Advice

1. Start with a small topology, such as 2×2 or 3×3.
2. First run `run_tick()` to ensure that a single step of migration works correctly.
3. Then switch to `run(n_steps=...)` for batch experiments.
4. If you want to compare spatial diffusion speeds, change only `migration_rate`; do not change both the topology and the kernel at the same time.

## 10. Chapter Summary

The practical usage of `SpatialPopulation` can be remembered as four steps:

1. Build a set of demes that share the same species.
2. Make those demes share the same configuration.
3. Choose either the adjacency path or the migration kernel path.
4. Debug with `run_tick()` and run batch experiments with `run(...)`.

---

## Related Chapters

- [Deep Dive into Simulation Kernels](simulation_kernels.md)
- [Hook System](hooks.md)
- [Modifier Mechanism](modifiers.md)
- [PopulationState and PopulationConfig](population_state_config.md)
