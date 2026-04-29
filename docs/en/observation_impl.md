# Observation History Recording: Developer Implementation Document

This document is intended for NATAL Core maintainers and contributors. It details the implementation principles, data flow, and module responsibilities of observation-based history recording.

## Overall Architecture

Observation history recording involves four layers, with parameters passed through NamedTuple bundles at the Kernel layer:

```
User-facing API         →  BasePopulation / SpatialPopulation's record_observation property
                           SpatialPopulation's _spatial_topo / _migration_params / _compact_meta
Observation layer      →  observation.py: Observation, ObservationFilter, build_mask
                           observation_record.py: CompactMeta, build_observation_row_spatial
Kernel layer           →  recording pathway in spatial_simulation_kernels.py
                           spatial_lifecycle_*.tmpl.py code generation templates
Export layer           →  state_translation.py: output_history / *observation_history_to_readable_dict
```

## NamedTuple Parameter Bundles

To avoid passing 13+ scattered parameters across multiple function layers, the spatial kernel dispatch uses four NamedTuples to converge parameters:

| NamedTuple | Definition Location | Parameters Converged | Lifecycle |
|-----------|-------------------|--------------------|-----------|
| `SpatialTopology` | `spatial_topology.py` | `rows`, `cols`, `wrap` | Fixed at construction time |
| `MigrationParams` | `spatial_topology.py` | `kernel`, `include_center`, `rate`, `adjust_on_edge` | Fixed at construction time |
| `HeterogeneousKernelParams` | `spatial_topology.py` | `deme_kernel_ids`, `d_row`, `d_col`, `weights`, `nnzs`, `total_sums`, `max_nnz` | Rebuilt each `run()` call |
| `CompactMeta` | `observation_record.py` | `offsets`, `deme_map`, `n_demes_per_group`, `selected_n`, `mode_aggregate`, `row_size` | Rebuilt when `record_observation` is set |

`migration_mode` and `adjacency` are not included in `MigrationParams` because they are dynamically resolved by `_effective_migration_route()` on each call (which can choose adjacency / kernel / auto), making them **routing strategies** rather than **migration parameters**.

## Data Flow

```
User defines groups
       ↓
ObservationFilter.build_filter(groups) → Observation object (with specs, labels, collapse_age)
       ↓
BasePopulation._build_observation_mask(obs) → 4D float64 mask (n_groups, n_sexes, n_ages, n_genotypes)
       ↓
_build_deme_info() → demean_modes dict → build_compact_metadata() → CompactMeta
       ↓
mask + CompactMeta passed to Numba kernel (observation_mask + compact_meta parameters)
       ↓
kernel every record_every steps:
  build_observation_row_spatial(ind, mask, compact_meta) → compact flat row
  row = [tick, compact_row]
       ↓
Python layer: _process_kernel_history() → history.append((tick, row.copy()))
       ↓
Export: output_history() auto-detects record_observation → dispatches to *observation_history_to_readable_dict
```

## Data Format Comparison

### Raw Mode (record_observation=None)

**Panmictic** per row:
```
[tick, ind[0,0,0], ind[0,0,1], ..., ind[n_sexes-1, n_ages-1, n_genotypes-1],
 sperm[0,0,0,0], ..., sperm[n_ages-1, n_genotypes-1, n_genotypes-1]]
```
Row size = `1 + n_sexes × n_ages × n_genotypes + n_ages × n_genotypes²`

**Spatial** per row:
```
[tick, flat_ind_deme_0, ..., flat_ind_deme_n-1, flat_sperm_deme_0, ..., flat_sperm_deme_n-1]
```
where `flat_ind_deme_d = ind_d[d].ravel()` (row size per deme = `n_sexes × n_ages × n_genotypes`)

### Observation Mode (record_observation is set)

**Panmictic** per row:
```
[tick, observed[0,0,0], observed[0,0,1], ..., observed[n_groups-1, n_sexes-1, n_ages-1]]
```
Row size = `1 + n_groups × n_sexes × n_ages`

**Spatial** per row:
```
[tick, observed[0,0,0,0], ..., observed[n_demes-1, n_groups-1, n_sexes-1, n_ages-1]]
```
Row size = `1 + n_demes × n_groups × n_sexes × n_ages`

## Core Module Details

### 1. Observation System (observation.py)

#### Observation Object

`Observation` is an immutable dataclass containing:
- `filter`: reference to the `ObservationFilter` that created it
- `diploid_genotypes`: the genotype sequence used for resolving genotype selectors
- `specs`: normalized group specifications `[(name, {key: value}), ...]`
- `labels`: group label tuple `(name_0, name_1, ...)`
- `collapse_age`: whether to collapse the age dimension

Key methods:
- `apply(individual_count)` → `(n_groups, n_sexes, n_ages)`: applies observation projection to a given count array
- `build_mask(n_sexes, n_ages, n_genotypes)` → `(n_groups, n_sexes, n_ages, n_genotypes)`: compiles a 4D binary mask for kernel use. Note this method always returns a 4D mask (`collapse_age=False`) because the kernel requires the full 3rd dimension
- `to_dict()` → metadata dict: serializes labels, collapse_age, n_groups and other metadata

#### ObservationFilter

`ObservationFilter(registry)` is responsible for compiling user-defined group specs into Observations:
- `build_filter(diploid_genotypes, groups, collapse_age)` → `Observation`: full compilation pipeline
- `create_observation(...)`: alias for `build_filter`
- `build_mask_from_specs(...)` → 4D float64 mask: core compilation function, fills the mask via a loop:
  ```python
  for gi in range(n_groups):
      for gidx in per_genotypes[gi]:
          for s in per_sexes[gi]:
              for a in range(n_ages):
                  if per_age_preds[gi](a):
                      mask[gi, s, a, gidx] = 1.0
  ```

#### apply_rule

Pure function `apply_rule(individual_count, rule)` → `observed`:
- 3D count × 4D mask: `sum(mask * count[None, :, :, :], axis=-1)`
- 2D count × 3D mask (discrete generation + collapse_age): similar broadcast + sum

### 2. Population Model Layer (base_population.py)

#### record_observation Property

```python
@property
def record_observation(self) -> Optional[Observation]: ...

@record_observation.setter
def record_observation(self, obs: Optional[Observation]) -> None:
    self._observation = obs
    if obs is not None:
        self._observation_mask = self._build_observation_mask(obs)
```

The setter compiles the 4D binary mask while setting the observation. `_observation_mask` is of type `NDArray[np.float64]` with shape `(n_groups, n_sexes, n_ages, n_genotypes)`.

#### set_observations Convenience Method

```python
def set_observations(self, groups, *, collapse_age=False):
    obs_filter = ObservationFilter(self.index_registry)
    self._observation = obs_filter.build_filter(
        diploid_genotypes=self.species,
        groups=groups,
        collapse_age=collapse_age,
    )
    self._observation_mask = self._build_observation_mask(self._observation)
```

Internal flow: create `ObservationFilter` → `build_filter` → compile mask → store `_observation` and `_observation_mask` separately.

#### _clone Compatibility

`_clone()` is used by `SpatialBuilder` for efficient deme cloning. During cloning, `_observation` and `_observation_mask` are reset to `None` (lines 281-282), because each clone needs to set observations independently — the observation mask depends on the deme's state shape (although typically the same, explicit setting is required).

### 3. Kernel Integration

#### Panmictic Kernels (simulation_kernels.py)

`run_with_hooks` / `run_discrete_with_hooks` internally call `build_observation_row_panmictic()`:

```python
if observation_mask is not None:
    flat_state[1:] = build_observation_row_panmictic(ind_count, observation_mask)
else:
    flat_state[1:1+ind_size] = ind_count.ravel()
    flat_state[1+ind_size:] = sperm_store.ravel()  # if applicable
```

Key parameters:
- `observation_mask: Optional[NDArray[np.float64]]` — 4D mask
- `build_observation_row_panmictic` is a standalone njit function in `observation_record.py`

#### Spatial Kernels (observation_record.py)

`build_observation_row_spatial()` is responsible for constructing the compact row, replacing the previous inline broadcast + `deme_selector` mask approach:

```python
# observation_record.py
@njit_switch(cache=True)
def build_observation_row_spatial(
    individual_count: NDArray[np.float64],  # (n_demes, n_sexes, n_ages, n_genotypes)
    observation_mask: NDArray[np.float64],  # (n_groups, n_sexes, n_ages, n_genotypes)
    compact: CompactMeta,
) -> NDArray[np.float64]:
    for gi in range(len(compact.offsets)):
        if compact.mode_aggregate[gi]:
            # aggregate: sum selected demes into one chunk
            agg = sum(observation_mask[gi] * individual_count[d] for d in selected)
            result[offset:offset+sex_ages] = agg.ravel()
        else:
            for di in range(nd):
                if di < compact.selected_n[gi]:
                    # selected deme → real data
                    result[...] = (observation_mask * ind).sum(axis=-1).ravel()
                else:
                    # unselected deme → -1.0 sentinel
                    result[...] = -1.0
```

#### Deme Selection: Three Modes

`CompactMeta` has three built-in per-group modes, controlled by the `"deme"` key in the group spec:

| Mode | spec format | Recording behavior | Export behavior |
|------|------------|-------------------|----------------|
| `mask` (default) | `"deme": [0, 2]` or `list` | All `n_demes` written, unselected = `-1.0` | Unselected shown as `"masked"`, not included in aggregate |
| `expand` | `"deme": {"demes": [0,2], "mode": "expand"}` | Only selected demes written | Only shows selected demes |
| `aggregate` | `"deme": {"demes": [0,2], "mode": "aggregate"}` | Summed into one chunk | Single summary statistic |

The -1.0 sentinel ensures that "deme truly 0 individuals" and "deme is masked" are distinguishable, avoiding the ambiguity of the old `deme_selector` zero-out approach.

#### Codegen Passing Path and Template Signatures

`_run_codegen_wrapper_steps()` passes NamedTuple bundles to templates:

```python
run_fn(
    ind_all, sperm_all,
    config_bank, deme_config_ids, registry, tick, n_steps,
    adjacency, migration_mode,
    self._spatial_topo,         # SpatialTopology (rows, cols, wrap)
    self._migration_params,     # MigrationParams (kernel, include_center, rate, adjust_on_edge)
    het,                        # HeterogeneousKernelParams | None
    record_interval, observation_mask, compact_meta,
)
```

The template `RUN_FN_NAME` signature has been reduced from 35 parameters to 17:

```python
def RUN_FN_NAME(
    ind, sperm, config_bank, deme_config_ids, registry, tick, n_steps,
    adjacency, migration_mode,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None,
    record_interval: int,
    observation_mask: Optional[np.ndarray],
    compact_meta: Optional[CompactMeta],
) -> ...:
```

Here `migration_mode` and `adjacency` are not included in `MigrationParams` because they are **routing strategies** (dynamically resolved by `_effective_migration_route()`), not **migration parameters** themselves.

### 4. Spatial-Specific Path (spatial_population.py)

#### record_observation Property

When set, it automatically calls `_build_deme_info()` to parse the `"deme"` spec, then calls `build_compact_metadata()` to construct `CompactMeta`:

```python
@record_observation.setter
def record_observation(self, obs: Optional[Observation]) -> None:
    self._observation = obs
    if obs is not None:
        ref_deme = self._demes[0]
        state = ref_deme.state
        self._observation_mask = obs.build_mask(...)
        self._rebuild_compact_meta()   # → _build_deme_info() + build_compact_metadata()
```

`_build_deme_info()` parses the `"deme"` key in group specs, supporting three formats:
- Missing / `"all"` → not in dict (default all demes)
- `list[int | (row, col)]` → `("mask", flat_indices)` (backward compatible)
- `{"demes": [...], "mode": "aggregate" | "expand" | "mask"}` → dict format with new semantics

#### Python Dispatch Path

When using Python dispatch (hook-aware fallback path), recording is triggered manually in the Python layer, reusing the same `build_observation_row_spatial()`:

```python
# run() → _should_use_python_dispatch() → True
if record_every > 0 and (self._tick % record_every == 0):
    self._record_snapshot()
for _ in range(n_steps):
    if self._run_python_dispatch_tick():
        was_stopped = True
        break
    if record_every > 0 and (self._tick % record_every == 0):
        self._record_snapshot()
```

`_record_snapshot()` calls the standalone njit function in observation mode:

```python
if self._observation_mask is not None and self._compact_meta is not None:
    row = build_observation_row_spatial(
        ind_all, self._observation_mask, self._compact_meta,
    )
    flat = np.empty(1 + self._compact_meta.row_size, dtype=np.float64)
    flat[0] = float(self._tick)
    flat[1:] = row
```

### 5. Export Layer (state_translation.py)

#### Automatic Dispatch Logic

```python
def output_history(population, observation=None, groups=None, ...):
    pop_obs = getattr(population, "record_observation", None)
    if pop_obs is not None and observation is None and groups is None:
        # Observation mode: directly parse compact snapshots
        payload = population_observation_history_to_readable_dict(...)
    else:
        # Raw mode or post-hoc observation: re-parse each snapshot
        payload = _build_history_observation_payload(...)
```

The spatial version is similar:

```python
def spatial_population_output_history(spatial_population, ...):
    obs = getattr(spatial_population, "record_observation", None)
    if obs is not None:
        payload = spatial_population_observation_history_to_readable_dict(...)
    else:
        payload = spatial_population_history_to_readable_dict(...)
```

#### population_observation_history_to_readable_dict

This function parses the compressed history array from observation mode. Workflow:
1. Retrieve the `record_observation` labels and collapse_age
2. For each row `[tick, observed.ravel()]`:
   - Reshape to `(n_groups, n_sexes, n_ages)`
   - Use `_build_observation_payload()` to convert the observed array into a nested dictionary `{sex: {age: {label: count}}}`
3. Return a structure with labels and snapshots

#### spatial_population_observation_history_to_readable_dict

Similar to the panmictic version but with an additional deme dimension:
1. For each row, reshape to `(n_demes, n_groups, n_sexes, n_ages)`
2. Expand per-deme payload by deme
3. Sum across demes to obtain aggregate

#### _build_observation_payload Utility Function

```python
def _build_observation_payload(observed, labels, sex_labels, include_zero_counts):
    """Convert an observed array to a nested dictionary {sex: {age: {label: count}}}"""
```

This function is a shared utility across all observation export paths, converting numeric arrays into human-readable nested dictionaries.

#### Fallback Mechanism

The observation history export functions (`population_observation_history_to_readable_dict`, `spatial_population_observation_history_to_readable_dict`) all include fallback logic:

```python
obs = getattr(population, "record_observation", None)
if obs is None:
    return population_history_to_readable_dict(population, ...)
```

This means even when called on old data without observations, these functions will not crash — they fall back to the raw history parsing path.

### 6. Post-hoc Observation Path

Post-hoc observation (without modifying the recording mode, only applying observations during export) is implemented through `_build_history_observation_payload`:

```python
def _build_history_observation_payload(population, history, observation, groups, collapse_age, ...):
    # For each row of the history array:
    # 1. Parse tick, individual_count, sperm_storage
    # 2. Apply the observation or a temporarily constructed Observation
    # 3. Assemble into observation-format payload
```

This path has higher performance overhead — it needs to reconstruct state from each history snapshot and re-apply the observation. It is suitable for short histories or scenarios where different grouping perspectives are needed after the fact.

## Key Design Decisions

### 1. Unified _history Storage

Regardless of raw or observation mode, `_history` is always a `List[Tuple[int, NDArray[np.float64]]]`. The only difference is the array content (raw flat state vs. observation-aggregated array). This keeps the `get_history()` interface consistent.

### 2. 4D Mask Always Complete

`Observation.build_mask()` always returns a 4D mask `(n_groups, n_sexes, n_ages, n_genotypes)`, even for discrete generation (age=1) or `collapse_age=True` scenarios. `collapse_age` is only stored as metadata in the Observation and is read by functions like `_build_observation_payload` during export.

The reason is that the kernel requires a uniform memory layout — dynamically switching dimensions based on `collapse_age` within a Numba njit function would cause type instability.

### 3. Kernel Observation Performs Aggregation Only, Not Selection

The observation mask is used inside the kernel only for genotype dimension aggregation (`sum(axis=-1)`), not for filtering demes or sexes. Deme filtering is handled by `deme_selector` (spatial-specific), while the sex and age axes are not compressed — the `observed` array retains the `(n_sexes, n_ages)` dimensions.

### 4. Spatial's _observation_mask is Shared

All demes share the same `_observation_mask` (because the number of genotypes and genotype names are consistent across all demes). Only `deme_selector` is per-deme.

## Verification Points

When modifying observation recording-related code, ensure the following behaviors remain unchanged:

1. **Raw data recording when no observation is set**: when `record_observation = None`, `run()` behaves identically to before
2. **Backward compatibility of observation mode**: observation mode historical data can be correctly exported with `output_history()`
3. **Post-hoc observation correctness**: `output_history(observation=obs)` produces consistent results in both raw history and observation history modes
4. **Spatial aggregate verification**: the spatial aggregate from observation mode equals the per-deme grouped summation result
5. **Clone compatibility**: demes cloned by `SpatialBuilder` can independently set observations
6. **Python dispatch path**: recording under the `_should_use_python_dispatch()` fallback path is also correct

Test commands:
```bash
pytest                                       # all tests
pyright src/natal/                            # type checking
ruff check src/natal/                         # lint
```
