# SpatialPopulation Initialization Optimization Plan

## Current State and Bottlenecks

### 1. Each deme independently goes through the builder pipeline

```python
# Current approach (51×51 = 2601 demes)
demes = [build_deme(species, idx, ...) for idx in range(2601)]
```

Each `build_deme` call goes through:
- Species genotype resolution (index lookup)
- Numba hook compilation (`_compile_hooks` → `CompiledEventHooks.from_compiled_hooks`)
- Config/fitness array allocation and population
- `_finalize_hooks` triggering codegen

These steps are completely duplicated across homogeneous demes. Measured: ~1s for 1000 demes, ~2.6s for 2601 demes.

### 2. Global config sharing is a manual operation

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
    deme.import_config(shared_config)
```

- If forgotten, each deme holds an independent config, wasting memory
- `import_config` can only share scalar fields; Numba compilation products remain independent

### 3. No batch expression for heterogeneous configs

Currently requires conditional branching in a loop:

```python
def build_deme(..., idx):
    k = 10000 if idx < n_demes // 2 else 5000
    ...
    .competition(carrying_capacity=k)
```

No declarative batch interface for defining spatial gradient patterns (high-left low-right, high-center low-edges, etc.).

### 4. No lazy initialization between demes

All demes must be ready before constructing `SpatialPopulation`.

---

## Design Goals

1. **Preserve Builder chained API syntax** — do not change the calling style of existing builders
2. **`batch_setting` for vectorized heterogeneous parameters** — embed heterogeneous data in chained calls
3. **Zero redundant construction for homogeneous demes** — share template compilation products
4. **Progressive complexity** — homogeneous scenarios need only a few parameters, heterogeneous scenarios add parameters

---

## Core Design: SpatialBuilder + batch_setting

### `batch_setting` Wrapper

`batch_setting` is a marker parameter wrapper, indicating "this is a parameter that varies across demes":

```python
batch_setting(value)                          # All demes identical (but explicitly marked as expandable)
batch_setting([10000, 5000, 8000, ...])        # List/array: by deme index
batch_setting(spatial=lambda i, x, y: ...)     # Callback: receives (deme_id, x, y), returns scalar
batch_setting({
    0: 10000,                                  # Specify only some demes
    range(1, 10): 5000,                        # Range mapping
    "rest": 8000,                              # Default for the rest
})
```

When a builder parameter is a `batch_setting` object, the builder internally switches to spatial batch mode.

### SpatialBuilder Chained API

```python
pop = SpatialPopulation.builder(species, n_demes=N, topology=HexGrid(rows=N, cols=N)) \
    .setup(name="spatial_demo") \
    .initial_state(
        female={"WT|WT": 5000, "Dr|WT": 50},      # Not batch → same for all demes
        male={"WT|WT": 5000, "Dr|WT": 50},
    ) \
    .reproduction(eggs_per_female=50) \
    .competition(
        carrying_capacity=batch_setting(spatial=lambda i, x, y: 10000 if x < N//2 else 5000),
        juvenile_growth_mode="concave",
        low_density_growth_rate=6.0,
    ) \
    .presets(drive) \
    .fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}}) \
    .migration(kernel=..., migration_rate=0.1) \
    .build()
```

### SpatialBuilder Internal Flow

```
At build() time:
1. Scan all parameters, extract batch_setting objects
2. If no batch_setting → homogeneous optimization path (see Phase 1a below)
3. If batch_setting exists:
   a. Expand each batch_setting into per-deme value lists (length n_demes)
   b. Group demes by config equivalence
   c. For each group:
      - Build template deme with the first deme's config (full builder pipeline once)
      - Clone remaining demes in the group from the template (shallow copy config, registry, hooks)
   d. Construct SpatialPopulation from cloned demes + topology/migration
```

---

## Implementation Roadmap

### Phase 1a: Homogeneous SpatialBuilder (no batch_setting)

Without `batch_setting`, all demes are completely identical. SpatialBuilder only needs to build one template, then N shallow copies.

```python
pop = SpatialPopulation.builder(species, n_demes=2601, ...) \
    .setup(...).initial_state(...).reproduction(...).competition(...) \
    .presets(drive).build()

# The following two lines are equivalent (decomposing internal logic):
# template = DiscreteGenerationPopulation.builder(...).build()
# demes = [clone(template) for _ in range(2601)]
# pop = SpatialPopulation(demes, topology=...)
```

Expected: 2601 demes ~50ms (excluding the template's first build of 2-3ms).

### Phase 1b: Heterogeneous SpatialBuilder (with batch_setting)

When at least one `batch_setting` parameter is detected, group by config equivalence.

```python
pop = SpatialPopulation.builder(species, n_demes=4, ...) \
    .setup(...).initial_state(...) \
    .competition(carrying_capacity=batch_setting([10000, 8000, 6000, 4000])) \
    .build()

# Internally: 4 different configs → 4 templates → each shallow copied (only one template built per group)
# If only 2 heterogeneous groups (e.g., first 2 K=10000, last 2 K=5000) → only 2 templates built
```

### Phase 1c: `batch_setting.spatial` Convenience API

Spatial gradient parameters can be quickly expressed via `batch_setting.spatial`:

```python
batch_setting.spatial(lambda x, y: 10000 if abs(x) < 5 else 5000, topology=hex_grid)
```

Receives topology coordinates, implicitly fills all deme positions.

### Phase 1d: `set_hook` Integration in SpatialBuilder

```python
SpatialPopulation.builder(...) \
    ...
    .set_hook(event="early", fn=my_hook, deme=batch_setting([0, 1, 2])) \
    .build()
```

The deme selector automatically expands to `set_hook` calls for each target deme.

---

## Subsequent Phases (as needed)

### Phase 2: Lightweight Per-Deme Constructor

When performance requirements exceed the DSL convenience of the builder, provide array-level construction that bypasses the builder:

```python
# Internal construction path (users do not need to call directly)
DemeFactory.quick(
    species=species,
    individual_count=np.array(...),  # (n_sexes, n_ages, n_genotypes)
    config=PopulationConfig(...),
    registry=shared_registry,
)
```

### Phase 3: SpatialPopulation Accepts Arrays Directly

```python
SpatialPopulation(
    demes=...,              # Traditional approach
    species=species,        # Alternative: provide arrays directly
    individual_counts=...,  # (n_demes, n_sexes, n_ages, n_genotypes)
    config=...,             # shared or bank
    topology=...,
    migration=...,
)
```

---

## `batch_setting` Detection and Expansion Mechanism

### Detect

```python
class batch_setting:
    """Marker wrapper for per-deme varying parameters."""
    
    _kind: Literal["scalar", "array", "spatial", "partial"]
    _data: Any
    
    def __init__(self, value):
        if callable(value) and "spatial" in hint: ...
        ...
```

Each builder setter method checks the parameter type:

```python
def competition(self, carrying_capacity=None, ...):
    if isinstance(carrying_capacity, batch_setting):
        self._batch_params["carrying_capacity"] = carrying_capacity
    else:
        self._params["carrying_capacity"] = carrying_capacity
    return self
```

### Expand

At `build()` time, `batch_setting` is expanded into a list of length `n_demes`:

```python
# Expansion algorithm
for name, bs in self._batch_params.items():
    if bs._kind == "scalar":
        values = [bs._data] * n_demes
    elif bs._kind == "array":
        values = list(bs._data)  # length must == n_demes
    elif bs._kind == "spatial":
        values = [bs._fn(self._topology.from_index(i)) for i in range(n_demes)]
    ...
```

### Group

Group by expanded config equivalence:

```python
groups: dict[tuple, list[int]] = {}
for i in range(n_demes):
    cfg_key = tuple(
        params[name][i] if isinstance(params[name], list) else params[name]
        for name in config_fields
    )
    groups.setdefault(cfg_key, []).append(i)
```

---

## Performance Expectations

| Scenario | Current | Optimized |
|----------|---------|-----------|
| Homogeneous 2601 demes | ~2.6s | ~50ms (+ template first build 2-3ms) |
| 2-group heterogeneous 2601 demes | ~2.6s | ~55ms (+ 2 templates at 2-3ms each) |
| Fully heterogeneous 2601 demes | ~2.6s | ~2.6s (builder cannot be skipped, but syntax is simplified) |

---

## Technical Risks

| Risk | Mitigation |
|------|------------|
| Config group key after batch expansion is unhashable (contains NumPy arrays) | Use `id(arr)` or serialized digest |
| Sharing `_compiled_hooks` reference when cloning demes leads to state leakage | Copy-on-write: duplicate on demand via `set_hook`/`remove_hook` |
| Does `PopulationConfig` support `_replace`? | It is a NamedTuple, confirmed usable |
| Relationship between SpatialBuilder and existing `DiscreteGenerationPopulationBuilder` | SpatialBuilder holds per-deme builders internally, reuses their validation logic |

---

## Interface Design Principles

1. **Do not break existing APIs** — leave `DiscreteGenerationPopulation.builder()` and existing `SpatialPopulation.__init__` untouched
2. **`batch_setting` is an optional enhancement** — demos still run without it; adding it provides cleaner syntax
3. **Zero-copy preferred** — share references for homogeneous configs/hooks, only copy differing behavior
4. **Validate upfront** — validate batch_setting length/coordinates against n_demes at `build()` time
