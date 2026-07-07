# Builder → Configurator Migration Guide

In v0.2.0, `PopulationBuilder` and its subclasses (`DiscreteGenerationPopulationBuilder`, `AgeStructuredPopulationBuilder`, `SpatialBuilder`) were replaced by the `Configurator` chain API.

## What Changed

| Before (v0.1.x) | After (v0.2.0) |
|---|---|
| `PopulationBuilder(species).build()` | `DiscreteGenerationPopulation.setup(species).build()` |
| `.competition(carrying_capacity=...)` returns Builder | Returns `Configurator` (same chain syntax) |
| Parameters deferred until `build()` | Parameters written immediately to config arrays |
| `SpatialBuilder` class | `SpatialConfigurator` via `pop.update()` |
| `PopulationBuilder` class | Accessible via `setup(legacy_path=True)` |

## What Stays the Same

The chain API syntax is **identical** — code like this still works unchanged:

```python
pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="MyPop", stochastic=True)
    .initial_state({"male": {"WT|WT": 500}, "female": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .build()
)
```

## API Differences

### 1. Import Paths

```python
# v0.1.x (removed)
from natal.population_builder import PopulationBuilder
from natal.genetic_presets import HomingDrive

# v0.2.0
from natal import HomingDrive  # or nt.HomingDrive
```

### 2. `setup()` Returns a Configurator

```python
# v0.2.0 — setup() returns a Configurator, not a Builder
configurator = nt.DiscreteGenerationPopulation.setup(species=sp)
print(type(configurator))  # <class 'natal.configurator.discrete.DiscreteConfigurator'>
```

### 3. Legacy Builder Path

If you depend on the old Builder API, pass `legacy_path=True` to `setup()`:

```python
builder = nt.DiscreteGenerationPopulation.setup(species=sp, legacy_path=True)
```

### 4. New Runtime Modification

Configurator supports runtime modification that the old Builder couldn't:

```python
# After build, modify parameters without rebuilding
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100)
```

### 5. Parameter Changes

- `female_age_based_survival_rates` → `female_age_based_survival` (all `_rates` suffixes dropped)
- `species_scale`, `base_carrying_capacity`, `base_expected_num_adult_females` removed
- `carrying_capacity` is now a direct 0-d ndarray

### 6. `SpatialBuilder` → `SpatialConfigurator`

```python
# v0.1.x
builder = SpatialBuilder(species, topology)
pop = builder.build()

# v0.2.0
from natal.spatial import SpatialPopulation
pop = SpatialPopulation.setup(species=sp, topology=grid).build()
```

## Key Behavioral Changes

1. **Immediate writes**: Configurator chain methods write to NumPy arrays immediately, not deferred to `build()`. This is invisible for most code.
2. **Default `Species.unordered=True`**: `A|a` and `a|A` now produce the same `Genotype` instance. Set `unordered=False` if parent-of-origin tracking is needed.
3. **Hook signature unified**: `(state, config, deme_id=-1)`. Old `(ind_count, tick)` no longer works.
4. **Default survival**: Age-structured models default to 100% survival at all ages (was decaying values).
5. **`set_param()` / `hook_set_param()`**: New low-level APIs for runtime modification by parameter name.
