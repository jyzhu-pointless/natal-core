# Builder to Configurator Migration Guide

In v0.2.0, `PopulationBuilder` and its subclasses (`DiscreteGenerationPopulationBuilder`,
`AgeStructuredPopulationBuilder`, `SpatialBuilder`) were replaced by the `Configurator`
chain API, and there is **no** `legacy_path` escape hatch -- the old Builder classes and
`setup(legacy_path=True)` do not exist.

## Change Summary

| Before (v0.1.x) | After (v0.2.0) |
|---|---|
| `PopulationBuilder` / `setup()` returns a Builder | `setup()` returns a `Configurator`; `build()` produces the population object |
| `.competition(...)` deferred to `build()` | parameters are written into the config arrays immediately (same chain syntax) |
| `SpatialBuilder(species, topology)` | `SpatialPopulation.builder(species, n_demes, pop_type)` returns a `SpatialConfigurator` |
| Hook signature `(ind_count, tick)` | `(state, config, deme_id)` (njit era) replaced by single-parameter `TickContext` callbacks / declarative `Op`s |

## What Did Not Change

The chain-API syntax is **identical** -- the following code needs no changes:

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

### 1. Import paths

```python
# The top-level user API is unchanged: import from natal (or `import natal as nt`)
from natal import Species, HomingDrive, Op
```

Concrete module paths live under `natal.frontend.*` (genetics structures/entities/patterns
are in `natal.frontend.genetics`, `natal.frontend.patterns`, ...).

### 2. `setup()` returns a Configurator

```python
# v0.2.0 -- setup() returns a Configurator, not a Builder
configurator = nt.DiscreteGenerationPopulation.setup(species=sp)
print(type(configurator))  # <class 'natal.frontend.configurator._base.Configurator'>
```

### 3. Runtime modification (new)

The Configurator enables runtime modification the old Builder could not do:

```python
# modify parameters after build without rebuilding
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100)

# preferred parameter surface: pop.params (bounds-validated + snapshot log)
pop.params.sex_ratio = 0.55
```

### 4. Parameter renames

- `female_age_based_survival_rates` -> `female_age_based_survival` (all `_rates` suffixes removed)
- `species_scale`, `base_carrying_capacity`, `base_expected_num_new_adult_females` removed
- `carrying_capacity` is now a direct plain scalar

### 5. `SpatialBuilder` -> `SpatialConfigurator`

```python
# v0.2.0 -- declare the deme count and population type first, then the spatial chain
from natal.frontend.spatial import SpatialPopulation

pop = (
    SpatialPopulation.builder(species=sp, n_demes=4, pop_type="age_structured")
    .setup(name="demo", stochastic=False)
    .age_structure(n_ages=4, new_adult_age=1)
    ...
    .migration(adjacency=adjacency, migration_rate=0.1)
    .build()
)
```

Note: there is no `SpatialPopulation.setup(...)` static method -- the entry point is
`SpatialPopulation.builder(...)`.

## Key Behavior Changes

1. **Immediate writes**: Configurator chain methods write into the NumPy arrays immediately rather than deferring to `build()`. Transparent to most code.
2. **`Species.unordered=True` by default**: `A|a` and `a|A` now produce the same `Genotype` instance. Set `unordered=False` to track parental origin.
3. **Unified hook shapes**: declarative (no params, returns `List[HookOp]`), single-parameter callback (`TickContext`), and selector callback (`selectors={...}`); the `(state, config, deme_id)` three-parameter signature is no longer available.
4. **Default survival**: age-structured models default to 100% survival at adult ages (`age >= new_adult_age`) and 0 at juvenile ages — without an explicit `female_age0_survival` / `male_age0_survival` the population dies out quickly (the discrete model defaults to `age0=1`, `age1=0`).
5. **Runtime parameter writes**: `pop.params.<name> = v` / `pop.update().<method>(...)` / `pop.params.tensor_write(...)` / `Op.set_param(...)`; every actual change is recorded in `pop.params_log`. `set_param(draft, name, value)` is a draft-level function (rebind the return value): it neither writes a running population nor logs a row; use `pop.params_log_details` for the complete audit.
