# Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding the population.
This chapter covers three modification approaches and the underlying mechanism.

## 1. `pop.update()` — Fluent Chain API

The simplest approach. Wraps the current config in a `Configurator` and writes immediately:

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .build()
)

# Single parameter
pop.update().competition(carrying_capacity=5000)

# Chain multiple parameters
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# Custom fields
pop.update().custom(temperature=35.0)
```

Each method calls `set_param(config, name, value)` which writes directly to
0-d ndarrays — no `freeze()` or rebuild needed.

## 2. `set_param()` — Low-Level String Interface

The underlying implementation of `pop.update()`. Useful in scripts, notebooks,
and objmode hooks:

```python
from natal.configurator import set_param

set_param(pop.config, "competition.carrying_capacity", 5000.0)
set_param(pop.config, "carrying_capacity", 5000.0)       # short name
set_param(pop.config, "reproduction.eggs_per_female", 100.0)
```

Resolution flow:
1. Look up parameter name in `parameters.py` registry (full name → short → alias)
2. Locate `PopulationConfig` field and index path
3. Write in-place: `config.carrying_capacity[()] = 5000.0`
4. Equilibrium-sensitive params (K / eggs / sex_ratio) auto-trigger `sync_equilibrium_metrics`

## 3. Direct Modification in Hooks

Fastest path — hook signature includes `config`, write directly in nopython:

```python
import natal as nt
from natal.discrete_population_config import DiscretePopulationConfig
from natal.population_state import DiscretePopulationState

@nt.hook(event="early", custom=True)
def environment_change(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    _deme_id: int,
) -> int:
    if state.n_tick == 10:
        config.carrying_capacity[()] *= 0.5
        config.expected_eggs_per_female[()] *= 0.7
        config.custom['temperature'][()] = 40.0
    return 0
```

### Performance Comparison

| Approach | Speed | Use Case |
|---|---|---|
| `config.field[()] = v` | Fastest (nopython) | You know the field name |
| `set_config_param(config, ID, v)` | Fast (nopython, integer routing) | Dynamic param selection |
| `hook_set_param(config, "name", v)` | Same as below (convenient single call) | Need string param names — cleaner syntax for single calls |
| `with objmode(): ...` | Same objmode cost | General Python fallback, can batch multiple calls in one block |

`hook_set_param` wraps `objmode` + `set_param` for convenient single-call usage. Performance is
identical to bare `with objmode()` — both cross the same Numba→Python boundary.

```python
from natal.configurator import hook_set_param

@nt.hook(event="early", custom=True)
def hook_with_names(state, config, deme_id):
    hook_set_param(config, "carrying_capacity", 5000.0)
    hook_set_param(config, "reproduction.eggs_per_female", 100.0)
    return 0
```

Note: multiple `hook_set_param` calls cross the objmode boundary multiple times.
For batch modification of many parameters, bare `with objmode()` is more efficient —
a single boundary crossing for all calls.

## 4. Custom Fields — `config.custom`

`config.custom` is a 0-d structured numpy array supporting arbitrary named fields.
Register at build time via `.custom()`, read/write via `[()]` at runtime:

```python
# Register at build time
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0, debug=False)
    .build()
)

# Read/write in hooks
@nt.hook(event="early", custom=True)
def seasonal_hook(state, config, _deme_id):
    season = int(config.custom['season_idx'][()])
    if season == 1:
        config.custom['temperature'][()] = 35.0

# Modify via update()
pop.update().custom(temperature=35.0, season_idx=1)
```

Supports three types: `bool`, `float`, `int`.

## 5. Spatial Population — Per-Deme Modification

`SpatialPopulation.update()` supports all-deme or single-deme modification,
matching the build-time `batch_setting` API:

```python
from natal.spatial_builder import batch_setting

# All demes
pop.update().competition(carrying_capacity=5000)

# Single deme (clone-on-write)
pop.update(deme=3).competition(carrying_capacity=8000)

# Batch per-deme (None = skip)
pop.update().competition(
    carrying_capacity=batch_setting([100, None, 300, None])
)
```

### Clone-on-Write

When multiple demes share config 0-d arrays (homogeneous setup), modifying a
single deme first copies those arrays to a private buffer, preventing other
demes from being affected. Detection uses `config.carrying_capacity` array
identity.

## 6. Under the Hood: 0-d ndarray + In-Place Write

All modification approaches converge on the same mechanism: 9 ecological
parameters are 0-d ndarrays; `field[()] = value` is atomic.

```
set_param(config, "carrying_capacity", 5000.0)
  → _resolve_param("carrying_capacity")  # lookup in parameters.py
  → config.carrying_capacity             # 0-d ndarray
  → carrying_capacity[()] = 5000.0       # in-place write
  → sync_equilibrium_metrics(config)     # auto-recompute
```

This guarantees identical behavior whether you use `pop.update()`, `set_param()`,
or direct hook modification.

## 7. Old vs New

| | Old (Builder) | New (Configurator) |
|---|---|---|
| Post-build modification | No native API | `pop.update()` chain |
| Hook-internal modification | Read-only (Op-based) | `config.field[()] = v` |
| Custom fields | `ConfigMutator` (removed) | `config.custom` + `.custom()` |
| Spatial per-deme | Not supported | `pop.update(deme=N)` + batch_setting |
| Low-level API | None | `set_param(config, name, value)` |
