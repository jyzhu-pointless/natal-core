# Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding the population.
This chapter covers three scenarios:

- **Between-tick**: Python-side via `pop.update()` or `set_param()`
- **Inside hooks**: Numba nopython via `config.field[()] = v`
- **Spatial**: per-deme modification + clone-on-write

---

## 1. Between-Tick: `pop.update()`

`pop.update()` wraps the current config in a `Configurator`. Chain methods are identical
to build-time and changes take effect immediately:

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

# Custom fields (read/write in hooks)
pop.update().custom(temperature=35.0)
```

Each call routes through `set_param(config, name, value)` → writes 0-d ndarray in-place.

---

## 2. Between-Tick: `set_param()` — Low-Level API

The underlying implementation of `pop.update()`. For scripts and notebooks:

```python
from natal.configurator import set_param

set_param(pop.config, "competition.carrying_capacity", 5000.0)

# Full name, short name, or alias all work
set_param(pop.config, "carrying_capacity", 5000.0)
set_param(pop.config, "reproduction.eggs_per_female", 100.0)
set_param(pop.config, "eggs_per_female", 100.0)  # alias
```

Resolution (four steps):

1. Lookup in `parameters.py` registry: full name → short → alias
2. Locate config field and array index
3. Write in-place: `config.carrying_capacity[()] = 5000.0`
4. K / eggs / sex_ratio auto-trigger `sync_equilibrium_metrics`

---

## 3. Modification Inside Hooks

Hook signature is ``(state, config) → int``. ``config`` is writable in-place;
changes are immediately visible to subsequent hooks and simulation steps.
For spatial models that need per-deme branching, an optional ``deme_id`` parameter
can be added, but most hooks do not need it.

### 3.1 Direct Write: `config.field[()] = v`

Fastest path. Numba nopython, pure C-level ndarray access:

```python
from natal.population_config import DiscretePopulationConfig
from natal.data import DiscretePopulationState

@nt.hook(event="early", custom=True)
def environment_change(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
) -> int:
    if state.n_tick == 10:
        config.carrying_capacity[()] *= 0.5
        config.eggs_per_female[()] *= 0.7
        config.custom['temperature'][()] = 40.0
    return 0
```

> Direct writes do NOT auto-sync equilibrium. Age-structured models need a manual
> `sync_equilibrium_metrics(config)` call.

### 3.2 `hook_set_param(config, "name", v)` — Objmode Wrapped

Wraps `objmode` + `set_param` for cleaner single-call syntax. Same performance as
bare `with objmode()` (identical Numba→Python boundary):

```python
from natal.configurator import hook_set_param

@nt.hook(event="early", custom=True)
def recovery_hook(state, config):
    if state.n_tick == 10:
        hook_set_param(config, "carrying_capacity", 5000.0)
        hook_set_param(config, "eggs_per_female", 100.0)
    return 0
```

Each call crosses the objmode boundary once. For **batch modification** of multiple
parameters, bare `with objmode()` is more efficient — one boundary for all calls.

### 3.3 Bare `with objmode()`

When you need logging, file I/O, or batch parameter changes inside a hook:

```python
from numba import objmode
from natal.configurator import set_param

@nt.hook(event="early", custom=True)
def batch_hook(state, config):
    if state.n_tick == 10:
        with objmode():
            print(f"[tick={state.n_tick}] emergency recovery")  # logging
            set_param(config, "carrying_capacity", 5000.0)
            set_param(config, "eggs_per_female", 100.0)
            set_param(config, "sex_ratio", 0.5)
    return 0
```

### Comparison

| Method | Performance | When to use |
|---|---|---|
| `config.field[()] = v` | Fastest (nopython) | You know the field name |
| `hook_set_param(config, "name", v)` | Objmode boundary (clean single call) | Need string parameter names |
| `with objmode(): set_param(...)` | Objmode boundary (batch efficient) | Batch changes or Python ecosystem |

---

## 4. Custom Fields: `config.custom`

0-d structured numpy array. Register at build time, read/write via `[()]` at runtime:

```python
# Build-time registration
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0)
    .build()
)

# Inside hooks
@nt.hook(event="early", custom=True)
def seasonal_hook(state, config):
    temp = config.custom['temperature'][()]
    if int(config.custom['season_idx'][()]) == 1:
        config.custom['temperature'][()] = 35.0

# Runtime modification
pop.update().custom(temperature=35.0, season_idx=1)
```

Supported types: `bool`, `float`, `int`.

> **Note**: Custom fields are NOT in the parameter registry, so `set_param()` and `hook_set_param()` cannot address them. Use direct array writes (`config.custom["temperature"][()] = value`) or `pop.update().custom(temperature=30.0)` instead.

---

## 5. Spatial Population — Per-Deme Modification

`SpatialPopulation.update()` mirrors the panmictic interface, with additional
per-deme and batch support:

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

**Clone-on-write**: In homogeneous setups, multiple demes share the same 0-d ndarrays.
Modifying a single deme first copies those arrays to a private buffer, isolating
the change from other demes.

---

## 6. Under the Hood

All modification paths converge on the same mechanism:

```
set_param / pop.update() / hook direct write
  → config.carrying_capacity           # 0-d ndarray
  → carrying_capacity[()] = 5000.0     # in-place write (atomic)
  → sync_equilibrium_metrics(config)   # K/eggs/sr auto-trigger
```

9 ecological parameters are 0-d ndarrays: K, eggs, sex_ratio, sperm_displacement_rate,
low_density_growth_rate, juvenile_growth_mode, generation_time, expected_competition_strength,
expected_survival_rate.

### `set_config()` — Whole-Config Replacement

`pop.set_config(new_config)` replaces the population's entire config object at once. This is useful when rebuilding a config from scratch (e.g. after modifying custom field schemas). The new config must have the same type (`PopulationConfig` or `DiscretePopulationConfig`) as the original.

Internally, Configurator's `custom()` method with new field names triggers this path: it rebuilds the custom structured array and calls `set_config()` to propagate the new config back to the population.

---

## 7. Parameter Reference

Parameters are grouped by domain, matching the Configurator chain API methods.

| Domain | Parameter | Aliases | Models | set_param |
|---|---|---|---|---|
| setup | `stochastic` | — | both | ❌ build-time only |
| setup | `continuous_sampling` | — | both | ❌ build-time only |
| setup | `fixed_egg_count` | — | both | ❌ build-time only |
| setup | `has_sex_chromosomes` | — | both | ❌ build-time only |
| age_structure | `n_ages` | — | age-structured | ❌ build-time only |
| age_structure | `new_adult_age` | — | age-structured | ❌ build-time only |
| age_structure | `generation_time` | — | age-structured | ❌ build-time only |
| survival | `female_age_based_survival` | — | age-structured | ✅ |
| survival | `male_age_based_survival` | — | age-structured | ✅ |
| reproduction | `eggs_per_female` | `expected_eggs_per_female` | both | ✅ |
| reproduction | `sex_ratio` | — | both | ✅ |
| reproduction | `sperm_displacement_rate` | — | both | ✅ |
| competition | `carrying_capacity` | — | both | ✅ |
| competition | `low_density_growth_rate` | — | both | ✅ |
| competition | `juvenile_growth_mode` | `growth_mode` | both | ✅ |
| fitness | `viability` | — | both | ❌ tensor |
| fitness | `fecundity` | — | both | ❌ tensor |
| fitness | `sexual_selection` | — | both | ❌ tensor |
| fitness | `zygote_viability` | — | both | ❌ tensor |
| migration | `migration_rate` | — | spatial | spatial-specific |

## 8. Old vs New

| | Old (Builder) | New (Configurator) |
|---|---|---|
| Post-build modification | Not supported | `pop.update()` |
| Hook modification | Declarative Op | `config.field[()] = v` |
| Custom fields | ConfigMutator (removed) | `config.custom` |
| Spatial per-deme | Not supported | `pop.update(deme=N)` + batch_setting |
| Low-level API | None | `set_param(config, name, value)` |
