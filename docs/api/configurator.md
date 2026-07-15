# configurator Module

Parameter configuration — build and runtime modification of population models.

## Overview

`Configurator` is the unified API for setting and modifying simulation parameters
identically at build time and runtime.

Key features:

- **Fluent chain API** — `.competition(carrying_capacity=10000).reproduction(eggs_per_female=50).build()`
- **Immediate writes** — every chain method writes to NumPy arrays in-place
- **Runtime modification** — `pop.update().competition(carrying_capacity=5000)` without rebuilding
- **Model-specific subclasses** — `DiscreteConfigurator` / `AgeStructuredConfigurator`
  with narrowed parameter signatures
- **Preset/modifier/fitness** — applied directly to config arrays, no deferred execution
- **Equilibrium sync** — `carrying_capacity` / `eggs_per_female` / `sex_ratio` changes
  auto-trigger `sync_equilibrium_metrics`

## Quick Start

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

# Build-time
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .custom(temperature=25.0)
    .build()
)

# Runtime
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)
```

## DiscreteConfigurator

Configurator for `DiscreteGenerationPopulation`. Parameters are narrowed to the
discrete-generation model.

```python
# Create
cfg = nt.Configurator.for_discrete(species)

# Or via setup()
cfg = nt.DiscreteGenerationPopulation.setup(species)

# Chain configuration — only discrete-relevant parameters are shown
cfg.age_structure(n_ages=2, new_adult_age=1)          # fixed 2 ages
cfg.reproduction(
    eggs_per_female=50,               # eggs per female per tick
    sex_ratio=0.5,                    # fraction female offspring
    female_adult_mating_rate=1.0,     # adult female mating probability
    male_adult_mating_rate=1.0,       # adult male mating probability
)
cfg.survival(
    female_age0_survival=0.9,         # female juvenile survival
    male_age0_survival=0.9,           # male juvenile survival
)
cfg.competition(
    carrying_capacity=10000,          # equilibrium carrying capacity K
    low_density_growth_rate=6.0,      # low-density growth rate r
    juvenile_growth_mode="concave",   # density-regulation function
)
```

## AgeStructuredConfigurator

Configurator for `AgeStructuredPopulation`. Supports per-age array parameters
and the Champer equilibrium model.

```python
cfg = nt.Configurator.for_age_structured(species)

cfg.age_structure(n_ages=8, new_adult_age=2)

# Per-age parameters accept flexible input:
#   scalar — fills all ages
#   list — per-age values
#   dict — sparse map {age: value}
#   callable — lambda age: ...
cfg.reproduction(
    eggs_per_female=100,
    sex_ratio=0.5,
    female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.0],
)
cfg.survival(
    female_age_based_survival=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0],
    male_age_based_survival=[1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.0],
)
cfg.competition(
    carrying_capacity=5000,
    low_density_growth_rate=6.0,
    juvenile_growth_mode="logistic",
    competition_strength=5.0,
    # Champer model — custom equilibrium distribution
    equilibrium_distribution=custom_dist,
)
```

## Shared Methods

Both Configurator subclasses expose these methods:

### `setup(**flags)`
```python
cfg.setup(name="MyPop", stochastic=False)
```
Configure simulation flags and population name.

### `initial_state(individual_count, sperm_storage=None)`
```python
cfg.initial_state(individual_count={
    "female": {"WT|WT": [0, 200, 150, 100]},
    "male":   {"WT|WT": [0, 200, 150, 100]},
})
```
Set the initial population distribution. *individual_count* is required,
format: `{sex: {genotype: age_data}}`.

### `custom(**fields)`
```python
cfg.custom(temperature=25.0, debug=True)
```
Register named fields stored in `config.custom`. Hooks read/write via
`config.custom['name'][()]`.

### `with_observation(groups, *, collapse_age=False)`
```python
cfg.with_observation({
    "adult_female": {"genotype": ["WT|WT"], "sex": "female", "age": [1]},
})
```
Register observation groups, applied at `build()` time. `groups` can be a dict
of name-to-spec, a list of specs, or `None` for one-group-per-genotype.
`collapse_age` controls whether the age axis is collapsed in exports.

> **Build-time only:** Calling `with_observation()` on a live Population (via `pop.update().with_observation(...)`) raises `RuntimeError`. Recording rules are frozen at build time and cannot change afterwards. Use `pop.create_observation()` for runtime queries.

### `presets(*presets)`
```python
cfg.presets(homing_drive)
```
Apply genetic presets immediately — writes directly to config arrays (not deferred).

### `reconfigure_preset(preset, **changes)`
```python
cfg.reconfigure_preset(homing_drive, homing_rate=0.95)
```
Modify a registered preset parameter and re-apply from baselines. Restores
baseline fitness/gamete arrays, applies the updated preset parameters, and
syncs equilibrium. Requires that the preset was first registered via
`presets()`.

### `modifiers(gamete_modifiers=None, zygote_modifiers=None)`
```python
cfg.modifiers(gamete_modifiers=[my_mod])
```
Register gamete/zygote modifiers, immediately rebuilding genotype/gamete maps.

### `fitness(viability=None, fecundity=None, sexual_selection=None, zygote_viability=None, mode="replace")`
```python
cfg.fitness(
    viability={"WT|WT": 0.8, "WT|Var": 1.0},
    fecundity={"female": {"WT|Var": 1.2}},
    mode="multiply",
)
```
Write fitness values to config arrays. Flat dicts apply to both sexes;
nested `{"female": {...}, "male": {...}}` for sex-specific values.
`mode="replace"` overwrites, `mode="multiply"` scales existing values.

### `hooks(*hook_items)`
```python
cfg.hooks(my_hook)
```
Register event hooks, forwarded to the Population constructor at `build()` time.

### `build(name=None, hooks=None)`
```python
pop = cfg.build(name="MyPop")
```
Sync equilibrium metrics and create the Population object.

### `apply()`
```python
cfg.apply()
```
Run equilibrium sync without creating a Population. Normally unnecessary —
`build()` calls `apply()` internally.

## Runtime Modification

### `pop.update()`
```python
# Single parameter
pop.update().competition(carrying_capacity=5000)

# Chained multiple parameters
pop.update().reproduction(eggs_per_female=100).competition(carrying_capacity=10000)

# Custom fields
pop.update().custom(temperature=35.0)
```
All changes write immediately to 0-d ndarrays — no `freeze()` or rebuild needed.

### Inside Hooks
```python
@nt.hook(event="early", custom=True)
def my_hook(state, config, deme_id):
    config.carrying_capacity[()] = 5000
    config.custom['temperature'][()] = 40.0
```

### Spatial Population
```python
# All demes
pop.update().competition(carrying_capacity=5000)

# Single deme (clone-on-write)
pop.update(deme=3).competition(carrying_capacity=8000)

# Batch per-deme
from natal.spatial import batch_setting
pop.update().competition(
    carrying_capacity=batch_setting([100, 200, 300, 400])
)
```

## Low-Level API

### `set_param(config, name, value)`
```python
from natal.configurator import set_param
set_param(config, "competition.carrying_capacity", 5000.0)
set_param(config, "carrying_capacity", 5000.0)  # short name also works
```
The foundation of all higher-level APIs. Resolves parameter names through the
`parameters.py` registry, locates the config field and index, and writes in-place.
Equilibrium-sensitive parameters (K / eggs / sex_ratio) auto-trigger sync.

### `hook_set_param(config, name, value)`
```python
from natal.configurator import hook_set_param

@nt.hook(event="early", custom=True)
def my_hook(state, config, deme_id):
    hook_set_param(config, "carrying_capacity", 5000.0)
    hook_set_param(config, "reproduction.eggs_per_female", 100.0)
    return 0
```
Wraps `objmode` + `set_param` for callable-from-njit convenience. Use when
you need string-name routing inside hooks. The fastest path remains direct
`config.field[()] = v`.

### `Configurator.for_config(config)`
```python
cfg = nt.Configurator.for_config(pop.config)
```
Returns `DiscreteConfigurator` or `AgeStructuredConfigurator` based on config type.

## Type Hierarchy

```
Configurator                  # base: setup, build, apply, presets, fitness, hooks...
├── DiscreteConfigurator      # + competition(discrete), reproduction(discrete), survival(discrete)
└── AgeStructuredConfigurator # + competition(Champer), reproduction(per-age), survival(per-age)
```


