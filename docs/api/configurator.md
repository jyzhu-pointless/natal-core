# configurator Module

Parameter configuration — build and runtime modification of population models.

## Overview

`Configurator` is the unified API for setting and modifying simulation parameters
identically at build time and runtime.

## API Reference

::: natal.frontend.configurator._base.Configurator
    options:
      heading_level: 3
      filters:
        - "!^_"

Key features:

- **Fluent chain API** — `.competition(carrying_capacity=10000).reproduction(eggs_per_female=50).build()`
- **Validated writes** — build-time methods update the model definition; runtime methods submit validated updates to the Rust session
- **Runtime modification** — `pop.update().competition(carrying_capacity=5000)` without rebuilding
- **One configurator** — the same `Configurator` dispatches parameters for discrete and age-structured models
- **Preset/modifier/fitness** — declarations compile into genetic tables; runtime reconfiguration updates the existing session
- **Equilibrium metrics** — derived on read (`pop.params.expected_*`); writes to the
  sensitive parameters (K / eggs / sex_ratio / the Champer overrides) refresh the
  draft cache

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

## Discrete-Generation Configuration

Use `Configurator.for_discrete(species)` or the population's `.setup(species)`
entry point. The following fragment assumes `species` is already defined.

```python
# Create
cfg = nt.Configurator.for_discrete(species)

# Or via setup()
cfg = nt.DiscreteGenerationPopulation.setup(species)

# Chain configuration — only discrete-relevant parameters are shown
# Discrete drafts are normalized to 2 ages at construction;
# age_structure() is not applicable here (it raises RuntimeError).
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
    juvenile_growth_mode="beverton_holt",   # density-regulation function
)
```

## Age-Structured Configuration

Use `Configurator.for_age_structured(species)`. It supports per-age parameters
and the Champer equilibrium model. The following fragment assumes `species`
and the optional `custom_dist` equilibrium distribution are already defined.

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

The unified Configurator exposes these methods. Fragments below assume a
build-time `cfg` and any referenced presets or modifiers already exist.

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

### `custom(**kwargs)`
```python
cfg.custom(temperature=25.0, debug=True)
```
Register typed named values. Read snapshots through `pop.config.custom`;
write runtime values with `pop.update().custom(...)` or, inside a callback,
`ctx.update().custom(...)`. Scalars are plain Python values, not 0-D arrays.

### `with_observation(groups, *, collapse_age=False)`
```python
cfg.with_observation({
    "adult_female": nt.IndividualSelector(
        ztype="WT|WT", sex="female", age=[1]
    ),
})
```
Register the canonical observation applied by `pop.observe()`. `groups` must be
a non-empty ordered mapping from non-empty string labels to
`IndividualSelector` values. The mapping order becomes the group-axis order.
`collapse_age=True` sums and removes the age axis.

Every built Population, including `SpatialPopulation`, has an immutable
`pop.observation`. If `with_observation()` is omitted, `build()` installs an
identity observation with one lossless group per ZType.

> **Build-time only:** Calling `with_observation()` on a live Population via
> `pop.update().with_observation(...)` raises `RuntimeError`. Observation rules
> cannot change after `build()`.

### `record_history(*, mode="raw", max_rows=None)`
```python
cfg.record_history(mode="observation", max_rows=5000)
```
Set the recording mode and capacity for the population's history. Must be called
during the build phase — calling on a runtime Configurator raises `RuntimeError`.

`with_observation()` and `record_history()` are **independent** — chain order
does not matter. A `with_observation()` call defines which groups to observe;
`record_history()` defines whether History stores full state or the values from
that observation. Omitting both methods therefore gives an identity
`pop.observation` and raw `pop.history`.

| mode | Description |
|------|-------------|
| `"raw"` (default) | Full-state recording — every genotype count and sperm storage is stored per snapshot |
| `"observation"` | Compressed — only values from the canonical observation are stored; an omitted `with_observation()` uses identity groups |

`max_rows` controls FIFO eviction: when the number of stored records exceeds
this value, the oldest entries are removed. `None` applies the population's
bounded default (`max_history`, 5000 rows); evicted rows also drop their
paired restore checkpoints.

`pop.history` is a typed `History` container. Raw history exposes `ticks`,
`individual_count`, and, where applicable, `sperm_storage`; observation-mode
history exposes `ticks` and `values`. A raw history can be projected later with
`pop.history.observe(pop.observation)`. This validates the Population layout
fingerprint before applying the observation.

### `presets(*presets)`
```python
cfg.presets(homing_drive)
```
Register genetic presets in the model definition. Construction compiles their
rules; runtime reconfiguration replaces validated tables without resetting the session.

### `reconfigure_preset(preset, **changes)`
```python
pop.update().reconfigure_preset(homing_drive, drive_conversion_rate=0.95)
```
Modify a registered preset parameter and re-apply from baselines. Restores
baseline fitness/gamete arrays and recompiles the genetic tables in place.
Equilibrium metrics follow on read. Requires that the preset was first
registered via `presets()`.

### `modifiers(gamete_modifiers=None, zygote_modifiers=None)`
```python
cfg.modifiers(gamete_modifiers=[my_mod])
```
Register gamete/zygote modifier rules for compilation into genotype/gamete maps.

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

### `hooks(*hook_items, event=None, priority=0, deme="*", name=None)`
```python
cfg.hooks(my_hook)
```
Register event hooks. On a build-time chain the registration is stored and
compiled when `build()` runs; on a runtime chain (`pop.update().hooks(...)`) it
takes effect immediately.

### `build(name=None, hook_items=None)`
```python
pop = cfg.build(name="MyPop")
```
Create the Population object: finalize declarations, apply optional index
compression, and freeze the observation and history layout. `hook_items`
accepts the same item shapes as `hooks()` and registers them together with
any hooks already stored by `hooks()`.

### `apply()`
```python
cfg.apply()
```
A no-op compatibility shim that returns `self`; `build()` calls it internally.
Equilibrium metrics are derived on read (`pop.params.expected_*`), so there are
no stored copies to sync.

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
Each runtime call validates and commits its own update. Separate chained calls
are separate transactions; the existing Rust session and random stream continue.

### Inside Hooks

This fragment continues the Quick Start population above. A Python callback
receives one `TickContext`; its updates commit with the callback's candidate
state when the callback completes successfully.

```python
@nt.hook(event="early")
def my_hook(ctx: nt.TickContext) -> int:
    ctx.update().competition(carrying_capacity=5000)
    ctx.update().custom(temperature=40.0)
    return 0

pop.update().hooks(my_hook)
pop.run(1)
assert pop.params.carrying_capacity == 5000
assert pop.config.custom["temperature"] == 40.0
```

### Spatial Population

This fragment assumes `spatial` is a built spatial population with four demes.
Use the validated parameter surface to change runtime ecology:

```python
# All demes.
spatial.params.tensor_write("carrying_capacity", 5000.0)

# One deme.
spatial.deme(3).write_ecology("carrying_capacity", 8000.0)

# A separate value for every deme.
spatial.params.tensor_write("carrying_capacity", [100.0, 200.0, 300.0, 400.0])
```

## Low-Level API

### `set_param(config, name, value)`
```python
from natal.frontend.configurator import set_param

draft = pop.config                                     # detached query snapshot
draft = set_param(draft, "competition.carrying_capacity", 5000.0)
draft = set_param(draft, "carrying_capacity", 5000.0)   # short name also works
```
A **draft-level** function: it resolves names through the `parameters.jsonc`
registry, locates the config field, and commits the value into the draft it was
given. It is *not* the runtime write path — `pop.update()` and
`pop.params` are, and they additionally sync the Rust session and append a
parameter-log row. Ecology scalars are NamedTuple slots written through
`_replace`, so the returned draft must be rebound; array-backed fields (custom
slots, vector/tensor contents) are mutated in place and return the same draft.
Equilibrium metrics are derived on read, so no stored copies need syncing.

### Declarative Parameter Updates

A parameter operation needs no Python callback. This fragment continues the
Quick Start population and keeps the declarative `Op` format:

```python
pop.update().hooks(nt.Op.set_param("carrying_capacity", "K * 0.5", event="early"))
```

For Python callback logic, use `ctx.params` or `ctx.update()` as shown above.
Direct mutation of a returned configuration snapshot does not update a session.

### `Configurator.for_config(config)`
```python
cfg = nt.Configurator.for_config(pop.config)
```
Returns the unified `Configurator` for the supplied draft. A configuration
snapshot obtained from `pop.config` is isolated; use `pop.update()` when the
intention is to modify a running population.

## Configurator Ownership

`Configurator` is one implementation for both model kinds. Construction keeps
model declarations; `pop.update()` binds it to runtime parameter updates.
`pop.config` is a query snapshot, so mutation through that snapshot cannot
replace the explicit runtime write path.
