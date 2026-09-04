# Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding the population.
This chapter covers three scenarios:

- **Between-tick**: Python-side via `pop.update()` or `pop.params.<name> = v`
- **Inside hooks**: via the callback hook's `pop.params` (`TickContext`), or declaratively via `Op.set_param`
- **Spatial**: per-deme writes (`pop.params.tensor_write` / `deme(i).write_ecology`)

---

## 1. Between-Tick Modification: `pop.update()`

`pop.update()` returns a `Configurator` wrapper over the current config. Chained methods are identical to the build chain and take effect immediately:

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

# one parameter
pop.update().competition(carrying_capacity=5000)

# several parameters chained
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# custom fields (readable/writable from hooks)
pop.update().custom(temperature=35.0)
```

Each call internally goes through `set_param(config, name, value)` -> in-place write of the plain scalar.

## 2. Between-Tick Modification: The `pop.params` Surface

`pop.params.<name> = value` is the preferred runtime write channel: attribute writes are jsonc-bounds-validated and reach the draft, the live Rust session, and the parameter snapshot log. Reads return the current value:

```python
pop.params.carrying_capacity = 5000.0
print(pop.params.carrying_capacity)  # current value
# vector / tensor parameters use a dedicated channel
pop.params.tensor_write("survival_rates", np.ones((2, 2)))
```

Every actual write appends one `(tick, name, old, new)` row to `pop.params_log`:

```python
for row in pop.params_log:
    print(f"tick={row[0]}  {row[1]}  {row[2]} -> {row[3]}")
```

---

## 3. Between-Tick Modification: The `set_param()` Low-Level API

The underlying implementation of `pop.update()`. Suitable for scripts and notebooks:

```python
from natal.frontend.configurator import set_param

set_param(pop.config, "competition.carrying_capacity", 5000.0)

# full name, short name, and aliases all work
set_param(pop.config, "carrying_capacity", 5000.0)
set_param(pop.config, "reproduction.eggs_per_female", 100.0)
set_param(pop.config, "eggs_per_female", 100.0)  # alias
```

Four internal steps:

1. Look up the `parameters.jsonc` registry: full name -> short name -> alias
2. Locate the config field and the array index
3. Write in place: `config.carrying_capacity[()] = 5000.0`
4. After K / eggs / sex_ratio changes, `sync_equilibrium_metrics` runs automatically

---

## 4. Inside Hooks

The callback hook's `pop.params` and `pop.update()` share the same writer stack, so their write semantics are identical:

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="early")
def heatwave(pop: TickContext) -> int:
    if pop.tick == 10:
        pop.params.carrying_capacity = 2000.0
        pop.params.eggs_per_female = 100.0
        pop.params.sex_ratio = 0.55
    return 0
```

Rules and notes:

- Attribute writes target ecology scalars (`carrying_capacity`, `eggs_per_female`,
  `sex_ratio`, `sperm_displacement_rate`, `low_density_growth_rate`, ... -- the same
  5-parameter target table as `Op.set_param` plus sibling ecology scalars);
  out-of-bounds values raise `ValueError`, and the write is visible to the later
  stages of the same tick.
- Vector/tensor parameters use `pop.params.tensor_write(name, values)`.
- Writing draft arrays directly (bypassing `pop.params`) does **not** sync the
  equilibrium automatically -- Age-structured models need a manual
  `sync_equilibrium_metrics(config)`; `pop.params` writes handle it.
- Declarative `Op.set_param("carrying_capacity", "K * 0.95", every=10)` is equivalent
  to running the same write chain on a schedule (no Python code), see
  [Hook System](2_hooks.md).

---

## 5. Custom Fields: `config.custom`

A 0-d structured numpy array. Fields are registered at build time with `.custom()` and initial values, read/written with `[()]` inside hooks, and changed at runtime via `pop.update().custom()`:

```python
# build
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0)
    .build()
)

# inside a hook
@nt.hook(event="early")
def seasonal_hook(pop: TickContext) -> int:
    temp = pop.state.config.custom['temperature'][()]
    if int(pop.state.config.custom['season_idx'][()]) == 1:
        pop.state.config.custom['temperature'][()] = 35.0
    return 0

# runtime
pop.update().custom(temperature=35.0, season_idx=1)
```

Supports `bool`, `float`, and `int`.

> **Note**: custom fields are not in the parameter registry, so `set_param()`,
> `pop.params`, and `Op.set_param` cannot reach them. Use a direct array write
> inside a hook (`config.custom["temperature"][()] = value`) or
> `pop.update().custom(...)`.

---

## 6. Spatial Per-Deme Modification

**The `SpatialPopulation.update()` chain API has been removed.** Runtime writes for spatial populations go through two entries:

### 6.1 `pop.params` (bulk writes, recommended)

`pop.params` exposes write-protected `(n_demes, ...)` ecology column views for reads; `tensor_write` validates the shape and routes values per deme through the shared write channel (column + deme draft + Rust session column stay in lockstep):

```python
from natal.frontend.spatial import batch_setting

# same value for all demes: per-deme shape broadcasts
pop.params.tensor_write("survival_rates", np.ones((2, 2)))

# per-deme values: the full (n_demes, ...) column
pop.params.tensor_write("carrying_capacity", np.array([5000.0, 5000.0, 8000.0, 8000.0]))

# migration_rate keeps its build-time sugar: scalar / per-sex mapping / (n_ages,) / (S, A) / (n_demes, S, A)
pop.params.tensor_write("migration_rate", {"F": 0.2, "M": 0.05})

# numeric reads (write-protected view)
print(pop.params.migration_rate.shape)  # (n_demes, S, A)
```

`migration_rate` has shape `(n_demes, S, A)` in the runtime contract (A is the age axis; smaller shapes broadcast automatically; a scalar applies to the adult ages of both sexes, juveniles get 0).

### 6.2 `deme(i).write_ecology` / `write_genetics` (single-deme writes)

`pop.deme(i)` returns a `DemeSlice` view: reads of `config`/`state`/`registry`/`name`/... are delegated to the underlying deme object; writes go through two dedicated methods:

- `write_ecology(field, value)`: writes both the ecology column and that deme's draft (per-field clone-on-write), so every execution path -- Python dispatch and the Rust session columns -- sees the same value.
- `write_genetics(field, values)`: first forks the deme's genetics variant (Rust side) and detaches the draft tables, so demes that previously shared those tables stay bitwise unchanged.

```python
pop.deme(3).write_ecology("carrying_capacity", 8000.0)
pop.deme(3).write_genetics("viability", new_table)
```

### 6.3 `batch_setting`: The Single Entry Point

At build time, `batch_setting([...])` is the **only** declaration entry for per-deme heterogeneous parameters: the kind is inferred from the values (`"scalar"` / `"array"` / lambda `"spatial"`), and the homogeneous vs. heterogeneous path forks automatically in `build()`. fitness/presets do not support `batch_setting` (they modify config-internal ndarrays, which cannot be expressed as scalars); the `spatial` kind lambda requires the builder to have received a `topology`.

---

## 7. Underlying Mechanism

All modification paths converge on the same operation:

```
set_param / pop.update() / pop.params / hook-side params writes / Op.set_param
  -> config.carrying_capacity           # plain scalar
  -> carrying_capacity[()] = 5000.0     # in-place write (atomic)
  -> sync_equilibrium_metrics(config)   # triggered automatically for K/eggs/sr
```

Ecology scalars (K, eggs, sex_ratio, sperm_displacement_rate, low_density_growth_rate,
juvenile_growth_mode, generation_time, expected_competition_strength,
expected_survival_rate) are all plain scalars.

### `set_config()` -- Whole-Configuration Replacement

`pop.set_config(new_config)` replaces the population's entire config object at once. Suitable after rebuilding the config from scratch (e.g. after changing the custom-field structure). The new config must have the same type (`ModelDraft`, and discrete models must satisfy the discrete normalization invariants).

The Configurator's `custom()` method triggers this path when adding new fields: it rebuilds the custom structured array and calls `set_config()` to write the new config back into the population.

---

## 8. Parameter Reference

Parameters are grouped by domain, matching the Configurator chain-API methods.

| Domain | Parameter | Alias | Model | set_param |
|---|---|---|---|---|
| setup | `stochastic` | -- | both | no (build time) |
| setup | `continuous_sampling` | -- | both | no (build time) |
| setup | `fixed_egg_count` | -- | both | no (build time) |
| setup | `has_sex_chromosomes` | -- | both | no (build time) |
| age_structure | `n_ages` | -- | age-structured | no (build time) |
| age_structure | `new_adult_age` | -- | age-structured | no (build time) |
| age_structure | `generation_time` | -- | age-structured | no (build time) |
| survival | `female_age_based_survival` | -- | age-structured | yes |
| survival | `male_age_based_survival` | -- | age-structured | yes |
| reproduction | `eggs_per_female` | `expected_eggs_per_female` | both | yes |
| reproduction | `sex_ratio` | -- | both | yes |
| reproduction | `sperm_displacement_rate` | -- | both | yes |
| competition | `carrying_capacity` | -- | both | yes |
| competition | `low_density_growth_rate` | -- | both | yes |
| competition | `juvenile_growth_mode` | `growth_mode` | both | yes |
| fitness | `viability` | -- | both | no (tensor) |
| fitness | `fecundity` | -- | both | no (tensor) |
| fitness | `sexual_selection` | -- | both | no (tensor) |
| fitness | `zygote_viability` | -- | both | no (tensor) |
| migration | `migration_rate` | -- | spatial | spatial only |

## 9. Old vs. New

| | Old (Builder / njit era) | New (Configurator) |
|---|---|---|
| Post-build modification | unsupported | `pop.update()` |
| Hook-side modification | `(state, config, deme_id)` direct writes | `pop.params` (`TickContext`) or `Op.set_param` |
| Parameter audit trail | none | `pop.params_log` ((tick, name, old, new)) |
| Custom fields | ConfigMutator (removed) | `config.custom` |
| Spatial per-deme writes | `SpatialPopulation.update()` chain (removed) | `pop.params.tensor_write` + `deme(i).write_ecology` |
| Low-level API | none | `set_param(config, name, value)` |
