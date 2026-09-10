# Runtime Parameter Modification

All parameters can be changed during simulation without rebuilding the population.
This chapter covers three scenarios:

- **Between-tick**: Python-side via `pop.update()` or `pop.params.<name> = v`
- **Inside hooks**: via the callback hook's `pop.params` (`TickContext`), or declaratively via `Op.set_param`
- **Spatial**: per-deme writes (`pop.params.tensor_write` / `deme(i).write_ecology`)

---

## 1. Between-Tick Modification: `pop.update()`

`pop.update()` returns a `RuntimeUpdater` **bound to the running population**. Domain methods have the same syntax as the build chain; every call is validated against the parameter route table and committed to the running population (draft and Rust session stay in sync), so later stages of the same tick see the new value:

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

# custom fields (write in callbacks; read pop.config.custom outside)
pop.update().custom(temperature=35.0)
```

Each call commits to the running population and appends a parameter-log row when the value actually changes; it operates on the running population, not on a query snapshot.

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

## 3. `set_param()`: A Draft-Level Low-Level API

`set_param()` writes only the **draft** you pass in; it never writes to a running population. `pop.config` is a query snapshot, so calling `set_param()` on it does not change the population — use `pop.update()` or `pop.params` to modify a running population. This function suits draft-only work (scripts, notebooks, offline configuration):

```python
from natal.frontend.builder import set_param

# Ecology scalars are NamedTuple slots: bind the returned draft
draft = pop.config
draft = set_param(draft, "competition.carrying_capacity", 5000.0)

# full name, short name, and aliases all work
draft = set_param(draft, "carrying_capacity", 5000.0)
draft = set_param(draft, "reproduction.eggs_per_female", 100.0)
draft = set_param(draft, "eggs_per_female", 100.0)  # alias
```

Key points:

1. Names resolve through the `parameters.jsonc` registry: full name -> short name -> alias.
2. Ecology scalars go through `NamedTuple._replace`, so the **return value must be rebound**; array-backed fields (custom slots, vector/tensor contents) are mutated in place and return the same draft.
3. A draft obtained from `pop.config` is an isolated snapshot: the write lands only on that copy, and even a rebound result never reaches the running population.
4. Equilibrium metrics (expected_competition_strength / expected_survival_rate) are derived on read — no stored copies to sync.

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
- Equilibrium metrics are not stored in the configuration: every read
  (`pop.params.expected_competition_strength`, ...) is derived from the current
  ecology, so it follows ecology writes immediately with no manual sync.
- Declarative `Op.set_param("carrying_capacity", "K * 0.95", every=10)` is equivalent
  to running the same write chain on a schedule (no Python code), see
  [Hook System](2_hooks.md).

---

## 5. Custom Fields: `config.custom`

`custom` is a 0-d structured numpy array. Fields are registered at build time with `.custom()` together with initial values; read them through `pop.config.custom["name"]` (a query snapshot) and write them through `pop.update().custom()`:

```python
# build
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0)
    .build()
)

# inside a hook: the write joins the event transaction and commits only on success
@nt.hook(event="early")
def seasonal_hook(ctx: TickContext) -> int:
    if ctx.tick == 0:
        ctx.update().custom(temperature=35.0)
    return 0

# outside a callback
pop.update().custom(temperature=35.0, season_idx=1)
print(pop.config.custom["temperature"])  # 35.0
```

`bool`, `float`, and `int` are supported, as are arrays of any rank; types and shapes are preserved.

> **Note**: custom fields are not in the parameter route table, and callbacks have
> **no public read path** for them: `TickContext` exposes no `config`, `ctx.state`
> has no `config` attribute, and `ctx.params` / `Op.set_param` accept registered
> parameters only (`Op.set_param` rejects a custom field at compile time with
> `ValueError`). Read custom values outside callbacks through
> `pop.config.custom["name"]`, and use callbacks for writes
> (`ctx.update().custom(...)`). Draft-level `set_param(draft, "temperature", v)`
> can write a custom slot, but like every `set_param` call it does not commit to a
> running population.

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
pop.deme(3).write_genetics("viability_fitness", new_table)
```

### 6.3 `batch_setting`: The Single Entry Point

At build time, `batch_setting([...])` is the **only** declaration entry for per-deme heterogeneous parameters: the kind is inferred from the values (`"scalar"` / `"array"` / lambda `"spatial"`), and the homogeneous vs. heterogeneous path forks automatically in `build()`. fitness/presets do not support `batch_setting` (they modify config-internal ndarrays, which cannot be expressed as scalars); the `spatial` kind lambda requires the builder to have received a `topology`.

---

## 7. Underlying Mechanism

Runtime write entries (`pop.update()`, `pop.params`, in-hook `ctx.params` /
`ctx.update()`, declarative `Op.set_param`) all converge on the same chain:

```
runtime write entry
  -> route-table name resolution + jsonc bounds validation
  -> draft field update (ecology scalars go through NamedTuple._replace)
  -> Rust session parameter sync + parameter snapshot log (tick, name, old, new)
  -> equilibrium metrics derived on read (no stored copies)
```

Draft-level `set_param(draft, name, value)` performs only the draft update: it does
not sync the session and never writes to a running population.

Ecology scalars (K, eggs, sex_ratio, sperm_displacement_rate, low_density_growth_rate,
juvenile_growth_mode, generation_time) are plain scalars in the runtime contract;
equilibrium metrics (expected_competition_strength, expected_survival_rate) are
read-only derived values read through `pop.params.<name>`; assigning to them raises
`AttributeError`.

### `set_config()` -- Whole-Configuration Replacement

`pop.set_config(new_config)` replaces the population's entire config object at once. Suitable after rebuilding the config from scratch (e.g. after changing the custom-field structure). The new config must have the same type (`ModelDraft`, and discrete models must satisfy the discrete normalization invariants).

The PopulationBuilder's `custom()` method triggers this path when adding new fields: it rebuilds the custom structured array and calls `set_config()` to write the new config back into the population.

---

## 8. Parameter Reference

Parameters are grouped by domain, matching the PopulationBuilder chain-API methods.

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

| | Old (Builder / njit era) | New (PopulationBuilder) |
|---|---|---|
| Post-build modification | unsupported | `pop.update()` |
| Hook-side modification | `(state, config, deme_id)` direct writes | `pop.params` (`TickContext`) or `Op.set_param` |
| Parameter audit trail | none | `pop.params_log` ((tick, name, old, new)) |
| Custom fields | ConfigMutator (removed) | `config.custom` |
| Spatial per-deme writes | `SpatialPopulation.update()` chain (removed) | `pop.params.tensor_write` + `deme(i).write_ecology` |
| Low-level API | none | `set_param(config, name, value)` |
