# SpatialPopulationBuilder Heterogeneous Config Sharing Mechanism

> **Implementation note**: this page describes the internal `ModelDraft._replace` sharing of large arrays during a heterogeneous build. The mechanism is still in use, but it is **not** the whole heterogeneous build — declaration freezing, signature grouping, and template cloning live in [SpatialPopulationBuilder: Batch Construction of Spatial Populations](spatial_population_builder.md).

## Problem

`SpatialPopulationBuilder._build_heterogeneous_demes()` compiles the builder pipeline once per **genetics** signature group (`_genetics_batch_names()` decides which batch kwargs belong to genetics; ecology-only differences do not split groups). Every additional variant inside a group still needs its own `ModelDraft`: either it is derived from the group's base config, or `_builder_for_group()` fully replays the builder pipeline (`setup → … → build()`), compiling a brand-new `ModelDraft` each time.

If the `_replace` fast path did not exist, every ecology-only variant would need a full replay, so all large arrays (`zygotes_to_gametes_map`, `gametes_to_zygotes_map`, `viability_fitness`, `fecundity_fitness`, etc.) would be duplicated, causing memory waste.

```
2601 demes, each with a unique carrying_capacity
→ 2601 full ModelDraft instances (one full replay per deme)
→ Large arrays copied 2601 times
```

## Solution: `_replace` Fast Path

`ModelDraft` is a `NamedTuple`, and its `_replace()` method creates a new instance while **sharing references to all fields that are not replaced**. Leveraging this property, the first variant of a group is compiled in full, and subsequent variants only replace the differing fields:

```
Variant 0: Full compile pipeline → base_config (all arrays)
Variant 1: base_config._replace(carrying_capacity=2000)       → shares all large arrays
Variant 2: base_config._replace(carrying_capacity=3000)       → shares all large arrays
...
Variant N: base_config._replace(initial_individual_count=arr) → rebuilds only initial_individual_count
```

## Parameter Discovery Mechanism

Instead of maintaining a hardcoded allowlist, parameters eligible for `_replace` are automatically discovered through a layered strategy:

### 1. Array Fields (Explicit)

Builder parameters that require dict → numpy array conversion, defined in `_ARRAY_KWARGS`:

| Builder kwarg | Config Field | Conversion Method |
|---|---|---|
| `individual_count` | `initial_individual_count` | `initial_state.resolve_*_initial_individual_count()` |
| `sperm_storage` | `initial_sperm_storage` | `initial_state.resolve_age_structured_initial_sperm_storage()` |

### 2. Multi-Field Mapping (Explicit)

Discrete-generation scalar builder kwargs that each write **one cell** of a unified `(2, n_ages)` vector config field, defined in `_DISCRETE_VECTOR_CELLS`. The target vector is copied before the write, so variants never alias the base:

| Builder kwarg | Config Field | Cell |
|---|---|---|
| `female_age0_survival` | `age_based_survival_rates` | `(0, 0)` |
| `male_age0_survival` | `age_based_survival_rates` | `(1, 0)` |
| `female_adult_mating_rate` | `age_based_mating_rates` | `(0, 1)` |
| `male_adult_mating_rate` | `age_based_mating_rates` | `(1, 1)` |

### 3. Renames (Explicit)

Builder kwarg names that differ from config field names, defined in `_KWARG_RENAMES`:

| Builder kwarg | Config Field |
|---|---|
| `eggs_per_female` | `eggs_per_female` |

### 4. Dynamic Discovery (Implicit)

For kwargs not in the three categories above, `hasattr(base_config, kwarg_name)` is used to check if it is a valid config field. For example, `low_density_growth_rate`, `juvenile_growth_mode`, `sex_ratio`, `sperm_displacement_rate`, etc., since the builder kwarg name matches the config field name, **no mapping configuration is needed** for automatic support.

Adding new batch-able scalar parameters typically does not require modifying the mapping tables — as long as the builder kwarg name matches the config field name.

Genetics-affecting kwargs (`presets`, `fitness` rows such as `viability` / `fecundity`, custom modifiers, etc.) never go through `_replace`: they split demes into separate genetics groups, each compiled by a full builder replay. Within a group, a non-genetics kwarg that fails the `hasattr` check also falls back to full replay; the replayed variant then shares the group's genetic product fields with the base config.

### Deliberately Unsupportable Heterogeneous Parameters

`stochastic` and `continuous_sampling` are simulation-mode-level parameters that should not vary between demes. `setup()` takes them directly and does not route them through the batch machinery, so these parameters **cannot** be provided via `batch_setting`.

## Equilibrium Metrics Are Derived on Read

Changes to `carrying_capacity`, `eggs_per_female`, and `sex_ratio` affect `expected_competition_strength` and `expected_survival_rate`, but the draft stores neither value: `ModelDraft` has no equilibrium fields, and nothing is recomputed after `_replace`. Both metrics are derived fresh on read — `pop.params.expected_competition_strength` / `pop.params.expected_survival_rate` call `derive_equilibrium_metrics_from_draft()` (`src/natal/frontend/model/ecology.py`), so a `_replace` variant is automatically consistent without any sync step.

## Array Field Conversion

The values for `individual_count` and `sperm_storage` are user-provided dicts (e.g., `{"female": {"WT|WT": 100}}`), which must be converted to numpy arrays before `_replace`. Conversion is done by the plain resolver functions in `natal.frontend.model.initial_state`:

- Age-structured: `resolve_age_structured_initial_individual_count(species, distribution, n_ages, new_adult_age)`
- Age-structured sperm storage: `resolve_age_structured_initial_sperm_storage(species, sperm_storage, n_ages, new_adult_age)`
- Discrete generation: `resolve_discrete_initial_individual_count(species, distribution)`

The result matches builder behavior.

## Cloning and Initial State

`_clone_deme()` delegates to `PopulationInstance._clone()` (`src/natal/frontend/population/base.py`), which copies the state arrays (`individual_count`, plus `sperm_storage` on age-structured states) from the template deme. A clone always shares the template's config object, so the copied state already matches the config — no post-clone overwrite is needed.

Demes whose initial state differs are never produced by cloning: their variant config (fresh `initial_individual_count` / `initial_sperm_storage` arrays computed by the resolvers) goes through the normal publish path (`_publish_and_build()`), which initializes the population state from the config.

## `_build_heterogeneous_demes` Flow

```
_build_heterogeneous_demes()
  │
  ├─ 1. Expand all batch_settings into per-deme value lists
  ├─ 2. _genetics_batch_names() → batch kwargs that affect the genetics section
  ├─ 3. Group demes by genetics-only signature
  │     (ecology differences do not split groups)
  │
  └─ 4. Per genetics group — compile candidates:
       │
       ├─ First deme → copy of the template builder (deme 0)
       │              or _builder_for_group() full replay
       │              → _compile_products() → base_config
       │
       └─ Other demes in the group:
            ├─ Same full signature as an earlier deme → reuse its candidate
            ├─ _can_use_replace(ecology kwargs, base_config)
            │   ├─ yes → _build_variant_config(sig_map, base_config)
            │   │        │
            │   │        ├─ Array fields → initial_state resolve_* → _replace
            │   │        ├─ Discrete scalars → copy vector, write one cell → _replace
            │   │        ├─ Renames → _replace(renamed_field=val)
            │   │        └─ Dynamic discovery → hasattr → _replace
            │   └─ no  → _builder_for_group() full replay, then
            │            _replace the genetic product fields from base_config
  │
  ├─ 5. _spatial_projection() → shared registry projection
  │     (union of every group's route tables)
  │
  └─ 6. Per genetics group — publish:
       ├─ Each unique config → builder._publish_and_build()
       │    (the first published deme becomes the genetics template)
       └─ Demes sharing a signature → _clone_deme() → _clone()
            (state arrays copied from the template; heavy arrays shared)
```

## Memory Impact

With 2601 demes and only `carrying_capacity` differing:

| Item | Before Optimization | After Optimization |
|---|---|---|
| `zygotes_to_gametes_map` | 2601 copies | 1 copy (shared) |
| `gametes_to_zygotes_map` | 2601 copies | 1 copy (shared) |
| `viability_fitness` | 2601 copies | 1 copy (shared) |
| `fecundity_fitness` | 2601 copies | 1 copy (shared) |
| `carrying_capacity` (scalar) | 2601 copies | 2601 copies (~60KB) |
| `initial_individual_count` | 2601 copies | 1 copy (all demes homogeneous) |

With 2601 demes and only `initial_individual_count` differing:

| Item | Before Optimization | After Optimization |
|---|---|---|
| `zygotes_to_gametes_map` | 2601 copies | 1 copy (shared) |
| `gametes_to_zygotes_map` | 2601 copies | 1 copy (shared) |
| All fitness arrays | 2601 copies | 1 copy (shared) |
| `initial_individual_count` | 2601 copies | 2601 copies (must differ) |

## File Location

The relevant implementation lives in `src/natal/frontend/spatial/builder.py`:

| Symbol | Role |
|---|---|
| `_ARRAY_KWARGS` | Set of parameters requiring dict→array conversion |
| `_DISCRETE_VECTOR_CELLS` | Discrete scalar kwargs → one cell of a unified vector field |
| `_KWARG_RENAMES` | Builder kwarg → config field renames |
| `_genetics_batch_names()` | Selects the batch kwargs that split genetics groups |
| `SpatialPopulationBuilder._build_heterogeneous_demes()` | Main heterogeneous build flow |
| `SpatialPopulationBuilder._builder_for_group()` | Replays one group into a complete-axis unpublished builder |
| `SpatialPopulationBuilder._can_use_replace(sig_map, base_config)` | Determines whether `_replace` can be used |
| `SpatialPopulationBuilder._build_variant_config()` | Creates variant config |
| `_clone_deme()` → `PopulationInstance._clone()` | Clones a published deme, sharing compiled state and config |
