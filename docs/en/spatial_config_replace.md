# SpatialPopulationBuilder Heterogeneous Config Sharing Mechanism

> **Implementation note**: this page describes the internal `ModelDraft._replace` sharing of large arrays during a heterogeneous build. The mechanism is still in use, but it is **not** the whole heterogeneous build — declaration freezing, signature grouping, and template cloning live in [SpatialPopulationBuilder: Batch Construction of Spatial Populations](spatial_population_builder.md).

## Problem

`SpatialPopulationBuilder._build_heterogeneous_demes()` compiles once per **genetics** signature group (`_genetics_batch_names()` decides which batch kwargs belong to genetics; ecology-only differences do not split groups). Every deme still needs its own `ModelDraft`; if every one were compiled from scratch, all large arrays (`zygotes_to_gametes_map`, `gametes_to_zygotes_map`, `viability_fitness`, `fecundity_fitness`, etc.) would be duplicated, causing memory waste.

## Solution: declaration projection plus shared genetic products

Each deme's concrete declarations (the journal with its per-deme batch values resolved) are projected through the single declaration interpreter (`builder/_declarations.py`) onto a fresh baseline — no builder method is re-executed. The genetics group compiles once via `compile_definition`; every deme in the group attaches the group's compiled genetic product fields unchanged, so those heavy arrays are shared:

```
Group compile: project group-0 declarations → compile_definition → products
Deme i:       project deme i's *differing* declarations onto products.config
              → shares the group's genetic product arrays
```

## Which parameters can differ per deme

There is no mapping table any more.  The delta projection re-uses the
chain methods' own write path (the route-table writer behind
`competition()` / `reproduction()` / `survival()` / `initial_state()`),
so every parameter those methods accept is projectable per deme:
scalars (`carrying_capacity`, `eggs_per_female`, `sex_ratio`, ...),
per-age vectors (`female_age_based_survival`, ...), and the initial
distribution (`individual_count` / `sperm_storage` dicts are resolved to
arrays by the plain resolvers in `natal.frontend.model.initial_state`).

Genetics-affecting kwargs (`presets`, `fitness` rows, custom modifiers)
do not project as deltas: they split demes into separate genetics
groups, each compiled once through `compile_definition`.  A group whose
genetics match the template inherits its compile cache, so recipes never
re-run for identical content.  Derived scalars (for example the Champer
egg override from `expected_num_new_adult_females`) freeze at the
group's computation unless the user re-declares them for the deme.

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
       ├─ _projected_group(values[first])
       │     resolve the deme's journal → project onto a fresh baseline
       │     (single interpreter; zero builder-method execution)
       ├─ _carrier_from_projection(...)
       │     group-0 with template-matching genetics inherits the
       │     template's compile cache (recipes never re-run)
       ├─ carrier._compile_products() → compile_definition
       │
       └─ Other demes in the group:
            ├─ Same full signature as an earlier deme → reuse its candidate
            └─ _projected_variant_config(group_config, values_i, values_first)
                  project only the *differing* declarations;
                  derived scalars stay frozen at the group's computation
  │
  ├─ 5. _spatial_projection() → shared registry projection
  │     (union of every group's route tables)
  │
  └─ 6. Per genetics group — publish:
       ├─ Each unique config → builder._publish_and_build()
## Cloning and Initial State

`_clone_deme()` delegates to `PopulationInstance._clone()` (`src/natal/frontend/population/base.py`), which copies the state arrays (`individual_count`, plus `sperm_storage` on age-structured states) from the template deme. A clone always shares the template's config object, so the copied state already matches the config — no post-clone overwrite is needed.

Demes whose initial state differs are never produced by cloning: their variant config (fresh `initial_individual_count` / `initial_sperm_storage` arrays computed by the resolvers) goes through the normal publish path (`_publish_and_build()`), which initializes the population state from the config.

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
| `_genetics_batch_names()` | Selects the batch kwargs that split genetics groups |
| `SpatialPopulationBuilder._build_heterogeneous_demes()` | Main heterogeneous build flow |
| `SpatialPopulationBuilder._projected_group()` | Resolves a deme's journal and projects it onto a fresh baseline |
| `SpatialPopulationBuilder._projected_variant_config()` | Projects one deme's differing declarations onto the group config |
| `builder/_declarations.py` | The single declaration interpreter both paths share |
| `_clone_deme()` → `PopulationInstance._clone()` | Clones a published deme, sharing compiled state and config |
