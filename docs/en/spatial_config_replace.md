# SpatialBuilder Heterogeneous Config Sharing Mechanism

## Problem

`SpatialBuilder._build_heterogeneous()` calls `_build_template_for_group()` for each config-equivalent group. This function fully replays the builder pipeline (`setup → … → build()`), calling `build_population_config()` each time to create a brand new `PopulationConfig`.

If only a few parameters differ between groups, all large arrays (`genotype_to_gametes_map`, `gametes_to_zygote_map`, `viability_fitness`, `fecundity_fitness`, etc.) are still duplicated, causing memory waste.

```
2601 demes, each with a unique carrying_capacity
→ 2601 full PopulationConfig instances
→ Large arrays copied 2601 times
```

## Solution: `_replace` Fast Path

`PopulationConfig` is a `NamedTuple`, and its `_replace()` method creates a new instance while **sharing references to all fields that are not replaced**. Leveraging this property, the first group is built in full, and subsequent groups only replace the differing fields:

```
Group 0: Full builder pipeline → base_config (all arrays)
Group 1: base_config._replace(carrying_capacity=2000)       → shares all large arrays
Group 2: base_config._replace(carrying_capacity=3000)       → shares all large arrays
...
Group N: base_config._replace(initial_individual_count=arr) → rebuilds only initial_individual_count
```

## Parameter Discovery Mechanism

Instead of maintaining a hardcoded allowlist, parameters eligible for `_replace` are automatically discovered through a layered strategy:

### 1. Array Fields (Explicit)

Builder parameters that require dict → numpy array conversion, defined in `_ARRAY_KWARGS`:

| Builder kwarg | Config Field | Conversion Method |
|---|---|---|
| `individual_count` | `initial_individual_count` | `PopulationConfigBuilder.resolve_*_initial_individual_count()` |
| `sperm_storage` | `initial_sperm_storage` | `PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage()` |

### 2. Multi-Field Mapping (Explicit)

One builder kwarg mapping to multiple config fields (requiring special scaling), defined in `_KWARG_MULTI_FIELD`:

| Builder kwarg | Config Fields |
|---|---|
| `carrying_capacity` | `base_carrying_capacity` (raw) + `carrying_capacity` (× population_scale) |
| `age_1_carrying_capacity` | Same as above |
| `old_juvenile_carrying_capacity` | Same as above |

### 3. Renames (Explicit)

Builder kwarg names that differ from config field names, defined in `_KWARG_RENAMES`:

| Builder kwarg | Config Field |
|---|---|
| `eggs_per_female` | `expected_eggs_per_female` |
| `expected_num_adult_females` | `base_expected_num_adult_females` |

### 4. Dynamic Discovery (Implicit)

For kwargs not in the three categories above, `hasattr(base_config, kwarg_name)` is used to check if it is a valid config field. For example, `low_density_growth_rate`, `juvenile_growth_mode`, `sex_ratio`, `sperm_displacement_rate`, etc., since the builder kwarg name matches the config field name, **no mapping configuration is needed** for automatic support.

Adding new batch-able scalar parameters typically does not require modifying the mapping tables — as long as the builder kwarg name matches the config field name.

Parameters not in any of the above categories (`presets`, `fitness`, survival rate arrays, etc.) fall back to full builder replay.

### Deliberately Unsupportable Heterogeneous Parameters

`stochastic` and `use_continuous_sampling` are simulation-mode-level parameters that should not vary between demes. The `setup()` method does not pass through `_detect_and_delegate`, so these parameters **cannot** be provided via `batch_setting`.

## Equilibrium Recalculation

Changes to `carrying_capacity`, `eggs_per_female`, and `sex_ratio` affect `expected_competition_strength` and `expected_survival_rate`. These parameters are flagged in `_EQUILIBRIUM_SENSITIVE_KWARGS`, and after `_replace` completes, `compute_equilibrium_metrics()` is automatically called to recalculate.

## Array Field Conversion

The values for `individual_count` and `sperm_storage` are user-provided dicts (e.g., `{"female": {"WT|WT": 100}}`), which must be converted to numpy arrays before `_replace`. Conversion is done via static methods on `PopulationConfigBuilder`:

- Age-structured: `resolve_age_structured_initial_individual_count(species, distribution, n_ages, new_adult_age)`
- Discrete generation: `resolve_discrete_initial_individual_count(species, distribution)`

The result is multiplied by `population_scale` to match builder behavior.

## State Overwrite After Cloning

`_clone_deme()` copies state data from the template. When `individual_count` or `sperm_storage` differs between groups, the `_replace` path additionally overwrites the corresponding state arrays after cloning:

```python
group_template = _clone_deme(base_template, config=variant_config, name=...)
# _clone copies state from base_template; overwrite with config values
state = group_template._require_state()
if "individual_count" in sig_map:
    state.individual_count[:] = variant_config.initial_individual_count
if "sperm_storage" in sig_map:
    ss = getattr(state, 'sperm_storage', None)
    if ss is not None:
        ss[:] = variant_config.initial_sperm_storage
```

## `_build_heterogeneous` Flow

```
_build_heterogeneous()
  │
  ├─ 1. Expand all batch_settings into per-deme value lists
  ├─ 2. Compute config signature (hash) for each deme
  ├─ 3. Group by signature
  │
  └─ 4. For each group:
       │
       ├─ First group → _build_template_for_group()
       │                 Full builder pipeline, produces base_config + base_template
       │
       ├─ Subsequent groups + _can_use_replace(sig_map, base_config)
       │   │  All kwargs pass through explicit mapping or dynamic hasattr discovery
       │   │
       │   ├─ _build_variant_config(sig_map, base_config)
       │   │   │
       │   │   ├─ Array fields → PopulationConfigBuilder.resolve_* → _replace
       │   │   ├─ Multi-field → _replace(base=raw, scaled=raw*pop_scale)
       │   │   ├─ Renames → _replace(renamed_field=val)
       │   │   ├─ Dynamic discovery → hasattr → _replace
       │   │   └─ Equilibrium-sensitive → compute_equilibrium_metrics()
       │   │
       │   └─ _clone_deme(base_template, variant_config)
       │        └─ Overwrite state corresponding to array-valued batch settings
       │
       └─ Subsequent groups + non-replaceable
           └─ _build_template_for_group()  (full replay, behavior unchanged)
```

## Memory Impact

With 2601 demes and only `carrying_capacity` differing:

| Item | Before Optimization | After Optimization |
|---|---|---|
| `genotype_to_gametes_map` | 2601 copies | 1 copy (shared) |
| `gametes_to_zygote_map` | 2601 copies | 1 copy (shared) |
| `viability_fitness` | 2601 copies | 1 copy (shared) |
| `fecundity_fitness` | 2601 copies | 1 copy (shared) |
| `carrying_capacity` (scalar) | 2601 copies | 2601 copies (~60KB) |
| `initial_individual_count` | 2601 copies | 1 copy (all demes homogeneous) |

With 2601 demes and only `initial_individual_count` differing:

| Item | Before Optimization | After Optimization |
|---|---|---|
| `genotype_to_gametes_map` | 2601 copies | 1 copy (shared) |
| `gametes_to_zygote_map` | 2601 copies | 1 copy (shared) |
| All fitness arrays | 2601 copies | 1 copy (shared) |
| `initial_individual_count` | 2601 copies | 2601 copies (must differ) |

## File Location

All changes are concentrated in `src/natal/spatial_builder.py`:

| Symbol | Role |
|---|---|
| `_ARRAY_KWARGS` | Set of parameters requiring dict→array conversion |
| `_KWARG_MULTI_FIELD` | Multi-field mapping (carrying_capacity variants) |
| `_KWARG_RENAMES` | Builder kwarg → config field renames |
| `_EQUILIBRIUM_SENSITIVE_KWARGS` | Set of parameters requiring equilibrium recalculation |
| `SpatialBuilder._build_heterogeneous()` | Main build logic |
| `SpatialBuilder._can_use_replace(sig_map, base_config)` | Determines whether `_replace` can be used |
| `SpatialBuilder._build_variant_config()` | Creates variant config |
