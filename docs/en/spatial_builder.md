# SpatialBuilder: Batch Construction of Spatial Populations

`SpatialBuilder` solves the redundant computation problem during multi-deme initialization using a "build template once, clone N-1 times" strategy. Construction time for 2601 homogeneous demes drops from ~2.6s to ~16ms.

## Quick Start

```python
from natal import Species, HexGrid, SpatialPopulation
from natal.spatial_builder import batch_setting

species = Species.from_dict(name="demo", structure={"chr1": {"loc": ["A", "B"]}})

# Homogeneous: all demes have identical parameters
pop = SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10)) \
    .setup(name="homo_demo", stochastic=False) \
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}}) \
    .reproduction(eggs_per_female=50) \
    .competition(carrying_capacity=10000) \
    .migration(migration_rate=0.1) \
    .build()

pop.run(10)
```

## Core Design

### Two-Layer Structure

```
SpatialPopulation.builder(...)
    │
    └─► SpatialBuilder         ← User-facing chained API
           │
           ├─ _template        ← AgeStructuredPopulationBuilder (or DiscreteGeneration...)
           │                     Always sees scalar parameters for a single deme
           ├─ _batch_settings  ← {param_name: BatchSetting}
           │                     Intercepted cross-deme varying parameters
           └─ _replay_log      ← [(method_name, kwargs), ...]
                                  Complete record of each chained call
```

`SpatialBuilder` does not modify existing builder classes; instead, it wraps them externally. During the chained call phase, it performs three tasks simultaneously:

1. **Delegates to `_template`** — the template builder always receives scalar values, maintaining correct internal state
2. **Detects `BatchSetting`** — intercepts and stores them in `_batch_settings`; template only sees `first_value()`
3. **Records in `_replay_log`** — preserves original arguments (including BatchSetting objects) for heterogeneous scenario replay

### Delegation Mechanism

Every chained method ultimately passes through `_detect_and_delegate`:

```python
# Example: .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000]))

def _detect_and_delegate(self, method_name, kwargs):
    concrete = {}
    for key, value in kwargs.items():
        if isinstance(value, BatchSetting):
            self._batch_settings[key] = value        # Store the original object
            first = value.first_value()               # Take the first scalar value
            if first is not None:
                concrete[key] = first                 # Template only sees scalar
        else:
            concrete[key] = value                     # Normal parameters pass through as-is

    self._replay_log.append((method_name, dict(kwargs)))  # Record original call

    method = getattr(self._template, method_name)
    method(**{k: v for k, v in concrete.items() if v is not None})
    return self
```

`presets()` and `hooks()` have positional arguments and use `_delegate_positional`, with the same logic.

### Parameter Aliasing

`competition()` is a unified entry point across `pop_type`, with internal parameter name normalization:

```
User passes carrying_capacity ─┐
                                ├─ age_structured → age_1_carrying_capacity (internal key)
User passes age_1_carrying_capacity ─┘
                                └─ discrete_generation → carrying_capacity (kept as-is)
```

Priority: `age_1_carrying_capacity` > `old_juvenile_carrying_capacity` > `carrying_capacity`.

This unifies key names in `_replay_log`, ensuring parameter names are consistent with the template builder signature during heterogeneous replay.

## Two Build Paths

`build()` automatically branches based on the presence of `_batch_settings`:

### Homogeneous Path (no batch_setting)

```
_build_homogeneous():
    1. template = self._template.build()     # Full pipeline once
    2. config = template.export_config()      # Export PopulationConfig
    3. demes = [template]
    4. for i in 1..n_demes:
           demes.append(_clone_deme(template, config))
    5. return SpatialPopulation(demes, ...)
```

### Heterogeneous Path (with batch_setting)

```
_build_heterogeneous():
    1. expanded = {name: batch.expand(n_demes, topology) for ...}
       # Expand all BatchSettings into per-deme value lists

    2. Compute config signature for each deme by (param_name, param_value) tuples
       # e.g., deme 0: (("age_1_carrying_capacity", 10000.0),)

    3. Group by signature → {sig: [deme_index, ...]}

    4. For each group:
       a. _build_template_for_group(sig_map)
          # Create new builder, replay _replay_log, replace batch params with group values
       b. Remaining demes in group = _clone_deme(group_template)

    5. Assemble all demes by index, construct SpatialPopulation
```

`_build_template_for_group` is the core of replay:

```python
def _build_template_for_group(self, sig_map):
    builder = AgeStructuredPopulationBuilder(self._species)  # Fresh builder

    for method_name, kwargs in self._replay_log:
        resolved = {}
        for key, value in kwargs.items():
            if key in sig_map:
                resolved[key] = sig_map[key]   # Replace with this group's scalar value
            elif isinstance(value, BatchSetting):
                resolved[key] = value.first_value()  # Uncovered batch takes first value
            else:
                resolved[key] = value           # Non-batch parameters pass through as-is

        getattr(builder, method_name)(**resolved)

    return builder.build()
```

## `_clone_deme`: Zero-Compilation-Overhead Cloning

Cloning creates instances via `__new__`, completely bypassing `__init__` to avoid repeated hook compilation and config construction.

```python
def _clone_deme(template, config, name):
    clone = AgeStructuredPopulation.__new__(AgeStructuredPopulation)

    # === Shared references (read-only during simulation) ===
    clone._species           = template._species
    clone._compiled_hooks    = template._compiled_hooks     # Already compiled hook functions
    clone._hook_executor     = template._hook_executor       # Hook execution engine
    clone._config            = config                        # PopulationConfig (shared)
    clone._index_registry    = template._index_registry      # Genotype lookup table
    clone._registry          = template._registry
    clone._gamete_modifiers  = template._gamete_modifiers    # Gamete modifiers
    clone._zygote_modifiers  = template._zygote_modifiers
    clone._genotypes_list    = template._genotypes_list
    clone._haploid_genotypes_list = template._haploid_genotypes_list

    # === Independent copies ===
    clone._name    = name
    clone._history = []
    clone._state   = State.create(...)                        # New state array
    clone._state_nn.individual_count[:] = template._state_nn.individual_count
    clone._state_nn.sperm_storage[:]    = template._state_nn.sperm_storage
    clone._initial_population_snapshot  = (copy of template's snapshot)

    return clone
```

| Attribute | Shared/Independent | Reason |
|-----------|-------------------|--------|
| `_compiled_hooks`, `_hook_executor` | Shared reference | Stateless, read-only registry |
| `_config` | Shared reference | Homogeneous demes have identical configs |
| `_index_registry`, `_registry` | Shared reference | Same species means same genotype indices |
| `_gamete_modifiers`, `_zygote_modifiers` | Shallow-copied list | Presets may operate on the same object |
| `_state` | Independently created | Each deme has its own individual counts |
| `_history`, `snapshots` | Independent empty list/dict | Each deme records its own simulation history |

## `BatchSetting`: Cross-Deme Varying Parameters

```python
from natal.spatial_builder import batch_setting

# List: index-to-index correspondence
batch_setting([10000, 5000, 5000, 8000])        # kind="scalar"

# NumPy array
batch_setting(np.array([10000, 5000, ...]))      # kind="array"

# Spatial function: (topology, deme_idx) -> float
batch_setting(lambda topo, i: 10000 if i < 50 else 5000)  # kind="spatial"
```

All three kinds are uniformly expanded into Python lists via `expand(n_demes, topology)` at `build()` time.

Parameters that accept `BatchSetting`: `carrying_capacity`, `age_1_carrying_capacity`, `eggs_per_female`, `sex_ratio`, `low_density_growth_rate`, `juvenile_growth_mode`, `expected_num_adult_females`.

## Performance Data

Test conditions: 2 alleles (4 genotypes), 200 individuals per deme initially.

| Scenario | Time | Notes |
|----------|------|-------|
| Homogeneous 100 demes | ~400ms | Includes first Numba compilation ~350ms |
| Homogeneous 2601 demes | ~16ms | Clone phase only, excluding first template build |
| 2-group heterogeneous 4 demes | ~6ms | One template per group + 1 clone each |
| First template build | ~2-3ms | Numba compilation leverages file cache, faster after first hit |

First template build time depends on hook count and Numba cache status; subsequent calls are typically < 5ms.

## Relationship with Existing API

`SpatialBuilder` does not modify any existing classes:

- `AgeStructuredPopulationBuilder` / `DiscreteGenerationPopulationBuilder` — unchanged, `SpatialBuilder` wraps them via composition
- `SpatialPopulation.__init__` — unchanged, `build()` ultimately calls it with the pre-built deme list
- The old per-deme construction approach still works

## Limitations

1. **`batch_setting` does not support fitness / presets** — fitness and presets modify NumPy arrays inside config (in-place), which are not well-suited for scalar value expression. For heterogeneous fitness, manually modify the corresponding deme's config arrays after build
2. **spatial kind requires topology** — `batch_setting(lambda topo, i: ...)` requires the topology parameter to have been passed to the builder, otherwise `expand()` will raise an error
3. **Homogeneous demes share the same `_config` reference** — if subsequent code directly modifies array fields of `pop.demes[0]._config` (not via `_replace`), it will affect all demes sharing that config. The correct approach is to create an independent copy via `config._replace(...)` first
