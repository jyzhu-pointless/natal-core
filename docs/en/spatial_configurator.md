# SpatialConfigurator: Batch Construction of Spatial Populations

`SpatialConfigurator` solves the redundant computation problem during multi-deme initialization using a "build template once, clone N-1 times" strategy.

## Quick Start

```python
import numpy as np
from natal import Species, HexGrid, SpatialPopulation

species = Species.from_dict(name="spatial_configurator_demo", structure={"chr1": {"loc": ["A", "B"]}})

# All demes use the same discrete-generation model.
pop = (
    SpatialPopulation.builder(
        species, n_demes=100, topology=HexGrid(10, 10),
        pop_type="discrete_generation",
    )
    .setup(name="homo_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}})
    .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    .reproduction(eggs_per_female=2, sex_ratio=0.5)
    .competition(carrying_capacity=10000)
    .migration(kernel=np.ones((3, 3)), migration_rate=0.1)
    .build()
)

pop.run(10)
assert pop.tick == 10
```

## Core Design

### Two-Layer Structure

```
SpatialPopulation.builder(...)
    │
    └─► SpatialConfigurator         ← User-facing chained API
           │
           ├─ _template        ← single-deme template Configurator (not a removed Builder class)
           │                     Always sees scalar parameters for a single deme
           ├─ _batch_settings  ← {param_name: BatchSetting}
           │                     Intercepted cross-deme varying parameters
           └─ _declaration_log      ← [(method_name, kwargs), ...]
                                  Complete record of each chained call
```

`SpatialConfigurator` adds no population class; it wraps a single-deme template `Configurator` externally. During the chained call phase it performs three tasks simultaneously:

1. **Delegates to `_template`** — the template `Configurator` always receives scalar values, maintaining correct internal state
2. **Detects `BatchSetting`** — intercepts and stores them in `_batch_settings`; template only sees `first_value()`
3. **Records in `_declaration_log`** — preserves original arguments (including BatchSetting objects) for heterogeneous scenario replay

### Frozen Declaration

`build()` first freezes declarations into a single `ModelDefinition`: the template inputs (single-deme settings, genetic rules, and hooks) are declaration fields, and `.spatial` stores the deme count, topology, expanded per-deme batch values, migration, spatial observation, history capacity, and compression declarations. The actual build creates an isolated compiler from this declaration and groups demes by genetic differences.

Batch functions expand once when inputs are frozen, and finalization reuses cached template genetics. Cold compilation consumes frozen concrete values without reevaluating batch functions. Definition queries copy NATAL arrays and containers, so modifying a query result cannot affect a later build. User presets, hooks, and their external resources preserve their identity and do not need to support deep copying.

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

    self._declaration_log.append((method_name, dict(kwargs)))  # Record original call

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

This unifies key names in `_declaration_log`, ensuring parameter names match the template `Configurator` method signatures during heterogeneous replay.

## Two Build Paths

`build()` automatically branches based on the presence of `_batch_settings`:

### Homogeneous Path (no batch_setting)

```
_build_homogeneous():
    1. template = self._template.build()     # Full pipeline once
    2. config = template.export_config()      # Export ModelDraft
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
          # Create a new template Configurator, replay _declaration_log, replace batch params with group values
       b. Remaining demes in group = _clone_deme(group_template)

    5. Assemble all demes by index, construct SpatialPopulation
```

`_build_template_for_group` is the core of replay:

```python
def _build_template_for_group(self, sig_map):
    # New single-deme template for this group (same entry as SpatialConfigurator.__init__)
    template = Configurator.from_species(self._species, discrete=(self._pop_type != "age_structured"))

    for method_name, kwargs in self._declaration_log:
        resolved = {}
        for key, value in kwargs.items():
            if key in sig_map:
                resolved[key] = sig_map[key]   # Replace with this group's scalar value
            elif isinstance(value, BatchSetting):
                resolved[key] = value.first_value()  # Uncovered batch takes first value
            else:
                resolved[key] = value           # Non-batch parameters pass through as-is

        getattr(template, method_name)(**resolved)

    return template.build()
```

After the first template of a group is fully built, later groups' variant configs share the large arrays of unreplaced fields through `ModelDraft._replace`; parameter discovery, equilibrium recomputation, and the fields that cannot be heterogeneous live in [Heterogeneous Config Sharing](spatial_config_replace.md).

## `_clone_deme`: Zero-Compilation-Overhead Cloning

`_clone_deme(template, config, name)` delegates to the population instance's `_clone(name=..., config=...)`: instances are created via `__new__`, completely bypassing `__init__`, so hook compilation and config construction never run twice.

```python
def _clone_deme(template, config, name):
    # config is handed to the clone by reference; in a homogeneous build every
    # clone shares the one exported config object
    return template._clone(name=name, config=config)
```

In a homogeneous build the template deme (index 0) keeps its own draft, while clones from index 1 on share the single config object returned by `template.export_config()` (large arrays shared by reference).

How `_clone` shares and copies state:

| Category | Contents | Reason |
|-----------|------|------|
| Shared reference | `_species`, `_config` (homogeneous clones share one exported config; the template deme keeps its own draft), `_index_registry`, `_registry`, `compiled_hook_descriptors`, native `HookProgram`, `_hook_runner`, `_genotypes_list`, `_haploid_genotypes_list` | Read-only during simulation; identical across homogeneous demes |
| Shallow-copied lists | `_presets`, `_manual_gamete`, `_manual_zygote`, `_gamete_modifiers`, `_zygote_modifiers` | Each deme can add or remove modifiers without affecting the others |
| Independent copies | `_state` (individual/sperm arrays copied from the template), `_initial_population_snapshot`, `_name`, `_deme_id`, `_tick`, `_params_log` (a fresh native log), `_reconfiguration_log`, `_run_program` | Each deme owns its runtime state and audit trail |
| Reset | Rust session bridge fields (`_rust_lifecycle_backend = None`, ...) | Clones start backend-less and create a session on first run |

> Sharing `_config` is a **build-time** deduplication, not a writable runtime sharing: modify a running deme through `deme(i).write_ecology(...)` / `write_genetics(...)` or `pop.params.tensor_write(...)`.

## `BatchSetting`: Cross-Deme Varying Parameters

```python
from natal.frontend.spatial import batch_setting

# List: index-to-index correspondence
batch_setting([10000, 5000, 5000, 8000])        # kind="scalar"

# NumPy array
batch_setting(np.array([10000, 5000, ...]))      # kind="array"

# Spatial function: (flat_index) -> float or (row, col) -> float
batch_setting(lambda i: 10000 if i < 50 else 5000)  # kind="spatial"
```

All three kinds are uniformly expanded into Python lists via `expand(n_demes, topology)` at `build()` time.

Parameters that accept `BatchSetting`: `carrying_capacity`, `age_1_carrying_capacity`, `eggs_per_female`, `sex_ratio`, `low_density_growth_rate`, `juvenile_growth_mode`, `expected_num_new_adult_females`.

## Construction Cost

A homogeneous build runs the full template build once and clones the remaining demes; a heterogeneous build groups demes by config signature, builds one template per group, then clones. First-template cost depends on the hook count and genetic scale, while cloning only copies state arrays.

This page reports no historical measurements: the former table carried no version or measurement conditions and cannot serve as a current performance guarantee. Measure your own model and workload for performance conclusions.

## Relationship with Existing API

`SpatialConfigurator` does not modify any existing classes:

- The old Builder classes (`AgeStructuredPopulationBuilder` / `DiscreteGenerationPopulationBuilder`) are removed; `SpatialConfigurator` is the only batch configuration path
- `SpatialPopulation.__init__` — unchanged, `build()` ultimately calls it with the pre-built deme list
- The old per-deme construction approach still works

## Limitations

1. **`batch_setting` does not support fitness / presets** — fitness and presets modify NumPy arrays inside config (in-place), which are not well-suited for scalar value expression. For heterogeneous fitness, manually modify the corresponding deme's config arrays after build
2. **spatial kind requires topology** — `batch_setting(lambda topo, i: ...)` requires the topology parameter to have been passed to the builder, otherwise `expand()` will raise an error
3. **Homogeneous demes share the same `_config` reference** — this is build-time deduplication, not writable runtime sharing. Writing array fields of `pop.demes[0]._config` directly bypasses session sync and affects every deme sharing that config; modify a single running deme with `deme(i).write_ecology(...)` / `write_genetics(...)`, or many demes with `pop.params.tensor_write(...)`
