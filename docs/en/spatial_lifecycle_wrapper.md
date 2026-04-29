# Spatial Model Lifecycle Wrapper Refactoring

## Background

The original spatial simulation kernel (`run_spatial_tick_with_migration`) was a "half-baked" implementation:

1. **No hook support** — it could only run lifecycle stages (reproduction → survival → aging) inside each deme, without executing user-registered hook events (first/early/late)
2. **No heterogeneous config support** — it only accepted a single shared `PopulationConfig`, requiring all demes to use identical parameters
3. **Python dispatch fallback was too broad** — with hooks or heterogeneous configs, the entire spatial run degraded to per-deme Python loops, completely unable to leverage Numba acceleration

Previously, the panmictic (single population) path had already solved the coexistence of hooks + njit via `compile_lifecycle_wrapper`. This modification extends the same approach to the spatial path.

## Refactoring Goals

Eliminate duplication between the spatial lifecycle sequence and the panmictic one: make the spatial prange body **directly call the panmictic lifecycle tick function**, instead of rewriting the stage invocation sequence.

## Changes

### 1. compiler.py — Add `deme_id` parameter to panmictic tick

In `_gen_lifecycle_source`, add `deme_id=-1` to the tick function signature:

```python
# Before refactoring
def _lifecycle_tick_<hash>(state, config, registry):
    ...
    result = _FIRST_HOOK(ind_count, tick)

# After refactoring
def _lifecycle_tick_<hash>(state, config, registry, deme_id=-1):
    ...
    result = _FIRST_HOOK(ind_count, tick, deme_id)
```

Corresponding calls to `_FIRST_HOOK`/`_EARLY_HOOK`/`_LATE_HOOK` and `execute_csr_event_program_with_state` in the tick body all pass `deme_id`.

This way, the spatial path can pass the actual deme index (`d`), allowing hooks to be aware of the deme context, while the panmictic path does not pass this parameter (defaulting to `-1`), keeping behavior unchanged.

### 2. compiler.py — Spatial lifecycle wrapper delegates to panmictic tick

The spatial prange body no longer arranges lifecycle stages itself; instead, it **imports the panmictic lifecycle tick function and calls it within prange**:

```
# Before refactoring
for d in prange(n_demes):
    cfg = config_bank[deme_config_ids[d]]
    ind = ind_all[d].copy()
    execute_csr(FIRST, ...)    ← Duplicated stage
    _FIRST_HOOK(...)
    run_reproduction(...)       ← Duplicated stage
    execute_csr(EARLY, ...)
    _EARLY_HOOK(...)
    run_survival(...)
    execute_csr(LATE, ...)
    _LATE_HOOK(...)
    run_aging(...)

# After refactoring
for d in prange(n_demes):
    cfg = config_bank[deme_config_ids[d]]
    ind = ind_all[d].copy()
    sperm = sperm_all[d].copy()
    state = PopulationState(tick, ind, sperm)    ← Construct State object
    (ind, sperm, _), result = _run_deme_tick(state, cfg, registry, d)
```

The spatial module no longer contains its own `_FIRST_HOOK`/`_EARLY_HOOK`/`_LATE_HOOK` global variables — hook globals are only set on the panmictic module. The panmictic tick function imported by the spatial module resolves hooks through its own module's globals.

### 3. compiler.py — Spatial source generation simplified

`_gen_spatial_lifecycle_source` now accepts `panmictic_stem` and `panmictic_tick_fn_name` parameters. The generated module source is significantly simplified:

```python
# Before refactoring: 6 imports, 3 hook globals
import numpy as np
from natal.kernels.simulation_kernels import (run_reproduction, ...)
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.hooks.executor import execute_csr_event_program_with_state
from natal.hooks.types import EVENT_FIRST, EVENT_EARLY, EVENT_LATE, ...
from natal.numba_utils import njit_switch, prange

_FIRST_HOOK = None
_EARLY_HOOK = None
_LATE_HOOK = None

# After refactoring: 3 imports + 1 panmictic tick import
import numpy as np
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.hooks.types import RESULT_CONTINUE, RESULT_STOP
from natal.numba_utils import njit_switch, prange
from natal.population_state import PopulationState
from natal._hook_codegen_lifecycle_structured_<key> import _lifecycle_tick_<key> as _run_deme_tick
```

### 4. compiler.py — CompiledEventHooks extension

4 new slots are added to `CompiledEventHooks`:

- `spatial_tick_fn` / `spatial_run_fn` — age-structured spatial lifecycle wrappers
- `spatial_discrete_tick_fn` / `spatial_discrete_run_fn` — discrete-generation spatial lifecycle wrappers

In `from_compiled_hooks()`, when Numba is enabled, in addition to the original panmictic wrappers, spatial wrappers are **pre-compiled**:

```python
if NUMBA_ENABLED:
    # Panmictic wrappers
    result.run_tick_fn, result.run_fn = compile_lifecycle_wrapper(...)
    result.run_discrete_tick_fn, result.run_discrete_fn = compile_lifecycle_wrapper(...)
    # Spatial wrappers (delegate to the above panmictic wrappers)
    result.spatial_tick_fn, result.spatial_run_fn = compile_spatial_lifecycle_wrapper(...)
    result.spatial_discrete_tick_fn, result.spatial_discrete_run_fn = compile_spatial_lifecycle_wrapper(...)
```

### 5. spatial_population.py — Runtime Adaptation

#### `_should_use_python_dispatch()` Narrowed

Original conditions:
```python
if not is_numba_enabled(): return True
if has_python_hooks() or has_compiled_hooks(): return True  # ← Too broad
return has_heterogeneous_configs()                           # ← Too broad
```

New conditions:
```python
if not is_numba_enabled(): return True
if has_python_hooks(): return True     # Only pure Python callbacks fall back
return False                           # Everything else goes through njit
```

This means:
- **CSR registry hooks** (declarative Op, njit selector hooks) → can execute via `execute_csr_event_program_with_state` within njit
- **User njit hooks** → execute within njit via module-level globals
- **Heterogeneous configs** → looked up by deme index via `config_bank` within njit
- Pure Python callable hooks → still require Python dispatch fallback

#### `_is_discrete_demes()` Helper Method

Determines deme type by checking whether the first deme's state has a `sperm_storage` attribute, then selects the structured or discrete spatial wrapper accordingly.

#### `_run_codegen_wrapper_tick()` Replacement

Changes from calling `run_spatial_tick_with_migration(single_config)` to calling `spatial_tick_fn(config_bank, deme_config_ids, registry, ...)`:

1. Call `_stack_deme_state_arrays()` to stack all deme states
2. Call `_heterogeneous_config_bank_and_ids()` to build the config bank
3. Select structured or discrete tick function based on `_is_discrete_demes()`
4. Pass registry (CSR hook data) and migration parameters
5. Write back state

#### `_run_codegen_wrapper_steps()` Replacement

Same as above, but uses `spatial_run_fn` to execute multiple ticks at once, supporting `record_interval` history recording.

## Spatial Model Complete Workflow

### Build Phase

```
Species + Drive + Demes → IndexRegistry / PopulationConfig / PopulationState
                          ↓
SpatialPopulation.__init__()
                          ↓
_compile_spatial_hooks_from_demes()
    → _collect_effective_compiled_hooks()     ← Collect hooks from all demes
    → _build_hook_program()                   ← Compile CSR HookProgram
    → CompiledEventHooks.from_compiled_hooks()
        → compile_combined_hook()             ← Merge same-event njit hooks
        → compile_lifecycle_wrapper()         ← Pre-compile panmictic wrapper
        → compile_spatial_lifecycle_wrapper() ← Pre-compile spatial wrapper
```

### Run Phase — `run_tick()`

```
spatial.run_tick()
  │
  ├─ _should_use_python_dispatch()?
  │    ├─ True  → _run_python_dispatch_tick()
  │    │            for deme in demes: deme.run_tick()
  │    │            run_spatial_migration(stacked_state)
  │    │
  │    └─ False → _run_codegen_wrapper_tick()
  │                  _stack_deme_state_arrays()
  │                  _heterogeneous_config_bank_and_ids()
  │                  spatial_tick_fn(config_bank, registry, ...)
```

### njit prange Internal Flow (One Tick)

```
_spatial_tick_<hash>(ind_all, sperm_all, config_bank, deme_config_ids, registry, tick, ...)
  │
  ├─ n_demes = ind_all.shape[0]
  ├─ stopped = zeros(n_demes, bool)
  │
  ├─ for d in prange(n_demes):              ← Execute each deme in parallel
  │    │
  │    ├─ cfg = config_bank[deme_config_ids[d]]  ← Heterogeneous config lookup
  │    │
  │    ├─ Construct PopulationState(tick, ind, sperm)
  │    │
  │    ├─ (ind, sperm, _), result = _run_deme_tick(state, cfg, registry, d)
  │    │    │                                    ← Delegate to panmictic tick
  │    │    ├─ [FIRST event]                       (with deme_id=d)
  │    │    │    execute_csr_event_program(registry, FIRST, ind, sperm, tick, d)
  │    │    │    _FIRST_HOOK(ind, tick, d)
  │    │    │
  │    │    ├─ Reproduction
  │    │    │
  │    │    ├─ [EARLY event]
  │    │    │    execute_csr_event_program(registry, EARLY, ind, sperm, tick, d)
  │    │    │    _EARLY_HOOK(ind, tick, d)
  │    │    │
  │    │    ├─ Survival / Competition
  │    │    │
  │    │    ├─ [LATE event]
  │    │    │    execute_csr_event_program(registry, LATE, ind, sperm, tick, d)
  │    │    │    _LATE_HOOK(ind, tick, d)
  │    │    │
  │    │    └─ Aging → returns (ind, sperm, tick+1), result
  │    │
  │    ├─ if result != CONTINUE: stopped[d] = True
  │    ├─ ind_all[d] = ind
  │    └─ sperm_all[d] = sperm
  │
  ├─ run_spatial_migration(                 ← Unified migration after prange
  │      ind_all, sperm_all, ...,
  │      config_bank[0], ...)
  │
  └─ Check stopped[] → return was_stopped
```

### Key Design Decisions

1. **Panmictic tick as the single source of truth**: the spatial prange body no longer repeats the lifecycle stage sequence; instead, it **delegates to the panmictic lifecycle tick**. The lifecycle order (FIRST → reproduction → EARLY → survival → LATE → aging) is **defined in only one place**; adding/changing stages does not miss the spatial path

2. **deme_id passing**: the panmictic tick's `deme_id=-1` default parameter allows both call paths to work correctly:
   - Panmictic call: no deme_id parameter → defaults to -1 → behavior unchanged
   - Spatial call: passes `d` (deme index) → hooks can perceive deme context

3. **Config bank always used**: even if all demes share the same config, it is still passed via config bank, keeping the generated module signature uniform

4. **Migration uses config_bank[0]**: the migration kernel only needs to read `is_stochastic` and `use_continuous_sampling`, which are verified to be consistent across all demes during spatial population construction

5. **Stop signal collection**: prange cannot directly break back to the main thread; a `stopped[n_demes]` boolean array is used to mark each deme's lifecycle. After prange completes, the stopped array is scanned serially

6. **Hook globals centralized in the panmictic module**: the spatial module no longer sets `_FIRST_HOOK` and other globals. These are only set on the panmictic module; the panmictic tick imported by spatial resolves hooks through its own module's globals. Each unique hook combination corresponds to a unique source hash, ensuring Numba `cache=True` works across processes

## User API: Simplified `@hook`

The `deme_selector` parameter of `@hook()` has been renamed to `deme` for more intuitive semantics:

```python
@hook(event="early", custom=True, deme="*")    # All demes (default)
@hook(event="early", custom=True, deme=3)       # Deme 3 only
@hook(event="early", custom=True, deme=[0,2,4]) # Specified list
```

No need to manually add `@njit` — the decorator handles it automatically:

```python
@hook(event="early", custom=True)
def my_hook(ind_count, tick, deme_id=-1):
    """Automatically njit-compiled when Numba is enabled, Python fallback when disabled."""
    if deme_id % 2 == 0:
        ind_count[0, 0, 0] *= 0.5
```

Used with `PopulationBuilder`:

```python
pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=sp, name="demo")
    .initial_state(...)
    .reproduction(...)
    .competition(...)
    .presets(drive)
    .hooks(my_hook)       # @hook-decorated function passed directly
    .build()
)
```
