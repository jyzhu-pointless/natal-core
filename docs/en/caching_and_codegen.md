# Numba Cache and Code Generation Mechanism

This document explains how Numba compilation caching works in `natal` and why code generation (codegen) is necessary to ensure cache validity across processes.

## Background: Numba's Caching Mechanism

Numba's `@njit(cache=True)` decorator can cache compiled functions to disk (`__pycache__`). The cache key is composed of the following elements:

- The function's qualified name (`__qualname__`)
- The function's bytecode hash, which is based on compiled bytecode rather than source text; changing comments alone does not affect the cache, but changing variable names does
- The type signatures of global variables referenced by the function

When the same Python function is re-imported in a subsequent run, Numba recalculates the cache key. If it matches the cache on disk, the compiled machine code is loaded directly, skipping the compilation process.

## Core Problem: Function Parameters and Cross-Process Cache Invalidation

### Function Parameter Approach (Not Cacheable)

In the initial no-codegen approach, hook functions were passed as parameters to lifecycle functions:

```python
@njit_switch(cache=True)
def run_discrete_tick_with_hooks(
    state, config, registry,
    first_hook: Callable,   # ← function as parameter
    early_hook: Callable,
    late_hook: Callable,
):
    ...
    result = first_hook(ind_count, tick)
```

When Numba compiles this function, the type of the `first_hook` parameter is a specific `Dispatcher` instance. Numba creates a **specialization** for each distinct set of Dispatcher parameters. The specialization's cache key includes **type identification information** (including overload fingerprints and other instance-bound information) for that Dispatcher.

The problem is: across process restarts, even with the same Python source code, Numba creates entirely new `Dispatcher` instances. The new instances carry different type identifiers (because internal states such as overload resolution are newly generated), causing the cache keys of previously compiled specializations to mismatch, forcing Numba to **recompile**.

### Global Variable Approach (Cacheable)

The codegen approach instead sets hook functions as module-level global variables:

```python
# Independent module created via code generation
_FIRST_HOOK = None   # injected later
_EARLY_HOOK = None
_LATE_HOOK = None

@njit_switch(cache=True)
def _lifecycle_tick_527c055(...):
    ...
    result = _FIRST_HOOK(ind_count, tick)  # ← global variable, not a parameter
```

When Numba compiles `_lifecycle_tick_527c055`, `_FIRST_HOOK` is a module-level global variable. Numba's cache key depends only on the function's own source text and function name, and the **type signature** of the global variable (`(ind_count, tick) -> int`). It **does not depend** on the identity of the specific Dispatcher object that the global variable points to.

Therefore, as long as the generated function name and source code remain unchanged (guaranteed by the hash_key), the cache key remains stable across processes.

## Code Generation Overview

There are currently two types of code generation:

### 1. Combined Hook Generation

Merges multiple @njit hooks for the same event into a single function to avoid the overhead of calling them individually.

**Entry point**: `compile_combined_hook(njit_fns, name)`

**Generation logic** (`natal/hooks/compiler.py`):

```python
lines = ["from natal.hook_dsl import njit_switch"]
lines.extend([f"{placeholder} = None" for placeholder in placeholder_names])
lines.append(f"def {fn_name}(ind_count, tick, deme_id=-1):")
for placeholder in placeholder_names:
    lines.append(f"    _result = {placeholder}(ind_count, tick, deme_id)")
    lines.append("    if _result != 0:")
    lines.append("        return _result")
lines.append("    return 0")
```

Generated module file:

```python
_FN_0 = None
_FN_1 = None

@njit_switch(cache=True)
def _combined_hook_19a81f6c(...):
    _result = _FN_0(ind_count, tick, deme_id)
    if _result != 0: return _result
    _result = _FN_1(ind_count, tick, deme_id)
    if _result != 0: return _result
    return 0
```

### 2. Lifecycle Wrapper Generation

Generates the complete lifecycle loop (reproduction → survival → aging, along with CSR operations and hook calls for the three events) as a standalone @njit function.

**Entry point**: `compile_lifecycle_wrapper(is_discrete, first_hook, early_hook, late_hook)`

Generates two functions:
- `_lifecycle_tick_{hash}`: Single tick, including CSR event execution and hook calls
- `_lifecycle_run_{hash}`: Multiple tick loop, including history recording

The `is_discrete` parameter determines whether to use the discrete-generation version (no sperm storage) or the age-structured version (with sperm storage).

Generated module file example (`hook_codegen/lifecycle_discrete_527c055.py`):

```python
import numpy as np
from natal.kernels.simulation_kernels import (
    run_discrete_reproduction, run_discrete_survival, run_discrete_aging,
)
from natal.hooks.executor import execute_csr_event_program_with_state
from natal.hooks.types import EVENT_FIRST, EVENT_EARLY, EVENT_LATE, ...

_FIRST_HOOK = None
_EARLY_HOOK = None
_LATE_HOOK = None

@njit_switch(cache=True)
def _lifecycle_tick_527c055(state, config, registry):
    ind_count = state.individual_count.copy()
    tick = state.n_tick
    # Execute FIRST event (CSR + hook)
    result = execute_csr_event_program_with_state(registry, EVENT_FIRST, ...)
    result = _FIRST_HOOK(ind_count, tick)
    # reproduction
    ind_count = run_discrete_reproduction(ind_count, config)
    # Execute EARLY event
    ...
    ind_count = run_discrete_survival(ind_count, config)
    # Execute LATE event
    ...
    ind_count = run_discrete_aging(ind_count)
    return (ind_count, tick + 1), RESULT_CONTINUE
```

## Key Functions and Data Flow

### 1. `stable_callable_identity(fn)` → `module:qualname`

Generates a cross-process stable identity string for identifying a callable. For @njit functions, takes `fn.py_func.__module__` and `__qualname__`; for regular functions, similarly:

```python
def _stable_callable_identity(fn):
    py_fn = getattr(fn, "py_func", fn)
    module_name = getattr(py_fn, "__module__", "<unknown>")
    qualname = getattr(py_fn, "__qualname__", getattr(py_fn, "__name__", "<unknown>"))
    return f"{module_name}:{qualname}"
```

### 2. `hash_key(parts)` → 16-character hex

Uses SHA-256 to compute the digest of a set of identifiers, truncating to the first 16 characters as the suffix for module and function names:

```python
def _hash_key(parts):
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]
```

### 3. File Generation and Loading

```python
# Generation
module_path = write_codegen_module(stem, source)
# → Written to .numba_cache/hook_codegen/{stem}.py

# Loading
module = load_codegen_module(stem, module_path)
# → Imported into sys.modules as natal._hook_codegen_{stem}

# Inject global variables
setattr(module, "_FIRST_HOOK", first_hook)

# Retrieve target function
tick_fn = getattr(module, fn_name)
```

**Timing constraint**: The `setattr` injection of global variables must be completed before the first call to `_lifecycle_tick_*`. This is because Numba checks the type signature of global variables when compiling/loading the cache: if `_FIRST_HOOK` is still `None` (`NoneType`) at first call, while the cache records a `Dispatcher` (a callable type), the type mismatch will cause cache invalidation. From the data flow perspective, the current code ensures this timing (injection occurs after module loading and before the first call), but this is a design constraint that needs to be maintained.

### 4. Complete Data Flow

```
User registers hook
      ↓
  compiler.hook() decorator
      ↓
  pop.register_compiled_hook(desc)
      ↓
  get_compiled_event_hooks()
      ↓
  CompiledEventHooks.from_compiled_hooks()
      ├─ compile_combined_hook() → merge same-event hooks
      └─ compile_lifecycle_wrapper() → generate tick/run functions
           ├─ _gen_lifecycle_source() → build source code
           ├─ write_codegen_module() → write to disk
           ├─ load_codegen_module() → import module
           └─ setattr(module, "_XXX_HOOK", hook) → inject global variables
      ↓
  Population calls hooks.run_fn() or hooks.run_discrete_fn()
```

### 5. Numba Singleton Default Hook

`_noop_hook` serves as a singleton `@njit(cache=True)` function used as the default value when no hooks are registered. All lifecycle modules reference the same `_noop_hook` as `_FIRST_HOOK` / `_EARLY_HOOK` / `_LATE_HOOK` when no hooks are present, so the Numba cache remains valid.

```python
@njit_switch(cache=True)
def _noop_hook(ind_count: np.ndarray, tick: int, deme_id: int = 0) -> int:
    return 0
```

## Why the Global Variable Approach Is Cacheable While the Parameter Approach Is Not

| Aspect | Function Parameter Approach | Global Variable Approach |
|---|---|---|
| What Numba sees | `first_hook` as a parameter, typed as a specific Dispatcher | `_FIRST_HOOK` as a global variable reference |
| Number of specializations | Creates a new specialization for each hook combination | Always a single compilation (determined by function name + source) |
| Cache key content | Function bytecode + Dispatcher type identification info | Function bytecode + function name |
| Cross-process stability | ❌ Dispatcher type identification info differs on each restart | ✅ Function name and bytecode remain unchanged across processes |
| Isolation | Implicit (parameter isolation) | Requires isolation through different module names |

The core of the global variable approach is: **Numba, when handling global variables, primarily checks type signature compatibility rather than object identity**. The `hash_key` ensures that different hook combinations generate different function names, while the same hook combination generates the same function name, so each combination has only one compilation result that is reusable across processes.

## Outstanding Issues

### No-Hook Scenario

When no hooks are registered, `from_compiled_hooks` still compiles the lifecycle wrapper (because `NUMBA_ENABLED` is True), and all three global variables are set to `_noop_hook`. The generated function name is fixed (the `stable_callable_identity` of `_noop_hook` is unique), so each population type (discrete/structured) has only one cached version.

When Numba is disabled (e.g., in test environments), `from_compiled_hooks` does not compile the lifecycle wrapper, and `run_fn` / `run_discrete_fn` remain `None`. The population code falls back to the parameter-based kernel functions (`run_with_hooks` / `run_discrete_with_hooks`), which automatically degrade to Python execution via `njit_switch`.
