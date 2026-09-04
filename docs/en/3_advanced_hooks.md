# Advanced Hook Tutorial

The [basic tutorial](2_hooks.md) covers declarative hooks (`Op.add`, `Op.scale`, etc.), which fit most routine scenarios.
When you need to operate on NumPy arrays directly for more flexible state changes (conditional branches, loops, custom arithmetic, hook-side runtime parameter writes), use the single-parameter callback hook or the selector-based hook.

## Single-Parameter Callback Hook (TickContext)

Callback hooks let you write code that operates on the simulation state directly; they are invoked on every firing. The parameter is a `TickContext` object with this public surface:

| Member | Type / semantics |
|---|---|
| `pop.tick` | Current simulation tick (read-only). |
| `pop.deme_id` | Deme index of this invocation (`-1` panmictic, read-only). |
| `pop.state` | Writable state view (short-term loan; writes take effect immediately). |
| `pop.params` | Writable parameter surface (same writer stack as `pop.params`; attribute writes are bounds-validated and reach the draft, the live Rust session, and the parameter snapshot log). |
| `pop.blueprint` | Read-only dimensions, name catalogs, and engine switches (`n_sexes`, `n_ages`, `n_ztypes`, `discrete`, `stochastic`, `continuous_sampling`, `extreme_speed_mode`, `ztype_names`, `gtype_names`). |
| `pop.metrics` | On-demand metrics view (recomputed on every access). |
| `pop.rng` | Deterministic random stream (derived from population slot, tick, deme, hook index; never touches global `numpy.random`). |
| `pop.update()` | Returns a runtime `Configurator` bound to the owning population (same syntax as the build chain). |
| `pop.stop()` / `pop.stop_requested` | Request/query run termination at the event boundary. |

### Basic Usage

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def custom_release_hook(pop: TickContext) -> int:
    # pop.state.individual_count is the individual-count NumPy array
    # with shape (sex, age, genotype)
    # sex=0 is female, sex=1 is male

    # release 100 individuals every 10 ticks
    if pop.tick % 10 == 0:
        # assume the genotype index of Var|WT is 1
        pop.state.individual_count[:, :, 1] += 100

    return 0  # 0 continues the simulation
```

### Array-Indexing Notes

`pop.state.individual_count` is dimension-ordered `(sex, age, genotype)`:

- `sex=0` is female (FEMALE), `sex=1` is male (MALE)
- use integer indices directly; take `.value` for enums (`Sex.MALE.value`)

```python
# correct
male_count = pop.state.individual_count[1, :, :].sum()
female_count = pop.state.individual_count[0, :, :].sum()

# or use .value
male_count = pop.state.individual_count[Sex.MALE.value, :, :].sum()
```

### Full Example (with runtime parameter writes)

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="early", priority=5)
def custom_culling_hook(pop: TickContext) -> int:
    # selective culling of one genotype
    if pop.tick > 50:
        # genotype index of WT|WT is 0
        wt_wt_count = pop.state.individual_count[:, :, 0].sum()
        if wt_wt_count > 10000:
            pop.state.individual_count[:, :, 0] = pop.state.individual_count[:, :, 0] * 0.9

    # hook-side runtime parameter write: takes effect immediately and lands in params_log
    pop.params.carrying_capacity = float(pop.params.carrying_capacity) * 0.5
    return 0
```

- Return `0` (or `RESULT_CONTINUE`) to continue; a non-zero value (or `RESULT_STOP`) stops immediately.
- `pop.params.<name> = v` is the recommended parameter-write channel inside hooks: the write is jsonc-bounds-validated, visible to later stages of the same tick, and appends one `(tick, name, old, new)` row to `pop.params_log`.
- `stop()` called **in the late event stops at the event boundary immediately** (the rest of the tick does not execute and the tick does not advance); `stop_requested` queries that state.

## Selector-Based Hooks

Selector-based hooks are a callback-hook variant where genotypes are addressed symbolically (e.g. `"Var|WT"`). The framework resolves the symbols to integer indices at registration; single-value selectors collapse to `int`, multi-value selectors are injected as `int32` ndarrays:

### Basic Usage

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", selectors={"target_gt": "Var|WT"}, priority=10)
def cap_target(pop: TickContext, target_gt: int) -> int:
    # target_gt is the selector-resolved genotype index (int for single values)
    if pop.tick % 10 == 0:
        pop.state.individual_count[:, :, target_gt] *= 0.95
    return 0
```

### Selector Resolution Rules

`selectors` values accept:

| Type | Example | Injected value |
|------|---------|-----------------|
| `str` (genotype label) | `"WT\|WT"` | single index (`int`) |
| `str` (wildcard) | `"*"` | all genotype indices (`int32` array) |
| `int` | `3` | used as index (`int`) |
| `range` | `range(3)` | `[0, 1, 2]` (`int32` array) |
| `list` / `tuple` | `["WT\|Dr", 4]` | multiple indices (`int32` array) |
| `Genotype` object | `species.genotypes[0]` | matching index (`int`) |

Single-value selectors collapse to `int`, multi-value selectors stay `np.ndarray[int32]`.
The function signature is `def hook(pop: TickContext, <selector name>) -> int` -- selector names are the keyword-argument names.

### Multi-Selector Example

```python
@hook(event="early", selectors={"drive": "Var|WT", "wt": "WT|WT"})
def balance_population(pop: TickContext, drive: int, wt: int) -> int:
    drive_count = pop.state.individual_count[:, :, drive].sum()
    wt_count = pop.state.individual_count[:, :, wt].sum()

    if drive_count > wt_count * 2:
        pop.state.individual_count[:, :, drive] *= 0.8

    return 0
```

> **Note**: selectors use exact string matching and do not support pattern syntax (`::`, `|*`, ...). For pattern matching, convert to index arrays with `GenotypeSelector` before registration.

## Random Sampling Inside Hooks

Randomness inside callback hooks must come from `pop.rng` (an `np.random.Generator`). Global `np.random` state is never touched:

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def stochastic_culling_hook(pop: TickContext) -> int:
    if pop.tick > 50:
        # pop.rng is an invocation-independent deterministic stream:
        # the same (slot, tick, deme, hook index) yields the same draws
        # on the reference and Rust backends
        survival_prob = 0.9
        n_current = pop.state.individual_count[:, :, 0]
        pop.state.individual_count[:, :, 0] = pop.rng.binomial(
            n_current.astype(int), survival_prob
        ).astype(float)
    return 0
```

The stream is derived from `population slot ^ (tick * 1_000_003) ^ ((deme_id + 7) * 6_559) ^ ((hook_index + 1) * 31)`. **Reproducibility scope**: for the same `setup(stochastic=True, seed=...)` and the same hook combination, the reference and Rust backends produce bit-reproducible deterministic (`stochastic=False`) trajectories as well as identical random draws from seed-driven streams; any global custom randomness (`np.random.seed(...)` etc.) is outside the promise.

## Execution Paths

The physical execution path of hooks is decided by the backend selected for the population (see [Backend Selection and Performance](4_backend_selection.md)):

- **Reference (Python) backend**: declarative Ops compile into a CSR plan interpreted in Python per event; callback hooks are invoked directly.
- **Rust (native extension) backend**: the CSR plan and dispatcher run inside the Rust session; single-parameter callbacks are bridged into the session (each invocation gets its own context wrapper).
- Both paths run the same event order and the same deterministic arithmetic; `stochastic=False` trajectories are bitwise identical.

`backend="numba"` has been removed -- selecting it raises a `ValueError` with a migration hint; use `"rust"` or `"python"` instead.

## Mixing Hook Types

A single event may mix declarative and callback shapes; all run in `priority` order (lower values first):

```python
from natal.frontend.hooks import hook, Op


# declarative hook: periodic release
@hook(event="first", priority=10)
def release_hook():
    return [Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 10 == 0")]


# selector-based hook
@hook(event="first", priority=7, selectors={"drive": "Var|WT"})
def check_drive_threshold(pop, drive):
    drive_count = pop.state.individual_count[:, :, drive].sum()
    if drive_count > 10000:
        pass
    return 0


# callback hook: light mortality
@hook(event="first", priority=5)
def custom_process_hook(pop):
    for age in range(pop.state.individual_count.shape[1]):
        pop.state.individual_count[:, age, :] *= 0.99
    return 0


pop = (
    nt.AgeStructuredPopulation.setup(species=sp)
    .hooks(release_hook, check_drive_threshold, custom_process_hook)
    .build()
)
```

The execution order of same-priority hooks is unspecified; priority semantics are consistent across backends.

## Performance Comparison

| Hook type | Performance | Flexibility | Readability | Typical use |
|----------|------|--------|--------|----------|
| Declarative | high | medium | high | most routine scenarios |
| Selector-based | high (baked indices) | medium | medium | scenarios targeting specific genotypes |
| Callback | medium (Python callback) | high | medium | compute-heavy logic, parameter and custom writes |

On the Rust backend, declarative Ops execute entirely inside the session -- the fastest path; callback hooks cross a Python<->Rust boundary per firing.

## Modifying Parameters at Runtime

Inside hooks, the recommended parameter-write channel is `pop.params`:

```python
import natal as nt
from natal.frontend.hooks.tick_context import TickContext

@nt.hook(event="early")
def heatwave(pop: TickContext) -> int:
    if pop.tick == 10:
        pop.params.carrying_capacity = 2000.0
        pop.params.sex_ratio = 0.55
    return 0
```

Semantics (uniform across backends):

- Writes are jsonc-bounds-validated; values outside the `parameters.jsonc` `bounds` raise `ValueError`;
- Visible to later stages of the same tick (reference path writes the draft directly; Rust path writes the session ecology columns directly);
- Every actual change appends to `pop.params_log` as `(tick, name, old, new)`;
- Vector/tensor parameters use `pop.params.tensor_write(name, values)`.

Custom fields are read/written via `pop.state` / `config.custom['name'][()]`, initialized at build time with `.custom(temperature=25.0)` and changed at runtime via `pop.update().custom(...)`. Custom fields are not in the parameter registry, so `pop.params` cannot reach them.

For chain-style updates inside a hook, use the Configurator returned by `pop.update()` (same syntax as the build chain).

## Slice-④ Consistency Terms

- **`stop()` in the late event**: halts immediately at the event boundary; the rest of the current tick does not execute and the tick does not advance.
- **After `stop()`**: `run()` must be preceded by `reset()`; otherwise `run()` raises.
- **Hook exceptions**: the reference backend re-raises the original type; the Rust backend wraps the bridged error as `RuntimeError` (the original information is preserved in `__cause__`), so the backend asymmetry of exception types is intentional.
- **Rust hook-side parameter writes merge after `run()`** (post-HB-2 fix): session-side writes evolve inside the session ecology columns; when `run()` returns, the audited transitions are appended to `params_log` under their own commit ticks and the final values are synchronized into the draft -- no dirty-bridge push-back needed.

## Related Sections

- [Hook System](2_hooks.md) - basic hook concepts and declarative hooks
- [Runtime Parameter Modification](3_runtime_modification.md)
- [Modifier Mechanism](3_modifiers.md)
- [Simulation Engine Deep Dive](4_simulation_engine.md)
- [Backend Selection and Performance](4_backend_selection.md)
