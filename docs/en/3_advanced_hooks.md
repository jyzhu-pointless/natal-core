# Advanced Hook Tutorial

The [basic tutorial](2_hooks.md) covers declarative hooks (`Op.add`, `Op.scale`, etc.), which fit most routine scenarios.
When you need to operate on NumPy arrays directly for more flexible state changes (conditional branches, loops, custom arithmetic, hook-side runtime parameter writes), use the single-parameter callback hook or the selector-based hook.

## Single-Parameter Callback Hook (TickContext)

Callback hooks let you write code that operates on the simulation state directly; they are invoked on every firing. The parameter is a `TickContext` object with this public surface:

| Member | Type / semantics |
|---|---|
| `pop.tick` | Current simulation tick (read-only). |
| `pop.deme_id` | Deme index of this invocation (`0` panmictic, the live deme index under a SpatialPopulation, read-only). |
| `pop.state` | Writable transaction candidate, materialized on first state or metrics access; commits when the callback succeeds. |
| `pop.params` | Writable parameter surface (same writer stack as `pop.params`; attribute writes are validated in the candidate and reach the Rust session and audit log when the callback succeeds). |
| `pop.blueprint` | Read-only dimensions, name catalogs, and engine switches (`n_sexes`, `n_ages`, `n_ztypes`, `discrete`, `stochastic`, `continuous_sampling`, `extreme_speed_mode`, `ztype_names`, `gtype_names`). |
| `pop.metrics` | On-demand metrics view (recomputed on every access). |
| `pop.rng` | Controlled sampler of the persistent Rust RNG stream for this deme; never touches global `numpy.random`. |
| `pop.update()` | Returns a runtime `RuntimeUpdater` bound to the owning population (same domain-method syntax as the build chain). |
| `pop.stop()` / `pop.stop_requested` | Request/query run termination at the event boundary. |

Parameter candidates are copied into Python only when the callback accesses parameters or configuration. Callbacks that only count visits, inspect state, or draw random numbers do not transfer parameter tensors. Validated writes update the native transaction directly, and an exception discards that callback's candidate.

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

Randomness inside callback hooks must come from `pop.rng` (the controlled Rust sampler). Global `np.random` state is never touched:

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def stochastic_culling_hook(pop: TickContext) -> int:
    if pop.tick > 50:
        # Draws advance this deme's persistent Rust stream.
        # Checkpoint restoration also restores the stream position.
        survival_prob = 0.9
        n_current = pop.state.individual_count[:, :, 0]
        pop.state.individual_count[:, :, 0] = pop.rng.binomial(
            n_current.astype(int), survival_prob
        ).astype(float)
    return 0
```

The sampler supports `random`, `uniform`, `normal`, `integers`, and `binomial`; binomial counts and probabilities can broadcast over arrays. Repeated accesses within a callback use the same stream. Failed callbacks discard their candidate draws. Replaying a checkpoint with the same hooks and parameters reproduces the subsequent draws. Retained RNG, parameter, and update handles reject access after the callback ends.

## Execution Paths

The native Rust engine is the only execution backend. Declarative Ops
compile into a CSR plan that runs inside the engine session; single-parameter
Python callbacks are bridged into the session (each invocation gets its own
context wrapper). Out-of-band surfaces -- `trigger_event` and finish events --
run the same CSR plan through the Rust-side interpreter.

## Mixing Hook Types

A single event may mix declarative and callback shapes. Within one event, both kinds execute interleaved in one cross-type `priority` order (lower values first; ties keep registration order):

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

Same-priority hooks run in registration order (stable sort); priority semantics are consistent across all three entry points: in-tick, `trigger_event`, and finish events.

## Choosing a Hook Shape

| Hook type | Flexibility | Readability | Typical use |
|----------|--------|--------|----------|
| Declarative | medium | high | most routine scenarios |
| Selector-based | high (baked indices) | medium | scenarios targeting specific genotypes |
| Callback | high | medium | compute-heavy logic, parameter or custom writes |

This table is qualitative guidance and carries no measured numbers. Declarative Ops
execute entirely inside the engine session; callback hooks cross a Python<->Rust
boundary per firing. Actual performance depends on model size and callback content —
measure it on your own workload.

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

Semantics (uniform across entry points):

- Writes are jsonc-bounds-validated; values outside the `parameters.jsonc` `bounds` raise `ValueError`;
- Visible to later stages of the same tick (both in-tick and explicit `trigger_event` writes commit through the owning Rust session);
- Every actual change appends to `pop.params_log` as `(tick, name, old, new)`;
- Vector/tensor parameters use `pop.params.tensor_write(name, values)`.

Outside a callback, read custom fields through `pop.config.custom['name']` (a query snapshot); initialize them at build time with `.custom(temperature=25.0)` and change them at runtime with `pop.update().custom(...)`. Custom values preserve bool/int/float types and the shape of arrays of any rank. Spatial custom values belong to their own deme and are included in checkpoints.

Callbacks have **no public read path** for custom fields: `TickContext` exposes no `config`, `ctx.state` has no `config` attribute, and `ctx.params` accepts registered parameters only. Callbacks can write them (`ctx.update().custom(...)`, committed with the event transaction); read them outside callbacks. Custom fields are not in the parameter route table, so `Op.set_param` rejects them at compile time with `ValueError`.

For chain-style updates inside a hook, use the `RuntimeUpdater` returned by `pop.update()` (same domain-method syntax as the build chain).

## Event transactions

- A callback commits state, ecology, genetic parameters, custom values, and RNG position together. Invalid state or an exception discards that callback's candidate; earlier successful callbacks remain committed.
- All hooks of an event (declarative and callback) execute in one cross-type `priority` order and see earlier writes. Declarative parameter writes commit once per native event — repeated writes to one parameter produce one audit row from its initial to final value — while each Python callback commits separately and produces its own rows.
- A successful parameter update is visible to later callbacks and stages of the same tick in ordinary and spatial models. Rust owns the current values and the parameter log; Python configuration reads return isolated snapshots. Spatial `ctx.params.tensor_write()` and deme parameter writes fork changed genetics inside the native session and leave other demes unchanged.
- Callback exceptions preserve their original Python type. A failed session requires reset or checkpoint restoration before another run.
- `stop()` halts at the current event boundary and preserves its state and phase. The tick does not advance; continuing requires reset or restoration of a Ready checkpoint.
- State arrays cross into Python only when a callback accesses state or metrics. A callback using only parameters or RNG creates no Python state arrays.

## Related Sections

- [Hook System](2_hooks.md) - basic hook concepts and declarative hooks
- [Runtime Parameter Modification](3_runtime_modification.md)
- [Modifier Mechanism](3_modifiers.md)
- [Simulation Engine Deep Dive](4_simulation_engine.md)
