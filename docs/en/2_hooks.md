# Hook System

Hooks are used to insert user logic at fixed points in the simulation workflow.

If you want to perform "additional operations" at certain stages of each tick -- such as periodic releases, conditional interventions, or threshold termination -- Hooks are the most direct way to do so.

## Hook Timing

Hook timing includes:

- `first`: Early stage of each tick.
- `early`: After the reproduction step, before the survival step.
- `late`: After the survival step, before the aging step.
- `finish`: When the simulation ends (not part of any single tick).

Among these, `finish` is a one-time event, while `first`, `early`, and `late` can be executed repeatedly across multiple ticks as needed.

When selecting an event, it is recommended to first clarify at which specific time point the intervention occurs, as this can significantly impact the interpretation of results.

## Declarative Hooks

For most users, it is recommended to use `@nt.hook` with `nt.Op.*`, registering in a chain on the population object:

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

@nt.hook(event="first", priority=10)
def periodic_release():
    return [
        nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0"),
        nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.98),
    ]


pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop", stochastic=True, continuous_sampling=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000, "Var|WT": 0},
        "male": {"WT|WT": 1000, "Var|WT": 0}
    })
    .survival(
        female_age_based_survival=0.85,
        male_age_based_survival=0.8
    )
    .reproduction(eggs_per_female=50.0)
    .competition(
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=10000
    )
    .hooks(periodic_release)
    .build()
)

pop.run(n_steps=200, record_every=10)
```

This approach is highly readable, easy to maintain, and makes it easier for teams to review the model logic.

## Three Hook Authoring Shapes

`@nt.hook` detects the shape from the function signature (at registration time):

| Shape | Signature | Notes |
|-------|-----------|-------|
| Declarative | No parameters, returns `List[HookOp]` | Called once at registration; the return value is compiled into a CSR plan |
| Callback | Single parameter `def hook(pop: TickContext) -> int` | Called once per tick; reads and writes state and parameters through `TickContext` |
| Selector callback | Single parameter + `selectors={...}` keyword args | Selector values are resolved at registration and injected on each call |

The legacy `(state, config, deme_id)` three-parameter signature is explicitly rejected (`TypeError` -- it is a leftover of the njit era with no migration channel). Callbacks return `0` (or `RESULT_CONTINUE`) to continue; a non-zero value (or `RESULT_STOP`) stops the simulation.

All four lifecycle events (`first`, `early`, `late`, and `finish`) execute in the native Rust session. The former Python CSR executor, samplers, and low-level execution exports are removed; only compiled `HookProgram` data and Python callback bridges remain.

`.hooks()` is the single registration entry point: call it in the build chain, or via `pop.update().hooks(...)` after construction.

## `Op` Operations

Common operations include:

- `Op.add`: Add individuals.
- `Op.subtract`: Remove individuals.
- `Op.scale`: Scale proportionally.
- `Op.set_count`: Set a target count.
- `Op.kill`: Handle by death probability.
- `Op.sample`: Sample without replacement.
- `Op.stop_if_*`: Stop the run when a condition holds. Includes:
  - `Op.stop_if_below`: stop when a genotype's count falls below a threshold.
  - `Op.stop_if_above`: stop when a genotype's count exceeds a threshold.
  - `Op.stop_if_zero`: stop when a genotype's count is zero.
  - `Op.stop_if_extinction`: stop when the total population is zero.
- `Op.set_param`: Schedule one ecology parameter on a tick plan (see below).
- `Op.convert`: One-to-one probabilistic zygote-type conversion (see below).

Think of them as "declarative transformations over the state tensor".

### `Op.set_param`: code-free parameter scheduling

`Op.set_param(param, value, every=1, start=0, when=None)` rewrites one runtime-mutable ecology scalar on a tick schedule:

```python
nt.Op.set_param("carrying_capacity", "K * 0.95", every=10)
```

- `value` is an arithmetic expression (compiled to RPN): operands are jsonc parameter names (`K` is the registered alias of `carrying_capacity`) or numeric literals; operators are `+ - * /` with parentheses. A plain number is sugar for a constant. The expression is evaluated **against the current values every firing tick**, so `"K * 0.95"` compounds.
- `every` / `start` control the firing plan: `tick >= start and (tick - start) % every == 0`; `when` adds an extra condition.
- The `event` argument defaults to `early`.

- Targets are exactly these **5 ecology parameters** (the same ecology scalars `ctx.params` writes by attribute and the Rust session holds as columns):

| Parameter | Notes |
|---|---|
| `carrying_capacity` | K |
| `eggs_per_female` | Eggs per female |
| `sex_ratio` | Sex ratio |
| `sperm_displacement_rate` | Sperm displacement rate |
| `low_density_growth_rate` | Low-density growth rate |

Vector and genetics-tensor parameters raise `ValueError` -- use `pop.update()` / `pop.params.tensor_write()` for those.

**Write semantics**:

- Out of a run, the write flushes through the same channel as `pop.params.<name> = ...` (route dispatch, session refresh, and the parameter snapshot log).
- Inside a `run()`, the write evolves within the session-owned ecology columns with the same event granularity and the same jsonc bounds (a non-finite or out-of-bounds value such as `"K / 0"` raises `ValueError` mid-run). Each committed transition is written at its event boundary to the native `ParameterLog`, and later reads obtain the current values from the session snapshot.

### `Op.convert`: one-to-one probabilistic conversion

`Op.convert(source, target, probability, when=None)` moves each individual currently in `source` to `target` with `probability`, independently per individual. Both patterns must each match **exactly one** ZType, otherwise `ValueError` at compile time.

- **Males**: only `individual_count` rows migrate (males carry no sperm label).
- **Females (age-structured)**: the virgin part and *every sperm bucket* `(female_z, male_z)` are binomially sampled and moved atomically to `(target_z, male_z)` -- the stored sperm genotype label follows the female row, the male axis is untouched. Totals are conserved exactly in deterministic mode and in expectation in stochastic mode.
- **Discrete-generation**: no sperm storage, so the op degenerates to plain per-individual binomial migration.

The canonical idiom uses a `probability=1.0` remainder step to absorb whatever the chain left over:

```python
# 30% of A|A become A|a, the rest become a|a
nt.Op.convert("A|A", "A|a", probability=0.3),
nt.Op.convert("A|A", "a|a", probability=1.0),
```

Multiple `convert` ops run in hook-priority order.

## Stochastics

When a Declarative Hook operation kills individuals (fewer individuals than before), the system samples -- depending on configuration -- to decide which individuals survive.

Declarative `Op` operations automatically choose their execution strategy from the `stochastic` setting at population creation (`setup`):

| Setting | `Op.scale` / `Op.set_count` / `Op.subtract` | `Op.kill` |
|------|--------------------------------|---------|
| `stochastic=True` | Binomial random sampling | Binomial survival per individual |
| `stochastic=False` | Deterministic scaling (direct multiply) | Deterministic scaling (multiply by survival probability) |

When `stochastic=True`, the `continuous_sampling` setting adds a second axis:

- `continuous_sampling=True`: continuous sampling (moment-matched Beta/Gamma distributions instead of binomial/Poisson).
- `continuous_sampling=False`: discrete sampling.

The advantage of declarative hooks: write the rule once with the same Op syntax, and the system switches between deterministic and stochastic modes automatically without touching the hook code.

## Condition Expressions (`when`)

`when` controls when an operation is active. Common forms:

- `tick == N`
- `tick % N == 0`
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

`and`, `or`, `not` and parentheses are supported.

Examples:

```python
when="tick >= 10 and tick < 50"
when="tick % 7 == 0 and not (tick == 14)"
```

## Registering Multiple Hooks

`.hooks()` accepts multiple hook functions:

```python
import natal as nt

@nt.hook(event="first", priority=10)
def release_hook():
    return [nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def culling_hook():
    return [nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.95, when="tick > 50")]

@nt.hook(event="late", priority=0)
def stop_hook():
    return [nt.Op.stop_if_above(genotypes="Var|WT", threshold=5000)]

pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(release_hook, culling_hook, stop_hook)
    .build()
)

pop.run(n_steps=100, record_every=10)
```

When multiple hooks exist, make the execution order explicit with `priority` to avoid implicit-order reproducibility issues.

## Execution Path

The native Rust engine is the only execution backend, so hooks have a single execution path:

- Declarative `Op`s compile into a CSR program (contiguous arrays + offset table) executed in event order inside the Rust session.
- Single-parameter callbacks (`TickContext`) cross the Python<->Rust boundary at event boundaries; each invocation gets its own context wrapper, and its writes join that invocation's event transaction — committed on success, discarded on failure.
- Within one event, declarative ops and Python callbacks interleave in one cross-type `priority` order (lower values first; ties keep registration order). The two kinds share a single comparable scale: whichever hook — callback or declarative — has the smaller `priority` always runs first, and later hooks see earlier writes.

Hooks are "Op is a hook": `Op` objects constitute the hook program, and a declarative `@hook` function is just the compiler entry point returning the Op list. There is no `initialize` event -- express initialization logic with the first tick of the `first` event (`when="tick == 1"`) or with the `finish` event.

In `SpatialPopulation`, local-hook `priority` only applies within a deme; no global order is defined across demes. See [Spatial Simulation](3_spatial_simulation.md).

## Relationship with `run` / `run_tick`

Hooks execute automatically in event order inside `run(...)` and `run_tick()`.

So users typically do not need to trigger hooks manually; just:

1. Define the hook.
2. Register it with `.hooks()` in the chain API.
3. Run the simulation normally.

## Minimal Example

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

@nt.hook(event="first", priority=0)
def release():
    return [nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def stop_if_no_female():
    return [nt.Op.stop_if_zero(sex="female")]

pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(release, stop_if_no_female)
    .build()
)

pop.run(n_steps=200, record_every=10)
```

## Single-Parameter Callback Hook (TickContext)

Use the single-parameter callback shape when you need to read and write state arrays or runtime parameters directly. The parameter is a `TickContext` with this public surface:

| Member | Type / semantics |
|---|---|
| `pop.tick` | Current simulation tick (read-only). |
| `pop.deme_id` | Deme index of this invocation (`0` panmictic, the live deme index under a SpatialPopulation, read-only). |
| `pop.state` | Writable state view (short-term loan; writes take effect immediately). |
| `pop.params` | Writable parameter surface (same writer stack as `pop.params`; attribute writes are bounds-validated and reach the draft, the live Rust session, and the parameter snapshot log). |
| `pop.blueprint` | Read-only dimensions, name catalogs, and engine switches (`n_sexes`, `n_ages`, `n_ztypes`, `ztype_names`, ...). |
| `pop.metrics` | On-demand metrics view (recomputed on every access). |
| `pop.rng` | Deterministic per-invocation random stream (derived from population slot, tick, deme, hook index; never touches global `numpy.random`). |
| `pop.update()` | Returns a runtime `Configurator` bound to the owning population (same syntax as the build chain). |
| `pop.stop()` / `pop.stop_requested` | Request/query run termination at the event boundary. |

```python
from natal.frontend.hooks.tick_context import TickContext

@nt.hook(event="early", priority=10)
def my_hook(pop: TickContext) -> int:
    pop.state.individual_count[1, :, :] += 100  # add 100 males
    pop.params.carrying_capacity = pop.params.carrying_capacity * 0.5
    if pop.tick > 50:
        pop.stop()
    return 0
```

For selector-based hooks and hook-side runtime parameter writes, see the [Advanced Hook Tutorial](3_advanced_hooks.md).

## Modifying Parameters Inside Hooks

See the [Advanced Hook Tutorial](3_advanced_hooks.md).

## Related Sections

- [Advanced Hook Tutorial](3_advanced_hooks.md)
- [Runtime Parameter Modification](3_runtime_modification.md)
- [Population Initialization](2_population_initialization.md)
- [Modifier Mechanism](3_modifiers.md)
- [Configurator API Reference](api/configurator.md)
- [Quickstart](1_quickstart.md)
