# Hook System

Hooks are used to insert user logic at fixed points in the simulation workflow.

If you want to perform "additional operations" at certain stages of each tick -- such as periodic releases, conditional interventions, or threshold termination -- Hooks are the most direct way to do so.

## Hook Timing

Hook timing includes:

- `initialization`: After simulation initialization completes, before entering the first tick.
- `first`: Early stage of each tick.
- `early`: After the reproduction step, before the survival step.
- `late`: After the survival step, before the aging step.
- `finish`: When the simulation ends.

Among these, `initialization` and `finish` are one-time events, while `first`, `early`, and `late` can be executed repeatedly across multiple ticks as needed.

When selecting an event, it is recommended to first clarify at which specific time point the intervention occurs, as this can significantly impact the interpretation of results.

## Declarative Hooks

For most users, it is recommended to use `@nt.hook` with `nt.Op.*`, registering in a chain on the population object:

```python
import natal as nt

@nt.hook(event="first", priority=10)
def periodic_release():
    return [
        nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0"),
        nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.98),
    ]


pop = (
    nt.AgeStructuredPopulation
    .setup(
        name="MyPop",
        stochastic=True,
        use_continuous_sampling=False
    )
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000, "Drive|WT": 0},
        "male": {"WT|WT": 1000, "Drive|WT": 0}
    })
    .survival(
        female_age_based_survival_rates=0.85,
        male_age_based_survival_rates=0.8
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

This approach offers high readability, low maintenance costs, and makes it easier for teams to review model rules.

## `Op` Operations

Common operations include:

- `Op.add`: Add individuals.
- `Op.subtract`: Remove individuals.
- `Op.scale`: Scale by a factor.
- `Op.set_count`: Set the target count.
- `Op.kill`: Process by death probability.
- `Op.sample`: Sample without replacement.
- `Op.stop_if_*`: Stop running when conditions are met. Includes:
  - `Op.stop_if_below`: Stop when the number of individuals of a specified genotype falls below a threshold.
  - `Op.stop_if_above`: Stop when the number of individuals of a specified genotype exceeds a threshold.
  - `Op.stop_if_zero`: Stop when the number of individuals of a specified genotype reaches zero.
  - `Op.stop_if_extinction`: Stop when the population size reaches zero.

Think of them as "declarative transformations on the state tensor."

## Stochasticity Handling

When individuals die as a result of Declarative Hook operations (the number of individuals becomes less than the original count), sampling may be performed based on the configuration to decide which individuals survive.

The `Op` operations in Declarative Hooks automatically select the execution method based on the `stochastic` configuration in `setup` (in the chain API) when the population is created:

| Configuration | `Op.scale` / `Op.set_count` / `Op.subtract` | `Op.kill` |
|--------------|---------------------------------------------|-----------|
| `stochastic=True` | Uses binomial distribution random sampling | Uses binomial distribution to determine each individual's survival |
| `stochastic=False` | Deterministic scaling (directly multiply by factor) | Deterministic scaling (multiply by survival probability) |

When `stochastic=True`, the sampling method can also be chosen via the `use_continuous_sampling` configuration:

- `use_continuous_sampling=True`: Uses continuous sampling (uses moment-matched Beta/Gamma distributions instead of Binomial/Poisson distributions)
- `use_continuous_sampling=False`: Uses discrete sampling

The advantage of Declarative Hooks is that you only need to write rules using the same Op syntax, and the system will automatically switch between deterministic and stochastic modes based on the configuration, without requiring Hook code modifications.

## Conditional Expressions (when)

`when` is used to control when an operation takes effect. Common expressions:

- `tick == N`
- `tick % N == 0`
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

Also supports `and`, `or`, `not` and parentheses combinations.

Examples:

```python
when="tick >= 10 and tick < 50"
when="tick % 7 == 0 and not (tick == 14)"
```

## Registering Multiple Hooks

The `.hooks()` method in the chain API supports passing multiple Hook functions:

```python
import natal as nt

@nt.hook(event="first", priority=10)
def release_hook():
    return [nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def culling_hook():
    return [nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.95, when="tick > 50")]

@nt.hook(event="late", priority=0)
def stop_hook():
    return [nt.Op.stop_if_above(genotypes="Drive|WT", threshold=5000)]

pop = (
    nt.AgeStructuredPopulation
    .setup(stochastic=True)
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

If multiple Hooks are present, it is recommended to use `priority` to explicitly specify the execution order, avoiding implicit ordering that could make results difficult to reproduce.

## Execution Modes

NATAL Core's Hook system supports two execution modes, controlled by the global `NUMBA_ENABLED` switch:

- **When `NUMBA_ENABLED=True` (default)**:
  - Declarative Hooks are compiled into pure data structures (CSR format) and executed efficiently inside Numba-compiled kernels
  - Custom Hooks and Selector-based Hooks must follow Numba syntax
  - Python-layer Hooks are rejected at registration time (except for `initialization` and `finish` events)

- **When `NUMBA_ENABLED=False`**:
  - Any registered Hook type (declarative CSR, njit, Python) will go through a unified Python event dispatch path in `run(...)` / `run_tick()`
  - In this path, the system executes all Hooks in order according to `priority`, without requiring manual triggering

When `NUMBA_ENABLED=True` globally, if declarative CSR, njit, and Python Hooks are mixed at the same event, the runtime will automatically switch to the unified Python event dispatch to ensure cross-type execution sorted by `priority`.

In `SpatialPopulation`, the `priority` of local Hooks only takes effect within a deme; no global order is defined between different demes.

## Relationship with `run` / `run_tick`

Hooks are automatically executed in event order during `run(...)` and `run_tick()`.

Therefore, users typically do not need to manually trigger Hooks; just:

1. Define the Hook.
2. Register it in the chain API using `.hooks()`.
3. Run the simulation normally.

## Minimal Example

```python
import natal as nt

@nt.hook(event="first", priority=0)
def release():
    return [nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def stop_if_no_female():
    return [nt.Op.stop_if_zero(sex="female", threshold=10000)]

pop = (
    nt.AgeStructuredPopulation
    .setup(stochastic=True)
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

## Related Sections

- [Advanced Hook Tutorial](3_advanced_hooks.md)
- [Population Initialization](2_population_initialization.md)
- [Modifier Mechanism](3_modifiers.md)
- [Simulation Kernels Deep Dive](4_simulation_kernels.md)
- [Numba Optimization Guide](4_numba_optimization.md)
- [Quick Start](1_quickstart.md)
