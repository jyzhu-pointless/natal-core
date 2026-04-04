# Hook System

Hooks are used to insert user logic at fixed points in the simulation flow.

If you want to perform “additional operations” at a certain stage of each tick, such as periodic releases, conditional interventions, or threshold termination, Hooks are the most direct way.

## 1. Value of Using Hooks

Hooks allow you to extend model behaviour without modifying the core kernel code. They are suitable for:

1. Periodic interventions (e.g., releasing individuals every few steps).
2. Conditional control (e.g., supplementing when numbers fall below a threshold).
3. Study flow control (e.g., early termination when a condition is met).

## 2. Event Time Points

NATAL provides four standard events:

- `initialization`: after simulation initialisation, before the first tick.
- `first`: early stage of each tick.
- `early`: after reproduction, before survival.
- `late`: after survival, before aging.
- `finish`: at the end of the simulation.

`initialization` is suitable for one‑time startup logic, for example:

1. Setting initial boundary conditions or thresholds.
2. Pre‑recording experimental metadata.
3. Performing a state correction or check before the first tick.

When choosing an event, clarify whether your intervention should happen “before initialisation”, “before survival”, or “after survival” – this significantly affects the interpretation of results.

## 3. Recommended Style: Declarative Hooks

For most users, it is recommended to use `@hook` together with `Op.*`, and then bind the hook to a population using `set_hook(...)`:

```python
from natal.hook_dsl import hook, Op


@hook(event="first", priority=10)
def periodic_release():
    return [
        Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0"),
        Op.scale(genotypes="WT|WT", ages="*", factor=0.98),
    ]


pop.set_hook("first", periodic_release)
```

This style is highly readable, low‑maintenance, and makes it easier for teams to review model rules.

## 4. Intuitive Understanding of `Op` Operations

Common operations include:

- `Op.add`: increase individual counts.
- `Op.subtract`: decrease individual counts.
- `Op.scale`: scale counts by a factor.
- `Op.set_count`: set a target count.
- `Op.kill`: kill individuals with a given probability.
- `Op.stop_if_*`: stop the simulation when a condition is met.

Think of them as “declarative transformations” applied to the state tensor.

## 5. Condition Expressions (`when`)

`when` controls when an operation takes effect. Common forms:

- `tick == N`
- `tick % N == 0`
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

And supports `and`, `or`, `not`, and parentheses.

Examples:

```python
when="tick >= 10 and tick < 50"
when="tick % 7 == 0 and not (tick == 14)"
```

## 6. Selector Syntax (Selecting Targets)

When you want to first select a target (e.g., a specific genotype) and then apply logic, you can use the selector pattern:

```python
from natal.hook_dsl import hook


@hook(event="late", selectors={"target_gt": "Drive|WT"})
def cap_target(pop, target_gt):
    arr = pop.state.individual_count
    if arr[:, :, target_gt].sum() > 5000:
        arr[:, :, target_gt] *= 0.95
```

This style is suitable for scenarios where you need to read the current state and apply conditional logic.

## 7. Registration Methods

### 7.1 Recommended: `set_hook`

```python
pop.set_hook("first", my_hook)
```

### 7.2 `register` Syntax as a Complement

```python
my_hook.register(pop)
```

If multiple hooks exist, it is recommended to specify execution order via `priority` to avoid implicit ordering that makes results hard to reproduce. `register` is mainly a convenience method provided by the `@hook` decorator; the core binding method remains `set_hook(...)`.

## 8. Relationship with `run` / `run_tick`

Hooks are automatically executed in event order when calling `run(...)` or `run_tick()`.

Therefore, users generally do not need to trigger hooks manually; you only need to:

1. Define the hook.
2. Bind it with `set_hook(...)`.
3. Run the simulation normally.

## 9. Practical Advice

1. Keep each hook single‑responsibility for easier verification.
2. Run short simulations to confirm behaviour before extending to long runs.
3. When multiple hooks interact, record their priorities and expected order.
4. For important experiments, save the configuration and hook definitions to support reproducibility.

## 10. Minimal Combined Example

```python
from natal.hook_dsl import hook, Op


@hook(event="initialization", priority=0)
def init_population():
    return [Op.stop_if_above(genotypes="Drive|WT", ages="*", threshold=1_000_000)]


@hook(event="first", priority=0)
def release():
    return [Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]


@hook(event="late", priority=5)
def cap(pop):
    # Additional state checks can be placed here
    return [Op.stop_if_above(genotypes="Drive|WT", ages="*", threshold=10000)]

# Assume a population `pop` has already been built via the builder

pop.set_hook("initialization", init_population)
pop.set_hook("first", release)
pop.set_hook("late", cap)

pop.run(n_steps=200, record_every=10)
```

## 11. Chapter Summary

The Hook system provides a stable, extensible way to inject behaviour into NATAL.

At the user level, follow three steps:

1. Define rules with `@hook`.
2. Prefer binding to the population using `set_hook(...)`.
3. If you still want to use the convenient `register` syntax from `@hook`, you can also use `register(...)`.
4. Execute with `run(...)` and verify results using the history.

---

## Related Chapters

- [Modifier Mechanism](modifiers.md)
- [Deep Dive into Simulation Kernels](simulation_kernels.md)
- [Numba Optimization Guide](numba_optimization.md)
- [Quick Start](quickstart.md)
