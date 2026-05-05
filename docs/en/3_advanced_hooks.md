# Advanced Hook Tutorial

The [Basic Tutorial](2_hooks.md) introduced declarative Hooks (`Op.add`, `Op.scale`, etc.), which are suitable for most common scenarios.
When you need to directly manipulate NumPy arrays for more flexible state modifications (e.g., conditional branching, loops, custom calculations),
you can use Custom Hooks or Selector-based Hooks.

## Custom Hooks

Custom Hooks allow you to directly write code to manipulate the simulation state, executed after Numba compilation.

### Basic Usage

```python
from natal.hooks import hook


@hook(event="late", priority=10)
def custom_release_hook(ind_count, tick, deme_id=-1):
    # ind_count is a NumPy array of individual counts
    # Shape: (sex, age, genotype)
    # sex=0 corresponds to female, sex=1 corresponds to male

    # Release 100 individuals every 10 ticks
    if tick % 10 == 0:
        # Assume the genotype index for Drive|WT is 1
        ind_count[:, :, 1] += 100

    return 0  # 0 means continue simulation
```

### Function Signature

Custom Hooks support two function signatures:

- `(ind_count, tick)` — does not receive deme information (simplified signature)
- `(ind_count, tick, deme_id=-1)` — receives the deme index; the actual index is passed when running across demes in a spatial population

The default value of `deme_id` is `-1`. When registered in a `SpatialPopulation`, the system automatically passes the current deme index;
when registered in a non-spatial population, this parameter can be omitted.

### Array Indexing Notes

The dimension order of `ind_count` is `(sex, age, genotype)`:

- `sex=0` corresponds to FEMALE, `sex=1` corresponds to MALE
- In Numba mode, `Sex.MALE` and similar enum types cannot be used directly for indexing; integer values or `.value` must be used

```python
# Correct approach
male_count = ind_count[1, :, :].sum()
female_count = ind_count[0, :, :].sum()

# Or using .value
male_count = ind_count[Sex.MALE.value, :, :].sum()
```

### Complete Example

```python
from natal.hooks import hook


@hook(event="early", priority=5)
def custom_culling_hook(ind_count, tick, deme_id=-1):
    # Selective culling of specific genotypes
    if tick > 50:
        # The genotype index for WT|WT is 0
        wt_wt_count = ind_count[:, :, 0].sum()
        if wt_wt_count > 10000:
            ind_count[:, :, 0] = ind_count[:, :, 0] * 0.9

    return 0
```

## Selector-based Hooks

Selector-based Hooks allow you to specify target genotypes by symbolic name (e.g. ``"Drive|WT"``).
The framework resolves symbols to integer indices at registration time and bakes them into
the compiled code.

### Basic Usage

```python
from natal.hooks import hook


@hook(event="late", selectors={"target_gt": "Drive|WT"}, priority=10)
def cap_target(ind_count, tick, target_gt):
    # target_gt is the genotype index (integer) resolved by the selector
    if tick % 10 == 0:
        ind_count[:, :, target_gt] *= 0.95
```

### Selector Resolution Rules

`selectors` values support the following types:

| Type | Example | Resolution |
|------|---------|-----------|
| `str` (genotype label) | `"WT\|WT"` | single index |
| `str` (wildcard) | `"*"` | all genotype indices |
| `int` | `3` | used directly |
| `range` | `range(3)` | `[0, 1, 2]` |
| `list` / `tuple` | `["WT\|Dr", 4]` | multiple indices |
| `Genotype` object | `species.genotypes[0]` | corresponding index |

> **Note**: Selectors use `IndexRegistry.resolve_genotype_index()` for exact string
> matching. Pattern syntax (`::`, `|*`, etc.) is not supported. Use `GenotypeSelector`
> to pre-resolve patterns to index arrays if needed.

### Parameter Passing Mode

Selector Hooks support three parameter passing modes via the `mode` argument:

| mode | Behavior | Example signature |
|------|---------|-------------------|
| `"auto"` (default) | Auto-detect: pack if param name not in keys | See below |
| `"expand"` | Each selector as an individual keyword argument | `fn(ind_count, tick, a, b)` |
| `"aggregate"` | All selectors packed into a single namedtuple | `fn(ind_count, tick, ctx)` |

#### mode="expand"

Each selector key becomes a separate function parameter:

```python
@hook(event="early", selectors={"drive": "Dr|WT", "wt": "WT|WT"}, mode="expand")
def balance_population(ind_count, tick, drive, wt):
    # drive and wt are both int (genotype indices)
    drive_count = ind_count[:, :, drive].sum()
    wt_count = ind_count[:, :, wt].sum()

    if drive_count > wt_count * 2:
        ind_count[:, :, drive] *= 0.8

    return 0
```

#### mode="aggregate"

All selectors are packed into a namedtuple, accessed via attributes:

```python
@hook(event="early", selectors={"drive": "Dr|WT", "wt": "WT|WT"}, mode="aggregate")
def balance_population(ind_count, tick, sel):
    # sel.drive and sel.wt are both int (genotype indices)
    # namedtuple attribute access is fully supported in Numba
    drive_count = ind_count[:, :, sel.drive].sum()
    wt_count = ind_count[:, :, sel.wt].sum()

    if drive_count > wt_count * 2:
        ind_count[:, :, sel.drive] *= 0.8

    return 0
```

**Advantage**: Adding a new selector only requires updating the `selectors` dict,
not the function signature. The parameter name can be any valid identifier
(e.g. `sel`, `ctx`, `params`).

#### mode="auto" (default)

When no `mode` is specified, the framework detects the style from the function
signature:

- Extra positional parameter name **not in** selector keys → aggregate mode
- Extra positional parameter name **in** selector keys → expand mode

```python
# "ctx" ∉ {"drive", "wt"} → auto aggregate
@hook(event="early", selectors={"drive": "Dr|WT", "wt": "WT|WT"})
def hook_agg(ind_count, tick, ctx):
    ctx.drive, ctx.wt  # namedtuple attributes

# "wt" ∈ {"wt"} → auto expand (backward compatible)
@hook(event="early", selectors={"wt": "WT|WT"})
def hook_exp(ind_count, tick, wt):
    ...  # wt is a plain int parameter
```

### Using deme_id

All three modes support the `deme_id` parameter for spatial populations:

```python
# Expand + deme_id
@hook(event="early", selectors={"target": "Dr|Dr"}, mode="expand")
def hook1(ind_count, tick, deme_id, target):
    if deme_id == 0:
        ind_count[:, :, target] = 0

# Aggregate + deme_id
@hook(event="early", selectors={"target": "Dr|Dr"}, mode="aggregate")
def hook2(ind_count, tick, deme_id, sel):
    if deme_id == 0:
        ind_count[:, :, sel.target] = 0
```

### Features

- Selectors are resolved at registration time and baked into generated Numba code
- Aggregate mode uses `collections.namedtuple`; field access is natively supported in Numba
- Single-value selectors (e.g. `"WT|WT"`) are unboxed to `int`; multi-value selectors remain `np.ndarray[int32]`
- Suitable for scenarios requiring logic based on specific targets

## Numba-Compatible Random Sampling

When performing random sampling in custom Hooks, it is recommended to use functions provided by the `natal.numba_compat` module. These functions are optimized to remain efficient in both Numba mode and pure Python mode:

```python
from natal.numba_compat import (
    binomial,
    binomial_2d,
    continuous_binomial,
    continuous_multinomial,
    set_numba_seed,
)
from natal.hooks import hook


@hook(event="late", priority=10)
def stochastic_culling_hook(ind_count, tick, deme_id=-1):
    if tick > 50:
        # Use binomial distribution for random culling
        # Assume 10% culling probability for genotype 0
        n_current = ind_count[:, :, 0]
        survival_prob = 0.9

        # continuous_binomial is more efficient for large counts
        ind_count[:, :, 0] = continuous_binomial(n_current, survival_prob)

    return 0
```

### Main API

| Function | Description |
|----------|-------------|
| `binomial(n, p)` | Binomial distribution sampling, returns number of successes in n trials |
| `binomial_2d(n, p, n_rows, n_cols)` | Element-wise binomial distribution sampling on a 2D array |
| `continuous_binomial(n, p)` | Continuous binomial distribution, returns floating point (more efficient for large counts) |
| `continuous_multinomial(n, p_array, out_counts)` | Continuous multinomial distribution |
| `multinomial(n, pvals)` | Multinomial distribution sampling |
| `set_numba_seed(seed)` | Set random seed (ensures reproducibility) |
| `clamp01(x)` | Clamp value to range [0, 1] |

### Use Cases

- **Adding randomness after deterministic operations**: First scale deterministically, then add noise with random sampling
- **Conditional random culling**: Dynamically determine culling probability based on current state
- **Batch sampling operations**: Use `binomial_2d` for batch sampling across entire arrays

```python
@hook(event="late", priority=10)
def age_specific_mortality(ind_count, tick, deme_id=-1):
    if tick % 10 == 0:
        # Apply different survival probabilities to each age group
        survival_rates = np.array([0.8, 0.9, 0.95, 0.98, 0.99])

        # Use binomial_2d for batch sampling
        n_ages = ind_count.shape[1]
        for age in range(n_ages):
            n_survivors = binomial_2d(
                ind_count[:, age, :],
                np.array([survival_rates[age]]),
                2,  # sex
                ind_count.shape[2]  # n_genotypes
            )
            ind_count[:, age, :] = n_survivors

    return 0
```

## Execution Mode and Compatibility

NATAL Core's Hook system automatically selects the execution path based on the global `NUMBA_ENABLED` switch:

| `NUMBA_ENABLED` | Custom Hook Behavior |
|----------------|---------------------|
| `True` (default) | Hook code must follow Numba syntax; the system automatically compiles with Numba |
| `False` | Hooks can use pure Python syntax, dispatched uniformly through `HookExecutor` |

### Why Numba Syntax Is Emphasized?

The framework enables Numba optimization by default, which means:

1. Custom Hooks are by default compiled with Numba via the `njit_switch` decorator
2. If the code contains unsupported Python features, it will error during registration or first execution
3. The performance advantage is particularly noticeable in large-scale simulations

### What If You Need to Use Hooks with Numba Disabled?

When `NUMBA_ENABLED=False`:

- All Hooks (declarative, selector, custom) are dispatched uniformly through `HookExecutor`
- The system executes all Hooks in order based on their `priority`
- No need to modify Hook definition code to switch execution paths

```python
from natal.numba_utils import numba_disabled

with numba_disabled():
    # In this context, NUMBA_ENABLED is False
    pop = builder.hooks(my_custom_hook).build()
    pop.run(n_steps=100)
```

## Mixing Different Types of Hooks

NATAL Core allows mixing different types of Hooks in the same event:

```python
from natal.hooks import hook, Op


# Declarative Hook: periodically release individuals
@hook(event="first", priority=10)
def release_hook():
    return [Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 10 == 0")]


# Selector-based Hook (aggregate mode): selector-driven operations
@hook(event="first", priority=7, selectors={"drive": "Drive|WT"}, mode="aggregate")
def check_drive_threshold(ind_count, tick, sel):
    drive_count = ind_count[:, :, sel.drive].sum()
    if drive_count > 10000:
        # Log or record state here
        pass
    return 0


# Custom Hook: efficient computation and state modification (deme_id=-1 is default for non-spatial)
@hook(event="first", priority=5)
def custom_process_hook(ind_count, tick, deme_id=-1):
    # Perform intensive computation
    for age in range(ind_count.shape[1]):
        ind_count[:, age, :] *= 0.99  # Slight mortality
    return 0


pop = builder.hooks(release_hook, check_drive_threshold, custom_process_hook).build()
```

### Execution Order

When mixing different types of Hooks:

- The system sorts Hooks by their `priority` value (smaller values have higher priority)
- Hooks with the same priority have an undefined execution order
- When `NUMBA_ENABLED=True`, selector and custom Hooks are merged into a single Numba function for execution

## Performance Comparison

| Hook Type | Performance | Flexibility | Readability | Use Cases |
|-----------|-------------|-------------|-------------|-----------|
| Declarative Hook | High | Medium | High | Most common scenarios |
| Selector-based Hook | High | High | Medium | Scenarios requiring logic based on specific targets |
| Custom Hook | Highest | High | Medium | Computationally intensive operations |

## Related Chapters

- [Hook System](2_hooks.md) - Basic Hook concepts and declarative Hook usage
- [Modifier Mechanism](3_modifiers.md) - Genetic modifier mechanism
- [Simulation Kernels Deep Dive](4_simulation_kernels.md) - How simulation kernels work
- [Numba Optimization Guide](4_numba_optimization.md) - Numba optimization techniques
