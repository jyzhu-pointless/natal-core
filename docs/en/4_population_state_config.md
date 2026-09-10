# `PopulationState` and `ModelDraft`

`PopulationState` and `ModelDraft` are the two most critical data objects in the NATAL simulation framework:

- `PopulationState` (and its discrete-generation counterpart `DiscretePopulationState`) is responsible for maintaining the dynamic state during simulation
- `ModelDraft` is responsible for storing simulation parameters and genetic mappings, serving as the configuration object read by engine at runtime

Understanding these two objects helps in organizing initialization, execution, and result interpretation more reliably.

`ModelDraft` is materialized into owned `Blueprint` and `Params` contracts at build time. Runtime parameter refreshes use the Params projection directly. The retired `hook_slot`, `SimState`, and `PlainPopulationState` compatibility surfaces are no longer part of the public API.

## Overview

After the user constructs a population via `setup(...).build()`, the framework internally follows this flow:

```text
User input parameters
  → ModelDraft (static configuration)
  → PopulationState / DiscretePopulationState (dynamic state)
  → run(...) / run_tick() continuously updates state
```

This can be understood as:

- `ModelDraft` answers "what are the model rules"
- `PopulationState` answers "what is the current system state"

## `PopulationState`: The State Object for Age-Structured Models

`PopulationState` is defined in `src/natal/frontend/data/state.py` and is essentially a `NamedTuple` container.

### Field Structure

```python
class PopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_ztypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_ztypes, n_ztypes)
```

Field descriptions:

- `n_tick`: Current time step
- `individual_count`: Individual count tensor, indexed by "sex-age-zygote type"
- `sperm_storage`: Structure for female sperm storage, indexed by "age-female ztype-male ztype"

## `DiscretePopulationState`: The State Object for Discrete-Generation Models

The discrete-generation model uses `DiscretePopulationState`, also defined in `src/natal/frontend/data/state.py`.

### Field Structure

```python
class DiscretePopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_ztypes)
```

Key differences from `PopulationState`:

- Does not include the `sperm_storage` field
- State updates are managed by the discrete-generation flow
- In the current discrete-generation implementation, the configuration is normalized to `n_ages=2`, `new_adult_age=1`

## `ModelDraft`: Model Rules and Mapping Configuration

`ModelDraft` is defined in `src/natal/frontend/model/draft.py` and contains the fixed parameters and matrices required to run the model.

### Configuration Groups

1. **Dimensions and Control Parameters**
  - `n_sexes`, `n_ages`, `n_ztypes`, `n_gtypes`, `n_glabs`, `n_slabs`
  - `stochastic`, `continuous_sampling`, `sex_ratio`

2. **Age-Related Parameters**
  - `age_based_survival_rates`
  - `age_based_mating_rates`
  - `female_age_based_fertility`
  - `age_based_relative_competition_strength`

3. **Fitness Parameters**
  - `viability_fitness` (shape: `(n_sexes, n_ages, n_ztypes)`)
  - `fecundity_fitness` (shape: `(n_sexes, n_ztypes)`)
  - `sexual_selection_fitness` (shape: `(n_ztypes, n_ztypes)`)

4. **Genetic Mapping Matrices**
  - `zygotes_to_gametes_map` (shape: `(n_sexes, n_ztypes, n_gtypes)`)
  - `gametes_to_zygotes_map` (shape: `(n_gtypes, n_gtypes, n_ztypes)`)

5. **Initial Distribution and Scaling Parameters**
  - `initial_individual_count`
  - `initial_sperm_storage`

### Draft Representation Versus the Runtime Contract

`ModelDraft` is a `NamedTuple` whose topology (which fields exist and their shapes) is **immutable** after construction. Two distinct uses matter:

- **The authoritative runtime data lives in the Rust session.** Every access to `pop.config` returns a **detached query snapshot** (`pop.config is pop.config` is False): field values are projected from the session and arrays are copies. Mutating the snapshot — scalar assignment, `set_param(pop.config, ...)`, or `snapshot.viability_fitness[...] = x` — never writes back to the population.
- **Runtime updates go through controlled channels.** Use `pop.params.<name>` or `pop.update()` for ecology scalars, and `pop.params.tensor_write(name, values)` for vectors and tensors. Inside a callback the same surfaces are `ctx.params` / `ctx.update()`, whose writes join the event transaction: they commit when the callback succeeds and are discarded when it fails.

```python
@nt.hook(event="early")
def heatwave(ctx: TickContext) -> int:
    if ctx.tick == 10:
        ctx.params.carrying_capacity = 2000.0  # jsonc bounds-checked; later stages of the tick see it
    return 0
```

```python
# Query: a snapshot, for reads and diagnostics
cfg = pop.config
print(cfg.n_ages, cfg.n_ztypes)
print(cfg.viability_fitness.shape)

# Write: a controlled channel
pop.params.carrying_capacity = 8000.0
pop.params.tensor_write("viability_fitness", new_table)
```

## Minimal Example: Inspecting State and Config

```python
from natal.frontend.genetics import Species
from natal.frontend.population import AgeStructuredPopulation
from natal.frontend.population import DiscreteGenerationPopulation

sp = Species.from_dict(name="Demo", structure={"chr1": {"A": ["A1", "A2"]}})

age_pop = (
    AgeStructuredPopulation
    .setup(sp, stochastic=False)
    .age_structure(n_ages=4, new_adult_age=2)
    .build()
)

dis_pop = (
    DiscreteGenerationPopulation
    .setup(sp, stochastic=False)
    .build()
)

print(type(age_pop.state).__name__)  # PopulationState
print(type(dis_pop.state).__name__)  # DiscretePopulationState

print(age_pop.config.n_ages, age_pop.config.new_adult_age)  # 4, 2
print(dis_pop.config.n_ages, dis_pop.config.new_adult_age)  # 2, 1
```

## Translating State to Readable Dict/JSON

For logging, frontend-backend communication, and debugging, NATAL provides the ability to translate state objects into human-readable structures.

The relevant API is located in `natal.frontend.output`:

- `population_state_to_dict` / `population_state_to_json`
- `discrete_population_state_to_dict` / `discrete_population_state_to_json`
- `population_to_readable_dict` / `population_to_readable_json`
- `population_history_to_readable_dict` / `population_history_to_readable_json`

Where:

- `PopulationState` translation results include `individual_count` and `sperm_storage`
- `DiscretePopulationState` translation results include `individual_count` (no `sperm_storage`)

Example:

```python
import natal as nt

# Assume pop is any constructed population (age-structured or discrete-generation)
readable = nt.population_to_readable_dict(pop)
print(readable["state_type"], readable["tick"])

# JSON output (for persistence or transmission)
payload = nt.population_to_readable_json(pop, indent=2)
print(payload[:200])

# History output (converted from flat snapshots)
hist_view = nt.population_history_to_readable_dict(pop)
print(hist_view["n_snapshots"], hist_view["snapshots"][-1]["tick"])
```

If you need to apply observation rules, use `pop.observe()` for the current state or `pop.history.observe(pop.observation)` for recorded history (see [Population Observation Rules](2_data_output.md)):

```python
# Project current state through the canonical observation
result = pop.observe()
print("Observation axes:", result.axes)
print("Observation values:", result.values)
```

If directly working with `PopulationState` / `DiscretePopulationState`, you can also call the corresponding functions and explicitly pass labels:

```python
from natal.frontend.output import population_state_to_dict

data = population_state_to_dict(
    state,
    genotype_labels=["WT|WT", "WT|Drive", "Drive|Drive"],
    sex_labels=["female", "male"],
)
```

---

## Related Sections

- [Population Initialization](2_population_initialization.md)
- [the Simulation Engine Deep Dive](4_simulation_engine.md)
- [Modifier Mechanism](3_modifiers.md)
- [Hook System](2_hooks.md)
- [Population Observation Rules](2_data_output.md)
