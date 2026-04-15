# `PopulationState` and `PopulationConfig`

This chapter introduces the two most critical data objects in NATAL simulations:

- `PopulationState` (and its discrete‑generation counterpart `DiscretePopulationState`): used to store the dynamic state during a simulation.
- `PopulationConfig`: used to store simulation parameters and genetic mappings; it is the configuration object read when running kernels.

Understanding these two objects helps you organise initialisation, execution, and result interpretation more reliably.

## 1. Building a Mental Model First

At the user level, model construction is usually done via the Builder or `setup(...).build()`. After construction, the framework internally produces:

```text
User input parameters
  → PopulationConfig (static configuration)
  → PopulationState / DiscretePopulationState (dynamic state)
  → run(...) / run_tick() continuously update the state
```

Think of it as:

- `PopulationConfig` answers “what are the model rules?”
- `PopulationState` answers “what is the current state of the system?”

## 2. `PopulationState`: State Object for the Age‑Structured Model

`PopulationState` is defined in `src/natal/population_state.py` and is essentially a `NamedTuple` container.

### 2.1 Field Structure

```python
class PopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_genotypes, n_genotypes)
```

Meaning of each field:

- `n_tick`: current time step.
- `individual_count`: main tensor of individual counts, indexed by “sex – age – genotype”.
- `sperm_storage`: expresses the sperm storage structure of females, indexed by “age – female genotype – male genotype”.

### 2.2 Why `NamedTuple`

This design balances “clear structure” and “numerical efficiency”:

- Scalar fields (e.g., `n_tick`) have stable semantics.
- Array fields can be updated in‑place, which is convenient for high‑frequency writes in the simulation loop.

### 2.3 Recommended Creation

```python
from natal.population_state import PopulationState

state = PopulationState.create(
    n_genotypes=6,
    n_sexes=2,
    n_ages=8,
    n_tick=0,
)
```

> Tip: You usually do not need to create the state manually; the framework creates and maintains it automatically during population initialisation.

## 3. `DiscretePopulationState`: State Object for the Discrete‑Generation Model

The discrete‑generation model uses `DiscretePopulationState`. It is also defined in `src/natal/population_state.py`.

### 3.1 Field Structure

```python
class DiscretePopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
```

Main differences from `PopulationState`:

- No `sperm_storage` field.
- State updates are maintained by the discrete‑generation workflow.
- In the current discrete‑generation implementation, the configuration is normalised to `n_ages=2`, `new_adult_age=1`.

### 3.2 Recommended Creation

```python
from natal.population_state import DiscretePopulationState

state = DiscretePopulationState.create(
    n_sexes=2,
    n_ages=2,
    n_genotypes=6,
    n_tick=0,
)
```

## 4. `PopulationConfig`: Model Rules and Mapping Configuration

`PopulationConfig` is defined in `src/natal/population_config.py`. It contains the fixed parameters and matrices needed to run the model.

### 4.1 Groups of Configuration Data

1. Dimensions and control parameters
  - `n_sexes`, `n_ages`, `n_genotypes`, `n_haploid_genotypes`, `n_glabs`
  - `is_stochastic`, `use_continuous_sampling`, `sex_ratio`
2. Age‑related parameters
  - `age_based_survival_rates`
  - `age_based_mating_rates`
  - `female_age_based_relative_fertility`
  - `age_based_relative_competition_strength`
3. Fitness parameters
  - `viability_fitness` (shape: `(n_sexes, n_ages, n_genotypes)`)
  - `fecundity_fitness` (shape: `(n_sexes, n_genotypes)`)
  - `sexual_selection_fitness` (shape: `(n_genotypes, n_genotypes)`)
4. Genetic mapping matrices
  - `genotype_to_gametes_map` (shape: `(n_sexes, n_genotypes, n_haploid_genotypes * n_glabs)`)
  - `gametes_to_zygote_map` (shape: `(n_hg*n_glabs, n_hg*n_glabs, n_genotypes)`)
5. Initial distributions and scaling parameters
  - `initial_individual_count`
  - `initial_sperm_storage`
  - `population_scale`, `base_carrying_capacity`, etc.

### 4.2 What to Pay Attention to When Using

For most users, the most common operation is reading the configuration rather than constructing it manually:

```python
cfg = pop.config
print(cfg.n_ages, cfg.n_genotypes)
print(cfg.viability_fitness.shape)
```

If you need to modify certain coefficients, first verify the dimensions and biological meaning of the target field before writing.

## 5. From Input to Execution: What the Compilation Steps Do

From the user’s perspective, the construction phase mainly accomplishes four things:

1. Parse inputs

- Convert genotype strings to internal objects.
- Normalise initial counts into fixed‑shape arrays.

2. Build genetic mappings

- Generate the genotype‑to‑gamete probability mapping.
- Generate the gamete‑combination‑to‑zygote probability mapping.

3. Assemble life history and fitness parameters

- Aggregate age‑dependent survival/mating/reproduction parameters.
- Write into viability / fecundity / sexual selection tensors.

4. Create the runnable object

- Bind `PopulationConfig`.
- Initialise the corresponding `PopulationState` or `DiscretePopulationState`.

## 6. How to Understand the Two Model Types

### 6.1 AgeStructuredPopulation

Suitable for scenarios where age classes are explicit and age‑dependent transitions and sperm storage need to be represented.

### 6.2 DiscreteGenerationPopulation

Suitable for non‑overlapping generation scenarios; the state structure is more compact and the workflow semantics are simpler.

## 7. Minimal Example: Inspecting state and config

```python
from natal.genetic_structures import Species
from natal.age_structured_population import AgeStructuredPopulation
from natal.discrete_generation_population import DiscreteGenerationPopulation

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

## 8. Practical Advice

1. Prefer using the Builder or `setup(...).build()` to generate objects; avoid manually assembling low‑level arrays.
2. Before modifying configuration arrays, first check the dimensions and then verify the biological interpretation.
3. In the discrete‑generation model, always organise input and analysis according to the semantics of age 0/1.

## 9. Translating State to Readable Dictionaries / JSON

To facilitate logging, communication between frontend and backend, and debugging, NATAL provides functions to translate state objects into human‑readable structures.

The relevant APIs are located in `natal.state_translation`:

- `population_state_to_dict` / `population_state_to_json`
- `discrete_population_state_to_dict` / `discrete_population_state_to_json`
- `population_to_readable_dict` / `population_to_readable_json`
- `population_history_to_readable_dict` / `population_history_to_readable_json`
- `population_to_observation_dict` / `population_to_observation_json`

Among them:

- The translation of `PopulationState` includes `individual_count` and `sperm_storage`.
- The translation of `DiscretePopulationState` includes `individual_count` (no `sperm_storage`).

Example:

```python
import natal as nt

# Assume pop is any already built population (age‑structured or discrete‑generation)
readable = nt.population_to_readable_dict(pop)
print(readable["state_type"], readable["tick"])

# JSON output (useful for persistence or transmission)
payload = nt.population_to_readable_json(pop, indent=2)
print(payload[:200])

# History output (converted from flattened snapshots)
hist_view = nt.population_history_to_readable_dict(pop)
print(hist_view["n_snapshots"], hist_view["snapshots"][-1]["tick"])
```

If you want to directly apply observation rules during translation (see [Population Observation Rules](observation_rules.md)), you can use the observation integration interface:

```python
observed = nt.population_to_observation_dict(
    pop,
    groups={
        "adult_wt_female": {
            "genotype": ["WT|WT"],
            "sex": "female",
            "age": [1],
        }
    },
    collapse_age=False,
)
print(observed["observed"]["adult_wt_female"])
```

If you are directly working with `PopulationState` / `DiscretePopulationState`, you can also call the corresponding functions and explicitly pass labels:

```python
from natal.state_translation import population_state_to_dict

data = population_state_to_dict(
    state,
    genotype_labels=["WT|WT", "WT|Drive", "Drive|Drive"],
    sex_labels=["female", "male"],
)
```

---

## Related Chapters

- [Builder System](builder_system.md)
- [Deep Dive into Simulation Kernels](simulation_kernels.md)
- [Modifier Mechanism](modifiers.md)
- [Hook System](hooks.md)
- [Population Observation Rules](observation_rules.md)
