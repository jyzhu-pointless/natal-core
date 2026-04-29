# `PopulationState` and `PopulationConfig`

`PopulationState` and `PopulationConfig` are the two most critical data objects in the NATAL simulation framework:

- `PopulationState` (and its discrete-generation counterpart `DiscretePopulationState`) is responsible for maintaining the dynamic state during simulation
- `PopulationConfig` is responsible for storing simulation parameters and genetic mappings, serving as the configuration object read by kernels at runtime

Understanding these two objects helps in organizing initialization, execution, and result interpretation more reliably.

## Overview

After the user constructs a population via the Builder or `setup(...).build()`, the framework internally follows this flow:

```text
User input parameters
  → PopulationConfig (static configuration)
  → PopulationState / DiscretePopulationState (dynamic state)
  → run(...) / run_tick() continuously updates state
```

This can be understood as:

- `PopulationConfig` answers "what are the model rules"
- `PopulationState` answers "what is the current system state"

## `PopulationState`: The State Object for Age-Structured Models

`PopulationState` is defined in `src/natal/population_state.py` and is essentially a `NamedTuple` container.

### Field Structure

```python
class PopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_genotypes, n_genotypes)
```

Field descriptions:

- `n_tick`: Current time step
- `individual_count`: Individual count tensor, indexed by "sex-age-genotype"
- `sperm_storage`: Structure for female sperm storage, indexed by "age-female genotype-male genotype"

## `DiscretePopulationState`: The State Object for Discrete-Generation Models

The discrete-generation model uses `DiscretePopulationState`, also defined in `src/natal/population_state.py`.

### Field Structure

```python
class DiscretePopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
```

Key differences from `PopulationState`:

- Does not include the `sperm_storage` field
- State updates are managed by the discrete-generation flow
- In the current discrete-generation implementation, the configuration is normalized to `n_ages=2`, `new_adult_age=1`

## `PopulationConfig`: Model Rules and Mapping Configuration

`PopulationConfig` is defined in `src/natal/population_config.py` and contains the fixed parameters and matrices required to run the model.

### Configuration Groups

1. **Dimensions and Control Parameters**
  - `n_sexes`, `n_ages`, `n_genotypes`, `n_haploid_genotypes`, `n_glabs`
  - `is_stochastic`, `use_continuous_sampling`, `sex_ratio`

2. **Age-Related Parameters**
  - `age_based_survival_rates`
  - `age_based_mating_rates`
  - `female_age_based_relative_fertility`
  - `age_based_relative_competition_strength`

3. **Fitness Parameters**
  - `viability_fitness` (shape: `(n_sexes, n_ages, n_genotypes)`)
  - `fecundity_fitness` (shape: `(n_sexes, n_genotypes)`)
  - `sexual_selection_fitness` (shape: `(n_genotypes, n_genotypes)`)

4. **Genetic Mapping Matrices**
  - `genotype_to_gametes_map` (shape: `(n_sexes, n_genotypes, n_haploid_genotypes * n_glabs)`)
  - `gametes_to_zygote_map` (shape: `(n_hg*n_glabs, n_hg*n_glabs, n_genotypes)`)

5. **Initial Distribution and Scaling Parameters**
  - `initial_individual_count`
  - `initial_sperm_storage`
  - `population_scale`, `base_carrying_capacity`, etc.

### What to Pay Attention to When Using

`PopulationConfig` is a **static object** containing all fixed parameters and genetic mapping matrices for the model. **It cannot and should not be modified during simulation.**

You can print the field values of `PopulationConfig` to confirm that the model parameters match expectations:

```python
cfg = pop.config
print(cfg.n_ages, cfg.n_genotypes)
print(cfg.viability_fitness.shape)
```

## Minimal Example: Inspecting State and Config

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

<!--TODO: may need to introduce history; need to introduce main methods of Population object-->

## Translating State to Readable Dict/JSON

For logging, frontend-backend communication, and debugging, NATAL provides the ability to translate state objects into human-readable structures.

The relevant API is located in `natal.state_translation`:

- `population_state_to_dict` / `population_state_to_json`
- `discrete_population_state_to_dict` / `discrete_population_state_to_json`
- `population_to_readable_dict` / `population_to_readable_json`
- `population_history_to_readable_dict` / `population_history_to_readable_json`
- `population_to_observation_dict` / `population_to_observation_json`

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

If you need to apply observation rules directly during translation (see [Population Observation Rules](2_data_output.md)), use the observation integration interface:

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

If directly working with `PopulationState` / `DiscretePopulationState`, you can also call the corresponding functions and explicitly pass labels:

```python
from natal.state_translation import population_state_to_dict

data = population_state_to_dict(
    state,
    genotype_labels=["WT|WT", "WT|Drive", "Drive|Drive"],
    sex_labels=["female", "male"],
)
```

---

## Related Sections

- [Population Initialization](2_population_initialization.md)
- [Simulation Kernels Deep Dive](4_simulation_kernels.md)
- [Modifier Mechanism](3_modifiers.md)
- [Hook System](2_hooks.md)
- [Population Observation Rules](2_data_output.md)
