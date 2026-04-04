# Builder System Explained (Complete Parameter Reference + Simulation Flow Mapping)

This chapter aims to explain Builder thoroughly:

1. What each public parameter is and its default value.
2. Which stage of the simulation the parameter affects.
3. How to configure following the “first make it run, then fine‑tune” principle.

If you read only one chapter about Builder, read this one.

## 1. Where Builder Fits in the Simulation

Builder’s responsibility is not to run the simulation directly, but to compile high‑level inputs into `PopulationConfig` and `PopulationState`, then hand them over to the simulation execution flow.

The flow can be simplified as:

```text
Builder chained configuration
  -> build()
  -> PopulationConfig / PopulationState
  -> run_tick / run
  -> reproduction -> survival -> aging (and hooks)
```

Related chapters:

- [PopulationState & PopulationConfig: Compilation and Configuration](population_state_config.md)
- [Deep Dive into Simulation Kernels](simulation_kernels.md)

## 2. Two Types of Builder

- `AgeStructuredPopulationBuilder`
  - Suitable for multi‑age‑class models.
  - Typical features: `n_ages` is configurable, supports age‑vector inputs, optional sperm storage.
- `DiscreteGenerationPopulationBuilder`
  - Suitable for discrete‑generation models.
  - Typical features: uses two age classes by default (`n_ages=2`, `new_adult_age=1`).

## 3. AgeStructuredPopulationBuilder Parameter Reference

### 3.1 `setup(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `name` | `str` | `"AgeStructuredPop"` | whole simulation | Only for identification and logging, does not change dynamics. Name experiments explicitly. |
| `stochastic` | `bool` | `True` | sampling stages (reproduction/survival) | `True` for stochastic, `False` for deterministic. Use `False` during tuning. |
| `use_continuous_sampling` | `bool` | `False` | probability sampling details | Controls sampling strategy. Keep default for most scenarios. |
| `use_fixed_egg_count` | `bool` | `False` | reproduction | `True` fixes egg number, `False` mimics more random oviposition. |

### 3.2 `age_structure(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `n_ages` | `int` | `8` | whole simulation (array dimensions) | Number of age classes. Constrains length of initial state, survival vectors etc. |
| `new_adult_age` | `int` | `2` | reproduction/survival | Starting age index for adults. Match species life history. |
| `generation_time` | `Optional[int]` | `None` | compilation parameter | Marks generation time, can be used for model interpretation. Avoid conflict with age definition. |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | competition/initial scaling | Auxiliary equilibrium distribution. Use only when explicitly doing steady‑state scaling. |

### 3.3 `initial_state(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `individual_count` | `dict` | required | initial state | Format `{sex: {genotype: age_data}}`. Not setting it will cause `build()` to error. |
| `sperm_storage` | `Optional[dict]` | `None` | reproduction (if sperm storage enabled) | Initial sperm inventory. Only needed when `use_sperm_storage=True`. |

`age_data` supports: scalar, list/tuple/array, `{age: value}` mapping. Must be non‑negative.

Code‑aligned format (from `PopulationConfigBuilder.resolve_age_structured_initial_individual_count` and related parsing functions):

```python
# 1) Scalar: distributed to all ages in [new_adult_age, n_ages)
{"female": {"WT|WT": 100.0}}

# 2) Sequence: written by age index, truncated if too long, missing entries default to 0
{"female": {"WT|WT": [0, 100, 80, 60]}}

# 3) Mapping: explicitly give some ages
{"female": {"WT|WT": {2: 100, 3: 80}}}
```

`sperm_storage` format is a three‑level mapping:

```python
{
  "WT|WT": {                 # female genotype
    "Drive|WT": [0, 0, 20],  # male genotype -> age_data
  }
}
```

Validation rules (source behaviour):

- Age indices must be in `[0, n_ages)`.
- All counts must be `>= 0`.
- Genotype strings must be resolvable by the current `Species`.

### 3.4 `survival(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional[Any]` | `None` | survival | Supports scalar/sequence/mapping/callable. `None` uses default curve. |
| `male_age_based_survival_rates` | `Optional[Any]` | `None` | survival | Same as above. |
| `generation_time` | `Optional[int]` | `None` | compilation parameter | Same name as in `age_structure`; later call overrides earlier. |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | scaling helper | Same name as in `age_structure`; later call overrides earlier. |

Practical advice:

- Survival rates should be in `[0, 1]`.
- Start with a smooth curve, then add age‑specific spikes.

Code‑aligned format (from `_resolve_survival_param`):

```python
# A) None -> uses default curve
.survival(female_age_based_survival_rates=None)

# B) Scalar -> same value for all ages
.survival(female_age_based_survival_rates=0.85)

# C) Sequence -> written by age index; missing entries become 0, extra truncated
.survival(female_age_based_survival_rates=[1.0, 1.0, 0.9, 0.7])

# D) Sparse mapping -> unspecified ages default to 1.0
.survival(female_age_based_survival_rates={0: 1.0, 1: 0.95, 2: 0.8})

# E) Callable -> must accept one age argument and return a numeric value
.survival(female_age_based_survival_rates=lambda age: 1.0 if age < 2 else 0.8)

# F) Sequence ending with None sentinel -> trailing None are filled with the last non‑None value
.survival(female_age_based_survival_rates=[1.0, 0.9, None])
```

Validation rules (source behaviour):

- Survival rates must be non‑negative (source does not enforce an upper bound of 1.0, but modelling suggests ≤1.0).
- Dictionary keys must be valid ages.
- A callable with an invalid signature will raise `TypeError`.

### 3.5 `reproduction(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional[list/ndarray]` | `None` | reproduction | Female mating rates by age. |
| `male_age_based_mating_rates` | `Optional[list/ndarray]` | `None` | reproduction | Male mating rates by age. |
| `female_age_based_relative_fertility` | `Optional[list/ndarray]` | `None` | reproduction | Female relative fertility weights by age. |
| `eggs_per_female` | `float` | `50.0` | reproduction | Base egg production per female. Start with a neutral value. |
| `use_fixed_egg_count` | `bool` | `False` | reproduction | Fixed/random oviposition switch. |
| `sex_ratio` | `float` | `0.5` | reproduction | Proportion of female offspring. Usually 0.5. |
| `use_sperm_storage` | `bool` | `True` | reproduction | Enable sperm storage mechanism. |
| `sperm_displacement_rate` | `float` | `0.05` | reproduction | Intensity of new sperm replacing old sperm. |

Format and length requirements (source behaviour):

- `female_age_based_mating_rates` / `male_age_based_mating_rates` / `female_age_based_relative_fertility`
  - When provided, they are converted to `np.array`.
  - During configuration compilation, their length must equal `n_ages`, otherwise `ValueError` is raised.

### 3.6 `competition(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `competition_strength` | `float` | `5.0` | juvenile density regulation | Competition intensity factor. |
| `juvenile_growth_mode` | `int\|str` | `"logistic"` | juvenile density regulation | Supports string or constant. Common value: `logistic`. |
| `low_density_growth_rate` | `float` | `6.0` | juvenile density regulation | Low‑density growth rate; too high may cause oscillations. |
| `old_juvenile_carrying_capacity` | `Optional[int]` | `None` | juvenile density regulation | Takes precedence over derivation from `expected_num_adult_females`. |
| `expected_num_adult_females` | `Optional[int]` | `None` | capacity derivation | If carrying capacity not given, can be used with egg count to derive capacity. |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | scaling helper | Same name as before; later call overrides earlier. |

### 3.7 `presets(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `*preset_list` | `Any` (varargs) | empty | post‑build processing | Presets are applied first; subsequent `fitness`/`modifiers`/`hooks` can override them. |

### 3.8 `fitness(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `viability` | `Optional[dict]` | `None` | survival | Viability fitness mapping. Supports genotype‑specific, sex‑specific, age‑specific, or sex+age‑specific. |
| `fecundity` | `Optional[dict]` | `None` | reproduction | Fecundity fitness mapping. |
| `sexual_selection` | `Optional[dict]` | `None` | reproduction (mating preference) | Supports flat or nested mapping. |
| `mode` | `str` | `"replace"` | fitness writing strategy | `replace` overwrites, `multiply` scales multiplicatively. |

Code‑aligned format (from `fitness()` docstring and `_iter_sexual_selection_entries`):

```python
# viability: genotype -> float
.fitness(viability={"WT|WT": 1.0, "Drive|Drive": 0.6})

# viability: genotype -> {sex: float}
.fitness(viability={"Drive|WT": {"female": 0.9, "male": 0.8}})

# viability: genotype -> {age: float}, applies equally to both sexes
.fitness(viability={"Drive|WT": {0: 0.95, 1: 0.85}})

# viability: genotype -> {sex: {age: float}}, allows fine‑grained sex+age specification
.fitness(viability={"Drive|WT": {"female": {1: 0.9}, "male": {2: 0.8}}})

# fecundity: genotype -> float or {sex: float}
.fitness(fecundity={"Drive|Drive": 0.7})

# sexual_selection flat format: {male_selector: value}, automatically treats female='*'
.fitness(sexual_selection={"Drive|WT": 1.2, "WT|WT": 1.0})

# sexual_selection nested format: {female_selector: {male_selector: value}}
.fitness(sexual_selection={"WT|WT": {"Drive|WT": 0.8, "WT|WT": 1.0}})
```

`GenotypeSelector` supports:

- Single selector: `"Drive|WT"` or a `Genotype` object.
- Union of selectors: `("Drive|WT", "Drive|Drive")`.

Age key constraints for `viability` (consistent with source):

- Age keys must be integers in `[0, n_ages)`.
- If using `{sex: ...}` form, can further provide `{age: float}`.
- If directly using `{age: float}`, it applies to both sexes.
- If age is not explicitly given, the default age is `new_adult_age - 1` (in discrete generation the default age is `0`).

### 3.9 `modifiers(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `gamete_modifiers` | `Optional[list[(hook_id,name,fn)]]` | `None` | gamete‑to‑zygote mapping compilation | Usually used for gamete‑stage modifications. |
| `zygote_modifiers` | `Optional[list[(hook_id,name,fn)]]` | `None` | zygote mapping compilation | Usually used for zygote‑stage modifications. |

Code‑aligned format:

```python
.modifiers(
  gamete_modifiers=[(10, "drive_gamete", my_gamete_modifier_fn)],
  zygote_modifiers=[(20, "drive_zygote", my_zygote_modifier_fn)],
)
```

Explanation:

- The tuple structure is fixed as `(priority_or_hook_id, optional_name, callable)`.
- During configuration compilation, they are sorted by the first field before being applied.

### 3.10 `hooks(...)`

| Parameter | Type | Default | Affected stage | Explanation & advice |
|---|---|---|---|---|
| `*hook_items` | `Callable` or `HookMap` | empty | event points (first/early/late/finish, etc.) | Can mix function and dictionary registration forms. An undeclared event will raise an error. |

Two valid input forms:

```python
# 1) Direct function (function must have @hook(event='...') metadata)
.hooks(my_hook_fn)

# 2) Mapping
.hooks({
  "late": [(my_hook_fn, "my_hook", 10)],
  "finish": [(finish_hook, "finish", 0)],
})
```

Common errors:

- Passing a plain function without `event` metadata -> `ValueError`.
- Passing something that is not callable nor a dictionary -> `TypeError`.

### 3.11 `build()`

`build()` takes no arguments but has strong constraints:

- `initial_state(...)` must be set beforehand.
- Execution order:
  - First constructs `PopulationConfig`.
  - Then creates the population object.
  - Then applies presets.
  - Then applies fitness/modifiers/hooks.

This is why `build()` should be placed last in the chain.

## 4. DiscreteGenerationPopulationBuilder Parameter Reference

Key differences for the discrete‑generation Builder:

- Uses `n_ages=2`, `new_adult_age=1` by default.
- You do **not** need `age_structure(...)`.

### 4.1 `setup(...)`

Same as for the age‑structured model:

- `name`
- `stochastic`
- `use_continuous_sampling`
- `use_fixed_egg_count`

### 4.2 `initial_state(...)`

| Parameter | Type | Default | Affected stage | Explanation |
|---|---|---|---|---|
| `individual_count` | `dict` | required | initial state | Still `{sex: {genotype: age_data}}`, but only ages 0 and 1 are supported. |

Discrete model `age_data` code‑aligned format (from `_resolve_discrete_age_distribution`):

```python
# Scalar -> (age0=0, age1=value)
{"female": {"WT|WT": 1000}}

# Sequence of length 1 -> (0, value)
{"female": {"WT|WT": [1000]}}

# Sequence of length 2 -> (age0, age1)
{"female": {"WT|WT": [200, 800]}}

# Mapping -> only keys 0 and 1 allowed
{"female": {"WT|WT": {0: 200, 1: 800}}}
```

Validation rules:

- List length must be `<= 2`.
- Dictionary only allows age keys `0` and `1`.
- All counts must be non‑negative.

### 4.3 `reproduction(...)`

| Parameter | Type | Default | Affected stage | Explanation |
|---|---|---|---|---|
| `eggs_per_female` | `float` | `50.0` | reproduction | Egg production per female. |
| `sex_ratio` | `float` | `0.5` | reproduction | Proportion of female offspring. |
| `female_adult_mating_rate` | `float` | `1.0` | reproduction | Adult female mating rate. |
| `male_adult_mating_rate` | `float` | `1.0` | reproduction | Adult male mating rate. |

### 4.4 `survival(...)`

| Parameter | Type | Default | Affected stage | Explanation |
|---|---|---|---|---|
| `female_age0_survival` | `float` | `1.0` | survival | Female juvenile survival. |
| `male_age0_survival` | `float` | `1.0` | survival | Male juvenile survival. |
| `adult_survival` | `float` | `0.0` | survival/aging boundary | Adult inter‑step survival. Setting to 0 approximates non‑overlapping generations. |

Modelling constraints:

- All three probabilities should be in `[0, 1]`.
- `adult_survival=0.0` is commonly used for strictly discrete generations.

### 4.5 `competition(...)`

| Parameter | Type | Default | Affected stage | Explanation |
|---|---|---|---|---|
| `juvenile_growth_mode` | `int\|str` | `"logistic"` | juvenile density regulation | Growth mode. |
| `low_density_growth_rate` | `float` | `1.0` | juvenile density regulation | Low‑density growth rate. |
| `carrying_capacity` | `Optional[int]` | `None` | density cap | Carrying capacity. |

### 4.6 `presets(...)` / `fitness(...)` / `modifiers(...)` / `hooks(...)` / `build()`

Semantics are the same as for the age‑structured model, with differences only in the discrete‑generation kernel and fixed age structure.

## 5. How Parameters Affect the Simulation Process

Looking at one tick:

1. reproduction
- Key parameters: mating rates, relative fertility, `eggs_per_female`, `sex_ratio`, `fecundity`, `sexual_selection`, sperm storage parameters.

2. survival
- Key parameters: survival rate vectors/scalars, `viability`.

3. aging
- Key parameters: `n_ages`, `new_adult_age` (age‑structured model) or `adult_survival` (discrete‑generation model).

4. density regulation (juvenile competition)
- Key parameters: `juvenile_growth_mode`, `low_density_growth_rate`, carrying‑capacity‑related parameters.

5. hook event points
- Logic registered via `hooks(...)` triggers at fixed event points for pre‑/post‑stage interventions.

## 6. Recommended Configuration Order (ready to copy)

### 6.1 Age‑Structured Model

1. `setup(...)`
2. `age_structure(...)`
3. `initial_state(...)`
4. `survival(...)`
5. `reproduction(...)`
6. `competition(...)`
7. `presets(...)` -> `fitness(...)` -> `modifiers(...)` -> `hooks(...)`
8. `build()`

### 6.2 Discrete‑Generation Model

1. `setup(...)`
2. `initial_state(...)`
3. `reproduction(...)`
4. `survival(...)`
5. `competition(...)`
6. `presets(...)` -> `fitness(...)` -> `modifiers(...)` -> `hooks(...)`
7. `build()`

## 7. Common Errors and Troubleshooting

1. Forgetting `initial_state(...)`
- Symptom: `build()` immediately errors.

2. Age vector length inconsistent with `n_ages`
- Symptom: Error during initialisation or compilation.

3. `sex_ratio` or probability parameters out of bounds
- Symptom: Abnormal results or runtime errors.

4. Setting the same parameter multiple times leading to overrides
- Symptom: Behaviour differs from expectation. For example, `generation_time` and `equilibrium_distribution` can be set in multiple methods; later calls override earlier ones.

## 8. Chapter Summary

Builder is not “syntactic sugar” – it organises model parameters into a reviewable, traceable, and tunable configuration workflow.

You can think of it as three layers:

1. Parameter layer: each parameter maps to a clear simulation stage.
2. Compilation layer: `build()` solidifies into configuration and state.
3. Execution layer: the simulation flow runs stage by stage, consuming these parameters.

**Next chapter**: [IndexRegistry Indexing Mechanism](index_registry.md)
