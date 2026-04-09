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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `name` | `str` | Population identifier name. | `"AgeStructuredPop"` | whole simulation | For logging and identification; does not affect dynamics; name experiments explicitly. |
| `stochastic` | `bool` | Choose stochastic or deterministic sampling. | `True` | sampling stages (reproduction/survival) | `True`=stochastic, `False`=deterministic; use `False` during tuning. |
| `use_continuous_sampling` | `bool` | Sampling strategy choice. | `False` | probability sampling details | Controls sampling method; keep default for most scenarios. |
| `use_fixed_egg_count` | `bool` | Whether egg count is fixed. | `False` | reproduction | `True` fixes egg count, `False` mimics random oviposition. |

### 3.2 `age_structure(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `n_ages` | `int` | Total number of age classes. | `8` | whole simulation (array dimensions) | Constrains initial state and survival vector array lengths; must be consistent with all age-related parameters. |
| `new_adult_age` | `int` | Age index when individual enters adult stage. | `2` | reproduction/survival | Should match target species' life history; ages below this are juveniles. |
| `generation_time` | `Optional[int]` | Generation time marker. | `None` | compilation parameter | For model interpretation; same name as `age_structure` parameter, later value overrides earlier; avoid conflict with age definition. |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | Auxiliary equilibrium distribution. | `None` | competition/initial scaling | Same name as `age_structure` parameter, later value overrides earlier; use only for explicit steady-state scaling. |

### 3.3 `initial_state(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `individual_count` | `Mapping[str, Mapping[Union[Genotype, str], Union[int, float, List, Tuple, NDArray, Dict[int, float]]]]` | Initial individual count distribution. Format `{sex: {genotype: age_data}}`. | required | initial state | Will cause `build()` to error if not set; supports scalar, sequence, mapping and other formats. |
| `sperm_storage` | `Optional[Mapping[Union[Genotype, str], Mapping[Union[Genotype, str], Union[int, float, List, Tuple, NDArray, Dict[int, float]]]]]` | Initial sperm inventory (if sperm storage enabled). | `None` | reproduction | Only needed when `use_sperm_storage=True`; format is three-level mapping. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional[Union[int, float, List[float], NDArray[np.float64], Dict[int, float], Callable]]` | Female survival rates by age. | `None` | survival | Supports scalar/sequence/mapping/callable formats; `None` uses default curve; recommend limiting to `[0, 1]`. |
| `male_age_based_survival_rates` | `Optional[Union[int, float, List[float], NDArray[np.float64], Dict[int, float], Callable]]` | Male survival rates by age. | `None` | survival | Same as `female_age_based_survival_rates`. |
| `generation_time` | `Optional[int]` | Generation time marker. | `None` | compilation parameter | Same name parameter as `age_structure`; later value overrides earlier. |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | Auxiliary equilibrium distribution. | `None` | scaling helper | Same name parameter as `age_structure`; later value overrides earlier. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional[list/ndarray]` | Female mating rates by age. | `None` | reproduction | Length must equal `n_ages`; uses defaults when not set. |
| `male_age_based_mating_rates` | `Optional[list/ndarray]` | Male mating rates by age. | `None` | reproduction | Length must equal `n_ages`; uses defaults when not set. |
| `female_age_based_relative_fertility` | `Optional[list/ndarray]` | Female relative fertility weight by age. | `None` | reproduction | Length must equal `n_ages`; tunes contribution to egg production by female age. |
| `eggs_per_female` | `float` | Base egg count per female individual. | `50.0` | reproduction | Used as baseline for population egg production; start from neutral value. |
| `use_fixed_egg_count` | `bool` | Whether egg count is fixed. | `False` | reproduction | `True` fixes egg count, `False` uses random oviposition. |
| `sex_ratio` | `float` | Proportion of females in offspring. | `0.5` | reproduction | Range should be in `[0, 1]`; 0.5 represents 1:1 sex ratio. Ignored when sex chromosome constraints deterministically assign offspring sex (e.g., XX/ZW female, XY/ZZ male). |
| `use_sperm_storage` | `bool` | Enable sperm storage mechanism. | `True` | reproduction | `True` enables storage, `False` disables; mating only in current generation. |
| `sperm_displacement_rate` | `float` | Rate of new sperm replacing old sperm. | `0.05` | reproduction | Range typically in `(0, 1]`; higher value = faster replacement. |

Format and length requirements (source behaviour):

- `female_age_based_mating_rates` / `male_age_based_mating_rates` / `female_age_based_relative_fertility`
  - When provided, they are converted to `np.array`.
  - During configuration compilation, their length must equal `n_ages`, otherwise `ValueError` is raised.

### 3.6 `competition(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `competition_strength` | `float` | Intensity factor of juvenile competition. | `5.0` | juvenile density regulation | Affects strength of density regulation effect; larger value = stronger control. |
| `juvenile_growth_mode` | `int\|str` | Juvenile growth density regulation mode. | `"logistic"` | juvenile density regulation | Supports `"logistic"`, `"beverton_holt"` etc.; commonly use `logistic`. |
| `low_density_growth_rate` | `float` | Intrinsic growth rate at low density. | `6.0` | juvenile density regulation | Represents multiplication factor without competition; too high may cause oscillations. |
| `age_1_carrying_capacity` | `Optional[int]` | Population carrying capacity at age=1. | `None` | juvenile density regulation | If specified, takes precedence over other sources (highest priority). |
| `old_juvenile_carrying_capacity` | `Optional[int]` | Alias for `age_1_carrying_capacity` (deprecated). | `None` | juvenile density regulation | Legacy parameter name; both parameters work but `age_1_carrying_capacity` is preferred. |
| `expected_num_adult_females` | `Optional[int]` | Expected count of adult females. | `None` | capacity derivation | Used to infer carrying capacity via equilibrium analysis (see note below). |
| `equilibrium_distribution` | `Optional[list/ndarray]` | Auxiliary equilibrium distribution. | `None` | scaling helper | Same name parameter as `age_structure`; later value overrides earlier. |

**Carrying capacity resolution logic:**

When neither `age_1_carrying_capacity` nor `old_juvenile_carrying_capacity` are specified, `expected_num_adult_females` is used to infer the carrying capacity through equilibrium distribution analysis:

1. If `age_1_carrying_capacity` or `old_juvenile_carrying_capacity` (legacy alias) is provided, that value is used (highest priority).
2. If `expected_num_adult_females` is provided, the system distributes this count across age classes using age-based survival rates.
3. Based on the equilibrium age distribution, it computes the expected age-0 egg production from adult females using mating rates and fertility weights.
4. The inferred carrying capacity (K at age=1) is computed from the age-0 production and base survival rate from age-0 to age-1.
5. If no carrying capacity source is available, the system attempts to infer from initial state (`initial_state()`) if provided.

This approach ensures that the carrying capacity is consistent with the equilibrium population distribution, rather than using a naive scaling factor.

### 3.7 `presets(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `*preset_list` | `Any` (varargs) | Preset configuration object list. | empty | post‑build processing | Presets applied first to establish baseline; subsequent `fitness`/`modifiers`/`hooks` can override preset values. |

### 3.8 `fitness(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `viability` | `Optional[Dict[GenotypeSelector, Union[float, Dict[Union[str, Sex, int], Union[float, Dict[int, float]]]]]]` | Survival fitness coefficient. | `None` | survival | Supports multi-level: by genotype, by sex, by age, by sex+age; default `None` does not modify. |
| `fecundity` | `Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]]` | Reproductive fitness coefficient. | `None` | reproduction | Specified by genotype and/or sex; default `None` does not modify. |
| `sexual_selection` | `Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]]` | Pair preference weight. | `None` | reproduction | Supports flat mapping `{male: value}` or nested mapping `{female: {male: value}}`. |
| `mode` | `str` | Fitness value writing mode. | `"replace"` | fitness writing strategy | `"replace"` overwrites original, `"multiply"` scales by factor. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `gamete_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | Gamete-stage conversion function list. | `None` | gamete‑to‑zygote mapping compilation | Format `(priority, name, function)` tuples; applied in priority order. |
| `zygote_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | Zygote-stage conversion function list. | `None` | zygote mapping compilation | Format `(priority, name, function)` tuples; applied in priority order. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `*hook_items` | `Callable` or `HookMap` | Hook functions or hook registration mapping. | empty | event points (first/early/late/finish, etc.) | Supports two forms: direct function (requires @hook metadata) or event mapping dictionary; undeclared events raise error. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `individual_count` | `dict` | Initial individual count distribution. Format `{sex: {genotype: age_data}}`. | required | initial state | Discrete generation model only supports age 0 and age 1; input formats same as age-structured model. |

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

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `eggs_per_female` | `float` | Egg count per female individual per generation. | `50.0` | reproduction | Baseline value for population egg production; start from neutral value. |
| `sex_ratio` | `float` | Proportion of females in offspring. | `0.5` | reproduction | Range should be in `[0, 1]`; 0.5 represents 1:1 sex ratio. Ignored when sex chromosome constraints deterministically assign offspring sex (e.g., XX/ZW female, XY/ZZ male). |
| `female_adult_mating_rate` | `float` | Adult female mating rate. | `1.0` | reproduction | Proportion of females participating in mating; range `[0, 1]`. |
| `male_adult_mating_rate` | `float` | Adult male mating rate. | `1.0` | reproduction | Proportion of males participating in mating; range `[0, 1]`. |

### 4.4 `survival(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `female_age0_survival` | `float` | Female juvenile (age 0) survival rate. | `1.0` | survival | Range `[0, 1]`; 1.0 = complete survival. |
| `male_age0_survival` | `float` | Male juvenile (age 0) survival rate. | `1.0` | survival | Range `[0, 1]`; 1.0 = complete survival. |
| `adult_survival` | `float` | Adult inter-generational survival rate. | `0.0` | survival/aging boundary | Range `[0, 1]`; 0.0 approximates strict non-overlapping generations; higher values allow adult persistence. |

Modelling constraints:

- All three probabilities should be in `[0, 1]`.
- `adult_survival=0.0` is commonly used for strictly discrete generations.

### 4.5 `competition(...)`

| Parameter | Type | Explanation | Default | Affected stage | Remarks |
|---|---|---|---|---|---|
| `juvenile_growth_mode` | `Union[int, str]` | Juvenile growth density regulation mode. | `"logistic"` | juvenile density regulation | Common: `"logistic"`; also supports other modes like `"beverton_holt"`. |
| `low_density_growth_rate` | `float` | Intrinsic growth multiplication at low density. | `1.0` | juvenile density regulation | Represents growth factor without competition; too high may cause oscillations. |
| `carrying_capacity` | `Optional[int]` | Juvenile carrying capacity. | `None` | density cap | Auto-derived if not set; uses explicitly specified value with priority. |

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
