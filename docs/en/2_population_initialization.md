# Population Initialization (Panmictic)

Population initialization is the first step of NATAL Core simulation, which configures and constructs a population through a chainable API.

> **Note**: This chapter covers the chainable configuration of **panmictic (single deme, well-mixed)** populations. For building multi-deme spatial populations (with topology, migration, and `batch_setting` heterogeneous configuration), please refer to the [Spatial Simulation Guide](3_spatial_simulation.md). The chainable syntax for spatial populations is essentially the same as this chapter, with the addition of a `.migration()` method and `batch_setting` support.

## Quick Start: Chainable API Configuration

NATAL Core provides a concise chainable API for configuring populations, which is the recommended usage:

```python
import natal as nt

# Chainable API configuration (recommended)
pop = (
    nt.AgeStructuredPopulation
    .setup(species=my_species)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 100},
        "male": {"WT|WT": 100}
    })
    .survival(female_age_based_survival_rates=0.85)
    .reproduction(eggs_per_female=50.0)
    .competition(age_1_carrying_capacity=1000)
    .hooks(my_hook)
    .build()
)
```

## Configuration Flow

The complete configuration flow is as follows:

```text
Population.setup() → chainable configuration method calls
  → build()
  → PopulationConfig / PopulationState
  → run_tick / run
  → reproduction → survival → aging (and hooks)
```

After configuration, you can use the returned population object to call `run()` or `run_tick()` methods to start the simulation (see [Simulation Kernels Deep Dive](4_simulation_kernels.md)).

## Two Types of Configuration Interfaces

NATAL Core provides two main population types:

- **`AgeStructuredPopulation`**: Suitable for multi-age-class models, supporting configurable numbers of age stages, per-age vector inputs, and optional sperm storage mechanisms.
- **`DiscreteGenerationPopulation`**: Suitable for discrete generation models, defaulting to two age stages (`n_ages=2`, `new_adult_age=1`), simplifying the modeling of non-overlapping generations.

## AgeStructuredPopulation Parameter Reference

### `setup(...)` – Basic Setup

Configure basic population information and randomness.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `name` | `str` | Population identification name | `"AgeStructuredPop"` | Entire workflow | Used only for logging and identification, does not affect dynamics; explicitly name experiments for clarity |
| `stochastic` | `bool` | Whether to use random sampling | `True` | Sampling stages (reproduction / survival, etc.) | `True` for random, `False` for deterministic; use `False` during parameter tuning |
| `use_continuous_sampling` | `bool` | Sampling strategy selection | `False` | Probability sampling details | Controls the sampling method; keep default for most scenarios |
| `use_fixed_egg_count` | `bool` | Whether egg count is fixed | `False` | reproduction | `True` for fixed egg count, `False` for more realistic random egg production |
| `species` | `Species` | Species object | Required | Entire workflow | Defines the genetic structure of the population; core configuration parameter |

### `age_structure(...)` – Age Structure

Configure the population's age structure, including the total number of age stages and the juvenile/adult division.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `n_ages` | `int` | Total number of age stages | `8` | Entire workflow (array dimensions) | Constrains array lengths for initial state, survival rates, etc.; must match lengths of all age-related parameters |
| `new_adult_age` | `int` | Age index at which individuals enter the adult stage | `2` | reproduction / survival | Recommended to align with the life history stage of the target species; individuals below this age are considered juveniles |
| `generation_time` | `Optional[int]` | Generation time marker | `None` | Compilation parameter | Used only for modeling interpretation; mutually exclusive with the same-named parameter in `age_structure`, the later one takes precedence |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | Explicit equilibrium distribution (2, n_ages) array | `None` | Competition metric derivation | Mutually exclusive with the same-named parameter in `survival` and `competition`; later one takes precedence; age=0 value is ignored (see competition section) |

### `initial_state(...)` – Initial State

Initial state parameters take effect at the start of the simulation, providing base data for various sampling functions in `algorithms.py`. The initial individual count distribution directly affects subsequent reproduction and survival calculations; sperm storage data is used by the `sample_mating` function during the reproduction phase.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `individual_count` | `Mapping` | Initial individual count distribution, format `{sex: {genotype: age_data}}` | Required | Initial state | If not set, `build()` will raise an error; supports scalar, sequence, mapping, and other formats |
| `sperm_storage` | `Optional[Mapping]` | Initial sperm storage (only needed when sperm storage is enabled) | `None` | reproduction | Required only when `use_sperm_storage=True`; format is a three-level mapping |

**Age data (`age_data`) format** (all counts must be non-negative):

- **Scalar**: Assigned to all ages in the range `[new_adult_age, n_ages)`
- **List / Tuple / Array**: Written sequentially by age index; excess elements are truncated, missing elements are filled with `0`
- **Dictionary**: Explicitly specifies values for specific ages, e.g., `{2: 100, 3: 80}`

Examples:

```python
# Scalar: all adult ages (>= new_adult_age) assigned 100
{"female": {"WT|WT": 100.0}}

# Sequence: written in age order
{"female": {"WT|WT": [0, 100, 80, 60]}}

# Mapping: assign values only to some ages
{"female": {"WT|WT": {2: 100, 3: 80}}}
```

**Sperm storage (`sperm_storage`) format** (three-level mapping):

```python
{
  "WT|WT": {                 # Female genotype
    "Drive|WT": [0, 0, 20],  # Male genotype -> sperm count by age
  }
}
```

Validation rules:

- Age index must be in the range `[0, n_ages)`
- All counts must be `>= 0`
- Genotype strings must be correctly parseable by the current `Species`

### `survival(...)` – Survival Parameters

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional` | Female per-age survival rates | `None` | survival | Supports scalar, sequence, mapping, function, etc.; `None` uses default curve; range `[0, 1]` |
| `male_age_based_survival_rates` | `Optional` | Male per-age survival rates | `None` | survival | Same as above |
| `generation_time` | `Optional[int]` | Generation time marker | `None` | Compilation parameter | Mutually exclusive with the same-named parameter in `age_structure`; later one takes precedence |
| `equilibrium_distribution` | `Optional` | Explicit equilibrium distribution (2, n_ages) array | `None` | Competition metric derivation | Mutually exclusive with the same-named parameter in `age_structure` and `competition`; later one takes precedence; age=0 value is ignored (see competition section) |

**Code examples** (from `_resolve_survival_param`):

```python
# A) None → use default curve
.survival(female_age_based_survival_rates=None)

# B) Scalar → same value for all ages
.survival(female_age_based_survival_rates=0.85)

# C) Sequence → written per age, missing filled with 0, excess truncated
.survival(female_age_based_survival_rates=[1.0, 1.0, 0.9, 0.7])

# D) Sparse mapping → unspecified ages default to 1.0
.survival(female_age_based_survival_rates={0: 1.0, 1: 0.95, 2: 0.8})

# E) Function → must accept an age parameter and return a value
.survival(female_age_based_survival_rates=lambda age: 1.0 if age < 2 else 0.8)

# F) Sequence with None sentinel at end → filled with last non-None value
.survival(female_age_based_survival_rates=[1.0, 0.9, None])
```

### `reproduction(...)` – Reproduction Parameters

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional` | Female per-age mating rates | `None` | reproduction | Length must equal `n_ages`; default values used when not set |
| `male_age_based_mating_rates` | `Optional` | Male per-age mating rates | `None` | reproduction | Length must equal `n_ages`; default values used when not set |
| `female_age_based_relative_fertility` | `Optional` | Female per-age relative fertility weights | `None` | reproduction | Length must equal `n_ages`; used to modulate egg production contribution across ages |
| `eggs_per_female` | `float` | Base number of eggs per female | `50.0` | reproduction | Baseline for population egg production; start with neutral value during tuning |
| `use_fixed_egg_count` | `bool` | Whether egg count is fixed | `False` | reproduction | `True` for fixed egg count, `False` for random egg production |
| `sex_ratio` | `float` | Proportion of female offspring | `0.5` | reproduction | Range `[0, 1]`; `0.5` means equal sex ratio. Ignored when sex chromosomes can determine offspring sex (e.g., XX/ZW for female, XY/ZZ for male) |
| `use_sperm_storage` | `bool` | Whether to enable sperm storage mechanism | `True` | reproduction | `True` enables, `False` disables (only current mating considered) |
| `sperm_displacement_rate` | `float` | Rate at which new sperm replaces old sperm | `0.05` | reproduction | Typical range `(0, 1]`; larger values mean faster replacement |

### `competition(...)` – Competition and Density Regulation

Competition parameters take effect during the survival phase of the population.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `competition_strength` | `float` | Relative competition factor for old juveniles (age=1) | `5.0` | Juvenile density regulation | Competition weights vary by age: age=0 fixed at `1.0`, age=1 uses `competition_strength` |
| `juvenile_growth_mode` | `Union[int, str]` | Density regulation mode for juvenile growth | `"logistic"` | Juvenile density regulation | Supports `"logistic"`, `"beverton_holt"`, etc.; usually `"logistic"` |
| `low_density_growth_rate` | `float` | Intrinsic growth rate at low density | `6.0` | Juvenile density regulation | Growth multiplier under no competition; overly large values can cause oscillations |
| `age_1_carrying_capacity` | `Optional[int]` | Carrying capacity at the age=1 stage | `None` | Juvenile density regulation | If explicitly specified, takes highest priority |
| `old_juvenile_carrying_capacity` | `Optional[int]` | Legacy parameter name (deprecated) with same function as `age_1_carrying_capacity` | `None` | Juvenile density regulation | `age_1_carrying_capacity` recommended; when both are set, `age_1_carrying_capacity` takes precedence |
| `expected_num_adult_females` | `Optional[int]` | Expected number of adult females, used to independently calculate expected egg production | `None` | Expected egg production derivation | Decoupled from `age_1_carrying_capacity`: one sets capacity, the other sets egg production (see below) |
| `equilibrium_distribution` | `Optional` | Explicit equilibrium distribution (2, n_ages) array | `None` | Competition metric derivation | Can be passed via `age_structure`, `survival`, or `competition`; later one takes precedence |

**Carrying capacity resolution logic**:

Carrying capacity $K$ and expected egg production are two independent concepts. The system follows a separation principle:

- `age_1_carrying_capacity` (or legacy alias `old_juvenile_carrying_capacity`) directly specifies the **carrying capacity at age=1 $K$**
- `expected_num_adult_females` independently specifies the **expected egg production** (does not back-calculate $K$)

The initialization path has three scenarios:

1. **Explicitly specified equilibrium distribution** (via `age_structure().equilibrium_distribution` or related parameters):
   - Directly reads the age=1 total from the equilibrium distribution as $K$
   - Calculates total expected egg production based on the equilibrium distribution, female mating rates, relative fertility, and egg production
   - Equilibrium survival rate = $K$ / total expected egg production

2. **Both `age_1_carrying_capacity` and `expected_num_adult_females` provided**:
   - `age_1_carrying_capacity` is directly used as $K$
   - Propagates `expected_num_adult_females` forward through survival rates to each adult age group, producing the female equilibrium distribution
   - Calculates expected egg production based on this distribution (considering mating rates and relative fertility)
   - The system independently calculates the equilibrium survival rate using $K$ (from `age_1_carrying_capacity`) and expected egg production (from `expected_num_adult_females`)

3. **Missing items inferred from initial state** (assuming the initial state is at equilibrium):
   - If $K$ is missing: uses the total count of age-1 individuals from the initial state
   - If expected egg production is missing: calculates expected egg production from the female distribution in the initial state

Regardless of the path taken, the system will genuinely construct the equilibrium distribution, then compute all competition metrics from it. This ensures consistency among $K$, expected egg production, and the equilibrium survival rate.

**Expected egg production formula**:

Total expected egg production is calculated as:

```
total_expected_eggs = Σ( N_f[age] × P_reproducing[age] × fertility[age] × eggs_per_female )
                     for all age ∈ [new_adult_age, n_ages)
```

Where:
- `N_f[age]`: Number of females at that age at equilibrium
  - When derived from `expected_num_adult_females`: propagated forward from new_adult_age using female survival rates
  - When derived from the equilibrium distribution: read directly from the female row in the distribution
- `P_reproducing[age]`: Proportion of females of that age participating in reproduction, from `female_age_based_reproduction_rates` (if not set, uses the female row of `female_age_based_mating_rates`)
- `fertility[age]`: Relative fertility weight for that age, from `female_age_based_relative_fertility` (defaults to all 1s)
- `eggs_per_female`: Base number of eggs per female

**Meaning of external_expected_eggs**:

When `expected_num_adult_females` is provided (path 2), the system uses it to calculate an expected egg production that is independent of the equilibrium distribution, called **external_expected_eggs**. This value is only used for survival rate calculation:

```
equilibrium_survival_rate = K / (external_expected_eggs × s_0_avg)
```

It does not affect competition intensity calculations (competition intensity always uses the egg production from the equilibrium distribution itself). This achieves independent control of capacity $K$ and expected egg production.

**Regarding age=0 in the equilibrium distribution**:

When passing an explicit equilibrium distribution, the age=0 value in the distribution is **ignored**. The system always calculates `produced_age_0` (expected age=0 egg production) from the distribution of adult females (`age >= new_adult_age`), rather than reading the age=0 value from the distribution. This is because between ticks, the age-0 count in the state is always 0 (age-0 individuals are produced at the start of each tick and density-regulated in the same tick).

Similarly, when the system automatically constructs an equilibrium distribution (propagating forward from $K$), age=0 is always set to 0.

### `presets(...)` – Preset Configuration

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `*preset_list` | `Any` variadic arguments | List of preset configuration objects | Empty | Post-processing after `build` | Presets are applied first to establish baseline configuration; subsequent settings via `fitness`, `modifiers`, `hooks`, etc. can override preset values |

### `fitness(...)` – Fitness Coefficients

Fitness parameters take effect at different stages of the simulation. `sexual_selection` affects mating probabilities in `compute_mating_probability_matrix` during reproduction, `fecundity` affects egg production in `fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction`, `viability` combines with age-specific survival rates in `compute_viability_survival_rates` during the survival phase, and `zygote_viability` is applied to newborn individuals immediately after the reproduction phase.

NATAL supports flexible fitness configuration schemes. In simulation, the following fitness types take effect at different stages:

- `viability`: Survival fitness coefficient, affects individual survival probability.
- `fecundity`: Fecundity fitness coefficient, affects individual reproductive capacity.
- `sexual_selection`: Mating preference weight, affects individual mate choice.
- `zygote_viability`: Zygote survival fitness coefficient, affects zygote survival probability.

> It is recommended to use `presets` to configure fitness coefficients for specific genotypes, rather than passing `fitness` directly.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `viability` | `Optional[Dict]` | Survival fitness coefficient | `None` | survival | Supports multi-level nesting: by genotype, by sex, by age, by sex+age. Default `None` means no modification. |
| `fecundity` | `Optional[Dict]` | Fecundity fitness coefficient | `None` | reproduction | Specified by genotype and/or sex. Default `None` means no modification. |
| `sexual_selection` | `Optional[Dict]` | Mating preference weight | `None` | reproduction | Supports flat mapping `{male: value}` or nested mapping `{female: {male: value}}`. |
| `zygote_viability` | `Optional[Dict]` | Zygote survival fitness coefficient | `None` | reproduction | Applied before the survival phase, represents the probability of a zygote surviving to become an individual. |
| `mode` | `str` | Fitness value write mode | `"replace"` | Fitness write strategy | `"replace"` overwrites existing values, `"multiply"` scales by a factor. |

**Code examples**:

```python
# viability: genotype → float
.fitness(viability={"WT|WT": 1.0, "Drive|Drive": 0.6})

# viability: genotype → {sex: float}
.fitness(viability={"Drive::WT": {"female": 0.9, "male": 0.8}})

# viability: genotype → {age: float}, shared between sexes
.fitness(viability={"Drive::WT": {0: 0.95, 1: 0.85}})

# viability: genotype → {sex: {age: float}}, can be sex+age specific
.fitness(viability={"Drive::WT": {"female": {1: 0.9}, "male": {2: 0.8}}})

# fecundity: genotype → float or {sex: float}
.fitness(fecundity={"Drive|Drive": 0.7})

# sexual_selection flat format: {male_selector: value}, female defaults to '*'
.fitness(sexual_selection={"Drive::WT": 1.2, "WT|WT": 1.0})

# sexual_selection nested format: {female_selector: {male_selector: value}}
.fitness(sexual_selection={"WT|WT": {"Drive::WT": 0.8, "WT|WT": 1.0}})

# zygote_viability fitness: genotype → float (both sexes)
.fitness(zygote_viability={"A|A": 0.5, "a|a": 0.8})

# zygote_viability fitness: genotype → {sex: float} (sex-specific)
.fitness(zygote_viability={"a|a": {"female": 0.3, "male": 0.4}})
```

**About genotype keys**:
- Can be `Genotype` objects
- Can be exact genotype strings or pattern matching strings (see [Genotype Pattern Matching](2_genotype_patterns.md))
- Can be a tuple containing the above types, e.g., `("Drive::WT", "Drive|Drive")`, representing the union of these genotypes

**`viability` age key constraints** (consistent with code):

- Age keys must be integers in the range `[0, n_ages)`
- If using `{sex: ...}` form, can further nest `{age: float}`
- If directly using `{age: float}`, the same value is used for both sexes
- When no age is explicitly specified, defaults to `new_adult_age - 1` (default age `0` in discrete generation models)

### `modifiers(...)` – Modifiers (Gamete/Zygote Conversion)

**Algorithm timing**: Modifiers take effect during the configuration compilation phase, building the mapping from gametes to zygotes. `gamete_modifiers` process gamete conversion before `compute_offspring_probability_tensor`, and `zygote_modifiers` process genotype conversion after zygote formation. These conversion functions affect subsequent reproduction and inheritance calculations.

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `gamete_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | List of gamete-stage conversion functions | `None` | gamete → zygote mapping compilation | Each element is a `(priority, name, function)` tuple; applied in priority order |
| `zygote_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | List of zygote-stage conversion functions | `None` | zygote mapping compilation | Each element is a `(priority, name, function)` tuple; applied in priority order |

**Example**:

```python
.modifiers(
  gamete_modifiers=[(10, "drive_gamete", my_gamete_modifier_fn)],
  zygote_modifiers=[(20, "drive_zygote", my_zygote_modifier_fn)],
)
```

Notes:

- The tuple structure is fixed as `(priority or hook ID, optional name, callable)`
- During configuration compilation, modifiers are sorted by the first field (priority) before being applied

### `hooks(...)` – Hook Functions

**Algorithm timing**: Hook functions take effect at specific event points in the simulation, such as `first`, `early`, `late`, `finish`, etc. These hooks can insert custom logic at different stages of the simulation, affecting population state or simulation behavior. Hook execution timing is determined by the event type and priority.

> For detailed documentation on `hooks`, see [Hook System](2_hooks.md).

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `*hook_items` | `Callable` or `HookMap` | Hook functions or hook registration mappings | Empty | Event points (first / early / late / finish, etc.) | Pass functions directly (with `@hook` decorator). |

**Example**:

```python
@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10")
    ]

# ...
.hooks(release_drive_carriers)
```

Common errors:

- Passing a regular function without `event` metadata → raises `ValueError`.
- Passing a value that is neither callable nor a dictionary → raises `TypeError`.

### `build()` – Compilation Build

The `build()` method takes no parameters but has strong constraints:

- `initial_state(...)` must be called before it to set the initial state.
- Execution order:
  1. Build `PopulationConfig`
  2. Create population object
  3. Apply presets
  4. Apply fitness / modifiers / hooks

Therefore, it is recommended to place `build()` at the end of the chain.

## DiscreteGenerationPopulation Parameter Reference

Key differences between the discrete generation model and the age-structured model:

- Defaults to `n_ages=2` and `new_adult_age=1`.
- Does not require calling `age_structure(...)`.

### `setup(...)`

Parameters are consistent with the age-structured model: `name`, `stochastic`, `use_continuous_sampling`, `use_fixed_egg_count`, `species`. `species` is required to define the genetic structure of the population.

### `initial_state(...)`

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `individual_count` | `dict` | Distribution of initial individual counts, format `{sex: {genotype: age_data}}` | Required | Initial state | Discrete generation model only supports age 0 and age 1; input format meanings are the same as the age-structured model |

**`age_data` resolution rules for discrete model** (from `_resolve_discrete_age_distribution`):

```python
# Scalar → (age0=0, age1=value)
{"female": {"WT|WT": 1000}}

# Length-1 sequence → (0, value)
{"female": {"WT|WT": [1000]}}

# Length-2 sequence → (age0, age1)
{"female": {"WT|WT": [200, 800]}}

# Mapping → only keys 0 or 1 allowed
{"female": {"WT|WT": {0: 200, 1: 800}}}
```

Validation rules:

- List length must be `<= 2`.
- Dictionary keys only allow `0` and `1`.
- All counts must be non-negative.

### `reproduction(...)`

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `eggs_per_female` | `float` | Number of eggs per female per generation | `50.0` | reproduction | Baseline for egg production; start with neutral value during tuning |
| `sex_ratio` | `float` | Proportion of female offspring | `0.5` | reproduction | Range `[0, 1]`; `0.5` means equal sex ratio. Ignored when sex chromosomes can determine offspring sex (e.g., XX/ZW for female, XY/ZZ for male) |
| `female_adult_mating_rate` | `float` | Adult female mating rate | `1.0` | reproduction | Proportion of females participating in mating; range `[0, 1]` |
| `male_adult_mating_rate` | `float` | Adult male mating rate | `1.0` | reproduction | Proportion of males participating in mating; range `[0, 1]` |

### `survival(...)`

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `female_age0_survival` | `float` | Female juvenile (age 0) survival rate | `1.0` | survival | Range `[0, 1]`; `1.0` means all survive |
| `male_age0_survival` | `float` | Male juvenile (age 0) survival rate | `1.0` | survival | Range `[0, 1]`; `1.0` means all survive |
| `adult_survival` | `float` | Adult survival rate between generations | `0.0` | survival / aging boundary | Range `[0, 1]`; set to `0` for strict non-overlapping generations, higher values allow adults to survive across generations |

Modeling advice:

- All three probabilities should ideally be constrained to `[0, 1]`.
- `adult_survival=0.0` is commonly used for strict discrete generation models.

### `competition(...)`

| Parameter | Type | Description | Default | Affected Stage | Notes |
|---|---|---|---|---|---|
| `juvenile_growth_mode` | `Union[int, str]` | Density regulation mode for juvenile growth | `"logistic"` | Juvenile density regulation | Commonly `"logistic"`, also supports `"beverton_holt"` and other modes |
| `low_density_growth_rate` | `float` | Intrinsic growth multiplier at low density | `1.0` | Juvenile density regulation | Growth multiplier under no competition; overly large values can cause oscillations |
| `carrying_capacity` | `Optional[int]` | Carrying capacity for juveniles | `None` | Density upper limit | If not set, the system will attempt automatic derivation; explicitly specified values take highest priority |

### `presets(...)` / `fitness(...)` / `modifiers(...)` / `hooks(...)` / `build()`

The semantics of these methods are fully consistent with the age-structured model, the only difference being that the discrete generation kernel uses a fixed age structure.

## Recommended Configuration Order
### Age-Structured Model

1. `setup(...)`
2. `age_structure(...)`
3. `initial_state(...)`
4. `survival(...)`
5. `reproduction(...)`
6. `competition(...)`
7. `presets(...)` → `fitness(...)` → `modifiers(...)` → `hooks(...)`
8. `build()`

### Discrete Generation Model

1. `setup(...)`
2. `initial_state(...)`
3. `reproduction(...)`
4. `survival(...)`
5. `competition(...)`
6. `presets(...)` → `fitness(...)` → `modifiers(...)` → `hooks(...)`
7. `build()`

## Common Errors and Troubleshooting

| Error Symptom | Possible Cause | Solution |
|---|---|---|
| `build()` raises an error | Forgot to set `initial_state(...)` | Call `initial_state(...)` before `build()` |
| Error during initialization or compilation | Age vector length does not match `n_ages` | Ensure all age-related parameter lengths equal `n_ages` |
| Abnormal results or runtime errors | `sex_ratio` or other probability parameters out of bounds | Check that parameters are within valid ranges (e.g., `[0, 1]`) |
| Behavior does not match expectations | Same-named parameter set multiple times leading to overwrite | Note that `generation_time`, `equilibrium_distribution` etc. can be set in multiple methods; later calls override earlier ones |

## Implementation Principles

The underlying chainable API uses a Builder object to manage all configurations. The order in which configurations take effect is:

1. **Basic configuration**: `setup()` and `age_structure()` set basic parameters
2. **State configuration**: `initial_state()` sets the initial population state
3. **Dynamics configuration**: `survival()`, `reproduction()`, `competition()` set population dynamics parameters
4. **Advanced configuration**: `hooks()`, `fitness()`, `modifiers()` set advanced features
5. **Final build**: `build()` compiles all configurations and creates the population object

### Detailed Working Mechanism

The chainable API's working mechanism is based on the following design principles:

1. **Class method startup**: `setup()` is a class method called directly on the class name, returning a configuration object instance
2. **Chainable calls**: Each configuration method returns the configuration object itself, supporting continuous calls
3. **Configuration validation**: At `build()` time, all required parameters are uniformly validated to ensure configuration completeness
4. **Configuration compilation**: The chain configuration is converted into underlying `PopulationConfig` and `PopulationState` objects

This design makes the configuration process both intuitive and flexible, while maintaining high performance at the underlying level.

## Chapter Summary

Population initialization provides a concise and intuitive configuration approach through a chainable API, organizing population parameters into a categorizable, chainable configuration workflow, and registering them into the underlying `PopulationConfig` at build time, thus achieving the unification of high-level usability and low-level high performance.

## Related Chapters

- [Hook System](2_hooks.md) - Detailed usage of hook functions
- [Genotype Pattern Matching](2_genotype_patterns.md) - Detailed genotype matching rules
- [PopulationState & PopulationConfig: Compilation and Configuration](4_population_state_config.md) - Detailed underlying configuration objects
- [Simulation Kernels Deep Dive](4_simulation_kernels.md) - Simulation execution flow and algorithm implementation

***

**Next Chapter**: [Hook System](2_hooks.md)
