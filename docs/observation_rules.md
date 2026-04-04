# Population Observation Rules

This chapter introduces the `observation` module, which is used to extract and aggregate data from the population state. This is a key component for data analysis, visualisation, and statistical inference.

---

## Core Concepts

### Why is `ObservationFilter` needed?

During a simulation, we often need to:

1. Extract specific sub‑populations from the complete `individual_count` array (which includes all genotypes, sexes, and ages)
2. Compress multi‑dimensional data into one‑dimensional vectors for statistical comparison
3. Support flexible grouping (e.g., “all adult females” or “larvae of a specific genotype”)

### Three Key Objects

| Object | Purpose |
|--------|---------|
| **ObservationFilter** | The main class for creating filtering rules and applying them |
| **Rule** | A NumPy mask array with shape `(n_groups, n_sexes, [n_ages], n_genotypes)` |
| **Observed** | Aggregated data after applying the rule, shape `(n_groups, n_sexes, [n_ages])` |

### Design Features

- **Pure function design**: `apply_rule()` is a side‑effect‑free NumPy operation
- **Flexible selectors**: Supports multiple formats for specifying genotypes, ages, and sexes
- **Unified pattern matching**: Genotype selection has been unified with `GeneticPattern` (supports ordered `|` and unordered `::`)
- **Performance optimised**: All operations are vectorised with NumPy, avoiding explicit loops

---

## ObservationFilter API

### Constructor

```python
from natal.observation import ObservationFilter
from natal.index_registry import IndexRegistry

# Create a filter
registry = pop.registry  # IndexRegistry instance
filter = ObservationFilter(registry)
```

**Parameters**:
- `registry`: An `IndexRegistry` object, used for resolving genotype names

### `build_filter` Method

The main method for constructing filtering rules.

```python
def build_filter(
    self,
    pop_or_state: Union[PopulationState, BasePopulation],
    *,
    diploid_genotypes: Optional[Union[Sequence, Species, BasePopulation]] = None,
    groups: Optional[Union[List, Tuple, Dict]] = None,
    collapse_age: bool = False,
) -> Tuple[np.ndarray, List[str]]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pop_or_state` | `PopulationState` \| `BasePopulation` | Population object or state |
| `diploid_genotypes` | `Sequence` \| `Species` \| `BasePopulation` | List of genotypes (supports multiple formats) |
| `groups` | `None` \| `List` \| `Dict` | Group specifications |
| `collapse_age` | `bool` | Whether to collapse all ages into one dimension |

**Returns**:
```python
(rule, labels)
# rule: np.ndarray, shape (n_groups, n_sexes, [n_ages], n_genotypes)
# labels: List[str], name of each group
```

#### `groups` Format

##### 1. `groups=None` (default, one group per genotype)

```python
rule, labels = filter.build_filter(pop, diploid_genotypes=pop.species)
# labels = ['g0', 'g1', 'g2', ...], one label per genotype
```

##### 2. `groups` as a list (unnamed groups)

```python
groups = [
    {"genotype": ["WT|WT"], "sex": "female"},
    {"genotype": ["WT|Drive"], "age": [2, 3, 4]},
]
rule, labels = filter.build_filter(pop, groups=groups)
# labels = ['group_0', 'group_1']
```

##### 3. `groups` as a dict (named groups)

```python
groups = {
    "all_females": {"sex": "female"},
    "adults": {"age": [2, 3, 4, 5, 6, 7]},
    "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]},
    "juvenile_drive": {
        "genotype": ["WT|Drive"],
        "age": [0, 1],
        "sex": "female"
    },
}
rule, labels = filter.build_filter(pop, groups=groups)
# labels = ['all_females', 'adults', 'drive_carriers', 'juvenile_drive']
```

#### Selector Specifications

Within the `groups` dictionary, each group specification supports the following keys:

##### `genotype` / `genotypes`

Specify genotypes. Supports multiple formats:

```python
# String (comma‑separated)
{"genotype": "WT|WT"}

# Pattern string (recommended)
# | represents ordered match (Maternal|Paternal)
# :: represents unordered match (the two copies of a homologous chromosome can be swapped)
{"genotype": "A1/B1|A2/B2; C1/D1::C2/D2"}

# List of strings
{"genotype": ["WT|WT", "WT|Drive", "Drive|Drive"]}

# Integer indices
{"genotype": [0, 2, 3]}

# Wildcard (all genotypes)
{"genotype": "*"}

# Not specified (default: all genotypes)
{}
```

##### Using Pattern Matching for Observation Groups (Recommended)

When the target group is complex, it is not recommended to write long genotype lists by hand. It is better to pass a pattern string directly to `groups["genotype"]` and let the observation module parse it uniformly.

```python
groups = {
    "target_female": {
        # Ordered match: Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_female_unordered": {
        # Unordered match (the two copies of a homologous chromosome can be swapped)
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}

rule, labels = filter.build_filter(pop, groups=groups)
```

This approach lets Observation and Preset share the same pattern semantics, reducing inconsistencies between “rule target” and “observation target”.

##### `sex`

Specify sex. Supports:

```python
# String
{"sex": "female"}  or  {"sex": "male"}
{"sex": "f"}       or  {"sex": "m"}

# Integer (Sex.FEMALE = 0, Sex.MALE = 1)
{"sex": 0}  or  {"sex": 1}

# List
{"sex": ["female", "male"]}

# Not specified or None (both sexes)
{}
```

##### `age`

Specify age. Supports multiple formats:

```python
# Explicit list
{"age": [2, 3, 4]}

# Closed interval [start, end] (inclusive)
{"age": [2, 7]}  # ages 2,3,4,5,6,7

# List of intervals (union)
{"age": [[0, 1], [4, 6]]}  # ages 0,1,4,5,6

# Callable (predicate function)
{"age": lambda a: a >= 2}

# Not specified (all ages)
{}
```

##### `unordered`

`unordered` is a compatibility parameter. For pattern strings, it is recommended to use `::` to express unordered semantics directly; `|` maintains ordered semantics (Maternal|Paternal).

```python
# Recommended: express ordered/unordered directly in the pattern
{"genotype": "A|a"}      # ordered
{"genotype": "A::a"}     # unordered
```

If you are using a non‑pattern selector object, `unordered=True` can still be used as a supplementary option:

```python
# Enable unordered matching
{"genotype": ["A|a"], "unordered": True}
# Matches both "A|a" and "a|A"

# Disable (default)
{"genotype": ["A|a"], "unordered": False}
# Matches only "A|a"
```

---

## Building Filtering Rules

### Example 1: Simple Sex Grouping

```python
from natal.observation import ObservationFilter

filter = ObservationFilter(pop.registry)

groups = {
    "females": {"sex": "female"},
    "males": {"sex": "male"},
}

rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups
)

print(labels)  # ['females', 'males']
print(rule.shape)  # (2, 2, 8, n_genotypes) if there are 8 ages
```

### Example 2: Age Stratification

```python
groups = {
    "juveniles": {"age": [0, 1]},
    "young_adults": {"age": [2, 3]},
    "old_adults": {"age": [4, 7]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### Example 3: Genotype‑Specific Observation

```python
# Track the time dynamics of specific genotypes
groups = {
    "WT_WT_all": {"genotype": ["WT|WT"]},
    "WT_WT_females": {"genotype": ["WT|WT"], "sex": "female"},
    "WT_WT_juvenile_f": {"genotype": ["WT|WT"], "sex": "female", "age": [0, 1]},
    "drive_all": {"genotype": ["WT|Drive", "Drive|Drive"]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### Example 4: Unordered Genotypes (Suppression/Drive Systems)

In a suppression system (e.g., CRISPR‑based suppression), `"S|+"` and `"+|S"` may be equivalent:

```python
groups = {
    "suppressed_hetero": {
        "genotype": "S::+",  # Recommended: direct unordered semantics
    },
    "suppressed_homo": {"genotype": ["S|S"]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### Example 5: Collapsing Age

Sometimes we want to ignore the age dimension and compare only sex and genotype directly:

```python
rule, labels = filter.build_filter(
    pop,
    groups={"females": {"sex": "female"}},
    collapse_age=True
)

# rule.shape will be (1, 2, n_genotypes), not (1, 2, 8, n_genotypes)
```

---

## Applying Rules

### `apply_rule` Function

```python
from natal.observation import apply_rule
import numpy as np

# Obtain observed values from a rule and the individual count array
observed = apply_rule(pop.state.individual_count, rule)

# observed shape: (n_groups, n_sexes, [n_ages]) or (n_groups, n_sexes)
```

**How it works**:
1. Multiply the rule array (which contains only 0 or 1) by the individual count array
2. Sum over the genotype dimension to obtain the total number of individuals in each group

**Example**:

```python
# pop.state.individual_count shape: (2, 8, 50)
# rule shape: (3, 2, 8, 50)
# observed shape: (3, 2, 8)

# observed[i, j, k] = total number of individuals in group i, sex j, age k
```

### Complete Workflow

```python
from natal.observation import ObservationFilter, apply_rule

# 1. Create a filter
filter = ObservationFilter(pop.registry)

# 2. Define groups (e.g., for PMCMC likelihood calculation)
groups = {
    "female_drive": {"genotype": ["WT|Drive", "Drive|Drive"], "sex": "female"},
    "male_drive": {"genotype": ["WT|Drive", "Drive|Drive"], "sex": "male"},
}

# 3. Build the rule
rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups
)

# 4. Apply the rule at each time step
observations = []
for _ in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)
    observations.append(observed)

# 5. Convert to an array for statistical analysis
observations = np.array(observations)  # shape: (100, 2, 2, 8)
```

---

## Practical Examples

### Example 1: Monitoring Gene Drive Spread

```python
from natal.observation import ObservationFilter, apply_rule
import numpy as np

pop = AgeStructuredPopulation(
    species=species,
    initial_individual_count={...},
    n_ages=8,
)

# Define observation targets: monitor drive allele frequency
filter = ObservationFilter(pop.registry)

groups = {
    "WT": {"genotype": ["WT|WT"]},
    "heterozygous": {"genotype": ["WT|Drive"]},
    "homozygous_drive": {"genotype": ["Drive|Drive"]},
}

rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups,
    collapse_age=True  # we don't care about age distribution
)

# Record time series
times = []
drive_freq = []

for t in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)

    # observed shape: (3, 2)
    # Calculate drive allele frequency
    het_count = observed[1].sum()  # heterozygotes
    hom_count = observed[2].sum()  # homozygotes
    total_alleles = 2 * observed[0].sum() + het_count + 2 * hom_count
    drive_alleles = het_count + 2 * hom_count

    drive_freq.append(drive_alleles / max(1, total_alleles))
    times.append(t)

# Visualise
import matplotlib.pyplot as plt
plt.plot(times, drive_freq)
plt.xlabel("Time (generations)")
plt.ylabel("Drive allele frequency")
plt.show()
```

### Example 2: Age‑Specific Monitoring

```python
# Focus on genotype distributions across different age groups
groups = {
    "juvenile_WT": {"age": [0, 1], "genotype": ["WT|WT"]},
    "juvenile_Drive": {"age": [0, 1], "genotype": ["WT|Drive", "Drive|Drive"]},
    "adult_WT": {"age": [2, 7], "genotype": ["WT|WT"]},
    "adult_Drive": {"age": [2, 7], "genotype": ["WT|Drive", "Drive|Drive"]},
}

rule, labels = filter.build_filter(pop, groups=groups)

# Observe each step
for t in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)

    # observed shape: (4, 2)
    # observed[0] = juvenile_WT females & males
    # observed[1] = juvenile_Drive females & males
    # ...

    print(f"t={t}: juveniles={observed[:2].sum():.0f}, "
          f"adults={observed[2:].sum():.0f}")
```

### Example 3: PMCMC Likelihood Calculation

```python
from natal.observation import ObservationFilter, apply_rule
from scipy.stats import poisson

# Suppose we have actual data observed_data
observed_data = np.array([
    [1000, 800],  # t=0: female, male counts
    [950, 750],
    [900, 700],
    # ...
])

# Set up the observation filter
filter = ObservationFilter(pop.registry)
groups = {
    "females": {"sex": "female"},
    "males": {"sex": "male"},
}
rule, labels = filter.build_filter(pop, groups=groups, collapse_age=True)

# Calculate likelihood
log_likelihood = 0.0
for t, data_t in enumerate(observed_data):
    pop.step()

    # Get simulated observations
    simulated = apply_rule(pop.state.individual_count, rule)
    # simulated shape: (2, 2) - 2 groups, 2 sexes

    # Sum over groups for each sex
    sim_females = simulated[0, 0] + simulated[1, 0]  # females
    sim_males = simulated[0, 1] + simulated[1, 1]    # males

    # Poisson likelihood
    log_likelihood += (
        poisson.logpmf(data_t[0], sim_females) +
        poisson.logpmf(data_t[1], sim_males)
    )

print(f"Log-likelihood: {log_likelihood:.2f}")
```

### Example 4: Integration with PMCMC Parameter Management (Recommended)

In the current PMCMC design, the mapping from parameters to model configuration is managed at the PMCMC layer, not inside the particle filter. It is recommended to update parameters via the `params_to_model_fn` callback.

```python
import numpy as np
from samplers.pmcmc import run_pmcmc
from samplers.likelihood import (
    make_init_sampler,
    make_transition_fn,
    make_obs_loglik_fn,
    LogLikelihoodEvaluator,
)
from samplers.parameter_mapping import make_fitness_config_applier

# Existing: config, observations, obs_rule, shapes, state_flat, param_idx, geno_idx
init_sampler = make_init_sampler(state_flat, n_sexes=shapes[0][0], n_ages=shapes[0][1], n_genotypes=shapes[0][2])
transition_fn, transition_args = make_transition_fn(config, shapes, param_idx=param_idx, geno_idx=geno_idx)
obs_loglik_fn = make_obs_loglik_fn(10.0, obs_rule, apply_rule, shapes=shapes)

evaluator = LogLikelihoodEvaluator(
    config=config,
    observations=observations,
    initial_state=state_flat,
    shapes=shapes,
    n_particles=300,
    sigma=10.0,
    obs_rule=obs_rule,
)

# Parameter mapping function (can be replaced with custom logic to modify arbitrary PopulationConfig fields)
params_to_model_fn = make_fitness_config_applier(
    config=config,
    param_idx=param_idx,
    geno_idx=geno_idx,
)

result = run_pmcmc(
    observations=observations,
    n_particles=300,
    init_sampler=init_sampler,
    transition_fn=transition_fn,
    obs_loglik_fn=obs_loglik_fn,
    params_init=np.array([0.5, 0.5]),
    n_iter=1000,
    step_sizes=np.array([0.1, 0.1]),
    log_prior_fn=lambda p: 0.0,
    transition_args=transition_args,
    params_to_model_fn=params_to_model_fn,
    loglik_evaluator=evaluator,
)
```

---

## Performance Tips

### 1. Reuse Rules

If you use the same grouping across multiple simulations, build the rule once and reuse it:

```python
# ✅ Recommended
rule, labels = filter.build_filter(pop, groups=groups)
for particle in range(100):
    pop.reset()
    for t in range(n_steps):
        pop.step()
        observed = apply_rule(pop.state.individual_count, rule)

# ❌ Inefficient
for particle in range(100):
    pop.reset()
    rule, _ = filter.build_filter(pop, groups=groups)  # rebuild each time
    for t in range(n_steps):
        pop.step()
        observed = apply_rule(pop.state.individual_count, rule)
```

### 2. Vectorise `apply_rule`

`apply_rule` supports batch application:

```python
# If you have multiple rules
rules = [rule1, rule2, rule3]
results = [apply_rule(pop.state.individual_count, r) for r in rules]
```

### 3. Collapse Age Early

If you do not need age resolution, use `collapse_age=True` to reduce dimensionality:

```python
# Faster
rule, _ = filter.build_filter(pop, groups=groups, collapse_age=True)

# Slower
rule, _ = filter.build_filter(pop, groups=groups, collapse_age=False)
```

---

## Common Errors

### Error 1: `diploid_genotypes` is `None`

```python
# ❌ Wrong
rule, labels = filter.build_filter(pop, groups=groups)

# ✅ Correct
rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,  # must be provided
    groups=groups
)
```

### Error 2: Genotype String Mismatch

```python
# Suppose the population’s genotypes are generated via to_string()
gt = pop.species.get_all_genotypes()[0]
print(gt.to_string())  # Output: "WT|WT"

# ❌ Will not match (typo)
groups = {"target": {"genotype": ["wt|wt"]}}

# ✅ Correct (case‑sensitive)
groups = {"target": {"genotype": ["WT|WT"]}}
```

### Error 3: Age Range Out of Bounds

```python
# If pop.n_ages = 8 (ages 0‑7)

# ❌ Wrong (age 8 does not exist)
groups = {"old": {"age": [6, 8]}}

# ✅ Correct
groups = {"old": {"age": [6, 7]}}
```

---

## Next Steps

- [API Entry](api/genetic_structures.md) – see complete method signatures
- [Hook System](hooks.md) – apply observation filters inside Hooks
- [Numba Optimization](numba_optimization.md) – performance tuning

---

**Back to Contents**: [Full Documentation Index](index.md)
