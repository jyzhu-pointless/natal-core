# Simulation Kernels Deep Dive

<!--TODO: Rewrite as a mathematical model introduction, requires many formulas-->

This chapter describes the simulation execution pipeline of NATAL from a user's perspective:

- What you call at the user layer;
- How the framework completes a single tick internally;
- Where history recording, state import/export, and Hooks fit into the flow.

After reading this chapter, you should be able to clearly answer two questions:

1. What stage computations does a single `pop.run(...)` perform internally.
2. When to use `run(...)`, `run_tick()`, `get_history()`, and `export_state()`.

## 1. User Entry Points and Execution Path

In daily use, you only need to program against the population object:

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

Internally, the execution path can be summarized as:

```text
population.run(...) / population.run_tick()
  → Retrieve compiled event hooks
  → Bind codegen runner
  → Sequentially invoke stage kernels (reproduction/survival/aging)
  → Update state and history
```

This means you do not need to manually organize low-level kernel calls; simply focus on parameters, Hooks, and result analysis.

## 2. Consistent Usage Across Both Population Types

### 2.1 `AgeStructuredPopulation`

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

### 2.2 `DiscreteGenerationPopulation`

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

Both share the same calling convention; the differences are primarily in the internal state structure and stage kernels.

## 3. Stage Order Within a Tick

Taking a standard tick as an example, the execution order is:

1. `first` user Hook
2. `reproduction` stage
3. `early` user Hook
4. `survival` stage
5. `late` user Hook
6. `aging` stage
7. `n_tick` is incremented

This order applies to both age-structured models and discrete-generation models; different models invoke their corresponding kernel implementations.

### 3.1 `AgeStructuredPopulation` Step-by-Step Algorithm

Taking one tick of `AgeStructuredPopulation` as an example, the three main stages can be further expanded as:

1. reproduction
  - Calculate effective male count weighted by age: `male_count[age, g] * male_mating_rate[age]`.
  - Construct the mating probability matrix `P(g_f -> g_m)` based on sexual selection fitness and effective male count.
  - Call `sample_mating(...)` to update `sperm_store` (including sperm displacement logic).
  - Call the fertilization function to generate age-0 new individuals (write female/male into `ind_count[:, 0, :]` respectively).
  - Apply zygote fitness to the newly generated age-0 individuals.
2. survival
  - First, apply density regulation to age-0 (juveniles): `NO_COMPETITION / FIXED / LOGISTIC / BEVERTON_HOLT`.
  - Then compute the combined survival rate of "age-based survival rate × viability."
  - Update both `individual_count` and `sperm_store` simultaneously using the combined survival rate, ensuring consistency between them.
3. aging
  - Shift all age classes forward by one.
  - Clear the new age-0 slot, ready for the next tick's reproduction.

Key point: AgeStructured follows the "long-term sperm storage" path, and `sperm_store` is synchronously updated across all three stages (reproduction/survival/aging).

### 3.2 `DiscreteGenerationPopulation` Step-by-Step Algorithm

`DiscreteGenerationPopulation` has a fixed `n_ages=2` (age0 = juvenile, age1 = adult), and each tick's algorithm is more compact:

1. reproduction
  - Only age1 adults participate in mating and fertilization.
  - Uses a temporary `temp_sperm_store` for the current step's fertilization; does not retain a long-term sperm bank across ticks.
  - Offspring are written into age0.
2. survival
  - First apply density regulation to age0 (also supports the four growth modes).
  - Apply the combined survival rate (age-based survival rate × viability) only to age0.
3. aging
  - Generational turnover: `age0 → age1`.
  - The original age1 is overwritten (i.e., "old adults exit" in discrete generations).

Key point: Discrete emphasizes "non-overlapping generations" and does not have the cross-age, cross-tick long-term sperm storage state found in AgeStructured.

### 3.3 Stochastic vs. Deterministic: Two Execution Semantics Under the Same Flow

The stage order is unchanged, but the numerical update method is determined by the configuration:

1. `is_stochastic=False` (deterministic)
  - Uses expected values/proportional scaling; results are typically continuous values (float).
  - Does not perform Binomial/Poisson sampling.
2. `is_stochastic=True` (stochastic)
  - Uses sampling-based updates (e.g., Binomial/Poisson/Multinomial, etc.); trajectories exhibit random fluctuations.
  - If `use_continuous_sampling=True`, continuous approximation sampling (e.g., Beta/Dirichlet/Gamma approximation) is used to improve differentiability/continuity and numerical stability in certain scenarios.

Additionally, the reproduction stage is affected by `use_fixed_egg_count`:

- `True`: Eggs are produced at a fixed expected count.
- `False`: Eggs are produced via a Poisson mechanism (resulting in random egg counts in stochastic mode).

## 4. Responsibilities of the `simulation_kernels` Module

`src/natal/kernels/simulation_kernels.py` primarily provides "stage-level kernel functions," including:

- Age-structured model: `run_reproduction`, `run_survival`, `run_aging`
- Discrete-generation model: `run_discrete_reproduction`, `run_discrete_survival`, `run_discrete_aging`

Additionally, this module provides lightweight wrapper functions for state/config import/export, facilitating integration with higher-level object methods.

### 4.1 Spatial Migration Backend Module Layout

Spatial migration kernels are now split into directory modules under `src/natal/kernels/migration/`:

- `adjacency.py`: Adjacency backend (dense/sparse row routing).
- `kernel.py`: Topology + migration-kernel backend.
- `__init__.py`: Package-level backend entry re-export.

The compatibility entry point `src/natal/kernels/spatial_migration_kernels.py` maintains the old API and dispatches according to backend mode:

- `migration_mode == 0` → adjacency backend (`adjacency.py`)
- `migration_mode == 1` → kernel-topology backend (`kernel.py`)

This allows the internal migration implementation to be organized into a maintainable modular structure without changing the user-facing entry point (e.g., `run_spatial_migration(...)`).

## 5. Relationship with `state`/`config`

During simulation, kernels read and write two core objects:

- `state`: The current population distribution and time step.
- `config`: Rule parameters such as survival rates, mating rates, fitness, and mapping matrices.

If you have read the previous chapter, you can think of this chapter as "how `state`/`config` are consumed and updated in each tick."

## 6. History Recording Mechanism

`run(...)` can record history data at intervals:

```python
pop.run(n_steps=200, record_every=10)
history = pop.get_history()
```

Practical advice:

- Smaller `record_every` values produce denser history, useful for diagnosing details.
- Larger `record_every` values produce more compact history, better suited for long-term simulations.
- If intermediate trajectories are not needed, set it to `0` to reduce memory usage.

## 7. State Export and Restoration

When you need to save snapshots, transfer state across scripts, or run forking experiments, use:

```python
state_flat, history = pop.export_state()
# ... save or process externally ...
pop.import_state(state_flat, history=history)
```

Typical scenarios:

1. Run to a critical time point and save a snapshot.
2. Fork multiple parameter branches from the same snapshot.
3. Compare trajectory differences under different strategies.

## 8. How Hooks Integrate into the Execution Pipeline

User-defined Hooks (e.g., `first`/`early`/`late`) are compiled and merged into the execution flow, then triggered by the runner at the corresponding stage.

This provides two benefits:

- The high-level API remains clean and simple to use.
- Execution maintains a consistent stage order, making results more explainable.

## 9. Recommended Usage Patterns

1. Batch simulations: Prefer `pop.run(...)`.
2. Single-step observation: Use `pop.run_tick()`.
3. Trajectory analysis: Combine `record_every` with `get_history()`.
4. Snapshot experiments: Use `export_state()` / `import_state()`.
5. Behavior extension: Use Hooks rather than manually assembling kernel calls.

## 10. Minimal Example

```python
# 1) Build population
pop = ...

# 2) Run continuously
pop.run(n_steps=100, record_every=10)

# 3) Single-step advance
pop.run_tick()

# 4) Retrieve history
history = pop.get_history()

# 5) Export and restore
state_flat, hist = pop.export_state()
pop.import_state(state_flat, history=hist)
```

## 11. Chapter Summary

The execution mechanism of NATAL can be understood as a three-layer分工:

- Population layer: Provides a stable user API and lifecycle management.
- Runner/Hook layer: Organizes stage flows and event logic into a unified execution chain.
- Kernel layer: Performs numerical computations for each stage.

In practical modeling, you typically only need to use the population API consistently and enhance controllability and explainability through Hooks and history when needed.

---

## Related Sections

- [PopulationState and PopulationConfig](4_population_state_config.md)
- [Numba Optimization Guide](4_numba_optimization.md)
- [Modifier Mechanism](3_modifiers.md)
- [Hook System](2_hooks.md)
