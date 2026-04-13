# Simulation Kernels Deep Dive

This chapter explains the simulation execution chain of NATAL from a user perspective:

- What you call at the user level.
- How the framework completes one tick internally.
- Where history recording, state export/import, and Hooks fit into the flow.

After reading this chapter, you should be able to clearly answer two questions:

1. What stage computations happen inside `pop.run(...)`.
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
  → obtain compiled event hooks
  → bind codegen runner
  → call stage kernels in order (reproduction/survival/aging)
  → update state and history
```

This means you do not need to manually organise low‑level kernel calls; just focus on parameters, Hooks, and result analysis.

## 2. Consistent Usage for Both Population Types

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

The calling convention is identical for both; differences lie mainly in the internal state structure and the stage kernels.

## 3. Stage Order Within One Tick

Taking a standard tick, the execution order is:

1. User `first` Hook
2. `reproduction` stage
3. User `early` Hook
4. `survival` stage
5. User `late` Hook
6. `aging` stage
7. `n_tick` incremented

This order holds for both the age‑structured model and the discrete‑generation model; different models call the corresponding kernel implementations.

### 3.1 Detailed Algorithm for `AgeStructuredPopulation` per Tick

For an `AgeStructuredPopulation` tick, the three main stages can be expanded further:

1. reproduction
  - Compute age‑weighted effective male count: `male_count[age, g] * male_mating_rate[age]`.
  - Build mating probability matrix `P(g_f -> g_m)` based on sexual selection fitness and effective male counts.
  - Call `sample_mating(...)` to update `sperm_store` (including sperm displacement logic).
  - Call fertilisation functions to generate age‑0 new individuals (females/males written to `ind_count[:, 0, :]` respectively).
  - Apply zygote fitness to newly formed offspring (age‑0 individuals) before survival stage.
2. survival
  - First apply density regulation to age‑0 (juveniles): `NO_COMPETITION / FIXED / LOGISTIC / BEVERTON_HOLT`.
  - Then compute the combined survival probability “age‑specific survival × viability”.
  - Update both `individual_count` and `sperm_store` with the combined survival probability, keeping them consistent.
3. aging
  - Shift all age classes forward by one.
  - Clear the new age‑0 slots, waiting for the next tick’s reproduction to write into them.

Key point: AgeStructured is the “long‑term sperm storage” path; `sperm_store` is synchronously updated in all three stages (reproduction/survival/aging).

### 3.2 Detailed Algorithm for `DiscreteGenerationPopulation` per Tick

`DiscreteGenerationPopulation` fixes `n_ages=2` (age0 = juvenile, age1 = adult). The algorithm per tick is more compact:

1. reproduction
  - Only adults at age1 are used for mating and fertilisation.
  - A temporary `temp_sperm_store` is used for fertilisation within the step; there is no long‑term sperm storage across ticks.
  - Offspring are written into age0.
2. survival
  - First apply density regulation to age0 (supports the same four growth modes).
  - Apply combined survival probability (age‑specific survival × viability) only to age0.
3. aging
  - Generation turnover: `age0 -> age1`.
  - The previous age1 is overwritten (i.e., “old adults exit” in discrete generations).

Key point: Discrete emphasises “non‑overlapping generations” and does not have the cross‑age, cross‑tick long‑term sperm storage state of AgeStructured.

### 3.3 Stochastic vs Deterministic: Two Execution Semantics Under the Same Stage Order

The stage order is unchanged, but how values are updated depends on the configuration:

1. `is_stochastic=False` (deterministic)
  - Uses expectations / proportional scaling; results are usually continuous values (float).
  - No Binomial/Poisson sampling.
2. `is_stochastic=True` (stochastic)
  - Uses sampling (e.g., Binomial/Poisson/Multinomial) for updates; trajectories exhibit random fluctuations.
  - If `use_continuous_sampling=True`, continuous approximations (e.g., Beta/Dirichlet/Gamma approximations) are used to improve differentiability/continuity and numerical stability in certain scenarios.

Additionally, the reproduction stage is influenced by `use_fixed_egg_count`:

- `True`: lays eggs according to a fixed expected number.
- `False`: lays eggs according to a Poisson mechanism (in stochastic mode this manifests as random egg numbers).

## 4. Responsibilities of the `simulation_kernels` Module

`src/natal/kernels/simulation_kernels.py` mainly provides “stage‑level kernel functions”, including:

- For the age‑structured model: `run_reproduction`, `run_survival`, `run_aging`.
- For the discrete‑generation model: `run_discrete_reproduction`, `run_discrete_survival`, `run_discrete_aging`.

The module also provides lightweight wrapper functions for importing/exporting state and configuration, making them easy to use together with the higher‑level object methods.

### 4.1 Spatial Migration Backend Layout

Spatial migration kernels are now organized as a package under
`src/natal/kernels/migration/`:

- `adjacency.py`: adjacency-row backend (dense/sparse row routing).
- `kernel.py`: topology + migration-kernel backend.
- `__init__.py`: package-level re-exports for backend entry points.

The compatibility facade `src/natal/kernels/spatial_migration_kernels.py`
keeps the legacy public API stable while dispatching by backend mode:

- `migration_mode == 0` -> adjacency backend (`adjacency.py`)
- `migration_mode == 1` -> kernel-topology backend (`kernel.py`)

This split keeps migration internals modular without changing user-facing
entry points such as `run_spatial_migration(...)`.

## 5. Relationship with `state` / `config`

When the simulation runs, the kernels read and write two core objects:

- `state`: the current population distribution and time step.
- `config`: rule parameters such as survival rates, mating rates, fitness values, mapping matrices, etc.

If you have read the previous chapter, you can think of this chapter as explaining “how `state`/`config` are consumed and updated in each tick”.

## 6. History Recording Mechanism

`run(...)` can write history data at intervals:

```python
pop.run(n_steps=200, record_every=10)
history = pop.get_history()
```

Practical advice:

- Smaller `record_every` gives denser history, good for diagnosing details.
- Larger `record_every` gives a sparser history, better for long‑term simulations.
- If intermediate trajectories are not needed, set it to `0` to reduce memory usage.

## 7. State Export and Restoration

When you need to save snapshots, pass state between scripts, or branch experiments, you can use:

```python
state_flat, history = pop.export_state()
# ... save or process externally ...
pop.import_state(state_flat, history=history)
```

Typical scenarios:

1. Save a snapshot after reaching a critical time point.
2. Derive multiple parameter branches from the same snapshot.
3. Compare trajectory differences under different strategies.

## 8. How Hooks Are Embedded in the Execution Chain

User‑defined Hooks (e.g., `first`/`early`/`late`) are compiled and merged into the execution flow, then triggered by the runner at the corresponding stage.

This brings two benefits:

- The high‑level API remains concise.
- The stage order stays uniform, making results more interpretable.

## 9. Recommended Usage Patterns

1. Batch simulations: prefer `pop.run(...)`.
2. Single‑step observation: use `pop.run_tick()`.
3. Analysing trajectories: combine `record_every` and `get_history()`.
4. Snapshot experiments: use `export_state()` / `import_state()`.
5. Extending behaviour: use Hooks, rather than manually assembling kernel calls.

## 10. Minimal Example

```python
# 1) Build the population
pop = ...

# 2) Run continuously
pop.run(n_steps=100, record_every=10)

# 3) Advance one step
pop.run_tick()

# 4) Retrieve history
history = pop.get_history()

# 5) Export and restore
state_flat, hist = pop.export_state()
pop.import_state(state_flat, history=hist)
```

## 11. Chapter Summary

You can view NATAL’s execution mechanism as a three‑layer division of labour:

- Population layer: provides a stable user API and lifecycle management.
- Runner / hook layer: organises the stage flow and event logic into a unified execution chain.
- Kernel layer: performs the numerical calculations for each stage.

In practice, you usually only need to use the population API stably, and improve controllability and interpretability through Hooks and history when needed.

---

## Related Chapters

- [PopulationState and PopulationConfig](population_state_config.md)
- [Numba Optimization Guide](numba_optimization.md)
- [Modifier Mechanism](modifiers.md)
- [Hook System](hooks.md)
