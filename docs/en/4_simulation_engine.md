# the Simulation Engine Deep Dive

<!--TODO: Rewrite as a mathematical model introduction, requires many formulas-->

This chapter describes the simulation execution pipeline of NATAL from a user's perspective:

- What you call at the user layer;
- How the framework completes a single tick internally;
- Where history recording, state import/export, and Hooks fit into the flow.

After reading this chapter, you should be able to clearly answer two questions:

1. What stage computations does a single `pop.run(...)` perform internally.
2. When to use `run(...)`, `run_tick()`, `pop.history`, and `export_state()`.

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
  → Run inside the native engine session
  → Sequentially execute stage kernels (reproduction/survival/aging)
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

Both share the same calling convention; the differences are primarily in the internal state structure and stage engine.

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

1. `stochastic=False` (deterministic)
  - Uses expected values/proportional scaling; results are typically continuous values (float).
  - Does not perform Binomial/Poisson sampling.
2. `stochastic=True` (stochastic)
  - Uses sampling-based updates (e.g., Binomial/Poisson/Multinomial, etc.); trajectories exhibit random fluctuations.
  - If `continuous_sampling=True`, continuous approximation sampling (e.g., Beta/Dirichlet/Gamma approximation) is used to improve differentiability/continuity and numerical stability in certain scenarios.

Additionally, the reproduction stage is affected by `fixed_egg_count`:

- `True`: Eggs are produced at a fixed expected count.
- `False`: Eggs are produced via a Poisson mechanism (resulting in random egg counts in stochastic mode).

## 4. Engine Implementation Layout

The native Rust extension `natal._engine_rs` is the only execution
engine. It owns the run state inside engine sessions (per-population for
panmictic models, one stacked session for spatial containers) and
implements the stage kernels for both model families:

- Age-structured model: reproduction, survival, aging (long-term sperm storage).
- Discrete-generation model: the compact two-age lifecycle with per-tick sperm.

### 4.1 Spatial Migration Layout

Migration runs inside the spatial engine session as the CSR stage after
the per-deme lifecycle. The frontend folds every migration declaration
(topology, adjacency matrix, or migration kernel) into one frozen CSR
plus a rate column at build time (`src/natal/frontend/spatial/migration.py`);
the session multiplies the rate column by that CSR each tick.

## 5. Relationship with `state`/`config`

During simulation, engine read and write two core objects:

- `state`: The current population distribution and time step.
- `config`: Rule parameters such as survival rates, mating rates, fitness, and mapping matrices.

If you have read the previous chapter, you can think of this chapter as "how `state`/`config` are consumed and updated in each tick."

## 6. History Recording Mechanism

`run(...)` can record history data at intervals:

```python
pop.run(n_steps=200, record_every=10)
history = pop.history.individual_count
```

Practical advice:

- Smaller `record_every` values produce denser history, useful for diagnosing details.
- Larger `record_every` values produce more compact history, better suited for long-term simulations.
- If intermediate trajectories are not needed, set it to `0` to reduce memory usage.

## 7. State Export and Restoration

When you need to save snapshots, transfer state across scripts, or run forking experiments, use:

```python
state_flat = pop.export_state()
# ... save or process externally ...
pop.import_state(state_flat)
# import_state() also clears the population's history, starting a fresh timeline
```

Typical scenarios:

1. Run to a critical time point and save a snapshot.
2. Fork multiple parameter branches from the same snapshot.
3. Compare trajectory differences under different strategies.

### 7.1 Random Streams (RNG) and the Bit-Reproducible Promise

- `build()` creates the Rust session automatically. The session owns its
  `SessionRng` and continues the stream across `run()` calls; no backend
  selection or enable step is required.
- In spatial models, deme `d` derives its stream from the session's base seed
  using `seed ^ d`. Lifecycle stages and migration consume that deme's stream.
- Inside Python hooks, `ctx.rng` is the event's controlled Rust sampler.
  Repeated access returns the same sampler and advances the stream. The
  sampler expires when the callback returns; restoring a checkpoint restores
  the recorded RNG state.
- **Promise scope**: with identical inputs (build parameters + seed + hook
  combination), deterministic (`stochastic=False`) trajectories are bitwise
  reproducible, and stochastic trajectories reproduce across processes under
  a fixed seed. Version-to-version bit-level stability is *not* promised
  (future algorithm fixes may change numerics).

### 7.2 Checkpoints and History Queries

Rust sessions own history values, checkpoints, and parameter logs. Every retained `mode="raw"` record has a complete checkpoint: individual and sperm state, tick, execution phase and status, RNG, ecology parameters (including migration and custom values), and log positions. Genetics tables are not rolled back. `mode="observation"` keeps only projected values, without hidden full raw history, and cannot restore checkpoints.

`pop.restore_checkpoint(tick)` accepts only an exact retained tick; an unrecorded or evicted tick fails before changing state. Restoration keeps history through that tick and truncates future parameter logs at the recorded positions, including updates made later at the same tick. Ordinary stable boundaries restore to `Ready`; manually recorded stopped or failed boundaries retain their execution status.

`record_snapshot()` records the current boundary between runs, including a stopped population; recording the same tick twice raises an error. `pop.history.boundary_metadata` returns an immutable sequence of `(tick, phase_cursor, status)` tuples. The phase cursor identifies a lifecycle position, such as `0` for a normal tick boundary and `2` for the early phase. Use it together with status to distinguish complete boundaries from interrupted execution.

`pop.params_log_details` returns `(tick, event, deme, parameter, old, new)` tuples. The deme is `0` for a non-spatial population; spatial populations expose logs through the corresponding deme's query interface. Values retain their Boolean, integer, floating-point, or array types; `None` represents the old value for additions and the new value for deletions. Arrays in query results are independent copies. `params_log` keeps the original four-column scalar projection, excluding arrays and additions/deletions; use `params_log_details` for the complete audit. Later updates or restores do not alter previously retrieved results.

`max_rows` bounds both retained records and checkpoints. Batch runs without Python callbacks record and evict within Rust. `clear_history()` clears records and their checkpoints while preserving current state, RNG, parameters, and parameter logs.

This complete example demonstrates bounded history and log rollback:

```python
import natal as nt

species = nt.Species.from_dict(
    "HistoryExample", {"Chr1": {"L1": ["W"]}}, gamete_labels=["default"]
)
pop = (
    nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False)
    .initial_state(individual_count={"female": {"W|W": 10}, "male": {"W|W": 10}})
    .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
    .competition(carrying_capacity=1000.0, low_density_growth_rate=2.0)
    .record_history(mode="raw", max_rows=3)
    .build()
)
pop.run(3, record_every=1)
assert pop.history.ticks == (1, 2, 3)
pop.update().competition(carrying_capacity=500.0)
assert pop.params_log_details[-1][3:] == ("carrying_capacity", 1000.0, 500.0)
pop.restore_checkpoint(1)
assert pop.params.carrying_capacity == 1000.0
assert pop.history.ticks == (1,)
assert pop.params_log_details == ()
assert pop.history.boundary_metadata == ((1, 0, "Ready"),)
```

## 8. How Hooks Integrate into the Execution Pipeline

User-defined Hooks (e.g., `first`/`early`/`late`) are compiled and merged into the execution flow, then triggered by the runner at the corresponding stage.

This provides two benefits:

- The high-level API remains clean and simple to use.
- Execution maintains a consistent stage order, making results more explainable.

## 9. Recommended Usage Patterns

1. Batch simulations: Prefer `pop.run(...)`.
2. Single-step observation: Use `pop.run_tick()`.
3. Trajectory analysis: Combine `record_every` with `pop.history`.
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
history = pop.history.individual_count

# 5) Export and restore
state_flat = pop.export_state()
pop.import_state(state_flat)
```

## 11. Chapter Summary

The execution mechanism of NATAL can be understood as a three-layer division of labor:

- Population layer: Provides a stable user API and lifecycle management.
- Runner/Hook layer: Organizes stage flows and event logic into a unified execution chain.
- Kernel layer: Performs numerical computations for each stage.

In practical modeling, you typically only need to use the population API consistently and enhance controllability and explainability through Hooks and history when needed.

---

## Related Sections

- [PopulationState and ModelDraft](4_population_state_config.md)
- [Modifier Mechanism](3_modifiers.md)
- [Hook System](2_hooks.md)
