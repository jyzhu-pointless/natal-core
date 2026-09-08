# Changelog

## Unreleased

### Breaking Changes

- **The Rust engine is the only execution backend**: the pure-Python
  reference package (`natal.backends.reference`) is deleted together with
  the `backend=` selector and the `disable_rust_backend` /
  `refresh_rust_backend` / `using_rust_backend` facades;
  `enable_rust_backend` remains the engine-session init entry, called by
  `build()` and lazily at the first run/tick boundary. Populations that
  skipped `build()` (clones, direct `SpatialPopulation` construction)
  create their session from their current state on the first run.
- **Numba backend removed**: the `natal.numba` package, the `backend="numba"`
  selector, `njit_switch`, `enable_numba/disable_numba`, the numba cache and
  codegen pipeline are gone. `backend="numba"` raises `ValueError` with a
  migration hint.
- **Spatial `pop.update()` chain removed**: `SpatialPopulation.update()` and the
  private `_SpatialUpdate` facade are gone; runtime spatial writes go through
  `pop.params.tensor_write(...)` and `deme(i).write_ecology(...)` /
  `write_genetics(...)`.
- **History / Observation API**: replace mutable runtime observation creation
  and legacy output helpers with a canonical build-time `Observation`, a
  self-describing `History`, `pop.observe()`, `pop.record_snapshot()`, and raw
  checkpoint restoration. Deleted legacy interfaces are not retained as
  compatibility aliases.
- **Forwarding shims removed**: the legacy top-level packages
  (`natal.data`, `natal.hooks`, `natal.engine`, ...) are gone; import from
  `natal.frontend.*`, `natal.backends.*`, `natal.contracts`, or the top-level
  lazy API (`nt.Op`, `nt.Species`, ...).

### New Features

- **Composable individual selectors**: add immutable `IndividualSelector`
  rules over ZType, sex, and age coordinates.
- **Structured history storage**: add immutable schemas, bounded history,
  read-only result ownership, post-hoc observation, and lifecycle-safe state
  restoration.

### Bug Fixes

- **Spatial runtime updates**: validate complete per-deme updates before commit,
  preserve shared configuration identity, and propagate replacement configs to
  every affected deme without leaving partial state on failure.
- **Preset modifier refresh**: rebuild gamete and zygote conversions from the
  Mendelian baseline so repeated refresh or runtime reconfiguration cannot
  compound drive rates; preserve build-time preset registration after `build()`.
- **Discrete Poisson lambda ceiling**: the Rust Poisson helper now returns the
  mean for lambdas at or above the library sampling ceiling instead of
  panicking between that ceiling (1.844e19) and the 2^104 resolution guard;
  a stochastic run whose per-pair egg total lands in that window (e.g. a
  census explosion under a large-scale configuration) now completes.
- **Spatial RNG streams**: per-deme streams were rebuilt from `seed ^ deme`
  every tick (and stochastic migration re-seeded per call), reusing identical
  random numbers across ticks; streams are now a persistent per-deme bank
  advancing across ticks.  Same-seed reproducibility and segmented-run
  bitwise identity hold; stochastic spatial trajectories differ from the
  defective old streams (plan R1).

### Changed

- **Spatial update internals**: replace the private `_SpatialUpdate` facade and
  method-name batching table with typed Configurator dispatch and explicit
  `batch_setting()` values.
- **Hook program literal pool rebasing**: declarative `set_param` value
  expressions carry per-hook RPN literal indices; concatenating hooks into
  one program (panmictic and spatial builders) now rebases those indices
  onto the shared pool, so a second literal-bearing hook no longer
  evaluates an earlier hook's literal.
- **Bounded recording memory (plan S4)**: plain populations now wire their
  `max_history` bound (default 5000 rows) into History; `record_history(max_rows=None)`
  applies the population default instead of unbounded growth, and evicted
  history rows drop their paired session checkpoints (plain, discrete, and
  spatial) so the checkpoint store stays bounded by the same budget.
- **Spatial full checkpoint restore (plan S4 CheckpointStore)**: the spatial
  session now stores restorable boundaries (stacked state, every per-deme RNG
  stream, and the ecology columns) at record-aligned raw-history ticks;
  `restore_checkpoint` rewinds state, randomness, and ecology so
  `restore -> run` replays the original stochastic trajectory bitwise, and
  restores the runnable state after a stop.  Demes whose drafts project the
  rolled-back ecology read the checkpoint values.
- **Python callbacks on the spatial Rust path**: deme hooks (``@nt.hook``)
  now run inside the spatial session's ticks — stable deme-order execution,
  private per-fire array copies, graceful stop, and ``ctx.update()`` param
  writes deferred to the next tick (plain-backend semantics).  The former
  "keep the reference backend for callback hooks" refusal is gone.
- **One spatial session for every model**: discrete-generation spatial
  populations now share the session-owned heterogeneous kernel with
  age-structured (per-deme RNG banks, declarative hooks and Python-callback
  registration, migration inside the tick).  The hook-less per-config-bank
  discrete backends, the homogeneous `SpatialEngineSession` /
  `RustSpatialLifecycleBackend` pair, and the `RustSpatialLifecycleBackend.run`
  state round trip are deleted; discrete spatial `reset()` now also restores
  the random source.  Fixes defect R2 (declarative hooks were silently
  skipped on the discrete spatial Rust path).
- **Rust spatial session owns the run state**: the heterogeneous session holds
  the stacked counts, sperm storage, tick, and per-deme RNG bank;
  `RustHeterogeneousSpatialLifecycleBackend.run(ind, sperm, tick)` is replaced
  by control-only `run_tick()` plus `state_snapshot()` / `set_state()` /
  `set_deme_state()` / `set_migration_rate()`; lifecycle then migration run
  inside Rust with the zero-rate skip preserved.  A hook stop now freezes the
  tick keeping the boundary state instead of raising `RuntimeError`.  Spatial
  `deme.state` returns an independent snapshot (the live write-through is
  retired — `deme.import_state(...)` is the write channel), and
  `SpatialPopulation.reset()` reseeds the RNG bank.

## v0.2.0b (2026.7.14)

### Breaking Changes

- **Parameter rename**: `expected_num_adult_females` → `expected_num_new_adult_females` (Configurator + `PopulationConfig`).

### New Features

- **Spatial index compression** (#32): unified-registry `compress()` prunes unreachable ztypes/gtypes across demes via BFS from seed genotypes; combined modifier maps keep drive-reachable types alive. Adds `bench_compress.py`.
- **Modifier system refactor** (#33):
  - **ztype/gtype naming**: `GameteHaploidGenomeConversionRule` → `GameteGtypeConversionRule`, `ZygoteGenotypeConversionRule` → `ZygoteZtypeConversionRule` (old names kept as aliases); `add_hg_convert` → `add_gtype_convert`.
  - **Matrix compilation**: `RuleSet.to_matrix()` compiles rules into dense transition matrices; a single `freq_vec @ M` replaces the Python rule-cascading loop.
  - **Declarative `Condition` DSL**: `sex()`, `ztype_has()`, `slab()`, `is_maternal()`, `is_paternal()`, combinable with `&` / `|`.
  - **New independent rules**: `GameteGlabConversionRule`, `ZygoteGlabRedirectRule`.
  - **Presets migrated** to the declarative base (`CytoplasmicPreset`, `Wolbachia`, `TransgenicBackground`, `HomingDrive`, `ToxinAntidoteDrive`); ~129 lines of dead code removed.

### Bug Fixes

- **#34**: zygote modifier `IndexError` when `somatic_labels > 1` — the argmax ztype index is now mapped back to its `Genotype` before indexing.
- **#36**: gamete modifier read the wrong ztype index with slabs > 1, causing the cargo allele to vanish — now indexes via `ztype_index(genotype, default_slab)`.
- **#37** (#38): compact spatial hook plan + copy-on-write shared storage.
- **perf**: removed `prange` from `_execute_single_csr_hook` to fix a regression.

### Changed

- `declared_zygote_types` parameter relaxed from `set` to `Sequence`.

### Documentation

- Expanded `AgeStructuredPopulation.setup()` docstring.
- Cleaned up drive-ridl demos; removed unused resistance params.

---

## v0.2.0a (2026.7.8)

### Breaking Changes

- **Module reorganization**: flat file structure → 17 subpackages. All import paths changed; `import natal as nt` remains backward-compatible.
- **Builder → Configurator**: `PopulationBuilder` replaced by `Configurator` chain API. Old Builder accessible via `legacy_path=True`.
- **Hook signature**: unified to `(state, config, deme_id=-1)`. Legacy 2-arg `(ind_count, tick)` removed. Custom hooks require `custom=True`.
- **Parameter rename**: `female_age_based_survival_rates` → `female_age_based_survival` (all `_rates` suffixes dropped).
- **Default survival**: age-structured models now default to 100% survival at all ages (`np.ones(n_ages - 1)`).
- **Species default**: `Species.unordered=True` by default — `A|a` and `a|A` are the same genotype.
- **Scale system removed**: `population_scale`, `base_carrying_capacity`, `base_expected_num_new_adult_females` and related getter/setter methods removed. `carrying_capacity` is now a direct 0-d ndarray.
- **SpatialBuilder → SpatialConfigurator**: old `SpatialBuilder` class removed.
- **Migration rate type**: `SpatialPopulation.migration_rate` returns `NDArray[float64]` (was `float`). Scalar rates apply only to adults.
- **V1 discrete lifecycle removed**: old template and kernels deleted; only V2 (`DiscretePopulationConfig`-based) path remains.
- **`gamete_labels` removed** from `AgeStructuredPopulation.setup()`.
- **Private `_nn` properties removed**: `_state_nn` / `_config_nn` / `_registry_nn` → public `state` / `config` / `registry`.

### Deprecations

- **Builder API** (`population_builder.py`): accessible via `setup(legacy_path=True)`.
- **`use_sperm_storage`**: emits `FutureWarning` — never functional; sperm storage always enabled.
- **`generate_numba_setter()`** and `_param_setter.py` removed.

### New Features

- **Configurator runtime modification**: `pop.update().competition(carrying_capacity=5000)` writes immediately to config arrays, no rebuild needed.
- **`reconfigure_preset()`**: modify registered preset parameters at runtime without double-application — `pop.update().reconfigure_preset(drive, homing_rate=0.95)`.
- **Unordered genotype canonicalization**: genotypes auto-normalized to canonical form when `Species.unordered=True` (default).
- **Wright-Fisher normalization**: improved discrete-generation sampling for small populations.
- **ZType registry refactoring**: flat dict-based indexing, complete genotype→ZType repair.
- **`fitness()` method on Configurator**: direct fitness writes with `replace`/`multiply` modes, `@slab`-aware patterns.
- **Fitness system** (`fitness/` subpackage): single-writer architecture via `apply_preset_fitness_patch` — all fitness modifications (viability, fecundity, sexual selection, zygote viability) go through one entry point.
- **Performance**: dedup haploid lookup, O(1) glab lookup, dead code removal.
- **Per-age viability**: `fitness()` and Configurator now support per-age viability arrays.
- **Preset priority**: `GeneticPreset` and its subclasses (`HomingDrive`, `ToxinAntidoteDrive`) now accept a `priority` parameter for deterministic modifier ordering.
- **Spatial runtime modification**: `pop.update()`, clone-on-write per-deme configs, `batch_setting()` helper.
- **Parameter descriptor registry** (`utils/parameters.py`): declarative registry of all configurable parameters with names, aliases, and domain grouping — powers `set_param()` and Configurator chain methods.
- **New demos**: `demo_config_and_params.py`, `demo_hook_modify_config.py`, `bench_numba.py`, `bench_compress.py`.
- **`ZygoteTypePattern.from_slab_key()`**: public helper to parse `"genotype@slab"` keys.
- **`equilibrium_individual_distribution`**: new field on `PopulationConfig` for custom equilibrium distributions.
- **Hook selector mode**: `mode="auto"|"expand"|"aggregate"` on `@hook` decorator — controls how selector keys map to hook function parameters.
- **Nested `sexual_selection`**: `{female_selector: {male_selector: val}}` format now supported in `configurator.fitness()`.
- **`set_config()` on `BasePopulation`**: new public method to replace a population's configuration at runtime.
- **`Configurator.from_species(species, discrete=True)`**: unified factory; `for_discrete()` and `for_age_structured()` are now one-line shorthands.
- **`Configurator.for_population()`**: static factory wiring a Configurator to an existing population for write-back support.
- **`initial_state()` on `DiscreteConfigurator`**: accepts flat JSON-style dicts for discrete-generation models.

### Hook System

- **Restructured**: `entry/` (declarative + decorator), `compile/` (template-driven codegen), `runtime/` (CSR + njit execution).
- **Unified dispatch**: mixed CSR + njit hooks now share one execution path.
- **`RESULT_SKIP`**: new return code disambiguates guard-skip from success-continue.
- **LifecycleWrappers** moved from hooks to `engine/lifecycle_wrappers.py`.
- Custom hooks intro added to `2_hooks.md`.

### Population

- **BasePopulation**: slimmed from ~1,743 to 532 lines; extracted `HookManagerMixin`, `ModifierPresetMixin`, `ObservationMixin`, `OutputMixin`.
- `update()` returns typed Configurator (`DiscreteConfigurator` / `AgeStructuredConfigurator`).

### UI

- Spatial dashboard overhaul with shared `ObservationPanel` helpers.
- Logo updated to new design.

### Rebrand

- NATAL expansion: **N**umba-**A**ccelerated → **N**umerical **A**ggregation. Emphasizes the aggregation modeling paradigm (group-level computation with statistical sampling) over the implementation backend. Current backend remains Numba.

### Documentation

- 18 stale root-level docs removed; superseded by `en/` / `zh/` versions.
- API reference restructured: 12 `:::` directives corrected, 6 new subpackage docs added, index reorganized by 17 subpackages.
- `simulation_kernels` → `simulation_engine` rename propagated throughout docs and configs.
- Genotype ordering descriptions corrected throughout: `|` vs `::` semantics updated for default `unordered=True`.
- MkDocs build fixed for Python 3.13 (pygments `None` filename bug).
- Logo and favicon added to all mkdocs configs.

### Testing

- **Numerical verification hardened** across 8 commits: weak assertions (`> 0`, `hasattr`, `print`) replaced with exact counts, `pytest.approx()`, `np.isfinite()` bounds, probability distribution invariants (`sum ≈ 1.0`), and shape validation. Affected files: spatial, algorithm, sampling, configurator update, population simulation, preset/modifier/hook-slab, and pattern tests.
- Non-numba test path (`NATAL_DISABLE_NUMBA=1`) fixed: 22 failures resolved.
- `test_hook_selector_mode.py`: `_FakeRegistry` replaced with real `Species` + `IndexRegistry`.
- `test_spatial_population_integration.py`: `_replace(raw_float)` replaced with `pop.update()`.
- New test suites: `test_unordered_genotypes.py`, `test_wright_fisher.py`, `test_base_population.py`, `test_hook_declarative.py`, `test_modifiers.py`, `test_parameters.py`, `test_population_state.py`, `test_observation_record.py`, `test_configurator.py` (strengthened), `test_lifecycle_wrappers.py`, `test_hook_executor.py`.

---

## 2026.4.29 (v0.1.3)
- **feat(spatial-topology)**: add `build_gaussian_kernel` public API with hex/square distance metric
- **feat(observation)**: add `CompactMeta` and `observation_record` module; integrate with builder and kernels via `record_observation` / `observation_mask`
- **feat(spatial-builder)**: add `SpatialBuilder` with `_replace` optimization for heterogeneous configs; support `with_observation`
- **feat(migration)**: add `normalize_kernel` option for boundary-aware migration; split kernel function and add heterogeneous routing
- **refactor(competition)**: separate carrying capacity from expected egg counts, remove backward-propagation bug
- **refactor(hooks)**: replace codegen with lifecycle wrappers for Numba caching; fix panmictic deme_id default to `-1`
- **refactor(observation)**: remove deprecated `unordered` parameter; use `CompactMeta` for spatial observation history export

## 2026.4.24 (v0.1.2)
- **feat(genetic_entities)**: check for duplicate gene names in species
- **feat(genetic_structures)**: add `Chromosome.get_locus`, `Species.get_gene/has_gene`; warn on duplicate names; fix recombination rate handling on position reorder/insertion
- **refactor(population)**: rename `zygote` fitness args to `zygote_viability`; `run()` default `record_every` from `1` to instance attribute
- **refactor(hooks)**: remove `numba` from `@hook`, auto-detect njit; add `custom` flag to allow custom hooks

## 2026.4.20 (v0.1.1)
- fix(algorithms): ensure `n_virgins_raw` is clamped to 0.0 when in the range `(-EPS, 0)` to prevent intermittent negative virgin count errors due to floating point precision issues
- fix(hooks.executor): round `current_count` to the nearest integer before comparison to `target_count` in discrete stochastic sampling paths where `current_count` may be stored as a float
- fix(genetic_presets): support different modes (multiplicative, dominant, recessive, custom) for zygote viability scaling; rename `zygote_fitness` to `zygote_viability_fitness` for clarity

## 2026.4.19 (v0.1.0-rc.2, v0.1.0)
- Remove redundant `parallel=True` decorators from adjacency migration wrapper functions that do not contain `prange`
- Move probability related logic from `algorithms.py` to `numba_compat.py`
- Change the DNA pattern of the logo from left-handed to right-handed helix
- Fix dashboard favicon loading after wheel installation by resolving `natal.svg` from package resources at runtime
- Update the `index` and `quickstart` parts of documentation

## 2026.4.17 (v0.1.0-rc.1)
- Refactor Observation system: make Observation reusable and state-independent by removing dimension coupling from state validation
- Decouple `ObservationFilter` from state-specific logic; dimension validation now occurs at apply-time via `Observation.apply()`
- Refocus API documentation: position `Observation` and state translation output functions as primary user entry points
- Discourage direct user instantiation of `Observation`; recommend population-level convenience methods instead
- Add `output_current_state()` and `output_history()` convenience methods to `BasePopulation` as primary interfaces
- Enhance demo files with observation and translator usage examples: `observation_history_demo.py`, `mosquito.py`, `discrete.py`
- Demonstrate pattern string filtering in demos: use `"Dr::*"` and `"R2|*"` patterns to show flexible genotype matching
- Refactor HexGrid to use parallelogram coordinates instead of odd-r offset coordinates for simpler neighbor calculation
- Update spatial visualization to support parallelogram grid layout with continuous diagonal offset
- Improve colorbar layout: change to horizontal orientation at bottom to avoid overlap with landscape
- Implement dynamic colorbar range adjustment: only update when current max exceeds 110% of historical max
- Enhance user experience: clicking deme no longer automatically switches to selected deme page
- Update spatial dashboard with improved layout and stable visualization ranges

## 2026.4.13
- Add Zygote Fitness support: new fitness type applied during reproduction stage before survival and competition
- Extend PopulationConfig with zygote_viability_fitness field and set_zygote_viability_fitness method
- Update Builder system to support zygote fitness configuration via fitness() method
- Extend Genetic Presets system with zygote allele-based fitness scaling support
- Integrate zygote fitness application in simulation kernels with proper stochastic sampling
- Add comprehensive unit tests for zygote fitness functionality
- Update documentation for PopulationConfig, Builder system, simulation kernels, and genetic presets
- Fix GeneticPattern parsing issues: `enumerate_genotypes_matching_pattern` now correctly recognizes unordered homologous chromosome identifier `::`; fixed parsing of single-character gene syntax with omitted `/`

## 2026.4.10
- Refactor hook dispatch flow: move Python dispatch runners out of population classes into hooks executor helpers, and remove DiscreteGenerationPopulation internal _step_* helpers
- Unify hook execution policy when Numba is disabled: any registered hook type now uses one sequential Python dispatch path
- Rework SpatialPopulation hook aggregation to pin compiled hooks to owning demes and rebuild one consistent aggregate hook registry after set/remove operations
- Simplify spatial wrapper template to run migration-enabled spatial tick kernel directly; keep local lifecycle plus migration responsibilities explicit in kernel/docs
- Add heterogeneous deme-config support on the njit spatial path via per-deme config-bank id routing, while preserving deme-level parallel execution
- Enforce migration-time consistency for `is_stochastic` and `use_continuous_sampling` across demes, and update spatial simulation guides accordingly (EN/ZH)
- Route heterogeneous deme-config execution through the unified hook-aware spatial timeline so hook semantics stay consistent regardless of config heterogeneity

## 2026.4.9
- Correct carrying capacity (equilibrium metrics) handling in population builders
- Enhance sex chromosome handling and genotype compatibility in population dynamics
- Add support for heterogeneous kernel routing in SpatialPopulation
