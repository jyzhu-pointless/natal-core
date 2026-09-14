# Changelog

## Unreleased

### Breaking Changes

- **Every engine now defaults to Beverton-Holt density regulation**. Omitting
  `growth_mode` / `juvenile_growth_mode` used to select `NO_COMPETITION` for
  discrete-generation populations (unbounded growth), `LOGISTIC` for
  age-structured ones and `LOGISTIC` for spatial discrete ones. All three entry
  points now default to `BEVERTON_HOLT`. Any model that never set the knob
  changes behaviour: discrete models stop growing without bound and converge to
  the carrying capacity, and age-structured models that oscillated under the
  logistic curve now settle. Set `growth_mode="no_competition"` (or
  `"logistic"`) explicitly to keep the previous curve — the modes themselves are
  unchanged. `BEVERTON_HOLT` is `g(x) = r / (1 + (r - 1) x)`: positive for every
  finite `x`, and `|g'(1)| < 1` for `r > 1`.
- **`competition_strength` no longer defaults to 5.0 on the spatial path**.
  `SpatialPopulationBuilder.competition()` used to inject 5.0 as the weight of
  the second juvenile age class while every other entry point left the whole
  competition-weight vector at 1.0. The default is now uniform: an unset
  `competition_strength` means 1.0 everywhere, so age 1 carries the same weight
  as age 0 unless the value is set explicitly. Spatial age-structured models
  that relied on the implicit 5.0 weight change dynamics; pass
  `competition_strength=5.0` to keep them.

### Fixed

- **A missing required gamete label is now a build error instead of a silent
  no-op**. `Wolbachia` inherits through a gamete label (`"wolbachia"` by
  default); when the species did not declare it, the preset registered no
  modifiers, raised nothing and the simulation ran with transmission silently
  absent while the fitness patch still applied. The missing label is now
  reported with the label name.
- **`competition_strength` on a model without a second juvenile age is
  rejected**. It writes `age_based_relative_competition_strength[1]`, so with
  `new_adult_age == 1` (every discrete model, and 2-age age-structured ones)
  age 0 is the only juvenile age and its weight is fixed at 1.0 — the value
  never reached the kernels.

### Documentation

- **Stale root-level plan documents removed**: `lifecycle-tick-unification-design.md`
  (the unified lifecycle tick has shipped), `P5_EXPLORATION.md` (superseded by the
  Rust callback path; its `research/p5_cfunc_bridge/` spike is gone),
  `RUST_BACKEND_PLAN.md`, `RUST_MODULE_ORGANIZATION_PLAN.md` (landed as
  `refactor(rust): organize kernel, session, model, output and hook modules`),
  `DOCS_CONSISTENCY_REPAIR_PLAN.md` (its D1–D7 items landed) and
  `RUST_ONLY_REFACTOR_PLAN.md`.
- `RUST_BACKEND_IMPLEMENTATION.md` is now a short index. Its previous body
  described Numba as the default backend beside Rust, pre-reorganization module
  paths and APIs that no longer exist; the maintained description lives in `docs/`.
- The frozen contracts recorded by the retired `RUST_ONLY_REFACTOR_PLAN.md` are
  carried over below; the tests and scripts that cited its sections now cite this
  section. The retired plan stays readable in history:
  `git show f04848f:RUST_ONLY_REFACTOR_PLAN.md`.

### Frozen contracts (carried over from the retired Rust-only refactor plan)

- **User surface (plan §2.1)**: the chained configuration API syntax; the
  `Species.from_dict` chromosome / locus / allele / sex-chromosome / label /
  recombination-rate declarations; preset rules (species binding, idempotent
  registration, priority, modifier order, fitness composition, reconfiguration
  semantics); and the declarative hook format (`Op.*`, `.hooks(...)`, selectors,
  condition expressions, events, priority, `every`/`start` scheduling, deme
  selection). Only the syntax is frozen — internal classes, inheritance, caches,
  array layouts and old import paths are not. The removed `backend=` selector is
  the explicit exception.
- **Preset semantics (plan §5.3)**: species binding, idempotent registration by
  object identity, priority ordering, manual-modifier ordering, dose effects,
  labels, sex and compressed-axis semantics are frozen. A preset reconfiguration
  rebuilding fitness and thereby overwriting manual fitness writes is accepted
  behavior and must not change silently.
- **One hand-written parameter inventory (plan §5.4)**: `src/natal/parameters.jsonc`
  owns names, aliases, types, bounds, target section, shape, writable phase and
  derived dependencies; the Rust mirror is generated from it and must not be
  hand-written twice. Rust re-validates Python input rather than trusting Python
  pre-checks; `apply` validates the whole batch before committing and
  `tensor_write` validates the whole candidate tensor before replacing; one public
  method call is one transaction, and chained calls are not jointly atomic.
- **Stop, error and run phases (plan §7.4)**: the accepted first/early/late stop
  short-circuit and the rule that a stopped population needs `reset()` before it
  can run again are frozen, as is keeping the modifications already in effect at
  that boundary. A failed parameter transaction changes no target value; a failed
  Python callback commits neither its candidate state nor its parameters and rolls
  back the RNG it consumed; a run error marks Failed without promising an automatic
  rollback to the start point. Nested runs, external writes during a run and stale
  hook contexts are rejected; Rust validation errors map to specific
  `ValueError`/`TypeError` rather than a catch-all `RuntimeError`.
- **Checkpoints and restore (plan §9)**: a checkpoint holds tick, phase cursor, run
  status, individual/sperm arrays, ecology parameters (including migration and
  custom), per-deme RNG, log position and structure/program compatibility. Genetic
  tensors do **not** roll back — bit-exact replay only holds while genetics and the
  program are unchanged. `restore_checkpoint(tick)` locates the record at the
  recoverable raw boundary, validates all inputs and compatibility, then atomically
  replaces state/ecology/RNG/cursors and truncates future history and the parameter
  log; a normal runnable boundary restores Ready instead of a leftover
  Stopped/Failed. Incompatible blueprints or programs are rejected without changing
  the session. `reset()` and restore are separate, and importing counts is not a
  checkpoint restore (no historical RNG). Checkpoints do not capture Python
  closures or external state.
- **History lifecycle (plan §8.3)**: History keeps only the query / label / export
  surface on the Python side; the Rust HistoryStore owns the values and schema and
  slices or aggregates before returning. Queries default to independent arrays, and
  an array handed out earlier must not change or dangle after run, clear, restore or
  ring eviction. The parameter log appends only on a successful commit where the
  value actually changed, carrying tick, event/phase, deme, parameter name, old and
  new value, through one commit path shared by direct writes, updates, preset
  recompiles and both hook kinds. `clear_history()` clears history and its retained
  checkpoints only — not current state, parameters or RNG. `record_snapshot()` after
  a stop stores explicit phase metadata for an unfinished tick instead of passing it
  off as a complete row.
- **Contract ledger (plan §11, S0)**: three machine-checkable ledgers — must-exist,
  must-not-exist and invariants — distinguish pre-existing known defects from new
  regressions.
- **Performance freeze (plan §13.1)**: scenarios, machine and thresholds are frozen
  at S0 time; a >10% regression in median wall time or peak memory on any scenario
  is a blocking finding, and thresholds must not be relaxed after seeing results.

## v0.3.0b0 (2026-09-13)

Release wheels bundle the Vue dashboard. Installation checks verify the actual
HTML, linked assets, and API outside the source checkout; Node.js is only needed
when building from source.

### Breaking Changes

- **Python 3.10 or later is required**. Release wheels cover CPython 3.10–3.13
  on Linux x86_64/ARM64, macOS Intel/ARM64, and Windows x86_64.
- **Conversion rules have one filter/target vocabulary**: use the gamete or
  zygote allele-conversion and whole-type-conversion rule families. Whole-type
  targets use `genotype@label`, with `*` preserving either component. Replace
  the retired redirect rules and legacy filter fields with the documented
  `filters` API. Legacy colon-separated genotype/label strings are rejected.
- **Published model layouts are fixed**: compilation starts with the complete
  species baseline, and publication projects all state and genetics arrays
  onto one consistent runtime layout. Runtime updates that make a pruned type
  newly reachable fail without partially committing the update. Declare the
  required reachable states before publication or build a new population.

- **The NiceGUI dashboards are removed**: `natal.frontend.ui` (Dashboard /
  PopulationDashboard / SpatialDashboard / launch), the `nt.ui.*` exports,
  and the `nicegui[highcharts]` dependency are gone. `launch_vue` (Vue 3 +
  FastAPI) is the interactive surface; the visualization helpers it still
  uses live inside `natal.frontend.webui`.
- **Migration-era package keys removed**: `natal.hooks`, `natal.data`, and
  the other pre-Phase-0 short keys now raise `AttributeError`. Import from
  the real paths (`natal.frontend.hooks`, ...) or the top-level lazy API.
- **Configurator → PopulationBuilder / RuntimeUpdater**: the `Configurator`
  chain is renamed back to `PopulationBuilder` (`from_species().setup()...`
  `build()`), spatial construction moved to
  `SpatialPopulation.builder(...)`, and runtime modification goes through
  `pop.update()` / `deme.update()` returning one shared `RuntimeUpdater`.
  The `natal.configurator` package is deleted.
- **Runtime hook registration removed**: `register_hooks()` and the whole
  post-registration machinery are gone. Hooks are declared on the build
  chain (`.hooks(...)`) and packed once at build; per-population factories
  and the `_pop_ref` / `_hook_context` internals no longer exist.
- **Model layer split**: `NormalizedModel`, `CompiledModel`, `_ComputedMaps`,
  and `RunProgram` are deleted. `natal.frontend.model` holds the frozen
  `ModelDefinition` (declaration snapshot) and the build-time `ModelDraft`.
- **Kernel Config snapshot layer removed**: the Rust kernels read the
  Blueprint / Params / genetics tensors directly instead of a per-tick
  rebuilt Config snapshot (`rust/src/kernels/config.rs` deleted).
- **`extreme_speed_mode` is chain-only**: `setup(extreme_speed_mode=...)`
  on the discrete-generation chain (modes: 3 deterministic, 1 multinomial,
  2 Poisson); the age-structured entry rejects non-zero, and the
  documented public low-level construction path is retired.
- **`DemeSlice` aligned surface**: `spatial.deme(i)` returns an explicit
  view of exactly the 15 `Population`-aligned members plus
  `index` / `write_ecology` / `write_genetics`; dynamic proxies and
  `_minimal_contract` are gone, and unlisted attributes raise
  `AttributeError`.
- **Undocumented exports removed without aliases**: the `_PUBLIC_EXPORTS`
  list is the whole public top-level API.
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

### Removed

- **Backend-selection documentation and tooling**: `docs/{en,zh}/4_backend_selection.md`
  and the API pages for the reference simulators are gone; guides now describe
  the single native engine. `demos/bench_backends.py` is deleted, the three
  `benchmarks/rust_backend_*.py` scripts measure the engine's own `run(n)`
  vs `run_tick()` paths, and the demos no longer pass the removed
  `backend=` selector. The MGDrivE1 cross-engine benchmark family keeps its
  validation/statistics plumbing but its engine entry points now document
  that they raise on invocation (the reference engine retired).
- **Directory auto-discovery of top-level exports**: `natal`'s public API is
  an explicit `_PUBLIC_EXPORTS` list; a module export reaches the top level
  only by being added to that list. `__init__.pyi` is generated from it.

### New Features

- **`launch_vue` — Vue 3 + FastAPI dashboard**: real-time curves,
  per-genotype inspection, hooks / genetics-matrix panels, a spatial hex
  landscape with click-to-inspect demes and a migration panel, and a debug
  tab (event log, parameter audit, between-tick state diff, raw arrays).
  The simulation runs server-side; closing the browser keeps the run going.
- **Composable individual selectors**: add immutable `IndividualSelector`
  rules over ZType, sex, and age coordinates.
- **Structured history storage**: add immutable schemas, bounded history,
  read-only result ownership, post-hoc observation, and lifecycle-safe state
  restoration.
- **Op-level `event` / `priority` on every factory**: `scale`, `set_count`,
  `add`, `subtract`, `kill`, `sample`, and the `stop_if_*` family accept
  `event=` and `priority=` like `Op.set_param` / `Op.convert` already did.

### Performance

- **Spatial ticks without the kernel Config layer**: reading the contracts
  directly cut spatial tick time by roughly 35% on the drive benchmarks.
- **Light direct-write refreshes**: `deme(i).write_ecology(...)`, the
  per-deme routing of `pop.params.tensor_write(...)`, and
  `deme(i).write_genetics(...)` hand the engine a source carrying only the
  named fields instead of a fully materialized contract (which copied
  every genetics table per write). Numeric outputs are bit-identical
  (pinned by the digest baselines).

### Bug Fixes

- **Sex-chromosome identity and validation**: preserve distinct XY/ZW chromosome
  identities through genotype parsing, serialization, indexing, and inheritance.
  Validate completed species structures before compiling a population.
- **Conversion probabilities and sequencing**: apply whole-type conversion
  rates, including label-only targets, and evaluate zygote `current` filters
  against the state produced by preceding rules. Repeated modifier refreshes
  compile from the species baseline rather than compounding previous changes.
- **Compressed and spatial lifecycles**: keep shared reachable-state closure,
  labeled sperm storage, fitness arrays, observation indices, and runtime
  refreshes aligned with the published layout. Zero-rate targets can remain
  pruned without breaking a later unchanged refresh.
- **Sex-specific survival**: apply zygote and juvenile survival on the correct
  sex axis, including stochastic Poisson thinning.

- **Manual `trigger_event("finish")` semantics**: a manually fired finish
  event is a rehearsal — `is_finished` now reads false during and after
  the event (it used to flip true-then-false because the session never
  reached Stopped). The production finish paths (finish_simulation,
  hook STOP, the spatial stop path) keep their true/true answers.
- **Selector value edges**: NumPy integer sex and age values are accepted
  like any `numbers.Integral`; booleans are rejected (``True`` used to
  silently mean sex 1); an empty sex label raises with the same
  "use None for a wildcard" guidance as an empty container.
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

- **Shared local and remote checks**: `scripts/ci_full.py` runs the same stages
  used by GitHub Actions. Release wheels are installed in isolated environments
  and tested before their exact artifacts are uploaded. Manual release runs
  default to a dry run, and release tags must match package versions.
- **Complete compilation without a complete offspring tensor**: derive the
  offspring tensor only for the final runtime axes. Frozen published registries
  prevent later registration or recompression from invalidating runtime indices.

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
- **Hook priority is op-level data with assignment resolution**: a call-level
  `.hooks(..., priority=P)` assigns one shared priority to the op items of
  that declaration (single ops included); without it the ops' own priorities
  are used and must agree within a packed list (mixed declarations raise
  `ValueError` at build time).  This fixes two silent drops: the call-level
  priority never reached a bare op, and an op-level priority (e.g.
  `Op.set_param(..., priority=5)`) was discarded when the op was packed into
  a list.  `.hooks()` now records `priority=None` (no assignment) instead of
  defaulting the declaration to `0`; decorated callbacks keep their decorator
  priority and are not reachable by the call-level assignment.

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
