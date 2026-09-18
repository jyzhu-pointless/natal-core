# Changelog

## Unreleased

### Changed

- `natal.Blueprint` carries a new `discrete_generation` flag, positioned
  between `has_sex_chromosomes` and `extreme_speed_mode` (23 fields, up from
  22). The frozen spec must say which engine it describes, because the
  equilibrium kernel reads per-age `fertility` differently in each (implicit
  1.0 vs `clamp01`). Keyword construction is unaffected; positional
  construction of the previously 22-field tuple is not. The field is derived
  from the draft the population was built from, not a user-settable parameter.
- The private `natal._engine_rs.equilibrium_metrics_flat` helper gained two
  required positional arguments (`has_sex_chromosomes`, `discrete_generation`)
  before its two defaulted sentinels; `src/natal/_engine_rs.pyi` matches the
  new order.

- The five pattern entries that match genetic content only —
  `Species.parse_genotype_pattern`, `Species.enumerate_genotypes_matching_pattern`,
  `Species.parse_haploid_genome_pattern`,
  `Species.enumerate_haploid_genomes_matching_pattern` and
  `GenotypePatternParser.parse_haploid_genome_pattern` — now reject an
  `@label` suffix with `PatternParseError`. The suffix used to be parsed and
  then never consulted, so a labelled query matched every label of the
  genotypes it named. `filter_*` and the selector resolvers inherit the
  rejection; the label-aware entries (`ZygoteTypePattern.parse`,
  `IndividualSelector`, conversion-rule `filters`,
  `GenotypePatternParser.parse_haplotype_pattern`) keep taking labels, and the
  new `GenotypePatternParser.require_unlabelled_pattern` is their shared guard.
- `GenotypePatternParser.parse` joins that list: it returns a `GenotypePattern`,
  which carries no label, so it now rejects `@label` instead of storing a
  suffix nothing reads. Content patterns no longer keep it at all —
  `GenotypePattern.lab` and `HaploidGenomePattern.lab` are gone (the latter was
  always `None`), and with them the `lab=` constructor argument. The label
  lives where it is matched: `ZygoteTypePattern.slab` and
  `GameteTypePattern.glab`. One visible consequence: `parse` returns the same
  cached object for `"A|a"` however a caller previously spelled the label,
  because the cache keys on the label-free spelling.
- `GameteTypePattern` pairs the gamete label with a complete
  `HaploidGenomePattern` through its new `glab` and `genome` attributes,
  replacing the flattened `HaplotypePath` + `lab` pair.
  `parse_haplotype_pattern` shares its content parsing with
  `parse_haploid_genome_pattern` (the new private `_parse_haploid_content`), so
  a multi-chromosome gamete selector describes each chromosome the way a
  content-only haploid pattern does; the old form merged every chromosome's
  loci into one path. `ZygoteTypePattern.from_slab_key` is removed: it had no
  callers, and `ZygoteTypePattern.parse` accepts the same `genotype@slab`
  spelling (without the exact-name genotype canonicalization).
- The `@` analysis has one spelling: `GenotypePatternParser.split_label_suffix`
  (renamed from the private `_strip_lab`), used by the label-aware entries and
  the conversion-target splitter. The `Species` genotype helpers no longer run
  their own copy of the content-only guard — the parser entry owns it — and
  the private `_parse_haplotype_path` no longer strips an `@` suffix: every
  entry resolves the label before splitting chromosomes, so that strip could
  never run on a label it was meant to remove.
- Conversion filter patterns are analysed once instead of three times: the
  strict validator owns the `@` scan and returns the label matcher, so a
  malformed label reports one message ("invalid filter label") on both the
  gamete and zygote stages instead of each module's own wording.

### Fixed

- The equilibrium calibration reads the offspring sex ratio the way the owning
  engine does: a species whose sex is determined by sex chromosomes ignores
  `sex_ratio` exactly as its tick already did, instead of letting a non-0.5
  value split the reference composition and move the equilibrium away from the
  declared carrying capacity (up to +56.9 % at 0.2 and -28.2 % at 0.7 in the
  reported probes; both engines shared the error).
- The equilibrium calibration consumes per-age `fertility` as the owning tick
  does: discrete generations use an implicit 1.0 (their tick reads no
  age-dependent fertility at all) and the age-structured path clamps the
  stored weight to `[0, 1]`. A raw `params.tensor_write("fertility", ...)`
  value outside the builder's domain no longer moves the equilibrium by the
  written factor.
- The Champer egg override (`competition(expected_num_new_adult_females=...)`)
  derives its total from the same per-age fertility weights the owning tick
  reads, so a discrete model ignores the stored tensor there and the
  age-structured path clamps it. Previously the raw stored values moved the
  override — and through it the realized equilibrium — by the written factor.

- A preset's `fitness_patch()` now rejects an unknown top-level key with a
  `ValueError` naming it and listing the supported keys. Unknown keys used to
  be skipped in silence, so the misspelled `viability_allele` in the documented
  example produced a completely ineffective patch.
- A preset's `fitness_patch()` honours an `@slab` label in a selector key,
  exactly as the `fitness()` chain does: only that slab is written, and a label
  no ZType carries is rejected. The label used to be dropped and every slab of
  the matched genotype was written.
- A spatial `presets()` or `hooks()` call that fails leaves neither a
  declaration-log entry nor a batch entry behind, and a failed call no longer
  overwrites a batch entry an earlier successful call committed.
- `initial_sperm_storage` input type errors raise `TypeError` instead of
  tripping an `assert`, so `python -O` no longer skips the checks and lets a
  malformed mapping reach an unrelated failure.
- A declarative hook that returns something other than a list, or a list
  carrying an element that is not a `HookOp`, raises `TypeError` at build time.
  It used to compile to a zero-operation hook and silently do nothing; an empty
  list is still a legal no-op.
- `compile_definition` rejects a declared fitness baseline that does not cover
  every field, and one whose shape does not match the draft, instead of
  silently truncating it or replacing it with `np.ones_like`.

## v0.3.0 (2026-09-16)

The first final 0.3 release retains the Rust engine and public model interfaces
from the beta series, with the numerical and spatial fixes below. Changes since
`v0.3.0b1` are listed here; users upgrading from 0.2 should also read the breaking
changes in the `v0.3.0b0` and `v0.3.0b1` sections.

The Population/Landscape separation and the broader builder, runtime-update,
preset, and hook redesign are deferred (TODO-019). Spatial Python callbacks
still run serially per deme; cross-deme global hooks and hook-driven migration
updates are not part of this release. The new density-regulation demo is an
independent design sketch, not a supported NATAL API.

Population-level readable exports currently require the registry genotype
labels to match the state axis. Multi-somatic-label states can raise a dimension
mismatch; use `pop.observe()` or project raw history through an Observation for
those models (TODO-020).

### Added

- Selector-based `Op.convert(from_=..., to=...)` changes selected individuals'
  genotype, label, age, or sex while conserving their counts and applying the
  documented sperm-storage transfer rules. `Op.clear_sperm_storage` clears
  selected females' stored sperm.
- Cytoplasmic presets use conversion rules with explicit source labels for
  gamete tagging and maternal inheritance.
- An independent density-regulation demo illustrates reference calibration,
  adult versus juvenile pressure, and interactions between two species.

### Fixed

- Homing-drive embryo editing is triggered by parental Cas9 deposition labels,
  including editing in offspring that do not inherit the drive. Maternal and
  enabled paternal channels act sequentially on the remaining target copies;
  inherited drive/Cas9 alone does not activate embryo editing.
- Staged discrete reproduction honors `fixed_egg_count`, disabling clutch
  Poisson noise while retaining the other enabled stochastic processes. Spatial
  discrete builders forward the flag and preserve a value supplied to `setup`
  when `reproduction` does not override it.
- The fused Wright-Fisher path applies ordinary age-0 survival and genotype
  viability after density regulation, matching the staged lifecycle's ordering.
  Its final-generation sampling still differs from staged sampling.
- Compensatory growth modes reject non-finite or subunit
  `low_density_growth_rate` values at construction and runtime updates.
- Spatial parameter views reject unsupported attribute assignments instead of
  accepting writes that leave the simulation unchanged. During spatial Python
  callbacks, population/deme snapshot reads now fail explicitly; use `ctx.state`
  and `ctx.metrics` to read the live callback state.
- Age-structure rebuilding requires a species-backed builder and preserves the
  species blueprint's genotype, label, and sex-chromosome layout.
- The RIDL batch demo's `--smoke --check` validates smoke outputs without
  comparing the reduced grid to the full-grid reference, including when the
  repeat count matches the reference. Full-grid reference checks are unchanged.

### Behavior Changes

- **The derived equilibrium reference now splits age 1 by the surviving sex
  ratio.** `equilibrium_metrics` divided the reference age-1 total with the raw
  offspring `sex_ratio`, so a model whose sexes differ in age-0 survival (for
  example `survival(female_age0_survival=0.9, male_age0_survival=0.8)`)
  calibrated against a composition it never reaches: the realised equilibrium
  missed `carrying_capacity` by 2.8%-37.5% (larger the closer
  `low_density_growth_rate` is to 1), and `ricker` could settle *below* K. The
  reference now uses `sex_ratio * s_f / (sex_ratio * s_f + (1 - sex_ratio) * s_m)`
  for the female share of the age-1 total, which makes the deterministic
  equilibrium exactly `K` for every compensatory curve in both engines. Models
  whose two sexes survive equally are bit-for-bit unchanged; recorded results
  for sex-asymmetric survival configurations shift once.
- **A non-zero `migration_rate` that cannot move anybody now warns.**
  Building a spatial population with a non-zero rate but no inter-deme edge —
  no `topology`, `kernel`, or `adjacency`, so the default adjacency is the
  identity matrix, or a kernel whose only non-zero weight is the excluded
  center — emits a `UserWarning`. Migration behaviour is unchanged: nothing
  mixed before and nothing mixes now.

### Documentation

- Declarative hook examples put `event` and `priority` on their `Op` objects;
  Python callbacks continue to use `@hook`.
- The bilingual developer guide covers the engine, model compilation, hooks,
  runtime updates, observations, and spatial execution.
- Initialization, spatial, and runtime-update examples include the required
  imports and model setup. Runtime guidance distinguishes writable parameters
  from build-time settings and documents per-deme fitness/preset declarations.
- The Wolbachia preset is documented as what it implements — a maternal
  cytoplasmic marker with optional per-slab fitness scaling — instead of
  claiming cytoplasmic incompatibility, with a pointer to the modifier recipe
  for building CI (`docs/*/4_index_registry.md`).
- `PointMutation`'s cascade section states the consequence of stacking
  single-target presets (a sequential model, `O(mu*nu)` away from the textbook
  two-way mutation equilibrium) and the exact rate correction that recovers the
  textbook recursion (`docs/*/2_genetic_presets.md`).
- The Wright-Fisher extreme-speed section states that the fused tick is its own
  model and that the staged path resamples the age-0 cohort, doubling the
  per-generation drift variance (`docs/*/2_population.md`).
- The population-initialization page records the surviving-sex-ratio
  composition premise of the derived equilibrium distribution.
- The conversion rule sets' docstring examples now show the working mount form
  (compile against the built population, then `pop.add_gamete_modifier(...)` or
  `add_zygote_modifier(...)`), which the pipeline accepts; the previous snippet
  raised `TypeError` at build time.

## v0.3.0b1 (2026-09-14)

This release turns three silent behaviors into explicit contracts: every engine
now defaults to Beverton-Holt density regulation, adjacency rows are read as
relative outbound weights, and rules must carry the age axis. It adds the
`PointMutation` preset, per-deme `migration_rate` declarations, and the missing
`RICKER` growth-mode constant. The wheel matrix and the release checks are
unchanged from `v0.3.0b0`.

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

- **Adjacency rows now mean relative outbound weights**. The builder
  row-normalizes every non-empty adjacency row to a probability vector before
  folding it into the migration CSR, so a sub-stochastic row (sum < 1, e.g.
  0.5) no longer drops the unrouted share every tick and a super-stochastic row
  (sum > 1) no longer creates mass. The default topology-derived adjacency —
  `build_adjacency_matrix(topology)` without `row_normalize` — hit exactly that
  path whenever `migration_rate > 0` was set without an explicit adjacency (for
  example `SquareGrid(1, 3)` has row sums `[1, 2, 1]`). All-zero rows (isolated
  demes) are left untouched and keep their mass at the source. A model that
  used a shrunken row to mean "migrate less" now migrates the full
  `migration_rate`; express that intent through `migration_rate` instead.
  `adjust_migration_on_edge` is now documented as the legacy ~1-ulp no-op it
  already was: the kernel fold had always divided each emitted row by its own
  sum, so a boundary deme already sent its full quota, just to fewer neighbors,
  each of which received a larger share.

- **`apply_rule` rules must carry the age axis**. The helper used to accept a
  3-D `(n_groups, n_sexes, n_ztypes)` rule (and a 2-D one) and infer that the
  rule had no age dimension. An age-collapsed selector mask and a genuinely
  age-free rule are the same array, so an OR'd selector could pass as an
  age-free rule and silently sum every age of the matched ZType. Every rule now
  keeps the age axis, degenerating to one class when the counts have none: a
  2-D `(sex, ztype)` count takes a `(n_groups, sex, 1, ztype)` rule, and a rule
  that drops the axis is rejected with a message naming the expected shape.
  `Observation.apply` accepts the same counts as before and now validates the
  mask through the same shared normalization.

### Added

- **Build-time `migration_rate` supports per-deme values**. A
  `batch_setting([...])` gives one rate declaration per deme (each element
  follows the scalar / age-vector / `(S, A)` / per-sex-mapping sugar), and
  `SpatialPopulationBuilder.migration()` also accepts the whole per-deme column
  directly: `(n_demes, S, A)`, or `(n_demes, n_ages)` broadcast across sexes —
  except that a 2-D shape of exactly `(S, A)` keeps its shared per-sex reading,
  so when `n_demes == n_sexes` the two 2-D shapes collide and only the 3-D
  column (or `batch_setting`) gives per-deme rates.
  Both land on `pop.params.migration_rate`, matching the existing runtime
  `params.tensor_write("migration_rate", ...)` channel. Boundary demes are now
  controlled through this rate, not through their adjacency rows.
- **`PointMutation` preset for spontaneous point mutation**. The new built-in
  preset converts a source allele into one or more target alleles in every
  gamete carrying it (no parent-genotype filter). The mutation is germline-only
  for now (the embryonic channel is deferred with TODO.legacy.md ARCH-021).
  Multi-target declarations compete: the preset compensates for the conversion
  cascade internally (`r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)`), so each target's realized
  share equals its declared rate instead of the earlier-declared target taking
  its mass first. Both modes reject non-finite and negative rates;
  `rate_mode="strict"` (default) additionally rejects rates above 1 and any
  declaration summing above 1, while `rate_mode="proportional"` reads them as
  proportions and scales them to 1. A declaration whose rates are all zero
  registers no conversion rule, and the preset also carries the allele-scaling
  fitness keywords (`viability_scaling`, `fecundity_scaling`,
  `sexual_selection_scaling`, `zygote_viability_scaling` and their `*_mode`
  companions), `effective_rates()` and the `source_allele` / `target_alleles`
  accessors. Available as `natal.PointMutation` /
  `natal.frontend.presets.PointMutation`.
- **`RICKER` growth-mode constant**. The fourth compensatory curve was reachable
  only through the string `"ricker"` or the bare integer `4` while the other
  modes had constants. `natal.RICKER` / `natal.frontend.model.RICKER` completes
  the set, and every constant now documents the curve it selects — `LOGISTIC`
  and `LINEAR` are the same curve under two historical names, and the numeric
  ids are the dispatch values of the Rust density-regulation kernel.

### Fixed

- **A species that declares none of the preset's maternal labels is now a build
  error instead of a silent no-op**. `Wolbachia` inherits through a gamete label
  (`"wolbachia"` by default); when the species declared none of the labels the
  preset matched, it registered no modifiers, raised nothing and the simulation
  ran with transmission silently absent while the fitness patch still applied.
  The unmatched labels are now reported by name. A preset that declares several
  maternal labels still only requires one of them to be present; the absent
  ones are not reported.
- **`competition_strength` on a model without a second juvenile age is
  rejected**. It writes `age_based_relative_competition_strength[1]`, so with
  `new_adult_age == 1` (every discrete model, and 2-age age-structured ones)
  age 0 is the only juvenile age and its weight is fixed at 1.0 — the value
  never reached the kernels.
- **`age_structure()` forwards the sex-chromosome masks**. The builder rebuilt
  its config from the species blueprint but carried only
  `has_sex_chromosomes`, so `female_only_by_sex_chrom` /
  `male_only_by_sex_chrom` were dropped and the draft fell back to the
  compatibility heuristic. With every baseline map row summing to 1 that
  heuristic cannot separate homogametic from heterogametic genotypes, and the
  kernel assigned sex from the compatibility ratio: roughly half of every
  sex-fixed genotype's mass landed on the wrong sex axis and was then handled
  with the other sex's survival and mating rates (measured 150 of 300 on the
  female ztype `X1|X1`). The masks are forwarded unexpanded, as
  `build_population_config` expects, and the mask-less combination is rejected
  there so no future rebuild path can silently reintroduce the fallback.
  `from_species()` was unaffected.
- **A zero equilibrium competition strength now extinguishes instead of
  disabling regulation**. `C*` is the reference point the compensatory curves
  are evaluated against. When it was zero the ratio guard substituted 1.0 (the
  neutral point of every curve) and the survival guard substituted 1.0, so the
  scaling was exactly 1.0 and regulation silently switched off. `C*` is zero
  exactly when the reference equilibrium carries no competing juvenile mass: a
  carrying capacity of zero, or a declared equilibrium distribution whose
  juvenile entries are all zero. A zero `eggs_per_female` is a third trigger
  only when `new_adult_age == 1` — every discrete model, and the 2-age
  age-structured ones — because age 0 is then the only competing age; from
  `new_adult_age == 2` on, the derived distribution still places
  `K · sex_ratio` juveniles at age 1, so `C*` stays positive. Such a model used
  to grow without bound (1000 adults reached 3,125,000 in five ticks) and now
  recruits nothing.

  Modes 2–4 therefore return a zero scaling. That matches `FIXED` only for the
  `K == 0` trigger: `FIXED` is evaluated against the carrying capacity rather
  than `C*`, so a positive `K` with an all-zero declared equilibrium still
  clamps at `K` instead of extinguishing (measured: 1000 after five ticks,
  against 0 for the compensatory modes). The divergence from the retired Python
  reference, which had the same guard combination, is deliberate. For a model
  whose only competing age is age 0 and which is driven purely from hooks
  (release/inundation runs) with `eggs_per_female == 0`, set
  `growth_mode="no_competition"` if the injected individuals must survive.
- **The word-vector ecology restore is now atomic**. `ecology_restore_words`
  wrote each field straight into the live section, so a field rejected
  part-way left the earlier fields already overwritten — while its docstring
  promised the clone-and-commit guarantee that `restore_ecology` provides. It
  now validates into a clone and commits in one assignment.
- **A zero or negative `sp_every` firing period is rejected**. The hook compiler
  enforces `every >= 1`, but `HookProgram` is a half-public wire type whose
  fields are public: a hand-built program reached `% every` with zero and
  panicked, and a negative period silently fired on multiples of its magnitude.
  Installing such a program now raises, and the interpreter returns an error
  rather than dividing by zero if one is assembled directly.
- **`collapse_age=True` observation histories can be read back**.
  `population_observation_history_to_readable_dict` reshaped every history row
  with the full age axis, but a collapsed recording writes one value per age
  class, so the conversion raised `ValueError: cannot reshape array of size 2
  into shape (1, 2, 2)` for any `n_ages > 1`. The row layout now follows the
  same collapsed/full distinction the recorder uses.
- **An out-of-range age key in `initial_state` is rejected instead of silently
  writing or dropping it**. The dict branch of the age-structured initial state
  tested only `age < n_ages`, and NumPy treats `-1` as the last index, so
  `{-1: 50.0}` landed on the oldest class while the sperm branch already raised.
  It now rejects both a negative index and a positive key at or beyond
  `n_ages`, which it used to drop without a word. The list branch keeps its
  existing silent drop.
- **Flattened-state parsing validates its length before slicing**.
  `parse_flattened_state` and `parse_flattened_discrete_state` accepted any
  1-D buffer and surfaced a truncated or oversized input as a NumPy reshape
  error (and an empty one as `IndexError`). The declared size is now checked
  first, so a malformed buffer fails with the expected and actual lengths, and
  a non-1-D input is rejected up front instead of surfacing a NumPy `TypeError`
  from an implicit scalar conversion.
- **`scripts/perf_freeze.py` runs to completion again**. Its scenario guards
  asked a `DemeSlice` for a `tick` attribute it does not expose, so the script
  aborted on the second scenario and the "frozen" performance gate protected
  nothing. The guard now checks the session tick, and a baseline recorded by a
  different extension build reports a warning and defers instead of raising a
  false blocking regression.
- **The deterministic migration engine now uses one bookkeeping order, so a raw
  sub-stochastic CSR row conserves mass**. The adjacency order used to subtract
  the full `value * rate` from the source and then deliver only
  `outbound * row_sum`, which lost (or, for a super-stochastic row, created) the
  remainder; the kernel order already distributed first and kept the
  `value - moved_total` residual. Both modes now use the kernel order, so the
  destination distribution is unchanged but the source keeps whatever was not
  delivered. Builder-folded rows are normalized to one, so models built through
  the public API are unchanged apart from last-ulp rounding of the source
  residual (`value - outbound` vs `value - Σ outbound·wᵢ`); the `phase0` spatial
  baseline stays bit-identical. For a build whose adjacency rows are already
  probability vectors — kernel routing, or adjacency mode after the row
  normalization described above — this ordering is the only difference. Raw
  hand-built CSR input is where the mass difference itself shows, while
  public-API models with un-normalized rows (the degree-weighted default among
  them) move because of the normalization change above. `stay_after_send`
  remains on the CSR for the frozen wire contract but no longer changes the
  numbers, and the retired "adjacency math reproduces the legacy Python order
  bitwise" claim is gone.
- **A kernel-routed spatial build no longer materializes the default dense
  adjacency**. The topology-derived default is an `(n_demes, n_demes)` matrix,
  and the row normalization added by the previous bullet reads and copies every
  element — so a model that routes through a migration kernel paid
  `O(n_demes²)` resident memory for a matrix the CSR fold never consults. A
  10,201-deme kernel model peaked at 1.71 GiB instead of 0.16 GiB, and
  `demos/spatial_hex_discrete.py` at its intended 501×501 grid (251,001 demes)
  requested hundreds of gigabytes and was killed by the OS during the build
  instead of running. The default matrix is now built only when adjacency
  routing will read it; an explicitly declared adjacency is still coerced and
  validated in either mode.

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
- **The Drive-RIDL remake demo is reproducible and self-checking**.
  `demos/drive_ridl_remake_batch.py` gained `--seed`, `--repeats`, `--smoke`,
  `--check`, `--no-plots` and `--write-reference` plus the `NATAL_RIDL_SEED`
  environment override, records the seed it used in a run manifest, writes a
  `#` provenance header on its four CSVs, and freezes or compares
  `demos/drive_ridl_remake_reference.json`. The quickstart demos
  (`mosquito.py`, `mosquito_ui.py`) dropped their stale
  `expected_num_new_adult_females` expectation.

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

### Validation

- Full test suites on Python 3.10, 3.11, 3.12, and 3.13, plus `ruff`, `pyright`,
  the generated public stub and the Rust gates (release CI).
- Twenty platform/interpreter wheel combinations, each installed outside the
  checkout and checked with dashboard HTML/assets/API requests and the
  end-to-end tests (release CI).
- The six numerical `phase0` baselines stay bit-identical.
- Eighteen demo smoke runs, including the 501×501 hex discrete grid at its full
  251,001-deme size (1.9 GiB peak RSS) and the six UI demos driven through the
  real app factory without serving.
- Documentation examples: 519 Python blocks across 78 pages classified, 448
  runnable; 302 pass and 144 fail. The failures are 118 page-context sketches
  that reference objects no block on their page creates, 14 missing imports,
  4 raw-versus-observation history conflicts, 2 stale allele examples, 2
  unguarded census divisions and 2 deliberate error demonstrations. `v0.3.0b0`
  shows the same profile (138 failing blocks), so this is a pre-existing
  documentation condition rather than a regression of this release.

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
