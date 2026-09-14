# A model's complete journey from declaration to results

Suppose you ask an agent to simulate 100 heterozygous females and 100 heterozygous males, Mendelian inheritance, and survival of half the juveniles. The agent reports 100 adults in the next generation: 25 AA, 50 Aa, and 25 aa. Where do those numbers come from? What do the builder, inheritance tensors, and Rust session each contribute?

This chapter follows one model through the entire path. You do not need to read the source first or memorize every function name. Understand what changes at each step, then use the implementation table to discuss concrete changes with an agent. The [next chapter](reproduction.md) expands the reproduction calculation.

## State the model requirements completely

We declare a diploid species with one locus on one autosome and three possible alleles: A, a, and X. Each individual carries two alleles at that locus. Parental origin is not distinguished, so A|a and a|A represent the same genotype. X illustrates the difference between types allowed by the species and types needed in this simulation. No mutation or conversion produces X here.

The table specifies the complete teaching scenario; it is not code with initialization omitted.

| Item | Declaration | Consequence |
| --- | --- | --- |
| Model | Ordinary staged discrete generations; fused acceleration off | Reproduction, survival, then complete generation replacement |
| Numerical mode | Deterministic | Counts are expectations and may be fractional |
| Initial adults | 100 of each sex, all A|a | Two parental pools |
| Initial juveniles | 0 | Empty age-0 slots |
| Inheritance | Mendelian segregation; only the default gamete and somatic labels | No additional conversions or label states |
| Adult mating rates | 1 for both sexes | All females may mate; all males contribute to partner weights |
| Eggs per female | 2 | Expected eggs per neutral pair |
| Reproductive participation | Internal discrete adult value of 1 | All mated females participate |
| Global sex ratio | 0.5, the female fraction | Equal female and male offspring of each type |
| All fitness values | 1 | No selection or additional reproductive losses |
| Baseline juvenile survival | 0.5 for both sexes | Fraction retained into adulthood |
| Density regulation | no_competition | No additional juvenile reduction from abundance or capacity |
| Index compression | Enabled | Remove types unreachable in this model |
| Output | Raw history, recorded every tick during execution | Preserve starting and completed boundaries |

Reproductive participation here is an internal value read by the kernel. The discrete builder uses its dedicated adult-mating vocabulary; the presence of internal age arrays does not permit arbitrary age-structured per-age arguments.

These requirements predict two separate properties: an inheritance ratio of 1:2:1, and a total falling from 200 to 100. A correct ratio does not establish a correct total.

## First transformation: the species becomes a type catalog

`Species` expresses genetic structure. `PopulationBuilder` organizes this simulation's counts, ecology, and behavior declarations. It needs an `IndexRegistry` to map meaningful types to integer array positions.

Three alleles produce six unordered diploid genotypes here. Gametes carry one allele, giving three gamete types. NATAL also supports labels: an individual ZType is a genotype plus a somatic label, and a GType is a haploid genotype plus a gamete label. With only default labels, these can provisionally be read as genotype and gamete categories.

The actual complete catalog for this example is:

| Complete ZType index | Type name | Initial female / male counts |
| --- | --- | --- |
| 0 | A\|A@default | 0 / 0 |
| 1 | A\|a@default | 100 / 100 |
| 2 | A\|X@default | 0 / 0 |
| 3 | a\|a@default | 0 / 0 |
| 4 | a\|X@default | 0 / 0 |
| 5 | X\|X@default | 0 / 0 |

The gamete catalog is A@default, a@default, X@default. `@default` is the actual name format verified here. Order follows species enumeration and label expansion; callers must not treat these indices as universal across species.

The catalog does more than accelerate lookup. Count column 3, inheritance type 3, and output name 3 must identify the same object. Otherwise, valid numerical computation can acquire the wrong biological interpretation.

## Second transformation: user parameters become a common draft

`ModelDraft` holds numerical construction materials. Discrete generations use two age slots: age 0 contains juveniles produced during this step, and age 1 contains reproducing adults. These are lifecycle roles, not two arbitrarily sized chronological intervals.

On the complete catalog, initial counts have shape `(2, 2, 6)`: sex, age, ZType. Each sex's age-0 row is zero, and its age-1 row is `(0, 100, 0, 0, 0, 0)`. Cells hold aggregate counts; there are no separate identity objects for the 200 individuals.

User scalars also become arrays read uniformly by kernels:

| Draft field | Values here | Interpretation |
| --- | --- | --- |
| `age_based_mating_rates` | Female `(0, 1)`; male `(0, 1)` | Juveniles do not mate; adult mating rate is 1 |
| `age_based_reproduction_rates` | `(0, 1)` | Only adults reproduce |
| `age_based_survival_rates` | `(0.5, 0)` for both sexes | Survival uses the age-0 value; aging replaces old adults |

`build_discrete_engine_config()` establishes discrete structure and defaults; `build_config_maps()` performs shared draft assembly. Declaration updates write relevant fields. Although an age-1 survival cell exists, setting it to 1 would not preserve adults across generations: the discrete replacement rule is not controlled by that cell.

This distinguishes a common internal representation from configurable semantics. Shared arrays support kernel access, but not every cell has the same configurable meaning in every model.

## Third transformation: inheritance rules become probability tables

Initial counts answer what exists now; inheritance tables answer what those parents can produce. Compilation needs both.

The builder captures a `ModelDefinition` through `_definition_for_compile()`, including declarations, the complete registry, a draft, presets, and modifiers. `_compile_products()` can reuse products for the same declaration, otherwise calling `compile_definition()`. It is therefore also inaccurate to say that every build necessarily recomputes all rules from scratch.

`CompiledProducts` bundles the draft, registry, and modifier lists. Its main inheritance products are two tables:

| Table | Complete-axis shape | Question answered |
| --- | --- | --- |
| `zygotes_to_gametes_map`, M | `(2, 6, 3)` | What gamete probabilities does this parent sex and type produce? |
| `gametes_to_zygotes_map`, F | `(3, 3, 6)` | What offspring results from this female and male gamete pair? |

The A|a parental M row is `(0.5, 0.5, 0)`, and the A/a gamete pair's F row selects A|a. AA and aa have zero initial counts, but their types and inheritance relationships are needed because the first generation can produce them.

With presets or modifiers, compilation starts from baselines and applies fitness and map declarations in their specified order. This example has no interventions, leaving Mendelian maps. Treating an already converted table as a new neutral baseline could apply a conversion twice on recompilation.

At this point, `offspring_tensor` is still a `(0, 0, 0)` placeholder. It means offspring probabilities have not yet been derived, not that every cross is infertile. These complete-axis products are not ready to serve directly as the final runtime model.

## Fourth transformation: determine which types execution needs

Compression follows inheritance reachability rather than deleting zero-count columns. `plan_projection()` starts with seeds such as initially present and explicitly retained types and follows inheritance maps. Age-structured models also account for types in initial sperm storage.

Here the closure is:

```text
Initial A|a
  → A and a gametes
  → A|A, A|a, and a|a offspring
  → Those offspring still produce only A and a gametes
  → No further types appear
```

AA and aa remain despite starting at zero. X is unreachable, so types containing it can be removed. The projection is:

| Complete index | Runtime index | Type |
| --- | --- | --- |
| 0 | 0 | A\|A@default |
| 1 | 1 | A\|a@default |
| 3 | 2 | a\|a@default |
| 2, 4, 5 | Removed | Types containing X |

`publish_products()` creates a new runtime registry and projects counts, fitness, M, F, and related arrays together. M becomes `(2, 3, 2)`, F becomes `(2, 2, 3)`, and counts become `(2, 2, 3)`. Compressing counts alone could send runtime index 2's aa abundance through the former A|X inheritance path.

It then derives P, `offspring_tensor`, on final axes. Its `(3, 3, 3)` axes are female parent, male parent, and offspring. That is 27 entries rather than the 216 of a six-type tensor. Deferring derivation avoids constructing a full cubic tensor only to discard most entries.

Publication also checks type identity, order, shapes, and names. Equal shapes do not establish compatible coordinates. Hook selectors must subsequently resolve against this final catalog.

To introduce X later, retain it at construction or rebuild with a new layout. A removed type is not an existing zero-valued cell that remains available for arbitrary writes.

## Fifth transformation: construction products enter an execution session

Publication does not start simulation. `_build_published()` creates the Python population and initializes its session. `RustDiscreteLifecycleBackend` uses `materialize()` to split the draft into two cross-language contracts:

| Contract | Contents in this example | Purpose |
| --- | --- | --- |
| `Blueprint` | Two sexes, two ages, three ZTypes, names, execution flags, initial counts, and related metadata | Fixed layout and initial conditions |
| `Params` | Eggs, sex ratio, ecological arrays, fitness, and inheritance tensors | Runtime parameter data |

`materialize()` establishes fresh array ownership. Rust `from_parts()` reads and validates Blueprint, `EcologyParams`, and `GeneticsTensors`, then creates owned state, RNG, tick, and execution status. The Python-exposed native class is `DiscreteEngineSession`, implemented by Rust `DiscreteGenerationSession`.

Two details matter here.

First, Rust participates before execution. `recompute_offspring_tensor()` already called a Rust numerical kernel to derive P. The useful division is Python orchestration and interfaces versus native numerical computation and runtime sessions; it does not coincide exactly with build versus run.

Second, drafts and contracts can include sperm fields in their shared structure, while the discrete session stores only individual counts and no cross-tick sperm bank. A field's presence in construction data does not establish runtime ownership or use.

After initializing the session, the builder finishes compiling the recording plan. That plan depends on final indices and shapes, keeping stored values aligned with names. The model is now runnable, but still contains its original 100 A|a adults of each sex.

## Sixth transformation: one tick produces new counts

Here `run_tick()` ultimately executes one iteration of the native batch path. The ordinary discrete tick has these boundaries. Triples below are ordered AA, Aa, aa, with counts shown separately for each sex.

| Boundary | Tick | Age 0 per sex | Age 1 per sex | Meaning |
| --- | --- | --- | --- | --- |
| Build complete / first | 0 | `(0, 0, 0)` | `(0, 100, 0)` | Parents ready |
| After reproduction / early | 0 | `(25, 50, 25)` | `(0, 100, 0)` | 200 offspring split equally by sex; parents remain |
| After survival / late | 0 | `(12.5, 25, 12.5)` | `(0, 100, 0)` | Half the juveniles remain |
| Stable boundary after aging | 1 | `(0, 0, 0)` | `(12.5, 25, 12.5)` | Offspring become adults and replace parents |

Hook positions determine future interventions' visible state even when no callbacks are installed. At early, summing the whole count array gives 400: 200 parents plus 200 offspring. That is not the final next-generation population.

The arithmetic is 100 pairs times 2 eggs = 200; Mendelian allocation gives `(50, 100, 50)`; each sex receives `(25, 50, 25)`; half survive to `(12.5, 25, 12.5)`. Fractions represent deterministic aggregate expectations. Rounding them and continuing is not an equivalent model.

Only successful completion of the tick advances time from 0 to 1. Stopping or failure can leave a partial execution boundary, so the final row cannot describe every exit path.

## Final transformation: one result has multiple query layouts

After execution, the Rust session is authoritative for current counts. Python `pop.state` returns a snapshot with copied arrays, refreshing its local cache first when necessary. `individual_count` has axes `(sex, age, ZType)` and shape `(2, 2, 3)` here. Mutating that returned array is not an engine update.

Default identity observation creates one group per ZType. For this single population, `pop.observe()` returns tick 1 with axes `(group, sex, age)` and shape `(3, 2, 2)`. The A|a group's adult value is 25 for each sex. Observation reorganizes named results; other groups can aggregate types, after which individual type counts cannot generally be reconstructed.

With raw history and recording every tick, one step produces ticks `(0, 1)`. `history.individual_count` uses `(record, sex, age, ZType)`, shape `(2, 2, 2, 3)`. History stores boundaries; it does not automatically add separate early and late rows from the table above.

Current state, current observation groups, and retained history ticks are therefore three separate questions. Numerical arrays need not share axes or information content.

## Judging development proposals

| Agent proposal | What this example establishes |
| --- | --- |
| Retain only initially nonzero genotypes | This would remove reachable AA and aa; check inheritance closure |
| Append an array column to introduce X | Registry, M, F, P, fitness, selectors, session, and recording layout also participate |
| Intervene by mutating `pop.state` | That changes a query snapshot; use controlled writes |
| Recreate the session on every run | Explain preservation of the timeline, parameters, history, and RNG |
| A 1:2:1 result proves reproduction is correct | Egg production could still be doubled or survival wrong; verify totals and stages |
| Check Rust availability only on the first run | Constructing P can already call Rust |

The first two proposals affect representable biological states, not merely index implementation. Nonoverlapping generations and deterministic counts are model semantics; compression and deferred derivation are implementation strategies; snapshots and controlled writes are interface contracts. Specify which category a requested change targets.

## Implementation and verification map

The narrative above explains the complete chain. This table helps an agent locate code and helps you check for omitted neighboring steps.

| Implementation entry | Responsibility here |
| --- | --- |
| [PopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_base.py): `_compile_products()`, `_publish_and_build()`, `_build_published()` | Compilation, publication, session and recording initialization |
| [build_registry()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_registry_builder.py) | Complete species catalog |
| [build_discrete_engine_config()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/assembly.py) | Internal two-age representation |
| [compile_definition()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/definition_compiler.py) | Declarations and candidate genetic products |
| [plan_projection() / publish_products()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/publication.py) | Reachability and coordinated projection |
| [materialize()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/contracts/materialize.py), [RustDiscreteLifecycleBackend](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) | Cross-language contracts and adaptation |
| [DiscreteGenerationSession](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/discrete_generation.rs) | Native state, batch loop, and recording |
| [compile_recording_plan()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py) | Output layout and names |

Catalogs, projection, shapes, early/late counts, final state, default observation, and raw history were checked with the same inputs. Existing [publication contracts](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication_contracts.py), [materialize contracts](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_contracts_materialize.py), and [recording lifecycle](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py) tests protect indices, isolation, and execution/recording boundaries respectively. They do not replace validation of every other model combination.

Continue with [How one reproduction stage is calculated](reproduction.md) to unpack the “200 offspring” step.
