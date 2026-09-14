# Model compilation and index publication

A biological type can have different integer indices in the complete species catalog and the compressed runtime layout. Model construction must keep genetic tables, initial counts, selectors, and result names on the same coordinates.

## ZTypes and GTypes

[IndexRegistry](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/registry/index.py) stores `(genotype, slab_label)` in `index_to_ztype` and `(haploid_genotype, glab_label)` in `index_to_gtype`. A ZType is therefore more than a bare genotype: its somatic label is part of its identity. A GType likewise includes its gamete label.

Let Z be the final ZType count, G the final GType count, and A the age count. Common arrays use the following layouts, with female before male on the sex axis.

| Draft field | Shape | Axes |
| --- | --- | --- |
| `initial_individual_count` | `(2, A, Z)` | Sex, age, individual type |
| `initial_sperm_storage` | `(A, Z, Z)` | Female age, female type, stored sperm's male type |
| `zygotes_to_gametes_map` | `(2, Z, G)` | Parent sex, parent type, gamete type |
| `gametes_to_zygotes_map` | `(G, G, Z)` | Female gamete, male gamete, offspring type |
| `offspring_tensor` | `(Z, Z, Z)` | Female parent, male parent, offspring type |
| `sexual_selection_fitness` | `(Z, Z)` | Female parent, male parent |

These are draft representations. Discrete-generation sessions do not retain sperm across ticks. A field's presence in the draft does not imply identical runtime state across models.

## Compilation on complete axes

[compile_definition()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/definition_compiler.py) requires an unpublished, complete species registry. It creates an isolated working copy through `CompileHost`, restarts from fitness baselines, orders presets by priority, and applies explicit fitness steps at their recorded positions. It then appends manual modifiers and rebuilds inheritance maps from the Mendelian baseline.

Each collected modifier must apply once. A failed compile publishes no candidate and restores preset species bindings. Reusing an already modified map as the next baseline could apply the same conversion again during recompilation.

[build_config_maps()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/assembly.py) fills defaults, validates dimensions, and assembles a complete-axis draft. Offspring derivation is deferred to final runtime axes, avoiding construction of a complete Z³ tensor whose entries may mostly be discarded.

## Publication changes coordinates together

[publish_products()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/publication.py) performs these steps:

1. Choose an explicit `IndexProjection`, a compression plan, or an identity projection.
2. Validate source counts, type identities, and order; equal dimensions do not establish equal coordinates.
3. Create a runtime registry and project initial state, fitness, and inheritance maps together through `_project_config()`.
4. Derive offspring on final axes or reuse a validated spatial genetic template.
5. Check shapes and names with `_validate_runtime_layout()` before publishing.

Original compilation products stay unpublished and reusable for isolated builds. `IndexProjection.z_full_to_runtime` and `g_full_to_runtime` use `-1` for removed types. Do not pass that sentinel directly as a NumPy index, where `-1` selects the last element.

## Compression retains more than nonzero individuals

`plan_projection()` seeds reachability with explicitly declared types, initially nonzero individuals, and both female and male type axes of initial sperm storage. It then computes inheritance closure. No seeds means retaining complete axes. The builder also adds extractable Hook type references to explicit retention seeds.

For example, suppose the complete catalog is A, B, C and runtime retains A, C. Runtime index 1 now means C. Compressing counts without the offspring tensor sends C's count through B's inheritance path. Dimensions can remain valid while the biological interpretation becomes wrong.

Consequently, `_build_published()` compiles Hook descriptors and constructs output layouts only after publication. Static reference collection cannot fully infer arbitrary Python callback behavior, so explicit retention declarations remain useful.

## Verification and change constraints

- [test_publication_contracts.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication_contracts.py) and [test_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication.py): publication, projection, and layout contracts.
- [test_offspring_single_derivation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_single_derivation.py): timing of offspring derivation.
- [test_spatial_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_publication.py): shared runtime layouts across demes.

When adding a field with Z or G axes, inspect projection, shape validation, names, materialization, and runtime recompilation together. Successful construction alone rarely detects a missing projection.
