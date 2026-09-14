# Spatial execution and migration

A deme is a local population unit within a spatial container. The container owns the shared timeline and stacked state. Running individual demes from Python cannot replace unified scheduling and migration.

## Shared layouts and heterogeneous parameters

[SpatialPopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/builder.py) organizes local declarations and shared publication. Compression must account for a common runtime layout; arrays cannot be stacked if ZType index 1 means a different type in each deme.

Native individual counts use `(deme, sex, age, ZType)`. Age-structured sperm storage uses `(deme, age, female ZType, male ZType)`. Demes can have different ecological parameters, while genetics are organized through variants.

Backend [ecology_columns_from_drafts() and genetics_variant_bank()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) explain these representations: ecological values form deme-addressed columns, and genetic tensors form a reusable variant bank. `fork_variant()` and `refresh_variant_tensors()` connect genetic branching and updates. Initial sharing does not imply that a write may silently affect sibling demes.

## Folding migration structure at build time

[fold_migration_csr()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/migration.py) converts adjacency matrices or migration kernels into `MigrationCSR`. CSR means compressed sparse row:

| Array | Meaning |
| --- | --- |
| `indptr` | Length D+1; adjacent pointers delimit one source deme's edges |
| `dest_idx` | Destination deme for each edge |
| `weights` | Edge weights corresponding to destinations |
| Rate column | Separate `(D, 2, A)` outbound rates by source, sex, and age |

Adjacency mode stores raw weights in ascending destination order. Kernel mode visits offsets in kernel row-major order, handles boundaries or wrapping, and normalizes. Wrapping can produce repeated destinations. The implementation retains those entries and their visitation order to preserve floating-point accumulation behavior; sorting or merging is not automatically equivalent.

The current denominator choice in `adjust_on_edge` cancels during final row normalization, generally leaving only rounding differences in the destination distribution. This differs from an intuition that boundaries must lose outbound mass. Read `_kernel_row_entries()` before changing it.

## Execution and migration are not one in-place sweep

In [kernels/spatial.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/spatial.rs), `schedule_deme_ticks()` schedules local ticks, while `run_spatial_tick_heterogeneous()` and `run_spatial_tick_discrete()` organize model-specific paths. Migration acts on updated deme states within the spatial lifecycle flow.

`migrate_csr_deterministic()` reads old inputs and accumulates into fresh outputs. Each source computes outbound mass, distributes it to CSR destinations, and keeps the residual locally. Fresh buffers prevent newly arrived individuals from emigrating again as part of another source in the same tick.

For one individual category, if A has 100, B has 0, and A sends at rate 0.2 exclusively to B, migration yields 80 and 20. This illustrates conservation during migration only, excluding earlier births or deaths.

The age-structured path handles unmated females separately from mated females categorized by stored male type, moving associated sperm storage at the female migration rate. `migrate_csr_stochastic_rngs()` samples outbound mass and destinations with source-deme RNGs. Sperm must not migrate again as independent individuals detached from females.

## Ownership and verification

[SpatialPopulation](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/population.py) manages the session and container operations. Local parameter access does not grant a deme independent control over running, resetting, or restoring the shared timeline.

- [test_spatial_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_publication.py): shared publication and variants.
- [test_spatial_migration_conservation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_migration_conservation.py): migration conservation.
- [test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py): deme execution restrictions, atomic local parameters, and the shared clock.

Migration changes need separate checks for total mass, female/storage association, boundaries and duplicate destinations, source traversal order, random streams, and stopping boundaries. Different totals before and after a complete lifecycle do not alone establish nonconservative migration.
