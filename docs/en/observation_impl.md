# Observation and History Implementation

This document is for NATAL Core maintainers and contributors. It explains the boundaries between the canonical Observation, typed History, and spatial recording paths. For the public workflow, see [Extracting Population Simulation Data](2_data_output.md).

## Implementation Overview

An Observation only defines how to derive observed values from population state. History only defines which kind of snapshot to store. A RecordingPlan connects them at build time, but they remain independent concepts:

```text
Configurator.with_observation(...)
  → compile an immutable canonical Observation

Configurator.record_history(mode=...)
  → select a raw or observation History schema

Population state
  ├─ pop.observe() → ObservationResult
  └─ sampling boundary → History
       ├─ raw mode: store complete state
       └─ observation mode: store the canonical Observation projection
```

Core modules and responsibilities:

| Module | Responsibility |
|--------|----------------|
| `output/observation.py` | Defines `Observation`, `ObservationResult`, `ObservationFilter`, and identity observations |
| `output/history.py` | Defines immutable schemas, typed array views, and post-hoc projection of raw History |
| `output/_recording.py` | Compiles the `RecordingPlan`, row width, and spatial layout at build time |
| `output/record.py` | Provides uniform observation-row encoding for non-spatial engines |
| `engine/templates/spatial_lifecycle_*.tmpl.py` | Runs the spatial lifecycle and returns regular raw batches |
| `spatial/population.py` | Applies the canonical Observation at the spatial container boundary, then commits History |

## Build-Time Public Interface

Non-spatial Configurators use:

```text
.with_observation(groups, collapse_age=False)
```

The spatial Configurator additionally accepts a deme selection and processing mode:

```text
.with_observation(
    groups,
    collapse_age=False,
    demes=None,
    deme_mode="preserve",
)
```

`groups` is an ordered mapping from non-empty labels to `IndividualSelector` instances. Its insertion order becomes the group-axis order in results.

The spatial arguments have the following semantics:

| Argument | Semantics |
|----------|-----------|
| `demes=None` | Select every deme in Population order |
| `demes=[2, 0]` | Select demes 2 and 0, preserving that order |
| `deme_mode="preserve"` | Keep one shared deme axis |
| `deme_mode="aggregate"` | Sum selected demes and remove the deme axis |

`demes` must be a non-empty, duplicate-free sequence of integer indices within the Population range. Every group shares the same ordered deme selection; separate groups cannot define different deme sets. `deme_mode` accepts only `"preserve"` and `"aggregate"`.

The Observation and History schema are frozen by `build()`. A runtime Configurator cannot replace the `with_observation()` or `record_history()` rules.

## Canonical Observation

Every built Population has one canonical `Observation`. If the user does not call `with_observation()`, the build creates an identity observation with one group per active ZType. Identity observations avoid a quadratic dense mask and use a ZType index map for a lossless axis transformation.

An `Observation` stores the following stable information:

- `labels`: group-axis labels.
- `collapse_age`: whether to sum and remove the age axis.
- `population_fingerprint`: prevents applying the rule to an incompatible Population layout.
- `deme_indices`: ordered spatial deme selection, or `None` for a non-spatial observation.
- `deme_mode`: whether a spatial observation preserves or aggregates the deme axis.
- A compiled selector mask, or a ZType index map for an identity observation.

### Numerical Projection

The age-structured non-spatial input shape is:

```text
(sex, age, ztype)
```

A regular observation produces a group-first result through this operation:

```text
mask[group, sex, age, ztype]
  × count[sex, age, ztype]
  → sum along ztype
  → result[group, sex, age]
```

Spatial input adds a regular deme axis at the front:

```text
(deme, sex, age, ztype)
```

Spatial projection first slices by `deme_indices`, then applies the same selector masks to every selected deme. `preserve` retains the sliced deme axis; `aggregate` explicitly sums that axis. Finally, `collapse_age=True` sums and removes the age axis.

This ordering guarantees two key invariants:

```text
preserve result == per-deme projection after directly slicing count in demes order
aggregate result == preserve result summed along the deme axis
```

### Public Result Axes

`pop.observe()` returns an `ObservationResult`; its `axes` explicitly describes the dimensions of `values`:

| Mode | `collapse_age=False` | `collapse_age=True` |
|------|----------------------|---------------------|
| Non-spatial | `(group, sex, age)` | `(group, sex)` |
| Spatial preserve | `(group, deme, sex, age)` | `(group, deme, sex)` |
| Spatial aggregate | `(group, sex, age)` | `(group, sex)` |

`labels["group"]` aligns one-to-one with the group axis. In preserve mode, the deme-axis order is exactly the order supplied through `demes`.

## History and RecordingPlan

`record_history()` independently selects the recording mode:

```text
.record_history(mode="raw", max_rows=None)
.record_history(mode="observation", max_rows=None)
```

At build time, `compile_recording_plan()` creates an immutable `HistorySchema` and computes a fixed `row_size` from the Population layout, Observation axes, and History mode. A spatial schema also stores the complete deme count and raw payload width per deme.

### Raw Mode

Spatial raw History always stores the complete state of every deme. It is unaffected by the Observation's `demes`, `deme_mode`, or `collapse_age`:

| Data | Shape |
|------|-------|
| `history.individual_count` | `(record, deme, sex, age, ztype)` |
| `history.sperm_storage` (age-structured) | `(record, deme, age, female_ztype, male_ztype)` |

Raw History can therefore call `history.observe(observation)` later to create an independent observation-mode History without modifying the original History.

### Observation Mode

Observation History stores only the numerical result of the canonical Observation:

| Mode | `collapse_age=False` | `collapse_age=True` |
|------|----------------------|---------------------|
| Non-spatial | `(record, group, sex, age)` | `(record, group, sex)` |
| Spatial preserve | `(record, group, deme, sex, age)` | `(record, group, deme, sex)` |
| Spatial aggregate | `(record, group, sex, age)` | `(record, group, sex)` |

The schema's `ObservationMetadata` stores group labels, age-collapse state, ordered deme selection, and deme mode. Reading `history.values` therefore does not depend on mutable external state from the current Population.

Observation History has discarded unrecorded ZTypes and unselected demes, so it cannot be projected post-hoc through a different Observation.

## Spatial Recording Path

Spatial recording does not execute the Observation inside the Numba wrapper. The wrapper runs lifecycle steps and migration, then returns a regular raw batch at stable tick boundaries:

```text
Numba spatial wrapper
  → [tick, all deme individual_count, all deme sperm_storage]
  → SpatialPopulation._process_kernel_history(...)
       ├─ raw History: validate and commit the complete batch
       └─ observation History:
            reshape into regular spatial count
            → canonical Observation.apply(...)
            → commit the fixed-shape projected row
```

The Python fallback calls `_record_snapshot()` at the same stable tick boundaries. Raw mode commits complete spatial state, while observation mode calls the same `Observation.apply()`. Both backends therefore share the same Observation semantics and History schema; only the place where the raw batch is produced differs.

The spatial wrapper transports raw batches to keep engine transport regular and fixed. The lifecycle kernel does not need to understand groups, deme selection, or aggregate rules. All Observation semantics remain concentrated in the canonical `Observation` and the spatial container boundary, rather than being reimplemented by the engine, Python fallback, and post-hoc projection paths.

## Removed Compact Spatial Layout

Spatial Observation no longer supports a different deme layout for each group. The following concepts from the old implementation have been removed from the spatial recording path:

- `CompactMeta` and `build_compact_metadata()`.
- `build_observation_row_spatial()`.
- Per-group `mask` / `expand` / `aggregate` layouts.
- The `-1.0` sentinel for an unselected deme.
- Ragged group offsets and different row widths per group.

Every group now shares one regular ndarray axis structure. Unselected demes are absent from preserve results and need no special numeric marker; a real zero count remains `0.0`. For different deme views, construct separate Observations from raw History, or record one wider shared selection and split it in caller code.

## Maintenance Invariants

Changes to Observation or History recording should verify at least these numerical relationships:

1. Spatial preserve is element-wise equal to per-deme projection after directly slicing in `demes` order.
2. Spatial aggregate is element-wise equal to the preserve result summed along the deme axis.
3. A spatial identity Observation loses no coordinate values in selected demes.
4. `collapse_age=True` is element-wise equal to the uncollapsed result summed along age.
5. Raw History retains every deme, ZType, and applicable sperm-storage value.
6. Post-hoc projection of raw History is element-wise equal to `Observation.apply()` at the same tick.
7. The Numba path and Python fallback produce identical ticks and payloads in deterministic simulations.
8. Unselected demes require no sentinel representation, and real zero counts are not confused with selection state.

Assertions must compare explicit axes and coordinate values. Comparing only totals or sorted flattened arrays cannot detect axis swaps or incorrect deme ordering.
