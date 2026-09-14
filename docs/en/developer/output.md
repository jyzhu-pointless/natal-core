# Observation, history, and checkpoints

Queries, history records, and restorable checkpoints serve different purposes. Their relationship determines memory use, recording intervals, and restoration behavior.

## Freezing the recording layout at build time

[compile_recording_plan()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py) constructs a `RecordingPlan` from the final registry, population state, and canonical `Observation`. `HistorySchema` retains layout and names; observation mode also builds a four-dimensional `(group, sex, age, ZType)` selector mask.

A group is a user-defined observation group. Its mask must use published ZType order, or valid numerical projection can acquire incorrect labels. `collapse_age` and spatial deme selection belong to observation metadata and should not be guessed by the presentation layer.

`project()` in [output/observation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/observation.rs) executes projection natively. A group count is the sum of state entries selected by its mask; the observation definition controls which age and deme axes remain. Projected recording therefore needs no full Python state transfer each tick.

## Two storage modes

| Mode | Native records | Restoration |
| --- | --- | --- |
| raw | Raw state rows and corresponding complete checkpoints | Retained exact ticks can be restored |
| observation | Predefined projection values | No hidden raw state and no checkpoint restoration |

[Rust HistoryData](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/history.rs) stores rows, boundary metadata, and shared-log relationships. Python [history.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/history.py) interprets layouts and wraps queries. Current-state `observe()` differs from history access: the former can project the current session, while the latter is limited to retained data.

Raw history can support post hoc observation; observation history has discarded unrecorded information. A few group totals cannot generally reconstruct every genotype count.

## Records represent execution boundaries

Session batch loops consider recording at boundaries, and the recording interval selects retained ticks. Consecutive runs share history; the recording layer deduplicates an exact continuation boundary. Distinguish automatic continuation from manually recording a duplicate tick.

`record_every=0` disables automatic recording for that run. Row limits evict old rows and their checkpoints, so having visited tick 20 does not imply that tick 20 remains restorable. `boundary_metadata` distinguishes ordinary boundaries from stopped or failed execution positions.

## Restoration is more than copying counts

Population `restore_checkpoint()` passes through the backend to session `restore_from_checkpoint()`. Restoration includes individual and sperm state, tick, phase, execution status, RNG, ecological parameters including custom and migration values, and parameter-log positions. Genetic tables do not roll back.

Only retained exact ticks are accepted. Successful restoration discards future history and truncates parameter logs at recorded positions. Later updates within the same tick can also belong to the discarded future. Filtering by `log.tick <= restored_tick` cannot express this behavior.

For example, if a parameter changes after the tick-10 checkpoint but before tick advancement, restoring tick 10 must use the checkpoint's parameters and log cursor. Recorded execution status must also be preserved rather than unconditionally reset to Ready.

`export_state()` / `import_state()` provide a separate state-transfer path and must not be confused with complete checkpoints. Reproducible replay requires checking restoration of random streams and runtime parameters, not counts alone.

## Verification and change constraints

- [test_history_observation_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_history_observation_contract.py): storage modes and queries.
- [test_observation_age_axis_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_observation_age_axis_contract.py): age-axis semantics.
- [test_restore_checkpoint_semantics.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_restore_checkpoint_semantics.py): exact restoration and parameter timelines.
- [test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py): continuous recording, eviction, and execution status.

When changing retention or restoration, check that rows, checkpoints, boundary metadata, and logs change together, and that rejected restoration leaves the current session unchanged.
