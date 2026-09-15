# Observation, history, and checkpoints (navigation)

Query results, history records, and restorable checkpoints serve different purposes. Their details now live in three topic chapters, so this page is a navigation entry: it states how the three relate, what the recording-mode trade-offs are, and keeps the code and test links.

## Three entry points

| Topic | Entry point now | Question answered |
| --- | --- | --- |
| How current state projects into grouped results | [How observation turns state into results](observation.md) | Grouping, axis order, names, and information loss |
| What history stores, how it evicts, how the parameter timeline aligns | [History recording and the parameter timeline](history.md) | Both modes, record intervals, label formats, the log |
| What returning to a tick requires, and whether genetics roll back | [Checkpoint restoration and experiment replay](checkpoints.md) | What a restore contains, and how it differs from importing state |

## How the three relate

| | Current observation | History | Checkpoint |
| --- | --- | --- | --- |
| Data source | a projection of the live session state | recorded rows | a checkpoint aligned with history rows |
| Time | a moment | a series | a past you can return to |
| Reversible | stateless and repeatable | read-only | restores and truncates what follows |
| Information loss | depends on the grouping | raw keeps everything, observation aggregates | covers running state only, never genetic tables |

In one sentence each: **observation is a projection, history is a record, a checkpoint is a rewind.** They share one grouping rule and one tick semantics, so what a query shows, what was recorded, and what can be restored line up item by item.

## Recording-mode trade-offs

- Need to regroup old data later, or to run branching experiments → use `raw`.
- Only care about a few fixed groups and want a smaller store → use `observation`, accepting that checkpoints cannot be restored.
- Need a storage ceiling → use `max_rows`, remembering that the checkpoints of evicted rows go with them.
- Need to know when a parameter changed → read `pop.params_log`, which is tick-aligned with the history.

## Code and test entry points

- [output/observation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/observation.py): observation groups, masks, projection.
- [output/_recording.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py): the recording plan and mask compilation.
- [output/history.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/history.py): layout, modes, row access, truncation.
- [rust/src/output/history.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/history.rs), [parameter_log.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/parameter_log.rs): the native row and log stores.
- [test_history_observation_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_history_observation_contract.py), [test_native_history_contracts.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_native_history_contracts.py), [test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py): recording and restore contracts.

Start this line at [how observation turns state into results](observation.md), or rebuild the overall picture from [architecture and responsibility boundaries](architecture.md).
