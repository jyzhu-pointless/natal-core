# Sessions and a single tick

This chapter follows the ordinary discrete-generation path. Python checks and adapts calls, a Rust session owns state, and kernels execute numerical stages. Batch execution does not return the complete state to Python after every stage.

## Entry points and calls

[DiscreteGenerationPopulation.run()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/population/discrete_generation.py) checks standalone execution ownership, reentrancy, failure, and stopping before resolving the recording interval and entering `_run_rust_lifecycle()`. `run_tick()` delegates to a one-step `run()` using the population's recording interval.

[RustDiscreteLifecycleBackend](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) materializes contracts and creates the native session during construction; its `run()` adapts native batch execution. The builder path establishes a session at build time, while some direct-construction paths defer it until execution. A new-session-per-run interpretation cannot explain continuous random streams and history.

Rust [DiscreteGenerationSession](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/discrete_generation.rs) uses `run_inner()` to organize the batch loop, logs, recording, and checkpoints. The [kernel run_tick()](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs) organizes stages within a step.

## Ordinary staged execution

| Stage | Implementation behavior | Observable state |
| --- | --- | --- |
| `first` | `HookProgram.execute_event()`, then commit event writes | Before reproduction |
| reproduction | `reproduction()`: mating, fertilization, zygote fitness | Offspring occupy age 0 |
| `early` | Execute Hooks and commit | Offspring have not passed survival |
| survival | `survival()`: juvenile density regulation and survival | Surviving offspring occupy age 0 |
| `late` | Execute Hooks and commit | Before aging |
| aging | `aging()`: age 0 replaces age 1; clear age 0 | Next generation's adults |

`EcoCtx.phase` changes at these boundaries. `stage_sources()` selects current ecology and genetics before each stage, allowing earlier event commits to affect later calculations in the same tick.

The session advances the tick after successful stage completion. A Hook stop result skips remaining stages; entering `run_tick` does not guarantee a tick increment. Failure also does not mean whole-tick rollback: completed stages and earlier callback commits may remain. See [Hook transactions](hooks.md).

## Execution status is not an independent Python flag

[ExecutionStatus](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/status.rs) defines `Ready`, `Running`, `Stopped`, and `Failed`. `begin()` accepts only Ready and transitions it to Running; rejection does not mutate status. Normal completion returns to a resumable boundary, whereas stopping and failure block direct continuation.

Python guards also prevent callbacks from recursively running the same population. Inspect native execution state, phase, and tick when diagnosing failures; counts alone do not identify the execution boundary. Checkpoint restoration restores recorded status, so restoring a Stopped checkpoint does not automatically make it Ready.

## Two paths to read separately

Age-structured kernels also implement reproduction, survival, and aging, but sperm storage persists through those stages. Aging shifts multiple age classes instead of replacing two generations.

The fused Wright–Fisher path uses `run_wf_tick()`, with `first` scheduled separately by the session. It does not execute the ordinary `early`, `late`, and three-stage combination. It has its own mode validation and update algorithm; it must not be described merely as a faster ordinary loop, and the table above does not apply unconditionally to every mode.

## Verification entry points

- [test_rust_discrete_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_rust_discrete_lifecycle.py): native discrete lifecycle execution.
- [test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py): stopping, failure, restoration, continuous recording, and native ticks.
- [test_frozen_lifecycle_rules.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_frozen_lifecycle_rules.py): lifecycle restrictions.

Changing stage order requires checking Hook-visible state, parameter visibility, stopping boundaries, recording, and RNG order. Equal total counts cannot establish that these contracts remain unchanged.
