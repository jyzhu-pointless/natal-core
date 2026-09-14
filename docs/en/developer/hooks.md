# Hooks and controlled updates

A Hook can be a compilable declarative operation or a Python callback. Both enter the same event schedule, but callbacks need controlled access across the Python/Rust boundary.

## From declarations to execution slots

In [hooks/_compile.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/_compile.py), `compile_hook_call()` resolves an individual call and `build_hook_program()` assembles the program. `HookLayoutContext` supplies layout context; selectors resolve against published indices. Compilation also handles priority, identity deduplication, and deme ranges.

Rust [HookProgram](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/hooks/interpreter.rs) executes native plan slots and Python callback slots in one priority order. Do not assume all declarative Hooks precede all Python Hooks: mixed-registration ordering is part of the contract.

An ordinary tick invokes `execute_event()` at first, early, and late. The following `EcoCtx.commit()` makes committed parameters visible to later stages. Manual events, finish, and fused modes have additional entry paths; ordinary ticks do not establish every event's behavior.

## Callback transaction lifetime

[TickContext](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/tick_context.py) and [EventTransaction / HookRng](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/_transaction.py) connect callback access to [Rust transactions](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/hooks/transaction.rs).

Transactions let a callback modify candidate data and commit on success. State and parameters are acquired on demand; a scalar-only callback should not pull the complete individual-count array. `HookRng` uses the controlled current random stream and expires when the callback returns. Retained context or parameter handles must not bypass lifetime checks or transaction routing.

Distinguish three boundaries: one callback's candidate changes, commits of multiple callbacks within one event, and the lifecycle stages of the entire tick. A later callback's failure does not undo earlier successful callbacks, and completed lifecycle stages are not automatically restored to the tick's starting state.

For example, if callback A successfully changes a parameter and callback B then fails in the same event, A's commit remains while B's uncommitted changes must not leak. `test_prior_callback_commit_survives_later_failure` asserts this scenario.

## Scalar updates and genetic recompilation

`RuntimeUpdater` in [builder/_runtime.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_runtime.py) routes updates to appropriate writers. Ordinary parameter writes validate candidates. Custom updates merge and normalize values, refresh native slots, and only then update the Python draft and audit entries.

Presets and genetic modifiers affect derived maps. `compile_runtime_candidate()` recompiles an isolated candidate, and `commit_genetic_update()` commits it. Mutating one map in place while retaining an old offspring tensor would leave inconsistent genetics. Runtime layouts are already published, so recompilation must also respect current axis identities.

`reconfigure_preset()` differs from appending a preset: its contract replays from neutral fitness, and callback transactions register necessary restoration actions for external objects. Check its actual semantics when extending updates rather than assuming every update preserves earlier manual fitness.

## Verification and change route

[test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py) covers expired channels, recursive execution, retention of earlier commits, on-demand data acquisition, custom-array isolation, and atomic native parameter updates.

For a new Hook operation, follow Python declaration, compiled slot, native interpreter, parameter or state commit, stopping/failure, and checkpoints. Establish selector resolution timing, callback ordering, writable fields, visibility to later stages, and the boundary retained after failure. Matching parameter and event names alone do not establish equivalent behavior across entry points.
