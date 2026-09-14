# Architecture and data flow

For a complete introduction, see [A model's complete journey from declaration to results](model_journey.md), which follows one model's inputs and outputs across these layers.

The journey from construction to observation has three distinct stages: users describe a model, Python compiles and publishes its numerical layout, and a Rust session executes and records it. Within the Python package, `frontend` primarily means the model frontend. The browser interface also has a repository-root `frontend/` directory; these are separate layers.

## From entry point to result

This is a responsibility flow, not a line-by-line call stack. See [model compilation](model.md) and [session execution](runtime.md) for concrete functions.

```text
Species / PopulationBuilder
    → ModelDefinition + ModelDraft
    → compile_definition → CompiledProducts (complete indices)
    → publish_products (final runtime indices)
    → Hook / Observation / RecordingPlan
    → materialize → Blueprint + Params
    → Rust backend → native session
    → lifecycle kernels → native history / observation
    → Python query results
```

[PopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_base.py) separates `_compile_products()`, `_publish_and_build()`, and `_build_published()`. This establishes a boundary between compilation products on complete axes and the final runtime model. Hook selectors and observation layouts depend on final indices and must not bind prematurely to pre-compression coordinates.

## Ownership by layer

| Layer | Object or module | Contents and constraints |
| --- | --- | --- |
| Declaration | `ModelDefinition` | Retains declarations for recompilation; compilation uses isolated working copies |
| Construction | `ModelDraft`, `CompiledProducts` | Candidate arrays, indices, and modifiers; not the authority for live state |
| Publication | `IndexProjection`, published `IndexRegistry` | Fixes runtime coordinates and projects all related axes together |
| Transfer contract | `Blueprint`, `Params` | Fields, dimensions, and numerical representations crossing into Rust |
| Execution | Rust `sessions/` | Current counts, tick, execution status, RNG, and runtime parameters |
| Output | Rust `output/` and Python wrappers | History rows, projections, parameter logs, and query metadata |

[materialize()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/contracts/materialize.py) splits a built draft into contract objects with fresh array ownership. Rust constructs its owned data from those contracts. This does not mean every update copies the entire model: `materialize_params()` skips Blueprint, while `contract_field_source()` projects only requested fields and its contiguous-array conversion does not always copy.

## Reads and writes take different routes

With a session present, `pop.state` and `pop.config` are Python query snapshots. Mutating their arrays is not a mechanism for changing the engine. Start with `RustDiscreteLifecycleBackend.state_snapshot()` and `config_snapshot_from_session()` to see how snapshots are reconstructed.

Runtime writes enter the native session through parameter channels or updaters; writes inside callbacks also follow transaction rules. Retained Python declarations support rebuilding and interpretation, but cannot establish the latest native values. Parameter reads therefore need the backend rather than unconditional access to an old draft.

Spatial populations concentrate execution ownership in one stacked session. Python deme objects provide access to local data but cannot independently advance the shared timeline; see [spatial execution](spatial.md).

## Source and verification entry points

- [rust_backend.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py): adaptation for contract conversion, session calls, snapshots, and parameter writes.
- [Rust model](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/model/mod.rs): native data organization; read alongside Python contracts.
- [test_contracts_materialize.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_contracts_materialize.py): field mappings, name directories, custom values, array isolation, and parameter-only materialization.
- [test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py): current native values, callback channels, and spatial execution ownership.

When changing data ownership, follow construction, transfer, execution, snapshots, and restoration. Identical results on a successful run alone cannot establish correct sharing and restoration behavior.
