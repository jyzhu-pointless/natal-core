# Making and validating changes

After understanding a function, identify the boundaries crossed by a change. This chapter provides investigation routes. Risk classification and quality gates remain defined in repository [AGENTS.en.md](https://github.com/jyzhu-pointless/natal-core/blob/main/AGENTS.en.md) and [quality_checks_spec.md](https://github.com/jyzhu-pointless/natal-core/blob/main/quality_checks_spec.md), avoiding a duplicate policy that can become stale.

## Find entry points by behavior

| Goal | Read first | Check next |
| --- | --- | --- |
| Add an ecological parameter | `parameters.jsonc`, `contracts/params.py`, `rust/src/model/ecology.rs` | Defaults, generated fields, materialization, runtime writes, logs, checkpoints |
| Change inheritance conversion | `genetics/matrices/`, `definition_compiler.py` | Compilation baselines, publication projection, offspring derivation, runtime recompilation |
| Change reproduction or survival | Relevant model in `rust/src/kernels/` | Stage order, sampling, age/sperm constraints, differences between models |
| Add a Hook operation | `hooks/_compile.py`, Rust `hooks/interpreter.rs` | Priority, selectors, transactions, stopping, failure |
| Change observation output | `_recording.py`, Python/Rust `output/` | Layouts, names, age and deme axes, both history modes |
| Change spatial sharing | `spatial/builder.py`, `genetics_variant_bank()` | Shared indices, variant branching, local writes, container ownership |

Generated parameter artifacts include [rust/src/generated/ecology_parameters.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/generated/ecology_parameters.rs). Trace the source and build process before choosing files to edit; hand edits to generated output may be overwritten.

## A concrete investigation sequence

For a new ecological parameter affecting survival, first establish units, default, valid range, whether it varies by deme, its stage of effect, and checkpoint behavior. These are behavioral questions before they are field-placement decisions.

Follow the declaration entry to `ModelDraft` and contract fields, then identify the native ecological column. Trace runtime writes to stage reads and check whether first and early updates affect this tick's survival as intended. Finally inspect snapshots, logs, and restoration so queried values agree with values consumed by kernels.

Validation should include a small model distinguishing old and new behavior and an invalid candidate that must not partially commit. If spatial use is supported, verify that updating one deme leaves other demes' values and genetic-sharing relationships unaffected. Tests that duplicate the complete algorithm are no substitute for boundary checks.

## Learn contracts from existing tests

Each chapter's verification links are investigation entry points. Read setup, event positions, and assertions before deciding whether more coverage is needed. Test names are clues; passing tests do not establish behavior they never assert.

Numerical changes need justified expected values or invariants. State and cross-language changes need ownership and failure-path coverage. Public API changes also require user documentation, both developer-guide languages, affected examples, and stubs to stay synchronized. Use the quality specification for exact commands.

## Maintaining this guide

Maintain corresponding Chinese and English pages and both MkDocs navigation files. Check source links when symbols move or change names, update layout tables when indexed fields are added, and revise execution-boundary explanations when stages or restoration change.

Separate current behavior from design history. Describe motivations as inferences when unsupported by code, tests, or design records. If documentation conflicts with implementation, establish the current contract and test evidence before correcting the explanation or raising a product issue. Do not silently change a model to make the prose true.

The separate browser interface lives in [root frontend/](https://github.com/jyzhu-pointless/natal-core/tree/main/frontend), with Python service entry points in [frontend/webui/](https://github.com/jyzhu-pointless/natal-core/tree/main/src/natal/frontend/webui). The core simulation guide ends here. Interface changes should additionally trace data through sessions, serialization, REST/WebSocket protocols, and display components.
