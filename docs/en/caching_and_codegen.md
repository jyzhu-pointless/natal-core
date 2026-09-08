# Execution Engine

The `natal.backends` package holds one adapter:

- `rust` — the adapter for the native `natal._engine_rs` extension (built
  with maturin; `rust_backend_available()` probes it). It is the only
  execution engine.

There is no source-code generation layer anymore: the engine is a
precompiled native module. Deterministic semantics are locked by the test
suite against hand-derived expectations and frozen golden values.
