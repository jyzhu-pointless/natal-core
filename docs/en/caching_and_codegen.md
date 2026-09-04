# Execution Backends

The `natal` engine ships one implementation per backend family under
`natal.backends`:

- `reference` — the pure-Python golden implementation (always available).
- `rust` — the adapter for the native `natal._engine_rs` extension (built
  with maturin; `rust_backend_available()` probes it).

There is no source-code generation layer anymore: the reference kernels are
plain NumPy functions and the Rust engine is a precompiled native module.
Deterministic semantics are asserted against the reference by the test
suite, and the per-deme Python dispatch is the only non-Rust execution
vehicle for spatial models.
