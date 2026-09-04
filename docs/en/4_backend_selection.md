# Backend Selection and Performance

`natal` ships two execution backends. `auto` (the default) picks the fast
native backend when it is available and falls back to the always-present
pure-Python reference:

| Backend | Selector | Notes |
|---------|----------|-------|
| Rust (native extension) | `backend="rust"` | Compiled `natal._engine_rs`; the fastest path. Built with `maturin develop`. |
| Pure-Python reference | `backend="python"` | Always available; reference-oracle semantics, best for debugging. |
| Auto | `backend="auto"` (default) | Rust when the extension imports, otherwise the reference. |

## Selecting a backend

Backends are selected at build time, per population:

```python
from natal.frontend.genetics import Species
from natal.frontend.population.age_structured import AgeStructuredPopulation

species = Species.from_dict("demo", {"chr1": {"loc": ["WT", "Dr"]}})
pop = AgeStructuredPopulation.setup(
    species, stochastic=False, backend="auto",  # auto / rust / python
).age_structure(4, 2).reproduction(eggs_per_female=50).build()

print(pop.using_rust_backend)  # True when the native extension is active
```

`pop.enable_rust_backend(seed=...)` opts a population into Rust after
construction; `pop.disable_rust_backend()` returns to the reference.

The retired compiled-backend selector (`backend="numba"`) raises
`ValueError` with a migration hint — use `"rust"` or `"python"`.

## What the backends share

Both backends execute the same tick order (first hook → reproduction →
early hook → survival → late hook → aging) and the same deterministic
arithmetic. Deterministic (`stochastic=False`) trajectories are bitwise
identical between them; stochastic runs use independent RNG streams.

Hooks run through the same declarative CSR interpreter on both paths;
single-parameter Python callbacks are bridged into the native session.

## Performance guidance

- The Rust backend is the main performance lever for large models
  (many genotypes, many demes, long runs).
- The reference backend is O(genotypes² · ages · demes) per tick and is
  intended as the golden oracle and debugging target, not for production
  throughput.
- `compress=True` at setup shrinks the genotype axis to reachable types
  and speeds both backends.
