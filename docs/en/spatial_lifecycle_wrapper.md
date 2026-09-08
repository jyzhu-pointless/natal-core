# Spatial Lifecycle Execution

This document describes the runtime execution architecture of `SpatialPopulation`
(after the slice-5 data plane landed). The former njit codegen spatial wrapper
pipeline (`compile_spatial_lifecycle_wrapper`, `NUMBA_ENABLED`, `numba`/`prange`
imports, the `natal.numba` utility layer) and the pure-Python reference engine
have been fully removed. The Rust native extension is the only execution
engine; every path shares the same hook plans and the same migration data
plane.

## Execution Model

One spatial tick runs in two phases:

```
per-deme lifecycle (per-deme granularity: first hook -> reproduction -> early
  hook -> density regulation + survival -> late hook -> aging)
      |
      v
unified migration (runtime rate column x frozen folded CSR)
```

- **Engine session**: the container's tick driver calls
  per-deme lifecycle inside the one session-owned spatial kernel; deme `d`
  derives its random stream from `seed ^ d` (see
  `enable_rust_backend(seed=...)`), and the migration stage runs inside the
  same session on the same per-deme streams.

## Migration Data Plane (slice-5)

At build time `fold_migration_csr()` folds the migration configuration into the
CSR triple (`indptr` / `dest_idx` / `weights`); at runtime it is just
`outbound * weight`:

- **adjacency mode**: each source row stores the raw adjacency values in
  destination-ascending order.
- **kernel mode**: the per-source historical row builder is reproduced in kernel
  row-major visit order; invalid (out-of-grid) offsets are dropped or wrapped;
  each emitted entry is scaled by the reciprocal of the kernel total -- or of
  the valid-row total when `adjust_on_edge=True`.
- Both modes end with a row normalization (a final emitted-row-sum division of
  the already-scaled entries during the fold), so boundary demes send their full
  outbound quota to valid destinations just like interior demes. The
  `adjust_on_edge` switch exists for historical bit-exact parity with the old
  pipeline, not to change the destination distribution.
- Migration rate and CSR are separate: the runtime `migration_rate` is a
  `(n_demes, S, A)` column (write-protected view); actual outflow =
  rate x weight.
- **Changing the topology means rebuilding**: the CSR is folded at build time;
  changing topology/adjacency/kernel parameters requires rebuilding the
  population (`pop.params.tensor_write("migration_rate", ...)` only changes the
  rate column, not the topology).

## Hook Execution

- Declarative hooks compile into CSR plans executed at event boundaries
  inside the engine session;
- Callback hooks (`TickContext`) are bridged across the boundary inside the
  session; out-of-band surfaces (`trigger_event`, finish events) invoke them
  directly;
- Per-deme `priority` only applies within a deme; no global order across demes;
- `@hook(..., deme=[0, 2])` pins a hook to specific demes (default `"*"`).

## User API

```python
from natal.frontend.spatial import batch_setting
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import build_adjacency_matrix

pop = (
    SpatialPopulation.builder(species=sp, n_demes=4, pop_type="age_structured")
    .setup(name="demo", stochastic=False)
    ...
    .migration(adjacency=..., migration_rate=0.1)
    .build()
)
pop.run(n_steps=10, record_every=1)
pop.params.tensor_write("migration_rate", {"F": 0.2, "M": 0.05})  # runtime rate change
```

After construction, runtime parameter writes have exactly two entries:
`pop.params.tensor_write(...)` (bulk, recommended) and
`deme(i).write_ecology(...)` / `write_genetics(...)` (single deme).
**The `SpatialPopulation.update()` chain has been removed.**
