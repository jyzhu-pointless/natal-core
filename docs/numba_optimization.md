# Numba Optimization Guide

This chapter introduces practical points related to Numba in NATAL, with the goal of helping you balance “debuggability” and “execution efficiency”.

This chapter does not pursue theoretical details, but focuses on the most common issues from a user perspective:

1. When the default configuration is fast enough.
2. When you should temporarily disable Numba for easier debugging.
3. How to run reproducible and interpretable performance tests.

## 1. The Role of Numba in NATAL

The core stage computations in NATAL are performed by numerical kernels, and Numba is responsible for JIT‑compiling those kernels into efficient machine code.

For users, a single rule of thumb:

- Numba is enabled by default, and that is usually the recommended configuration.

You do not need to manually decorate framework kernels; you only need to learn how to “temporarily disable” Numba during debugging, and keep it “enabled by default” in production.

## 2. Default Behaviour and How to Switch

### 2.1 Default Behaviour

The project enables Numba by default, so you get stable performance at typical scales.

### 2.2 Temporary Disable (Recommended for Debugging)

```python
from natal.numba_utils import numba_disabled

with numba_disabled():
    pop.run(n_steps=10)
```

This method has a clear scope; exiting the `with` block automatically restores the original state.

### 2.3 Global Enable / Disable

```python
from natal.numba_utils import disable_numba, enable_numba

disable_numba()
pop.run(n_steps=10)

enable_numba()
pop.run(n_steps=100)
```

This is suitable for batch debugging, but remember to re‑enable Numba when you finish debugging.

## 3. When to Disable Numba

Consider disabling Numba first in the following scenarios:

1. Error messages are not intuitive and make it hard to locate the business logic.
2. You are debugging a Hook or a combination of parameters and want to first confirm “logical correctness”.
3. You want to quickly print intermediate variables to verify whether the values meet expectations.

A typical workflow:

1. Run a small‑scale example inside `numba_disabled()` to ensure it works.
2. After confirming the logic, re‑enable Numba.
3. Run at production scale and record performance.

## 4. Debugging Output and Common Patterns

In the context of numerical kernels, prefer simple, stable output methods.

Example:

```python
# A safer, quick output form
print("x =", x)
```

Compared to complex string concatenation, this form behaves more consistently across different execution paths.

## 5. Advice for Performance Testing

Performance testing should be separated from “model correctness verification”.

### 5.1 Suggested Workflow

1. Fix model parameters and the random seed.
2. Run a warm‑up pass first to avoid the impact of first‑time compilation.
3. Then take the actual timing, and record environment information (machine, Python version, dependency versions).

### 5.2 Example

```python
import time

# Warm‑up
pop.run(n_steps=1, record_every=0)
pop.reset()

start = time.perf_counter()
pop.run(n_steps=200, record_every=0)
elapsed = time.perf_counter() - start

print(f"Elapsed: {elapsed:.3f}s, per tick: {elapsed / 200:.6f}s")
```

## 6. Main Factors Affecting Performance

At the user modelling level, focus primarily on:

1. The size of the state space (especially the number of genotypes and age classes).
2. The computational complexity and trigger frequency of Hooks.
3. The history recording density (`record_every` setting).

When tuning, starting with “reducing unnecessary recording” and “simplifying Hook computations” is often more direct than tweaking low‑level parameters.

## 7. About Caching and First Run

JIT compilation introduces an overhead on the first run – this is normal.

Common practice:

- Run a few steps as a warm‑up before taking official timings.
- If the environment or code changes significantly, re‑warm‑up before comparing performance data.

### 7.1 Cache State Can Significantly Affect Conclusions

In large Spatial scenarios, comparing two sets of data with inconsistent cache states can easily lead to wrong conclusions.

In practice, it is recommended to fix the measurement protocol:

1. Explicitly state whether you are comparing “first run” (including compilation) or “steady‑state run” (excluding first compilation).
2. Keep the same caching strategy in each comparison (e.g., always warm‑up first, or always clear and then warm‑up under the same conditions).
3. Run at least multiple repetitions and report intervals/means, not just a single result.

A common pitfall: after optimising, the code itself is faster, but because the cache state differs, the observed numbers may look “unchanged” or even “slower”.

### 7.2 A Recent Example from Spatial Kernels

Taking the `demos/spatial_hex.py` scenario as an example, after unifying the caching and warm‑up protocol, `spatial.run(5)` stably falls in the range of `0.6‑0.7s`. This number is meant as a reference range for “steady‑state end‑to‑end runtime”, not the first‑run compilation time.

## 8. A Practical Checklist

Before starting large‑scale experiments, quickly check:

- Have you completed small‑scale correctness verification?
- Have you performed a warm‑up before running the actual batch?
- Does `record_every` meet your current analysis needs (rather than being the densest default)?
- Have you kept necessary runtime logs and parameter snapshots?

## 9. Recommended Workflow

1. Early modelling phase: prioritise interpretability and debuggability.
2. After parameters are finalised: restore the default Numba configuration and run long‑term simulations.
3. When reporting results: record both “runtime” and “key biological metrics”, not just speed.

## 10. Chapter Summary

In NATAL, Numba is a “performance accelerator that works by default”, not a complex component that users must manually maintain.

For most modelling tasks, following these principles is enough:

- Keep it enabled by default during production runs.
- Temporarily disable it for logic debugging.
- Use reproducible procedures when evaluating performance.

---

## Related Chapters

- [Deep Dive into Simulation Kernels](simulation_kernels.md)
- [PopulationState and PopulationConfig](population_state_config.md)
- [Hook System](hooks.md)
- [Quick Start](quickstart.md)
