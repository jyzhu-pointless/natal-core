# Numba Optimization Guide

This chapter covers practical Numba-related topics in NATAL, aiming to help you strike a balance between "debuggability" and "execution efficiency."

This chapter does not pursue theoretical details but focuses on the most common usage-level issues:

1. When the default setting is already fast enough.
2. When you should temporarily disable Numba for debugging.
3. How to conduct reproducible and explainable performance tests.

## 1. The Role of Numba in NATAL

The core stage computations in NATAL are performed by numerical kernels, and Numba handles JIT-compiling these kernels into efficient machine code.

For users, there is one key takeaway:

- Numba is enabled by default, and this is typically the recommended configuration.

You do not need to manually add decorators to framework kernels; you only need to learn to "temporarily disable" it during debugging and keep it "enabled by default" in production.

## 2. Default Behavior and Toggle Methods

### 2.1 Default Behavior

The project enables Numba by default to ensure stable performance at typical simulation scales.

### 2.2 Temporarily Disable (Recommended Debugging Approach)

```python
from natal.numba_utils import numba_disabled

with numba_disabled():
    pop.run(n_steps=10)
```

This approach has a clear scope, and the original state is automatically restored upon exiting the `with` block.

### 2.3 Global Enable/Disable

```python
from natal.numba_utils import disable_numba, enable_numba

disable_numba()
pop.run(n_steps=10)

enable_numba()
pop.run(n_steps=100)
```

Suitable for batch debugging, but remember to restore the setting after debugging is complete.

## 3. When to Disable Numba

Consider disabling Numba for troubleshooting in the following scenarios:

1. Error messages are not intuitive enough, making it difficult to locate the business logic.
2. You are debugging Hooks or parameter combinations and want to first confirm "logical correctness."
3. You want to quickly print intermediate variables to verify numerical values.

Common workflow:

1. First, run a small-scale example with `numba_disabled()`.
2. Confirm the logic is correct, then re-enable Numba.
3. Run at full scale and record performance.

## 4. Debug Output and Common Practices

In the context of numerical kernels, prefer simple, stable output methods.

Example:

```python
# A more reliable form of quick output
print("x =", x)
```

Compared to complex string concatenation, this form is more likely to maintain consistent behavior across different execution paths.

## 5. Recommendations for Performance Testing

Performance testing should be conducted separately from "model correctness validation."

### 5.1 Recommended Process

1. Fix model parameters and random seeds.
2. First, run a warm-up pass to avoid the initial compilation overhead affecting statistics.
3. Then run the actual timing and record environmental information (machine, Python version, dependency versions).

### 5.2 Example

```python
import time

# Warm-up
pop.run(n_steps=1, record_every=0)
pop.reset()

start = time.perf_counter()
pop.run(n_steps=200, record_every=0)
elapsed = time.perf_counter() - start

print(f"Elapsed: {elapsed:.3f}s, per tick: {elapsed / 200:.6f}s")
```

## 6. Main Factors Affecting Performance

At the user modeling layer, the following factors typically deserve the most attention:

1. State space size (especially the number of genotypes and age classes).
2. The computational complexity and trigger frequency of Hooks.
3. History recording density (the `record_every` setting).

When tuning, starting with "reducing unnecessary recording" and "simplifying Hook computations" is often more effective than fine-tuning low-level parameters.

## 7. Caching and First-Run Behavior

JIT compilation incurs a first-run overhead, which is normal.

Common practices:

- Run a small number of steps for warm-up before formal timing.
- If the environment or code changes significantly, re-warm-up before comparing performance data.

### 7.1 Cache State Can Significantly Affect Conclusions

In large Spatial scenarios, comparing two sets of data with "inconsistent cache states" can easily lead to incorrect conclusions.

In practice, it is recommended to standardize the "measurement approach":

1. Clearly state whether you are comparing "first-run" (including compilation) or "steady-state" (excluding first compilation).
2. Maintain a consistent caching strategy across comparisons (e.g., always warm up, or clear the cache and then warm up under the same conditions).
3. Run multiple repetitions and report ranges/means, not just single results.

A common pitfall is: the optimized code is actually faster, but because the cache states before and after differ, the observed result may appear "unchanged" or even "slower."

### 7.2 Example with Spatial Kernel Metrics

Taking the `demos/spatial_hex.py` scenario as an example, after standardizing the cache and warm-up conditions, `spatial.run(5)` consistently lands in the `0.6-0.7s` range. This figure serves as a reference interval for "steady-state end-to-end runtime," not first-compilation time.

## 8. A Practical Checklist

Before starting large-scale experiments, a quick check:

- Has small-scale correctness validation been completed?
- Has warm-up been performed before the formal batch run?
- Does `record_every` meet the current analysis needs (rather than using the default densest setting)?
- Have the necessary run logs and parameter snapshots been preserved?

## 9. Recommended Work Mode

1. Early modeling phase: Prioritize explainability and debuggability.
2. After parameters are finalized: Restore default Numba configuration and run long-term simulations.
3. When producing reports: Record both "runtime" and "key biological indicators" to avoid focusing solely on speed.

## 10. Chapter Summary

Numba in NATAL is a "performance accelerator available by default," not a complex component that users must manually maintain.

For most modeling tasks, following these principles is sufficient:

- Keep Numba enabled by default for formal runs.
- Temporarily disable it for logic troubleshooting.
- Use reproducible workflows for performance evaluation.

---

## Related Sections

- [Simulation Kernels Deep Dive](4_simulation_kernels.md)
- [PopulationState and PopulationConfig](4_population_state_config.md)
- [Hook System](2_hooks.md)
- [Quick Start](1_quickstart.md)
