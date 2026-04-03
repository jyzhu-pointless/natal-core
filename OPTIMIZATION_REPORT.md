# Spatial Simulation Kernel Optimization Report

## Executive Summary
This optimization cycle focused on the spatial simulation kernels used by age-structured populations, especially migration and reproduction hot paths in Numba kernels.

After aligning cache/warm-up conditions and re-running the same scenario, end-to-end runtime for `demos/spatial_hex.py` (`spatial.run(5)`, 100x100 HexGrid) is now in the **0.6-0.7s** range.

## Final Performance Metrics
- Baseline (historical runs): around 2.8-3.0s for `spatial.run(5)` in the same large-grid scenario.
- Latest stable range (cache state aligned): **0.6-0.7s** for `spatial.run(5)`.
- Equivalent per-tick envelope: about 0.12-0.14s/tick.
- Quality gates: pytest, pyright, ruff all pass.

## What Was Optimized

### 1. Reproduction-side allocation and recomputation reductions
**Files**:
- `src/natal/algorithms.py`
- `src/natal/kernels/simulation_kernels.py`
- `src/natal/kernels/spatial_simulation_kernels.py`

**Key changes**:
- Removed avoidable temporary allocations and defensive copies in hot loops.
- Reworked mating/fertilization internals to reduce transient tensor pressure.
- Precomputed offspring probability tensor once per tick and reused across demes.

**Why it helps**:
- Less allocation and fewer large temporary arrays reduce memory bandwidth pressure.
- Reusing config-only tensors removes repeated per-deme work.

### 2. Kernel-mode migration complexity correction
**File**:
- `src/natal/kernels/spatial_simulation_kernels.py`

**Key changes**:
- Added kernel-specialized routing path for `migration_mode == 1`.
- Built compact kernel offset table once per migration call.
- Built sparse destination rows on-the-fly per source deme inside `prange`.
- Avoided constructing global `n_demes x n_demes` migration row tables in kernel mode.

**Why it helps**:
- For fixed small kernels, routing work scales close to `O(kernel_nnz * n_demes)`.
- This avoids dense/global row-materialization overhead that can dominate runtime.

## Core Algorithmic Shape (Current)

### Reproduction stage
- For each deme:
  - Recover virgin-female mass from `female_total - stored_sperm_total`.
  - Handle virgin and sperm-coupled mass consistently.
  - Use precomputed offspring probability tensor for genotype transitions.

### Migration stage
- For each source deme:
  - Build compact valid destinations from kernel offsets + topology boundary policy.
  - Sample or deterministically split outbound mass bucket-wise.
  - Accumulate destination updates in thread-local buffers.
- Reduce thread-local buffers at the end for synchronized migration semantics.

## Measurement Notes (Important)
- Numba performance comparisons are sensitive to warm-up and cache state.
- Use a fixed procedure: warm up first, then run repeated timed measurements.
- Compare stable ranges, not one-off values.

## Validation
- `pytest`
- `pyright`
- `ruff check src demos`

All quality gates pass after these optimization and documentation updates.

## Conclusion
The current spatial kernel implementation is no longer bottlenecked by earlier migration row-construction patterns in kernel mode, and the latest aligned measurement indicates `spatial.run(5)` has moved into the 0.6-0.7s runtime band for the benchmarked 100x100 HexGrid scenario.
