# Benchmarking & Profiling Scripts

This directory contains performance analysis and optimization scripts for natal-core.

## Scripts Overview

### Profiling Scripts

- **profile_spatial_hex_run.py**: Profile spatial.run(5) with 100x100 demes using cProfile
  ```bash
  python benchmarks/profile_spatial_hex_run.py
  ```
  Identifies Python-layer hotspots and cumulative function call patterns.

- **profile_single_deme_repro.py**: Profile single-deme reproduction (older approach, may need updates)
  ```bash
  python benchmarks/profile_single_deme_repro.py
  ```

- **profile_repro_direct.py**: Direct kernel profiling using low-level SimulationKernels API
  ```bash
  python benchmarks/profile_repro_direct.py
  ```

### Benchmarking Scripts

- **benchmark_kernel_overhead.py**: Compare state management overhead vs kernel execution time
  ```bash
  python benchmarks/benchmark_kernel_overhead.py
  ```
  Measures cost of stacking/unstacking arrays and compares with pure kernel time.

- **benchmark_stages.py**: Per-tick performance analysis
  ```bash
  python benchmarks/benchmark_stages.py
  ```
  Measures individual tick execution cost across multiple runs.

## Usage Tips

### Running from Project Root
All scripts use PYTHONPATH-relative imports, so run from project root:
```bash
cd /Users/pointless/Desktop/work/natal-core
python benchmarks/profile_spatial_hex_run.py
```

### Interpreting Results

**cProfile Output**:
- Look for functions with high `cumtime` and high `ncalls`
- `tottime` shows time spent in function itself
- `cumtime` includes time in called functions

**Benchmark Output**:
- Average ± std dev gives performance stability indication
- Variance > 10% suggests system-level noise or thermal effects
- Single best/worst runs often excluded from analysis

### Key Metrics from Last Run
- **Spatial.run(5) total**: ~2.74-2.88s (5 runs)
- **Per-tick cost**: ~0.574s ± 0.053s
- **Python overhead**: 1.4% (state management)
- **Numba kernel time**: 98.6% (bottleneck)

## Performance Analysis Results

📊 See `OPTIMIZATION_REPORT.md` in project root for detailed findings.

### Recent Optimization
Copy removal in spatial kernels (run_spatial_reproduction, run_spatial_survival, run_spatial_aging) achieved:
- 29% speedup in state stack/unstack (3.74s → 2.64s)
- 56% faster pytest suite (15.83s → 6.89s)
- 1-2% overall demo improvement

### Bottleneck Identified
98.6% of time spent in @njit Numba code:
- Not visible in cProfile (Numba functions appear as single call)
- Dominated by algorithmic cost (mating, fertilization, offspring generation)
- Further optimization requires:
  - Numba algorithm tuning
  - Parallelization flags adjustment
  - Memory layout optimization

## Future Optimization Directions

1. **Numba Algorithm Level**
   - Profile mating probability matrix computation
   - Optimize offspring probability tensor construction
   - Reduce allocations in inner loops

2. **Parallelization**
   - Test different n_jobs values
   - Verify prange scheduling is optimal
   - Check thread synchronization overhead

3. **State Management** (higher risk)
   - Cache frequently accessed arrays
   - Consider mutable state structure (refactor from NamedTuple)
   - Batch state updates
