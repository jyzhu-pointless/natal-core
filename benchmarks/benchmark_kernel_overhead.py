# type:ignore
"""
Compare execution time of run_spatial_fn directly to identify if time is in Numba or Python.
"""

import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
from demos.spatial_hex import build_hex_spatial_population


def benchmark_runs():
    """Run 3 iterations of spatial.run(5) and measure times."""
    times = []

    for run_idx in range(3):
        spatial = build_hex_spatial_population()
        spatial.run(1)  # warmup

        start = time.perf_counter()
        spatial.run(5)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Run {run_idx + 1}: {elapsed:.4f}s")

    print(f"\nAverage: {np.mean(times):.4f}s (±{np.std(times):.4f}s)")
    return times


def benchmark_kernel_time():
    """Measure just the Numba kernel call without state management."""
    spatial = build_hex_spatial_population()
    spatial.run(1)  # warmup

    # Manually reconstruct what run() does, but time only the kernel call
    n_steps = 5

    # Setup
    hooks = spatial._demes[0].get_compiled_event_hooks()
    run_fn = hooks.run_spatial_fn
    registry = hooks.registry
    config = spatial._shared_config()

    # Method 1: Measure including state stack/unstack
    print("=" * 60)
    print("Method 1: Including state stack/unstack")
    start = time.perf_counter()
    for _ in range(3):
        ind_all, sperm_all = spatial._stack_deme_state_arrays()
        final_state_tuple, _history, was_stopped = run_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=int(spatial._tick),
            n_ticks=int(n_steps),
            adjacency=spatial._adjacency,
            migration_mode=spatial._migration_mode_code,
            topology_rows=int(spatial._topology.rows),
            topology_cols=int(spatial._topology.cols),
            topology_wrap=bool(spatial._topology.wrap),
            migration_kernel=spatial._migration_kernel_array(),
            kernel_include_center=bool(spatial._kernel_include_center),
            migration_rate=float(spatial._migration_rate),
        )
        spatial._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))
    elapsed = time.perf_counter() - start
    print(f"  3 runs (with stack/unstack): {elapsed:.4f}s ({elapsed/3:.4f}s per run)")

    # Reset for next test
    spatial = build_hex_spatial_population()
    spatial.run(1)  # warmup
    hooks = spatial._demes[0].get_compiled_event_hooks()
    run_fn = hooks.run_spatial_fn
    registry = hooks.registry
    config = spatial._shared_config()

    # Method 2: Measure with pre-stacked arrays
    print("=" * 60)
    print("Method 2: Pre-stacked arrays (one allocation)")
    start = time.perf_counter()

    # Stack once
    ind_all, sperm_all = spatial._stack_deme_state_arrays()

    # Time 3 runs with same arrays
    t0 = time.perf_counter()
    for _ in range(3):
        final_state_tuple, _history, was_stopped = run_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=0,
            n_ticks=5,
            adjacency=spatial._adjacency,
            migration_mode=spatial._migration_mode_code,
            topology_rows=int(spatial._topology.rows),
            topology_cols=int(spatial._topology.cols),
            topology_wrap=bool(spatial._topology.wrap),
            migration_kernel=spatial._migration_kernel_array(),
            kernel_include_center=bool(spatial._kernel_include_center),
            migration_rate=float(spatial._migration_rate),
        )
        # Don't apply state for this benchmark

    elapsed = time.perf_counter() - t0
    print(f"  3 runs (pre-stacked, no unstack): {elapsed:.4f}s ({elapsed/3:.4f}s per run)")


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: spatial.run(5) with 100x100 demes (3 runs)")
    print("=" * 60)
    benchmark_runs()

    print("\n")
    benchmark_kernel_time()
