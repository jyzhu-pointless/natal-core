# type:ignore
"""
Micro-benchmark to measure individual tick costs in spatial simulation.
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


def benchmark_individual_stages():
    """Measure cost of individual ticks."""
    spatial = build_hex_spatial_population()
    spatial.run(1)  # warmup

    print("=" * 60)
    print("Per-tick Performance Analysis")
    print("=" * 60)

    times = []
    for run_idx in range(3):
        spatial = build_hex_spatial_population()
        spatial.run(1)  # warmup

        start = time.perf_counter()
        spatial.run(1)   # measure 1 tick
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Run {run_idx + 1}, 1 tick: {elapsed:.4f}s")

    print(f"\nAverage per tick: {np.mean(times):.4f}s (±{np.std(times):.4f}s)")


if __name__ == "__main__":
    benchmark_individual_stages()
