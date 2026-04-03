# type:ignore
"""
Simple profiling wrapper around spatial_hex.py to identify reproduction hotspots.
"""

import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cProfile
import pstats
import io
import time
from demos.spatial_hex import build_hex_spatial_population


def profile_spatial_run():
    """Profile a single spatial run(5) call."""
    print("=" * 80)
    print("PROFILING: spatial.run(5) with 100x100 demes")
    print("=" * 80)

    spatial = build_hex_spatial_population()

    # Warmup
    spatial.run(1)

    # Profile the actual run
    profiler = cProfile.Profile()
    profiler.enable()

    spatial.run(5)

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(60)

    print(s.getvalue())


if __name__ == "__main__":
    profile_spatial_run()
