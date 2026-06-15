"""Numba vs pure-Python performance benchmark.

Each mode runs in a separate subprocess because ``NUMBA_DISABLE_JIT``
must be set before ``import numba``.

Run:
    python demos/bench_numba.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

WARMUP = 1
BENCH = 100
WORKER = os.path.join(os.path.dirname(__file__), "_bench_worker.py")


def run_subprocess(disable_numba: bool) -> float:
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1"} if disable_numba else None
    proc = subprocess.run(
        [sys.executable, WORKER, str(WARMUP), str(BENCH)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    for line in proc.stderr.splitlines():
        if "Traceback" in line or "Error:" in line:
            print(f"  {line}", file=sys.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("ELAPSED="):
            return float(line.split("=")[1])
    return 0.0


def main() -> None:
    print("=" * 70)
    print("  Numba vs pure-Python benchmark")
    print("=" * 70)
    print("  Model:     25 genotypes x 8 ages x 2 sexes")
    print("  Initial:   1 000 adults / genotype / sex  ->  50 000 total")
    print(f"  Warmup:    {WARMUP} ticks")
    print(f"  Benchmark: {BENCH} ticks")
    print()

    print("  [1/2] Numba JIT (default) ... ", end="", flush=True)
    t0 = time.perf_counter()
    t_numba = run_subprocess(disable_numba=False)
    tw_numba = time.perf_counter() - t0
    print(f"{t_numba:.3f}s sim  ({tw_numba:.1f}s wall)")

    print("  [2/2] Pure Python ............ ", end="", flush=True)
    t0 = time.perf_counter()
    t_python = run_subprocess(disable_numba=True)
    tw_python = time.perf_counter() - t0
    print(f"{t_python:.3f}s sim  ({tw_python:.1f}s wall)")

    print()
    print("=" * 70)
    if t_numba > 0:
        print(f"  Numba:   {t_numba:8.3f}s  ({BENCH / t_numba:8.0f} t/s)  "
              f"wall: {tw_numba:.0f}s")
        print(f"  Python:  {t_python:8.3f}s  ({BENCH / t_python:8.0f} t/s)  "
              f"wall: {tw_python:.0f}s")
        print(f"  Speedup: {t_python / t_numba:.0f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
