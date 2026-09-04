"""Reference vs Rust performance benchmark.

Each mode runs in a separate subprocess so the two backends never share a
process state.

Run:
    python demos/bench_backends.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Literal, cast

_Backend = Literal["python", "rust"]

WARMUP = 1
BENCH = 100
WORKER = __file__


def run_subprocess(backend: str) -> float:  # str: first-class CLI argument
    """Run the worker once under *backend* and return the elapsed seconds."""
    proc = subprocess.run(
        [sys.executable, WORKER, "--worker", backend, str(WARMUP), str(BENCH)],
        capture_output=True, text=True, timeout=300,
    )
    for line in proc.stderr.splitlines():
        if "Traceback" in line or "Error:" in line:
            print(f"  {line}", file=sys.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("ELAPSED="):
            return float(line.split("=")[1])
    return 0.0


def _worker(backend: str, warmup: int, bench: int) -> None:
    """Run one benchmark workload for *backend*.

    Args:
        backend: One of the setup() backend selectors ("python"/"rust").
    """
    from natal.frontend.genetics import Species
    from natal.frontend.population.age_structured import AgeStructuredPopulation

    sp = Species.from_dict(
        name="bench_sub",
        structure={"chr1": {"loc": ["A1", "A2", "A3", "A4", "A5"]}},
        gamete_labels=["default"],
    )
    genos = [str(g) for g in sp.get_all_genotypes()]
    dist = {g: [0, 1000] for g in genos}
    pop = (
        AgeStructuredPopulation.setup(sp, stochastic=False, backend=cast(_Backend, backend))
        .age_structure(n_ages=8, new_adult_age=2)
        .initial_state(individual_count={"female": dist, "male": dist})
        .reproduction(eggs_per_female=50, sex_ratio=0.5)
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
        .competition(carrying_capacity=50_000, juvenile_growth_mode="logistic")
        .build()
    )

    pop.run(warmup, finish=False)
    t0 = time.perf_counter()
    pop.run(bench, finish=False)
    elapsed = time.perf_counter() - t0
    print(f"ELAPSED={elapsed:.6f}")


def main() -> None:
    """Benchmark both execution backends and print the comparison."""
    import sys as _sys

    if "--worker" in _sys.argv:
        _backend = _sys.argv[_sys.argv.index("--worker") + 1]
        _warmup = int(_sys.argv[_sys.argv.index("--worker") + 2])
        _bench = int(_sys.argv[_sys.argv.index("--worker") + 3])
        _worker(_backend, _warmup, _bench)
        return

    from natal.backends.rust.rust_backend import rust_backend_available

    has_rust = rust_backend_available()
    print("=" * 70)
    print("  Reference vs Rust benchmark")
    print("=" * 70)
    print("  Model:     25 genotypes x 8 ages x 2 sexes")
    print("  Initial:   1 000 adults / genotype / sex  ->  50 000 total")
    print(f"  Warmup:    {WARMUP} ticks")
    print(f"  Benchmark: {BENCH} ticks")
    print()

    print("  [1/2] Pure-Python reference .... ", end="", flush=True)
    t0 = time.perf_counter()
    t_python = run_subprocess("python")
    tw_python = time.perf_counter() - t0
    print(f"{t_python:.3f}s sim  ({tw_python:.1f}s wall)")

    if has_rust:
        print("  [2/2] Rust                  .... ", end="", flush=True)
        t0 = time.perf_counter()
        t_rust = run_subprocess("rust")
        tw_rust = time.perf_counter() - t0
        print(f"{t_rust:.3f}s sim  ({tw_rust:.1f}s wall)")
    else:
        t_rust = 0.0
        tw_rust = 0.0
        print("  [2/2] Rust                  .... unavailable (extension not built)")

    print()
    print("=" * 70)
    print(f"  Python:  {t_python:8.3f}s  ({BENCH / t_python:8.0f} t/s)  "
          f"wall: {tw_python:.0f}s")
    if has_rust:
        print(f"  Rust:    {t_rust:8.3f}s  ({BENCH / t_rust:8.0f} t/s)  "
              f"wall: {tw_rust:.0f}s")
        print(f"  Speedup: {t_python / t_rust:.0f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
