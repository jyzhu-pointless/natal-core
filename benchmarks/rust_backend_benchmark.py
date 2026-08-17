"""Compare the Rust and Numba age-structured lifecycle backends.

Run with the release extension installed, e.g.:

    maturin build --release
    pip install --force-reinstall --no-deps <wheel>
    python benchmarks/rust_backend_benchmark.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import natal as nt
from natal.engine.backends.rust_backend import rust_backend_available

N_TICKS = 100
REPEATS = 3


def build(stochastic: bool, name: str):
    """Build a moderately sized 4-locus population."""
    species = nt.Species.from_dict(
        name=f"rust_bench_{stochastic}_{name}",
        structure={
            "chr1": {
                "l1": ["A", "B"],
                "l2": ["C", "D"],
                "l3": ["E", "F"],
                "l4": ["G", "H"],
            }
        },
        gamete_labels=["default"],
    )
    pop = (
        nt.AgeStructuredPopulation.setup(species, stochastic=stochastic, name=name)
        .age_structure(5, 3)
        .reproduction(
            eggs_per_female=8.0,
            fixed_egg_count=True,
            female_age_based_mating_rate=1.0,
            male_age_based_mating_rate=1.0,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
        )
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
        .competition(juvenile_growth_mode=1, carrying_capacity=20000)
        .build()
    )
    n_ztypes = pop.config.n_ztypes
    rng = np.random.default_rng(42)
    pop.state.individual_count[:] = rng.integers(0, 50, size=(2, 5, n_ztypes))
    pop.state.sperm_storage[:] = rng.integers(0, 3, size=(5, n_ztypes, n_ztypes))
    pop.state.sperm_storage[:3, :, :] = 0.0
    for age in range(3, 5):
        for female_ztype in range(n_ztypes):
            total = pop.state.sperm_storage[age, female_ztype, :].sum()
            if total > pop.state.individual_count[0, age, female_ztype]:
                pop.state.sperm_storage[age, female_ztype, :] = 0.0
    return pop


def measure_run(pop, n_steps: int) -> float:
    """Measure the in-kernel ``run(n)`` path."""
    start = time.perf_counter()
    pop.run(n_steps, record_every=0)
    return time.perf_counter() - start


def measure_tick_loop(pop, n_steps: int) -> float:
    """Measure repeated ``run_tick()`` calls."""
    start = time.perf_counter()
    for _ in range(n_steps):
        pop.run_tick()
    return time.perf_counter() - start


def benchmark(stochastic: bool) -> None:
    """Warm up, then measure both backends and report medians."""
    reference = build(stochastic, "numba")
    rust_pop = build(stochastic, "rust").enable_rust_backend(seed=1)

    measure_run(reference, 2)
    measure_run(rust_pop, 2)

    timings: dict[str, list[float]] = {
        "numba run(n)": [],
        "rust run(n)": [],
        "numba run_tick loop": [],
        "rust run_tick loop": [],
    }
    for _ in range(REPEATS):
        timings["numba run(n)"].append(measure_run(reference, N_TICKS))
        timings["rust run(n)"].append(measure_run(rust_pop, N_TICKS))
        timings["numba run_tick loop"].append(measure_tick_loop(reference, N_TICKS))
        timings["rust run_tick loop"].append(measure_tick_loop(rust_pop, N_TICKS))

    print(f"\n=== stochastic={stochastic} ===")
    print(f"{'path':22s} {'total_ms':>10s} {'per_tick_ms':>12s}")
    for label, values in timings.items():
        total_ms = statistics.median(values) * 1000.0
        print(f"{label:22s} {total_ms:10.1f} {total_ms / N_TICKS:12.3f}")

    nonzero_sperm = int(np.count_nonzero(reference.state.sperm_storage > 1e-12))
    nonzero_sperm_total = int(reference.state.sperm_storage.size)
    print(
        f"note: deterministic vs stochastic workloads differ; "
        f"compare only within the same mode. "
        f"final sperm nonzero: {nonzero_sperm}/{nonzero_sperm_total}."
    )


def main() -> None:
    if not rust_backend_available():
        raise SystemExit("natal._engine_rs is not built.")
    benchmark(stochastic=False)
    benchmark(stochastic=True)


if __name__ == "__main__":
    main()
