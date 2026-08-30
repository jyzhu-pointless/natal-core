"""Benchmark the Rust spatial backend against the Numba spatial kernel.

Measures both the in-kernel ``run(n)`` path and repeated ``run_tick()`` calls
for a moderately sized spatial population with deterministic and stochastic
modes.

Run with the release extension installed, e.g.:

    maturin build --release
    pip install --force-reinstall --no-deps <wheel>
    python benchmarks/rust_backend_spatial_benchmark.py
"""

from __future__ import annotations

import statistics
import time

import natal as nt
from natal.engine.backends.rust_backend import rust_backend_available
from natal.spatial.population import SpatialPopulation
from natal.spatial.topology import SquareGrid, build_adjacency_matrix

N_DEMES = 16
N_TICKS = 20
REPEATS = 3


def build_species(name: str) -> nt.Species:
    """Return a moderately sized biallelic species."""
    return nt.Species.from_dict(
        name=name,
        structure={
            "chr1": {
                "l1": ["A", "B"],
                "l2": ["C", "D"],
                "l3": ["E", "F"],
            }
        },
        gamete_labels=["default"],
    )


def build_deme(
    species: nt.Species,
    name: str,
    stochastic: bool,
    adult_count: float = 1000.0,
) -> nt.AgeStructuredPopulation:
    """Build one age-structured deme with non-empty adult population."""
    return (
        nt.AgeStructuredPopulation.setup(
            species, stochastic=stochastic, name=name
        )
        .age_structure(4, 2)
        .initial_state(
            individual_count={
                "female": {"A/C/E|A/C/E": [0.0, adult_count, adult_count, adult_count]},
                "male": {"A/C/E|A/C/E": [0.0, adult_count, adult_count, adult_count]},
            }
        )
        .reproduction(
            eggs_per_female=8.0,
            fixed_egg_count=True,
            female_age_based_mating_rate=1.0,
            male_age_based_mating_rate=1.0,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
        )
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
        .competition(juvenile_growth_mode=1, carrying_capacity=10000)
        .build()
    )


def build_spatial(species: nt.Species, stochastic: bool, prefix: str) -> SpatialPopulation:
    """Build a spatial population with a square-grid topology."""
    demes = [
        build_deme(species, f"{prefix}_deme_{i}", stochastic)
        for i in range(N_DEMES)
    ]
    adjacency = build_adjacency_matrix(
        SquareGrid(
            rows=4,
            cols=4,
            neighborhood="von_neumann",
            wrap=False,
        ),
        row_normalize=True,
    )
    return SpatialPopulation(demes=demes, adjacency=adjacency, migration_rate=0.2)


def measure_run(pop: SpatialPopulation, n_steps: int) -> float:
    """Measure the in-kernel ``run(n)`` path without history recording."""
    start = time.perf_counter()
    pop.run(n_steps, record_every=0, clear_history_on_start=True)
    return time.perf_counter() - start


def measure_tick_loop(pop: SpatialPopulation, n_steps: int) -> float:
    """Measure repeated ``run_tick()`` calls."""
    start = time.perf_counter()
    for _ in range(n_steps):
        pop.run_tick()
    return time.perf_counter() - start


def benchmark(stochastic: bool) -> None:
    """Warm up and compare Numba vs Rust spatial backends."""
    species = build_species(f"spatial_bench_{stochastic}")
    reference = build_spatial(species, stochastic, "numba")
    rust_pop = build_spatial(species, stochastic, "rust").enable_rust_backend(seed=1)

    # Warm up both code paths.
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

    print(f"\n=== stochastic={stochastic} (n_demes={N_DEMES}, n_ticks={N_TICKS}) ===")
    print(f"{'path':24s} {'total_ms':>10s} {'per_tick_ms':>12s} {'speedup':>8s}")
    for label, values in timings.items():
        total_ms = statistics.median(values) * 1000.0
        per_tick = total_ms / N_TICKS
        speedup = ""
        if label.startswith("rust"):
            base_label = "numba" + label[len("rust"):]
            base_values = timings.get(base_label)
            if base_values:
                base_ms = statistics.median(base_values) * 1000.0
                speedup = f"{base_ms / total_ms:.2f}x"
        print(f"{label:24s} {total_ms:10.1f} {per_tick:12.3f} {speedup:>8s}")


def main() -> None:
    if not rust_backend_available():
        raise SystemExit("natal._engine_rs is not built.")
    benchmark(stochastic=False)
    benchmark(stochastic=True)


if __name__ == "__main__":
    main()
