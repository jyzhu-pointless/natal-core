"""Slice-5 parity baseline: deterministic trajectories for spatial models.

Fixes the exact numerical trajectories of the default homogeneous and
heterogeneous spatial models so the slice-5 refactoring can be verified
bit-for-bit.  Each scenario runs a small deterministic simulation through
all three execution paths (Python dispatch, the compiled backend codegen, Rust) and
prints a digest of the stacked state after each tick.

Usage:
    python scripts/slice5_parity_baseline.py            # print digests
    python scripts/slice5_parity_baseline.py --save     # write JSON baseline
    python scripts/slice5_parity_baseline.py --check    # compare against baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _digest(arrays: list[np.ndarray]) -> str:
    """Stable digest of a list of float arrays (bytes, C-order)."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _state_arrays(pop) -> list[np.ndarray]:
    ind = np.stack([d.state.individual_count for d in pop.demes], axis=0)
    sperm_list = []
    for d in pop.demes:
        s = getattr(d.state, "sperm_storage", None)
        if s is None:
            ind_shape = d.state.individual_count.shape
            s = np.zeros((ind_shape[1], ind_shape[2], ind_shape[2]))
        sperm_list.append(s)
    sperm = np.stack(sperm_list, axis=0)
    return [ind, sperm]


def _species():
    from natal.frontend.genetics import Species

    return Species.from_dict(
        "s5",
        {
            "Chr1": {
                "R": ["r0", "r1"],
            }
        },
    )


def _scenario_homogeneous():
    """Homogeneous 3x2 square-grid model with a Gaussian kernel."""
    from natal.frontend.spatial import SpatialPopulation, SquareGrid, build_gaussian_kernel

    species = _species()
    topo = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=False)
    kernel = build_gaussian_kernel("square", size=3, sigma=0.9)
    pop = (
        SpatialPopulation.builder(species, n_demes=6, topology=topo)
        .setup(name="s5h", stochastic=False, continuous_sampling=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={"female": {"r0|r0": 400}, "male": {"r0|r0": 300}})
        .survival(female_age_based_survival=[0.9, 0.95, 0.9], male_age_based_survival=[0.9, 0.95, 0.9])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9, 0.9],
            male_age_based_mating_rate=[0.0, 0.9, 0.9],
            age_based_reproduction_rate=[0.0, 0.9, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=5000,
            low_density_growth_rate=6,
        )
        .migration(kernel=kernel, migration_rate=0.25)
        .build()
    )
    digests = {"t0": _digest(_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_state_arrays(pop))
    return digests


def _scenario_homogeneous_discrete():
    """Homogeneous discrete-generation hex model (mirrors spatial_hex_discrete)."""
    from natal.frontend.spatial import HexGrid, SpatialPopulation, build_gaussian_kernel

    species = _species()
    topo = HexGrid(rows=2, cols=2, wrap=False)
    kernel = build_gaussian_kernel("hex", size=3, sigma=0.8)
    pop = (
        SpatialPopulation.builder(
            species, n_demes=4, topology=topo, pop_type="discrete_generation"
        )
        .setup(name="s5d", stochastic=False)
        .initial_state(individual_count={"female": {"r0|r0": 500}, "male": {"r0|r0": 500}})
        .reproduction(eggs_per_female=50)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .migration(kernel=kernel, migration_rate=0.5)
        .build()
    )
    digests = {"t0": _digest(_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_state_arrays(pop))
    return digests


def _scenario_adjacency():
    """Adjacency-mode migration with an explicit dense matrix."""
    from natal.frontend.spatial import SpatialPopulation

    species = _species()
    adjacency = np.array(
        [
            [0.0, 0.6, 0.4, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.2, 0.3, 0.0, 0.5],
            [0.0, 0.7, 0.3, 0.0],
        ]
    )
    pop = (
        SpatialPopulation.builder(species, n_demes=4)
        .setup(name="s5a", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={"female": {"r0|r0": 800}, "male": {"r0|r0": 600}})
        .survival(female_age_based_survival=[0.9, 0.95], male_age_based_survival=[0.9, 0.95])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=4000,
            low_density_growth_rate=6,
        )
        .migration(adjacency=adjacency, migration_rate=0.3)
        .build()
    )
    digests = {"t0": _digest(_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_state_arrays(pop))
    return digests


def _scenario_heterogeneous():
    """Heterogeneous model: per-deme K gradient + two fitness variants."""
    from natal.frontend.spatial import SpatialPopulation, SquareGrid, batch_setting

    species = _species()
    topo = SquareGrid(rows=3, cols=2, neighborhood="von_neumann", wrap=False)
    k_values = [9000, 500, 9000, 9000, 500, 9000]
    pop = (
        SpatialPopulation.builder(species, n_demes=6, topology=topo)
        .setup(name="s5het", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting(
                [
                    {"female": {"r0|r0": 400}, "male": {"r0|r0": 300}},
                    {"female": {"r0|r1": 350}, "male": {"r0|r1": 250}},
                    {"female": {"r0|r0": 400}, "male": {"r0|r0": 300}},
                    {"female": {"r0|r0": 400}, "male": {"r0|r0": 300}},
                    {"female": {"r0|r1": 350}, "male": {"r0|r1": 250}},
                    {"female": {"r0|r0": 400}, "male": {"r0|r0": 300}},
                ]
            )
        )
        .survival(female_age_based_survival=[0.9, 0.95], male_age_based_survival=[0.9, 0.95])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=batch_setting(k_values),
            low_density_growth_rate=6,
        )
        .fitness(
            viability={"r1|r1": 0.4},
            mode="multiply",
        )
        .migration(adjacency=None, migration_rate=0.2)
        .build()
    )
    # adjacency=None + topology -> topology-derived adjacency.
    digests = {"t0": _digest(_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_state_arrays(pop))
    return digests


def _scenario_age_structured() -> dict[str, str]:
    """Deterministic age-structured panmictic model (reference path)."""
    from natal.frontend.builder import PopulationBuilder
    from natal.frontend.genetics import Species

    species = Species.from_dict(
        "s5age",
        {
            "Chr1": {
                "R": ["r0", "r1"],
            }
        },
    )
    pop = (
        PopulationBuilder.from_species(species)
        .age_structure(3, 1)
        .setup(name="s5age", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"r0|r0": 400},
                "male": {"r0|r0": 300},
            }
        )
        .survival(female_age_based_survival=[0.9, 0.95, 0.9], male_age_based_survival=[0.9, 0.95, 0.9])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9, 0.9],
            male_age_based_mating_rate=[0.0, 0.9, 0.9],
            age_based_reproduction_rate=[0.0, 0.9, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=5000,
            low_density_growth_rate=6,
        )
        .build()
    )
    digests = {"t0": _digest(_age_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_age_state_arrays(pop))
    return digests


def _scene_age_rust() -> dict[str, str]:
    """Same deterministic age-structured model driven through Rust."""
    from natal.frontend.builder import PopulationBuilder
    from natal.frontend.genetics import Species

    species = Species.from_dict(
        "s5age_rust",
        {
            "Chr1": {
                "R": ["r0", "r1"],
            }
        },
    )
    pop = (
        PopulationBuilder.from_species(species)
        .age_structure(3, 1)
        .setup(name="s5age_rust", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"r0|r0": 400},
                "male": {"r0|r0": 300},
            }
        )
        .survival(female_age_based_survival=[0.9, 0.95, 0.9], male_age_based_survival=[0.9, 0.95, 0.9])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9, 0.9],
            male_age_based_mating_rate=[0.0, 0.9, 0.9],
            age_based_reproduction_rate=[0.0, 0.9, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=5000,
            low_density_growth_rate=6,
        )
        .build()
        ._initialize_session(seed=3)
    )
    digests = {"t0": _digest(_age_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_age_state_arrays(pop))
    return digests


def _scenario_discrete_generation() -> dict[str, str]:
    """Deterministic discrete-generation panmictic model (reference path)."""
    from natal.frontend.builder import PopulationBuilder
    from natal.frontend.genetics import Species

    species = Species.from_dict(
        "s5disc",
        {
            "Chr1": {
                "R": ["r0", "r1"],
            }
        },
    )
    pop = (
        PopulationBuilder.for_discrete(species)
        .setup(name="s5disc", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"r0|r0": 400},
                "male": {"r0|r0": 400},
            }
        )
        .reproduction(eggs_per_female=50)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .build()
    )
    digests = {"t0": _digest(_discrete_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_discrete_state_arrays(pop))
    return digests


def _scene_discrete_rust() -> dict[str, str]:
    """Same deterministic discrete-generation model driven through Rust."""
    from natal.frontend.builder import PopulationBuilder
    from natal.frontend.genetics import Species

    species = Species.from_dict(
        "s5disc_rust",
        {
            "Chr1": {
                "R": ["r0", "r1"],
            }
        },
    )
    pop = (
        PopulationBuilder.for_discrete(species)
        .setup(name="s5disc_rust", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"r0|r0": 400},
                "male": {"r0|r0": 400},
            }
        )
        .reproduction(eggs_per_female=50)
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=1000,
            low_density_growth_rate=6,
        )
        .build()
        ._initialize_session(seed=17)
    )
    digests = {"t0": _digest(_discrete_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_discrete_state_arrays(pop))
    return digests


def _age_state_arrays(pop) -> list[np.ndarray]:
    """Digest arrays for an age-structured population state."""
    return [
        np.ascontiguousarray(pop.state.individual_count),
        np.ascontiguousarray(pop.state.sperm_storage),
    ]


def _discrete_state_arrays(pop) -> list[np.ndarray]:
    """Digest arrays for a discrete-generation population state."""
    return [np.ascontiguousarray(pop.state.individual_count)]


def _run_all() -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    results["homogeneous_kernel"] = _scenario_homogeneous()
    results["homogeneous_discrete"] = _scenario_homogeneous_discrete()
    results["adjacency_dense"] = _scenario_adjacency()
    results["heterogeneous_ecology"] = _scenario_heterogeneous()
    results["age_structured"] = _scenario_age_structured()
    results["discrete_generation"] = _scenario_discrete_generation()

    # Rust path (only when the extension is importable).
    try:
        from natal.backends.rust.rust_backend import rust_backend_available

        if rust_backend_available():
            results["rust_homogeneous_kernel"] = _rust_scenario(kernel_mode=True)
            results["rust_adjacency_dense"] = _rust_scenario(kernel_mode=False)
            results["rust_age_structured"] = _scene_age_rust()
            results["rust_discrete_generation"] = _scene_discrete_rust()
        else:
            print("[baseline] rust extension unavailable; rust scenarios skipped",
                  file=sys.stderr)
    except Exception as err:  # pragma: no cover - diagnostic path
        print(f"[baseline] rust scenarios failed: {err}", file=sys.stderr)

    return results


def _rust_scenario(kernel_mode: bool) -> dict[str, str]:
    """Rust-backed spatial run: lifecycle session + rust migration kernels."""
    from natal.frontend.spatial import SpatialPopulation, SquareGrid, build_gaussian_kernel

    species = _species()
    topo = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=False)
    builder = (
        SpatialPopulation.builder(species, n_demes=9, topology=topo)
        .setup(name="s5rust", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={"female": {"r0|r0": 400}, "male": {"r0|r0": 300}})
        .survival(female_age_based_survival=[0.9, 0.95], male_age_based_survival=[0.9, 0.95])
        .reproduction(
            eggs_per_female=80,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=5000,
            low_density_growth_rate=6,
        )
    )
    if kernel_mode:
        # NOTE: current rust kernel-migration contract requires kernel dims to
        # equal the topology dims; a 3x3 grid takes a 3x3 kernel.
        kernel = build_gaussian_kernel("square", size=3, sigma=0.9)
        builder = builder.migration(kernel=kernel, migration_rate=0.25)
    else:
        adjacency = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                if abs(i - j) == 1:
                    adjacency[i, j] = 1.0
                # Row-normalize below.
        row = adjacency.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        adjacency = adjacency / row
        builder = builder.migration(adjacency=adjacency, migration_rate=0.25)
    pop = builder.build()._initialize_session(seed=7)
    digests = {"t0": _digest(_state_arrays(pop))}
    for _ in range(4):
        pop.run_tick()
        digests[f"t{pop.tick}"] = _digest(_state_arrays(pop))
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write baseline JSON")
    parser.add_argument("--check", action="store_true", help="compare against baseline")
    args = parser.parse_args()

    results = _run_all()
    for name, digests in results.items():
        print(name)
        for tick, digest in digests.items():
            print(f"  {tick}: {digest}")

    baseline_path = REPO / "tests" / "data" / "slice5_parity_baseline.json"
    if args.save:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"[baseline] saved -> {baseline_path}", file=sys.stderr)
        return 0
    if args.check:
        if not baseline_path.exists():
            print(f"[baseline] missing baseline file {baseline_path}", file=sys.stderr)
            return 2
        expected = json.loads(baseline_path.read_text())
        failures = []
        for name, digests in expected.items():
            got = results.get(name)
            if got != digests:
                failures.append(f"{name}: expected={digests} got={got}")
        if failures:
            print("[baseline] PARITY FAILURES:", file=sys.stderr)
            for line in failures:
                print(f"  {line}", file=sys.stderr)
            return 1
        print("[baseline] parity OK", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
