"""Scaling benchmark for the deterministic spatial age-structured XPU model.

The benchmark builds the same age-structured lifecycle as
``reference_cpu.py`` on grids from 5x5 up to 20x20 and reports warm CPU vs XPU
run times.
"""

from __future__ import annotations

import argparse
import time

import numba
import numpy as np
import torch

import natal as nt
import reference_cpu
from gpu_model import SpatialAgeStructuredXPU

GRID_SIZES: list[tuple[int, int]] = [
    (5, 5),
    (10, 10),
    (15, 15),
    (20, 20),
]
N_TICKS = reference_cpu.N_TICKS

# Migration is now local stencil/neighborhood based on both CPU and XPU, so
# the dense O(D^2) adjacency bottleneck is gone. This is a practical upper
# guard for the current prototype; 200x200 is intentionally allowed.
MAX_SAFE_GRID = 200


def parse_args() -> argparse.Namespace:
    """Parse optional single-grid benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Scaling benchmark for spatial age-structured XPU model."
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        metavar="N",
        help="Run one square grid of size N x N (e.g. --grid 20).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=N_TICKS,
        help=f"Number of ticks per run (default: {N_TICKS}).",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help=(
            "Allow --grid larger than the practical upper guard. "
            "Only use this if you know the machine has enough RAM/VRAM."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run the stochastic age-structured lifecycle instead of deterministic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_states_for_size(rows: int, cols: int) -> list[dict[str, dict[str, list[float]]]]:
    """Build scaled initial states for an arbitrary grid size."""
    n_demes = rows * cols
    center = n_demes // 2

    base_wt_female = reference_cpu._scaled_initial_vector(
        reference_cpu._BASE_INITIAL_DISTRIBUTION["female"]["WT|WT"],
        reference_cpu.N_AGES,
    )
    base_wt_male = reference_cpu._scaled_initial_vector(
        reference_cpu._BASE_INITIAL_DISTRIBUTION["male"]["WT|WT"],
        reference_cpu.N_AGES,
    )
    base_drive_female = reference_cpu._scaled_initial_vector(
        reference_cpu._BASE_INITIAL_DISTRIBUTION["female"]["WT|Drive"],
        reference_cpu.N_AGES,
    )
    base_drive_male = reference_cpu._scaled_initial_vector(
        reference_cpu._BASE_INITIAL_DISTRIBUTION["male"]["WT|Drive"],
        reference_cpu.N_AGES,
    )

    base_state = {
        "female": {
            "WT|WT": base_wt_female,
            "WT|Drive": base_drive_female,
        },
        "male": {
            "WT|WT": base_wt_male,
            "WT|Drive": base_drive_male,
        },
    }

    states = [base_state for _ in range(n_demes)]

    release_female = reference_cpu.CENTER_DRIVE_RELEASE["female"]["WT|Drive"]
    release_male = reference_cpu.CENTER_DRIVE_RELEASE["male"]["WT|Drive"]

    center_state = {
        "female": {
            "WT|WT": [
                base_wt_female[a] - release_female[a] for a in range(reference_cpu.N_AGES)
            ],
            "WT|Drive": release_female,
        },
        "male": {
            "WT|WT": [
                base_wt_male[a] - release_male[a] for a in range(reference_cpu.N_AGES)
            ],
            "WT|Drive": release_male,
        },
    }
    states[center] = center_state
    return states


def build_cpu_population(
    rows: int, cols: int, *, stochastic: bool = False
) -> nt.SpatialPopulation:
    """Build an age-structured CPU population for a custom grid size."""
    species = nt.Species.from_dict(
        name=f"{reference_cpu.SPECIES_NAME}_{rows}x{cols}",
        structure={"chr1": {"loc": reference_cpu.ALLELES}},
    )
    states = build_states_for_size(rows, cols)
    topology = reference_cpu.SquareGrid(rows=rows, cols=cols)

    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=rows * cols,
            topology=topology,
            pop_type="age_structured",
        )
        .setup(
            name=f"bench_age_{rows}x{cols}",
            stochastic=stochastic,
            continuous_sampling=False,
        )
        .age_structure(
            n_ages=reference_cpu.N_AGES,
            new_adult_age=reference_cpu.NEW_ADULT_AGE,
        )
        .initial_state(individual_count=reference_cpu.batch_setting(states))
        .survival(
            female_age_based_survival=reference_cpu.FEMALE_SURVIVAL,
            male_age_based_survival=reference_cpu.MALE_SURVIVAL,
        )
        .reproduction(
            female_age_based_mating_rate=reference_cpu.FEMALE_MATING_RATE,
            male_age_based_mating_rate=reference_cpu.MALE_MATING_RATE,
            eggs_per_female=reference_cpu.EGGS_PER_FEMALE,
            fixed_egg_count=reference_cpu.FIXED_EGG_COUNT,
            sex_ratio=reference_cpu.SEX_RATIO,
            sperm_displacement_rate=reference_cpu.SPERM_DISPLACEMENT_RATE,
        )
        .presets(reference_cpu.build_drive())
        .fitness(
            viability=reference_cpu.VIABILITY_FITNESS,
            fecundity=reference_cpu.FECUNDITY_FITNESS,
        )
        .competition(
            competition_strength=reference_cpu.COMPETITION_STRENGTH,
            juvenile_growth_mode=reference_cpu.JUVENILE_GROWTH_MODE,
            low_density_growth_rate=reference_cpu.LOW_DENSITY_GROWTH_RATE,
            old_juvenile_carrying_capacity=reference_cpu.OLD_JUVENILE_CARRYING_CAPACITY,
            expected_num_new_adult_females=reference_cpu.EXPECTED_NUM_NEW_ADULT_FEMALES,
        )
        .migration(
            kernel=reference_cpu.MIGRATION_KERNEL,
            migration_rate=reference_cpu.MIGRATION_RATE,
            strategy=reference_cpu.MIGRATION_STRATEGY,
            adjust_migration_on_edge=reference_cpu.MIGRATION_ADJUST_ON_EDGE,
        )
        .build()
    )


def time_cpu_warm(
    pop: nt.SpatialPopulation,
    n_ticks: int,
    *,
    stochastic: bool = False,
    seed: int = 42,
) -> float:
    """Warm natal-core, reset, then time n_ticks without recording."""
    pop.run(n_steps=1, record_every=0)
    pop.reset()
    if stochastic:
        np.random.seed(seed)
    start = time.perf_counter()
    pop.run(n_steps=n_ticks, record_every=0)
    return time.perf_counter() - start


def time_xpu_warm(
    ind: np.ndarray,
    sperm: np.ndarray,
    cfg: object,
    rows: int,
    cols: int,
    n_ticks: int,
    device: torch.device,
    *,
    stochastic: bool = False,
    seed: int = 42,
) -> float:
    """Warm XPU kernels, then time a fresh model."""
    common = dict(
        individual_count=ind,
        sperm_storage=sperm,
        config=cfg,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=n_ticks,
        device=device,
        stochastic=stochastic,
        seed=seed,
        grid_shape=(rows, cols),
        wrap=False,
        migration_kernel=reference_cpu.MIGRATION_KERNEL,
        adjust_migration_on_edge=reference_cpu.MIGRATION_ADJUST_ON_EDGE,
    )
    _warm = SpatialAgeStructuredXPU(**common)
    _warm.run_no_history()

    model = SpatialAgeStructuredXPU(**common)
    torch.xpu.synchronize()
    start = time.perf_counter()
    model.run_no_history()
    torch.xpu.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()

    if args.grid is not None:
        if args.grid > MAX_SAFE_GRID and not args.allow_large:
            raise SystemExit(
                f"--grid {args.grid} exceeds the current practical guard "
                f"(MAX_SAFE_GRID={MAX_SAFE_GRID}). Pass --allow-large only if you "
                "have enough RAM/VRAM and know what you are doing."
            )
        grid_sizes = [(args.grid, args.grid)]
    else:
        grid_sizes = GRID_SIZES

    ticks = int(args.ticks)
    device = torch.device("xpu")
    mode = "stochastic" if args.stochastic else "deterministic"

    print(f"Device: {device}, xpu_available={torch.xpu.is_available()}")
    print(
        f"CPU threads: numba={numba.get_num_threads()}, "
        f"torch={torch.get_num_threads()}"
    )
    print(f"Mode: {mode}, seed={args.seed}")
    print(f"Ticks per run: {ticks}")
    print(f"{'demes':>7} {'rows':>5} {'CPU_s':>8} {'XPU_s':>8} {'CPU/XPU':>8}")
    results: list[tuple[int, int, int, float, float, float]] = []

    for rows, cols in grid_sizes:
        n_demes = rows * cols
        print(f"\nBenchmarking {rows}x{cols} = {n_demes} demes ...", flush=True)
        pop = build_cpu_population(rows, cols, stochastic=args.stochastic)

        ind = np.stack(
            [deme.state.individual_count for deme in pop.demes], axis=0
        )
        sperm = np.stack(
            [deme.state.sperm_storage for deme in pop.demes], axis=0
        )
        cfg = pop.deme(0).config

        cpu_t = time_cpu_warm(
            pop, ticks, stochastic=args.stochastic, seed=args.seed
        )
        xpu_t = time_xpu_warm(
            ind,
            sperm,
            cfg,
            rows,
            cols,
            ticks,
            device,
            stochastic=args.stochastic,
            seed=args.seed,
        )

        speedup = cpu_t / xpu_t if xpu_t > 0 else float("inf")
        results.append((n_demes, rows, cols, cpu_t, xpu_t, speedup))
        print(f"{n_demes:>7} {rows:>5} {cpu_t:>8.4f} {xpu_t:>8.4f} {speedup:>8.2f}")

    print("\n==== Summary ====")
    print(f"Mode: {mode}")
    print(f"{'demes':>7} {'CPU_s':>8} {'XPU_s':>8} {'CPU/XPU':>8}")
    for n_demes, _r, _c, cpu_t, xpu_t, speedup in results:
        print(f"{n_demes:>7} {cpu_t:>8.4f} {xpu_t:>8.4f} {speedup:>8.2f}")


if __name__ == "__main__":
    main()
