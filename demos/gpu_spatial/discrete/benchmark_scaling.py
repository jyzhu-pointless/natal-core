"""Quick scaling benchmark: natal-core CPU vs PyTorch XPU for different deme grids.

The goal is to find a deme count at which the Intel Arc iGPU starts to become
useful relative to the Numba CPU implementation. We report warm timings after
discarding one-time compile / initialisation costs.

This is a demo script outside natal-core's strict ``src`` type-check scope.
"""

# pyright: reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import natal as nt
import reference_cpu
from gpu_model import SpatialDiscreteXPU

GRID_SIZES: list[tuple[int, int]] = [
    (5, 5),
    (10, 10),
    (15, 15),
    (20, 20),
    (30, 30),
    (40, 40),
    (50, 50),
    (100, 100),
    (200, 200),
    (400, 400),
]
N_TICKS = 20


def build_cpu_population(
    rows: int, cols: int, *, stochastic: bool = False
) -> nt.SpatialPopulation:
    """Build a homogeneous CPU reference population for an arbitrary grid."""
    species = reference_cpu.build_species()
    n_demes = rows * cols
    center = n_demes // 2

    wild: dict[str, dict[str, float]] = {
        "female": {"WT|WT": 500.0},
        "male": {"WT|WT": 500.0},
    }
    release: dict[str, dict[str, float]] = {
        "female": {"WT|WT": 450.0, "Dr|WT": 50.0},
        "male": {"WT|WT": 450.0, "Dr|WT": 50.0},
    }
    states: list[dict[str, dict[str, float]]] = [
        dict(wild) for _ in range(n_demes)
    ]
    states[center] = release

    topology = reference_cpu.SquareGrid(rows=rows, cols=cols)

    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=n_demes,
            topology=topology,
            pop_type="discrete_generation",
        )
        .setup(
            name=f"bench_{rows}x{cols}",
            stochastic=stochastic,
            continuous_sampling=False,
        )
        .initial_state(individual_count=reference_cpu.batch_setting(states))
        .reproduction(
            eggs_per_female=reference_cpu.EGGS_PER_FEMALE,
            sex_ratio=reference_cpu.SEX_RATIO,
        )
        .survival(
            female_age0_survival=reference_cpu.FEMALE_AGE0_SURVIVAL,
            male_age0_survival=reference_cpu.MALE_AGE0_SURVIVAL,
        )
        .competition(
            carrying_capacity=int(reference_cpu.CARRYING_CAPACITY),
            juvenile_growth_mode="fixed",
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
    """Warm natal-core, reset, then time n_ticks without history recording."""
    pop.run(n_steps=1, record_every=0)
    pop.reset()
    if stochastic:
        np.random.seed(seed)
    start = time.perf_counter()
    pop.run(n_steps=n_ticks, record_every=0)
    return time.perf_counter() - start


def time_xpu_warm(
    state: np.ndarray,
    cfg: object,
    rows: int,
    cols: int,
    n_ticks: int,
    device: torch.device,
    *,
    stochastic: bool = False,
    seed: int = 42,
) -> float:
    """Run once to warm XPU kernels, then time a fresh model."""
    common = dict(
        state=state,
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
    _warm = SpatialDiscreteXPU(**common)
    _warm.run_no_history()

    model = SpatialDiscreteXPU(**common)
    torch.xpu.synchronize()
    start = time.perf_counter()
    model.run_no_history()
    torch.xpu.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run the stochastic discrete lifecycle instead of deterministic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-deme",
        type=int,
        default=2500,
        help="Only benchmark grids with n_demes <= this value.",
    )
    args = parser.parse_args()

    device = torch.device("xpu")
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"Device: {device}, xpu_available={torch.xpu.is_available()}")
    print(f"Mode: {mode}, seed={args.seed}")
    print(f"{'demes':>7} {'rows':>5} {'CPU_s':>8} {'XPU_s':>8} {'XPU/CPU':>8}")
    results: list[tuple[int, int, int, float, float, float]] = []

    grid_sizes = [
        (rows, cols)
        for rows, cols in GRID_SIZES
        if rows * cols <= args.max_deme
    ]
    for rows, cols in grid_sizes:
        n_demes = rows * cols
        print(f"\nBenchmarking {rows}x{cols} = {n_demes} demes ...", flush=True)
        pop = build_cpu_population(rows, cols, stochastic=args.stochastic)

        state = np.stack(
            [deme.state.individual_count for deme in pop.demes], axis=0
        )
        cfg = pop.deme(0).config

        cpu_t = time_cpu_warm(
            pop, N_TICKS, stochastic=args.stochastic, seed=args.seed
        )
        xpu_t = time_xpu_warm(
            state, cfg, rows, cols, N_TICKS, device,
            stochastic=args.stochastic, seed=args.seed,
        )

        ratio = xpu_t / cpu_t
        results.append((n_demes, rows, cols, cpu_t, xpu_t, ratio))
        print(f"{n_demes:>7} {rows:>5} {cpu_t:>8.4f} {xpu_t:>8.4f} {ratio:>8.2f}")

    print("\n==== Summary ====")
    print(f"Mode: {mode}")
    print(f"{'demes':>7} {'CPU_s':>8} {'XPU_s':>8} {'XPU/CPU':>8}")
    for n_demes, _r, _c, cpu_t, xpu_t, ratio in results:
        flag = "  <- GPU faster" if ratio < 1.0 else ""
        print(f"{n_demes:>7} {cpu_t:>8.4f} {xpu_t:>8.4f} {ratio:>8.2f}{flag}")


if __name__ == "__main__":
    main()
