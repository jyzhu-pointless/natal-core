"""Validate SLiM genotype-scaling models against NATAL stochastic dynamics.

For each neutral MGDrivE1-compatible cube, both engines simulate the same
panmictic stochastic lifecycle with a release on transition 24.  The script
compares the final adult genotype distributions across independent repeats
and exits nonzero if any standardized mean difference exceeds a tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.mgdrive1.cluster.run_natal_condition import _build_scenario
from benchmarks.mgdrive1.slim_panmmictic import (
    CUBE_GENOTYPES,
    benchmark_slim,
)
from benchmarks.mgdrive1.spatial_benchmark import run_spatial
from natal.numba.compat import set_numba_seed


def _natal_replicates(
    *,
    cube_dir: Path,
    cube: str,
    repeats: int,
    n_days: int,
    seed: int,
) -> list[np.ndarray]:
    """Run NATAL stochastic panmictic replicates for one cube."""
    scenario, release, _, _ = _build_scenario(
        cube_dir=cube_dir,
        cube_kind=cube,
        population_scale=1,
        rows=1,
        cols=1,
        stochastic=True,
        migration_rate=0.0,
    )
    set_numba_seed(seed)
    # Exclude JIT warm-up from the sampled replicates.
    from benchmarks.mgdrive1.spatial_benchmark import step_spatial

    step_spatial(
        scenario.state,
        scenario.config,
        migration_kernel=scenario.migration_kernel,
        rows=1,
        cols=1,
        migration_rate=0.0,
        stochastic=True,
    )
    outcomes: list[np.ndarray] = []
    for repeat in range(repeats):
        set_numba_seed(seed + repeat)
        result = run_spatial(
            scenario.state,
            scenario.config,
            migration_kernel=scenario.migration_kernel,
            rows=1,
            cols=1,
            migration_rate=0.0,
            n_days=n_days,
            release_day=25,
            release_deme=0,
            release=release,
            stochastic=True,
        )
        male = np.asarray(result.adult_male[0], dtype=float)
        female = np.asarray(result.adult_female[0], dtype=float)
        outcomes.append(np.concatenate(([male.sum() + female.sum()], male, female.ravel())))
    return outcomes


def _slim_replicates(
    *,
    cube: str,
    repeats: int,
    n_days: int,
    seed: int,
) -> list[np.ndarray]:
    """Run SLiM stochastic panmictic replicates for one cube."""
    outcomes: list[np.ndarray] = []
    for record in benchmark_slim(
        repeats=repeats,
        n_days=n_days,
        seed=seed,
        cube=cube,
    ):
        outcomes.append(
            np.concatenate(
                (
                    [
                        float(
                            record.adult_male.sum()
                            + record.adult_female.sum()
                            + record.unmated_female_total
                        )
                    ],
                    record.adult_male,
                    record.adult_female.ravel(),
                )
            )
        )
    return outcomes


def _compare(left: list[np.ndarray], right: list[np.ndarray]) -> tuple[float, float]:
    """Return max and mean absolute standardized mean differences."""
    left_matrix = np.vstack(left)
    right_matrix = np.vstack(right)
    pooled = np.sqrt(
        (left_matrix.var(axis=0) + right_matrix.var(axis=0)) / 2.0
    )
    scale = np.where(pooled > 0.0, pooled, 1.0)
    standardized = np.abs(left_matrix.mean(axis=0) - right_matrix.mean(axis=0)) / scale
    return float(standardized.max()), float(standardized.mean())


def main() -> None:
    """Run the validation and report a per-cube summary table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--cube-dir",
        type=Path,
        default=Path(
            __file__
        ).resolve().parents[2]
        / "benchmark-results"
        / "northstar_20260808"
        / "cube_data",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Max allowed standardized mean difference in pooled-SD units",
    )
    args = parser.parse_args()
    if args.repeats < 4:
        parser.error("--repeats must be at least 4")

    print(
        f"{'cube':<14}{'G':>4}{'max|d|':>10}{'mean|d|':>10}  verdict"
    )
    failed = False
    for cube in ("multiallele6", "twolocus9", "twolocus18"):
        natal = _natal_replicates(
            cube_dir=args.cube_dir,
            cube=cube,
            repeats=args.repeats,
            n_days=args.days,
            seed=args.seed,
        )
        slim = _slim_replicates(
            cube=cube,
            repeats=args.repeats,
            n_days=args.days,
            seed=args.seed + 10_000,
        )
        maximum, mean = _compare(natal, slim)
        verdict = "PASS" if maximum <= args.tolerance else "FAIL"
        failed = failed or maximum > args.tolerance
        print(
            f"{cube:<14}{CUBE_GENOTYPES[cube]:>4}"
            f"{maximum:>10.3f}{mean:>10.3f}  {verdict}"
        )
    if failed:
        raise SystemExit(
            f"standardized differences exceed tolerance {args.tolerance}"
        )


if __name__ == "__main__":
    main()
