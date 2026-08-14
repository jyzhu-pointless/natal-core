"""Generate daily cross-engine trajectories for visual validation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from natal.numba.compat import set_numba_seed

from .lifecycle import DailyRelease
from .slim_panmmictic import SLIM_SCRIPT
from .spatial_benchmark import (
    SpatialPatchState,
    build_hex_benchmark_scenario,
    step_spatial,
)

TrajectoryTask = Callable[[Path], None]


def _summarize_state(state: SpatialPatchState) -> tuple[float, float]:
    """Return total and recessive-genotype adult abundance.

    Args:
        state: Current spatial lifecycle state.

    Returns:
        Total adults and adults with the ``aa`` diploid genotype.
    """
    adult_total = float(
        state.adult_male.sum()
        + state.adult_female.sum()
        + state.unmated_female.sum()
    )
    aa_adult = float(
        state.adult_male[:, 2].sum()
        + state.adult_female[:, 2, :].sum()
        + state.unmated_female[:, 2].sum()
    )
    return adult_total, aa_adult


def _natal_trajectories(
    *,
    scenario_name: str,
    rows: int,
    cols: int,
    stochastic: bool,
    repeats: int,
    n_days: int,
    seed: int | None,
) -> pd.DataFrame:
    """Generate NATAL daily trajectories for one matched scenario.

    Args:
        scenario_name: Frozen scenario label.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        stochastic: Whether to sample demographic events.
        repeats: Number of independent trajectories.
        n_days: Number of daily transitions.
        seed: Base random seed for the single-deme panmictic scenario. Spatial
            scenarios must leave it unset because their parallel RNG state is
            intentionally runtime-managed.

    Returns:
        Long-form daily trajectory table.
    """
    scenario = build_hex_benchmark_scenario(
        rows=rows,
        cols=cols,
        migration_rate=0.0 if rows * cols == 1 else 0.05,
        stochastic=stochastic,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        unmated_female=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        adult_female=np.zeros((3, 3), dtype=np.float64),
        eggs=np.zeros(3, dtype=np.float64),
    )
    center_deme = ((rows - 1) // 2) * cols + (cols - 1) // 2
    if rows * cols == 1 and seed is None:
        raise ValueError("panmictic NATAL trajectories require a seed")
    if rows * cols > 1 and seed is not None:
        raise ValueError("spatial NATAL trajectories do not accept a seed")
    rows_out: list[dict[str, str | int | float]] = []
    for repeat_index in range(repeats):
        if seed is not None:
            # Single-deme execution has no parallel-deme RNG ambiguity.
            set_numba_seed(seed + repeat_index)
        state = scenario.state
        for transition in range(n_days + 1):
            adult_total, aa_adult = _summarize_state(state)
            rows_out.append(
                {
                    "engine": "NATAL Core",
                    "scenario": scenario_name,
                    "mode": (
                        "stochastic" if stochastic else "deterministic"
                    ),
                    "repeat": repeat_index + 1,
                    "transition": transition,
                    "adult_total": adult_total,
                    "aa_adult": aa_adult,
                }
            )
            if transition == n_days:
                continue
            absolute_day = 1 + transition + 1
            state = step_spatial(
                state,
                scenario.config,
                migration_kernel=scenario.migration_kernel,
                rows=rows,
                cols=cols,
                migration_rate=scenario.migration_rate,
                stochastic=stochastic,
                release=(
                    (center_deme, release)
                    if absolute_day == 25
                    else None
                ),
            )
    return pd.DataFrame(rows_out)


def _slim_trajectories(
    *,
    repeats: int,
    n_days: int,
    seed: int,
) -> pd.DataFrame:
    """Generate daily SLiM panmictic trajectories.

    Args:
        repeats: Number of independent SLiM subprocesses.
        n_days: Number of daily transitions.
        seed: Base random seed.

    Returns:
        Long-form daily trajectory table.

    Raises:
        FileNotFoundError: If the SLiM executable is unavailable.
        RuntimeError: If a subprocess fails or emits an invalid trace.
    """
    executable = os.environ.get("CLUSTER_SLIM_BIN") or shutil.which("slim")
    if executable is None:
        raise FileNotFoundError("SLiM executable was not found")
    rows_out: list[dict[str, str | int | float]] = []
    for repeat_index in range(repeats):
        process = subprocess.run(
            (
                executable,
                "-s",
                str(seed + repeat_index),
                "-d",
                f"N_DAYS={n_days}",
                "-d",
                "TRACE=T",
                str(SLIM_SCRIPT),
            ),
            capture_output=True,
            check=False,
            text=True,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"SLiM trace failed: {message}")
        trace_lines = [
            line.removeprefix("TRACE:")
            for line in process.stdout.splitlines()
            if line.startswith("TRACE:")
        ]
        if len(trace_lines) != n_days + 1:
            raise RuntimeError("SLiM trace has an invalid number of days")
        for line in trace_lines:
            values = line.split(",")
            if len(values) != 3:
                raise RuntimeError("SLiM trace row must contain three values")
            transition, adult_total, aa_adult = map(int, values)
            rows_out.append(
                {
                    "engine": "SLiM",
                    "scenario": "panmictic",
                    "mode": "stochastic",
                    "repeat": repeat_index + 1,
                    "transition": transition,
                    "adult_total": adult_total,
                    "aa_adult": aa_adult,
                }
            )
        if (repeat_index + 1) % 100 == 0:
            print(
                f"SLiM trajectories {repeat_index + 1}/{repeats}",
                flush=True,
            )
    return pd.DataFrame(rows_out)


def _write_natal(
    destination: Path,
    *,
    scenario_name: str,
    rows: int,
    cols: int,
    stochastic: bool,
    repeats: int,
    n_days: int,
    seed: int | None,
) -> None:
    """Generate and write one NATAL trajectory table."""
    _natal_trajectories(
        scenario_name=scenario_name,
        rows=rows,
        cols=cols,
        stochastic=stochastic,
        repeats=repeats,
        n_days=n_days,
        seed=seed,
    ).to_csv(destination, index=False)


def _write_slim(
    destination: Path,
    *,
    repeats: int,
    n_days: int,
    seed: int,
) -> None:
    """Generate and write the SLiM trajectory table."""
    _slim_trajectories(
        repeats=repeats,
        n_days=n_days,
        seed=seed,
    ).to_csv(destination, index=False)


def _run_mgdrive(
    destination: Path,
    *,
    mode: str,
    rows: int,
    cols: int,
    n_days: int,
    repeats: int,
    seed: int,
) -> None:
    """Run the R trajectory generator.

    Args:
        mode: Deterministic or stochastic execution.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        n_days: Number of daily transitions.
        repeats: Number of trajectories.
        seed: Base random seed.
        destination: Long-form output CSV.
    """
    script = Path(__file__).with_name("generate_spatial_trajectory.R")
    subprocess.run(
        (
            "Rscript",
            str(script),
            mode,
            str(rows),
            str(cols),
            str(n_days),
            str(repeats),
            str(seed),
            str(destination),
        ),
        check=True,
        env=os.environ.copy(),
    )


def _valid(path: Path, *, expected_rows: int) -> bool:
    """Return whether a cached trajectory table is complete."""
    return path.is_file() and len(pd.read_csv(path)) == expected_rows


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize means, standard deviations, and empirical 95% intervals."""
    def quantile_025(values: pd.Series) -> float:
        """Return the empirical lower 95% interval boundary."""
        return float(values.quantile(0.025))

    def quantile_975(values: pd.Series) -> float:
        """Return the empirical upper 95% interval boundary."""
        return float(values.quantile(0.975))

    group_columns = ["engine", "scenario", "mode", "transition"]
    summary = raw.groupby(group_columns, sort=False).agg(
        n=("repeat", "nunique"),
        adult_total_mean=("adult_total", "mean"),
        adult_total_sd=("adult_total", "std"),
        adult_total_q025=("adult_total", quantile_025),
        adult_total_q975=("adult_total", quantile_975),
        aa_adult_mean=("aa_adult", "mean"),
        aa_adult_sd=("aa_adult", "std"),
        aa_adult_q025=("aa_adult", quantile_025),
        aa_adult_q975=("aa_adult", quantile_975),
    )
    return summary.reset_index().fillna(0.0)


def main() -> None:
    """Generate or resume all daily validation trajectories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--spatial-repeats", type=int, default=100)
    parser.add_argument("--panmictic-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mgdrive-r-lib",
        type=Path,
        default=Path("/private/tmp/natal-r-lib"),
    )
    args = parser.parse_args()
    days = int(args.days)
    spatial_repeats = int(args.spatial_repeats)
    panmictic_repeats = int(args.panmictic_repeats)
    seed = int(args.seed)
    output_dir = Path(args.output_dir)
    mgdrive_r_lib = Path(args.mgdrive_r_lib)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MGDRIVE_R_LIB"] = str(mgdrive_r_lib)
    tasks: tuple[tuple[str, int, TrajectoryTask], ...] = (
        (
            "natal-spatial-deterministic.csv",
            days + 1,
            partial(
                _write_natal,
                scenario_name="spatial",
                rows=15,
                cols=15,
                stochastic=False,
                repeats=1,
                n_days=days,
                seed=None,
            ),
        ),
        (
            "mgdrive-spatial-deterministic.csv",
            days + 1,
            partial(
                _run_mgdrive,
                mode="deterministic",
                rows=15,
                cols=15,
                n_days=days,
                repeats=1,
                seed=seed,
            ),
        ),
        (
            "natal-spatial-stochastic.csv",
            (days + 1) * spatial_repeats,
            partial(
                _write_natal,
                scenario_name="spatial",
                rows=15,
                cols=15,
                stochastic=True,
                repeats=spatial_repeats,
                n_days=days,
                seed=None,
            ),
        ),
        (
            "mgdrive-spatial-stochastic.csv",
            (days + 1) * spatial_repeats,
            partial(
                _run_mgdrive,
                mode="stochastic",
                rows=15,
                cols=15,
                n_days=days,
                repeats=spatial_repeats,
                seed=seed,
            ),
        ),
        (
            "natal-panmictic-stochastic.csv",
            (days + 1) * panmictic_repeats,
            partial(
                _write_natal,
                scenario_name="panmictic",
                rows=1,
                cols=1,
                stochastic=True,
                repeats=panmictic_repeats,
                n_days=days,
                seed=seed,
            ),
        ),
        (
            "slim-panmictic-stochastic.csv",
            (days + 1) * panmictic_repeats,
            partial(
                _write_slim,
                repeats=panmictic_repeats,
                n_days=days,
                seed=seed,
            ),
        ),
    )
    completed_paths: list[Path] = []
    for name, expected_rows, task in tasks:
        path = output_dir / name
        if _valid(path, expected_rows=expected_rows):
            print(f"SKIP {name}", flush=True)
        else:
            print(f"START {name}", flush=True)
            task(path)
            if not _valid(path, expected_rows=expected_rows):
                raise RuntimeError(f"{name} is incomplete")
            print(f"DONE {name}", flush=True)
        completed_paths.append(path)
    raw = pd.concat(
        [pd.read_csv(path) for path in completed_paths],
        ignore_index=True,
    )
    raw.to_csv(output_dir / "trajectory_raw.csv", index=False)
    _summary(raw).to_csv(
        output_dir / "trajectory_summary.csv",
        index=False,
    )
    print("COMPLETE trajectory validation", flush=True)


if __name__ == "__main__":
    main()
