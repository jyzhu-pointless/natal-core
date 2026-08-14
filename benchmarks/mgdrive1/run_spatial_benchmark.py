"""Run the matched NATAL Core and MGDrivE1 spatial benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from statistics import fmean

import numpy as np
from numpy.typing import NDArray

from .spatial_benchmark import BenchmarkRecord, benchmark_natal


def _standardized_difference(
    mean_difference: NDArray[np.float64],
    standard_error: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return z-scores without hiding zero-variance disagreements.

    Args:
        mean_difference: Difference between engine means.
        standard_error: Combined standard error for each comparison.

    Returns:
        Finite z-scores when standard error is positive, zero for exact
        zero-variance agreement, and signed infinity for zero-variance
        disagreement.
    """
    result = np.zeros_like(mean_difference)
    positive_error = standard_error > 0.0
    np.divide(
        mean_difference,
        standard_error,
        out=result,
        where=positive_error,
    )
    disagreement = (~positive_error) & (mean_difference != 0.0)
    result[disagreement] = np.copysign(
        np.inf,
        mean_difference[disagreement],
    )
    return result


def _write_natal_records(
    path: Path,
    records: tuple[BenchmarkRecord, ...],
    *,
    rows: int,
    cols: int,
    n_days: int,
) -> None:
    """Write NATAL benchmark records in the MGDrivE1-compatible schema.

    Args:
        path: Output CSV path.
        records: Timed NATAL records.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        n_days: Number of simulated days.
    """
    fieldnames = [
        "engine",
        "mode",
        "rows",
        "cols",
        "n_days",
        "repeat",
        "elapsed_seconds",
        "male_AA",
        "male_Aa",
        "male_aa",
        "female_AA_mate_AA",
        "female_AA_mate_Aa",
        "female_AA_mate_aa",
        "female_Aa_mate_AA",
        "female_Aa_mate_Aa",
        "female_Aa_mate_aa",
        "female_aa_mate_AA",
        "female_aa_mate_Aa",
        "female_aa_mate_aa",
        "spatial_male_aa_radius2",
        "spatial_female_aa_radius2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            values = [
                *record.adult_male.tolist(),
                *record.adult_female.reshape(-1).tolist(),
            ]
            row: dict[str, str | int | float] = {
                "engine": "NATAL-Core-local-kernel",
                "mode": record.mode,
                "rows": rows,
                "cols": cols,
                "n_days": n_days,
                "repeat": record.repeat,
                "elapsed_seconds": record.elapsed_seconds,
            }
            for name, value in zip(fieldnames[7:19], values, strict=True):
                row[name] = value
            row["spatial_male_aa_radius2"] = record.male_aa_radius2
            row["spatial_female_aa_radius2"] = record.female_aa_radius2
            writer.writerow(row)


def _run_mgdrive(
    *,
    mode: str,
    rows: int,
    cols: int,
    n_days: int,
    repeats: int,
    seed: int,
    output_path: Path,
) -> None:
    """Run the pinned MGDrivE1 R benchmark process.

    Args:
        mode: Deterministic or stochastic mode.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        n_days: Number of simulated days.
        repeats: Number of timed replicates.
        seed: Base random seed.
        output_path: Output CSV path.
    """
    script = Path(__file__).with_name("benchmark_spatial.R")
    subprocess.run(
        [
            "Rscript",
            str(script),
            mode,
            str(rows),
            str(cols),
            str(n_days),
            str(repeats),
            str(seed),
            str(output_path),
        ],
        check=True,
        env=os.environ.copy(),
    )


# Public names let the resumable formal runner reuse the same engine adapter
# and output schema as the single-run command.
run_mgdrive = _run_mgdrive
write_natal_records = _write_natal_records


def _summary(
    natal_records: tuple[BenchmarkRecord, ...],
    *,
    mgdrive_paths: tuple[Path, ...],
) -> dict[str, float]:
    """Build a compact timing summary.

    Args:
        natal_records: Timed NATAL records.
        mgdrive_paths: Existing MGDrivE1 result CSV paths.

    Returns:
        JSON-serializable timing summary.
    """
    result: dict[str, float] = {}
    for mode in ("deterministic", "stochastic"):
        elapsed = [
            record.elapsed_seconds
            for record in natal_records
            if record.mode == mode
        ]
        if elapsed:
            result[f"natal_{mode}_mean_seconds"] = fmean(elapsed)
    for path in mgdrive_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        mode = rows[0]["mode"]
        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        mean_elapsed = fmean(elapsed)
        result[f"mgdrive_{mode}_mean_seconds"] = mean_elapsed
        natal_elapsed = result.get(f"natal_{mode}_mean_seconds")
        if isinstance(natal_elapsed, float):
            result[f"{mode}_speedup"] = mean_elapsed / natal_elapsed
        natal_mode_records = tuple(
            record for record in natal_records if record.mode == mode
        )
        numeric_names = [
            name
            for name in rows[0]
            if name.startswith(("male_", "female_"))
        ]
        natal_values = np.stack(
            [
                np.concatenate(
                    (
                        record.adult_male,
                        record.adult_female.reshape(-1),
                    )
                )
                for record in natal_mode_records
            ]
        )
        mgdrive_values = np.array(
            [
                [float(row[name]) for name in numeric_names]
                for row in rows
            ],
            dtype=np.float64,
        )
        if mode == "deterministic":
            result["deterministic_max_abs_error"] = float(
                np.max(np.abs(natal_values - mgdrive_values))
            )
        elif natal_values.shape[0] > 1 and mgdrive_values.shape[0] > 1:
            standard_error = np.sqrt(
                natal_values.var(axis=0, ddof=1) / natal_values.shape[0]
                + mgdrive_values.var(axis=0, ddof=1)
                / mgdrive_values.shape[0]
            )
            mean_difference = (
                natal_values.mean(axis=0) - mgdrive_values.mean(axis=0)
            )
            category_z = _standardized_difference(
                mean_difference,
                standard_error,
            )
            natal_total = natal_values.sum(axis=1)
            mgdrive_total = mgdrive_values.sum(axis=1)
            total_standard_error = np.sqrt(
                natal_total.var(ddof=1) / natal_total.size
                + mgdrive_total.var(ddof=1) / mgdrive_total.size
            )
            total_z = _standardized_difference(
                np.array(
                    [natal_total.mean() - mgdrive_total.mean()],
                    dtype=np.float64,
                ),
                np.array([total_standard_error], dtype=np.float64),
            )
            result["stochastic_total_mean_z"] = float(total_z[0])
            result["stochastic_max_category_mean_abs_z"] = float(
                np.max(np.abs(category_z))
            )
        spatial_names = [
            "spatial_male_aa_radius2",
            "spatial_female_aa_radius2",
        ]
        natal_spatial = np.array(
            [
                [record.male_aa_radius2, record.female_aa_radius2]
                for record in natal_mode_records
            ],
            dtype=np.float64,
        )
        mgdrive_spatial = np.array(
            [
                [float(row[name]) for name in spatial_names]
                for row in rows
            ],
            dtype=np.float64,
        )
        if mode == "deterministic":
            result["deterministic_spatial_moment_max_abs_error"] = float(
                np.max(np.abs(natal_spatial - mgdrive_spatial))
            )
        elif natal_spatial.shape[0] > 1:
            spatial_standard_error = np.sqrt(
                natal_spatial.var(axis=0, ddof=1) / natal_spatial.shape[0]
                + mgdrive_spatial.var(axis=0, ddof=1)
                / mgdrive_spatial.shape[0]
            )
            spatial_difference = (
                natal_spatial.mean(axis=0) - mgdrive_spatial.mean(axis=0)
            )
            spatial_z = _standardized_difference(
                spatial_difference,
                spatial_standard_error,
            )
            result["stochastic_spatial_moment_max_abs_z"] = float(
                np.max(np.abs(spatial_z))
            )
    return result


def _validate_summary(summary: dict[str, float]) -> None:
    """Reject cross-engine results that fail numerical equivalence.

    Args:
        summary: Timing and correctness metrics from both engines.

    Raises:
        RuntimeError: If deterministic error exceeds ``1e-8`` or any
            stochastic mean difference exceeds three combined standard errors.
    """
    for name in (
        "deterministic_max_abs_error",
        "deterministic_spatial_moment_max_abs_error",
    ):
        if name not in summary:
            continue
        value = summary[name]
        if not np.isfinite(value):
            raise RuntimeError(f"{name} is not finite")
        if value > 1e-8:
            raise RuntimeError(f"{name} exceeds the 1e-8 equivalence limit")
    for name in (
        "stochastic_total_mean_z",
        "stochastic_max_category_mean_abs_z",
        "stochastic_spatial_moment_max_abs_z",
    ):
        if name not in summary:
            continue
        value = summary[name]
        if not np.isfinite(value):
            raise RuntimeError(f"{name} is not finite")
        if abs(value) > 3.0:
            raise RuntimeError(f"{name} exceeds the three-sigma limit")


def main() -> None:
    """Run requested engines and write raw CSV plus a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15)
    parser.add_argument("--cols", type=int, default=15)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stochastic-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--with-mgdrive", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("both", "deterministic", "stochastic"),
        default="both",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natal_records: list[BenchmarkRecord] = []
    mgdrive_paths: list[Path] = []
    modes = (
        (False, True)
        if args.mode == "both"
        else (args.mode == "stochastic",)
    )
    for stochastic in modes:
        mode = "stochastic" if stochastic else "deterministic"
        mode_repeats = (
            args.stochastic_repeats if stochastic else args.repeats
        )
        records = benchmark_natal(
            stochastic=stochastic,
            rows=args.rows,
            cols=args.cols,
            n_days=args.days,
            repeats=mode_repeats,
        )
        natal_records.extend(records)
        _write_natal_records(
            args.output_dir / f"natal-{mode}.csv",
            records,
            rows=args.rows,
            cols=args.cols,
            n_days=args.days,
        )
        if args.with_mgdrive:
            mgdrive_path = args.output_dir / f"mgdrive-{mode}.csv"
            _run_mgdrive(
                mode=mode,
                rows=args.rows,
                cols=args.cols,
                n_days=args.days,
                repeats=mode_repeats,
                seed=args.seed,
                output_path=mgdrive_path,
            )
            mgdrive_paths.append(mgdrive_path)

    summary = _summary(
        tuple(natal_records),
        mgdrive_paths=tuple(mgdrive_paths),
    )
    _validate_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
