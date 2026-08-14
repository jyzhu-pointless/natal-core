"""Freeze and statistically analyze the formal cross-engine benchmarks."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import stdtr

from .run_slim_benchmark import (
    compare_stochastic_records,
    holm_adjust,
    write_consistency,
)
from .slim_panmmictic import PanmicticRecord

MALE_COLUMNS = ("male_AA", "male_Aa", "male_aa")
FEMALE_COLUMNS = (
    "female_AA_mate_AA",
    "female_AA_mate_Aa",
    "female_AA_mate_aa",
    "female_Aa_mate_AA",
    "female_Aa_mate_Aa",
    "female_Aa_mate_aa",
    "female_aa_mate_AA",
    "female_aa_mate_Aa",
    "female_aa_mate_aa",
)
ADULT_COLUMNS = (*MALE_COLUMNS, *FEMALE_COLUMNS)
SPATIAL_MOMENT_COLUMNS = (
    "spatial_male_aa_radius2",
    "spatial_female_aa_radius2",
)


def _read_panmictic(path: Path) -> tuple[PanmicticRecord, ...]:
    """Read one aggregate panmictic result as validated records.

    Args:
        path: Aggregate NATAL or SLiM CSV.

    Returns:
        Immutable records used by the existing equivalence implementation.
    """
    table = pd.read_csv(path)
    records: list[PanmicticRecord] = []
    for _, row in table.iterrows():
        records.append(
            PanmicticRecord(
                engine=str(row["engine"]),
                mode=str(row["mode"]),
                repeat=int(row["repeat"]),
                day=int(row["day"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
                population_size=int(row["population_size"]),
                aquatic_total=int(row["aquatic_total"]),
                unmated_female_total=int(row["unmated_female_total"]),
                adult_male=row[list(MALE_COLUMNS)].to_numpy(dtype=float),
                adult_female=row[list(FEMALE_COLUMNS)]
                .to_numpy(dtype=float)
                .reshape(3, 3),
            )
        )
    return tuple(records)


def _performance_tests(source: Path) -> pd.DataFrame:
    """Calculate block-paired log-runtime tests and speedup intervals.

    Args:
        source: Directory containing aggregate formal CSV files.

    Returns:
        One row per planned runtime comparison.
    """
    comparisons = (
        (
            "spatial_deterministic",
            "NATAL Core",
            "MGDrivE1",
            "natal-spatial-deterministic.csv",
            "mgdrive-spatial-deterministic.csv",
        ),
        (
            "spatial_stochastic",
            "NATAL Core",
            "MGDrivE1",
            "natal-spatial-stochastic.csv",
            "mgdrive-spatial-stochastic.csv",
        ),
        (
            "panmictic_stochastic",
            "NATAL Core",
            "MGDrivE1",
            "natal-panmmictic.csv",
            "mgdrive-panmmictic.csv",
        ),
        (
            "panmictic_stochastic",
            "NATAL Core",
            "SLiM",
            "natal-panmmictic.csv",
            "slim-panmmictic.csv",
        ),
        (
            "panmictic_stochastic",
            "MGDrivE1",
            "SLiM",
            "mgdrive-panmmictic.csv",
            "slim-panmmictic.csv",
        ),
    )
    rows: list[dict[str, str | int | float]] = []
    raw_p: list[float] = []
    for scenario, left_engine, right_engine, left_name, right_name in comparisons:
        left = pd.read_csv(source / left_name)
        right = pd.read_csv(source / right_name)
        left_block = left.assign(
            log_elapsed=np.log(left["elapsed_seconds"])
        ).groupby("block")["log_elapsed"].mean()
        right_block = right.assign(
            log_elapsed=np.log(right["elapsed_seconds"])
        ).groupby("block")["log_elapsed"].mean()
        if not left_block.index.equals(right_block.index):
            raise RuntimeError(f"{scenario} block identifiers do not match")
        log_ratio = right_block.to_numpy() - left_block.to_numpy()
        interval = stats.t.interval(
            0.95,
            log_ratio.size - 1,
            loc=log_ratio.mean(),
            scale=stats.sem(log_ratio),
        )
        test = stats.ttest_rel(right_block, left_block)
        p_value = float(test.pvalue)
        raw_p.append(p_value)
        rows.append(
            {
                "scenario": scenario,
                "left_engine": left_engine,
                "right_engine": right_engine,
                "n_per_engine": len(left),
                "n_blocks": len(left_block),
                "left_geomean_seconds": float(
                    np.exp(np.log(left["elapsed_seconds"]).mean())
                ),
                "right_geomean_seconds": float(
                    np.exp(np.log(right["elapsed_seconds"]).mean())
                ),
                "right_to_left_speedup": float(np.exp(log_ratio.mean())),
                "speedup_ci95_low": float(np.exp(interval[0])),
                "speedup_ci95_high": float(np.exp(interval[1])),
                "paired_block_t": float(test.statistic),
                "raw_p": p_value,
            }
        )
    adjusted = holm_adjust(np.asarray(raw_p, dtype=float))
    result = pd.DataFrame(rows)
    result["holm_p"] = adjusted
    return result


def _equivalence_row(
    metric: str,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, str | int | float | bool] | None:
    """Calculate a Welch test and ±0.5 pooled-SD TOST result.

    Args:
        metric: Outcome label.
        left: NATAL replicate outcomes.
        right: MGDrivE1 replicate outcomes.

    Returns:
        Statistical row, or ``None`` for exact zero-variance agreement.
    """
    left_variance = float(left.var(ddof=1))
    right_variance = float(right.var(ddof=1))
    pooled_sd = float(
        np.sqrt(
            (
                (left.size - 1) * left_variance
                + (right.size - 1) * right_variance
            )
            / (left.size + right.size - 2)
        )
    )
    if pooled_sd == 0.0:
        if float(left.mean()) != float(right.mean()):
            raise RuntimeError(f"{metric} has unequal constant outcomes")
        return None
    difference = float(left.mean() - right.mean())
    standard_error = float(
        np.sqrt(left_variance / left.size + right_variance / right.size)
    )
    degrees_freedom = standard_error**4 / (
        (left_variance / left.size) ** 2 / (left.size - 1)
        + (right_variance / right.size) ** 2 / (right.size - 1)
    )
    ci_low, ci_high = stats.t.interval(
        0.90,
        degrees_freedom,
        loc=difference,
        scale=standard_error,
    )
    margin = 0.5 * pooled_sd
    lower_p = float(
        stdtr(degrees_freedom, -(difference + margin) / standard_error)
    )
    upper_p = float(
        stdtr(degrees_freedom, (difference - margin) / standard_error)
    )
    tost_p = max(lower_p, upper_p)
    welch = stats.ttest_ind(left, right, equal_var=False)
    return {
        "metric": metric,
        "natal_mean": float(left.mean()),
        "mgdrive_mean": float(right.mean()),
        "standardized_difference": difference / pooled_sd,
        "ci90_low": float(ci_low / pooled_sd),
        "ci90_high": float(ci_high / pooled_sd),
        "welch_p": float(welch.pvalue),
        "holm_p": 0.0,
        "tost_p": tost_p,
        "equivalent": tost_p < 0.05,
    }


def _spatial_consistency(source: Path) -> pd.DataFrame:
    """Calculate stochastic spatial outcome equivalence tests.

    Args:
        source: Directory containing aggregate formal CSV files.

    Returns:
        Informative per-outcome statistics with Holm correction.
    """
    natal = pd.read_csv(source / "natal-spatial-stochastic.csv")
    mgdrive = pd.read_csv(source / "mgdrive-spatial-stochastic.csv")
    metrics = [
        (
            "adult_total",
            natal[list(ADULT_COLUMNS)].sum(axis=1).to_numpy(dtype=float),
            mgdrive[list(ADULT_COLUMNS)].sum(axis=1).to_numpy(dtype=float),
        )
    ]
    for column in (*ADULT_COLUMNS, *SPATIAL_MOMENT_COLUMNS):
        metrics.append(
            (
                column,
                natal[column].to_numpy(dtype=float),
                mgdrive[column].to_numpy(dtype=float),
            )
        )
    rows = [
        result
        for metric, left, right in metrics
        if (result := _equivalence_row(metric, left, right)) is not None
    ]
    result = pd.DataFrame(rows)
    result["holm_p"] = holm_adjust(
        result["welch_p"].to_numpy(dtype=float)
    )
    return result


def _deterministic_validation(source: Path) -> pd.DataFrame:
    """Calculate exact deterministic agreement metrics.

    Args:
        source: Directory containing aggregate formal CSV files.

    Returns:
        Adult-state and spatial-moment maximum absolute errors.
    """
    natal = pd.read_csv(source / "natal-spatial-deterministic.csv")
    mgdrive = pd.read_csv(source / "mgdrive-spatial-deterministic.csv")
    return pd.DataFrame(
        {
            "scenario": ["Spatial 15 x 15 (30 d)"] * 2,
            "quantity": ["adult states", "spatial moments"],
            "max_abs_error": [
                float(
                    np.max(
                        np.abs(
                            natal[list(ADULT_COLUMNS)].to_numpy(dtype=float)
                            - mgdrive[list(ADULT_COLUMNS)].to_numpy(dtype=float)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            natal[list(SPATIAL_MOMENT_COLUMNS)].to_numpy(
                                dtype=float
                            )
                            - mgdrive[list(SPATIAL_MOMENT_COLUMNS)].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            ],
            "equivalence_limit": [1e-8, 1e-8],
        }
    )


def main() -> None:
    """Freeze formal tables and write all precomputed statistical tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    args = parser.parse_args()
    spatial = args.frozen_root / "spatial_30d"
    panmictic = args.frozen_root / "slim_30d"
    spatial.mkdir(parents=True, exist_ok=True)
    panmictic.mkdir(parents=True, exist_ok=True)
    copy_map = {
        "natal-spatial-deterministic.csv": spatial / "natal-deterministic.csv",
        "mgdrive-spatial-deterministic.csv": spatial / "mgdrive-deterministic.csv",
        "natal-spatial-stochastic.csv": spatial / "natal-stochastic.csv",
        "mgdrive-spatial-stochastic.csv": spatial / "mgdrive-stochastic.csv",
        "natal-panmmictic.csv": panmictic / "natal-panmmictic.csv",
        "mgdrive-panmmictic.csv": panmictic / "mgdrive-panmmictic.csv",
        "slim-panmmictic.csv": panmictic / "slim-panmmictic.csv",
    }
    for source_name, destination in copy_map.items():
        shutil.copyfile(args.source / source_name, destination)

    natal = _read_panmictic(args.source / "natal-panmmictic.csv")
    slim = _read_panmictic(args.source / "slim-panmmictic.csv")
    panmictic_results = compare_stochastic_records(natal, slim)
    write_consistency(panmictic / "consistency.csv", panmictic_results)
    _performance_tests(args.source).to_csv(
        args.frozen_root / "performance_tests.csv",
        index=False,
    )
    _spatial_consistency(args.source).to_csv(
        spatial / "consistency.csv",
        index=False,
    )
    _deterministic_validation(args.source).to_csv(
        args.frozen_root / "deterministic_validation.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
