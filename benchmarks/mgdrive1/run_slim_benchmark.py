"""Run and statistically compare NATAL and SLiM panmictic replicates."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from scipy import stats
from scipy.special import stdtr

from .slim_panmmictic import (
    ConsistencyResult,
    PanmicticRecord,
    benchmark_natal_panmmictic,
    benchmark_slim,
)

Metric = tuple[str, Callable[[PanmicticRecord], float]]


def _metrics() -> tuple[Metric, ...]:
    """Return informative panmictic outcomes used for statistical comparison."""
    return (
        ("population_size", lambda record: float(record.population_size)),
        ("aquatic_total", lambda record: float(record.aquatic_total)),
        (
            "adult_total",
            lambda record: float(
                record.adult_male.sum()
                + record.adult_female.sum()
                + record.unmated_female_total
            ),
        ),
        ("male_AA", lambda record: float(record.adult_male[0])),
        ("male_aa", lambda record: float(record.adult_male[2])),
        (
            "female_AA_mate_AA",
            lambda record: float(record.adult_female[0, 0]),
        ),
        (
            "female_AA_mate_aa",
            lambda record: float(record.adult_female[0, 2]),
        ),
        (
            "female_aa_mate_AA",
            lambda record: float(record.adult_female[2, 0]),
        ),
        (
            "female_aa_mate_aa",
            lambda record: float(record.adult_female[2, 2]),
        ),
    )


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Apply Holm's family-wise multiple-comparison correction.

    Args:
        p_values: One-dimensional unadjusted p-values.

    Returns:
        Adjusted p-values in the input order.
    """
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, p_values[index] * (p_values.size - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def compare_stochastic_records(
    natal_records: Sequence[PanmicticRecord],
    slim_records: Sequence[PanmicticRecord],
) -> tuple[ConsistencyResult, ...]:
    """Compare final distributions with Welch, Holm, and TOST procedures.

    TOST uses a practical equivalence margin of ±0.5 pooled standard deviations.
    Metrics with zero pooled variance are omitted because their standardized
    effect and equivalence margin are undefined.

    Args:
        natal_records: Independent NATAL replicates.
        slim_records: Independent SLiM replicates.

    Returns:
        Per-metric results with Holm-adjusted p-values.

    Raises:
        ValueError: If either engine has fewer than two replicates.
        RuntimeError: If a statistical result is non-finite.
    """
    if len(natal_records) < 2 or len(slim_records) < 2:
        raise ValueError("statistical comparison requires two or more replicates")
    preliminary: list[ConsistencyResult] = []
    for metric_name, extractor in _metrics():
        natal = np.array(
            [extractor(record) for record in natal_records],
            dtype=np.float64,
        )
        slim = np.array(
            [extractor(record) for record in slim_records],
            dtype=np.float64,
        )
        natal_variance = float(natal.var(ddof=1))
        slim_variance = float(slim.var(ddof=1))
        pooled_sd = np.sqrt(
            (
                (natal.size - 1) * natal_variance
                + (slim.size - 1) * slim_variance
            )
            / (natal.size + slim.size - 2)
        )
        if pooled_sd == 0.0:
            if float(natal.mean()) != float(slim.mean()):
                raise RuntimeError(
                    f"{metric_name} has unequal constant outcomes"
                )
            continue
        difference = float(natal.mean() - slim.mean())
        standard_error = np.sqrt(
            natal_variance / natal.size + slim_variance / slim.size
        )
        degrees_freedom = standard_error**4 / (
            (natal_variance / natal.size) ** 2 / (natal.size - 1)
            + (slim_variance / slim.size) ** 2 / (slim.size - 1)
        )
        welch = stats.ttest_ind(natal, slim, equal_var=False)
        ci_low, ci_high = stats.t.interval(
            0.90,
            degrees_freedom,
            loc=difference,
            scale=standard_error,
        )
        margin = 0.5 * pooled_sd
        lower_p = float(
            cast(
                np.float64,
                stdtr(
                    degrees_freedom,
                    -(difference + margin) / standard_error,
                ),
            )
        )
        upper_p = float(
            cast(
                np.float64,
                stdtr(
                    degrees_freedom,
                    (difference - margin) / standard_error,
                ),
            )
        )
        tost_p = float(max(lower_p, upper_p))
        values = np.array(
            [
                difference / pooled_sd,
                ci_low / pooled_sd,
                ci_high / pooled_sd,
                float(welch.pvalue),
                tost_p,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{metric_name} produced non-finite statistics")
        preliminary.append(
            ConsistencyResult(
                metric=metric_name,
                natal_mean=float(natal.mean()),
                slim_mean=float(slim.mean()),
                standardized_difference=float(values[0]),
                ci90_low=float(values[1]),
                ci90_high=float(values[2]),
                welch_p=float(values[3]),
                holm_p=0.0,
                tost_p=tost_p,
                equivalent=tost_p < 0.05,
            )
        )

    if not preliminary:
        raise RuntimeError("comparison has no informative stochastic metrics")
    adjusted = _holm_adjust(
        np.array([result.welch_p for result in preliminary], dtype=np.float64)
    )
    return tuple(
        replace(result, holm_p=float(adjusted[index]))
        for index, result in enumerate(preliminary)
    )


def validate_consistency(results: Sequence[ConsistencyResult]) -> None:
    """Reject a family-wise significant cross-engine distribution difference.

    Args:
        results: Corrected per-metric comparisons.

    Raises:
        RuntimeError: If results are empty, contain non-finite or internally
            inconsistent statistics, show a Holm-adjusted difference, or fail
            the TOST equivalence margin.
    """
    if not results:
        raise RuntimeError("consistency validation requires informative metrics")
    for result in results:
        values = (
            result.natal_mean,
            result.slim_mean,
            result.standardized_difference,
            result.ci90_low,
            result.ci90_high,
            result.welch_p,
            result.holm_p,
            result.tost_p,
        )
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError(f"{result.metric} contains non-finite statistics")
        if not (
            0.0 <= result.welch_p <= 1.0
            and 0.0 <= result.holm_p <= 1.0
            and 0.0 <= result.tost_p <= 1.0
        ):
            raise RuntimeError(f"{result.metric} contains an invalid p-value")
        if result.holm_p < result.welch_p:
            raise RuntimeError(
                f"{result.metric} has an inconsistent Holm correction"
            )
        if not (
            result.ci90_low
            <= result.standardized_difference
            <= result.ci90_high
        ):
            raise RuntimeError(
                f"{result.metric} has an inconsistent confidence interval"
            )
        if result.holm_p < 0.05:
            raise RuntimeError(
                f"{result.metric} differs after Holm correction"
            )
        expected_equivalence = result.tost_p < 0.05
        interval_equivalence = (
            result.ci90_low > -0.5 and result.ci90_high < 0.5
        )
        if expected_equivalence != interval_equivalence:
            raise RuntimeError(
                f"{result.metric} has inconsistent TOST and confidence interval"
            )
        if result.equivalent != expected_equivalence:
            raise RuntimeError(
                f"{result.metric} has an inconsistent equivalence flag"
            )
        if not expected_equivalence:
            raise RuntimeError(
                f"{result.metric} does not satisfy the TOST equivalence margin"
            )


def _write_records(path: Path, records: Sequence[PanmicticRecord]) -> None:
    """Write raw engine observations to CSV.

    Args:
        path: Destination CSV.
        records: Records from one engine.
    """
    fields = (
        "engine",
        "mode",
        "repeat",
        "day",
        "elapsed_seconds",
        "population_size",
        "aquatic_total",
        "unmated_female_total",
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
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        for record in records:
            writer.writerow(
                (
                    record.engine,
                    record.mode,
                    record.repeat,
                    record.day,
                    record.elapsed_seconds,
                    record.population_size,
                    record.aquatic_total,
                    record.unmated_female_total,
                    *record.adult_male,
                    *record.adult_female.ravel(),
                )
            )


def _write_consistency(
    path: Path,
    results: Sequence[ConsistencyResult],
) -> None:
    """Write per-metric statistical comparisons to CSV.

    Args:
        path: Destination CSV.
        results: Corrected stochastic comparisons.
    """
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(ConsistencyResult.__dataclass_fields__)
        for result in results:
            writer.writerow(
                (
                    result.metric,
                    result.natal_mean,
                    result.slim_mean,
                    result.standardized_difference,
                    result.ci90_low,
                    result.ci90_high,
                    result.welch_p,
                    result.holm_p,
                    result.tost_p,
                    result.equivalent,
                )
            )


# Public names let benchmark orchestration reuse the validated CSV/statistics
# mechanisms without reaching through another module's private interface.
holm_adjust = _holm_adjust
write_records = _write_records
write_consistency = _write_consistency


def main() -> None:
    """Run both panmictic engines and write raw and statistical results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results-slim"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natal_records = benchmark_natal_panmmictic(
        repeats=args.repeats,
        n_days=args.days,
        seed=args.seed,
    )
    slim_records = benchmark_slim(
        repeats=args.repeats,
        n_days=args.days,
        seed=args.seed,
    )
    results = compare_stochastic_records(natal_records, slim_records)
    validate_consistency(results)
    _write_records(args.output_dir / "natal-panmmictic.csv", natal_records)
    _write_records(args.output_dir / "slim-panmmictic.csv", slim_records)
    _write_consistency(args.output_dir / "consistency.csv", results)

    natal_times = np.array(
        [record.elapsed_seconds for record in natal_records],
        dtype=np.float64,
    )
    slim_times = np.array(
        [record.elapsed_seconds for record in slim_records],
        dtype=np.float64,
    )
    performance = stats.ttest_ind(
        np.log(natal_times),
        np.log(slim_times),
        equal_var=False,
    )
    summary = {
        "days": args.days,
        "repeats": args.repeats,
        "natal_mean_seconds": float(natal_times.mean()),
        "slim_mean_seconds": float(slim_times.mean()),
        "slim_to_natal_lifecycle_ratio": float(
            slim_times.mean() / natal_times.mean()
        ),
        "timing_scope": "both engines exclude initialization and compilation",
        "log_runtime_welch_p": float(performance.pvalue),
        "minimum_holm_p": min(result.holm_p for result in results),
        "equivalent_metrics": sum(result.equivalent for result in results),
        "tested_metrics": len(results),
        "maximum_abs_standardized_difference": max(
            abs(result.standardized_difference) for result in results
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
