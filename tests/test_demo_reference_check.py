"""The Drive-RIDL reference comparator must be exact about what it compares.

The demo freezes its own full-grid run as the repository reference (the original
SLiM data is unavailable).  These tests pin the comparator's contract on small
synthetic grids, where the cost is negligible: NaN travels as JSON ``null``,
counts must match exactly, the suppressed-cell pattern must match, and mean
weeks are compared with the stored tolerance — while a run with a different
seed is skipped rather than failed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

_DEMO = Path(__file__).resolve().parent.parent / "demos" / "drive_ridl_remake_batch.py"


def _load_demo():
    """Load the demo module by path (demos/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location("drive_ridl_remake_batch", _DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    """The loaded demo module (imports matplotlib once)."""
    return _load_demo()


def _grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two 2x2 grids: counts plus mean weeks with one unsuppressed (NaN) cell."""
    mean = np.array([[10.5, np.nan], [20.25, 30.0]])
    counts = np.array([[2, 0], [1, 1]], dtype=np.int64)
    fitness_mean = np.array([[np.nan, 7.0], [8.0, 9.5]])
    fitness_counts = np.array([[0, 1], [1, 2]], dtype=np.int64)
    return mean, counts, fitness_mean, fitness_counts


def test_reference_round_trip_and_value_comparison(demo, tmp_path: Path) -> None:
    """A frozen reference round-trips and compares equal to its own run."""
    mean, counts, fitness_mean, fitness_counts = _grids()
    path = tmp_path / "ref.json"
    demo.write_reference(
        path,
        seed=0,
        repeats=20,
        mean_suppression_weeks=mean,
        success_counts=counts,
        fitness_mean_suppression_weeks=fitness_mean,
        fitness_success_counts=fitness_counts,
    )
    compared = demo.check_reference(
        path,
        seed=0,
        repeats=20,
        mean_suppression_weeks=mean.copy(),
        success_counts=counts.copy(),
        fitness_mean_suppression_weeks=fitness_mean.copy(),
        fitness_success_counts=fitness_counts.copy(),
    )
    assert compared is True


@pytest.mark.parametrize("kind", ["count", "mean", "nan_pattern"])
def test_reference_rejects_value_drift(demo, tmp_path: Path, kind: str) -> None:
    """Counts, mean values and the suppressed pattern are all checked."""
    mean, counts, fitness_mean, fitness_counts = _grids()
    path = tmp_path / "ref.json"
    demo.write_reference(
        path,
        seed=0,
        repeats=20,
        mean_suppression_weeks=mean,
        success_counts=counts,
        fitness_mean_suppression_weeks=fitness_mean,
        fitness_success_counts=fitness_counts,
    )
    mutated_counts = counts.copy()
    mutated_mean = mean.copy()
    if kind == "count":
        mutated_counts[0, 0] += 1
    elif kind == "mean":
        mutated_mean[0, 0] += 1.0
    else:
        mutated_mean[0, 1] = 3.0  # fill an unsuppressed cell with a value
    with pytest.raises(AssertionError):
        demo.check_reference(
            path,
            seed=0,
            repeats=20,
            mean_suppression_weeks=mutated_mean,
            success_counts=mutated_counts,
            fitness_mean_suppression_weeks=fitness_mean,
            fitness_success_counts=fitness_counts,
        )


def test_reference_is_skipped_for_a_different_run(demo, tmp_path: Path) -> None:
    """A different seed or replicate count cannot be compared, only skipped."""
    mean, counts, fitness_mean, fitness_counts = _grids()
    path = tmp_path / "ref.json"
    demo.write_reference(
        path,
        seed=0,
        repeats=20,
        mean_suppression_weeks=mean,
        success_counts=counts,
        fitness_mean_suppression_weeks=fitness_mean,
        fitness_success_counts=fitness_counts,
    )
    assert (
        demo.check_reference(
            path,
            seed=1,
            repeats=20,
            mean_suppression_weeks=mean,
            success_counts=counts,
            fitness_mean_suppression_weeks=fitness_mean,
            fitness_success_counts=fitness_counts,
        )
        is False
    )
    assert (
        demo.check_reference(
            path,
            seed=0,
            repeats=3,
            mean_suppression_weeks=mean,
            success_counts=counts,
            fitness_mean_suppression_weeks=fitness_mean,
            fitness_success_counts=fitness_counts,
        )
        is False
    )
