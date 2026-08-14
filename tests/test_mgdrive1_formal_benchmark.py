"""Numerical contracts for formal MGDrivE1 benchmark utilities."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchmarks.mgdrive1.analyze_formal_benchmark as analysis
import benchmarks.mgdrive1.generate_trajectory_validation as trajectory
import benchmarks.mgdrive1.run_formal_benchmark as formal
from benchmarks.mgdrive1.lifecycle import DeterministicConfig
from benchmarks.mgdrive1.spatial_benchmark import (
    HexBenchmarkScenario,
    SpatialPatchState,
)


def _panmictic_rows(engine: str, elapsed: tuple[float, ...]) -> pd.DataFrame:
    """Build conserved explicit-population records for CSV tests.

    Args:
        engine: Engine label stored in every record.
        elapsed: Positive runtimes, one per block.

    Returns:
        Valid panmictic aggregate table.
    """
    rows: list[dict[str, str | int | float]] = []
    for index, seconds in enumerate(elapsed, start=1):
        adult_values = {
            column: index + offset
            for offset, column in enumerate(analysis.ADULT_COLUMNS)
        }
        adult_total = sum(adult_values.values())
        rows.append(
            {
                "engine": engine,
                "mode": "stochastic",
                "repeat": index,
                "day": 30,
                "block": index,
                "elapsed_seconds": seconds,
                "population_size": 100 + adult_total,
                "aquatic_total": 100,
                "unmated_female_total": 0,
                **adult_values,
            }
        )
    return pd.DataFrame(rows)


def _spatial_rows(
    engine: str,
    elapsed: tuple[float, ...],
    *,
    offset: float,
) -> pd.DataFrame:
    """Build spatial aggregate records with nonzero outcome variance.

    Args:
        engine: Engine label.
        elapsed: Positive runtimes, one per block.
        offset: Engine-specific numerical displacement.

    Returns:
        Spatial aggregate table.
    """
    rows: list[dict[str, str | int | float]] = []
    for index, seconds in enumerate(elapsed, start=1):
        outcomes = {
            column: float(index + column_index) + offset
            for column_index, column in enumerate(analysis.ADULT_COLUMNS)
        }
        rows.append(
            {
                "engine": engine,
                "repeat": index,
                "block": index,
                "elapsed_seconds": seconds,
                **outcomes,
                "spatial_male_aa_radius2": 10.0 * index + offset,
                "spatial_female_aa_radius2": 20.0 * index + offset,
            }
        )
    return pd.DataFrame(rows)


def _write_analysis_source(source: Path) -> None:
    """Write all seven formal aggregate inputs.

    Args:
        source: Destination directory.
    """
    source.mkdir(parents=True, exist_ok=True)
    _panmictic_rows("NATAL Core", (1.0, 1.8, 3.2, 5.5)).to_csv(
        source / "natal-panmmictic.csv", index=False
    )
    _panmictic_rows("MGDrivE1", (1.7, 3.5, 5.1, 9.8)).to_csv(
        source / "mgdrive-panmmictic.csv", index=False
    )
    _panmictic_rows("SLiM", (2.5, 4.1, 8.3, 11.2)).to_csv(
        source / "slim-panmmictic.csv", index=False
    )
    for mode in ("deterministic", "stochastic"):
        _spatial_rows(
            "NATAL Core",
            (1.2, 2.1, 3.8, 6.4),
            offset=0.0,
        ).to_csv(source / f"natal-spatial-{mode}.csv", index=False)
        _spatial_rows(
            "MGDrivE1",
            (2.0, 3.9, 7.2, 9.1),
            offset=0.1 if mode == "stochastic" else 1e-9,
        ).to_csv(source / f"mgdrive-spatial-{mode}.csv", index=False)


def test_analysis_main_freezes_complete_numerical_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze inputs and calculate every planned numerical comparison.

    Args:
        tmp_path: Temporary source and frozen directories.
        monkeypatch: Pytest fixture replacing command-line arguments.
    """
    source = tmp_path / "source"
    frozen = tmp_path / "frozen"
    _write_analysis_source(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_formal_benchmark",
            "--source",
            str(source),
            "--frozen-root",
            str(frozen),
        ],
    )

    analysis.main()

    assert len(pd.read_csv(frozen / "performance_tests.csv")) == 5
    deterministic = pd.read_csv(frozen / "deterministic_validation.csv")
    np.testing.assert_allclose(
        deterministic["max_abs_error"],
        [1e-9, 1e-9],
        rtol=0.0,
        atol=1e-14,
    )
    spatial = pd.read_csv(frozen / "spatial_30d" / "consistency.csv")
    assert len(spatial) == 15
    assert np.all((spatial["holm_p"] >= 0.0) & (spatial["holm_p"] <= 1.0))
    assert len(pd.read_csv(frozen / "slim_30d" / "natal-panmmictic.csv")) == 4
    records = analysis._read_panmictic(source / "natal-panmmictic.csv")
    assert len(records) == 4
    assert not records[0].adult_male.flags.writeable
    np.testing.assert_array_equal(records[0].adult_male, [1.0, 2.0, 3.0])


def test_analysis_rejects_mismatched_blocks_and_constant_disagreement(
    tmp_path: Path,
) -> None:
    """Reject invalid pairing and zero-variance numerical disagreement.

    Args:
        tmp_path: Temporary malformed aggregate directory.
    """
    _write_analysis_source(tmp_path)
    malformed = pd.read_csv(tmp_path / "mgdrive-spatial-deterministic.csv")
    malformed.loc[0, "block"] = 99
    malformed.to_csv(tmp_path / "mgdrive-spatial-deterministic.csv", index=False)

    with pytest.raises(RuntimeError, match="block identifiers"):
        analysis._performance_tests(tmp_path)
    with pytest.raises(RuntimeError, match="unequal constant"):
        analysis._equivalence_row(
            "adult_total",
            np.ones(4),
            np.full(4, 2.0),
        )
    assert analysis._equivalence_row(
        "adult_total",
        np.ones(4),
        np.ones(4),
    ) is None


def _trajectory_scenario() -> HexBenchmarkScenario:
    """Build a three-genotype state with exact adult totals.

    Returns:
        Minimal scenario used by trajectory tests.
    """
    config = DeterministicConfig.neutral(
        n_genotypes=3,
        time_egg=1,
        time_larva=1,
        time_pupa=1,
        beta=0.0,
        adult_mortality=0.0,
        aquatic_mortality=0.0,
        alpha=1.0,
    )
    state = SpatialPatchState(
        aquatic=np.zeros((1, 3, 3), dtype=np.float64),
        adult_male=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        adult_female=np.zeros((1, 3, 3), dtype=np.float64),
        unmated_female=np.array([[0.0, 0.0, 4.0]], dtype=np.float64),
    )
    return HexBenchmarkScenario(
        config=config,
        state=state,
        migration_kernel=np.ones((1, 1), dtype=np.float64),
        rows=1,
        cols=1,
        migration_rate=0.0,
    )


def test_natal_trajectory_seed_axes_release_order_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover seed-axis contracts and the day-25 release transition.

    Args:
        monkeypatch: Pytest fixture replacing scenario and lifecycle boundaries.
    """
    scenario = _trajectory_scenario()
    seeds: list[int] = []
    releases: list[object] = []
    monkeypatch.setattr(
        trajectory,
        "build_hex_benchmark_scenario",
        lambda **_kwargs: scenario,
    )
    monkeypatch.setattr(trajectory, "set_numba_seed", seeds.append)

    def preserve_state(*args: object, **kwargs: object) -> SpatialPatchState:
        """Record the scheduled release and preserve exact state values."""
        releases.append(kwargs["release"])
        return scenario.state

    monkeypatch.setattr(trajectory, "step_spatial", preserve_state)

    table = trajectory._natal_trajectories(
        scenario_name="panmictic",
        rows=1,
        cols=1,
        stochastic=True,
        repeats=2,
        n_days=24,
        seed=100,
    )

    assert len(table) == 50
    assert seeds == [100, 101]
    np.testing.assert_allclose(table["adult_total"], 10.0)
    np.testing.assert_allclose(table["aa_adult"], 7.0)
    assert sum(release is not None for release in releases) == 2
    assert releases[23] is not None
    assert releases[47] is not None
    total, aa_total = trajectory._summarize_state(scenario.state)
    assert total == 10.0
    assert aa_total == 7.0

    with pytest.raises(ValueError, match="require a seed"):
        trajectory._natal_trajectories(
            scenario_name="panmictic",
            rows=1,
            cols=1,
            stochastic=True,
            repeats=1,
            n_days=0,
            seed=None,
        )
    with pytest.raises(ValueError, match="do not accept"):
        trajectory._natal_trajectories(
            scenario_name="spatial",
            rows=2,
            cols=2,
            stochastic=True,
            repeats=1,
            n_days=0,
            seed=1,
        )


def test_slim_trajectories_validate_process_and_trace_contracts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Parse exact SLiM traces and reject every external failure shape.

    Args:
        monkeypatch: Pytest fixture replacing executable and subprocess calls.
        capsys: Pytest fixture capturing progress output.
    """
    monkeypatch.setattr(trajectory.shutil, "which", lambda _name: "/bin/slim")
    commands: list[tuple[str, ...]] = []

    def successful_run(command: tuple[str, ...], **_kwargs: object) -> object:
        """Return one exact day-zero trace and capture the command."""
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="TRACE:0,10,3\n", stderr="")

    monkeypatch.setattr(trajectory.subprocess, "run", successful_run)
    table = trajectory._slim_trajectories(repeats=100, n_days=0, seed=200)

    assert len(table) == 100
    assert commands[0][2] == "200"
    assert commands[-1][2] == "299"
    np.testing.assert_array_equal(table["adult_total"], np.full(100, 10))
    assert "100/100" in capsys.readouterr().out

    monkeypatch.setattr(trajectory.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="not found"):
        trajectory._slim_trajectories(repeats=1, n_days=0, seed=1)

    monkeypatch.setattr(trajectory.shutil, "which", lambda _name: "/bin/slim")
    monkeypatch.setattr(
        trajectory.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="failed",
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        trajectory._slim_trajectories(repeats=1, n_days=0, seed=1)

    monkeypatch.setattr(
        trajectory.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="number of days"):
        trajectory._slim_trajectories(repeats=1, n_days=0, seed=1)

    monkeypatch.setattr(
        trajectory.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="TRACE:0,10\n",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="three values"):
        trajectory._slim_trajectories(repeats=1, n_days=0, seed=1)


def test_trajectory_writers_runner_cache_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write owned tables, preserve runner axes, and summarize exact moments.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest fixture replacing generators and subprocess execution.
    """
    raw = pd.DataFrame(
        {
            "engine": ["NATAL Core", "NATAL Core"],
            "scenario": ["panmictic", "panmictic"],
            "mode": ["stochastic", "stochastic"],
            "repeat": [1, 2],
            "transition": [0, 0],
            "adult_total": [10.0, 14.0],
            "aa_adult": [2.0, 4.0],
        }
    )
    monkeypatch.setattr(trajectory, "_natal_trajectories", lambda **_kwargs: raw)
    monkeypatch.setattr(trajectory, "_slim_trajectories", lambda **_kwargs: raw)
    natal_path = tmp_path / "natal.csv"
    slim_path = tmp_path / "slim.csv"
    trajectory._write_natal(
        natal_path,
        scenario_name="panmictic",
        rows=1,
        cols=1,
        stochastic=True,
        repeats=2,
        n_days=0,
        seed=7,
    )
    trajectory._write_slim(slim_path, repeats=2, n_days=0, seed=7)
    assert trajectory._valid(natal_path, expected_rows=2)
    assert not trajectory._valid(natal_path, expected_rows=3)
    assert not trajectory._valid(tmp_path / "missing.csv", expected_rows=2)
    np.testing.assert_allclose(pd.read_csv(natal_path)["adult_total"], [10.0, 14.0])
    summary = trajectory._summary(raw)
    assert summary.loc[0, "n"] == 2
    assert summary.loc[0, "adult_total_mean"] == 12.0
    assert summary.loc[0, "adult_total_sd"] == pytest.approx(np.sqrt(8.0))
    assert summary.loc[0, "aa_adult_mean"] == 3.0

    captured: dict[str, object] = {}

    def capture_run(command: tuple[str, ...], **kwargs: object) -> None:
        """Capture the R command and environment without external execution."""
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(trajectory.subprocess, "run", capture_run)
    destination = tmp_path / "mgdrive.csv"
    trajectory._run_mgdrive(
        destination,
        mode="stochastic",
        rows=2,
        cols=3,
        n_days=4,
        repeats=5,
        seed=6,
    )
    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[-7:] == ("stochastic", "2", "3", "4", "5", "6", str(destination))
    assert captured["check"] is True
    assert isinstance(captured["env"], dict)


def test_trajectory_main_resumes_and_rejects_incomplete_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume valid tables and reject a task that writes too few rows.

    Args:
        tmp_path: Temporary trajectory output directory.
        monkeypatch: Pytest fixture replacing external task writers.
    """
    output = tmp_path / "complete"
    output.mkdir()
    base_columns = {
        "engine": "test",
        "scenario": "test",
        "mode": "deterministic",
        "repeat": 1,
        "transition": 0,
        "adult_total": 10.0,
        "aa_adult": 2.0,
    }
    pd.DataFrame([base_columns]).to_csv(
        output / "natal-spatial-deterministic.csv",
        index=False,
    )

    def write_requested(destination: Path, **kwargs: object) -> None:
        """Write exactly the requested trajectory row count."""
        repeats = int(kwargs["repeats"])
        n_days = int(kwargs["n_days"])
        pd.DataFrame(
            [
                {**base_columns, "repeat": index + 1}
                for index in range(repeats * (n_days + 1))
            ]
        ).to_csv(destination, index=False)

    monkeypatch.setattr(trajectory, "_write_natal", write_requested)
    monkeypatch.setattr(trajectory, "_write_slim", write_requested)
    monkeypatch.setattr(trajectory, "_run_mgdrive", write_requested)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_trajectory_validation",
            "--days",
            "0",
            "--spatial-repeats",
            "2",
            "--panmictic-repeats",
            "3",
            "--output-dir",
            str(output),
        ],
    )

    trajectory.main()

    assert len(pd.read_csv(output / "trajectory_raw.csv")) == 12
    assert len(pd.read_csv(output / "trajectory_summary.csv")) == 1
    assert os.environ["MGDRIVE_R_LIB"] == "/private/tmp/natal-r-lib"

    broken = tmp_path / "broken"

    def write_nothing(_destination: Path, **_kwargs: object) -> None:
        """Leave the requested output absent."""

    monkeypatch.setattr(trajectory, "_write_natal", write_nothing)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_trajectory_validation",
            "--days",
            "0",
            "--spatial-repeats",
            "1",
            "--panmictic-repeats",
            "1",
            "--output-dir",
            str(broken),
        ],
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        trajectory.main()


def _write_repeat_csv(path: Path, repeats: tuple[int, ...]) -> None:
    """Write a minimal repeat-bearing task result.

    Args:
        path: CSV destination.
        repeats: Local repeat identifiers.
    """
    pd.DataFrame(
        {
            "engine": ["test"] * len(repeats),
            "repeat": repeats,
            "value": [10.0 * repeat for repeat in repeats],
        }
    ).to_csv(path, index=False)


def test_formal_csv_normalization_aggregation_and_errors(tmp_path: Path) -> None:
    """Normalize exact IDs, aggregate schemas, and preserve sources on errors.

    Args:
        tmp_path: Temporary block directory.
    """
    source = tmp_path / "source.csv"
    normalized = tmp_path / "normalized.csv"
    _write_repeat_csv(source, (1, 2))
    formal._normalize_block(
        source,
        normalized,
        block=3,
        repeat_offset=4,
        expected_rows=2,
    )
    rows = list(csv.DictReader(normalized.open(newline="", encoding="utf-8")))
    assert [int(row["block"]) for row in rows] == [3, 3]
    assert [int(row["repeat"]) for row in rows] == [5, 6]
    assert not source.exists()
    assert formal._completed(normalized, expected_rows=2)
    assert not formal._completed(normalized, expected_rows=1)
    assert not formal._completed(tmp_path / "missing.csv", expected_rows=1)

    second_source = tmp_path / "source2.csv"
    second = tmp_path / "normalized2.csv"
    _write_repeat_csv(second_source, (1, 2))
    formal._normalize_block(
        second_source,
        second,
        block=4,
        repeat_offset=6,
        expected_rows=2,
    )
    aggregate = tmp_path / "aggregate.csv"
    formal._aggregate((normalized, second), aggregate, expected_rows=4)
    aggregate_rows = list(csv.DictReader(aggregate.open(newline="", encoding="utf-8")))
    assert [int(row["repeat"]) for row in aggregate_rows] == [5, 6, 7, 8]

    missing_repeat = tmp_path / "missing-repeat.csv"
    pd.DataFrame({"value": [1.0]}).to_csv(missing_repeat, index=False)
    with pytest.raises(RuntimeError, match="repeat column"):
        formal._normalize_block(
            missing_repeat,
            tmp_path / "unused.csv",
            block=1,
            repeat_offset=0,
            expected_rows=1,
        )
    assert missing_repeat.exists()

    wrong_rows = tmp_path / "wrong-rows.csv"
    _write_repeat_csv(wrong_rows, (1,))
    with pytest.raises(RuntimeError, match="expected 2"):
        formal._normalize_block(
            wrong_rows,
            tmp_path / "unused2.csv",
            block=1,
            repeat_offset=0,
            expected_rows=2,
        )
    assert wrong_rows.exists()

    mismatch = tmp_path / "mismatch.csv"
    pd.DataFrame({"other": [1.0]}).to_csv(mismatch, index=False)
    with pytest.raises(RuntimeError, match="mismatched"):
        formal._aggregate((normalized, mismatch), tmp_path / "bad.csv", expected_rows=3)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="header"):
        formal._aggregate((empty,), tmp_path / "bad-empty.csv", expected_rows=0)
    with pytest.raises(RuntimeError, match="expected 5"):
        formal._aggregate((normalized,), tmp_path / "bad-count.csv", expected_rows=5)


def test_formal_task_matrix_preserves_engine_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke all seven tasks and retain each seed/stochastic contract.

    Args:
        tmp_path: Temporary task destinations.
        monkeypatch: Pytest fixture replacing all engine boundaries.
    """
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        formal,
        "benchmark_natal_panmmictic",
        lambda **kwargs: calls.append(("natal-pan", kwargs)) or (),
    )
    monkeypatch.setattr(
        formal,
        "benchmark_slim",
        lambda **kwargs: calls.append(("slim", kwargs)) or (),
    )
    monkeypatch.setattr(
        formal,
        "benchmark_natal",
        lambda **kwargs: calls.append(("natal-spatial", kwargs)) or (),
    )
    monkeypatch.setattr(
        formal,
        "write_records",
        lambda path, _records: path.write_text("repeat\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        formal,
        "write_natal_records",
        lambda path, _records, **kwargs: (
            calls.append(("write-spatial", kwargs)),
            path.write_text("repeat\n", encoding="utf-8"),
        ),
    )
    monkeypatch.setattr(
        formal,
        "run_mgdrive",
        lambda **kwargs: calls.append(("mgdrive", kwargs)),
    )
    tasks = formal._build_tasks(
        rows=2,
        cols=3,
        n_days=4,
        panmictic_per_block=5,
        spatial_per_block=6,
    )

    for index, (name, repeats, task) in enumerate(tasks):
        task(tmp_path / f"{name}.csv", repeats, 100 + index)

    assert len(tasks) == 7
    assert sum(name == "mgdrive" for name, _ in calls) == 3
    natal_pan = next(payload for name, payload in calls if name == "natal-pan")
    assert natal_pan == {"repeats": 5, "n_days": 4, "seed": 100}
    slim = next(payload for name, payload in calls if name == "slim")
    assert slim == {"repeats": 5, "n_days": 4, "seed": 102}
    spatial_calls = [payload for name, payload in calls if name == "natal-spatial"]
    assert [payload["stochastic"] for payload in spatial_calls] == [False, True]
    assert all("seed" not in payload for payload in spatial_calls)


def test_formal_main_runs_rotated_blocks_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run rotated blocks, aggregate exact IDs, then skip complete work.

    Args:
        tmp_path: Temporary formal output directory.
        monkeypatch: Pytest fixture replacing the task matrix and CLI.
    """
    library = tmp_path / "r-lib"
    library.mkdir()
    output = tmp_path / "formal"
    calls: list[str] = []

    def task_one(path: Path, repeats: int, seed: int) -> None:
        """Write one complete local-repeat block."""
        calls.append(f"one:{seed}")
        _write_repeat_csv(path, tuple(range(1, repeats + 1)))

    def task_two(path: Path, repeats: int, seed: int) -> None:
        """Write the second complete local-repeat block."""
        calls.append(f"two:{seed}")
        _write_repeat_csv(path, tuple(range(1, repeats + 1)))

    monkeypatch.setattr(
        formal,
        "_build_tasks",
        lambda **_kwargs: (("one", 2, task_one), ("two", 2, task_two)),
    )
    argv = [
        "run_formal_benchmark",
        "--panmictic-repeats",
        "4",
        "--spatial-repeats",
        "4",
        "--blocks",
        "2",
        "--seed",
        "10",
        "--output-dir",
        str(output),
        "--mgdrive-r-lib",
        str(library),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    formal.main()
    formal.main()

    assert calls == ["one:10", "two:10", "two:12", "one:12"]
    one = pd.read_csv(output / "one.csv")
    assert one["repeat"].tolist() == [1, 2, 3, 4]
    assert one["block"].tolist() == [1, 1, 2, 2]
    assert os.environ["MGDRIVE_R_LIB"] == str(library)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--blocks", "0"),
        ("--blocks", "3", "--panmictic-repeats", "4"),
        (
            "--blocks",
            "3",
            "--panmictic-repeats",
            "6",
            "--spatial-repeats",
            "4",
        ),
    ],
)
def test_formal_main_rejects_invalid_block_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    """Reject invalid block axes before creating benchmark output.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest fixture replacing command-line arguments.
        arguments: Invalid CLI axis override.
    """
    output = tmp_path / "formal"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_formal_benchmark", "--output-dir", str(output), *arguments],
    )

    with pytest.raises(ValueError):
        formal.main()

    assert not output.exists()


def test_formal_main_requires_existing_r_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing R library before output mutation.

    Args:
        tmp_path: Temporary paths.
        monkeypatch: Pytest fixture replacing command-line arguments.
    """
    output = tmp_path / "formal"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_formal_benchmark",
            "--output-dir",
            str(output),
            "--mgdrive-r-lib",
            str(tmp_path / "missing"),
        ],
    )

    with pytest.raises(FileNotFoundError):
        formal.main()

    assert not output.exists()
