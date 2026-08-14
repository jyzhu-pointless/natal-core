"""Adversarial contracts for the panmictic SLiM comparison."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.mgdrive1 import run_slim_benchmark as runner
from benchmarks.mgdrive1 import slim_panmmictic as panmictic
from benchmarks.mgdrive1.run_slim_benchmark import (
    _holm_adjust,
    _write_consistency,
    _write_records,
    compare_stochastic_records,
    validate_consistency,
)
from benchmarks.mgdrive1.slim_panmmictic import (
    CUBE_GENOTYPES,
    CUBE_RELEASE_INDEX,
    ConsistencyResult,
    PanmicticRecord,
    benchmark_natal_panmmictic,
    benchmark_slim,
    parse_slim_output,
)

SLIM = shutil.which("slim")
VALID_OUT = (
    "SLiM 5.0\n"
    "OUT:30,100,94,0,6,1,2,3,0,0,0,0,0,0,0,0,0\n"
    "TIME:0.25\n"
)


def _valid_out(n_genotypes: int = 3) -> str:
    """Build a valid OUT record for any supported genotype-state count."""
    male = [1.0, 2.0, 3.0] + [0.0] * (n_genotypes - 3)
    female = np.zeros(n_genotypes * n_genotypes)
    fields = [30, 100, 94, 0, 6, *male, *female]
    return (
        "SLiM 5.0\n"
        "OUT:" + ",".join(str(value) for value in fields) + "\n"
        "TIME:0.25\n"
    )


def _record(
    *,
    repeat: int = 1,
    population_size: int | None = None,
    aquatic_total: int | None = None,
    unmated_female_total: int = 2,
    elapsed_seconds: float = 0.1,
    male: np.ndarray | None = None,
    female: np.ndarray | None = None,
    engine: str = "test",
) -> PanmicticRecord:
    """Build one valid synthetic benchmark record."""
    if male is None:
        male = np.array([10.0, 2.0, 1.0])
    if female is None:
        female = np.array(
            [[8.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
        )
    adult_total = int(round(float(male.sum() + female.sum())))
    if aquatic_total is None:
        aquatic_total = (
            70
            if population_size is None
            else population_size - unmated_female_total - adult_total
        )
    if population_size is None:
        population_size = aquatic_total + unmated_female_total + adult_total
    return PanmicticRecord(
        engine=engine,
        mode="stochastic",
        repeat=repeat,
        day=30,
        elapsed_seconds=elapsed_seconds,
        population_size=population_size,
        aquatic_total=aquatic_total,
        unmated_female_total=unmated_female_total,
        adult_male=male,
        adult_female=female,
    )


def _result(**changes: float | str | bool) -> ConsistencyResult:
    """Build one finite statistical result and apply requested replacements."""
    base = ConsistencyResult(
        metric="population_size",
        natal_mean=100.0,
        slim_mean=101.0,
        standardized_difference=-0.1,
        ci90_low=-0.3,
        ci90_high=0.1,
        welch_p=0.6,
        holm_p=0.8,
        tost_p=0.04,
        equivalent=True,
    )
    return replace(base, **changes)


def test_parse_slim_output_returns_owned_read_only_counts() -> None:
    """Parse the public record without leaking mutable array views."""
    result = parse_slim_output(VALID_OUT, repeat=2)

    assert result.engine == "SLiM-5.0-individual"
    assert result.mode == "stochastic"
    assert result.repeat == 2
    assert result.day == 30
    assert result.population_size == 100
    assert result.aquatic_total == 94
    assert result.unmated_female_total == 0
    np.testing.assert_array_equal(result.adult_male, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.adult_female, np.zeros((3, 3)))
    assert result.adult_male.flags.owndata
    assert result.adult_female.flags.owndata
    assert not result.adult_male.flags.writeable
    assert not result.adult_female.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.adult_male[0] = 99.0


def test_record_copies_caller_owned_arrays() -> None:
    """Keep a frozen record independent of later caller mutations."""
    male = np.array([1.0, 2.0, 3.0])
    female = np.arange(9.0).reshape(3, 3)
    record = _record(male=male, female=female)

    male[:] = 0.0
    female[:] = 0.0

    np.testing.assert_array_equal(record.adult_male, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        record.adult_female,
        np.arange(9.0).reshape(3, 3),
    )
    assert not np.shares_memory(record.adult_male, male)
    assert not np.shares_memory(record.adult_female, female)


@pytest.mark.parametrize(
    "changes",
    [
        {"adult_male": np.array([0.5, 0.0, 0.0])},
        {"adult_female": np.full((3, 3), 0.5)},
    ],
)
def test_record_rejects_fractional_individual_counts(
    changes: dict[str, np.ndarray],
) -> None:
    """Reject fractional values from an explicitly individual-based model."""
    arguments = {
        "engine": "test",
        "mode": "stochastic",
        "repeat": 1,
        "day": 0,
        "elapsed_seconds": 0.0,
        "population_size": 1,
        "aquatic_total": 0,
        "unmated_female_total": 0,
        "adult_male": np.zeros(3),
        "adult_female": np.zeros((3, 3)),
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match="integers"):
        PanmicticRecord(**arguments)


def test_record_rejects_population_total_disagreement() -> None:
    """Require population size to equal aquatic plus every adult category."""
    with pytest.raises(ValueError, match="total conservation"):
        _record(population_size=101, aquatic_total=70)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat", True),
        ("repeat", 1.5),
        ("day", False),
        ("day", 0.5),
        ("population_size", True),
        ("population_size", 100.5),
        ("aquatic_total", False),
        ("aquatic_total", 70.5),
        ("unmated_female_total", True),
        ("unmated_female_total", 2.5),
    ],
)
def test_record_rejects_bool_and_fractional_integer_scalars(
    field: str,
    value: bool | float,
) -> None:
    """Reject bool and fractional values on every explicit integer field."""
    arguments = {
        "engine": "test",
        "mode": "stochastic",
        "repeat": 1,
        "day": 30,
        "elapsed_seconds": 0.1,
        "population_size": 100,
        "aquatic_total": 70,
        "unmated_female_total": 2,
        "adult_male": np.array([10.0, 2.0, 1.0]),
        "adult_female": np.array(
            [[8.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
        ),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="must be integers"):
        PanmicticRecord(**arguments)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "deterministic"}, "stochastic"),
        ({"repeat": 0}, "repeat"),
        ({"day": -1}, "repeat"),
        ({"elapsed_seconds": -0.1}, "elapsed_seconds"),
        ({"elapsed_seconds": np.inf}, "elapsed_seconds"),
        ({"population_size": -1}, "population counts"),
        ({"aquatic_total": -1}, "population counts"),
        ({"unmated_female_total": -1}, "population counts"),
        ({"adult_male": np.zeros(2)}, "3 genotypes"),
        ({"adult_female": np.zeros((3, 2))}, "3 by 3"),
        ({"adult_male": np.array([0.0, np.nan, 0.0])}, "adult counts"),
        ({"adult_female": np.full((3, 3), -1.0)}, "adult counts"),
    ],
)
def test_record_rejects_each_invalid_scalar_and_axis(
    changes: dict[str, str | int | float | np.ndarray],
    message: str,
) -> None:
    """Reject every documented scalar, shape, and finite-count violation."""
    arguments = {
        "engine": "test",
        "mode": "stochastic",
        "repeat": 1,
        "day": 0,
        "elapsed_seconds": 0.0,
        "population_size": 1,
        "aquatic_total": 0,
        "unmated_female_total": 0,
        "adult_male": np.zeros(3),
        "adult_female": np.zeros((3, 3)),
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        PanmicticRecord(**arguments)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "exactly one"),
        (VALID_OUT + VALID_OUT, "exactly one"),
        ("OUT:30,1,2\nTIME:0.1\n", "17 fields"),
        (
            "OUT:30,100,94,0,6,one,2,3,0,0,0,0,0,0,0,0,0\nTIME:0.1\n",
            "numeric",
        ),
        (
            "OUT:30,100,94,0,6,nan,2,3,0,0,0,0,0,0,0,0,0\nTIME:0.1\n",
            "finite",
        ),
        (
            "OUT:30.5,100,94,0,6,1,2,3,0,0,0,0,0,0,0,0,0\nTIME:0.1\n",
            "integers",
        ),
    ],
)
def test_parse_slim_output_rejects_malformed_external_records(
    stdout: str,
    message: str,
) -> None:
    """Reject absent, duplicate, malformed, and non-finite public records."""
    with pytest.raises(RuntimeError, match=message):
        parse_slim_output(stdout, repeat=1)


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        ("", "exactly one TIME"),
        ("TIME:0.1\nTIME:0.2\n", "exactly one TIME"),
        ("TIME:not-a-number\n", "numeric"),
        ("TIME:nan\n", "finite"),
        ("TIME:-0.1\n", "finite"),
    ],
)
def test_parse_slim_output_rejects_invalid_timing_record(
    timing: str,
    message: str,
) -> None:
    """Reject missing, duplicate, malformed, and negative lifecycle timing."""
    stdout = (
        "OUT:30,100,94,0,6,1,2,3,0,0,0,0,0,0,0,0,0\n" + timing
    )

    with pytest.raises(RuntimeError, match=message):
        parse_slim_output(stdout, repeat=1)


def test_parse_slim_output_rejects_invalid_external_counts() -> None:
    """Translate invalid externally supplied counts into a parser error."""
    stdout = (
        "OUT:30,-1,94,0,6,1,2,3,0,0,0,0,0,0,0,0,0\n"
        "TIME:0.1\n"
    )

    with pytest.raises(RuntimeError, match="population"):
        parse_slim_output(stdout, repeat=1)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        (
            "OUT:30,101,94,0,6,1,2,3,0,0,0,0,0,0,0,0,0\nTIME:0.1\n",
            "population fields",
        ),
        (
            "OUT:30,101,94,0,7,1,2,3,0,0,0,0,0,0,0,0,0\nTIME:0.1\n",
            "adult fields",
        ),
    ],
)
def test_parse_slim_output_rejects_internally_inconsistent_totals(
    stdout: str,
    message: str,
) -> None:
    """Reject both population-level and adult-level conservation violations."""
    with pytest.raises(RuntimeError, match=message):
        parse_slim_output(stdout, repeat=1)


@pytest.mark.parametrize("n_genotypes", [3, 6, 9, 18])
def test_parse_slim_output_supports_scaled_genotype_state_space(
    n_genotypes: int,
) -> None:
    """Parse variable-length OUT records for every MGDrivE cube size."""
    male = [1.0, 2.0, 3.0] + [0.0] * (n_genotypes - 3)
    stdout = _valid_out(n_genotypes)

    result = parse_slim_output(stdout, repeat=1, n_genotypes=n_genotypes)

    assert result.n_genotypes == n_genotypes
    assert result.day == 30
    assert result.population_size == 100
    assert result.adult_male.shape == (n_genotypes,)
    assert result.adult_female.shape == (n_genotypes, n_genotypes)
    np.testing.assert_array_equal(result.adult_male, male)
    np.testing.assert_array_equal(
        result.adult_female,
        np.zeros((n_genotypes, n_genotypes)),
    )


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
def test_benchmark_slim_rejects_invalid_repeats_without_running(
    repeats: int | float,
) -> None:
    """Reject non-integral repetition counts before executable lookup."""
    with pytest.raises(ValueError, match="repeats"):
        benchmark_slim(repeats=repeats, n_days=1)


@pytest.mark.parametrize("n_days", [-1, 1.5, True])
def test_benchmark_slim_rejects_invalid_days_without_running(
    n_days: int | float,
) -> None:
    """Reject non-integral day counts before executable lookup."""
    with pytest.raises(ValueError, match="n_days"):
        benchmark_slim(repeats=1, n_days=n_days)


@pytest.mark.parametrize("population_scale", [0, -1, 1.5, True])
def test_benchmark_slim_rejects_invalid_population_scale_without_running(
    population_scale: int | float,
) -> None:
    """Reject nonpositive or non-integral equilibrium multipliers."""
    with pytest.raises(ValueError, match="population_scale"):
        benchmark_slim(
            repeats=1,
            n_days=0,
            population_scale=population_scale,
        )


def test_benchmark_slim_reports_missing_executable_and_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish a missing executable from a missing model file."""
    monkeypatch.setattr(panmictic.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="executable"):
        benchmark_slim(repeats=1, n_days=0)

    missing_executable = tmp_path / "missing-slim"
    with pytest.raises(FileNotFoundError, match="missing-slim"):
        benchmark_slim(
            repeats=1,
            n_days=0,
            executable=missing_executable,
        )

    executable = tmp_path / "slim"
    executable.touch()
    with pytest.raises(FileNotFoundError, match="model"):
        benchmark_slim(
            repeats=1,
            n_days=0,
            executable=executable,
            script=tmp_path / "missing.slim",
        )


def test_benchmark_slim_builds_seeded_subprocess_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use independent seeds and the requested day constant per replicate."""
    executable = tmp_path / "slim"
    script = tmp_path / "model.slim"
    executable.touch()
    script.touch()
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        capture_output: bool,
        check: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert not check
        assert text
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=VALID_OUT, stderr="")

    monkeypatch.setattr(panmictic.subprocess, "run", fake_run)

    records = benchmark_slim(
        repeats=2,
        n_days=30,
        seed=91,
        executable=executable,
        script=script,
    )

    assert [record.repeat for record in records] == [1, 2]
    assert all(record.elapsed_seconds >= 0.0 for record in records)
    assert commands == [
        (
            str(executable),
            "-s",
            "91",
            "-d",
            "N_DAYS=30",
            "-d",
            "POP_SCALE=1",
            "-d",
            "CUBE='mendelian3'",
            str(script),
        ),
        (
            str(executable),
            "-s",
            "92",
            "-d",
            "N_DAYS=30",
            "-d",
            "POP_SCALE=1",
            "-d",
            "CUBE='mendelian3'",
            str(script),
        ),
    ]


def test_benchmark_slim_passes_cube_constant_to_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the selected MGDrivE cube to the SLiM model."""
    executable = tmp_path / "slim"
    script = tmp_path / "model.slim"
    executable.touch()
    script.touch()
    captured: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        panmictic.subprocess,
        "run",
        lambda command, **kwargs: (
            captured.append(command)
            or subprocess.CompletedProcess(
                command,
                0,
                stdout=_valid_out(18),
                stderr="",
            )
        ),
    )

    benchmark_slim(
        repeats=1,
        n_days=30,
        seed=91,
        executable=executable,
        script=script,
        cube="twolocus18",
    )

    assert any(
        captured[0][index] == "-d"
        and captured[0][index + 1] == "CUBE='twolocus18'"
        for index in range(len(captured[0]) - 1)
    )


def test_benchmark_slim_rejects_unknown_cube_and_mismatched_genotypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported cubes and genotype-count disagreements upfront."""
    executable = tmp_path / "slim"
    script = tmp_path / "model.slim"
    executable.touch()
    script.touch()

    with pytest.raises(ValueError, match="unsupported SLiM cube"):
        benchmark_slim(
            repeats=1,
            n_days=0,
            executable=executable,
            script=script,
            cube="not-a-cube",
        )
    with pytest.raises(ValueError, match="requires 9 genotypes"):
        benchmark_slim(
            repeats=1,
            n_days=0,
            executable=executable,
            script=script,
            cube="twolocus9",
            n_genotypes=6,
        )


def test_benchmark_slim_uses_located_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use PATH lookup when no executable override is supplied."""
    executable = tmp_path / "slim"
    script = tmp_path / "model.slim"
    executable.touch()
    script.touch()
    monkeypatch.setattr(
        panmictic.shutil,
        "which",
        lambda _: str(executable),
    )
    monkeypatch.setattr(
        panmictic.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=VALID_OUT,
            stderr="",
        ),
    )

    result = benchmark_slim(repeats=1, n_days=30, script=script)

    assert result[0].population_size == 100


@pytest.mark.parametrize(
    ("stderr", "stdout", "message"),
    [
        ("fatal stderr", "ignored stdout", "fatal stderr"),
        ("", "fallback stdout", "fallback stdout"),
    ],
)
def test_benchmark_slim_surfaces_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    stdout: str,
    message: str,
) -> None:
    """Surface stderr, or stdout when stderr is empty, on nonzero exit."""
    executable = tmp_path / "slim"
    script = tmp_path / "model.slim"
    executable.touch()
    script.touch()
    monkeypatch.setattr(
        panmictic.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            7,
            stdout=stdout,
            stderr=stderr,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        benchmark_slim(
            repeats=1,
            n_days=1,
            executable=executable,
            script=script,
        )


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
def test_slim_check_accepts_the_model() -> None:
    """Have SLiM parse and validate the checked-in model."""
    process = subprocess.run(
        (str(SLIM), "-check", str(panmictic.SLIM_SCRIPT)),
        capture_output=True,
        check=False,
        text=True,
    )

    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
@pytest.mark.parametrize("cube", sorted(CUBE_GENOTYPES))
def test_slim_check_accepts_each_genotype_cube(cube: str) -> None:
    """Validate every neutral MGDrivE-compatible genetic architecture."""
    process = subprocess.run(
        (
            str(SLIM),
            "-check",
            "-d",
            f"CUBE='{cube}'",
            str(panmictic.SLIM_SCRIPT),
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
def test_slim_day_zero_matches_exact_integer_initial_population() -> None:
    """Represent the exact 32,750 individuals at MGDrivE absolute day one."""
    record = benchmark_slim(
        repeats=1,
        n_days=0,
        seed=20260807,
        executable=Path(str(SLIM)),
    )[0]

    assert record.day == 0
    assert record.population_size == 32750
    assert record.aquatic_total == 32250
    assert record.unmated_female_total == 0
    np.testing.assert_array_equal(record.adult_male, [250.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        record.adult_female,
        np.diag([250.0, 0.0, 0.0]),
    )


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
@pytest.mark.parametrize(
    ("cube", "wildtype"),
    [
        ("mendelian3", 0),
        ("multiallele6", 0),
        ("twolocus9", 8),
        ("twolocus18", 0),
    ],
)
def test_slim_day_zero_matches_each_cube_wildtype_equilibrium(
    cube: str,
    wildtype: int,
) -> None:
    """Keep every cube's initial adults wildtype and mated to wildtype."""
    n_genotypes = CUBE_GENOTYPES[cube]
    record = benchmark_slim(
        repeats=1,
        n_days=0,
        seed=20260807,
        executable=Path(str(SLIM)),
        cube=cube,
    )[0]
    male = np.zeros(n_genotypes)
    male[wildtype] = 250.0
    female = np.zeros((n_genotypes, n_genotypes))
    female[wildtype, wildtype] = 250.0

    assert record.day == 0
    assert record.population_size == 32750
    assert record.aquatic_total == 32250
    assert record.unmated_female_total == 0
    np.testing.assert_array_equal(record.adult_male, male)
    np.testing.assert_array_equal(record.adult_female, female)


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
def test_slim_population_scale_multiplies_initial_equilibrium() -> None:
    """Scale every equilibrium compartment while keeping release counts fixed."""
    record = benchmark_slim(
        repeats=1,
        n_days=0,
        seed=20260807,
        executable=Path(str(SLIM)),
        population_scale=2,
    )[0]

    assert record.population_size == 65500
    assert record.aquatic_total == 64500
    np.testing.assert_array_equal(record.adult_male, [500.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        record.adult_female,
        np.diag([500.0, 0.0, 0.0]),
    )


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
def test_slim_release_occurs_on_transition_24() -> None:
    """Introduce exactly ten aa adults of each sex on transition 24."""
    before = benchmark_slim(
        repeats=1,
        n_days=23,
        seed=42,
        executable=Path(str(SLIM)),
    )[0]
    released = benchmark_slim(
        repeats=1,
        n_days=24,
        seed=42,
        executable=Path(str(SLIM)),
    )[0]

    assert before.adult_male[2] == 0.0
    assert before.adult_female[2].sum() == 0.0
    assert released.adult_male[2] == 10.0
    assert released.adult_female[2].sum() == 10.0


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
@pytest.mark.parametrize("cube", sorted(CUBE_GENOTYPES))
def test_slim_release_occurs_on_transition_24_for_each_cube(
    cube: str,
) -> None:
    """Release the cube-specific genotype on transition 24 in every model."""
    release = CUBE_RELEASE_INDEX[cube]
    before = benchmark_slim(
        repeats=1,
        n_days=23,
        seed=42,
        executable=Path(str(SLIM)),
        cube=cube,
    )[0]
    released = benchmark_slim(
        repeats=1,
        n_days=24,
        seed=42,
        executable=Path(str(SLIM)),
        cube=cube,
    )[0]

    assert before.adult_male[release] == 0.0
    assert before.adult_female[release].sum() == 0.0
    assert released.adult_male[release] == 10.0
    assert released.adult_female[release].sum() == 10.0


@pytest.mark.skipif(SLIM is None, reason="SLiM executable is not installed")
def test_real_one_day_distributions_have_no_holm_significant_difference() -> None:
    """Compare independent stochastic samples at the real engine boundary."""
    natal = benchmark_natal_panmmictic(repeats=6, n_days=1, seed=100)
    slim = benchmark_slim(
        repeats=6,
        n_days=1,
        seed=200,
        executable=Path(str(SLIM)),
    )

    results = compare_stochastic_records(natal, slim)

    assert results
    assert all(0.0 <= result.holm_p <= 1.0 for result in results)
    with pytest.raises(RuntimeError, match="TOST"):
        validate_consistency(results)


@pytest.mark.parametrize("repeats", [0, 1.5, True])
def test_benchmark_natal_rejects_invalid_repeats(
    repeats: int | float,
) -> None:
    """Reject invalid NATAL replicate counts before scenario construction."""
    with pytest.raises(ValueError, match="repeats"):
        benchmark_natal_panmmictic(repeats=repeats, n_days=0)


@pytest.mark.parametrize("n_days", [-1, 1.5, True])
def test_benchmark_natal_rejects_invalid_days(n_days: int | float) -> None:
    """Reject invalid NATAL day counts before scenario construction."""
    with pytest.raises(ValueError, match="n_days"):
        benchmark_natal_panmmictic(repeats=1, n_days=n_days)


def _assert_natal_day_zero() -> None:
    """Assert the shared initial population through the selected execution path."""
    records = benchmark_natal_panmmictic(repeats=2, n_days=0, seed=20260807)
    assert [record.repeat for record in records] == [1, 2]
    for record in records:
        assert record.day == 0
        assert record.population_size == 32750
        assert record.aquatic_total == 32250
        assert record.unmated_female_total == 0
        np.testing.assert_array_equal(record.adult_male, [250.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            record.adult_female,
            np.diag([250.0, 0.0, 0.0]),
        )


@pytest.mark.numba_on
def test_natal_jit_day_zero_matches_explicit_population() -> None:
    """Exercise the JIT benchmark seam with exact initial counts."""
    _assert_natal_day_zero()


@pytest.mark.numba_off
def test_natal_fallback_day_zero_matches_explicit_population() -> None:
    """Exercise the Python fallback benchmark seam with exact initial counts."""
    _assert_natal_day_zero()


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    """Preserve family-wise error control and original metric order."""
    raw = np.array([0.04, 0.001, 0.03, 0.8])
    adjusted = _holm_adjust(raw)

    np.testing.assert_allclose(adjusted, [0.09, 0.004, 0.09, 0.8])
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0.0)
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1.0)


def test_statistical_comparison_uses_finite_welch_holm_and_tost() -> None:
    """Produce finite effect, confidence, difference, and equivalence results."""
    natal = tuple(
        _record(
            repeat=index + 1,
            aquatic_total=70 + index,
            male=np.array([10.0 + index, 2.0, 1.0 + index]),
            female=np.array(
                [
                    [8.0 + index, 1.0, float(index)],
                    [2.0, 1.0, 0.0],
                    [2.0 + index, 0.0, 1.0 + index],
                ]
            ),
        )
        for index in range(6)
    )
    slim = tuple(
        _record(
            repeat=index + 1,
            aquatic_total=71 + index,
            male=np.array([11.0 + index, 2.0, 1.0 + index]),
            female=np.array(
                [
                    [9.0 + index, 1.0, float(index)],
                    [2.0, 1.0, 0.0],
                    [2.0 + index, 0.0, 1.0 + index],
                ]
            ),
        )
        for index in range(6)
    )

    results = compare_stochastic_records(natal, slim)

    assert {result.metric for result in results} == {
        "population_size",
        "aquatic_total",
        "adult_total",
        "male_AA",
        "male_aa",
        "female_AA_mate_AA",
        "female_AA_mate_aa",
        "female_aa_mate_AA",
        "female_aa_mate_aa",
    }
    assert all(np.isfinite(result.standardized_difference) for result in results)
    assert all(result.ci90_low < result.ci90_high for result in results)
    assert all(0.0 <= result.welch_p <= result.holm_p <= 1.0 for result in results)
    assert all(0.0 <= result.tost_p <= 1.0 for result in results)
    assert all(result.equivalent == (result.tost_p < 0.05) for result in results)
    with pytest.raises(RuntimeError, match="TOST"):
        validate_consistency(results)


def test_statistical_comparison_requires_two_replicates_per_engine() -> None:
    """Reject either underidentified sample axis independently."""
    one = (_record(),)
    two = (_record(), _record(repeat=2))

    with pytest.raises(ValueError, match="two or more"):
        compare_stochastic_records(one, two)
    with pytest.raises(ValueError, match="two or more"):
        compare_stochastic_records(two, one)


def test_statistical_comparison_rejects_unequal_constant_outcomes() -> None:
    """Never silently discard a zero-variance cross-engine disagreement."""
    natal = (_record(), _record(repeat=2))
    slim = (
        _record(population_size=101, aquatic_total=71),
        _record(repeat=2, population_size=101, aquatic_total=71),
    )

    with pytest.raises(RuntimeError, match="unequal constant outcomes"):
        compare_stochastic_records(natal, slim)


def test_statistical_comparison_rejects_no_informative_metrics() -> None:
    """Reject two completely constant samples instead of returning no evidence."""
    records = (_record(), _record(repeat=2))

    with pytest.raises(RuntimeError, match="no informative"):
        compare_stochastic_records(records, records)


def test_statistical_comparison_rejects_nonfinite_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a non-finite result returned by the statistical backend."""
    natal = (
        _record(population_size=100),
        _record(repeat=2, population_size=102, aquatic_total=72),
    )
    slim = (
        _record(population_size=101, aquatic_total=71),
        _record(repeat=2, population_size=103, aquatic_total=73),
    )
    monkeypatch.setattr(
        runner.stats,
        "ttest_ind",
        lambda *args, **kwargs: type("Welch", (), {"pvalue": np.nan})(),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        compare_stochastic_records(natal, slim)


@pytest.mark.parametrize(
    "field",
    [
        "natal_mean",
        "slim_mean",
        "standardized_difference",
        "ci90_low",
        "ci90_high",
        "welch_p",
        "holm_p",
        "tost_p",
    ],
)
def test_validate_consistency_rejects_each_nonfinite_field(field: str) -> None:
    """Reject NaN in every statistic that feeds acceptance or reporting."""
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_consistency([_result(**{field: np.nan})])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("welch_p", -0.01),
        ("welch_p", 1.01),
        ("holm_p", -0.01),
        ("holm_p", 1.01),
        ("tost_p", -0.01),
        ("tost_p", 1.01),
    ],
)
def test_validate_consistency_rejects_out_of_range_p_values(
    field: str,
    value: float,
) -> None:
    """Require each probability to remain in the closed unit interval."""
    with pytest.raises(RuntimeError, match="invalid p-value"):
        validate_consistency([_result(**{field: value})])


def test_validate_consistency_rejects_holm_below_raw_p_value() -> None:
    """Require multiplicity correction never to reduce a raw p-value."""
    with pytest.raises(RuntimeError, match="inconsistent Holm correction"):
        validate_consistency([_result(welch_p=0.7, holm_p=0.6)])


@pytest.mark.parametrize(
    ("effect", "ci_low", "ci_high"),
    [
        (-0.4, -0.3, 0.1),
        (0.2, -0.3, 0.1),
        (-0.1, 0.1, -0.3),
    ],
)
def test_validate_consistency_rejects_effect_outside_confidence_interval(
    effect: float,
    ci_low: float,
    ci_high: float,
) -> None:
    """Require an ordered confidence interval containing its point estimate."""
    with pytest.raises(RuntimeError, match="inconsistent confidence interval"):
        validate_consistency(
            [
                _result(
                    standardized_difference=effect,
                    ci90_low=ci_low,
                    ci90_high=ci_high,
                )
            ]
        )


def test_validate_consistency_rejects_holm_significant_difference() -> None:
    """Mechanically reject a family-wise significant engine difference."""
    with pytest.raises(RuntimeError, match="Holm"):
        validate_consistency([_result(welch_p=0.01, holm_p=0.049)])


def test_validate_consistency_rejects_empty_results() -> None:
    """Reject an empty metric family because it contains no evidence."""
    with pytest.raises(RuntimeError, match="informative metrics"):
        validate_consistency([])


@pytest.mark.parametrize(
    ("tost_p", "equivalent", "ci_low", "ci_high"),
    [(0.04, False, -0.3, 0.1), (0.2, True, -0.6, 0.1)],
)
def test_validate_consistency_rejects_inconsistent_equivalence_flag(
    tost_p: float,
    equivalent: bool,
    ci_low: float,
    ci_high: float,
) -> None:
    """Require the cached equivalence flag to agree exactly with TOST."""
    with pytest.raises(RuntimeError, match="inconsistent equivalence flag"):
        validate_consistency(
            [
                _result(
                    tost_p=tost_p,
                    equivalent=equivalent,
                    ci90_low=ci_low,
                    ci90_high=ci_high,
                )
            ]
        )


@pytest.mark.parametrize(
    ("tost_p", "equivalent", "ci_low", "ci_high"),
    [
        (0.04, True, -0.6, 0.1),
        (0.04, True, -0.5, 0.1),
        (0.04, True, -0.3, 0.5),
        (0.2, False, -0.3, 0.1),
    ],
)
def test_validate_consistency_rejects_tost_ci_contradiction(
    tost_p: float,
    equivalent: bool,
    ci_low: float,
    ci_high: float,
) -> None:
    """Make TOST significance agree with strict containment in ±0.5 SD."""
    with pytest.raises(RuntimeError, match="inconsistent TOST"):
        validate_consistency(
            [
                _result(
                    tost_p=tost_p,
                    equivalent=equivalent,
                    ci90_low=ci_low,
                    ci90_high=ci_high,
                )
            ]
        )


def test_validate_consistency_requires_and_accepts_tost_equivalence() -> None:
    """Require formal equivalence after passing the corrected difference test."""
    with pytest.raises(RuntimeError, match="TOST"):
        validate_consistency(
            [
                _result(
                    tost_p=0.2,
                    equivalent=False,
                    ci90_low=-0.6,
                    ci90_high=0.1,
                )
            ]
        )

    validate_consistency([_result()])


def test_csv_writers_preserve_complete_record_and_statistic_schema(
    tmp_path: Path,
) -> None:
    """Write all genotype cells and all corrected statistics in stable order."""
    records_path = tmp_path / "records.csv"
    consistency_path = tmp_path / "consistency.csv"
    record = _record()
    result = _result()

    _write_records(records_path, [record])
    _write_consistency(consistency_path, [result])

    with records_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 2
    assert len(rows[0]) == 20
    assert rows[0][0:8] == [
        "engine",
        "mode",
        "repeat",
        "day",
        "elapsed_seconds",
        "population_size",
        "aquatic_total",
        "unmated_female_total",
    ]
    assert rows[1][0] == "test"
    assert [float(value) for value in rows[1][8:11]] == [10.0, 2.0, 1.0]
    assert [float(value) for value in rows[1][11:20]] == list(
        record.adult_female.ravel()
    )

    with consistency_path.open(newline="", encoding="utf-8") as source:
        statistic_rows = list(csv.reader(source))
    assert statistic_rows[0] == list(ConsistencyResult.__dataclass_fields__)
    assert statistic_rows[1][0] == "population_size"
    assert statistic_rows[1][-1] == "True"


def test_cli_writes_outputs_and_prints_finite_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise CLI argument plumbing, CSV output, and JSON summary."""
    natal = tuple(
        _record(
            repeat=index + 1,
            population_size=100 + index,
            aquatic_total=70 + index,
            elapsed_seconds=0.1 + 0.01 * index,
            engine="NATAL",
        )
        for index in range(3)
    )
    slim = tuple(
        _record(
            repeat=index + 1,
            population_size=101 + index,
            aquatic_total=71 + index,
            elapsed_seconds=1.0 + 0.1 * index,
            engine="SLiM",
        )
        for index in range(3)
    )
    consistency = (_result(),)
    natal_calls: list[tuple[int, int, int]] = []
    slim_calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        runner,
        "benchmark_natal_panmmictic",
        lambda *, repeats, n_days, seed: (
            natal_calls.append((repeats, n_days, seed)) or natal
        ),
    )
    monkeypatch.setattr(
        runner,
        "benchmark_slim",
        lambda *, repeats, n_days, seed: (
            slim_calls.append((repeats, n_days, seed)) or slim
        ),
    )
    monkeypatch.setattr(
        runner,
        "compare_stochastic_records",
        lambda natal_records, slim_records: consistency,
    )
    monkeypatch.setattr(runner, "validate_consistency", lambda results: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_slim_benchmark",
            "--days",
            "12",
            "--repeats",
            "3",
            "--seed",
            "77",
            "--output-dir",
            str(tmp_path),
        ],
    )

    runner.main()

    summary = json.loads(capsys.readouterr().out)
    assert natal_calls == [(3, 12, 77)]
    assert slim_calls == [(3, 12, 77)]
    assert summary["days"] == 12
    assert summary["repeats"] == 3
    assert summary["slim_to_natal_lifecycle_ratio"] == pytest.approx(10.0)
    assert summary["timing_scope"] == (
        "both engines exclude initialization and compilation"
    )
    assert summary["minimum_holm_p"] == 0.8
    assert summary["equivalent_metrics"] == 1
    assert summary["tested_metrics"] == 1
    assert summary["maximum_abs_standardized_difference"] == 0.1
    assert 0.0 <= summary["log_runtime_welch_p"] <= 1.0
    assert (tmp_path / "natal-panmmictic.csv").is_file()
    assert (tmp_path / "slim-panmmictic.csv").is_file()
    assert (tmp_path / "consistency.csv").is_file()


def test_cli_parser_rejects_noninteger_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed numeric CLI input before any benchmark is run."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_slim_benchmark", "--days", "not-an-integer"],
    )

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 2


def test_removed_spatial_and_deterministic_interfaces_are_absent() -> None:
    """Keep the SLiM benchmark deliberately panmictic and stochastic-only."""
    assert not hasattr(panmictic, "benchmark_slim_spatial")
    assert not hasattr(panmictic, "benchmark_slim_deterministic")
    assert not hasattr(runner, "compare_deterministic_records")
