"""Run the individual-based SLiM panmictic comparison.

The SLiM model follows the stochastic MGDrivE1-compatible daily lifecycle used
by NATAL's benchmark kernel. It deliberately excludes spatial migration.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from natal.numba.compat import set_numba_seed

from .lifecycle import DailyRelease
from .spatial_benchmark import (
    build_hex_benchmark_scenario,
    run_spatial,
    step_spatial,
)

FloatArray = NDArray[np.float64]
SLIM_SCRIPT = Path(__file__).with_name("mgdrive1_panmmictic.slim")
CUBE_GENOTYPES = {
    "mendelian3": 3,
    "multiallele6": 6,
    "twolocus9": 9,
    "twolocus18": 18,
}
CUBE_RELEASE_INDEX = {
    "mendelian3": 2,
    "multiallele6": 4,
    "twolocus9": 0,
    "twolocus18": 11,
}


def _validate_cube(cube: str, n_genotypes: int | None) -> int:
    """Return the resolved genotype count for a supported MGDrivE cube."""
    if cube not in CUBE_GENOTYPES:
        raise ValueError(f"unsupported SLiM cube: {cube}")
    expected = CUBE_GENOTYPES[cube]
    if n_genotypes is not None and n_genotypes != expected:
        raise ValueError(
            f"cube {cube} requires {expected} genotypes, got {n_genotypes}"
        )
    return expected


def _read_only(value: FloatArray) -> FloatArray:
    """Return an owned, read-only float64 array.

    Args:
        value: Array supplied by the caller.

    Returns:
        Owned, read-only array.
    """
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PanmicticRecord:
    """Store one stochastic panmictic benchmark result.

    Attributes:
        engine: Simulation engine label.
        mode: Simulation mode; always ``"stochastic"`` for this benchmark.
        repeat: One-based replicate index.
        day: Number of completed daily transitions.
        elapsed_seconds: End-to-end elapsed wall time.
        population_size: Total explicit individuals in the focal population.
        aquatic_total: Total explicit aquatic-stage individuals.
        unmated_female_total: Adult females without stored sperm.
        adult_male: Adult male counts by genotype.
        adult_female: Mated adult female counts by female and mate genotype.
        n_genotypes: Number of genotype compartments in the genetic model.
    """

    engine: str
    mode: str
    repeat: int
    day: int
    elapsed_seconds: float
    population_size: int
    aquatic_total: int
    unmated_female_total: int
    adult_male: FloatArray
    adult_female: FloatArray
    n_genotypes: int = 3

    def __post_init__(self) -> None:
        """Take ownership of arrays and validate record dimensions.

        Raises:
            ValueError: If a scalar or genotype axis is invalid.
        """
        integer_scalars = (
            self.repeat,
            self.day,
            self.population_size,
            self.aquatic_total,
            self.unmated_female_total,
        )
        if any(type(value) is not int for value in integer_scalars):
            raise ValueError("indices and explicit population counts must be integers")
        if type(self.n_genotypes) is not int or self.n_genotypes < 1:
            raise ValueError("n_genotypes must be a positive integer")
        object.__setattr__(self, "adult_male", _read_only(self.adult_male))
        object.__setattr__(
            self,
            "adult_female",
            _read_only(self.adult_female),
        )
        if self.mode != "stochastic":
            raise ValueError("SLiM comparison supports stochastic mode only")
        if self.repeat < 1 or self.day < 0:
            raise ValueError("repeat and day are outside their valid ranges")
        if (
            not np.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if min(
            self.population_size,
            self.aquatic_total,
            self.unmated_female_total,
        ) < 0:
            raise ValueError("population counts must be nonnegative")
        if self.adult_male.shape != (self.n_genotypes,):
            raise ValueError(
                f"adult_male must contain {self.n_genotypes} genotypes"
            )
        if self.adult_female.shape != (
            self.n_genotypes,
            self.n_genotypes,
        ):
            raise ValueError(
                f"adult_female must be {self.n_genotypes} by "
                f"{self.n_genotypes}"
            )
        if (
            not np.all(np.isfinite(self.adult_male))
            or not np.all(np.isfinite(self.adult_female))
            or np.any(self.adult_male < 0.0)
            or np.any(self.adult_female < 0.0)
        ):
            raise ValueError("adult counts must be finite and nonnegative")
        if (
            not np.array_equal(self.adult_male, np.rint(self.adult_male))
            or not np.array_equal(
                self.adult_female,
                np.rint(self.adult_female),
            )
        ):
            raise ValueError("explicit individual counts must be integers")
        adult_total = int(
            round(float(self.adult_male.sum() + self.adult_female.sum()))
        )
        if self.population_size != (
            self.aquatic_total
            + self.unmated_female_total
            + adult_total
        ):
            raise ValueError("population counts violate total conservation")


@dataclass(frozen=True)
class ConsistencyResult:
    """Store one cross-engine stochastic distribution comparison.

    Attributes:
        metric: Compared outcome name.
        natal_mean: NATAL replicate mean.
        slim_mean: SLiM replicate mean.
        standardized_difference: Mean difference in pooled-SD units.
        ci90_low: Lower 90% confidence bound in pooled-SD units.
        ci90_high: Upper 90% confidence bound in pooled-SD units.
        welch_p: Unadjusted two-sided Welch-test p-value.
        holm_p: Family-wise Holm-adjusted p-value.
        tost_p: Two-one-sided-test p-value for a ±0.5 pooled-SD margin.
        equivalent: Whether the TOST p-value is below 0.05.
    """

    metric: str
    natal_mean: float
    slim_mean: float
    standardized_difference: float
    ci90_low: float
    ci90_high: float
    welch_p: float
    holm_p: float
    tost_p: float
    equivalent: bool


def parse_slim_output(
    stdout: str,
    *,
    repeat: int,
    n_genotypes: int = 3,
) -> PanmicticRecord:
    """Parse the final public ``OUT`` record emitted by the SLiM model.

    Args:
        stdout: Complete SLiM standard output.
        repeat: One-based replicate index.
        n_genotypes: Number of genotype compartments in the OUT record.

    Returns:
        Parsed immutable benchmark result.

    Raises:
        RuntimeError: If the external output lacks one valid final record.
    """
    output_lines = [
        line.removeprefix("OUT:")
        for line in stdout.splitlines()
        if line.startswith("OUT:")
    ]
    if len(output_lines) != 1:
        raise RuntimeError("SLiM output must contain exactly one OUT record")
    timing_lines = [
        line.removeprefix("TIME:")
        for line in stdout.splitlines()
        if line.startswith("TIME:")
    ]
    if len(timing_lines) != 1:
        raise RuntimeError("SLiM output must contain exactly one TIME record")
    try:
        elapsed_seconds = float(timing_lines[0])
    except ValueError as error:
        raise RuntimeError("SLiM TIME field must be numeric") from error
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise RuntimeError("SLiM TIME field must be finite and nonnegative")
    if type(n_genotypes) is not int or n_genotypes < 1:
        raise ValueError("n_genotypes must be a positive integer")
    expected_fields = 5 + n_genotypes + n_genotypes**2
    fields = output_lines[0].split(",")
    if len(fields) != expected_fields:
        raise RuntimeError(
            f"SLiM OUT record must contain {expected_fields} fields"
        )
    try:
        values = np.array(fields, dtype=np.float64)
    except ValueError as error:
        raise RuntimeError("SLiM OUT fields must be numeric") from error
    if not np.all(np.isfinite(values)):
        raise RuntimeError("SLiM OUT fields must be finite")

    integer_values = np.rint(values).astype(np.int64)
    if not np.array_equal(values, integer_values.astype(np.float64)):
        raise RuntimeError("SLiM population fields must be integers")
    if integer_values[1] != integer_values[2:5].sum():
        raise RuntimeError("SLiM population fields violate total conservation")
    if integer_values[4] != integer_values[5:].sum():
        raise RuntimeError("SLiM adult fields violate total conservation")
    adult_male = values[5 : 5 + n_genotypes]
    adult_female = values[5 + n_genotypes :].reshape(
        n_genotypes,
        n_genotypes,
    )
    return PanmicticRecord(
        engine="SLiM-5.0-individual",
        mode="stochastic",
        repeat=repeat,
        day=int(integer_values[0]),
        elapsed_seconds=elapsed_seconds,
        population_size=int(integer_values[1]),
        aquatic_total=int(integer_values[2]),
        unmated_female_total=int(integer_values[3]),
        adult_male=adult_male,
        adult_female=adult_female,
        n_genotypes=n_genotypes,
    )


def benchmark_slim(
    *,
    repeats: int,
    n_days: int,
    seed: int = 20260807,
    executable: Path | None = None,
    script: Path = SLIM_SCRIPT,
    population_scale: int = 1,
    cube: str = "mendelian3",
    n_genotypes: int | None = None,
) -> tuple[PanmicticRecord, ...]:
    """Run independent stochastic SLiM panmictic replicates.

    Args:
        repeats: Number of independent subprocess replicates.
        n_days: Number of MGDrivE-compatible daily transitions.
        seed: Base SLiM random seed.
        executable: Optional explicit SLiM executable path.
        script: SLiM model script.
        population_scale: Integer multiplier for the equilibrium population.
        cube: MGDrivE1 inheritance cube name selecting the genetic model.
        n_genotypes: Optional genotype count; validated against the cube.

    Returns:
        Immutable records in replicate order.

    Raises:
        ValueError: If repetitions or days are invalid.
        FileNotFoundError: If SLiM or the model script is unavailable.
        RuntimeError: If SLiM exits unsuccessfully or emits invalid output.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if type(n_days) is not int or n_days < 0:
        raise ValueError("n_days must be a nonnegative integer")
    if type(population_scale) is not int or population_scale < 1:
        raise ValueError("population_scale must be a positive integer")
    resolved_genotypes = _validate_cube(cube, n_genotypes)
    executable_path = executable
    if executable_path is None:
        configured = os.environ.get("CLUSTER_SLIM_BIN")
        located = configured or shutil.which("slim")
        if located is None:
            raise FileNotFoundError("SLiM executable was not found")
        executable_path = Path(located)
    if not executable_path.is_file():
        raise FileNotFoundError(f"SLiM executable was not found: {executable_path}")
    if not script.is_file():
        raise FileNotFoundError(f"SLiM model was not found: {script}")

    records: list[PanmicticRecord] = []
    for repeat_index in range(repeats):
        process = subprocess.run(
            (
                str(executable_path),
                "-s",
                str(seed + repeat_index),
                "-d",
                f"N_DAYS={n_days}",
                "-d",
                f"POP_SCALE={population_scale}",
                "-d",
                f"CUBE='{cube}'",
                str(script),
            ),
            capture_output=True,
            check=False,
            text=True,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"SLiM failed with exit {process.returncode}: {message}")
        records.append(
            parse_slim_output(
                process.stdout,
                repeat=repeat_index + 1,
                n_genotypes=resolved_genotypes,
            )
        )
    return tuple(records)


def benchmark_natal_panmmictic(
    *,
    repeats: int,
    n_days: int,
    seed: int = 20260807,
) -> tuple[PanmicticRecord, ...]:
    """Run NATAL's matching stochastic single-patch lifecycle.

    Args:
        repeats: Number of independent timed replicates.
        n_days: Number of MGDrivE-compatible daily transitions.
        seed: Base NATAL random seed.

    Returns:
        Immutable records in replicate order.

    Raises:
        ValueError: If repetitions or days are invalid.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if type(n_days) is not int or n_days < 0:
        raise ValueError("n_days must be a nonnegative integer")
    scenario = build_hex_benchmark_scenario(
        rows=1,
        cols=1,
        migration_rate=0.0,
        stochastic=True,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        unmated_female=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        adult_female=np.zeros((3, 3), dtype=np.float64),
        eggs=np.zeros(3, dtype=np.float64),
    )

    # Exclude Numba compilation from timed replicates.
    set_numba_seed(seed)
    step_spatial(
        scenario.state,
        scenario.config,
        migration_kernel=scenario.migration_kernel,
        rows=1,
        cols=1,
        migration_rate=0.0,
        stochastic=True,
    )

    records: list[PanmicticRecord] = []
    for repeat_index in range(repeats):
        # A single deme has no parallel-deme RNG ambiguity, so the existing
        # panmictic benchmark can retain its reproducible seed contract.
        set_numba_seed(seed + repeat_index)
        start = perf_counter()
        result = run_spatial(
            scenario.state,
            scenario.config,
            migration_kernel=scenario.migration_kernel,
            rows=1,
            cols=1,
            migration_rate=0.0,
            stochastic=True,
            n_days=n_days,
            release_day=25,
            release_deme=0,
            release=release,
        )
        elapsed_seconds = perf_counter() - start
        adult_male = result.adult_male[0]
        adult_female = result.adult_female[0]
        unmated_total = int(round(float(result.unmated_female.sum())))
        aquatic_total = int(round(float(result.aquatic.sum())))
        adult_total = int(
            round(float(adult_male.sum() + adult_female.sum()))
        )
        records.append(
            PanmicticRecord(
                engine="NATAL-Core-MG-compatible",
                mode="stochastic",
                repeat=repeat_index + 1,
                day=n_days,
                elapsed_seconds=elapsed_seconds,
                population_size=aquatic_total
                + unmated_total
                + adult_total,
                aquatic_total=aquatic_total,
                unmated_female_total=unmated_total,
                adult_male=adult_male,
                adult_female=adult_female,
            )
        )
    return tuple(records)
