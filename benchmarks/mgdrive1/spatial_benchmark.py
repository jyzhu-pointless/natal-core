"""NATAL-backed deterministic and stochastic MGDrivE1 spatial benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from natal.engine.simulation.mgdrive1_compatible import (
    advance_mgdrive1_lifecycle,
)
from natal.engine.spatial_migrator import apply_spatial_adjacency_migration
from natal.spatial.topology import HexGrid, build_gaussian_kernel

from .lifecycle import DailyRelease, DeterministicConfig, PatchState

FloatArray = NDArray[np.float64]


def _read_only(value: FloatArray) -> FloatArray:
    """Return an owned, read-only float64 array.

    Args:
        value: Input array.

    Returns:
        Owned array with mutation disabled.
    """
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SpatialPatchState:
    """Store MGDrivE1 compartments for multiple demes.

    Attributes:
        aquatic: Deme-by-genotype-by-aquatic-day counts.
        adult_male: Deme-by-genotype adult male counts.
        adult_female: Deme-by-female-by-mate genotype mated counts.
        unmated_female: Deme-by-genotype unmated female counts.
    """

    aquatic: FloatArray
    adult_male: FloatArray
    adult_female: FloatArray
    unmated_female: FloatArray

    def __post_init__(self) -> None:
        """Own and validate all spatial state arrays.

        Raises:
            ValueError: If an axis, count, or numerical value is invalid.
        """
        for name in (
            "aquatic",
            "adult_male",
            "adult_female",
            "unmated_female",
        ):
            object.__setattr__(self, name, _read_only(getattr(self, name)))
        if self.aquatic.ndim != 3:
            raise ValueError("aquatic must be deme by genotype by aquatic day")
        n_demes, n_genotypes, _ = self.aquatic.shape
        if self.adult_male.shape != (n_demes, n_genotypes):
            raise ValueError("adult_male spatial axes do not match aquatic")
        if self.adult_female.shape != (
            n_demes,
            n_genotypes,
            n_genotypes,
        ):
            raise ValueError("adult_female spatial axes do not match aquatic")
        if self.unmated_female.shape != (n_demes, n_genotypes):
            raise ValueError("unmated_female spatial axes do not match aquatic")
        arrays = (
            self.aquatic,
            self.adult_male,
            self.adult_female,
            self.unmated_female,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("spatial state arrays must contain finite values")
        if any(np.any(value < 0.0) for value in arrays):
            raise ValueError("spatial population counts must be nonnegative")


@dataclass(frozen=True)
class HexBenchmarkScenario:
    """Bundle one local-kernel hex benchmark scenario.

    Attributes:
        config: Shared MGDrivE1 lifecycle configuration.
        state: Initial state across all demes.
        migration_kernel: Compact local Gaussian kernel.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        migration_rate: Daily adult migration probability.
    """

    config: DeterministicConfig
    state: SpatialPatchState
    migration_kernel: FloatArray
    rows: int
    cols: int
    migration_rate: float

    def __post_init__(self) -> None:
        """Protect the compact kernel from mutation."""
        object.__setattr__(
            self,
            "migration_kernel",
            _read_only(self.migration_kernel),
        )


@dataclass(frozen=True)
class BenchmarkRecord:
    """Store one timed NATAL benchmark replicate.

    Attributes:
        mode: Deterministic or stochastic execution mode.
        repeat: One-based replicate index.
        elapsed_seconds: Warm-cache simulation time.
        adult_male: Final male genotype totals.
        adult_female: Final female-by-mate genotype totals.
        male_aa_radius2: Spatial second moment of ``aa`` adult males.
        female_aa_radius2: Spatial second moment of ``aa`` adult females.
    """

    mode: str
    repeat: int
    elapsed_seconds: float
    adult_male: FloatArray
    adult_female: FloatArray
    male_aa_radius2: float
    female_aa_radius2: float

    def __post_init__(self) -> None:
        """Protect final aggregate arrays from mutation."""
        object.__setattr__(self, "adult_male", _read_only(self.adult_male))
        object.__setattr__(
            self,
            "adult_female",
            _read_only(self.adult_female),
        )


def stack_patch_state(
    state: PatchState,
    *,
    n_demes: int,
    stochastic: bool,
) -> SpatialPatchState:
    """Replicate one patch state across homogeneous demes.

    Args:
        state: Single-patch initial state.
        n_demes: Number of homogeneous demes.
        stochastic: Whether to round counts like MGDrivE1 stochastic setup.

    Returns:
        Spatial state with independent deme axes.

    Raises:
        ValueError: If ``n_demes`` is not positive.
    """
    if type(n_demes) is not int or n_demes < 1:
        raise ValueError("n_demes must be a positive integer")

    def replicate(value: FloatArray) -> FloatArray:
        """Copy one patch array across the requested deme axis."""
        result = np.repeat(value[np.newaxis, ...], n_demes, axis=0)
        return np.rint(result) if stochastic else result

    return SpatialPatchState(
        aquatic=replicate(state.aquatic),
        adult_male=replicate(state.adult_male),
        adult_female=replicate(state.adult_female),
        unmated_female=replicate(state.unmated_female),
    )


def _advance_lifecycle(
    state: SpatialPatchState,
    config: DeterministicConfig,
    *,
    stochastic: bool,
    release: tuple[int, DailyRelease] | None,
) -> SpatialPatchState:
    """Advance all demes through the patch-local lifecycle.

    Args:
        state: Spatial state at the start of the day.
        config: Shared lifecycle configuration.
        stochastic: Whether to sample MGDrivE1 demographic events.
        release: Optional target deme and daily release.

    Returns:
        State after local population dynamics and before migration.
    """
    n_demes = int(state.aquatic.shape[0])
    n_genotypes = int(state.aquatic.shape[1])
    aquatic_duration = int(state.aquatic.shape[2])
    expected_duration = (
        config.time_egg + config.time_larva + config.time_pupa
    )
    if config.inheritance.shape[0] != n_genotypes:
        raise ValueError("state and config genotype axes do not match")
    if aquatic_duration != expected_duration:
        raise ValueError("state aquatic duration does not match config")
    if release is not None:
        release_deme, release_values = release
        if release_deme < 0 or release_deme >= n_demes:
            raise ValueError("release_deme is outside the spatial state")
        if release_values.adult_male.size != n_genotypes:
            raise ValueError("release and state genotype axes do not match")

    release_deme = -1
    release_values = DailyRelease(
        adult_male=np.zeros(n_genotypes, dtype=np.float64),
        unmated_female=np.zeros(n_genotypes, dtype=np.float64),
        adult_female=np.zeros(
            (n_genotypes, n_genotypes),
            dtype=np.float64,
        ),
        eggs=np.zeros(n_genotypes, dtype=np.float64),
    )
    if release is not None:
        release_deme, release_values = release
    aquatic, adult_male, adult_female, unmated_female = (
        advance_mgdrive1_lifecycle(
            state.aquatic,
            state.adult_male,
            state.adult_female,
            state.unmated_female,
            config.time_egg,
            config.time_larva,
            config.time_pupa,
            config.beta,
            config.adult_mortality,
            config.aquatic_mortality,
            config.alpha,
            config.inheritance,
            config.mating_fitness,
            config.female_fraction,
            config.adult_survival_modifier,
            config.female_emergence,
            config.male_emergence,
            config.fertility_modifier,
            stochastic,
            release_deme,
            release_values.adult_male,
            release_values.unmated_female,
            release_values.adult_female,
            release_values.eggs,
        )
    )
    return SpatialPatchState(
        aquatic=aquatic,
        adult_male=adult_male,
        adult_female=adult_female,
        unmated_female=unmated_female,
    )


def _migrate_adults(
    state: SpatialPatchState,
    *,
    migration_kernel: FloatArray,
    rows: int,
    cols: int,
    migration_rate: float,
    stochastic: bool,
) -> SpatialPatchState:
    """Migrate adults through NATAL Core's compact topology-kernel backend.

    Args:
        state: State after patch-local lifecycle events.
        migration_kernel: Compact source-relative kernel.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        migration_rate: Daily adult migration probability.
        stochastic: Whether to sample multinomial migration.

    Returns:
        State after synchronized adult migration.
    """
    if migration_rate <= 0.0:
        return state
    n_demes, n_genotypes = state.adult_male.shape
    if rows * cols != n_demes:
        raise ValueError("rows * cols must equal the number of demes")
    individuals = np.zeros(
        (n_demes, 2, 1, n_genotypes),
        dtype=np.float64,
    )
    # MGDrivE1 migration moves mated females and males only. Unmated females
    # remain in their patch until mating resolves them.
    individuals[:, 0, 0] = state.adult_female.sum(axis=2)
    individuals[:, 1, 0] = state.adult_male
    sperm = state.adult_female[:, np.newaxis, :, :].copy()
    migrated_individuals, migrated_sperm = (
        apply_spatial_adjacency_migration(
            ind_count_all=individuals,
            sperm_store_all=sperm,
            adjacency=np.zeros((1, 1), dtype=np.float64),
            migration_mode=1,
            topology_rows=rows,
            topology_cols=cols,
            topology_wrap=False,
            migration_kernel=np.asarray(
                migration_kernel,
                dtype=np.float64,
            ),
            kernel_include_center=False,
            rate=np.array([migration_rate], dtype=np.float64),
            stochastic=stochastic,
            continuous_sampling=False,
            adjust_migration_on_edge=True,
        )
    )
    adult_female = migrated_sperm[:, 0]
    return SpatialPatchState(
        aquatic=state.aquatic,
        adult_male=migrated_individuals[:, 1, 0],
        adult_female=adult_female,
        unmated_female=state.unmated_female,
    )


def step_spatial(
    state: SpatialPatchState,
    config: DeterministicConfig,
    *,
    migration_kernel: FloatArray,
    rows: int,
    cols: int,
    migration_rate: float,
    stochastic: bool,
    release: tuple[int, DailyRelease] | None = None,
) -> SpatialPatchState:
    """Advance one spatial MGDrivE1 day using NATAL local migration.

    Args:
        state: Spatial state at the start of the day.
        config: Shared lifecycle configuration.
        migration_kernel: Compact local Gaussian kernel.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        migration_rate: Daily adult migration probability.
        stochastic: Whether demographic and migration events are sampled.
        release: Optional target deme and release for this day.

    Returns:
        New state after lifecycle and synchronized migration.
    """
    local_state = _advance_lifecycle(
        state,
        config,
        stochastic=stochastic,
        release=release,
    )
    return _migrate_adults(
        local_state,
        migration_kernel=migration_kernel,
        rows=rows,
        cols=cols,
        migration_rate=migration_rate,
        stochastic=stochastic,
    )


def run_spatial(
    state: SpatialPatchState,
    config: DeterministicConfig,
    *,
    migration_kernel: FloatArray,
    rows: int,
    cols: int,
    migration_rate: float,
    n_days: int,
    initial_day: int = 1,
    release_day: int | None = None,
    release_deme: int | None = None,
    release: DailyRelease | None = None,
    stochastic: bool,
) -> SpatialPatchState:
    """Run a spatial benchmark scenario without recording daily history.

    Args:
        state: Initial spatial state.
        config: Shared lifecycle configuration.
        migration_kernel: Compact local Gaussian kernel.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        migration_rate: Daily adult migration probability.
        n_days: Number of daily transitions.
        initial_day: MGDrivE1 time represented by the input state.
        release_day: Optional absolute MGDrivE1 release day.
        release_deme: Optional target deme for the release.
        release: Optional release counts.
        stochastic: Whether to sample demographic and migration events.

    Returns:
        Final spatial state.

    Raises:
        ValueError: If release parameters are incomplete or days are invalid.
    """
    if type(n_days) is not int or n_days < 0:
        raise ValueError("n_days must be a nonnegative integer")
    release_values = (release_day, release_deme, release)
    supplied = tuple(value is not None for value in release_values)
    if any(supplied) and not all(supplied):
        raise ValueError("release_day, release_deme, and release must be set")
    result = SpatialPatchState(
        aquatic=state.aquatic,
        adult_male=state.adult_male,
        adult_female=state.adult_female,
        unmated_female=state.unmated_female,
    )
    for offset in range(1, n_days + 1):
        day = initial_day + offset
        scheduled_release = (
            (release_deme, release)
            if release_day == day
            and release_deme is not None
            and release is not None
            else None
        )
        result = step_spatial(
            result,
            config,
            migration_kernel=migration_kernel,
            rows=rows,
            cols=cols,
            migration_rate=migration_rate,
            stochastic=stochastic,
            release=scheduled_release,
        )
    return result


def adult_totals(
    state: SpatialPatchState,
) -> tuple[FloatArray, FloatArray]:
    """Aggregate adult male and mated-female counts over demes.

    Args:
        state: Spatial state to summarize.

    Returns:
        Male genotype totals and female-by-mate genotype totals.
    """
    return (
        _read_only(state.adult_male.sum(axis=0)),
        _read_only(state.adult_female.sum(axis=0)),
    )


def recessive_spatial_moments(
    state: SpatialPatchState,
    *,
    rows: int,
    cols: int,
) -> tuple[float, float]:
    """Measure squared hex distance carried by recessive adults.

    Args:
        state: Spatial state to summarize.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.

    Returns:
        Unnormalized second moments for ``aa`` males and females.
    """
    if rows * cols != state.aquatic.shape[0]:
        raise ValueError("rows * cols must equal the number of demes")
    row_index, col_index = np.indices((rows, cols), dtype=np.float64)
    center_row = (rows - 1) // 2
    center_col = (cols - 1) // 2
    dr = row_index.reshape(-1) - center_row
    dc = col_index.reshape(-1) - center_col
    distance_squared = dr**2 + dc**2 + dr * dc
    male_aa = state.adult_male[:, 2]
    female_aa = state.adult_female[:, 2, :].sum(axis=1)
    return (
        float(np.dot(male_aa, distance_squared)),
        float(np.dot(female_aa, distance_squared)),
    )


def build_hex_benchmark_scenario(
    *,
    rows: int = 15,
    cols: int = 15,
    kernel_size: int = 5,
    sigma: float = 1.0,
    migration_rate: float = 0.05,
    stochastic: bool,
) -> HexBenchmarkScenario:
    """Build the homogeneous MGDrivE1 15-by-15 hex scenario.

    Args:
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        kernel_size: Odd side length of the local Gaussian kernel.
        sigma: Gaussian dispersal scale in hex-cell units.
        migration_rate: Daily adult migration probability.
        stochastic: Whether to round the initial state to integer counts.

    Returns:
        Ready-to-run local-kernel benchmark scenario.
    """
    from .lifecycle import build_mendelian_equilibrium

    config, patch_state, _ = build_mendelian_equilibrium(
        time_egg=5,
        time_larva=6,
        time_pupa=4,
        beta=20.0,
        adult_mortality=0.09,
        daily_population_growth=1.175,
        adult_equilibrium=500.0,
    )
    kernel = build_gaussian_kernel(
        HexGrid,
        size=kernel_size,
        sigma=sigma,
    )
    return HexBenchmarkScenario(
        config=config,
        state=stack_patch_state(
            patch_state,
            n_demes=rows * cols,
            stochastic=stochastic,
        ),
        migration_kernel=kernel,
        rows=rows,
        cols=cols,
        migration_rate=migration_rate,
    )


def benchmark_natal(
    *,
    stochastic: bool,
    rows: int = 15,
    cols: int = 15,
    n_days: int = 30,
    repeats: int = 30,
    kernel_size: int = 5,
    sigma: float = 1.0,
    migration_rate: float = 0.05,
) -> tuple[BenchmarkRecord, ...]:
    """Benchmark NATAL's compact local-kernel execution path.

    The first one-day call warms Numba compilation and is excluded from timing.
    Every timed replicate starts from the same equilibrium state. Stochastic
    execution uses Numba's runtime-managed random state.

    Args:
        stochastic: Whether to sample demographic and migration events.
        rows: Number of hex-grid rows.
        cols: Number of hex-grid columns.
        n_days: Number of daily transitions per replicate.
        repeats: Number of timed replicates.
        kernel_size: Odd side length of the local Gaussian kernel.
        sigma: Gaussian dispersal scale in hex-cell units.
        migration_rate: Daily adult migration probability.

    Returns:
        Immutable benchmark records for every replicate.

    Raises:
        ValueError: If ``repeats`` is not positive.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    scenario = build_hex_benchmark_scenario(
        rows=rows,
        cols=cols,
        kernel_size=kernel_size,
        sigma=sigma,
        migration_rate=migration_rate,
        stochastic=stochastic,
    )
    release = DailyRelease(
        adult_male=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        unmated_female=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        adult_female=np.zeros((3, 3), dtype=np.float64),
        eggs=np.zeros(3, dtype=np.float64),
    )
    center_deme = ((rows - 1) // 2) * cols + (cols - 1) // 2

    # Warm the compiled lifecycle and migration paths before timed replicates.
    step_spatial(
        scenario.state,
        scenario.config,
        migration_kernel=scenario.migration_kernel,
        rows=rows,
        cols=cols,
        migration_rate=migration_rate,
        stochastic=stochastic,
    )

    records: list[BenchmarkRecord] = []
    for repeat_index in range(repeats):
        start = perf_counter()
        result = run_spatial(
            scenario.state,
            scenario.config,
            migration_kernel=scenario.migration_kernel,
            rows=rows,
            cols=cols,
            migration_rate=migration_rate,
            n_days=n_days,
            release_day=25,
            release_deme=center_deme,
            release=release,
            stochastic=stochastic,
        )
        elapsed = perf_counter() - start
        male, female = adult_totals(result)
        male_radius2, female_radius2 = recessive_spatial_moments(
            result,
            rows=rows,
            cols=cols,
        )
        records.append(
            BenchmarkRecord(
                mode="stochastic" if stochastic else "deterministic",
                repeat=repeat_index + 1,
                elapsed_seconds=elapsed,
                adult_male=male,
                adult_female=female,
                male_aa_radius2=male_radius2,
                female_aa_radius2=female_radius2,
            )
        )
    return tuple(records)
