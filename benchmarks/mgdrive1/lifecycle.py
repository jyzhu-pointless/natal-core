"""Deterministic single-patch lifecycle compatible with MGDrivE1.

The event order and equations follow ``R/Patch-Simulation.R`` from MGDrivE
commit ``f7ec820e8a6b0f4fa5697b190f6cb0b1d2d02311``. This module deliberately
starts as a correctness reference for the benchmark; acceleration is added only
after its numerical behavior is locked down.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mendelian_inheritance() -> FloatArray:
    """Return the default MGDrivE1 ``AA``, ``Aa``, ``aa`` inheritance cube.

    Returns:
        Female-by-male-by-offspring inheritance probabilities.
    """
    return np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.5, 0.5, 0.0],
                [0.25, 0.5, 0.25],
                [0.0, 0.5, 0.5],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )


def build_mendelian_equilibrium(
    *,
    time_egg: int,
    time_larva: int,
    time_pupa: int,
    beta: float,
    adult_mortality: float,
    daily_population_growth: float,
    adult_equilibrium: float,
) -> tuple[DeterministicConfig, PatchState, int]:
    """Build the default MGDrivE1 Mendelian equilibrium scenario.

    Args:
        time_egg: Number of daily egg cohorts.
        time_larva: Number of daily larval cohorts.
        time_pupa: Number of daily pupal cohorts.
        beta: Wild-type eggs laid per mated female per day.
        adult_mortality: Daily adult mortality probability.
        daily_population_growth: Low-density daily population growth.
        adult_equilibrium: Total equilibrium adult population.

    Returns:
        Configuration, equilibrium patch state, and integer larval equilibrium.

    Raises:
        ValueError: If the bionomic parameters cannot define an equilibrium.
    """
    stage_times = (time_egg, time_larva, time_pupa)
    if any(
        type(value) is not int
        or value < 1
        for value in stage_times
    ):
        raise ValueError("aquatic stage durations must be positive integers")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if (
        not np.isfinite(adult_mortality)
        or adult_mortality <= 0.0
        or adult_mortality >= 1.0
    ):
        raise ValueError("adult_mortality must be finite and between 0 and 1")
    if (
        not np.isfinite(daily_population_growth)
        or daily_population_growth <= 1.0
    ):
        raise ValueError("daily_population_growth must be finite and above 1")
    if not np.isfinite(adult_equilibrium) or adult_equilibrium < 0.0:
        raise ValueError("adult_equilibrium must be nonnegative and finite")

    total_aquatic_time = time_egg + time_larva + time_pupa
    generation_time = total_aquatic_time + 1.0 / adult_mortality
    generation_growth = daily_population_growth**generation_time
    aquatic_mortality = 1.0 - (
        (generation_growth * adult_mortality)
        / ((beta / 2.0) * (1.0 - adult_mortality))
    ) ** (1.0 / total_aquatic_time)

    aquatic_survival = 1.0 - aquatic_mortality
    egg_stage_survival = aquatic_survival**time_egg
    larval_stage_survival = aquatic_survival**time_larva
    alpha_first = (
        beta * egg_stage_survival * (adult_equilibrium / 2.0)
    ) / (generation_growth - 1.0)
    larval_growth_ratio = larval_stage_survival / generation_growth
    alpha_second = (1.0 - larval_growth_ratio) / (
        1.0 - larval_growth_ratio ** (1.0 / time_larva)
    )
    alpha = alpha_first * alpha_second
    larval_equilibrium = int(round(alpha * (generation_growth - 1.0)))

    config = DeterministicConfig(
        time_egg=time_egg,
        time_larva=time_larva,
        time_pupa=time_pupa,
        beta=beta,
        adult_mortality=adult_mortality,
        aquatic_mortality=aquatic_mortality,
        alpha=alpha,
        inheritance=mendelian_inheritance(),
        mating_fitness=np.ones((3, 3), dtype=np.float64),
        female_fraction=np.full(3, 0.5, dtype=np.float64),
        adult_survival_modifier=np.ones(3, dtype=np.float64),
        female_emergence=np.ones(3, dtype=np.float64),
        male_emergence=np.ones(3, dtype=np.float64),
        fertility_modifier=np.ones(3, dtype=np.float64),
    )

    larval_daily_survival = (
        alpha / (alpha + larval_equilibrium)
    ) ** (1.0 / time_larva) * aquatic_survival
    larval_weights = larval_daily_survival ** np.arange(
        time_larva,
        dtype=np.float64,
    )
    larval_counts = (
        larval_equilibrium * larval_weights / larval_weights.sum()
    )

    aquatic_wild_type = np.zeros(total_aquatic_time, dtype=np.float64)
    larva_start = time_egg
    larva_end = larva_start + time_larva
    aquatic_wild_type[larva_start:larva_end] = larval_counts
    for age in range(time_egg - 1, -1, -1):
        aquatic_wild_type[age] = (
            aquatic_wild_type[age + 1] / aquatic_survival
        )
    aquatic_wild_type[larva_end] = (
        aquatic_wild_type[larva_end - 1] * larval_daily_survival
    )
    for age in range(larva_end + 1, total_aquatic_time):
        aquatic_wild_type[age] = (
            aquatic_wild_type[age - 1] * aquatic_survival
        )

    aquatic = np.zeros((3, total_aquatic_time), dtype=np.float64)
    aquatic[0] = aquatic_wild_type
    adult_male = np.zeros(3, dtype=np.float64)
    adult_male[0] = adult_equilibrium / 2.0
    adult_female = np.zeros((3, 3), dtype=np.float64)
    adult_female[0, 0] = adult_equilibrium / 2.0
    state = PatchState(
        aquatic=aquatic,
        adult_male=adult_male,
        adult_female=adult_female,
        unmated_female=np.zeros(3, dtype=np.float64),
    )
    return config, state, larval_equilibrium


def _float_array(value: FloatArray) -> FloatArray:
    """Return an owned, read-only float64 copy.

    Args:
        value: Array supplied by the caller.

    Returns:
        Owned, read-only float64 array.
    """
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DeterministicConfig:
    """Store MGDrivE1 parameters needed by the deterministic patch lifecycle.

    Attributes:
        time_egg: Number of daily egg cohorts.
        time_larva: Number of daily larval cohorts.
        time_pupa: Number of daily pupal cohorts.
        beta: Wild-type eggs laid per mated female per day.
        adult_mortality: Daily adult mortality probability.
        aquatic_mortality: Daily density-independent aquatic mortality.
        alpha: Larval density-dependence centering parameter.
        inheritance: Female-by-male-by-offspring inheritance probabilities.
        mating_fitness: Female-by-male mating weights.
        female_fraction: Genotype-specific female fraction at emergence.
        adult_survival_modifier: Genotype-specific adult survival multiplier.
        female_emergence: Genotype-specific female emergence success.
        male_emergence: Genotype-specific male emergence success.
        fertility_modifier: Female-genotype fertility multiplier.
    """

    time_egg: int
    time_larva: int
    time_pupa: int
    beta: float
    adult_mortality: float
    aquatic_mortality: float
    alpha: float
    inheritance: FloatArray
    mating_fitness: FloatArray
    female_fraction: FloatArray
    adult_survival_modifier: FloatArray
    female_emergence: FloatArray
    male_emergence: FloatArray
    fertility_modifier: FloatArray

    def __post_init__(self) -> None:
        """Take ownership of arrays and validate the configuration.

        Raises:
            ValueError: If a scalar or genotype axis is invalid.
        """
        stage_times = (self.time_egg, self.time_larva, self.time_pupa)
        if any(
            type(value) is not int
            or value < 1
            for value in stage_times
        ):
            raise ValueError("aquatic stage durations must be positive integers")
        scalar_bounds = (
            ("beta", self.beta, 0.0, np.inf),
            ("adult_mortality", self.adult_mortality, 0.0, 1.0),
            ("aquatic_mortality", self.aquatic_mortality, 0.0, 1.0),
            ("alpha", self.alpha, 0.0, np.inf),
        )
        for name, value, lower, upper in scalar_bounds:
            if (
                not np.isfinite(value)
                or value < lower
                or value > upper
                or (name == "alpha" and value == 0.0)
            ):
                raise ValueError(f"{name} is outside its valid range")

        for name in (
            "inheritance",
            "mating_fitness",
            "female_fraction",
            "adult_survival_modifier",
            "female_emergence",
            "male_emergence",
            "fertility_modifier",
        ):
            object.__setattr__(self, name, _float_array(getattr(self, name)))

        if self.inheritance.ndim != 3:
            raise ValueError("inheritance must have three genotype axes")
        n_genotypes = self.inheritance.shape[0]
        if self.inheritance.shape != (
            n_genotypes,
            n_genotypes,
            n_genotypes,
        ):
            raise ValueError("inheritance genotype axes must have equal length")
        expected_matrix = (n_genotypes, n_genotypes)
        expected_vector = (n_genotypes,)
        if self.mating_fitness.shape != expected_matrix:
            raise ValueError("mating_fitness must be genotype by genotype")
        for name in (
            "female_fraction",
            "adult_survival_modifier",
            "female_emergence",
            "male_emergence",
            "fertility_modifier",
        ):
            if getattr(self, name).shape != expected_vector:
                raise ValueError(f"{name} must have one value per genotype")

        arrays = (
            self.inheritance,
            self.mating_fitness,
            self.female_fraction,
            self.adult_survival_modifier,
            self.female_emergence,
            self.male_emergence,
            self.fertility_modifier,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("configuration arrays must contain finite values")
        if np.any(self.inheritance < 0.0):
            raise ValueError("inheritance probabilities must be nonnegative")
        if not np.allclose(
            self.inheritance.sum(axis=2),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("inheritance probabilities must sum to one")
        if np.any(self.mating_fitness < 0.0):
            raise ValueError("mating_fitness must be nonnegative")
        for name in (
            "female_fraction",
            "adult_survival_modifier",
            "female_emergence",
            "male_emergence",
            "fertility_modifier",
        ):
            value = getattr(self, name)
            if np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{name} must be between zero and one")

    @classmethod
    def neutral(
        cls,
        *,
        n_genotypes: int,
        time_egg: int,
        time_larva: int,
        time_pupa: int,
        beta: float,
        adult_mortality: float,
        aquatic_mortality: float,
        alpha: float,
    ) -> DeterministicConfig:
        """Build a no-fitness-cost configuration with uniform inheritance.

        Args:
            n_genotypes: Number of genotype compartments.
            time_egg: Number of daily egg cohorts.
            time_larva: Number of daily larval cohorts.
            time_pupa: Number of daily pupal cohorts.
            beta: Eggs laid per mated female per day.
            adult_mortality: Daily adult mortality probability.
            aquatic_mortality: Daily aquatic mortality probability.
            alpha: Larval density-dependence centering parameter.

        Returns:
            Neutral deterministic configuration.

        Raises:
            ValueError: If ``n_genotypes`` is not positive.
        """
        if (
            isinstance(n_genotypes, bool)
            or n_genotypes < 1
        ):
            raise ValueError("n_genotypes must be a positive integer")
        inheritance = np.full(
            (n_genotypes, n_genotypes, n_genotypes),
            1.0 / n_genotypes,
            dtype=np.float64,
        )
        return cls(
            time_egg=time_egg,
            time_larva=time_larva,
            time_pupa=time_pupa,
            beta=beta,
            adult_mortality=adult_mortality,
            aquatic_mortality=aquatic_mortality,
            alpha=alpha,
            inheritance=inheritance,
            mating_fitness=np.ones(
                (n_genotypes, n_genotypes), dtype=np.float64
            ),
            female_fraction=np.full(n_genotypes, 0.5, dtype=np.float64),
            adult_survival_modifier=np.ones(n_genotypes, dtype=np.float64),
            female_emergence=np.ones(n_genotypes, dtype=np.float64),
            male_emergence=np.ones(n_genotypes, dtype=np.float64),
            fertility_modifier=np.ones(n_genotypes, dtype=np.float64),
        )


@dataclass(frozen=True)
class PatchState:
    """Store the four population compartments of one MGDrivE1 patch.

    Attributes:
        aquatic: Genotype-by-aquatic-day population matrix.
        adult_male: Adult male count by genotype.
        adult_female: Mated females by female and mate genotype.
        unmated_female: Unmated adult females by genotype.
    """

    aquatic: FloatArray
    adult_male: FloatArray
    adult_female: FloatArray
    unmated_female: FloatArray

    def __post_init__(self) -> None:
        """Take ownership of arrays and validate all state axes.

        Raises:
            ValueError: If an axis, count, or numerical value is invalid.
        """
        object.__setattr__(self, "aquatic", _float_array(self.aquatic))
        object.__setattr__(self, "adult_male", _float_array(self.adult_male))
        object.__setattr__(
            self,
            "adult_female",
            _float_array(self.adult_female),
        )
        object.__setattr__(
            self,
            "unmated_female",
            _float_array(self.unmated_female),
        )
        if self.aquatic.ndim != 2:
            raise ValueError("aquatic must be genotype by aquatic day")
        n_genotypes = self.aquatic.shape[0]
        if self.adult_male.shape != (n_genotypes,):
            raise ValueError("adult_male must have one value per genotype")
        if self.adult_female.shape != (n_genotypes, n_genotypes):
            raise ValueError(
                "adult_female must be female genotype by mate genotype"
            )
        if self.unmated_female.shape != (n_genotypes,):
            raise ValueError(
                "unmated_female must have one value per genotype"
            )
        arrays = (
            self.aquatic,
            self.adult_male,
            self.adult_female,
            self.unmated_female,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("state arrays must contain finite values")
        if any(np.any(value < 0.0) for value in arrays):
            raise ValueError("population counts must be nonnegative")


@dataclass(frozen=True)
class DailyRelease:
    """Store releases applied during one MGDrivE1 simulation day.

    Adult releases occur after emergence and before mating. Egg releases occur
    after oviposition, matching ``oneDay_PopDynamics_Patch``.

    Attributes:
        adult_male: Released adult males by genotype.
        unmated_female: Released unmated females by genotype.
        adult_female: Released mated females by female and mate genotype.
        eggs: Released eggs by genotype.
    """

    adult_male: FloatArray
    unmated_female: FloatArray
    adult_female: FloatArray
    eggs: FloatArray

    def __post_init__(self) -> None:
        """Take ownership of release arrays and validate their axes.

        Raises:
            ValueError: If release axes, counts, or values are invalid.
        """
        for name in (
            "adult_male",
            "unmated_female",
            "adult_female",
            "eggs",
        ):
            object.__setattr__(self, name, _float_array(getattr(self, name)))
        n_genotypes = self.adult_male.size
        if self.adult_male.shape != (n_genotypes,):
            raise ValueError("adult_male release must be one-dimensional")
        if self.unmated_female.shape != (n_genotypes,):
            raise ValueError(
                "unmated_female release must have one value per genotype"
            )
        if self.adult_female.shape != (n_genotypes, n_genotypes):
            raise ValueError(
                "adult_female release must be female by mate genotype"
            )
        if self.eggs.shape != (n_genotypes,):
            raise ValueError("egg release must have one value per genotype")
        arrays = (
            self.adult_male,
            self.unmated_female,
            self.adult_female,
            self.eggs,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("release arrays must contain finite values")
        if any(np.any(value < 0.0) for value in arrays):
            raise ValueError("release counts must be nonnegative")


def step_deterministic(
    state: PatchState,
    config: DeterministicConfig,
    release: DailyRelease | None = None,
) -> PatchState:
    """Advance one patch by one day using the MGDrivE1 event order.

    Args:
        state: Population state at the start of the day.
        config: Deterministic lifecycle parameters.
        release: Optional releases scheduled for this day.

    Returns:
        New population state after one day.

    Raises:
        ValueError: If state and configuration axes are incompatible.
    """
    n_genotypes = state.adult_male.size
    if config.inheritance.shape[0] != n_genotypes:
        raise ValueError("state and config genotype axes do not match")
    expected_duration = (
        config.time_egg + config.time_larva + config.time_pupa
    )
    if state.aquatic.shape[1] != expected_duration:
        raise ValueError("state aquatic duration does not match config")
    if release is not None and release.adult_male.size != n_genotypes:
        raise ValueError("release and state genotype axes do not match")

    aquatic = state.aquatic.copy()
    adult_male = state.adult_male.copy()
    adult_female = state.adult_female.copy()
    unmated_female = state.unmated_female.copy()

    aquatic_survival = 1.0 - config.aquatic_mortality
    adult_survival = (
        (1.0 - config.adult_mortality)
        * config.adult_survival_modifier
    )

    # MGDrivE1 kills existing adults before adding today's newly emerged adults.
    adult_male *= adult_survival
    adult_female *= adult_survival[:, None]

    egg_end = config.time_egg
    larva_start = egg_end
    larva_end = egg_end + config.time_larva
    pupa_start = larva_end
    pupa_end = pupa_start + config.time_pupa

    emerging = state.aquatic[:, pupa_end - 1] * aquatic_survival
    for age in range(pupa_end - 2, pupa_start - 1, -1):
        aquatic[:, age + 1] = state.aquatic[:, age] * aquatic_survival

    larval_total = float(state.aquatic[:, larva_start:larva_end].sum())
    density_survival = (
        config.alpha / (config.alpha + larval_total)
    ) ** (1.0 / config.time_larva)
    larval_survival = density_survival * aquatic_survival
    for age in range(larva_end - 1, larva_start - 1, -1):
        aquatic[:, age + 1] = state.aquatic[:, age] * larval_survival

    for age in range(egg_end - 1, -1, -1):
        aquatic[:, age + 1] = state.aquatic[:, age] * aquatic_survival

    # MGDrivE1 applies one additional adult-survival factor during emergence.
    emerging *= 1.0 - config.adult_mortality
    adult_male += (
        emerging
        * (1.0 - config.female_fraction)
        * config.male_emergence
    )
    unmated_female += (
        emerging * config.female_fraction * config.female_emergence
    )

    if release is not None:
        adult_male += release.adult_male
        unmated_female += release.unmated_female
        adult_female += release.adult_female

    for female_genotype in range(adult_male.size):
        if unmated_female[female_genotype] <= 0.0:
            continue
        weights = (
            adult_male * config.mating_fitness[female_genotype, :]
        )
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            adult_female[female_genotype, :] += (
                unmated_female[female_genotype] * weights / weight_sum
            )
            unmated_female[female_genotype] = 0.0
        else:
            unmated_female[female_genotype] *= adult_survival[
                female_genotype
            ]

    fertile_females = (
        adult_female * (config.beta * config.fertility_modifier[:, None])
    )
    for offspring_genotype in range(adult_male.size):
        aquatic[offspring_genotype, 0] = float(
            (
                fertile_females
                * config.inheritance[:, :, offspring_genotype]
            ).sum()
        )
    if release is not None:
        aquatic[:, 0] += release.eggs

    return PatchState(
        aquatic=aquatic,
        adult_male=adult_male,
        adult_female=adult_female,
        unmated_female=unmated_female,
    )


def run_deterministic(
    state: PatchState,
    config: DeterministicConfig,
    *,
    n_days: int,
    initial_day: int = 1,
    releases: Mapping[int, DailyRelease] | None = None,
) -> PatchState:
    """Advance a deterministic patch for multiple days.

    Args:
        state: Population state at the start of the run.
        config: Deterministic lifecycle parameters.
        n_days: Number of daily transitions.
        initial_day: MGDrivE1 time represented by the input state.
        releases: Releases keyed by their absolute MGDrivE1 simulation day.

    Returns:
        Owned final patch state.

    Raises:
        ValueError: If a duration, day, or release key is invalid.
    """
    return run_deterministic_trajectory(
        state,
        config,
        n_days=n_days,
        initial_day=initial_day,
        releases=releases,
    )[-1]


def run_deterministic_trajectory(
    state: PatchState,
    config: DeterministicConfig,
    *,
    n_days: int,
    initial_day: int = 1,
    releases: Mapping[int, DailyRelease] | None = None,
) -> tuple[PatchState, ...]:
    """Return the initial state and every subsequent daily state.

    Args:
        state: Population state at the start of the run.
        config: Deterministic lifecycle parameters.
        n_days: Number of daily transitions.
        initial_day: MGDrivE1 time represented by the input state.
        releases: Releases keyed by their absolute MGDrivE1 simulation day.

    Returns:
        Tuple containing ``n_days + 1`` owned, read-only states.

    Raises:
        ValueError: If a duration, day, or release key is invalid.
    """
    if (
        type(n_days) is not int
        or n_days < 0
    ):
        raise ValueError("n_days must be a nonnegative integer")
    if type(initial_day) is not int or initial_day < 0:
        raise ValueError("initial_day must be a nonnegative integer")
    release_schedule = {} if releases is None else dict(releases)
    if any(type(day) is not int for day in release_schedule):
        raise ValueError("release days must be integers")
    result = PatchState(
        aquatic=state.aquatic,
        adult_male=state.adult_male,
        adult_female=state.adult_female,
        unmated_female=state.unmated_female,
    )
    trajectory = [result]
    for offset in range(1, n_days + 1):
        day = initial_day + offset
        result = step_deterministic(
            result,
            config,
            release=release_schedule.get(day),
        )
        trajectory.append(result)
    return tuple(trajectory)
