"""Population state containers based on NamedTuple.

These containers keep scalar metadata immutable while allowing in-place mutation
of NumPy array contents, which remains compatible with Numba kernels.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, NamedTuple
from numpy.typing import NDArray

__all__ = [
    "PopulationState",
    "DiscretePopulationState",
    "PlainPopulationState",
    "PlainDiscretePopulationState",
    "to_plain_population_state",
    "to_plain_discrete_population_state",
    "from_plain_population_state",
    "from_plain_discrete_population_state",
    "parse_flattened_state",
]


class PopulationState(NamedTuple):
    """Age-structured state container.

    Scalars are immutable (use ``_replace`` to rebuild); array values remain
    mutable in-place.
    """

    n_tick: int
    individual_count: NDArray[np.float64]
    sperm_storage: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        n_genotypes: int,
        n_sexes: Optional[int] = None,
        n_ages: int = 2,
        n_tick: int = 0,
        individual_count: Optional[NDArray[np.float64]] = None,
        sperm_storage: Optional[NDArray[np.float64]] = None,
    ) -> "PopulationState":
        if n_sexes is None:
            n_sexes = 2
        assert n_genotypes > 0, "n_genotypes must be positive"
        assert n_ages > 0, "n_ages must be positive"
        assert n_tick >= 0, "n_tick must be non-negative"

        if individual_count is None:
            ind = np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64)
        else:
            expected_shape = (n_sexes, n_ages, n_genotypes)
            assert individual_count.shape == expected_shape, (
                f"Invalid shape for individual_count: expected {expected_shape}, got {individual_count.shape}"
            )
            ind = individual_count.astype(np.float64)

        if sperm_storage is None:
            sperm = np.zeros((n_ages, n_genotypes, n_genotypes), dtype=np.float64)
        else:
            expected_shape = (n_ages, n_genotypes, n_genotypes)
            assert sperm_storage.shape == expected_shape, (
                f"Invalid shape for sperm_storage: expected {expected_shape}, got {sperm_storage.shape}"
            )
            sperm = sperm_storage.astype(np.float64)

        return cls(n_tick=np.int32(n_tick), individual_count=ind, sperm_storage=sperm)

    def get_count(self, sex: int, age: int, genotype_index: int) -> float:
        return self.individual_count[sex, age, genotype_index]

    def add_count(self, sex: int, age: int, genotype_index: int, count: float) -> None:
        self.individual_count[sex, age, genotype_index] += count

    def set_count(self, sex: int, age: int, genotype_index: int, count: float) -> None:
        self.individual_count[sex, age, genotype_index] = count

    def get_stored_sperm(self, age: int, female_genotype_index: int, male_genotype_index: int) -> float:
        return self.sperm_storage[age, female_genotype_index, male_genotype_index]

    def set_stored_sperm(self, age: int, female_genotype_index: int, male_genotype_index: int, count: float) -> None:
        self.sperm_storage[age, female_genotype_index, male_genotype_index] += count

    def flatten_all(self) -> NDArray[np.float64]:
        tick_arr = np.array([float(self.n_tick)], dtype=np.float64)
        return np.concatenate((tick_arr, self.individual_count.flatten(), self.sperm_storage.flatten()))


class DiscretePopulationState(NamedTuple):
    """Discrete-generation state container."""

    n_tick: int
    individual_count: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        n_sexes: int,
        n_ages: int,
        n_genotypes: int,
        n_tick: int = 0,
        individual_count: Optional[NDArray[np.float64]] = None,
    ) -> "DiscretePopulationState":
        assert n_sexes > 0, "n_sexes must be positive"
        assert n_ages > 0, "n_ages must be positive"
        assert n_genotypes > 0, "n_genotypes must be positive"
        assert n_tick >= 0, "n_tick must be non-negative"

        if individual_count is None:
            ind = np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64)
        else:
            expected_shape = (n_sexes, n_ages, n_genotypes)
            assert individual_count.shape == expected_shape, (
                f"Invalid shape for individual_count: expected {expected_shape}, got {individual_count.shape}"
            )
            ind = individual_count.astype(np.float64)

        return cls(n_tick=np.int32(n_tick), individual_count=ind)

    def flatten_all(self) -> NDArray[np.float64]:
        tick_arr = np.array([float(self.n_tick)], dtype=np.float64)
        return np.concatenate((tick_arr, self.individual_count.flatten()))


# Backward-compatible aliases
PlainPopulationState = PopulationState
PlainDiscretePopulationState = DiscretePopulationState


def to_plain_population_state(state: PopulationState, copy: bool = True) -> PlainPopulationState:
    ind = state.individual_count.copy() if copy else state.individual_count
    sperm = state.sperm_storage.copy() if copy else state.sperm_storage
    return PopulationState(n_tick=int(state.n_tick), individual_count=ind, sperm_storage=sperm)


def to_plain_discrete_population_state(
    state: DiscretePopulationState,
    copy: bool = True,
) -> PlainDiscretePopulationState:
    ind = state.individual_count.copy() if copy else state.individual_count
    return DiscretePopulationState(n_tick=int(state.n_tick), individual_count=ind)


def from_plain_population_state(plain: PlainPopulationState) -> PopulationState:
    return PopulationState(
        n_tick=int(plain.n_tick),
        individual_count=plain.individual_count,
        sperm_storage=plain.sperm_storage,
    )


def from_plain_discrete_population_state(plain: PlainDiscretePopulationState) -> DiscretePopulationState:
    return DiscretePopulationState(
        n_tick=int(plain.n_tick),
        individual_count=plain.individual_count,
    )


def parse_flattened_state(
    flat_array: NDArray[np.float64],
    n_sexes: Union[int, np.integer],
    n_ages: Union[int, np.integer],
    n_genotypes: Union[int, np.integer],
    copy: bool = True,
) -> PopulationState:
    n_tick = np.int32(flat_array[0])
    end = 1 + n_sexes * n_ages * n_genotypes
    individual_count = flat_array[1:end].reshape((n_sexes, n_ages, n_genotypes))
    sperm_storage = flat_array[end:].reshape((n_ages, n_genotypes, n_genotypes))

    if copy:
        individual_count = individual_count.copy()
        sperm_storage = sperm_storage.copy()

    return PopulationState(
        n_tick=n_tick,
        individual_count=individual_count,
        sperm_storage=sperm_storage,
    )
