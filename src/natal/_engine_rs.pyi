"""Type stub for the optional native Rust extension ``natal._engine_rs``."""

import numpy as np
from numpy.typing import NDArray

def age_structured_aging(
    individual_count: NDArray[np.float64],
    sperm_storage: NDArray[np.float64],
) -> None:
    """Advance age classes in place for an age-structured deme."""


def discrete_aging(individual_count: NDArray[np.float64]) -> None:
    """Move age-0 juveniles to age-1 adults in place for a discrete-generation deme."""


class EngineSession:
    """Rust-owned simulation configuration, RNG, and CSR hook program."""

    def __init__(self, config: object, seed: int = 0) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def reproduction(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
    ) -> None: ...
    def survival(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
    ) -> None: ...
    def aging(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
    ) -> None: ...
    def tick(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
        tick: int,
        deme_id: int,
    ) -> int: ...
    def run(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
        tick: int,
        n_ticks: int,
        record_interval: int,
        observation_mask: NDArray[np.float64] | None = None,
    ) -> tuple[int, NDArray[np.float64], bool]: ...
