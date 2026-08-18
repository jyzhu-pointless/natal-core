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


class DiscreteEngineSession:
    """Rust-owned discrete-generation configuration, RNG, and CSR hooks."""

    def __init__(self, config: object, seed: int = 0) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def tick(
        self,
        individual_count: NDArray[np.float64],
        tick: int,
        wf: bool,
    ) -> int: ...
    def run(
        self,
        individual_count: NDArray[np.float64],
        tick: int,
        n_ticks: int,
        record_interval: int,
        wf: bool,
        observation_mask: NDArray[np.float64] | None = None,
    ) -> tuple[int, NDArray[np.float64], bool]: ...


class SpatialEngineSession:
    """Rust-owned homogeneous spatial multi-deme lifecycle session."""

    def __init__(self, config: object, seed: int = 0) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> int: ...


class HeterogeneousSpatialEngineSession:
    """Rust-owned heterogeneous spatial config-bank lifecycle session."""

    def __init__(
        self,
        config_bank: list[object],
        deme_config_ids: NDArray[np.int64],
        seed: int = 0,
    ) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> int: ...


def migrate_adjacency_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one deterministic dense-adjacency migration step."""


def migrate_adjacency_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stochastic dense-adjacency migration step."""


def migrate_kernel_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    migration_kernel: NDArray[np.float64],
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one deterministic topology-kernel migration step."""


def migrate_kernel_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    migration_kernel: NDArray[np.float64],
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stochastic topology-kernel migration step."""
