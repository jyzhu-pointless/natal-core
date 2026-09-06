"""Type stub for the optional native Rust extension ``natal._engine_rs``."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

def age_structured_aging(
    individual_count: NDArray[np.float64],
    sperm_storage: NDArray[np.float64],
) -> None:
    """Advance age classes in place for an age-structured deme."""


def discrete_aging(individual_count: NDArray[np.float64]) -> None:
    """Move age-0 juveniles to age-1 adults in place for a discrete-generation deme."""


def equilibrium_metrics(
    blueprint: object,
    params: object,
) -> tuple[float, float]:
    """Compute (expected_competition_strength, expected_survival_rate) from contracts."""


class EngineSession:
    """Rust-owned contract pair, RNG, and CSR hook program."""

    def __init__(self, blueprint: object, params: object, seed: int = 0) -> None: ...
    def refresh_params(self, fields: list[str], source: object) -> None: ...
    def apply(self, writes: dict[str, float]) -> None: ...
    def get_scalar(self, name: str) -> float: ...
    def get_tensor(self, name: str) -> NDArray[np.float64]: ...
    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def set_python_callbacks(
        self,
        first: list[Callable[..., int]],
        early: list[Callable[..., int]],
        late: list[Callable[..., int]],
    ) -> None: ...
    def clear_python_callbacks(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def drain_eco_journal(self) -> list[tuple[int, int, float, float]]: ...
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
    def snapshot_state(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
        tick: int,
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64], list[int], dict[str, object]]: ...
    def restore_state(
        self,
        individual_count: NDArray[np.float64],
        sperm_storage: NDArray[np.float64],
        tick: int,
        ind_flat: NDArray[np.float64],
        sperm_flat: NDArray[np.float64],
        rng_words: list[int],
        ecology: dict[str, object],
    ) -> int: ...


class DiscreteEngineSession:
    """Rust-owned discrete-generation contract pair, RNG, and CSR hooks."""

    def __init__(self, blueprint: object, params: object, seed: int = 0) -> None: ...
    def refresh_params(self, fields: list[str], source: object) -> None: ...
    def apply(self, writes: dict[str, float]) -> None: ...
    def get_scalar(self, name: str) -> float: ...
    def get_tensor(self, name: str) -> NDArray[np.float64]: ...
    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def set_python_callbacks(
        self,
        first: list[Callable[..., int]],
        early: list[Callable[..., int]],
        late: list[Callable[..., int]],
    ) -> None: ...
    def clear_python_callbacks(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def drain_eco_journal(self) -> list[tuple[int, int, float, float]]: ...
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
    def snapshot_state(
        self,
        individual_count: NDArray[np.float64],
        tick: int,
    ) -> tuple[int, NDArray[np.float64], list[int], dict[str, object]]: ...
    def restore_state(
        self,
        individual_count: NDArray[np.float64],
        tick: int,
        ind_flat: NDArray[np.float64],
        rng_words: list[int],
        ecology: dict[str, object],
    ) -> int: ...


class SpatialEngineSession:
    """Rust-owned homogeneous spatial multi-deme lifecycle session."""

    def __init__(self, blueprint: object, params: object, seed: int = 0) -> None: ...
    def refresh_params(self, fields: list[str], source: object) -> None: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def drain_eco_journal(self) -> list[tuple[int, int, int, float, float]]: ...
    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> int: ...


class HeterogeneousSpatialEngineSession:
    """Rust-owned heterogeneous spatial session over the variant bank.

    One shared blueprint, one columnized ecology set (per-deme ``Params``
    columns), a bank of shared genetics variants, and a per-deme variant
    index — no per-deme contract clones.
    """

    def __init__(
        self,
        blueprint: object,
        ecology_columns: dict[str, NDArray[np.float64] | NDArray[np.int64]],
        tensor_bank: list[dict[str, NDArray[np.float64]]],
        deme_variant_ids: NDArray[np.int64],
        seed: int = 0,
    ) -> None: ...
    def refresh_deme_ecology(self, deme: int, fields: list[str], source: object) -> None: ...
    def refresh_variant_tensors(self, variant_id: int, fields: list[str], source: object) -> None: ...
    def n_variants(self) -> int: ...
    def fork_variant(self, deme: int) -> int: ...
    def set_hook_program(self, program: object) -> None: ...
    def clear_hook_program(self) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def drain_eco_journal(self) -> list[tuple[int, int, int, float, float]]: ...
    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> int: ...


def migrate_csr_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    stay_after: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one deterministic CSR migration step."""


def migrate_csr_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stochastic CSR migration step."""


def compute_offspring_tensor(
    meiosis: NDArray[np.float64],
    fusion: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the offspring probability tensor from meiosis and fusion tables."""
