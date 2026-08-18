"""Rust backend adapter for the optional native extension ``natal._engine_rs``.

The module exposes both the original standalone aging kernels and the stateful
``RustLifecycleBackend`` for full age-structured ticks.  If the native
extension is missing, Numba/Python backends remain unaffected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.hooks.types import HookProgram

__all__ = [
    "RustDiscreteLifecycleBackend",
    "RustHeterogeneousSpatialLifecycleBackend",
    "RustLifecycleBackend",
    "RustSpatialLifecycleBackend",
    "rust_backend_available",
    "rust_backend_signature",
    "rust_backend_supports_custom_hooks",
    "rust_migrate_adjacency_deterministic",
    "rust_migrate_adjacency_stochastic",
    "rust_migrate_kernel_deterministic",
    "rust_migrate_kernel_stochastic",
    "rust_run_age_structured_aging",
    "rust_run_discrete_aging",
]


def rust_backend_signature(
    config: PopulationConfig | DiscretePopulationConfig,
    hook_program: HookProgram | None = None,
) -> str:
    """Return a stable signature for the Rust backend's snapshot inputs.

    The signature covers every config field and every CSR hook array.  It is
    used by populations to detect runtime ``pop.update()`` / hook changes and
    rebuild the Rust session when required.

    Args:
        config: Population configuration snapshot.
        hook_program: Optional declarative CSR hook program.

    Returns:
        A 32-character hex digest.
    """
    hasher = hashlib.sha256()
    for name in config._fields:
        value = getattr(config, name)
        hasher.update(name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = cast(NDArray[np.float64], value)
            hasher.update(str(array.shape).encode("utf-8"))
            hasher.update(str(array.dtype).encode("utf-8"))
            hasher.update(np.ascontiguousarray(array).tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))
    if hook_program is not None:
        for name in hook_program._fields:
            value = getattr(hook_program, name)
            hasher.update(("hook:" + name).encode("utf-8"))
            if isinstance(value, np.ndarray):
                array = cast(NDArray[np.float64], value)
                hasher.update(str(array.shape).encode("utf-8"))
                hasher.update(str(array.dtype).encode("utf-8"))
                hasher.update(np.ascontiguousarray(array).tobytes())
            else:
                hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def rust_backend_available() -> bool:
    """Return whether the compiled Rust extension can be imported.

    Returns:
        True when ``natal._engine_rs`` was built with maturin, False otherwise.
    """
    try:
        from natal import _engine_rs
    except ImportError:
        return False
    return _engine_rs is not None


def rust_run_age_structured_aging(
    state: PopulationState,
    config: PopulationConfig,
) -> PopulationState:
    """Run the age-structured aging stage in Rust.

    The Python-owned state is copied first, matching the reference
    ``run_aging`` semantics; the Rust kernel then mutates the copies in place
    through zero-copy NumPy views.

    Args:
        state: Current population state.
        config: Population configuration (accepted for stage-signature
            compatibility; Rust derives ``n_ages`` from the array shape).

    Returns:
        A new ``PopulationState`` with age classes advanced by one tick.

    Raises:
        RuntimeError: If the Rust extension is not built.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
    sperm_store = np.array(state.sperm_storage, dtype=np.float64, order="C", copy=True)
    _engine_rs.age_structured_aging(ind_count, sperm_store)
    return PopulationState(
        n_tick=state.n_tick,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )


def rust_run_discrete_aging(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
) -> DiscretePopulationState:
    """Run the discrete-generation aging stage in Rust.

    The Python-owned state is copied first, matching the reference
    ``run_discrete_aging`` semantics; the Rust kernel then mutates the copy in
    place through a zero-copy NumPy view.

    Args:
        state: Current discrete population state.
        config: Discrete population configuration (accepted for stage-signature
            compatibility and currently unused by both implementations).

    Returns:
        A new ``DiscretePopulationState`` with juveniles moved to the adult
        age class.

    Raises:
        RuntimeError: If the Rust extension is not built.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
    _engine_rs.discrete_aging(ind_count)
    return DiscretePopulationState(
        n_tick=state.n_tick,
        individual_count=ind_count,
    )


class RustLifecycleBackend:
    """Stateful Rust age-structured lifecycle backend.

    The backend owns a Rust ``EngineSession`` that keeps a copy of the
    configuration, its RNG, and the declarative CSR hook program.  State
    arrays stay Python-owned and are copied before each tick, preserving the
    reference lifecycle's immutable-input contract.

    Custom hook callables are deliberately unsupported: a population with
    custom hooks must use the Numba path.  ``rust_backend_supports_custom_hooks``
    always returns False and exists so callers can select a backend uniformly.
    """

    supports_custom_hooks = False

    def __init__(
        self,
        config: PopulationConfig,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust lifecycle backend.

        Args:
            config: A fully built age-structured ``PopulationConfig``.  The
                Rust side copies the numeric arrays it needs at construction
                time; later in-place config updates are intentionally not
                reflected until the backend is rebuilt.
            hook_program: Optional declarative CSR hook program.  Custom hook
                callables must not be present; they are not executable here.
            seed: Seed for the Rust RNG used by stochastic sampling.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        self._session = _engine_rs.EngineSession(config, seed)
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def run_tick(
        self,
        state: PopulationState,
        deme_id: int = -1,
    ) -> tuple[PopulationState, int]:
        """Run one full age-structured tick in Rust.

        The stage order is first hook → reproduction → early hook → survival →
        late hook → aging, matching ``natal.engine.lifecycle.run_structured_tick``.

        Args:
            state: Current population state.  It is not modified; the returned
                state contains the tick result.
            deme_id: Deme index used by CSR deme selectors.  ``-1`` is the
                panmictic default.

        Returns:
            ``(next_state, result_code)`` where result code is ``0``
            (continue) or ``1`` (a declarative stop operation triggered).
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        sperm_store = np.array(state.sperm_storage, dtype=np.float64, order="C", copy=True)
        result = self._session.tick(ind_count, sperm_store, int(state.n_tick), int(deme_id))
        if result == 0:
            next_tick = int(state.n_tick) + 1
        else:
            next_tick = int(state.n_tick)
        return (
            PopulationState(
                n_tick=next_tick,
                individual_count=ind_count,
                sperm_storage=sperm_store,
            ),
            int(result),
        )

    def run_tick_inplace(self, state: PopulationState) -> tuple[PopulationState, int]:
        """Run one tick in place, sharing the caller-owned arrays.

        Unlike :meth:`run_tick`, this method does **not** copy the state
        arrays.  The input arrays are mutated directly and the returned
        ``PopulationState`` shares them.  Use only when the caller accepts
        in-place mutation.

        Args:
            state: Current population state.

        Returns:
            ``(next_state, result_code)`` where ``next_state`` wraps the same
            arrays as *state*.

        Raises:
            ValueError: If either array is not C-contiguous float64.
        """
        ind_count = state.individual_count
        sperm_store = state.sperm_storage
        if ind_count.dtype != np.float64 or not ind_count.flags.c_contiguous:
            raise ValueError("individual_count must be C-contiguous float64")
        if sperm_store.dtype != np.float64 or not sperm_store.flags.c_contiguous:
            raise ValueError("sperm_storage must be C-contiguous float64")
        result = self._session.tick(ind_count, sperm_store, int(state.n_tick), -1)
        next_tick = int(state.n_tick) + 1 if result == 0 else int(state.n_tick)
        return (
            PopulationState(
                n_tick=next_tick,
                individual_count=ind_count,
                sperm_storage=sperm_store,
            ),
            int(result),
        )

    def run(
        self,
        state: PopulationState,
        n_steps: int,
        record_every: int = 0,
        observation_mask: NDArray[np.float64] | None = None,
    ) -> tuple[PopulationState, NDArray[np.float64], bool]:
        """Run up to ``n_steps`` ticks inside Rust with optional recording.

        This is the batch counterpart of :meth:`run_tick`.  The caller-owned
        state is copied once, all ticks execute in Rust, and flattened history
        rows (when requested) are returned without Python per-tick callbacks.

        Args:
            state: Current population state.  It is not modified.
            n_steps: Number of ticks to execute.
            record_every: Record interval in ticks.  ``0`` disables recording.
            observation_mask: Optional ``(n_groups, n_sexes, n_ages, n_ztypes)``
                observation mask; when provided, rows contain per-group sums
                over the ztype axis instead of raw state.

        Returns:
            ``(next_state, history_rows, was_stopped)``.  ``history_rows`` is
            a 2-D float64 array, possibly with zero rows when recording is
            disabled.
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        sperm_store = np.array(state.sperm_storage, dtype=np.float64, order="C", copy=True)
        if observation_mask is not None:
            observation_mask = np.ascontiguousarray(observation_mask, dtype=np.float64)
        final_tick, history_rows, was_stopped = self._session.run(
            ind_count,
            sperm_store,
            int(state.n_tick),
            int(n_steps),
            int(record_every),
            observation_mask,
        )
        return (
            PopulationState(
                n_tick=int(final_tick),
                individual_count=ind_count,
                sperm_storage=sperm_store,
            ),
            history_rows,
            bool(was_stopped),
        )

    def reseed(self, seed: int) -> None:
        """Reseed the Rust RNG used by stochastic sampling.

        Args:
            seed: New RNG seed.
        """
        self._session.reseed(seed)


def rust_backend_supports_custom_hooks() -> bool:
    """Return whether the Rust backend can execute custom hook callables.

    Returns:
        Always False for now; custom-hook populations must use the Numba path.
    """
    return False


class RustDiscreteLifecycleBackend:
    """Stateful Rust backend for discrete-generation populations.

    Supports both the standard three-stage lifecycle and the fused
    Wright-Fisher tick (``extreme_speed_mode > 0``).  Custom hook callables
    are unsupported and must use the Numba fallback.
    """

    supports_custom_hooks = False

    def __init__(
        self,
        config: DiscretePopulationConfig,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust discrete-generation backend.

        Args:
            config: A fully built ``DiscretePopulationConfig``.
            hook_program: Optional declarative CSR hook program.
            seed: Seed for the Rust RNG.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        self._wf = int(getattr(config, "extreme_speed_mode", 0)) > 0
        self._session = _engine_rs.DiscreteEngineSession(config, seed)
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def run_tick(self, state: DiscretePopulationState) -> tuple[DiscretePopulationState, int]:
        """Run one discrete-generation or Wright-Fisher tick in Rust.

        Args:
            state: Current population state.  It is not modified.

        Returns:
            ``(next_state, result_code)``.
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        result = self._session.tick(ind_count, int(state.n_tick), self._wf)
        next_tick = int(state.n_tick) + 1 if result == 0 else int(state.n_tick)
        return (
            DiscretePopulationState(n_tick=next_tick, individual_count=ind_count),
            int(result),
        )

    def run_tick_inplace(
        self, state: DiscretePopulationState
    ) -> tuple[DiscretePopulationState, int]:
        """Run one tick in place, sharing the caller-owned array.

        Unlike :meth:`run_tick`, this method does **not** copy the state
        array.  The input array is mutated directly and the returned state
        shares it.

        Args:
            state: Current discrete population state.

        Returns:
            ``(next_state, result_code)``.

        Raises:
            ValueError: If the array is not C-contiguous float64.
        """
        ind_count = state.individual_count
        if ind_count.dtype != np.float64 or not ind_count.flags.c_contiguous:
            raise ValueError("individual_count must be C-contiguous float64")
        result = self._session.tick(ind_count, int(state.n_tick), self._wf)
        next_tick = int(state.n_tick) + 1 if result == 0 else int(state.n_tick)
        return (
            DiscretePopulationState(n_tick=next_tick, individual_count=ind_count),
            int(result),
        )

    def run(
        self,
        state: DiscretePopulationState,
        n_steps: int,
        record_every: int = 0,
        observation_mask: NDArray[np.float64] | None = None,
    ) -> tuple[DiscretePopulationState, NDArray[np.float64], bool]:
        """Run up to ``n_steps`` ticks inside Rust with optional recording.

        Args:
            state: Current population state.  It is not modified.
            n_steps: Number of ticks to execute.
            record_every: Record interval in ticks.  ``0`` disables recording.
            observation_mask: Optional ``(n_groups, 2, 2, n_ztypes)`` mask.

        Returns:
            ``(next_state, history_rows, was_stopped)``.
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        if observation_mask is not None:
            observation_mask = np.ascontiguousarray(observation_mask, dtype=np.float64)
        final_tick, history_rows, was_stopped = self._session.run(
            ind_count,
            int(state.n_tick),
            int(n_steps),
            int(record_every),
            self._wf,
            observation_mask,
        )
        return (
            DiscretePopulationState(n_tick=int(final_tick), individual_count=ind_count),
            history_rows,
            bool(was_stopped),
        )

    def reseed(self, seed: int) -> None:
        """Reseed the Rust RNG.

        Args:
            seed: New RNG seed.
        """
        self._session.reseed(seed)


class RustSpatialLifecycleBackend:
    """Rust backend for homogeneous spatial multi-deme lifecycle runs.

    This P4 slice runs the age-structured lifecycle for every deme and does
    not include migration yet.  Config and hooks are shared across demes.
    """

    supports_custom_hooks = False

    def __init__(
        self,
        config: PopulationConfig,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust spatial lifecycle backend.

        Args:
            config: Shared age-structured ``PopulationConfig``.
            hook_program: Optional shared declarative CSR hook program.
            seed: Base seed; deme *d* uses ``seed + d`` for its RNG.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        self._session = _engine_rs.SpatialEngineSession(config, seed)
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run one tick for all demes.

        Args:
            individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
            sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
            tick: Current tick.

        Returns:
            ``(individual_count_all, sperm_storage_all, next_tick)``.  The
            input arrays are copied first.
        """
        ind = np.array(individual_count_all, dtype=np.float64, order="C", copy=True)
        sperm = np.array(sperm_storage_all, dtype=np.float64, order="C", copy=True)
        next_tick = self._session.run(ind, sperm, int(tick))
        return ind, sperm, int(next_tick)

    def reseed(self, seed: int) -> None:
        """Reseed the deme RNG stream.

        Args:
            seed: New base seed.
        """
        self._session.reseed(seed)


class RustHeterogeneousSpatialLifecycleBackend:
    """Rust backend for heterogeneous spatial config-bank lifecycle runs.

    Each deme selects a ``PopulationConfig`` from *config_bank* via
    *deme_config_ids*.  Migration is not included yet.
    """

    supports_custom_hooks = False

    def __init__(
        self,
        config_bank: Sequence[PopulationConfig],
        deme_config_ids: NDArray[np.int64],
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust heterogeneous spatial backend.

        Args:
            config_bank: Sequence of unique ``PopulationConfig`` objects.
            deme_config_ids: Int64 array mapping each deme to a config index.
            hook_program: Optional shared declarative CSR hook program.
            seed: Base seed; deme *d* uses ``seed + d``.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        self._session = _engine_rs.HeterogeneousSpatialEngineSession(
            list(config_bank), np.ascontiguousarray(deme_config_ids, dtype=np.int64), seed
        )
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run one tick for all demes with per-deme configs.

        Args:
            individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
            sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
            tick: Current tick.

        Returns:
            ``(individual_count_all, sperm_storage_all, next_tick)``.
        """
        ind = np.array(individual_count_all, dtype=np.float64, order="C", copy=True)
        sperm = np.array(sperm_storage_all, dtype=np.float64, order="C", copy=True)
        next_tick = self._session.run(ind, sperm, int(tick))
        return ind, sperm, int(next_tick)

    def reseed(self, seed: int) -> None:
        """Reseed the deme RNG stream.

        Args:
            seed: New base seed.
        """
        self._session.reseed(seed)


def rust_migrate_adjacency_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one deterministic dense-adjacency migration step in Rust.

    Args:
        individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
        sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
        adjacency: Dense outbound adjacency ``(n_demes, n_demes)``.
        rate: Migration probability, scalar or per-age vector.

    Returns:
        ``(individual_count_all, sperm_storage_all)`` after migration.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    ind = np.ascontiguousarray(individual_count_all, dtype=np.float64)
    sperm = np.ascontiguousarray(sperm_storage_all, dtype=np.float64)
    adjacency = np.ascontiguousarray(adjacency, dtype=np.float64)
    rate = np.ascontiguousarray(rate, dtype=np.float64)
    return _engine_rs.migrate_adjacency_deterministic(ind, sperm, adjacency, rate)


def rust_migrate_adjacency_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one stochastic dense-adjacency migration step in Rust.

    Each source deme uses its own RNG stream derived from ``seed + deme_id``,
    matching the per-deme RNG policy used by spatial lifecycle ticks.

    Args:
        individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
        sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
        adjacency: Dense outbound adjacency ``(n_demes, n_demes)``.
        rate: Migration probability, scalar or per-age vector.
        seed: RNG seed.
        continuous_sampling: Use continuous Beta/Dirichlet sampling.

    Returns:
        ``(individual_count_all, sperm_storage_all)`` after migration.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    ind = np.ascontiguousarray(individual_count_all, dtype=np.float64)
    sperm = np.ascontiguousarray(sperm_storage_all, dtype=np.float64)
    adjacency = np.ascontiguousarray(adjacency, dtype=np.float64)
    rate = np.ascontiguousarray(rate, dtype=np.float64)
    return _engine_rs.migrate_adjacency_stochastic(
        ind, sperm, adjacency, rate, seed, continuous_sampling
    )


def rust_migrate_kernel_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    migration_kernel: NDArray[np.float64],
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one deterministic topology-kernel migration step in Rust."""
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    return _engine_rs.migrate_kernel_deterministic(
        np.ascontiguousarray(individual_count_all, dtype=np.float64),
        np.ascontiguousarray(sperm_storage_all, dtype=np.float64),
        np.ascontiguousarray(migration_kernel, dtype=np.float64),
        topology_wrap,
        kernel_include_center,
        np.ascontiguousarray(rate, dtype=np.float64),
    )


def rust_migrate_kernel_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    migration_kernel: NDArray[np.float64],
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one stochastic topology-kernel migration step in Rust.

    Each source deme uses its own RNG stream derived from ``seed + deme_id``,
    matching the per-deme RNG policy used by spatial lifecycle ticks.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "and re-run."
        ) from err
    return _engine_rs.migrate_kernel_stochastic(
        np.ascontiguousarray(individual_count_all, dtype=np.float64),
        np.ascontiguousarray(sperm_storage_all, dtype=np.float64),
        np.ascontiguousarray(migration_kernel, dtype=np.float64),
        topology_wrap,
        kernel_include_center,
        np.ascontiguousarray(rate, dtype=np.float64),
        seed,
        continuous_sampling,
    )
