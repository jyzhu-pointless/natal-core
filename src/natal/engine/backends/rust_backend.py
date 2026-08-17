"""Rust backend adapter for the optional native extension ``natal._engine_rs``.

The module exposes both the original standalone aging kernels and the stateful
``RustLifecycleBackend`` for full age-structured ticks.  If the native
extension is missing, Numba/Python backends remain unaffected.
"""

from __future__ import annotations

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
    "RustLifecycleBackend",
    "rust_backend_available",
    "rust_backend_supports_custom_hooks",
    "rust_run_age_structured_aging",
    "rust_run_discrete_aging",
]


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
