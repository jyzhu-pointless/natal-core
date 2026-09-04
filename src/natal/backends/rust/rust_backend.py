"""Rust backend adapter for the optional native extension ``natal._engine_rs``.

The module exposes both the original standalone aging kernels and the stateful
``RustLifecycleBackend`` for full age-structured ticks.  If the native
extension is missing, the pure-Python reference backend remains unaffected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from natal.contracts.materialize import Materialized, materialize
from natal.frontend.data import (
    DiscretePopulationState,
    ModelDraft,
    PopulationState,
)
from natal.frontend.hooks.types import ECO_PARAM_NAMES

if TYPE_CHECKING:
    from natal.contracts.params import Params
    from natal.frontend.hooks.types import HookProgram

# Memory checkpoint tuples returned by the session-level snapshot API.
# (tick, ind_flat, sperm_flat, rng_words, ecology)
AgeCheckpoint: TypeAlias = tuple[
    int, NDArray[np.float64], NDArray[np.float64], list[int], dict[str, object]
]
# (tick, ind_flat, rng_words, ecology) — discrete models have no sperm.
DiscreteCheckpoint: TypeAlias = tuple[
    int, NDArray[np.float64], list[int], dict[str, object]
]
# One audited set_param transition in log form: (tick, name, old, new).
EcoJournalEntry: TypeAlias = tuple[int, str, float, float]
# Spatial log rows carry the deme as a name prefix ("deme{i}:{name}")
# because params_log rows have no deme dimension.
_SPATIAL_ROW_PREFIX = "deme"

__all__ = [
    "RustDiscreteLifecycleBackend",
    "RustHeterogeneousSpatialLifecycleBackend",
    "RustLifecycleBackend",
    "RustSpatialLifecycleBackend",
    "ecology_columns_from_drafts",
    "genetics_variant_bank",
    "rust_backend_available",
    "rust_migrate_csr_deterministic",
    "rust_migrate_csr_stochastic",
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


# Marker prefix the Rust commit gate puts on set_param bounds failures
# (non-finite or out-of-bounds values, e.g. "K / 0" -> inf).  The Python
# flush channel raises ValueError for the same condition, so the adapters
# convert the exception type to keep error contracts backend-agnostic.
_SET_PARAM_BOUNDS_MARKER = "set_param value out of bounds"


def _unwrap_set_param_bounds_error(err: RuntimeError) -> NoReturn:
    """Re-raise a Rust set_param bounds failure as ``ValueError``.

    Args:
        err: The ``RuntimeError`` raised by a Rust session run.

    Raises:
        ValueError: When the Rust message carries the bounds marker
            (re-raised with the identical message, naming the parameter,
            tick, and offending value).
        RuntimeError: The original error for every other failure mode.
    """
    message = str(err)
    if message.startswith(_SET_PARAM_BOUNDS_MARKER):
        raise ValueError(message) from err
    raise err


_T = TypeVar("_T")


def _session_call(fn: Callable[[], _T]) -> _T:
    """Invoke a Rust session call with the bounds-error conversion applied.

    Args:
        fn: Zero-argument callable wrapping one Rust session method call.

    Returns:
        Whatever *fn* returns.
    """
    try:
        return fn()
    except RuntimeError as err:
        _unwrap_set_param_bounds_error(err)


def rust_run_age_structured_aging(
    state: PopulationState,
    config: ModelDraft,
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
    config: ModelDraft,
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

    The backend owns a Rust ``EngineSession`` built from the contract
    objects materialized out of the draft (``from_parts``).  The session
    owns its blueprint/params copies, its RNG, and the declarative CSR hook
    program; runtime value changes flow through
    :meth:`refresh_params` (directed pull, no rebuild, no RNG reset).
    State arrays stay Python-owned and are copied before each tick,
    preserving the reference lifecycle's immutable-input contract.

    Single-parameter Python callbacks are supported through the session's
    ``python_callbacks`` channel; the population bridges them at
    ``enable_rust_backend`` time.
    """

    def __init__(
        self,
        config: ModelDraft,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust lifecycle backend.

        Args:
            config: A fully built age-structured ``ModelDraft``.  It is
                materialized into the contract pair once; the session then
                owns its copies and the draft can retire.
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
        contracts: Materialized = materialize(config)
        # PyO3 #[new]: the class constructor *is* from_parts(bp, params, seed).
        self._session = _engine_rs.EngineSession(
            contracts.blueprint, contracts.params, seed
        )
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def refresh_params(self, fields: list[str], params_obj: Params) -> None:
        """Pull exactly *fields* from the contract params into the session.

        The session is not rebuilt and the RNG keeps streaming; only the
        named scalars and tensors are overwritten.

        Args:
            fields: Contract field names (sorted for determinism).
            params_obj: A ``natal.contracts.Params`` carrying the current
                values for those fields.
        """
        self._session.refresh_params(fields, params_obj)

    def apply(self, writes: dict[str, float]) -> None:
        """Batch scalar write straight into the session-owned params.

        Args:
            writes: Contract scalar field names to values.
        """
        self._session.apply(writes)

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Whole-tensor contents write straight into the session params.

        Args:
            field: Contract tensor field name.
            values: Flat row-major contents; size must match the blueprint.
        """
        self._session.tensor_write(field, values)

    def set_python_callbacks(
        self,
        first: list[Callable[..., int]],
        early: list[Callable[..., int]],
        late: list[Callable[..., int]],
    ) -> None:
        """Register Python callables fired at Rust event boundaries.

        Args:
            first: Callables invoked after the ``first`` CSR event.
            early: Callables invoked after the ``early`` CSR event.
            late: Callables invoked after the ``late`` CSR event.
        """
        self._session.set_python_callbacks(first, early, late)

    def clear_python_callbacks(self) -> None:
        """Clear all registered Python callbacks."""
        self._session.clear_python_callbacks()

    def snapshot_checkpoint(self, state: PopulationState) -> AgeCheckpoint:
        """Capture a memory checkpoint of the session-owned state.

        Args:
            state: Current population state providing the arrays and tick.

        Returns:
            ``(tick, ind_flat, sperm_flat, rng_words, ecology)``.  The RNG
            words capture the full generator state, so restoring continues
            the exact stream.
        """
        return self._session.snapshot_state(
            state.individual_count, state.sperm_storage, int(state.n_tick)
        )

    def restore_checkpoint(
        self, state: PopulationState, snapshot: AgeCheckpoint
    ) -> PopulationState:
        """Restore a checkpoint produced by :meth:`snapshot_checkpoint`.

        Args:
            state: Live population state whose arrays are overwritten.
            snapshot: The tuple returned by :meth:`snapshot_checkpoint`.

        Returns:
            A new state carrying the restored arrays and the checkpointed
            tick (states are NamedTuples, so the tick travels with a fresh
            instance).
        """
        tick, ind_flat, sperm_flat, rng_words, ecology = snapshot
        restored_tick = self._session.restore_state(
            state.individual_count,
            state.sperm_storage,
            int(tick),
            ind_flat,
            sperm_flat,
            rng_words,
            ecology,
        )
        return PopulationState(
            n_tick=int(restored_tick),
            individual_count=state.individual_count,
            sperm_storage=state.sperm_storage,
        )

    def run_tick(
        self,
        state: PopulationState,
        deme_id: int = -1,
    ) -> tuple[PopulationState, int]:
        """Run one full age-structured tick in Rust.

        The stage order is first hook → reproduction → early hook → survival →
        late hook → aging, matching ``natal.backends.reference.lifecycle.run_structured_tick``.

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
        result = _session_call(
            lambda: self._session.tick(
                ind_count, sperm_store, int(state.n_tick), int(deme_id)
            )
        )
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
        result = _session_call(
            lambda: self._session.tick(ind_count, sperm_store, int(state.n_tick), -1)
        )
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
        Parameter writes made by in-run ``Op.set_param`` hooks accumulate in
        the session audit journal — drain them with
        :meth:`drain_eco_journal` after this call to keep the population's
        ``params_log`` and draft in sync.

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

        Raises:
            ValueError: When an in-run ``Op.set_param`` value fails the Rust
                bounds gate (non-finite or outside the jsonc bounds); the
                message names the parameter, tick, and value.
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        sperm_store = np.array(state.sperm_storage, dtype=np.float64, order="C", copy=True)
        if observation_mask is not None:
            observation_mask = np.ascontiguousarray(observation_mask, dtype=np.float64)
        final_tick, history_rows, was_stopped = _session_call(
            lambda: self._session.run(
                ind_count,
                sperm_store,
                int(state.n_tick),
                int(n_steps),
                int(record_every),
                observation_mask,
            )
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

    def drain_eco_journal(self) -> list[EcoJournalEntry]:
        """Drain the session's accumulated set_param audit journal.

        Returns:
            ``(tick, name, old, new)`` rows (parameter names resolved
            through ``ECO_PARAM_NAMES``), one per committed value change
            since the previous drain, in commit order.  Values were already
            bounds-validated on the Rust side.
        """
        return [
            (int(tick), ECO_PARAM_NAMES[param_id], float(old), float(new))
            for tick, param_id, old, new in self._session.drain_eco_journal()
        ]

    def reseed(self, seed: int) -> None:
        """Reseed the Rust RNG used by stochastic sampling.

        Args:
            seed: New RNG seed.
        """
        self._session.reseed(seed)


class RustDiscreteLifecycleBackend:
    """Stateful Rust backend for discrete-generation populations.

    Supports both the standard three-stage lifecycle and the fused
    Wright-Fisher tick (``extreme_speed_mode > 0``), plus single-parameter
    Python callbacks via ``set_python_callbacks``.
    """

    def __init__(
        self,
        config: ModelDraft,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust discrete-generation backend.

        Args:
            config: A fully built discrete-normalized ``ModelDraft``.  It is
                materialized into the contract pair once; the session then
                owns its copies.
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
        contracts: Materialized = materialize(config)
        # PyO3 #[new]: the class constructor *is* from_parts(bp, params, seed).
        self._session = _engine_rs.DiscreteEngineSession(
            contracts.blueprint, contracts.params, seed
        )
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def refresh_params(self, fields: list[str], params_obj: Params) -> None:
        """Pull exactly *fields* from the contract params into the session.

        The session is not rebuilt and the RNG keeps streaming.

        Args:
            fields: Contract field names (sorted for determinism).
            params_obj: A ``natal.contracts.Params`` carrying the current
                values for those fields.
        """
        self._session.refresh_params(fields, params_obj)

    def apply(self, writes: dict[str, float]) -> None:
        """Batch scalar write straight into the session-owned params."""
        self._session.apply(writes)

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        """Whole-tensor contents write straight into the session params."""
        self._session.tensor_write(field, values)

    def set_python_callbacks(
        self,
        first: list[Callable[..., int]],
        early: list[Callable[..., int]],
        late: list[Callable[..., int]],
    ) -> None:
        """Register Python callables fired at Rust event boundaries."""
        self._session.set_python_callbacks(first, early, late)

    def clear_python_callbacks(self) -> None:
        """Clear all registered Python callbacks."""
        self._session.clear_python_callbacks()

    def snapshot_checkpoint(self, state: DiscretePopulationState) -> DiscreteCheckpoint:
        """Capture a memory checkpoint (no sperm storage in discrete models).

        Args:
            state: Current discrete population state.

        Returns:
            ``(tick, ind_flat, rng_words, ecology)``.
        """
        return self._session.snapshot_state(
            state.individual_count, int(state.n_tick)
        )

    def restore_checkpoint(
        self, state: DiscretePopulationState, snapshot: DiscreteCheckpoint
    ) -> DiscretePopulationState:
        """Restore a checkpoint produced by :meth:`snapshot_checkpoint`.

        Args:
            state: Live discrete population state whose array is overwritten.
            snapshot: The tuple returned by :meth:`snapshot_checkpoint`.

        Returns:
            A new state carrying the restored array and the checkpointed
            tick.
        """
        tick, ind_flat, rng_words, ecology = snapshot
        restored_tick = self._session.restore_state(
            state.individual_count,
            int(tick),
            ind_flat,
            rng_words,
            ecology,
        )
        return DiscretePopulationState(
            n_tick=int(restored_tick),
            individual_count=state.individual_count,
        )

    def run_tick(self, state: DiscretePopulationState) -> tuple[DiscretePopulationState, int]:
        """Run one discrete-generation or Wright-Fisher tick in Rust.

        Args:
            state: Current population state.  It is not modified.

        Returns:
            ``(next_state, result_code)``.
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        result = _session_call(
            lambda: self._session.tick(ind_count, int(state.n_tick), self._wf)
        )
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
        result = _session_call(
            lambda: self._session.tick(ind_count, int(state.n_tick), self._wf)
        )
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

        In-run ``Op.set_param`` writes accumulate in the session audit
        journal — drain them with :meth:`drain_eco_journal` after this call.

        Args:
            state: Current population state.  It is not modified.
            n_steps: Number of ticks to execute.
            record_every: Record interval in ticks.  ``0`` disables recording.
            observation_mask: Optional ``(n_groups, 2, 2, n_ztypes)`` mask.

        Returns:
            ``(next_state, history_rows, was_stopped)``.

        Raises:
            ValueError: When an in-run ``Op.set_param`` value fails the Rust
                bounds gate (non-finite or outside the jsonc bounds).
        """
        ind_count = np.array(state.individual_count, dtype=np.float64, order="C", copy=True)
        if observation_mask is not None:
            observation_mask = np.ascontiguousarray(observation_mask, dtype=np.float64)
        final_tick, history_rows, was_stopped = _session_call(
            lambda: self._session.run(
                ind_count,
                int(state.n_tick),
                int(n_steps),
                int(record_every),
                self._wf,
                observation_mask,
            )
        )
        return (
            DiscretePopulationState(n_tick=int(final_tick), individual_count=ind_count),
            history_rows,
            bool(was_stopped),
        )

    def drain_eco_journal(self) -> list[EcoJournalEntry]:
        """Drain the session's accumulated set_param audit journal.

        Returns:
            ``(tick, name, old, new)`` rows (parameter names resolved
            through ``ECO_PARAM_NAMES``), one per committed value change
            since the previous drain, in commit order.  Values were already
            bounds-validated on the Rust side.
        """
        return [
            (int(tick), ECO_PARAM_NAMES[param_id], float(old), float(new))
            for tick, param_id, old, new in self._session.drain_eco_journal()
        ]

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

    def __init__(
        self,
        config: ModelDraft,
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust spatial lifecycle backend.

        Args:
            config: Shared age-structured ``ModelDraft`` (materialized once).
            hook_program: Optional shared declarative CSR hook program.
            seed: Base seed; deme *d* uses ``seed ^ d`` for its RNG.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        contracts: Materialized = materialize(config)
        # PyO3 #[new]: the class constructor *is* from_parts(bp, params, seed).
        self._session = _engine_rs.SpatialEngineSession(
            contracts.blueprint, contracts.params, seed
        )
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    def refresh_params(self, fields: list[str], params_obj: Params) -> None:
        """Pull exactly *fields* from the contract params into the session.

        Args:
            fields: Contract field names (sorted for determinism).
            params_obj: A ``natal.contracts.Params`` carrying the current
                values for those fields.
        """
        self._session.refresh_params(fields, params_obj)

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

        Raises:
            ValueError: When an in-run ``Op.set_param`` value fails the Rust
                bounds gate (non-finite or outside the jsonc bounds).
        """
        ind = np.array(individual_count_all, dtype=np.float64, order="C", copy=True)
        sperm = np.array(sperm_storage_all, dtype=np.float64, order="C", copy=True)
        next_tick = _session_call(lambda: self._session.run(ind, sperm, int(tick)))
        return ind, sperm, int(next_tick)

    def drain_eco_journal(self) -> list[EcoJournalEntry]:
        """Drain the session's per-deme set_param audit journal.

        Spatial journal rows carry the deme as a name prefix —
        ``(tick, "deme{i}:{name}", old, new)`` — because a single
        ``params_log`` row ``(tick, name, old, new)`` has no deme
        dimension.  Split the prefix on ``":"`` to recover the deme id and
        the plain parameter name.

        Returns:
            ``(tick, "deme{i}:{name}", old, new)`` rows, one per committed
            value change since the previous drain, in deme-then-commit
            order.  Values were already bounds-validated on the Rust side.
        """
        return [
            (
                int(tick),
                f"{_SPATIAL_ROW_PREFIX}{deme}:{ECO_PARAM_NAMES[param_id]}",
                float(old),
                float(new),
            )
            for deme, tick, param_id, old, new in self._session.drain_eco_journal()
        ]

    def reseed(self, seed: int) -> None:
        """Reseed the deme RNG stream.

        Args:
            seed: New base seed.
        """
        self._session.reseed(seed)


class RustHeterogeneousSpatialLifecycleBackend:
    """Rust backend for heterogeneous spatial runs over the variant bank.

    Slice-5 stage-2 boundary: the session receives one shared blueprint,
    one columnized ecology set (per-deme ``Params`` columns), a bank of
    genetics ``TensorSet`` variants, and a per-deme variant index.  Demes
    with identical genetics share one bank entry regardless of how their
    ecology differs, so bank size scales with genetics diversity only.
    """

    def __init__(
        self,
        blueprint: object,
        ecology_columns: Mapping[str, NDArray[np.float64] | NDArray[np.int64]],
        tensor_bank: Sequence[Mapping[str, NDArray[np.float64]]],
        deme_variant_ids: NDArray[np.int64],
        hook_program: HookProgram | None = None,
        seed: int = 0,
    ) -> None:
        """Create a Rust heterogeneous spatial backend.

        Args:
            blueprint: The spatial contract ``Blueprint`` (real ``n_demes``
                and migration CSR).
            ecology_columns: Mapping of ecology field name to a flat column
                array (scalar columns length ``n_demes``, vector columns
                ``n_demes`` times their per-deme extent, ``growth_mode``
                int64).  See :func:`ecology_columns_from_drafts`.
            tensor_bank: Sequence of ``{genetics tensor name: flat array}``
                mappings.  See :func:`genetics_variant_bank`.
            deme_variant_ids: Int64 array mapping each deme to a bank index.
            hook_program: Optional shared declarative CSR hook program.
            seed: Base seed; deme *d* uses ``seed ^ d``.
        """
        try:
            from natal import _engine_rs
        except ImportError as err:
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "and re-run."
            ) from err
        # PyO3 #[new]: the class constructor *is* from_parts(...).
        # growth_mode travels as int64; every other column is float64.
        boundary_columns: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
            name: (
                np.ascontiguousarray(column, dtype=np.int64)
                if name == "growth_mode"
                else np.ascontiguousarray(column, dtype=np.float64)
            )
            for name, column in ecology_columns.items()
        }
        self._session = _engine_rs.HeterogeneousSpatialEngineSession(
            blueprint,
            boundary_columns,
            [
                {key: np.ascontiguousarray(value) for key, value in tensors.items()}
                for tensors in tensor_bank
            ],
            np.ascontiguousarray(deme_variant_ids, dtype=np.int64),
            seed,
        )
        if hook_program is not None:
            self._session.set_hook_program(hook_program)

    @property
    def n_variants(self) -> int:
        """int: Number of genetics variants in the session bank."""
        return int(self._session.n_variants())

    def fork_variant(self, deme: int) -> int:
        """Fork one deme's genetics variant into a private bank entry.

        The deme is re-pointed at the fresh clone; other demes keep
        sharing the original tables.

        Args:
            deme: Deme whose genetics diverge.

        Returns:
            The new variant id, writable via
            :meth:`refresh_variant_tensors`.
        """
        return int(self._session.fork_variant(deme))

    def refresh_deme_ecology(self, deme: int, fields: list[str], params_obj: Params) -> None:
        """Pull exactly *fields* for one deme's ecology column.

        Args:
            deme: Deme column index.
            fields: Contract field names (sorted for determinism).
            params_obj: A ``natal.contracts.Params`` carrying the current
                values.
        """
        self._session.refresh_deme_ecology(deme, fields, params_obj)

    def refresh_variant_tensors(
        self, variant_id: int, fields: list[str], params_obj: Params
    ) -> None:
        """Pull exactly *fields* for one genetics bank variant.

        Args:
            variant_id: Index into the variant bank.
            fields: Genetics tensor names (sorted for determinism).
            params_obj: A ``natal.contracts.Params`` carrying the current
                values.
        """
        self._session.refresh_variant_tensors(variant_id, fields, params_obj)

    def run(
        self,
        individual_count_all: NDArray[np.float64],
        sperm_storage_all: NDArray[np.float64],
        tick: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run one tick for all demes with per-deme ecology columns.

        Args:
            individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
            sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
            tick: Current tick.

        Returns:
            ``(individual_count_all, sperm_storage_all, next_tick)``.

        Raises:
            ValueError: When an in-run ``Op.set_param`` value fails the Rust
                bounds gate (non-finite or outside the jsonc bounds).
        """
        ind = np.array(individual_count_all, dtype=np.float64, order="C", copy=True)
        sperm = np.array(sperm_storage_all, dtype=np.float64, order="C", copy=True)
        next_tick = _session_call(lambda: self._session.run(ind, sperm, int(tick)))
        return ind, sperm, int(next_tick)

    def drain_eco_journal(self) -> list[EcoJournalEntry]:
        """Drain the session's per-deme set_param audit journal.

        Spatial journal rows carry the deme as a name prefix —
        ``(tick, "deme{i}:{name}", old, new)`` — because a single
        ``params_log`` row ``(tick, name, old, new)`` has no deme
        dimension.  Split the prefix on ``":"`` to recover the deme id and
        the plain parameter name.

        Returns:
            ``(tick, "deme{i}:{name}", old, new)`` rows, one per committed
            value change since the previous drain, in deme-then-commit
            order.  Values were already bounds-validated on the Rust side.
        """
        return [
            (
                int(tick),
                f"{_SPATIAL_ROW_PREFIX}{deme}:{ECO_PARAM_NAMES[param_id]}",
                float(old),
                float(new),
            )
            for deme, tick, param_id, old, new in self._session.drain_eco_journal()
        ]

    def reseed(self, seed: int) -> None:
        """Reseed the deme RNG stream.

        Args:
            seed: New base seed.
        """
        self._session.reseed(seed)


# Draft fields carrying the eight genetics tables (contract names differ
# only for the meiosis map; see _GENETICS_CONTRACT_FIELDS).
_GENETICS_CONTRACT_FIELDS: tuple[tuple[str, str], ...] = (
    ("viability_fitness", "viability_fitness"),
    ("fecundity_fitness", "fecundity_fitness"),
    ("sexual_selection_fitness", "sexual_selection_fitness"),
    ("zygote_viability_fitness", "zygote_viability_fitness"),
    ("offspring_tensor", "offspring_tensor"),
    ("zygotes_to_gametes_map", "meiosis_map"),
    ("female_ztype_compatibility", "female_ztype_compatibility"),
    ("male_ztype_compatibility", "male_ztype_compatibility"),
)

# Draft scalar/vector sources of the columnized ecology fields, in the
# session boundary's field-name space (mirrors contracts.materialize).
_ECOLOGY_SCALAR_SOURCES: tuple[tuple[str, str, float], ...] = (
    ("carrying_capacity", "carrying_capacity", 0.0),
    ("eggs_per_female", "eggs_per_female", 0.0),
    ("sex_ratio", "sex_ratio", 0.5),
    ("sperm_displacement_rate", "sperm_displacement_rate", 0.0),
    ("low_density_growth_rate", "low_density_growth_rate", 0.0),
)

# Draft vector fields mapped to their columnized contract names.
_ECOLOGY_VECTOR_SOURCES: tuple[tuple[str, str], ...] = (
    ("age_based_survival_rates", "survival_rates"),
    ("age_based_mating_rates", "mating_rates"),
    ("age_based_reproduction_rates", "reproduction_rates"),
    ("female_age_based_fertility", "fertility"),
    ("age_based_relative_competition_strength", "competition_weights"),
)


def ecology_columns_from_drafts(
    drafts: Sequence[ModelDraft],
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Gather per-deme ecology from drafts into columnized arrays.

    The layout is the heterogeneous Rust session's stage-2 boundary: every
    scalar becomes an ``(n_demes,)`` column (``growth_mode`` int64, the
    external-eggs sentinel ``None`` becomes ``-1.0``), and every ecology
    vector is tiled into a flat ``(n_demes, ...)`` row-major column.

    Args:
        drafts: One fully built ``ModelDraft`` per deme, in deme order.

    Returns:
        A mapping of contract field name to flat column arrays (float64,
        int64 for ``growth_mode``).  ``migration_rate`` is *not* included;
        callers append the spatial rate column themselves.
    """
    n_demes = len(drafts)
    columns: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {}
    for contract_name, draft_field, _default in _ECOLOGY_SCALAR_SOURCES:
        columns[contract_name] = np.array(
            [float(getattr(draft, draft_field)) for draft in drafts],
            dtype=np.float64,
        )
    external = np.array(
        [
            float(draft.external_expected_eggs)
            if draft.external_expected_eggs is not None
            else -1.0
            for draft in drafts
        ],
        dtype=np.float64,
    )
    columns["external_expected_eggs"] = external
    columns["growth_mode"] = np.array(
        [int(draft.juvenile_growth_mode) for draft in drafts], dtype=np.int64
    )
    for draft_field, contract_name in _ECOLOGY_VECTOR_SOURCES:
        first = np.asarray(getattr(drafts[0], draft_field), dtype=np.float64)
        stacked = np.empty((n_demes,) + first.shape, dtype=np.float64)
        for index, draft in enumerate(drafts):
            stacked[index] = getattr(draft, draft_field)
        columns[contract_name] = stacked.reshape(n_demes, -1).ravel()
    # Equilibrium declaration: keep the derive-mode sentinel empty unless
    # at least one draft declares a distribution (heterogeneous declared
    # and derived demes cannot share one column set).
    declared = [d for d in drafts if getattr(d, "equilibrium_individual_distribution", None) is not None
                and np.asarray(d.equilibrium_individual_distribution).size]
    if declared:
        first = np.asarray(declared[0].equilibrium_individual_distribution, dtype=np.float64)
        stacked = np.zeros((n_demes,) + first.shape, dtype=np.float64)
        for index, draft in enumerate(drafts):
            eq = getattr(draft, "equilibrium_individual_distribution", None)
            if eq is not None and np.asarray(eq).size:
                stacked[index] = eq
        columns["equilibrium_distribution"] = stacked.reshape(n_demes, -1).ravel()
    else:
        columns["equilibrium_distribution"] = np.zeros(0, dtype=np.float64)
    return columns


def genetics_variant_bank(
    drafts: Sequence[ModelDraft],
) -> tuple[list[dict[str, NDArray[np.float64]]], NDArray[np.int64]]:
    """Deduplicate drafts into a genetics variant bank by tensor content.

    Demes whose eight genetics tables are bitwise equal share one bank
    entry, so the bank size follows genetics diversity only — ecological
    batch differences (carrying capacity, survival, initial state) never
    split variants.

    Args:
        drafts: One fully built ``ModelDraft`` per deme, in deme order.

    Returns:
        ``(bank, deme_variant_ids)`` where *bank* holds one
        ``{contract tensor name: flat float64 copy}`` mapping per unique
        genetics content and *deme_variant_ids* maps each deme to its bank
        index.
    """
    bank: list[dict[str, NDArray[np.float64]]] = []
    by_content: dict[bytes, int] = {}
    by_identity: dict[tuple[int, ...], int] = {}
    ids = np.zeros(len(drafts), dtype=np.int64)
    for index, draft in enumerate(drafts):
        identity = tuple(id(getattr(draft, field)) for field, _ in _GENETICS_CONTRACT_FIELDS)
        cached = by_identity.get(identity)
        if cached is None:
            tensors = {
                contract: np.array(getattr(draft, field), dtype=np.float64).ravel()
                for field, contract in _GENETICS_CONTRACT_FIELDS
            }
            content = b"".join(
                np.ascontiguousarray(tensors[contract], dtype=np.float64).tobytes()
                for contract, _ in sorted(tensors.items())
            )
            cached = by_content.get(content)
            if cached is None:
                cached = len(bank)
                bank.append(tensors)
                by_content[content] = cached
            by_identity[identity] = cached
        ids[index] = cached
    return bank, ids


def rust_migrate_csr_deterministic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    stay_after: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one deterministic CSR migration step in Rust.

    Runtime migration is the ``(n_demes, n_sexes, n_ages)`` rate column
    multiplied by the frozen CSR routing table folded onto the Blueprint
    at build time.

    Args:
        individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
        sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
        indptr: CSR migration row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        rate: ``(n_demes, 2, n_ages)`` migration-rate column (flat float64).
        stay_after: Deterministic bookkeeping order (kernel mode uses
            ``True``).

    Returns:
        ``(individual_count_all, sperm_storage_all)`` after migration.
    """
    try:
        from natal import _engine_rs
    except ImportError as err:
        raise RuntimeError(
            "natal._engine_rs is not available; build it with `maturin develop` "
            "before enabling the Rust backend."
        ) from err
    ind = np.ascontiguousarray(individual_count_all, dtype=np.float64)
    sperm = np.ascontiguousarray(sperm_storage_all, dtype=np.float64)
    return _engine_rs.migrate_csr_deterministic(
        ind,
        sperm,
        np.ascontiguousarray(indptr, dtype=np.int64),
        np.ascontiguousarray(dest_idx, dtype=np.int64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(np.ravel(rate), dtype=np.float64),
        stay_after,
    )


def rust_migrate_csr_stochastic(
    individual_count_all: NDArray[np.float64],
    sperm_storage_all: NDArray[np.float64],
    indptr: NDArray[np.int64],
    dest_idx: NDArray[np.int64],
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    seed: int,
    continuous_sampling: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one stochastic CSR migration step in Rust.

    Each source deme uses its own RNG stream derived from ``seed ^ deme_id``,
    matching the per-deme RNG policy used by spatial lifecycle ticks.

    Args:
        individual_count_all: Stacked state ``(n_demes, 2, n_ages, n_z)``.
        sperm_storage_all: Stacked storage ``(n_demes, n_ages, n_z, n_z)``.
        indptr: CSR migration row pointer, length ``n_demes + 1``.
        dest_idx: CSR destination index per entry.
        weights: CSR normalized outbound weight per entry.
        rate: ``(n_demes, 2, n_ages)`` migration-rate column (flat float64).
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
            "before enabling the Rust backend."
        ) from err
    ind = np.ascontiguousarray(individual_count_all, dtype=np.float64)
    sperm = np.ascontiguousarray(sperm_storage_all, dtype=np.float64)
    return _engine_rs.migrate_csr_stochastic(
        ind,
        sperm,
        np.ascontiguousarray(indptr, dtype=np.int64),
        np.ascontiguousarray(dest_idx, dtype=np.int64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(np.ravel(rate), dtype=np.float64),
        seed,
        continuous_sampling,
    )
