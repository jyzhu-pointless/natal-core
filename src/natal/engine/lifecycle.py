"""Unified lifecycle tick orchestration and codegen assembler.

This module is the single source of truth for the simulation tick order:

    first hook -> reproduction -> early hook -> survival -> late hook
    -> aging

and for the Wright-Fisher fused tick (first hook only).

The functions below are intentionally plain Python and are not decorated.
The Numba-enabled path embeds their source via ``inspect.getsource`` and
assembles a generated module; the Numba-disabled path calls the same
functions directly.  There are no lifecycle template files anymore.
"""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.engine.age_structured_simulator import (
    run_aging,
    run_reproduction,
    run_survival,
)
from natal.engine.discrete_generation_simulator import (
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
)
from natal.engine.simulation.discrete_generation import run_wf_tick as _run_wf_wide
from natal.engine.spatial_migrator import run_spatial_migration
from natal.hooks.runtime.csr_kernel import execute_csr_event_program_with_state
from natal.hooks.types import (
    EVENT_EARLY,
    EVENT_FIRST,
    EVENT_LATE,
    RESULT_CONTINUE,
    RESULT_STOP,
    HookProgram,
)

if TYPE_CHECKING:
    from natal.spatial.topology import (
        HeterogeneousKernelParams,
        MigrationParams,
        SpatialTopology,
    )

try:
    from numba import (  # pyright: ignore[reportMissingTypeStubs] — numba ships no type stubs
        prange,
    )
except ImportError:
    prange = range  # type: ignore[assignment] — prange fallback keeps the Python path importable without numba

__all__ = [
    "assemble_lifecycle_module",
    "run",
    "run_discrete_tick",
    "run_structured_tick",
    "run_wf_tick",
]

_StateT = TypeVar("_StateT", PopulationState, DiscretePopulationState)
_LifecycleState = PopulationState | DiscretePopulationState
_LifecycleConfig = PopulationConfig | DiscretePopulationConfig
_LifecycleHook = Callable[..., int]


def _run_event(
    event_id: int,
    state: _LifecycleState,
    config: _LifecycleConfig,
    registry: HookProgram,
    event_hook: _LifecycleHook,
    deme_id: int,
    has_sperm_storage: bool,
    sperm_store: Optional[NDArray[np.float64]],
) -> int:
    """Execute one event: declarative CSR ops, then the combined hook.

    Args:
        event_id: Numeric event id (EVENT_FIRST, EVENT_EARLY, EVENT_LATE).
        state: Current population state.  Its arrays are mutated in-place by
            CSR declarative operations and by the combined hook.
        config: Current population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        event_hook: Combined hook with signature
            ``(state, config, deme_id) -> int``.
        deme_id: Deme index.  ``-1`` is the panmictic default.
        has_sperm_storage: Whether *sperm_store* contains real data.  When
            False, *sperm_store* must be ``None`` (no dummy array).
        sperm_store: Sperm-storage array or ``None`` for discrete models.

    Returns:
        ``RESULT_CONTINUE`` or ``RESULT_STOP``.
    """
    result = execute_csr_event_program_with_state(
        registry,
        event_id,
        state.individual_count,
        sperm_store,
        state.n_tick,
        bool(config.stochastic),
        has_sperm_storage,
        bool(config.continuous_sampling),
        deme_id,
    )
    if result != RESULT_CONTINUE:
        return RESULT_STOP
    result = event_hook(state, config, deme_id)
    return RESULT_STOP if result != 0 else RESULT_CONTINUE


def run_structured_tick(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = -1,
) -> tuple[PopulationState, int]:
    """Execute one age-structured tick with hooks at each lifecycle stage.

    Args:
        state: Current population state with sperm storage.
        config: Population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``-1`` is the panmictic default.

    Returns:
        ``(next_state, result_code)``.
    """
    tick = state.n_tick
    current = PopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    result = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, True,
        current.sperm_storage,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_reproduction(current, config)

    result = _run_event(
        EVENT_EARLY, current, config, registry, early_hook, deme_id, True,
        current.sperm_storage,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_survival(current, config)

    result = _run_event(
        EVENT_LATE, current, config, registry, late_hook, deme_id, True,
        current.sperm_storage,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_aging(current, config)
    return PopulationState(
        n_tick=tick + 1,
        individual_count=current.individual_count,
        sperm_storage=current.sperm_storage,
    ), RESULT_CONTINUE


def run_discrete_tick(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = -1,
) -> tuple[DiscretePopulationState, int]:
    """Execute one discrete-generation tick with hooks at each stage.

    Args:
        state: Current discrete population state.
        config: Discrete population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``-1`` is the panmictic default.

    Returns:
        ``(next_state, result_code)``.
    """
    tick = state.n_tick
    current = DiscretePopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
    )

    result = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, False, None,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_discrete_reproduction(current, config)

    result = _run_event(
        EVENT_EARLY, current, config, registry, early_hook, deme_id, False, None,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_discrete_survival(current, config)

    result = _run_event(
        EVENT_LATE, current, config, registry, late_hook, deme_id, False, None,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    current = run_discrete_aging(current, config)
    return DiscretePopulationState(
        n_tick=tick + 1,
        individual_count=current.individual_count,
    ), RESULT_CONTINUE


def run_wf_tick(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = -1,
) -> tuple[DiscretePopulationState, int]:
    """Execute one Wright-Fisher fused tick.

    Only the ``first`` hook runs; ``early_hook`` and ``late_hook`` are
    accepted for signature uniformity but deliberately unused because the
    Wright-Fisher tick has no intermediate lifecycle stages.

    Args:
        state: Current discrete population state.
        config: Discrete population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Accepted for signature uniformity; unused.
        late_hook: Accepted for signature uniformity; unused.
        deme_id: Deme index.  ``-1`` is the panmictic default.

    Returns:
        ``(next_state, result_code)``.
    """
    tick = state.n_tick
    current = DiscretePopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
    )

    result = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, False, None,
    )
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP

    new_ind = _run_wf_wide(
        ind_count=current.individual_count,
        offspring_tensor=config.offspring_tensor,
        fecundity_f=config.fecundity_f,
        fecundity_m=config.fecundity_m,
        sexual_selection=config.sexual_selection_fitness,
        viability_f=config.viability_f,
        viability_m=config.viability_m,
        eggs_per_female=float(config.eggs_per_female[()]),
        sex_ratio=float(config.sex_ratio[()]),
        female_compat=config.female_ztype_compatibility,
        male_compat=config.male_ztype_compatibility,
        female_only=config.female_only_by_sex_chrom,
        male_only=config.male_only_by_sex_chrom,
        has_sex_chromosomes=config.has_sex_chromosomes,
        mode=int(config.extreme_speed_mode),
        stochastic=bool(config.stochastic),
        mating_rate_f=config.female_adult_mating_rate,
        mating_rate_m=config.male_adult_mating_rate,
        reproduction_rate=config.reproduction_rate,
        carrying_capacity=float(config.carrying_capacity[()]),
        juvenile_growth_mode=int(config.juvenile_growth_mode[()]),
        low_density_growth_rate=float(config.low_density_growth_rate[()]),
        expected_competition_strength=float(config.expected_competition_strength[()]),
        expected_survival_rate=float(config.expected_survival_rate[()]),
    )
    return DiscretePopulationState(
        n_tick=tick + 1,
        individual_count=new_ind,
    ), RESULT_CONTINUE


def run(
    tick_fn: Callable[..., tuple[_StateT, int]],
    state: _StateT,
    config: _LifecycleConfig,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int,
    n_steps: int,
    record_every: int,
    record_fn: Callable[[_StateT], None],
) -> tuple[_StateT, bool]:
    """Run *n_steps* lifecycle ticks in pure Python.

    This is the Numba-disabled loop.  Recording uses the supplied Python
    callback so the population layer can append snapshots through the normal
    History path.

    Args:
        tick_fn: Single-tick function with signature
            ``(state, config, registry, first_hook, early_hook, late_hook,
            deme_id) -> (state, result)``.
        state: Initial population state.
        config: Population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``-1`` is the panmictic default.
        n_steps: Number of ticks to execute.
        record_every: Record after ticks satisfying
            ``tick % record_every == 0``.  ``0`` disables recording.
        record_fn: Callback receiving the completed state after each
            recorded tick.

    Returns:
        ``(final_state, was_stopped)``.
    """
    current = state
    for _ in range(n_steps):
        current, result = tick_fn(
            current, config, registry, first_hook, early_hook, late_hook, deme_id,
        )
        if result != RESULT_CONTINUE:
            return current, True
        if record_every > 0 and (current.n_tick % record_every == 0):
            record_fn(current)
    return current, False


def _run_loop_structured(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    record_interval: int = 0,
    observation_mask: Optional[NDArray[np.float64]] = None,
    n_obs_groups: int = 0,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], int],
    Optional[NDArray[np.float64]],
    bool,
]:
    """Run multiple age-structured ticks with in-kernel array recording.

    Args:
        state: Initial population state.
        config: Population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        n_ticks: Number of ticks to execute.
        record_interval: Recording interval.  ``0`` disables recording.
        observation_mask: Optional 4-D mask ``(n_groups, n_sexes, n_ages,
            n_ztypes)``.  When set, observation rows are recorded instead
            of raw state rows.
        n_obs_groups: Number of observation groups in *observation_mask*.

    Returns:
        ``((individual_count, sperm_storage, tick), history, was_stopped)``.
    """
    was_stopped = False
    ind_count = state.individual_count.copy()
    sperm_store = state.sperm_storage.copy()
    tick = state.n_tick

    if observation_mask is not None:
        n_sexes_ = ind_count.shape[0]
        n_ages_ = ind_count.shape[1]
        flatten_size = 1 + n_obs_groups * n_sexes_ * n_ages_
    else:
        ind_size = ind_count.size
        sperm_size = sperm_store.size
        flatten_size = 1 + ind_size + sperm_size

    if record_interval > 0:
        estimated_size = (n_ticks // record_interval) + 2
        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
    history_count = 0

    if record_interval > 0 and (tick % record_interval == 0):
        flat_state = np.zeros(flatten_size, dtype=np.float64)
        flat_state[0] = tick
        if observation_mask is not None:
            observed = np.sum(observation_mask * ind_count[None, :, :, :], axis=-1)
            flat_state[1:] = observed.flatten()
        else:
            flat_state[1:1 + ind_count.size] = ind_count.flatten()
            flat_state[1 + ind_count.size:] = sperm_store.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_ticks):
        next_state, result = _TICK_FN_STRUCTURED(
            PopulationState(
                n_tick=tick, individual_count=ind_count, sperm_storage=sperm_store,
            ),
            config,
            registry,
        )
        ind_count = next_state.individual_count
        sperm_store = next_state.sperm_storage
        tick = next_state.n_tick

        if (
            result == RESULT_CONTINUE
            and record_interval > 0
            and (tick % record_interval == 0)
        ):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick
            if observation_mask is not None:
                observed = np.sum(observation_mask * ind_count[None, :, :, :], axis=-1)
                flat_state[1:] = observed.flatten()
            else:
                flat_state[1:1 + ind_count.size] = ind_count.flatten()
                flat_state[1 + ind_count.size:] = sperm_store.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

        if result != RESULT_CONTINUE:
            was_stopped = True
            break

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind_count, sperm_store, tick), history_result, was_stopped


def _run_loop_discrete(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    record_interval: int = 0,
    observation_mask: Optional[NDArray[np.float64]] = None,
    n_obs_groups: int = 0,
) -> tuple[
    tuple[NDArray[np.float64], int],
    Optional[NDArray[np.float64]],
    bool,
]:
    """Run multiple discrete or Wright-Fisher ticks with in-kernel recording.

    Args:
        state: Initial discrete population state.
        config: Discrete population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        n_ticks: Number of ticks to execute.
        record_interval: Recording interval.  ``0`` disables recording.
        observation_mask: Optional 4-D mask ``(n_groups, n_sexes, n_ages,
            n_ztypes)``.  When set, observation rows are recorded instead
            of raw state rows.
        n_obs_groups: Number of observation groups in *observation_mask*.

    Returns:
        ``((individual_count, tick), history, was_stopped)``.
    """
    was_stopped = False
    ind_count = state.individual_count.copy()
    tick = state.n_tick

    if observation_mask is not None:
        n_sexes_ = ind_count.shape[0]
        n_ages_ = ind_count.shape[1]
        flatten_size = 1 + n_obs_groups * n_sexes_ * n_ages_
    else:
        flatten_size = 1 + ind_count.size

    if record_interval > 0:
        estimated_size = (n_ticks // record_interval) + 2
        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
    history_count = 0

    if record_interval > 0 and (tick % record_interval == 0):
        flat_state = np.zeros(flatten_size, dtype=np.float64)
        flat_state[0] = tick
        if observation_mask is not None:
            observed = np.sum(observation_mask * ind_count[None, :, :, :], axis=-1)
            flat_state[1:] = observed.flatten()
        else:
            flat_state[1:] = ind_count.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_ticks):
        next_state, result = _TICK_FN_DISCRETE(
            DiscretePopulationState(n_tick=tick, individual_count=ind_count),
            config,
            registry,
        )
        ind_count = next_state.individual_count
        tick = next_state.n_tick

        if (
            result == RESULT_CONTINUE
            and record_interval > 0
            and (tick % record_interval == 0)
        ):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick
            if observation_mask is not None:
                observed = np.sum(observation_mask * ind_count[None, :, :, :], axis=-1)
                flat_state[1:] = observed.flatten()
            else:
                flat_state[1:] = ind_count.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

        if result != RESULT_CONTINUE:
            was_stopped = True
            break

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind_count, tick), history_result, was_stopped


def _spatial_tick_shell_structured(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[PopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Run one spatial structured tick: per-deme lifecycle, then migration.

    Args:
        ind_count_all: Stacked individual counts ``(n_demes, n_sexes,
            n_ages, n_ztypes)``.
        sperm_store_all: Stacked sperm-storage arrays.
        config_bank: Numba-typed list of unique ``PopulationConfig`` values.
        deme_config_ids: ``(n_demes,)`` int64 config ids into *config_bank*.
        registry: ``HookProgram`` with CSR declarative operations.
        tick: Current tick (shared by all demes).
        spatial_topo: Spatial topology with rows, cols, and wrap flags.
        migration: Migration parameters.
        het_kernel: Optional heterogeneous kernel parameters.  ``None``
            means all demes share one kernel.

    Returns:
        ``(ind_count_all, sperm_store_all, tick + 1, was_stopped)``.
    """
    n_demes = ind_count_all.shape[0]
    stopped = np.zeros(n_demes, dtype=np.bool_)
    for d in prange(n_demes):
        cfg = config_bank[int(deme_config_ids[d])]
        ind = ind_count_all[d].copy()
        sperm = sperm_store_all[d].copy()

        next_state, result = _DEME_TICK_STRUCTURED(
            PopulationState(n_tick=tick, individual_count=ind, sperm_storage=sperm),
            cfg, registry, d,
        )
        ind = next_state.individual_count
        sperm = next_state.sperm_storage

        if result != RESULT_CONTINUE:
            stopped[d] = True

        ind_count_all[d] = ind
        sperm_store_all[d] = sperm

    ind_count_all, sperm_store_all = run_spatial_migration(
        ind_count_all, sperm_store_all, migration.adjacency, migration.mode_code,
        spatial_topo.rows, spatial_topo.cols, spatial_topo.wrap,
        migration.kernel, migration.include_center,
        config_bank[0], migration.rate, migration.adjust_on_edge,
        (het_kernel.deme_kernel_ids if het_kernel is not None else None),
        (het_kernel.d_row if het_kernel is not None else None),
        (het_kernel.d_col if het_kernel is not None else None),
        (het_kernel.weights if het_kernel is not None else None),
        (het_kernel.nnzs if het_kernel is not None else None),
        (het_kernel.total_sums if het_kernel is not None else None),
        (het_kernel.max_nnz if het_kernel is not None else 0),
    )

    was_stopped = False
    for i in range(n_demes):
        if stopped[i]:
            was_stopped = True
            break
    return ind_count_all, sperm_store_all, tick + 1, was_stopped


def _spatial_run_shell_structured(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[PopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    n_steps: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
    record_interval: int = 0,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], int],
    Optional[NDArray[np.float64]],
    bool,
]:
    """Run multiple spatial structured ticks with optional recording.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked sperm-storage arrays.
        config_bank: Numba-typed list of unique ``PopulationConfig`` values.
        deme_config_ids: ``(n_demes,)`` int64 config ids into *config_bank*.
        registry: ``HookProgram`` with CSR declarative operations.
        tick: Current tick.
        n_steps: Number of ticks to execute.
        spatial_topo: Spatial topology.
        migration: Migration parameters.
        het_kernel: Optional heterogeneous kernel parameters.
        record_interval: Recording interval.  ``0`` disables recording.

    Returns:
        ``((ind_count_all, sperm_store_all, tick), history, was_stopped)``.
    """
    was_stopped = False
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    tick_cur = tick

    flatten_size = 1 + ind.size + sperm.size

    if record_interval > 0:
        estimated_size = (n_steps // record_interval) + 2
        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
    history_count = 0

    if record_interval > 0 and (tick_cur % record_interval == 0):
        flat_state = np.zeros(flatten_size, dtype=np.float64)
        flat_state[0] = tick_cur
        flat_state[1:1 + ind.size] = ind.flatten()
        flat_state[1 + ind.size:] = sperm.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_steps):
        ind, sperm, tick_cur, step_stopped = _SPATIAL_TICK_STRUCTURED(
            ind, sperm, config_bank, deme_config_ids, registry, tick_cur,
            spatial_topo, migration, het_kernel,
        )
        if step_stopped:
            was_stopped = True
            break

        if record_interval > 0 and (tick_cur % record_interval == 0):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick_cur
            flat_state[1:1 + ind.size] = ind.flatten()
            flat_state[1 + ind.size:] = sperm.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind, sperm, tick_cur), history_result, was_stopped


def _spatial_tick_shell_discrete(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[DiscretePopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Run one spatial discrete tick: per-deme lifecycle, then migration.

    Args:
        ind_count_all: Stacked individual counts ``(n_demes, n_sexes,
            n_ages, n_ztypes)``.
        sperm_store_all: Stacked zero sperm-storage arrays kept for the
            shared migration signature.
        config_bank: Numba-typed list of unique
            ``DiscretePopulationConfig`` values.
        deme_config_ids: ``(n_demes,)`` int64 config ids into *config_bank*.
        registry: ``HookProgram`` with CSR declarative operations.
        tick: Current tick (shared by all demes).
        spatial_topo: Spatial topology with rows, cols, and wrap flags.
        migration: Migration parameters.
        het_kernel: Optional heterogeneous kernel parameters.

    Returns:
        ``(ind_count_all, sperm_store_all, tick + 1, was_stopped)``.
    """
    n_demes = ind_count_all.shape[0]
    stopped = np.zeros(n_demes, dtype=np.bool_)
    for d in prange(n_demes):
        cfg = config_bank[int(deme_config_ids[d])]
        ind = ind_count_all[d].copy()

        next_state, result = _DEME_TICK_DISCRETE(
            DiscretePopulationState(n_tick=tick, individual_count=ind),
            cfg, registry, d,
        )
        ind = next_state.individual_count

        if result != RESULT_CONTINUE:
            stopped[d] = True

        ind_count_all[d] = ind

    ind_count_all, sperm_store_all = run_spatial_migration(
        ind_count_all, sperm_store_all, migration.adjacency, migration.mode_code,
        spatial_topo.rows, spatial_topo.cols, spatial_topo.wrap,
        migration.kernel, migration.include_center,
        config_bank[0], migration.rate, migration.adjust_on_edge,
        (het_kernel.deme_kernel_ids if het_kernel is not None else None),
        (het_kernel.d_row if het_kernel is not None else None),
        (het_kernel.d_col if het_kernel is not None else None),
        (het_kernel.weights if het_kernel is not None else None),
        (het_kernel.nnzs if het_kernel is not None else None),
        (het_kernel.total_sums if het_kernel is not None else None),
        (het_kernel.max_nnz if het_kernel is not None else 0),
    )

    was_stopped = False
    for i in range(n_demes):
        if stopped[i]:
            was_stopped = True
            break
    return ind_count_all, sperm_store_all, tick + 1, was_stopped


def _spatial_run_shell_discrete(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[DiscretePopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    n_steps: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
    record_interval: int = 0,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], int],
    Optional[NDArray[np.float64]],
    bool,
]:
    """Run multiple spatial discrete ticks with optional recording.

    Args:
        ind_count_all: Stacked individual counts.
        sperm_store_all: Stacked zero sperm-storage arrays.
        config_bank: Numba-typed list of unique
            ``DiscretePopulationConfig`` values.
        deme_config_ids: ``(n_demes,)`` int64 config ids into *config_bank*.
        registry: ``HookProgram`` with CSR declarative operations.
        tick: Current tick.
        n_steps: Number of ticks to execute.
        spatial_topo: Spatial topology.
        migration: Migration parameters.
        het_kernel: Optional heterogeneous kernel parameters.
        record_interval: Recording interval.  ``0`` disables recording.

    Returns:
        ``((ind_count_all, sperm_store_all, tick), history, was_stopped)``.
    """
    was_stopped = False
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    tick_cur = tick

    flatten_size = 1 + ind.size + sperm.size

    if record_interval > 0:
        estimated_size = (n_steps // record_interval) + 2
        history_array = np.zeros((estimated_size, flatten_size), dtype=np.float64)
    else:
        history_array = np.zeros((0, flatten_size), dtype=np.float64)
    history_count = 0

    if record_interval > 0 and (tick_cur % record_interval == 0):
        flat_state = np.zeros(flatten_size, dtype=np.float64)
        flat_state[0] = tick_cur
        flat_state[1:1 + ind.size] = ind.flatten()
        flat_state[1 + ind.size:] = sperm.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_steps):
        ind, sperm, tick_cur, step_stopped = _SPATIAL_TICK_DISCRETE(
            ind, sperm, config_bank, deme_config_ids, registry, tick_cur,
            spatial_topo, migration, het_kernel,
        )
        if step_stopped:
            was_stopped = True
            break

        if record_interval > 0 and (tick_cur % record_interval == 0):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick_cur
            flat_state[1:1 + ind.size] = ind.flatten()
            flat_state[1 + ind.size:] = sperm.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind, sperm, tick_cur), history_result, was_stopped


def _TICK_FN_STRUCTURED(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[PopulationState, int]:
    """Source sentinel replaced by the generated structured tick function.

    Args:
        state: Population state passed to the generated tick.
        config: Population configuration passed to the generated tick.
        registry: ``HookProgram`` passed to the generated tick.
        deme_id: Deme id passed to the generated tick.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_TICK_FN_STRUCTURED is only valid in generated modules")


def _TICK_FN_DISCRETE(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[DiscretePopulationState, int]:
    """Source sentinel replaced by the generated discrete tick function.

    Args:
        state: Discrete population state passed to the generated tick.
        config: Discrete population configuration passed to the tick.
        registry: ``HookProgram`` passed to the generated tick.
        deme_id: Deme id passed to the generated tick.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_TICK_FN_DISCRETE is only valid in generated modules")


def _DEME_TICK_STRUCTURED(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[PopulationState, int]:
    """Source sentinel replaced by the generated panmictic tick import.

    Args:
        state: Population state passed to the panmictic tick.
        config: Population configuration passed to the panmictic tick.
        registry: ``HookProgram`` passed to the panmictic tick.
        deme_id: Deme id passed to the panmictic tick.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_DEME_TICK_STRUCTURED is only valid in generated modules")


def _DEME_TICK_DISCRETE(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[DiscretePopulationState, int]:
    """Source sentinel replaced by the generated panmictic tick import.

    Args:
        state: Discrete population state passed to the panmictic tick.
        config: Discrete population configuration passed to the tick.
        registry: ``HookProgram`` passed to the panmictic tick.
        deme_id: Deme id passed to the panmictic tick.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_DEME_TICK_DISCRETE is only valid in generated modules")


def _SPATIAL_TICK_STRUCTURED(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[PopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Source sentinel replaced by the generated spatial tick function.

    Args:
        ind_count_all: Stacked individual counts passed to the spatial tick.
        sperm_store_all: Stacked sperm arrays passed to the spatial tick.
        config_bank: Numba-typed config list passed to the spatial tick.
        deme_config_ids: Per-deme config ids passed to the spatial tick.
        registry: ``HookProgram`` passed to the spatial tick.
        tick: Current tick passed to the spatial tick.
        spatial_topo: Spatial topology passed to the spatial tick.
        migration: Migration parameters passed to the spatial tick.
        het_kernel: Optional heterogeneous kernel parameters.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_SPATIAL_TICK_STRUCTURED is only valid in generated modules")


def _SPATIAL_TICK_DISCRETE(  # pyright: ignore[reportUnusedFunction] — embedded via inspect.getsource
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config_bank: list[DiscretePopulationConfig],
    deme_config_ids: NDArray[np.int64],
    registry: HookProgram,
    tick: int,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Source sentinel replaced by the generated spatial tick function.

    Args:
        ind_count_all: Stacked individual counts passed to the spatial tick.
        sperm_store_all: Stacked zero sperm arrays passed to the tick.
        config_bank: Numba-typed discrete config list passed to the tick.
        deme_config_ids: Per-deme config ids passed to the spatial tick.
        registry: ``HookProgram`` passed to the spatial tick.
        tick: Current tick passed to the spatial tick.
        spatial_topo: Spatial topology passed to the spatial tick.
        migration: Migration parameters passed to the spatial tick.
        het_kernel: Optional heterogeneous kernel parameters.

    Returns:
        Never returns; this placeholder is replaced before code generation.

    Raises:
        NotImplementedError: If called directly outside a generated module.
    """
    raise NotImplementedError("_SPATIAL_TICK_DISCRETE is only valid in generated modules")


# ---------------------------------------------------------------------------
# Codegen assembler
# ---------------------------------------------------------------------------


def _replace_identifier_outside_docstring(source: str, old: str, new: str) -> str:
    """Replace identifier tokens outside the first docstring in *source*.

    Lifecycle source functions are "generation friendly": parameter names
    may only be rewritten in executable code.  Docstring text is left
    untouched so generated help text is not mangled by hook-name
    substitution.

    Args:
        source: Source text of one top-level function.
        old: Identifier token to replace.
        new: Replacement identifier token.

    Returns:
        Source text with replacements applied outside the docstring.
    """
    lines = source.splitlines()
    triple_indices = [i for i, line in enumerate(lines) if '"""' in line]
    if len(triple_indices) >= 2:
        start, end = triple_indices[0], triple_indices[1]
    else:
        start, end = -1, -1
    for i, line in enumerate(lines):
        if i == start or i == end or (start < i < end):
            continue
        lines[i] = re.sub(
            rf"\b{re.escape(old)}\b", new, line,
        )
    return "\n".join(lines)


def _replace_function_signature(source: str, params: str, new_name: str) -> str:
    """Replace the first top-level ``def`` signature in *source*.

    Args:
        source: Source text of one top-level function.
        params: The replacement parameter list (after the function name).
        new_name: The replacement function name.

    Returns:
        Source text with the ``def`` line replaced.

    Raises:
        ValueError: If no top-level ``def`` is found.
    """
    match = re.search(
        r"^def [A-Za-z_]\w*\([^)]*\)(?:\s*->\s*[^:]+)?:",
        source,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("Expected a top-level function definition in source")
    return source[: match.start()] + f"def {new_name}({params}):" + source[match.end() :]


def _rename_function(source: str, new_name: str) -> str:
    """Rename the first top-level ``def`` while preserving its signature.

    Spatial shells keep their full kernel signature; only the function name
    is generated.

    Args:
        source: Source text of one top-level function.
        new_name: Replacement function name.

    Returns:
        Source text with only the function name replaced.

    Raises:
        ValueError: If no top-level ``def`` is found.
    """
    match = re.match(r"^def ([A-Za-z_]\w*)\(", source)
    if match is None:
        raise ValueError("Expected a top-level function definition in source")
    return source[: match.start(1)] + new_name + source[match.end(1) :]


_TICK_SOURCES = {
    "structured": "run_structured_tick",
    "discrete": "run_discrete_tick",
    "wf": "run_wf_tick",
    "spatial_structured": "_spatial_tick_shell_structured",
    "spatial_discrete": "_spatial_tick_shell_discrete",
}
_LOOP_SOURCES = {
    "structured": "_run_loop_structured",
    "discrete": "_run_loop_discrete",
    "wf": "_run_loop_discrete",
    "spatial_structured": "_spatial_run_shell_structured",
    "spatial_discrete": "_spatial_run_shell_discrete",
}
_HOOK_PARAMS = ["first_hook", "early_hook", "late_hook"]


def assemble_lifecycle_module(
    mode: str,
    tick_fn_name: str,
    run_fn_name: str,
    panmictic_stem: str = "",
    panmictic_tick_fn_name: str = "",
) -> str:
    """Assemble a generated lifecycle module for *mode*.

    The generated module contains the shared ``_run_event`` source, one tick
    function, and one multi-tick run function.  Hook parameters in the tick
    source are rewritten to module globals (``_FIRST_HOOK``,
    ``_EARLY_HOOK``, ``_LATE_HOOK``) which are injected with ``setattr``
    after loading, matching the previous template behaviour.

    Args:
        mode: One of ``"structured"``, ``"discrete"``, ``"wf"``,
            ``"spatial_structured"``, or ``"spatial_discrete"``.
        tick_fn_name: Name for the generated tick function.
        run_fn_name: Name for the generated run function.
        panmictic_stem: For spatial modes, the generated panmictic module
            stem used in the import path.
        panmictic_tick_fn_name: For spatial modes, the generated panmictic
            tick function name.

    Returns:
        Complete module source as a string.

    Raises:
        ValueError: If *mode* is unknown or spatial mode is missing its
            panmictic reference.
    """
    if mode not in _TICK_SOURCES:
        raise ValueError(f"Unknown lifecycle mode: {mode!r}")
    if mode.startswith("spatial_") and (
        not panmictic_stem or not panmictic_tick_fn_name
    ):
        raise ValueError(
            "Spatial lifecycle modes require panmictic stem and tick name"
        )

    source_fn_name = _TICK_SOURCES[mode]
    tick_source = inspect.getsource(globals()[source_fn_name])
    if mode in {"spatial_structured", "spatial_discrete"}:
        tick_source = _rename_function(tick_source, tick_fn_name)
    else:
        tick_source = _replace_function_signature(
            tick_source, "state, config, registry, deme_id=-1", tick_fn_name
        )
        for param in _HOOK_PARAMS:
            tick_source = _replace_identifier_outside_docstring(
                tick_source, param, f"_{param.upper()}"
            )

    loop_source = inspect.getsource(globals()[_LOOP_SOURCES[mode]])
    if mode in {"spatial_structured", "spatial_discrete"}:
        loop_source = _rename_function(loop_source, run_fn_name)
    else:
        loop_source = _replace_function_signature(
            loop_source,
            "state, config, registry, n_ticks, record_interval=0, "
            "observation_mask=None, n_obs_groups=0",
            run_fn_name,
        )
    if mode == "spatial_structured":
        loop_source = loop_source.replace("_SPATIAL_TICK_STRUCTURED", tick_fn_name)
        tick_source = tick_source.replace("_DEME_TICK_STRUCTURED", "_run_deme_tick")
    elif mode == "spatial_discrete":
        loop_source = loop_source.replace("_SPATIAL_TICK_DISCRETE", tick_fn_name)
        tick_source = tick_source.replace("_DEME_TICK_DISCRETE", "_run_deme_tick")
    elif mode == "discrete" or mode == "wf":
        loop_source = loop_source.replace("_TICK_FN_DISCRETE", tick_fn_name)
    else:
        loop_source = loop_source.replace("_TICK_FN_STRUCTURED", tick_fn_name)

    event_source = inspect.getsource(_run_event)
    header = _assemble_header(mode, panmictic_stem, panmictic_tick_fn_name)

    tick_decorator = (
        "@njit_switch(cache=True, parallel=True)"
        if mode.startswith("spatial_")
        else "@njit_switch(cache=True)"
    )
    event_block = f"@njit_switch(cache=True, inline='always')\n{event_source}"
    tick_block = f"{tick_decorator}\n{tick_source}"
    loop_block = f"@njit_switch(cache=True)\n{loop_source}"
    return "\n\n".join([header, event_block, tick_block, loop_block])


def _assemble_header(
    mode: str, panmictic_stem: str, panmictic_tick_fn_name: str
) -> str:
    """Build the generated module import and hook-global header.

    Args:
        mode: Lifecycle mode passed to :func:`assemble_lifecycle_module`.
        panmictic_stem: Panmictic generated-module stem for spatial modes.
        panmictic_tick_fn_name: Panmictic tick function name for spatial
            modes.

    Returns:
        Generated module header source as a string.
    """
    lines = [
        "from __future__ import annotations",
        "from typing import Callable, Optional",
        "import numpy as np",
        "from natal.data import (",
        "    DiscretePopulationConfig,",
        "    DiscretePopulationState,",
        "    PopulationConfig,",
        "    PopulationState,",
        ")",
        "from natal.hooks.types import (",
        "    EVENT_EARLY,",
        "    EVENT_FIRST,",
        "    EVENT_LATE,",
        "    RESULT_CONTINUE,",
        "    RESULT_STOP,",
        "    HookProgram,",
        ")",
        "from natal.hooks.runtime.csr_kernel import execute_csr_event_program_with_state",
        "from natal.numba.utils import njit_switch",
    ]
    if mode == "structured":
        lines += [
            "from natal.engine.age_structured_simulator import (",
            "    run_aging,",
            "    run_reproduction,",
            "    run_survival,",
            ")",
        ]
    elif mode == "discrete":
        lines += [
            "from natal.engine.discrete_generation_simulator import (",
            "    run_discrete_aging,",
            "    run_discrete_reproduction,",
            "    run_discrete_survival,",
            ")",
        ]
    elif mode == "wf":
        lines += [
            "from natal.engine.simulation.discrete_generation import (",
            "    run_wf_tick as _run_wf_wide,",
            ")",
        ]
    elif mode == "spatial_structured":
        lines += [
            f"from natal._hook_codegen_{panmictic_stem} import (",
            f"    {panmictic_tick_fn_name} as _raw_run_deme_tick,",
            ")",
            "from numba import prange",
            "from natal.engine.spatial_migrator import run_spatial_migration",
            "from natal.spatial.topology import (",
            "    HeterogeneousKernelParams,",
            "    MigrationParams,",
            "    SpatialTopology,",
            ")",
        ]
    else:
        lines += [
            f"from natal._hook_codegen_{panmictic_stem} import (",
            f"    {panmictic_tick_fn_name} as _raw_run_deme_tick,",
            ")",
            "from numba import prange",
            "from natal.engine.spatial_migrator import run_spatial_migration",
            "from natal.spatial.topology import (",
            "    HeterogeneousKernelParams,",
            "    MigrationParams,",
            "    SpatialTopology,",
            ")",
        ]

    lines += [
        "",
        "",
        "def _default_hook(",
        "    _state: object, _config: object | None = None, _deme_id: int = -1,",
        ") -> int:",
        "    return 0",
        "",
        "",
        "_FIRST_HOOK: Callable[[object, object, int], int] = _default_hook",
        "_EARLY_HOOK: Callable[[object, object, int], int] = _default_hook",
        "_LATE_HOOK: Callable[[object, object, int], int] = _default_hook",
        "",
        "_run_deme_tick = _raw_run_deme_tick"
        if mode.startswith("spatial_")
        else "",
    ]
    return "\n".join(lines)
