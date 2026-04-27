"""Codegen template for age-structured lifecycle wrappers.

Behaves like :file:`lifecycle_discrete.tmpl.py` but includes sperm storage
management throughout the lifecycle: FIRST → reproduction → EARLY → survival
→ LATE → aging.
"""

from typing import Callable, Optional

import numpy as np

from natal.hooks.executor import execute_csr_event_program_with_state
from natal.hooks.types import (
    EVENT_EARLY,
    EVENT_FIRST,
    EVENT_LATE,
    RESULT_CONTINUE,
    RESULT_STOP,
    HookProgram,
)
from natal.kernels.simulation_kernels import (
    run_aging,
    run_reproduction,
    run_survival,
)
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig
from natal.population_state import PopulationState


def _default_hook(
    _ind_count: np.ndarray, _tick: int, _deme_id: int = -1,
) -> int:
    return 0

# Same setattr injection mechanism as lifecycle_discrete.tmpl.py.
_FIRST_HOOK: Callable[[np.ndarray, int, int], int] = _default_hook
_EARLY_HOOK: Callable[[np.ndarray, int, int], int] = _default_hook
_LATE_HOOK: Callable[[np.ndarray, int, int], int] = _default_hook


@njit_switch(cache=True)
def TICK_FN_NAME(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[tuple[np.ndarray, np.ndarray, int], int]:
    """Execute one lifecycle tick with sperm storage management.

    Reproduction, survival, and aging all read and write ``sperm_store``,
    so the return tuple is ``(individual_count, sperm_store, tick)``.

    Args:
        state: Current population state (with sperm storage).
        config: Population configuration.
        registry: HookProgram for CSR event programs.
        deme_id: Deme index. Omitted for panmictic calls (default -1),
                 passed as actual deme index for spatial calls.

    Returns:
        ((individual_count, sperm_store, tick), result).
    """
    ind_count = state.individual_count.copy()
    tick = state.n_tick
    sperm_store = state.sperm_storage.copy()
    is_stochastic = bool(config.is_stochastic)
    use_continuous = bool(config.use_continuous_sampling)

    # FIRST event
    result = execute_csr_event_program_with_state(
        registry, EVENT_FIRST, ind_count, sperm_store, tick,
        is_stochastic, True, use_continuous, deme_id,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = _FIRST_HOOK(ind_count, tick, deme_id)
    if result != 0:
        return (ind_count, sperm_store, tick), RESULT_STOP

    ind_count, sperm_store = run_reproduction(ind_count, sperm_store, config)

    # EARLY event
    result = execute_csr_event_program_with_state(
        registry, EVENT_EARLY, ind_count, sperm_store, tick,
        is_stochastic, True, use_continuous, deme_id,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = _EARLY_HOOK(ind_count, tick, deme_id)
    if result != 0:
        return (ind_count, sperm_store, tick), RESULT_STOP

    ind_count, sperm_store = run_survival(ind_count, sperm_store, config)

    # LATE event
    result = execute_csr_event_program_with_state(
        registry, EVENT_LATE, ind_count, sperm_store, tick,
        is_stochastic, True, use_continuous, deme_id,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    result = _LATE_HOOK(ind_count, tick, deme_id)
    if result != 0:
        return (ind_count, sperm_store, tick), RESULT_STOP

    ind_count, sperm_store = run_aging(ind_count, sperm_store, config)
    return (ind_count, sperm_store, tick + 1), RESULT_CONTINUE


@njit_switch(cache=True)
def RUN_FN_NAME(
    state: PopulationState,
    config: PopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    record_interval: int = 0,
    observation_mask: Optional[np.ndarray] = None,
    n_obs_groups: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray, int], Optional[np.ndarray], bool]:
    """Execute multiple lifecycle ticks in sequence. History rows are
    ``[tick, ind..., sperm...]`` (raw) or ``[tick, observed...]`` (observation).

    Args:
        state: Initial population state (with sperm storage).
        config: Population configuration.
        registry: HookProgram for CSR event programs.
        n_ticks: Number of ticks to execute.
        record_interval: Recording interval (0 means no recording).
        observation_mask: Optional 4D mask ``(n_groups, n_sexes, n_ages, n_genotypes)``.
            When provided, history stores observation-reduced snapshots instead
            of the full flattened state.
        n_obs_groups: Number of observation groups (first axis of mask).

    Returns:
        ((individual_count, sperm_store, tick), history_array_or_None, was_stopped).
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
        temp_state = PopulationState(
            n_tick=tick, individual_count=ind_count, sperm_storage=sperm_store,
        )
        current_state, result = TICK_FN_NAME(temp_state, config, registry)
        ind_count, sperm_store, tick = current_state

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

        if result != RESULT_CONTINUE:
            was_stopped = True
            break

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind_count, sperm_store, tick), history_result, was_stopped
