"""Codegen template for Wright-Fisher extreme-speed lifecycle wrappers.

Follows the same pattern as lifecycle_discrete_v2.tmpl.py but replaces the
mate→fertilize→survive→aging chain with a single run_wf_tick call.
Only FIRST hooks are supported — there is no natural insertion point for
EARLY (between reproduction and survival) or LATE (between survival and
aging) since WF fuses all three steps into one.

The compiler (:func:`compile_lifecycle_wrapper`) reads this file and performs
string substitution:
  - ``TICK_FN_NAME`` → actual tick function name
  - ``RUN_FN_NAME`` → actual run function name

The ``_FIRST_HOOK`` module-level global is injected via ``setattr`` after
loading.
"""

from typing import Callable, Optional

import numpy as np

from natal.engine.simulation.discrete_generation import run_wf_tick
from natal.hooks.runtime.csr_kernel import execute_csr_event_program_with_state
from natal.hooks.types import (
    EVENT_FIRST,
    RESULT_CONTINUE,
    RESULT_STOP,
    HookProgram,
)
from natal.numba_utils import njit_switch
from natal.population_config import DiscretePopulationConfig
from natal.population_state import DiscretePopulationState


def _default_hook(
    _state: object, _config: object = None, _deme_id: int = -1,
) -> int:
    return 0


_FIRST_HOOK: Callable[[object, object, int], int] = _default_hook


@njit_switch(cache=True)
def TICK_FN_NAME(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    deme_id: int = -1,
) -> tuple[tuple[np.ndarray, int], int]:
    ind_count = state.individual_count.copy()
    tick = state.n_tick
    dummy_sperm_store = np.zeros((0, 0, 0), dtype=np.float64)
    stochastic = bool(config.stochastic)
    use_continuous = bool(config.continuous_sampling)
    mode = int(config.extreme_speed_mode)

    # ---- FIRST hooks ----
    result = execute_csr_event_program_with_state(
        registry, EVENT_FIRST, ind_count, dummy_sperm_store, tick,
        stochastic, False, use_continuous, deme_id,
    )
    if result != RESULT_CONTINUE:
        return (ind_count, tick), RESULT_STOP
    result = _FIRST_HOOK(
        DiscretePopulationState(n_tick=tick, individual_count=ind_count),
        config, deme_id,
    )
    if result != 0:
        return (ind_count, tick), RESULT_STOP

    # ---- WF tick (reproduction + competition + sampling + aging in one step) ----
    ind_count = run_wf_tick(
        ind_count=ind_count,
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
        mode=mode,
        stochastic=stochastic,
        mating_rate_f=config.female_adult_mating_rate,
        mating_rate_m=config.male_adult_mating_rate,
        reproduction_rate=config.reproduction_rate,
        carrying_capacity=float(config.carrying_capacity[()]),
        juvenile_growth_mode=int(config.juvenile_growth_mode[()]),
        low_density_growth_rate=float(config.low_density_growth_rate[()]),
        expected_competition_strength=float(config.expected_competition_strength[()]),
        expected_survival_rate=float(config.expected_survival_rate[()]),
    )

    # No EARLY / LATE hooks — WF fuses reproduction + survival + aging
    # into a single atomic step, so there is no natural insertion point
    # between those stages.

    return (ind_count, tick + 1), RESULT_CONTINUE


@njit_switch(cache=True)
def RUN_FN_NAME(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    registry: HookProgram,
    n_ticks: int,
    record_interval: int = 0,
    observation_mask: Optional[np.ndarray] = None,
    n_obs_groups: int = 0,
) -> tuple[tuple[np.ndarray, int], Optional[np.ndarray], bool]:
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
        current_state, result = TICK_FN_NAME(
            DiscretePopulationState(n_tick=tick, individual_count=ind_count),
            config, registry,
        )
        ind_count, tick = current_state

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

        if result != RESULT_CONTINUE:
            was_stopped = True
            break

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind_count, tick), history_result, was_stopped
