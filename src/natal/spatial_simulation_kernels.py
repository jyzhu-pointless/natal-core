"""Spatial simulation kernels.

This module hosts multi-deme kernels separate from single-deme simulation_kernels.
The core ``run_spatial_tick`` function is Numba-switchable and operates on stacked
state arrays across demes.

Hook execution model in this module:
1) Hooks are still executed by the existing Python HookExecutor.
2) Heavy numeric work (reproduction/survival/aging + optional migration) runs in njit.
3) Spatial wrappers pass ``deme_id`` to HookExecutor so one hook can target
    one/many demes via ``deme_selector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig
from natal.population_state import PopulationState
from natal.simulation_kernels import run_aging, run_reproduction, run_survival
from natal.hook_dsl import EVENT_EARLY, EVENT_FIRST, EVENT_LATE

__all__ = [
    "RESULT_CONTINUE",
    "RESULT_STOP",
    "run_spatial_reproduction",
    "run_spatial_survival",
    "run_spatial_aging",
    "run_spatial_tick",
    "run_spatial_tick_with_adjacency_migration",
    "run_spatial_hook_event",
    "run_spatial_tick_with_hooks",
]

RESULT_CONTINUE = 0
RESULT_STOP = 1


@njit_switch(cache=True)
def _apply_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    rate: float,
) -> NDArray[np.float64]:
    """Apply adjacency-based migration over the first axis (deme axis)."""
    n_demes = ind_count_all.shape[0]
    flat = ind_count_all.reshape(n_demes, -1)
    inflow = np.zeros_like(flat)

    for src in range(n_demes):
        for dst in range(n_demes):
            w = adjacency[src, dst]
            if w == 0.0:
                continue
            inflow[dst, :] += w * flat[src, :]

    out_flat = (1.0 - rate) * flat + rate * inflow
    return out_flat.reshape(ind_count_all.shape)


@njit_switch(cache=True)
def run_spatial_reproduction(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run reproduction for all demes in one njit batch."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()

    n_demes = ind.shape[0]

    for deme_id in range(n_demes):
        ind_d, sperm_d = run_reproduction(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d

    return ind, sperm


@njit_switch(cache=True)
def run_spatial_survival(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run survival for all demes in one njit batch."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()

    n_demes = ind.shape[0]

    for deme_id in range(n_demes):
        ind_d, sperm_d = run_survival(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d

    return ind, sperm


@njit_switch(cache=True)
def run_spatial_aging(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run aging for all demes in one njit batch and increment tick."""
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()

    n_demes = ind.shape[0]

    for deme_id in range(n_demes):
        ind_d, sperm_d = run_aging(ind[deme_id], sperm[deme_id], config)
        ind[deme_id] = ind_d
        sperm[deme_id] = sperm_d

    return ind, sperm, int(tick) + 1


@njit_switch(cache=True)
def run_spatial_tick(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick for all demes: reproduction -> survival -> aging."""
    ind, sperm = run_spatial_reproduction(ind_count_all, sperm_store_all, config)
    ind, sperm = run_spatial_survival(ind, sperm, config)
    ind, sperm, tick_next = run_spatial_aging(ind, sperm, config, tick)
    return ind, sperm, tick_next


@njit_switch(cache=True)
def run_spatial_tick_with_adjacency_migration(
    ind_count_all: NDArray[np.float64],
    sperm_store_all: NDArray[np.float64],
    config: PopulationConfig,
    tick: int,
    adjacency: NDArray[np.float64],
    migration_rate: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run one spatial tick then apply adjacency migration on individual counts."""
    ind, sperm, tick_next = run_spatial_tick(ind_count_all, sperm_store_all, config, tick)
    if migration_rate > 0.0:
        ind = _apply_adjacency_migration(ind, adjacency, migration_rate)
    return ind, sperm, tick_next


@dataclass
class _SpatialHookPopulationView:
    """Lightweight hook execution view for one deme state/config pair."""

    state: PopulationState
    _config: PopulationConfig


def _resolve_deme_config(
    deme_configs: Union[PopulationConfig, List[PopulationConfig]],
    deme_id: int,
) -> PopulationConfig:
    if isinstance(deme_configs, list):
        if deme_id < 0 or deme_id >= len(deme_configs):
            raise IndexError(f"deme config index out of range: {deme_id}")
        return deme_configs[deme_id]
    return deme_configs


def run_spatial_hook_event(
    event_id: int,
    hook_executor: Any,
    deme_states: List[PopulationState],
    deme_configs: Union[PopulationConfig, List[PopulationConfig]],
) -> int:
    """Execute one hook event across all demes with deme-aware filtering."""
    for deme_id, state in enumerate(deme_states):
        cfg = _resolve_deme_config(deme_configs, deme_id)
        view = _SpatialHookPopulationView(state=state, _config=cfg)
        result = hook_executor.execute_event(event_id, view, state.n_tick, deme_id=deme_id)
        if result == RESULT_STOP:
            return RESULT_STOP
    return RESULT_CONTINUE


def run_spatial_tick_with_hooks(
    deme_states: List[PopulationState],
    config: PopulationConfig,
    hook_executor: Optional[Any] = None,
    adjacency: Optional[NDArray[np.float64]] = None,
    migration_rate: float = 0.0,
) -> Tuple[List[PopulationState], int]:
    """Python orchestration wrapper with hook dispatch around njit spatial tick.

    This wrapper keeps hook execution in Python while delegating heavy numeric
    per-deme stepping to njit kernels.

    Event/stage order in this spatial wrapper:
    - first
    - reproduction (njit batch)
    - early
    - survival (njit batch)
    - late
    - aging (njit batch)
    - migration (optional, njit batch)
    """
    if not deme_states:
        return [], RESULT_CONTINUE

    working_states = [
        PopulationState(
            n_tick=int(state.n_tick),
            individual_count=state.individual_count.copy(),
            sperm_storage=state.sperm_storage.copy(),
        )
        for state in deme_states
    ]

    if hook_executor is not None:
        # Spatial selector filtering happens inside HookExecutor via deme_id.
        if run_spatial_hook_event(EVENT_FIRST, hook_executor, working_states, config) == RESULT_STOP:
            return working_states, RESULT_STOP

    ind_all = np.stack([s.individual_count for s in working_states], axis=0)
    sperm_all = np.stack([s.sperm_storage for s in working_states], axis=0)
    tick = int(working_states[0].n_tick)

    # reproduction
    ind_all, sperm_all = run_spatial_reproduction(ind_all, sperm_all, config)
    for deme_id in range(len(working_states)):
        working_states[deme_id] = working_states[deme_id]._replace(
            individual_count=ind_all[deme_id],
            sperm_storage=sperm_all[deme_id],
        )

    if hook_executor is not None:
        # EARLY runs after reproduction and before survival.
        if run_spatial_hook_event(EVENT_EARLY, hook_executor, working_states, config) == RESULT_STOP:
            return working_states, RESULT_STOP

    # Hooks may mutate state arrays in-place; rebuild stacked arrays.
    ind_all = np.stack([s.individual_count for s in working_states], axis=0)
    sperm_all = np.stack([s.sperm_storage for s in working_states], axis=0)

    # survival
    ind_all, sperm_all = run_spatial_survival(ind_all, sperm_all, config)
    for deme_id in range(len(working_states)):
        working_states[deme_id] = working_states[deme_id]._replace(
            individual_count=ind_all[deme_id],
            sperm_storage=sperm_all[deme_id],
        )

    if hook_executor is not None:
        # LATE runs after survival and before aging.
        if run_spatial_hook_event(EVENT_LATE, hook_executor, working_states, config) == RESULT_STOP:
            return working_states, RESULT_STOP

    # Hooks may mutate state arrays in-place; rebuild stacked arrays.
    ind_all = np.stack([s.individual_count for s in working_states], axis=0)
    sperm_all = np.stack([s.sperm_storage for s in working_states], axis=0)

    # aging
    ind_all, sperm_all, tick = run_spatial_aging(ind_all, sperm_all, config, tick)

    # migration is always the final stage in one spatial tick.
    if adjacency is not None and migration_rate > 0.0:
        ind_all = _apply_adjacency_migration(ind_all, adjacency, migration_rate)

    for deme_id in range(len(working_states)):
        working_states[deme_id] = working_states[deme_id]._replace(
            individual_count=ind_all[deme_id],
            sperm_storage=sperm_all[deme_id],
            n_tick=tick,
        )

    return working_states, RESULT_CONTINUE
