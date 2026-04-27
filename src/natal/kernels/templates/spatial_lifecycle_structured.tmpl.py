"""Codegen template for spatial age-structured lifecycle wrappers.

Same delegation pattern as :file:`spatial_lifecycle_discrete.tmpl.py` but
with sperm storage management — per-deme lifecycle state includes
``sperm_store``, and migration exchanges both individuals and sperm across
demes.

The compiler (:func:`compile_spatial_lifecycle_wrapper`) substitutes
``TICK_FN_NAME``, ``RUN_FN_NAME``, ``PANMICTIC_STEM``, and
``PANMICTIC_TICK_FN_NAME`` placeholders.

Hook globals (``_FIRST_HOOK``/``_EARLY_HOOK``/``_LATE_HOOK``) live on the
panmictic module, not here. See :file:`spatial_lifecycle_discrete.tmpl.py`
for details.
"""
from typing import Callable, Optional, cast

import numpy as np
from natal._hook_codegen_PANMICTIC_STEM import (  # type: ignore[reportMissingImports]
    PANMICTIC_TICK_FN_NAME as _raw_run_deme_tick,  # type: ignore[reportUnknownVariableType]
)
from numba import prange  # type: ignore[reportMissingTypeStubs]

from natal.hooks.types import RESULT_CONTINUE, HookProgram
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig
from natal.population_state import PopulationState

# pyright cannot see the dynamically-generated panmictic module, so cast
# the imported tick function to the expected signature for type safety.
_run_deme_tick: Callable[..., tuple[tuple[np.ndarray, np.ndarray, int], int]] = cast(
    Callable[..., tuple[tuple[np.ndarray, np.ndarray, int], int]], _raw_run_deme_tick
)


@njit_switch(cache=True, parallel=True)
def TICK_FN_NAME(
    ind_count_all: np.ndarray,
    sperm_store_all: np.ndarray,
    config_bank: list[PopulationConfig],
    deme_config_ids: np.ndarray,
    registry: HookProgram,
    tick: int,
    adjacency: np.ndarray,
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: np.ndarray,
    kernel_include_center: bool,
    migration_rate: float,
    adjust_migration_on_edge: bool = False,
    deme_kernel_ids: np.ndarray | None = None,
    kernel_d_row: np.ndarray | None = None,
    kernel_d_col: np.ndarray | None = None,
    kernel_weights: np.ndarray | None = None,
    kernel_nnzs: np.ndarray | None = None,
    kernel_total_sums: np.ndarray | None = None,
    max_nnz: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Execute one spatial tick with sperm storage: per-deme lifecycle in prange, then migration.

    Each deme runs the age-structured panmictic lifecycle tick (with sperm
    storage) independently inside ``prange``. After all demes complete,
    ``run_spatial_migration`` exchanges both individuals and sperm across demes.

    Args:
        ind_count_all: 2-D array of individual counts, shape (n_demes, n_age_groups).
        sperm_store_all: 3-D array of sperm storage, shape (n_demes, ...).
        config_bank: List of PopulationConfigs indexed by deme_config_ids.
        deme_config_ids: Integer array mapping each deme to a config index.
        registry: HookProgram for CSR event programs.
        tick: Current tick number (same for all demes).
        adjacency: Adjacency matrix for migration.
        migration_mode: Migration mode identifier.
        topology_rows: Number of rows in spatial topology.
        topology_cols: Number of columns in spatial topology.
        topology_wrap: Whether topology wraps around (toroidal).
        migration_kernel: Migration kernel weight matrix.
        kernel_include_center: Whether kernel includes center cell.
        migration_rate: Per-capita migration rate.
        adjust_migration_on_edge: Whether to adjust migration rates on boundaries.

    Returns:
        (ind_count_all, sperm_store_all, tick + 1, was_stopped).
    """
    n_demes = ind_count_all.shape[0]
    stopped = np.zeros(n_demes, dtype=np.bool_)
    for d in prange(n_demes):
        cfg = config_bank[int(deme_config_ids[d])]
        ind = ind_count_all[d].copy()
        sperm = sperm_store_all[d].copy()

        state = PopulationState(n_tick=tick, individual_count=ind, sperm_storage=sperm)
        (ind, sperm, _next_tick), result = _run_deme_tick(state, cfg, registry, d)

        if result != RESULT_CONTINUE:
            stopped[d] = True

        ind_count_all[d] = ind
        sperm_store_all[d] = sperm

    ind_count_all, sperm_store_all = run_spatial_migration(
        ind_count_all, sperm_store_all, adjacency, migration_mode,
        topology_rows, topology_cols, topology_wrap,
        migration_kernel, kernel_include_center,
        config_bank[0], migration_rate, adjust_migration_on_edge,
        deme_kernel_ids, kernel_d_row, kernel_d_col,
        kernel_weights, kernel_nnzs, kernel_total_sums, max_nnz,
    )

    was_stopped = False
    for i in range(n_demes):
        if stopped[i]:
            was_stopped = True
            break
    return ind_count_all, sperm_store_all, tick + 1, was_stopped


@njit_switch(cache=True)
def RUN_FN_NAME(
    ind_count_all: np.ndarray,
    sperm_store_all: np.ndarray,
    config_bank: list[PopulationConfig],
    deme_config_ids: np.ndarray,
    registry: HookProgram,
    tick: int,
    n_steps: int,
    adjacency: np.ndarray,
    migration_mode: int,
    topology_rows: int,
    topology_cols: int,
    topology_wrap: bool,
    migration_kernel: np.ndarray,
    kernel_include_center: bool,
    migration_rate: float,
    adjust_migration_on_edge: bool = False,
    deme_kernel_ids: np.ndarray | None = None,
    kernel_d_row: np.ndarray | None = None,
    kernel_d_col: np.ndarray | None = None,
    kernel_weights: np.ndarray | None = None,
    kernel_nnzs: np.ndarray | None = None,
    kernel_total_sums: np.ndarray | None = None,
    max_nnz: int = 0,
    record_interval: int = 0,
    observation_mask: Optional[np.ndarray] = None,
    n_obs_groups: int = 0,
    deme_selector: Optional[np.ndarray] = None,
) -> tuple[tuple[np.ndarray, np.ndarray, int], Optional[np.ndarray], bool]:
    """Execute multiple spatial ticks with sperm storage, with optional history recording.

    Calls TICK_FN_NAME in a loop and optionally records flattened snapshots
    of the full state (individuals + sperm) at each ``record_interval`` tick.
    When ``observation_mask`` is provided, history stores observation-reduced
    snapshots instead.

    Args:
        Same as TICK_FN_NAME, plus:
        n_steps: Number of ticks to execute.
        record_interval: Recording interval (0 means no recording).
        observation_mask: Optional 4D mask ``(n_groups, n_sexes, n_ages, n_genotypes)``.
        n_obs_groups: Number of observation groups.
        deme_selector: Optional per-group deme filter ``(n_groups, n_demes)``.

    Returns:
        ((ind_count_all, sperm_store_all, tick), history_array_or_None, was_stopped).
    """
    was_stopped = False
    ind = ind_count_all.copy()
    sperm = sperm_store_all.copy()
    tick_cur = tick

    if observation_mask is not None:
        n_demes_ = ind.shape[0]
        n_sexes_ = ind.shape[1]
        n_ages_ = ind.shape[2]
        flatten_size = 1 + n_demes_ * n_obs_groups * n_sexes_ * n_ages_
    else:
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
        if observation_mask is not None:
            observed = np.sum(observation_mask[None, :, :, :, :] * ind[:, None, :, :, :], axis=-1)
            if deme_selector is not None:
                observed = observed * deme_selector.T[:, :, None, None]
            flat_state[1:] = observed.flatten()
        else:
            flat_state[1:1 + ind.size] = ind.flatten()
            flat_state[1 + ind.size:] = sperm.flatten()
        history_array[history_count, :] = flat_state
        history_count += 1

    for _ in range(n_steps):
        ind, sperm, tick_cur, step_stopped = TICK_FN_NAME(
            ind, sperm, config_bank, deme_config_ids, registry, tick_cur,
            adjacency, migration_mode, topology_rows, topology_cols, topology_wrap,
            migration_kernel, kernel_include_center, migration_rate, adjust_migration_on_edge,
            deme_kernel_ids, kernel_d_row, kernel_d_col,
            kernel_weights, kernel_nnzs, kernel_total_sums, max_nnz,
        )
        if step_stopped:
            was_stopped = True
            break

        if record_interval > 0 and (tick_cur % record_interval == 0):
            flat_state = np.zeros(flatten_size, dtype=np.float64)
            flat_state[0] = tick_cur
            if observation_mask is not None:
                observed = np.sum(observation_mask[None, :, :, :, :] * ind[:, None, :, :, :], axis=-1)
                if deme_selector is not None:
                    observed = observed * deme_selector.T[:, :, None, None]
                flat_state[1:] = observed.flatten()
            else:
                flat_state[1:1 + ind.size] = ind.flatten()
                flat_state[1 + ind.size:] = sperm.flatten()
            history_array[history_count, :] = flat_state
            history_count += 1

    if record_interval > 0:
        history_result = history_array[:history_count, :]
    else:
        history_result = None
    return (ind, sperm, tick_cur), history_result, was_stopped
