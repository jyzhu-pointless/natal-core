# pyright: ignore[reportUnusedImport, reportUnusedVariable, reportUnknownVariableType, reportUnknownParameterType, reportMissingParameterType, reportUnusedExpression, reportUndefinedVariable]
# ruff: noqa

"""Codegen template for unified mixed-type hook dispatch (see compile/codegen.py)."""

from typing import Callable, Optional

import numpy as np

from natal.hooks.runtime.csr_kernel import execute_single_csr_hook
from natal.numba_utils import njit_switch

# HookProgram array globals (injected via setattr).  All 15 arrays are
# required by _execute_single_csr_hook and must be present on the module.
_HP_N_HOOKS: np.int32 = np.int32(0)
_HP_OP_OFFSETS: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_OP_TYPES_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_GIDX_OFFSETS_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_GIDX_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_AGE_OFFSETS_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_AGE_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_SEX_MASKS_DATA: np.ndarray = np.zeros(0, dtype=np.bool_)
_HP_PARAMS_DATA: np.ndarray = np.zeros(0, dtype=np.float64)
_HP_CONDITION_OFFSETS_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_CONDITION_TYPES_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_CONDITION_PARAMS_DATA: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_DEME_SELECTOR_TYPES: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_DEME_SELECTOR_OFFSETS: np.ndarray = np.zeros(0, dtype=np.int32)
_HP_DEME_SELECTOR_DATA: np.ndarray = np.zeros(0, dtype=np.int32)

# Njit function placeholders (injected via setattr).
_FN_0: Callable[..., int] = (
    lambda _state, _config=None, _deme_id=-1: 0  # type: ignore[assignment]
)


@njit_switch(cache=True)
def _unified_hook_TEMPLATE(state, config=None, deme_id=-1):  # type: ignore[no-untyped-def]
    ind_count = state.individual_count  # type: ignore[union-attr]
    tick = state.n_tick  # type: ignore[union-attr]
    stochastic = config.stochastic  # type: ignore[union-attr]
    continuous_sampling = config.continuous_sampling  # type: ignore[union-attr]
    sperm_store = "PLACEHOLDER_SPERM_SETUP"

# === PLACEHOLDER: DO NOT DELETE ===
# This comment line is replaced at codegen time by compile_unified_event_hook
# with the priority-ordered schedule body.  It must stay at column 0 — the
# replacement text carries its own indentation.
# PLACEHOLDER_SCHEDULE_BODY
# === END PLACEHOLDER ===

    return 0
