"""CSR execution engine — Numba-accelerated declarative hook kernels.

The hot loop operates on flattened ndarrays and avoids Python objects entirely.
These functions are called from two contexts:

* **Numba fast path** — lifecycle templates call ``execute_csr_event_arrays``
  (batch) or unified functions from ``compile.codegen.compile_unified_event_hook``
  (mixed CSR + njit interleaved by priority).
* **Python fallback** — ``HookExecutor`` (in ``natal.hooks.runtime.fallback``)
  calls ``execute_csr_event_arrays`` for individual descriptors when Numba
  is disabled.

Return value protocol
---------------------
Every hook execution function returns an int:

``RESULT_CONTINUE`` (0)
    All operations completed; proceed to the next hook.
``RESULT_SKIP`` (0, alias)
    Hook not applicable in this context (e.g. wrong deme).  Same runtime
    behaviour as ``RESULT_CONTINUE``; the distinct name is for readability.
``RESULT_STOP`` (1)
    Abort the current event immediately.  Subsequent hooks for the same
    event are skipped, but the next event still executes normally.

Sperm storage
-------------
Age-structured populations carry a ``sperm_storage`` array.  When a female
count is reduced, sperm categories must be scaled by the same survival rate
to stay coherent.  Discrete-generation models have no sperm storage
(``has_sperm_storage=False``) and use the simpler ``_apply_target_without_sperm``
path for all cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from natal import numba_compat as nbc
from natal.numba_utils import njit_switch

from ..types import (
    COND_OP_AND,
    COND_OP_NOT,
    COND_OP_OR,
    COND_TICK_GT,
    RESULT_CONTINUE,
    RESULT_SKIP,
    RESULT_STOP,
    DemeSelector,
    HookProgram,
)

# ---------------------------------------------------------------------------
# Deme selector helpers
# ---------------------------------------------------------------------------


def deme_selector_matches(selector: DemeSelector, deme_id: int) -> bool:
    """Return whether *deme_id* should execute under *selector* (Python path).

    Supported forms: ``"*"`` (wildcard), ``int`` (single deme), or
    ``list`` / ``tuple`` / ``range`` (set of demes).
    """
    if selector == "*":
        return True
    if isinstance(selector, int):
        return selector == deme_id
    if isinstance(selector, range):
        return deme_id in selector
    return deme_id in selector


@njit_switch(cache=True)
def njit_deme_selector_matches(
    sel_type: int,
    start: int,
    end: int,
    data: np.ndarray,
    deme_id: int,
) -> bool:
    """Numba-compatible deme selector check against serialised arrays.

    The selector is encoded in the HookProgram's ``deme_selector_*``
    arrays with these *sel_type* values:

    ==========  ============================================
    sel_type    Meaning
    ==========  ============================================
    0           ``"*"`` (ANY) — always True
    1           single integer — ``data[start] == deme_id``
    2           ``range`` — ``start <= deme_id < end``
    3           list/tuple — iterate ``data[start:end]``
    ==========  ============================================
    """
    if sel_type == 0:  # ANY — wildcard
        return True
    if sel_type == 1:  # SINGLE
        return data[start] == deme_id
    if sel_type == 2:  # RANGE
        return deme_id >= data[start] and deme_id < data[start + 1]
    if sel_type == 3:  # LIST — linear scan
        if start >= end:
            return False
        for i in range(start, end):
            if data[i] == deme_id:
                return True
        return False
    return True  # Unknown type — allow (conservative)


# ---------------------------------------------------------------------------
# CSR condition evaluation (RPN — Reverse Polish Notation)
# ---------------------------------------------------------------------------

# Condition token type constants inlined for njit scope (avoids attribute
# lookup).  Values must match types.py:COND_*.
_COND_ALWAYS = 0
_COND_TICK_EQ = 1
_COND_TICK_MOD = 2
_COND_TICK_GE = 3
_COND_TICK_LT = 4
_COND_TICK_LE = 5
_COND_TICK_GT = 6


@njit_switch(cache=True)
def _check_csr_condition(cond_type: int, cond_param: int, tick: int) -> bool:
    """Evaluate a single atomic condition token against the current tick.

    Each declarative op can carry a ``when`` clause (e.g. ``"tick >= 100"``)
    that gets parsed into an RPN token stream.  This function handles the
    *leaf* tokens — tick comparisons like ``tick == 5`` or ``tick % 3 == 0``.
    Logical operators (AND/OR/NOT) have higher token values and are handled
    by ``_eval_csr_condition_program``.
    """
    if cond_type == _COND_ALWAYS:
        return True
    if cond_type == _COND_TICK_EQ:
        return tick == cond_param
    if cond_type == _COND_TICK_MOD:
        return cond_param > 0 and tick % cond_param == 0
    if cond_type == _COND_TICK_GE:
        return tick >= cond_param
    if cond_type == _COND_TICK_LT:
        return tick < cond_param
    if cond_type == _COND_TICK_LE:
        return tick <= cond_param
    if cond_type == _COND_TICK_GT:
        return tick > cond_param
    if cond_type >= COND_OP_AND:
        return False  # Logical operators should never reach the atomic evaluator.
    return True


@njit_switch(cache=True)
def _eval_csr_condition_program(
    cond_types: np.ndarray,
    cond_params: np.ndarray,
    cond_start: int,
    cond_end: int,
    tick: int,
) -> bool:
    """Evaluate an RPN condition program spanning ``[cond_start, cond_end)``.

    Each operation's ``when`` clause is compiled to a postfix token stream
    stored in the flattened ``condition_types_data`` and
    ``condition_params_data`` arrays.  The evaluation uses an int8 stack:
    leaf tokens push 0 or 1; AND/OR/NOT pop and push the result.

    Returns:
        True if the condition is satisfied or if the span is empty,
        False otherwise.
    """
    max_len = cond_end - cond_start
    if max_len <= 0:
        return True  # No condition — always execute.

    # int8 stack — values are only ever 0 or 1, minimal footprint.
    stack = np.zeros(max_len + 1, dtype=np.int8)
    top = 0  # Next free slot (one past the last pushed value).

    for idx in range(cond_start, cond_end):
        token_type = cond_types[idx]
        token_param = cond_params[idx]

        # Leaf: atomic predicate → push 0 or 1.
        if token_type <= COND_TICK_GT:
            val = 1 if _check_csr_condition(token_type, token_param, tick) else 0
            stack[top] = val
            top += 1
            continue

        # NOT: pop one, negate, push.
        if token_type == COND_OP_NOT:
            if top < 1:
                return False
            stack[top - 1] = 0 if stack[top - 1] else 1
            continue

        # AND: pop two, AND, push.
        if token_type == COND_OP_AND:
            if top < 2:
                return False
            rhs = stack[top - 1]
            lhs = stack[top - 2]
            top -= 2
            stack[top] = 1 if (lhs and rhs) else 0
            top += 1
            continue

        # OR: pop two, OR, push.
        if token_type == COND_OP_OR:
            if top < 2:
                return False
            rhs = stack[top - 1]
            lhs = stack[top - 2]
            top -= 2
            stack[top] = 1 if (lhs or rhs) else 0
            top += 1
            continue

        return False  # Unknown token.

    if top != 1:
        return False
    return stack[0] != 0


# Public alias for tests and external consumers.
eval_csr_condition_program = _eval_csr_condition_program


# ---------------------------------------------------------------------------
# Target-count application helpers for survival sampling
# ---------------------------------------------------------------------------
#
# Hook operations express a *target count* (e.g. "set to 20", "scale by 0.5").
# When target < current, removal is modelled as *survival* — each individual
# survives with probability = target / current.  This keeps ``Op.scale(0.5)``
# semantically identical to "50 % survival" and ensures sperm storage scaling
# stays coherent.


@njit_switch(cache=True)
def _sample_survivors(
    n_base: float,
    survival_prob: float,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> float:
    """Return the survivor count after applying *survival_prob* to *n_base*.

    Args:
        n_base: Current count (may be fractional for continuous models).
        survival_prob: Per-individual survival probability in [0, 1].
        stochastic_flag: If False, use deterministic multiplication.
        dirichlet_flag: If True, keep counts continuous (no integer rounding).

    Returns:
        Survivor count — continuous if *dirichlet_flag*, else integer-rounded.
    """
    if n_base <= 0.0:
        return 0.0
    if stochastic_flag:
        if dirichlet_flag:
            return nbc.continuous_binomial(n_base, survival_prob)
        return float(np.random.binomial(int(round(n_base)), survival_prob))
    return n_base * survival_prob


@njit_switch(cache=True)
def _apply_target_without_sperm(
    current_count: float,
    target_count: float,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> float:
    """Apply a target count update for populations *without* sperm storage.

    Used for males in all models and for all individuals in discrete-generation
    models.  When *target_count* >= *current_count*, individuals are simply
    added.  When *target_count* < *current_count*, a survival process is
    applied with probability = target / current.
    """
    if stochastic_flag and not dirichlet_flag:
        current_count = float(round(current_count))

    if target_count >= current_count:
        return target_count  # Adding individuals — no survival needed.
    if current_count <= 0.0:
        return 0.0

    survival_prob = max(0.0, min(1.0, target_count / current_count))
    return _sample_survivors(current_count, survival_prob, stochastic_flag, dirichlet_flag)


@njit_switch(cache=True)
def _apply_target_with_sperm(
    current_count: float,
    target_count: float,
    sperm_row: np.ndarray,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> float:
    """Apply a target count update for the female branch with sperm storage.

    Used for age-structured models where female counts are linked to sperm
    category counts.  When reducing the female count, sperm categories are
    scaled (or sampled) by the **same survival rate**, keeping the population
    state coherent.

    The female count is conceptually split into *virgins* (no stored sperm)
    and *mated* females (one entry per gamete-male genotype).  Each subgroup
    survives independently; the results are summed back into the total.

    Args:
        current_count: Total female count before the operation.
        target_count: Desired female count after the operation.
        sperm_row: ``sperm_storage[age, gidx, :]`` — per-genotype-male sperm
            counts for this (age, female-genotype) cell.  Mutated in-place.
        stochastic_flag: If False, use deterministic proportional scaling.
        dirichlet_flag: If True, use continuous sampling (no integer rounding).

    Returns:
        New total female count = surviving virgins + surviving mated.
    """
    if stochastic_flag and not dirichlet_flag:
        current_count = float(round(current_count))

    if target_count >= current_count:
        return target_count  # Adding — sperm storage unchanged.

    if current_count <= 0.0:
        for gm_idx in range(sperm_row.shape[0]):
            sperm_row[gm_idx] = 0.0
        return 0.0

    survival_prob = max(0.0, min(1.0, target_count / current_count))

    # Deterministic: proportionally scale sperm and total.
    if not stochastic_flag:
        for gm_idx in range(sperm_row.shape[0]):
            sperm_row[gm_idx] *= survival_prob
        return target_count

    # Stochastic: sample each sperm category independently.
    n_f_raw = float(current_count)
    total_sperm_count = 0.0
    for gm_idx in range(sperm_row.shape[0]):
        total_sperm_count += float(sperm_row[gm_idx])

    n_virgins_raw = n_f_raw - total_sperm_count
    if n_virgins_raw >= -nbc.EPS:
        n_virgins_raw = max(0.0, n_virgins_raw)
    if n_virgins_raw < 0.0:
        print(
            "n_virgins<0 in _apply_target_with_sperm:",
            n_virgins_raw,
            "n_f_raw=",
            n_f_raw,
            "total_sperm=",
            total_sperm_count,
        )
        raise ValueError("Invalid state: n_virgins < 0 in _apply_target_with_sperm")

    n_virgins = n_virgins_raw if dirichlet_flag else float(int(round(n_virgins_raw)))

    new_sperm_sum = 0.0
    for gm_idx in range(sperm_row.shape[0]):
        if dirichlet_flag:
            n_sperm = sperm_row[gm_idx]
        else:
            n_sperm = float(int(round(sperm_row[gm_idx])))
        sperm_row[gm_idx] = _sample_survivors(n_sperm, survival_prob, True, dirichlet_flag)
        new_sperm_sum += sperm_row[gm_idx]

    survivors_virgins = _sample_survivors(n_virgins, survival_prob, True, dirichlet_flag)
    return new_sperm_sum + survivors_virgins


# ===================================================================
# Declarative CSR execution — the hot loop
# ===================================================================
#
# Two callable granularities:
#
#   ``_execute_single_csr_hook(hook_idx, ...)``
#       Per-hook primitive.  Extracted so unified mixed-type dispatch
#       can interleave CSR hooks with njit calls at arbitrary positions
#       in a priority-ordered schedule.
#
#   ``execute_csr_event_arrays(event_id, ...)``
#       Batch dispatch for one event.  Used by lifecycle templates and
#       ``HookExecutor`` (Python fallback).  Delegates each hook to
#       ``_execute_single_csr_hook``.
# ===================================================================

# OpType enum values inlined for njit scope (avoids attribute lookup).
_OP_SCALE = 0
_OP_SET = 1
_OP_ADD = 2
_OP_SUBTRACT = 3
_OP_KILL = 4
_OP_SAMPLE = 5
_OP_STOP_IF_ZERO = 6
_OP_STOP_IF_BELOW = 7
_OP_STOP_IF_ABOVE = 8
_OP_STOP_IF_EXTINCTION = 9


@njit_switch(cache=True, parallel=True)
def _execute_single_csr_hook(
    hook_idx: int,
    n_hooks: int | np.integer[Any],
    op_offsets: np.ndarray,
    op_types_data: np.ndarray,
    gidx_offsets_data: np.ndarray,
    gidx_data: np.ndarray,
    age_offsets_data: np.ndarray,
    age_data: np.ndarray,
    sex_masks_data: np.ndarray,
    params_data: np.ndarray,
    condition_offsets_data: np.ndarray,
    condition_types_data: np.ndarray,
    condition_params_data: np.ndarray,
    deme_selector_types: np.ndarray,
    deme_selector_offsets: np.ndarray,
    deme_selector_data: np.ndarray,
    individual_count: np.ndarray,
    sperm_storage: np.ndarray,
    has_sperm_storage: bool,
    tick: int,
    stochastic: bool,
    continuous_sampling: bool,
    deme_id: int,
) -> int:
    """Execute a single CSR hook at global index *hook_idx*.

    This is the **per-hook CSR primitive**.  Given a global index into the
    flattened ``HookProgram`` arrays, it:

    1. Bounds-checks *hook_idx* (returns ``RESULT_SKIP`` if invalid).
    2. Checks the serialised deme selector (returns ``RESULT_SKIP`` if
       *deme_id* doesn't match).
    3. Iterates over the hook's operations — ``op_offsets[hook_idx]``
       to ``op_offsets[hook_idx + 1]``.
    4. For each operation:
       a. Evaluates the RPN condition (``when`` clause); skips if unmet.
       b. Reads genotype / age / sex selectors (CSR ranges).
       c. For each selected (sex, age, genotype) cell, computes a target
          count from the operation type and applies it via
          ``_apply_target_with_sperm`` or ``_apply_target_without_sperm``.
       d. For ``stop_if_*`` operations, aggregates the selected cells and
          returns ``RESULT_STOP`` if the threshold is met.
    5. Returns ``RESULT_CONTINUE`` if all operations completed normally.

    This function was extracted from the inner loop of
    ``execute_csr_event_arrays`` so that ``compile.codegen.compile_unified_event_hook``
    can call individual CSR hooks at specific positions in a priority-ordered
    schedule, interleaved with njit function calls.

    Returns:
        ``RESULT_CONTINUE`` (0) — all ops executed normally.
        ``RESULT_SKIP`` (0) — hook not applicable (wrong deme or OOB).
        ``RESULT_STOP`` (1) — a ``stop_if_*`` operation triggered.
    """
    # Guard: bounds check.
    if hook_idx < 0 or hook_idx >= n_hooks:
        return RESULT_SKIP

    # Guard: deme selector.  Encoding: 0=ANY, 1=SINGLE, 2=RANGE, 3=LIST.
    if not njit_deme_selector_matches(
        deme_selector_types[hook_idx],
        deme_selector_offsets[hook_idx],
        deme_selector_offsets[hook_idx + 1],
        deme_selector_data,
        deme_id,
    ):
        return RESULT_SKIP

    # op_offsets is a prefix-sum array: op_offsets[i] is the start index of
    # hook i's operations in the flattened op_*_data arrays.
    op_start = op_offsets[hook_idx]
    op_end = op_offsets[hook_idx + 1]

    for op_idx in range(op_start, op_end):
        # ---- Condition evaluation ----
        cond_start = condition_offsets_data[op_idx]
        cond_end = condition_offsets_data[op_idx + 1]

        if not _eval_csr_condition_program(
            condition_types_data,
            condition_params_data,
            cond_start,
            cond_end,
            tick,
        ):
            continue  # Condition not met — skip this operation.

        op_type = op_types_data[op_idx]
        param = params_data[op_idx]

        # ---- Genotype / age / sex selectors (CSR ranges) ----
        gidx_start = gidx_offsets_data[op_idx]
        gidx_end = gidx_offsets_data[op_idx + 1]
        age_start = age_offsets_data[op_idx]
        age_end = age_offsets_data[op_idx + 1]

        # sex_masks_data is flat: [f0, m0, f1, m1, ...].
        sex_mask_idx = op_idx * 2
        sex_female = sex_masks_data[sex_mask_idx]
        sex_male = sex_masks_data[sex_mask_idx + 1]

        # Mutation ops (0..5): iterate sex × age × genotype, with prange
        # on the innermost gidx loop.  Each (sex, age, gidx) cell is
        # independent — different gidx values write to distinct rows of
        # individual_count and sperm_storage, so no data races.
        #
        # Stop ops (6..9) are handled separately below with a serial
        # reduction — prange is NOT used there.
        if op_type <= _OP_SAMPLE:
            for sex_idx in range(2):
                if sex_idx == 0 and not sex_female:
                    continue
                if sex_idx == 1 and not sex_male:
                    continue

                for age_idx_ptr in range(age_start, age_end):
                    age = age_data[age_idx_ptr]

                    for gidx_ptr in prange(gidx_start, gidx_end):
                        gidx = gidx_data[gidx_ptr]
                        current = individual_count[sex_idx, age, gidx]

                        # Compute target count from operation type.
                        if op_type == _OP_SCALE:
                            target = max(0.0, current * param)
                        elif op_type == _OP_SET:
                            target = max(0.0, param)
                        elif op_type == _OP_ADD:
                            target = max(0.0, current + param)
                        elif op_type == _OP_SUBTRACT:
                            target = max(0.0, current - param)
                        elif op_type == _OP_KILL:
                            target = max(0.0, current * (1.0 - param))
                        elif op_type == _OP_SAMPLE:
                            target = min(current, max(0.0, param))
                        else:
                            target = current

                        if sex_idx == 0 and has_sperm_storage:
                            individual_count[sex_idx, age, gidx] = _apply_target_with_sperm(
                                current,
                                target,
                                sperm_storage[age, gidx, :],
                                stochastic,
                                continuous_sampling,
                            )
                        else:
                            individual_count[sex_idx, age, gidx] = _apply_target_without_sperm(
                                current,
                                target,
                                stochastic,
                                continuous_sampling,
                            )

        # ---- STOP_IF: aggregate selected cells, check threshold ----
        if op_type in (_OP_STOP_IF_ZERO, _OP_STOP_IF_BELOW, _OP_STOP_IF_ABOVE):
            selected_total = 0.0
            for sex_idx in range(2):
                if sex_idx == 0 and not sex_female:
                    continue
                if sex_idx == 1 and not sex_male:
                    continue
                for age_idx_ptr in range(age_start, age_end):
                    age = age_data[age_idx_ptr]
                    for gidx_ptr in range(gidx_start, gidx_end):
                        gidx = gidx_data[gidx_ptr]
                        selected_total += individual_count[sex_idx, age, gidx]

            if op_type == _OP_STOP_IF_ZERO and selected_total <= 0.0:
                return RESULT_STOP
            if op_type == _OP_STOP_IF_BELOW and selected_total < param:
                return RESULT_STOP
            if op_type == _OP_STOP_IF_ABOVE and selected_total > param:
                return RESULT_STOP
        elif op_type == _OP_STOP_IF_EXTINCTION:
            if individual_count.sum() <= 0.0:
                return RESULT_STOP

    return RESULT_CONTINUE


# Public alias — imported by compile.codegen.compile_unified_event_hook and tests.
execute_single_csr_hook = _execute_single_csr_hook


@njit_switch(cache=True)
def execute_csr_event_arrays(
    n_events: int | np.integer[Any],
    n_hooks: int | np.integer[Any],
    hook_offsets: np.ndarray,
    n_ops_list: np.ndarray,  # pyright: ignore[reportUnusedParameter] — positional caller compatibility
    op_offsets: np.ndarray,
    op_types_data: np.ndarray,
    gidx_offsets_data: np.ndarray,
    gidx_data: np.ndarray,
    age_offsets_data: np.ndarray,
    age_data: np.ndarray,
    sex_masks_data: np.ndarray,
    params_data: np.ndarray,
    condition_offsets_data: np.ndarray,
    condition_types_data: np.ndarray,
    condition_params_data: np.ndarray,
    deme_selector_types: np.ndarray,
    deme_selector_offsets: np.ndarray,
    deme_selector_data: np.ndarray,
    event_id: int,
    individual_count: np.ndarray,
    sperm_storage: np.ndarray,
    has_sperm_storage: bool,
    tick: int,
    stochastic: bool,
    continuous_sampling: bool,
    deme_id: int,
) -> int:
    """Execute all hooks for one event from flattened CSR arrays.

    Resolves *event_id* to a hook range via ``hook_offsets``, then calls
    ``_execute_single_csr_hook`` for each hook.  The function signature
    mirrors ``HookProgram`` fields positionally so callers can unpack
    the NamedTuple directly.

    Three-level CSR traversal::

        event_id  →  hook_offsets[event_id]  →  hook range
        hook_idx  →  op_offsets[hook_idx]    →  op range
        op_idx    →  gidx/age/cond offsets   →  cell range

    Returns:
        ``RESULT_CONTINUE`` (0) — all hooks executed normally.
        ``RESULT_STOP`` (1) — a hook returned STOP.
    """
    if event_id < 0 or event_id >= n_events:
        return 0

    # hook_offsets is a prefix-sum: [event_id] is the first hook,
    # [event_id + 1] is one past the last.
    hook_start = hook_offsets[event_id]
    hook_end = hook_offsets[event_id + 1]

    for hook_idx in range(hook_start, hook_end):
        result = _execute_single_csr_hook(
            hook_idx=hook_idx,
            n_hooks=n_hooks,
            op_offsets=op_offsets,
            op_types_data=op_types_data,
            gidx_offsets_data=gidx_offsets_data,
            gidx_data=gidx_data,
            age_offsets_data=age_offsets_data,
            age_data=age_data,
            sex_masks_data=sex_masks_data,
            params_data=params_data,
            condition_offsets_data=condition_offsets_data,
            condition_types_data=condition_types_data,
            condition_params_data=condition_params_data,
            deme_selector_types=deme_selector_types,
            deme_selector_offsets=deme_selector_offsets,
            deme_selector_data=deme_selector_data,
            individual_count=individual_count,
            sperm_storage=sperm_storage,
            has_sperm_storage=has_sperm_storage,
            tick=tick,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            deme_id=deme_id,
        )
        if result != RESULT_CONTINUE:
            return result  # Propagate STOP immediately.

    return RESULT_CONTINUE


def build_hook_program(program: HookProgram) -> HookProgram:
    """Return *program* unchanged (forward-compat hook point).

    Exists as a hook for potential schema upgrades or validation logic.
    Currently a no-op.
    """
    return program


@njit_switch(cache=True)
def execute_csr_event_program_with_state(
    program: HookProgram,
    event_id: int,
    individual_count: np.ndarray,
    sperm_storage: np.ndarray,
    tick: int,
    stochastic: bool,
    has_sperm_storage: bool,
    continuous_sampling: bool,
    deme_id: int = 0,
) -> int:
    """Execute one event from a ``HookProgram``, unpacking all fields.

    Primary adapter between the HookProgram NamedTuple and the flat-array
    interface of ``execute_csr_event_arrays``.  Lifecycle templates call
    this function directly.

    Args:
        program: Compiled HookProgram containing all declarative ops.
        event_id: Which event to execute (EVENT_FIRST=0, EVENT_EARLY=1, …).
        individual_count: 3-D array ``[sex, age, genotype]``, mutated in-place.
        sperm_storage: 3-D array ``[age, genotype, gamete_male]``.  Pass a
            dummy ``(0,0,0)`` array for discrete-generation models.
        tick: Current simulation tick (used for ``when`` clause evaluation).
        stochastic: Whether to use stochastic survival sampling.
        has_sperm_storage: Whether *sperm_storage* contains real data.
        continuous_sampling: Whether to use continuous-Dirichlet sampling.
        deme_id: Deme index for spatial models (0 for panmictic).

    Returns:
        ``RESULT_CONTINUE`` or ``RESULT_STOP``.
    """
    return execute_csr_event_arrays(
        n_events=program.n_events,
        n_hooks=program.n_hooks,
        hook_offsets=program.hook_offsets,
        n_ops_list=program.n_ops_list,
        op_offsets=program.op_offsets,
        op_types_data=program.op_types_data,
        gidx_offsets_data=program.gidx_offsets_data,
        gidx_data=program.gidx_data,
        age_offsets_data=program.age_offsets_data,
        age_data=program.age_data,
        sex_masks_data=program.sex_masks_data,
        params_data=program.params_data,
        condition_offsets_data=program.condition_offsets_data,
        condition_types_data=program.condition_types_data,
        condition_params_data=program.condition_params_data,
        deme_selector_types=program.deme_selector_types,
        deme_selector_offsets=program.deme_selector_offsets,
        deme_selector_data=program.deme_selector_data,
        event_id=event_id,
        individual_count=individual_count,
        sperm_storage=sperm_storage,
        has_sperm_storage=has_sperm_storage,
        tick=tick,
        stochastic=stochastic,
        continuous_sampling=continuous_sampling,
        deme_id=deme_id,
    )


@njit_switch(cache=True)
def execute_csr_event_program(
    program: HookProgram,
    event_id: int,
    individual_count: np.ndarray,
    tick: int,
) -> int:
    """Execute one event with deterministic defaults and no sperm storage.

    Convenience wrapper for quick tests or simple discrete-generation
    setups.  For production use, prefer ``execute_csr_event_program_with_state``
    which exposes the full state flags.
    """
    dummy_sperm = np.zeros((0, 0, 0), dtype=np.float64)
    return execute_csr_event_program_with_state(
        program,
        event_id,
        individual_count,
        dummy_sperm,
        tick,
        stochastic=False,
        has_sperm_storage=False,
        continuous_sampling=False,
        deme_id=0,
    )
