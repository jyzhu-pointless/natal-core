"""CSR interpreter — reference-oracle declarative hook kernel.

Role since slice 4 (26 decisions): this module is the **reference oracle**
for declarative CSR hooks.  The Rust ``hooks.rs`` interpreter is the
primary executor for the Rust backend; this Python interpreter
serves the reference backend (inside the lifecycle
wrappers via ``execute_csr_event_program_with_state``, and from the
Python orchestration layer via ``HookExecutor``) and provides the
parity baseline the Rust executor is tested against.  Full retirement of
the duplicate interpreter is deferred to slice 6.

Return value protocol
---------------------
Every hook execution function returns an int:

``RESULT_CONTINUE`` (0)
    All operations completed; proceed to the next hook.
``RESULT_SKIP`` (0, alias)
    Hook not applicable in this context (e.g. wrong deme).  Same runtime
    behavior as ``RESULT_CONTINUE``; the distinct name is for readability.
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

from typing import Any, Optional

import numpy as np

# prange removed — parallel=True on _execute_single_csr_hook was causing
# OpenMP overhead (4-5x slowdown) for small genotype counts. See #perf.
import natal.backends.reference.sampling as sampling
from natal.frontend.hooks.types import (
    COND_OP_AND,
    COND_OP_NOT,
    COND_OP_OR,
    COND_TICK_GT,
    RESULT_CONTINUE,
    RESULT_SKIP,
    RESULT_STOP,
    RPN_ADD,
    RPN_DIV,
    RPN_LITERAL,
    RPN_MUL,
    RPN_PARAM,
    RPN_SUB,
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
def njit_deme_selector_matches(
    sel_type: int,
    start: int,
    end: int,
    data: np.ndarray,
    deme_id: int,
) -> bool:
    """Deme selector check against serialized arrays.

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
# When target < current, removal is modeled as *survival* — each individual
# survives with probability = target / current.  This keeps ``Op.scale(0.5)``
# semantically identical to "50 % survival" and ensures sperm storage scaling
# stays coherent.
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
            return sampling.continuous_binomial(n_base, survival_prob)
        return float(sampling.binomial(int(round(n_base)), survival_prob))
    return n_base * survival_prob
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
        sperm_row: ``sperm_storage[age, zidx, :]`` — per-genotype-male sperm
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
    if n_virgins_raw >= -sampling.EPS:
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
def _eval_rpn_value(
    rpn_kinds: np.ndarray,
    rpn_payload: np.ndarray,
    rpn_start: int,
    rpn_end: int,
    sp_literals: np.ndarray,
    eco_values: np.ndarray,
) -> float:
    """Evaluate one ``Op.set_param`` RPN value program with a stack machine.

    Operands push either a literal (pool-indexed) or the *current* value
    of an ecology parameter (``ECO_PARAM_NAMES``-indexed slot of
    *eco_values*); binary operators pop two and push the result.  The
    token stream was depth-validated at compile time, so the machine can
    trust its input.

    Division by zero follows explicit IEEE-754 semantics (``x/0`` = ±inf,
    ``0/0`` = nan) instead of raising: the identical branch runs in the
    kernel and the Rust interpreter, and exceptions would diverge
    across backends.

    Args:
        rpn_kinds: Flattened RPN token-kind array.
        rpn_payload: Flattened per-token payload array.
        rpn_start: First token index of this op's program.
        rpn_end: One past the last token index.
        sp_literals: Shared float64 literal pool.
        eco_values: Current ecology values indexed by ECO param id.

    Returns:
        The evaluated expression value.
    """
    stack = np.zeros(rpn_end - rpn_start + 1, dtype=np.float64)
    top = 0
    for idx in range(rpn_start, rpn_end):
        kind = rpn_kinds[idx]
        if kind == RPN_LITERAL:
            stack[top] = sp_literals[rpn_payload[idx]]
            top += 1
        elif kind == RPN_PARAM:
            stack[top] = eco_values[rpn_payload[idx]]
            top += 1
        else:
            rhs = stack[top - 1]
            lhs = stack[top - 2]
            top -= 1
            if kind == RPN_ADD:
                stack[top - 1] = lhs + rhs
            elif kind == RPN_SUB:
                stack[top - 1] = lhs - rhs
            elif kind == RPN_MUL:
                stack[top - 1] = lhs * rhs
            elif kind == RPN_DIV:
                # Explicit IEEE semantics (see docstring) keeps the
                # Python and Rust interpreters bit-identical.
                if rhs == 0.0:
                    if lhs > 0.0:
                        stack[top - 1] = np.inf
                    elif lhs < 0.0:
                        stack[top - 1] = -np.inf
                    else:
                        stack[top - 1] = np.nan
                else:
                    stack[top - 1] = lhs / rhs
            # Unknown token kinds cannot occur: compile-time validation
            # guarantees only RPN_* tokens enter the stream.
    return stack[0]
def _convert_count(
    n_base: float,
    prob: float,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> float:
    """Return how many of *n_base* individuals convert at *prob*.

    Shares the sampling conventions of ``_sample_survivors`` so
    ``Op.convert`` draws come from the same CSR sampling channel:
    deterministic mode moves ``n * prob`` exactly; stochastic mode uses
    discrete or continuous binomial sampling.

    Args:
        n_base: Count eligible for conversion.
        prob: Per-individual conversion probability.
        stochastic_flag: Whether to sample stochastically.
        dirichlet_flag: Whether continuous sampling is enabled.

    Returns:
        The converted count (continuous in Dirichlet mode).
    """
    if n_base <= 0.0:
        return 0.0
    if stochastic_flag:
        if dirichlet_flag:
            return sampling.continuous_binomial(n_base, prob)
        return float(sampling.binomial(int(round(n_base)), prob))
    return n_base * prob


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
_OP_SET_PARAM = 10
_OP_CONVERT = 11
def _execute_single_csr_hook(
    hook_idx: int,
    n_hooks: int | np.integer[Any],
    op_offsets: np.ndarray,
    op_types_data: np.ndarray,
    zidx_offsets_data: np.ndarray,
    zidx_data: np.ndarray,
    age_offsets_data: np.ndarray,
    age_data: np.ndarray,
    sex_masks_data: np.ndarray,
    params_data: np.ndarray,
    condition_offsets_data: np.ndarray,
    condition_types_data: np.ndarray,
    condition_params_data: np.ndarray,
    sp_param_ids_data: np.ndarray,
    sp_every_data: np.ndarray,
    sp_start_data: np.ndarray,
    rpn_offsets_data: np.ndarray,
    rpn_kinds_data: np.ndarray,
    rpn_payload_data: np.ndarray,
    sp_literals_data: np.ndarray,
    convert_source_z_data: np.ndarray,
    convert_target_z_data: np.ndarray,
    deme_selector_types: np.ndarray,
    deme_selector_offsets: np.ndarray,
    deme_selector_data: np.ndarray,
    individual_count: np.ndarray,
    sperm_storage: Optional[np.ndarray],
    has_sperm_storage: bool,
    tick: int,
    stochastic: bool,
    continuous_sampling: bool,
    deme_id: int,
    eco_values: Optional[np.ndarray] = None,
) -> int:
    """Execute a single CSR hook at global index *hook_idx*.

    This is the **per-hook CSR primitive**.  Given a global index into the
    flattened ``HookProgram`` arrays, it:

    1. Bounds-checks *hook_idx* (returns ``RESULT_SKIP`` if invalid).
    2. Checks the serialized deme selector (returns ``RESULT_SKIP`` if
       *deme_id* doesn't match).
    3. Iterates over the hook's operations — ``op_offsets[hook_idx]``
       to ``op_offsets[hook_idx + 1]``.
    4. For each operation:
       a. Evaluates the RPN condition (``when`` clause); skips if unmet.
       b. Reads genotype / age / sex selectors (CSR ranges).
       c. For each selected (sex, age, genotype) cell, computes a target
          count from the operation type and applies it via
          ``_apply_target_with_sperm`` or ``_apply_target_without_sperm``.
       d. For ``OP_SET_PARAM``: checks the (start, every) schedule,
          evaluates the RPN value program against the *current* ecology
          values, and writes the result into *eco_values* (the caller
          flushes it through the parameter write channel).
       e. For ``OP_CONVERT``: binomially migrates individuals (and, for
          females, every sperm bucket plus the virgin part) from the
          source ZType to the target ZType, conserving totals.
       f. For ``stop_if_*`` operations, aggregates the selected cells and
          returns ``RESULT_STOP`` if the threshold is met.
    5. Returns ``RESULT_CONTINUE`` if all operations completed normally.

    Args:
        eco_values: Live scratch array indexed by ``ECO_PARAM_NAMES``
            position.  ``OP_SET_PARAM`` reads current values and writes
            new values here; ``None`` allocates a local scratch (for
            callers with no set_param ops).

    Returns:
        ``RESULT_CONTINUE`` (0) — all ops executed normally.
        ``RESULT_SKIP`` (0) — hook not applicable (wrong deme or OOB).
        ``RESULT_STOP`` (1) — a ``stop_if_*`` operation triggered.
    """
    if eco_values is None:
        eco_scratch = np.zeros(5, dtype=np.float64)
    else:
        eco_scratch = eco_values

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
        zidx_start = zidx_offsets_data[op_idx]
        zidx_end = zidx_offsets_data[op_idx + 1]
        age_start = age_offsets_data[op_idx]
        age_end = age_offsets_data[op_idx + 1]

        # sex_masks_data is flat: [f0, m0, f1, m1, ...].
        sex_mask_idx = op_idx * 2
        sex_female = sex_masks_data[sex_mask_idx]
        sex_male = sex_masks_data[sex_mask_idx + 1]

        # Mutation ops (0..5): iterate sex × age × genotype serially.
        # Each (sex, age, zidx) cell is independent — different zidx
        # values write to distinct rows of individual_count and
        # sperm_storage, so no data races.
        #
        # Stop ops (6..9) are handled separately below with a serial
        # reduction.
        if op_type <= _OP_SAMPLE:
            for sex_idx in range(2):
                if sex_idx == 0 and not sex_female:
                    continue
                if sex_idx == 1 and not sex_male:
                    continue

                for age_idx_ptr in range(age_start, age_end):
                    age = age_data[age_idx_ptr]

                    for zidx_ptr in range(zidx_start, zidx_end):
                        zidx = zidx_data[zidx_ptr]
                        current = individual_count[sex_idx, age, zidx]

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

                        if sex_idx == 0 and sperm_storage is not None:
                            individual_count[sex_idx, age, zidx] = _apply_target_with_sperm(
                                current,
                                target,
                                sperm_storage[age, zidx, :],
                                stochastic,
                                continuous_sampling,
                            )
                        else:
                            individual_count[sex_idx, age, zidx] = _apply_target_without_sperm(
                                current,
                                target,
                                stochastic,
                                continuous_sampling,
                            )

        # ---- OP_SET_PARAM: schedule check, RPN evaluation, eco write ----
        # The kernel only computes; the caller owns the write-back channel
        # (route dispatch / dirty bridge / params log on Python, session
        # ecology columns on Rust).
        elif op_type == _OP_SET_PARAM:
            start_tick = sp_start_data[op_idx]
            every_ticks = sp_every_data[op_idx]
            if tick >= start_tick and (tick - start_tick) % every_ticks == 0:
                rpn_start = rpn_offsets_data[op_idx]
                rpn_end = rpn_offsets_data[op_idx + 1]
                value = _eval_rpn_value(
                    rpn_kinds_data,
                    rpn_payload_data,
                    rpn_start,
                    rpn_end,
                    sp_literals_data,
                    eco_scratch,
                )
                eco_scratch[sp_param_ids_data[op_idx]] = value

        # ---- OP_CONVERT: one-to-one probabilistic ZType migration ----
        # Males migrate plain counts; females migrate the virgin part and
        # every sperm bucket atomically (female label follows the row,
        # male axis untouched).  Totals are conserved by construction:
        # every moved unit is subtracted from source and added to target.
        elif op_type == _OP_CONVERT:
            src_z = convert_source_z_data[op_idx]
            dst_z = convert_target_z_data[op_idx]
            prob = param
            n_ages_dim = individual_count.shape[1]
            for age in range(n_ages_dim):
                # Males carry no sperm label — plain count migration.
                male_base = individual_count[1, age, src_z]
                moved_male = _convert_count(
                    male_base, prob, stochastic, continuous_sampling
                )
                individual_count[1, age, src_z] -= moved_male
                individual_count[1, age, dst_z] += moved_male

                if sperm_storage is not None and has_sperm_storage:
                    n_male_z = sperm_storage.shape[2]
                    # Virgin count is fixed before the bucket loop: the
                    # loop moves buckets out of the source row, so the
                    # pre-loop row sum is the mated total.
                    sperm_row_sum = 0.0
                    for mz in range(n_male_z):
                        sperm_row_sum += sperm_storage[age, src_z, mz]
                    moved_mated = 0.0
                    for mz in range(n_male_z):
                        bucket = sperm_storage[age, src_z, mz]
                        moved_bucket = _convert_count(
                            bucket, prob, stochastic, continuous_sampling
                        )
                        sperm_storage[age, src_z, mz] -= moved_bucket
                        sperm_storage[age, dst_z, mz] += moved_bucket
                        moved_mated += moved_bucket
                    virgins = individual_count[0, age, src_z] - sperm_row_sum
                    if virgins < 0.0:
                        virgins = 0.0
                    moved_virgin = _convert_count(
                        virgins, prob, stochastic, continuous_sampling
                    )
                    individual_count[0, age, src_z] -= moved_mated + moved_virgin
                    individual_count[0, age, dst_z] += moved_mated + moved_virgin
                else:
                    # Discrete models / no sperm storage: plain migration.
                    female_base = individual_count[0, age, src_z]
                    moved_female = _convert_count(
                        female_base, prob, stochastic, continuous_sampling
                    )
                    individual_count[0, age, src_z] -= moved_female
                    individual_count[0, age, dst_z] += moved_female

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
                    for zidx_ptr in range(zidx_start, zidx_end):
                        zidx = zidx_data[zidx_ptr]
                        selected_total += individual_count[sex_idx, age, zidx]

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


# Public alias — used by tests and external parity checks.
execute_single_csr_hook = _execute_single_csr_hook
def execute_csr_event_arrays(
    n_events: int | np.integer[Any],
    n_hooks: int | np.integer[Any],
    hook_offsets: np.ndarray,
    n_ops_list: np.ndarray,  # pyright: ignore[reportUnusedParameter] — positional caller compatibility
    op_offsets: np.ndarray,
    op_types_data: np.ndarray,
    zidx_offsets_data: np.ndarray,
    zidx_data: np.ndarray,
    age_offsets_data: np.ndarray,
    age_data: np.ndarray,
    sex_masks_data: np.ndarray,
    params_data: np.ndarray,
    condition_offsets_data: np.ndarray,
    condition_types_data: np.ndarray,
    condition_params_data: np.ndarray,
    sp_param_ids_data: np.ndarray,
    sp_every_data: np.ndarray,
    sp_start_data: np.ndarray,
    rpn_offsets_data: np.ndarray,
    rpn_kinds_data: np.ndarray,
    rpn_payload_data: np.ndarray,
    sp_literals_data: np.ndarray,
    convert_source_z_data: np.ndarray,
    convert_target_z_data: np.ndarray,
    deme_selector_types: np.ndarray,
    deme_selector_offsets: np.ndarray,
    deme_selector_data: np.ndarray,
    event_id: int,
    individual_count: np.ndarray,
    sperm_storage: Optional[np.ndarray],
    has_sperm_storage: bool,
    tick: int,
    stochastic: bool,
    continuous_sampling: bool,
    deme_id: int,
    eco_values: Optional[np.ndarray] = None,
) -> int:
    """Execute all hooks for one event from flattened CSR arrays.

    Resolves *event_id* to a hook range via ``hook_offsets``, then calls
    ``_execute_single_csr_hook`` for each hook.  The function signature
    mirrors ``HookProgram`` fields positionally so callers can unpack
    the NamedTuple directly.

    Three-level CSR traversal::

        event_id  →  hook_offsets[event_id]  →  hook range
        hook_idx  →  op_offsets[hook_idx]    →  op range
        op_idx    →  zidx/age/cond offsets   →  cell range

    Args:
        eco_values: Live scratch array indexed by ``ECO_PARAM_NAMES``
            position for ``OP_SET_PARAM`` evaluation; ``None`` allocates
            a local scratch.

    Returns:
        ``RESULT_CONTINUE`` (0) — all hooks executed normally.
        ``RESULT_STOP`` (1) — a hook returned STOP.
    """
    if eco_values is None:
        eco_scratch = np.zeros(5, dtype=np.float64)
    else:
        eco_scratch = eco_values

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
            zidx_offsets_data=zidx_offsets_data,
            zidx_data=zidx_data,
            age_offsets_data=age_offsets_data,
            age_data=age_data,
            sex_masks_data=sex_masks_data,
            params_data=params_data,
            condition_offsets_data=condition_offsets_data,
            condition_types_data=condition_types_data,
            condition_params_data=condition_params_data,
            sp_param_ids_data=sp_param_ids_data,
            sp_every_data=sp_every_data,
            sp_start_data=sp_start_data,
            rpn_offsets_data=rpn_offsets_data,
            rpn_kinds_data=rpn_kinds_data,
            rpn_payload_data=rpn_payload_data,
            sp_literals_data=sp_literals_data,
            convert_source_z_data=convert_source_z_data,
            convert_target_z_data=convert_target_z_data,
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
            eco_values=eco_scratch,
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
def execute_csr_event_program_with_state(
    program: HookProgram,
    event_id: int,
    individual_count: np.ndarray,
    sperm_storage: Optional[np.ndarray],
    tick: int,
    stochastic: bool,
    has_sperm_storage: bool,
    continuous_sampling: bool,
    deme_id: int = 0,
    eco_values: Optional[np.ndarray] = None,
) -> int:
    """Execute one event from a ``HookProgram``, unpacking all fields.

    Primary adapter between the HookProgram NamedTuple and the flat-array
    interface of ``execute_csr_event_arrays``.  Lifecycle templates call
    this function directly.

    Args:
        program: Compiled HookProgram containing all declarative ops.
        event_id: Which event to execute (EVENT_FIRST=0, EVENT_EARLY=1, …).
        individual_count: 3-D array ``[sex, age, genotype]``, mutated in-place.
        sperm_storage: 3-D array ``[age, genotype, gamete_male]`` or
            ``None`` when *has_sperm_storage* is False (discrete models).
        tick: Current simulation tick (used for ``when`` clause evaluation).
        stochastic: Whether to use stochastic survival sampling.
        has_sperm_storage: Whether *sperm_storage* contains real data.
        continuous_sampling: Whether to use continuous-Dirichlet sampling.
        deme_id: Deme index for spatial models (0 for panmictic).
        eco_values: Live scratch array indexed by ``ECO_PARAM_NAMES``
            position; ``OP_SET_PARAM`` reads current values and writes
            new values here.  ``None`` allocates a local scratch.

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
        zidx_offsets_data=program.zidx_offsets_data,
        zidx_data=program.zidx_data,
        age_offsets_data=program.age_offsets_data,
        age_data=program.age_data,
        sex_masks_data=program.sex_masks_data,
        params_data=program.params_data,
        condition_offsets_data=program.condition_offsets_data,
        condition_types_data=program.condition_types_data,
        condition_params_data=program.condition_params_data,
        sp_param_ids_data=program.sp_param_ids,
        sp_every_data=program.sp_every,
        sp_start_data=program.sp_start,
        rpn_offsets_data=program.rpn_offsets,
        rpn_kinds_data=program.rpn_kinds,
        rpn_payload_data=program.rpn_payload,
        sp_literals_data=program.sp_literals,
        convert_source_z_data=program.convert_source_z,
        convert_target_z_data=program.convert_target_z,
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
        eco_values=eco_values,
    )
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
