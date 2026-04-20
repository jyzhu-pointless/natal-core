"""CSR execution kernels and Python-level hook executor.

Runtime flow for one event:

1) Evaluate declarative CSR plans in njit kernels (fast data path)
2) Execute compiled custom ``njit_fn`` hooks
3) Execute Python wrappers (only when Numba mode allows it)

The hot loop is intentionally array-driven and avoids Python objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from natal import numba_compat as nbc
from natal.numba_utils import njit_switch

from .types import (
    COND_OP_AND,
    COND_OP_NOT,
    COND_OP_OR,
    COND_TICK_GT,
    EVENT_ID_MAP,
    NUM_EVENTS,
    RESULT_CONTINUE,
    RESULT_STOP,
    CompiledHookDescriptor,
    DemeSelector,
    HookProgram,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation


def deme_selector_matches(selector: DemeSelector, deme_id: int) -> bool:
    """Return whether one deme id should execute under ``selector``.

    Supported forms:
    - "*" for all demes
    - int for one deme
    - list/tuple/range for a set of demes
    """
    if selector == "*":
        return True
    if isinstance(selector, int):
        return selector == deme_id
    if isinstance(selector, range):
        return deme_id in selector
    return deme_id in selector


@njit_switch(cache=True)
def njit_deme_selector_matches(sel_type: int, start: int, end: int, data: np.ndarray, deme_id: int) -> bool:
    """Numba-compatible version of deme_selector_matches.

    Types: 0=ANY, 1=SINGLE, 2=RANGE, 3=LIST
    """
    if sel_type == 0:  # ANY ("*")
        return True
    if sel_type == 1:  # SINGLE (int)
        return data[start] == deme_id
    if sel_type == 2:  # RANGE (range)
        return deme_id >= data[start] and deme_id < data[start + 1]
    if sel_type == 3:  # LIST (list/tuple)
        if start >= end:
            return False
        for i in range(start, end):
            if data[i] == deme_id:
                return True
        return False
    return True


@njit_switch(cache=True)
def _check_csr_condition(cond_type: int, cond_param: int, tick: int) -> bool:
    """Evaluate one atomic condition token.

    Logical operators are not handled here and are expected to be evaluated by
    ``_eval_csr_condition_program``.
    """
    if cond_type == 0:
        return True
    if cond_type == 1:
        return tick == cond_param
    if cond_type == 2:
        return cond_param > 0 and tick % cond_param == 0
    if cond_type == 3:
        return tick >= cond_param
    if cond_type == 4:
        return tick < cond_param
    if cond_type == 5:
        return tick <= cond_param
    if cond_type == 6:
        return tick > cond_param
    if cond_type == 100:
        return False
    if cond_type == 101:
        return False
    if cond_type == 102:
        return False
    return True


@njit_switch(cache=True)
def _eval_csr_condition_program(
    cond_types: np.ndarray,
    cond_params: np.ndarray,
    cond_start: int,
    cond_end: int,
    tick: int,
) -> bool:
    """Evaluate an RPN condition program span for one operation.

    The condition program is stored in flattened arrays; ``cond_start`` and
    ``cond_end`` define the current operation's token span.
    """
    max_len = cond_end - cond_start
    if max_len <= 0:
        return True

    # int8 stack keeps memory tiny and works well in njit mode.
    stack = np.zeros(max_len + 1, dtype=np.int8)
    top = 0

    for idx in range(cond_start, cond_end):
        token_type = cond_types[idx]
        token_param = cond_params[idx]

        if token_type <= COND_TICK_GT:
            val = 1 if _check_csr_condition(token_type, token_param, tick) else 0
            stack[top] = val
            top += 1
            continue

        if token_type == COND_OP_NOT:
            if top < 1:
                return False
            stack[top - 1] = 0 if stack[top - 1] else 1
            continue

        if token_type == COND_OP_AND:
            if top < 2:
                return False
            rhs = stack[top - 1]
            lhs = stack[top - 2]
            top -= 2
            stack[top] = 1 if (lhs and rhs) else 0
            top += 1
            continue

        if token_type == COND_OP_OR:
            if top < 2:
                return False
            rhs = stack[top - 1]
            lhs = stack[top - 2]
            top -= 2
            stack[top] = 1 if (lhs or rhs) else 0
            top += 1
            continue

        return False

    if top != 1:
        return False
    return stack[0] != 0


# Public alias for tests/interop layers.
eval_csr_condition_program = _eval_csr_condition_program


@njit_switch(cache=True)
def _sample_survivors(
    n_base: float,
    survival_prob: float,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> float:
    """Sample survivors using deterministic or stochastic policy."""
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
    """Apply target count update for male branch or no-sperm populations."""
    if stochastic_flag and not dirichlet_flag:
        current_count = float(round(current_count))

    if target_count >= current_count:
        return target_count
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
    """Apply target update for female branch while keeping sperm row consistent.

    When reducing female count, we scale or sample sperm categories with the
    same survival rate so sperm storage remains coherent with female counts.
    """
    if stochastic_flag and not dirichlet_flag:
        current_count = float(round(current_count))

    if target_count >= current_count:
        return target_count

    if current_count <= 0.0:
        for gm_idx in range(sperm_row.shape[0]):
            sperm_row[gm_idx] = 0.0
        return 0.0

    survival_prob = max(0.0, min(1.0, target_count / current_count))

    if not stochastic_flag:
        ratio = survival_prob
        for gm_idx in range(sperm_row.shape[0]):
            sperm_row[gm_idx] *= ratio
        return target_count

    n_f_raw = float(current_count)

    total_sperm_count = 0.0
    for gm_idx in range(sperm_row.shape[0]):
        total_sperm_count += float(sperm_row[gm_idx])

    # Validate on raw mass first, then convert to sampling counts.
    n_virgins_raw = n_f_raw - total_sperm_count
    if n_virgins_raw >= -nbc.EPS:
        # Prevent negative virgins.
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


@njit_switch(cache=True)
def execute_csr_event_arrays(
    n_events: int | np.integer[Any],
    n_hooks: int | np.integer[Any],
    hook_offsets: np.ndarray,
    n_ops_list: np.ndarray,
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
    is_stochastic: bool,
    use_continuous_sampling: bool,
    deme_id: int,
) -> int:
    """Execute one event from flattened CSR arrays.

    Inputs are plain arrays extracted from ``HookProgram``. This function is
    the hottest part of declarative hook runtime.
    """
    if event_id < 0 or event_id >= n_events:
        return 0

    # Event span -> hook span -> op span (three-level CSR traversal)
    hook_start = hook_offsets[event_id]
    hook_end = hook_offsets[event_id + 1]

    for hook_idx in range(hook_start, hook_end):
        if hook_idx < 0 or hook_idx >= n_hooks:
            continue

        # Filtering by deme_id using serialized selector data
        if not njit_deme_selector_matches(
            deme_selector_types[hook_idx],
            deme_selector_offsets[hook_idx],
            deme_selector_offsets[hook_idx + 1],
            deme_selector_data,
            deme_id,
        ):
            continue

        op_start = op_offsets[hook_idx]
        op_end = op_offsets[hook_idx + 1]

        for op_idx in range(op_start, op_end):
            cond_start = condition_offsets_data[op_idx]
            cond_end = condition_offsets_data[op_idx + 1]

            if not _eval_csr_condition_program(
                condition_types_data,
                condition_params_data,
                cond_start,
                cond_end,
                tick,
            ):
                continue

            op_type = op_types_data[op_idx]
            param = params_data[op_idx]

            gidx_start = gidx_offsets_data[op_idx]
            gidx_end = gidx_offsets_data[op_idx + 1]
            age_start = age_offsets_data[op_idx]
            age_end = age_offsets_data[op_idx + 1]

            sex_mask_idx = op_idx * 2
            sex_female = sex_masks_data[sex_mask_idx]
            sex_male = sex_masks_data[sex_mask_idx + 1]

            for sex_idx in range(2):
                if sex_idx == 0 and not sex_female:
                    continue
                if sex_idx == 1 and not sex_male:
                    continue

                for age_idx_ptr in range(age_start, age_end):
                    age = age_data[age_idx_ptr]

                    for gidx_ptr in range(gidx_start, gidx_end):
                        gidx = gidx_data[gidx_ptr]
                        current = individual_count[sex_idx, age, gidx]

                        # Convert each operation to a target count first, then
                        # route through one unified update function so survival
                        # semantics are consistent across operators.
                        if op_type == 0:     # Op.scale
                            target = max(0.0, current * param)
                        elif op_type == 1:   # Op.set
                            target = max(0.0, param)
                        elif op_type == 2:   # Op.add
                            target = max(0.0, current + param)
                        elif op_type == 3:   # Op.subtract
                            target = max(0.0, current - param)
                        elif op_type == 4:   # Op.kill
                            target = max(0.0, current * (1.0 - param))
                        elif op_type == 5:   # Op.sample
                            target = min(current, max(0.0, param))
                        else:
                            target = current

                        if op_type <= 5:   # Op.scale, Op.set, Op.add, Op.subtract, Op.kill, Op.sample
                            if sex_idx == 0 and has_sperm_storage:
                                individual_count[sex_idx, age, gidx] = _apply_target_with_sperm(
                                    current,
                                    target,
                                    sperm_storage[age, gidx, :],
                                    is_stochastic,
                                    use_continuous_sampling,
                                )
                            else:
                                individual_count[sex_idx, age, gidx] = _apply_target_without_sperm(
                                    current,
                                    target,
                                    is_stochastic,
                                    use_continuous_sampling,
                                )

            # STOP_IF_* operators aggregate selected cells and may short-circuit
            # event execution with RESULT_STOP.
            if op_type == 6 or op_type == 7 or op_type == 8:   # Op.stop_if_zero, Op.stop_if_below, Op.stop_if_above
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

                if op_type == 6 and selected_total <= 0.0:
                    return RESULT_STOP
                if op_type == 7 and selected_total < param:
                    return RESULT_STOP
                if op_type == 8 and selected_total > param:
                    return RESULT_STOP
            elif op_type == 9:   # Op.stop_if_extinction
                if individual_count.sum() <= 0.0:
                    return RESULT_STOP

    return RESULT_CONTINUE


def build_hook_program(program: HookProgram) -> HookProgram:
    """Compatibility hook for future HookProgram validation/migration."""
    return program


@njit_switch(cache=True)
def execute_csr_event_program_with_state(
    program: HookProgram,
    event_id: int,
    individual_count: np.ndarray,
    sperm_storage: np.ndarray,
    tick: int,
    is_stochastic: bool,
    has_sperm_storage: bool,
    use_continuous_sampling: bool,
    deme_id: int = 0,
) -> int:
    """Execute event directly from ``HookProgram`` while exposing state flags."""
    return execute_csr_event_arrays(
        program.n_events,
        program.n_hooks,
        program.hook_offsets,
        program.n_ops_list,
        program.op_offsets,
        program.op_types_data,
        program.gidx_offsets_data,
        program.gidx_data,
        program.age_offsets_data,
        program.age_data,
        program.sex_masks_data,
        program.params_data,
        program.condition_offsets_data,
        program.condition_types_data,
        program.condition_params_data,
        program.deme_selector_types,
        program.deme_selector_offsets,
        program.deme_selector_data,
        event_id,
        individual_count,
        sperm_storage,
        has_sperm_storage,
        tick,
        is_stochastic,
        use_continuous_sampling,
        deme_id,
    )


@njit_switch(cache=True)
def execute_csr_event_program(
    program: HookProgram,
    event_id: int,
    individual_count: np.ndarray,
    tick: int,
) -> int:
    """Compatibility wrapper with deterministic defaults and no sperm storage."""
    dummy_sperm = np.zeros((0, 0, 0), dtype=np.float64)
    return execute_csr_event_program_with_state(
        program,
        event_id,
        individual_count,
        dummy_sperm,
        tick,
        False,
        False,
        False,  # use_continuous_sampling
        0,      # deme_id
    )


class HookExecutor:
    """Python-layer coordinator for all hook execution modes.

    This class is used by population event dispatch where both njit and Python
    callback hooks must coexist around the declarative CSR core.
    """

    def __init__(
        self,
        registry: HookProgram,
        hooks_by_event: Dict[int, List[CompiledHookDescriptor]],
    ) -> None:
        self.registry = registry
        self.hooks_by_event = hooks_by_event

    @staticmethod
    def from_compiled_hooks(
        registry: HookProgram,
        compiled_hooks: List[CompiledHookDescriptor],
    ) -> HookExecutor:
        """Group descriptors by event and sort by priority."""
        from collections import defaultdict

        hooks_by_event: Dict[int, List[CompiledHookDescriptor]] = defaultdict(list)
        for desc in compiled_hooks:
            event_id = EVENT_ID_MAP.get(desc.event)
            if event_id is not None:
                if desc.plan is not None or desc.njit_fn is not None or desc.py_wrapper is not None:
                    hooks_by_event[event_id].append(desc)

        for event_id in hooks_by_event:
            hooks_by_event[event_id].sort(key=lambda x: x.priority)

        return HookExecutor(registry, dict(hooks_by_event))

    def execute_event(
        self,
        event_id: int,
        population: BasePopulation[Any],
        tick: int,
        deme_id: int = 0,
    ) -> int:
        """Run one event with priority ordering across hook types."""
        if event_id < 0 or event_id >= NUM_EVENTS:
            return RESULT_CONTINUE

        ind_count = population.state.individual_count

        # Prepare optional sperm-storage arrays for kernels that require them.
        sperm_store = getattr(population.state, "sperm_storage", None)
        has_sperm_storage = sperm_store is not None and sperm_store.size > 0
        if not has_sperm_storage:
            sperm_store = np.zeros((0, 0, 0), dtype=np.float64)
        assert sperm_store is not None
        is_stochastic = bool(getattr(getattr(population, "_config", None), "is_stochastic", False))
        use_continuous_sampling = bool(
            getattr(getattr(population, "_config", None), "use_continuous_sampling", False)
        )

        from ..numba_utils import NUMBA_ENABLED

        # Unified timeline: descriptors are pre-sorted by priority (stable).
        for desc in self.hooks_by_event.get(event_id, []):
            if not deme_selector_matches(desc.deme_selector, deme_id):
                continue

            if desc.plan is not None:
                result = execute_csr_event_arrays(
                    np.int32(1),
                    np.int32(1),
                    np.array([0, 1], dtype=np.int32),
                    np.array([desc.plan.n_ops], dtype=np.int32),
                    np.array([0, desc.plan.n_ops], dtype=np.int32),
                    desc.plan.op_types,
                    desc.plan.gidx_offsets,
                    desc.plan.gidx_data,
                    desc.plan.age_offsets,
                    desc.plan.age_data,
                    desc.plan.sex_masks.flatten(),
                    desc.plan.params,
                    desc.plan.condition_offsets,
                    desc.plan.condition_types,
                    desc.plan.condition_params,
                    np.array([0], dtype=np.int32),   # selector ANY
                    np.array([0, 0], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    0,
                    ind_count,
                    sperm_store,
                    has_sperm_storage,
                    tick,
                    is_stochastic,
                    use_continuous_sampling,
                    deme_id,
                )
                if result == RESULT_STOP:
                    return RESULT_STOP

            if desc.njit_fn is not None:
                try:
                    result = desc.njit_fn(ind_count, tick, deme_id)
                    if result == RESULT_STOP:
                        return RESULT_STOP
                except Exception as e:
                    raise RuntimeError(f"Error in njit hook '{desc.name}': {e}") from e

            if desc.py_wrapper is not None and desc.njit_fn is None:
                if NUMBA_ENABLED:
                    raise RuntimeError(
                        f"Python py_wrapper hook '{desc.name}' is not allowed when Numba is enabled."
                    )
                try:
                    desc.py_wrapper(population)
                except Exception as e:
                    raise RuntimeError(f"Error in py_wrapper hook '{desc.name}': {e}") from e

        return RESULT_CONTINUE

    def get_hooks_for_event(self, event_id: int) -> List[CompiledHookDescriptor]:
        return self.hooks_by_event.get(event_id, [])


def run_discrete_with_hooks(
    population: Any,
    *,
    n_steps: int,
    record_every: int,
    finish: bool,
    clear_history_on_start: bool,
) -> Any:
    """Run one discrete population through HookExecutor-coordinated timeline."""
    from natal.kernels import simulation_kernels as sk
    from natal.population_state import DiscretePopulationState

    population.ensure_hook_executor()

    if clear_history_on_start:
        population.clear_history()

    if record_every > 0 and (population._tick % record_every == 0):
        population.create_history_snapshot()

    was_stopped = False
    for _ in range(n_steps):
        if population.trigger_event("first", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        population._state_nn.individual_count[:] = sk.run_discrete_reproduction(
            population._state_nn.individual_count,
            population._config_nn,
        )

        if population.trigger_event("early", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        population._state_nn.individual_count[:] = sk.run_discrete_survival(
            population._state_nn.individual_count,
            population._config_nn,
        )

        if population.trigger_event("late", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        population._state_nn.individual_count[:] = sk.run_discrete_aging(
            population._state_nn.individual_count,
        )

        population._tick += 1
        population._state = DiscretePopulationState(
            n_tick=int(population._tick),
            individual_count=population._state_nn.individual_count,
        )

        if record_every > 0 and (population._tick % record_every == 0):
            population.create_history_snapshot()

    if was_stopped:
        population._finished = True
        population.trigger_event("finish")
    elif finish:
        population.finish_simulation()

    return population


def run_age_structured_with_hooks(
    population: Any,
    *,
    n_steps: int,
    record_every: int,
    finish: bool,
    clear_history_on_start: bool,
) -> Any:
    """Run one age-structured population through HookExecutor timeline."""
    from natal.kernels import simulation_kernels as sk
    from natal.population_state import PopulationState

    population.ensure_hook_executor()

    if clear_history_on_start:
        population.clear_history()

    if record_every > 0 and (population._tick % record_every == 0):
        population.create_history_snapshot()

    was_stopped = False
    for _ in range(n_steps):
        if population.trigger_event("first", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        ind_next, sperm_next = sk.run_reproduction(
            population._state_nn.individual_count,
            population._state_nn.sperm_storage,
            population._config_nn,
        )
        population._state_nn.individual_count[:] = ind_next
        population._state_nn.sperm_storage[:] = sperm_next

        if population.trigger_event("early", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        ind_next, sperm_next = sk.run_survival(
            population._state_nn.individual_count,
            population._state_nn.sperm_storage,
            population._config_nn,
        )
        population._state_nn.individual_count[:] = ind_next
        population._state_nn.sperm_storage[:] = sperm_next

        if population.trigger_event("late", deme_id=0) != RESULT_CONTINUE:
            was_stopped = True
            break

        ind_next, sperm_next = sk.run_aging(
            population._state_nn.individual_count,
            population._state_nn.sperm_storage,
            population._config_nn,
        )
        population._state_nn.individual_count[:] = ind_next
        population._state_nn.sperm_storage[:] = sperm_next

        population._tick += 1
        population._state = PopulationState(
            n_tick=int(population._tick),
            individual_count=population._state_nn.individual_count,
            sperm_storage=population._state_nn.sperm_storage,
        )

        if record_every > 0 and (population._tick % record_every == 0):
            population.create_history_snapshot()

    if was_stopped:
        population._finished = True
        population.trigger_event("finish")
    elif finish:
        population.finish_simulation()

    return population


