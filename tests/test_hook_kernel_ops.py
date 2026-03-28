#!/usr/bin/env python3
"""Smoke tests for declarative kernel ops (KILL + STOP_IF_*)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hook_dsl import (  # noqa: E402
    COND_ALWAYS,
    RESULT_CONTINUE,
    RESULT_STOP,
    OpType,
    execute_csr_event_arrays,
    parse_condition,
)


def run_single_op(
    op_type: int,
    param: float,
    condition: str | None,
    initial_female: float,
    initial_male: float = 0.0,
    initial_sperm: float = 0.0,
    initial_sperm_row: list[float] | None = None,
    is_stochastic: bool = False,
    use_dirichlet_sampling: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    ind = np.zeros((2, 1, 1), dtype=np.float64)
    if initial_sperm_row is None:
        sperm = np.zeros((1, 1, 1), dtype=np.float64)
        sperm[0, 0, 0] = initial_sperm
    else:
        sperm = np.zeros((1, 1, len(initial_sperm_row)), dtype=np.float64)
        for gm, val in enumerate(initial_sperm_row):
            sperm[0, 0, gm] = float(val)
    ind[0, 0, 0] = initial_female
    ind[1, 0, 0] = initial_male

    if condition is None:
        cond_types = np.array([COND_ALWAYS], dtype=np.int32)
        cond_params = np.array([0], dtype=np.int32)
    else:
        cond_types, cond_params = parse_condition(condition)

    result = execute_csr_event_arrays(
        n_events=np.int32(1),
        n_hooks=np.int32(1),
        hook_offsets=np.array([0, 1], dtype=np.int32),
        n_ops_list=np.array([1], dtype=np.int32),
        op_offsets=np.array([0, 1], dtype=np.int32),
        op_types_data=np.array([op_type], dtype=np.int32),
        gidx_offsets_data=np.array([0, 1], dtype=np.int32),
        gidx_data=np.array([0], dtype=np.int32),
        age_offsets_data=np.array([0, 1], dtype=np.int32),
        age_data=np.array([0], dtype=np.int32),
        sex_masks_data=np.array([True, True], dtype=np.bool_),
        params_data=np.array([param], dtype=np.float64),
        condition_offsets_data=np.array([0, len(cond_types)], dtype=np.int32),
        condition_types_data=cond_types,
        condition_params_data=cond_params,
        deme_selector_types=np.array([0], dtype=np.int32),  # 0=ANY ("*")
        deme_selector_offsets=np.array([0, 0], dtype=np.int32),
        deme_selector_data=np.array([], dtype=np.int32),
        event_id=0,
        individual_count=ind,
        sperm_storage=sperm,
        has_sperm_storage=True,
        tick=10,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling,
        deme_id=0
    )
    return int(result), ind, sperm


def assert_close(actual: float, expected: float, eps: float = 1e-9) -> None:
    assert abs(actual - expected) <= eps, f"Expected {expected}, got {actual}"


if __name__ == "__main__":
    # KILL: deterministic scaling count *= (1 - p)
    result, ind, sperm = run_single_op(
        int(OpType.KILL),
        0.25,
        None,
        initial_female=8.0,
        initial_sperm=4.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 6.0)
    assert_close(float(sperm[0, 0, 0]), 3.0)

    # KILL gated by composed condition
    result, ind, _ = run_single_op(
        int(OpType.KILL),
        0.5,
        "tick >= 10 and not (tick == 14)",
        initial_female=4.0,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 2.0)

    # KILL stochastic: female survivors are sampled, sperm follows same ratio
    np.random.seed(123)
    result, ind, sperm = run_single_op(
        int(OpType.KILL),
        0.5,
        None,
        initial_female=10.0,
        initial_sperm=6.0,
        is_stochastic=True,
    )
    assert result == RESULT_CONTINUE
    survivors = float(ind[0, 0, 0])
    assert 0.0 <= survivors <= 10.0
    sampled_sperm = float(sperm[0, 0, 0])
    assert 0.0 <= sampled_sperm <= 6.0
    assert survivors + 1e-9 >= sampled_sperm

    # KILL stochastic + sperm categories: female survivors should equal
    # sampled_sperm_sum + sampled_virgins (algorithm-consistent order).
    np.random.seed(7)
    result, ind, sperm = run_single_op(
        int(OpType.KILL),
        0.3,
        None,
        initial_female=9.0,
        initial_sperm_row=[2.0, 1.0, 0.0],
        is_stochastic=True,
    )
    assert result == RESULT_CONTINUE
    sampled_sperm_sum = float(sperm[0, 0, :].sum())
    sampled_total = float(ind[0, 0, 0])
    assert sampled_total + 1e-9 >= sampled_sperm_sum

    # Continuous approximation branch should produce finite non-negative values.
    np.random.seed(11)
    result, ind, sperm = run_single_op(
        int(OpType.KILL),
        0.4,
        None,
        initial_female=7.5,
        initial_sperm=2.5,
        is_stochastic=True,
        use_dirichlet_sampling=True,
    )
    assert result == RESULT_CONTINUE
    assert float(ind[0, 0, 0]) >= 0.0
    assert float(sperm[0, 0, 0]) >= 0.0

    # SCALE (>1): equivalent to reverse-kill growth, add virgins only
    result, ind, sperm = run_single_op(
        int(OpType.SCALE),
        1.5,
        None,
        initial_female=4.0,
        initial_sperm=1.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 6.0)
    assert_close(float(sperm[0, 0, 0]), 1.0)

    # SCALE (<1): equivalent to kill-like reduction on sperm+virgins
    result, ind, sperm = run_single_op(
        int(OpType.SCALE),
        0.5,
        None,
        initial_female=8.0,
        initial_sperm=4.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 4.0)
    assert_close(float(sperm[0, 0, 0]), 2.0)

    # SUBTRACT deterministic updates sperm via the same target logic.
    result, ind, sperm = run_single_op(
        int(OpType.SUBTRACT),
        3.0,
        None,
        initial_female=10.0,
        initial_sperm=5.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 7.0)
    assert_close(float(sperm[0, 0, 0]), 3.5)

    # SAMPLE deterministic: cap count and shrink sperm row proportionally when reducing.
    result, ind, sperm = run_single_op(
        int(OpType.SAMPLE),
        6.0,
        None,
        initial_female=10.0,
        initial_sperm=4.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 6.0)
    assert_close(float(sperm[0, 0, 0]), 2.4)

    # SET deterministic: >current behaves like add virgins, <current like subtract.
    result, ind, sperm = run_single_op(
        int(OpType.SET),
        8.0,
        None,
        initial_female=5.0,
        initial_sperm=2.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 8.0)
    assert_close(float(sperm[0, 0, 0]), 2.0)

    result, ind, sperm = run_single_op(
        int(OpType.SET),
        2.0,
        None,
        initial_female=8.0,
        initial_sperm=4.0,
        is_stochastic=False,
    )
    assert result == RESULT_CONTINUE
    assert_close(float(ind[0, 0, 0]), 2.0)
    assert_close(float(sperm[0, 0, 0]), 1.0)

    # SUBTRACT stochastic with sperm categories should keep sampled total >= sperm sum.
    np.random.seed(19)
    result, ind, sperm = run_single_op(
        int(OpType.SUBTRACT),
        2.0,
        None,
        initial_female=9.0,
        initial_sperm_row=[2.0, 1.0, 1.0],
        is_stochastic=True,
    )
    assert result == RESULT_CONTINUE
    assert float(ind[0, 0, 0]) + 1e-9 >= float(sperm[0, 0, :].sum())

    # STOP_IF_ZERO
    result, _, _ = run_single_op(int(OpType.STOP_IF_ZERO), 0.0, None, initial_female=0.0)
    assert result == RESULT_STOP

    # STOP_IF_BELOW
    result, _, _ = run_single_op(int(OpType.STOP_IF_BELOW), 5.0, None, initial_female=4.0)
    assert result == RESULT_STOP

    # STOP_IF_ABOVE
    result, _, _ = run_single_op(int(OpType.STOP_IF_ABOVE), 5.0, None, initial_female=6.0)
    assert result == RESULT_STOP

    # STOP_IF_EXTINCTION
    result, _, _ = run_single_op(int(OpType.STOP_IF_EXTINCTION), 0.0, None, initial_female=0.0)
    assert result == RESULT_STOP

    # STOP_IF_ZERO with false condition should not stop
    result, _, _ = run_single_op(
        int(OpType.STOP_IF_ZERO),
        0.0,
        "tick == 999 or tick == 1000",
        initial_female=0.0,
    )
    assert result == RESULT_CONTINUE

    print("All hook kernel-op tests passed.")
