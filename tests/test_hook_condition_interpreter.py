#!/usr/bin/env python3
"""Smoke tests for hook DSL condition interpreter (and/or/not)."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hook_dsl import _parse_condition, _eval_csr_condition_program  # noqa: E402


def eval_expr(expr: str, tick: int) -> bool:
    cond_types, cond_params = _parse_condition(expr)
    return bool(_eval_csr_condition_program(cond_types, cond_params, 0, len(cond_types), tick))


def expect_true(expr: str, tick: int) -> None:
    assert eval_expr(expr, tick), f"Expected True for tick={tick}: {expr}"


def expect_false(expr: str, tick: int) -> None:
    assert not eval_expr(expr, tick), f"Expected False for tick={tick}: {expr}"


def expect_error(expr: str) -> None:
    try:
        _parse_condition(expr)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for expression: {expr}")


if __name__ == "__main__":
    # Basic atoms
    expect_true("tick == 10", 10)
    expect_false("tick == 10", 9)

    # and / or
    expect_true("tick >= 10 and tick < 20", 12)
    expect_false("tick >= 10 and tick < 20", 22)
    expect_true("tick == 3 or tick == 5", 5)
    expect_false("tick == 3 or tick == 5", 4)

    # not and precedence (not > and > or)
    expect_true("not tick == 3", 4)
    expect_false("not tick == 3", 3)
    expect_true("tick == 1 or tick == 2 and tick < 2", 1)
    expect_false("tick == 1 or tick == 2 and tick < 2", 2)

    # Parentheses
    expect_true("(tick == 1 or tick == 2) and tick < 2", 1)
    expect_false("(tick == 1 or tick == 2) and tick < 2", 2)

    # Mod predicate + composition
    expect_true("tick % 5 == 0 and tick >= 10", 10)
    expect_false("tick % 5 == 0 and tick >= 10", 5)

    # Nested not
    expect_true("not (tick == 1 or tick == 2)", 3)
    expect_false("not (tick == 1 or tick == 2)", 1)

    # Invalid expression cases
    expect_error("tick >= 10 and")
    expect_error("tick >= 10 or or tick < 20")
    expect_error("(tick >= 10")
    expect_error("tick >= 10 and population.get_total_count() < 10")

    print("All hook condition interpreter tests passed.")
