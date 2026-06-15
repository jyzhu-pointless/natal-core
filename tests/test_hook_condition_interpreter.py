#!/usr/bin/env python3
"""Pytest tests for hook DSL condition interpreter (and/or/not).

Tests the full pipeline: atomic parsing, tokenization, RPN conversion,
compile-time validation, and runtime evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.hooks import eval_csr_condition_program, parse_condition
from natal.hooks.declarative import (
    _parse_atomic_condition,
    _parse_condition,
    _to_rpn_condition,
    _tokenize_condition_expr,
)
from natal.hooks.types import (
    COND_ALWAYS,
    COND_OP_AND,
    COND_OP_NOT,
    COND_OP_OR,
    COND_TICK_EQ,
    COND_TICK_GE,
    COND_TICK_GT,
    COND_TICK_LE,
    COND_TICK_LT,
    COND_TICK_MOD,
)

# ── _parse_atomic_condition ──────────────────────────────────────────────────


class TestParseAtomicCondition:
    """Tests for ``_parse_atomic_condition`` -- one predicate string to token."""

    def test_tick_eq(self) -> None:
        assert _parse_atomic_condition("tick == 5") == (COND_TICK_EQ, 5)

    def test_tick_ge(self) -> None:
        assert _parse_atomic_condition("tick >= 10") == (COND_TICK_GE, 10)

    def test_tick_gt(self) -> None:
        assert _parse_atomic_condition("tick > 3") == (COND_TICK_GT, 3)

    def test_tick_le(self) -> None:
        assert _parse_atomic_condition("tick <= 7") == (COND_TICK_LE, 7)

    def test_tick_lt(self) -> None:
        assert _parse_atomic_condition("tick < 2") == (COND_TICK_LT, 2)

    def test_tick_mod(self) -> None:
        assert _parse_atomic_condition("tick % 10 == 0") == (COND_TICK_MOD, 10)

    def test_tick_mod_not_equal_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported atomic condition"):
            _parse_atomic_condition("tick % 10 != 0")

    def test_field_not_tick_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported atomic condition"):
            _parse_atomic_condition("foo > 5")


# ── _tokenize_condition_expr ─────────────────────────────────────────────────


class TestTokenizeConditionExpr:
    """Tests for ``_tokenize_condition_expr`` -- string to token list."""

    def test_single_atom(self) -> None:
        tokens = _tokenize_condition_expr("tick == 5")
        assert tokens == [(COND_TICK_EQ, 5)]

    def test_and_expression(self) -> None:
        tokens = _tokenize_condition_expr("tick >= 10 and tick < 20")
        assert tokens == [
            (COND_TICK_GE, 10),
            (COND_OP_AND, 0),
            (COND_TICK_LT, 20),
        ]

    def test_or_expression(self) -> None:
        tokens = _tokenize_condition_expr("tick == 1 or tick == 2")
        assert tokens == [
            (COND_TICK_EQ, 1),
            (COND_OP_OR, 0),
            (COND_TICK_EQ, 2),
        ]

    def test_not_unary(self) -> None:
        tokens = _tokenize_condition_expr("not tick == 3")
        assert tokens == [
            (COND_OP_NOT, 0),
            (COND_TICK_EQ, 3),
        ]

    def test_parentheses(self) -> None:
        tokens = _tokenize_condition_expr(
            "(tick == 1 or tick == 2) and tick < 2"
        )
        assert tokens == [
            (-(ord("(")), 0),
            (COND_TICK_EQ, 1),
            (COND_OP_OR, 0),
            (COND_TICK_EQ, 2),
            (-(ord(")")), 0),
            (COND_OP_AND, 0),
            (COND_TICK_LT, 2),
        ]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            _tokenize_condition_expr("")

    def test_invalid_syntax_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported condition syntax"):
            _tokenize_condition_expr("foo > 5")


# ── _to_rpn_condition ────────────────────────────────────────────────────────


class TestToRpnCondition:
    """Tests for ``_to_rpn_condition`` -- token list to RPN arrays."""

    def test_single_atom(self) -> None:
        types, params = _to_rpn_condition([(COND_TICK_EQ, 5)])
        assert len(types) == 1
        assert types[0] == COND_TICK_EQ
        assert params[0] == 5

    def test_and_expression_rpn(self) -> None:
        tokens = [
            (COND_TICK_GE, 10),
            (COND_OP_AND, 0),
            (COND_TICK_LT, 20),
        ]
        types, params = _to_rpn_condition(tokens)
        # Infix: GE10 AND LT20  →  RPN: GE10 LT20 AND
        np.testing.assert_array_equal(types, [COND_TICK_GE, COND_TICK_LT, COND_OP_AND])
        np.testing.assert_array_equal(params, [10, 20, 0])

    def test_or_expression_rpn(self) -> None:
        tokens = [
            (COND_TICK_EQ, 1),
            (COND_OP_OR, 0),
            (COND_TICK_EQ, 2),
        ]
        types, params = _to_rpn_condition(tokens)
        # Infix: EQ1 OR EQ2  →  RPN: EQ1 EQ2 OR
        np.testing.assert_array_equal(types, [COND_TICK_EQ, COND_TICK_EQ, COND_OP_OR])
        np.testing.assert_array_equal(params, [1, 2, 0])

    def test_not_precedence(self) -> None:
        """NOT is right-associative with highest precedence → operand-first RPN."""
        tokens = [
            (COND_OP_NOT, 0),
            (COND_TICK_EQ, 3),
        ]
        types, params = _to_rpn_condition(tokens)
        # Infix: NOT EQ3  →  RPN: EQ3 NOT
        np.testing.assert_array_equal(types, [COND_TICK_EQ, COND_OP_NOT])
        np.testing.assert_array_equal(params, [3, 0])

    def test_parentheses_change_evaluation_order(self) -> None:
        """``(a or b) and c`` produces ``a b or c and``, not ``a b and c or``."""
        tokens = [
            (-(ord("(")), 0),
            (COND_TICK_EQ, 1),
            (COND_OP_OR, 0),
            (COND_TICK_EQ, 2),
            (-(ord(")")), 0),
            (COND_OP_AND, 0),
            (COND_TICK_LT, 2),
        ]
        types, params = _to_rpn_condition(tokens)
        # Infix: (EQ1 OR EQ2) AND LT2  →  RPN: EQ1 EQ2 OR LT2 AND
        np.testing.assert_array_equal(
            types,
            [COND_TICK_EQ, COND_TICK_EQ, COND_OP_OR, COND_TICK_LT, COND_OP_AND],
        )
        np.testing.assert_array_equal(params, [1, 2, 0, 2, 0])

    def test_extra_right_paren_raises(self) -> None:
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            _to_rpn_condition([(COND_TICK_EQ, 5), (-(ord(")")), 0)])

    def test_not_enough_operands_raises(self) -> None:
        """Binary operator with only one operand → compile-time error."""
        tokens = [(COND_TICK_EQ, 5), (COND_OP_AND, 0)]
        with pytest.raises(ValueError, match="malformed binary operator"):
            _to_rpn_condition(tokens)


# ── _parse_condition ─────────────────────────────────────────────────────────


class TestParseCondition:
    """Tests for ``_parse_condition`` -- convenience wrapper (combines above)."""

    def test_none_returns_always(self) -> None:
        types, params = _parse_condition(None)
        np.testing.assert_array_equal(types, [COND_ALWAYS])
        np.testing.assert_array_equal(params, [0])

    def test_valid_expression_produces_rpn(self) -> None:
        types, params = _parse_condition("tick >= 5 and tick < 10")
        # Expect RPN: GE5 LT10 AND
        np.testing.assert_array_equal(
            types, [COND_TICK_GE, COND_TICK_LT, COND_OP_AND]
        )
        np.testing.assert_array_equal(params, [5, 10, 0])


# ── Runtime evaluation (original script assertions) ──────────────────────────


class TestEvalConditionProgram:
    """Integration tests: parse → RPN → ``eval_csr_condition_program``.

    Retains all assertions from the original ``if __name__ == "__main__"``
    script.
    """

    @staticmethod
    def _eval(expr: str, tick: int) -> bool:
        cond_types, cond_params = parse_condition(expr)
        return bool(
            eval_csr_condition_program(
                cond_types, cond_params, 0, len(cond_types), tick
            )
        )

    # --- basic atoms ---

    def test_tick_eq(self) -> None:
        assert self._eval("tick == 10", 10) is True
        assert self._eval("tick == 10", 9) is False

    # --- and / or ---

    def test_and_range(self) -> None:
        assert self._eval("tick >= 10 and tick < 20", 12) is True
        assert self._eval("tick >= 10 and tick < 20", 22) is False

    def test_or_alternatives(self) -> None:
        assert self._eval("tick == 3 or tick == 5", 5) is True
        assert self._eval("tick == 3 or tick == 5", 4) is False

    # --- not and precedence (not > and > or) ---

    def test_not_unary(self) -> None:
        assert self._eval("not tick == 3", 4) is True
        assert self._eval("not tick == 3", 3) is False

    def test_and_higher_precedence_than_or(self) -> None:
        """``tick == 1 or tick == 2 and tick < 2`` binds as
        ``tick == 1 or (tick == 2 and tick < 2)``."""
        assert self._eval("tick == 1 or tick == 2 and tick < 2", 1) is True
        assert self._eval("tick == 1 or tick == 2 and tick < 2", 2) is False

    # --- parentheses ---

    def test_parentheses_override_precedence(self) -> None:
        assert self._eval(
            "(tick == 1 or tick == 2) and tick < 2", 1
        ) is True
        assert self._eval(
            "(tick == 1 or tick == 2) and tick < 2", 2
        ) is False

    # --- mod predicate + composition ---

    def test_mod_and_range(self) -> None:
        assert self._eval("tick % 5 == 0 and tick >= 10", 10) is True
        assert self._eval("tick % 5 == 0 and tick >= 10", 5) is False

    # --- nested not ---

    def test_not_with_parentheses(self) -> None:
        assert self._eval("not (tick == 1 or tick == 2)", 3) is True
        assert self._eval("not (tick == 1 or tick == 2)", 1) is False

    # --- invalid expressions ---

    def test_trailing_operator_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_condition("tick >= 10 and")

    def test_duplicate_operator_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_condition("tick >= 10 or or tick < 20")

    def test_unmatched_left_paren_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_condition("(tick >= 10")

    def test_unsupported_syntax_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_condition(
                "tick >= 10 and population.get_total_count() < 10"
            )
