"""Tests for hook condition parsing and compilation."""

from __future__ import annotations

import numpy as np
import pytest

from natal.frontend.hooks import parse_condition
from natal.frontend.hooks.entry.declarative import (
    _parse_atomic_condition,
    _parse_condition,
    _to_rpn_condition,
    _tokenize_condition_expr,
)
from natal.frontend.hooks.types import (
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


class TestParseAtomicCondition:
    """Test conversion of one predicate string to a token."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("tick == 5", (COND_TICK_EQ, 5)),
            ("tick >= 10", (COND_TICK_GE, 10)),
            ("tick > 3", (COND_TICK_GT, 3)),
            ("tick <= 7", (COND_TICK_LE, 7)),
            ("tick < 2", (COND_TICK_LT, 2)),
            ("tick % 10 == 0", (COND_TICK_MOD, 10)),
        ],
    )
    def test_tick_predicates(self, expression: str, expected: tuple[int, int]) -> None:
        assert _parse_atomic_condition(expression) == expected

    @pytest.mark.parametrize("expression", ["tick % 10 != 0", "foo > 5"])
    def test_unsupported_predicates_raise(self, expression: str) -> None:
        with pytest.raises(ValueError, match="Unsupported atomic condition"):
            _parse_atomic_condition(expression)


class TestTokenizeConditionExpr:
    """Test conversion from expressions to infix tokens."""

    def test_operators_and_parentheses(self) -> None:
        tokens = _tokenize_condition_expr("(tick == 1 or tick == 2) and tick < 2")
        assert tokens == [
            (-(ord("(")), 0), (COND_TICK_EQ, 1), (COND_OP_OR, 0),
            (COND_TICK_EQ, 2), (-(ord(")")), 0), (COND_OP_AND, 0),
            (COND_TICK_LT, 2),
        ]

    def test_not_unary(self) -> None:
        assert _tokenize_condition_expr("not tick == 3") == [
            (COND_OP_NOT, 0), (COND_TICK_EQ, 3)
        ]

    @pytest.mark.parametrize("expression", ["", "foo > 5"])
    def test_invalid_expression_raises(self, expression: str) -> None:
        message = "cannot be empty" if not expression else "Unsupported condition syntax"
        with pytest.raises(ValueError, match=message):
            _tokenize_condition_expr(expression)


class TestToRpnCondition:
    """Test infix token conversion to native RPN arrays."""

    def test_operator_precedence(self) -> None:
        tokens = _tokenize_condition_expr("(tick == 1 or tick == 2) and tick < 2")
        types, params = _to_rpn_condition(tokens)
        np.testing.assert_array_equal(
            types, [COND_TICK_EQ, COND_TICK_EQ, COND_OP_OR, COND_TICK_LT, COND_OP_AND]
        )
        np.testing.assert_array_equal(params, [1, 2, 0, 2, 0])

    def test_not_precedence(self) -> None:
        types, params = _to_rpn_condition([(COND_OP_NOT, 0), (COND_TICK_EQ, 3)])
        np.testing.assert_array_equal(types, [COND_TICK_EQ, COND_OP_NOT])
        np.testing.assert_array_equal(params, [3, 0])

    def test_malformed_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            _to_rpn_condition([(COND_TICK_EQ, 5), (-(ord(")")), 0)])
        with pytest.raises(ValueError, match="malformed binary operator"):
            _to_rpn_condition([(COND_TICK_EQ, 5), (COND_OP_AND, 0)])


class TestParseCondition:
    """Test the public parser convenience wrapper."""

    def test_none_returns_always(self) -> None:
        types, params = _parse_condition(None)
        np.testing.assert_array_equal(types, [COND_ALWAYS])
        np.testing.assert_array_equal(params, [0])

    def test_valid_expression_produces_rpn(self) -> None:
        types, params = parse_condition("tick >= 5 and tick < 10")
        np.testing.assert_array_equal(types, [COND_TICK_GE, COND_TICK_LT, COND_OP_AND])
        np.testing.assert_array_equal(params, [5, 10, 0])

    @pytest.mark.parametrize(
        "expression", ["tick >= 10 and", "tick >= 10 or or tick < 20", "(tick >= 10"]
    )
    def test_invalid_expression_raises(self, expression: str) -> None:
        with pytest.raises(ValueError):
            parse_condition(expression)
