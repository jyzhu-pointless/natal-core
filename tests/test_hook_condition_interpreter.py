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


class TestZeroModuloDivisor:
    """A zero modulo divisor is an illegal condition, rejected at parse time.

    Regression target (CR-5): ``tick % 0 == 0`` used to compile into a
    ``COND_TICK_MOD`` token with parameter 0, which the native interpreter
    then masked into an always-false predicate — silently disabling the
    declaring hook instead of surfacing the malformed condition.
    """

    def test_parse_atomic_condition_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _parse_atomic_condition("tick % 0 == 0")

    def test_parse_condition_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            parse_condition("tick % 0 == 0")

    def test_compound_expression_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            parse_condition("tick >= 1 and tick % 0 == 0")

    def test_op_declaration_rejects_zero_at_build(self) -> None:
        """The declaring build chain fails; nothing is registered."""
        import natal as nt

        op = nt.Op.scale(ages=[0], factor=0.5, when="tick % 0 == 0", event="early")
        species = nt.Species.from_dict(
            name="zmod_species", structure={"chr1": {"loc": ["WT", "Dr"]}}
        )
        builder = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name="zmod", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 50.0}, "male": {"WT|WT": 50.0}}
            )
            .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .hooks(op)
        )
        with pytest.raises(ValueError, match="positive integer"):
            builder.build()
