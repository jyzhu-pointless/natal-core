# pyright: ignore[reportUnusedImport, reportUnusedVariable, reportUnusedParameter, reportUnknownParameterType, reportMissingParameterType, reportUnusedExpression, reportUndefinedVariable]
# ruff: noqa

"""Codegen template for selector hook wrapper (see entry/selector.py)."""

from typing import Callable

from natal.numba.utils import njit_switch

# PLACEHOLDER_NAMEDTUPLE_IMPORT
# PLACEHOLDER_NAMEDTUPLE_DEF

_USER_FN: Callable[..., int] = lambda _s, _c=None, _d=-1: 0  # type: ignore[assignment]
# PLACEHOLDER_SEL_GLOBALS


@njit_switch(cache=True)
def PLACEHOLDER_FN_NAME(state, config=None, deme_id=-1):  # type: ignore[no-untyped-def]
# === PLACEHOLDER: DO NOT DELETE ===
# This comment line is replaced at codegen time by _compile_selector_njit_wrapper
# with the wrapper body.  It must stay at column 0 — the
# replacement text carries its own indentation.
# PLACEHOLDER_WRAPPER_BODY
# === END PLACEHOLDER ===

    return 0
