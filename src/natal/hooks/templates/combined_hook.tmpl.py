# pyright: ignore[reportUnusedImport, reportUnusedVariable, reportUnknownVariableType, reportUnknownParameterType, reportMissingParameterType, reportUnusedExpression, reportUndefinedVariable]
# ruff: noqa

"""Codegen template for combined njit-only hook dispatch (see compile/codegen.py)."""

from typing import Callable

from natal.numba.utils import njit_switch

# Njit function placeholders — generated at codegen time.
# PLACEHOLDER_FN_DECLARATIONS (replaced with _FN_0, _FN_1, ... type-stub lines)


@njit_switch(cache=True)
def _combined_hook_TEMPLATE(state, config=None, deme_id=-1):  # type: ignore[no-untyped-def]
# === PLACEHOLDER: DO NOT DELETE ===
# This comment line is replaced at codegen time by compile_combined_hook
# with the njit call schedule body.  It must stay at column 0 — the
# replacement text carries its own indentation.
# PLACEHOLDER_SCHEDULE_BODY
# === END PLACEHOLDER ===

    return 0
