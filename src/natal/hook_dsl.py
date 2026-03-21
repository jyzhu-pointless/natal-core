"""Backward-compatible hook DSL imports.

The implementation now lives under ``natal.hook``.
This module remains as a stable compatibility layer for existing imports.
"""

from natal.hook import *  # noqa: F401,F403
from natal.hook.compiler import _njit_switch

__all__ = [
	# Core types
	'OpType',
	'HookOp',
	'Op',
	'CompiledHookPlan',
	'CompiledHookDescriptor',
	'HookProgram',
	'HookExecutor',
	'execute_csr_event_arrays',
	'execute_csr_event_program',
	'build_hook_program',
	# Numba-friendly hook utilities
	'noop_hook',
	'compile_combined_hook',
	'CompiledEventHooks',
	# Decorator
	'hook',
	# Compilation
	'compile_declarative_hook',
	'compile_selector_hook',
	# Constants
	'COND_ALWAYS',
	'COND_TICK_EQ',
	'COND_TICK_MOD',
	'COND_TICK_GE',
	'COND_TICK_GT',
	'COND_TICK_LE',
	'COND_TICK_LT',
	# Event IDs and naming
	'EVENT_FIRST',
	'EVENT_EARLY',
	'EVENT_LATE',
	'EVENT_FINISH',
	'EVENT_NAMES',
	'EVENT_ID_MAP',
	# Result codes
	'RESULT_CONTINUE',
	'RESULT_STOP',
]
