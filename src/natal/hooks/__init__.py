"""Forwarding shim: the ``hooks`` package now lives at
``natal.frontend.hooks``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.hooks.compile as _m0
import natal.frontend.hooks.compile.codegen as _m1
import natal.frontend.hooks.compile.container as _m2
import natal.frontend.hooks.entry as _m3
import natal.frontend.hooks.entry.declarative as _m4
import natal.frontend.hooks.entry.decorator as _m5
import natal.frontend.hooks.entry.selector as _m6
import natal.frontend.hooks.runtime as _m7
import natal.frontend.hooks.runtime.csr_kernel as _m8
import natal.frontend.hooks.runtime.fallback as _m9
import natal.frontend.hooks.types as _m10
from natal.frontend.hooks import (
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
    EVENT_EARLY,
    EVENT_FINISH,
    EVENT_FIRST,
    EVENT_ID_MAP,
    EVENT_LATE,
    EVENT_NAMES,
    NUM_EVENTS,
    RESULT_CONTINUE,
    RESULT_SKIP,
    RESULT_STOP,
    CompiledEventHooks,
    CompiledHookDescriptor,
    CompiledHookPlan,
    DemeSelector,
    HookExecutor,
    HookOp,
    HookProgram,
    Op,
    OpType,
    build_hook_program,
    compile_combined_hook,
    compile_declarative_hook,
    compile_selector_hook,
    deme_selector_matches,
    eval_csr_condition_program,
    execute_csr_event_arrays,
    execute_csr_event_program,
    execute_csr_event_program_with_state,
    execute_single_csr_hook,
    hook,
    noop_hook,
    parse_condition,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.hooks.compile"] = _m0
_sys.modules["natal.hooks.compile.codegen"] = _m1
_sys.modules["natal.hooks.compile.container"] = _m2
_sys.modules["natal.hooks.entry"] = _m3
_sys.modules["natal.hooks.entry.declarative"] = _m4
_sys.modules["natal.hooks.entry.decorator"] = _m5
_sys.modules["natal.hooks.entry.selector"] = _m6
_sys.modules["natal.hooks.runtime"] = _m7
_sys.modules["natal.hooks.runtime.csr_kernel"] = _m8
_sys.modules["natal.hooks.runtime.fallback"] = _m9
_sys.modules["natal.hooks.types"] = _m10

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "COND_ALWAYS",
    "COND_OP_AND",
    "COND_OP_NOT",
    "COND_OP_OR",
    "COND_TICK_EQ",
    "COND_TICK_GE",
    "COND_TICK_GT",
    "COND_TICK_LE",
    "COND_TICK_LT",
    "COND_TICK_MOD",
    "CompiledEventHooks",
    "CompiledHookDescriptor",
    "CompiledHookPlan",
    "DemeSelector",
    "EVENT_EARLY",
    "EVENT_FINISH",
    "EVENT_FIRST",
    "EVENT_ID_MAP",
    "EVENT_LATE",
    "EVENT_NAMES",
    "HookExecutor",
    "HookOp",
    "HookProgram",
    "NUM_EVENTS",
    "Op",
    "OpType",
    "RESULT_CONTINUE",
    "RESULT_SKIP",
    "RESULT_STOP",
    "build_hook_program",
    "compile_combined_hook",
    "compile_declarative_hook",
    "compile_selector_hook",
    "deme_selector_matches",
    "eval_csr_condition_program",
    "execute_csr_event_arrays",
    "execute_csr_event_program",
    "execute_csr_event_program_with_state",
    "execute_single_csr_hook",
    "hook",
    "noop_hook",
    "parse_condition",
]
