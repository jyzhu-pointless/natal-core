"""Core hook types, constants, and low-level utilities.

This module intentionally avoids high-level compilation/execution logic and
only defines shared primitives used by other hook modules.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from natal.numba_utils import get_numba_cache_dir

if TYPE_CHECKING:
    from natal.index_core import IndexCore


_HOOK_CODEGEN_DIR = Path(get_numba_cache_dir()) / "hook_codegen"
_HOOK_CODEGEN_LOCK = threading.Lock()


def _stable_callable_identity(fn: Callable) -> str:
    """Build a stable identity string for a callable across process runs."""
    py_fn = getattr(fn, "py_func", fn)
    module_name = getattr(py_fn, "__module__", "<unknown>")
    qualname = getattr(py_fn, "__qualname__", getattr(py_fn, "__name__", "<unknown>"))
    return f"{module_name}:{qualname}"


def _hash_key(parts: List[str]) -> str:
    """Compute a deterministic short hash key for generated wrapper identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _write_codegen_module(stem: str, source: str) -> Path:
    """Write generated wrapper module to stable path if it doesn't exist."""
    # Code generation can happen from multiple registration calls; lock to
    # avoid race conditions when two threads try to create the same file.
    with _HOOK_CODEGEN_LOCK:
        _HOOK_CODEGEN_DIR.mkdir(parents=True, exist_ok=True)
        module_path = _HOOK_CODEGEN_DIR / f"{stem}.py"
        if not module_path.exists():
            module_path.write_text(source, encoding="utf-8")
        return module_path


def _load_codegen_module(stem: str, module_path: Path):
    """Load a generated wrapper module from file, reusing sys.modules when possible."""
    module_name = f"natal._hook_codegen_{stem}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generated hook module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_numba_dispatcher(fn: Callable) -> bool:
    """Return True when callable is a numba dispatcher (has ``py_func``)."""
    return hasattr(fn, "py_func")


def _validate_numba_hook_required(fn, hook_name: str, reason: str) -> None:
    from ..numba_utils import NUMBA_ENABLED

    if NUMBA_ENABLED and not _is_numba_dispatcher(fn):
        raise TypeError(
            f"{hook_name} must be an @njit function when Numba is enabled due to {reason}. "
            f"Got {type(fn).__name__}. "
            "Either decorate with @njit or temporarily disable Numba using: "
            "from natal.numba_utils import numba_disabled; with numba_disabled(): ..."
        )


def is_njit_function(fn) -> bool:
    """Back-compatible alias for checking Numba dispatcher callables."""
    return _is_numba_dispatcher(fn)


def validate_hook_for_numba(hook, hook_name: str = "hook") -> None:
    """Back-compatible hook validator for Numba-enabled mode."""
    _validate_numba_hook_required(hook, hook_name, "hook registration")


# Condition type constants
COND_ALWAYS = 0
COND_TICK_EQ = 1
COND_TICK_MOD = 2
COND_TICK_GE = 3
COND_TICK_LT = 4
COND_TICK_LE = 5
COND_TICK_GT = 6

# Logical condition opcodes (RPN program)
COND_OP_AND = 100
COND_OP_OR = 101
COND_OP_NOT = 102

# Execution result codes
RESULT_CONTINUE = 0
RESULT_STOP = 1

# Event ID constants (for HookProgram)
EVENT_FIRST = 0
EVENT_EARLY = 1
EVENT_LATE = 2
EVENT_FINISH = 3
NUM_EVENTS = 4

EVENT_NAMES = ["first", "early", "late", "finish"]
EVENT_ID_MAP = {name: i for i, name in enumerate(EVENT_NAMES)}


@dataclass
class CompiledHookPlan:
    """Compiled declarative plan with CSR-style flattened arrays.

    Variable-length fields (genotypes/ages/conditions) are represented via
    ``*_offsets`` + ``*_data`` to keep kernel inputs contiguous and compact.
    """

    n_ops: int
    op_types: np.ndarray
    gidx_offsets: np.ndarray
    gidx_data: np.ndarray
    age_offsets: np.ndarray
    age_data: np.ndarray
    sex_masks: np.ndarray
    params: np.ndarray
    condition_offsets: np.ndarray
    condition_types: np.ndarray
    condition_params: np.ndarray

    def to_tuple(self) -> Tuple:
        return (
            self.n_ops,
            self.op_types,
            self.gidx_offsets,
            self.gidx_data,
            self.age_offsets,
            self.age_data,
            self.sex_masks,
            self.params,
            self.condition_offsets,
            self.condition_types,
            self.condition_params,
        )


@dataclass
class CompiledHookDescriptor:
    """Unified descriptor for all hook modes.

    Exactly one of ``plan``, ``njit_fn``, or ``py_wrapper`` is typically used
    as the primary execution payload for a descriptor.
    """

    name: str
    event: str
    priority: int = 0
    plan: Optional[CompiledHookPlan] = None
    selectors: Dict[str, np.ndarray] = field(default_factory=dict)
    static_arrays: Tuple[np.ndarray, ...] = field(default_factory=tuple)
    meta: Dict[str, int] = field(default_factory=dict)
    njit_fn: Optional[Callable] = None
    py_wrapper: Optional[Callable] = None


class HookProgram(NamedTuple):
    """Event-grouped plain-data CSR representation for declarative hooks."""

    n_events: np.int32
    n_hooks: np.int32
    hook_offsets: np.ndarray
    n_ops_list: np.ndarray
    op_offsets: np.ndarray
    op_types_data: np.ndarray
    gidx_offsets_data: np.ndarray
    gidx_data: np.ndarray
    age_offsets_data: np.ndarray
    age_data: np.ndarray
    sex_masks_data: np.ndarray
    params_data: np.ndarray
    condition_offsets_data: np.ndarray
    condition_types_data: np.ndarray
    condition_params_data: np.ndarray
