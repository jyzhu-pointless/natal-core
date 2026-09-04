"""Core hook types, constants, and low-level utilities.

This module intentionally avoids high-level compilation/execution logic and
only defines shared primitives used by other hook modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    pass

# Any callable that can serve as a hook body (noop, njit, combined, kernel).
HookCallable = Callable[..., Any]


class OpType(IntEnum):
    """Operation opcodes consumed by the runtime kernel.

    We intentionally keep integer values stable because these values are
    serialized into ``CompiledHookPlan.op_types`` and interpreted in the
    executor hot-loop.
    """

    SCALE = 0
    SET = 1
    ADD = 2
    SUBTRACT = 3
    KILL = 4
    SAMPLE = 5
    STOP_IF_ZERO = 6
    STOP_IF_BELOW = 7
    STOP_IF_ABOVE = 8
    STOP_IF_EXTINCTION = 9
    SET_PARAM = 10
    CONVERT = 11


# Canonical id table for ``Op.set_param`` targets and RPN operands.  The
# order is a cross-backend wire contract: the Python kernel, the
# kernel, and the Rust interpreter all index their ecology-value array by
# position in this tuple, so the order must never change once released.
# Membership rule: jsonc ecology-section scalars that are runtime-mutable
# 0-d draft arrays *and* Rust session f64 columns — dimensions (n_ages …),
# mode enums (growth_mode), derived caches (generation_time …), vectors,
# and genetics tensors are all excluded on purpose (they raise ValueError
# at compile time).
ECO_PARAM_NAMES: Tuple[str, ...] = (
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
)

# RPN value-expression token kinds (``Op.set_param`` value payloads).
# 0/1 push one operand (literal pool index / ECO param id); 2..5 are the
# binary arithmetic operators.  Mirrored in ``rust/src/hooks.rs``.
RPN_LITERAL = 0
RPN_PARAM = 1
RPN_ADD = 2
RPN_SUB = 3
RPN_MUL = 4
RPN_DIV = 5


@dataclass
class HookOp:
    """Single declarative operation before compilation.

    Fields in this class can still be symbolic (for example genotype labels).
    The compiler resolves all symbolic fields into concrete integer arrays.

    An ``HookOp`` can be registered directly as a declarative hook via
    ``.hooks(op, ...)``; *event* / *priority* may ride on the op itself or
    be supplied by the registration call.
    """

    op_type: OpType
    genotypes: Union[str, List[str], Literal["*"]] = "*"
    ages: Union[int, List[int], range, Literal["*"]] = "*"
    sex: Literal["female", "male", "both"] = "both"
    param: float = 1.0
    condition: Optional[str] = None
    event: Optional[str] = None
    priority: int = 0
    # ``Op.set_param`` payload: route name of the target scalar, the raw
    # value expression (RPN source or a plain number), and the firing
    # schedule (``tick >= start and (tick - start) % every == 0``).
    param_name: Optional[str] = None
    value_expr: Optional[Union[str, float, int]] = None
    every: int = 1
    start: int = 0
    # ``Op.convert`` payload: target zygote-type pattern (the source lives
    # in ``genotypes``); ``param`` carries the conversion probability.
    target_z: Optional[str] = None


DemeSelector = Union[int, List[int], Tuple[int, ...], range, Literal["*"]]


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
RESULT_CONTINUE = 0   # "executed successfully, proceed to the next hook"
RESULT_SKIP = 0       # "not applicable in this context, skip this hook silently"
RESULT_STOP = 1       # "abort the current event immediately"

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
    zidx_offsets: np.ndarray
    zidx_data: np.ndarray
    age_offsets: np.ndarray
    age_data: np.ndarray
    sex_masks: np.ndarray
    params: np.ndarray
    condition_offsets: np.ndarray
    condition_types: np.ndarray
    condition_params: np.ndarray
    # -- OP_SET_PARAM data area (per-op columns; -1 = not a set_param op) --
    # sp_param_ids: index into the fixed ECO_PARAM_NAMES table.
    # sp_every / sp_start: firing schedule (``tick >= start`` and
    # ``(tick - start) % every == 0``).
    sp_param_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    sp_every: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    sp_start: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    # rpn_offsets: CSR prefix offsets into the flattened token arrays.
    # rpn_kinds: RPN_LITERAL / RPN_PARAM / RPN_ADD..RPN_DIV per token.
    # rpn_payload: literal-pool index or ECO param id per operand token.
    # sp_literals: float64 literal pool shared by all ops of this plan.
    rpn_offsets: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))
    rpn_kinds: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    rpn_payload: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    sp_literals: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # -- OP_CONVERT data area (per-op ztype ids; -1 = not a convert op) --
    convert_source_z: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    convert_target_z: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    def to_tuple(self) -> Tuple[object, ...]:
        """Convert this plan to a flat tuple for HDF5 / array storage.

        Returns:
            A tuple of all fields in declaration order.
        """
        return (
            self.n_ops,
            self.op_types,
            self.zidx_offsets,
            self.zidx_data,
            self.age_offsets,
            self.age_data,
            self.sex_masks,
            self.params,
            self.condition_offsets,
            self.condition_types,
            self.condition_params,
            self.sp_param_ids,
            self.sp_every,
            self.sp_start,
            self.rpn_offsets,
            self.rpn_kinds,
            self.rpn_payload,
            self.sp_literals,
            self.convert_source_z,
            self.convert_target_z,
        )


def _empty_selector_map() -> Dict[str, np.ndarray]:
    """Return an empty selector map (default factory for dataclass fields)."""
    return {}


def _empty_meta_map() -> Dict[str, int]:
    """Return an empty meta map (default factory for dataclass fields)."""
    return {}


@dataclass
class CompiledHookDescriptor:
    """Unified descriptor for all hook forms.

    The payload is binary (26-decision): exactly one of ``plan`` (CSR
    declarative ops) or ``callback`` (single-parameter Python callable) is
    used as the execution payload.  The legacy ``njit_fn`` / ``py_wrapper``
    payloads were removed with the njit hook-wrapper mechanism.
    """

    name: str
    event: str
    priority: int = 0
    deme_selector: DemeSelector = "*"
    plan: Optional[CompiledHookPlan] = None
    callback: Optional[Callable[[object], Optional[int]]] = None
    selectors: Dict[str, np.ndarray] = field(default_factory=_empty_selector_map)
    meta: Dict[str, int] = field(default_factory=_empty_meta_map)
    ops: Optional[List[HookOp]] = None
    # Originating object (decorated/plain function or op group tuple) used
    # for identity-based idempotent registration; ``None`` for descriptors
    # built without one.
    source: Optional[object] = None


class HookProgram(NamedTuple):
    """Event-grouped plain-data CSR representation for declarative hooks."""

    n_events: np.int32
    n_hooks: np.int32
    hook_offsets: np.ndarray
    n_ops_list: np.ndarray
    op_offsets: np.ndarray
    op_types_data: np.ndarray
    zidx_offsets_data: np.ndarray
    zidx_data: np.ndarray
    age_offsets_data: np.ndarray
    age_data: np.ndarray
    sex_masks_data: np.ndarray
    params_data: np.ndarray
    condition_offsets_data: np.ndarray
    condition_types_data: np.ndarray
    condition_params_data: np.ndarray
    deme_selector_types: np.ndarray
    deme_selector_offsets: np.ndarray
    deme_selector_data: np.ndarray
    # OP_SET_PARAM / OP_CONVERT CSR data area (flattened across all hooks).
    # Per-op columns use -1 for ops of other types; rpn_offsets is the
    # shared prefix-sum over the flattened RPN token arrays.
    sp_param_ids: np.ndarray = np.array([], dtype=np.int32)
    sp_every: np.ndarray = np.array([], dtype=np.int32)
    sp_start: np.ndarray = np.array([], dtype=np.int32)
    rpn_offsets: np.ndarray = np.array([0], dtype=np.int32)
    rpn_kinds: np.ndarray = np.array([], dtype=np.int32)
    rpn_payload: np.ndarray = np.array([], dtype=np.int32)
    sp_literals: np.ndarray = np.array([], dtype=np.float64)
    convert_source_z: np.ndarray = np.array([], dtype=np.int32)
    convert_target_z: np.ndarray = np.array([], dtype=np.int32)
    # True when any op is OP_SET_PARAM: populations carrying such ops route
    # to the Python lifecycle orchestration so writes reach the route
    # table / dirty bridge / params snapshot log (single write channel).
    has_set_param: bool = False


def empty_hook_program(n_events: int = NUM_EVENTS) -> HookProgram:
    """Build an all-empty CSR ``HookProgram``.

    Used as the neutral element of the run program: populations without
    declarative hooks still need a well-shaped program so kernels and the
    Rust bridge can consume it without special cases.

    Args:
        n_events: Number of lifecycle events (default 4).

    Returns:
        An empty ``HookProgram`` with correct offsets.
    """
    return HookProgram(
        n_events=np.int32(n_events),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(n_events + 1, dtype=np.int32),
        n_ops_list=np.array([], dtype=np.int32),
        op_offsets=np.array([0], dtype=np.int32),
        op_types_data=np.array([], dtype=np.int32),
        zidx_offsets_data=np.array([0], dtype=np.int32),
        zidx_data=np.array([], dtype=np.int32),
        age_offsets_data=np.array([0], dtype=np.int32),
        age_data=np.array([], dtype=np.int32),
        sex_masks_data=np.array([], dtype=np.bool_),
        params_data=np.array([], dtype=np.float64),
        condition_offsets_data=np.array([0], dtype=np.int32),
        condition_types_data=np.array([], dtype=np.int32),
        condition_params_data=np.array([], dtype=np.int32),
        deme_selector_types=np.array([], dtype=np.int32),
        deme_selector_offsets=np.array([0], dtype=np.int32),
        deme_selector_data=np.array([], dtype=np.int32),
        sp_param_ids=np.array([], dtype=np.int32),
        sp_every=np.array([], dtype=np.int32),
        sp_start=np.array([], dtype=np.int32),
        rpn_offsets=np.array([0], dtype=np.int32),
        rpn_kinds=np.array([], dtype=np.int32),
        rpn_payload=np.array([], dtype=np.int32),
        sp_literals=np.array([], dtype=np.float64),
        convert_source_z=np.array([], dtype=np.int32),
        convert_target_z=np.array([], dtype=np.int32),
        has_set_param=False,
    )


class RunProgram(NamedTuple):
    """Program-level plan bundle owned by a population.

    Domain-B landing slot: the CSR hook program and the frozen recording
    plan travel together as the population's *program*.  The density
    program is still config-driven (no plan object exists yet), so it is
    deliberately absent instead of being fabricated as a placeholder.

    Attributes:
        hooks: CSR declarative hook plan (empty when no Op hooks).
        recording: The frozen :class:`RecordingPlan`, or ``None`` before
            the Configurator installs it at the end of ``build()``.
    """

    hooks: HookProgram
    recording: Optional[object] = None
