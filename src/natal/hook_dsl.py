"""Declarative Hook DSL with automatic Numba compilation.

This module provides a domain-specific language for defining population hooks
that can be automatically compiled to Numba-accelerated code. Users define
hooks using high-level operations (scale, add, kill, etc.) with genotype/age
selectors, and the system handles index resolution and code generation.

Three usage modes:
1. Declarative: Return a list of Op.* operations (fully automatic)
2. Selector-based: Use @hook(selectors={...}) to pre-resolve indices
3. Native Numba: Use @hook(numba=True) for user-provided njit functions

Example:
    >>> from natal.hook_dsl import hook, Op
    >>> 
    >>> # Declarative mode
    >>> @hook(event='early')
    >>> def reduce_juveniles():
    ...     return [
    ...         Op.scale(genotypes='AA', ages=[0, 1], factor=0.9),
    ...         Op.add(genotypes='*', ages=0, delta=50, when='tick % 10 == 0'),
    ...     ]
    >>> 
    >>> # Selector mode
    >>> @hook(event='first', selectors={'target_gt': 'Dw'})
    >>> def release_drive(pop, target_gt):
    ...     if pop.tick == 10:
    ...         pop.state.individual_count[1, 2, target_gt] += 100
    >>> 
    >>> # Register to population
    >>> reduce_juveniles.register(pop)
"""

from __future__ import annotations

import re
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Union, Callable, Tuple, Any, 
    Literal, TYPE_CHECKING
)

from natal.numba_utils import jitclass_switch, njit_switch

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.base_population import BasePopulation
    from natal.index_core import IndexCore

# =============================================================================
# Operation Types and Data Structures
# =============================================================================

class OpType(IntEnum):
    """Hook operation types (Numba compatible integer enum)."""
    SCALE = 0       # count *= factor
    SET = 1         # count = value
    ADD = 2         # count += delta
    SUBTRACT = 3    # count -= delta (min 0)
    KILL = 4        # binomial sampling with survival prob = 1 - param
    SAMPLE = 5      # count = min(count, sample_size)
    # Termination condition checks (return True to stop simulation)
    STOP_IF_ZERO = 6        # stop if selected count == 0
    STOP_IF_BELOW = 7       # stop if selected count < threshold
    STOP_IF_ABOVE = 8       # stop if selected count > threshold
    STOP_IF_EXTINCTION = 9  # stop if total population == 0


# Condition type constants
COND_ALWAYS = 0      # No condition
COND_TICK_EQ = 1     # tick == N
COND_TICK_MOD = 2    # tick % N == 0
COND_TICK_GE = 3     # tick >= N
COND_TICK_LT = 4     # tick < N
COND_TICK_LE = 5     # tick <= N
COND_TICK_GT = 6     # tick > N


# Execution result codes
RESULT_CONTINUE = 0   # Continue simulation
RESULT_STOP = 1       # Stop simulation (termination condition met)


def is_njit_function(fn) -> bool:
    """Check if a function is a Numba @njit compiled function.
    
    Args:
        fn: Any callable
        
    Returns:
        bool: True if fn is a Numba dispatcher (has py_func attribute)
    """
    return hasattr(fn, 'py_func')


def validate_hook_for_numba(hook, hook_name: str = "hook") -> None:
    """Validate that a hook is Numba-compatible when Numba is enabled.
    
    Numba is enabled by default for performance. This validation ensures hooks
    are properly decorated with @njit when Numba is active.
    
    Args:
        hook: The hook function to validate
        hook_name: Name for error messages
        
    Raises:
        TypeError: If Numba is enabled but hook is not @njit
    """
    from .numba_utils import NUMBA_ENABLED
    
    if NUMBA_ENABLED:
        if not is_njit_function(hook):
            raise TypeError(
                f"{hook_name} must be an @njit function when Numba is enabled. "
                f"Got {type(hook).__name__}. "
                f"Either decorate with @njit or temporarily disable Numba using: "
                f"from natal.numba_utils import numba_disabled; with numba_disabled(): ..."
            )


# Event ID constants (for HookRegistry)
EVENT_FIRST = 0
EVENT_REPRODUCTION = 1
EVENT_EARLY = 2
EVENT_SURVIVAL = 3
EVENT_LATE = 4
EVENT_FINISH = 5
NUM_EVENTS = 6

EVENT_NAMES = ['first', 'reproduction', 'early', 'survival', 'late', 'finish']

# Build reverse mapping for quick lookup
EVENT_ID_MAP = {name: i for i, name in enumerate(EVENT_NAMES)}


@dataclass
class HookOp:
    """Single hook operation in declarative form.
    
    Attributes:
        op_type: Operation type enum.
        genotypes: Genotype selector ('*' for all, string, or list).
        ages: Age selector ('*' for all, int, list, or range).
        sex: Sex selector ('female', 'male', or 'both').
        param: Operation parameter (factor/value/delta/probability).
        condition: Optional condition string (e.g., 'tick % 10 == 0').
    """
    op_type: OpType
    genotypes: Union[str, List[str], Literal['*']] = '*'
    ages: Union[int, List[int], range, Literal['*']] = '*'
    sex: Literal['female', 'male', 'both'] = 'both'
    param: float = 1.0
    condition: Optional[str] = None


class Op:
    """Factory class for creating HookOp instances with convenient methods."""
    
    @staticmethod
    def scale(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        factor: float = 1.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Scale counts by a factor: count *= factor.
        
        Args:
            genotypes: Target genotypes ('*', 'AA', ['AA', 'Aa'], etc.)
            ages: Target ages ('*', 0, [0, 1], range(0, 3), etc.)
            sex: Target sex ('female', 'male', 'both')
            factor: Multiplication factor
            when: Optional condition (e.g., 'tick % 10 == 0')
        
        Returns:
            HookOp: Configured operation
        """
        return HookOp(OpType.SCALE, genotypes, ages, sex, factor, when)
    
    @staticmethod
    def set_count(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        value: float = 0.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Set counts to a specific value: count = value.
        
        Args:
            genotypes: Target genotypes
            ages: Target ages
            sex: Target sex
            value: Value to set
            when: Optional condition
        
        Returns:
            HookOp: Configured operation
        """
        return HookOp(OpType.SET, genotypes, ages, sex, value, when)
    
    @staticmethod
    def add(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        delta: float = 0.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Add to counts: count += delta.
        
        Args:
            genotypes: Target genotypes
            ages: Target ages
            sex: Target sex
            delta: Amount to add
            when: Optional condition
        
        Returns:
            HookOp: Configured operation
        """
        return HookOp(OpType.ADD, genotypes, ages, sex, delta, when)
    
    @staticmethod
    def subtract(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        delta: float = 0.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Subtract from counts: count = max(0, count - delta).
        
        Args:
            genotypes: Target genotypes
            ages: Target ages
            sex: Target sex
            delta: Amount to subtract
            when: Optional condition
        
        Returns:
            HookOp: Configured operation
        """
        return HookOp(OpType.SUBTRACT, genotypes, ages, sex, delta, when)
    
    @staticmethod
    def kill(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        prob: float = 0.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Kill individuals with given probability.
        
        Each individual has `prob` probability of dying (binomial sampling).
        
        Args:
            genotypes: Target genotypes
            ages: Target ages
            sex: Target sex
            prob: Death probability (0-1)
            when: Optional condition
        
        Returns:
            HookOp: Configured operation
        """
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        return HookOp(OpType.KILL, genotypes, ages, sex, prob, when)
    
    @staticmethod
    def sample(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        size: int = 0,
        when: Optional[str] = None
    ) -> HookOp:
        """Sample at most `size` individuals: count = min(count, size).
        
        Args:
            genotypes: Target genotypes
            ages: Target ages
            sex: Target sex
            size: Maximum count to keep
            when: Optional condition
        
        Returns:
            HookOp: Configured operation
        """
        return HookOp(OpType.SAMPLE, genotypes, ages, sex, float(size), when)
    
    # =========================================================================
    # Termination Conditions
    # =========================================================================
    
    @staticmethod
    def stop_if_zero(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        when: Optional[str] = None
    ) -> HookOp:
        """Stop simulation if selected population count equals zero.
        
        This operation checks if the sum of counts for the selected
        genotypes/ages/sex equals zero, and signals termination if so.
        
        Args:
            genotypes: Target genotypes to check
            ages: Target ages to check
            sex: Target sex to check
            when: Optional condition (e.g., 'tick >= 10' to only check after tick 10)
        
        Returns:
            HookOp: Configured termination check
        
        Example:
            >>> # Stop if all 'AA' individuals die
            >>> Op.stop_if_zero(genotypes='AA')
            >>> 
            >>> # Stop if males go extinct
            >>> Op.stop_if_zero(sex='male')
        """
        return HookOp(OpType.STOP_IF_ZERO, genotypes, ages, sex, 0.0, when)
    
    @staticmethod
    def stop_if_below(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        threshold: float = 1.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Stop simulation if selected population count falls below threshold.
        
        Args:
            genotypes: Target genotypes to check
            ages: Target ages to check
            sex: Target sex to check
            threshold: Stop if count < threshold
            when: Optional condition
        
        Returns:
            HookOp: Configured termination check
        """
        return HookOp(OpType.STOP_IF_BELOW, genotypes, ages, sex, float(threshold), when)
    
    @staticmethod
    def stop_if_above(
        genotypes: Union[str, List[str], Literal['*']] = '*',
        ages: Union[int, List[int], range, Literal['*']] = '*',
        sex: Literal['female', 'male', 'both'] = 'both',
        threshold: float = 1000000.0,
        when: Optional[str] = None
    ) -> HookOp:
        """Stop simulation if selected population count exceeds threshold.
        
        Args:
            genotypes: Target genotypes to check
            ages: Target ages to check
            sex: Target sex to check
            threshold: Stop if count > threshold
            when: Optional condition
        
        Returns:
            HookOp: Configured termination check
        """
        return HookOp(OpType.STOP_IF_ABOVE, genotypes, ages, sex, float(threshold), when)
    
    @staticmethod
    def stop_if_extinction(when: Optional[str] = None) -> HookOp:
        """Stop simulation if total population goes extinct.
        
        Convenience method that checks if the entire population is zero.
        
        Args:
            when: Optional condition
        
        Returns:
            HookOp: Configured termination check
        """
        return HookOp(OpType.STOP_IF_EXTINCTION, '*', '*', 'both', 0.0, when)


# =============================================================================
# Compiled Hook Structures (Numba-friendly)
# =============================================================================

@dataclass
class CompiledHookPlan:
    """Compiled hook execution plan with CSR-packed arrays (Numba-friendly).
    
    All operations are packed into parallel arrays to avoid Python objects
    during execution.
    
    Attributes:
        n_ops: Number of operations.
        op_types: Operation type for each op (shape: n_ops).
        gidx_offsets: CSR-style offsets for genotype indices (shape: n_ops+1).
        gidx_data: Concatenated genotype indices.
        age_offsets: CSR-style offsets for age indices (shape: n_ops+1).
        age_data: Concatenated age indices.
        sex_masks: Boolean mask for each sex per op (shape: n_ops, 2).
        params: Operation parameters (shape: n_ops).
        condition_types: Condition type per op (shape: n_ops).
        condition_params: Condition parameter per op (shape: n_ops).
    """
    n_ops: int
    op_types: np.ndarray           # dtype=int32
    gidx_offsets: np.ndarray       # dtype=int32
    gidx_data: np.ndarray          # dtype=int32
    age_offsets: np.ndarray        # dtype=int32
    age_data: np.ndarray           # dtype=int32
    sex_masks: np.ndarray          # dtype=bool, shape (n_ops, 2)
    params: np.ndarray             # dtype=float64
    condition_types: np.ndarray    # dtype=int32
    condition_params: np.ndarray   # dtype=int32
    
    def to_tuple(self) -> Tuple:
        """Convert to tuple for passing to Numba functions."""
        return (
            self.n_ops,
            self.op_types,
            self.gidx_offsets,
            self.gidx_data,
            self.age_offsets,
            self.age_data,
            self.sex_masks,
            self.params,
            self.condition_types,
            self.condition_params,
        )


@dataclass 
class CompiledHookDescriptor:
    """Descriptor for a compiled hook.
    
    Attributes:
        name: Hook function name.
        event: Event name ('first', 'early', 'survival', 'late', etc.).
        priority: Execution priority (lower = earlier).
        plan: Compiled execution plan (for declarative hooks).
        selectors: Resolved selector arrays (for selector-based hooks).
        static_arrays: Precomputed static arrays.
        meta: Metadata dict (n_genotypes, n_ages, etc.).
        njit_fn: Optional Numba-jitted function.
        py_wrapper: Python wrapper function.
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


# =============================================================================
# HookRegistry (Numba JitClass for hook management)
# =============================================================================

from numba import types as nb_types

# Define spec for HookRegistry jitclass
spec_hook_registry = [
    ('n_events', nb_types.int32),
    ('n_hooks', nb_types.int32),
    ('hook_offsets', nb_types.int32[::1]),      # CSR: event -> hook
    ('n_ops_list', nb_types.int32[::1]),        # operations per hook
    ('op_offsets', nb_types.int32[::1]),        # CSR: hook -> operation
    
    # Packed operation data (all operations concatenated)
    ('op_types_data', nb_types.int32[::1]),
    ('gidx_offsets_data', nb_types.int32[::1]),
    ('gidx_data', nb_types.int32[::1]),
    ('age_offsets_data', nb_types.int32[::1]),
    ('age_data', nb_types.int32[::1]),
    ('sex_masks_data', nb_types.bool_[::1]),
    ('params_data', nb_types.float64[::1]),
    ('condition_types_data', nb_types.int32[::1]),
    ('condition_params_data', nb_types.int32[::1]),
]

@jitclass_switch(spec_hook_registry)
class HookRegistry:
    """Numba-compiled registry for managing all population hooks.
    
    Stores all hooks in packed arrays with CSR-style indexing for
    efficient event-based execution.
    """
    
    def __init__(self, n_events, n_hooks, hook_offsets, n_ops_list, op_offsets,
                 op_types_data, gidx_offsets_data, gidx_data, 
                 age_offsets_data, age_data, sex_masks_data,
                 params_data, condition_types_data, condition_params_data):
        self.n_events = n_events
        self.n_hooks = n_hooks
        self.hook_offsets = hook_offsets
        self.n_ops_list = n_ops_list
        self.op_offsets = op_offsets
        self.op_types_data = op_types_data
        self.gidx_offsets_data = gidx_offsets_data
        self.gidx_data = gidx_data
        self.age_offsets_data = age_offsets_data
        self.age_data = age_data
        self.sex_masks_data = sex_masks_data
        self.params_data = params_data
        self.condition_types_data = condition_types_data
        self.condition_params_data = condition_params_data
    
    def execute_csr_event(self, event_id, individual_count, tick):
        """Execute CSR-packed declarative operations for a given event.
        
        This method only handles Op.* operations that have been compiled
        into CSR arrays. User functions (njit_fn, py_wrapper) are handled
        by HookExecutor in the Python layer.
        
        Args:
            event_id: Integer event ID (0-5)
            individual_count: Population count array (sex, age, genotype)
            tick: Current simulation tick
        
        Returns:
            int: RESULT_CONTINUE (0) or RESULT_STOP (1)
        """
        if event_id < 0 or event_id >= self.n_events:
            return 0  # Invalid event ID, skip
        
        # Get hooks for this event using CSR indexing
        hook_start = self.hook_offsets[event_id]
        hook_end = self.hook_offsets[event_id + 1]
        
        # Execute each hook
        for hook_idx in range(hook_start, hook_end):
            result = self._execute_hook(hook_idx, individual_count, tick)
            if result != 0:
                return result
        
        return 0  # RESULT_CONTINUE
    
    def _execute_hook(self, hook_idx, individual_count, tick):
        """Execute a single hook by index."""
        if hook_idx < 0 or hook_idx >= self.n_hooks:
            return 0
        
        # Get operation range for this hook
        op_start = self.op_offsets[hook_idx]
        op_end = self.op_offsets[hook_idx + 1]
        
        # Execute each operation in the hook
        for op_idx in range(op_start, op_end):
            result = self._execute_operation(op_idx, individual_count, tick)
            if result != 0:
                return result
        
        return 0
    
    def _execute_operation(self, op_idx, individual_count, tick):
        """Execute a single operation."""
        # Check condition
        cond_type = self.condition_types_data[op_idx]
        cond_param = self.condition_params_data[op_idx]
        
        if not self._check_condition(cond_type, cond_param, tick):
            return 0  # Condition not met, skip
        
        # Get operation parameters
        op_type = self.op_types_data[op_idx]
        param = self.params_data[op_idx]
        
        # Get genotype indices (CSR format)
        gidx_start = self.gidx_offsets_data[op_idx]
        gidx_end = self.gidx_offsets_data[op_idx + 1]
        
        # Get age indices (CSR format)
        age_start = self.age_offsets_data[op_idx]
        age_end = self.age_offsets_data[op_idx + 1]
        
        # Get sex mask (2 elements: [female, male])
        sex_mask_idx = op_idx * 2
        sex_female = self.sex_masks_data[sex_mask_idx]
        sex_male = self.sex_masks_data[sex_mask_idx + 1]
        
        # Apply operation to all selected cells
        for sex_idx in range(2):
            if sex_idx == 0 and not sex_female:  # 0=female
                continue
            if sex_idx == 1 and not sex_male:    # 1=male
                continue
            
            for age_idx_ptr in range(age_start, age_end):
                age = self.age_data[age_idx_ptr]
                
                for gidx_ptr in range(gidx_start, gidx_end):
                    gidx = self.gidx_data[gidx_ptr]
                    
                    # Apply operation
                    current = individual_count[sex_idx, age, gidx]
                    
                    if op_type == 0:  # SCALE
                        individual_count[sex_idx, age, gidx] = current * param
                    elif op_type == 1:  # SET
                        individual_count[sex_idx, age, gidx] = param
                    elif op_type == 2:  # ADD
                        individual_count[sex_idx, age, gidx] = current + param
                    elif op_type == 3:  # SUBTRACT
                        individual_count[sex_idx, age, gidx] = max(0.0, current - param)
                    elif op_type == 5:  # SAMPLE
                        individual_count[sex_idx, age, gidx] = min(current, param)
        
        return 0
    
    def _check_condition(self, cond_type, cond_param, tick):
        """Check if operation should execute based on tick condition."""
        if cond_type == 0:  # COND_ALWAYS
            return True
        elif cond_type == 1:  # COND_TICK_EQ
            return tick == cond_param
        elif cond_type == 2:  # COND_TICK_MOD
            return cond_param > 0 and tick % cond_param == 0
        elif cond_type == 3:  # COND_TICK_GE
            return tick >= cond_param
        elif cond_type == 4:  # COND_TICK_LT
            return tick < cond_param
        elif cond_type == 5:  # COND_TICK_LE
            return tick <= cond_param
        elif cond_type == 6:  # COND_TICK_GT
            return tick > cond_param
        return True


# =============================================================================
# HookExecutor (Python-layer coordinator)
# =============================================================================

class HookExecutor:
    """协调式hook执行器，管理所有已编译的hooks。
    
    将hooks按event分类，并协调三层执行：
    1. CSR操作（Numba，极快）
    2. njit_fn hooks（Numba用户函数，快）
    3. py_wrapper hooks（Python包装器，支持动态修改）
    
    职责：
    - 按event分类管理hooks
    - 按优先级排序
    - 协调CSR操作 + 用户函数的执行顺序
    """
    
    def __init__(self, registry, hooks_by_event):
        """初始化执行器。
        
        Args:
            registry: HookRegistry实例（处理CSR操作）
            hooks_by_event: Dict[int, List[CompiledHookDescriptor]]，按event分类
        """
        self.registry = registry
        self.hooks_by_event = hooks_by_event
    
    @staticmethod
    def from_compiled_hooks(
        registry: HookRegistry,
        compiled_hooks: List['CompiledHookDescriptor']
    ) -> 'HookExecutor':
        """从已编译的hooks构建执行器。
        
        Args:
            registry: HookRegistry实例
            compiled_hooks: 所有CompiledHookDescriptor列表
        
        Returns:
            HookExecutor: 新的执行器实例
        """
        from collections import defaultdict
        
        hooks_by_event = defaultdict(list)
        for desc in compiled_hooks:
            event_id = EVENT_ID_MAP.get(desc.event)
            if event_id is not None:
                # 只收集有执行逻辑的hooks
                if desc.plan is not None or desc.njit_fn is not None or desc.py_wrapper is not None:
                    hooks_by_event[event_id].append(desc)
        
        # 按优先级排序
        for event_id in hooks_by_event:
            hooks_by_event[event_id].sort(key=lambda x: x.priority)
        
        return HookExecutor(registry, dict(hooks_by_event))
    
    def execute_event(
        self,
        event_id: int,
        population: 'BasePopulation',
        tick: int
    ) -> int:
        """执行某个事件的所有hooks。
        
        执行顺序：
        1. CSR操作（Numba快速路径）
        2. njit_fn hooks（按优先级）
        3. py_wrapper hooks（按优先级）
        
        Args:
            event_id: Integer event ID (0-5)
            population: BasePopulation实例
            tick: 当前simulation tick
        
        Returns:
            int: RESULT_CONTINUE (0) 继续运行，或 RESULT_STOP (1) 停止
        """
        if event_id < 0 or event_id >= NUM_EVENTS:
            return RESULT_CONTINUE
        
        ind_count = population.state.individual_count
        
        # === Phase 1: CSR操作（Numba编译，极快） ===
        result = self.registry.execute_csr_event(event_id, ind_count, tick)
        if result == RESULT_STOP:
            return RESULT_STOP
        
        # === Phase 2: njit_fn hooks（用户自定义Numba函数） ===
        for desc in self.hooks_by_event.get(event_id, []):
            if desc.njit_fn is not None:
                try:
                    result = desc.njit_fn(ind_count, tick)
                    if result == RESULT_STOP:
                        return RESULT_STOP
                except Exception as e:
                    raise RuntimeError(f"Error in njit hook '{desc.name}': {e}")
        
        # === Phase 3: py_wrapper hooks（Python包装函数） ===
        for desc in self.hooks_by_event.get(event_id, []):
            if desc.py_wrapper is not None and desc.njit_fn is None:
                try:
                    # py_wrapper通常用于selector模式或动态修改
                    desc.py_wrapper(population)
                except Exception as e:
                    raise RuntimeError(f"Error in py_wrapper hook '{desc.name}': {e}")
        
        return RESULT_CONTINUE
    
    def get_hooks_for_event(self, event_id: int) -> List['CompiledHookDescriptor']:
        """获取特定事件的所有hooks。
        
        Args:
            event_id: Event ID
        
        Returns:
            List[CompiledHookDescriptor]: Hooks列表（已按优先级排序）
        """
        return self.hooks_by_event.get(event_id, [])


# =============================================================================
# Selector Resolution
# =============================================================================

def _resolve_genotypes(
    selector: Union[str, List[str], Literal['*']],
    index_core: 'IndexCore',
    diploid_genotypes: List[Any],
    n_genotypes: int
) -> np.ndarray:
    """Resolve genotype selector to index array.
    
    Args:
        selector: Genotype selector ('*', 'AA', ['AA', 'Aa'], etc.)
        index_core: IndexCore instance for resolution
        diploid_genotypes: List of diploid genotype objects
        n_genotypes: Total number of genotypes
    
    Returns:
        np.ndarray: Array of genotype indices (dtype=int32)
    """
    if selector == '*':
        return np.arange(n_genotypes, dtype=np.int32)
    
    if isinstance(selector, str):
        selector = [selector]
    
    indices = []
    for s in selector:
        if isinstance(s, int):
            indices.append(s)
        else:
            # Try to resolve via index_core
            idx = index_core.resolve_genotype_index(diploid_genotypes, s, strict=True)
            if idx is None:
                raise ValueError(f"Cannot resolve genotype: {s}")
            indices.append(idx)
    
    return np.array(indices, dtype=np.int32)


def _resolve_ages(
    selector: Union[int, List[int], range, Literal['*']],
    n_ages: int
) -> np.ndarray:
    """Resolve age selector to index array.
    
    Args:
        selector: Age selector ('*', 0, [0, 1], range(0, 3), etc.)
        n_ages: Total number of ages
    
    Returns:
        np.ndarray: Array of age indices (dtype=int32)
    """
    if selector == '*':
        return np.arange(n_ages, dtype=np.int32)
    
    if isinstance(selector, int):
        return np.array([selector], dtype=np.int32)
    
    if isinstance(selector, range):
        return np.array(list(selector), dtype=np.int32)
    
    return np.array(list(selector), dtype=np.int32)


def _resolve_sex(selector: Literal['female', 'male', 'both']) -> np.ndarray:
    """Resolve sex selector to boolean mask.
    
    Args:
        selector: Sex selector ('female', 'male', 'both')
    
    Returns:
        np.ndarray: Boolean mask of shape (2,) for [female, male]
    """
    if selector == 'both':
        return np.array([True, True], dtype=np.bool_)
    elif selector == 'female':
        return np.array([True, False], dtype=np.bool_)
    elif selector == 'male':
        return np.array([False, True], dtype=np.bool_)
    else:
        raise ValueError(f"Unknown sex selector: {selector}")


def _parse_condition(condition: Optional[str]) -> Tuple[int, int]:
    """Parse condition string to (condition_type, condition_param).
    
    Supported formats:
        - None: always execute
        - 'tick == N': execute when tick equals N
        - 'tick % N == 0': execute every N ticks
        - 'tick >= N': execute when tick >= N
        - 'tick > N': execute when tick > N
        - 'tick < N': execute when tick < N
        - 'tick <= N': execute when tick <= N
    
    Args:
        condition: Condition string or None
    
    Returns:
        Tuple[int, int]: (condition_type, condition_param)
    """
    if condition is None:
        return (COND_ALWAYS, 0)
    
    condition = condition.strip()
    
    # tick == N
    match = re.match(r'tick\s*==\s*(\d+)', condition)
    if match:
        return (COND_TICK_EQ, int(match.group(1)))
    
    # tick % N == 0
    match = re.match(r'tick\s*%\s*(\d+)\s*==\s*0', condition)
    if match:
        return (COND_TICK_MOD, int(match.group(1)))
    
    # tick >= N
    match = re.match(r'tick\s*>=\s*(\d+)', condition)
    if match:
        return (COND_TICK_GE, int(match.group(1)))
    
    # tick > N
    match = re.match(r'tick\s*>\s*(\d+)', condition)
    if match:
        return (COND_TICK_GT, int(match.group(1)))
    
    # tick <= N
    match = re.match(r'tick\s*<=\s*(\d+)', condition)
    if match:
        return (COND_TICK_LE, int(match.group(1)))
    
    # tick < N
    match = re.match(r'tick\s*<\s*(\d+)', condition)
    if match:
        return (COND_TICK_LT, int(match.group(1)))
    
    raise ValueError(f"Unsupported condition format: {condition}")


def _resolve_selector_to_array(
    spec: Any,
    index_core: 'IndexCore',
    diploid_genotypes: List[Any]
) -> np.ndarray:
    """Resolve a flexible selector spec to an integer array.
    
    Used for @hook(selectors={...}) parameter resolution.
    
    Args:
        spec: Selector specification (int, str, list, range, etc.)
        index_core: IndexCore for genotype resolution
        diploid_genotypes: List of diploid genotype objects
    
    Returns:
        np.ndarray: Resolved indices (dtype=int32)
    """
    # Already an int
    if isinstance(spec, int):
        return np.array([spec], dtype=np.int32)
    
    # Range
    if isinstance(spec, range):
        return np.array(list(spec), dtype=np.int32)
    
    # String (genotype name)
    if isinstance(spec, str):
        if spec == '*':
            return np.arange(len(diploid_genotypes), dtype=np.int32)
        idx = index_core.resolve_genotype_index(diploid_genotypes, spec, strict=True)
        if idx is None:
            raise ValueError(f"Cannot resolve genotype: {spec}")
        return np.array([idx], dtype=np.int32)
    
    # List/tuple
    if isinstance(spec, (list, tuple)):
        indices = []
        for item in spec:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                idx = index_core.resolve_genotype_index(diploid_genotypes, item, strict=True)
                if idx is None:
                    raise ValueError(f"Cannot resolve genotype: {item}")
                indices.append(idx)
            else:
                # Try direct lookup
                try:
                    idx = index_core.genotype_to_index.get(item)
                    if idx is not None:
                        indices.append(idx)
                    else:
                        raise ValueError(f"Cannot resolve selector item: {item}")
                except Exception:
                    raise ValueError(f"Cannot resolve selector item: {item}")
        return np.array(indices, dtype=np.int32)
    
    # Genotype object
    try:
        idx = index_core.genotype_to_index.get(spec)
        if idx is not None:
            return np.array([idx], dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Cannot resolve selector spec: {spec}") from e

# =============================================================================
# Compilation Functions
# =============================================================================

def compile_declarative_hook(
    ops: List[HookOp],
    pop: 'AgeStructuredPopulation',
    event: str,
    priority: int = 0,
    name: str = "declarative_hook"
) -> CompiledHookDescriptor:
    """Compile a list of declarative operations to a CompiledHookDescriptor.
    
    Args:
        ops: List of HookOp operations
        pop: Population instance for index resolution
        event: Event name
        priority: Execution priority
        name: Hook name
    
    Returns:
        CompiledHookDescriptor: Compiled descriptor
    """
    index_core = pop._index_core
    diploid_genotypes = index_core.index_to_genotype
    n_genotypes = index_core.num_genotypes()
    n_ages = pop._config.n_ages
    
    # Collect compiled data
    op_types_list = []
    gidx_offsets = [0]
    gidx_data_list = []
    age_offsets = [0]
    age_data_list = []
    sex_masks_list = []
    params_list = []
    condition_types_list = []
    condition_params_list = []
    
    for op in ops:
        # Operation type
        op_types_list.append(int(op.op_type))
        
        # Genotype indices
        gidx_array = _resolve_genotypes(op.genotypes, index_core, diploid_genotypes, n_genotypes)
        gidx_data_list.extend(gidx_array.tolist())
        gidx_offsets.append(len(gidx_data_list))
        
        # Age indices
        age_array = _resolve_ages(op.ages, n_ages)
        age_data_list.extend(age_array.tolist())
        age_offsets.append(len(age_data_list))
        
        # Sex mask
        sex_mask = _resolve_sex(op.sex)
        sex_masks_list.append(sex_mask)
        
        # Parameter
        params_list.append(float(op.param))
        
        # Condition
        cond_type, cond_param = _parse_condition(op.condition)
        condition_types_list.append(cond_type)
        condition_params_list.append(cond_param)
    
    n_ops = len(ops)
    
    plan = CompiledHookPlan(
        n_ops=n_ops,
        op_types=np.array(op_types_list, dtype=np.int32),
        gidx_offsets=np.array(gidx_offsets, dtype=np.int32),
        gidx_data=np.array(gidx_data_list, dtype=np.int32) if gidx_data_list else np.array([], dtype=np.int32),
        age_offsets=np.array(age_offsets, dtype=np.int32),
        age_data=np.array(age_data_list, dtype=np.int32) if age_data_list else np.array([], dtype=np.int32),
        sex_masks=np.vstack(sex_masks_list) if sex_masks_list else np.zeros((0, 2), dtype=np.bool_),
        params=np.array(params_list, dtype=np.float64),
        condition_types=np.array(condition_types_list, dtype=np.int32),
        condition_params=np.array(condition_params_list, dtype=np.int32),
    )
    
    meta = {
        'n_genotypes': n_genotypes,
        'n_ages': n_ages,
    }
    
    return CompiledHookDescriptor(
        name=name,
        event=event,
        priority=priority,
        plan=plan,
        meta=meta,
    )


def compile_selector_hook(
    func: Callable,
    pop: 'AgeStructuredPopulation',
    event: str,
    selectors_spec: Dict[str, Any],
    priority: int = 0,
    numba_mode: bool = False,
) -> CompiledHookDescriptor:
    """Compile a selector-based hook.
    
    Supports two modes:
    1. Python mode (default): User function receives (pop, **resolved_selectors)
    2. Numba mode: User function is @njit with signature (ind_count, tick, sel1, sel2, ...)
       and receives pre-resolved integer indices as positional arguments
    
    Args:
        func: User function (Python callable or @njit function)
        pop: Population instance
        event: Event name
        selectors_spec: Dict of selector specifications to resolve
        priority: Execution priority
        numba_mode: If True, generates Numba-compatible wrapper
    
    Returns:
        CompiledHookDescriptor: Compiled descriptor with njit_fn (Numba mode)
                                or py_wrapper (Python mode)
    
    Example (Python mode):
        >>> @hook(event='early', selectors={'target_gidx': 'AA'})
        ... def my_hook(pop, target_gidx):
        ...     pop.state.individual_count[:, :, target_gidx] *= 0.9
    
    Example (Numba mode):
        >>> @njit
        ... def my_numba_hook(ind_count, tick, target_gidx):
        ...     ind_count[:, :, target_gidx] *= 0.9
        ...     return 0
        >>> 
        >>> @hook(event='early', selectors={'target_gidx': 'AA'}, numba_mode=True)
        ... def wrapper(ind_count, tick, target_gidx):
        ...     return my_numba_hook(ind_count, tick, target_gidx)
    """
    index_core = pop._index_core
    diploid_genotypes = index_core.index_to_genotype
    
    # Resolve all selectors
    resolved = {}
    for name, spec in selectors_spec.items():
        resolved[name] = _resolve_selector_to_array(spec, index_core, diploid_genotypes)
    
    meta = {
        'n_genotypes': index_core.num_genotypes(),
        'n_ages': pop._n_ages,
    }
    
    # Check if function is already @njit decorated
    is_njit_fn = hasattr(func, 'py_func')  # Numba dispatcher has py_func attribute
    
    if numba_mode or is_njit_fn:
        # Numba mode: generate a wrapper that calls the function with hardcoded indices
        njit_fn = _compile_selector_njit_wrapper(func, resolved)
        
        return CompiledHookDescriptor(
            name=func.__name__,
            event=event,
            priority=priority,
            selectors=resolved,
            meta=meta,
            njit_fn=njit_fn,
        )
    else:
        # Python mode: create a Python wrapper
        def py_wrapper(population):
            # For single-element arrays, pass the scalar value
            kwargs = {}
            for k, v in resolved.items():
                if len(v) == 1:
                    kwargs[k] = int(v[0])
                else:
                    kwargs[k] = v
            func(population, **kwargs)
        
        return CompiledHookDescriptor(
            name=func.__name__,
            event=event,
            priority=priority,
            selectors=resolved,
            meta=meta,
            py_wrapper=py_wrapper,
        )


def _compile_selector_njit_wrapper(
    user_fn: Callable,
    resolved_selectors: Dict[str, np.ndarray]
) -> Callable:
    """Generate a Numba-compatible wrapper for selector-based hook.
    
    Generates code like:
        @njit(cache=False)
        def selector_wrapper(ind_count, tick):
            return user_fn(ind_count, tick, 2, 3)  # hardcoded indices
    
    Args:
        user_fn: User's @njit function
        resolved_selectors: Dict of {param_name: index_array}
    
    Returns:
        Callable: @njit wrapper function
    """
    # Build argument list with hardcoded values
    arg_lines = []
    for name, indices in resolved_selectors.items():
        if len(indices) == 1:
            # Single value - pass as scalar
            arg_lines.append(f"{name}={int(indices[0])}")
        else:
            # Multiple values - need to handle differently
            # For now, pass the first one (TODO: support arrays)
            arg_lines.append(f"{name}={int(indices[0])}")
    
    args_str = ", ".join(arg_lines)
    
    fn_name = f"_selector_wrapper_{user_fn.__name__}"
    
    code = f"""
@_njit_switch(cache=False)
def {fn_name}(ind_count, tick):
    return _user_fn(ind_count, tick, {args_str})
"""
    
    # Create execution namespace
    global_ns = {
        '_njit_switch': _njit_switch,
        '_user_fn': user_fn,
        'np': np,
    }
    
    exec(code, global_ns)
    return global_ns[fn_name]


# =============================================================================
# Numba Executor - Combined Hook Compilation
# =============================================================================

# Use njit_switch instead of direct numba.njit for configurable compilation
_njit_switch = njit_switch

# No-op hook function (used as default when no hooks are registered)
@_njit_switch(cache=False)
def _noop_hook(ind_count, tick):
    """Default no-op hook that does nothing and returns CONTINUE."""
    return 0

# Re-export for external use
noop_hook = _noop_hook


def compile_combined_hook(njit_fns: List[Callable], name: str = "combined_hook") -> Callable:
    """Compile multiple njit hook functions into a single combined function.
    
    This uses dynamic code generation to create a new @njit function that
    calls each hook in sequence, allowing the combined function to be
    called from within other @njit functions (like run_tick).
    
    Args:
        njit_fns: List of @njit decorated hook functions.
                  Each must have signature: (ind_count, tick) -> int
        name: Name for the generated function
    
    Returns:
        Callable: A single @njit function that executes all hooks in order.
                  Returns RESULT_STOP (1) if any hook returns non-zero,
                  otherwise returns RESULT_CONTINUE (0).
    
    Example:
        >>> @njit
        ... def hook1(ind_count, tick):
        ...     ind_count[0, 0, :] *= 0.9
        ...     return 0
        >>> 
        >>> @njit
        ... def hook2(ind_count, tick):
        ...     if tick % 10 == 0:
        ...         ind_count[1, 2, 0] += 100
        ...     return 0
        >>> 
        >>> combined = compile_combined_hook([hook1, hook2], "my_hooks")
        >>> # combined can now be called from within @njit functions
    """
    if len(njit_fns) == 0:
        return _noop_hook
    
    if len(njit_fns) == 1:
        return njit_fns[0]
    
    # Create global namespace with all function references
    fn_names = [f'_fn_{i}' for i in range(len(njit_fns))]
    global_ns = {fn_name: fn for fn_name, fn in zip(fn_names, njit_fns)}
    global_ns['_njit_switch'] = _njit_switch
    
    # Generate code that calls each function in sequence
    # Note: cache=False because dynamically generated functions can't be cached
    lines = [
        '@_njit_switch(cache=False)',
        f'def {name}(ind_count, tick):',
    ]
    for fn_name in fn_names:
        lines.append(f'    _result = {fn_name}(ind_count, tick)')
        lines.append('    if _result != 0:')
        lines.append('        return _result')
    lines.append('    return 0')
    
    code = '\n'.join(lines)
    
    # Execute to create the function
    exec(code, global_ns)
    
    return global_ns[name]


class CompiledEventHooks:
    """Container for compiled hooks organized by event.
    
    This class holds the combined @njit functions for each event,
    ready to be passed directly to run_tick.
    
    Attributes:
        first: Combined hook for FIRST event (or noop_hook)
        reproduction: Combined hook for REPRODUCTION event
        early: Combined hook for EARLY event
        survival: Combined hook for SURVIVAL event
        late: Combined hook for LATE event
        finish: Combined hook for FINISH event
        registry: HookRegistry for CSR operations
    """
    
    __slots__ = ('first', 'reproduction', 'early', 'survival', 'late', 
                 'finish', 'registry', '_event_hooks')
    
    def __init__(self):
        self.first = _noop_hook
        self.reproduction = _noop_hook
        self.early = _noop_hook
        self.survival = _noop_hook
        self.late = _noop_hook
        self.finish = _noop_hook
        self.registry = None  # Optional HookRegistry for CSR ops
        self._event_hooks = {
            'first': _noop_hook,
            'reproduction': _noop_hook,
            'early': _noop_hook,
            'survival': _noop_hook,
            'late': _noop_hook,
            'finish': _noop_hook,
        }
    
    def get_hook(self, event_name: str) -> Callable:
        """Get the combined hook function for an event."""
        return self._event_hooks.get(event_name, _noop_hook)
    
    def set_hook(self, event_name: str, hook_fn: Callable) -> None:
        """Set the combined hook function for an event."""
        self._event_hooks[event_name] = hook_fn
        setattr(self, event_name, hook_fn)
    
    @staticmethod
    def from_compiled_hooks(
        compiled_hooks: List['CompiledHookDescriptor'],
        registry: Optional['HookRegistry'] = None
    ) -> 'CompiledEventHooks':
        """Build CompiledEventHooks from a list of CompiledHookDescriptor.
        
        Collects all njit_fn hooks for each event and compiles them into
        combined functions.
        
        Args:
            compiled_hooks: List of CompiledHookDescriptor
            registry: Optional HookRegistry for CSR operations
        
        Returns:
            CompiledEventHooks: Ready-to-use hook container
        """
        result = CompiledEventHooks()
        result.registry = registry
        
        # Group njit hooks by event
        hooks_by_event: Dict[str, List[Callable]] = {name: [] for name in EVENT_NAMES}
        
        for desc in compiled_hooks:
            if desc.njit_fn is not None and desc.event in hooks_by_event:
                hooks_by_event[desc.event].append((desc.priority, desc.njit_fn))
        
        # Sort by priority and compile combined functions
        for event_name, hook_list in hooks_by_event.items():
            if hook_list:
                # Sort by priority (lower = earlier)
                hook_list.sort(key=lambda x: x[0])
                njit_fns = [fn for _, fn in hook_list]
                combined = compile_combined_hook(njit_fns, f"combined_{event_name}_hooks")
                result.set_hook(event_name, combined)
        
        return result




# =============================================================================
# Hook Decorator
# =============================================================================

def hook(
    event: Optional[str] = None,
    selectors: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    numba: bool = False
):
    """Decorator for defining compiled hooks.
    
    Decorates a function to mark it as a hook. The event can be specified
    either in the decorator or later when calling `pop.set_hook()`.
    
    Three usage modes:
    
    1. Declarative (function returns list of Op.*):
        @hook()
        def my_hook():
            return [Op.scale(...), Op.add(...)]
        pop.set_hook('early', my_hook)
    
    2. Selector-based (pre-resolved indices injected):
        @hook(selectors={'target': 'AA'})
        def my_hook(pop, target):
            pop.state.individual_count[:, :, target] *= 0.9
        pop.set_hook('first', my_hook)
    
    3. Native Numba (user-provided njit function):
        @hook(numba=True)
        def my_hook(ind_count, tick, selectors, rng_seed):
            # Pure numba code
            pass
        pop.set_hook('late', my_hook)
    
    You can also specify event in decorator (legacy style):
        @hook(event='early')
        def my_hook():
            return [Op.scale(...)]
        my_hook.register(pop)
    
    Args:
        event: Event name (optional, can be specified in set_hook instead)
        selectors: Dict of selector specs to pre-resolve (for mode 2)
        priority: Execution priority (lower = earlier)
        numba: If True, treat as native Numba function (mode 3)
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Store metadata on function for auto-detection
        func._hook_meta = {
            'event': event,
            'selectors': selectors or {},
            'priority': priority,
            'numba_mode': numba,
        }
        func._hook_compiled = None  # Will be set on register
        
        # Keep legacy style for backward compatibility
        func._hook_event = event
        func._hook_selectors_spec = selectors or {}
        func._hook_priority = priority
        func._hook_numba_mode = numba
        
        def register(pop: 'BasePopulation', event_override: Optional[str] = None) -> CompiledHookDescriptor:
            """Register this hook to a population and return the compiled descriptor.
            
            Args:
                pop: Population instance
                event_override: Override event name (if not set in decorator)
            
            Returns:
                CompiledHookDescriptor
            """
            actual_event = event_override or event
            if actual_event is None:
                raise ValueError(
                    f"Event not specified for hook '{func.__name__}'. "
                    "Specify in decorator @hook(event='...') or call pop.set_hook('event', hook)"
                )
            
            if numba:
                # Mode 3: Native Numba
                desc = CompiledHookDescriptor(
                    name=func.__name__,
                    event=actual_event,
                    priority=priority,
                    njit_fn=func,
                    meta={'n_genotypes': pop._index_core.num_genotypes(), 'n_ages': pop._n_ages},
                )
            elif selectors:
                # Mode 2: Selector-based (with optional Numba support)
                desc = compile_selector_hook(func, pop, actual_event, selectors, priority, numba_mode=numba)
            else:
                # Mode 1: Declarative (or plain Python hook)
                try:
                    result = func()
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], HookOp):
                        desc = compile_declarative_hook(result, pop, actual_event, priority, func.__name__)
                    else:
                        # Plain Python hook
                        desc = CompiledHookDescriptor(
                            name=func.__name__,
                            event=actual_event,
                            priority=priority,
                            py_wrapper=lambda p, f=func: f(p),
                            meta={'n_genotypes': pop._index_core.num_genotypes(), 'n_ages': pop._config.n_ages},
                        )
                except TypeError:
                    # Function requires arguments - treat as plain hook
                    desc = CompiledHookDescriptor(
                        name=func.__name__,
                        event=actual_event,
                        priority=priority,
                        py_wrapper=func,
                        meta={'n_genotypes': pop._index_core.num_genotypes(), 'n_ages': pop._config.n_ages},
                    )
            
            func._hook_compiled = desc
            pop._register_compiled_hook(desc)
            return desc
        
        func.register = register
        return func
    
    return decorator

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core types
    'OpType',
    'HookOp',
    'Op',
    'CompiledHookPlan',
    'CompiledHookDescriptor',
    'HookRegistry',
    'HookExecutor',
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
    # Event IDs
    'EVENT_FIRST',
    'EVENT_REPRODUCTION',
    'EVENT_EARLY',
    'EVENT_SURVIVAL',
    'EVENT_LATE',
    'EVENT_FINISH',
    # Result codes
    'RESULT_CONTINUE',
    'RESULT_STOP',
]
