"""Declarative hook authoring and compilation.

This module is the "front-end compiler" for Op-based hooks:

1) User code returns a list of ``HookOp`` objects via ``Op.*`` helpers.
2) Symbolic selectors (genotype/age/sex) are resolved to integer arrays.
3) Condition strings are compiled into an RPN token stream.
4) Everything is packed into a ``CompiledHookPlan`` (CSR-like arrays).

The resulting plan is pure data and can be executed inside njit kernels.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .types import (
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
    CompiledHookDescriptor,
    CompiledHookPlan,
    DemeSelector,
    HookOp,
    OpType,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation
    from natal.index_registry import IndexRegistry

class Op:
    """Factory helpers for building declarative operations.

    The methods here only build data objects and do not touch population state.
    Compilation happens later in ``compile_declarative_hook``.
    """

    @staticmethod
    def scale(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        factor: float = 1.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create a scaling operation that multiplies counts by a factor.

        Args:
            genotypes: Genotype selector ("*" for all, specific genotype, or list)
            ages: Age selector ("*" for all, specific age, range, or list)
            sex: Sex selector ("female", "male", or "both")
            factor: Scaling factor (e.g., 0.5 halves the count, 2.0 doubles it)
            when: Optional condition expression (e.g., "tick >= 100")

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.SCALE, genotypes, ages, sex, factor, when)

    @staticmethod
    def set_count(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        value: float = 0.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that sets counts to a specific value.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            value: Target count value (individuals will be added/removed to match)
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.SET, genotypes, ages, sex, value, when)

    @staticmethod
    def add(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        delta: float = 0.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that adds a fixed number of individuals.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            delta: Number of individuals to add (can be negative to remove)
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.ADD, genotypes, ages, sex, delta, when)

    @staticmethod
    def subtract(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        delta: float = 0.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that subtracts a fixed number of individuals.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            delta: Number of individuals to subtract
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.SUBTRACT, genotypes, ages, sex, delta, when)

    @staticmethod
    def kill(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        prob: float = 0.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create a probabilistic killing operation.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            prob: Probability of killing each selected individual (0.0 to 1.0)
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        return HookOp(OpType.KILL, genotypes, ages, sex, prob, when)

    @staticmethod
    def sample(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        size: int = 0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create a sampling operation that selects individuals without replacement.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            size: Number of individuals to sample
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.SAMPLE, genotypes, ages, sex, float(size), when)

    @staticmethod
    def stop_if_zero(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that stops the simulation if selected count reaches zero.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.STOP_IF_ZERO, genotypes, ages, sex, 0.0, when)

    @staticmethod
    def stop_if_below(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        threshold: float = 1.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that stops the simulation if count falls below threshold.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            threshold: Minimum count threshold
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.STOP_IF_BELOW, genotypes, ages, sex, float(threshold), when)

    @staticmethod
    def stop_if_above(
        genotypes: Union[str, List[str], Literal["*"]] = "*",
        ages: Union[int, List[int], range, Literal["*"]] = "*",
        sex: Literal["female", "male", "both"] = "both",
        threshold: float = 1_000_000.0,
        when: Optional[str] = None,
    ) -> HookOp:
        """Create an operation that stops the simulation if count exceeds threshold.

        Args:
            genotypes: Genotype selector
            ages: Age selector
            sex: Sex selector
            threshold: Maximum count threshold
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.STOP_IF_ABOVE, genotypes, ages, sex, float(threshold), when)

    @staticmethod
    def stop_if_extinction(when: Optional[str] = None) -> HookOp:
        """Create an operation that stops the simulation if total population goes extinct.

        Args:
            when: Optional condition expression

        Returns:
            HookOp: Operation descriptor for compilation
        """
        return HookOp(OpType.STOP_IF_EXTINCTION, "*", "*", "both", 0.0, when)


def _resolve_genotypes(
    selector: Union[str, List[str], Literal["*"]],
    index_registry: IndexRegistry,
    diploid_genotypes: List[Any],
    n_genotypes: int,
) -> np.ndarray:
    """Resolve genotype selector syntax into concrete genotype indices.

    Supported input forms:
    - ``"*"``
    - one label (``"AA"``) or a label list
    - raw integer index or index list

    Args:
        selector: Genotype selector expression
        index_registry: Registry for genotype name resolution
        diploid_genotypes: List of all diploid genotypes in the population
        n_genotypes: Total number of genotypes

    Returns:
        np.ndarray: Array of genotype indices (int32)

    Raises:
        ValueError: If genotype cannot be resolved
    """
    if selector == "*":
        return np.arange(n_genotypes, dtype=np.int32)

    if isinstance(selector, str):
        selector = [selector]

    indices: List[int] = []
    for item in selector:
        if isinstance(item, int):
            indices.append(item)
            continue

        idx = index_registry.resolve_genotype_index(diploid_genotypes, item, strict=True)
        if idx is None:
            raise ValueError(f"Cannot resolve genotype: {item}")
        indices.append(idx)

    return np.array(indices, dtype=np.int32)


def _resolve_ages(selector: Union[int, List[int], range, Literal["*"]], n_ages: int) -> np.ndarray:
    """Resolve age selector syntax to an int32 index vector.

    Args:
        selector: Age selector ("*" for all, integer, list, or range)
        n_ages: Total number of age classes in the population

    Returns:
        np.ndarray: Array of age indices (int32)
    """
    if selector == "*":
        return np.arange(n_ages, dtype=np.int32)
    if isinstance(selector, int):
        return np.array([selector], dtype=np.int32)
    if isinstance(selector, range):
        return np.array(list(selector), dtype=np.int32)
    return np.array(list(selector), dtype=np.int32)


def _resolve_sex(selector: Literal["female", "male", "both"]) -> np.ndarray:
    """Encode sex selector as a two-slot boolean mask: [female, male].

    Args:
        selector: Sex selector ("female", "male", or "both")

    Returns:
        np.ndarray: Boolean mask array [female_selected, male_selected]

    Raises:
        ValueError: If selector is not recognized
    """
    if selector == "both":
        return np.array([True, True], dtype=np.bool_)
    if selector == "female":
        return np.array([True, False], dtype=np.bool_)
    if selector == "male":
        return np.array([False, True], dtype=np.bool_)
    raise ValueError(f"Unknown sex selector: {selector}")


def _parse_atomic_condition(atom: str) -> Tuple[int, int]:
    """Parse one atomic predicate into ``(cond_type, parameter)``.

    Example:
    - ``tick % 10 == 0`` -> ``(COND_TICK_MOD, 10)``
    - ``tick >= 5`` -> ``(COND_TICK_GE, 5)``

    Args:
        atom: Atomic condition string (e.g., "tick >= 5")

    Returns:
        Tuple[int, int]: Condition type and parameter

    Raises:
        ValueError: If condition syntax is not supported
    """
    atom = atom.strip()

    match = re.fullmatch(r"tick\s*%\s*(\d+)\s*==\s*0", atom)
    if match:
        return (COND_TICK_MOD, int(match.group(1)))

    match = re.fullmatch(r"tick\s*==\s*(\d+)", atom)
    if match:
        return (COND_TICK_EQ, int(match.group(1)))

    match = re.fullmatch(r"tick\s*>=\s*(\d+)", atom)
    if match:
        return (COND_TICK_GE, int(match.group(1)))

    match = re.fullmatch(r"tick\s*>\s*(\d+)", atom)
    if match:
        return (COND_TICK_GT, int(match.group(1)))

    match = re.fullmatch(r"tick\s*<=\s*(\d+)", atom)
    if match:
        return (COND_TICK_LE, int(match.group(1)))

    match = re.fullmatch(r"tick\s*<\s*(\d+)", atom)
    if match:
        return (COND_TICK_LT, int(match.group(1)))

    raise ValueError(f"Unsupported atomic condition: {atom}")


def _tokenize_condition_expr(condition: str) -> List[Tuple[int, int]]:
    """Tokenize condition expression into operator/predicate tuples.

    Parentheses are encoded as negative sentinels so we can reuse one compact
    token representation all the way to the shunting-yard stage.

    Args:
        condition: Condition expression string (e.g., "tick >= 5 and tick % 10 == 0")

    Returns:
        List[Tuple[int, int]]: List of tokens (type, parameter)

    Raises:
        ValueError: If condition syntax is invalid
    """
    s = condition.strip()
    if not s:
        raise ValueError("Condition expression cannot be empty")

    tokens: List[Tuple[int, int]] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            tokens.append((-(ord("(")), 0))
            i += 1
            continue

        if ch == ")":
            tokens.append((-(ord(")")), 0))
            i += 1
            continue

        m = re.match(r"(and|or|not)\b", s[i:])
        if m:
            word = m.group(1)
            if word == "and":
                tokens.append((COND_OP_AND, 0))
            elif word == "or":
                tokens.append((COND_OP_OR, 0))
            else:
                tokens.append((COND_OP_NOT, 0))
            i += len(word)
            continue

        m = re.match(r"tick\s*%\s*\d+\s*==\s*0", s[i:])
        if m:
            atom = m.group(0)
            tokens.append(_parse_atomic_condition(atom))
            i += len(atom)
            continue

        m = re.match(r"tick\s*(==|>=|>|<=|<)\s*\d+", s[i:])
        if m:
            atom = m.group(0)
            tokens.append(_parse_atomic_condition(atom))
            i += len(atom)
            continue

        raise ValueError(f"Unsupported condition syntax near: {s[i:]!r}")

    return tokens


def _to_rpn_condition(tokens: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert infix condition tokens to RPN (Reverse Polish Notation).

    Why RPN:
    - the runtime kernel can evaluate RPN with a tiny fixed-size stack
    - no recursion, no Python objects, and predictable control flow in njit

    Args:
        tokens: List of tokens from _tokenize_condition_expr

    Returns:
        Tuple[np.ndarray, np.ndarray]: RPN token types and parameters

    Raises:
        ValueError: If condition expression is malformed
    """
    output_types: List[int] = []
    output_params: List[int] = []
    op_stack: List[Tuple[int, int]] = []

    precedence = {COND_OP_OR: 1, COND_OP_AND: 2, COND_OP_NOT: 3}
    right_assoc = {COND_OP_NOT}

    for tok_type, tok_param in tokens:
        if 0 <= tok_type <= COND_TICK_GT:
            # Atomic predicates (tick comparisons) go directly to output
            output_types.append(tok_type)
            output_params.append(tok_param)
            continue

        if tok_type in (COND_OP_AND, COND_OP_OR, COND_OP_NOT):
            # Handle operator precedence using shunting-yard algorithm
            while op_stack:
                top_type, top_param = op_stack[-1]
                if top_type < 0:  # Parenthesis marker
                    break
                if top_type not in precedence:
                    break
                p_top = precedence[top_type]
                p_cur = precedence[tok_type]
                should_pop = (p_top > p_cur) or (p_top == p_cur and tok_type not in right_assoc)
                if not should_pop:
                    break
                out_t, out_p = op_stack.pop()
                output_types.append(out_t)
                output_params.append(out_p)
            op_stack.append((tok_type, tok_param))
            continue

        if tok_type == -(ord("(")):
            # Left parenthesis - push to stack
            op_stack.append((tok_type, tok_param))
            continue

        if tok_type == -(ord(")")):
            # Right parenthesis - pop until matching left parenthesis
            found_left = False
            while op_stack:
                top_type, top_param = op_stack.pop()
                if top_type == -(ord("(")):
                    found_left = True
                    break
                output_types.append(top_type)
                output_params.append(top_param)
            if not found_left:
                raise ValueError("Mismatched parentheses in condition")
            continue

        raise ValueError(f"Unknown condition token type: {tok_type}")

    # Pop remaining operators from stack
    while op_stack:
        top_type, top_param = op_stack.pop()
        if top_type in (-(ord("(")), -(ord(")"))):
            raise ValueError("Mismatched parentheses in condition")
        output_types.append(top_type)
        output_params.append(top_param)

    if not output_types:
        raise ValueError("Invalid condition expression")

    # Validate stack behavior early so malformed expressions fail at compile
    # time rather than deep inside the runtime loop.
    depth = 0
    for tok in output_types:
        if 0 <= tok <= COND_TICK_GT:
            depth += 1
        elif tok == COND_OP_NOT:
            if depth < 1:
                raise ValueError("Invalid condition expression: malformed 'not'")
        elif tok in (COND_OP_AND, COND_OP_OR):
            if depth < 2:
                raise ValueError("Invalid condition expression: malformed binary operator")
            depth -= 1
        else:
            raise ValueError(f"Invalid condition expression token: {tok}")

    if depth != 1:
        raise ValueError("Invalid condition expression: missing logical operator")

    return (np.array(output_types, dtype=np.int32), np.array(output_params, dtype=np.int32))


def _parse_condition(condition: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compile optional condition string into runtime token arrays.

    Args:
        condition: Optional condition string (None means always true)

    Returns:
        Tuple[np.ndarray, np.ndarray]: RPN token types and parameters

    Raises:
        ValueError: If condition syntax is invalid
    """
    if condition is None:
        return (np.array([COND_ALWAYS], dtype=np.int32), np.array([0], dtype=np.int32))

    tokens = _tokenize_condition_expr(condition)
    return _to_rpn_condition(tokens)

# Public alias for external callers.
parse_condition = _parse_condition


def compile_declarative_hook(
    ops: List[HookOp],
    pop: BasePopulation[Any],
    event: str,
    priority: int = 0,
    deme_selector: DemeSelector = "*",
    name: str = "declarative_hook",
) -> CompiledHookDescriptor:
    """Compile declarative ops into a ``CompiledHookDescriptor``.

    The compiler packs all per-op fields into parallel arrays. Offsets arrays
    (``*_offsets``) define CSR spans for variable-length selector/condition
    data and avoid Python object usage in runtime kernels.
    """
    # Get population configuration and registry for resolving genotype/age indices
    index_registry = pop.index_registry
    diploid_genotypes = index_registry.index_to_genotype
    n_genotypes = index_registry.num_genotypes()
    n_ages = pop.config.n_ages

    # Initialize data structures for storing compiled hook operations
    # These will be packed into parallel arrays for efficient runtime execution

    # 1. Operation type stream - stores the operation code for each hook
    op_types_list: List[int] = []

    # 2. Genotype selection data (CSR format)
    # gidx_offsets: CSR offsets defining genotype index ranges for each operation
    # gidx_data: Flattened list of all genotype indices across all operations
    gidx_offsets: List[int] = [0]  # Start with offset 0 for the first operation
    gidx_data_list: List[int] = []

    # 3. Age selection data (CSR format)
    # age_offsets: CSR offsets defining age index ranges for each operation
    # age_data: Flattened list of all age indices across all operations
    age_offsets: List[int] = [0]  # Start with offset 0 for the first operation
    age_data_list: List[int] = []

    # 4. Sex selection and operation parameters
    # sex_masks: Boolean masks for male/female selection (2D array: [op][sex])
    # params: Numeric parameters for each operation (e.g., fitness values)
    sex_masks_list: List[NDArray[np.bool_]] = []
    params_list: List[float] = []

    # 5. Condition expression data (CSR format)
    # condition_offsets: CSR offsets defining condition token ranges for each operation
    # condition_types: Flattened list of condition operation types
    # condition_params: Flattened list of condition parameters
    condition_offsets: List[int] = [0]  # Start with offset 0 for the first operation
    condition_types_list: List[int] = []
    condition_params_list: List[int] = []

    # Process each hook operation and compile it into the packed arrays
    for op in ops:
        # 1) Operation type - convert enum to integer for efficient runtime lookup
        op_types_list.append(int(op.op_type))

        # 2) Genotype span - resolve genotype selectors to actual genotype indices
        # Example: "A1/A1" -> [0], "*" -> [0, 1, 2, ..., n_genotypes-1]
        gidx_array = _resolve_genotypes(op.genotypes, index_registry, diploid_genotypes, n_genotypes)
        gidx_data_list.extend(gidx_array.tolist())
        gidx_offsets.append(len(gidx_data_list))  # Record end offset for this operation

        # 3) Age span - resolve age selectors to actual age indices
        # Example: "0-5" -> [0, 1, 2, 3, 4, 5], "*" -> [0, 1, ..., n_ages-1]
        age_array = _resolve_ages(op.ages, n_ages)
        age_data_list.extend(age_array.tolist())
        age_offsets.append(len(age_data_list))  # Record end offset for this operation

        # 4) Sex mask + numeric parameter
        # Convert sex selector to boolean mask [male_selected, female_selected]
        sex_masks_list.append(_resolve_sex(op.sex))
        params_list.append(float(op.param))  # Convert parameter to float

        # 5) Compiled condition token span
        # Parse condition expression into RPN (Reverse Polish Notation) tokens
        cond_types, cond_params = _parse_condition(op.condition)
        condition_types_list.extend(cond_types.tolist())
        condition_params_list.extend(cond_params.tolist())
        condition_offsets.append(len(condition_types_list))  # Record end offset

    # Create the compiled execution plan with all packed arrays
    plan = CompiledHookPlan(
        n_ops=len(ops),  # Total number of operations

        # Operation type stream - each element is an integer operation code
        op_types=np.array(op_types_list, dtype=np.int32),

        # Genotype selection data in CSR format
        # gidx_offsets[i] to gidx_offsets[i+1] defines genotype indices for operation i
        gidx_offsets=np.array(gidx_offsets, dtype=np.int32),
        gidx_data=np.array(gidx_data_list, dtype=np.int32) if gidx_data_list else np.array([], dtype=np.int32),

        # Age selection data in CSR format
        # age_offsets[i] to age_offsets[i+1] defines age indices for operation i
        age_offsets=np.array(age_offsets, dtype=np.int32),
        age_data=np.array(age_data_list, dtype=np.int32) if age_data_list else np.array([], dtype=np.int32),

        # Sex selection masks - 2D boolean array [n_ops x 2]
        # Each row: [male_selected, female_selected]
        sex_masks=np.vstack(sex_masks_list) if sex_masks_list else np.zeros((0, 2), dtype=np.bool_),

        # Operation parameters - numeric values for each operation
        params=np.array(params_list, dtype=np.float64),

        # Condition expression data in CSR format
        # condition_offsets[i] to condition_offsets[i+1] defines condition tokens for operation i
        condition_offsets=np.array(condition_offsets, dtype=np.int32),
        condition_types=np.array(condition_types_list, dtype=np.int32),
        condition_params=np.array(condition_params_list, dtype=np.int32),
    )

    # Return the complete hook descriptor with metadata
    return CompiledHookDescriptor(
        name=name,                    # Human-readable name for debugging
        event=event,                  # Simulation event when this hook triggers
        priority=priority,            # Execution priority (higher = earlier)
        deme_selector=deme_selector, # Which demes this hook applies to
        plan=plan,                    # Compiled execution plan
        meta={"n_genotypes": n_genotypes, "n_ages": n_ages},  # Population metadata
        ops=ops,                     # Original operations for reference/debugging
    )
