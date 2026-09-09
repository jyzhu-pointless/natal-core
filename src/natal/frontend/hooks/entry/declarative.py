"""Declarative hook authoring and compilation.

This module is the "front-end compiler" for Op-based hooks:

1) User code returns a list of ``HookOp`` objects via ``Op.*`` helpers.
2) Symbolic selectors (genotype/age/sex) are resolved to integer arrays.
3) Condition strings are compiled into an RPN token stream.
4) Everything is packed into a ``CompiledHookPlan`` (CSR-like arrays).

The resulting plan is pure data and is executed by the native Rust engine.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Species
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
    ECO_PARAM_NAMES,
    RPN_ADD,
    RPN_DIV,
    RPN_LITERAL,
    RPN_MUL,
    RPN_PARAM,
    RPN_SUB,
    CompiledHookDescriptor,
    CompiledHookPlan,
    DemeSelector,
    HookLayout,
    HookOp,
    OpType,
)
from natal.frontend.patterns import resolve_zygote_type as _resolve_zygote_type
from natal.frontend.registry.index import IndexRegistry

# Fast membership view of the fixed set_param target/operand table.
_ECO_PARAM_SET = frozenset(ECO_PARAM_NAMES)


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

    @staticmethod
    def set_param(
        param: str,
        value: Union[str, float, int],
        every: int = 1,
        start: int = 0,
        when: Optional[str] = None,
        event: Optional[str] = None,
        priority: int = 0,
    ) -> HookOp:
        """Create a no-code parameter schedule operation (issue 26).

        The op rewrites one runtime-mutable ecology scalar on a tick
        schedule, e.g.::

            nt.Op.set_param("carrying_capacity", "K * 0.95", every=10)

        The *value* is an arithmetic expression over the current
        parameter values: operands are jsonc parameter names (``K`` is a
        registered alias of ``carrying_capacity``) or numeric literals,
        operators are ``+ - * /`` with the usual precedence and optional
        parentheses.  A plain number is accepted as sugar for a constant
        expression.  The expression is evaluated **every firing tick
        against the current values**, so ``"K * 0.95"`` compounds.

        Only ecology-section scalars that are runtime-mutable 0-d draft
        arrays *and* Rust session columns are legal targets (see
        ``ECO_PARAM_NAMES``: carrying_capacity, eggs_per_female, sex_ratio,
        sperm_displacement_rate, low_density_growth_rate).  Vector and
        genetics-tensor parameters raise ``ValueError`` — use
        ``pop.update()`` / ``pop.params.tensor_write()`` for those.

        The op fires at an event boundary (``first`` / ``early`` /
        ``late``; default ``early``) whenever
        ``tick >= start and (tick - start) % every == 0`` and the optional
        ``when`` condition holds.  On the Python and Rust lifecycles the
        write flushes through the same channel as
        ``pop.params.<name> = ...`` (route dispatch, Rust dirty bridge,
        and the ``(tick, name, old, new)`` parameter snapshot log) and is
        visible to the later stages of the same tick.  On the Rust
        ``run()`` path the write evolves inside the session-owned ecology
        columns with the same event granularity and the same jsonc bounds
        (a non-finite or out-of-bounds value, e.g. ``"K / 0"``, raises
        ``ValueError`` mid-run); when ``run()`` returns, the audited
        transitions are appended to ``params_log`` and the final values
        are synchronized into the draft.

        Args:
            param: Target parameter name (jsonc name or alias).
            value: RPN-source expression string or a plain number.
            every: Fire every *every* ticks (>= 1).
            start: First tick the schedule is active (>= 0).
            when: Optional extra condition expression.
            event: Event boundary at which the op fires (default early).
            priority: Hook priority when registered standalone.

        Returns:
            HookOp: Operation descriptor for compilation.

        Raises:
            ValueError: If *every* < 1 or *start* < 0 (the target name
                and the expression are validated at compile time).
        """
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        return HookOp(
            OpType.SET_PARAM,
            "*",
            "*",
            "both",
            0.0,
            when,
            event,
            priority,
            param_name=param,
            value_expr=value,
            every=every,
            start=start,
        )

    @staticmethod
    def convert(
        source: str,
        target: str,
        probability: float,
        when: Optional[str] = None,
        event: Optional[str] = None,
        priority: int = 0,
    ) -> HookOp:
        """Create a one-to-one probabilistic zygote-type conversion (issue 35).

        Each individual currently in the *source* zygote type moves to the
        *target* zygote type with *probability*, independently per
        individual (per age class).  Both patterns must each match exactly
        one ZType; anything else raises ``ValueError`` at compile time.

        Semantics by model:

        - **Males**: only ``individual_count`` rows migrate (males carry
          no sperm label).
        - **Females (age-structured)**: the virgin part and *every sperm
          bucket* ``(female_z, male_z)`` are binomially sampled and moved
          atomically to ``(target_z, male_z)`` — the stored sperm genotype
          label follows the female row, the male axis is untouched.
          Total counts are conserved exactly in deterministic mode and in
          expectation in stochastic mode.
        - **Discrete-generation**: no sperm storage, so the op degenerates
          to plain per-individual binomial migration.

        Multiple ``convert`` ops execute in hook-priority order, so a
        one-to-many split is expressed as a chain of conditional binomial
        draws with a ``probability=1.0`` remainder step::

            # 30 % of A|A become A|a, the rest become a|a
            nt.Op.convert("A|A", "A|a", probability=0.3),
            nt.Op.convert("A|A", "a|a", probability=1.0),

        Args:
            source: Genotype pattern that must match exactly one ZType.
            target: Genotype pattern that must match exactly one ZType.
            probability: Per-individual conversion probability in [0, 1].
            when: Optional condition expression.
            event: Event boundary at which the op fires (default early).
            priority: Hook priority when registered standalone.

        Returns:
            HookOp: Operation descriptor for compilation.

        Raises:
            ValueError: If *probability* is outside [0, 1] (pattern
                resolution is validated at compile time).
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got {probability}"
            )
        return HookOp(
            OpType.CONVERT,
            source,
            "*",
            "both",
            float(probability),
            when,
            event,
            priority,
            target_z=target,
        )


def _resolve_genotypes(
    selector: Union[str, List[str], Literal["*"]],
    index_registry: IndexRegistry,
    species: Species,
    n_ztypes: int,
) -> np.ndarray:
    """Resolve genotype selector syntax into concrete ZType indices.

    Supported input forms:
    - ``"*"`` — all ZTypes
    - genotype label (``"AA"``) or label list — resolved via ZygoteTypePattern
    - ``@slab`` syntax (``"AA@infected"``) — genotype with slab constraint
    - raw integer index or index list

    Args:
        selector: Genotype selector expression
        index_registry: Registry for genotype name resolution
        species: Species for genotype pattern resolution
        n_ztypes: Total number of ZType indices

    Returns:
        np.ndarray: Array of ZType indices (int32)

    Raises:
        ValueError: If genotype cannot be resolved
    """
    if selector == "*":
        return np.arange(n_ztypes, dtype=np.int32)

    if isinstance(selector, str):
        selector = [selector]

    indices: List[int] = []
    for item in selector:
        if isinstance(item, int):
            indices.append(item)
            continue

        z_indices = _resolve_zygote_type(item, species, index_registry)
        if not z_indices:
            raise ValueError(f"Cannot resolve genotype: {item}")
        indices.extend(z_indices)

    return np.array(indices, dtype=np.int32)


# _resolve_zygote_type is imported from natal.frontend.patterns


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

    Examples:
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
    - no recursion, no Python objects, and predictable control flow in Rust

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


# ---------------------------------------------------------------------------
# Value-expression compiler (Op.set_param RPN payloads)
# ---------------------------------------------------------------------------

_VALUE_TOKEN_RE = re.compile(
    r"(?P<number>\d+\.\d*|\.\d+|\d+)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>[-+*/()])"
)


def _resolve_eco_param_id(name: str) -> int:
    """Resolve one operand/target name to its fixed ECO param id.

    Args:
        name: jsonc parameter name (full key, short name, or alias).

    Returns:
        int: Index into ``ECO_PARAM_NAMES``.

    Raises:
        ValueError: If the name is unknown, or known but not one of the
            runtime-mutable ecology scalars (vectors and genetics tensors
            are called out explicitly in the message).
    """
    from natal.frontend.configurator._routes import lookup_or_none

    entry = lookup_or_none(name)
    if entry is None:
        raise ValueError(
            f"Unknown parameter name {name!r}; expected one of "
            f"{sorted(_ECO_PARAM_SET)} (or a registered alias)."
        )
    if entry.name in _ECO_PARAM_SET:
        return ECO_PARAM_NAMES.index(entry.name)
    kind_hint = (
        "a vector parameter"
        if entry.kind in ("age_vec", "sex_row", "slot")
        else "a tensor parameter"
        if entry.kind == "geno_tensor"
        else f"a {entry.kind!r} parameter"
    )
    raise ValueError(
        f"Op.set_param target {name!r} resolves to {kind_hint} "
        f"({entry.name}); only ecology scalars "
        f"{sorted(_ECO_PARAM_SET)} are settable by Op.set_param."
    )


def _compile_value_expr(
    expr: Union[str, float, int, None, bool],
) -> Tuple[List[int], List[int], List[float]]:
    """Compile a set_param value expression into flat RPN token arrays.

    The expression grammar is: identifiers (parameter names), numeric
    literals, binary ``+ - * /`` with standard precedence, and optional
    parentheses.  Shunting-yard produces the postfix token stream; a
    depth simulation rejects malformed programs at compile time so the
    runtime stack machine can trust its input.

    Args:
        expr: Expression string, or a plain number (constant sugar).

    Returns:
        Tuple of ``(kinds, payload, literals)`` — token kinds
        (``RPN_LITERAL`` / ``RPN_PARAM`` / ``RPN_ADD..RPN_DIV``), the
        per-operand payload (literal pool index or ECO param id), and
        the float64 literal pool.

    Raises:
        ValueError: On lexing failures, unknown identifiers, mismatched
            parentheses, or a malformed operator/operand sequence.
        TypeError: On non-numeric, non-string inputs (``bool``/``None``).
    """
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return ([RPN_LITERAL], [0], [float(expr)])
    if not isinstance(expr, str):
        raise TypeError(
            "set_param value must be a string expression or a number, "
            f"got {type(expr).__name__}"
        )

    source = expr.strip()
    if not source:
        raise ValueError("set_param value expression cannot be empty")

    # -- Pass 1: lex into (kind, payload) infix tokens.
    infix: List[Tuple[str, object]] = []
    literals: List[float] = []
    pos = 0
    n = len(source)
    while pos < n:
        ch = source[pos]
        if ch.isspace():
            pos += 1
            continue
        match = _VALUE_TOKEN_RE.match(source, pos)
        if match is None:
            raise ValueError(
                f"Unsupported set_param value syntax near: {source[pos:]!r}"
            )
        if match.lastgroup == "number":
            literals.append(float(match.group("number")))
            infix.append(("operand", ("lit", len(literals) - 1)))
        elif match.lastgroup == "ident":
            param_id = _resolve_eco_param_id(match.group("ident"))
            infix.append(("operand", ("param", param_id)))
        else:
            infix.append(("op", match.group("op")))
        pos = match.end()

    # -- Pass 2: shunting-yard to postfix.
    kinds: List[int] = []
    payload: List[int] = []
    op_stack: List[Tuple[str, int]] = []  # (symbol, precedence)
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}

    for token_kind, value in infix:
        if token_kind == "operand":
            tag, index = cast("Tuple[str, int]", value)
            kinds.append(RPN_LITERAL if tag == "lit" else RPN_PARAM)
            payload.append(index)
            continue
        symbol = cast("str", value)
        if symbol == "(":
            op_stack.append((symbol, 0))
            continue
        if symbol == ")":
            while op_stack and op_stack[-1][0] != "(":
                sym, _ = op_stack.pop()
                kinds.append(_RPN_OP_CODE[sym])
                payload.append(0)
            if not op_stack:
                raise ValueError(
                    f"Mismatched parentheses in set_param value: {source!r}"
                )
            op_stack.pop()  # discard "("
            continue
        # Binary operator: pop strictly-higher-precedence stack tops.
        prec = precedence[symbol]
        while (
            op_stack
            and op_stack[-1][0] != "("
            and op_stack[-1][1] >= prec
        ):
            sym, _ = op_stack.pop()
            kinds.append(_RPN_OP_CODE[sym])
            payload.append(0)
        op_stack.append((symbol, prec))

    while op_stack:
        sym, _ = op_stack.pop()
        if sym == "(":
            raise ValueError(
                f"Mismatched parentheses in set_param value: {source!r}"
            )
        kinds.append(_RPN_OP_CODE[sym])
        payload.append(0)

    # -- Pass 3: depth simulation — malformed programs fail here, not in
    # the runtime stack machine.
    depth = 0
    for kind in kinds:
        if kind in (RPN_LITERAL, RPN_PARAM):
            depth += 1
        else:
            if depth < 2:
                raise ValueError(
                    f"Malformed set_param value expression: {source!r}"
                )
            depth -= 1
    if depth != 1:
        raise ValueError(f"Malformed set_param value expression: {source!r}")

    return kinds, payload, literals


# Operator symbol -> RPN opcode (kept as a module table for symmetry with
# the condition compiler's precedence map).
_RPN_OP_CODE: Dict[str, int] = {
    "+": RPN_ADD,
    "-": RPN_SUB,
    "*": RPN_MUL,
    "/": RPN_DIV,
}


def _compile_convert_endpoint(
    pattern: str,
    species: Species,
    index_registry: IndexRegistry,
    role: str,
) -> int:
    """Resolve one convert endpoint pattern to its single ZType index.

    Args:
        pattern: Genotype pattern string (e.g. ``"A|A"``).
        species: Species for pattern resolution.
        index_registry: Registry for ZType index resolution.
        role: ``"source"`` or ``"target"`` — used in error messages.

    Returns:
        int: The unique matching ZType index.

    Raises:
        ValueError: If the pattern matches zero or multiple ZTypes; the
            message carries the full match list for debugging.
    """
    matches = _resolve_zygote_type(pattern, species, index_registry)
    if len(matches) == 1:
        return matches[0]
    # resolve_ztype_indices only returns in-range indices, so the name
    # projection below is total.
    from natal.contracts.blueprint import format_type_name

    names = [
        format_type_name(gt, slab)
        for gt, slab in (index_registry.index_to_ztype[z] for z in matches)
    ]
    raise ValueError(
        f"Op.convert {role} pattern {pattern!r} must match exactly one "
        f"zygote type, but matched {len(matches)}: {names}"
    )


def compile_declarative_hook(
    ops: List[HookOp],
    pop: HookLayout,
    event: str,
    priority: int = 0,
    deme_selector: DemeSelector = "*",
    name: str = "declarative_hook",
) -> CompiledHookDescriptor:
    """Compile declarative ops into a ``CompiledHookDescriptor``.

    The compiler packs all per-op fields into parallel arrays. Offsets arrays
    (``*_offsets``) define CSR spans for variable-length selector/condition
    data and avoid Python object usage in runtime engine.

    Args:
        ops: Declarative operations to compile.
        pop: Layout provider (a built population or the builder's
            build-time context); only its ``index_registry``, ``species``,
            and ``config.n_ages`` are read.
        event: Event this hook fires at.
        priority: Execution priority — lower values run first.
        deme_selector: Deme selector carried on the descriptor.
        name: Human-readable descriptor name.
    """
    # Get population configuration and registry for resolving genotype/age indices
    index_registry = pop.index_registry
    species = index_registry.index_to_genotype[0].species if index_registry.index_to_genotype else pop.species
    n_ztypes = index_registry.n_ztypes
    n_ages = pop.config.n_ages

    # Initialize data structures for storing compiled hook operations
    # These will be packed into parallel arrays for efficient runtime execution

    # 1. Operation type stream - stores the operation code for each hook
    op_types_list: List[int] = []

    # 2. Genotype selection data (CSR format)
    # zidx_offsets: CSR offsets defining genotype index ranges for each operation
    # zidx_data: Flattened list of all genotype indices across all operations
    zidx_offsets: List[int] = [0]  # Start with offset 0 for the first operation
    zidx_data_list: List[int] = []

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

    # 6. OP_SET_PARAM data area: per-op schedule columns, a shared RPN
    # token stream (CSR via rpn_offsets), and a shared literal pool.
    sp_param_ids: List[int] = []
    sp_every_list: List[int] = []
    sp_start_list: List[int] = []
    rpn_offsets: List[int] = [0]
    rpn_kinds_list: List[int] = []
    rpn_payload_list: List[int] = []
    sp_literals_list: List[float] = []

    # 7. OP_CONVERT data area: resolved single-ZType endpoints per op.
    convert_source_z: List[int] = []
    convert_target_z: List[int] = []

    # Process each hook operation and compile it into the packed arrays
    for op in ops:
        # 1) Operation type - convert enum to integer for efficient runtime lookup
        op_types_list.append(int(op.op_type))

        # 2) Genotype span - resolve genotype selectors to actual genotype indices
        # Examples: "A1|A1" -> [0], "*" -> [0, 1, 2, ..., n_genotypes-1]
        zidx_array = _resolve_genotypes(op.genotypes, index_registry, species, n_ztypes)
        zidx_data_list.extend(zidx_array.tolist())
        zidx_offsets.append(len(zidx_data_list))  # Record end offset for this operation

        # 3) Age span - resolve age selectors to actual age indices
        # Examples: "0-5" -> [0, 1, 2, 3, 4, 5], "*" -> [0, 1, ..., n_ages-1]
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

        # 6) OP_SET_PARAM payload: validate the target name, compile the
        # value expression, and store the schedule columns.  Non-set_param
        # ops record the -1 sentinel so per-op columns stay aligned.
        if op.op_type == OpType.SET_PARAM:
            if op.param_name is None:
                raise ValueError("Op.set_param requires a parameter name")
            # A None / bool / non-scalar value expression raises TypeError
            # inside _compile_value_expr (type-of-input error); only the
            # target-name check above is a value error.
            sp_param_ids.append(_resolve_eco_param_id(op.param_name))
            sp_every_list.append(int(op.every))
            sp_start_list.append(int(op.start))
            kinds, payload, literals = _compile_value_expr(
                op.value_expr  # type: ignore[arg-type]  # None/bool reach the runtime TypeError arm by design
            )
            rpn_kinds_list.extend(kinds)
            # Literal payloads are indices into the plan-wide shared pool;
            # rebase each expression's local indices onto the current pool
            # length before appending, or later set_param ops would silently
            # read earlier ops' literals.
            pool_base = len(sp_literals_list)
            rpn_payload_list.extend(
                p + pool_base if k == RPN_LITERAL else p
                for k, p in zip(kinds, payload)
            )
            rpn_offsets.append(len(rpn_kinds_list))
            sp_literals_list.extend(literals)
        else:
            sp_param_ids.append(-1)
            sp_every_list.append(1)
            sp_start_list.append(0)
            rpn_offsets.append(len(rpn_kinds_list))

        # 7) OP_CONVERT payload: both endpoints must resolve to exactly
        # one ZType; anything else fails here with the match list.
        if op.op_type == OpType.CONVERT:
            source_pattern = op.genotypes if isinstance(op.genotypes, str) else str(op.genotypes)
            if op.target_z is None:
                raise ValueError("Op.convert requires a target genotype pattern")
            source_z = _compile_convert_endpoint(
                source_pattern, species, index_registry, "source"
            )
            target_z = _compile_convert_endpoint(
                op.target_z, species, index_registry, "target"
            )
            if source_z == target_z:
                raise ValueError(
                    f"Op.convert source and target must differ, both "
                    f"resolve to ztype {source_z}"
                )
            convert_source_z.append(source_z)
            convert_target_z.append(target_z)
        else:
            convert_source_z.append(-1)
            convert_target_z.append(-1)

    # Create the compiled execution plan with all packed arrays
    plan = CompiledHookPlan(
        n_ops=len(ops),  # Total number of operations

        # Operation type stream - each element is an integer operation code
        op_types=np.array(op_types_list, dtype=np.int32),

        # Genotype selection data in CSR format
        # zidx_offsets[i] to zidx_offsets[i+1] defines genotype indices for operation i
        zidx_offsets=np.array(zidx_offsets, dtype=np.int32),
        zidx_data=np.array(zidx_data_list, dtype=np.int32) if zidx_data_list else np.array([], dtype=np.int32),

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

        # OP_SET_PARAM data area (per-op columns + RPN token stream +
        # shared literal pool)
        sp_param_ids=np.array(sp_param_ids, dtype=np.int32),
        sp_every=np.array(sp_every_list, dtype=np.int32),
        sp_start=np.array(sp_start_list, dtype=np.int32),
        rpn_offsets=np.array(rpn_offsets, dtype=np.int32),
        rpn_kinds=np.array(rpn_kinds_list, dtype=np.int32),
        rpn_payload=np.array(rpn_payload_list, dtype=np.int32),
        sp_literals=np.array(sp_literals_list, dtype=np.float64),

        # OP_CONVERT data area (single-ZType endpoints per op)
        convert_source_z=np.array(convert_source_z, dtype=np.int32),
        convert_target_z=np.array(convert_target_z, dtype=np.int32),
    )

    # Return the complete hook descriptor with metadata
    return CompiledHookDescriptor(
        name=name,                    # Human-readable name for debugging
        event=event,                  # Simulation event when this hook triggers
        priority=priority,            # Execution priority (lower = earlier)
        deme_selector=deme_selector, # Which demes this hook applies to
        plan=plan,                    # Compiled execution plan
        meta={"n_ztypes": index_registry.n_ztypes, "n_ages": n_ages},  # Population metadata
        ops=ops,                     # Original operations for reference/debugging
    )
