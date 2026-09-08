"""Adversarial addendum tests for ``Op.set_param`` / ``Op.convert``.

Complements ``tests/test_ops_setparam_convert.py`` (which covers the
factory happy paths, one-fire scheduling, one sperm-bucket matrix, and a
start=0 parity program).  This file attacks the same surface from the
angles that file does not reach:

1. RPN compile-time attacks — unary minus, parenthesis errors, operand/
   operator juxtaposition, tensor/vector *operands* (not targets),
   400-token expressions, whitespace, and registration atomicity.
2. RPN numeric matrix — compound expressions equal a same-order Python
   evaluation bit-for-bit; explicit IEEE-754 division semantics
   (``x/0`` = ±inf, ``0/0`` = nan) at the kernel level; compounding
   schedules re-read operands written by earlier ops.
3. Scheduling semantics — ``start``/``every`` boundaries (fire at
   3/8/13), ``when`` AND-composed with the schedule, declaration order
   within one plan and priority chaining across descriptors, non-``early``
   event boundaries.
4. Convert numerical invariants (focus) — a non-trivial three-bucket
   sperm matrix with per-bucket deterministic ``n*p`` migration, exact
   per-trial row conservation plus 3-sigma bucket statistics over 200
   stochastic runs, three-way split chains, relay chains whose
   declaration order is observable, and probability boundary values.
5. Compile-time validation addenda — zero-match *target*, group
   registration atomicity for convert.
6. Schedule-state persistence with a non-zero ``start`` and a ``when``
   clause, verified per tick across separate ``run()`` call boundaries.
7. Write-channel integrity — exact ``(tick, name, old, new)`` rows on an
   ``every=2`` schedule, snapshot ownership of ``params_log``, and
   schedule persistence across ``run`` call boundaries.
8. Spatial models — deme-selector ``set_param`` writes only the selected
   deme columns; ``convert`` applies per deme independently.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import natal as nt  # noqa: E402
from natal.frontend.configurator import Configurator  # noqa: E402
from natal.frontend.hooks.entry.declarative import Op  # noqa: E402
from natal.frontend.hooks.types import (  # noqa: E402
    ECO_PARAM_NAMES,
    RPN_ADD,
    RPN_DIV,
    RPN_LITERAL,
    RPN_MUL,
    RPN_PARAM,
    RPN_SUB,
    CompiledHookPlan,
    HookOp,
)
from natal.frontend.population.age_structured import (  # noqa: E402
    AgeStructuredPopulation,  # noqa: E402
)
from natal.frontend.spatial.population import SpatialPopulation  # noqa: E402

try:
    from natal.backends.rust.rust_backend import rust_backend_available

    RUST_AVAILABLE = rust_backend_available()
except Exception:  # pragma: no cover - import guard for unbuilt extensions
    RUST_AVAILABLE = False

_SPECIES_COUNTER = 0


def _fresh_species() -> nt.Species:
    """Return a uniquely-named two-allele species (cache-safe)."""
    global _SPECIES_COUNTER
    _SPECIES_COUNTER += 1
    return nt.Species.from_dict(
        name=f"AddendumSpecies{_SPECIES_COUNTER}",
        structure={"chr1": {"loc": ["A", "a"]}},
        gamete_labels=["default"],
    )


def _build_age_structured(
    species: nt.Species,
    name: str,
    *,
    carrying_capacity: float = 800.0,
    sperm: Optional[dict] = None,
) -> AgeStructuredPopulation:
    """Build a deterministic age-structured population with sperm storage."""
    if sperm is None:
        sperm = {"A|A": {"A|A": 3, "A|a": 3}}
    return (
        Configurator.from_species(species)
        .age_structure(3, 1)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 20, "A|a": 10},
                "male": {"A|A": 15, "A|a": 5},
            },
            sperm_storage=sperm,
        )
        .competition(juvenile_growth_mode=1, carrying_capacity=carrying_capacity)
        .build()
    )


def _mirror_convert_row(
    female_src: float,
    buckets: List[float],
    prob: float,
) -> Tuple[float, List[float], float]:
    """Mirror the kernel's deterministic female-row convert arithmetic.

    Reproduces the exact float64 operation sequence of the CSR kernel's
    ``OP_CONVERT`` branch (pre-loop row sum, sequential bucket moves,
    clamped virgins, one combined move) so tests can assert bit-for-bit
    equality instead of approximations.
    """
    sperm_row_sum = 0.0
    for bucket in buckets:
        sperm_row_sum += bucket
    moved_mated = 0.0
    src_after: List[float] = []
    for bucket in buckets:
        moved = bucket * prob
        src_after.append(bucket - moved)
        moved_mated += moved
    virgins = female_src - sperm_row_sum
    if virgins < 0.0:
        virgins = 0.0
    moved_virgin = virgins * prob
    moved_total = moved_mated + moved_virgin
    return female_src - moved_total, src_after, moved_total


def _invoke_plan(
    plan: CompiledHookPlan,
    individual_count: np.ndarray,
    sperm_storage: Optional[np.ndarray],
    *,
    tick: int,
    stochastic: bool,
    eco_values: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Run one compiled plan through the CSR kernel on handcrafted arrays.

    Mirrors the flat-array call shape used by ``HookExecutor`` so tests
    can drive a single plan in isolation (no lifecycle, no write channel)
    — the right level for kernel-semantics locks like IEEE division.
    """
    from natal.frontend.hooks.runtime.csr_kernel import execute_csr_event_arrays

    if seed is not None:
        np.random.seed(seed)
    has_sperm = sperm_storage is not None and sperm_storage.size > 0
    execute_csr_event_arrays(
        n_events=np.int32(1),
        n_hooks=np.int32(1),
        hook_offsets=np.array([0, 1], dtype=np.int32),
        n_ops_list=np.array([plan.n_ops], dtype=np.int32),
        op_offsets=np.array([0, plan.n_ops], dtype=np.int32),
        op_types_data=plan.op_types,
        zidx_offsets_data=plan.zidx_offsets,
        zidx_data=plan.zidx_data,
        age_offsets_data=plan.age_offsets,
        age_data=plan.age_data,
        sex_masks_data=plan.sex_masks.ravel(),
        params_data=plan.params,
        condition_offsets_data=plan.condition_offsets,
        condition_types_data=plan.condition_types,
        condition_params_data=plan.condition_params,
        sp_param_ids_data=plan.sp_param_ids,
        sp_every_data=plan.sp_every,
        sp_start_data=plan.sp_start,
        rpn_offsets_data=plan.rpn_offsets,
        rpn_kinds_data=plan.rpn_kinds,
        rpn_payload_data=plan.rpn_payload,
        sp_literals_data=plan.sp_literals,
        convert_source_z_data=plan.convert_source_z,
        convert_target_z_data=plan.convert_target_z,
        deme_selector_types=np.array([0], dtype=np.int32),
        deme_selector_offsets=np.array([0, 0], dtype=np.int32),
        deme_selector_data=np.array([], dtype=np.int32),
        event_id=0,
        individual_count=individual_count,
        sperm_storage=sperm_storage,
        has_sperm_storage=has_sperm,
        tick=tick,
        stochastic=stochastic,
        continuous_sampling=False,
        deme_id=0,
        eco_values=eco_values,
    )
    return eco_values


def _compile_plan(pop: AgeStructuredPopulation, ops: List[HookOp]) -> CompiledHookPlan:
    """Compile ops into a plan without registering them on the population."""
    plan = nt.hooks.compile_declarative_hook(ops, pop, "early").plan
    assert plan is not None
    return plan


# ---------------------------------------------------------------------------
# Section 0 — wire contracts
# ---------------------------------------------------------------------------


def test_eco_param_names_and_rpn_constants_are_a_wire_contract() -> None:
    """ECO table order and RPN opcode values are frozen cross-backend contracts.

    The Python kernel and ``rust/src/hooks/interpreter.rs`` both index
    the ecology-value array by position in ``ECO_PARAM_NAMES`` and decode
    RPN tokens by integer kind, so any change here silently corrupts every
    backend at once.  Lock both tables.
    """
    assert ECO_PARAM_NAMES == (
        "carrying_capacity",
        "eggs_per_female",
        "sex_ratio",
        "sperm_displacement_rate",
        "low_density_growth_rate",
    )
    assert (RPN_LITERAL, RPN_PARAM, RPN_ADD, RPN_SUB, RPN_MUL, RPN_DIV) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


# ---------------------------------------------------------------------------
# Section A — RPN compile-time attacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    ["-K", "K * -1", "1 - -1", "- 1"],
)
def test_rpn_unary_minus_rejected(expr: str) -> None:
    """Unary minus has no grammar production — every spelling fails at compile.

    The shunting-yard treats ``-`` as strictly binary, so a leading or
    doubled minus produces an operator-first RPN program that the depth
    simulation rejects with ``ValueError`` before anything is registered.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, f"unary{abs(hash(expr)) % 10000}")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", expr)], event="early"
        )
    assert len(pop.compiled_hook_descriptors) == 0


@pytest.mark.parametrize(
    "expr",
    ["((K + 1", "K + 1)", ")K(", "(()", "K + ()", "(((K))"],
)
def test_rpn_parenthesis_errors_rejected(expr: str) -> None:
    """Unbalanced or empty parentheses fail at compile with no registration."""
    species = _fresh_species()
    pop = _build_age_structured(species, f"paren{abs(hash(expr)) % 10000}")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", expr)], event="early"
        )
    assert len(pop.compiled_hook_descriptors) == 0


@pytest.mark.parametrize(
    "expr",
    ["1 1", "K K", "K + +", "K * * 2", "0.5K", "K + * 2"],
)
def test_rpn_juxtaposition_and_operator_runs_rejected(expr: str) -> None:
    """Two operands or two operators in a row yield depth != 1 → ValueError."""
    species = _fresh_species()
    pop = _build_age_structured(species, f"juxt{abs(hash(expr)) % 10000}")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", expr)], event="early"
        )
    assert len(pop.compiled_hook_descriptors) == 0


@pytest.mark.parametrize("expr", ["", "   "])
def test_rpn_blank_expression_rejected(expr: str) -> None:
    """Empty (or whitespace-only) expressions never reach the runtime."""
    species = _fresh_species()
    pop = _build_age_structured(species, "blank")
    with pytest.raises(ValueError, match="empty"):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", expr)], event="early"
        )


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("viability * 2", "tensor"),
        ("female_age_based_survival + 1", "vector"),
        ("male_age0_survival * K", "vector"),
    ],
)
def test_rpn_tensor_and_vector_operands_rejected(expr: str, message: str) -> None:
    """Tensor/vector names are rejected as *operands*, with a targeted message.

    ``test_ops_setparam_convert.py`` covers them as targets only; an
    expression like ``"viability * 2"`` must fail the same way because the
    RPN stack machine has 0-d scalar slots only.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, f"operand{abs(hash(expr)) % 10000}")
    with pytest.raises(ValueError, match=message):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", expr)], event="early"
        )


def test_rpn_unknown_identifier_in_long_expression_rejected() -> None:
    """A 200-term expression with one bogus identifier fails as a whole.

    Depth-explosion and unknown-name handling compose: the lexer rejects
    the bogus identifier wherever it appears, and nothing is registered.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "longbad")
    long_expr = "K" + " + 1" * 199 + " + bogus"
    with pytest.raises(ValueError, match="bogus"):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", long_expr)], event="early"
        )
    assert len(pop.compiled_hook_descriptors) == 0


def test_rpn_long_operator_run_rejected() -> None:
    """A 200-operator run cannot produce a depth-1 program → ValueError."""
    species = _fresh_species()
    pop = _build_age_structured(species, "longops")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", "K" + " +" * 200)], event="early"
        )


@pytest.mark.parametrize("expr", ["K / 0", "0 / 0", "1 / 0", "K / (K - K)"])
def test_rpn_division_by_zero_literals_compile(expr: str) -> None:
    """Divide-by-zero *expressions* compile fine — IEEE handling is runtime.

    The compile-time depth simulation only validates program shape; the
    numeric result (inf/nan) is produced by the kernel's explicit IEEE-754
    branch and locked separately at the kernel level.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, f"div0c{abs(hash(expr)) % 10000}")
    pop.register_hooks([Op.set_param("carrying_capacity", expr)], event="early")
    assert len(pop.compiled_hook_descriptors) == 1


def test_failed_op_group_registration_is_atomic() -> None:
    """One bad op rejects the whole group — zero descriptors, zero writes."""
    species = _fresh_species()
    pop = _build_age_structured(species, "atomic")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [
                Op.set_param("carrying_capacity", "K * 0.5"),
                Op.set_param("carrying_capacity", "K + +"),
                Op.convert("A|A", "A|a", probability=0.25),
            ],
            event="early",
        )
    assert len(pop.compiled_hook_descriptors) == 0
    assert pop._run_program.hooks.n_hooks == 0  # noqa: SLF001 — contract lock
    # The population stays runnable and untouched.
    pop.run(1, record_every=0)
    assert pop.params.carrying_capacity == 800.0
    assert pop.params_log == ()


# ---------------------------------------------------------------------------
# Section B — RPN numeric matrix
# ---------------------------------------------------------------------------


_RPN_CASES: List[Tuple[str, Tuple[float, float], Callable[[float, float], float]]] = [
    ("(eggs_per_female + 2) / sex_ratio", (100.0, 0.5), lambda e, r: (e + 2) / r),
    ("(eggs_per_female + 2) / sex_ratio", (7.5, 0.25), lambda e, r: (e + 2) / r),
    ("K - eggs_per_female - 2", (100.0, 0.5), lambda e, r: 800.0 - e - 2),
    ("K - eggs_per_female - 2", (33.25, 0.5), lambda e, r: 800.0 - e - 2),
    (
        "sex_ratio * K + eggs_per_female / 10",
        (100.0, 0.5),
        lambda e, r: r * 800.0 + e / 10,
    ),
    (
        "sex_ratio * K + eggs_per_female / 10",
        (7.5, 0.25),
        lambda e, r: r * 800.0 + e / 10,
    ),
]


@pytest.mark.parametrize(("expr", "values", "mirror"), _RPN_CASES)
def test_rpn_expression_matrix_bitwise_vs_python(
    expr: str, values: Tuple[float, float], mirror: Callable[[float, float], float]
) -> None:
    """Compound expressions equal a same-order Python evaluation bit-for-bit.

    Each case installs concrete (eggs_per_female, sex_ratio) draft values,
    fires the op once via ``trigger_event`` (no lifecycle noise), and
    compares with ``==`` — not ``approx`` — because both sides run the
    identical float64 operation sequence.
    """
    eggs, ratio = values
    species = _fresh_species()
    pop = _build_age_structured(
        species, f"matrix{abs(hash(expr + str(values))) % 10000}"
    )
    pop.params.eggs_per_female = eggs
    pop.params.sex_ratio = ratio
    pop.register_hooks([Op.set_param("carrying_capacity", expr)], event="early")
    pop.trigger_event("early")
    assert pop.params.carrying_capacity == mirror(eggs, ratio)


def test_rpn_ieee_division_kernel_semantics() -> None:
    """Kernel-level division follows explicit IEEE-754 semantics.

    ``x/0`` = +inf for x>0, -inf for x<0, nan for x=0, and nan propagates
    through later multiplication.  Checked by invoking the kernel directly
    with a handcrafted eco scratch so the params write channel (bounds
    validation) cannot mask the raw float result.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "ieee")

    def eco_draft() -> np.ndarray:
        return np.array([800.0, 100.0, 0.5, 0.05, 6.0])

    plan = _compile_plan(pop, [Op.set_param("carrying_capacity", "K / 0")])
    eco = _invoke_plan(
        plan, np.zeros((2, 3, 3)), None, tick=0, stochastic=False,
        eco_values=eco_draft(),
    )
    assert eco is not None and math.isinf(eco[0]) and eco[0] > 0

    plan = _compile_plan(pop, [Op.set_param("carrying_capacity", "0 / 0")])
    eco = _invoke_plan(
        plan, np.zeros((2, 3, 3)), None, tick=0, stochastic=False,
        eco_values=eco_draft(),
    )
    assert eco is not None and math.isnan(eco[0])

    plan = _compile_plan(pop, [Op.set_param("carrying_capacity", "(0 - K) / 0")])
    eco = _invoke_plan(
        plan, np.zeros((2, 3, 3)), None, tick=0, stochastic=False,
        eco_values=eco_draft(),
    )
    assert eco is not None and math.isinf(eco[0]) and eco[0] < 0

    plan = _compile_plan(pop, [Op.set_param("carrying_capacity", "(K / 0) * 0")])
    eco = _invoke_plan(
        plan, np.zeros((2, 3, 3)), None, tick=0, stochastic=False,
        eco_values=eco_draft(),
    )
    assert eco is not None and math.isnan(eco[0])


def test_rpn_inf_result_crashes_at_write_channel_bounds() -> None:
    """Non-finite results pass the kernel but die at the flush, mid-run.

    Locked current behavior: the kernel computes inf (IEEE branch) and the
    flush routes it through ``pop.params`` bounds validation, which raises
    ``ValueError`` for ``carrying_capacity`` (bounds [0, 1e12]) *during the
    run*, not at compile time.  Reported as an implementation wart — the
    documented IEEE semantics are unreachable through the public write
    channel for bounded parameters.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "infcrash")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K / 0")], event="early"
    )
    with pytest.raises(ValueError, match="carrying_capacity"):
        pop.run(1, record_every=0)


def test_rpn_long_expressions_compile_and_evaluate_bitwise() -> None:
    """A 400-term sum and a 60-deep parenthesis nest evaluate bit-exactly.

    Long well-formed programs must compile (no recursion in shunting-yard)
    and produce the identical left-associative float64 result as a Python
    loop performing the same operations in the same order.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "longok")
    expr = "K" + " + 1" * 400
    pop.register_hooks([Op.set_param("eggs_per_female", expr)], event="early")
    pop.trigger_event("early")
    expected = 800.0
    for _ in range(400):
        expected += 1.0
    assert pop.params.eggs_per_female == expected

    species = _fresh_species()
    pop = _build_age_structured(species, "deepparen")
    expr = "(" * 60 + "K * 0.5" + ")" * 60
    pop.register_hooks(
        [Op.set_param("eggs_per_female", expr)], event="early"
    )
    pop.trigger_event("early")
    assert pop.params.eggs_per_female == 800.0 * 0.5


def test_rpn_chained_self_reference_every_tick_bitwise() -> None:
    """``"K * 0.9"`` every=1 compounds identically to ``k *= 0.9`` per fire.

    Mirrors with sequential multiplication (not ``0.9 ** n``): one rounded
    power and n rounded multiplies differ in the last ulp, and the spec is
    the per-fire evaluation order.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "compound")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.9", every=1)], event="early"
    )
    k_manual = 800.0
    for tick in range(12):
        pop.run(1, record_every=0)
        k_manual = k_manual * 0.9
        assert pop.params.carrying_capacity == k_manual, f"tick {tick}"


def test_rpn_cross_param_chain_within_one_tick() -> None:
    """Op 2 sees op 1's write within the same tick (shared eco scratch).

    The second expression deliberately contains no numeric literals: it
    writes a pure function of the two live param operands, isolating the
    chained-operand invariant from literal-pool indexing (which the
    regression lock in Section X covers separately).
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "crossparam")
    pop.register_hooks(
        [
            Op.set_param("eggs_per_female", "eggs_per_female * 2.0"),
            Op.set_param("carrying_capacity", "eggs_per_female * sex_ratio"),
        ],
        event="early",
    )
    pop.trigger_event("early")
    # eggs doubled first (100 -> 200), then K = 200 * 0.5 in the same tick.
    assert pop.params.eggs_per_female == 200.0
    assert pop.params.carrying_capacity == 200.0 * 0.5


def test_set_param_alias_target_and_operand() -> None:
    """Aliases (``K``, ``expected_eggs_per_female``) resolve on both sides."""
    species = _fresh_species()
    pop = _build_age_structured(species, "alias1")
    pop.register_hooks(
        [Op.set_param("K", "K * 2.0")], event="early"
    )
    pop.trigger_event("early")
    assert pop.params.carrying_capacity == 1600.0

    species = _fresh_species()
    pop = _build_age_structured(species, "alias2")
    pop.register_hooks(
        [Op.set_param("eggs_per_female", "expected_eggs_per_female + 5.0")],
        event="early",
    )
    pop.trigger_event("early")
    assert pop.params.eggs_per_female == 105.0


# ---------------------------------------------------------------------------
# Section C — scheduling semantics
# ---------------------------------------------------------------------------


def test_schedule_start3_every5_fires_exactly_at_3_8_13() -> None:
    """``tick >= start and (tick - start) % every == 0`` → fires 3, 8, 13."""
    species = _fresh_species()
    pop = _build_age_structured(species, "sched35813")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5", every=5, start=3)],
        event="early",
    )
    pop.run(15, record_every=0)

    k = 800.0
    expected_rows: List[Tuple[int, str, float, float]] = []
    for fire_tick in (3, 8, 13):
        new_k = k * 0.5
        expected_rows.append((fire_tick, "carrying_capacity", k, new_k))
        k = new_k
    assert pop.params_log == tuple(expected_rows)
    assert pop.params.carrying_capacity == k


def test_schedule_when_clause_ands_with_schedule() -> None:
    """The ``when`` RPN gates firing on top of the every/start schedule."""
    species = _fresh_species()
    pop = _build_age_structured(species, "whenand")
    pop.register_hooks(
        [
            Op.set_param(
                "carrying_capacity", "K * 0.5", every=1, when="tick < 3"
            )
        ],
        event="early",
    )
    k_manual = 800.0
    for tick in range(6):
        pop.run(1, record_every=0)
        if tick < 3:  # schedule fires while the when clause holds
            k_manual = k_manual * 0.5
        assert pop.params.carrying_capacity == k_manual, f"tick {tick}"
    assert tuple(row[0] for row in pop.params_log) == (0, 1, 2)


def test_same_plan_two_set_param_ops_last_declaration_wins() -> None:
    """Within one plan, later declarations overwrite earlier ones.

    The follower expression is literal-free on purpose: it writes the
    *current* eggs value into K, proving op 2 both ran after op 1 and won
    the write (the shared-literal-pool regression lock in Section X covers
    literal-bearing follower expressions).
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "lastwins")
    pop.register_hooks(
        [
            Op.set_param("carrying_capacity", 111.0),
            Op.set_param("carrying_capacity", "eggs_per_female"),
        ],
        event="early",
    )
    pop.trigger_event("early")
    assert pop.params.carrying_capacity == 100.0  # eggs default, not 111.0
    # Both fired ops flush the same final value: exactly one committed row.
    assert pop.params_log == ((0, "carrying_capacity", 800.0, 100.0),)


def test_set_param_priority_chain_across_descriptors_sees_earlier_write() -> None:
    """A later-priority descriptor evaluates against the earlier write.

    Both orderings are locked: with the constant writer at priority 0 the
    multiplier sees 999 and produces 499.5; with the multiplier first the
    constant simply overwrites.  This pins cross-descriptor chaining (the
    eco scratch stays live across descriptors of one event) and the
    final event commit. Intermediate scratch writes are coalesced in the log.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "prioconstfirst")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", 999.0)], event="early", priority=0, name="w"
    )
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5")],
        event="early",
        priority=1,
        name="m",
    )
    pop.trigger_event("early")
    assert pop.params_log == (
        (0, "carrying_capacity", 800.0, 999.0 * 0.5),
    )
    assert pop.params.carrying_capacity == 999.0 * 0.5

    species = _fresh_species()
    pop = _build_age_structured(species, "priomultfirst")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", 999.0)], event="early", priority=1, name="w"
    )
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5")],
        event="early",
        priority=0,
        name="m",
    )
    pop.trigger_event("early")
    assert pop.params_log == (
        (0, "carrying_capacity", 800.0, 999.0),
    )
    assert pop.params.carrying_capacity == 999.0


def test_set_param_first_event_boundary_schedule() -> None:
    """``event='first'`` ops fire on the first boundary with the schedule.

    every=2 over six ticks must fire at ticks 0, 2, 4 — proving the
    schedule check is event-boundary agnostic, not early-specific.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "firstevent")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5", every=2, event="first")],
        event="first",
    )
    pop.run(6, record_every=0)
    k = 800.0
    expected_rows: List[Tuple[int, str, float, float]] = []
    for fire_tick in (0, 2, 4):
        new_k = k * 0.5
        expected_rows.append((fire_tick, "carrying_capacity", k, new_k))
        k = new_k
    assert pop.params_log == tuple(expected_rows)


# ---------------------------------------------------------------------------
# Section D — convert numerical invariants (focus)
# ---------------------------------------------------------------------------


def _overwrite_adult_rows(
    pop: AgeStructuredPopulation,
    *,
    female_aa: float,
    male_aa: float,
    zero_female_aax: bool,
) -> None:
    """Place explicit counts in the two adult age rows of the A|A column.

    ``initial_state`` spreads the adult dict over ages 1..n-1; tests that
    need exact per-row bookkeeping overwrite the A|A column so every adult
    row carries the same known value (and, optionally, zero the A|a
    destination column).
    """
    state = pop.state
    ind = state.individual_count
    ind[:, :, 0] = 0.0
    ind[0, 1:, 0] = female_aa
    ind[1, 1:, 0] = male_aa
    if zero_female_aax:
        ind[:, :, 1] = 0.0
    pop.import_state(state)


def test_convert_nontrivial_three_bucket_matrix_deterministic_exact() -> None:
    """Per-bucket atomic migration on a 3-male-z sperm matrix, bit-exact.

    Each adult female A|A row carries buckets (12, 8, 4) over male labels
    (A|A, A|a, a|a) plus 16 virgins; deterministic conversion at p=0.25
    must move exactly ``bucket * p`` per bucket (independent draws), keep
    the female row sum and the male sperm axis bit-identical, and leave
    males as a plain 25% migration.  Expected values come from the
    kernel-order mirror and are asserted with ``==``.
    """
    species = _fresh_species()
    pop = _build_age_structured(
        species,
        "threebucket",
        sperm={"A|A": {"A|A": 12.0, "A|a": 8.0, "a|a": 4.0}},
    )
    # 40 females per adult row: virgins = 40 - 24 = 16 per row.
    _overwrite_adult_rows(pop, female_aa=40.0, male_aa=20.0, zero_female_aax=False)

    ind_before = pop.state.individual_count.copy()
    sperm_before = pop.state.sperm_storage.copy()
    pop.register_hooks(
        [Op.convert("A|A", "A|a", probability=0.25)], event="early"
    )
    pop.trigger_event("early")
    ind = pop.state.individual_count
    sperm = pop.state.sperm_storage

    p = 0.25
    for age in range(1, 3):
        buckets = [12.0, 8.0, 4.0]
        f_src_after, src_buckets, moved_total = _mirror_convert_row(40.0, buckets, p)
        # Female source row: 40 -> 30 exactly; destination gains the same.
        assert ind[0, age, 0] == f_src_after, f"age {age}"
        assert ind[0, age, 1] == float(ind_before[0, age, 1]) + moved_total
        # Per-bucket: each bucket moved exactly bucket * p, independently.
        for mz, bucket in enumerate(buckets):
            assert sperm[age, 0, mz] == src_buckets[mz]
            assert sperm[age, 1, mz] == float(sperm_before[age, 1, mz]) + bucket * p
        # Female row sum conserved bit-exactly.
        assert ind[0, age, :].sum() == float(ind_before[0, age, :].sum())
        # Male row is a plain 25% migration.
        assert ind[1, age, 0] == 20.0 - 20.0 * p
        assert ind[1, age, 1] == float(ind_before[1, age, 1]) + 20.0 * p
    # Male sperm axis frozen: per-male-label column totals bit-identical.
    np.testing.assert_array_equal(
        sperm.sum(axis=(0, 1)), sperm_before.sum(axis=(0, 1))
    )
    # Grand totals conserved.
    assert ind.sum() + sperm.sum() == ind_before.sum() + sperm_before.sum()


def test_convert_stochastic_bucket_means_within_3sigma_and_exact_conservation() -> None:
    """200 seeded stochastic runs: bucket means within 3-sigma, rows exact.

    Per-trial invariants (exact, not statistical): the female and male row
    sums are conserved bit-exactly (integer-valued binomial moves), and
    the male sperm-label column totals never move.  Statistical: each
    destination bucket's mean over 200 trials sits within 3 standard
    errors of ``n * p``, which a biased or mis-parameterized kernel would
    violate (sum of independent binomials for the combined female total).
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "stochbuckets")
    plan = _compile_plan(pop, [Op.convert("A|A", "A|a", probability=0.25)])

    p = 0.25
    buckets = [12.0, 8.0, 4.0]
    n_female, n_male = 40.0, 20.0
    n_trials = 200
    dst_bucket_sums = [0.0, 0.0, 0.0]
    dst_female_sum = 0.0
    dst_male_sum = 0.0
    for trial in range(n_trials):
        ind = np.zeros((2, 1, 3), dtype=np.float64)
        ind[0, 0, 0] = n_female
        ind[1, 0, 0] = n_male
        sperm = np.zeros((1, 3, 3), dtype=np.float64)
        sperm[0, 0, :] = buckets
        _invoke_plan(plan, ind, sperm, tick=0, stochastic=True, seed=trial)
        # Exact per-trial conservation.
        assert ind[0, 0, :].sum() == n_female
        assert ind[1, 0, :].sum() == n_male
        np.testing.assert_array_equal(sperm.sum(axis=(0, 1)), buckets)
        for mz in range(3):
            dst_bucket_sums[mz] += float(sperm[0, 1, mz])
        dst_female_sum += float(ind[0, 0, 1])
        dst_male_sum += float(ind[1, 0, 1])

    for mz, n_base in enumerate(buckets):
        mean = dst_bucket_sums[mz] / n_trials
        sigma3 = 3.0 * math.sqrt(n_base * p * (1.0 - p) / n_trials)
        assert abs(mean - n_base * p) <= sigma3, f"bucket {mz}: {mean}"
    # Combined female destination (buckets + virgins, one binomial(40, p)).
    female_mean = dst_female_sum / n_trials
    female_sigma3 = 3.0 * math.sqrt(n_female * p * (1.0 - p) / n_trials)
    assert abs(female_mean - n_female * p) <= female_sigma3
    male_mean = dst_male_sum / n_trials
    male_sigma3 = 3.0 * math.sqrt(n_male * p * (1.0 - p) / n_trials)
    assert abs(male_mean - n_male * p) <= male_sigma3
    # Sanity floor for the statistics themselves: variance must be present.
    assert dst_male_sum > 0.0


def test_convert_three_way_split_chain_exact() -> None:
    """0.3 then 0.5-of-the-remainder splits A|A three ways, exactly.

    Expected values come from applying the kernel-order mirror twice
    (chained), so every float operation matches the kernel's sequence.
    """
    species = _fresh_species()
    pop = _build_age_structured(
        species,
        "threeway",
        sperm={"A|A": {"A|A": 12.0, "A|a": 8.0, "a|a": 4.0}},
    )
    _overwrite_adult_rows(pop, female_aa=40.0, male_aa=20.0, zero_female_aax=False)

    pop.register_hooks(
        [
            Op.convert("A|A", "A|a", probability=0.3),
            Op.convert("A|A", "a|a", probability=0.5),
        ],
        event="early",
    )
    pop.trigger_event("early")
    ind = pop.state.individual_count
    sperm = pop.state.sperm_storage

    buckets = [12.0, 8.0, 4.0]
    # First draw: p1 = 0.3 out of A|A; second: p2 = 0.5 of the remainder.
    f_after1, buckets_after1, moved1 = _mirror_convert_row(40.0, buckets, 0.3)
    f_after2, buckets_after2, moved2 = _mirror_convert_row(
        f_after1, buckets_after1, 0.5
    )
    for age in range(1, 3):
        assert ind[0, age, 0] == f_after2
        assert ind[0, age, 1] == 10.0 + moved1
        assert ind[0, age, 2] == moved2
        for mz, bucket in enumerate(buckets):
            # Destination buckets accumulate the raw moved amounts
            # (``dst += bucket * p``), not the source-side differences.
            moved_b1 = bucket * 0.3
            moved_b2 = buckets_after1[mz] * 0.5
            assert sperm[age, 1, mz] == moved_b1
            assert sperm[age, 2, mz] == moved_b2
    # Male chain per adult row: 20 -> 14 after 0.3, then 7 after 0.5.
    m1 = 20.0 * 0.3
    m2 = (20.0 - m1) * 0.5
    for age in range(1, 3):
        assert ind[1, age, 0] == 20.0 - m1 - m2
        assert ind[1, age, 1] == 5.0 + m1
        assert ind[1, age, 2] == m2
    # Grand total conserved: 2 adult rows.
    total = ind.sum() + sperm.sum()
    assert total == 2.0 * ((40.0 + 10.0 + 20.0 + 5.0) + 24.0)


def test_convert_relay_chain_declaration_order_visible() -> None:
    """A→B then B→C is distinguishable from the reversed registration.

    Both orders conserve totals; only the final distribution differs,
    which proves convert ops execute (and chain) in declaration order.
    """
    species = _fresh_species()
    pop = _build_age_structured(
        species,
        "relay",
        sperm={"A|A": {"A|A": 6.0, "A|a": 4.0, "a|a": 2.0}},
    )
    _overwrite_adult_rows(pop, female_aa=40.0, male_aa=20.0, zero_female_aax=True)

    pop.register_hooks(
        [
            Op.convert("A|A", "A|a", probability=0.5),
            Op.convert("A|a", "a|a", probability=0.5),
        ],
        event="early",
    )
    pop.trigger_event("early")
    ind = pop.state.individual_count
    sperm = pop.state.sperm_storage
    # Females per adult row: A|A -> 20, A|a -> 10 (half of the arrivals), a|a -> 10.
    for age in range(1, 3):
        assert ind[0, age, 0] == 20.0
        assert ind[0, age, 1] == 10.0
        assert ind[0, age, 2] == 10.0
        # Male: 20 -> 10 -> 5 split across A|a / a|a.
        assert ind[1, age, 0] == 10.0
        assert ind[1, age, 1] == 5.0
        assert ind[1, age, 2] == 5.0
        # Buckets relay: A|a row keeps half the arrivals, a|a gets the rest.
        for mz, bucket in enumerate((6.0, 4.0, 2.0)):
            arrived = bucket * 0.5
            assert sperm[age, 1, mz] == arrived * 0.5
            assert sperm[age, 2, mz] == arrived * 0.5
    assert ind.sum() + sperm.sum() == 2.0 * (40.0 + 20.0 + 12.0)

    # Reversed registration: the A|a->a|a op runs on an empty row first.
    species = _fresh_species()
    pop2 = _build_age_structured(
        species,
        "relayrev",
        sperm={"A|A": {"A|A": 6.0, "A|a": 4.0, "a|a": 2.0}},
    )
    _overwrite_adult_rows(pop2, female_aa=40.0, male_aa=20.0, zero_female_aax=True)
    pop2.register_hooks(
        [
            Op.convert("A|a", "a|a", probability=0.5),
            Op.convert("A|A", "A|a", probability=0.5),
        ],
        event="early",
    )
    pop2.trigger_event("early")
    ind2 = pop2.state.individual_count
    for age in range(1, 3):
        assert ind2[0, age, 0] == 20.0
        assert ind2[0, age, 1] == 20.0
        assert ind2[0, age, 2] == 0.0
        assert ind2[1, age, 0] == 10.0
        assert ind2[1, age, 1] == 10.0
        assert ind2[1, age, 2] == 0.0


def test_convert_probability_boundaries_zero_and_one() -> None:
    """p=0 leaves the state bit-identical; p=1 empties the source row."""
    species = _fresh_species()
    pop = _build_age_structured(species, "pzero")
    before = pop.state.individual_count.copy()
    sperm_before = pop.state.sperm_storage.copy()
    pop.register_hooks(
        [Op.convert("A|A", "A|a", probability=0.0)], event="early"
    )
    pop.trigger_event("early")
    np.testing.assert_array_equal(pop.state.individual_count, before)
    np.testing.assert_array_equal(pop.state.sperm_storage, sperm_before)

    species = _fresh_species()
    pop = _build_age_structured(species, "pone")
    pop.register_hooks(
        [Op.convert("A|A", "A|a", probability=1.0)], event="early"
    )
    pop.trigger_event("early")
    ind = pop.state.individual_count
    sperm = pop.state.sperm_storage
    # Source ztype fully drained (every age row, both sexes, every bucket).
    assert float(ind[:, :, 0].sum()) == 0.0
    assert float(sperm[:, 0, :].sum()) == 0.0
    # Destination received exactly the drained totals on top of its own
    # initial content.
    assert float(ind[0, :, 1].sum()) == float(
        before[0, :, 1].sum() + before[0, :, 0].sum()
    )
    assert float(ind[1, :, 1].sum()) == float(
        before[1, :, 1].sum() + before[1, :, 0].sum()
    )
    assert float(sperm[:, 1, :].sum()) == float(
        sperm_before[:, 1, :].sum() + sperm_before[:, 0, :].sum()
    )

    # NaN probability is rejected by the factory (out-of-range check).
    with pytest.raises(ValueError, match="probability"):
        Op.convert("A|A", "A|a", probability=float("nan"))


# ---------------------------------------------------------------------------
# Section E — compile-time validation addenda
# ---------------------------------------------------------------------------


def test_convert_zero_match_target_raises() -> None:
    """A target pattern matching no ZType fails at compile time."""
    species = _fresh_species()
    pop = _build_age_structured(species, "zero_dst")
    with pytest.raises(ValueError, match="target pattern"):
        pop.register_hooks(
            [Op.convert("A|A", "Z|Z", probability=0.5)], event="early"
        )


def test_convert_group_registration_atomic_on_bad_op() -> None:
    """A bad convert in a group rejects the whole registration."""
    species = _fresh_species()
    pop = _build_age_structured(species, "atomic_conv")
    with pytest.raises(ValueError):
        pop.register_hooks(
            [
                Op.convert("A|A", "A|a", probability=0.25),
                Op.convert("Z|Z", "A|a", probability=0.25),
            ],
            event="early",
        )
    assert len(pop.compiled_hook_descriptors) == 0


# ---------------------------------------------------------------------------
# Section F — schedule-state persistence across run boundaries
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_start1_every2_when_schedule_persists_across_run_calls() -> None:
    """start=1, every=2, ``when`` schedules survive run-call boundaries.

    Every tick runs through a separate ``run()`` call, so schedule state
    (tick counter, session column, draft value) must persist across
    run-call boundaries.  The engine session column is compared against
    the population draft and the hand-compounded ``K * 0.9`` chain after
    every tick.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "addparity_rust")
    pop.register_hooks(
        [
            Op.set_param(
                "carrying_capacity",
                "K * 0.9",
                every=2,
                start=1,
                when="tick >= 1",
            ),
            Op.convert("A|A", "A|a", probability=0.25),
            Op.kill(genotypes="A|a", prob=0.1),
        ],
        event="early",
        name="addparity_program",
    )
    pop._initialize_session(seed=11)
    session = pop._rust_lifecycle_backend._session  # noqa: SLF001 — bridge

    k_manual = 800.0
    for tick in range(8):
        pop.run(1, record_every=0)
        if tick >= 1 and (tick - 1) % 2 == 0:
            k_manual = k_manual * 0.9
        rust_k = float(session.get_scalar("carrying_capacity"))
        assert rust_k == pop.params.carrying_capacity, f"tick {tick}"
        assert rust_k == k_manual, f"tick {tick}"


# ---------------------------------------------------------------------------
# Section G — write-channel integrity
# ---------------------------------------------------------------------------


def test_params_log_exact_rows_every2_chained_values() -> None:
    """``every=2`` produces rows exactly on fire ticks with chained values."""
    species = _fresh_species()
    pop = _build_age_structured(species, "logevery2")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.9", every=2)], event="early"
    )
    pop.run(8, record_every=0)
    k = 800.0
    expected_rows: List[Tuple[int, str, float, float]] = []
    for fire_tick in (0, 2, 4, 6):
        new_k = k * 0.9
        expected_rows.append((fire_tick, "carrying_capacity", k, new_k))
        k = new_k
    assert pop.params_log == tuple(expected_rows)
    # No rows on non-fire ticks: 8 ticks, 4 fires, 4 rows.
    assert len(pop.params_log) == 4


def test_params_log_snapshot_is_an_owned_copy() -> None:
    """``params_log`` returns a frozen snapshot, not a live view.

    Ownership contract: callers cannot observe or mutate future writes
    through a previously returned tuple.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "logown")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5", every=1)], event="early"
    )
    pop.run(1, record_every=0)
    snapshot = pop.params_log
    assert snapshot == ((0, "carrying_capacity", 800.0, 400.0),)
    pop.run(1, record_every=0)
    # The earlier snapshot is unchanged; the fresh read grew.
    assert snapshot == ((0, "carrying_capacity", 800.0, 400.0),)
    assert len(pop.params_log) == 2
    assert pop.params_log is not pop.params_log


def test_schedule_persists_across_run_call_boundaries() -> None:
    """A schedule keeps firing across separate ``run()`` invocations.

    State-transition coverage for the write channel: run(3) + run(3) +
    run(2) must produce fires at ticks 0, 2, 4, 6 with one continuous
    value chain — no reset at run boundaries.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "runbounds")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5", every=2)], event="early"
    )
    pop.run(3, record_every=0)
    pop.run(3, record_every=0)
    pop.run(2, record_every=0)
    assert tuple(row[0] for row in pop.params_log) == (0, 2, 4, 6)
    k = 800.0
    for _ in range(4):
        k = k * 0.5
    assert pop.params.carrying_capacity == k


# ---------------------------------------------------------------------------
# Section H — spatial models
# ---------------------------------------------------------------------------


def test_spatial_set_param_selector_writes_only_selected_deme_columns() -> None:
    """A spatial-level deme selector writes only the selected demes.

    demes 0 and 2 receive the schedule (value + one log row each); deme 1
    keeps its draft value and an empty params log.
    """
    species = _fresh_species()

    def build_deme(name: str) -> AgeStructuredPopulation:
        return _build_age_structured(species, name, carrying_capacity=500.0)

    demes = [build_deme(f"addsp_d{d}") for d in range(3)]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.register_hooks(
        [Op.set_param("carrying_capacity", 321.0, every=1)],
        event="early",
        deme=[0, 2],
    )
    spatial.run(1, record_every=0)
    assert demes[0].params.carrying_capacity == 321.0
    assert demes[2].params.carrying_capacity == 321.0
    assert demes[1].params.carrying_capacity == 500.0
    assert demes[0].params_log == ((0, "carrying_capacity", 500.0, 321.0),)
    assert demes[2].params_log == ((0, "carrying_capacity", 500.0, 321.0),)
    assert demes[1].params_log == ()


def test_spatial_convert_applies_per_deme_independently() -> None:
    """Convert runs inside each deme on the deme's own state.

    With migration_rate=0 and per-deme early triggers, every deme shows
    the exact same deterministic per-deme migration (p of each sex row
    plus every sperm bucket, mirrored in kernel order), and per-deme
    totals are conserved.
    """
    species = _fresh_species()

    def build_deme(name: str) -> AgeStructuredPopulation:
        return _build_age_structured(species, name)

    demes = [build_deme(f"addconv_d{d}") for d in range(3)]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.register_hooks(
        [Op.convert("A|A", "A|a", probability=0.25)], event="early"
    )

    p = 0.25
    for deme_id, deme in enumerate(demes):
        before_ind = deme.state.individual_count.copy()
        before_sperm = deme.state.sperm_storage.copy()
        result = spatial.trigger_event("early", deme_id=deme_id)
        assert result == 0
        ind = deme.state.individual_count
        sperm = deme.state.sperm_storage
        for age in range(ind.shape[1]):
            female = float(before_ind[0, age, 0])
            buckets = [float(before_sperm[age, 0, mz]) for mz in range(3)]
            if female == 0.0 and all(bucket == 0.0 for bucket in buckets):
                continue
            f_after, buckets_after, moved = _mirror_convert_row(
                female, buckets, p
            )
            assert ind[0, age, 0] == f_after, f"deme {deme_id} age {age}"
            assert ind[0, age, 1] == float(before_ind[0, age, 1]) + moved
            for mz in range(3):
                assert sperm[age, 0, mz] == buckets_after[mz]
                assert sperm[age, 1, mz] == float(before_sperm[age, 1, mz]) + (
                    buckets[mz] * p
                )
            male = float(before_ind[1, age, 0])
            assert ind[1, age, 0] == male - male * p
            assert ind[1, age, 1] == float(before_ind[1, age, 1]) + male * p
        # Per-deme conservation.
        assert ind.sum() + sperm.sum() == before_ind.sum() + before_sperm.sum()


# ---------------------------------------------------------------------------
# Section X — shared-literal-pool regression lock
# ---------------------------------------------------------------------------


def test_second_set_param_literal_reads_its_own_literal() -> None:
    """Each set_param expression must resolve its own literal pool slots.

    Regression guard for the fixed shared-literal-pool defect: literal
    payloads are rebased onto the plan-wide pool at compile time, so
    eggs gets 6.0 from the first op and carrying_capacity reads *its own*
    777.0 — never the first expression's 6.0.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "poolbug")
    pop.register_hooks(
        [
            Op.set_param("eggs_per_female", 6.0),
            Op.set_param("carrying_capacity", 777.0),
        ],
        event="early",
    )
    pop.trigger_event("early")
    assert pop.params.eggs_per_female == 6.0
    assert pop.params.carrying_capacity == 777.0


# ═══════════════════════════════════════════════════════════════════════════
# Hard-blocker regression guards (evaluator round: rust paths + error paths)
# ═══════════════════════════════════════════════════════════════════════════


def _hb_species(name: str):
    import natal as nt

    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _hb_spatial_pop(name: str, event: str):
    """Viable 2-deme age-structured spatial model with a set_param hook."""
    import natal as nt

    sp = _hb_species(name)
    return (
        nt.SpatialPopulation.builder(sp, n_demes=2)
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"A|A": 300}, "male": {"A|A": 300}}
        )
        .survival(female_age_based_survival=[0.9, 0.95], male_age_based_survival=[0.9, 0.95])
        .reproduction(
            eggs_per_female=50,
            sex_ratio=0.5,
            female_age_based_mating_rate=[0.0, 0.9],
            male_age_based_mating_rate=[0.0, 0.9],
            age_based_reproduction_rate=[0.0, 1.0],
            female_age_based_fertility=[0.0, 1.0],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=800.0,
            low_density_growth_rate=2.0,
        )
        .migration(migration_rate=0.05)
        .hooks(
            nt.Op.set_param(
                "carrying_capacity",
                "carrying_capacity * 0.5",
                event=event,
            )
        )
        .build()
    )


class TestHb1SpatialEventLevelSemantics:
    """HB-1: spatial set_param must be event-level.

    Before the fix the spatial kernels committed set_param writes only
    at tick granularity, so a first-event write could not influence the
    same tick's reproduction/survival (tick 0 diverged silently).
    """

    @pytest.mark.parametrize("event", ["first", "early", "late"])
    def test_spatial_set_param_semantics(self, event: str) -> None:
        """Event-level commits compound per deme within each tick.

        Semantics assertions: the K chain halves once per tick per deme,
        every write lands in the audit journal at its own tick, and the
        dynamics stay alive.
        """
        try:
            from natal.backends.rust.rust_backend import rust_backend_available
        except ImportError:
            pytest.skip("rust backend module unavailable")
        if not rust_backend_available():
            pytest.skip("rust extension not built")

        pop = _hb_spatial_pop(f"hb1_sem_{event}", event)
        pop.run(3, record_every=0)
        total = float(np.asarray(pop.deme(0).state.individual_count).sum())

        # K chain: exactly one halving per tick, on every deme draft.
        np.testing.assert_array_equal(
            np.asarray(pop.params.carrying_capacity), [100.0, 100.0]
        )
        # Audit: one change row per fired tick per deme; rows carry the
        # commit tick.
        rows = [r for r in pop.demes[0].params_log if r[1] == "carrying_capacity"]
        assert [(r[0], r[3]) for r in rows] == [(0, 400.0), (1, 200.0), (2, 100.0)]
        assert total > 0.0


class TestHb2RustRunChannelMerge:
    """HB-2: after a rust run, draft/params_log must reflect session writes."""

    def _panmictic(self, name: str):
        import natal as nt

        sp = _hb_species(name)
        return nt.DiscreteGenerationPopulation.setup(sp, stochastic=False).initial_state(
            individual_count={
                "female": {"A|A": [0.0, 400.0]},
                "male": {"A|A": [0.0, 400.0]},
            }
        )

    def test_rust_run_merges_journal_into_params_log_and_draft(self) -> None:
        try:
            from natal.backends.rust.rust_backend import rust_backend_available
        except ImportError:
            pytest.skip("rust backend module unavailable")
        if not rust_backend_available():
            pytest.skip("rust extension not built")
        import natal as nt

        op = nt.Op.set_param("carrying_capacity", "carrying_capacity * 0.5")

        pop = (
            self._panmictic("hb2_run")
            .competition(
                juvenile_growth_mode="beverton_holt",
                carrying_capacity=800.0,
                low_density_growth_rate=2.0,
            )
            .reproduction(eggs_per_female=10.0)
            .hooks(op, event="early")
            .build()
        )
        pop.run(3, record_every=0)

        # The draft reflects the final session value: 800 halved per tick.
        assert pop.params.carrying_capacity == 100.0
        # The audit log has one change row per fired tick with the
        # hand-compounded chain 800 -> 400 -> 200 -> 100.
        rows = [r for r in pop.params_log if r[1] == "carrying_capacity"]
        assert rows == [
            (0, "carrying_capacity", 800.0, 400.0),
            (1, "carrying_capacity", 400.0, 200.0),
            (2, "carrying_capacity", 200.0, 100.0),
        ]

    def test_rust_run_inf_expression_raises_with_param_name(self) -> None:
        try:
            from natal.backends.rust.rust_backend import rust_backend_available
        except ImportError:
            pytest.skip("rust backend module unavailable")
        if not rust_backend_available():
            pytest.skip("rust extension not built")
        import natal as nt

        pop = (
            self._panmictic("hb2_inf")
            .competition(
                juvenile_growth_mode="beverton_holt",
                carrying_capacity=800.0,
                low_density_growth_rate=2.0,
            )
            .reproduction(eggs_per_female=10.0)
            .hooks(
                nt.Op.set_param("carrying_capacity", "carrying_capacity / 0"),
                event="first",
            )
            .build()
        )
        pop._initialize_session(seed=0)
        with pytest.raises((ValueError, RuntimeError), match="carrying_capacity"):
            pop.run(2, record_every=0)


class TestHb3ValueExpressionTypeError:
    """HB-3: non-scalar value expressions raise TypeError, not ValueError."""

    @pytest.mark.parametrize("bad", [True, False, None])
    def test_bool_and_none_values_raise_type_error(self, bad: object) -> None:
        """The bad-kind value surfaces as TypeError at compile (build) time."""

        sp = _hb_species("hb3_sp")
        with pytest.raises(TypeError, match="string expression or a number"):
            _hb_minimal_age_build(sp, bad)

    def test_illegal_character_raises_value_error(self) -> None:
        """An illegal character in the expression is a value error."""

        sp = _hb_species("hb3_sp2")
        with pytest.raises(ValueError):
            _hb_minimal_age_build(sp, "K @ 2")


def _hb_minimal_age_build(sp: object, bad_value: object) -> None:
    """Build a minimal age-structured population carrying one set_param op."""
    import natal as nt

    (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)  # pyright: ignore[reportArgumentType]  # test helper takes Species
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"A|A": [10, 100]}, "male": {"A|A": [10, 100]}}
        )
        .competition(carrying_capacity=500.0, juvenile_growth_mode="beverton_holt")
        .hooks(
            nt.Op.set_param(
                "carrying_capacity", bad_value, event="early"
            )  # type: ignore[arg-type]  # negative contract: deliberately wrong value kind
        )
        .build()
    )
