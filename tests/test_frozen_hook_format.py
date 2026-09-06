"""Frozen user-surface contract samples: the declarative hook format.

RUST_ONLY_REFACTOR_PLAN.md section 2.1 freezes the declarative hook
format: ``Op.*`` actions, ``.hooks(...)`` registration, selectors,
condition expressions, events, priority, scheduling (every/start), and
deme selection.  These tests are executable samples of that syntax with
numerical anchors on deterministic dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import natal as nt


def _species() -> nt.Species:
    """Return the shared two-allele species for hook samples."""
    return nt.Species.from_dict(
        name="FrozenHookSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _builder(
    pop_name: str,
    *hook_items: nt.HookOp | Callable[..., Any],  # Any: arbitrary hook callable per the frozen .hooks() contract
    **hook_kwargs: str,
) -> nt.Configurator:
    """Return a minimal discrete builder chain with optional hooks.

    Args:
        pop_name: Population name.
        hook_items: ``Op`` objects, ``@nt.hook``-decorated functions, or
            plain single-parameter callbacks.
        hook_kwargs: Forwarded registration keywords (``event``,
            ``name``, ...).

    Returns:
        The builder chain.
    """
    chain = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(), name=pop_name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )
    if hook_items:
        chain = chain.hooks(*hook_items, **hook_kwargs)
    return chain


# ══════════════════════════════════════════════════════════════════════════════
# Op action surface
# ══════════════════════════════════════════════════════════════════════════════


def test_every_action_op_constructs_registers_and_runs() -> None:
    """Every ``Op`` action constructs, registers on ``first``, and runs.

    With non-binding stop thresholds and a sample size that keeps
    everyone, the composed census is exactly computable: scale 10 by
    0.98, pin WT|WT to 7, add 5 then subtract 1 on every genotype, so
    adults enter reproduction as (11, 4, 4) per sex.  Random mating of
    that genotype mix is Hardy-Weinberg with Dr frequency 6/19, giving
    19·(13/19)², 19·2·(13/19)(6/19), 19·(6/19)² per sex.
    """
    ops = [
        nt.Op.scale(genotypes="*", ages="*", sex="both", factor=0.98),
        nt.Op.set_count(genotypes="WT|WT", sex="both", value=7.0),
        nt.Op.add(genotypes="*", sex="both", delta=5.0),
        nt.Op.subtract(genotypes="*", sex="both", delta=1.0),
        nt.Op.kill(genotypes="*", sex="both", prob=0.0),
        nt.Op.sample(genotypes="*", sex="both", size=100),
        nt.Op.stop_if_below(genotypes="*", sex="both", threshold=0.0),
        nt.Op.stop_if_above(genotypes="*", sex="both", threshold=1.0e9),
        nt.Op.stop_if_zero(sex="female"),
    ]
    pop = _builder("FrozenAllOpsPop", *ops, event="first").build()

    pop.run(1)

    # No stop op fired: the tick advanced.
    assert pop.tick == 1
    counts = pop.state.individual_count
    np.testing.assert_allclose(counts[:, 1, :], [[169 / 19, 156 / 19, 36 / 19]] * 2)
    # Conservation: (11, 4, 4) parents per sex recruit exactly 19 per sex.
    np.testing.assert_allclose(counts[:, 1, :].sum(axis=1), [19.0, 19.0])
    np.testing.assert_allclose(counts[:, 0, :], 0.0)


def test_op_set_count_pins_exact_census() -> None:
    """``Op.set_count`` replaces the selected census exactly."""
    pop = _builder(
        "FrozenSetCountPop",
        nt.Op.set_count(genotypes="WT|WT", sex="both", value=7.0),
        event="first",
    ).build()

    pop.run(1)

    counts = pop.state.individual_count
    # Discrete populations keep adults on the canonical age-1 layer.
    np.testing.assert_allclose(counts[:, 1, 0], [7.0, 7.0])


def test_op_convert_full_rate_shifts_genotype_then_breeds_mendelian() -> None:
    """``Op.convert(p=1.0)`` moves every individual, then breeding spreads.

    After converting 10 WT|WT pairs to WT|Dr, one generation of
    WT|Dr × WT|Dr reproduction yields the 1:2:1 ratio 2.5/5/2.5.
    """
    pop = _builder(
        "FrozenConvertPop",
        nt.Op.convert("WT|WT", "WT|Dr", 1.0),
        event="first",
    ).build()

    pop.run(1)

    counts = pop.state.individual_count
    # Both sexes recruit the identical 1:2:1 ratio.
    np.testing.assert_allclose(counts[:, 1, :], [[2.5, 5.0, 2.5]] * 2)
    np.testing.assert_allclose(counts[:, 0, :], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Events and priority
# ══════════════════════════════════════════════════════════════════════════════


def test_event_order_first_once_early_late_per_tick_finish_once() -> None:
    """``first`` fires once, ``early``/``late`` per tick, ``finish`` once."""
    events: list[str] = []

    def recorder(name: str) -> Callable[[nt.DiscreteGenerationPopulation], int]:
        def cb(pop: nt.DiscreteGenerationPopulation) -> int:
            events.append(name)
            return 0

        cb.__name__ = f"record_{name}"
        return cb

    pop = (
        _builder("FrozenEventOrderPop")
        .hooks(recorder("first"), event="first")
        .hooks(recorder("early"), event="early")
        .hooks(recorder("late"), event="late")
        .hooks(recorder("finish"), event="finish")
        .build()
    )

    pop.run(2, finish=True)

    assert events == ["first", "early", "late", "first", "early", "late", "finish"]


def test_priority_orders_callbacks_within_one_event() -> None:
    """Same-event callbacks execute in ascending priority order."""
    order: list[str] = []

    def late_cb(pop: nt.DiscreteGenerationPopulation) -> int:
        order.append("p10")
        return 0

    def early_cb(pop: nt.DiscreteGenerationPopulation) -> int:
        order.append("p0")
        return 0

    late_cb.__name__ = "frozen_late"
    early_cb.__name__ = "frozen_early"

    pop = (
        _builder("FrozenPriorityHookPop")
        .hooks(late_cb, event="early", priority=10)
        .hooks(early_cb, event="early", priority=0)
        .build()
    )

    pop.run(1)

    assert order == ["p0", "p10"]


def test_bare_op_registration_defaults_to_early_event() -> None:
    """An undecorated ``Op`` registered without ``event=`` runs as ``early``.

    Per tick the lifecycle order is first → reproduction → early →
    survival → late, so the default-early ``set_count`` overwrites the
    freshly reproduced census with its pinned value.
    """
    pop = _builder(
        "FrozenBareOpPop", nt.Op.set_count(genotypes="*", sex="both", value=3.0)
    ).build()

    compiled = pop.get_compiled_hooks("early")
    assert len(compiled) == 1

    pop.run(1)
    counts = pop.state.individual_count
    # The census pin proves the default-early op executed on the living
    # tick: every genotype cell is exactly the set value.
    np.testing.assert_allclose(counts[:, 1, :], [[3.0, 3.0, 3.0]] * 2)
    np.testing.assert_allclose(counts[:, 0, :], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Conditions and scheduling
# ══════════════════════════════════════════════════════════════════════════════


def test_when_condition_false_leaves_dynamics_untouched() -> None:
    """A never-true ``when`` gate is observationally identical to no hook."""
    gated = _builder(
        "FrozenWhenGatedPop",
        nt.Op.add(genotypes="*", sex="both", delta=1000.0, when="tick >= 99"),
        event="first",
    ).build()
    plain = _builder("FrozenWhenPlainPop").build()

    gated.run(3)
    plain.run(3)

    np.testing.assert_array_equal(
        gated.state.individual_count, plain.state.individual_count
    )


def test_when_condition_true_fires_only_on_matching_ticks() -> None:
    """A ``tick % 2 == 0`` gate commits its write only on ticks 0, 2, 4.

    The parameter log is the audit trail: each committed ``set_param``
    write lands as one ``(tick, name, old, new)`` row, so the gate
    schedule is observable exactly.
    """
    pop = _builder(
        "FrozenWhenParityPop",
        nt.Op.set_param(
            "carrying_capacity", "K * 0.5", when="tick % 2 == 0"
        ),
        event="late",
    ).build()

    pop.run(5)

    log = pop.params_log
    assert [row[0] for row in log] == [0, 2, 4]
    assert [row[2] for row in log] == [100000.0, 50000.0, 25000.0]
    assert [row[3] for row in log] == [50000.0, 25000.0, 12500.0]
    assert pop.params.carrying_capacity == 12500.0


def test_set_param_every_start_schedule() -> None:
    """``Op.set_param(every=2, start=2)`` applies on ticks 2, 4, ... ."""
    pop = _builder(
        "FrozenSetParamPop",
        nt.Op.set_param("carrying_capacity", "K * 0.5", every=2, start=2),
        event="late",
    ).build()

    pop.run(5)

    # K: 100000 → 50000 (tick 2) → 25000 (tick 4).
    assert pop.params.carrying_capacity == 25000.0
    # The log pins the schedule: no write before the start tick.
    assert [row[0] for row in pop.params_log] == [2, 4]


def test_set_param_every_without_start_begins_at_tick_zero() -> None:
    """``every=`` without ``start=`` defaults to firing from tick 0."""
    pop = _builder(
        "FrozenEveryNoStartPop",
        nt.Op.set_param("carrying_capacity", "K * 0.5", every=2),
        event="late",
    ).build()

    pop.run(5)

    # Fires on ticks 0, 2, 4 — one row earlier than the start=2 form.
    assert [row[0] for row in pop.params_log] == [0, 2, 4]
    assert pop.params.carrying_capacity == 12500.0


# ══════════════════════════════════════════════════════════════════════════════
# Decorator author forms and deme selection
# ══════════════════════════════════════════════════════════════════════════════


def test_decorator_selector_callback_receives_pop_and_resolved_index() -> None:
    """``@nt.hook(selectors=...)`` injects resolved indices as parameters."""
    received: list[tuple[int, int]] = []

    @nt.hook(event="early", selectors={"het": "WT|Dr"})
    def watch(pop: nt.DiscreteGenerationPopulation, het: int) -> int:
        received.append((int(pop.tick), het))
        return 0

    pop = _builder("FrozenSelectorCbPop", watch).build()

    pop.run(2)

    # WT|Dr is genotype index 1 in this species.
    assert received == [(0, 1), (1, 1)]


def test_hooks_name_labels_the_compiled_group() -> None:
    """``.hooks(..., name=...)`` labels the compiled descriptor group."""
    pop = _builder(
        "FrozenNamedHookPop",
        nt.Op.set_count(genotypes="*", sex="both", value=3.0),
        event="early",
        name="census_pin",
    ).build()

    compiled = pop.get_compiled_hooks("early")
    assert [d.name for d in compiled] == ["census_pin"]

    pop.run(1)
    np.testing.assert_allclose(
        pop.state.individual_count[:, 1, :], [[3.0, 3.0, 3.0]] * 2
    )


def test_stop_if_extinction_halts_and_closes_the_population() -> None:
    """``Op.stop_if_extinction`` stops as soon as the census hits zero."""
    pop = _builder(
        "FrozenExtinctPop",
        nt.Op.kill(genotypes="*", sex="both", prob=1.0),
        nt.Op.stop_if_extinction(),
        event="first",
    ).build()

    pop.run(5)

    # The kill emptied the census at the first event, so the stop op
    # short-circuited the very first tick and the run closed the door.
    assert pop.tick == 0
    np.testing.assert_allclose(pop.state.individual_count, 0.0)
    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(1)


def test_deme_scoped_hook_touches_only_selected_demes() -> None:
    """``@nt.hook(deme=[...])`` confines declarative ops to those demes."""

    @nt.hook(event="first", deme=[0, 2])
    def boost_selected() -> list[nt.HookOp]:
        return [nt.Op.add(genotypes="*", sex="both", delta=100.0)]

    def _spatial(name: str, with_hook: bool) -> nt.SpatialPopulation:
        chain = (
            nt.SpatialPopulation.builder(
                _species(), n_demes=4, pop_type="discrete_generation"
            )
            .setup(name=name, stochastic=False)
            .initial_state(
                individual_count=nt.batch_setting(
                    [
                        {"female": {"WT|WT": 50.0}, "male": {"WT|WT": 50.0}},
                    ]
                    * 4
                )
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
            .migration(
                adjacency=np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                ),
                migration_rate=0.0,
            )
        )
        if with_hook:
            chain = chain.hooks(boost_selected)
        return chain.build()

    hooked = _spatial("FrozenDemeHookOn", with_hook=True)
    plain = _spatial("FrozenDemeHookOff", with_hook=False)

    hooked.run(1)
    plain.run(1)

    hooked_states = [d.state.individual_count for d in hooked.demes]
    plain_states = [d.state.individual_count for d in plain.demes]
    # Without the hook all four demes are bit-identical (same census,
    # same parameters, zero migration).
    for s in plain_states[1:]:
        np.testing.assert_array_equal(s, plain_states[0])
    # The two selected demes receive the identical boost and stay
    # bit-identical to each other...
    np.testing.assert_array_equal(hooked_states[0], hooked_states[2])
    # ...while the unselected demes match the hook-free baseline exactly.
    np.testing.assert_array_equal(hooked_states[1], plain_states[1])
    np.testing.assert_array_equal(hooked_states[3], plain_states[3])
    # The boost is real: selected demes grew strictly beyond baseline.
    assert hooked_states[0].sum() > plain_states[0].sum()
