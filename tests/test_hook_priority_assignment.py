"""Priority assignment semantics of ``.hooks()`` op declarations.

Priority is op-level data with assignment resolution (one packed group
carries one priority):

- a call-level ``priority`` assigns one shared priority to the op items
  of that declaration (single ops included);
- without it the ops' own priorities are used and must agree within a
  list — a mixed declaration raises ``ValueError`` at build time;
- decorated functions keep their decorator priority and are not
  reachable by the call-level assignment;
- plain single-parameter callables fall back to ``0`` when the call
  gives none.

These tests pin the resolution rules; cross-descriptor execution
ordering is covered by ``test_ops_addendum_adversarial`` and
``test_ops_setparam_convert``.
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.hooks import Op


def _build(item: object, **hooks_kwargs: object) -> tuple[tuple[int, ...], int]:
    """Build a minimal population declaring *item*; return priorities.

    Args:
        item: The hook declaration passed to ``.hooks()``.
        **hooks_kwargs: Extra keyword arguments forwarded to ``.hooks()``.

    Returns:
        A ``(priorities, total)`` pair: the compiled descriptors'
        priorities in declaration order, and the post-run total count.
    """
    sp = nt.Species.from_dict(
        name=f"prio_{abs(hash((repr(item), repr(sorted(hooks_kwargs.items()))))) % 10**8}",
        structure={"auto": {"A": ["WT", "Var"]}},
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=sp, name="t", stochastic=False)
        .initial_state({"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
        .reproduction(eggs_per_female=0)
        .competition(growth_mode="fixed")
        .hooks(item, **hooks_kwargs)  # type: ignore[arg-type]  # test-only declaration shapes
        .build()
    )
    priorities = tuple(d.priority for d in pop.compiled_hook_descriptors)
    pop.run(n_steps=2, record_every=0)
    return priorities, int(pop.get_total_count())


class TestCallLevelAssignment:
    """A call-level priority assigns the ops of that declaration."""

    def test_single_op_gets_call_priority(self) -> None:
        """`.hooks(op, priority=7)` compiles the op at priority 7."""
        priorities, _ = _build(
            Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0),
            event="early",
            priority=7,
        )
        assert priorities == (7,)

    def test_call_priority_overrides_factory_priority(self) -> None:
        """The registration call is the last declaration: it wins over
        an op-level priority set through the ``Op.set_param`` factory."""
        op = Op.set_param("carrying_capacity", 100, priority=5)
        priorities, _ = _build(op, event="early", priority=7)
        assert priorities == (7,)

    def test_list_members_share_call_priority(self) -> None:
        """Packing = one shared priority: the call value applies to the
        whole group even when members carry their own values."""
        op_plain = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op_scheduled = Op.set_param("carrying_capacity", 100, priority=5)
        priorities, _ = _build([op_plain, op_scheduled], event="early", priority=3)
        assert priorities == (3,)

    def test_explicit_zero_overrides_factory_priority(self) -> None:
        """``priority=0`` is an assignment, not an absence: it must
        override an op's own non-zero priority (this is exactly the
        distinction the ``None`` default exists to preserve)."""
        op = Op.set_param("carrying_capacity", 100, priority=5)
        priorities, _ = _build(op, event="early", priority=0)
        assert priorities == (0,)

    def test_mixed_group_with_call_priority_is_assigned(self) -> None:
        """A call-level priority resolves a mixed group by assignment
        instead of raising: the contradiction only exists without one."""
        op_plain = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op_scheduled = Op.set_param("carrying_capacity", 100, priority=5)
        priorities, _ = _build([op_plain, op_scheduled], event="early", priority=2)
        assert priorities == (2,)


class TestOwnPriorityWithoutCallLevel:
    """Without a call-level priority the ops' own values are used."""

    def test_single_op_defaults_to_zero(self) -> None:
        priorities, _ = _build(
            Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0), event="early"
        )
        assert priorities == (0,)

    def test_factory_priority_survives_packing(self) -> None:
        """An op-level priority set via the factory survives when the
        packed group agrees on it."""
        op = Op.set_param("carrying_capacity", 100, priority=5)
        priorities, _ = _build([op], event="early")
        assert priorities == (5,)

    def test_unanimous_group_keeps_shared_value(self) -> None:
        op_a = Op.set_param("carrying_capacity", 100, priority=5)
        op_b = Op.set_param("eggs_per_female", 10, priority=5)
        priorities, _ = _build([op_a, op_b], event="early")
        assert priorities == (5,)

    def test_mixed_group_without_call_priority_is_rejected(self) -> None:
        """One packed hook carries one priority: a list mixing priorities
        without a call-level assignment is a contradictory declaration
        and fails at build time."""
        op_plain = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op_scheduled = Op.set_param("carrying_capacity", 100, priority=5)
        with pytest.raises(ValueError, match=r"priorities \[0, 5\]"):
            _build([op_plain, op_scheduled], event="early")

    def test_empty_op_list_keeps_noop_descriptor(self) -> None:
        """An empty op list has nothing to disagree about: it keeps the
        historical no-op descriptor instead of raising."""
        priorities, _ = _build([], event="early")
        assert priorities == (0,)


class TestFactoryLevelFields:
    """Every Op factory exposes event / priority as op-level data."""

    def test_factory_priority_on_classic_op_bare(self) -> None:
        """A classic factory op carries its own priority when passed bare."""
        priorities, _ = _build(
            Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0, priority=5),
            event="early",
        )
        assert priorities == (5,)

    def test_factory_priority_on_classic_op_packed(self) -> None:
        """The same op-level priority survives packing (unanimous group)."""
        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0, priority=5)
        priorities, _ = _build([op], event="early")
        assert priorities == (5,)

    def test_call_priority_overrides_factory_value(self) -> None:
        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0, priority=5)
        priorities, _ = _build(op, event="early", priority=8)
        assert priorities == (8,)

    def test_factory_event_wins_over_call_event(self) -> None:
        """An op-level event rides on any factory op, not only set_param."""
        sp = nt.Species.from_dict(
            name="prio_factory_event", structure={"auto": {"A": ["WT", "Var"]}}
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=sp, name="t", stochastic=False)
            .initial_state({"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
            .reproduction(eggs_per_female=0)
            .competition(growth_mode="fixed")
            .hooks(
                Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0, event="late"),
                event="early",
            )
            .build()
        )
        assert [d.event for d in pop.compiled_hook_descriptors] == ["late"]


class TestCallableItems:
    """Callables follow their own channels."""

    def test_decorated_callback_keeps_decorator_priority(self) -> None:
        """The call-level priority is an op-group assignment; a
        decorated callback keeps its decorator priority."""

        @nt.hook(event="early", priority=3)
        def cb(pop: object) -> int:
            return 0

        priorities, _ = _build(cb, priority=9)
        assert priorities == (3,)

    def test_plain_callable_falls_back_to_zero(self) -> None:
        def cb(pop: object) -> int:
            return 0

        priorities, _ = _build(cb, event="early")
        assert priorities == (0,)

    def test_plain_callable_gets_call_priority(self) -> None:
        def cb(pop: object) -> int:
            return 0

        priorities, _ = _build(cb, event="early", priority=4)
        assert priorities == (4,)
