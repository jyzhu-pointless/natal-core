#!/usr/bin/env python3
"""Event execution ordering and stop semantics for the hook forms.

Cross-type priority semantics: within one event, CSR declarative plans
and single-parameter Python callbacks execute interleaved in one stable
ascending-priority order (lower values run first; ties keep registration
order).  A stop request from either kind aborts the event, skipping every
later hook of either kind.

The legacy njit-hook interleaving is gone — njit hooks have no migration
channel, so every former njit hook below is a Python callback with the
same mutation semantics.
"""

from __future__ import annotations

from typing import List

import pytest  # type: ignore

import natal as nt
from natal.frontend.hooks import Op, hook


def _make_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chr1": {
                "loc": ["WT", "Dr"],
            }
        },
    )


def _build_discrete_population(name: str) -> nt.DiscreteGenerationPopulation:
    species = _make_species(name)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 10.0]},
                "male": {"WT|WT": [0.0, 10.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


def test_callback_priority_ordering_first_event() -> None:
    pop = _build_discrete_population("cb_priority_first_event")
    calls: List[str] = []
    observed: dict[str, float] = {}

    @hook(event="first", priority=0)
    def first_probe_a(pop):
        _ = pop
        calls.append("probe_a")
        observed["probe_a_seen"] = 10.0

    @hook(event="first", priority=1)
    def first_probe_b(pop):
        calls.append("probe_b")
        observed["probe_b_seen"] = float(pop.state.individual_count[1, 1, 0])

    @hook(event="early", priority=0)
    def early_probe(pop):
        calls.append("early_probe")
        observed["early_seen"] = float(pop.state.individual_count[1, 1, 0])

    pop.update().hooks(first_probe_a, first_probe_b, early_probe)

    pop.run(n_steps=1)

    assert calls[:2] == ["probe_a", "probe_b"]
    assert observed["probe_b_seen"] == 10.0


def test_mixed_csr_then_callbacks_in_order() -> None:
    """CSR(pri=0) → cb(pri=1) → cb(pri=2): priority order across types."""
    pop = _build_discrete_population("mixed_csr_then_cb")
    calls: List[str] = []
    observed: dict[str, float] = {}

    @hook(event="first", priority=0)
    def first_csr_early_pri():
        # priority 0: runs before both callbacks despite being a plan
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="first", priority=1)
    def cb_one(pop):
        calls.append("cb_one")
        pop.state.individual_count[1, 1, 0] += 2.0
        return 0

    @hook(event="first", priority=2)
    def cb_two(pop):
        calls.append("cb_two")
        return 0

    @hook(event="early", priority=0)
    def early_probe(pop):
        calls.append("early_probe")
        observed["early_seen"] = float(pop.state.individual_count[1, 1, 0])

    pop.update().hooks(first_csr_early_pri, cb_one, cb_two, early_probe)
    pop.run(n_steps=1)

    # Observed at the early boundary (before aging wipes age-1):
    # 10 + csr(3) + cb_one(2) = 15: the priority-0 plan ran first.
    assert calls == ["cb_one", "cb_two", "early_probe"]
    assert observed["early_seen"] == 15.0


def test_callback_hooks_run_without_manual_trigger() -> None:
    pop = _build_discrete_population("cb_auto_run")
    calls: List[str] = []

    def python_hook(pop) -> None:
        _ = pop
        calls.append("called")

    pop.update().hooks(python_hook, event="first")
    pop.run(n_steps=1)

    assert calls == ["called"]


def _build_simple_discrete_population(name: str) -> nt.DiscreteGenerationPopulation:
    """Minimal panmictic population for hook execution testing.

    All 10 individuals start at age 0.  Survival=1.0 keeps them alive
    across the single tick (discrete aging shifts age 0 → age 1 at tick end,
    but hooks observe the post-survival, pre-aging state at age 0).
    """
    species = _make_species(name)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [10.0, 0.0]},
                "male": {"WT|WT": [10.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .build()
    )


def test_unified_csr_before_callback() -> None:
    """CSR(pri=0) → callback(pri=1): the callback sees the CSR mutation."""
    pop = _build_simple_discrete_population("unified_csr_before_cb")

    @hook(event="first", priority=0)
    def csr_hook():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20)]

    @hook(event="first", priority=1)
    def cb_hook(pop):
        pop.state.individual_count[1, 0, 0] += 100.0
        return 0

    pop.update().hooks(csr_hook, cb_hook)

    pop.run(n_steps=1)

    # CSR set=20 → callback add=100 → age1 = 120 after aging.
    assert pop.state.individual_count[1, 1, 0] == 120.0


def test_callback_beats_lower_priority_csr() -> None:
    """callback(pri=0) → CSR(pri=1): one priority order across both kinds.

    The callback adds 100 first (10 → 110); the higher-value plan then
    sets the slot to 20 outright.  The final 20 proves the callback ran
    before the plan — the pre-interleaving engine always ran plans first
    and would finish at 120.
    """
    pop = _build_simple_discrete_population("cb_beats_csr")

    @hook(event="early", priority=0)
    def cb_hook(pop):
        pop.state.individual_count[1, 0, 0] += 100.0
        return 0

    @hook(event="early", priority=1)
    def csr_hook():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20)]

    pop.update().hooks(cb_hook, csr_hook)

    pop.run(n_steps=1)

    # callback +100 (110) ran first, then set_count(20) overwrote it.
    assert pop.state.individual_count[1, 1, 0] == 20.0


def test_callback_observes_priority_earlier_csr_write() -> None:
    """A callback reads the mutation of a smaller-priority CSR plan."""
    pop = _build_simple_discrete_population("cb_sees_csr")

    @hook(event="early", priority=0)
    def csr_hook():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20)]

    observed: dict[str, float] = {}

    @hook(event="early", priority=1)
    def cb_hook(pop):
        observed["male_age0"] = float(pop.state.individual_count[1, 0, 0])
        return 0

    pop.update().hooks(csr_hook, cb_hook)

    pop.run(n_steps=1)

    assert observed["male_age0"] == 20.0
    assert pop.state.individual_count[1, 1, 0] == 20.0


def test_callbacks_share_state_mutations_in_order() -> None:
    """Two callbacks on one event see each other's mutations in priority order."""
    pop = _build_simple_discrete_population("cb_share_state")

    @hook(event="first", priority=0)
    def cb_a(pop):
        pop.state.individual_count[1, 0, 0] += 10.0
        return 0

    @hook(event="first", priority=1)
    def cb_b(pop):
        pop.state.individual_count[1, 0, 0] += 5.0
        return 0

    pop.update().hooks(cb_a, cb_b)

    pop.run(n_steps=1)

    # 10 + 10 + 5 = 25 at age-1 after aging.
    assert pop.state.individual_count[1, 1, 0] == 25.0


def test_same_priority_callbacks_stable_order() -> None:
    """Callbacks with equal priority execute in registration order."""
    pop = _build_simple_discrete_population("same_priority_cb")

    @hook(event="first", priority=0)
    def first_registered(pop):
        pop.state.individual_count[1, 0, 0] = 100.0
        return 0

    @hook(event="first", priority=0)
    def second_registered(pop):
        pop.state.individual_count[1, 0, 0] += 1.0
        return 0

    pop.update().hooks(first_registered, second_registered)
    pop.run(n_steps=1)

    # first sets to 100, second adds 1 → 101 at age-1
    assert pop.state.individual_count[1, 1, 0] == 101.0


def test_same_priority_mixed_tie_keeps_registration_order() -> None:
    """Equal-priority CSR plans and callbacks tie in registration order.

    Two interleavings in one event, all at priority 0: the plan registered
    first runs before the callback registered after it, and the callback
    registered first observes the pre-plan state before the later plan.
    """
    pop = _build_simple_discrete_population("tie_mixed_order")
    observed: List[float] = []

    @hook(event="first", priority=0)
    def plan_set_twenty():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20.0)]

    @hook(event="first", priority=0)
    def cb_after_plan(pop):
        observed.append(float(pop.state.individual_count[1, 0, 0]))
        return 0

    @hook(event="first", priority=0)
    def cb_before_plan(pop):
        observed.append(float(pop.state.individual_count[1, 0, 0]))
        return 0

    @hook(event="first", priority=0)
    def plan_set_thirty():
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=30.0)]

    pop.update().hooks(plan_set_twenty, cb_after_plan, cb_before_plan, plan_set_thirty)

    pop.run(n_steps=1)

    # Registration order under one priority: plan(20) -> cb sees 20 ->
    # cb sees 20 -> plan(30). Aging moves the slot to age 1.
    assert observed == [20.0, 20.0]
    assert pop.state.individual_count[1, 1, 0] == 30.0


def test_csr_on_other_event_still_executes() -> None:
    """CSR hooks on a non-mixed event still execute (registry completeness)."""
    pop = _build_simple_discrete_population("csr_other_event")

    @hook(event="first", priority=0)
    def first_cb(pop):
        pop.state.individual_count[1, 0, 0] += 100.0
        return 0

    @hook(event="early", priority=0)
    def early_csr():
        return [Op.add(genotypes="WT|WT", ages=0, sex="male", delta=50)]

    pop.update().hooks(first_cb, early_csr)

    pop.run(n_steps=1)

    # first: 10+100 = 110 → early: +50 = 160 (post-aging age-1).
    assert pop.state.individual_count[1, 1, 0] == 160.0


# ---------------------------------------------------------------------------
# STOP_IF end-to-end lifecycle tests
# ---------------------------------------------------------------------------


def test_stop_if_zero_shortcircuits_remaining_callbacks() -> None:
    """Op.stop_if_zero aborts the event, skipping later callbacks."""
    pop = _build_simple_discrete_population("stop_if_zero")

    @hook(event="first", priority=0)
    def csr_kill():
        # Set male to 0 → triggers stop_if_zero below.
        return [Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=0)]

    @hook(event="first", priority=1)
    def csr_stop():
        return [Op.stop_if_zero(genotypes="WT|WT", ages=0, sex="male")]

    @hook(event="first", priority=2)
    def cb_should_be_skipped(pop):
        pop.state.individual_count[1, 0, 0] += 999.0  # should never execute
        return 0

    pop.update().hooks(csr_kill, csr_stop, cb_should_be_skipped)

    pop.run(n_steps=1)

    # csr_kill set to 0 → csr_stop sees 0 → STOP → callback skipped
    assert pop.state.individual_count[1, 1, 0] == 0.0


def test_stop_if_extinction_shortcircuits_remaining_callbacks() -> None:
    """Op.stop_if_extinction aborts when total population reaches 0."""
    pop = _build_simple_discrete_population("stop_if_extinction")

    @hook(event="first", priority=0)
    def csr_kill():
        # Set both sexes to 0.
        return [
            Op.set_count(genotypes="WT|WT", ages=0, sex="female", value=0),
            Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=0),
        ]

    @hook(event="first", priority=1)
    def csr_stop():
        return [Op.stop_if_extinction()]

    @hook(event="first", priority=2)
    def cb_should_be_skipped(pop):
        pop.state.individual_count[1, 0, 0] += 999.0
        return 0

    pop.update().hooks(csr_kill, csr_stop, cb_should_be_skipped)

    pop.run(n_steps=1)

    assert pop.state.individual_count[1, 1, 0] == 0.0


def test_stop_if_zero_condition_not_met_continues() -> None:
    """Op.stop_if_zero does NOT abort when count > 0."""
    pop = _build_simple_discrete_population("stop_if_zero_continue")

    @hook(event="first", priority=0)
    def csr_stop():
        # Male count is 10 > 0 → condition not met → continue.
        return [Op.stop_if_zero(genotypes="WT|WT", ages=0, sex="male")]

    @hook(event="first", priority=1)
    def cb_should_run(pop):
        pop.state.individual_count[1, 0, 0] += 5.0
        return 0

    pop.update().hooks(csr_stop, cb_should_run)

    pop.run(n_steps=1)

    # STOP not triggered → callback runs → 10 + 5 = 15 at age-1
    assert pop.state.individual_count[1, 1, 0] == 15.0


# ---------------------------------------------------------------------------
# Op type end-to-end lifecycle tests
# ---------------------------------------------------------------------------


def test_op_scale_end_to_end() -> None:
    """Op.scale should multiply individual counts by a factor."""
    pop = _build_simple_discrete_population("op_scale")

    pop.update().hooks(
        Op.scale(genotypes="WT|WT", ages=0, sex="male", factor=0.3),
        event="first",
    )
    pop.run(n_steps=1)

    # 10 * 0.3 = 3 at age-1 (deterministic, no stochastic)
    assert pop.state.individual_count[1, 1, 0] == 3.0


def test_op_sample_end_to_end() -> None:
    """Op.sample should clamp individual counts to at most the given size."""
    pop = _build_simple_discrete_population("op_sample")

    pop.update().hooks(
        Op.sample(genotypes="WT|WT", ages=0, sex="male", size=4),
        event="first",
    )
    pop.run(n_steps=1)

    # 10 clamped to 4 at age-1
    assert pop.state.individual_count[1, 1, 0] == 4.0


def test_op_kill_end_to_end() -> None:
    """Op.kill should remove a fraction of individuals."""
    pop = _build_simple_discrete_population("op_kill")

    pop.update().hooks(
        Op.kill(genotypes="WT|WT", ages=0, sex="male", prob=0.6),
        event="first",
    )
    pop.run(n_steps=1)

    # 10 * (1 - 0.6) = 4 at age-1
    assert pop.state.individual_count[1, 1, 0] == 4.0


def test_op_subtract_end_to_end() -> None:
    """Op.subtract should remove a fixed number of individuals."""
    pop = _build_simple_discrete_population("op_sub")

    pop.update().hooks(
        Op.subtract(genotypes="WT|WT", ages=0, sex="male", delta=7),
        event="first",
    )
    pop.run(n_steps=1)

    # 10 - 7 = 3 at age-1
    assert pop.state.individual_count[1, 1, 0] == 3.0


def test_op_bare_hook_no_event_kwarg() -> None:
    """An Op carrying its own event registers without the event kwarg."""
    pop = _build_simple_discrete_population("op_self_event")

    op = Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=99)
    op.event = "early"
    pop.update().hooks(op)
    pop.run(n_steps=1)

    assert pop.state.individual_count[1, 1, 0] == 99.0


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_no_hooks_runs_normally() -> None:
    """Population with zero registered hooks should run without error."""
    pop = _build_simple_discrete_population("no_hooks")
    pop.run(n_steps=1)
    assert pop.state.individual_count[1, 1, 0] == 10.0


def test_single_callback_hook() -> None:
    """A single callback hook on a single event should execute correctly."""
    pop = _build_simple_discrete_population("single_cb")

    @hook(event="first", priority=0)
    def single(pop):
        pop.state.individual_count[1, 0, 0] += 42.0
        return 0

    pop.update().hooks(single)
    pop.run(n_steps=1)

    assert pop.state.individual_count[1, 1, 0] == 52.0


def test_single_csr_hook() -> None:
    """A single CSR hook on a single event should execute correctly."""
    pop = _build_simple_discrete_population("single_csr")

    pop.update().hooks(
        Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=99),
        event="early",
    )
    pop.run(n_steps=1)

    assert pop.state.individual_count[1, 1, 0] == 99.0


def test_legacy_signature_rejected() -> None:
    """The njit-era (state, config, deme_id) signature is rejected up front."""
    pop = _build_simple_discrete_population("legacy_rejected")

    @hook(event="first")
    def legacy_hook(state, config, deme_id):  # pragma: no cover - rejected
        _ = (state, config, deme_id)
        return 0

    with pytest.raises(TypeError, match="def hook\\(pop\\) -> int"):
        pop.update().hooks(legacy_hook)
