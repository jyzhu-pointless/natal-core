"""Adversarial tests for the hook domain and recording.

Every semantic asserted here was probed empirically on the Rust
lifecycle backend before being locked in.  Directions covered (beyond
``tests/test_hooks_slice4.py``):

1. ``stop()`` semantics per event (first/early/late), the
   finished-guard / ``reset()`` recovery state machine, and stop-flag
   non-leakage across runs.
2. The two-layer state loan: hook writes feed subsequent engine stages;
   external tampering with ``pop._state`` between runs cannot reach the
   session-owned engine.
3. ``metrics`` exact numeric agreement for a known allele mixture,
   including the zero-total degenerate case.
4. Parameter snapshot completeness: no-change runs append zero rows, the
   log tuple is an immutable per-call snapshot, and hook writes carry
   the hook's tick.
5. Negative contract: the njit-era registration surface
   (``set_hook`` / ``get_hooks`` / ``remove_hook`` / ``hook_entries`` /
   ``njit_fn`` / ``py_wrapper`` / ``hook_set_param`` / ``unified_hook``)
   is inaccessible; a rejected registration leaves the population
   untouched.
6. Op-as-hook identity dedup across build-time and runtime entrypoints;
   distinct ops stack with exact arithmetic.
7. Runtime ``pop.update().hooks(...)`` is bitwise-equivalent to
   build-time ``.hooks(...)``.
8. ``ctx.rng`` reproducibility under global ``numpy.random`` pollution
   and per-hook-index stream independence.
9. Error path: a raising hook propagates as a Python exception (wrapped
   as ``RuntimeError`` on the Rust bridge), the session survives, and a
   disarmed re-run completes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks import Op
from natal.frontend.hooks.tick_context import TickContext

# ---------------------------------------------------------------------------
# Builders and helpers
# ---------------------------------------------------------------------------


def _build(
    name: str,
    *,
    hooks: list[object] | None = None,
    hook_calls: list | None = None,
    alleles: tuple[str, ...] = ("WT", "Dr"),
    carrying_capacity: float | None = None,
) -> nt.DiscreteGenerationPopulation:
    """Quiescent discrete population (eggs=0, juveniles only, no selection).

    Initialized with 100 juveniles of each homozygote per sex in age 0
    and no adults, so discrete reproduction short-circuits (no mating-age
    females) and hook mutations of age 0 stay exactly observable through
    aging.
    """
    species = nt.Species.from_dict(
        name=f"s4x_{name}",
        structure={"chr1": {"loc": list(alleles)}},
    )
    allele_names = {f"{a}|{a}": [100.0, 0.0] for a in alleles}
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": allele_names,
                "male": allele_names,
            }
        )
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    )
    if carrying_capacity is not None:
        builder = builder.competition(carrying_capacity=carrying_capacity)
    if hooks:
        builder = builder.hooks(*hooks)
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


def _ztype_index(ctx: TickContext, base: str) -> int:
    """Resolve a ztype id from a catalog base name (``"A|A"``)."""
    for idx, name in enumerate(ctx.blueprint.ztype_names):
        if name.split(":")[0].split("@")[0] == base:
            return idx
    raise AssertionError(f"ztype {base!r} missing from catalog")


def _live_ctx(pop: nt.DiscreteGenerationPopulation) -> TickContext:
    """A read-only context view over the population's current state."""
    return TickContext(pop, tick=pop.tick, deme_id=-1, state=pop._state)  # pyright: ignore[reportPrivateUsage]  # TickContext borrows the live arrays (short-term loan)


# ---------------------------------------------------------------------------
# 1. stop() semantics per event
# ---------------------------------------------------------------------------


def test_first_stop_prevents_downstream_events_and_tick() -> None:
    """A first-event stop skips early/late/aging and freezes the tick."""
    events: list[str] = []

    @nt.hook(event="first")
    def stopper(pop: TickContext) -> int:
        events.append("first")
        pop.state.individual_count[:, 0, 0] += 1.0  # first-event marker
        pop.stop()
        return 0

    @nt.hook(event="early")
    def early_marker(pop: TickContext) -> int:
        events.append("early")
        pop.state.individual_count[:, 0, 0] += 10.0
        return 0

    @nt.hook(event="late")
    def late_marker(pop: TickContext) -> int:
        events.append("late")
        pop.state.individual_count[:, 0, 0] += 100.0
        return 0

    pop = _build(
        "s4x_first_stop",
        hooks=[stopper, early_marker, late_marker],
    )
    pop.run(n_steps=5)

    assert events == ["first"]  # early/late never fired, no retry
    assert pop.tick == 0
    assert pop._finished
    ic = pop.state.individual_count
    # The first-event marker (+1) applied; the early (+10) and late (+100)
    # markers did not.  Aging did not run either (age 1 still empty).
    assert float(ic[0, 0, 0]) == 101.0
    assert float(ic[1, 0, 0]) == 101.0
    assert float(ic[:, 1, :].sum()) == 0.0


def test_early_stop_skips_late_and_aging() -> None:
    """An early-event stop skips the late event and the aging stage."""
    events: list[str] = []

    @nt.hook(event="first")
    def first_marker(pop: TickContext) -> int:
        events.append("first")
        return 0

    @nt.hook(event="early")
    def early_stopper(pop: TickContext) -> int:
        events.append("early")
        pop.state.individual_count[:, 0, 2] += 1.0  # early marker
        pop.stop()
        return 0

    @nt.hook(event="late")
    def late_marker(pop: TickContext) -> int:
        events.append("late")
        pop.state.individual_count[:, 0, 2] += 100.0  # would betray late
        return 0

    pop = _build(
        "s4x_early_stop",
        hooks=[first_marker, early_stopper, late_marker],
    )
    pop.run(n_steps=3)  # must not resume after the stop

    assert events == ["first", "early"]
    assert pop.tick == 0
    assert pop._finished
    ic = pop.state.individual_count
    # Late marker absent (+100), aging skipped (age 1 empty).  The early
    # marker added exactly 1 per sex to the 100+100 Dr|Dr juveniles.
    assert float(ic[:, 0, 2].sum()) == 202.0
    assert float(ic[:, 1, 2].sum()) == 0.0


def test_late_stop_halts_at_event_boundary_before_aging() -> None:
    """A late-event stop ends the run immediately; aging does not run.

    Implemented semantics (identical on rust / python): the tick
    does NOT complete — the stop aborts at the event boundary, so the
    tick counter stays put and the aging stage is skipped.
    """
    events: list[str] = []

    @nt.hook(event="late")
    def late_stopper(pop: TickContext) -> int:
        events.append("late")
        pop.state.individual_count[:, 0, 2] += 3.0  # late marker
        pop.stop()
        return 0

    pop = _build(
        "s4x_late_stop",
        hooks=[late_stopper],
    )
    pop.run(n_steps=2)

    assert events == ["late"]  # fired once; the run ended, no second tick
    assert pop.tick == 0
    assert pop._finished
    ic = pop.state.individual_count
    # Late marker applied to both sexes; aging skipped (age 1 empty).
    assert float(ic[:, 0, 2].sum()) == 206.0  # 200 Dr|Dr juveniles +3+3
    assert float(ic[:, 1, 2].sum()) == 0.0


def test_stop_guard_blocks_rerun_until_reset() -> None:
    """After a stop the finished-guard rejects run(); reset() recovers."""
    calls: list[int] = []
    stop_already = {"done": False}

    @nt.hook(event="first")
    def stop_once(pop: TickContext) -> int:
        calls.append(pop.tick)
        if not stop_already["done"]:
            stop_already["done"] = True
            pop.stop()
        return 0

    pop = _build("s4x_guard", hooks=[stop_once])
    pop.run(n_steps=3)
    assert pop._finished
    assert calls == [0]

    with pytest.raises(RuntimeError, match="has finished"):
        pop.run(n_steps=1)  # the stop flag keeps the guard closed

    pop.reset()  # the sanctioned recovery: clears flag, restores state
    assert pop.tick == 0
    assert not pop._finished
    pop.run(n_steps=2)
    # The hook ran once per tick after the reset (stop did not leak).
    assert calls == [0, 0, 1]
    assert pop.tick == 2
    assert not pop._finished


def test_nonzero_return_stops_on_rust_backend() -> None:
    """Returning nonzero from a callback stops the Rust batch run."""
    calls: list[int] = []

    @nt.hook(event="early")
    def return_stopper(pop: TickContext) -> int:
        calls.append(pop.tick)
        return 7  # any nonzero code must stop

    pop = _build("s4x_rust_ret", hooks=[return_stopper])
    pop.run(n_steps=4)

    assert calls == [0]
    assert pop.tick == 0
    assert pop._finished


# ---------------------------------------------------------------------------
# 2. Two-layer state discipline (short-term loan vs long-term isolation)
# ---------------------------------------------------------------------------


def test_hook_state_write_feeds_subsequent_engine_stages() -> None:
    """A hook's state write is consumed by the lifecycle stages after it."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def boost(pop: TickContext) -> int:
        captured.append(pop)
        pop.state.individual_count[:, 0, 0] += 5.0
        return 0

    pop = _build("s4x_loan", hooks=[boost])
    pop.run(n_steps=1)

    # Aging (a later stage in the same tick) consumed the hook's values:
    # (100 + 5) juveniles per sex moved from age 0 to age 1, exactly.
    assert float(pop.state.individual_count[:, 1, 0].sum()) == 210.0
    assert float(pop.state.individual_count[:, 0, 0].sum()) == 0.0
    assert len(captured) == 1


def test_external_state_tampering_between_runs_rejected() -> None:
    """External state tampering between runs is contained.

    The session owns the state outright, so a tampered Python
    cache simply cannot reach the engine — the next run's trajectory is
    bit-identical to an untampered twin.
    """
    pop = _build("s4x_tamper")
    twin = _build("s4x_tamper_twin")
    pop.run(n_steps=1)
    twin.run(n_steps=1)

    # The public state snapshots since R5, so write the live container
    # (the cache) directly to attempt the tamper.
    pop._state.individual_count[0, 0, 0] = 777.0  # pyright: ignore[reportPrivateUsage]

    pop.run(n_steps=1)
    twin.run(n_steps=1)
    np.testing.assert_array_equal(
        pop.state.individual_count, twin.state.individual_count
    )


def test_blueprint_view_is_read_only_and_cached() -> None:
    """Blueprint attributes reject assignment and the view is stable."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        captured.append(pop)
        return 0

    pop = _build("s4x_bp_ro", hooks=[capture])
    pop.run(n_steps=1)

    bp = captured[0].blueprint
    assert captured[0].blueprint is bp  # cached per context
    assert isinstance(bp.ztype_names, tuple)  # immutable catalog
    assert bp.continuous_sampling is False
    assert bp.extreme_speed_mode == 0
    assert len(bp.gtype_names) > 0  # catalog projected for gtypes too
    with pytest.raises(AttributeError):
        bp.n_sexes = 5  # type: ignore[misc]
    with pytest.raises(AttributeError):
        bp.ztype_names = ("X",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. metrics: exact numbers for a known mixture
# ---------------------------------------------------------------------------


def test_metrics_mixture_exact_frequencies() -> None:
    """female A|A 600 + male A|B 400 → p(A)=0.8, p(B)=0.2, exact."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        _ = pop.metrics  # Materialize a detached state during the active event.
        captured.append(pop)
        return 0

    pop = _build(
        "s4x_mixture",
        hooks=[capture],
        alleles=("A", "B"),
    )
    aa = _ztype_index(_live_ctx(pop), "A|A")
    ab = _ztype_index(_live_ctx(pop), "A|B")
    # Overwrite the quiescent start with the target mixture (age 0).
    # Setup writes reach the live container (pop.state is a snapshot since R5).
    state = pop.state
    state.individual_count[:] = 0.0  # pyright: ignore[reportPrivateUsage]
    state.individual_count[0, 0, aa] = 600.0  # female A|A
    state.individual_count[1, 0, ab] = 400.0  # male A|B

    pop.import_state(state)
    pop.trigger_event("first")
    ctx = captured[0]
    catalog = ctx.blueprint.ztype_names

    # Counts project through the name catalog (slab-qualified keys).
    counts = ctx.metrics.genotype_counts
    assert set(counts) == set(catalog)
    assert counts["A|A:default"] == 600.0
    assert counts["A|B:default"] == 400.0
    assert counts["B|B:default"] == 0.0

    # Aggregates.
    assert ctx.metrics.total == 1000.0
    np.testing.assert_allclose(ctx.metrics.by_sex, [600.0, 400.0])
    np.testing.assert_allclose(ctx.metrics.by_age, [1000.0, 0.0])
    assert ctx.metrics.genotype_frequencies == {
        "A|A:default": 0.6,
        "A|B:default": 0.4,
        "B|B:default": 0.0,
    }

    # Allele frequencies: two gene copies per diploid individual.
    af = ctx.metrics.allele_frequencies["loc"]
    a_copies = 2.0 * 600.0 + 400.0
    total_copies = 2.0 * 1000.0
    assert af["A"] == a_copies / total_copies  # bit-exact reference
    assert af["A"] == pytest.approx(0.8)
    assert af["B"] == pytest.approx(0.2)
    assert af["A"] + af["B"] == 1.0


def test_metrics_zero_total_maps_frequencies_to_zero() -> None:
    """An empty population yields 0.0 genotype frequencies, no alleles."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        _ = pop.metrics  # Materialize a detached state during the active event.
        captured.append(pop)
        return 0

    pop = _build("s4x_zero", hooks=[capture])
    # Setup write reaches the live container (pop.state is a snapshot since R5).
    state = pop.state
    state.individual_count[:] = 0.0  # pyright: ignore[reportPrivateUsage]
    pop.import_state(state)
    pop.trigger_event("first")

    ctx = captured[0]
    assert ctx.metrics.total == 0.0
    assert ctx.metrics.genotype_frequencies == dict.fromkeys(
        ctx.blueprint.ztype_names, 0.0
    )
    assert ctx.metrics.allele_frequencies == {"loc": {}}


# ---------------------------------------------------------------------------
# 4. Parameter snapshot log completeness
# ---------------------------------------------------------------------------


def test_params_log_snapshot_semantics_and_quiet_run() -> None:
    """No-change runs append zero rows; the log hands out safe snapshots."""
    pop = _build("s4x_log", carrying_capacity=100_000.0)
    assert pop.params_log == ()

    pop.update().competition(carrying_capacity=9000.0)
    first_log = pop.params_log
    assert first_log == ((0, "carrying_capacity", 100_000.0, 9000.0),)

    pop.run(n_steps=3)  # a run without any write records nothing
    assert pop.params_log == first_log

    pop.update().competition(carrying_capacity=8000.0)
    # The earlier snapshot is detached from the internal list.
    assert first_log == ((0, "carrying_capacity", 100_000.0, 9000.0),)
    assert pop.params_log == (
        (0, "carrying_capacity", 100_000.0, 9000.0),
        (3, "carrying_capacity", 9000.0, 8000.0),
    )
    assert pop.params_log is not pop.params_log  # fresh snapshot per call
    with pytest.raises(TypeError):
        pop.params_log[0] = "x"  # type: ignore[index]


def test_params_log_hook_write_tick_attribution() -> None:
    """A hook write lands as exactly one row stamped with the hook tick."""
    @nt.hook(event="first")
    def writer(pop: TickContext) -> int:
        if pop.tick >= 2:  # declared at build; gated to fire from tick 2 on
            pop.params.carrying_capacity = 555.0
        return 0

    pop = _build("s4x_hooklog", carrying_capacity=100_000.0, hooks=[writer])
    pop.run(n_steps=2)  # now at tick 2 (no fire before the gate)
    assert pop.params_log == ()
    pop.run(n_steps=1)

    assert pop.params_log == ((2, "carrying_capacity", 100_000.0, 555.0),)


# ---------------------------------------------------------------------------
# 5. Negative contract: the njit-era surface is gone
# ---------------------------------------------------------------------------


def test_removed_hook_surface_inaccessible() -> None:
    """Deleted njit-era registration surfaces must not be reachable."""
    pop = _build("s4x_neg")

    # Population-level registration surface (njit era).
    for attr in ("set_hook", "get_hooks", "remove_hook", "hook_entries"):
        assert not hasattr(pop, attr), f"pop.{attr} must not exist"

    # Hooks package surface.
    import natal.frontend.hooks as hooks_pkg

    for attr in ("hook_set_param", "unified_hook", "set_hook", "is_njit_function"):
        assert not hasattr(hooks_pkg, attr), (
            f"natal.frontend.hooks.{attr} must not exist"
        )

    # Descriptor payload: binary (plan | callback) only.
    @nt.hook(event="first")
    def probe(pop: TickContext) -> int:
        _ = pop
        return 0

    pop = _build("s4x_neg_probe", hooks=[probe])
    desc = pop.get_compiled_hooks("first")[0]
    for attr in ("njit_fn", "py_wrapper", "static_arrays"):
        assert not hasattr(desc, attr), f"descriptor.{attr} must not exist"
    assert desc.callback is probe
    assert desc.plan is None

    # Codegen module and template directory removed from the package.
    hooks_dir = Path(nt.hooks.__file__).parent
    assert not (hooks_dir / "templates").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("natal.frontend.hooks.compile.codegen")


def test_rejected_declaration_leaves_population_unbuildable() -> None:
    """A TypeError from a bad signature rejects the whole build."""
    def legacy(state: object, config: object, deme_id: object) -> int:
        _ = (state, config, deme_id)
        return 0

    with pytest.raises(TypeError, match="exactly one parameter"):
        _build("s4x_reject_state", hook_calls=[((legacy,), {"event": "first"})])

    # A clean build of the same shape stays usable.
    pop = _build("s4x_reject_state_clean")
    assert pop.get_compiled_hooks() == []  # nothing declared
    assert pop.params_log == ()
    pop.run(n_steps=1)  # the population stays usable
    assert pop.tick == 1


# ---------------------------------------------------------------------------
# 6. Op-as-hook: identity dedup and exact stacking
# ---------------------------------------------------------------------------


def test_op_identity_dedup_across_build_and_runtime() -> None:
    """The same Op object registers once across all entrypoints.

    The probe op is a non-idempotent ``add``: if identity dedup ever
    broke, a double registration would double the delta and the exact
    arithmetic below would fail.
    """
    op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=7.0)
    op.event = "first"
    distinct = Op.add(genotypes="Dr|Dr", ages=0, sex="both", delta=5.0)

    pop = _build(
        "s4x_dedup",
        hooks=[op],  # build-time declaration
        hook_calls=[
            ((op,), {"event": "first"}),  # duplicate declaration: deduped
            ((distinct,), {"event": "first"}),
        ],
    )
    assert len(pop.get_compiled_hooks("first")) == 2  # 1 deduped + 1 distinct

    # Both ops apply exactly (observed mid-tick, before survival noise):
    # add to a lower current takes the exact additive path (target >=
    # current returns target_count verbatim).
    pop.trigger_event("first")
    ic = pop.state.individual_count
    wt = _ztype_index(_live_ctx(pop), "WT|WT")
    dr = _ztype_index(_live_ctx(pop), "Dr|Dr")
    assert float(ic[1, 0, wt]) == 107.0  # +7 applied exactly once
    assert float(ic[0, 0, wt]) == 100.0  # male-only op left females alone
    assert float(ic[0, 0, dr]) == 105.0  # 100 + 5, exact
    assert float(ic[1, 0, dr]) == 105.0


def test_op_group_set_then_add_stacks_exactly() -> None:
    """set_count followed by add applies both ops in order: 30 + 12 = 42.

    Setting 30 onto a current of 100 exercises the deterministic thinning
    path (``current * (target / current)``); the reference expression is
    the bit-exact definition of that arithmetic.
    """
    pop = _build(
        "s4x_stack",
        hook_calls=[
            (
                ([
                    Op.set_count(genotypes="WT|WT", ages=0, sex="both", value=30.0),
                    Op.add(genotypes="WT|WT", ages=0, sex="both", delta=12.0),
                ],),
                {"event": "first"},
            )
        ],
    )
    assert len(pop.get_compiled_hooks("first")) == 1  # one group descriptor

    pop.trigger_event("first")
    ic = pop.state.individual_count
    wt = _ztype_index(_live_ctx(pop), "WT|WT")
    expected = 100.0 * (30.0 / 100.0) + 12.0  # thinning, then exact add
    assert float(ic[0, 0, wt]) == expected
    assert float(ic[1, 0, wt]) == expected

    pop.run(n_steps=1)  # the stacked value survives aging (survival rounds)
    assert float(pop.state.individual_count[:, 1, wt].sum()) == pytest.approx(
        2.0 * expected
    )


# ---------------------------------------------------------------------------
# 7. Runtime .hooks() entry equals build-time .hooks()
# ---------------------------------------------------------------------------


def test_build_hooks_entry_matches_inline_hook_items() -> None:
    """Declaring the same hook set via .hooks() or hook items agrees bit-for-bit."""

    def make_tweak() -> Callable[[TickContext], int]:
        @nt.hook(event="early")
        def tweak(pop: TickContext) -> int:
            pop.state.individual_count[0, 0, 2] += 3.0
            return 0

        return tweak

    op_build = Op.set_count(genotypes="Dr|Dr", ages=0, sex="male", value=11.0)
    op_build.event = "first"
    op_inline = Op.set_count(genotypes="Dr|Dr", ages=0, sex="male", value=11.0)
    op_inline.event = "first"

    pa = _build(
        "s4x_rt_a",
        hooks=[make_tweak(), op_build],
    )
    pb = _build(
        "s4x_rt_b",
        hook_calls=[((make_tweak(), op_inline), {"event": "first"})],
    )

    pa.run(n_steps=3)
    pb.run(n_steps=3)

    assert np.array_equal(pa.state.individual_count, pb.state.individual_count), (
        "both declaration spellings must be bitwise-equivalent"
    )


# ---------------------------------------------------------------------------
# 8. ctx.rng determinism and global-stream isolation
# ---------------------------------------------------------------------------


def test_rng_stream_reproducible_under_global_pollution() -> None:
    """Same pop name → identical stream regardless of the global RNG state."""
    draws: list[list[float]] = []

    def make_sampler() -> Callable[[TickContext], int]:
        @nt.hook(event="first")
        def sampler(pop: TickContext) -> int:
            draws.append(list(pop.rng.random(4)))
            return 0

        return sampler

    np.random.seed(123)
    p1 = _build("s4x_rng_fixed", hooks=[make_sampler()])
    p1.run(n_steps=2)
    d1 = list(draws)
    draws.clear()

    np.random.seed(999)  # pollute the global stream differently
    p2 = _build("s4x_rng_fixed", hooks=[make_sampler()])
    p2.run(n_steps=2)
    d2 = list(draws)
    draws.clear()

    assert d1 == d2 and len(d1) == 2  # tick 0 and tick 1 replay exactly

    p3 = _build("s4x_rng_other", hooks=[make_sampler()])
    p3.run(n_steps=2)
    assert list(draws) == d1  # names do not replace the session seed


def test_hook_samplers_share_an_advancing_controlled_stream() -> None:
    """Callbacks consume successive draws and their contexts expire on return."""
    draws: dict[str, list[float]] = {}
    contexts: list[TickContext] = []
    stable_identity: list[bool] = []

    def make(tag: str) -> Callable[[TickContext], int]:
        """Create one callback with an independently observable draw group."""
        @nt.hook(event="first")
        def sampler(ctx: TickContext) -> int:
            """Consume four values from the current Rust stream."""
            stable_identity.append(ctx.rng is ctx.rng)
            contexts.append(ctx)
            draws[tag] = list(ctx.rng.random(4))
            return 0
        return sampler

    pop = _build("s4x_rng_idx", hooks=[make("a"), make("b")])
    pop.run(n_steps=1)
    assert all(stable_identity)
    assert len(draws["a"]) == len(draws["b"]) == 4
    assert draws["a"] != draws["b"]
    for ctx in contexts:
        with pytest.raises(RuntimeError, match="expired"):
            ctx.rng.random()


def test_hook_rng_does_not_touch_numpy_global_stream() -> None:
    """A hook consuming ctx.rng leaves the legacy global stream untouched."""
    draws: list[list[float]] = []

    @nt.hook(event="first")
    def heavy_sampler(pop: TickContext) -> int:
        draws.append(list(pop.rng.random(16)))
        return 0

    np.random.seed(20260902)
    expected_after = np.random.random(8)  # the next 8 global draws
    np.random.seed(20260902)

    with_hook = _build("s4x_rng_poll_a", hooks=[heavy_sampler])
    with_hook.run(n_steps=1)
    observed_after_hook = np.random.random(8)

    np.random.seed(20260902)
    without_hook = _build("s4x_rng_poll_b")
    without_hook.run(n_steps=1)
    observed_after_plain = np.random.random(8)

    assert len(draws[0]) == 16
    np.testing.assert_array_equal(observed_after_hook, observed_after_plain)
    np.testing.assert_array_equal(observed_after_hook, expected_after)


# ---------------------------------------------------------------------------
# 5b. Registration error paths and identity rules (BasePopulation hook surface)
# ---------------------------------------------------------------------------


def test_declaration_rejects_invalid_items() -> None:
    """Non-Op list entries and unsupported item types raise TypeError."""
    with pytest.raises(TypeError, match="only HookOp"):
        _build("s4x_bad_items_a", hook_calls=[((["not-an-op"],), {"event": "first"})])

    with pytest.raises(TypeError, match="Unsupported hook item"):
        _build("s4x_bad_items_b", hook_calls=[((42,), {"event": "first"})])

    pop = _build("s4x_bad_items_clean")
    assert pop.get_compiled_hooks() == []


def test_declarative_op_without_event_defaults_to_early() -> None:
    """A bare Op with no event declares at the documented ``early`` default."""
    pop = _build(
        "s4x_op_no_event",
        hook_calls=[((Op.set_count(genotypes="WT|WT", ages=0, value=1.0),), {})],
    )
    descs = pop.get_compiled_hooks()
    assert len(descs) == 1
    assert descs[0].event == "early"


def test_declaration_event_overrides_op_early_default() -> None:
    """An explicit declaration-level event wins over the early default."""
    pop = _build(
        "s4x_op_event_override",
        hook_calls=[
            (
                (Op.set_param("carrying_capacity", "K * 0.95", every=10),),
                {"event": "late"},
            )
        ],
    )
    descs = pop.get_compiled_hooks()
    assert len(descs) == 1
    assert descs[0].event == "late"


def test_meta_object_without_register_rejected() -> None:
    """An object carrying hook meta must expose register()."""
    class FakeDecorated:
        meta = {"event": "first"}

        def __call__(self) -> int:  # pragma: no cover - must not be reached
            raise AssertionError("declaration must reject before calling")

    with pytest.raises(TypeError, match="expose register"):
        _build("s4x_meta_no_register", hook_calls=[((FakeDecorated(),), {})])


def test_plain_callable_without_event_rejected() -> None:
    """A plain single-parameter callable needs an explicit event."""
    def cb(pop: TickContext) -> int:
        _ = pop
        return 0

    with pytest.raises(ValueError, match="No event specified"):
        _build("s4x_plain_no_event", hook_calls=[((cb,), {})])


def test_op_group_dedup_by_tuple_identity() -> None:
    """Declaring the same op list twice yields one group descriptor."""
    group = [
        Op.set_count(genotypes="WT|WT", ages=0, sex="both", value=5.0),
        Op.add(genotypes="Dr|Dr", ages=0, sex="both", delta=1.0),
    ]
    pop = _build(
        "s4x_group_dedup",
        hook_calls=[
            ((group,), {"event": "first"}),
            ((list(group),), {"event": "first"}),  # same op objects, new list
        ],
    )

    assert len(pop.get_compiled_hooks("first")) == 1


def test_trigger_event_unknown_event_is_noop() -> None:
    """An unknown event name returns CONTINUE and leaves state alone."""
    pop = _build("s4x_trigger_bogus")
    before = pop.state.individual_count.copy()

    assert pop.trigger_event("no-such-event") == 0
    np.testing.assert_array_equal(pop.state.individual_count, before)


def test_has_python_hooks_alias_agrees() -> None:
    """has_python_hooks() mirrors has_python_callbacks()."""
    empty = _build("s4x_alias_empty")
    assert empty.has_python_callbacks() is False
    assert empty.has_python_hooks() == empty.has_python_callbacks()

    @nt.hook(event="first")
    def cb(pop: TickContext) -> int:
        _ = pop
        return 0

    pop = _build("s4x_alias_cb", hooks=[cb])
    assert pop.has_python_callbacks() is True
    assert pop.has_python_hooks() is True


# ---------------------------------------------------------------------------
# 5c. Deme-selector serialization and panmictic filtering
# ---------------------------------------------------------------------------


def test_deme_selector_serialization_and_panmictic_filter() -> None:
    """int/range/list selectors serialize into the program and filter demes.

    The panmictic default trigger executes as deme 0, so the range
    selector ``[0, 2)`` matches it; a deme no selector covers applies
    no marker.
    """

    def make_cb(tag: str) -> Callable[[TickContext], int]:
        @nt.hook(event="first")
        def cb(pop: TickContext) -> int:
            pop.state.individual_count[0, 0, 0] += 1000.0
            _ = tag
            return 0

        return cb

    selectors: list[object] = [2, range(0, 2), [1, 3]]
    descriptors: list["nt.hooks.CompiledHookDescriptor"] = []
    for idx, sel in enumerate(selectors):
        cb = make_cb(f"t{idx}")
        descriptors.append(
            nt.hooks.CompiledHookDescriptor(
                name=f"sel_{idx}",
                event="first",
                deme_selector=sel,  # type: ignore[arg-type]
                callback=cb,
                source=cb,
            )
        )

    # Inject through the same internal channel clones travel through.
    species = nt.Species.from_dict(
        name="s4x_deme_sel_inject", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    template = _build("s4x_deme_sel")
    pop = type(template)(
        species=template.species,
        population_config=template.config,
        name="s4x_deme_sel",
        hook_descriptors=descriptors,
    )
    before = pop.state.individual_count.copy()

    # All three selector shapes landed in the CSR program (types 1/2/3).
    types = pop._hook_program.deme_selector_types.tolist()
    assert sorted(t for t in types if t != 0) == [1, 2, 3]

    # Deme 5 matches no selector: no marker applied.
    assert pop.trigger_event("first", deme_id=5) == 0
    np.testing.assert_array_equal(pop.state.individual_count, before)

    # The panmictic default trigger executes as deme 0: the range
    # selector [0, 2) matches and applies its marker at [0, 0, 0] only.
    assert pop.trigger_event("first") == 0
    expected = before.copy()
    expected[0, 0, 0] += 1000.0
    np.testing.assert_array_equal(pop.state.individual_count, expected)


def test_runner_skips_non_tick_event_descriptors() -> None:
    """A hook on 'initialization' never enters the in-tick runner."""
    fired: list[int] = []

    @nt.hook(event="initialization")
    def init_hook(pop: TickContext) -> int:
        fired.append(pop.tick)
        return 0

    pop = _build("s4x_init_event", hooks=[init_hook])
    pop.run(n_steps=1)

    assert fired == []  # 'initialization' has no in-tick event id
    assert pop.tick == 1

    runner = pop._ensure_hook_runner()
    assert runner.has_callbacks() is False  # the runner never indexed it


# ---------------------------------------------------------------------------
# 9. Error path: raising hooks
# ---------------------------------------------------------------------------


def test_hook_exception_requires_recovery_before_session_reuse() -> None:
    """Original exceptions propagate; Failed sessions require restore or reset."""
    armed = {"live": True}

    @nt.hook(event="first")
    def fragile(ctx: TickContext) -> int:
        """Fail one callback, then remain disarmed after explicit recovery."""
        if armed["live"]:
            armed["live"] = False
            raise ValueError("hook boom")
        return 0

    pop = _build("s4x_boom", hooks=[fragile])
    initial = pop.export_state().copy()
    with pytest.raises(ValueError, match="hook boom"):
        pop.run(n_steps=2)
    assert not pop._running
    np.testing.assert_array_equal(pop.export_state(), initial)
    with pytest.raises(RuntimeError):
        pop.run(n_steps=1)
    pop.reset()
    np.testing.assert_array_equal(pop.export_state(), initial)
    pop.run(n_steps=1)
    assert pop.tick == 1


class TestRustDiscreteFemaleOp:
    """rust discrete x female-targeted CSR mutation op must not panic.

    Regression guard: execute_event unconditionally sliced the sperm
    storage row for female targets, but the discrete engine passes an
    empty sperm slice (no sperm dimension) -> range-out-of-range panic.
    """

    def test_rust_discrete_female_scale_op_no_panic(self) -> None:
        try:
            from natal.backends.rust.rust_backend import rust_backend_available
        except ImportError:
            pytest.skip("rust backend module unavailable")
        if not rust_backend_available():
            pytest.skip("rust extension not built")
        import natal as nt

        sp = nt.Species.from_dict(
            name="slice4_female_op",
            structure={"chr1": {"loc": ["A"]}},
            gamete_labels=["default"],
        )
        op = nt.Op.scale("A|A", factor=0.5)
        pop = (
            nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
            .initial_state(
                individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}}
            )
            .hooks(op, event="early")
            .build()
        )
        pop._initialize_session(seed=0)
        pop.run(3, record_every=0)
        total = float(pop.state.individual_count.sum())
        assert total > 0.0 and total == total  # finite and positive
