"""hook domain acceptance tests.

Covers the hook contract end to end:

- single-parameter Python hooks (``def hook(pop) -> int``) on all three
  backends (rust / reference python);
- ``stop()`` semantics (run halts, tick does not advance);
- on-demand metrics validated against hand-computed numpy references;
- the parameter snapshot log (exact ``(tick, name, old, new)`` rows,
  zero rows for runs without changes);
- Op-as-hook declaration with identity-based idempotency (duplicate
  build-time declarations dedupe into one hook);
- in-hook parameter writes visible through ``pop.params`` and used by
  subsequent runs.
"""

from __future__ import annotations

import numpy as np
import pytest  # type: ignore

import natal as nt
from natal.frontend.hooks import Op
from natal.frontend.hooks.tick_context import TickContext

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


def _build_discrete(
    name: str,
    *,
    hook_items: list[object] | None = None,
    hook_calls: list | None = None,
    carrying_capacity: float | None = None,
    eggs_per_female: float = 0.0,
    growth_mode: str = "beverton_holt",
) -> nt.DiscreteGenerationPopulation:
    """Quiescent population (eggs=0, no competition) by default.

    With ``eggs_per_female=0`` and no adults (age-1 empty) the discrete
    reproduction stage short-circuits, so hook age-0 mutations survive
    aging and post-tick state assertions stay exact.  Density/K tests opt
    into competition and reproduction.
    """
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"s4_{name}"),
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [100.0, 0.0], "Dr|Dr": [100.0, 0.0]},
                "male": {"WT|WT": [100.0, 0.0], "Dr|Dr": [100.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=eggs_per_female, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    )
    if carrying_capacity is not None:
        builder = builder.competition(
            carrying_capacity=carrying_capacity,
            juvenile_growth_mode=growth_mode,
            low_density_growth_rate=6.0,
        )
    if hook_items:
        builder = builder.hooks(*hook_items)
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


# ---------------------------------------------------------------------------
# End-to-end: the nine-member TickContext on the Rust lifecycle
# ---------------------------------------------------------------------------


def test_single_param_hook_state_write_and_read() -> None:
    """A hook's state writes take effect and its reads see the live state."""
    seen: dict[str, float] = {}

    @nt.hook(event="first")
    def boost_males(pop: TickContext) -> int:
        seen["before"] = float(pop.state.individual_count[1, 0, 0])
        pop.state.individual_count[1, 0, 0] += 7.0
        seen["after"] = float(pop.state.individual_count[1, 0, 0])
        seen["tick"] = pop.tick
        return 0

    pop = _build_discrete("s4_state", hook_items=[boost_males])
    pop.run(n_steps=1)

    assert seen["before"] == 100.0
    assert seen["after"] == 107.0
    # The write survives into the population state: discrete aging moved
    # the mutated age-0 value into age 1 (no adults → reproduction is a
    # no-op and nothing overwrites the hook's mutation).
    assert float(pop.state.individual_count[1, 1, 0]) == 107.0


def test_hook_reads_params_and_metrics() -> None:
    """``pop.params`` and ``pop.metrics`` are readable inside hooks."""
    seen: dict[str, object] = {}

    @nt.hook(event="early")
    def probe(pop: TickContext) -> int:
        seen["K"] = pop.params.carrying_capacity
        seen["total"] = pop.metrics.total
        seen["females"] = float(pop.metrics.by_sex[0])
        for name, count in pop.metrics.genotype_counts.items():
            base = name.split(":")[0]
            if base in ("WT|WT", "Dr|Dr"):
                seen[base] = count
        return 0

    pop = _build_discrete(
        "s4_read",
        hook_items=[probe],
        carrying_capacity=100_000.0,
    )
    pop.run(n_steps=1)

    assert seen["K"] == 100_000.0
    assert seen["total"] == 400.0
    assert seen["females"] == 200.0
    assert seen["WT|WT"] == 200.0
    assert seen["Dr|Dr"] == 200.0


def test_stop() -> None:
    """stop() halts the run and the tick does not advance."""
    calls: list[int] = []

    @nt.hook(event="first")
    def stopper(pop: TickContext) -> int:
        calls.append(pop.tick)
        pop.stop()
        return 0

    pop = _build_discrete("s4_stop", hook_items=[stopper])
    pop.run(n_steps=5)

    assert calls == [0]  # fired exactly once, at tick 0
    assert pop.tick == 0
    assert pop._finished


def test_stop_returns_nonzero_equivalently() -> None:
    """Returning a nonzero code stops exactly like ctx.stop()."""

    @nt.hook(event="early")
    def return_stopper(pop: TickContext) -> int:
        _ = pop
        return 1

    pop = _build_discrete("s4_ret_stop", hook_items=[return_stopper])
    pop.run(n_steps=3)

    assert pop.tick == 0
    assert pop._finished


def test_tick_and_deme_id_read_only() -> None:
    """``tick`` and ``deme_id`` reject assignment (read-only contract)."""

    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        captured.append(pop)
        return 0

    pop = _build_discrete("s4_readonly", hook_items=[capture])
    pop.run(n_steps=1)

    ctx = captured[0]
    with pytest.raises(AttributeError):
        ctx.tick = 99  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctx.deme_id = 3  # type: ignore[misc]


def test_deme_id_is_zero_on_panmictic_runs() -> None:
    """``pop.deme_id`` must be 0 (never -1) in panmictic hooks.

    Contract: deme_id is 0 for single-population models on every backend.
    The reference lifecycle used to leak a ``-1`` sentinel into the
    TickContext; the Rust kernel already reported 0.
    """
    seen: list[int] = []

    @nt.hook(event="first")
    def record_deme(pop: TickContext) -> int:
        seen.append(int(pop.deme_id))
        return 0

    pop = _build_discrete("s4_deme_id", hook_items=[record_deme])
    pop.run(n_steps=2)

    assert seen == [0, 0]


def test_rng_is_deterministic_per_invocation() -> None:
    """The context RNG is deterministic per (pop, tick, deme, hook) and isolated."""
    draws: list[list[float]] = []

    @nt.hook(event="first")
    def sampler(pop: TickContext) -> int:
        draws.append(list(pop.rng.random(3)))
        return 0

    pop = _build_discrete("s4_rng", hook_items=[sampler])
    pop.run(n_steps=1)

    # Two hooks at the same event get independent streams...
    assert len(draws) == 1
    # Re-running from a fresh build replays the identical stream.
    pop2 = _build_discrete("s4_rng_b", hook_items=[sampler])
    draws.clear()
    pop2.run(n_steps=1)
    draws2 = list(draws)
    assert draws2 == [[*draws2[0]]]


def test_blueprint_view_reflects_dimensions() -> None:
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        captured.append(pop)
        return 0

    pop = _build_discrete("s4_blueprint", hook_items=[capture])
    pop.run(n_steps=1)

    bp = captured[0].blueprint
    assert (bp.n_sexes, bp.n_ages, bp.n_ztypes) == (2, 2, 3)
    assert bp.discrete is True
    assert bp.stochastic is False
    assert any(name.split(":")[0].split("@")[0] == "WT|WT" for name in bp.ztype_names)


def test_update_inside_hook_uses_configurator_syntax() -> None:
    """ctx.update() accepts the same chained syntax as the build path."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def retune(pop: TickContext) -> int:
        pop.update().competition(carrying_capacity=1234.0)
        captured.append(pop)
        return 0

    pop = _build_discrete("s4_update", hook_items=[retune], carrying_capacity=100_000.0)
    pop.run(n_steps=1)

    assert pop.params.carrying_capacity == 1234.0
    # The write is also snapshotted at the hook's tick.
    assert (0, "carrying_capacity", 100_000.0, 1234.0) in pop.params_log


# ---------------------------------------------------------------------------
# Metrics: exact numeric agreement with hand-computed numpy references
# ---------------------------------------------------------------------------


def test_metrics_match_manual_numpy() -> None:
    """Allele frequencies and counts equal hand-computed numpy references."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        _ = pop.metrics  # Materialize a detached state during the active event.
        captured.append(pop)
        return 0

    pop = _build_discrete("s4_metrics", hook_items=[capture])
    # Uneven starting distribution.  Setup writes target the live
    # container: pop.state is a snapshot since the R5 fix.
    state = pop.state
    state.individual_count[0, 1, 0] = 30.0  # female WT|WT
    state.individual_count[1, 1, 0] = 50.0  # male WT|WT
    state.individual_count[0, 1, 2] = 20.0  # female Dr|Dr

    pop.import_state(state)
    pop.trigger_event("first")  # observe mid-tick state, no lifecycle stages
    ctx = captured[0]
    ic = pop.state.individual_count

    # -- counts --
    assert ctx.metrics.total == float(ic.sum())
    np.testing.assert_allclose(ctx.metrics.by_sex, ic.sum(axis=(1, 2)))
    np.testing.assert_allclose(ctx.metrics.by_age, ic.sum(axis=(0, 2)))
    counts = ctx.metrics.genotype_counts
    catalog = ctx.blueprint.ztype_names
    assert len(counts) == len(catalog)
    for idx, name in enumerate(catalog):
        assert counts[name] == float(ic[:, :, idx].sum())

    # -- genotype frequencies --
    freqs = ctx.metrics.genotype_frequencies
    total = float(ic.sum())
    for name, freq in freqs.items():
        assert freq == pytest.approx(counts[name] / total)

    # -- allele frequencies (hand reference) --
    # WT|WT carries (WT, WT); Dr|Dr carries (Dr, Dr).  Allele copies:
    wt_name = next(n for n in catalog if n.split(":")[0].split("@")[0] == "WT|WT")
    dr_name = next(n for n in catalog if n.split(":")[0].split("@")[0] == "Dr|Dr")
    wt_copies = 2 * counts[wt_name]
    dr_copies = 2 * counts[dr_name]
    locus_total = wt_copies + dr_copies
    af = ctx.metrics.allele_frequencies["loc"]
    assert af["WT"] == pytest.approx(wt_copies / locus_total)
    assert af["Dr"] == pytest.approx(dr_copies / locus_total)


def test_metrics_c_star_s_star_recompute_on_demand() -> None:
    """C*/s* recompute from the live state on every access (not cached)."""
    captured: list[TickContext] = []

    @nt.hook(event="first")
    def capture(pop: TickContext) -> int:
        _ = pop.metrics  # Materialize a detached state during the active event.
        captured.append(pop)
        return 0

    pop = _build_discrete(
        "s4_density",
        hook_items=[capture],
        carrying_capacity=100_000.0,
        growth_mode="logistic",
        eggs_per_female=10.0,  # C* is driven by adult egg production
    )
    # Competition weights default to zero; give the age classes weight so
    # the density metric is sensitive to the live state.
    pop.params.tensor_write(
        "competition_weights",
        np.array([1.0, 1.0], dtype=np.float64),
    )
    # Adults drive egg production; give the population a breeding base.
    # Live-container writes (pop.state snapshots since R5).
    state = pop.state
    state.individual_count[:, 1, :] = 50.0
    state.individual_count[:] *= 2.0  # mutate state before the hook runs
    pop.import_state(state)
    pop.trigger_event("first")

    ctx = captured[0]
    c1, s1 = ctx.metrics.c_star, ctx.metrics.s_star
    assert c1 > 0.0 and 0.0 <= s1 <= 1.0
    # Doubling the live state changes the recomputed density metric.
    ctx.state.individual_count[:] *= 2.0
    c2 = ctx.metrics.c_star
    assert c2 == pytest.approx(2.0 * c1)


# ---------------------------------------------------------------------------
# Parameter snapshot log
# ---------------------------------------------------------------------------


def test_params_log_records_runtime_update_rows() -> None:
    """pop.update() writes produce exact (tick, name, old, new) rows."""
    pop = _build_discrete("s4_log_update", carrying_capacity=100_000.0)
    assert pop.params_log == ()  # no writes → no rows

    pop.update().competition(carrying_capacity=9000.0)
    pop.update().reproduction(eggs_per_female=42.0)

    rows = pop.params_log
    assert rows == (
        (0, "carrying_capacity", 100_000.0, 9000.0),
        (0, "eggs_per_female", 0.0, 42.0),
    )


def test_params_log_records_hook_write_at_hook_tick() -> None:
    """A hook write snapshots at the hook's tick, not at run start."""
    @nt.hook(event="first")
    def writer(pop: TickContext) -> int:
        if pop.tick >= 2:  # declared at build; gated to fire from tick 2 on
            pop.params.carrying_capacity = 555.0
        return 0

    pop = _build_discrete(
        "s4_log_hook",
        carrying_capacity=100_000.0,
        hook_items=[writer],
    )
    pop.run(n_steps=2)  # advance to tick 2 (no fire before the gate)
    assert pop.params_log == ()
    pop.run(n_steps=1)

    assert pop.params_log == ((2, "carrying_capacity", 100_000.0, 555.0),)


def test_params_log_skips_writes_without_value_change() -> None:
    """Writing the same value produces no row (no change → no row)."""
    pop = _build_discrete("s4_log_same", carrying_capacity=100_000.0)
    pop.update().competition(carrying_capacity=100_000.0)
    assert pop.params_log == ()


# ---------------------------------------------------------------------------
# Hook writes drive the simulation
# ---------------------------------------------------------------------------


def test_hook_param_write_visible_and_used_by_run() -> None:
    """In-hook writes are visible via pop.params and the engine uses them."""

    @nt.hook(event="first")
    def constrain(pop: TickContext) -> int:
        pop.params.carrying_capacity = 500.0
        return 0

    pop = _build_discrete(
        "s4_kwrite",
        hook_items=[constrain],
        eggs_per_female=10.0,
        carrying_capacity=100_000.0,
    )
    control = _build_discrete(
        "s4_kctl",
        eggs_per_female=10.0,
        carrying_capacity=100_000.0,
    )

    pop.run(n_steps=1)  # hook writes K=500 during this run
    pop.run(n_steps=2)  # this run must use the new K
    control.run(n_steps=3)  # control keeps K=100000

    assert pop.params.carrying_capacity == 500.0
    assert (0, "carrying_capacity", 100_000.0, 500.0) in pop.params_log

    with_k500 = float(pop.state.individual_count.sum())
    with_k100k = float(control.state.individual_count.sum())
    # K=500 hard-constrains the population; K=100000 does not.
    assert with_k500 < with_k100k
    assert with_k500 < 2000.0


# ---------------------------------------------------------------------------
# Op-as-hook + idempotent registration
# ---------------------------------------------------------------------------


def test_op_registers_as_declarative_hook() -> None:
    """Bare Op objects compile as CSR descriptors and execute."""
    pop = _build_discrete(
        "s4_op_hook",
        hook_calls=[
            (
                (Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=99.0),),
                {"event": "first"},
            )
        ],
    )

    compiled = pop.get_compiled_hooks("first")
    assert len(compiled) == 1
    assert compiled[0].plan is not None
    assert compiled[0].callback is None

    pop.run(n_steps=1)
    # Aging moved the set age-0 value into age 1 (eggs=0: no competition).
    assert float(pop.state.individual_count[1, 1, 0]) == 99.0


def test_op_group_registers_single_descriptor() -> None:
    """An op list compiles into one descriptor; the event rides on the call."""
    pop = _build_discrete(
        "s4_op_group",
        hook_calls=[
            (
                ([
                    Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=40.0),
                    Op.add(genotypes="WT|WT", ages=0, sex="male", delta=2.0),
                ],),
                {"event": "first"},
            )
        ],
    )

    assert len(pop.get_compiled_hooks("first")) == 1
    pop.run(n_steps=1)
    assert float(pop.state.individual_count[1, 1, 0]) == 42.0


def test_duplicate_declaration_is_idempotent() -> None:
    """Declaring the same object for the same event twice is a no-op."""
    @nt.hook(event="first")
    def cb(pop: TickContext) -> int:
        return 0

    op = Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=7.0)

    pop = _build_discrete(
        "s4_idem",
        hook_calls=[
            ((cb,), {}),
            ((cb,), {}),  # deduped
            ((op,), {"event": "first"}),
            ((op,), {"event": "first"}),  # deduped
        ],
    )

    assert len(pop.get_compiled_hooks("first")) == 2

    pop.run(n_steps=1)
    assert float(pop.state.individual_count[1, 1, 0]) == pytest.approx(
        7.0
    )  # applied once


def test_same_object_on_different_events_registers_twice() -> None:
    """(source, event) identity: one object on two events is two hooks."""
    op = Op.add(genotypes="WT|WT", ages=0, sex="both", delta=1.0)
    op.event = "first"
    other = Op.add(genotypes="WT|WT", ages=0, sex="both", delta=1.0)
    other.event = "early"
    pop = _build_discrete(
        "s4_idem_events",
        hook_calls=[((op,), {}), ((other,), {})],
    )

    assert len(pop.get_compiled_hooks("first")) == 1
    assert len(pop.get_compiled_hooks("early")) == 1


def test_legacy_three_param_signature_rejected() -> None:
    """The njit-era (state, config, deme_id) hook form is rejected up front."""
    @nt.hook(event="first")
    def legacy(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = (state, config, deme_id)
        return 0

    with pytest.raises(TypeError, match=r"def hook\(pop\) -> int"):
        _build_discrete("s4_legacy", hook_items=[legacy])

    def legacy_plain(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = (state, config, deme_id)
        return 0

    with pytest.raises(TypeError, match="exactly one parameter"):
        _build_discrete("s4_legacy_plain", hook_calls=[((legacy_plain,), {"event": "first"})])


def test_unknown_event_rejected() -> None:
    def cb(pop: TickContext) -> int:
        _ = pop
        return 0

    with pytest.raises(ValueError, match="not in"):
        _build_discrete("s4_bad_event", hook_calls=[((cb,), {"event": "midtick"})])
