"""Session-owned spatial state contracts.

The heterogeneous Rust spatial session owns the stacked counts, sperm
storage, tick, and one persistent per-deme RNG stream.  ``run_tick``
carries control parameters only; the lifecycle and the migration consume
the same per-deme streams inside Rust; deme reads pull lazily from one
session snapshot; per-deme imports push back into the session.

These contracts include the promoted R1 repro: per-deme streams advance
across ticks instead of being rebuilt from ``seed ^ deme_id`` every tick.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.backends.rust.rust_backend import rust_backend_available
from natal.frontend.hooks.entry.declarative import Op
from natal.frontend.spatial.population import SpatialPopulation

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _ring_adjacency(n: int) -> np.ndarray:
    """Return the dense adjacency of an open chain 0-1-...-(n-1)."""
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        matrix[i, i + 1] = 1.0
        matrix[i + 1, i] = 1.0
    return matrix


def _build(
    name: str,
    seed: int,
    *,
    stochastic: bool = True,
    n_demes: int = 4,
    rate: float = 0.25,
    late_ops: tuple = (),
) -> SpatialPopulation:
    """Build an age-structured spatial population and enable Rust."""
    builder = (
        nt.SpatialPopulation.builder(
            _species(f"{name}sp"), n_demes=n_demes, pop_type="age_structured"
        )
        .setup(name=name, stochastic=stochastic)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
                * n_demes
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(n_demes), migration_rate=rate)
    )
    if late_ops:
        builder = builder.hooks(*late_ops)
    population = builder.build()
    population._initialize_session(seed=seed)
    return population


def _stacked(population: SpatialPopulation) -> np.ndarray:
    """Return the stacked per-deme individual counts (copy)."""
    return np.stack([deme.state.individual_count for deme in population.demes])


def test_persistent_deme_streams_advance_across_ticks() -> None:
    """R1 promotion: streams advance; a rebuilt twin resuming the counts
    from tick 1 must diverge from the uninterrupted run."""
    a = _build("own_r1a", seed=42)
    a.run(1)
    imported = [
        {
            "n_tick": 1,
            "individual_count": deme.state.individual_count.copy(),
            "sperm_storage": deme.state.sperm_storage.copy(),
        }
        for deme in a.demes
    ]
    a.run(1)
    final_a = _stacked(a)

    b = _build("own_r1b", seed=42)
    def restore_counts(ctx: nt.TickContext) -> int:
        state = imported[ctx.deme_id]
        ctx.state.individual_count[:] = state["individual_count"]
        ctx.state.sperm_storage[:] = state["sperm_storage"]
        return 0
    b.register_hooks(restore_counts, event="first")
    b.run(1)
    final_b = _stacked(b)

    assert not np.array_equal(final_a, final_b), (
        "per-deme RNG must advance across ticks: A(two ticks) equals "
        "B(one tick from A's tick-1 state) bitwise"
    )


def test_segmented_runs_match_one_run_bitwise() -> None:
    """run(a+b) equals run(a); run(b) bitwise with persistent streams."""
    whole = _build("own_segwhole", seed=43)
    whole.run(3)
    final_whole = _stacked(whole)

    split = _build("own_segsplit", seed=43)
    split.run(1)
    split.run(2)
    final_split = _stacked(split)

    np.testing.assert_array_equal(final_whole, final_split)


def test_same_seed_reproduces_the_trajectory() -> None:
    """Two same-seed populations produce identical stochastic runs."""
    first = _build("own_repro1", seed=44)
    first.run(3)
    second = _build("own_repro2", seed=44)
    second.run(3)
    np.testing.assert_array_equal(_stacked(first), _stacked(second))


def test_scoped_state_transaction_continues_from_committed_counts() -> None:
    """A scoped state transaction reaches the only native owner.

    With zero migration and no surviving individuals or stored sperm,
    subsequent reproduction cannot resurrect the extinguished deme.
    """
    target = _build("own_imp_target", seed=45, stochastic=False, n_demes=3, rate=0.0)
    target.run(1, record_every=0)
    def extinguish(ctx: nt.TickContext) -> int:
        ctx.state.individual_count[:] = 0
        ctx.state.sperm_storage[:] = 0
        return 0
    target.register_hooks(extinguish, event="first", deme=1)
    target.trigger_event("first", deme_id=1)
    target.run(2, record_every=0)
    np.testing.assert_array_equal(
        target.demes[1].state.individual_count,
        np.zeros((2, 3, 3), dtype=np.float64),
        err_msg="the imported extinction must drive the session's later ticks",
    )
    # The untouched demes keep growing past the import boundary.
    assert (target.demes[0].state.individual_count > 0.0).any()
    assert (target.demes[2].state.individual_count > 0.0).any()


def test_stop_keeps_boundary_state_and_freezes_the_tick() -> None:
    """A late stop keeps the boundary state and freezes the tick."""
    # The population grows ~4x per tick; a 150.0 age-total cap fires the
    # stop during the first late event.
    population = _build(
        "own_stop",
        seed=46,
        stochastic=False,
        late_ops=(Op.stop_if_above("*", "*", "both", threshold=150.0),),
    )
    population.run(5)
    final = _stacked(population)
    assert population._tick < 5  # noqa: SLF001 — stop froze the container tick
    assert np.isfinite(final).all()
    assert (final > 0.0).any(), "stopped run keeps the modifications up to the boundary"
    with pytest.raises(RuntimeError, match="finished"):
        population.run(1)


def test_retained_snapshot_cannot_mutate_the_run() -> None:
    """A deme.state snapshot is independent; callback transactions own writes."""
    population = _build("own_snap", seed=47)
    retained = population.demes[0].state.individual_count
    before = retained.copy()
    population.run(2)
    current = population.demes[0].state.individual_count
    assert not np.array_equal(before, current), "the run advanced deme 0"
    (
        np.testing.assert_array_equal(retained, before),
        ("a retained snapshot must not observe later ticks"),
    )
    retained[...] = 0.0
    (
        np.testing.assert_array_equal(
            population.demes[0].state.individual_count, current
        ),
        "mutating a retained snapshot must not touch the run state",
    )


def test_restore_checkpoint_continues_the_session_run() -> None:
    """restore_checkpoint reaches the session: restore(1) then run(1)
    equals the uninterrupted run(2) bitwise (deterministic)."""
    whole = _build("own_res_whole", seed=50, stochastic=False)
    whole.run(2, record_every=1)
    expected = _stacked(whole)

    restored = _build("own_res_restored", seed=50, stochastic=False)
    restored.run(2, record_every=1)
    restored.restore_checkpoint(1)
    restored.run(1, record_every=1)
    np.testing.assert_array_equal(_stacked(restored), expected)
    assert restored._tick == whole._tick  # noqa: SLF001


def test_reset_reaches_the_session() -> None:
    """reset() resets the session too: reset then run(1) equals a fresh
    population's run(1) bitwise (stochastic streams reseeded from the
    original base seed by the rebuild-free reset path)."""
    fresh = _build("own_reset_fresh", seed=51)
    fresh.run(1)
    expected = _stacked(fresh)

    reseted = _build("own_reset_used", seed=51)
    reseted.run(2, record_every=0)
    reseted.reset()
    reseted.run(1, record_every=0)
    np.testing.assert_array_equal(_stacked(reseted), expected)


def test_runtime_migration_rate_write_reaches_the_session() -> None:
    """tensor_write("migration_rate") changes the next Rust tick's outcome.

    Control and treatment run the same seeded engine; only the treatment
    has the rate column raised mid-run, so any divergence in the tick-2
    state is the write reaching the session.
    """
    control = _build("own_rate_ctl", seed=52, stochastic=False)
    treatment = _build("own_rate_trt", seed=52, stochastic=False)
    control.run(1, record_every=0)
    treatment.run(1, record_every=0)
    np.testing.assert_array_equal(_stacked(control), _stacked(treatment))

    new_rate = np.full((4, 2, 3), 0.5, dtype=np.float64)
    treatment.params.tensor_write("migration_rate", new_rate)
    control.run(1, record_every=0)
    treatment.run(1, record_every=0)
    assert not np.array_equal(_stacked(control), _stacked(treatment))


def test_public_readers_are_fresh_after_control_ticks() -> None:
    """Counts, aggregates, allele frequencies, and deme export reflect the
    tick immediately — even with recording disabled."""
    population = _build("own_fresh", seed=53)
    before_total = population.get_total_count()
    population.run(2, record_every=0)
    after_total = population.get_total_count()
    assert after_total > before_total, "counts must advance with the run"
    assert population.aggregate_individual_count().sum() == float(after_total)
    frequencies = population.compute_allele_frequencies()
    assert set(frequencies) == {"WT", "Dr"}
    # export_state flattens [n_tick, ind.ravel(), sperm.ravel()] off the
    # same (refreshed) caches.
    state = population.demes[0].state
    exported = np.asarray(population.demes[0].export_state())
    assert exported[0] == state.n_tick
    np.testing.assert_array_equal(
        exported[1 : 1 + state.individual_count.size],
        np.asarray(state.individual_count).ravel(),
    )


def test_parallel_and_sequential_schedulers_agree_bitwise() -> None:
    """Callback-carrying (sequential) and callback-free (parallel) runs
    consume the per-deme streams identically: same trajectory bitwise."""

    def make_backend(seed: int, callbacks: bool):
        population = _build("own_sched", seed=seed)
        backend = population._rust_spatial_backend  # noqa: SLF001
        assert backend is not None
        if callbacks:
            fired: list[int] = []

            def noop(ind, sperm, tick, deme_id):  # noqa: ANN001
                fired.append(deme_id)
                return 0

            backend.set_python_callbacks([noop], [], [])
            return backend, fired
        return backend, None

    reference, _ = make_backend(54, callbacks=False)
    for _ in range(3):
        reference.run_tick()
    tick, ind_flat, sperm_flat = reference.state_snapshot()

    sequential, fired = make_backend(54, callbacks=True)
    for _ in range(3):
        sequential.run_tick()
    s_tick, s_ind, s_sperm = sequential.state_snapshot()

    assert s_tick == tick
    np.testing.assert_array_equal(ind_flat, s_ind)
    np.testing.assert_array_equal(sperm_flat, s_sperm)
    # The sequential scheduler really ran the callbacks — 4 demes x 3 ticks.
    assert fired is not None and len(fired) == 12


def test_discrete_spatial_rust_tick_keeps_the_migration_tail() -> None:
    """The legacy per-bank discrete path must still migrate.

    A homogeneous square with rate 0.5 and one asymmetric deme must
    equalize: the rust path tracks the python-dispatch reference bitwise
    (deterministic).  Guards the S3b fusion against dropping the shared
    migration stage again.
    """

    def build(name: str, seed: int, enable_rust: bool) -> SpatialPopulation:
        pop = (
            nt.SpatialPopulation.builder(
                _species(f"{name}sp"), n_demes=4, pop_type="discrete_generation"
            )
            .setup(name=name, stochastic=False)
            .initial_state(
                individual_count=nt.batch_setting(
                    [
                        {"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}},
                        {"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}},
                        {"female": {"WT|WT": 400.0}, "male": {"WT|WT": 400.0}},
                        {"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}},
                    ]
                )
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
            .migration(adjacency=np.ones((4, 4)), migration_rate=0.5)
            .build()
        )
        if enable_rust:
            pop._initialize_session(seed=seed)
        return pop

    reference = build("own_d mig_ref".replace(" ", ""), 61, enable_rust=False)
    rusted = build("own_dmig_rs", 61, enable_rust=True)
    for _ in range(3):
        reference.run(1, record_every=0)
        rusted.run(1, record_every=0)
    np.testing.assert_array_equal(
        _stacked(rusted),
        _stacked(reference),
        err_msg="discrete spatial rust path dropped the migration stage",
    )


def test_declarative_hooks_run_on_discrete_spatial_rust() -> None:
    """R2 promotion: one Program for every model — a declarative halve-K
    hook lands on the discrete spatial Rust path exactly like the plain
    discrete twin (K 10000 -> 2500 over two late-event firings)."""
    @nt.hook(event="late")
    def halve_k() -> list:
        return [nt.Op.set_param("carrying_capacity", "K * 0.5")]

    spatial = (
        nt.SpatialPopulation.builder(
            _species("own_r2sp"), n_demes=4, pop_type="discrete_generation"
        )
        .setup(name="own_r2_spatial", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}},
                ]
                * 4
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(4), migration_rate=0.0)
        .hooks(halve_k)
        .build()
    )
    spatial._initialize_session(seed=7)
    spatial.run(2, record_every=0)
    ks = [deme.params.carrying_capacity for deme in spatial.demes]
    assert ks == [2500.0, 2500.0, 2500.0, 2500.0]

    plain = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species("own_r2plain"), name="own_r2_plain", stochastic=False
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .hooks(halve_k)
        .build()
    )
    plain.run(2, record_every=0)
    assert plain.params.carrying_capacity == 2500.0


def test_discrete_spatial_stochastic_migration_matches_python_dispatch() -> None:
    """The fused discrete kernel's stochastic migration tail matches the
    python-dispatch reference statistically (mass conservation + integer
    counts), and discrete reset restores the random source."""
    def build(name: str, seed: int, enable: bool):
        pop = (
            nt.SpatialPopulation.builder(
                _species(f"{name}sp"), n_demes=4, pop_type="discrete_generation"
            )
            .setup(name=name, stochastic=True)
            .initial_state(
                individual_count=nt.batch_setting(
                    [
                        {"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}},
                    ]
                    * 4
                )
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
            .migration(adjacency=np.ones((4, 4)), migration_rate=0.25)
            .build()
        )
        if enable:
            pop._initialize_session(seed=seed)
        return pop

    population = build("own_dmig", 62, enable=True)
    population.run(2, record_every=0)
    totals = [_stacked(population)[d].sum() for d in range(4)]
    assert all(float(t).is_integer() for t in totals)
    # Reset restores state AND the random source: reset then run(1) equals
    # a fresh same-seed population's run(1) bitwise.
    population.run(1, record_every=0)
    population.reset()
    fresh = build("own_dmig_fresh", 62, enable=True)
    fresh.run(1, record_every=0)
    population.reset()
    population.run(1, record_every=0)
    np.testing.assert_array_equal(_stacked(population), _stacked(fresh))


# ── S3c: Python callbacks on the spatial session ─────────────────────────────


def _build_callback_population(
    name: str,
    seed: int,
    hooks: tuple,
    *,
    n_demes: int = 3,
    stochastic: bool = False,
) -> SpatialPopulation:
    """Build a small age-structured spatial population with *hooks*."""
    builder = (
        nt.SpatialPopulation.builder(
            _species(f"{name}sp"), n_demes=n_demes, pop_type="age_structured"
        )
        .setup(name=name, stochastic=stochastic)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
                * n_demes
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .migration(adjacency=np.eye(n_demes), migration_rate=0.0)
    )
    if hooks:
        builder = builder.hooks(*hooks)
    population = builder.build()
    population._initialize_session(seed=seed)
    return population


def test_python_callbacks_fire_per_deme_and_keep_trajectories() -> None:
    """Callbacks fire once per deme per tick; a no-op hook leaves the
    deterministic trajectory bitwise unchanged (scheduler parity)."""
    fires: list[tuple[int, int]] = []

    @nt.hook(event="early")
    def counter(context: nt.TickContext) -> None:
        fires.append((int(context.tick), int(context.deme_id)))

    plain = _build_callback_population("own_cb_plain", 73, ())
    hooked = _build_callback_population("own_cb_hook", 73, (counter,))
    plain.run(2, record_every=0)
    hooked.run(2, record_every=0)
    assert sorted(fires) == sorted(
        (tick, deme) for tick in (0, 1) for deme in range(3)
    )
    np.testing.assert_array_equal(_stacked(plain), _stacked(hooked))


def test_python_hook_update_lands_for_the_next_tick() -> None:
    """ctx.update writes defer to the next tick (plain-backend semantics).

    A late hook halving K must bind the FOLLOWING tick's density
    regulation: the hooked trajectory diverges from the un-hooked twin
    and the deme draft reads the written value.
    """

    @nt.hook(event="late")
    def halve(context: nt.TickContext) -> None:
        context.update().competition(carrying_capacity=500.0)

    hooked = _build_callback_population(
        "own_upd_h", 74,
        (halve,),
    )
    twin = _build_callback_population("own_upd_n", 74, ())
    hooked.run(2, record_every=0)
    twin.run(2, record_every=0)
    assert hooked.demes[0].params.carrying_capacity == 500.0
    assert twin.demes[0].params.carrying_capacity == 100000.0
    assert not np.array_equal(_stacked(hooked), _stacked(twin))


def test_python_hook_stop_on_discrete_spatial_freezes_the_tick() -> None:
    """A Python stop on the discrete spatial path is a graceful freeze."""

    @nt.hook(event="first")
    def stopper(context: nt.TickContext) -> None:
        context.stop()

    population = (
        nt.SpatialPopulation.builder(
            _species("own_cbsp_sp"), n_demes=2, pop_type="discrete_generation"
        )
        .setup(name="own_cb_stop", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [{"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}}] * 2
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
        .migration(adjacency=np.eye(2), migration_rate=0.0)
        .hooks(stopper)
        .build()
    )
    population._initialize_session(seed=75)
    population.run(3, record_every=0)
    assert population._tick == 0  # noqa: SLF001 — stop froze the tick
    with pytest.raises(RuntimeError, match="finished"):
        population.run(1)


def test_mixed_declarative_and_python_hooks_share_one_program() -> None:
    """A declarative set_param and a Python callback coexist: both land."""
    seen: list[int] = []

    @nt.hook(event="late")
    def observer(context: nt.TickContext) -> None:
        seen.append(int(context.deme_id))

    @nt.hook(event="late")
    def halve_k() -> list:
        return [nt.Op.set_param("carrying_capacity", "K * 0.5")]

    population = (
        nt.SpatialPopulation.builder(
            _species("own_mixsp"), n_demes=2, pop_type="discrete_generation"
        )
        .setup(name="own_mix", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [{"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}}] * 2
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .migration(adjacency=np.eye(2), migration_rate=0.0)
        .hooks(halve_k, observer)
        .build()
    )
    population._initialize_session(seed=76)
    population.run(2, record_every=0)
    assert [deme.params.carrying_capacity for deme in population.demes] == [
        2500.0,
        2500.0,
    ]
    assert sorted(seen) == [0, 0, 1, 1]


def test_python_callbacks_fire_on_every_registered_event() -> None:
    """F2 guard: hooks on multiple events all fire (first+early+late)."""
    seen_events: list[str] = []

    @nt.hook(event="first")
    def on_first(context: nt.TickContext) -> None:
        seen_events.append("first")

    @nt.hook(event="early")
    def on_early(context: nt.TickContext) -> None:
        seen_events.append("early")

    @nt.hook(event="late")
    def on_late(context: nt.TickContext) -> None:
        seen_events.append("late")

    population = _build_callback_population(
        "own_multi", 77, (on_first, on_early, on_late)
    )
    population.run(2, record_every=0)
    # 3 demes x 2 ticks per event.
    assert sorted(seen_events) == sorted(
        ["first"] * 6 + ["early"] * 6 + ["late"] * 6
    )


def test_python_update_routes_to_every_owning_deme() -> None:
    """F1 guard: a build-time hook writing ctx.update lands on ALL demes
    (cloned demes must not share one runner bound to deme 0)."""

    @nt.hook(event="late")
    def halve(context: nt.TickContext) -> None:
        context.update().competition(carrying_capacity=500.0)

    population = _build_callback_population("own_route", 78, (halve,))
    population.run(2, record_every=0)
    assert [
        deme.params.carrying_capacity for deme in population.demes
    ] == [500.0, 500.0, 500.0]


def test_declarative_and_python_same_tick_writes_compose() -> None:
    """A declarative eggs write and a python K write in the same tick must
    compose: the declarative commit binds the NEXT tick's journal old
    value and the python write lands beside it (no clobber)."""
    journal_seen: list[tuple] = []

    @nt.hook(event="first")
    def bump_eggs() -> list:
        return [nt.Op.set_param("eggs_per_female", "eggs_per_female + 1", every=1)]

    @nt.hook(event="first")
    def write_k(context: nt.TickContext) -> None:
        journal_seen.append(len(journal_seen))
        context.update().competition(carrying_capacity=500.0)

    population = (
        nt.SpatialPopulation.builder(
            _species("own_composesp"), n_demes=1, pop_type="age_structured"
        )
        .setup(name="own_compose", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .migration(adjacency=np.eye(1), migration_rate=0.0)
        .hooks(bump_eggs, write_k)
        .build()
    )
    population._initialize_session(seed=79)
    population.run(3, record_every=0)
    eggs_log = [
        row for row in population.demes[0].params_log
        if row[1] == "eggs_per_female"
    ]
    # Tick 0 commits 4->5; tick 1 must run with eggs=5 and commit 5->6 —
    # the python K write (a different field) must not revert the tick-0
    # declarative commit, and the K write lands beside it.
    assert eggs_log[0] == (0, "eggs_per_female", 4.0, 5.0)
    assert eggs_log[1] == (1, "eggs_per_female", 5.0, 6.0)
    assert population.demes[0].params.carrying_capacity == 500.0


def test_python_vector_write_reaches_every_sharing_deme() -> None:
    """F8 guard: a ctx.update of a VECTOR ecology field on a homogeneous
    build (demes sharing one config object, vectors written in place)
    must refresh the session column of every sharing deme."""

    @nt.hook(event="late")
    def slash_survival(context: nt.TickContext) -> None:
        context.update().survival(female_age_based_survival=[1.0, 0.1, 0.0])

    population = _build_callback_population(
        "own_vec", 80, (slash_survival,)
    )
    twin = _build_callback_population("own_vec_n", 80, ())
    population.run(2, record_every=0)
    twin.run(2, record_every=0)
    # All demes read the new vector...
    for deme in population.demes:
        assert float(deme.params.survival_rates[0][1]) == 0.1
    # ...and every deme's TRAJECTORY diverged from the un-hooked twin
    # (each session column was refreshed, not just deme 0's).
    stacked_hooked = _stacked(population)
    stacked_twin = _stacked(twin)
    for deme_index in range(3):
        assert not np.array_equal(
            stacked_hooked[deme_index], stacked_twin[deme_index]
        ), f"deme {deme_index} kept the stale survival column"


# ── S4a: spatial session CheckpointStore ─────────────────────────────────────


def _build_stochastic_spatial(name: str, seed: int) -> SpatialPopulation:
    """Build a stochastic 4-deme age-structured spatial population."""
    return (
        nt.SpatialPopulation.builder(
            _species(f"{name}sp"), n_demes=4, pop_type="age_structured"
        )
        .setup(name=name, stochastic=True)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
                * 4
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(4), migration_rate=0.25)
        .build()
    )._initialize_session(seed=seed)


def test_spatial_stochastic_restore_replays_bitwise() -> None:
    """Full checkpoint restore (state + RNG bank + ecology) replays the
    original stochastic trajectory bitwise: restore(1) then run(3) equals
    the uninterrupted run(4)."""
    whole = _build_stochastic_spatial("own_ckpt_whole", 90)
    whole.run(4, record_every=1)
    final_whole = _stacked(whole)

    restored = _build_stochastic_spatial("own_ckpt_restored", 90)
    restored.run(4, record_every=1)
    restored.restore_checkpoint(1)
    restored.run(3, record_every=1)
    final_restored = _stacked(restored)

    np.testing.assert_array_equal(final_whole, final_restored)


def test_spatial_restore_rolls_back_ecology_and_revives() -> None:
    """Restore rolls the ecology columns back and revives the runnable
    state after a declarative hook changed K mid-run."""
    @nt.hook(event="late")
    def halve_k() -> list:
        return [nt.Op.set_param("carrying_capacity", "K * 0.5")]

    population = (
        nt.SpatialPopulation.builder(
            _species("own_ckpt_ecosp"), n_demes=2, pop_type="age_structured"
        )
        .setup(name="own_ckpt_eco", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
                * 2
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=80000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(2), migration_rate=0.0)
        .hooks(halve_k)
        .build()
    )._initialize_session(seed=91)
    population.run(2, record_every=1)
    assert population.demes[0].params.carrying_capacity == 20000.0

    population.restore_checkpoint(0)
    assert population.demes[0].params.carrying_capacity == 80000.0
    assert population.demes[1].params.carrying_capacity == 80000.0
    # Restoring a runnable boundary revives the runnable state (plan 9).
    population.run(1, record_every=0)


def test_spatial_clear_history_drops_the_checkpoints() -> None:
    """clear_history also drops the session checkpoints: a cleared tick no
    longer restores."""
    population = _build_stochastic_spatial("own_ckpt_clear", 92)
    population.run(3, record_every=1)
    population.clear_history()
    with pytest.raises(ValueError, match="No history available"):
        population.restore_checkpoint(1)


def test_restore_ignores_out_of_run_param_writes() -> None:
    """A param write BETWEEN runs must not poison the restored boundary:
    restore(2) after a runtime migration_rate write continues from the
    checkpoint's ecology (the clean twin), not the written one."""
    written = _build_stochastic_spatial("own_ckpt_written", 93)
    written.run(2, record_every=1)
    written.params.tensor_write("migration_rate", 0.5)
    written.run(1, record_every=1)
    written.restore_checkpoint(2)
    written.run(2, record_every=1)

    clean = _build_stochastic_spatial("own_ckpt_clean", 93)
    clean.run(2, record_every=1)
    clean.run(2, record_every=1)
    np.testing.assert_array_equal(_stacked(written), _stacked(clean))
    # The restored migration rate is the checkpoint's (the build's 0.25
    # ring), not the 0.5 written between the runs.
    np.testing.assert_array_equal(
        written.params.migration_rate, clean.params.migration_rate
    )
    assert float(written.params.migration_rate.max()) == 0.25


def test_spatial_restore_rolls_back_vector_columns() -> None:
    """Vector ecology columns (survival_rates) roll back on restore: the
    container column AND the public params read return the checkpoint
    values, not a stale pre-restore write."""
    population = _build_stochastic_spatial("own_ckpt_vec", 94)
    population.run(2, record_every=1)
    original = np.asarray(population.params.survival_rates).copy()

    modified = original.copy()
    modified[:, 1] = 0.05
    population.params.tensor_write("survival_rates", modified)
    population.run(1, record_every=1)
    population.restore_checkpoint(2)

    np.testing.assert_array_equal(
        np.asarray(population.params.survival_rates), original
    )
    for deme_id, deme in enumerate(population.demes):
        np.testing.assert_array_equal(
            np.asarray(deme.params.survival_rates), original[deme_id]
        )


def test_checkpoint_eviction_tracks_history_eviction() -> None:
    """Evicted history rows take their checkpoints: the store stays
    bounded by the same budget, and the newest boundary stays
    restorable."""
    population = _build_stochastic_spatial("own_evict", 95)
    # Shrink the history bound: only the newest 2 rows survive.
    population._history_obj.max_rows = 2  # pyright: ignore[reportPrivateUsage]
    population.run(5, record_every=1)
    assert len(population.history.ticks) == 2
    oldest_tick = population.history.ticks[0]
    # The newest restorable boundary still replays bitwise against a twin
    # that only recorded the same window.
    restored = _build_stochastic_spatial("own_evict_twin", 95)
    restored.run(5, record_every=1)
    restored.restore_checkpoint(int(oldest_tick))
    restored.run(5 - int(oldest_tick), record_every=1)
    np.testing.assert_array_equal(_stacked(restored), _stacked(population))


def test_plain_record_snapshot_pairs_checkpoint_eviction() -> None:
    """Plain-model manual snapshots also drop evicted checkpoints:
    restoring an evicted tick raises cleanly instead of half-restoring
    (the tick is rewound, then the missing history row aborts)."""
    population = (
        nt.AgeStructuredPopulation.setup(
            species=_species("own_snap_evictsp"),
            name="own_snap_evict",
            stochastic=True,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 100.0, 0.0]},
                "male": {"WT|WT": [0.0, 100.0, 0.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0)
        .record_history(mode="raw", max_rows=2)
        .build()
    )
    population._initialize_session(seed=96)
    population.run(3, record_every=2)
    assert population.history.ticks == (0, 2)
    population.record_snapshot()
    assert population.history.ticks == (2, 3)
    # The tick-0 checkpoint was evicted with its row: restoring it raises
    # cleanly and leaves the population at the manual snapshot tick.
    with pytest.raises(ValueError):
        population.restore_checkpoint(0)
    assert population.tick == 3


def test_spatial_program_rebases_set_param_literals() -> None:
    """Two declarative set_param hooks with literals each read their OWN
    literal: the second hook must not consume the first hook's pool slot
    (spatial program concatenation rebases RPN literal indices)."""

    @nt.hook(event="first")
    def double_eggs() -> list:
        return [nt.Op.set_param("eggs_per_female", "4.0 * 2", every=1)]

    @nt.hook(event="late")
    def halve_k() -> list:
        return [nt.Op.set_param("carrying_capacity", "K * 0.5", every=1)]

    population = (
        nt.SpatialPopulation.builder(
            _species("own_rpnsp"), n_demes=2, pop_type="discrete_generation"
        )
        .setup(name="own_rpn", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [{"female": {"WT|WT": 100.0}, "male": {"WT|WT": 100.0}}] * 2
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
        .migration(adjacency=np.eye(2), migration_rate=0.0)
        .hooks(double_eggs, halve_k)
        .build()
    )
    population._initialize_session(seed=99)
    population.run(1, record_every=0)
    assert population.demes[0].params.eggs_per_female == 8.0
    assert population.demes[0].params.carrying_capacity == 5e11
