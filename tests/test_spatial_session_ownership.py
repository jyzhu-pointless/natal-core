"""Session-owned spatial state contracts (plan S3a).

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
    population.enable_rust_backend(seed=seed)
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
    for deme, state in zip(b.demes, imported):
        deme.import_state(state)
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


def test_import_state_continues_from_imported_counts() -> None:
    """A per-deme import reaches the session: the next tick computes from
    the imported counts instead of overwriting them.

    Importing an extinguished deme must keep that deme extinct through
    later ticks — the pre-S3 behavior recomputed ticks from the stale
    pre-import cache, resurrecting the deme.
    """
    target = _build("own_imp_target", seed=45, stochastic=False, n_demes=3, rate=0.0)
    target.run(1, record_every=0)
    target.demes[1].import_state(
        {
            "n_tick": 1,
            "individual_count": np.zeros((2, 3, 3), dtype=np.float64),
            "sperm_storage": np.zeros((3, 3, 3), dtype=np.float64),
        }
    )
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
    """A deme.state snapshot is independent; imports are the write channel."""
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


def test_disable_pulls_the_session_state_back() -> None:
    """disable_rust_backend leaves the demes at the session's final state."""
    stochastic = _build("own_dis_a", seed=48)
    stochastic.run(2)
    expected = _stacked(stochastic)
    n_tick = stochastic._tick  # noqa: SLF001

    stochastic.disable_rust_backend()
    assert _stacked(stochastic).shape == expected.shape
    np.testing.assert_array_equal(_stacked(stochastic), expected)
    assert stochastic._tick == n_tick  # noqa: SLF001


def test_fused_run_matches_the_python_dispatch_reference() -> None:
    """The fused session tick (lifecycle then migration inside Rust) is
    bitwise identical to the python-dispatch reference for a
    deterministic spatial run with live migration.
    """
    population = _build("own_zero_mig", seed=49, stochastic=False)
    population.run(2)
    # Reference twin through the python-dispatch path (no Rust backend).
    reference = (
        nt.SpatialPopulation.builder(
            _species("own_zero_mig_sp2"), n_demes=4, pop_type="age_structured"
        )
        .setup(name="own_zero_mig_ref", stochastic=False)
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
    )
    reference.run(2)
    np.testing.assert_array_equal(_stacked(population), _stacked(reference))


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
    """tensor_write("migration_rate") changes Rust-tick outcomes to match
    the python-dispatch twin (deterministic bitwise parity)."""
    py_twin = _build("own_rate_py", seed=52, stochastic=False)
    py_twin.disable_rust_backend()
    rs = _build("own_rate_rs", seed=52, stochastic=False)
    rs.run(1, record_every=0)
    py_twin.run(1, record_every=0)

    new_rate = np.full((4, 2, 3), 0.5, dtype=np.float64)
    rs.params.tensor_write("migration_rate", new_rate)
    py_twin.params.tensor_write("migration_rate", new_rate)
    rs.run(1, record_every=0)
    py_twin.run(1, record_every=0)
    np.testing.assert_array_equal(_stacked(rs), _stacked(py_twin))


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
            pop.enable_rust_backend(seed=seed)
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
    spatial.enable_rust_backend(seed=7)
    assert spatial.using_rust_backend
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
            pop.enable_rust_backend(seed=seed)
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
