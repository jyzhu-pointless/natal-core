#!/usr/bin/env python3
"""Spatial container run/hook tests.

Migrated to the current architecture:

- the retired compiled-backend disable guard is a no-op context manager
  (the reference Python dispatch is the only non-Rust execution path);
- hook declarations live in the build chain (``.hooks(...)``); plans are
  compiled once at ``build()`` — single-parameter callbacks and
  ``@hook`` decorators carry event/priority/deme metadata;
- the njit-era ``njit_fn``/``py_wrapper`` descriptor payloads are gone —
  custom hooks are single-parameter ``TickContext`` callbacks;
- the 100-deme subprocess regression keeps the homogeneous-deme scale
  check without numba/prange.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest

import natal as nt
from natal.frontend.data import DiscretePopulationState, PopulationState
from natal.frontend.genetics import Species
from natal.frontend.hooks import Op
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.spatial.population import SpatialPopulation


def _make_species(prefix: str = "SpatialRunSpecies") -> Species:
    """Build a two-allele single-locus species."""
    return Species.from_dict(
        prefix,
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


def _make_population_config(species: Species, name: str = "config_template") -> object:
    """Build a quiescent age-structured draft (state changes only via hooks)."""
    return (
        nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 100.0, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, 100.0, 0.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 1.0, 0.0],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0, 0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0, 0.0, 0.0],
            eggs_per_female=0.0,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_new_adult_females=100,
        )
        .build()
        .export_config()
    )


def _make_discrete_population_config(species: Species, name: str) -> object:
    """Build a quiescent discrete-generation draft."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
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
        .export_config()
    )


class _RunDemePopulation:
    """Lightweight deme double: state changes only through run_tick."""

    def __init__(
        self,
        species: Species,
        name: str,
        config: object,
    ) -> None:
        self._species = species
        self._name = name
        self._tick = 0
        self._finished = False
        from natal._engine_rs import ParameterLog

        self._params_log = ParameterLog()
        self._config = config
        self.config = config
        self.finish_events = 0
        self._state = PopulationState(
            n_tick=0,  # restored
            individual_count=np.zeros(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
                (2, config.n_ages, config.n_ztypes), dtype=np.float64
            ),
            sperm_storage=np.zeros(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
                (config.n_ages, config.n_ztypes, config.n_ztypes), dtype=np.float64
            ),
        )

    @property
    def tick(self) -> int:
        """Read the native tick metadata mirrored by the spatial wrapper."""
        return self._tick

    @property
    def species(self) -> Species:
        """Species: shared by all demes of one container."""
        return self._species

    @property
    def state(self) -> PopulationState:
        """PopulationState: snapshot face (unused by the orchestration)."""
        return self._state

    def _live_state(self) -> PopulationState:
        """Live container — the spatial write-back channels call this.

        Duck-type adapter: since R5 the production populations expose the
        live container through this accessor while ``state`` snapshots.
        """
        return self._state

    def export_config(self) -> object:
        """Return the shared draft."""
        return self._config

    def has_python_callbacks(self) -> bool:
        """The double carries no Python callbacks."""
        return False

    def clear_history(self) -> None:
        """No-op: doubles have no history."""

    def run_tick(self) -> _RunDemePopulation:
        """Advance one fake tick."""
        self._tick += 1
        return self

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Record finish events, one per call."""
        _ = deme_id
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def reset(self) -> None:
        """Return the fake deme to tick 0."""
        self._tick = 0

    def get_total_count(self) -> int:
        """Return the summed individual count."""
        return int(self._state.individual_count.sum())

    def get_female_count(self) -> int:
        """Return the summed female count."""
        return int(self._state.individual_count[0].sum())

    def get_male_count(self) -> int:
        """Return the summed male count."""
        return int(self._state.individual_count[1].sum())


class _RunDiscreteDemePopulation:
    """Lightweight discrete-generation deme double."""

    def __init__(self, species: Species, name: str, config: object) -> None:
        self._species = species
        self._name = name
        self._tick = 0
        self._finished = False
        from natal._engine_rs import ParameterLog

        self._params_log = ParameterLog()
        self._config = config
        self.config = config
        self.finish_events = 0
        self._state = DiscretePopulationState(
            n_tick=0,
            individual_count=np.zeros(  # restored  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
                (2, config.n_ages, config.n_ztypes),
                dtype=np.float64,
            ),
        )

    @property
    def tick(self) -> int:
        """Read the native tick metadata mirrored by the spatial wrapper."""
        return self._tick

    @property
    def species(self) -> Species:
        """Species: shared by all demes of one container."""
        return self._species

    @property
    def state(self) -> DiscretePopulationState:
        """DiscretePopulationState: snapshot face (unused by orchestration)."""
        return self._state

    def _live_state(self) -> DiscretePopulationState:
        """Live container — the spatial write-back channels call this."""
        return self._state

    def export_config(self) -> object:
        """Return the shared draft."""
        return self._config

    def has_python_callbacks(self) -> bool:
        """The double carries no Python callbacks."""
        return False

    def clear_history(self) -> None:
        """No-op: doubles have no history."""

    def run_tick(self) -> _RunDiscreteDemePopulation:
        """Advance one fake tick."""
        self._tick += 1
        return self

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Record finish events, one per call."""
        _ = deme_id
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def reset(self) -> None:
        """Return the fake deme to tick 0."""
        self._tick = 0

    def get_total_count(self) -> int:
        """Return the summed individual count."""
        return int(self._state.individual_count.sum())

    def get_female_count(self) -> int:
        """Return the summed female count."""
        return int(self._state.individual_count[0].sum())

    def get_male_count(self) -> int:
        """Return the summed male count."""
        return int(self._state.individual_count[1].sum())


def test_spatial_population_run_tick_updates_all_demes():
    """run_tick runs every deme's lifecycle through the shared session.

    The quiescent draft keeps full survival through age 3 and no
    reproduction, so one session tick moves the seeded age-1 cohort
    (100 per sex, WT|WT) to age 2 in place in every deme, and both
    identical demes stay perfectly in step.
    """
    species = _make_species("spatial_run_tick")
    shared_config = _make_population_config(species)

    d0 = _RunDemePopulation(species, "d0", shared_config)
    d1 = _RunDemePopulation(species, "d1", shared_config)
    for deme in (d0, d1):
        deme._state.individual_count[:, 1, 0] = 100.0  # noqa: SLF001 — seed the double's live state

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)
    sp._initialize_session(seed=0)

    sp.run_tick()

    assert sp.tick == 1
    assert d0.tick == 1 and d1.tick == 1
    assert d0.tick == d1.tick == sp.tick
    for deme in sp.demes:
        counts = deme.state.individual_count
        assert counts[:, 1, 0].sum() == 0.0, "age-1 cohort must vacate"
        assert counts[0, 2, 0] == 100.0, "female cohort ages 1 -> 2"
        assert counts[1, 2, 0] == 100.0, "male cohort ages 1 -> 2"
        assert counts.sum() == 200.0, "quiescent draft conserves counts"
    np.testing.assert_array_equal(
        d0.state.individual_count, d1.state.individual_count
    )


def test_spatial_population_run_stop_marks_finish():
    """A stopped tick halts the run, finishes every deme, fires finish."""
    from natal.frontend.hooks.tick_context import TickContext

    species = _make_species("spatial_run_stop")
    finish_events: list[int] = []

    @nt.hook(event="first", deme=0)
    def stop_on_deme_zero(pop: TickContext) -> int:
        """Deme 0 stops the very first tick."""
        return pop.stop()

    @nt.hook(event="finish")
    def record_finish(pop: TickContext) -> int:
        finish_events.append(int(pop.deme_id))
        return 0

    demes = [
        _build_test_deme(
            f"stop_mark_d{i}",
            species,
            hook_calls=[((record_finish,), {})]
            + ([((stop_on_deme_zero,), {})] if i == 0 else []),
        )
        for i in range(2)
    ]

    sp = SpatialPopulation(demes, migration_rate=0.0)
    sp._initialize_session(seed=0)

    sp.run(n_steps=5, record_every=1)

    # The stopped tick does not advance; every deme is finished and the
    # container's mark-all pass fired each deme's finish exactly once.
    assert sp.tick == 0
    assert all(deme.is_finished for deme in demes)
    assert finish_events == [0, 1]


def test_spatial_stop_path_finish_hooks_see_own_deme_ids() -> None:
    """Finish hooks observe the firing deme's own index on the stop path.

    Regression guard: ``_mark_all_demes_stopped`` used to trigger finish
    without a deme id, so every stop-path finish hook saw the default
    instead of its deme index.  The stopping deme's own lifecycle finish
    (fired once with its live id) is pre-existing behavior pinned here.
    """
    species = _make_species("spatial_stop_finish_ids")
    finish_ids: list[int] = []

    @nt.hook(event="first", deme=1)
    def stop_on_deme_one(pop: TickContext) -> int:
        """Deme 1 requests the stop; every deme's finish then fires."""
        return pop.stop()

    @nt.hook(event="finish")
    def record_finish(pop: TickContext) -> int:
        finish_ids.append(int(pop.deme_id))
        return 0

    demes = [
        _build_test_deme(
            f"stop_id_d{i}",
            species,
            hook_calls=[((record_finish,), {})]
            + ([((stop_on_deme_one,), {})] if i == 1 else []),
        )
        for i in range(3)
    ]

    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.run(n_steps=5)

    # The engine freezes the tick at the stop boundary; the container's
    # mark-all pass then fires every deme's finish in list order with
    # each deme's own index.
    assert finish_ids == [0, 1, 2]
    assert all(deme.is_finished for deme in demes)


def test_spatial_population_stochastic_discrete_migration_preserves_integer_counts():
    """Stochastic discrete ticks keep every per-deme count integral.

    A discrete-generation tick replaces the adults with the offspring
    generation, so total conservation across a tick is not the invariant;
    the engine's multinomial sampling must still land on whole
    individuals everywhere (no fractional leakage).
    """
    species = _make_species("spatial_run_stochastic_discrete")
    shared_config = _make_discrete_population_config(species, "stoch_disc")  # restored
    shared_config = shared_config._replace(
        stochastic=True,
        eggs_per_female=np.array(4.0),
        age_based_mating_rates=np.full((2, 2), 1.0),
        age_based_reproduction_rates=np.array([0.0, 1.0]),  # type: ignore[list-item]  # duck-typed double draft
    )  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    d0 = _RunDiscreteDemePopulation(species, "d0", shared_config)
    d1 = _RunDiscreteDemePopulation(species, "d1", shared_config)  # restored
    width = shared_config.n_ztypes  # type: ignore[attr-defined]  # duck-typed double
    seed_counts = np.zeros((2, 2, width), dtype=np.float64)
    seed_counts[0, 1, 0] = 3.0  # females, adults
    seed_counts[1, 1, 0] = 2.0  # males, adults
    d0._state = d0.state._replace(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        individual_count=seed_counts
    )

    sp = SpatialPopulation(
        [d0, d1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )
    sp._initialize_session(seed=0)
    sp.run_tick()

    total_counts = [float(deme.state.individual_count.sum()) for deme in sp.demes]
    assert sum(total_counts) > 0.0
    # Deme 1 held no breeding adults, so any mass there arrived by
    # migration of the new adult generation.
    assert total_counts[1] > 0.0
    for deme in sp.demes:
        assert np.allclose(
            deme.state.individual_count, np.round(deme.state.individual_count)
        )


def test_spatial_population_stochastic_age_migration_preserves_sperm_consistency():
    """Stochastic age-structured migration conserves sex/sperm mass."""
    species = _make_species("spatial_run_stochastic_age")
    shared_config = _make_population_config(species)  # restored
    shared_config = shared_config._replace(stochastic=True, continuous_sampling=False)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    d0 = _RunDemePopulation(species, "d0", shared_config)
    d1 = _RunDemePopulation(species, "d1", shared_config)  # restored
    width = shared_config.n_ztypes  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    ind = np.zeros((2, shared_config.n_ages, width), dtype=np.float64)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    ind[0, 1, 0] = 5.0
    ind[1, 1, 0] = 4.0  # restored
    sperm = np.zeros(
        (shared_config.n_ages, width, width),  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        dtype=np.float64,
    )
    sperm[1, 0, 0] = 3.0  # restored
    d0._state = PopulationState(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        n_tick=0, individual_count=ind, sperm_storage=sperm
    )

    np.random.seed(23)

    sp = SpatialPopulation(
        [d0, d1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )
    sp._initialize_session(seed=0)
    sp.run_tick()

    total_females = sum(
        float(deme.state.individual_count[0].sum()) for deme in sp.demes
    )
    total_males = sum(float(deme.state.individual_count[1].sum()) for deme in sp.demes)
    total_sperm = sum(float(deme.state.sperm_storage.sum()) for deme in sp.demes)
    assert np.isclose(total_females, 5.0)
    assert np.isclose(total_males, 4.0)
    assert np.isclose(total_sperm, 3.0)

    for deme in sp.demes:
        female_total = float(deme.state.individual_count[0, :, :].sum())
        sperm_total = float(deme.state.sperm_storage[:, :, :].sum())
        assert female_total >= sperm_total
        assert np.allclose(
            deme.state.individual_count, np.round(deme.state.individual_count)
        )
        assert np.allclose(deme.state.sperm_storage, np.round(deme.state.sperm_storage))


# ---------------------------------------------------------------------------
# Per-deme hook dispatch semantics
# ---------------------------------------------------------------------------


def _build_test_deme(
    name: str,
    species: nt.Species,
    hook_calls: list | None = None,
) -> nt.DiscreteGenerationPopulation:
    """Build an independent quiescent discrete deme (own hook storage).

    Args:
        name: Population name.
        species: Genetic architecture.
        hook_calls: Optional ``(items, kwargs)`` pairs declared through
            ``.hooks()`` in this deme's build chain.
    """
    chain = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [10.0, 0.0]},
                "male": {"WT|WT": [10.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    )
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items, **kwargs)
    return chain.build()


def test_spatial_hook_priority_runs_in_run_tick_and_run() -> None:
    """Hooks fire in priority order, per deme, for run_tick and run."""
    calls: list[str] = []

    @nt.hook(event="first", priority=0)
    def first_a(pop: object) -> int:
        """Priority-0 callback: record and bump age-0 females."""
        _ = pop
        calls.append("a")
        return 0

    @nt.hook(event="first", priority=1)
    def first_b(pop: object) -> int:
        """Priority-1 callback: record and bump age-0 females."""
        _ = pop
        calls.append("b")
        return 0

    sp = SpatialPopulation(
        [
            _build_test_deme(
                "prio_d0",
                _make_species("spatial_prio"),
                hook_calls=[((first_a, first_b), {})],
            )
        ],
        migration_rate=0.0,
    )
    sp._initialize_session(seed=0)

    sp.run_tick()
    sp.run(n_steps=1)

    assert calls == ["a", "b", "a", "b"]


def test_spatial_mixed_priority_is_local_per_deme() -> None:
    """Hooks registered on one deme never fire on another deme."""
    species = _make_species("spatial_local_priority_per_deme")
    calls: list[str] = []

    @nt.hook(event="first", priority=0)
    def d0_hook(pop: object) -> int:
        """Deme-0-only callback: +2 to age-0 females."""
        calls.append("d0")  # restored
        pop.state.individual_count[0, 0, 0] += 2.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    @nt.hook(event="first", priority=0)
    def d1_hook(pop: object) -> int:
        """Deme-1-only callback: +4 to age-0 males."""
        calls.append("d1")  # restored
        pop.state.individual_count[1, 0, 0] += 4.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    d0 = _build_test_deme("local_d0", species, hook_calls=[((d0_hook,), {})])
    d1 = _build_test_deme("local_d1", species, hook_calls=[((d1_hook,), {})])

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.run_tick()

    assert calls == ["d0", "d1"]
    assert float(spatial.deme(0).state.individual_count.sum()) == 22.0
    assert float(spatial.deme(1).state.individual_count.sum()) == 24.0


def test_spatial_reference_run_hooks_see_live_deme_ids() -> None:
    """During a reference spatial run every deme's hook sees its own index.

    Regression guard for the ``deme_id=-1`` sentinel that used to leak
    from the reference lifecycle: hooks read ``pop.deme_id`` through the
    TickContext and must observe the live deme index, never -1.
    """
    species = _make_species("spatial_deme_id_value")
    seen: list[int] = []

    @nt.hook(event="first", priority=0)
    def record_deme(pop: TickContext) -> int:
        """Record the executing deme id exactly as reported."""
        seen.append(int(pop.deme_id))
        return 0

    demes = [
        _build_test_deme(f"deme_id_d{i}", species, hook_calls=[((record_deme,), {})])
        for i in range(3)
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.run(n_steps=2)

    # Each of the 3 demes fires its first-event hook once per tick.
    assert sorted(seen) == [0, 0, 1, 1, 2, 2]


def test_spatial_reference_deme_selector_targets_one_deme() -> None:
    """A ``deme=1`` hook fires only on deme 1 during a reference run.

    Deme-selector filtering consumes the same ``deme_id`` value; before
    the fix the sentinel -1 matched no selector, so deme-targeted hooks
    silently never ran on the reference backend.
    """
    species = _make_species("spatial_deme_id_selector")
    hits: list[int] = []

    @nt.hook(event="first", priority=0, deme=1)
    def only_deme_one(pop: TickContext) -> int:
        """Record the deme ids where this selector-matched hook fires."""
        hits.append(int(pop.deme_id))
        return 0

    demes = [
        _build_test_deme(f"sel_deme_d{i}", species, hook_calls=[((only_deme_one,), {})])
        for i in range(3)
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.run(n_steps=1)

    assert hits == [1]


def test_spatial_compiled_local_hooks_still_take_effect() -> None:
    """A local stop hook halts the first tick and blocks further runs."""
    species = _make_species("spatial_compiled_local_hook_effect")
    d1 = _build_test_deme("csr_local_d1", species)

    @nt.hook(event="first", priority=0)
    def stop_immediately(pop: object) -> int:
        """Request termination on the first event."""  # restored
        pop.stop()  # type: ignore[attr-defined]  # duck-typed double call: shapes verified by assertions below
        return 0

    d0 = _build_test_deme("csr_local_d0", species, hook_calls=[((stop_immediately,), {})])
    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial.run_tick()

    assert d0.is_finished and d1.is_finished
    with pytest.raises(RuntimeError):
        spatial.run_tick()


# ========================================================================
# Compact spatial hook plan tests (issue #37)
# ========================================================================


def _build_quiescent_age_pop(
    species: nt.Species,
    n_demes: int,
    name: str = "quiescent",
    hook_calls: list | None = None,
) -> SpatialPopulation:
    """Build a homogeneous quiescent age-structured population.

    Args:
        species: Genetic architecture.
        n_demes: Deme count.
        name: Population name.
        hook_calls: Optional ``(items, kwargs)`` pairs declared through
            ``.hooks()`` in the spatial build chain.
    """
    chain = (
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, pop_type="age_structured"
        )
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 100.0, 0.0]},
                "male": {"WT|WT": [0.0, 100.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 1.0, 0.0],
            male_age_based_survival=[1.0, 1.0, 0.0],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0, 0.0],
            eggs_per_female=0.0,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_new_adult_females=100,
        )
    )
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items)
    return chain.build()


def _build_discrete_pop(
    species: nt.Species,
    n_demes: int,
    name: str = "discrete",
    hook_calls: list | None = None,
) -> SpatialPopulation:
    """Build a homogeneous discrete-generation population via builder."""
    chain = (
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 100},
                "male": {"WT|WT": 100},
            }
        )
        .reproduction(eggs_per_female=10.0)
        .competition(
            carrying_capacity=500,
            low_density_growth_rate=6.0,
            juvenile_growth_mode="beverton_holt",
        )
    )
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items)
    return chain.build()


# -----------------------------------------------------------------------
# Compact plan structure tests
# -----------------------------------------------------------------------
def test_compact_plan_folds_identical_sequences_to_wildcard() -> None:
    """All demes sharing one descriptor sequence → single wildcard slot."""
    species = _make_species("compact_wildcard")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-0 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    spatial = _build_quiescent_age_pop(
        species, n_demes=3, name="compact_wildcard_build", hook_calls=[((my_hook,), {})]
    )

    expanded = spatial._collect_effective_compiled_hooks()
    compact = spatial._collect_compact_spatial_hooks()

    assert len(expanded) == 3
    assert {int(desc.deme_selector) for desc in expanded} == {0, 1, 2}

    assert len(compact) == 1
    assert compact[0].deme_selector == "*"
    assert compact[0].callback is not None


def test_compact_plan_preserves_expanded_view() -> None:
    """Public get_compiled_hooks still returns per-deme pinned descriptors."""
    species = _make_species("compact_expanded_view")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    spatial = _build_quiescent_age_pop(
        species, n_demes=2, name="compact_expanded_view_build", hook_calls=[((my_hook,), {})]
    )

    public = spatial.get_compiled_hooks()
    assert len(public) == 2
    assert {int(desc.deme_selector) for desc in public} == {0, 1}


def test_compact_plan_subset_selector() -> None:
    """Descriptor with subset selector stays as tuple, not wildcard."""
    species = _make_species("compact_subset")

    @nt.hook(event="first", priority=0, deme=(0, 1))
    def demes_0_1_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    spatial = _build_quiescent_age_pop(
        species,
        n_demes=3,
        name="compact_subset_build",
        hook_calls=[((demes_0_1_hook,), {})],
    )

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 1
    sel = compact[0].deme_selector
    assert isinstance(sel, tuple)
    assert set(sel) == {0, 1}


def test_compact_plan_different_order_not_merged() -> None:
    """[mul2, add1] vs [add1, mul2]: different order, different results.

    mul2 (×2) and add1 (+1) are non-commutative:
      [mul2, add1] on 100 → 301 total;  [add1, mul2] → 302 total.
    Deme 1 reuses deme 0's descriptors in reversed order, so the compact
    plan must produce two separate groups based on order alone.
    """
    species = _make_species("compact_order")

    @nt.hook(event="first", priority=0)
    def mul2(pop: object) -> int:
        """Double age-1 females."""  # restored
        pop.state.individual_count[0, 1, 0] *= 2.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    @nt.hook(event="first", priority=0)
    def add1(pop: object) -> int:
        """Add one age-1 female."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    def build_ordered_deme(name: str, hooks_in_order: list) -> nt.DiscreteGenerationPopulation:
        """Build a quiescent age deme with one ordered declaration pair."""
        chain = (
            nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 100.0, 0.0]},
                    "male": {"WT|WT": [0.0, 100.0, 0.0]},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 1.0, 0.0],
                male_age_based_survival=[1.0, 1.0, 0.0],
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 0.0, 0.0],
                male_age_based_mating_rate=[0.0, 0.0, 0.0],
                eggs_per_female=0.0,
            )
            .competition(
                juvenile_growth_mode="logistic",
                expected_num_new_adult_females=100,
            )
        )
        for hook in hooks_in_order:
            chain = chain.hooks(hook)
        return chain.build()

    sp = SpatialPopulation(
        [
            build_ordered_deme("compact_order_d0", [mul2, add1]),
            build_ordered_deme("compact_order_d1", [add1, mul2]),
        ],
        migration_rate=0.0,
    )
    sp._initialize_session(seed=0)

    compact = sp._collect_compact_spatial_hooks()
    selectors = {d.deme_selector for d in compact}
    assert 0 in selectors and 1 in selectors
    assert len(compact) == 4

    sp.run_tick()

    total0 = float(sp.deme(0).state.individual_count.sum())
    total1 = float(sp.deme(1).state.individual_count.sum())
    assert total0 == 301.0, (
        f"deme[0] mul2→add1: female 100×2+1=201, +100 male = 301, got {total0}"
    )
    assert total1 == 302.0, (
        f"deme[1] add1→mul2: female (100+1)×2=202, +100 male = 302, got {total1}"
    )


def test_compact_plan_empty_hook_sequence_skipped() -> None:
    """Deme with no compiled hooks contributes no descriptors to compact plan."""
    species = _make_species("compact_empty")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    d0 = _build_test_deme("ce_d0", species, hook_calls=[((my_hook,), {})])
    d1 = _build_test_deme("ce_d1", species)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial._initialize_session(seed=0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == 0


# -----------------------------------------------------------------------
# Build-time declaration contracts (former shared-storage tests)
# -----------------------------------------------------------------------
def test_set_hook_duplicate_declaration_compiles_once() -> None:
    """Declaring the same hook object twice yields one descriptor.

    Formerly the identity-idempotent registration dedupe; the same
    (source, event) identity now dedupes at build-time compilation.
    """
    species = _make_species("set_hook_shared")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species,
        n_demes=3,
        name="set_hook_shared_build",
        hook_calls=[((my_hook,), {}), ((my_hook,), {})],
    )
    assert len(sp._deme_object(0).compiled_hook_descriptors) == 1  # pyright: ignore[reportPrivateUsage]  # deme slot-level plan (not on the aligned slice surface)

    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"


def test_container_register_hooks_is_deleted() -> None:
    """Post-build registration no longer exists at the container level."""
    species = _make_species("set_hook_cow")
    sp = _build_quiescent_age_pop(species, n_demes=3)

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Would-be late registration body (never installed)."""
        _ = pop
        return 0

    with pytest.raises(AttributeError):
        sp.register_hooks(my_hook)  # type: ignore[attr-defined]  # negative contract: deleted surface


def test_set_hook_shared_storage_subset_cow_execution() -> None:
    """A deme-targeted declaration fires only on its selected deme."""
    species = _make_species("cow_exec")
    n_demes = 5
    target = 2

    @nt.hook(event="first", priority=0, deme=target)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species, n_demes, name="cow_exec_build", hook_calls=[((add_one,), {})]
    )

    sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_subset_callback_hook_no_leak() -> None:
    """A deme-targeted callback never leaks to other demes."""
    species = _make_species("no_leak")
    n_demes = 5
    target = 2

    @nt.hook(event="first", priority=0, deme=target)
    def py_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species, n_demes, name="no_leak_build", hook_calls=[((py_hook,), {})]
    )
    sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_empty_selector_noop() -> None:
    """An empty selector matches no deme, so the hook never fires."""
    species = _make_species("empty_sel")

    @nt.hook(event="first", priority=0, deme=())
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species, n_demes=3, name="empty_sel_build", hook_calls=[((add_one,), {})]
    )
    sp.run_tick()

    for i in range(3):
        assert float(sp.deme(i).state.individual_count.sum()) == 200.0


def test_set_hook_cow_subsequent_mutation_no_leak() -> None:
    """Targeted plus wildcard declarations stay isolated per deme.

    Formerly a copy-on-write storage isolation check; the same execution
    contract is expressed with deme-targeted decorator metadata compiled
    at build: hook_a fires on deme 0 only, hook_b on every deme.
    """
    species = _make_species("cow_iso")

    @nt.hook(event="first", priority=0, deme=0)
    def hook_a(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    @nt.hook(event="first", priority=0)
    def hook_b(pop: object) -> int:
        """Bump age-1 males on every fire."""  # restored
        pop.state.individual_count[1, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species,
        n_demes=3,
        name="cow_iso_build",
        hook_calls=[((hook_a, hook_b), {})],
    )

    sp.run_tick()

    for i in range(3):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 202.0 if i == 0 else 201.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


# -----------------------------------------------------------------------
# Run-tick integration tests
# -----------------------------------------------------------------------
def test_compact_plan_run_tick_deterministic_state() -> None:
    """Quiescent model: hook +1 on age-1 female → total per deme = 200 + 1."""
    species = _make_species("compact_det")

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species, n_demes=3, name="compact_det_build", hook_calls=[((add_one,), {})]
    )

    sp.run_tick()

    for i in range(3):
        total = float(sp.deme(i).state.individual_count.sum())
        assert total == 201.0, f"deme[{i}]: expected 201, got {total}"
        state = sp.deme(i).state.individual_count
        assert np.all(state >= 0.0)
        assert not np.any(np.isnan(state))


def test_compact_plan_csr_then_callback_ordering() -> None:
    """Within one event the priority order interleaves both hook kinds.

    callback(pri=0) +1 then CSR(pri=1) ×2 on age-1 female=100 → 202
    (the pre-interleaving plan-first order would give 201).
    """
    species = _make_species("compact_mixed_exact")

    @nt.hook(event="early", priority=0)
    def add1(pop: object) -> int:
        """Add one age-1 female."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    @nt.hook(event="early", priority=1)
    def mul2_csr() -> list[object]:
        """Scale age-1 females by two."""
        return [Op.scale(genotypes="WT|WT", ages=1, sex="female", factor=2.0)]

    sp = _build_quiescent_age_pop(
        species,
        n_demes=2,
        name="compact_mixed_exact_build",
        hook_calls=[((mul2_csr,), {}), ((add1,), {})],
    )

    sp.run_tick()

    # Callback runs first: female[age=1] 100+1=101, then CSR ×2 → 202.
    # Survival keeps age 1 (rate 1.0); aging moves the result into age 2.
    for i in range(2):
        assert float(sp.deme(i).state.individual_count[0, 2, 0]) == 202.0


def test_compact_plan_deme_targeted_callback_keeps_wildcard_slots_aligned() -> None:
    """A deme-targeted callback must not shift other demes' callback slots.

    Regression guard for the cross-type slot wiring: the compact slot column
    numbers callbacks inside the reference deme's selector-filtered
    sequence, while the cross-deme bridges index each deme's *unfiltered*
    callback list. When a targeted callback (here ``deme=1``, priority 0)
    sorts before a wildcard callback (priority 5), the wildcard slot of the
    excluded deme dispatched the targeted callback instead — skipped by the
    selector — and the wildcard callback silently never ran on that deme.
    """
    species = _make_species("compact_targeted_shift")
    calls: list[tuple[str, int]] = []

    @nt.hook(event="first", priority=0, deme=1)
    def targeted(pop: TickContext) -> int:
        """Fires only on deme 1 and sorts before the wildcard callback."""
        calls.append(("A", int(pop.deme_id)))
        return 0

    @nt.hook(event="first", priority=5)
    def wildcard(pop: TickContext) -> int:
        """Fires on every deme after the targeted callback on deme 1."""
        calls.append(("B", int(pop.deme_id)))
        return 0

    sp = _build_quiescent_age_pop(
        species,
        n_demes=2,
        name="compact_targeted_shift_build",
        hook_calls=[((targeted,), {}), ((wildcard,), {})],
    )

    sp.run_tick()

    # Deme 0 runs the wildcard only (the targeted selector excludes it);
    # deme 1 runs the targeted callback first (priority 0 < 5), then the
    # wildcard. The failing wiring dropped ("B", 0) entirely.
    assert calls == [("B", 0), ("A", 1), ("B", 1)]


def test_compact_plan_heterogeneous_groups_identity_mapped_bridges() -> None:
    """Heterogeneous compact groups map callback slots by identity.

    Copy-on-write splits deme 2 onto its own hook storage, so the compact
    plan carries three groups (0 | 1 | 2). Every group's wildcard callback
    is the same descriptor object, and the group-2-only callback exists in
    no other deme's runner — the bridges must translate each slot by
    callback identity (per-deme runner index), never by positional
    coincidence across groups.
    """
    species = _make_species("compact_hetero_bridges")
    calls: list[tuple[str, int]] = []

    @nt.hook(event="first", priority=0)
    def everywhere(pop: TickContext) -> int:
        """Wildcard callback registered before the deme-1-targeted one."""
        calls.append(("AB", int(pop.deme_id)))
        return 0

    @nt.hook(event="first", priority=1, deme=2)
    def deme2_only(pop: TickContext) -> int:
        """Deme-2-targeted declaration (priority 1)."""
        calls.append(("C", int(pop.deme_id)))
        return 0

    @nt.hook(event="first", priority=0, deme=1)
    def target_deme_one(pop: TickContext) -> int:
        """Ties with ``everywhere`` at priority 0; registration order wins."""
        calls.append(("T1", int(pop.deme_id)))
        return 0

    sp = _build_quiescent_age_pop(
        species,
        n_demes=3,
        name="compact_hetero_bridges_build",
        hook_calls=[((everywhere, target_deme_one, deme2_only), {})],
    )

    sp.run_tick()

    # Deme 0: everywhere only. Deme 1: everywhere then T1 (priority-0 tie,
    # registration order). Deme 2: everywhere then C.
    assert calls == [
        ("AB", 0),
        ("AB", 1),
        ("T1", 1),
        ("AB", 2),
        ("C", 2),
    ]


def test_builder_homogeneous_demes_share_compiled_hooks() -> None:
    """Builder-created homogeneous population: all demes share hook storage."""
    species = _make_species("builder_share")
    n_demes = 5

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_discrete_pop(
        species, n_demes, hook_calls=[((add_one,), {})]
    )
    # Homogeneous clones share one descriptor tuple by identity.
    ref = sp._deme_object(0).compiled_hook_descriptors  # pyright: ignore[reportPrivateUsage]  # deme slot-level plan (not on the aligned slice surface)
    for i in range(1, n_demes):
        assert sp._deme_object(i).compiled_hook_descriptors is ref  # pyright: ignore[reportPrivateUsage]  # deme slot-level plan (not on the aligned slice surface)

    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"


def test_builder_set_hook_subset_cow_combined() -> None:
    """Subset registration via builder-created population: COW isolates target."""
    species = _make_species("builder_cow")
    n_demes = 5
    target = 2
    @nt.hook(event="first", priority=0, deme=target)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(
        species, n_demes, name="builder_cow_build", hook_calls=[((add_one,), {})]
    )

    sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


# -----------------------------------------------------------------------
# 100-deme subprocess regression
# -----------------------------------------------------------------------
_ONE_HUNDRED_DEMES_TEST_CODE = """
import numpy as np
import natal as nt

sp_species = nt.Species.from_dict(
    "An100Deme", structure={"chr1": {"A": ["WT", "Drive"]}},
    gamete_labels=["default"], unordered=False,
)

S_IDX = 0
N_DEMES = 100


@nt.hook(event="early")
def infect_susceptible_females(pop):
    females = pop.state.individual_count[0]
    sperm = pop.state.sperm_storage
    for age in range(1, pop.blueprint.n_ages):
        n_s = int(round(females[age, S_IDX]))
        n_mated = 0
        for mi in range(pop.blueprint.n_ztypes):
            n_mated += int(round(sperm[age, S_IDX, mi]))
        n_virgins = n_s - n_mated
        if n_virgins < 0:
            return 1
        n_moved = int(pop.rng.integers(0, n_virgins + 1))
        n_moved_sperm = 0
        for mi in range(pop.blueprint.n_ztypes):
            nb = int(round(sperm[age, S_IDX, mi]))
            nbm = int(pop.rng.integers(0, nb + 1))
            sperm[age, S_IDX, mi] -= nbm
            sperm[age, 2, mi] += nbm
            n_moved_sperm += nbm
        females[age, S_IDX] -= n_moved
        females[age, 2] += n_moved
    return 0


builder = (
    nt.SpatialPopulation.builder(sp_species, n_demes=N_DEMES)
    .setup(stochastic=True)
    .age_structure(n_ages=8, new_adult_age=1)
    .initial_state({
        "female": {"WT|WT": [0, 500, 500, 0, 0, 0, 0, 0]},
        "male":   {"WT|WT": [0, 500, 500, 0, 0, 0, 0, 0]},
    })
    .survival(female_age_based_survival=[1]*8,
              male_age_based_survival=[1]*8)
    .reproduction(
        eggs_per_female=0,
        female_age_based_mating_rate=[0, 0, 0, 0, 0, 0, 0, 0],
        male_age_based_mating_rate=[0, 0, 0, 0, 0, 0, 0, 0],
    )
    .competition(juvenile_growth_mode="logistic",
                 expected_num_new_adult_females=500)
    .migration(migration_rate=0)
)
builder.hooks(infect_susceptible_females)
pop = builder.build()

compact = pop._collect_compact_spatial_hooks()
assert len(compact) == 1, f"expected 1 compact slot, got {len(compact)}"
assert compact[0].deme_selector == "*", f"expected wildcard, got {compact[0].deme_selector}"

registry = pop.hooks
assert registry is not None
assert int(registry.n_hooks) == 1, f"expected 1 hook slot in registry, got {int(registry.n_hooks)}"

for _run_i in range(3):
    pop.run_tick()

    for d in range(N_DEMES):
        state = pop.demes[d].state.individual_count
        assert np.all(np.isfinite(state)), f"run {_run_i} deme[{d}] has non-finite values"
        assert np.all(state >= 0), f"run {_run_i} deme[{d}] has negative counts"
        initial_total = 2000
        final_total = state.sum()
        assert abs(float(final_total) - initial_total) < 1e-9, (
            f"run {_run_i} deme[{d}] total changed: {final_total} != {initial_total}"
        )
"""


def test_homogeneous_100_deme_subprocess_no_crash() -> None:
    """100-deme homogeneous population with a custom hook runs cleanly."""
    result = subprocess.run(
        [sys.executable, "-c", _ONE_HUNDRED_DEMES_TEST_CODE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess returned {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_spatial_builder_custom_slots_reach_all_demes() -> None:
    """``.custom()`` on the spatial chain carries slots to every deme.

    Homogeneous path: the template draft carries the slots and clones
    share it.  Heterogeneous path: every group template replays the
    same kwargs, so all groups carry them too.
    """
    species = _make_species("spatial_custom_slots")

    homogeneous = (
        SpatialPopulation.builder(species, n_demes=3)
        .setup(stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [100.0, 0.0]},
                "male": {"WT|WT": [100.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .custom(x=1, tag=2.5)
        .build()
    )
    for i, d in enumerate(homogeneous.demes):
        assert d.config.custom["x"] == 1, i
        assert d.config.custom["tag"] == 2.5, i

    heterogeneous = (
        SpatialPopulation.builder(species, n_demes=4)
        .setup(stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [100.0, 0.0]},
                "male": {"WT|WT": [100.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(
            carrying_capacity=nt.batch_setting([1000.0, 1000.0, 2000.0, 2000.0])
        )
        .custom(x=7)
        .build()
    )
    for i, d in enumerate(heterogeneous.demes):
        assert d.config.custom["x"] == 7, i


# -----------------------------------------------------------------------
# Container run window: cross-deme reads degrade to last-published values
# -----------------------------------------------------------------------
def test_cross_deme_reads_inside_a_run_window_degrade_safely() -> None:
    """A hook in deme 0 reading deme 1 never touches the borrowed session.

    During a container run the native session is mutably borrowed; the
    run window is published on every managed deme, so cross-deme
    lifecycle/state/count reads resolve to the last-published values
    (pre-run: the build state) instead of raising ``Already mutably
    borrowed``.  After the run ends, the same reads are fresh.
    """
    species = _make_species("run_window_reads")
    holder: dict[str, SpatialPopulation] = {}
    observed: list[dict[str, object]] = []

    @nt.hook(event="first", priority=0, deme=0)
    def cross_deme_reader(pop: object) -> int:
        """Read sibling-deme values through the container inside the window."""
        container = holder["run_window_reads"]
        observed.append(
            {
                "total": container.deme(1).get_total_count(),
                "state_tick": container.deme(1).state.n_tick,
                "slot_tick": container._deme_object(1).tick,  # pyright: ignore[reportPrivateUsage]  # lifecycle projection lives on the slot
                "slot_finished": container._deme_object(1).is_finished,  # pyright: ignore[reportPrivateUsage]
            }
        )
        return 0

    pop = _build_quiescent_age_pop(
        species, n_demes=2, name="run_window_reads", hook_calls=[((cross_deme_reader,), {})]
    )
    holder["run_window_reads"] = pop

    initial_total = pop.deme(1).get_total_count()
    assert initial_total == 200.0
    pop.run(2, record_every=0)

    # Two firings (two ticks x one deme-0 slot): every in-window read saw
    # the last-published boundary, and no native borrow error surfaced.
    assert len(observed) == 2
    for snapshot in observed:
        assert snapshot["total"] == 200.0
        assert snapshot["state_tick"] == 0
        assert snapshot["slot_tick"] == 0
        assert snapshot["slot_finished"] is False
    # After the run the same reads are fresh and match the session state.
    assert pop.tick == 2
    assert pop._deme_object(1).tick == 2  # pyright: ignore[reportPrivateUsage]  # clock projects the session again
    _tick, ind_all, _sperm = pop._native_stacked_state()  # pyright: ignore[reportPrivateUsage]  # session truth after the run
    assert pop.deme(1).get_total_count() == float(ind_all[1].sum())


def test_native_deme_count_accessor_matches_stacked_state() -> None:
    """The additive session accessor agrees with the stacked state sums."""
    species = _make_species("native_counts_accessor")
    pop = _build_quiescent_age_pop(species, n_demes=3, name="native_counts_accessor")
    pop.run(1, record_every=0)
    backend = pop._rust_spatial_backend
    assert backend is not None
    _tick, ind_flat, _sperm_flat = backend.state_snapshot()
    n_demes = pop.n_demes
    draft = pop.deme(0).export_config()
    ind_all = ind_flat.reshape(n_demes, int(draft.n_sexes), int(draft.n_ages), int(draft.n_ztypes))
    for i in range(pop.n_demes):
        total, female, male = backend.counts(i)
        assert total == float(ind_all[i].sum())
        assert female == float(ind_all[i, 0].sum())
        assert male == float(ind_all[i, 1].sum())
        # The per-deme state projection carries the same plane.
        plane = pop.deme(i).state.individual_count
        deme_plane = backend.state_snapshot_deme(i)[1]
        assert deme_plane.tolist() == plane.ravel().tolist()
