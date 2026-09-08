#!/usr/bin/env python3
"""Spatial container run/hook tests.

Migrated to the current architecture:

- the retired compiled-backend disable guard is a no-op context manager
  (the reference Python dispatch is the only non-Rust execution path);
- ``set_hook``/``hook_id``-based registration is replaced by
  ``register_hooks`` (single-parameter callbacks, ``@hook`` decorators);
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
        *,
        individual_delta: float = 0.0,
        sperm_delta: float = 0.0,
        stop_after_run_tick: bool = False,
    ) -> None:
        self._species = species
        self._name = name
        self.tick = 0
        self._finished = False
        self._config = config
        self.config = config
        self._individual_delta = float(individual_delta)
        self._sperm_delta = float(sperm_delta)
        self._stop_after_run_tick = bool(stop_after_run_tick)
        self.finish_events = 0
        self._state = PopulationState(
            n_tick=0,  # restored
            individual_count=np.zeros((2, config.n_ages, 1), dtype=np.float64),  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
            sperm_storage=np.zeros((config.n_ages, 1, 1), dtype=np.float64),  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        )

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

    def clear_history(self) -> None:
        """No-op: doubles have no history."""

    def run_tick(self) -> _RunDemePopulation:
        """Advance one fake tick, optionally stopping."""
        if self._individual_delta != 0.0:
            self._state = self._state._replace(
                individual_count=(
                    self._state.individual_count + self._individual_delta
                ),
                sperm_storage=self._state.sperm_storage + self._sperm_delta,
            )
        if self._stop_after_run_tick:
            self._finished = True
        self.tick += 1
        return self

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Record finish events, one per call."""
        _ = deme_id
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def reset(self) -> None:
        """Return the fake deme to tick 0."""
        self.tick = 0

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
        self.tick = 0
        self._finished = False
        self._config = config
        self.config = config
        self.finish_events = 0
        self._state = DiscretePopulationState(
            n_tick=0,
            individual_count=np.zeros(  # restored
                (2, config.n_ages, 1),
                dtype=np.float64,  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
            ),
        )

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

    def clear_history(self) -> None:
        """No-op: doubles have no history."""

    def run_tick(self) -> _RunDiscreteDemePopulation:
        """Advance one fake tick."""
        self.tick += 1
        return self

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Record finish events, one per call."""
        _ = deme_id
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def reset(self) -> None:
        """Return the fake deme to tick 0."""
        self.tick = 0

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
    """run_tick drives every deme's tick and aggregates the container."""
    species = _make_species("spatial_run_tick")
    shared_config = _make_population_config(species)

    d0 = _RunDemePopulation(
        species, "d0", shared_config, individual_delta=1.0, sperm_delta=2.0
    )
    d1 = _RunDemePopulation(
        species, "d1", shared_config, individual_delta=1.0, sperm_delta=2.0
    )

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)

    sp.run_tick()

    assert sp.tick == 1
    assert d0.tick == 1 and d1.tick == 1
    assert float(d0.state.individual_count.sum()) == 8.0
    assert float(d1.state.individual_count.sum()) == 8.0
    assert float(d0.state.sperm_storage.sum()) == 8.0
    assert float(d1.state.sperm_storage.sum()) == 8.0


def test_spatial_population_run_stop_marks_finish():
    """A stopped deme halts the run, finishes every deme, fires finish."""
    species = _make_species("spatial_run_stop")
    shared_config = _make_population_config(species)

    d0 = _RunDemePopulation(species, "d0", shared_config, stop_after_run_tick=True)
    d1 = _RunDemePopulation(species, "d1", shared_config)

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)

    sp.run(n_steps=5, record_every=1)

    # The stopped tick does not advance; d1 never runs its tick.
    assert sp.tick == 0
    assert d0.tick == 1 and d1.tick == 0
    assert d0._finished and d1._finished
    assert d0.finish_events == 1 and d1.finish_events == 1


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

    demes = [_build_test_deme(f"stop_id_d{i}", species) for i in range(3)]
    for deme in demes:
        deme.register_hooks(record_finish)
    demes[1].register_hooks(stop_on_deme_one)

    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial.run(n_steps=5)

    # Deme 1 stops first: its lifecycle fires its own finish (id 1), then
    # the container's mark-all pass fires every deme in list order with
    # each deme's own index.  Demes 0 and 2 finish only via mark-all.
    assert finish_ids == [1, 0, 1, 2]
    assert all(deme._finished for deme in demes)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface


def test_spatial_population_stochastic_discrete_migration_preserves_integer_counts():
    """Stochastic discrete migration keeps per-deme counts integral."""
    species = _make_species("spatial_run_stochastic_discrete")
    shared_config = _make_discrete_population_config(species, "stoch_disc")  # restored
    shared_config = shared_config._replace(stochastic=True, continuous_sampling=False)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    d0 = _RunDiscreteDemePopulation(species, "d0", shared_config)
    d1 = _RunDiscreteDemePopulation(species, "d1", shared_config)  # restored
    d0._state = d0.state._replace(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        individual_count=np.array(
            [
                [[0.0], [3.0]],
                [[0.0], [2.0]],
            ],
            dtype=np.float64,
        )
    )

    np.random.seed(17)

    sp = SpatialPopulation(
        [d0, d1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )
    sp.run_tick()

    total_counts = [float(deme.state.individual_count.sum()) for deme in sp.demes]
    assert np.isclose(sum(total_counts), 5.0)
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
    ind = np.zeros((2, shared_config.n_ages, 1), dtype=np.float64)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    ind[0, 1, 0] = 5.0
    ind[1, 1, 0] = 4.0  # restored
    sperm = np.zeros((shared_config.n_ages, 1, 1), dtype=np.float64)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
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
        female_total = float(deme.state.individual_count[0, 1, 0])
        sperm_total = float(deme.state.sperm_storage[1, 0, 0])
        assert female_total >= sperm_total
        assert np.allclose(
            deme.state.individual_count, np.round(deme.state.individual_count)
        )
        assert np.allclose(deme.state.sperm_storage, np.round(deme.state.sperm_storage))


# ---------------------------------------------------------------------------
# Per-deme hook dispatch semantics
# ---------------------------------------------------------------------------


def _build_test_deme(name: str, species: nt.Species) -> nt.DiscreteGenerationPopulation:
    """Build an independent quiescent discrete deme (own hook storage)."""
    return (
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
        .build()
    )


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
        [_build_test_deme("prio_d0", _make_species("spatial_prio"))],
        migration_rate=0.0,
    )
    sp.register_hooks(first_a)
    sp.register_hooks(first_b)

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

    d0 = _build_test_deme("local_d0", species)
    d1 = _build_test_deme("local_d1", species)
    d0.register_hooks(d0_hook)
    d1.register_hooks(d1_hook)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
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

    demes = [_build_test_deme(f"deme_id_d{i}", species) for i in range(3)]
    for deme in demes:
        deme.register_hooks(record_deme)
    spatial = SpatialPopulation(demes, migration_rate=0.0)
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

    demes = [_build_test_deme(f"sel_deme_d{i}", species) for i in range(3)]
    for deme in demes:
        deme.register_hooks(only_deme_one)
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial.run(n_steps=1)

    assert hits == [1]


def test_spatial_compiled_local_hooks_still_take_effect() -> None:
    """A local stop hook halts the first tick and blocks further runs."""
    species = _make_species("spatial_compiled_local_hook_effect")
    d0 = _build_test_deme("csr_local_d0", species)
    d1 = _build_test_deme("csr_local_d1", species)

    @nt.hook(event="first", priority=0)
    def stop_immediately(pop: object) -> int:
        """Request termination on the first event."""  # restored
        pop.stop()  # type: ignore[attr-defined]  # duck-typed double call: shapes verified by assertions below
        return 0

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.register_hooks(stop_immediately, deme=0)
    spatial.run_tick()

    assert d0._finished and d1._finished
    with pytest.raises(RuntimeError):
        spatial.run_tick()


# ========================================================================
# Compact spatial hook plan tests (issue #37)
# ========================================================================


def _build_quiescent_age_pop(
    species: nt.Species,
    n_demes: int,
    name: str = "quiescent",
) -> SpatialPopulation:
    """Build a homogeneous quiescent age-structured population."""
    return (
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
        .build()
    )


def _build_discrete_pop(
    species: nt.Species,
    n_demes: int,
    name: str = "discrete",
) -> SpatialPopulation:
    """Build a homogeneous discrete-generation population via builder."""
    return (
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
        .build()
    )


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

    d0 = _build_test_deme("cw_d0", species)
    d0.register_hooks(my_hook)
    d1 = _build_test_deme("cw_d1", species)
    d2 = _build_test_deme("cw_d2", species)
    # d1 and d2 reuse d0's compiled descriptor list (identical sequences).  # restored
    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    d2.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

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

    d0 = _build_test_deme("cev_d0", species)
    d0.register_hooks(my_hook)
    d1 = _build_test_deme("cev_d1", species)  # restored
    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    public = spatial.get_compiled_hooks()
    assert len(public) == 2
    assert {int(desc.deme_selector) for desc in public} == {0, 1}


def test_compact_plan_subset_selector() -> None:
    """Descriptor with subset selector stays as tuple, not wildcard."""
    species = _make_species("compact_subset")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    d0 = _build_test_deme("cs_d0", species)
    d0.register_hooks(my_hook)
    d1 = _build_test_deme("cs_d1", species)  # restored
    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    d2 = _build_test_deme("cs_d2", species)

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

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

    sp = _build_quiescent_age_pop(species, n_demes=2)
    # Clear deme 0's compiled plan so only the two hooks below are installed.  # restored
    sp.deme(0).compiled_hook_descriptors = []  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    sp.deme(0).register_hooks(mul2)
    sp.deme(0).register_hooks(add1)
    descs0 = list(sp.deme(0).compiled_hook_descriptors)

    # Deme 1 reuses the same descriptors in reversed order (non-commutative).  # restored
    sp.deme(1).compiled_hook_descriptors = [descs0[1], descs0[0]]  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    sp._refresh_spatial_hooks()

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

    d0 = _build_test_deme("ce_d0", species)
    d0.register_hooks(my_hook)
    d1 = _build_test_deme("ce_d1", species)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == 0


# -----------------------------------------------------------------------
# register_hooks shared-storage tests
# -----------------------------------------------------------------------
def test_set_hook_shared_storage_registers_once() -> None:
    """register_hooks on shared-storage demes appends one descriptor."""
    species = _make_species("set_hook_shared")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)
    # Baseline: all demes currently share one descriptor list object.  # restored
    count_before = len(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    sp.register_hooks(my_hook)  # restored
    count_after = len(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert count_after == count_before + 1

    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"


def test_set_hook_shared_storage_subset_cow_structure() -> None:
    """Subset registration copy-on-writes so non-targeted demes stay clean."""
    species = _make_species("set_hook_cow")

    @nt.hook(event="first", priority=0)
    def my_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)
    # All demes share one list; registration on a subset triggers copy-on-write.  # restored
    shared_id = id(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    count_before = len(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    sp.register_hooks(my_hook, deme=0)
    # Targeted deme 0 gets a fresh list; non-targeted demes keep the shared one.  # restored
    assert id(sp.deme(0).compiled_hook_descriptors) != shared_id  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert len(sp.deme(0).compiled_hook_descriptors) == count_before + 1  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    # Non-targeted demes must still share the original list object.  # restored
    assert id(sp.deme(1).compiled_hook_descriptors) == shared_id  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert len(sp.deme(1).compiled_hook_descriptors) == count_before  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert id(sp.deme(2).compiled_hook_descriptors) == shared_id  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert len(sp.deme(2).compiled_hook_descriptors) == count_before  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface


def test_set_hook_shared_storage_subset_cow_execution() -> None:
    """Subset registration: only targeted deme runs hook, others unchanged."""
    species = _make_species("cow_exec")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp.register_hooks(add_one, deme=target)

    sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_subset_callback_hook_no_leak() -> None:
    """Callback hook registered on a subset does not leak to other demes."""
    species = _make_species("no_leak")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @nt.hook(event="first", priority=0)
    def py_hook(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp.register_hooks(py_hook, deme=target)
    sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_empty_selector_noop() -> None:
    """Empty selector (no matching demes) leaves all state unchanged."""
    species = _make_species("empty_sel")
    sp = _build_quiescent_age_pop(species, n_demes=3)

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    totals_before = [float(sp.deme(i).state.individual_count.sum()) for i in range(3)]
    sp.register_hooks(add_one, deme=[])

    sp.run_tick()

    for i in range(3):
        assert float(sp.deme(i).state.individual_count.sum()) == totals_before[i]


def test_set_hook_cow_subsequent_mutation_no_leak() -> None:
    """After COW subset registration, a later wildcard hook stays isolated."""
    species = _make_species("cow_iso")

    @nt.hook(event="first", priority=0)
    def hook_a(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    @nt.hook(event="first", priority=0)
    def hook_b(pop: object) -> int:
        """Bump age-1 males on every fire."""  # restored
        pop.state.individual_count[1, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)

    # Subset registration on deme 0 triggers COW.
    sp.register_hooks(hook_a, deme=0)  # restored
    count_after_a = len(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    count_non_target = len(sp.deme(1).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    # Register hook_b on all demes via wildcard selector.
    sp.register_hooks(hook_b)  # restored
    assert len(sp.deme(0).compiled_hook_descriptors) == count_after_a + 1  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert len(sp.deme(1).compiled_hook_descriptors) == count_non_target + 1  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    assert len(sp.deme(2).compiled_hook_descriptors) == count_non_target + 1  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    # Wildcard registration keeps deme 0's COW-isolated list separate.  # restored
    assert id(sp.deme(0).compiled_hook_descriptors) != id(  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        sp.deme(1).compiled_hook_descriptors  # type: ignore[attr-defined]  # duck-typed double call: shapes verified by assertions below
    )


# -----------------------------------------------------------------------
# Run-tick integration tests
# -----------------------------------------------------------------------
def test_compact_plan_run_tick_deterministic_state() -> None:
    """Quiescent model: hook +1 on age-1 female → total per deme = 200 + 1."""
    species = _make_species("compact_det")
    sp = _build_quiescent_age_pop(species, n_demes=3)

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp.register_hooks(add_one)

    sp.run_tick()

    for i in range(3):
        total = float(sp.deme(i).state.individual_count.sum())
        assert total == 201.0, f"deme[{i}]: expected 201, got {total}"
        state = sp.deme(i).state.individual_count
        assert np.all(state >= 0.0)
        assert not np.any(np.isnan(state))


def test_compact_plan_csr_then_callback_ordering() -> None:
    """Within one event CSR plans run before Python callbacks.

    CSR ×2 then callback +1 on age-1 female=100 → 201 (reversed order
    would give 202).
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

    sp = _build_quiescent_age_pop(species, n_demes=2)
    sp.register_hooks(mul2_csr)
    sp.register_hooks(add1)

    sp.run_tick()

    # CSR runs first: female[age=1] 100×2=200, then callback +1 → 201.
    # Survival keeps age 1 (rate 1.0); aging moves the result into age 2.
    for i in range(2):
        assert float(sp.deme(i).state.individual_count[0, 2, 0]) == 201.0


def test_builder_homogeneous_demes_share_compiled_hooks() -> None:
    """Builder-created homogeneous population: all demes share hook storage."""
    species = _make_species("builder_share")
    n_demes = 5
    sp = _build_discrete_pop(species, n_demes)
    # Pin the single shared list object before further registration.  # restored
    ref_id = id(sp.deme(0).compiled_hook_descriptors)  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
    for i in range(1, n_demes):  # restored
        assert id(sp.deme(i).compiled_hook_descriptors) == ref_id  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp.register_hooks(add_one)
    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"


def test_builder_set_hook_subset_cow_combined() -> None:
    """Subset registration via builder-created population: COW isolates target."""
    species = _make_species("builder_cow")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @nt.hook(event="first", priority=0)
    def add_one(pop: object) -> int:
        """Bump age-1 females on every fire."""  # restored
        pop.state.individual_count[0, 1, 0] += 1.0  # type: ignore[attr-defined]  # duck-typed double: intentionally violates the typed surface
        return 0

    sp.register_hooks(add_one, deme=target)

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

registry = pop.hooks.registry
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
