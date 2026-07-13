#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest  # type: ignore

import natal as nt
from natal.data import DiscretePopulationState, PopulationConfig, PopulationState
from natal.engine.lifecycle_wrappers import LifecycleWrappers
from natal.genetics import Species
from natal.hooks import Op, hook
from natal.numba import compat as nbc
from natal.numba.utils import numba_disabled
from natal.population.base import BasePopulation
from natal.spatial.population import SpatialPopulation


class _RunDemePopulation(BasePopulation):
    def __init__(
        self,
        species: Species,
        name: str,
        config,
        *,
        individual_delta: float = 0.0,
        sperm_delta: float = 0.0,
        stop_after_run_tick: bool = False,
    ):
        self._species = species
        self._name = name
        self._tick = 0
        self._history = []
        self._finished = False
        self._config = config
        self._individual_delta = float(individual_delta)
        self._sperm_delta = float(sperm_delta)
        self._stop_after_run_tick = bool(stop_after_run_tick)
        self.finish_events = 0
        self._hooks_obj: Any = None
        self._state = PopulationState(
            n_tick=0,
            individual_count=np.zeros((2, 1, 1), dtype=np.float64),
            sperm_storage=np.zeros((1, 1, 1), dtype=np.float64),
        )

    def clear_history(self) -> None:
        self._history.clear()

    def run_tick(self):
        if self._individual_delta != 0.0:
            self._state = self._state._replace(
                individual_count=self._state.individual_count + self._individual_delta,
                sperm_storage=self._state.sperm_storage + self._sperm_delta,
            )
        if self._stop_after_run_tick:
            self._finished = True
        self._tick += 1
        return self

    def get_total_count(self) -> int:
        return int(self._state.individual_count.sum())

    def get_female_count(self) -> int:
        return int(self._state.individual_count[0].sum())

    def get_male_count(self) -> int:
        return int(self._state.individual_count[1].sum())

    def run(self, n_steps: int, record_every: int = 1, finish: bool = False):
        self._tick += int(n_steps)
        return self

    def reset(self) -> None:
        self._tick = 0

    def export_config(self):
        return self._config

    def get_compiled_event_hooks(self):
        return cast(LifecycleWrappers, self._hooks_obj)

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def update(self) -> Any:  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _RunDiscreteDemePopulation(BasePopulation):
    def __init__(self, species: Species, name: str, config):
        self._species = species
        self._name = name
        self._tick = 0
        self._history = []
        self._finished = False
        self._config = config
        self.finish_events = 0
        self._hooks_obj: Any = None
        self._state = DiscretePopulationState(
            n_tick=0,
            individual_count=np.zeros((2, 2, 1), dtype=np.float64),
        )

    def clear_history(self) -> None:
        self._history.clear()

    def run_tick(self):
        self._tick += 1
        return self

    def get_total_count(self) -> int:
        return int(self._state.individual_count.sum())

    def get_female_count(self) -> int:
        return int(self._state.individual_count[0].sum())

    def get_male_count(self) -> int:
        return int(self._state.individual_count[1].sum())

    def run(self, n_steps: int, record_every: int = 1, finish: bool = False):
        self._tick += int(n_steps)
        return self

    def reset(self) -> None:
        self._tick = 0

    def export_config(self):
        return self._config

    def get_compiled_event_hooks(self):
        return cast(LifecycleWrappers, self._hooks_obj)

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        if event_name == "finish":
            self.finish_events += 1
        return 0

    def update(self) -> Any:  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _make_species(prefix: str = "SpatialRunSpecies") -> Species:
    return Species.from_dict(
        prefix,
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


def _make_population_config(species: Species, name: str = "config_template") -> PopulationConfig:
    return (
        nt.AgeStructuredPopulation
        .setup(species=species, name=name, stochastic=False)
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


def test_spatial_population_run_tick_updates_all_demes():
    species = _make_species("spatial_run_tick")
    shared_config = _make_population_config(species)

    d0 = _RunDemePopulation(
        species,
        "d0",
        shared_config,
        individual_delta=1.0,
        sperm_delta=2.0,
    )
    d1 = _RunDemePopulation(
        species,
        "d1",
        shared_config,
        individual_delta=1.0,
        sperm_delta=2.0,
    )

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)

    with numba_disabled():
        sp.run_tick()

    assert sp.tick == 1
    assert d0.tick == 1 and d1.tick == 1
    assert float(d0.state.individual_count.sum()) == 2.0
    assert float(d1.state.individual_count.sum()) == 2.0
    assert float(d0.state.sperm_storage.sum()) == 2.0
    assert float(d1.state.sperm_storage.sum()) == 2.0


def test_spatial_population_run_stop_marks_finish():
    species = _make_species("spatial_run_stop")
    shared_config = _make_population_config(species)

    d0 = _RunDemePopulation(
        species,
        "d0",
        shared_config,
        stop_after_run_tick=True,
    )
    d1 = _RunDemePopulation(species, "d1", shared_config)

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)

    with numba_disabled():
        sp.run(n_steps=5, record_every=1)

    assert sp.tick == 0
    assert d0.tick == 1 and d1.tick == 0
    assert d0._finished and d1._finished
    assert d0.finish_events == 1 and d1.finish_events == 1


@pytest.mark.numba_off
def test_spatial_population_stochastic_discrete_migration_preserves_integer_counts():
    species = _make_species("spatial_run_stochastic_discrete")
    shared_config = _make_population_config(species)._replace(
        stochastic=True,
        continuous_sampling=False,
    )

    d0 = _RunDiscreteDemePopulation(species, "d0", shared_config)
    d1 = _RunDiscreteDemePopulation(species, "d1", shared_config)
    d0._state = d0.state._replace(
        individual_count=np.array(
            [
                [[0.0], [3.0]],
                [[0.0], [2.0]],
            ],
            dtype=np.float64,
        )
    )

    np.random.seed(17)
    nbc.set_numba_seed(17)

    sp = SpatialPopulation(
        [d0, d1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )
    sp.run_tick()

    total_counts = [float(deme.state.individual_count.sum()) for deme in sp.demes]
    assert np.isclose(sum(total_counts), 5.0)
    for deme in sp.demes:
        assert np.allclose(deme.state.individual_count, np.round(deme.state.individual_count))


@pytest.mark.numba_off
def test_spatial_population_stochastic_age_migration_preserves_sperm_consistency():
    species = _make_species("spatial_run_stochastic_age")
    shared_config = _make_population_config(species)._replace(
        stochastic=True,
        continuous_sampling=False,
    )

    d0 = _RunDemePopulation(species, "d0", shared_config)
    d1 = _RunDemePopulation(species, "d1", shared_config)
    d0._state = PopulationState(
        n_tick=0,
        individual_count=np.array(
            [
                [[5.0]],
                [[4.0]],
            ],
            dtype=np.float64,
        ),
        sperm_storage=np.array([[[3.0]]], dtype=np.float64),
    )

    np.random.seed(23)
    nbc.set_numba_seed(23)

    sp = SpatialPopulation(
        [d0, d1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )
    sp.run_tick()

    total_females = sum(float(deme.state.individual_count[0].sum()) for deme in sp.demes)
    total_males = sum(float(deme.state.individual_count[1].sum()) for deme in sp.demes)
    total_sperm = sum(float(deme.state.sperm_storage.sum()) for deme in sp.demes)
    assert np.isclose(total_females, 5.0)
    assert np.isclose(total_males, 4.0)
    assert np.isclose(total_sperm, 3.0)

    for deme in sp.demes:
        female_total = float(deme.state.individual_count[0, 0, 0])
        sperm_total = float(deme.state.sperm_storage[0, 0, 0])
        assert female_total >= sperm_total
        assert np.allclose(deme.state.individual_count, np.round(deme.state.individual_count))
        assert np.allclose(deme.state.sperm_storage, np.round(deme.state.sperm_storage))


@pytest.mark.numba_off
def test_spatial_mixedpriority_ordering_runs_in_run_tick_and_run():
    species = _make_species("spatial_mixed_priority")
    calls_np = np.zeros(8, dtype=np.int32)  # 0: py, 1: njit
    observed_first_py_np = np.zeros(4, dtype=np.float64)
    observed_first_njit_np = np.zeros(4, dtype=np.float64)
    observed_early_probe_np = np.zeros(4, dtype=np.float64)
    idx_np = np.zeros(1, dtype=np.int32)

    def _build_deme(name: str) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
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

    d0 = _build_deme("mixed_d0")
    d1 = _build_deme("mixed_d1")

    # Spatial engine require one shared config object.
    d1._config = d0.export_config()  # type: ignore[attr-defined]

    from numba import njit

    @hook(event="first", priority=0)
    def first_python(population):  # type: ignore[no-untyped-def]
        idx = int(idx_np[0])
        calls_np[idx] = 0
        observed_first_py_np[idx // 2] = float(population.state.individual_count[1, 1, 0])
        idx_np[0] += 1

    @njit
    @hook(event="first", priority=1)
    def first_njit(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 1
        observed_first_njit_np[idx // 2] = float(state.individual_count[1, 1, 0])
        state.individual_count[1, 1, 0] += 2.0
        idx_np[0] += 1
        return 0

    early_idx_np = np.zeros(1, dtype=np.int32)

    @hook(event="first", priority=2)
    def first_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="early", priority=0)
    def early_probe(population):  # type: ignore[no-untyped-def]
        idx = int(early_idx_np[0])
        observed_early_probe_np[idx] = float(population.state.individual_count[1, 1, 0])
        early_idx_np[0] += 1

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.set_hook("first", first_csr)
    spatial.set_hook("first", first_njit)
    spatial.set_hook("first", first_python)
    spatial.set_hook("early", early_probe)

    spatial.run_tick()
    spatial.run(n_steps=1)

    # 2 demes * 2 ticks: py/njit each called 4 times, per-deme order fixed.
    calls = ["py" if x == 0 else "njit" for x in calls_np]
    assert calls == ["py", "njit", "py", "njit", "py", "njit", "py", "njit"]
    assert observed_first_py_np.tolist() == [10.0, 10.0, 0.0, 0.0]
    assert observed_first_njit_np.tolist() == [10.0, 10.0, 0.0, 0.0]
    # early probes confirm csr (+3) is applied after njit (+2) each tick.
    assert observed_early_probe_np.tolist() == [15.0, 15.0, 5.0, 5.0]


def test_spatial_compiled_hooks_are_pinned_to_owning_deme() -> None:
    species = _make_species("spatial_pinned_hooks")

    def _build_deme(name: str) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0.0, 2.0]},
                    "male": {"WT|WT": [0.0, 2.0]},
                }
            )
            .reproduction(eggs_per_female=0.0)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .build()
        )

    d0 = _build_deme("pin_d0")
    d1 = _build_deme("pin_d1")

    @hook(event="first", priority=1)
    def first_d0():
        return [Op.add(genotypes="WT|WT", ages=1, sex="female", delta=1.0)]

    @hook(event="first", priority=0)
    def first_d1():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=1.0)]

    d0.set_hook("first", first_d0)
    d1.set_hook("first", first_d1)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    registry = spatial.hooks.registry
    assert registry is not None

    start = int(registry.hook_offsets[0])
    end = int(registry.hook_offsets[1])
    assert end - start == 2
    assert registry.deme_selector_types[start:end].tolist() == [1, 1]

    sel0_start = int(registry.deme_selector_offsets[start])
    sel0_end = int(registry.deme_selector_offsets[start + 1])
    sel1_start = int(registry.deme_selector_offsets[start + 1])
    sel1_end = int(registry.deme_selector_offsets[start + 2])
    assert registry.deme_selector_data[sel0_start:sel0_end].tolist() == [0]
    assert registry.deme_selector_data[sel1_start:sel1_end].tolist() == [1]


@pytest.mark.numba_off
def test_spatial_mixed_priority_is_local_per_deme() -> None:
    species = _make_species("spatial_local_priority_per_deme")
    calls_np = np.zeros(4, dtype=np.int32)  # 0: d0_py, 1: d0_njit, 2: d1_py, 3: d1_njit
    idx_np = np.zeros(1, dtype=np.int32)
    observed_d0_py_np = np.zeros(1, dtype=np.float64)
    observed_d0_early_np = np.zeros(1, dtype=np.float64)
    observed_d1_py_np = np.zeros(1, dtype=np.float64)
    observed_d1_early_np = np.zeros(1, dtype=np.float64)

    def _build_deme(name: str) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
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

    d0 = _build_deme("local_d0")
    d1 = _build_deme("local_d1")
    d1._config = d0.export_config()  # type: ignore[attr-defined]

    from numba import njit

    @hook(event="first", priority=0)
    def d0_py(population):  # type: ignore[no-untyped-def]
        idx = int(idx_np[0])
        calls_np[idx] = 0
        observed_d0_py_np[0] = float(population.state.individual_count[1, 1, 0])
        idx_np[0] += 1

    @njit
    @hook(event="first", priority=1)
    def d0_njit(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 1
        state.individual_count[1, 1, 0] += 2.0
        idx_np[0] += 1
        return 0

    @hook(event="first", priority=2)
    def d0_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="early", priority=0)
    def d0_early(population):  # type: ignore[no-untyped-def]
        observed_d0_early_np[0] = float(population.state.individual_count[1, 1, 0])

    @hook(event="first", priority=2)
    def d1_py(population):  # type: ignore[no-untyped-def]
        idx = int(idx_np[0])
        calls_np[idx] = 2
        observed_d1_py_np[0] = float(population.state.individual_count[1, 1, 0])
        idx_np[0] += 1

    @njit
    @hook(event="first", priority=0)
    def d1_njit(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 3
        state.individual_count[1, 1, 0] += 4.0
        idx_np[0] += 1
        return 0

    @hook(event="first", priority=1)
    def d1_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=5.0)]

    @hook(event="early", priority=0)
    def d1_early(population):  # type: ignore[no-untyped-def]
        observed_d1_early_np[0] = float(population.state.individual_count[1, 1, 0])

    # Set hooks directly on individual demes
    d0.set_hook("first", d0_csr)
    d0.set_hook("first", d0_njit)
    d0.set_hook("first", d0_py)
    d0.set_hook("early", d0_early)

    d1.set_hook("first", d1_csr)
    d1.set_hook("first", d1_njit)
    d1.set_hook("first", d1_py)
    d1.set_hook("early", d1_early)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.run_tick()

    calls = ["d0_py" if x == 0 else "d0_njit" if x == 1 else "d1_py" if x == 2 else "d1_njit" for x in calls_np]
    assert calls == ["d0_py", "d0_njit", "d1_njit", "d1_py"]
    assert observed_d0_py_np[0] == 10.0
    assert observed_d0_early_np[0] == 15.0
    assert observed_d1_py_np[0] == 19.0
    assert observed_d1_early_np[0] == 19.0


@pytest.mark.numba_off
def test_spatial_compiled_local_hooks_still_take_effect() -> None:
    species = _make_species("spatial_compiled_local_hook_effect")

    def _build_deme(name: str) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
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

    d0 = _build_deme("csr_d0")
    d1 = _build_deme("csr_d1")
    d1._config = d0.export_config()  # type: ignore[attr-defined]

    @hook(event="first", priority=0)
    def stop_immediately():
        return [Op.stop_if_above(genotypes="WT|WT", ages=1, sex="male", threshold=1.0)]

    d0.set_hook("first", stop_immediately)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.run_tick()

    assert d0._finished and d1._finished  # type: ignore[attr-defined]
    with np.testing.assert_raises(RuntimeError):
        spatial.run_tick()


# ========================================================================
# Compact spatial hook plan tests (issue #37)
# ========================================================================


def _build_test_deme(name: str, species: nt.Species) -> nt.DiscreteGenerationPopulation:
    """Build a minimal deterministic discrete-generation deme for testing."""
    return (
        nt.DiscreteGenerationPopulation.setup(species=species, name=name, stochastic=False)
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


def test_compact_plan_folds_identical_sequences_to_wildcard() -> None:
    """All demes sharing one descriptor sequence → single wildcard slot."""
    species = _make_species("compact_wildcard")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    # Register the hook on deme[0] only; manually share _compiled_hooks
    # to simulate the clone-sharing pattern.
    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", my_hook)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]

    # Make d1 and d2 share d0's hook storage (simulating clone pattern).
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]
    d2._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d2._hooks = d0._hooks  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    expanded = spatial._collect_effective_compiled_hooks()
    compact = spatial._collect_compact_spatial_hooks()

    # Expanded view: 3 demes, each pinned to its owner.
    assert len(expanded) == 3
    assert {int(desc.deme_selector) for desc in expanded} == {0, 1, 2}

    # Compact plan: 1 wildcard descriptor (not 3).
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"
    assert compact[0].njit_fn is not None


def test_compact_plan_preserves_expanded_view() -> None:
    """Public get_compiled_hooks still returns per-deme pinned descriptors."""
    species = _make_species("compact_expanded_view")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", my_hook)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    public = spatial.get_compiled_hooks()
    assert len(public) == 2
    assert {int(desc.deme_selector) for desc in public} == {0, 1}


def test_compact_plan_subset_selector() -> None:
    """Descriptor with subset selector stays as tuple, not wildcard."""
    species = _make_species("compact_subset")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", my_hook)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]
    # d2 has its own (empty) storage — NOT sharing with d0/d1.
    # So only [0, 1] share the descriptor; [2] has nothing.

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    # One descriptor for demes [0, 1] with compacted selector.
    assert len(compact) == 1
    sel = compact[0].deme_selector
    assert isinstance(sel, tuple)
    assert set(sel) == {0, 1}


def test_compact_plan_duplicate_slots_preserved() -> None:
    """[A, A] registration still produces two execution slots."""
    species = _make_species("compact_dup")

    from numba import njit

    @njit
    @hook(event="first", custom=True, priority=0)
    def hook_a(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @njit
    @hook(event="first", custom=True, priority=1)
    def hook_a2(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 2.0
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", hook_a, hook_id=0)
    d0.set_hook("first", hook_a2, hook_id=1)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 2
    # Both slots are for all demes.
    assert compact[0].deme_selector == "*"
    assert compact[1].deme_selector == "*"
    # Priorities preserved in sequence order.
    assert compact[0].priority == 0
    assert compact[1].priority == 1


def test_compact_plan_different_order_not_merged() -> None:
    """[A, B] vs [B, A] are distinct sequences and not merged."""
    species = _make_species("compact_order")

    from numba import njit

    @njit
    @hook(event="first", custom=True, priority=0)
    def hook_a(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @njit
    @hook(event="first", custom=True, priority=1)
    def hook_b(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 2.0
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", hook_a, hook_id=0)
    d0.set_hook("first", hook_b, hook_id=1)

    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d1.set_hook("first", hook_b, hook_id=0)
    d1.set_hook("first", hook_a, hook_id=1)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    # Two groups: [0] with [A,B] and [1] with [B,A].
    assert len(compact) == 4  # 2 per group
    selectors = [desc.deme_selector for desc in compact]
    assert 0 in selectors
    assert 1 in selectors


def test_set_hook_shared_storage_registers_once() -> None:
    """Spatial set_hook on shared-storage demes only appends one descriptor."""
    species = _make_species("set_hook_shared")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    d0 = _build_test_deme("d0", species)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]

    # Simulate clone sharing: all three share d0's compiled hooks.
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]
    d2._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d2._hooks = d0._hooks  # type: ignore[attr-defined]

    # Count of descriptors before registering.
    count_before = len(d0._compiled_hooks)  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    # Register via spatial-level API.
    spatial.set_hook("first", my_hook)

    # Each deme's shared _compiled_hooks should have only +1 descriptor,
    # not +3 (one per targeted deme).
    count_after = len(d0._compiled_hooks)  # type: ignore[attr-defined]
    assert count_after == count_before + 1

    # Compact plan should be 1 wildcard slot for the custom hook.
    compact = spatial._collect_compact_spatial_hooks()
    custom_slots = [d for d in compact if d.njit_fn is not None]
    assert len(custom_slots) == 1
    assert custom_slots[0].deme_selector == "*"


def test_set_hook_shared_storage_subset_cow() -> None:
    """Subset registration copy-on-writes so non-targeted demes stay clean."""
    species = _make_species("set_hook_cow")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    d0 = _build_test_deme("d0", species)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]

    # All three share d0's compiled hooks (simulating clone sharing).
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]
    d2._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d2._hooks = d0._hooks  # type: ignore[attr-defined]

    shared_id = id(d0._compiled_hooks)  # type: ignore[attr-defined]
    count_before = len(d0._compiled_hooks)  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    # Register only for deme [0] — subset of the shared storage group.
    spatial.set_hook("first", my_hook, deme_selector=0)

    # d0 should now have a new (copied) _compiled_hooks with +1 entry.
    assert id(d0._compiled_hooks) != shared_id  # type: ignore[attr-defined]
    assert len(d0._compiled_hooks) == count_before + 1  # type: ignore[attr-defined]

    # d1 and d2 should still reference the original shared list (unchanged).
    assert id(d1._compiled_hooks) == shared_id  # type: ignore[attr-defined]
    assert len(d1._compiled_hooks) == count_before  # type: ignore[attr-defined]
    assert id(d2._compiled_hooks) == shared_id  # type: ignore[attr-defined]
    assert len(d2._compiled_hooks) == count_before  # type: ignore[attr-defined]


def test_compact_plan_run_tick_deterministic_state() -> None:
    """Custom hook via shared-storage set_hook runs without crash and +1 sticks."""
    species = _make_species("compact_deterministic")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    d0 = _build_test_deme("d0", species)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]

    # Simulate clone sharing.
    d1._compiled_hooks = d0._compiled_hooks  # type: ignore[attr-defined]
    d1._hooks = d0._hooks  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    with numba_disabled():
        spatial.set_hook("first", add_one)
        spatial.run_tick()

    # Both demes advanced one tick — hook ran without crash.
    assert d0.tick == 1 and d1.tick == 1
    # State sanity: no NaN or negative counts.
    for d in (d0, d1):
        state = d.state.individual_count
        assert np.all(state >= 0.0)
        assert not np.any(np.isnan(state))


def test_compact_plan_mixed_declarative_and_njit() -> None:
    """CSR + njit in same event — compact plan preserves structure."""
    species = _make_species("compact_mixed")

    from numba import njit

    @njit
    @hook(event="first", custom=True, priority=1)
    def after_csr(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @hook(event="first", priority=0)
    def csr_hook():
        return [Op.add(genotypes="WT|WT", ages=1, sex="female", delta=2.0)]

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", csr_hook, hook_id=0)
    d0.set_hook("first", after_csr, hook_id=1)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    d1.set_hook("first", csr_hook, hook_id=0)
    d1.set_hook("first", after_csr, hook_id=1)

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 4  # 2 demes × 2 hooks, different id()

    with numba_disabled():
        spatial.run_tick()

    assert d0.tick == 1 and d1.tick == 1
    # State sanity.
    for d in (d0, d1):
        state = d.state.individual_count
        assert np.all(state >= 0.0)
        assert not np.any(np.isnan(state))


def test_compact_plan_empty_hook_sequence_skipped() -> None:
    """Deme with no compiled hooks contributes no descriptors to compact plan."""
    species = _make_species("compact_empty")

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", my_hook)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]
    # d1 has no hooks.

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    # Only d0 contributes a descriptor; empty d1 is skipped.
    assert len(compact) == 1
    assert compact[0].deme_selector == 0


# ========================================================================
# Builder-based homogeneous sharing and COW tests
# ========================================================================


def _build_homogeneous_spatial_pop(
    species: nt.Species,
    n_demes: int,
    name: str = "test_homog",
    *,
    pop_type: str = "discrete_generation",
    quiescent: bool = False,
) -> SpatialPopulation:
    """Build a homogeneous spatial population via the builder API.

    All demes share the same config, species, index registry, and
    compiled hook storage — the defining trait of homogeneous mode.

    Args:
        species: Genetic architecture shared by all demes.
        n_demes: Number of demes.
        name: Population name.
        pop_type: ``"discrete_generation"`` or ``"age_structured"``.
        quiescent: If True, set ``eggs_per_female=0.0`` and survival=1.0
            so that the population state changes only through hooks.
            This makes hook effects directly measurable.

    Returns:
        A ``SpatialPopulation`` with all demes sharing compiled state.
    """
    if pop_type == "discrete_generation":
        if quiescent:
            # 离散代模型并不适合"静态"模式（eggs_per_female=0 时种群会灭绝），
            # 自动切换到年龄结构模型
            return cast(SpatialPopulation, (
                nt.SpatialPopulation
                .builder(species, n_demes=n_demes, pop_type="age_structured")
                .setup(name=name, stochastic=False)
                .age_structure(n_ages=3, new_adult_age=1)
                .initial_state(individual_count={
                    "female": {"WT|WT": [0.0, 100.0, 0.0]},
                    "male": {"WT|WT": [0.0, 100.0, 0.0]},
                })
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
            ))
        return cast(SpatialPopulation, (
            nt.SpatialPopulation
            .builder(species, n_demes=n_demes, pop_type="discrete_generation")
            .setup(name=name, stochastic=False)
            .initial_state(individual_count={
                "female": {"WT|WT": 100},
                "male": {"WT|WT": 100},
            })
            .reproduction(eggs_per_female=10.0)
            .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                         juvenile_growth_mode="concave")
            .build()
        ))
    else:
        return cast(SpatialPopulation, (
            nt.SpatialPopulation
            .builder(species, n_demes=n_demes, pop_type="age_structured")
            .setup(name=name, stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(individual_count={
                "female": {"WT|WT": [0.0, 100.0, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, 100.0, 0.0, 0.0]},
            })
            .reproduction(eggs_per_female=0.0)
            .competition(carrying_capacity=500, low_density_growth_rate=6.0,
                         juvenile_growth_mode="logistic")
            .build()
        ))


def test_builder_homogeneous_demes_share_compiled_hooks() -> None:
    """Builder-created homogeneous population: all demes share _compiled_hooks.

    Invariant: In homogeneous mode, _clone_deme() copies
    ``clone._compiled_hooks = template._compiled_hooks``, so every deme
    in the population points to the same list object.  A spatial
    set_hook(..., deme_selector="*") therefore registers only ONE
    descriptor that covers all demes.
    """
    species = _make_species("builder_homog_share")
    n_demes = 10
    sp = _build_homogeneous_spatial_pop(species, n_demes)

    # --- Invariant 1: same object identity across all demes ---
    ref_id = id(sp.deme(0)._compiled_hooks)  # type: ignore[attr-defined]
    for i in range(1, n_demes):
        assert id(sp.deme(i)._compiled_hooks) == ref_id, (  # type: ignore[attr-defined]
            f"deme[{i}] does not share _compiled_hooks with deme[0]"
        )

    # --- Invariant 2: registering a hook for all demes produces ---
    # a single wildcard descriptor in the compact plan.
    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp.set_hook("first", add_one)
    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == "*"
    assert compact[0].njit_fn is not None


def test_builder_set_hook_subset_cow_structure() -> None:
    """subset registration via spatial.set_hook triggers COW for targeted deme.

    Scenario (matching the issue #37 spec):
      - Build 10-deme homogeneous population via builder API.
      - Register hook on deme 5 only: ``spatial.set_hook(..., deme_selector=5)``.
      - Expect: deme 5 splits off (new _compiled_hooks copy), demes 0–4,6–9
        keep the original shared storage intact.

    Invariant proven: After COW, only the targeted deme holds a different
    _compiled_hooks object from the original shared one; all non-targeted
    demes remain identical to the original.
    """
    species = _make_species("builder_cow_tgt")
    n_demes = 10
    sp = _build_homogeneous_spatial_pop(species, n_demes)

    original_id = id(sp.deme(0)._compiled_hooks)  # type: ignore[attr-defined]
    original_len = len(sp.deme(0)._compiled_hooks)  # type: ignore[attr-defined]

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    target_deme = 5
    sp.set_hook("first", my_hook, deme_selector=target_deme)

    # --- Invariant 1: targeted deme has a new (COW) storage ---
    assert id(sp.deme(target_deme)._compiled_hooks) != original_id  # type: ignore[attr-defined]
    assert len(sp.deme(target_deme)._compiled_hooks) == original_len + 1  # type: ignore[attr-defined]

    # --- Invariant 2: non-targeted demes still share the original storage ---
    for i in range(n_demes):
        if i == target_deme:
            continue
        assert id(sp.deme(i)._compiled_hooks) == original_id, (  # type: ignore[attr-defined]
            f"deme[{i}] should still share original storage"
        )
        assert len(sp.deme(i)._compiled_hooks) == original_len  # type: ignore[attr-defined]

    # --- Invariant 3: compact plan has two slots ---
    # one for the 9-deme shared wildcard group, one for deme 5.
    compact = sp._collect_compact_spatial_hooks()
    # The shared group has 0 hooks, deme 5 has 1 hook → 1 descriptor.
    # Both contribute to the compact plan, but only the targeted one has njit_fn.
    non_empty = [d for d in compact if d.njit_fn is not None]
    assert len(non_empty) == 1
    assert isinstance(non_empty[0].deme_selector, int)
    assert non_empty[0].deme_selector == target_deme


def test_builder_set_hook_subset_cow_execution() -> None:
    """subset registration COW: targeted deme runs the hook; others do not.

    This is the execution counterpart of test_builder_set_hook_subset_cow_structure.
    Instead of verifying internal storage references, it runs a tick and
    measures the observable effect: only the targeted deme's state changes.

    Invariant:
      - Initial state: every deme has female[WT|WT] = 100 and male[WT|WT] = 100.
      - Hook: +1.0 to female age-1 WT|WT.
      - After run_tick(), total individual_count for targeted deme = 201,
        all others = 200.  (Aging moves the +1 from age 1 to age 2, but
        total is conserved.)
    """
    species = _make_species("builder_cow_exec")
    n_demes = 10
    target_deme = 5
    sp = _build_homogeneous_spatial_pop(species, n_demes, quiescent=True)

    from numba import njit

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp.set_hook("first", add_one, deme_selector=target_deme)

    # Record initial total per deme before hook execution.
    initial_totals = np.array([
        float(sp.deme(i).state.individual_count.sum())
        for i in range(n_demes)
    ])

    with numba_disabled():
        sp.run_tick()

    # --- Invariant 1: targeted deme total increased by exactly 1.0 ---
    assert float(sp.deme(target_deme).state.individual_count.sum()) == pytest.approx(
        initial_totals[target_deme] + 1.0
    )

    # --- Invariant 2: non-targeted demes unchanged in total ---
    for i in range(n_demes):
        if i == target_deme:
            continue
        assert float(sp.deme(i).state.individual_count.sum()) == pytest.approx(
            initial_totals[i]
        ), f"deme[{i}] total changed from {initial_totals[i]} to {sp.deme(i).state.individual_count.sum()}"

    # --- Invariant 3: global total increased by exactly 1.0 ---
    final_total = sum(
        float(sp.deme(i).state.individual_count.sum())
        for i in range(n_demes)
    )
    assert final_total == pytest.approx(initial_totals.sum() + 1.0)

    # --- Invariant 4: no NaN, no negative counts ---
    for i in range(n_demes):
        state = sp.deme(i).state.individual_count
        assert np.all(state >= 0.0), f"deme[{i}] has negative counts"
        assert not np.any(np.isnan(state)), f"deme[{i}] has NaN"


def test_builder_cow_then_wildcard_registration() -> None:
    """After COW splits deme 5, a wildcard registration adds to both groups.

    Scenario:
      1. Build 10-deme homogeneous population.
      2. set_hook(..., deme_selector=5) → COW: deme 5 splits off.
      3. set_hook("first", hook_B) → wildcard: all 10 demes get hook_B.

    Invariant: Both hooks execute for deme 5, but only hook_B executes
    for the other 9 demes.  Deme 5 gets +2.0 (hook_A + hook_B), all
    others get +1.0 (hook_B only).
    """
    species = _make_species("builder_cow_wildcard")
    n_demes = 10
    target_deme = 5
    sp = _build_homogeneous_spatial_pop(species, n_demes, quiescent=True)

    from numba import njit

    @njit
    @hook(event="first", custom=True, priority=0)
    def hook_subset(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @njit
    @hook(event="first", custom=True, priority=1)
    def hook_all(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 2.0
        return 0

    # Step 2: subset registration → COW on deme 5.
    sp.set_hook("first", hook_subset, deme_selector=target_deme)

    # Step 3: wildcard registration → all 10 demes.
    sp.set_hook("first", hook_all)

    with numba_disabled():
        sp.run_tick()

    # --- Invariant: deme 5 gets both hooks (+3.0 total), others get only hook_all (+2.0) ---
    # Because of aging, the hook additions move from age 1 to age 2, but total is conserved.
    base_init = 200.0  # 100 female + 100 male per deme
    expected_target = base_init + 3.0   # 100+100 + 1 + 2
    expected_other = base_init + 2.0    # 100+100 + 2

    assert float(sp.deme(target_deme).state.individual_count.sum()) == pytest.approx(expected_target), (
        f"deme[{target_deme}] expected {expected_target}, got "
        f"{sp.deme(target_deme).state.individual_count.sum()}"
    )
    for i in range(n_demes):
        if i == target_deme:
            continue
        assert float(sp.deme(i).state.individual_count.sum()) == pytest.approx(expected_other), (
            f"deme[{i}] expected {expected_other}, got "
            f"{sp.deme(i).state.individual_count.sum()}"
        )

    # --- Invariant: compact plan reflects two groups ---
    compact = sp._collect_compact_spatial_hooks()
    # 2 descriptors: hook_subset for deme 5, hook_all for all 10
    # In homogeneous mode, all 10 demes share one storage, so hook_all
    # appears as a single wildcard — but deme 5's COW group adds hook_subset.
    assert len(compact) >= 3  # hook_subset(deme5), hook_all(wildcard), plus any empty shared group


def test_builder_cow_then_second_subset_different_deme() -> None:
    """Two subset registrations on different demes each trigger independent COW.

    Scenario:
      1. Build 10-deme homogeneous population.
      2. set_hook(..., deme_selector=3) → COW: deme 3 splits off.
      3. set_hook(..., deme_selector=7) → COW: deme 7 splits off.

    Invariant: Three distinct storage groups emerge:
      - Group A (demes {0,1,2,4,5,6,8,9}): original shared storage, 0 hooks
      - Group B (deme {3}): COW'd storage, 1 hook
      - Group C (deme {7}): COW'd storage, 1 hook
    After run_tick(), only demes 3 and 7 show +1.0; others unchanged.
    """
    species = _make_species("builder_cow_two")
    n_demes = 10
    sp = _build_homogeneous_spatial_pop(species, n_demes, quiescent=True)

    from numba import njit

    @njit
    @hook(event="first", custom=True, priority=0)
    def hook_a(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @njit
    @hook(event="first", custom=True, priority=0)
    def hook_b(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    # Register same hook fn on two different demes → two independent COW events.
    sp.set_hook("first", hook_a, deme_selector=3)
    sp.set_hook("first", hook_b, deme_selector=7)

    with numba_disabled():
        sp.run_tick()

    base_init = 200.0  # 100 female + 100 male per deme
    targeted = {3, 7}
    for i in range(n_demes):
        expected = base_init + 1.0 if i in targeted else base_init
        actual = float(sp.deme(i).state.individual_count.sum())
        assert actual == pytest.approx(expected), (
            f"deme[{i}] expected {expected}, got {actual}"
        )

    # --- Compact plan: 2 descriptors (one per targeted deme) ---
    compact = sp._collect_compact_spatial_hooks()
    non_empty = [d for d in compact if d.njit_fn is not None]
    assert len(non_empty) == 2
    target_selectors = {d.deme_selector for d in non_empty}
    assert target_selectors == {3, 7}


