#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
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
        self.hooks_obj: Any = None
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
        return cast(LifecycleWrappers, self.hooks_obj)

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
        self.hooks_obj: Any = None
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
        return cast(LifecycleWrappers, self.hooks_obj)

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
    calls_np = np.zeros(8, dtype=np.int32)
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

    d1._config = d0.export_config()  # type: ignore[attr-defined]

    from numba import njit

    @hook(event="first", priority=0)
    def first_python(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 0
        observed_first_py_np[idx // 2] = float(state.individual_count[1, 1, 0])
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
    def early_probe(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        idx = int(early_idx_np[0])
        observed_early_probe_np[idx] = float(state.individual_count[1, 1, 0])
        early_idx_np[0] += 1

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.set_hook("first", first_csr)
    spatial.set_hook("first", first_njit)
    spatial.set_hook("first", first_python)
    spatial.set_hook("early", early_probe)

    spatial.run_tick()
    spatial.run(n_steps=1)

    calls = ["py" if x == 0 else "njit" for x in calls_np]
    assert calls == ["py", "njit", "py", "njit", "py", "njit", "py", "njit"]
    assert observed_first_py_np.tolist() == [10.0, 10.0, 0.0, 0.0]
    assert observed_first_njit_np.tolist() == [10.0, 10.0, 0.0, 0.0]
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
    calls_np = np.zeros(4, dtype=np.int32)
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
    def d0_py(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 0
        observed_d0_py_np[0] = float(state.individual_count[1, 1, 0])
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
    def d0_early(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        observed_d0_early_np[0] = float(state.individual_count[1, 1, 0])

    @hook(event="first", priority=2)
    def d1_py(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        idx = int(idx_np[0])
        calls_np[idx] = 2
        observed_d1_py_np[0] = float(state.individual_count[1, 1, 0])
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
    def d1_early(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        observed_d1_early_np[0] = float(state.individual_count[1, 1, 0])

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

from numba import njit


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


def _build_quiescent_age_pop(
    species: nt.Species, n_demes: int, name: str = "quiescent",
) -> SpatialPopulation:
    """Build a homogeneous age-structured population where state only changes via hooks."""
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


def _build_discrete_pop(
    species: nt.Species, n_demes: int, name: str = "discrete",
) -> SpatialPopulation:
    """Build a homogeneous discrete-generation population via builder."""
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


# -----------------------------------------------------------------------
# Compact plan structure tests
# -----------------------------------------------------------------------
def test_compact_plan_folds_identical_sequences_to_wildcard() -> None:
    """All demes sharing one descriptor sequence → single wildcard slot."""
    species = _make_species("compact_wildcard")

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
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]

    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]
    d1.hook_entries = d0.hook_entries  # type: ignore[attr-defined]
    d2.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]
    d2.hook_entries = d0.hook_entries  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    expanded = spatial._collect_effective_compiled_hooks()
    compact = spatial._collect_compact_spatial_hooks()

    assert len(expanded) == 3
    assert {int(desc.deme_selector) for desc in expanded} == {0, 1, 2}

    assert len(compact) == 1
    assert compact[0].deme_selector == "*"
    if nt.is_numba_enabled():
        assert compact[0].njit_fn is not None
    else:
        assert compact[0].py_wrapper is not None


def test_compact_plan_preserves_expanded_view() -> None:
    """Public get_compiled_hooks still returns per-deme pinned descriptors."""
    species = _make_species("compact_expanded_view")

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
    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]
    d1.hook_entries = d0.hook_entries  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    public = spatial.get_compiled_hooks()
    assert len(public) == 2
    assert {int(desc.deme_selector) for desc in public} == {0, 1}


def test_compact_plan_subset_selector() -> None:
    """Descriptor with subset selector stays as tuple, not wildcard."""
    species = _make_species("compact_subset")

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
    d1.compiled_hook_descriptors = d0.compiled_hook_descriptors  # type: ignore[attr-defined]
    d1.hook_entries = d0.hook_entries  # type: ignore[attr-defined]
    d2 = _build_test_deme("d2", species)
    d2._config = d0.export_config()  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1, d2], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 1
    sel = compact[0].deme_selector
    assert isinstance(sel, tuple)
    assert set(sel) == {0, 1}


def test_compact_plan_duplicate_slots_preserved() -> None:
    """Same hook registered twice: two compact slots, executes twice per deme."""
    species = _make_species("compact_dup")

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=2)

    sp.set_hook("first", add_one, hook_id=0)
    sp.set_hook("first", add_one, hook_id=1)

    compact = sp._collect_compact_spatial_hooks()
    assert len(compact) == 2
    assert compact[0].deme_selector == "*"
    assert compact[1].deme_selector == "*"

    with numba_disabled():
        sp.run_tick()

    for i in range(2):
        total = float(sp.deme(i).state.individual_count.sum())
        assert total == 202.0, f"deme[{i}]: expected 202 (base 200 + 2×1), got {total}"


def test_compact_plan_different_order_not_merged() -> None:
    """[A, B] vs [B, A] with same descriptor objects: different order, different results.

    mul2 (×2) and add1 (+1) are non-commutative:
      [mul2, add1] on 100 → 301 total;  [add1, mul2] → 302 total.
    Deme 1 reuses deme 0's descriptors in reversed order so the compact plan
    must produce two separate groups based on order alone.
    """
    species = _make_species("compact_order")

    @njit
    @hook(event="first", custom=True)
    def mul2(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] *= 2.0
        return 0

    @njit
    @hook(event="first", custom=True)
    def add1(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=2)

    sp.deme(0).compiled_hook_descriptors = []
    sp.deme(0).hook_entries = {event: [] for event in sp.deme(0).ALLOWED_EVENTS}
    sp.deme(0).set_hook("first", mul2, hook_id=0)
    sp.deme(0).set_hook("first", add1, hook_id=1)
    descs0 = list(sp.deme(0).compiled_hook_descriptors)

    # Deme 1 reuses the SAME descriptor objects, only the order differs.
    sp.deme(1).compiled_hook_descriptors = [descs0[1], descs0[0]]
    sp.deme(1).hook_entries = {event: [] for event in sp.deme(1).ALLOWED_EVENTS}

    sp._hooks = sp._compile_spatial_hooks_from_demes()

    compact = sp._collect_compact_spatial_hooks()
    selectors = {d.deme_selector for d in compact}
    assert 0 in selectors and 1 in selectors
    assert len(compact) == 4

    with numba_disabled():
        sp.run_tick()

    total0 = float(sp.deme(0).state.individual_count.sum())
    total1 = float(sp.deme(1).state.individual_count.sum())
    assert total0 == 301.0, f"deme[0] mul2→add1: female 100×2+1=201, +100 male = 301, got {total0}"
    assert total1 == 302.0, f"deme[1] add1→mul2: female (100+1)×2=202, +100 male = 302, got {total1}"


def test_compact_plan_empty_hook_sequence_skipped() -> None:
    """Deme with no compiled hooks contributes no descriptors to compact plan."""
    species = _make_species("compact_empty")

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        return 0

    d0 = _build_test_deme("d0", species)
    d0.set_hook("first", my_hook)
    d1 = _build_test_deme("d1", species)
    d1._config = d0.export_config()  # type: ignore[attr-defined]

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)

    compact = spatial._collect_compact_spatial_hooks()
    assert len(compact) == 1
    assert compact[0].deme_selector == 0


# -----------------------------------------------------------------------
# set_hook shared-storage tests
# -----------------------------------------------------------------------
def test_set_hook_shared_storage_registers_once() -> None:
    """Spatial set_hook on shared-storage demes only appends one descriptor."""
    species = _make_species("set_hook_shared")

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)

    count_before = len(sp.deme(0).compiled_hook_descriptors)
    sp.set_hook("first", my_hook)
    count_after = len(sp.deme(0).compiled_hook_descriptors)
    assert count_after == count_before + 1

    compact = sp._collect_compact_spatial_hooks()
    if nt.is_numba_enabled():
        custom_slots = [d for d in compact if d.njit_fn is not None]
    else:
        custom_slots = [d for d in compact if d.py_wrapper is not None]
    assert len(custom_slots) == 1
    assert custom_slots[0].deme_selector == "*"


def test_set_hook_shared_storage_subset_cow_structure() -> None:
    """Subset registration copy-on-writes so non-targeted demes stay clean."""
    species = _make_species("set_hook_cow")

    @njit
    @hook(event="first", custom=True)
    def my_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)

    shared_id = id(sp.deme(0).compiled_hook_descriptors)
    count_before = len(sp.deme(0).compiled_hook_descriptors)

    sp.set_hook("first", my_hook, deme_selector=0)

    assert id(sp.deme(0).compiled_hook_descriptors) != shared_id
    assert len(sp.deme(0).compiled_hook_descriptors) == count_before + 1

    assert id(sp.deme(1).compiled_hook_descriptors) == shared_id
    assert len(sp.deme(1).compiled_hook_descriptors) == count_before
    assert id(sp.deme(2).compiled_hook_descriptors) == shared_id
    assert len(sp.deme(2).compiled_hook_descriptors) == count_before


def test_set_hook_shared_storage_subset_cow_execution() -> None:
    """Subset registration: only targeted deme runs hook, others unchanged."""
    species = _make_species("cow_exec")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp.set_hook("first", add_one, deme_selector=target)

    with numba_disabled():
        sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_subset_python_hook_no_leak() -> None:
    """Python py-wrapper hook on subset does not leak to untargeted demes."""
    species = _make_species("no_leak")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @hook(event="first", priority=0)
    def py_hook(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = config, deme_id
        state.individual_count[0, 1, 0] += 1.0

    with numba_disabled():
        sp.set_hook("first", py_hook, deme_selector=target)
        sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


def test_set_hook_empty_selector_noop() -> None:
    """Empty selector (no matching demes) leaves all state unchanged."""
    species = _make_species("empty_sel")
    sp = _build_quiescent_age_pop(species, n_demes=3)

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    totals_before = [float(sp.deme(i).state.individual_count.sum()) for i in range(3)]
    sp.set_hook("first", add_one, deme_selector=[])

    with numba_disabled():
        sp.run_tick()

    for i in range(3):
        assert float(sp.deme(i).state.individual_count.sum()) == totals_before[i]


def test_set_hook_cow_subsequent_mutation_no_leak() -> None:
    """After COW subset registration, subsequent hook on target stays isolated."""
    species = _make_species("cow_iso")

    @njit
    @hook(event="first", custom=True)
    def hook_a(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @njit
    @hook(event="first", custom=True)
    def hook_b(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[1, 1, 0] += 1.0
        return 0

    sp = _build_quiescent_age_pop(species, n_demes=3)

    # Subset registration on deme 0 triggers COW.
    sp.set_hook("first", hook_a, deme_selector=0)
    count_after_a = len(sp.deme(0).compiled_hook_descriptors)
    count_non_target = len(sp.deme(1).compiled_hook_descriptors)

    # Register hook_b on all demes via wildcard selector.
    sp.set_hook("first", hook_b)
    assert len(sp.deme(0).compiled_hook_descriptors) == count_after_a + 1
    assert len(sp.deme(1).compiled_hook_descriptors) == count_non_target + 1
    assert len(sp.deme(2).compiled_hook_descriptors) == count_non_target + 1

    assert id(sp.deme(0).compiled_hook_descriptors) != id(sp.deme(1).compiled_hook_descriptors)


# -----------------------------------------------------------------------
# Run-tick integration tests
# -----------------------------------------------------------------------
def test_compact_plan_run_tick_deterministic_state() -> None:
    """Quiescent model: hook +1 on age-1 female → total per deme = 200 + 1."""
    species = _make_species("compact_det")
    sp = _build_quiescent_age_pop(species, n_demes=3)

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp.set_hook("first", add_one)

    with numba_disabled():
        sp.run_tick()

    for i in range(3):
        total = float(sp.deme(i).state.individual_count.sum())
        assert total == 201.0, f"deme[{i}]: expected 201, got {total}"
        state = sp.deme(i).state.individual_count
        assert np.all(state >= 0.0)
        assert not np.any(np.isnan(state))


def test_compact_plan_mixed_csr_njit_exact_ordering() -> None:
    """CSR ×2 then njit +1 on age-1 female=100 → 201; reversed order → 202.

    Uses the same descriptor objects for both demes, only the registration
    order differs.  Aging moves age 1 → age 2, so the hook effect is
    observed at age 2 after the tick.  Compact plan must produce two groups.
    """
    species = _make_species("compact_mixed_exact")

    @njit
    @hook(event="early", custom=True)
    def add1(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    @hook(event="early")
    def mul2_csr():
        return [Op.scale(genotypes="WT|WT", ages=1, sex="female", factor=2.0)]

    sp = _build_quiescent_age_pop(species, n_demes=2)

    sp.deme(0).compiled_hook_descriptors = []
    sp.deme(0).hook_entries = {event: [] for event in sp.deme(0).ALLOWED_EVENTS}
    sp.deme(0).set_hook("early", mul2_csr, hook_id=0)
    sp.deme(0).set_hook("early", add1, hook_id=1)

    sp.deme(1).compiled_hook_descriptors = []
    sp.deme(1).hook_entries = {event: [] for event in sp.deme(1).ALLOWED_EVENTS}
    sp.deme(1).set_hook("early", add1, hook_id=0)
    sp.deme(1).set_hook("early", mul2_csr, hook_id=1)
    # Share descriptors for the order-sensitive compact test.
    descs0 = list(sp.deme(0).compiled_hook_descriptors)
    sp.deme(1).compiled_hook_descriptors = [descs0[1], descs0[0]]

    sp._hooks = sp._compile_spatial_hooks_from_demes()

    with numba_disabled():
        sp.run_tick()

    # d0: mul2_csr → add1: female[age=1] 100×2=200 +1=201 → aging to age 2
    assert float(sp.deme(0).state.individual_count[0, 2, 0]) == 201.0
    # d1: add1 → mul2_csr: female[age=1] (100+1)=101 ×2=202 → aging to age 2
    assert float(sp.deme(1).state.individual_count[0, 2, 0]) == 202.0

    compact = sp._collect_compact_spatial_hooks()
    selectors = {d.deme_selector for d in compact}
    assert 0 in selectors and 1 in selectors
    assert len(compact) == 4


def test_builder_homogeneous_demes_share_compiled_hooks() -> None:
    """Builder-created homogeneous population: all demes share compiled_hook_descriptors."""
    species = _make_species("builder_share")
    n_demes = 5
    sp = _build_discrete_pop(species, n_demes)

    ref_id = id(sp.deme(0).compiled_hook_descriptors)
    for i in range(1, n_demes):
        assert id(sp.deme(i).compiled_hook_descriptors) == ref_id

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


def test_builder_set_hook_subset_cow_combined() -> None:
    """Subset registration via builder-created population: COW isolates target."""
    species = _make_species("builder_cow")
    n_demes = 5
    target = 2
    sp = _build_quiescent_age_pop(species, n_demes)

    @njit
    @hook(event="first", custom=True)
    def add_one(state, config, deme_id):  # type: ignore[no-untyped-def]
        _ = deme_id
        state.individual_count[0, 1, 0] += 1.0
        return 0

    sp.set_hook("first", add_one, deme_selector=target)

    with numba_disabled():
        sp.run_tick()

    for i in range(n_demes):
        total = float(sp.deme(i).state.individual_count.sum())
        expected = 201.0 if i == target else 200.0
        assert total == expected, f"deme[{i}]: {total} != {expected}"


# -----------------------------------------------------------------------
# 100-deme Numba/prange subprocess regression test
# -----------------------------------------------------------------------
_PRANGE_TEST_CODE = '''
import os
import sys

import numpy as np
import natal as nt
from natal.hooks import hook, RESULT_STOP
from natal.numba import binomial

sp = nt.Species.from_dict(
    name="An", structure={"chr1": {"A": ["WT", "Drive"]}},
    somatic_labels=["S", "E", "I"], gamete_labels=["default"], unordered=False,
)

S_IDX = 0
P = 0.3
N_DEMES = 100

@hook(event="early", custom=True)
def infect_susceptible_females(state, config, deme_id=-1):
    _ = deme_id
    females = state.individual_count[0]
    sperm = state.sperm_storage
    for age in range(config.new_adult_age, config.n_ages):
        n_s = int(round(females[age, S_IDX]))
        n_mated = 0
        for mi in range(config.n_ztypes):
            n_mated += int(round(sperm[age, S_IDX, mi]))
        n_virgins = n_s - n_mated
        if n_virgins < 0:
            return RESULT_STOP
        n_moved = int(binomial(n_virgins, P))
        n_moved_sperm = 0
        for mi in range(config.n_ztypes):
            nb = int(round(sperm[age, S_IDX, mi]))
            nbm = int(binomial(nb, P))
            sperm[age, S_IDX, mi] -= nbm
            sperm[age, 2, mi] += nbm
            n_moved_sperm += nbm
        females[age, S_IDX] -= n_moved
        females[age, 2] += n_moved
    return 0

builder = (
    nt.SpatialPopulation.builder(sp, n_demes=N_DEMES)
    .setup(stochastic=True).age_structure(n_ages=8, new_adult_age=1)
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

registry = pop.hooks.hooks.registry
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

sys.exit(0)
'''


def test_prange_100_deme_no_crash() -> None:
    """100-deme homogeneous population with custom hook + binomial under prange."""
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = "2"
    result = subprocess.run(
        [sys.executable, "-c", _PRANGE_TEST_CODE],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, (
        f"subprocess returned {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
