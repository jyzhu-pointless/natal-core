#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, cast

import numpy as np

import natal as nt
from natal.base_population import BasePopulation
from natal.numba_utils import numba_disabled
from natal.genetic_structures import Species
from natal.hook_dsl import CompiledEventHooks, Op, hook
from natal import numba_compat as nbc
from natal.population_config import PopulationConfig
from natal.population_state import DiscretePopulationState, PopulationState
from natal.spatial_population import SpatialPopulation


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
        return cast(CompiledEventHooks, self._hooks_obj)

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        if event_name == "finish":
            self.finish_events += 1
        return 0


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
        return cast(CompiledEventHooks, self._hooks_obj)

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        if event_name == "finish":
            self.finish_events += 1
        return 0


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
            female_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
            male_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
        )
        .reproduction(
            female_age_based_mating_rates=[0.0, 0.0, 0.0, 0.0],
            male_age_based_mating_rates=[0.0, 0.0, 0.0, 0.0],
            eggs_per_female=0.0,
            use_sperm_storage=False,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_adult_females=100,
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


def test_spatial_population_stochastic_discrete_migration_preserves_integer_counts():
    species = _make_species("spatial_run_stochastic_discrete")
    shared_config = _make_population_config(species)._replace(
        is_stochastic=True,
        use_continuous_sampling=False,
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


def test_spatial_population_stochastic_age_migration_preserves_sperm_consistency():
    species = _make_species("spatial_run_stochastic_age")
    shared_config = _make_population_config(species)._replace(
        is_stochastic=True,
        use_continuous_sampling=False,
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
            .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
            .build()
        )

    d0 = _build_deme("mixed_d0")
    d1 = _build_deme("mixed_d1")

    # Spatial kernels require one shared config object.
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
    def first_njit(ind_count, tick, deme_id):  # type: ignore[no-untyped-def]
        _ = (tick, deme_id)
        idx = int(idx_np[0])
        calls_np[idx] = 1
        observed_first_njit_np[idx // 2] = float(ind_count[1, 1, 0])
        ind_count[1, 1, 0] += 2.0
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
            .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
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
            .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
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
    def d0_njit(ind_count, tick, deme_id):  # type: ignore[no-untyped-def]
        _ = (tick, deme_id)
        idx = int(idx_np[0])
        calls_np[idx] = 1
        ind_count[1, 1, 0] += 2.0
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
    def d1_njit(ind_count, tick, deme_id):  # type: ignore[no-untyped-def]
        _ = (tick, deme_id)
        idx = int(idx_np[0])
        calls_np[idx] = 3
        ind_count[1, 1, 0] += 4.0
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
            .survival(female_age0_survival=1.0, male_age0_survival=1.0, adult_survival=1.0)
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
