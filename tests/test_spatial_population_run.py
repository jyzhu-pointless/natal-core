#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import natal as nt
from natal.base_population import BasePopulation
from natal.genetic_structures import Species
from natal.hook_dsl import CompiledEventHooks, Op, hook
from natal import numba_compat as nbc
from natal.population_state import DiscretePopulationState, PopulationState
from natal.spatial_population import SpatialPopulation


class _RunDemePopulation(BasePopulation):
    def __init__(self, species: Species, name: str, config):
        self._species = species
        self._name = name
        self._tick = 0
        self._history = []
        self._finished = False
        self._config = config
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


def test_spatial_population_run_tick_updates_all_demes():
    species = _make_species("spatial_run_tick")
    shared_config = object()

    d0 = _RunDemePopulation(species, "d0", shared_config)
    d1 = _RunDemePopulation(species, "d1", shared_config)

    def _run_spatial_tick(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
    ):
        assert config is shared_config
        return (ind_count_all + 1.0, sperm_store_all + 2.0, int(tick) + 1), 0

    def _run_spatial(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        n_ticks,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
        record_interval,
    ):
        return (ind_count_all, sperm_store_all, int(tick)), None, False

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)
    sp.run_tick()

    assert sp.tick == 1
    assert d0.tick == 1 and d1.tick == 1
    assert float(d0.state.individual_count.sum()) == 2.0
    assert float(d1.state.individual_count.sum()) == 2.0
    assert float(d0.state.sperm_storage.sum()) == 2.0
    assert float(d1.state.sperm_storage.sum()) == 2.0


def test_spatial_population_run_stop_marks_finish():
    species = _make_species("spatial_run_stop")
    shared_config = object()

    d0 = _RunDemePopulation(species, "d0", shared_config)
    d1 = _RunDemePopulation(species, "d1", shared_config)

    def _run_spatial_tick(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
    ):
        return (ind_count_all, sperm_store_all, int(tick)), 0

    def _run_spatial(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        n_ticks,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
        record_interval,
    ):
        return (ind_count_all + 3.0, sperm_store_all, int(tick) + 2), None, True

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

    sp = SpatialPopulation([d0, d1], migration_rate=0.0)
    sp.run(n_steps=5, record_every=1)

    assert sp.tick == 2
    assert d0.tick == 2 and d1.tick == 2
    assert d0._finished and d1._finished
    assert d0.finish_events == 1 and d1.finish_events == 1


def test_spatial_population_stochastic_discrete_migration_preserves_integer_counts():
    species = _make_species("spatial_run_stochastic_discrete")
    shared_config = SimpleNamespace(
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

    def _run_spatial_tick(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
    ):
        return (ind_count_all, sperm_store_all, int(tick) + 1), 0

    def _run_spatial(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        n_ticks,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
        record_interval,
    ):
        return (ind_count_all, sperm_store_all, int(tick)), None, False

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

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
    shared_config = SimpleNamespace(
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

    def _run_spatial_tick(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
    ):
        return (ind_count_all, sperm_store_all, int(tick) + 1), 0

    def _run_spatial(
        ind_count_all,
        sperm_store_all,
        config,
        registry,
        tick,
        n_ticks,
        adjacency,
        migration_mode,
        topology_rows,
        topology_cols,
        topology_wrap,
        migration_kernel,
        kernel_include_center,
        migration_rate,
        record_interval,
    ):
        return (ind_count_all, sperm_store_all, int(tick)), None, False

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

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


def test_spatial_mixed_hook_priority_ordering_runs_in_run_tick_and_run():
    species = _make_species("spatial_mixed_priority")
    calls: list[str] = []
    observed: dict[str, list[float]] = {
        "first_py": [],
        "first_njit": [],
        "early_probe": [],
    }

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

    @hook(event="first", priority=0)
    def first_python(population):  # type: ignore[no-untyped-def]
        calls.append("py")
        observed["first_py"].append(float(population.state.individual_count[1, 1, 0]))

    @hook(event="first", priority=1, numba=True)
    def first_njit(ind_count, tick, deme_id):  # type: ignore[no-untyped-def]
        _ = (tick, deme_id)
        calls.append("njit")
        observed["first_njit"].append(float(ind_count[1, 1, 0]))
        ind_count[1, 1, 0] += 2.0
        return 0

    @hook(event="first", priority=2)
    def first_csr():
        return [Op.add(genotypes="WT|WT", ages=1, sex="male", delta=3.0)]

    @hook(event="early", priority=0)
    def early_probe(population):  # type: ignore[no-untyped-def]
        observed["early_probe"].append(float(population.state.individual_count[1, 1, 0]))

    spatial = SpatialPopulation([d0, d1], migration_rate=0.0)
    spatial.set_hook("first", first_csr)
    spatial.set_hook("first", first_njit)
    spatial.set_hook("first", first_python)
    spatial.set_hook("early", early_probe)

    spatial.run_tick()
    spatial.run(n_steps=1)

    # 2 demes * 2 ticks: py/njit each called 4 times, per-deme order fixed.
    assert calls == ["py", "njit", "py", "njit", "py", "njit", "py", "njit"]
    assert observed["first_py"] == [10.0, 10.0, 0.0, 0.0]
    assert observed["first_njit"] == [10.0, 10.0, 0.0, 0.0]
    # early probes confirm csr (+3) is applied after njit (+2) each tick.
    assert observed["early_probe"] == [15.0, 15.0, 5.0, 5.0]
