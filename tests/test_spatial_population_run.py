#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from natal.base_population import BasePopulation
from natal.genetic_structures import Species
from natal.hook_dsl import CompiledEventHooks
from natal.population_state import PopulationState
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

    def _run_spatial_tick(ind, sperm, config, registry, tick, adjacency, migration_rate):
        assert config is shared_config
        return (ind + 1.0, sperm + 2.0, int(tick) + 1), 0

    def _run_spatial(ind, sperm, config, registry, tick, n_ticks, adjacency, migration_rate, record_interval):
        return (ind, sperm, int(tick)), None, False

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

    sp = SpatialPopulation([d0, d1], migration_rate=0.1)
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

    def _run_spatial_tick(ind, sperm, config, registry, tick, adjacency, migration_rate):
        return (ind, sperm, int(tick)), 0

    def _run_spatial(ind, sperm, config, registry, tick, n_ticks, adjacency, migration_rate, record_interval):
        return (ind + 3.0, sperm, int(tick) + 2), None, True

    hooks = SimpleNamespace(
        run_spatial_tick_fn=_run_spatial_tick,
        run_spatial_fn=_run_spatial,
        registry=object(),
    )
    d0._hooks_obj = hooks
    d1._hooks_obj = hooks

    sp = SpatialPopulation([d0, d1], migration_rate=0.2)
    sp.run(n_steps=5, record_every=1)

    assert sp.tick == 2
    assert d0.tick == 2 and d1.tick == 2
    assert d0._finished and d1._finished
    assert d0.finish_events == 1 and d1.finish_events == 1
