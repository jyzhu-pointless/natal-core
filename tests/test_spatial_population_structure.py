#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from typing import Sequence, cast

from natal.base_population import BasePopulation
from natal.genetic_structures import Species
from natal.spatial_population import SpatialPopulation


class _DummyDemePopulation(BasePopulation):
    def __init__(self, species: Species, name: str):
        self._species = species
        self._name = name
        self._tick = 0
        self._history = []
        self._state = type("S", (), {"individual_count": np.zeros((2, 1, 1), dtype=np.float64)})()

    def clear_history(self) -> None:
        self._history.clear()

    def run_tick(self):
        self._tick += 1
        return self

    def get_total_count(self) -> int:
        return 0

    def get_female_count(self) -> int:
        return 0

    def get_male_count(self) -> int:
        return 0

    def run(self, n_steps: int, record_every: int = 1, finish: bool = False):
        self._tick += int(n_steps)
        return self

    def reset(self) -> None:
        self._tick = 0


def _make_species(prefix: str = "SpatialPopSpecies") -> Species:
    return Species.from_dict(
        prefix,
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


def test_spatial_population_is_not_base_population_subclass():
    assert not issubclass(SpatialPopulation, BasePopulation)


def test_spatial_population_demes_must_be_base_population_instances():
    species = _make_species("spatial_struct_1")
    deme0 = _DummyDemePopulation(species, "d0")
    deme1 = _DummyDemePopulation(species, "d1")

    sp = SpatialPopulation([deme0, deme1], migration_rate=0.25)

    assert sp.n_demes == 2
    assert isinstance(sp.deme(0), BasePopulation)
    assert isinstance(sp.deme(1), BasePopulation)
    assert sp.species is species
    assert sp.adjacency.shape == (2, 2)


def test_spatial_population_rejects_non_base_population_deme():
    species = _make_species("spatial_struct_2")
    deme0 = _DummyDemePopulation(species, "d0")
    bad_demes = cast(Sequence[BasePopulation], [deme0, object()])

    try:
        SpatialPopulation(bad_demes)
        assert False, "Expected TypeError for non-BasePopulation deme"
    except TypeError:
        pass
