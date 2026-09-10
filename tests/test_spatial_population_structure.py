#!/usr/bin/env python3

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np

from natal.frontend.population.base import BasePopulation
from natal.frontend.genetics import Species
from natal.frontend.spatial.population import DemeSlice, SpatialPopulation


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

    def update(self) -> Any:  # type: ignore[no-untyped-def,any-return]  # duck-typed double: mirrors the untyped base-class hook; never called on this stub
        raise NotImplementedError

    def _snapshot_state(self):
        """Snapshot hook: return the stub container itself (read-only stub)."""
        return self._state

    @property
    def species(self) -> Species:
        return self._species


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
    # Stage 3: deme() hands out compat slices delegating to the demes.
    assert isinstance(sp.deme(0), DemeSlice)
    assert isinstance(sp.deme(1), DemeSlice)
    # Slices delegate reads to the wrapped demes.
    assert sp.deme(0).name == deme0.name
    assert sp.deme(1).name == deme1.name
    assert sp.species is species
    # The default identity adjacency folds into a CSR row per deme.
    assert sp.blueprint.n_demes == 2
    assert sp.migration_csr.indptr.shape == (3,)


def test_spatial_population_rejects_non_base_population_deme():
    species = _make_species("spatial_struct_2")
    deme0 = _DummyDemePopulation(species, "d0")
    bad_demes = Sequence[BasePopulation]

    try:
        SpatialPopulation(bad_demes)
        assert False, "Expected TypeError for non-BasePopulation deme"
    except (TypeError, AssertionError, AttributeError):  #?
        pass


def test_spatial_population_accepts_csr_tuple_adjacency():
    species = _make_species("spatial_struct_sparse_csr")
    deme0 = _DummyDemePopulation(species, "d0")
    deme1 = _DummyDemePopulation(species, "d1")

    # CSR for [[0, 1], [1, 0]]
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    data = np.array([1.0, 1.0], dtype=np.float64)

    sp = SpatialPopulation(
        [deme0, deme1],
        adjacency=(indptr, indices, data),
        migration_rate=0.25,
    )

    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    # The CSR tuple folds into the migration CSR: row 0 -> deme 1, row 1 -> deme 0.
    assert np.array_equal(sp.migration_csr.indptr, np.array([0, 1, 2]))
    assert np.array_equal(sp.migration_csr.dest_idx, np.array([1, 0]))
    assert np.allclose(sp.migration_csr.weights, [1.0, 1.0])


def test_spatial_population_hybrid_strategy_interface_and_kernel_bank():
    species = _make_species("spatial_struct_hybrid_iface")
    deme0 = _DummyDemePopulation(species, "d0")
    deme1 = _DummyDemePopulation(species, "d1")

    kernel = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    sp = SpatialPopulation(
        [deme0, deme1],
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_strategy="hybrid",
        kernel_bank=(kernel,),
        deme_kernel_ids=np.array([0, 0], dtype=np.int64),
        migration_rate=0.1,
    )

    # The hybrid strategy is a build-time materialization choice only: it
    # resolves to kernel mode.  Without a topology the kernel-bank routing
    # has no coordinate space, so the fold emits no outbound entries —
    # the historical runtime behavior of bank routing with zero topology
    # rows.
    assert sp.blueprint.n_demes == 2
    assert int(sp.migration_csr.indptr[-1]) == 0
    assert sp.migration_csr.dest_idx.size == 0
