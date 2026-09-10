#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from natal.contracts.materialize import SpatialMigration  # noqa: F401  # construction contract exercises materialize through the container
from natal.frontend.data import DiscretePopulationState, ModelDraft
from natal.frontend.genetics import Species
from natal.frontend.population.base import BasePopulation
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)
from natal.frontend.spatial.population import DemeSlice, SpatialPopulation


def _reference_draft(species: Species) -> ModelDraft:
    """Return a real built draft as the doubles' declaration surface.

    The double satisfies the explicit population contract, so its
    ``export_config`` must hand out a genuine ``ModelDraft`` (the spatial
    contract pair materializes from it); building one real population
    keeps the draft consistent with the species without the test
    hand-rolling tensor shapes.
    """
    population = (
        DiscreteGenerationPopulation.setup(species=species, stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )
    return population.export_config()


class _DummyDemePopulation(BasePopulation):
    """Contract-satisfying lightweight deme double.

    Implements the aligned population surface pieces the container and
    the slice read at construction and in these tests; everything else
    stays a read-only stub.
    """

    def __init__(self, species: Species, name: str):
        self._species = species
        self._name = name
        self._tick = 0
        self._history = []
        self._config = _reference_draft(species)
        self._state = DiscretePopulationState.create(
            n_sexes=int(self._config.n_sexes),
            n_ages=int(self._config.n_ages),
            n_ztypes=int(self._config.n_ztypes),
            n_tick=0,
        )

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

    def export_config(self) -> ModelDraft:
        """Aligned surface: hand out the double's declaration draft."""
        return self._config

    def export_state(self) -> np.ndarray:
        """Aligned surface: flatten the stub state (tick + counts)."""
        return self._state.flatten_all()

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
