"""Check isolation of native parameters and copied declaration metadata together."""

from copy import deepcopy
from typing import Literal

import numpy as np
import pytest

import natal as nt
from tests._config_assertions import assert_config_equal
from tests.test_review_history_regressions import _history_population


@pytest.mark.parametrize("model", ["age", "discrete", "spatial"])
def test_every_mutable_config_field_is_detached(model: Literal["age", "discrete", "spatial"]) -> None:
    """Mutating any exported metadata/native array cannot reach later snapshots."""
    owner = _history_population(f"ConfigSnapshotAllFields_{model}", model)
    pop = owner.demes[0] if isinstance(owner, nt.SpatialPopulation) else owner
    pop.update().custom(flag=True, cohort=7, temperature=2.5, grid=np.arange(12.0).reshape(2, 2, 3))
    expected = pop.config
    before_draft = deepcopy(pop._config)
    snapshot = pop.config
    for name in snapshot._fields:
        value = getattr(snapshot, name)
        previous = getattr(expected, name)
        if isinstance(value, np.ndarray):
            assert isinstance(previous, np.ndarray)
            assert not np.shares_memory(value, previous), name
            assert value.dtype == previous.dtype
            value[...] = np.logical_not(value) if value.dtype == np.bool_ else value + 1
    grid = snapshot.custom["grid"]
    assert isinstance(grid, np.ndarray)
    grid.fill(-1)
    snapshot.custom["cohort"] = 999
    snapshot.custom["new_slot"] = True
    fresh = pop.config
    assert_config_equal(fresh, expected)
    assert_config_equal(pop._config, before_draft)
    assert type(fresh.custom["flag"]) is bool
    assert type(fresh.custom["cohort"]) is int
    assert type(fresh.custom["temperature"]) is float


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_config_equilibrium_presence_and_custom_arrays_remain_native_snapshots(model: Literal["age", "discrete"]) -> None:
    """Optional sentinels and tensors reflect Rust writes without sharing their storage."""
    owner = _history_population(f"ConfigSnapshotOptional_{model}", model)
    assert not isinstance(owner, nt.SpatialPopulation)
    assert owner.config.equilibrium_individual_distribution is None
    assert owner.config.external_expected_eggs is None
    backend = owner._rust_lifecycle_backend
    assert backend is not None
    declared = np.arange(1.0, 2 * owner.config.n_ages + 1).reshape(2, -1)
    backend._session.tensor_write("equilibrium_distribution", declared.ravel())
    backend._session.apply({"external_expected_eggs": 123.0})
    snapshot = owner.config
    assert snapshot.equilibrium_individual_distribution is not None
    np.testing.assert_array_equal(snapshot.equilibrium_individual_distribution, declared)
    assert snapshot.external_expected_eggs == 123.0
    snapshot.equilibrium_individual_distribution.fill(0)
    np.testing.assert_array_equal(owner.config.equilibrium_individual_distribution, declared)
    backend._session.tensor_write("equilibrium_distribution", np.zeros(0))
    backend._session.apply({"external_expected_eggs": -1.0})
    assert owner.config.equilibrium_individual_distribution is None
    assert owner.config.external_expected_eggs is None


def test_spatial_reinitialization_rejects_missing_export_before_replacing_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed deme cannot replace the active session or its seed during handoff."""
    owner = _history_population("ConfigSnapshotInvalidHandoff", "spatial")
    assert isinstance(owner, nt.SpatialPopulation)
    backend = owner._rust_spatial_backend
    seed = owner._rust_spatial_seed
    before = [deme.export_state().copy() for deme in owner.demes]
    with monkeypatch.context() as patch:
        patch.setattr(owner.demes[1], "export_config", None)
        with pytest.raises(TypeError, match=r"deme\[1\].*export_config"):
            owner._initialize_session(seed=19)
    assert owner._rust_spatial_backend is backend
    assert owner._rust_spatial_seed == seed
    for deme, expected in zip(owner.demes, before, strict=True):
        np.testing.assert_array_equal(deme.export_state(), expected)
    owner.run(1)
    assert owner.tick == 1


def test_compact_handoff_sharing_is_confined_to_detached_internal_arrays() -> None:
    """Only compact handoff arrays share; normal exports and live demes remain isolated."""
    owner = _history_population("ConfigSnapshotCompactHandoff", "spatial")
    assert isinstance(owner, nt.SpatialPopulation)
    expected = owner.demes[0].config
    compact = owner._export_deme_drafts(compact=True)
    ordinary = owner._export_deme_drafts()
    assert compact[0].viability_fitness is compact[1].viability_fitness
    assert not np.shares_memory(ordinary[0].viability_fitness, ordinary[1].viability_fitness)
    compact[0].viability_fitness.fill(0)
    for deme, snapshot in zip(owner.demes, ordinary, strict=True):
        np.testing.assert_array_equal(deme.config.viability_fitness, expected.viability_fitness)
        np.testing.assert_array_equal(snapshot.viability_fitness, expected.viability_fitness)
