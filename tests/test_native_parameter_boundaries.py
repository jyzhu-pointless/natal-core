"""Native parameter validation and owned stage contracts, bypassing Python guards."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_review_runtime_regressions import ModelKind, _population


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("field,value", [("growth_mode", 1.5), ("growth_mode", 8), ("external_expected_eggs", -2), ("external_expected_eggs", np.nan)])
def test_native_scalar_batch_rejects_invalid_domains_atomically(model: ModelKind, field: str, value: float) -> None:
    """Native validation must reject domains before changing a valid sibling field."""
    pop = _population(f"NativeDomain_{model}_{field}_{value}", model)
    session = pop._rust_lifecycle_backend._session
    old = session.get_scalar("carrying_capacity")
    with pytest.raises(ValueError):
        session.apply({"carrying_capacity": 500, field: value})
    assert session.get_scalar("carrying_capacity") == old


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_custom_replacement_validates_and_detaches(model: ModelKind) -> None:
    """The native public slot setter preserves dtype and never aliases Python arrays."""
    pop = _population(f"NativeCustomSetter_{model}", model)
    session = pop._rust_lifecycle_backend._session
    data = np.array([[2.0, 3.0]])
    session.set_custom_slots({"flag": False, "n": 42, "rate": 0.5, "array": data})
    data[:] = -999
    first = session.get_custom_slots()
    assert first["flag"] is False and type(first["n"]) is int
    np.testing.assert_array_equal(first["array"], [[2, 3]])
    with pytest.raises((TypeError, ValueError)):
        session.set_custom_slots({"array": object()})
    first["array"][:] = 0
    np.testing.assert_array_equal(session.get_custom_slots()["array"], [[2, 3]])


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_parameter_tensor_queries_are_isolated_and_reject_unknown_fields(model: ModelKind) -> None:
    """Ecology query columns are detached, including optional empty declarations."""
    pop = _population(f"NativeQuery_{model}", model)
    session = pop._rust_lifecycle_backend._session
    for field in ("survival_rates", "mating_rates", "reproduction_rates", "fertility", "competition_weights", "equilibrium_distribution", "migration_rate"):
        before = session.get_tensor(field)
        expected = before.copy()
        before[:] = -100
        np.testing.assert_array_equal(session.get_tensor(field), expected)
    with pytest.raises(KeyError):
        session.get_tensor("not_a_parameter")
    with pytest.raises(KeyError):
        session.get_scalar("not_a_parameter")


def test_native_age_stages_obey_mendelian_survival_and_aging_limits() -> None:
    """100 wild-type mothers lay two eggs each at sex ratio 1/2; adults survive at .8."""
    pop = _population("NativeAgeStages", "age", stochastic=False)
    session = pop._rust_lifecycle_backend._session
    shape = pop._live_state().individual_count.shape
    session.reproduction()
    _, counts, _ = session.state_snapshot()
    np.testing.assert_array_equal(np.asarray(counts).reshape(shape)[:, 0, :], [[100, 0, 0], [100, 0, 0]])
    session.survival()
    _, counts, _ = session.state_snapshot()
    values = np.asarray(counts).reshape(shape)
    np.testing.assert_array_equal(values[:, 1, 0], [80, 80])
    session.aging()
    tick, counts, _ = session.state_snapshot()
    values = np.asarray(counts).reshape(shape)
    assert tick == 0
    np.testing.assert_array_equal(values[:, 2, 0], [80, 80])
    np.testing.assert_array_equal(values[:, 0, :], 0)


def test_spatial_explicit_event_logs_before_first_run() -> None:
    """A manual native event has an audit owner even without prior recording."""
    import natal as nt

    species = nt.Species.from_dict(name="NativeExplicitSpatialLog", structure={"chr": {"loc": ["WT", "Dr"]}})
    @nt.hook(event="early")
    def set_capacity() -> list[object]:
        return [nt.Op.set_param("carrying_capacity", 123)]

    spatial = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
               .competition(carrying_capacity=800)
               .hooks(set_capacity)
               .build())
    spatial.trigger_event("early", deme_id=1)
    assert spatial.demes[1].params.carrying_capacity == 123
    assert spatial.demes[0].params.carrying_capacity == 800
    assert spatial.demes[1].params_log[-1][1:] == ("carrying_capacity", 800, 123)
    assert spatial.demes[0].params_log == ()


def test_ordinary_equilibrium_checkpoint_restores_declaration_and_derive_mode() -> None:
    """Native presence, public snapshots, and continuation agree across both restores."""
    pop = _population("OrdinaryEquilibriumRestore", "age", stochastic=False)
    control = _population("OrdinaryEquilibriumControl", "age", stochastic=False)
    pop.record_snapshot()
    declared = np.array([[0., 100., 0.], [0., 100., 0.]])
    pop.update().competition(equilibrium_distribution=declared)
    np.testing.assert_array_equal(pop.config.equilibrium_individual_distribution, declared)
    pop.run(1)
    pop.restore_checkpoint(0)
    assert pop.config.equilibrium_individual_distribution is None
    pop.run(1)
    control.run(1)
    np.testing.assert_array_equal(pop.export_state(), control.export_state())
    pop.update().competition(equilibrium_distribution=declared)
    pop.clear_history()
    pop.record_snapshot()
    pop._rust_lifecycle_backend._session.tensor_write("equilibrium_distribution", np.zeros(0))
    assert pop.config.equilibrium_individual_distribution is None
    pop.restore_checkpoint(1)
    np.testing.assert_array_equal(pop.config.equilibrium_individual_distribution, declared)


@pytest.mark.parametrize("presence,expected", [(np.array([2, 0]), "0 or 1"), (np.array([1, 0]), "presence"), (np.array([0]), "expected")])
def test_native_equilibrium_presence_rejects_inconsistent_columns(presence: np.ndarray, expected: str) -> None:
    """Invalid flags, missing backing data, and wrong deme counts fail construction."""
    from natal import _engine_rs
    from natal.backends.rust.rust_backend import ecology_columns_from_drafts, genetics_variant_bank
    from natal.contracts.materialize import SpatialMigration, materialize

    pop = _population("InvalidPresenceColumns", "age", stochastic=False)
    draft = pop.config
    columns = ecology_columns_from_drafts([draft, draft])
    columns["equilibrium_declared"] = presence.astype(np.int64)
    bank, ids = genetics_variant_bank([draft, draft])
    migration = SpatialMigration(indptr=np.zeros(3, dtype=np.int64), dest_idx=np.zeros(0, dtype=np.int64), weights=np.zeros(0), rate=np.zeros((2, 2, 3)))
    blueprint = materialize(draft, migration).blueprint
    with pytest.raises(ValueError, match=expected):
        _engine_rs.HeterogeneousSpatialEngineSession(blueprint, columns, bank, ids, np.stack([draft.initial_individual_count] * 2), np.stack([draft.initial_sperm_storage] * 2), 0)
