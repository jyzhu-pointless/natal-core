"""Cross-field validation for density-dependent growth rates.

The scalar bounds remain broad because ``none`` and ``fixed`` do not consume
``r``.  Compensatory curves require a finite intrinsic rate ``r >= 1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr": {"locus": ["A", "a"]}},
        gamete_labels=["default"],
    )


def _builder(name: str) -> nt.PopulationBuilder:
    return (
        nt.PopulationBuilder.from_species(_species(name))
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"A|A": 10},
                "male": {"A|A": 10},
            }
        )
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .survival(female_age_based_survival=0.5, male_age_based_survival=0.5)
    )


@pytest.mark.parametrize("mode", ["linear", "logistic", "beverton_holt", "ricker"])
def test_compensatory_modes_accept_boundary_r_one(mode: str) -> None:
    config = _builder(f"r_one_{mode}").competition(
        growth_mode=mode,
        carrying_capacity=100.0,
        low_density_growth_rate=1.0,
    ).config
    assert config.low_density_growth_rate == 1.0


@pytest.mark.parametrize("mode", ["linear", "logistic", "beverton_holt", "ricker"])
@pytest.mark.parametrize("bad_r", [0.5, math.nan, math.inf])
def test_compensatory_modes_reject_invalid_r(mode: str, bad_r: float) -> None:
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        _builder(f"bad_{mode}_{bad_r}").competition(
            growth_mode=mode,
            carrying_capacity=100.0,
            low_density_growth_rate=bad_r,
        )


@pytest.mark.parametrize("mode", ["no_competition", "fixed"])
def test_none_and_fixed_retain_low_r_domain(mode: str) -> None:
    config = _builder(f"low_r_{mode}").competition(
        growth_mode=mode,
        carrying_capacity=100.0,
        low_density_growth_rate=0.5,
    ).config
    assert config.low_density_growth_rate == 0.5


def test_mode_switch_and_atomic_runtime_update() -> None:
    pop = _builder("runtime_growth_contract").competition(
        growth_mode="fixed", carrying_capacity=100.0, low_density_growth_rate=0.5
    ).build()
    pop.update().competition(growth_mode="beverton_holt", low_density_growth_rate=1.0)
    assert pop.params.growth_mode == 3
    assert pop.params.low_density_growth_rate == 1.0

    before = (pop.params.growth_mode, pop.params.low_density_growth_rate)
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        pop.update().competition(growth_mode="ricker", low_density_growth_rate=0.5)
    assert (pop.params.growth_mode, pop.params.low_density_growth_rate) == before


def test_params_mode_switch_requires_valid_existing_r() -> None:
    pop = _builder("params_growth_contract").competition(
        growth_mode="fixed", carrying_capacity=100.0, low_density_growth_rate=0.5
    ).build()
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        pop.params.growth_mode = 4
    assert pop.params.growth_mode == 1
    assert pop.params.low_density_growth_rate == 0.5


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("bad_r", [0.5, math.nan, math.inf])
def test_native_growth_batch_rejects_before_any_commit(model: str, bad_r: float) -> None:
    """Bypassing Python guards cannot commit invalid r or a valid sibling write."""
    from tests.test_review_runtime_regressions import _population

    pop = _population(f"native_r_{model}_{bad_r}", model, stochastic=False)
    session = pop._rust_lifecycle_backend._session
    before = session.get_scalar("carrying_capacity")
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        session.apply({"carrying_capacity": 123, "low_density_growth_rate": bad_r})
    assert session.get_scalar("carrying_capacity") == before
    assert session.get_scalar("low_density_growth_rate") == 2.0
    # Validate the final pair, independent of mapping iteration order.
    session.apply({"low_density_growth_rate": 0.5, "growth_mode": 1})
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        session.apply({"growth_mode": 4})
    assert session.get_scalar("growth_mode") == 1
    session.apply({"low_density_growth_rate": 1, "growth_mode": 4})
    assert session.get_scalar("growth_mode") == 4


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_invalid_growth_hook_preserves_event_state_and_parameters(model: str) -> None:
    """An invalid first-event r rolls back the whole ecology scratch commit."""
    from tests.test_review_runtime_regressions import _population

    pop = _population(
        f"hook_r_{model}", model, stochastic=False,
        hook_calls=[((nt.Op.set_param("carrying_capacity", 123),
                      nt.Op.set_param("low_density_growth_rate", 0.5)), {"event": "first"})],
    )
    before = pop.state.individual_count.copy()
    log_before = pop.params_log
    with pytest.raises((ValueError, RuntimeError), match="low_density_growth_rate"):
        pop.run(1)
    assert pop.params.carrying_capacity == 100000
    assert pop.params.low_density_growth_rate == 2
    np.testing.assert_array_equal(pop.state.individual_count, before)
    assert pop.params_log == log_before


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
def test_spatial_growth_update_rejects_without_mutating_other_demes(model: str) -> None:
    """A rejected deme update preserves ecology and permits a valid pair switch."""
    builder = (
        nt.SpatialPopulation.builder(_species(f"spatial_r_{model}"), n_demes=2, pop_type=model)
        .setup(stochastic=False)
    )
    if model == "age_structured":
        builder = builder.age_structure(n_ages=3, new_adult_age=1)
    pop = builder.competition(carrying_capacity=100, low_density_growth_rate=2).build()
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        pop.deme(1).update().competition(carrying_capacity=123, low_density_growth_rate=0.5)
    for deme in pop.demes:
        assert deme.params.carrying_capacity == 100
        assert deme.params.low_density_growth_rate == 2
    pop.deme(1).update().competition(growth_mode="fixed", low_density_growth_rate=0.5)
    assert pop.demes[1].params.low_density_growth_rate == 0.5
    assert pop.demes[0].params.low_density_growth_rate == 2


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("mode", [2, 3, 4])
def test_native_spatial_constructor_validates_every_deme_growth_pair(model: str, mode: int) -> None:
    """Raw ecology columns cannot hide an invalid rate in a later deme."""
    from natal import _engine_rs
    from natal.backends.rust.rust_backend import ecology_columns_from_drafts, genetics_variant_bank
    from natal.contracts.materialize import SpatialMigration, materialize
    from tests.test_review_runtime_regressions import _population

    pop = _population(f"raw_r_{model}_{mode}", model, stochastic=False)
    draft = pop.config
    columns = ecology_columns_from_drafts([draft, draft])
    columns["growth_mode"] = np.array([mode, mode], dtype=np.int64)
    columns["low_density_growth_rate"] = np.array([2.0, 0.5])
    bank, ids = genetics_variant_bank([draft, draft])
    migration = SpatialMigration(
        indptr=np.zeros(3, dtype=np.int64), dest_idx=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0), rate=np.zeros((2, 2, draft.n_ages)),
    )
    blueprint = materialize(draft, migration).blueprint
    with pytest.raises(ValueError, match="low_density_growth_rate"):
        _engine_rs.HeterogeneousSpatialEngineSession(
            blueprint, columns, bank, ids,
            np.stack([draft.initial_individual_count] * 2),
            np.stack([draft.initial_sperm_storage] * 2), 0,
        )
