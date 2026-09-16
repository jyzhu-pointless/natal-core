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
        hook_calls=[((nt.Op.set_param("carrying_capacity", 123, event="first"),
                      nt.Op.set_param("low_density_growth_rate", 0.5, event="first")), {})],
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


# ---------------------------------------------------------------------------
# Equilibrium calibration and sex-specific juvenile survival
# ---------------------------------------------------------------------------
# The derived reference distribution splits the age-1 total by the *surviving*
# sex ratio -- the offspring sex ratio filtered by each sex's own age-0
# survival -- so the calibrated equilibrium is the declared carrying capacity
# even when the two sexes survive differently.  Splitting by the raw sex ratio
# instead made the realised equilibrium miss K by 2.8%-37.5% (larger as r
# approaches 1) whenever `female_age0_survival != male_age0_survival`.
# Equal-survival models are unaffected bit-for-bit; their convergence is
# covered by tests/test_default_growth_mode.py.

_DISCRETE_CASES = [
    # (growth_mode, sex_ratio, female age-0 survival, male age-0 survival, r)
    ("beverton_holt", 0.5, 0.9, 0.8, 3.0),
    ("beverton_holt", 0.5, 0.8, 0.3, 3.0),
    ("beverton_holt", 0.4, 0.8, 0.3, 2.0),
    ("ricker", 0.5, 0.9, 0.8, 3.0),
]


@pytest.mark.parametrize("growth_mode,sex_ratio,s_f,s_m,r", _DISCRETE_CASES)
def test_discrete_equilibrium_reaches_k_under_sex_specific_survival(
    growth_mode: str, sex_ratio: float, s_f: float, s_m: float, r: float
) -> None:
    """Discrete engine: the adult equilibrium is exactly K.

    Catches a reference distribution whose age-1 sex split ignores the
    per-sex age-0 survival: with (0.5, 0.9, 0.8, 3) the pre-fix calibration
    settled at 2055.6 instead of 2000, (0.5, 0.8, 0.3, 3) at 2312.5, and
    (0.4, 0.8, 0.3, 2) at 2750.
    """
    carrying_capacity = 2000.0
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"sex_surv_{growth_mode}_{s_f}_{s_m}_{r}"),
            name=f"sex_surv_{growth_mode}_{s_f}_{s_m}_{r}",
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"A|A": carrying_capacity / 2},
                "male": {"A|A": carrying_capacity / 2},
            }
        )
        .survival(female_age0_survival=s_f, male_age0_survival=s_m)
        .reproduction(eggs_per_female=6.0, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode=growth_mode,
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=r,
        )
        .build()
    )
    population.run(400)
    adults = float(np.asarray(population.state.individual_count)[:, 1, :].sum())
    assert adults == pytest.approx(carrying_capacity, rel=1e-9)


def test_age_structured_equilibrium_reaches_k_under_sex_specific_survival() -> None:
    """Age-structured engine: the age-1 total is exactly K.

    Same calibration core as the discrete path; with s_f = 0.8 and s_m = 0.3
    the pre-fix calibration settled at 2312.5 instead of 2000.
    """
    carrying_capacity = 2000.0
    population = (
        nt.AgeStructuredPopulation.setup(
            species=_species("age_sex_surv"), name="age_sex_surv", stochastic=False
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"A|A": {1: 500.0}}, "male": {"A|A": {1: 500.0}}}
        )
        .survival(
            female_age_based_survival=[0.8, 0.8, 0.5],
            male_age_based_survival=[0.3, 0.3, 0.5],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0],
            eggs_per_female=20.0,
            sex_ratio=0.5,
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=3.0,
        )
        .build()
    )
    population.run(600)
    age_distribution = population.get_age_distribution("both")
    assert float(age_distribution[1]) == pytest.approx(carrying_capacity, rel=1e-9)


def test_age_structured_juvenile_weights_stay_at_k_under_sex_specific_survival() -> None:
    """Age 1 is a juvenile here, so both sexes' age-1 counts enter C*.

    ``new_adult_age = 2`` puts the age-1 row inside the competition strength,
    which makes the male half of the reference split load-bearing.  The
    realised equilibrium composition is the surviving sex ratio,
    ``sex_ratio * s_f / (sex_ratio * s_f + (1 - sex_ratio) * s_m) = 0.8 / 1.1``
    of the age-1 total, so both halves are checked.  A reference that keeps
    the raw offspring ratio for the male half is far from self-consistent:
    a declared variant with that male entry settles at 2939.2 instead of
    2000 for these inputs.
    """
    carrying_capacity = 2000.0
    population = (
        nt.AgeStructuredPopulation.setup(
            species=_species("age_sex_surv_juvenile_weight"),
            name="age_sex_surv_juvenile_weight",
            stochastic=False,
        )
        .age_structure(n_ages=4, new_adult_age=2)
        .initial_state(
            individual_count={"female": {"A|A": {2: 500.0}}, "male": {"A|A": {2: 500.0}}}
        )
        .survival(
            female_age_based_survival=[0.8, 0.8, 0.8, 0.8],
            male_age_based_survival=[0.3, 0.7, 0.7, 0.7],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 0.0, 1.0, 1.0],
            eggs_per_female=20.0,
            sex_ratio=0.5,
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=2.0,
        )
        .build()
    )
    population.run(800)
    age_distribution = population.get_age_distribution("both")
    assert float(age_distribution[1]) == pytest.approx(carrying_capacity, rel=1e-9)
    individual_count = np.asarray(population.state.individual_count)
    female_age_1 = float(individual_count[0, 1, :].sum())
    male_age_1 = float(individual_count[1, 1, :].sum())
    expected_female_share = 0.5 * 0.8 / (0.5 * 0.8 + 0.5 * 0.3)
    assert female_age_1 / (female_age_1 + male_age_1) == pytest.approx(
        expected_female_share, rel=1e-9
    )
