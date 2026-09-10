"""Slice-① acceptance: the contract layer (ModelDraft -> Blueprint + Params).

Every assertion pins a numeric or identity invariant:

- **Values**: the contract layer exposes exactly the draft's values
  (scalars, vectors, tensors, custom slots) for both granularities.
- **Ownership**: every contract array is a fresh copy — mutating the
  draft after materialization must not leak into the contract, and the
  two ``Params`` objects from two materializations must be independent.
- **Frozen/mutable split**: ``Blueprint`` is a NamedTuple (immutable
  fields) holding only rebuild-to-change data; the genetics tensors,
  equilibrium declaration, and Champer override live in ``Params`` and
  are absent from ``Blueprint`` (negative contract).
- **Discrete normalization**: the discrete scalars
  (``female_age0_survival`` etc.) normalize into the unified
  ``(2, n_ages)`` vectors at construction; the per-sex view fields and
  the dead ``female_fertility`` scalar no longer exist anywhere.
- **Name directory**: the Blueprint carries canonical
  ``"<genotype>:<label>"`` strings for every ztype/gtype index.
- **Sentinels**: ``equilibrium_distribution`` shape ``(0, 0)`` selects
  derivation mode; ``external_expected_eggs < 0`` means unused.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

import natal as nt
import natal.contracts
from natal.contracts import (
    CONTRACTS_VERSION,
    Blueprint,
    gtype_names_from_registry,
    materialize,
    ztype_names_from_registry,
)
from natal.contracts.materialize import materialize_params
from natal.frontend.data import ModelDraft

materialize_module = import_module("natal.contracts.materialize")


def _age_config() -> ModelDraft:
    sp = nt.Species.from_dict(
        name="contracts_materialize_species",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"A|A": 200, "A|B": 100},
                "male": {"A|A": 150, "A|B": 150},
            }
        )
        .reproduction(
            eggs_per_female=10.0,
            sex_ratio=0.5,
            female_age_based_mating_rate=1.0,
            male_age_based_mating_rate=1.0,
            age_based_reproduction_rate=1.0,
            female_age_based_fertility=1.0,
        )
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.8)
        .competition(
            juvenile_growth_mode=3,
            carrying_capacity=1234.0,
            low_density_growth_rate=4.0,
        )
        .build()
    ).config


def _discrete_config() -> ModelDraft:
    sp = nt.Species.from_dict(
        name="contracts_materialize_discrete",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"A|A": [0.0, 100.0]},
                "male": {"A|A": [0.0, 100.0]},
            }
        )
        .reproduction(
            eggs_per_female=6.0,
            female_adult_mating_rate=0.9,
            male_adult_mating_rate=0.8,
        )
        .competition(juvenile_growth_mode=2, carrying_capacity=500.0)
        .build()
    ).config


# ── values: age-structured ────────────────────────────────────────────────────


def test_age_structured_params_values_match_draft() -> None:
    cfg = _age_config()
    mat = materialize(cfg)
    p = mat.params
    assert p.carrying_capacity == float(cfg.carrying_capacity)
    assert p.eggs_per_female == float(cfg.eggs_per_female)
    assert p.sex_ratio == float(cfg.sex_ratio)
    assert p.low_density_growth_rate == float(cfg.low_density_growth_rate)
    assert p.growth_mode == int(cfg.juvenile_growth_mode)
    assert p.external_expected_eggs == -1.0
    np.testing.assert_allclose(p.survival_rates, cfg.age_based_survival_rates)
    np.testing.assert_allclose(p.mating_rates, cfg.age_based_mating_rates)
    np.testing.assert_allclose(
        p.reproduction_rates, cfg.age_based_reproduction_rates
    )
    np.testing.assert_allclose(p.fertility, cfg.female_age_based_fertility)
    np.testing.assert_allclose(
        p.competition_weights, cfg.age_based_relative_competition_strength
    )


def test_age_structured_genetics_values_match_draft() -> None:
    cfg = _age_config()
    mat = materialize(cfg)
    p = mat.params
    np.testing.assert_allclose(p.viability_fitness, cfg.viability_fitness)
    np.testing.assert_allclose(p.fecundity_fitness, cfg.fecundity_fitness)
    np.testing.assert_allclose(
        p.sexual_selection_fitness, cfg.sexual_selection_fitness
    )
    np.testing.assert_allclose(
        p.zygote_viability_fitness, cfg.zygote_viability_fitness
    )
    np.testing.assert_allclose(p.offspring_tensor, cfg.offspring_tensor)
    np.testing.assert_allclose(p.meiosis_map, cfg.zygotes_to_gametes_map)
    np.testing.assert_allclose(
        p.female_ztype_compatibility, cfg.female_ztype_compatibility
    )
    np.testing.assert_allclose(
        p.male_ztype_compatibility, cfg.male_ztype_compatibility
    )


def test_blueprint_values_match_draft() -> None:
    cfg = _age_config()
    bp = materialize(cfg).blueprint
    assert bp.n_sexes == cfg.n_sexes
    assert bp.n_ages == cfg.n_ages
    assert bp.n_ztypes == cfg.n_ztypes
    assert bp.n_gtypes == cfg.n_gtypes
    assert bp.n_glabs == cfg.n_glabs
    assert bp.new_adult_age == cfg.new_adult_age
    np.testing.assert_array_equal(bp.adult_ages, cfg.adult_ages)
    assert bp.stochastic == cfg.stochastic
    assert bp.continuous_sampling == cfg.continuous_sampling
    assert bp.fixed_egg_count == cfg.fixed_egg_count
    assert bp.has_sex_chromosomes == cfg.has_sex_chromosomes
    assert bp.extreme_speed_mode == 0
    np.testing.assert_array_equal(
        bp.female_only_by_sex_chrom, cfg.female_only_by_sex_chrom
    )
    np.testing.assert_array_equal(
        bp.male_only_by_sex_chrom, cfg.male_only_by_sex_chrom
    )
    np.testing.assert_allclose(
        bp.initial_individual_count, cfg.initial_individual_count
    )


def test_blueprint_shape_invariants() -> None:
    cfg = _age_config()
    bp = materialize(cfg).blueprint
    n_z = cfg.n_ztypes
    assert bp.ztype_names == tuple(cfg.ztype_names)
    assert len(bp.ztype_names) == n_z
    assert len(bp.gtype_names) == cfg.n_gtypes
    assert bp.initial_individual_count.shape == (2, cfg.n_ages, n_z)
    assert bp.initial_sperm_storage.shape == (cfg.n_ages, n_z, n_z)
    assert bp.female_only_by_sex_chrom.shape == (n_z,)
    assert bp.male_only_by_sex_chrom.shape == (n_z,)
    assert bp.initial_individual_count[:, 1, :].sum() == pytest.approx(600.0)


# ── values: discrete normalization ───────────────────────────────────────────


def test_discrete_normalization_into_unified_vectors() -> None:
    cfg = _discrete_config()
    # The API kwargs landed in the unified (2, 2) vectors.
    assert cfg.age_based_survival_rates[0, 0] == pytest.approx(1.0)
    assert cfg.age_based_survival_rates[1, 0] == pytest.approx(1.0)
    assert cfg.age_based_survival_rates[:, 1].sum() == pytest.approx(0.0)
    assert cfg.age_based_mating_rates[0, 1] == pytest.approx(0.9)
    assert cfg.age_based_mating_rates[1, 1] == pytest.approx(0.8)
    assert cfg.age_based_mating_rates[:, 0].sum() == pytest.approx(0.0)
    assert cfg.age_based_reproduction_rates[1] == pytest.approx(1.0)
    assert cfg.age_based_reproduction_rates[0] == pytest.approx(0.0)
    assert cfg.discrete_generation is True


def test_discrete_materialized_params_match_vectors() -> None:
    cfg = _discrete_config()
    mat = materialize(cfg)
    p = mat.params
    assert p.carrying_capacity == pytest.approx(500.0)
    assert p.eggs_per_female == pytest.approx(6.0)
    assert p.growth_mode == 2
    np.testing.assert_allclose(
        p.survival_rates, [[1.0, 0.0], [1.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(
        p.mating_rates, [[0.0, 0.9], [0.0, 0.8]], atol=1e-12
    )
    np.testing.assert_allclose(
        p.reproduction_rates, [0.0, 1.0], atol=1e-12
    )
    # Discrete models start with zero sperm storage (the discrete
    # population never allocates a sperm dimension at state creation).
    assert mat.blueprint.initial_sperm_storage.sum() == pytest.approx(0.0)


# ── ownership: copies, not aliases ───────────────────────────────────────────


@pytest.mark.parametrize("field", [
    "survival_rates", "mating_rates", "reproduction_rates",
    "fertility", "competition_weights", "viability_fitness",
    "fecundity_fitness", "sexual_selection_fitness",
    "zygote_viability_fitness", "offspring_tensor", "meiosis_map",
])
def test_params_arrays_are_independent_copies(field: str) -> None:
    cfg = _age_config()
    mat = materialize(cfg)
    draft_arr = getattr(cfg, {
        "survival_rates": "age_based_survival_rates",
        "mating_rates": "age_based_mating_rates",
        "reproduction_rates": "age_based_reproduction_rates",
        "fertility": "female_age_based_fertility",
        "competition_weights": "age_based_relative_competition_strength",
        "meiosis_map": "zygotes_to_gametes_map",
    }.get(field, field))
    params_arr = getattr(mat.params, field)
    assert params_arr is not draft_arr
    params_arr[...] = 7.0
    assert float(np.asarray(draft_arr).ravel()[0]) != 7.0


def test_blueprint_initial_state_is_a_copy() -> None:
    cfg = _age_config()
    bp = materialize(cfg).blueprint
    sentinel = float(cfg.initial_individual_count.sum())
    # R4: the blueprint's arrays are read-only — the strongest form of the
    # isolation contract (no write path exists at all, so the draft can
    # never be corrupted through the contract).
    assert not bp.initial_individual_count.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        bp.initial_individual_count[...] = -1.0
    assert float(cfg.initial_individual_count.sum()) == sentinel


def test_two_materializations_are_independent() -> None:
    cfg = _age_config()
    m1 = materialize(cfg)
    m2 = materialize(cfg)
    m1.params.carrying_capacity = 9_999.0
    m1.params.viability_fitness[...] = 0.0
    assert m2.params.carrying_capacity != 9_999.0
    assert m2.params.viability_fitness.max() > 0.0


def test_params_is_mutable_in_place() -> None:
    cfg = _age_config()
    p = materialize(cfg).params
    before = p.carrying_capacity
    p.carrying_capacity = 4_321.0
    assert p.carrying_capacity == 4_321.0
    assert before == 1234.0
    # Vector contents mutate in place.
    p.survival_rates[0, 0] = 0.5
    assert p.survival_rates[0, 0] == pytest.approx(0.5)


# ── frozen/mutable split (negative contract) ─────────────────────────────────


def test_blueprint_is_a_frozen_namedtuple() -> None:
    # NamedTuple marker + tuple base + immutable fields.
    assert hasattr(Blueprint, "_fields")
    assert issubclass(Blueprint, tuple)
    with pytest.raises(AttributeError):
        materialize(_age_config()).blueprint.n_ages = 5  # type: ignore[misc]


@pytest.mark.parametrize("field", [
    "viability_fitness", "fecundity_fitness", "sexual_selection_fitness",
    "zygote_viability_fitness", "offspring_tensor", "meiosis_map",
    "female_ztype_compatibility", "male_ztype_compatibility",
    "equilibrium_distribution", "external_expected_eggs",
    "carrying_capacity", "survival_rates",
])
def test_genetics_and_ecology_fields_absent_from_blueprint(field: str) -> None:
    bp = materialize(_age_config()).blueprint
    assert not hasattr(bp, field)


def test_removed_legacy_fields_do_not_exist() -> None:
    cfg = _discrete_config()
    for legacy in (
        "female_age0_survival", "male_age0_survival",
        "female_adult_mating_rate", "male_adult_mating_rate",
        "reproduction_rate", "female_fertility",
        "meiosis_f", "meiosis_m", "fecundity_f", "fecundity_m",
        "viability_f", "viability_m",
    ):
        assert not hasattr(cfg, legacy)


def test_removed_public_names_are_inaccessible() -> None:
    with pytest.raises(ImportError):
        from natal.contracts import (
            ParamsBlock,  # type: ignore[attr-defined]  # noqa: F401
        )
    with pytest.raises(ImportError):
        from natal.contracts import (
            build_params_dtype,  # type: ignore[attr-defined]  # noqa: F401
        )
    with pytest.raises(ImportError):
        from natal.frontend.data import (  # type: ignore[attr-defined]  # noqa: F401
            PopulationConfig,
        )
    with pytest.raises(ImportError):
        from natal.frontend.data import (  # type: ignore[attr-defined]  # noqa: F401
            DiscretePopulationConfig,
        )
    with pytest.raises(ImportError):
        from natal.frontend.data import (
            PlainPopulationConfig,  # type: ignore[attr-defined]  # noqa: F401
        )
    with pytest.raises(ImportError):
        from natal.frontend.data import (  # type: ignore[attr-defined]  # noqa: F401
            to_plain_population_config,
        )


def test_contracts_all_surface() -> None:
    expected = {
        "CONTRACTS_VERSION", "Blueprint", "CustomValue",
        "Materialized", "Params", "format_type_name",
        "gtype_names_from_registry", "materialize", "ztype_names_from_registry",
    }
    assert set(natal.contracts.__all__) == expected
    assert CONTRACTS_VERSION >= 2


# ── custom slots ─────────────────────────────────────────────────────────────


def test_custom_slots_roundtrip() -> None:
    sp = nt.Species.from_dict(
        name="contracts_materialize_custom",
        structure={"chr1": {"loc": ["A"]}},
        gamete_labels=["default"],
    )
    pop = (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .custom(release_fraction=0.25, enable_check=True)
        .build()
    )
    p = materialize(pop.config).params
    assert p.custom_slots["release_fraction"] == pytest.approx(0.25)
    assert p.custom_slots["enable_check"] is True


def test_custom_slots_default_empty() -> None:
    p = materialize(_age_config()).params
    assert p.custom_slots == {}


# ── sentinels ────────────────────────────────────────────────────────────────


def test_equilibrium_sentinel_shape() -> None:
    p = materialize(_age_config()).params
    assert p.equilibrium_distribution.shape == (0, 0)


def test_retired_sim_state_contract_is_absent() -> None:
    """The unused legacy SimState contract is absent from all surfaces."""
    for statement in (
        "from natal import SimState",
        "from natal.contracts import SimState",
        "from natal.contracts.state import SimState",
    ):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            exec(statement, {})


def test_retired_plain_state_surfaces_are_absent() -> None:
    """Self-conversion aliases and functions are removed from state APIs."""
    names = (
        "PlainPopulationState", "PlainDiscretePopulationState",
        "to_plain_population_state", "to_plain_discrete_population_state",
        "from_plain_population_state", "from_plain_discrete_population_state",
    )
    for module in ("natal", "natal.frontend.data", "natal.frontend.data.state"):
        for name in names:
            with pytest.raises((ImportError, AttributeError)):
                getattr(__import__(module, fromlist=[name]), name)


# ── name directory helpers ───────────────────────────────────────────────────


def test_name_directory_from_registry() -> None:
    sp = nt.Species.from_dict(
        name="contracts_materialize_registry_names",
        structure={"chr1": {"loc": ["A"]}},
        gamete_labels=["default"],
    )
    from natal.frontend.builder._registry_builder import build_registry

    registry = build_registry(sp)
    names = ztype_names_from_registry(registry.index_to_ztype)
    gnames = gtype_names_from_registry(registry.index_to_gtype)
    assert len(names) == registry.n_ztypes > 0
    assert len(gnames) == registry.n_gtypes > 0
    # Canonical rendering: "<genotype>:<slab>" with the genotype first.
    assert names[0] == f"{registry.index_to_genotype[0]}:default"


def test_name_directory_exposes_canonical_strings() -> None:
    cfg = _age_config()
    bp = materialize(cfg).blueprint
    # "A|A" (wild-type genotype, default slab) must appear in the directory.
    assert any(n.startswith("A|A:") for n in bp.ztype_names)


def test_custom_slots_accept_the_documented_value_kinds() -> None:
    """custom_slots stores bool/int/float/ndarray as materialized."""
    sp = nt.Species.from_dict(
        name="contracts_materialize_custom_kinds",
        structure={"chr1": {"loc": ["A"]}},
        gamete_labels=["default"],
    )
    pop = (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .custom(flag=True, count=3, ratio=0.5)
        .build()
    )
    slots = materialize(pop.config).params.custom_slots
    assert slots["flag"] is True
    assert slots["count"] == 3
    assert slots["ratio"] == 0.5


def test_params_class_surface() -> None:
    p = materialize(_age_config()).params
    snapshot = p.snapshot_ecology()
    # Ecology section only — no genetics keys in the snapshot.
    assert "viability_fitness" not in snapshot
    assert "offspring_tensor" not in snapshot
    assert snapshot["carrying_capacity"] == pytest.approx(1234.0)


def test_params_only_materialization_skips_blueprint_and_preserves_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Params-only projection matches materialize and owns copied arrays."""
    draft = _age_config()
    monkeypatch.setattr(
        materialize_module,
        "_blueprint",
        lambda *_args: pytest.fail("Params-only materialization built a Blueprint"),
    )
    params = materialize_params(draft)
    assert params.carrying_capacity == pytest.approx(1234.0)
    params.survival_rates[0, 0] = -1.0
    assert draft.age_based_survival_rates[0, 0] != -1.0


def test_params_only_materialization_copies_provided_migration_rate() -> None:
    """A supplied migration-rate column is copied into Params."""
    draft = _age_config()
    migration_rate = np.full((2, 2, 2), 0.25, dtype=np.float64)
    params = materialize_params(draft, migration_rate)
    migration_rate[0, 0, 0] = 0.75
    assert params.migration_rate[0, 0, 0] == pytest.approx(0.25)
