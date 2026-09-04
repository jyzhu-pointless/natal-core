"""Slice-① gap-filling tests for the redrawn contract layer.

Complements ``test_contracts_materialize.py`` with the five strict
test categories it leaves open.  Every assertion pins a numeric or
identity invariant:

- **Negative contract**: the deleted names (``ParamsBlock``-family,
  the legacy config classes, the 12 per-sex/dead draft fields) are
  unreachable both via ``import`` and structurally (``_fields``).
- **Ownership**: every contract array is a fresh copy of the draft
  array (``is not`` plus write-isolation in both directions), the two
  compatibility arrays and a declared equilibrium are covered (the
  existing file omits them), and ``snapshot_ecology`` exposes a fresh
  custom-slots dict while arrays stay shared by reference (documented).
- **State transitions**: materialization is idempotent, the contract
  binds only at materialization time, and a runtime ``update()`` on a
  live discrete population is visible to a later materialization while
  the discrete normalization (zero adult survival) is preserved.
- **Axis combinations**: (age x discrete) x (custom x none) x
  (equilibrium declared x sentinel) — full shape matrix plus
  probability invariants (meiosis rows sum to 1, offspring rows sum
  to 1).
- **Error paths**: the four ``_require_discrete_config`` ``ValueError``
  branches and its ``TypeError`` branch, each proven to leave the draft
  untouched; ``format_type_name`` separator/uniqueness properties; and
  ztype-name alignment after ``compress_config``.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Sequence

import numpy as np
import pytest

import natal as nt
import natal.contracts
from natal.contracts import (
    Blueprint,
    CustomValue,
    Materialized,
    Params,
    format_type_name,
    gtype_names_from_registry,
    materialize,
    ztype_names_from_registry,
)
from natal.contracts.params import EcologySnapshot
from natal.frontend.data import ModelDraft, compress_config
from natal.frontend.population.discrete_generation import (
    _require_discrete_config,
)

# ── builders ──────────────────────────────────────────────────────────────────


def _species(name: str) -> nt.Species:
    """Create a two-locus species with one somatic/gamete label."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _age_draft(
    name: str = "contracts_slice1_age", with_custom: bool = True
) -> ModelDraft:
    """Build an age-structured draft with exactly known demographics."""
    sp = _species(name)
    chain = nt.AgeStructuredPopulation.setup(sp, stochastic=False)
    if with_custom:
        chain = chain.custom(release_fraction=0.25, enable_check=True)
    return (
        chain.initial_state(
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


def _discrete_draft(
    name: str = "contracts_slice1_discrete", with_custom: bool = True
) -> ModelDraft:
    """Build a discrete-generation draft with exactly known demographics."""
    sp = _species(name)
    chain = nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
    if with_custom:
        chain = chain.custom(release_fraction=0.25, enable_check=True)
    return (
        chain.initial_state(
            individual_count={
                "female": {"A|A": [0.0, 100.0], "A|B": [0.0, 50.0]},
                "male": {"A|A": [0.0, 100.0], "A|B": [0.0, 50.0]},
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


# Exact field lists pinned by the schema tests below.
_PARAMS_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclass_fields(Params)
)


def _assert_slots_equal(
    left: dict[str, CustomValue], right: dict[str, CustomValue]
) -> None:
    """Assert two custom-slots mappings carry identical values.

    ndarray values are compared elementwise (exact), everything else
    with ``==``.
    """
    assert set(left) == set(right)
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, np.ndarray):
            assert isinstance(right_value, np.ndarray)
            np.testing.assert_array_equal(left_value, right_value)
        else:
            assert left_value == right_value


def _assert_materializations_equal(
    first: Materialized, second: Materialized
) -> None:
    """Assert two materializations are numerically identical.

    Every Blueprint and Params leaf must hold the exact same value
    (arrays elementwise, scalars exactly) — the idempotency invariant.
    """
    for name in Blueprint._fields:
        left = getattr(first.blueprint, name)
        right = getattr(second.blueprint, name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    for name in _PARAMS_FIELDS:
        left = getattr(first.params, name)
        right = getattr(second.params, name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            _assert_slots_equal(left, right)
        else:
            assert left == right


# ── negative contract ─────────────────────────────────────────────────────────


def test_deleted_contracts_names_are_inaccessible() -> None:
    # The ParamsBlock family must not be importable from natal.contracts.
    with pytest.raises(ImportError):
        from natal.contracts import make_params_block  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
    with pytest.raises(ImportError):
        from natal.contracts import (  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
            build_params_dtype,
        )
    # And the underlying modules must not carry them under any spelling.
    assert not hasattr(natal.contracts, "ParamsBlock")
    assert not hasattr(natal.contracts, "build_params_dtype")
    assert not hasattr(natal.contracts, "make_params_block")
    assert not hasattr(natal.contracts.params, "ParamsBlock")
    assert not hasattr(natal.contracts.params, "build_params_dtype")
    assert not hasattr(natal.contracts.params, "make_params_block")


def test_deleted_data_names_are_inaccessible() -> None:
    # natal.frontend.data: the from_ alias is gone together with the
    # names already covered by test_contracts_materialize.py.
    with pytest.raises(ImportError):
        from natal.frontend.data import (  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
            from_plain_population_config,
        )
    # natal.frontend.data (legacy shim): all five legacy config names are gone.
    with pytest.raises(ImportError):
        from natal.frontend.data import DiscretePopulationConfig  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
    with pytest.raises(ImportError):
        from natal.frontend.data import PlainPopulationConfig  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
    with pytest.raises(ImportError):
        from natal.frontend.data import PopulationConfig  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
    with pytest.raises(ImportError):
        from natal.frontend.data import from_plain_population_config  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist
    with pytest.raises(ImportError):
        from natal.frontend.data import to_plain_population_config  # type: ignore[attr-defined]  # noqa: F401  # negative contract: name must not exist


def test_model_draft_schema_has_no_legacy_fields() -> None:
    # Structural proof (stronger than per-instance hasattr): the 12
    # discrete scalars, per-sex views and the dead female_fertility are
    # not fields of the merged draft type at all.
    legacy = (
        "female_age0_survival", "male_age0_survival",
        "female_adult_mating_rate", "male_adult_mating_rate",
        "reproduction_rate", "female_fertility",
        "meiosis_f", "meiosis_m",
        "fecundity_f", "fecundity_m",
        "viability_f", "viability_m",
    )
    assert set(legacy).isdisjoint(ModelDraft._fields)


def test_blueprint_field_schema_is_exact() -> None:
    # The frozen blueprint carries exactly the 7 dimensions, 5 execution
    # flags, 2 name directories, 2 sex-chromosome masks, 2 initial states
    # and the slice-5 spatial block (deme count + migration CSR) —
    # nothing else (genetics tensors, equilibrium, Champer override must
    # not sneak back in).
    assert Blueprint._fields == (
        "n_sexes", "n_ages", "n_ztypes", "n_gtypes", "n_glabs",
        "new_adult_age", "adult_ages",
        "stochastic", "continuous_sampling", "fixed_egg_count",
        "has_sex_chromosomes", "extreme_speed_mode",
        "ztype_names", "gtype_names",
        "female_only_by_sex_chrom", "male_only_by_sex_chrom",
        "initial_individual_count", "initial_sperm_storage",
        "n_demes", "migration_indptr", "migration_dest_idx",
        "migration_weights",
    )


def test_params_field_schema_is_exact() -> None:
    # The flat mutable container carries exactly the ecology section
    # (7 scalars + 6 vectors + the slice-5 spatial migration-rate
    # column), the genetics section (8 tables) and the custom slots —
    # in declaration order, migration_rate before custom_slots (the
    # dataclass rule pins both of them after the non-default fields).
    assert _PARAMS_FIELDS == (
        "carrying_capacity", "eggs_per_female", "sex_ratio",
        "sperm_displacement_rate", "low_density_growth_rate",
        "growth_mode", "external_expected_eggs",
        "survival_rates", "mating_rates", "reproduction_rates",
        "fertility", "competition_weights", "equilibrium_distribution",
        "viability_fitness", "fecundity_fitness",
        "sexual_selection_fitness", "zygote_viability_fitness",
        "offspring_tensor", "meiosis_map",
        "female_ztype_compatibility", "male_ztype_compatibility",
        "migration_rate",
        "custom_slots",
    )


def test_params_is_slots_with_identity_equality() -> None:
    # slots=True: instances carry no __dict__ (memory discipline).
    p = materialize(_age_draft()).params
    assert not hasattr(p, "__dict__")
    # eq=False: equality is object identity — two Params with identical
    # content are distinct values (mutability would corrupt value
    # semantics).
    q = materialize(_age_draft()).params
    assert p == p
    assert q != p


# ── ownership ─────────────────────────────────────────────────────────────────


_BLUEPRINT_ARRAY_FIELDS: tuple[tuple[str, str], ...] = (
    # (blueprint field, draft field) — identical names here.
    ("adult_ages", "adult_ages"),
    ("female_only_by_sex_chrom", "female_only_by_sex_chrom"),
    ("male_only_by_sex_chrom", "male_only_by_sex_chrom"),
    ("initial_individual_count", "initial_individual_count"),
    ("initial_sperm_storage", "initial_sperm_storage"),
)


@pytest.mark.parametrize(("bp_field", "draft_field"), _BLUEPRINT_ARRAY_FIELDS)
def test_blueprint_arrays_are_owned_copies(
    bp_field: str, draft_field: str
) -> None:
    cfg = _age_draft()
    bp = materialize(cfg).blueprint
    draft_arr = getattr(cfg, draft_field)
    contract_arr = getattr(bp, bp_field)
    # Identity: the contract never aliases the draft.
    assert contract_arr is not draft_arr
    # Write isolation: mutating the contract leaves the draft intact.
    sentinel = np.array(draft_arr, copy=True)
    contract_arr[...] = 1 if contract_arr.dtype == np.bool_ else -7.0
    np.testing.assert_array_equal(draft_arr, sentinel)


@pytest.mark.parametrize("field", [
    "female_ztype_compatibility", "male_ztype_compatibility",
])
def test_compatibility_arrays_are_owned_copies(field: str) -> None:
    cfg = _age_draft()
    p = materialize(cfg).params
    draft_arr = getattr(cfg, field)
    contract_arr = getattr(p, field)
    assert contract_arr is not draft_arr
    sentinel = draft_arr.copy()
    contract_arr[...] = 7.0
    np.testing.assert_array_equal(draft_arr, sentinel)


def test_declared_equilibrium_is_an_owned_copy() -> None:
    eq = np.array([[0.3, 0.7], [0.3, 0.7]], dtype=np.float64)
    cfg = _age_draft()._replace(equilibrium_individual_distribution=eq)
    p = materialize(cfg).params
    assert p.equilibrium_distribution is not eq
    # Values round-trip exactly (materialize must not renormalize).
    np.testing.assert_array_equal(p.equilibrium_distribution, eq)
    p.equilibrium_distribution[...] = -1.0
    np.testing.assert_array_equal(eq, np.array([[0.3, 0.7], [0.3, 0.7]]))


def test_custom_slot_3d_array_is_a_float64_copy() -> None:
    heat = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    sp = _species("contracts_slice1_heat")
    cfg = (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .custom(heat=heat)
        .initial_state(
            individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}}
        )
        .build()
    ).config
    p = materialize(cfg).params
    slot = p.custom_slots["heat"]
    assert isinstance(slot, np.ndarray)
    assert slot.dtype == np.float64
    # Identity: the slot is a fresh array, not a view into the draft's
    # structured array.
    assert slot is not cfg.custom["heat"]
    np.testing.assert_array_equal(slot, heat)
    # Write isolation in both directions.
    slot[0, 0, 0] = -99.0
    assert float(cfg.custom["heat"][0, 0, 0]) == pytest.approx(0.0)


def test_every_contract_array_is_distinct_across_materializations() -> None:
    cfg = _age_draft()
    m1 = materialize(cfg)
    m2 = materialize(cfg)
    for name in Blueprint._fields:
        left = getattr(m1.blueprint, name)
        if isinstance(left, np.ndarray):
            assert left is not getattr(m2.blueprint, name), name
    for name in _PARAMS_FIELDS:
        left = getattr(m1.params, name)
        if isinstance(left, np.ndarray):
            assert left is not getattr(m2.params, name), name


def test_snapshot_ecology_custom_slots_dict_is_isolated() -> None:
    p = materialize(_age_draft()).params
    snap = p.snapshot_ecology()
    slots = snap["custom_slots"]
    assert isinstance(slots, dict)
    # The dict itself is a fresh mapping.
    assert slots is not p.custom_slots
    assert set(slots) == set(p.custom_slots) == {"release_fraction", "enable_check"}
    # Mutating the snapshot's dict must not touch the live slots.
    slots["extra"] = 1.0
    del slots["release_fraction"]
    assert "extra" not in p.custom_slots
    assert "release_fraction" in p.custom_slots


def test_snapshot_ecology_exact_keys_and_scalar_values() -> None:
    p = materialize(_age_draft()).params
    snap = p.snapshot_ecology()
    # Exactly the ecology section — the 13 non-genetic names plus the
    # slice-5 migration-rate column and the nested custom-slots mapping;
    # no genetics key may appear.
    assert set(snap) == {
        "carrying_capacity", "eggs_per_female", "sex_ratio",
        "sperm_displacement_rate", "low_density_growth_rate",
        "growth_mode", "external_expected_eggs",
        "survival_rates", "mating_rates", "reproduction_rates",
        "fertility", "competition_weights", "equilibrium_distribution",
        "migration_rate",
        "custom_slots",
    }
    # Panmictic materialization: one all-zero (n_sexes, n_ages) column.
    assert snap["migration_rate"].shape == (1, 2, p.survival_rates.shape[1])
    assert not snap["migration_rate"].any()
    assert snap["carrying_capacity"] == pytest.approx(1234.0)
    assert snap["growth_mode"] == 3
    assert snap["external_expected_eggs"] == pytest.approx(-1.0)
    # Documented shallow behavior: array entries are shared references
    # (callers needing isolation copy further).
    assert snap["survival_rates"] is p.survival_rates


# ── state transitions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("granularity", ["age", "discrete"])
def test_materialize_is_idempotent(granularity: str) -> None:
    # Same draft, two materializations: numerically identical contracts
    # for both granularities.
    name = f"contracts_slice1_idem_{granularity}"
    cfg = _age_draft(name) if granularity == "age" else _discrete_draft(name)
    first = materialize(cfg)
    second = materialize(cfg)
    _assert_materializations_equal(first, second)


def test_contract_binds_at_materialization_time() -> None:
    cfg = _age_draft()
    first = materialize(cfg).params
    # Mutate the draft after the first materialization (NamedTuple scalar
    # rebind plus an in-place vector write).
    cfg = cfg._replace(carrying_capacity=777.0)
    cfg.age_based_survival_rates[0, 0] = 0.42
    second = materialize(cfg).params
    # The second materialization reflects the new draft values exactly.
    assert second.carrying_capacity == pytest.approx(777.0)
    assert second.survival_rates[0, 0] == pytest.approx(0.42)
    # The first contract still carries the values captured at its
    # materialization — and is not aliased to the draft.
    assert first.carrying_capacity == pytest.approx(1234.0)
    assert first.survival_rates[0, 0] == pytest.approx(0.9)
    cfg.age_based_survival_rates[0, 0] = 0.1
    assert first.survival_rates[0, 0] == pytest.approx(0.9)


def test_runtime_update_visible_to_later_materialization() -> None:
    sp = _species("contracts_slice1_runtime")
    pop = (
        nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"A|A": [0.0, 100.0]},
                "male": {"A|A": [0.0, 100.0]},
            }
        )
        .reproduction(eggs_per_female=6.0)
        .build()
    )
    before = materialize(pop.config).params
    assert before.survival_rates[0, 0] == pytest.approx(1.0)
    assert before.eggs_per_female == pytest.approx(6.0)

    pop.update().survival(
        female_age0_survival=0.7, male_age0_survival=0.6
    )
    pop.update().reproduction(eggs_per_female=9.0)

    after = materialize(pop.config)
    p = after.params
    # Juvenile column carries the runtime values exactly.
    assert p.survival_rates[0, 0] == pytest.approx(0.7)
    assert p.survival_rates[1, 0] == pytest.approx(0.6)
    # The discrete normalization survives the runtime write: adults are
    # still fully replaced (non-overlapping generations).
    assert float(p.survival_rates[:, 1].sum()) == 0.0
    assert p.eggs_per_female == pytest.approx(9.0)
    # The updated draft still satisfies the discrete invariants.
    assert _require_discrete_config(pop.config) is pop.config


# ── axis combinations ─────────────────────────────────────────────────────────


_EQUILIBRIUM = np.array([[0.3, 0.7], [0.3, 0.7]], dtype=np.float64)


def _combo_draft(
    granularity: str, with_custom: bool, with_equilibrium: bool
) -> ModelDraft:
    """Build the draft for one cell of the 2x2x2 combination matrix."""
    name = f"contracts_slice1_combo_{granularity}_{with_custom}_{with_equilibrium}"
    builder = _age_draft if granularity == "age" else _discrete_draft
    cfg = builder(name, with_custom=with_custom)
    if with_equilibrium:
        cfg = cfg._replace(equilibrium_individual_distribution=_EQUILIBRIUM)
    return cfg


@pytest.mark.parametrize("granularity", ["age", "discrete"])
@pytest.mark.parametrize("with_custom", [False, True])
@pytest.mark.parametrize("with_equilibrium", [False, True])
def test_axis_combination_shapes_and_values(
    granularity: str, with_custom: bool, with_equilibrium: bool
) -> None:
    cfg = _combo_draft(granularity, with_custom, with_equilibrium)
    mat = materialize(cfg)
    bp, p = mat.blueprint, mat.params

    n_ages, n_z, n_g = cfg.n_ages, cfg.n_ztypes, cfg.n_gtypes
    assert (n_ages, n_z, n_g) == (2, 3, 2)  # two-locus species probe

    # -- Blueprint shapes (written out in full) --
    assert bp.n_sexes == 2
    assert bp.n_ages == n_ages
    assert bp.n_ztypes == n_z
    assert bp.n_gtypes == n_g
    assert bp.n_glabs == 1
    assert bp.new_adult_age == cfg.new_adult_age
    np.testing.assert_array_equal(bp.adult_ages, [1])
    assert bp.initial_individual_count.shape == (2, n_ages, n_z)
    assert bp.initial_sperm_storage.shape == (n_ages, n_z, n_z)
    assert bp.female_only_by_sex_chrom.shape == (n_z,)
    assert bp.male_only_by_sex_chrom.shape == (n_z,)
    assert len(bp.ztype_names) == n_z
    assert len(bp.gtype_names) == n_g
    # Name directory: canonical "<genotype>:<slab>" strings, aligned
    # with the draft directory.
    assert bp.ztype_names == tuple(cfg.ztype_names)
    assert "A|A:default" in bp.ztype_names

    # -- Params shapes --
    assert p.survival_rates.shape == (2, n_ages)
    assert p.mating_rates.shape == (2, n_ages)
    assert p.reproduction_rates.shape == (n_ages,)
    assert p.fertility.shape == (n_ages,)
    assert p.competition_weights.shape == (n_ages,)
    assert p.viability_fitness.shape == (2, n_ages, n_z)
    assert p.fecundity_fitness.shape == (2, n_z)
    assert p.sexual_selection_fitness.shape == (n_z, n_z)
    assert p.zygote_viability_fitness.shape == (2, n_z)
    assert p.offspring_tensor.shape == (n_z, n_z, n_z)
    assert p.meiosis_map.shape == (2, n_z, n_g)
    assert p.female_ztype_compatibility.shape == (n_z,)
    assert p.male_ztype_compatibility.shape == (n_z,)

    # -- Probability invariants (hold for every cell of the matrix) --
    np.testing.assert_allclose(p.meiosis_map.sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        p.offspring_tensor.sum(axis=-1), 1.0, atol=1e-12
    )

    # -- Granularity-specific demographic values --
    if granularity == "age":
        assert cfg.discrete_generation is False
        np.testing.assert_allclose(
            p.survival_rates, [[0.9, 0.9], [0.8, 0.8]], atol=1e-12
        )
        assert p.growth_mode == 3
        assert p.carrying_capacity == pytest.approx(1234.0)
    else:
        assert cfg.discrete_generation is True
        np.testing.assert_allclose(
            p.survival_rates, [[1.0, 0.0], [1.0, 0.0]], atol=1e-12
        )
        np.testing.assert_allclose(
            p.mating_rates, [[0.0, 0.9], [0.0, 0.8]], atol=1e-12
        )
        np.testing.assert_allclose(
            p.reproduction_rates, [0.0, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            p.fertility, [0.0, 1.0], atol=1e-12
        )
        assert p.growth_mode == 2
        assert p.carrying_capacity == pytest.approx(500.0)

    # -- Custom-slot axis --
    if with_custom:
        assert p.custom_slots == {"release_fraction": 0.25, "enable_check": True}
    else:
        assert p.custom_slots == {}

    # -- Equilibrium axis: declared copy vs (0, 0) sentinel --
    if with_equilibrium:
        assert p.equilibrium_distribution.shape == (2, n_ages)
        np.testing.assert_array_equal(p.equilibrium_distribution, _EQUILIBRIUM)
    else:
        assert p.equilibrium_distribution.shape == (0, 0)


# ── error paths ───────────────────────────────────────────────────────────────


def _corrupted_draft(kind: str) -> ModelDraft:
    """Produce a draft violating exactly one discrete invariant.

    Args:
        kind: One of ``"n_ages"``, ``"new_adult_age"``, ``"adult_ages"``,
            ``"adult_survival"``.

    Returns:
        A ``ModelDraft`` violating the named invariant (arrays are
        copied before in-place edits, so the base draft is untouched).
    """
    cfg = _discrete_draft()
    if kind == "n_ages":
        return cfg._replace(n_ages=3)
    if kind == "new_adult_age":
        return cfg._replace(new_adult_age=0)
    if kind == "adult_ages":
        return cfg._replace(adult_ages=np.array([0], dtype=np.int64))
    assert kind == "adult_survival"
    survival = cfg.age_based_survival_rates.copy()
    survival[0, 1] = 0.5
    return cfg._replace(age_based_survival_rates=survival)


@pytest.mark.parametrize(
    ("kind", "pattern"),
    [
        ("n_ages", r"n_ages == 2"),
        ("new_adult_age", r"new_adult_age == 1"),
        ("adult_ages", r"adult_ages must be \[1\]"),
        ("adult_survival", r"zero adult survival"),
    ],
)
def test_require_discrete_config_rejects_bad_drafts(
    kind: str, pattern: str
) -> None:
    bad = _corrupted_draft(kind)
    # Snapshot the validator's input to prove it never mutates its
    # argument on the error path.
    survival_before = bad.age_based_survival_rates.copy()
    adult_ages_before = bad.adult_ages.copy()
    with pytest.raises(ValueError, match=pattern):
        _require_discrete_config(bad)
    np.testing.assert_array_equal(
        bad.age_based_survival_rates, survival_before
    )
    np.testing.assert_array_equal(bad.adult_ages, adult_ages_before)


def test_require_discrete_config_rejects_non_draft() -> None:
    with pytest.raises(TypeError, match="ModelDraft"):
        _require_discrete_config(object())
    with pytest.raises(TypeError, match="ModelDraft"):
        _require_discrete_config("not a draft")


def test_discrete_population_rejects_two_age_overlapping_draft() -> None:
    # The adult-survival marker is what separates a discrete draft from
    # a 2-age overlapping (age-structured) draft with the same
    # n_ages/new_adult_age/adult_ages: the latter must be rejected,
    # otherwise the discrete engine would produce silent nonsense.
    sp = _species("contracts_slice1_marker")
    age_cfg = (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .survival(female_age_based_survival=0.9, male_age_based_survival=0.8)
        .build()
    ).config
    assert age_cfg.n_ages == 2
    assert float(age_cfg.age_based_survival_rates[:, 1].sum()) > 0.0
    with pytest.raises(ValueError, match="zero adult survival"):
        nt.DiscreteGenerationPopulation(sp, age_cfg)


def test_format_type_name_separator_and_roundtrip() -> None:
    # Exact canonical form.
    assert format_type_name("A|a", "wolb") == "A|a:wolb"
    # The first ':' is an unambiguous separator for pattern-syntax
    # genotypes (which never contain ':'), so both parts are recoverable.
    genotype_part, sep, label_part = "A|a:wolb".rpartition(":")
    assert (genotype_part, sep, label_part) == ("A|a", ":", "wolb")
    # Works for real genotype entities (rendered via str).
    sp = _species("contracts_slice1_fmt")
    gt = sp.get_genotype_from_str("A|B")
    assert format_type_name(gt, "wolb") == "A|B:wolb"


def test_format_type_name_is_injective_without_colons() -> None:
    # Distinct (genotype, label) pairs never collide — the separator
    # cannot be confused with pattern characters (| and @).
    entries: Sequence[tuple[str, str]] = [
        ("A|A", "wt"), ("A|a", "wt"), ("A|a", "wolb"), ("a|a", "wt|extra"),
    ]
    names = [format_type_name(g, lab) for g, lab in entries]
    assert len(set(names)) == len(names)
    for (genotype, label), name in zip(entries, names):
        assert name == f"{genotype}:{label}"


def test_name_directory_helpers_render_exact_strings() -> None:
    # Exact canonical rendering (index order preserved).
    assert ztype_names_from_registry([("A|A", "wt"), ("A|a", "wolb")]) == (
        "A|A:wt",
        "A|a:wolb",
    )
    assert gtype_names_from_registry([("A", "wt"), ("a", "wolb")]) == (
        "A:wt",
        "a:wolb",
    )
    # Empty registries produce empty directories.
    assert ztype_names_from_registry([]) == ()
    assert gtype_names_from_registry([]) == ()


def test_compress_config_slices_ztype_names_and_axes() -> None:
    cfg = _age_draft()
    mask = np.array([0, -1, 2], dtype=np.int32)
    compressed = compress_config(cfg, mask)
    # The name directory is sliced with the same mask as the axes.
    assert compressed.n_ztypes == 2
    assert compressed.ztype_names == (cfg.ztype_names[0], cfg.ztype_names[2])
    # Z-indexed arrays keep their remaining axes and drop pruned slots.
    assert compressed.viability_fitness.shape == (2, cfg.n_ages, 2)
    assert compressed.initial_individual_count.shape == (2, cfg.n_ages, 2)
    assert compressed.fecundity_fitness.shape == (2, 2)
    assert compressed.zygote_viability_fitness.shape == (2, 2)
    assert compressed.sexual_selection_fitness.shape == (2, 2)
    assert compressed.female_ztype_compatibility.shape == (2,)
    assert compressed.male_ztype_compatibility.shape == (2,)
    assert compressed.female_only_by_sex_chrom.shape == (2,)
    assert compressed.male_only_by_sex_chrom.shape == (2,)
    assert compressed.initial_sperm_storage.shape == (cfg.n_ages, 2, 2)
    # The surviving slots carry exactly the original values.
    np.testing.assert_array_equal(
        compressed.viability_fitness, cfg.viability_fitness[:, :, [0, 2]]
    )
    np.testing.assert_array_equal(
        compressed.initial_individual_count,
        cfg.initial_individual_count[:, :, [0, 2]],
    )


def test_compress_then_materialize_keeps_directory_aligned() -> None:
    cfg = _age_draft()
    compressed = compress_config(
        cfg, np.array([0, -1, 2], dtype=np.int32)
    )
    mat = materialize(compressed)
    # Blueprint dimensions and the name directory stay aligned after a
    # compression happened before materialization.
    assert mat.blueprint.n_ztypes == 2
    assert len(mat.blueprint.ztype_names) == mat.blueprint.n_ztypes
    assert mat.blueprint.ztype_names == compressed.ztype_names
    assert mat.params.viability_fitness.shape == (2, cfg.n_ages, 2)


def test_build_time_compression_aligns_both_directories() -> None:
    # End-to-end: rebuild_config_maps compresses ztypes AND gtypes; both
    # directories must end up aligned with the compressed indices.
    sp = _species("contracts_slice1_compress")
    cfg = (
        nt.AgeStructuredPopulation.setup(
            sp,
            stochastic=False,
            compress=True,
            declared_zygote_types=["A|A"],
        )
        .initial_state(
            individual_count={"female": {"A|A": 200}, "male": {"A|A": 150}}
        )
        .build()
    ).config
    assert cfg.n_ztypes == len(cfg.ztype_names) == 1
    assert cfg.n_gtypes == len(cfg.gtype_names) == 1
    assert cfg.ztype_names == ("A|A:default",)
    assert cfg.gtype_names == ("A:default",)
    mat = materialize(cfg)
    assert len(mat.blueprint.ztype_names) == mat.blueprint.n_ztypes == 1
    assert len(mat.blueprint.gtype_names) == mat.blueprint.n_gtypes == 1
    assert mat.params.meiosis_map.shape == (2, 1, 1)
    assert mat.params.offspring_tensor.shape == (1, 1, 1)
    np.testing.assert_allclose(mat.params.meiosis_map.sum(axis=-1), 1.0)


def test_champer_external_eggs_roundtrip() -> None:
    # Champer calibration: expected_num_new_adult_females=500 with
    # eggs_per_female=10 -> external override = 500 * 10 = 5000 exactly.
    sp = _species("contracts_slice1_champer")
    cfg = (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .reproduction(eggs_per_female=10.0)
        .competition(expected_num_new_adult_females=500.0)
        .initial_state(
            individual_count={"female": {"A|A": 200}, "male": {"A|A": 150}}
        )
        .build()
    ).config
    assert cfg.external_expected_eggs == pytest.approx(5000.0)
    # The declared override replaces the -1.0 sentinel in Params.
    p = materialize(cfg).params
    assert p.external_expected_eggs == pytest.approx(5000.0)
    # And the ecology snapshot carries it too (it is an ecology field).
    snap: EcologySnapshot = p.snapshot_ecology()
    assert snap["external_expected_eggs"] == pytest.approx(5000.0)
