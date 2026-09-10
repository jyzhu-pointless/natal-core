"""Direct-write light source contracts (perf residual R2).

The spatial direct-write channels (``DemeSlice.write_ecology``,
container ``params.tensor_write`` per-deme routing, and
``DemeSlice.write_genetics``) hand the Rust refresh functions a
duck-typed source carrying only the named fields
(:func:`natal.contracts.materialize.contract_field_source`) instead of
a fully materialized contract.

Pinned here:

- **Equivalence**: for every mappable contract field, the light source
  attribute equals the attribute of a full
  ``materialize(draft).params`` bit-for-bit (floats exact, arrays
  bitwise), so the engine reads identical values either way.
- **No full pull**: a direct write never calls ``materialize``; the
  refresh still reaches the session (the write is visible on the
  authoritative read surface).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

import natal as nt
from natal.contracts.materialize import contract_field_source, materialize

# The package re-exports the ``materialize`` function, which shadows the
# same-named submodule on plain attribute access; import_module returns
# the real module for monkeypatching.
materialize_mod = importlib.import_module("natal.contracts.materialize")

if TYPE_CHECKING:
    from natal.frontend.model.draft import ModelDraft


@pytest.fixture(scope="module")
def species() -> nt.Species:
    """Two-allele species shared by the light-source tests."""
    return nt.Species.from_dict(
        name="__test_light_source__",
        structure={"auto": {"A": ["WT", "Dr"]}},
    )


@pytest.fixture(scope="module")
def rich_draft(species: nt.Species) -> ModelDraft:
    """A fully populated age-structured deme draft (declaration copy)."""
    pop = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .setup(name="light_src_equiv", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 5.0]},
                        "male": {"WT|WT": [0.0, 100.0, 5.0]},
                    },
                ]
                * 2
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.2],
            male_age_based_survival=[1.0, 0.85, 0.1],
        )
        .competition(carrying_capacity=1234.5, low_density_growth_rate=2.5)
        .build()
    )
    return pop.deme(0).config


def test_light_source_matches_full_materialization_per_field(
    rich_draft: ModelDraft,
) -> None:
    """Every mappable field: light source == full params, bit-for-bit.

    Catches a conversion drift between ``contract_field_source`` and
    ``_params`` (a wrong None sentinel, a dtype change, a C-order
    violation would all surface here).
    """
    params = materialize(rich_draft).params
    for name in materialize_mod._FIELD_SOURCES:  # pyright: ignore[reportPrivateUsage]  # the mapping itself is the contract under test
        source = contract_field_source(rich_draft, (name,))
        assert not hasattr(source, "unexpected"), "source exposes only named fields"
        light = getattr(source, name)
        full = getattr(params, name)
        if isinstance(light, np.ndarray):
            assert light.dtype == np.float64
            assert np.array_equal(light, np.asarray(full))
        else:
            assert float(light) == float(full)


def test_light_source_exposes_only_requested_names(rich_draft: ModelDraft) -> None:
    """A two-field request yields exactly two attributes.

    Catches a regression to whole-contract carriage on the genetics
    refresh path (meiosis writes pull two names).
    """
    source = contract_field_source(
        rich_draft, ("meiosis_map", "offspring_tensor")
    )
    assert sorted(vars(source)) == ["meiosis_map", "offspring_tensor"]


def test_unknown_field_name_is_rejected(rich_draft: ModelDraft) -> None:
    """Unmapped names fail loudly at source construction."""
    with pytest.raises(KeyError, match="no contract field source"):
        contract_field_source(rich_draft, ("not_a_field",))


def _built_discrete(species: nt.Species, name: str):
    return (
        nt.SpatialPopulation.builder(
            species, n_demes=4, topology=nt.SquareGrid(2, 2), pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500.0, low_density_growth_rate=6.0)
        .build()
    )


def test_direct_ecology_write_never_materializes(
    species: nt.Species, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_ecology`` and column ``tensor_write`` skip materialize.

    Catches a regression to the full-contract carriage: the per-write
    copy of every genetics table and the blueprint.  The refresh itself
    must still land — the authoritative column read shows the new value.
    """

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("direct write materialized the full contract")

    monkeypatch.setattr(materialize_mod, "materialize", _forbidden)

    pop = _built_discrete(species, "LightSrcNoMat")
    pop.deme(1).write_ecology("carrying_capacity", 321.0)
    assert float(pop.params.carrying_capacity[1]) == 321.0

    pop.params.tensor_write(
        "carrying_capacity", np.array([11.0, 22.0, 33.0, 44.0])
    )
    assert pop.deme(3).config.carrying_capacity == 44.0


def test_direct_genetics_write_never_materializes(
    species: nt.Species, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_genetics`` (single tensor and meiosis-with-derived) skips
    materialize while forking and refreshing the variant bank."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("direct write materialized the full contract")

    monkeypatch.setattr(materialize_mod, "materialize", _forbidden)

    pop = _built_discrete(species, "LightSrcNoMatGen")
    base = np.asarray(pop.deme(0).config.viability_fitness)
    pop.deme(0).write_genetics("viability_fitness", np.full_like(base, 0.5))
    assert pop.deme(0).config.viability_fitness.flat[0] == pytest.approx(0.5)
