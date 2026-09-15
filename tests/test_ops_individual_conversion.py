"""Public selector conversion and clearing contracts with exact count outcomes."""

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.types import HookOp


def _build(*ops: HookOp) -> nt.AgeStructuredPopulation:
    """Build all label types so hook targets are available without compression."""
    species = nt.Species.from_dict(
        "individual_conversion_basic", {"chr": {"loc": ["A", "B"]}},
        somatic_labels=["uninfected", "infected"],
    )
    return (
        nt.AgeStructuredPopulation.setup(species=species, stochastic=False, compress=False)
        .age_structure(4, 2)
        .initial_state(individual_count={
            "female": {"A|A@uninfected": [0, 20, 80, 0]},
            "male": {"A|A@uninfected": [0, 0, 40, 0]},
        })
        .competition(carrying_capacity=200)
        .hooks(*ops, event="first")
        .build()
    )


def test_age_regression_relabels_only_selected_carriers_and_keeps_sperm() -> None:
    """Regressing adult females preserves genotype and transfers their stored sperm."""
    pop = _build(nt.Op.convert(
        from_=nt.IndividualSelector(sex="female", age=2, ztype="*@uninfected"),
        to=nt.IndividualSelector(age=1, ztype="*@infected"), probability=0.25,
    ))
    genotype = pop.species.get_genotype_from_str("A|A")
    source = pop.registry.ztype_index(genotype, "uninfected")
    target = pop.registry.ztype_index(genotype, "infected")
    initial = pop.state
    initial.sperm_storage[2, source, source] = 40
    pop.import_state(initial)
    expected = initial.individual_count.copy()
    expected[0, 2, source] -= 20
    expected[0, 1, target] += 20
    expected_sperm = initial.sperm_storage.copy()
    expected_sperm[2, source, source] -= 10
    expected_sperm[1, target, source] += 10
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count, expected)
    np.testing.assert_array_equal(pop.state.sperm_storage, expected_sperm)


def test_separate_ops_cascade_through_new_age() -> None:
    """A later op processes both pre-existing and newly converted age-one females."""
    pop = _build(
        nt.Op.convert(
            from_=nt.IndividualSelector(sex="female", age=2),
            to=nt.IndividualSelector(age=1), probability=0.5,
        ),
        nt.Op.convert(
            from_=nt.IndividualSelector(sex="female", age=1),
            to=nt.IndividualSelector(age=0), probability=1.0,
        ),
    )
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count[0].sum(axis=1), [60, 0, 40, 0])
    assert pop.state.individual_count[1].sum() == 40


def test_clear_male_selection_does_not_clear_female_carriers() -> None:
    """Selecting males never targets the male-type columns of female sperm storage."""
    pop = _build(nt.Op.clear_sperm_storage(selector=nt.IndividualSelector(sex="male")))
    initial = pop.state
    initial.sperm_storage[2, 0, 0] = 20
    pop.import_state(initial)
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.sperm_storage, initial.sperm_storage)
    np.testing.assert_array_equal(pop.state.individual_count, initial.individual_count)


@pytest.mark.parametrize("target", [
    nt.IndividualSelector(age=[1, 2]),
    nt.IndividualSelector(sex=["female", "male"]),
    nt.IndividualSelector(age=1) | nt.IndividualSelector(age=2),
    nt.IndividualSelector(age=4),
    nt.IndividualSelector(ztype="*@missing"),
    nt.IndividualSelector(ztype="{A,B}|*@infected"),
])
def test_invalid_targets_fail_at_build(target: nt.IndividualSelector) -> None:
    """Ambiguous and unavailable target states cannot publish a runnable population."""
    with pytest.raises(ValueError):
        _build(nt.Op.convert(from_=nt.IndividualSelector(), to=target, probability=0.25))


def test_selector_and_legacy_endpoints_cannot_mix() -> None:
    """Mixed keyword families must not override one another silently."""
    with pytest.raises(TypeError, match="mix"):
        nt.Op.convert(
            source="A|A", target="B|B", probability=0.25,
            from_=nt.IndividualSelector(), to=nt.IndividualSelector(age=1),
        )


@pytest.mark.parametrize("change_genotype", [False, True])
def test_sex_conversion_requires_a_compatible_destination_genotype(change_genotype: bool) -> None:
    """XX cannot become male unless the conversion also supplies a legal XY genotype."""
    species = nt.Species.from_dict(
        "conversion_xy_constraints",
        {
            "chrX": {"sex_type": "X", "loci": {"x": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {"y": ["Y1"]}},
        },
        unordered=False,
    )
    op = nt.Op.convert(
        from_=nt.IndividualSelector(sex="female", ztype="X1|X1"),
        to=nt.IndividualSelector(sex="male", ztype="X1|Y1" if change_genotype else None),
        probability=0.5,
    )
    builder = (
        nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False, compress=False)
        .initial_state(individual_count={"female": {"X1|X1": 100}})
        .competition(carrying_capacity=100)
        .hooks(op, event="first")
    )
    if not change_genotype:
        with pytest.raises(ValueError, match="incompatible"):
            builder.build()
        return
    pop = builder.build()
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count.sum(axis=(1, 2)), [50, 50])


def test_clear_has_no_effect_on_discrete_model_without_sperm_storage() -> None:
    """A clear operation is valid for the model with no sperm carrier array."""
    species = nt.Species.from_dict("clear_discrete", {"chr": {"loc": ["A"]}})
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False)
        .initial_state(individual_count={"female": {"A|A": 30}, "male": {"A|A": 20}})
        .competition(carrying_capacity=50)
        .hooks(nt.Op.clear_sperm_storage(selector=nt.IndividualSelector()), event="first")
        .build()
    )
    initial = pop.state.individual_count.copy()
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count, initial)
