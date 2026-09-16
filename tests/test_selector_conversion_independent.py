"""Independent per-carrier invariants for selector conversions and sperm clearing."""

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks.types import HookOp, OpType


def test_legacy_conversion_probability_remains_required() -> None:
    """Omitting the historically required probability must not silently mean no-op."""
    with pytest.raises(TypeError):
        nt.Op.convert("A|A", "B|B")


@pytest.mark.parametrize("case", ["mixed-type", "missing-legacy-target", "clear-wrong-type"])
def test_operation_factories_reject_incomplete_or_wrong_endpoint_types(case: str) -> None:
    """Untyped callers get explicit argument errors instead of silently ignored selectors."""
    with pytest.raises(TypeError):
        if case == "mixed-type":
            nt.Op.convert(from_=nt.IndividualSelector(), to="A|A", probability=1)
        elif case == "missing-legacy-target":
            nt.Op.convert(source="A|A", probability=1)
        else:
            nt.Op.clear_sperm_storage(selector="*")


@pytest.mark.parametrize("op", [
    HookOp(OpType.CLEAR_SPERM_STORAGE),
    HookOp(OpType.CONVERT, source_selector=nt.IndividualSelector()),
])
def test_incomplete_operation_descriptors_are_rejected_at_build(op: HookOp) -> None:
    """Direct HookOp declarations must supply the same mandatory selector fields."""
    with pytest.raises((TypeError, ValueError)):
        _population(op)


def test_malformed_target_pattern_has_public_value_error() -> None:
    """The Op compiler reports malformed target syntax consistently with other bad targets."""
    with pytest.raises(ValueError, match="target pattern"):
        _population(nt.Op.convert(
            from_=nt.IndividualSelector(), to=nt.IndividualSelector(ztype="A@@default"), probability=1,
        ))


@pytest.mark.parametrize("sex", [-1, 2])
def test_invalid_target_sex_is_rejected_before_population_is_built(sex: int) -> None:
    """A target sex outside the model axes must fail compilation, not execution."""
    with pytest.raises(ValueError):
        _population(nt.Op.convert(
            from_=nt.IndividualSelector(sex="female", age=2),
            to=nt.IndividualSelector(sex=sex), probability=1,
        ))


def _population(op: HookOp | list[HookOp], *, stochastic: bool = False) -> nt.AgeStructuredPopulation:
    """Build an isolated event fixture; no births, deaths, or aging are executed."""
    species = nt.Species.from_dict(
        name="independent_selector_conversion",
        structure={"chr": {"loc": ["A", "B"]}},
    )
    return (
        nt.AgeStructuredPopulation.setup(species=species, stochastic=stochastic)
        .age_structure(4, 2)
        .initial_state(individual_count={"female": {"A|A": [0, 0, 16, 0]}})
        .competition(carrying_capacity=100)
        .hooks(op, event="first")
        .build()
    )


@pytest.mark.parametrize("destination_sex", ["female", "male"])
def test_stochastic_conversion_moves_mated_carriers_and_buckets_together(destination_sex: str) -> None:
    """Every source female is mated, so retained count must equal retained buckets exactly."""
    op = nt.Op.convert(
        from_=nt.IndividualSelector(ztype="A|A", sex="female", age=2),
        to=nt.IndividualSelector(ztype="B|B", sex=destination_sex, age=1),
        probability=0.5,
    )
    pop = _population(op, stochastic=True)
    registry = pop.registry
    source = registry.ztype_index(pop.species.get_genotype_from_str("A|A"), "default")
    target = registry.ztype_index(pop.species.get_genotype_from_str("B|B"), "default")
    initial = pop.state
    initial.sperm_storage[2, source, source] = 4
    initial.sperm_storage[2, source, target] = 12
    converted = []
    for seed in range(128):
        # Independent seeded sessions; reset initial state before triggering
        # only the event under test, so reproductive dynamics cannot confound it.
        pop._initialize_session(seed=seed)
        pop.import_state(initial)
        pop.trigger_event("first")
        result = pop.state
        counts, sperm = result.individual_count, result.sperm_storage
        assert counts.sum() == 16
        assert counts[0, 2, source] == sperm[2, source].sum()
        destination = 0 if destination_sex == "female" else 1
        moved = counts[destination, 1, target]
        converted.append(moved)
        if destination_sex == "female":
            assert moved == sperm[1, target].sum()
            np.testing.assert_array_equal(sperm.sum(axis=(0, 1)), initial.sperm_storage.sum(axis=(0, 1)))
        else:
            assert sperm[1].sum() == 0
        assert np.all(sperm.sum(axis=-1) <= counts[0])

    # Each conversion total is Binomial(16, 0.5). Across 128 seeds, the
    # sample mean has SD sqrt(4/128); 6 SD tolerates RNG noise but rejects
    # a missing conversion channel or materially incorrect probability.
    assert abs(float(np.mean(converted)) - 8) < 6 * np.sqrt(4 / 128)
    # Binomial(16, .5) has fourth central moment 3*4**2 + 4*(1-6*.25)=46.
    # The exact variance of unbiased sample variance is
    # (mu4 - (n-3)/(n-1)*sigma**4)/n. A six-SD envelope also rejects
    # replacing binomial sampling by deterministic expected counts.
    variance_sd = np.sqrt((46 - (125 / 127) * 16) / 128)
    assert abs(float(np.var(converted, ddof=1)) - 4) < 6 * variance_sd


def test_clear_sperm_union_preserves_unselected_cross_coordinates_and_counts() -> None:
    """Two correlated ZType/age atoms must not clear their rectangular cross-product."""
    selector = (
        nt.IndividualSelector(ztype="A|A", sex="female", age=1)
        | nt.IndividualSelector(ztype="B|B", sex="female", age=2)
    )
    pop = _population(nt.Op.clear_sperm_storage(selector=selector))
    registry = pop.registry
    a = registry.ztype_index(pop.species.get_genotype_from_str("A|A"), "default")
    b = registry.ztype_index(pop.species.get_genotype_from_str("B|B"), "default")
    initial = pop.state
    initial.individual_count[:] = 0
    initial.sperm_storage[:] = 0
    for age, ztype, value in ((1, a, 4), (1, b, 8), (2, a, 12), (2, b, 16)):
        initial.individual_count[0, age, ztype] = value
        initial.sperm_storage[age, ztype, b] = value
    pop.import_state(initial)
    expected = initial.sperm_storage.copy()
    expected[1, a] = 0
    expected[2, b] = 0
    for _ in range(2):
        pop.trigger_event("first")
        np.testing.assert_array_equal(pop.state.individual_count, initial.individual_count)
        np.testing.assert_array_equal(pop.state.sperm_storage, expected)


def test_conversion_source_union_is_correlated_and_deduplicated() -> None:
    """Repeated atoms convert once, without selecting the other age/type combinations."""
    first = nt.IndividualSelector(ztype="A|A", sex="female", age=1)
    second = nt.IndividualSelector(ztype="B|B", sex="female", age=2)
    pop = _population(nt.Op.convert(
        from_=first | second | first,
        to=nt.IndividualSelector(age=0), probability=0.5,
    ))
    registry = pop.registry
    a = registry.ztype_index(pop.species.get_genotype_from_str("A|A"), "default")
    b = registry.ztype_index(pop.species.get_genotype_from_str("B|B"), "default")
    initial = pop.state
    initial.individual_count[:] = 0
    for age, ztype, value in ((1, a, 4), (1, b, 8), (2, a, 12), (2, b, 16)):
        initial.individual_count[0, age, ztype] = value
    pop.import_state(initial)
    expected = initial.individual_count.copy()
    expected[0, 1, a] -= 2
    expected[0, 0, a] += 2
    expected[0, 2, b] -= 8
    expected[0, 0, b] += 8
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count, expected)


def test_male_to_female_conversion_does_not_rewrite_previously_stored_sperm() -> None:
    """Male identity changes cannot retroactively change sperm already held by females."""
    pop = _population(nt.Op.convert(
        from_=nt.IndividualSelector(ztype="A|A", sex="male", age=2),
        to=nt.IndividualSelector(ztype="B|B", sex="female", age=1),
        probability=0.25,
    ))
    registry = pop.registry
    a = registry.ztype_index(pop.species.get_genotype_from_str("A|A"), "default")
    b = registry.ztype_index(pop.species.get_genotype_from_str("B|B"), "default")
    initial = pop.state
    initial.individual_count[1, 2, a] = 40
    initial.sperm_storage[2, a, a] = 12
    pop.import_state(initial)
    expected = initial.individual_count.copy()
    expected[1, 2, a] -= 10
    expected[0, 1, b] += 10
    pop.trigger_event("first")
    np.testing.assert_array_equal(pop.state.individual_count, expected)
    np.testing.assert_array_equal(pop.state.sperm_storage, initial.sperm_storage)


def test_identity_conversion_does_not_consume_the_random_stream() -> None:
    """Adding an identity operation must leave subsequent seeded conversions identical."""
    actual = nt.Op.convert(
        from_=nt.IndividualSelector(ztype="A|A", sex="female", age=2),
        to=nt.IndividualSelector(ztype="B|B"), probability=0.5,
    )
    identity = nt.Op.convert(
        from_=nt.IndividualSelector(), to=nt.IndividualSelector(), probability=0.5,
    )
    reference = _population(actual, stochastic=True)
    with_identity = _population([identity, actual], stochastic=True)
    for seed in range(16):
        # reset rearms each population and reinstalls its initial state.
        for pop in (reference, with_identity):
            pop.reset()
            pop._initialize_session(seed=seed)
            pop.trigger_event("first")
        np.testing.assert_array_equal(with_identity.state.individual_count, reference.state.individual_count)
        np.testing.assert_array_equal(with_identity.state.sperm_storage, reference.state.sperm_storage)


@pytest.mark.parametrize("replace_genotype", [False, True])
def test_sex_conversion_respects_structural_genotype_eligibility(replace_genotype: bool) -> None:
    """XX cannot become male alone; an explicit legal XY replacement can change sex."""
    species = nt.Species.from_dict(
        name="independent_xy_sex_conversion",
        structure={
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )
    target = nt.IndividualSelector(sex="male", ztype="X1|Y1" if replace_genotype else None)
    builder = (
        nt.AgeStructuredPopulation.setup(species=species, stochastic=False)
        .age_structure(3, 1)
        .initial_state(individual_count={"female": {"X1|X1": [0, 10, 0]}, "male": {"X1|Y1": [0, 6, 0]}})
        .competition(carrying_capacity=100)
        .hooks(nt.Op.convert(from_=nt.IndividualSelector(), to=target, probability=1), event="first")
    )
    if not replace_genotype:
        with pytest.raises(ValueError, match="sex.*genotype"):
            builder.build()
        return
    pop = builder.build()
    pop.trigger_event("first")
    assert pop.state.individual_count[0].sum() == 0
    assert pop.state.individual_count[1].sum() == 16
