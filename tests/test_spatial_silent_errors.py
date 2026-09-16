"""Regression contracts for ignored declarations and stale spatial reads."""

from typing import Literal

import numpy as np
import pytest

import natal as nt


Model = Literal["age_structured", "discrete_generation"]


def _builder(model: Model = "discrete_generation"):
    species = nt.Species.from_dict(name="silent_errors", structure={"auto": {"A": ["WT"]}})
    builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type=model).setup(stochastic=False)
    if model == "age_structured":
        builder = builder.age_structure(n_ages=2, new_adult_age=1)
    return (
        builder.initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .competition(carrying_capacity=500)
    )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_discrete_reproduction_preserves_fixed_egg_count(fixed: bool, heterogeneous: bool) -> None:
    """Both build paths forward the declared execution flag to every native deme."""
    eggs = nt.batch_setting([2.0, 3.0]) if heterogeneous else 2.0
    pop = _builder().reproduction(eggs_per_female=eggs, fixed_egg_count=fixed).build()
    assert pop.blueprint.fixed_egg_count is fixed
    assert all(pop.deme(i).config.fixed_egg_count is fixed for i in range(2))


@pytest.mark.parametrize("field", ["carrying_capacity", "migration_rate", "misspelled_parameter"])
def test_spatial_params_assignment_is_rejected(field: str) -> None:
    """Neither temporary nor retained views silently accept parameter assignments."""
    pop = _builder().build()
    view = pop.params
    before = pop.params.carrying_capacity.copy()
    for target in (view, pop.params):
        with pytest.raises(AttributeError):
            setattr(target, field, 5)
    np.testing.assert_array_equal(pop.params.carrying_capacity, before)
    np.testing.assert_array_equal(view.carrying_capacity, before)
    view.tensor_write("carrying_capacity", 250.0)
    np.testing.assert_array_equal(pop.params.carrying_capacity, [250.0, 250.0])


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("explicit_event", [False, True])
def test_callback_state_queries_fail_instead_of_returning_old_cache(
    model: Model, warm_cache: bool, explicit_event: bool,
) -> None:
    """State and count queries reject callback reads regardless of cache freshness."""
    visits: list[int] = []

    @nt.hook(event="early")
    def inspect(ctx: nt.TickContext) -> None:
        assert ctx.metrics.total == float(ctx.state.individual_count.sum())
        visits.append(ctx.deme_id)
        for index in range(2):
            deme = pop.deme(index)
            queries = (
                lambda: deme.state,
                deme.get_total_count,
                deme.get_female_count,
                deme.get_male_count,
                deme.export_state,
            )
            for query in queries:
                with pytest.raises(RuntimeError, match="ctx.state or ctx.metrics"):
                    query()
        for query in (
            pop.get_total_count, pop.get_female_count, pop.get_male_count,
            pop.aggregate_individual_count, pop.aggregate_state,
        ):
            with pytest.raises(RuntimeError, match="ctx.state or ctx.metrics"):
                query()

    pop = _builder(model).reproduction(eggs_per_female=2).hooks(inspect).build()
    control = _builder(model).reproduction(eggs_per_female=2).build()
    if warm_cache:
        for index in range(2):
            _ = pop.deme(index).state
    if explicit_event:
        pop.trigger_event("early", deme_id=0)
        control.trigger_event("early", deme_id=0)
        assert visits == [0]
    else:
        pop.run(2, record_every=0)
        control.run(2, record_every=0)
        assert visits == [0, 1, 0, 1]
    np.testing.assert_array_equal(pop.aggregate_individual_count(), control.aggregate_individual_count())
    assert pop.get_total_count() == control.get_total_count()
    for index in range(2):
        np.testing.assert_array_equal(pop.deme(index).state.individual_count, control.deme(index).state.individual_count)


@pytest.mark.parametrize("explicit_event", [False, True])
def test_callback_query_failure_releases_query_guard(explicit_event: bool) -> None:
    """An uncaught forbidden query cannot leave boundary reads disabled."""
    @nt.hook(event="early")
    def reject(ctx: nt.TickContext) -> None:
        pop.deme(1).get_total_count()

    pop = _builder().hooks(reject).build()
    with pytest.raises(RuntimeError, match="ctx.state or ctx.metrics"):
        if explicit_event:
            pop.trigger_event("early", deme_id=0)
        else:
            pop.run(1, record_every=0)
    total = pop.aggregate_individual_count().sum()
    assert pop.get_total_count() == total
    assert sum(pop.deme(i).state.individual_count.sum() for i in range(2)) == total


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("setup_flag", [False, True])
@pytest.mark.parametrize("reproduction_flag", [None, False, True])
def test_reproduction_flag_precedence(
    model: Model, heterogeneous: bool, setup_flag: bool, reproduction_flag: bool | None,
) -> None:
    """Only an explicit reproduction flag overrides the setup declaration."""
    eggs = nt.batch_setting([2.0, 3.0]) if heterogeneous else 2.0
    builder = _builder(model).setup(fixed_egg_count=setup_flag)
    if reproduction_flag is None:
        pop = builder.reproduction(eggs_per_female=eggs).build()
    else:
        pop = builder.reproduction(eggs_per_female=eggs, fixed_egg_count=reproduction_flag).build()
    expected = setup_flag if reproduction_flag is None else reproduction_flag
    assert pop.blueprint.fixed_egg_count is expected
    assert all(pop.deme(i).config.fixed_egg_count is expected for i in range(2))
