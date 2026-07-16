"""Public contract tests for canonical Observation and History configuration."""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.output import History
from natal.patterns import IndividualSelector
from natal.ui.dashboard_population import Dashboard

InvalidGroups: TypeAlias = (
    None
    | list[IndividualSelector]
    | tuple[IndividualSelector, ...]
    | dict[str, dict[str, str]]
    | dict[str, IndividualSelector]
    | dict[int, IndividualSelector]
)


def _configurator(name: str) -> nt.AgeStructuredConfigurator:
    """Create a deterministic four-age configurator for contract tests.

    Args:
        name: Identifier used to name the species and population.

    Returns:
        A chainable configurator ready for further customization.
    """
    species = nt.Species.from_dict(
        name=f"{name}_species",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )
    return (
        nt.AgeStructuredPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
            continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [1.0, 2.0, 3.0, 4.0]},
                "male": {"WT|WT": [5.0, 6.0, 7.0, 8.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.9, 0.8],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=500,
            expected_num_new_adult_females=10,
        )
    )


def _build_population(
    name: str,
    *,
    groups: OrderedDict[str, IndividualSelector] | None = None,
    collapse_age: bool = False,
    history_mode: Literal["raw", "observation"] | None = None,
) -> nt.AgeStructuredPopulation:
    """Build a population with independently configurable output policies.

    Args:
        name: Identifier used to name the species and population.
        groups: Observation groups mapping labels to selectors (identity if None).
        collapse_age: Whether to sum over the age axis.
        history_mode: ``"raw"`` or ``"observation"``; no recording if None.

    Returns:
        A built population whose configuration is frozen.
    """
    configurator = _configurator(name)
    if groups is not None:
        configurator.with_observation(groups=groups, collapse_age=collapse_age)
    if history_mode is not None:
        configurator.record_history(mode=history_mode)
    return configurator.build()


def _coordinate_unique_counts(n_ztypes: int) -> NDArray[np.float64]:
    """Return values whose decimal positions encode ztype, sex, and age.

    Args:
        n_ztypes: Number of ZType slots to allocate.

    Returns:
        A ``(2, 4, n_ztypes)`` array with coordinate-encoding values.
    """
    counts = np.empty((2, 4, n_ztypes), dtype=np.float64)
    for sex in range(2):
        for age in range(4):
            for ztype in range(n_ztypes):
                counts[sex, age, ztype] = 100.0 * ztype + 10.0 * sex + age + 1.0
    return counts


def _ztype_labels(population: nt.AgeStructuredPopulation) -> tuple[str, ...]:
    """Return the stable canonical labels for the population's active ZTypes.

    Args:
        population: A built population whose index registry provides the labels.

    Returns:
        Tuple of ``"Genotype@slab"`` strings in registry order.
    """
    return tuple(
        f"{genotype}@{slab}"
        for genotype, slab in population.index_registry.index_to_ztype
    )


def test_default_raw_population_has_canonical_identity_observation() -> None:
    """A default raw build still owns a lossless canonical Observation."""
    population = _build_population("contract_default_identity")

    assert population.observation is not None
    assert population.observation.labels == _ztype_labels(population)
    history = population.history
    assert isinstance(history, History)
    assert history.schema.mode == "raw"
    assert (
        population.observation.population_fingerprint
        == history.schema.population.fingerprint
    )


def test_identity_projection_preserves_group_sex_age_coordinates() -> None:
    """Identity projection maps every source coordinate to the declared axes."""
    population = _build_population("contract_identity_axes")
    observation = population.observation
    assert observation is not None
    counts = _coordinate_unique_counts(population.index_registry.n_ztypes)
    population.state.individual_count[...] = counts

    expected = np.moveaxis(counts, 2, 0)
    np.testing.assert_array_equal(observation.apply(counts), expected)

    result = population.observe()
    assert result is not None
    assert result.tick == 0
    assert result.axes == ("group", "sex", "age")
    assert result.labels == {"group": _ztype_labels(population)}
    np.testing.assert_array_equal(result.values, expected)
    assert result.values.shape == (3, 2, 4)
    assert float(result.values.sum()) == float(counts.sum())


def test_collapse_age_sums_only_the_age_axis() -> None:
    """collapse_age removes age while preserving exact group and sex coordinates."""
    groups = OrderedDict(
        (
            ("wild", IndividualSelector(ztype="WT|WT")),
            ("heterozygous", IndividualSelector(ztype="WT|Dr")),
            ("drive", IndividualSelector(ztype="Dr|Dr")),
        )
    )
    population = _build_population(
        "contract_collapse_age",
        groups=groups,
        collapse_age=True,
    )
    counts = _coordinate_unique_counts(population.index_registry.n_ztypes)
    population.state.individual_count[...] = counts

    expected = np.moveaxis(counts, 2, 0).sum(axis=2)
    result = population.observe()
    assert result is not None
    assert result.axes == ("group", "sex")
    assert result.labels == {"group": tuple(groups)}
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.values.sum(axis=0), counts.sum(axis=(1, 2)))


def test_explicit_observation_does_not_change_raw_history_mode() -> None:
    """with_observation defines projection rules without changing storage mode."""
    groups = OrderedDict((("all", IndividualSelector()),))
    population = _build_population(
        "contract_explicit_observation_raw_history",
        groups=groups,
        history_mode="raw",
    )

    assert population.observation is not None
    assert population.observation.labels == ("all",)
    history = population.history
    assert isinstance(history, History)
    assert history.schema.mode == "raw"


def test_observation_history_uses_automatic_identity_observation() -> None:
    """Observation recording without explicit groups uses canonical identity."""
    population = _build_population(
        "contract_identity_observation_history",
        history_mode="observation",
    )

    assert population.observation is not None
    assert population.observation.labels == _ztype_labels(population)
    history = population.history
    assert isinstance(history, History)
    assert history.schema.mode == "observation"


def test_explicit_observation_can_use_observation_history() -> None:
    """Explicit projection rules can independently select observation storage."""
    groups = OrderedDict(
        (
            (
                "drive",
                IndividualSelector(ztype="WT|Dr")
                | IndividualSelector(ztype="Dr|Dr"),
            ),
        )
    )
    population = _build_population(
        "contract_explicit_observation_history",
        groups=groups,
        history_mode="observation",
    )

    assert population.observation.labels == ("drive",)
    assert population.history.schema.mode == "observation"


def test_observation_and_history_configuration_order_is_irrelevant() -> None:
    """Swapping the two chain calls produces the same frozen policies."""
    groups = OrderedDict((("wild", IndividualSelector(ztype="WT|WT")),))
    first_observation = (
        _configurator("contract_order_observation_first")
        .with_observation(groups=groups, collapse_age=True)
        .record_history(mode="observation", max_rows=7)
        .build()
    )
    first_history = (
        _configurator("contract_order_history_first")
        .record_history(mode="observation", max_rows=7)
        .with_observation(groups=groups, collapse_age=True)
        .build()
    )

    assert first_observation.observation.labels == first_history.observation.labels
    assert (
        first_observation.observation.collapse_age
        is first_history.observation.collapse_age
        is True
    )
    assert first_observation.history.schema.mode == first_history.history.schema.mode
    assert first_observation.history.max_rows == first_history.history.max_rows == 7


def test_runtime_configurator_rejects_output_schema_mutation() -> None:
    """A built Population cannot replace Observation or History policy."""
    population = _build_population("contract_runtime_mutation")

    with pytest.raises(RuntimeError, match="build phase"):
        population.update().with_observation(
            groups={"all": IndividualSelector()}
        )
    with pytest.raises(RuntimeError, match="build phase"):
        population.update().record_history(mode="observation")


def test_dashboard_rebuilds_typed_state_from_public_raw_history() -> None:
    """Dashboard consumes typed History arrays instead of legacy flat rows."""
    population = _build_population("contract_dashboard_typed_history")
    population.run(n_steps=1, record_every=1)
    history = population.history
    assert len(history) == 2

    dashboard = object.__new__(Dashboard)
    dashboard.pop = population
    first_state = dashboard._raw_history_state(0)
    assert first_state.n_tick == history.ticks[0]
    np.testing.assert_array_equal(
        first_state.individual_count,
        history.individual_count[0],
    )


@pytest.mark.parametrize(
    "groups",
    [
        None,
        [],
        (),
        {},
        {"legacy": {"ztype": "WT|WT"}},
        {"": IndividualSelector(ztype="WT|WT")},
        {1: IndividualSelector(ztype="WT|WT")},
    ],
    ids=[
        "none",
        "list",
        "tuple",
        "empty",
        "legacy-value",
        "empty-label",
        "non-string-label",
    ],
)
def test_with_observation_rejects_noncanonical_groups(groups: InvalidGroups) -> None:
    """Groups must be a non-empty ordered string-to-selector mapping.

    Args:
        groups: An invalid groups value supplied via parametrize.
    """
    with pytest.raises((TypeError, ValueError)):
        _configurator(f"contract_invalid_groups_{id(groups)}").with_observation(
            groups=groups
        ).build()


# ── Negative contract tests: deleted interfaces must not be accessible ──


def test_deleted_record_observation_not_accessible() -> None:
    """pop.record_observation must not be callable or settable."""
    pop = _build_population("neg_record_obs", history_mode="raw")
    assert not hasattr(pop, "record_observation")
    assert "record_observation" not in dir(pop)


def test_deleted_set_observations_not_accessible() -> None:
    """pop.set_observations must not be callable."""
    pop = _build_population("neg_set_obs", history_mode="raw")
    assert not hasattr(pop, "set_observations")
    assert "set_observations" not in dir(pop)


def test_deleted_create_observation_not_accessible() -> None:
    """pop.create_observation must not be callable."""
    pop = _build_population("neg_create_obs", history_mode="raw")
    assert not hasattr(pop, "create_observation")
    assert "create_observation" not in dir(pop)


def test_deleted_get_history_not_accessible() -> None:
    """pop.get_history must not be callable."""
    pop = _build_population("neg_get_hist", history_mode="raw")
    assert not hasattr(pop, "get_history")
    assert "get_history" not in dir(pop)


def test_deleted_output_current_state_not_importable() -> None:
    """output_current_state must not be importable from natal.output.translation."""
    with pytest.raises(ImportError):
        from natal.output.translation import output_current_state  # noqa: F401


def test_deleted_output_history_not_importable() -> None:
    """output_history must not be importable from natal.output.translation."""
    with pytest.raises(ImportError):
        from natal.output.translation import output_history  # noqa: F401


def test_deleted_spatial_population_output_history_not_importable() -> None:
    """spatial_population_output_history must not be importable."""
    with pytest.raises(ImportError):
        from natal.output.translation import spatial_population_output_history  # noqa: F401


def test_history_has_no_public_append() -> None:
    """History must not expose a public append method."""
    pop = _build_population("neg_append", history_mode="raw")
    pop.run(1)
    assert not hasattr(pop.history, "append")
    assert "append" not in dir(pop.history)


def test_history_has_no_public_to_numpy() -> None:
    """History must not expose a public to_numpy method."""
    pop = _build_population("neg_tonumpy", history_mode="raw")
    pop.run(1)
    assert not hasattr(pop.history, "to_numpy")
    assert "to_numpy" not in dir(pop.history)


def test_observationfilter_not_publicly_exported() -> None:
    """ObservationFilter must not be importable from natal.output."""
    with pytest.raises(ImportError):
        from natal.output import ObservationFilter  # noqa: F401


def test_historybatch_not_publicly_exported() -> None:
    """HistoryBatch must not be importable from natal.output."""
    with pytest.raises(ImportError):
        from natal.output import HistoryBatch  # noqa: F401


def test_history_observe_rejects_legacy_mask() -> None:
    """History.observe() must reject bare ndarray (legacy mask path
    deleted)."""
    pop = _build_population("neg_legacy_mask", history_mode="raw")
    pop.run(1)
    with pytest.raises((TypeError, AttributeError)):
        pop.history.observe(np.zeros((1, 2, 1, 1)))  # type: ignore[arg-type]


def test_observationfilter_create_observation_deleted() -> None:
    """ObservationFilter.create_observation must not be callable."""
    from natal.output.observation import ObservationFilter
    assert not hasattr(ObservationFilter, "create_observation")
    assert "create_observation" not in dir(ObservationFilter)
