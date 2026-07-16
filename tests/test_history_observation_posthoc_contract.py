"""Numerical contracts for direct spatial output and post-hoc observation."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.numba.utils import numba_disabled
from natal.output import History
from natal.patterns import IndividualSelector
from natal.spatial.configurator import batch_setting


def _species(name: str) -> nt.Species:
    """Build the shared biallelic species used by the contract fixtures.

    Args:
        name: Species identifier.

    Returns:
        A biallelic species with one locus and default gamete labels.
    """
    return nt.Species.from_dict(
        name,
        {"Chr1": {"L1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _discrete_deme(
    species: nt.Species,
    name: str,
    *,
    female_wild: float,
    male_wild: float,
    custom_observation: bool = False,
    collapse_age: bool = False,
) -> nt.DiscreteGenerationPopulation:
    """Build one deterministic deme with coordinate-distinct initial counts.

    Args:
        species: The shared species definition.
        name: Population identifier.
        female_wild: Base count for female WT|WT individuals.
        male_wild: Base count for male WT|WT individuals.
        custom_observation: Whether to install the canonical two-group rule.
        collapse_age: Whether to sum over the age axis in observation.

    Returns:
        A built discrete-generation deme with raw history recording.
    """
    configurator = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {
                    "WT|WT": female_wild,
                    "WT|Dr": female_wild + 1.0,
                },
                "male": {
                    "WT|WT": male_wild,
                    "Dr|Dr": male_wild + 1.0,
                },
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=2.0,
            carrying_capacity=1000.0,
        )
    )
    if custom_observation:
        configurator.with_observation(
            groups=_observation_groups(),
            collapse_age=collapse_age,
        )
    return configurator.record_history(mode="raw").build()


def _observation_groups() -> OrderedDict[str, IndividualSelector]:
    """Return the canonical two-group rule used by post-hoc projections.

    Returns:
        OrderedDict mapping ``"wild"`` and ``"drive"`` labels to selectors.
    """
    return OrderedDict(
        (
            ("wild", IndividualSelector(ztype="WT|WT")),
            (
                "drive",
                IndividualSelector(ztype="WT|Dr")
                | IndividualSelector(ztype="Dr|Dr"),
            ),
        )
    )


def _direct_spatial_discrete(name: str) -> nt.SpatialPopulation:
    """Construct a spatial population directly, without SpatialConfigurator.

    Args:
        name: Base identifier for species and demes.

    Returns:
        A two-deme spatial population with identity observation.
    """
    species = _species(f"{name}_species")
    demes = [
        _discrete_deme(
            species,
            f"{name}_deme_0",
            female_wild=10.0,
            male_wild=20.0,
        ),
        _discrete_deme(
            species,
            f"{name}_deme_1",
            female_wild=30.0,
            male_wild=40.0,
        ),
    ]
    return nt.SpatialPopulation(demes, migration_rate=0.0, name=name)


def _stack_counts(population: nt.SpatialPopulation) -> NDArray[np.float64]:
    """Copy the stable individual-count tensor from every deme.

    Args:
        population: A built spatial population.

    Returns:
        Stacked ``(n_demes, sex, age, ztype)`` count array.
    """
    return np.stack([deme.state.individual_count.copy() for deme in population.demes])


def _configured_spatial_discrete(name: str) -> nt.SpatialPopulation:
    """Build a raw spatial History with custom canonical observation groups.

    Args:
        name: Base identifier for species and population.

    Returns:
        A two-deme spatial population with custom observation and raw history.
    """
    return (
        nt.SpatialPopulation.builder(
            _species(f"{name}_species"),
            n_demes=2,
            topology=None,
            pop_type="discrete_generation",
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count=batch_setting(
                [
                    {
                        "female": {"WT|WT": 10.0, "WT|Dr": 11.0},
                        "male": {"WT|WT": 20.0, "Dr|Dr": 21.0},
                    },
                    {
                        "female": {"WT|WT": 30.0, "WT|Dr": 31.0},
                        "male": {"WT|WT": 40.0, "Dr|Dr": 41.0},
                    },
                ]
            )
        )
        .reproduction(eggs_per_female=2.0)
        .competition(carrying_capacity=1000.0)
        .with_observation(groups=_observation_groups(), collapse_age=False)
        .record_history(mode="raw")
        .build()
    )


def test_direct_spatial_constructor_installs_identity_observation_and_raw_history() -> None:
    """Direct construction records exact typed snapshots without a configurator."""
    with numba_disabled():
        population = _direct_spatial_discrete("direct_output_defaults")
        initial_counts = _stack_counts(population)
        initial_projection = population.observe()
        population.run(1, record_every=1)
        final_counts = _stack_counts(population)

    expected_identity = np.moveaxis(initial_counts, -1, 0)
    assert population.observation.labels == (
        "WT|WT@default",
        "WT|Dr@default",
        "Dr|Dr@default",
    )
    assert initial_projection.axes == ("group", "deme", "sex", "age")
    np.testing.assert_array_equal(initial_projection.values, expected_identity)
    assert isinstance(population.history, History)
    assert population.history.schema.mode == "raw"
    assert (
        population.observation.population_fingerprint
        == population.history.schema.population.fingerprint
    )
    assert population.history.ticks == (0, 1)
    assert population.history.individual_count.shape == (2, 2, 2, 2, 3)
    np.testing.assert_array_equal(
        population.history.individual_count,
        np.stack((initial_counts, final_counts)),
    )


def test_nonspatial_raw_history_posthoc_collapse_equals_each_projection() -> None:
    """Post-hoc collapse projects every raw record with the supplied rule."""
    species = _species("posthoc_nonspatial_species")
    with numba_disabled():
        population = _discrete_deme(
            species,
            "posthoc_nonspatial",
            female_wild=11.0,
            male_wild=23.0,
            custom_observation=True,
            collapse_age=True,
        )
        population.run(1, record_every=1)
    observation = population.observation

    expected = np.stack(
        [observation.apply(record) for record in population.history.individual_count]
    )
    observed_history = population.history.observe(observation)

    assert (
        observation.population_fingerprint
        == population.history.schema.population.fingerprint
    )
    assert observed_history.schema.mode == "observation"
    assert observed_history.ticks == population.history.ticks == (0, 1)
    assert observed_history.values.shape == (2, 2, 2)
    np.testing.assert_array_equal(observed_history.values, expected)


def test_spatial_raw_history_posthoc_observation_preserves_every_deme() -> None:
    """Post-hoc spatial values are group-first and exact for every record/deme."""
    with numba_disabled():
        population = _configured_spatial_discrete("posthoc_spatial")
        population.run(1, record_every=1)
    observation = population.observation

    expected = np.stack(
        [
            np.stack(
                [observation.apply(deme_counts) for deme_counts in record],
                axis=1,
            )
            for record in population.history.individual_count
        ]
    )
    observed_history = population.history.observe(observation)

    assert (
        observation.population_fingerprint
        == population.history.schema.population.fingerprint
    )
    assert observed_history.ticks == (0, 1)
    assert observed_history.values.shape == (2, 2, 2, 2, 2)
    np.testing.assert_array_equal(observed_history.values, expected)


def test_foreign_same_shape_observation_is_rejected_by_layout_fingerprint() -> None:
    """An Observation from a different same-shaped species cannot project History."""
    native_species = _species("native_layout_species")
    native = _discrete_deme(
        native_species,
        "native_layout",
        female_wild=10.0,
        male_wild=20.0,
    )
    foreign_species = nt.Species.from_dict(
        "foreign_layout_species",
        {"Chr1": {"L1": ["A", "B"]}},
        gamete_labels=["default"],
    )
    foreign = (
        nt.DiscreteGenerationPopulation.setup(
            species=foreign_species,
            name="foreign_layout",
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"A|A": 10.0, "A|B": 11.0},
                "male": {"A|A": 20.0, "B|B": 21.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=2.0,
            carrying_capacity=1000.0,
        )
        .record_history(mode="raw")
        .build()
    )

    assert native.state.individual_count.shape == foreign.state.individual_count.shape
    assert (
        native.history.schema.population.fingerprint
        != foreign.observation.population_fingerprint
    )
    expected_message = (
        "Observation population layout does not match History: expected "
        f"{native.history.schema.population.fingerprint}, got "
        f"{foreign.observation.population_fingerprint}."
    )
    with pytest.raises(ValueError) as error:
        native.history.observe(foreign.observation)
    assert str(error.value) == expected_message


def test_spatial_age_restore_checkpoint_restores_all_demes_and_truncates() -> None:
    """Checkpoint restore resets counts, sperm, all ticks, and future records."""
    with numba_disabled():
        population = (
            nt.SpatialPopulation.builder(
                _species("restore_spatial_age_species"),
                n_demes=2,
                topology=None,
                pop_type="age_structured",
            )
            .setup(name="restore_spatial_age", stochastic=False)
            .age_structure(n_ages=4, new_adult_age=2)
            .initial_state(
                individual_count=batch_setting(
                    [
                        {
                            "female": {"WT|WT": [1.0, 2.0, 3.0, 4.0]},
                            "male": {"WT|WT": [5.0, 6.0, 7.0, 8.0]},
                        },
                        {
                            "female": {"WT|WT": [9.0, 10.0, 11.0, 12.0]},
                            "male": {"WT|WT": [13.0, 14.0, 15.0, 16.0]},
                        },
                    ]
                ),
                sperm_storage=batch_setting(
                    [
                        {"WT|WT": {"WT|WT": {2: 17.0, 3: 18.0}}},
                        {"WT|WT": {"WT|WT": {2: 19.0, 3: 20.0}}},
                    ]
                ),
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.8, 0.0],
                male_age_based_survival=[1.0, 0.9, 0.8, 0.0],
            )
            .reproduction(
                eggs_per_female=2.0,
                female_age_based_mating_rate=[0.0, 0.0, 0.3, 0.5],
                male_age_based_mating_rate=[0.0, 0.0, 0.3, 0.5],
            )
            .competition(
                carrying_capacity=1000.0,
                expected_num_new_adult_females=10.0,
            )
            .record_history(mode="raw")
            .build()
        )
        population.run(2, record_every=1)

    assert population.history.ticks == (0, 1, 2)
    expected_counts = population.history.individual_count[1].copy()
    recorded_sperm = population.history.sperm_storage
    assert recorded_sperm is not None
    expected_sperm = recorded_sperm[1].copy()
    for deme in population.demes:
        deme.state.individual_count.fill(901.0)
        deme.state.sperm_storage.fill(902.0)

    population.restore_checkpoint(1)

    assert population.tick == 1
    assert population.history.ticks == (0, 1)
    for deme_index, deme in enumerate(population.demes):
        assert deme.tick == 1
        assert deme.state.n_tick == 1
        np.testing.assert_array_equal(
            deme.state.individual_count,
            expected_counts[deme_index],
        )
        np.testing.assert_array_equal(
            deme.state.sperm_storage,
            expected_sperm[deme_index],
        )
