"""Numerical runtime contracts for History and Observation integration."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.numba.utils import numba_disabled
from natal.output import History
from natal.patterns import IndividualSelector
from natal.spatial.configurator import (
    _float_value,  # type: ignore[reportPrivateUsage]  # directly verify replay-log type boundary
    _object_sequence,  # type: ignore[reportPrivateUsage]  # directly verify positional replay boundary
    batch_setting,
)
from natal.ui.spatial_dashboard import SpatialDashboard


def _species(name: str) -> nt.Species:
    """Build the biallelic species used by all runtime contract tests.

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


def _discrete_population(
    name: str,
    *,
    observation_history: bool,
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic non-spatial discrete population.

    Args:
        name: Base identifier for species and population.
        observation_history: If True, install observation groups and
            record observation history; otherwise record raw history.

    Returns:
        A built discrete-generation population.
    """
    configurator = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"{name}_species"),
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=2.0,
            carrying_capacity=100,
        )
    )
    if observation_history:
        configurator.with_observation(
            groups=OrderedDict(
                (
                    ("wild", IndividualSelector(ztype="WT|WT")),
                    (
                        "drive",
                        IndividualSelector(ztype="WT|Dr")
                        | IndividualSelector(ztype="Dr|Dr"),
                    ),
                )
            ),
            collapse_age=True,
        ).record_history(mode="observation")
    else:
        configurator.record_history(mode="raw")
    return configurator.build()


def _spatial_discrete(
    name: str,
    *,
    observation_history: bool,
) -> nt.SpatialPopulation:
    """Build two deterministic discrete demes with distinguishable counts.

    Args:
        name: Base identifier for species and population.
        observation_history: If True, record observation history;
            otherwise record raw history (both use explicit groups).

    Returns:
        A built two-deme spatial population.
    """
    configurator = (
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
                        "female": {"WT|WT": 10.0, "WT|Dr": 20.0},
                        "male": {"WT|WT": 30.0, "Dr|Dr": 40.0},
                    },
                    {
                        "female": {"WT|WT": 50.0, "Dr|Dr": 60.0},
                        "male": {"WT|Dr": 70.0, "Dr|Dr": 80.0},
                    },
                ]
            )
        )
        .reproduction(eggs_per_female=2.0)
        .competition(carrying_capacity=1000.0)
    )
    if observation_history:
        configurator.with_observation(
            groups=OrderedDict(
                (
                    ("wild", IndividualSelector(ztype="WT|WT")),
                    ("drive", IndividualSelector(ztype="Dr|Dr")),
                )
            ),
            collapse_age=True,
        ).record_history(mode="observation")
    else:
        configurator.with_observation(
            groups=OrderedDict(
                (
                    ("wild", IndividualSelector(ztype="WT|WT")),
                    ("drive", IndividualSelector(ztype="Dr|Dr")),
                )
            )
        ).record_history(mode="raw")
    return configurator.build()


def _stack_individual_count(population: nt.SpatialPopulation) -> NDArray[np.float64]:
    """Copy all deme count tensors at one stable tick.

    Args:
        population: A built spatial population.

    Returns:
        Stacked ``(n_demes, sex, age, ztype)`` count array.
    """
    return np.stack([deme.state.individual_count.copy() for deme in population.demes])


def _stack_sperm_storage(population: nt.SpatialPopulation) -> NDArray[np.float64]:
    """Copy all deme sperm tensors at one stable tick.

    Args:
        population: A built spatial population.

    Returns:
        Stacked ``(n_demes, n_ages, n_ztypes, n_ztypes)`` sperm array.
    """
    return np.stack([deme.state.sperm_storage.copy() for deme in population.demes])


def test_nonspatial_observation_history_equals_direct_projection_each_tick() -> None:
    """Recorded collapsed values equal the canonical projection at every tick."""
    with numba_disabled():
        population = _discrete_population(
            "nonspatial_observation_history",
            observation_history=True,
        )
        expected_tick_zero = population.observation.apply(
            population.state.individual_count.copy()
        )
        population.run(n_steps=1, record_every=1)
        expected_tick_one = population.observation.apply(
            population.state.individual_count.copy()
        )

    expected = np.stack((expected_tick_zero, expected_tick_one))
    assert population.history.ticks == (0, 1)
    assert population.history.values.shape == (2, 2, 2)
    np.testing.assert_array_equal(population.history.values, expected)


def test_spatial_raw_history_preserves_exact_discrete_deme_snapshots() -> None:
    """Public spatial raw History retains every deme coordinate at each tick."""
    with numba_disabled():
        population = _spatial_discrete(
            "spatial_raw_history",
            observation_history=False,
        )
        expected_tick_zero = _stack_individual_count(population)
        population.run(1, record_every=1)
        expected_tick_one = _stack_individual_count(population)

    assert isinstance(population.history, History)
    assert population.history.ticks == (0, 1)
    expected = np.stack((expected_tick_zero, expected_tick_one))
    assert population.history.individual_count.shape == (2, 2, 2, 2, 3)
    np.testing.assert_array_equal(population.history.individual_count, expected)


def test_spatial_observe_is_group_first_and_preserves_deme_coordinates() -> None:
    """Spatial projection maps known ZTypes to group,deme,sex,age exactly."""
    with numba_disabled():
        population = _spatial_discrete(
            "spatial_observe_axes",
            observation_history=False,
        )
    counts = np.arange(1.0, 25.0).reshape(2, 2, 2, 3)
    for deme_index, deme in enumerate(population.demes):
        deme.state.individual_count[...] = counts[deme_index]

    result = population.observe()
    expected = np.stack((counts[..., 0], counts[..., 2]), axis=0)

    assert result.tick == 0
    assert result.axes == ("group", "deme", "sex", "age")
    assert result.labels == {"group": ("wild", "drive")}
    assert result.values.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result.values, expected)


def test_spatial_collapsed_observation_history_equals_each_stable_projection() -> None:
    """Spatial observation History stores group,deme,sex without losing a deme."""
    with numba_disabled():
        population = _spatial_discrete(
            "spatial_observation_history",
            observation_history=True,
        )
        expected_tick_zero = population.observe().values.copy()
        population.run(1, record_every=1)
        expected_tick_one = population.observe().values.copy()

    expected = np.stack((expected_tick_zero, expected_tick_one))
    assert population.history.ticks == (0, 1)
    assert population.history.values.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(population.history.values, expected)


def test_spatial_age_raw_history_preserves_exact_sperm_and_count_snapshots() -> None:
    """Age-structured raw History retains the full per-deme sperm tensor."""
    with numba_disabled():
        population = (
            nt.SpatialPopulation.builder(
                _species("spatial_age_raw_species"),
                n_demes=2,
                topology=None,
                pop_type="age_structured",
            )
            .setup(name="spatial_age_raw", stochastic=False)
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
        expected_count_zero = _stack_individual_count(population)
        expected_sperm_zero = _stack_sperm_storage(population)
        population.run(1, record_every=1)
        expected_count_one = _stack_individual_count(population)
        expected_sperm_one = _stack_sperm_storage(population)

    sperm_storage = population.history.sperm_storage
    assert sperm_storage is not None
    assert population.history.individual_count.shape == (2, 2, 2, 4, 3)
    assert sperm_storage.shape == (2, 2, 4, 3, 3)
    np.testing.assert_array_equal(
        population.history.individual_count,
        np.stack((expected_count_zero, expected_count_one)),
    )
    np.testing.assert_array_equal(
        sperm_storage,
        np.stack((expected_sperm_zero, expected_sperm_one)),
    )


def test_nonspatial_kernel_rows_collapse_age_before_history_commit() -> None:
    """Kernel observation rows are reduced to the frozen collapsed schema."""
    with numba_disabled():
        population = _discrete_population(
            "nonspatial_kernel_collapse",
            observation_history=True,
        )
    uncollapsed = np.arange(1.0, 9.0).reshape(2, 2, 2)
    row = np.concatenate((np.array([7.0]), uncollapsed.ravel()))[np.newaxis, :]

    population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify engine-to-History boundary
        row,
        clear_history_on_start=False,
    )

    assert population.history.ticks == (7,)
    assert population.history.values.shape == (1, 2, 2)
    np.testing.assert_array_equal(
        population.history.values[0],
        uncollapsed.sum(axis=-1),
    )


def test_spatial_kernel_rows_project_observation_and_trim_raw_transport() -> None:
    """Spatial engine transport commits only the values declared by each schema."""
    with numba_disabled():
        observed_population = _spatial_discrete(
            "spatial_kernel_observation",
            observation_history=True,
        )
        raw_population = _spatial_discrete(
            "spatial_kernel_raw",
            observation_history=False,
        )

    observed_counts = _stack_individual_count(observed_population)
    observed_row = np.concatenate(
        (np.array([7.0]), observed_counts.ravel(), np.array([901.0, 902.0]))
    )[np.newaxis, :]
    expected_observation = observed_population.observe().values
    observed_population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify raw spatial transport projection
        observed_row,
        clear_history_on_start=False,
    )

    assert observed_population.history.ticks == (7,)
    np.testing.assert_array_equal(
        observed_population.history.values[0],
        expected_observation,
    )

    raw_counts = _stack_individual_count(raw_population)
    raw_row = np.concatenate(
        (np.array([9.0]), raw_counts.ravel(), np.array([903.0, 904.0]))
    )[np.newaxis, :]
    raw_population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify transport-only payload removal
        raw_row,
        clear_history_on_start=False,
    )

    assert raw_population.history.ticks == (9,)
    assert raw_population.history._to_numpy().shape == (1, raw_population.history.schema.row_size)
    np.testing.assert_array_equal(raw_population.history.individual_count[0], raw_counts)


def test_spatial_output_accessors_reject_unbuilt_state_and_shape_empty_history() -> None:
    """Spatial output accessors distinguish unbuilt state from empty history."""
    unbuilt = object.__new__(nt.SpatialPopulation)
    unbuilt._history_obj = None  # type: ignore[reportPrivateUsage]  # construct the pre-build boundary
    unbuilt._observation = None  # type: ignore[reportPrivateUsage]  # construct the pre-build boundary

    try:
        _ = unbuilt.history
    except RuntimeError as error:
        assert str(error) == "History is not initialized for this population."
    else:
        raise AssertionError("Unbuilt spatial history must be rejected")

    try:
        _ = unbuilt.observation
    except RuntimeError as error:
        assert str(error) == "Observation is not initialized for this population."
    else:
        raise AssertionError("Unbuilt spatial observation must be rejected")
    with pytest.raises(RuntimeError, match="History is not initialized"):
        _ = unbuilt.history

    with numba_disabled():
        built = _spatial_discrete(
            "spatial_empty_history",
            observation_history=False,
        )
    assert built.history._to_numpy().shape == (0, built.history.schema.row_size)


def test_spatial_dashboard_rebuilds_exact_totals_from_typed_raw_history() -> None:
    """Spatial charts consume History tensors without legacy flat-row parsing."""
    with numba_disabled():
        population = _spatial_discrete(
            "spatial_dashboard_history",
            observation_history=False,
        )
        population.run(1, record_every=1)

    dashboard = object.__new__(SpatialDashboard)
    dashboard.pop = population
    dashboard._chart_history = []
    dashboard._allele_freq_history = {}
    dashboard._last_chart_tick = -1
    dashboard._rebuild_chart_history()

    expected = [
        [float(tick), float(counts.sum())]
        for tick, counts in zip(
            population.history.ticks,
            population.history.individual_count,
        )
    ]
    assert dashboard._chart_history == expected
    assert dashboard._last_chart_tick == population.history.ticks[-1]
    assert set(dashboard._allele_freq_history) == {"WT", "Dr"}


def test_age_structured_snapshot_collapses_canonical_observation_exactly() -> None:
    """Age snapshot recording removes only age and preserves group and sex."""
    with numba_disabled():
        population = (
            nt.AgeStructuredPopulation.setup(
                species=_species("age_snapshot_species"),
                name="age_snapshot",
                stochastic=False,
                continuous_sampling=False,
            )
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [1.0, 2.0, 3.0, 4.0]},
                    "male": {"Dr|Dr": [5.0, 6.0, 7.0, 8.0]},
                }
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
                male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
                eggs_per_female=2.0,
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.8],
                male_age_based_survival=[1.0, 0.9, 0.8],
            )
            .competition(
                juvenile_growth_mode="concave",
                old_juvenile_carrying_capacity=500.0,
                expected_num_new_adult_females=10.0,
            )
            .with_observation(
                groups=OrderedDict(
                    (
                        ("wild", IndividualSelector(ztype="WT|WT")),
                        ("drive", IndividualSelector(ztype="Dr|Dr")),
                    )
                ),
                collapse_age=True,
            )
            .record_history(mode="observation")
            .build()
        )
    expected = population.observation.apply(population.state.individual_count)

    population.record_snapshot()

    assert population.history.ticks == (0,)
    assert population.history.values.shape == (1, 2, 2)
    np.testing.assert_array_equal(population.history.values[0], expected)


def test_spatial_replay_type_boundaries_accept_numpy_scalar_and_reject_text() -> None:
    """Replay helpers preserve numeric value and reject string-as-sequence input."""
    assert _float_value(np.int64(7), name="carrying_capacity") == 7.0
    with pytest.raises(
        TypeError,
        match="carrying_capacity must be numeric, got str",
    ):
        _float_value("seven", name="carrying_capacity")
    with pytest.raises(
        TypeError,
        match="preset_list must be a sequence, got str",
    ):
        _object_sequence("not positional arguments", name="preset_list")


# ── Lifecycle transition tests ────────────────────────────────────────────────


def test_restore_then_run_continues_from_target_tick() -> None:
    """restore → run: population continues from the restored tick."""
    species = _species("lc_restore")
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species, name="lc_restore_pop",
            stochastic=False, continuous_sampling=False,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": [1.0, 2.0, 3.0]},
            "male": {"WT|WT": [4.0, 5.0, 6.0]},
        })
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9],
            male_age_based_survival=[1.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=200,
            expected_num_new_adult_females=5,
        )
        .record_history(mode="raw")
        .build()
    )
    pop.run(5)
    assert pop.tick == 5
    # Restore to tick 2
    pop.restore_checkpoint(2)
    assert pop.tick == 2
    # Continue
    pop.run(3)
    assert pop.tick == 5, f"Expected tick 5 after restore+run, got {pop.tick}"
    assert len(pop.history) == 6, f"Expected 6 records (0-2 + 3-5), got {len(pop.history)}"


def test_finish_then_snapshot_records_current_tick() -> None:
    """finish → snapshot: manual snapshot works after finish_simulation."""
    species = _species("lc_finish")
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species, name="lc_finish_pop",
            stochastic=False, continuous_sampling=False,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": [1.0, 2.0, 3.0]},
            "male": {"WT|WT": [4.0, 5.0, 6.0]},
        })
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9],
            male_age_based_survival=[1.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=200,
            expected_num_new_adult_females=5,
        )
        .record_history(mode="raw")
        .build()
    )
    pop.run(3, finish=True, record_every=0)
    # No auto-record → tick 3 not yet in history
    assert len(pop.history) == 0
    pop.record_snapshot()
    assert pop.tick in pop.history.ticks, (
        f"Snapshot did not record tick {pop.tick}"
    )
    # Duplicate must be rejected
    with pytest.raises(ValueError):
        pop.record_snapshot()


def test_import_then_run_starts_from_imported_tick() -> None:
    """import → run: engine starts from the imported checkpoint tick."""
    species = _species("lc_import")
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species, name="lc_import_pop",
            stochastic=False, continuous_sampling=False,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": [1.0, 2.0, 3.0]},
            "male": {"WT|WT": [4.0, 5.0, 6.0]},
        })
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9],
            male_age_based_survival=[1.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=200,
            expected_num_new_adult_females=5,
        )
        .record_history(mode="raw")
        .build()
    )
    pop.run(4)
    exported = pop.export_state()
    assert len(pop.history) == 5  # ticks 0-4

    pop.import_state(exported)
    assert len(pop.history) == 0, "import_state must clear history"
    assert pop.tick == 4, f"Expected tick 4 after import, got {pop.tick}"

    pop.run(2)
    assert pop.tick == 6, f"Expected tick 6 after import+run(2), got {pop.tick}"
    assert len(pop.history) == 3, (
        f"Expected 3 records (4,5,6) after import+run, got {len(pop.history)}"
    )


def test_clear_then_record_starts_fresh() -> None:
    """clear → record: after clearing history, a new snapshot can be recorded."""
    species = _species("lc_clear")
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species, name="lc_clear_pop",
            stochastic=False, continuous_sampling=False,
        )
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": [1.0, 2.0, 3.0]},
            "male": {"WT|WT": [4.0, 5.0, 6.0]},
        })
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9],
            male_age_based_survival=[1.0, 0.9],
        )
        .competition(
            juvenile_growth_mode="concave",
            old_juvenile_carrying_capacity=200,
            expected_num_new_adult_females=5,
        )
        .record_history(mode="raw")
        .build()
    )
    pop.run(3)
    assert len(pop.history) == 4  # ticks 0-3
    pop.clear_history()
    assert len(pop.history) == 0, "clear_history must empty history"
    pop.record_snapshot()
    assert len(pop.history) == 1, "record_snapshot must work after clear"
    assert pop.history.ticks == (3,), f"Expected tick (3,), got {pop.history.ticks}"
