"""RED contracts for regular spatial Observation and History axes."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypeAlias

import numpy as np
import pytest

import natal as nt
import natal.output.record as record_module
from natal.engine.spatial_simulator import run_spatial_steps_with_migration
from natal.numba.utils import numba_disabled, numba_enabled
from natal.patterns import IndividualSelector

DemeMode: TypeAlias = Literal["preserve", "aggregate"]


def _species(name: str) -> nt.Species:
    """Build the biallelic species used by every spatial contract.

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


def _groups() -> dict[str, IndividualSelector]:
    """Return two exhaustive, coordinate-distinct ZType groups.

    Returns:
        Dictionary mapping ``"wild"`` and ``"drive"`` labels to selectors.
    """
    return {
        "wild": IndividualSelector(ztype="WT|WT"),
        "drive": (
            IndividualSelector(ztype="WT|Dr")
            | IndividualSelector(ztype="Dr|Dr")
        ),
    }


def _build_discrete(
    name: str,
    *,
    n_demes: int = 3,
    demes: list[int] | None = None,
    deme_mode: DemeMode = "preserve",
    collapse_age: bool = False,
    history_mode: Literal["raw", "observation"] = "observation",
    identity: bool = False,
) -> nt.SpatialPopulation:
    """Build a deterministic three-deme discrete population.

    Args:
        name: Base identifier for species and population.
        n_demes: Number of demes (default 3).
        demes: Selected deme indices for observation (None means all).
        deme_mode: ``"preserve"`` or ``"aggregate"``.
        collapse_age: Whether to sum over the age axis.
        history_mode: ``"raw"`` or ``"observation"``.
        identity: If True, skip explicit observation groups (use canonical identity).

    Returns:
        A built spatial population with the requested history mode.
    """
    configurator = (
        nt.SpatialPopulation.builder(
            _species(f"{name}_species"),
            n_demes=n_demes,
            topology=None,
            pop_type="discrete_generation",
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
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
    if not identity:
        configurator.with_observation(
            _groups(),
            collapse_age=collapse_age,
            demes=demes,
            deme_mode=deme_mode,
        )
    return configurator.record_history(mode=history_mode).build()


def _build_age(
    name: str,
    *,
    n_demes: int = 3,
    history_mode: Literal["raw", "observation"] = "raw",
) -> nt.SpatialPopulation:
    """Build an age-structured population with spatial sperm storage.

    Args:
        name: Base identifier for species and population.
        n_demes: Number of demes (default 3).
        history_mode: ``"raw"`` or ``"observation"``.

    Returns:
        A built age-structured spatial population.
    """
    return (
        nt.SpatialPopulation.builder(
            _species(f"{name}_species"),
            n_demes=n_demes,
            topology=None,
            pop_type="age_structured",
        )
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [1.0, 2.0, 3.0]},
                "male": {"WT|Dr": [4.0, 5.0, 6.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 1.0],
            male_age_based_survival=[1.0, 1.0],
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0, 0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0, 0.0],
        )
        .competition(
            juvenile_growth_mode=nt.NO_COMPETITION,
        )
        .with_observation(
            _groups(),
            demes=[1] if n_demes > 1 else [0],
            deme_mode="aggregate",
        )
        .record_history(mode=history_mode)
        .build()
    )


def _install_coordinate_counts(population: nt.SpatialPopulation) -> np.ndarray:
    """Install counts whose decimal place identifies every source axis.

    Args:
        population: A built spatial population.

    Returns:
        Stacked ``(n_demes, sex, age, ztype)`` array with coordinate values.
    """
    installed: list[np.ndarray] = []
    for deme_index, deme in enumerate(population.demes):
        shape = deme.state.individual_count.shape
        coordinates = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        counts = coordinates + 1.0 + 1000.0 * deme_index
        deme.state.individual_count[:] = counts
        installed.append(counts)
    return np.stack(installed)


def _install_valid_coordinate_sperm(
    population: nt.SpatialPopulation,
    counts: np.ndarray,
) -> np.ndarray:
    """Install exact binary sperm fractions bounded by female counts.

    Args:
        population: A built spatial population.
        counts: The per-deme count array used to derive sperm values.

    Returns:
        Stacked ``(n_demes, n_ages, n_ztypes, n_ztypes)`` sperm array.
    """
    installed: list[np.ndarray] = []
    for deme_index, deme in enumerate(population.demes):
        sperm = np.zeros_like(deme.state.sperm_storage)
        female_counts = counts[deme_index, 0]
        sperm[:, :, 0] = female_counts / 4.0
        sperm[:, :, 1] = female_counts / 8.0
        deme.state.sperm_storage[:] = sperm
        installed.append(sperm)
    return np.stack(installed)


def _advance_age_state(
    counts: np.ndarray,
    sperm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance both raw tensors by one age-only deterministic tick.

    Args:
        counts: Current ``(n_demes, sex, age, ztype)`` count array.
        sperm: Current ``(n_demes, n_ages, n_ztypes, n_ztypes)`` sperm array.

    Returns:
        A ``(next_counts, next_sperm)`` tuple after age shifting.
    """
    next_counts = np.zeros_like(counts)
    next_counts[:, :, 1:, :] = counts[:, :, :-1, :]
    next_sperm = np.zeros_like(sperm)
    next_sperm[:, 1:, :, :] = sperm[:, :-1, :, :]
    return next_counts, next_sperm


def _age_state_sequence(
    initial_counts: np.ndarray,
    initial_sperm: np.ndarray,
    *,
    n_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute expected tick-boundary tensors without lifecycle helpers.

    Args:
        initial_counts: Count array at tick 0.
        initial_sperm: Sperm array at tick 0.
        n_steps: Number of ticks to advance.

    Returns:
        A ``(count_sequence, sperm_sequence)`` tuple, each of length
        ``n_steps + 1`` (including the initial tick).
    """
    count_sequence = [initial_counts.copy()]
    sperm_sequence = [initial_sperm.copy()]
    for _ in range(n_steps):
        next_counts, next_sperm = _advance_age_state(
            count_sequence[-1],
            sperm_sequence[-1],
        )
        count_sequence.append(next_counts)
        sperm_sequence.append(next_sperm)
    return count_sequence, sperm_sequence


def _manual_groups(
    population: nt.SpatialPopulation,
    counts: np.ndarray,
) -> np.ndarray:
    """Project wild versus drive ZTypes without using Observation code.

    Args:
        population: A built spatial population (for genotype labels).
        counts: Raw ``(n_demes, sex, age, ztype)`` count array.

    Returns:
        A ``(2, n_demes, sex, age)`` array: index 0 for wild, 1 for drive.
    """
    genotype_labels = tuple(
        str(genotype)
        for genotype, _slab in population.deme(0).index_registry.index_to_ztype
    )
    wild_indices = [
        index for index, label in enumerate(genotype_labels) if label == "WT|WT"
    ]
    drive_indices = [
        index for index, label in enumerate(genotype_labels) if label != "WT|WT"
    ]
    assert len(wild_indices) == 1
    assert len(drive_indices) == 2
    return np.stack(
        [
            counts[..., wild_indices].sum(axis=-1),
            counts[..., drive_indices].sum(axis=-1),
        ],
        axis=0,
    )


def _manual_spatial_projection(
    population: nt.SpatialPopulation,
    counts: np.ndarray,
    *,
    deme_mode: DemeMode,
    collapse_age: bool,
) -> np.ndarray:
    """Derive the configured projection without calling Observation code.

    Args:
        population: A built spatial population.
        counts: Raw ``(n_demes, sex, age, ztype)`` count array.
        deme_mode: ``"preserve"`` or ``"aggregate"``.
        collapse_age: Whether to sum over the age axis.

    Returns:
        The expected projected values for the configured groups and deme policy.
    """
    expected = _manual_groups(population, counts)[:, [2, 0]]
    if collapse_age:
        expected = expected.sum(axis=-1)
    if deme_mode == "aggregate":
        expected = expected.sum(axis=1)
    return expected


def test_preserve_selects_regular_shared_deme_axis() -> None:
    """Preserve equals direct per-deme projection in requested deme order."""
    population = _build_discrete(
        "phase6_preserve",
        demes=[2, 0],
        deme_mode="preserve",
    )
    counts = _install_coordinate_counts(population)
    expected = _manual_groups(population, counts)[:, [2, 0], :, :]

    direct = population.observation.apply(counts)
    result = population.observe()
    population.record_snapshot()

    np.testing.assert_array_equal(direct, expected)
    assert result.axes == ("group", "deme", "sex", "age")
    assert result.values.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result.values, expected)
    assert population.history.axes == (
        "record",
        "group",
        "deme",
        "sex",
        "age",
    )
    assert population.history.values.shape == (1, 2, 2, 2, 2)
    np.testing.assert_array_equal(population.history.values[0], expected)


def test_aggregate_equals_explicit_sum_of_preserved_demes() -> None:
    """Aggregate removes deme and equals preserve summed along that axis."""
    preserve = _build_discrete(
        "phase6_aggregate_reference",
        demes=[2, 0],
        deme_mode="preserve",
    )
    aggregate = _build_discrete(
        "phase6_aggregate",
        demes=[2, 0],
        deme_mode="aggregate",
    )
    preserve_counts = _install_coordinate_counts(preserve)
    aggregate_counts = _install_coordinate_counts(aggregate)
    np.testing.assert_array_equal(aggregate_counts, preserve_counts)
    expected_preserve = _manual_groups(preserve, preserve_counts)[:, [2, 0]]
    expected_aggregate = expected_preserve.sum(axis=1)

    direct_aggregate = aggregate.observation.apply(aggregate_counts)
    preserve_result = preserve.observe()
    aggregate_result = aggregate.observe()
    aggregate.record_snapshot()

    np.testing.assert_array_equal(direct_aggregate, expected_aggregate)
    np.testing.assert_array_equal(preserve_result.values, expected_preserve)
    assert aggregate_result.axes == ("group", "sex", "age")
    assert aggregate_result.values.shape == (2, 2, 2)
    np.testing.assert_array_equal(aggregate_result.values, expected_aggregate)
    np.testing.assert_array_equal(
        aggregate_result.values,
        preserve_result.values.sum(axis=1),
    )
    assert aggregate.history.axes == ("record", "group", "sex", "age")
    assert aggregate.history.values.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(aggregate.history.values[0], expected_aggregate)


@pytest.mark.parametrize(
    ("deme_mode", "expected_axes", "expected_history_axes"),
    [
        (
            "preserve",
            ("group", "deme", "sex"),
            ("record", "group", "deme", "sex"),
        ),
        (
            "aggregate",
            ("group", "sex"),
            ("record", "group", "sex"),
        ),
    ],
)
def test_collapse_age_removes_age_axis_and_sums_values(
    deme_mode: DemeMode,
    expected_axes: tuple[str, ...],
    expected_history_axes: tuple[str, ...],
) -> None:
    """collapse_age performs a real reduction for either deme policy.

    Args:
        deme_mode: ``"preserve"`` or ``"aggregate"`` (from parametrize).
        expected_axes: Expected observation result axis names.
        expected_history_axes: Expected history axis names.
    """
    population = _build_discrete(
        f"phase6_collapse_{deme_mode}",
        demes=[2, 0],
        deme_mode=deme_mode,
        collapse_age=True,
    )
    counts = _install_coordinate_counts(population)
    expected = _manual_groups(population, counts)[:, [2, 0]].sum(axis=-1)
    if deme_mode == "aggregate":
        expected = expected.sum(axis=1)

    direct = population.observation.apply(counts)
    result = population.observe()
    population.record_snapshot()

    np.testing.assert_array_equal(direct, expected)
    assert result.axes == expected_axes
    assert result.values.shape == expected.shape
    np.testing.assert_array_equal(result.values, expected)
    assert population.history.axes == expected_history_axes
    assert population.history.values.shape == (1, *expected.shape)
    np.testing.assert_array_equal(population.history.values[0], expected)


def test_spatial_identity_preserves_every_deme_coordinate() -> None:
    """Identity maps ZType to group without losing any spatial coordinate."""
    population = _build_discrete(
        "phase6_identity",
        identity=True,
        history_mode="observation",
    )
    counts = _install_coordinate_counts(population)
    expected = np.moveaxis(counts, -1, 0)

    direct = population.observation.apply(counts)
    result = population.observe()
    population.record_snapshot()

    np.testing.assert_array_equal(direct, expected)
    assert result.axes == ("group", "deme", "sex", "age")
    assert result.values.shape == expected.shape
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(population.history.values[0], expected)
    assert float(result.values.sum()) == float(counts.sum())


def test_single_deme_spatial_discrete_raw_history_keeps_deme_axis() -> None:
    """One spatial deme remains distinct from a panmictic raw layout."""
    population = _build_discrete(
        "phase6_single_deme_discrete_raw",
        n_demes=1,
        demes=[0],
        history_mode="raw",
    )
    counts = _install_coordinate_counts(population)

    population.record_snapshot()

    assert population.history.axes == (
        "record",
        "deme",
        "sex",
        "age",
        "ztype",
    )
    assert population.history.individual_count.shape == (1, 1, 2, 2, 3)
    np.testing.assert_array_equal(population.history.individual_count[0], counts)
    restored_tick, restored_counts, restored_sperm = (
        population.history.restore_state(0)
    )
    assert restored_tick == 0
    assert restored_counts.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(restored_counts, counts)
    assert restored_sperm is None


def test_single_deme_spatial_age_raw_history_keeps_sperm_deme_axis() -> None:
    """One spatial age deme keeps leading deme axes for both raw tensors."""
    population = _build_age(
        "phase6_single_deme_age_raw",
        n_demes=1,
    )
    counts = _install_coordinate_counts(population)
    sperm = _install_valid_coordinate_sperm(population, counts)

    population.record_snapshot()

    assert population.history.axes == (
        "record",
        "deme",
        "sex",
        "age",
        "ztype",
    )
    assert population.history.individual_count.shape == (1, 1, 2, 3, 3)
    np.testing.assert_array_equal(population.history.individual_count[0], counts)
    assert population.history.sperm_storage is not None
    assert population.history.sperm_storage.shape == (1, 1, 3, 3, 3)
    np.testing.assert_array_equal(population.history.sperm_storage[0], sperm)
    restored_tick, restored_counts, restored_sperm = (
        population.history.restore_state(0)
    )
    assert restored_tick == 0
    assert restored_counts.shape == (1, 2, 3, 3)
    np.testing.assert_array_equal(restored_counts, counts)
    assert restored_sperm is not None
    assert restored_sperm.shape == (1, 3, 3, 3)
    np.testing.assert_array_equal(restored_sperm, sperm)


def test_single_deme_spatial_observation_preserve_keeps_deme_axis() -> None:
    """Preserve mode never erases the spatial axis when its length is one."""
    population = _build_discrete(
        "phase6_single_deme_observation",
        n_demes=1,
        demes=[0],
        history_mode="observation",
    )
    counts = _install_coordinate_counts(population)
    expected = _manual_groups(population, counts)

    result = population.observe()
    population.record_snapshot()

    assert population.observation.axes == ("group", "deme", "sex", "age")
    assert result.axes == ("group", "deme", "sex", "age")
    assert result.values.shape == (2, 1, 2, 2)
    np.testing.assert_array_equal(result.values, expected)
    assert population.history.axes == (
        "record",
        "group",
        "deme",
        "sex",
        "age",
    )
    assert population.history.values.shape == (1, 2, 1, 2, 2)
    np.testing.assert_array_equal(population.history.values[0], expected)


def test_single_deme_spatial_posthoc_observation_keeps_deme_axis() -> None:
    """Post-hoc projection preserves the one-deme spatial layout exactly."""
    population = _build_discrete(
        "phase6_single_deme_posthoc",
        n_demes=1,
        demes=[0],
        history_mode="raw",
    )
    counts = _install_coordinate_counts(population)
    expected = _manual_groups(population, counts)
    population.record_snapshot()

    observed = population.history.observe(population.observation)

    assert observed.ticks == (0,)
    assert observed.axes == (
        "record",
        "group",
        "deme",
        "sex",
        "age",
    )
    assert observed.values.shape == (1, 2, 1, 2, 2)
    np.testing.assert_array_equal(observed.values[0], expected)


@pytest.mark.parametrize("history_mode", ["raw", "observation"])
def test_spatial_get_history_exports_exact_flat_payload(
    history_mode: Literal["raw", "observation"],
) -> None:
    """The spatial compatibility adapter preserves schema flattening order.

    Args:
        history_mode: ``"raw"`` or ``"observation"`` (from parametrize).
    """
    population = _build_discrete(
        f"phase6_get_history_{history_mode}",
        demes=[2, 0],
        history_mode=history_mode,
    )
    counts = _install_coordinate_counts(population)
    population.record_snapshot()
    payload = (
        counts
        if history_mode == "raw"
        else _manual_groups(population, counts)[:, [2, 0]]
    )
    expected = np.concatenate((np.array([0.0]), payload.ravel()))[np.newaxis, :]

    assert population.history._to_numpy().shape == (1, expected.shape[1])
    np.testing.assert_array_equal(population.history._to_numpy(), expected)


def test_raw_history_ignores_observation_demes_and_preserves_sperm() -> None:
    """Raw mode stores all deme counts and sperm regardless of Observation scope."""
    population = _build_age("phase6_raw_age")
    counts = _install_coordinate_counts(population)
    sperm_rows: list[np.ndarray] = []
    for deme_index, deme in enumerate(population.demes):
        shape = deme.state.sperm_storage.shape
        sperm = (
            np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            + 1.0
            + 10000.0 * deme_index
        )
        deme.state.sperm_storage[:] = sperm
        sperm_rows.append(sperm)
    expected_sperm = np.stack(sperm_rows)

    population.record_snapshot()

    assert population.history.individual_count.shape == (1, *counts.shape)
    np.testing.assert_array_equal(population.history.individual_count[0], counts)
    assert population.history.sperm_storage is not None
    assert population.history.sperm_storage.shape == (1, *expected_sperm.shape)
    np.testing.assert_array_equal(
        population.history.sperm_storage[0],
        expected_sperm,
    )


@pytest.mark.parametrize("deme_mode", ["preserve", "aggregate"])
@pytest.mark.parametrize("collapse_age", [False, True])
def test_numba_and_python_spatial_observation_history_are_identical(
    deme_mode: DemeMode,
    collapse_age: bool,
) -> None:
    """Both spatial backends record identical ticks and regular payloads.

    Args:
        deme_mode: ``"preserve"`` or ``"aggregate"`` (from parametrize).
        collapse_age: Whether to sum over the age axis (from parametrize).
    """
    with numba_enabled():
        kernel = _build_discrete(
            f"phase6_kernel_{deme_mode}_{collapse_age}",
            demes=[2, 0],
            deme_mode=deme_mode,
            collapse_age=collapse_age,
        )
    python = _build_discrete(
        f"phase6_python_{deme_mode}_{collapse_age}",
        demes=[2, 0],
        deme_mode=deme_mode,
        collapse_age=collapse_age,
    )
    kernel_counts = _install_coordinate_counts(kernel)
    python_counts = _install_coordinate_counts(python)
    np.testing.assert_array_equal(kernel_counts, python_counts)
    expected_initial = _manual_spatial_projection(
        kernel,
        kernel_counts,
        deme_mode=deme_mode,
        collapse_age=collapse_age,
    )

    with numba_enabled():
        kernel.run(2, record_every=1)
    with numba_disabled():
        python.run(2, record_every=1)

    assert kernel.history.ticks == python.history.ticks == (0, 1, 2)
    assert kernel.history.axes == python.history.axes
    assert kernel.history.values.shape == python.history.values.shape
    np.testing.assert_array_equal(kernel.history.values, python.history.values)
    kernel_final = np.stack(
        [deme.state.individual_count for deme in kernel.demes]
    )
    python_final = np.stack(
        [deme.state.individual_count for deme in python.demes]
    )
    np.testing.assert_array_equal(kernel_final, python_final)
    expected_final = _manual_spatial_projection(
        kernel,
        kernel_final,
        deme_mode=deme_mode,
        collapse_age=collapse_age,
    )
    np.testing.assert_array_equal(kernel.history.values[0], expected_initial)
    np.testing.assert_array_equal(kernel.history.values[-1], expected_final)
    np.testing.assert_array_equal(kernel.observe().values, expected_final)
    np.testing.assert_array_equal(python.observe().values, expected_final)


def test_spatial_apply_rebuilds_lazy_mask_for_default_all_demes() -> None:
    """A lazy spatial rule projects every default-selected deme coordinate."""
    population = _build_discrete("phase6_lazy_all_demes")
    counts = _install_coordinate_counts(population)
    lazy_observation = replace(
        population.observation,
        mask=None,
    )
    expected = _manual_groups(population, counts)

    projected = lazy_observation.apply(counts)

    assert lazy_observation.deme_indices == (0, 1, 2)
    assert lazy_observation.axes == ("group", "deme", "sex", "age")
    assert projected.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(projected, expected)


def test_spatial_observation_metadata_describes_regular_axes() -> None:
    """Serialized metadata preserves one shared ordered deme selection."""
    population = _build_discrete(
        "phase6_metadata",
        demes=[2, 0],
        deme_mode="aggregate",
        collapse_age=True,
    )

    assert population.observation.to_dict() == {
        "labels": ["wild", "drive"],
        "collapse_age": True,
        "n_groups": 2,
        "demes": [2, 0],
        "deme_mode": "aggregate",
    }


@pytest.mark.parametrize("deme_indices", [(), (-1,), (3,)])
def test_spatial_apply_rejects_invalid_deme_selection(
    deme_indices: tuple[int, ...],
) -> None:
    """Direct spatial projection rejects empty and out-of-layout selections.

    Args:
        deme_indices: Invalid deme selection tuple (from parametrize).
    """
    population = _build_discrete("phase6_apply_invalid_demes")
    counts = _install_coordinate_counts(population)
    observation = replace(population.observation, deme_indices=deme_indices)
    message = (
        "Observation selects no demes"
        if not deme_indices
        else "Observation deme selection is outside the population layout"
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        observation.apply(counts)


def test_spatial_apply_rejects_unsupported_rank() -> None:
    """A tensor with an extra axis cannot be mistaken for spatial counts."""
    population = _build_discrete("phase6_apply_invalid_rank")
    invalid = np.zeros((1, 3, 2, 2, 3), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="^Unsupported individual_count ndim: 5$",
    ):
        population.observation.apply(invalid)


def test_spatial_output_properties_require_initialized_policy() -> None:
    """A partially constructed spatial Population cannot expose output state."""
    population = _build_discrete("phase6_uninitialized_output")
    population._observation = None  # type: ignore[reportPrivateUsage]  # emulate an incomplete build boundary
    population._history_obj = None  # type: ignore[reportPrivateUsage]  # emulate an incomplete build boundary

    with pytest.raises(
        RuntimeError,
        match=r"^Observation is not initialized for this population\.$",
    ):
        _ = population.observation
    with pytest.raises(
        RuntimeError,
        match=r"^History is not initialized for this population\.$",
    ):
        _ = population.history


def test_empty_deme_selection_is_rejected_explicitly() -> None:
    """An Observation that selects no spatial deme is invalid."""
    with pytest.raises(ValueError, match="^Observation selects no demes$"):
        _build_discrete(
            "phase6_empty_demes",
            demes=[],
            deme_mode="preserve",
        )


@pytest.mark.parametrize("removed_mode", ["mask", "expand"])
def test_removed_spatial_deme_modes_are_rejected(removed_mode: str) -> None:
    """Legacy ragged spatial modes cannot enter the canonical Observation.

    Args:
        removed_mode: A removed mode string (from parametrize).
    """
    with pytest.raises(
        ValueError,
        match=(
            "^deme_mode must be 'preserve' or 'aggregate', "
            f"got '{removed_mode}'$"
        ),
    ):
        _build_discrete(
            f"phase6_removed_{removed_mode}",
            demes=[0, 2],
            deme_mode=removed_mode,  # type: ignore[arg-type]  # runtime contract intentionally supplies a removed mode
        )


@pytest.mark.parametrize("invalid_demes", [[True], [1.5]])
def test_configurator_rejects_non_integer_deme_indices(
    invalid_demes: list[object],
) -> None:
    """Boolean and floating-point indices cannot enter a spatial rule.

    Args:
        invalid_demes: A list of non-integer deme values (from parametrize).
    """
    with pytest.raises(
        TypeError,
        match="^demes must contain integer deme indices$",
    ):
        _build_discrete(
            "phase6_non_integer_deme",
            demes=invalid_demes,  # type: ignore[arg-type]  # deliberate runtime validation input
        )


@pytest.mark.parametrize(
    ("invalid_demes", "expected_message"),
    [
        ([-1], r"demes must be within \[0, 3\), got \(-1,\)"),
        ([3], r"demes must be within \[0, 3\), got \(3,\)"),
        ([1, 1], "demes must not contain duplicate indices"),
    ],
)
def test_configurator_rejects_invalid_integer_deme_selections(
    invalid_demes: list[int],
    expected_message: str,
) -> None:
    """Out-of-range and duplicate indices fail before Population creation.

    Args:
        invalid_demes: Invalid deme indices (from parametrize).
        expected_message: Expected error message pattern.
    """
    with pytest.raises(ValueError, match=f"^{expected_message}$"):
        _build_discrete(
            "phase6_invalid_integer_deme",
            demes=invalid_demes,
        )


def test_raw_spatial_engine_transport_serializes_regular_state_rows() -> None:
    """The generic spatial engine records exact raw boundary tensors."""
    population = _build_age("phase6_raw_engine_transport")
    counts = _install_coordinate_counts(population)
    sperm = np.stack(
        [deme.state.sperm_storage.copy() for deme in population.demes]
    )
    expected_initial = np.concatenate(
        (
            np.array([0.0]),
            counts.ravel(),
            sperm.ravel(),
        )
    )

    with numba_disabled():
        final_state, history, was_stopped = run_spatial_steps_with_migration.py_func(  # type: ignore[attr-defined]  # tester must trace the Python transport implementation
            counts,
            sperm,
            population.deme(0).config,
            tick=0,
            n_steps=1,
            adjacency=np.zeros((3, 3), dtype=np.float64),
            migration_mode=0,
            topology_rows=0,
            topology_cols=0,
            topology_wrap=False,
            migration_kernel=np.zeros((1, 1), dtype=np.float64),
            kernel_include_center=False,
            migration_rate=np.zeros(1, dtype=np.float64),
            record_interval=1,
        )

    final_counts, final_sperm, final_tick = final_state
    expected_final = np.concatenate(
        (
            np.array([1.0]),
            final_counts.ravel(),
            final_sperm.ravel(),
        )
    )
    assert history is not None
    assert history.shape == (2, expected_initial.size)
    assert final_tick == 1
    assert was_stopped is False
    np.testing.assert_array_equal(history[0], expected_initial)
    np.testing.assert_array_equal(history[1], expected_final)


def test_numba_and_python_spatial_raw_history_are_identical() -> None:
    """Both backends serialize every raw deme coordinate at each boundary."""
    with numba_enabled():
        kernel = _build_discrete(
            "phase6_kernel_raw",
            demes=[2, 0],
            deme_mode="aggregate",
            history_mode="raw",
        )
    python = _build_discrete(
        "phase6_python_raw",
        demes=[2, 0],
        deme_mode="aggregate",
        history_mode="raw",
    )
    initial = _install_coordinate_counts(kernel)
    python_initial = _install_coordinate_counts(python)
    np.testing.assert_array_equal(python_initial, initial)

    with numba_enabled():
        kernel.run(2, record_every=1)
    with numba_disabled():
        python.run(2, record_every=1)

    kernel_final = np.stack(
        [deme.state.individual_count for deme in kernel.demes]
    )
    python_final = np.stack(
        [deme.state.individual_count for deme in python.demes]
    )
    assert kernel.history.ticks == python.history.ticks == (0, 1, 2)
    assert kernel.history.individual_count.shape == (3, *initial.shape)
    assert python.history.individual_count.shape == (3, *initial.shape)
    np.testing.assert_array_equal(kernel.history._to_numpy(), python.history._to_numpy())
    np.testing.assert_array_equal(kernel.history.individual_count[0], initial)
    np.testing.assert_array_equal(python.history.individual_count[0], initial)
    np.testing.assert_array_equal(kernel_final, python_final)
    np.testing.assert_array_equal(kernel.history.individual_count[-1], kernel_final)
    np.testing.assert_array_equal(python.history.individual_count[-1], python_final)


@pytest.mark.parametrize("history_mode", ["raw", "observation"])
def test_age_spatial_backends_continue_with_sparse_exact_history(
    history_mode: Literal["raw", "observation"],
) -> None:
    """Age backends preserve exact counts, sperm, and sparse boundaries.

    Args:
        history_mode: ``"raw"`` or ``"observation"`` (from parametrize).
    """
    with numba_enabled():
        kernel = _build_age(
            f"phase6_age_kernel_{history_mode}",
            history_mode=history_mode,
        )
    python = _build_age(
        f"phase6_age_python_{history_mode}",
        history_mode=history_mode,
    )
    initial_counts = _install_coordinate_counts(kernel)
    python_counts = _install_coordinate_counts(python)
    initial_sperm = _install_valid_coordinate_sperm(kernel, initial_counts)
    python_sperm = _install_valid_coordinate_sperm(python, python_counts)
    expected_counts, expected_sperm = _age_state_sequence(
        initial_counts,
        initial_sperm,
        n_steps=5,
    )
    expected_ticks = (0, 1, 2, 4)

    np.testing.assert_array_equal(python_counts, initial_counts)
    np.testing.assert_array_equal(python_sperm, initial_sperm)
    np.testing.assert_array_equal(
        initial_sperm.sum(axis=-1),
        initial_counts[:, 0] * 3.0 / 8.0,
    )

    with numba_enabled():
        kernel.run(1, record_every=1)
    with numba_disabled():
        python.run(1, record_every=1)

    assert kernel.tick == python.tick == 1
    for population in (kernel, python):
        boundary_counts = np.stack(
            [deme.state.individual_count for deme in population.demes]
        )
        boundary_sperm = np.stack(
            [deme.state.sperm_storage for deme in population.demes]
        )
        np.testing.assert_array_equal(boundary_counts, expected_counts[1])
        np.testing.assert_array_equal(boundary_sperm, expected_sperm[1])

    with numba_enabled():
        kernel.run(4, record_every=2)
    with numba_disabled():
        python.run(4, record_every=2)

    assert kernel.tick == python.tick == 5
    assert kernel.history.ticks == python.history.ticks == expected_ticks
    for population in (kernel, python):
        final_counts = np.stack(
            [deme.state.individual_count for deme in population.demes]
        )
        final_sperm = np.stack(
            [deme.state.sperm_storage for deme in population.demes]
        )
        np.testing.assert_array_equal(final_counts, expected_counts[5])
        np.testing.assert_array_equal(final_sperm, expected_sperm[5])

    if history_mode == "raw":
        expected_history_counts = np.stack(
            [expected_counts[tick] for tick in expected_ticks]
        )
        expected_history_sperm = np.stack(
            [expected_sperm[tick] for tick in expected_ticks]
        )
        for population in (kernel, python):
            assert population.history.axes == (
                "record",
                "deme",
                "sex",
                "age",
                "ztype",
            )
            np.testing.assert_array_equal(
                population.history.individual_count,
                expected_history_counts,
            )
            assert population.history.sperm_storage is not None
            np.testing.assert_array_equal(
                population.history.sperm_storage,
                expected_history_sperm,
            )
    else:
        expected_history_values = np.stack(
            [
                _manual_groups(kernel, expected_counts[tick])[:, [1]].sum(axis=1)
                for tick in expected_ticks
            ]
        )
        for population in (kernel, python):
            assert population.history.axes == (
                "record",
                "group",
                "sex",
                "age",
            )
            np.testing.assert_array_equal(
                population.history.values,
                expected_history_values,
            )
            with pytest.raises(
                ValueError,
                match="^sperm_storage is only available in raw mode$",
            ):
                _ = population.history.sperm_storage

    np.testing.assert_array_equal(
        kernel.history._to_numpy(),
        python.history._to_numpy(),
    )


def test_spatial_observation_has_no_compact_or_sentinel_representation() -> None:
    """Regular observation arrays replace CompactMeta and sentinel padding."""
    assert hasattr(record_module, "CompactMeta") is False
    population = _build_discrete(
        "phase6_no_compact",
        demes=[2, 0],
        deme_mode="preserve",
    )
    counts = _install_coordinate_counts(population)
    expected = _manual_groups(population, counts)[:, [2, 0]]
    population.record_snapshot()

    assert hasattr(population._recording_plan, "compact_layout") is False  # type: ignore[reportPrivateUsage]  # architectural deletion contract
    assert "_compact_meta" not in vars(population)
    assert "_deme_modes" not in vars(population)
    assert population.history.axes == (
        "record",
        "group",
        "deme",
        "sex",
        "age",
    )
    assert population.history.values.shape == (1, 2, 2, 2, 2)
    np.testing.assert_array_equal(population.history.values[0], expected)
    assert not np.any(population.history.values == -1.0)
