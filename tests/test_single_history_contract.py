"""Contracts requiring one typed History for every non-spatial runtime path."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Literal, TypeAlias

import numpy as np
import pytest

import natal as nt
import natal.engine.lifecycle as lifecycle_engine
from natal.hooks import Op, hook
from natal.numba.utils import numba_disabled, numba_enabled
from natal.output import History, HistorySchema, PopulationLayout
from natal.output.history import HistoryBatch
from natal.patterns import IndividualSelector

Model: TypeAlias = Literal["age", "discrete", "wright_fisher"]
HistoryMode: TypeAlias = Literal["raw", "observation"]
ExecutionPath: TypeAlias = Literal["kernel", "python"]
ContinuationModel: TypeAlias = Literal[
    "age",
    "discrete",
    "wright_fisher",
    "spatial",
]
NonSpatialPopulation: TypeAlias = (
    nt.AgeStructuredPopulation | nt.DiscreteGenerationPopulation
)


@hook(event="first")
def _noop_history_hook() -> list[Op]:
    """Force Python dispatch when Numba is disabled without changing state.

    Returns:
        Empty operation list.
    """
    return []


@hook(event="first")
def _stop_on_initial_population() -> list[Op]:
    """Stop before lifecycle mutation whenever the initial population is nonzero.

    Returns:
        Single stop operation with a zero threshold.
    """
    return [Op.stop_if_above(threshold=0.0)]


def _species(name: str) -> nt.Species:
    """Build the biallelic species shared by deterministic fixtures.

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


def _observation_groups() -> dict[str, IndividualSelector]:
    """Return two exhaustive groups whose result has no age axis.

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


def _build_population(
    model: Model,
    mode: HistoryMode,
    name: str,
    *,
    install_noop_hook: bool = False,
) -> NonSpatialPopulation:
    """Build one deterministic population for the requested engine path.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
        mode: ``"raw"`` or ``"observation"``.
        name: Base identifier for species and population.
        install_noop_hook: Whether to install a no-op hook for Python dispatch.

    Returns:
        A built non-spatial population with history recording.
    """
    species = _species(f"{name}_species")
    if model == "age":
        configurator = (
            nt.AgeStructuredPopulation.setup(
                species=species,
                name=name,
                stochastic=False,
                continuous_sampling=False,
            )
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {
                        "WT|WT": [1.0, 2.0, 3.0, 4.0],
                        "WT|Dr": [5.0, 6.0, 7.0, 8.0],
                    },
                    "male": {
                        "WT|WT": [9.0, 10.0, 11.0, 12.0],
                        "Dr|Dr": [13.0, 14.0, 15.0, 16.0],
                    },
                },
                sperm_storage={
                    "WT|WT": {"WT|WT": {1: 17.0, 2: 18.0}},
                    "WT|Dr": {"Dr|Dr": {2: 19.0, 3: 20.0}},
                },
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.8],
                male_age_based_survival=[1.0, 0.9, 0.8],
            )
            .reproduction(
                eggs_per_female=2.0,
                female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
                male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            )
            .competition(
                juvenile_growth_mode="concave",
                old_juvenile_carrying_capacity=1000.0,
                expected_num_new_adult_females=10.0,
            )
        )
    else:
        configurator = (
            nt.DiscreteGenerationPopulation.setup(
                species=species,
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
            .reproduction(eggs_per_female=2.0)
            .competition(
                juvenile_growth_mode="concave",
                low_density_growth_rate=2.0,
                carrying_capacity=1000.0,
            )
        )

    if mode == "observation":
        configurator.with_observation(
            groups=_observation_groups(),
            collapse_age=True,
        )
    if install_noop_hook:
        configurator.hooks(_noop_history_hook)
    population = configurator.record_history(mode=mode).build()
    if model == "wright_fisher":
        object.__setattr__(
            population,
            "_config",
            population.config._replace(extreme_speed_mode=1),
        )
    return population


def _execution_context(path: ExecutionPath) -> AbstractContextManager[None]:
    """Return the explicit context selecting one execution implementation.

    Args:
        path: ``"kernel"`` for Numba JIT or ``"python"`` for Python dispatch.

    Returns:
        A context manager that enables or disables Numba.
    """
    return numba_enabled() if path == "kernel" else numba_disabled()


def _run_population(
    model: Model,
    mode: HistoryMode,
    path: ExecutionPath,
) -> NonSpatialPopulation:
    """Build and run two recorded ticks through one execution path.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
        mode: ``"raw"`` or ``"observation"``.
        path: ``"kernel"`` or ``"python"``.

    Returns:
        A population that has advanced two ticks under the chosen path.
    """
    with _execution_context(path):
        population = _build_population(
            model,
            mode,
            f"single_{model}_{mode}_{path}",
            install_noop_hook=True,
        )
        population.run(2, record_every=1)
    return population


def _spatial_population(name: str) -> nt.SpatialPopulation:
    """Build a deterministic two-deme raw spatial population.

    Args:
        name: Base identifier for species and population.

    Returns:
        A built two-deme spatial population with raw history recording.
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
            individual_count={
                "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
            }
        )
        .reproduction(eggs_per_female=2.0)
        .competition(carrying_capacity=1000.0)
        .record_history(mode="raw")
        .build()
    )


@pytest.mark.parametrize("model", ["age", "discrete", "wright_fisher"])
@pytest.mark.parametrize("mode", ["raw", "observation"])
def test_population_owns_only_public_typed_history(
    model: Model,
    mode: HistoryMode,
) -> None:
    """Build and run never create a legacy second history container.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
        mode: ``"raw"`` or ``"observation"``.
    """
    with numba_disabled():
        population = _build_population(model, mode, f"only_history_{model}_{mode}")
        legacy_present_after_build = "_history" in vars(population)
        population.run(1, record_every=1)

    assert isinstance(population.history, History)
    assert legacy_present_after_build is False
    assert "_history" not in vars(population)
    assert population.history.ticks == (0, 1)


@pytest.mark.parametrize("model", ["age", "discrete", "wright_fisher"])
@pytest.mark.parametrize("mode", ["raw", "observation"])
def test_kernel_and_python_history_are_numerically_identical(
    model: Model,
    mode: HistoryMode,
) -> None:
    """Both execution implementations commit identical typed snapshots.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
        mode: ``"raw"`` or ``"observation"``.
    """
    kernel = _run_population(model, mode, "kernel")
    python = _run_population(model, mode, "python")

    assert kernel.history.ticks == python.history.ticks == (0, 1, 2)
    if mode == "observation":
        assert kernel.history.values.shape == python.history.values.shape == (3, 2, 2)
        np.testing.assert_array_equal(kernel.history.values, python.history.values)
        return

    np.testing.assert_array_equal(
        kernel.history.individual_count,
        python.history.individual_count,
    )
    if model == "age":
        kernel_sperm = kernel.history.sperm_storage
        python_sperm = python.history.sperm_storage
        assert kernel_sperm is not None
        assert python_sperm is not None
        np.testing.assert_array_equal(kernel_sperm, python_sperm)
    else:
        assert kernel.history.sperm_storage is None
        assert python.history.sperm_storage is None


@pytest.mark.parametrize("path", ["kernel", "python"])
@pytest.mark.parametrize("mode", ["raw", "observation"])
def test_continuation_filters_boundary_and_clear_starts_new_timeline(
    path: ExecutionPath,
    mode: HistoryMode,
) -> None:
    """Continuation is strictly increasing; clear retains only the new run.

    Args:
        path: ``"kernel"`` or ``"python"``.
        mode: ``"raw"`` or ``"observation"``.
    """
    with _execution_context(path):
        population = _build_population("discrete", mode, f"continue_{path}_{mode}")
        population.run(2, record_every=1)
        population.run(2, record_every=1, clear_history_on_start=False)
        assert population.history.ticks == (0, 1, 2, 3, 4)
        assert all(
            right > left
            for left, right in zip(
                population.history.ticks,
                population.history.ticks[1:],
            )
        )
        population.run(2, record_every=1, clear_history_on_start=True)

    assert population.history.ticks == (4, 5, 6)


def _build_python_continuation_population(
    model: ContinuationModel,
) -> NonSpatialPopulation | nt.SpatialPopulation:
    """Build a raw population whose continuation begins in Python dispatch.

    Args:
        model: ``"age"``, ``"discrete"``, ``"wright_fisher"``, or ``"spatial"``.

    Returns:
        A built population whose next run will use Python dispatch.
    """
    if model == "spatial":
        return _spatial_population("continuation_payload_spatial")
    return _build_population(
        model,
        "raw",
        f"continuation_payload_{model}",
        install_noop_hook=True,
    )


def _continuation_mutable_count(
    population: NonSpatialPopulation | nt.SpatialPopulation,
    model: ContinuationModel,
) -> np.ndarray:
    """Return the mutable count array represented by the current History tail.

    Args:
        population: A built population.
        model: ``"age"``, ``"discrete"``, ``"wright_fisher"``, or ``"spatial"``.

    Returns:
        The mutable individual-count array for the first deme or population.
    """
    if model == "spatial":
        assert isinstance(population, nt.SpatialPopulation)
        return population.demes[0].state.individual_count
    assert not isinstance(population, nt.SpatialPopulation)
    return population.state.individual_count


@pytest.mark.parametrize(
    "model",
    ["age", "discrete", "wright_fisher", "spatial"],
)
def test_python_continuation_rejects_same_tick_changed_payload_atomically(
    model: ContinuationModel,
) -> None:
    """A changed continuation boundary raises before any lifecycle stage advances.

    Args:
        model: ``"age"``, ``"discrete"``, ``"wright_fisher"``, or ``"spatial"``.
    """
    with numba_disabled():
        population = _build_python_continuation_population(model)
        population.run(1, record_every=1)
        original_history = population.history._to_numpy().copy()
        boundary_tick = population.tick
        current_count = _continuation_mutable_count(population, model)
        current_count[0, 0, 0] += 0.5
        changed_state = current_count.copy()

        with pytest.raises(ValueError, match="boundary payload does not match"):
            population.run(
                1,
                record_every=1,
                clear_history_on_start=False,
            )

    assert population.tick == boundary_tick == 1
    np.testing.assert_array_equal(population.history._to_numpy(), original_history)
    np.testing.assert_array_equal(
        _continuation_mutable_count(population, model),
        changed_state,
    )
    assert population.history.ticks == (0, 1)


@pytest.mark.parametrize(
    "model",
    ["age", "discrete", "wright_fisher", "spatial"],
)
def test_python_continuation_accepts_unchanged_boundary(
    model: ContinuationModel,
) -> None:
    """An equal continuation boundary is filtered once and simulation advances.

    Args:
        model: ``"age"``, ``"discrete"``, ``"wright_fisher"``, or ``"spatial"``.
    """
    with numba_disabled():
        population = _build_python_continuation_population(model)
        population.run(1, record_every=1)
        population.run(1, record_every=1, clear_history_on_start=False)

    assert population.tick == 2
    assert population.history.ticks == (0, 1, 2)


@pytest.mark.parametrize("model", ["age", "discrete", "wright_fisher"])
@pytest.mark.parametrize("mode", ["raw", "observation"])
def test_get_history_adapter_exports_only_public_history(
    model: Model,
    mode: HistoryMode,
) -> None:
    """Compatibility adapter is an exact export of the sole typed container.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
        mode: ``"raw"`` or ``"observation"``.
    """
    with numba_disabled():
        population = _build_population(model, mode, f"adapter_{model}_{mode}")
        population.run(1, record_every=1)

    assert "_history" not in vars(population)
    h_arr = population.history._to_numpy()
    assert h_arr.ndim == 2, f"Expected 2-D history array, got shape {h_arr.shape}"
    assert h_arr.shape[0] == 2, f"Expected 2 records (ticks 0 and 1), got {h_arr.shape[0]}"
    assert h_arr[0, 0] == 0.0  # first tick is 0
    assert h_arr[1, 0] == 1.0  # second tick is 1


@pytest.mark.parametrize("model", ["age", "discrete", "wright_fisher"])
def test_clear_history_preserves_and_empties_the_single_container(model: Model) -> None:
    """clear_history mutates the existing typed History and no other storage.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
    """
    with numba_disabled():
        population = _build_population(model, "raw", f"clear_{model}")
        population.run(1, record_every=1)
    history = population.history
    assert history.ticks == (0, 1)

    population.clear_history()

    assert population.history is history
    assert history.is_empty
    assert history.ticks == ()
    assert "_history" not in vars(population)


def _manual_history() -> tuple[History, HistorySchema]:
    """Return an empty raw History with a five-value row schema.

    Returns:
        A ``(History, HistorySchema)`` pair for a 1-deme discrete layout.
    """
    layout = PopulationLayout(
        kind="discrete_generation",
        n_demes=1,
        n_sexes=2,
        n_ages=2,
        n_ztypes=1,
        has_sperm_storage=False,
        sex_labels=("female", "male"),
        ztype_labels=("WT|WT@default",),
    )
    schema = HistorySchema(
        mode="raw",
        population=layout,
        row_size=5,
    )
    return History(schema), schema


def _history_rows(*ticks: int) -> np.ndarray:
    """Return coordinate-distinct rows for the supplied ticks.

    Args:
        *ticks: Tick values to embed in row order.

    Returns:
        A ``(len(ticks), 5)`` array with unique per-row payloads.
    """
    rows = np.empty((len(ticks), 5), dtype=np.float64)
    for index, tick in enumerate(ticks):
        rows[index] = np.array(
            [float(tick), 10.0 * index + 1.0, 10.0 * index + 2.0,
             10.0 * index + 3.0, 10.0 * index + 4.0],
            dtype=np.float64,
        )
    return rows


def test_history_append_rejects_duplicate_batch_atomically() -> None:
    """A batch containing duplicate ticks raises without committing any row."""
    history, schema = _manual_history()
    history._append(HistoryBatch(schema=schema, rows=_history_rows(0, 1)))

    with pytest.raises(ValueError, match="strictly increasing|unique"):
        history._append(HistoryBatch(schema=schema, rows=_history_rows(2, 2)))

    assert history.ticks == (0, 1)
    assert history.n_records == 2


def test_history_append_rejects_out_of_order_batch_atomically() -> None:
    """A descending batch raises without partially appending its first row."""
    history, schema = _manual_history()
    history._append(HistoryBatch(schema=schema, rows=_history_rows(0, 1)))

    with pytest.raises(ValueError, match="strictly increasing|unique"):
        history._append(HistoryBatch(schema=schema, rows=_history_rows(3, 2)))

    assert history.ticks == (0, 1)
    assert history.n_records == 2


def test_history_rejects_evicted_older_tick_by_current_order() -> None:
    """Eviction does not permit appending a forgotten tick before the tail."""
    history, schema = _manual_history()
    history.max_rows = 2
    history._append(HistoryBatch(schema=schema, rows=_history_rows(0, 1, 2)))
    assert history.ticks == (1, 2)

    with pytest.raises(ValueError, match="strictly increasing"):
        history._append(HistoryBatch(schema=schema, rows=_history_rows(0)))

    assert history.ticks == (1, 2)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_manual_snapshot_rejects_current_tick_without_duplicate_write(
    model: Model,
) -> None:
    """A strict manual snapshot rejects an already-recorded current tick.

    Args:
        model: ``"age"`` or ``"discrete"``.
    """
    population = _build_population(model, "raw", f"snapshot_duplicate_{model}")
    population.record_snapshot()
    original = population.history._to_numpy().copy()

    with pytest.raises(ValueError, match="already contains tick 0"):
        population.record_snapshot()

    assert population.history.ticks == (0,)
    np.testing.assert_array_equal(population.history._to_numpy(), original)


def test_snapshot_and_kernel_transport_require_initialized_history() -> None:
    """Recording boundaries reject a population whose typed History is absent."""
    population = _build_population("discrete", "raw", "missing_typed_history")
    schema = population.history.schema
    row = np.zeros((1, schema.row_size), dtype=np.float64)
    population._history_obj = None  # type: ignore[reportPrivateUsage]  # exercise the invalid pre-build boundary

    with pytest.raises(RuntimeError, match="History is not initialized"):
        population.record_snapshot()
    with pytest.raises(RuntimeError, match="History is not initialized"):
        population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify engine transport boundary
            row,
            clear_history_on_start=False,
        )


def test_snapshot_rejects_missing_state_or_observation() -> None:
    """Snapshot requires both the runtime state and observation projection."""
    missing_state = _build_population("discrete", "raw", "missing_state")
    missing_state._state = None  # type: ignore[reportPrivateUsage]  # exercise invalid construction boundary
    with pytest.raises(RuntimeError, match="state is not initialized"):
        missing_state.record_snapshot()

    missing_observation = _build_population(
        "discrete",
        "observation",
        "missing_observation",
    )
    missing_observation._observation = None  # type: ignore[reportPrivateUsage]  # exercise invalid projection boundary
    with pytest.raises(RuntimeError, match="Observation is not initialized"):
        missing_observation.record_snapshot()


def test_kernel_transport_rejects_rows_before_current_history_tail() -> None:
    """Engine transport cannot append a timeline older than the committed tail."""
    with numba_enabled():
        population = _build_population("discrete", "raw", "old_kernel_rows")
        population.run(1, record_every=1)
    old_row = population.history._to_numpy()[0:1].copy()

    with pytest.raises(ValueError, match="starts before"):
        population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify stale engine batch rejection
            old_row,
            clear_history_on_start=False,
        )

    assert population.history.ticks == (0, 1)


def test_spatial_manual_duplicate_and_continuation_boundary_are_distinct() -> None:
    """Manual duplicate raises while automatic spatial continuation filters overlap."""
    with numba_disabled():
        manual = _spatial_population("spatial_manual_duplicate")
        manual.record_snapshot()
        original = manual.history._to_numpy().copy()
        with pytest.raises(ValueError, match="already contains tick 0"):
            manual.record_snapshot()
        np.testing.assert_array_equal(manual.history._to_numpy(), original)

    with numba_enabled():
        continued = _spatial_population("spatial_continuation")
        continued.run(1, record_every=1)
        continued.run(1, record_every=1, clear_history_on_start=False)

    assert continued.history.ticks == (0, 1, 2)


def test_spatial_automatic_boundary_reuse_is_exact_noop() -> None:
    """Automatic spatial recording reuses an equal current boundary once."""
    population = _spatial_population("spatial_automatic_duplicate")
    population.record_snapshot()
    original = population.history._to_numpy().copy()

    population._record_snapshot(  # type: ignore[reportPrivateUsage]  # exercise automatic boundary reuse
        allow_existing=True
    )

    np.testing.assert_array_equal(population.history._to_numpy(), original)
    assert population.history.ticks == (0,)


def test_spatial_kernel_transport_rejects_rows_before_current_tail() -> None:
    """Spatial engine batches older than the committed tail fail atomically."""
    with numba_disabled():
        population = _spatial_population("spatial_old_kernel_rows")
        population.run(1, record_every=1)
    original = population.history._to_numpy().copy()

    with pytest.raises(ValueError, match="starts before"):
        population._process_kernel_history(  # type: ignore[reportPrivateUsage]  # verify stale spatial engine batch rejection
            original[0:1],
            clear_history_on_start=False,
        )

    np.testing.assert_array_equal(population.history._to_numpy(), original)


def test_output_mixin_clear_delegates_to_existing_typed_history() -> None:
    """The shared clear implementation empties the same History instance."""
    population = _build_population("discrete", "raw", "mixin_clear")
    population.record_snapshot()
    history = population.history

    super(nt.DiscreteGenerationPopulation, population).clear_history()

    assert population.history is history
    assert history.is_empty
    assert "_history" not in vars(population)


def test_nonspatial_age_restore_uses_only_typed_history() -> None:
    """Age restore resets exact counts, sperm, tick, and truncates typed rows."""
    with numba_enabled():
        population = _build_population("age", "raw", "age_typed_restore")
        population.run(2, record_every=1)
    expected_count = population.history.individual_count[1].copy()
    sperm = population.history.sperm_storage
    assert sperm is not None
    expected_sperm = sperm[1].copy()
    population.state.individual_count.fill(901.0)
    population.state.sperm_storage.fill(902.0)

    population.restore_checkpoint(1)

    assert population.tick == 1
    assert population.history.ticks == (0, 1)
    np.testing.assert_array_equal(population.state.individual_count, expected_count)
    np.testing.assert_array_equal(population.state.sperm_storage, expected_sperm)


@pytest.mark.parametrize("mode", ["raw", "observation"])
def test_clone_inherits_recording_policy_with_independent_history(
    mode: HistoryMode,
) -> None:
    """Clone policy is shared, while its mask, state, and History rows are isolated.

    Args:
        mode: ``"raw"`` or ``"observation"``.
    """
    source = _build_population("discrete", mode, f"clone_source_{mode}")
    source.record_snapshot()
    source_rows = source.history._to_numpy().copy()
    source_state = source.state.individual_count.copy()

    clone = source._clone(f"clone_target_{mode}")  # type: ignore[reportPrivateUsage]  # clone contract is intentionally internal

    assert clone.observation is source.observation
    assert clone._recording_plan is source._recording_plan  # type: ignore[reportPrivateUsage]  # immutable plan must be shared
    assert clone.history.schema == source.history.schema
    assert clone.history is not source.history
    assert clone.history.ticks == ()
    if mode == "observation":
        source_mask = source._observation_mask  # type: ignore[reportPrivateUsage]  # verify mutable engine mask isolation
        clone_mask = clone._observation_mask  # type: ignore[reportPrivateUsage]  # verify mutable engine mask isolation
        assert source_mask is not None
        assert clone_mask is not None
        assert clone_mask is not source_mask
        np.testing.assert_array_equal(clone_mask, source_mask)
    else:
        assert source._observation_mask is None  # type: ignore[reportPrivateUsage]  # raw recording has no projection mask
        assert clone._observation_mask is None  # type: ignore[reportPrivateUsage]  # raw recording has no projection mask

    with numba_disabled():
        clone.run(1, record_every=1)

    np.testing.assert_array_equal(source.history._to_numpy(), source_rows)
    np.testing.assert_array_equal(source.state.individual_count, source_state)
    assert clone.history.ticks == (0, 1)
    if mode == "raw":
        np.testing.assert_array_equal(clone.history.individual_count[0], source_state)
    else:
        expected = source.observation.apply(source_state)
        np.testing.assert_array_equal(clone.history.values[0], expected)


def test_clone_preserves_uninitialized_history_boundary() -> None:
    """An internal pre-schema clone preserves the absence of History."""
    source = _build_population("discrete", "raw", "clone_without_history")
    source._history_obj = None  # type: ignore[reportPrivateUsage]  # exercise the construction-only clone branch

    clone = source._clone("clone_without_history_target")  # type: ignore[reportPrivateUsage]  # clone contract is intentionally internal

    assert clone._history_obj is None  # type: ignore[reportPrivateUsage]  # no schema means no History can be fabricated


@pytest.mark.parametrize("model", ["age", "discrete", "wright_fisher"])
def test_compiled_stop_at_tick_zero_keeps_only_initial_boundary(
    model: Model,
) -> None:
    """A stopped lifecycle must not record a second post-tick boundary at tick zero.

    Args:
        model: ``"age"``, ``"discrete"``, or ``"wright_fisher"``.
    """
    with numba_enabled():
        population = _build_population(model, "raw", f"compiled_stop_{model}")
        population.set_hook("first", _stop_on_initial_population)
        wrappers = population.get_compiled_event_hooks()
        if model == "age":
            assert wrappers.run_fn is not None
        elif model == "discrete":
            assert wrappers.run_discrete_fn is not None
        else:
            assert wrappers.run_wf_fn is not None
        initial_count = population.state.individual_count.copy()
        initial_sperm = (
            population.state.sperm_storage.copy()
            if model == "age"
            else None
        )
        population.run(3, record_every=1)

    assert population.tick == 0
    assert population.history.ticks == (0,)
    np.testing.assert_array_equal(
        population.history.individual_count,
        initial_count[np.newaxis, ...],
    )
    if model == "age":
        history_sperm = population.history.sperm_storage
        assert history_sperm is not None
        assert initial_sperm is not None
        np.testing.assert_array_equal(
            history_sperm,
            initial_sperm[np.newaxis, ...],
        )
    else:
        assert population.history.sperm_storage is None


def test_continuation_same_tick_different_payload_is_atomic_error() -> None:
    """Continuation overlap requires equality of both tick and complete payload."""
    history, schema = _manual_history()
    history._append(HistoryBatch(schema=schema, rows=_history_rows(0, 1)))
    original = history._to_numpy().copy()
    conflicting = original[-1:].copy()
    conflicting[0, 3] += 0.5

    with pytest.raises(ValueError, match="boundary payload does not match"):
        history._append_continuation(  # type: ignore[reportPrivateUsage]  # exercise the engine-only overlap contract
            HistoryBatch(schema=schema, rows=conflicting)
        )

    np.testing.assert_array_equal(history._to_numpy(), original)
    assert history.ticks == (0, 1)


def test_continuation_rejects_schema_mismatch_and_accepts_empty_batch() -> None:
    """Continuation transport validates schema before treating empty rows as no-op."""
    history, schema = _manual_history()
    original = _history_rows(0, 1)
    history._append(HistoryBatch(schema=schema, rows=original))
    mismatched_schema = replace(schema, row_size=6)

    with pytest.raises(ValueError, match="schema does not match"):
        history._append_continuation(  # type: ignore[reportPrivateUsage]  # exercise the engine-only schema boundary
            HistoryBatch(
                schema=mismatched_schema,
                rows=np.zeros((0, 6), dtype=np.float64),
            )
        )

    history._append_continuation(  # type: ignore[reportPrivateUsage]  # empty same-schema engine batch is an exact no-op
        HistoryBatch(
            schema=schema,
            rows=np.zeros((0, schema.row_size), dtype=np.float64),
        )
    )
    np.testing.assert_array_equal(history._to_numpy(), original)
    assert history.ticks == (0, 1)


def test_spatial_python_dispatch_suppresses_only_deme_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deme run_tick executes with temporary recording suppression before migration.

    Args:
        monkeypatch: Pytest monkeypatch fixture for run_tick replacement.
    """
    with numba_disabled():
        population = _spatial_population("spatial_python_single_history")
        run_tick_calls: list[str] = []
        seen_record_every: list[int] = []
        original_run = nt.DiscreteGenerationPopulation._run_python_lifecycle

        def tracked_run(
            deme: nt.DiscreteGenerationPopulation,
            tick_fn: object,
            n_steps: int,
            record_every: int,
            finish: bool,
            clear_history_on_start: bool,
        ) -> nt.DiscreteGenerationPopulation:
            """Record the delegated deme name and disabled deme recording.

            Args:
                deme: The deme whose unified lifecycle is being tracked.
                tick_fn: The unified single-tick function.
                n_steps: Number of ticks (always one for spatial delegation).
                record_every: Per-deme recording interval (must be zero).
                finish: Whether the delegated run finishes the deme.
                clear_history_on_start: Whether the delegated run clears
                    deme history first.

            Returns:
                The same deme object after one tick.
            """
            _ = tick_fn, n_steps, finish, clear_history_on_start
            run_tick_calls.append(deme.name)
            seen_record_every.append(record_every)
            return original_run(
                deme,
                tick_fn=lifecycle_engine.run_discrete_tick,
                n_steps=n_steps,
                record_every=record_every,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )

        monkeypatch.setattr(
            nt.DiscreteGenerationPopulation,
            "_run_python_lifecycle",
            tracked_run,
        )
        original_intervals = (3, 5)
        for deme, interval in zip(population.demes, original_intervals):
            deme.record_every = interval
        initial_per_deme = np.stack(
            [deme.state.individual_count.copy() for deme in population.demes]
        )
        population.run(1, record_every=1)

    assert tuple(run_tick_calls) == tuple(deme.name for deme in population.demes)
    assert seen_record_every == [0, 0]
    assert tuple(deme.record_every for deme in population.demes) == original_intervals
    assert tuple(deme.history.ticks for deme in population.demes) == ((), ())
    assert population.history.ticks == (0, 1)
    np.testing.assert_array_equal(
        population.history.individual_count[0],
        initial_per_deme,
    )
    final_per_deme = np.stack(
        [deme.state.individual_count for deme in population.demes]
    )
    np.testing.assert_array_equal(
        population.history.individual_count[1],
        final_per_deme,
    )


def test_spatial_python_dispatch_restores_absent_recording_attribute() -> None:
    """Temporary suppression removes record_every when a deme lacked it initially."""
    with numba_disabled():
        population = _spatial_population("spatial_python_absent_interval")
        first, second = population.demes
        delattr(first, "record_every")
        second.record_every = 7
        population.run(1, record_every=0)

    assert hasattr(first, "record_every") is False
    assert second.record_every == 7
    assert first.history.ticks == ()
    assert second.history.ticks == ()
    assert population.history.ticks == ()
    assert population.tick == first.tick == second.tick == 1


@pytest.mark.parametrize("had_record_every", [True, False])
def test_spatial_recording_suppression_restores_after_deme_exception(
    monkeypatch: pytest.MonkeyPatch,
    had_record_every: bool,
) -> None:
    """Deme recording policy is restored in finally and errors propagate unchanged.

    Args:
        monkeypatch: Pytest monkeypatch fixture for run_tick replacement.
        had_record_every: Whether the first deme initially has a record_every
            attribute (supplied via parametrize).
    """
    with numba_disabled():
        population = _spatial_population(
            f"spatial_suppression_error_{had_record_every}"
        )
        first = population.demes[0]
        original_interval = 9
        if had_record_every:
            first.record_every = original_interval
        else:
            delattr(first, "record_every")
        seen_intervals: list[int] = []
        expected_error = RuntimeError("deme lifecycle sentinel")

        def failing_run(
            deme: nt.DiscreteGenerationPopulation,
            tick_fn: object,
            n_steps: int,
            record_every: int,
            finish: bool,
            clear_history_on_start: bool,
        ) -> nt.DiscreteGenerationPopulation:
            """Record the delegated interval before raising the sentinel.

            Args:
                deme: The deme whose interval is being inspected.
                tick_fn: The unified single-tick function.
                n_steps: Number of ticks (always one for spatial delegation).
                record_every: Per-deme recording interval (must be zero).
                finish: Whether the delegated run finishes the deme.
                clear_history_on_start: Whether the delegated run clears
                    deme history first.

            Returns:
                Never returns; always raises.

            Raises:
                RuntimeError: Always, as a simulation sentinel.
            """
            _ = deme, tick_fn, n_steps, finish, clear_history_on_start
            seen_intervals.append(record_every)
            raise expected_error

        monkeypatch.setattr(
            nt.DiscreteGenerationPopulation,
            "_run_python_lifecycle",
            failing_run,
        )

        with pytest.raises(RuntimeError, match="deme lifecycle sentinel") as captured:
            population.run(1, record_every=1)

    assert captured.value is expected_error
    assert seen_intervals == [0]
    if had_record_every:
        assert first.record_every == original_interval
    else:
        assert hasattr(first, "record_every") is False
    assert population.tick == 0
    assert population.history.ticks == (0,)
    assert tuple(deme.history.ticks for deme in population.demes) == ((), ())


def test_spatial_raw_history_posthoc_observation_projects_every_deme() -> None:
    """Post-hoc observation projects each spatial raw row without mixing demes."""
    with numba_disabled():
        population = _spatial_population("spatial_posthoc_observation")
        population.run(1, record_every=1)
    raw_count = population.history.individual_count

    observed = population.history.observe(population.observation)

    expected = np.stack(
        [
            np.stack(
                [
                    population.observation.apply(raw_count[record, deme])
                    for deme in range(population.n_demes)
                ],
                axis=1,
            )
            for record in range(population.history.n_records)
        ]
    )
    assert observed.ticks == population.history.ticks == (0, 1)
    np.testing.assert_array_equal(observed.values, expected)


def test_age_legacy_state_adapters_round_trip_typed_history() -> None:
    """Age state adapters preserve exact state arrays and sole History rows."""
    with numba_disabled():
        source = _build_population("age", "raw", "age_adapter_source")
        source.run(1, record_every=1)
    state_flat = source.export_state()
    history_rows = source.history._to_numpy()
    assert source._get_kernel_config() is source.config  # type: ignore[reportPrivateUsage]  # simulator adapter must expose the active immutable config
    np.testing.assert_array_equal(state_flat, source.state.flatten_all())
    np.testing.assert_array_equal(history_rows, source.history._to_numpy())

    target = _build_population("age", "raw", "age_adapter_target")
    target.import_state(state_flat)

    assert target.tick == source.tick
    np.testing.assert_array_equal(
        target.state.individual_count,
        source.state.individual_count,
    )
    np.testing.assert_array_equal(
        target.state.sperm_storage,
        source.state.sperm_storage,
    )
    assert "_history" not in vars(target)

    object_target = _build_population("age", "raw", "age_adapter_object_target")
    object_target.import_state(source.state)
    assert object_target.tick == source.tick
    np.testing.assert_array_equal(
        object_target.state.individual_count,
        source.state.individual_count,
    )
    np.testing.assert_array_equal(
        object_target.state.sperm_storage,
        source.state.sperm_storage,
    )


def test_age_reset_clears_typed_history_and_restores_initial_state() -> None:
    """Reset restores the exact initial arrays and empties the sole History."""
    population = _build_population("age", "raw", "age_reset_history")
    initial_count = population.state.individual_count.copy()
    initial_sperm = population.state.sperm_storage.copy()
    with numba_disabled():
        population.run(1, record_every=1)
    assert population.history.ticks == (0, 1)

    population.reset()

    assert population.tick == 0
    assert population.history.ticks == ()
    np.testing.assert_array_equal(population.state.individual_count, initial_count)
    np.testing.assert_array_equal(population.state.sperm_storage, initial_sperm)


def test_age_import_rejects_bad_sperm_shape_without_mutation() -> None:
    """A failed age-state import preserves arrays, tick, and History."""
    population = _build_population("age", "raw", "age_atomic_import")
    population.record_snapshot()
    initial_count = population.state.individual_count.copy()
    initial_sperm = population.state.sperm_storage.copy()
    initial_tick = population.tick
    initial_history = population.history.individual_count
    replacement_count = np.full_like(initial_count, 17.0)

    with pytest.raises(ValueError, match="sperm_storage shape"):
        population.import_state(
            {
                "n_tick": 42,
                "individual_count": replacement_count,
                "sperm_storage": np.zeros((0,), dtype=np.float64),
            }
        )

    np.testing.assert_array_equal(population.state.individual_count, initial_count)
    np.testing.assert_array_equal(population.state.sperm_storage, initial_sperm)
    assert population.tick == initial_tick
    assert population.history.ticks == (initial_tick,)
    np.testing.assert_array_equal(
        population.history.individual_count,
        initial_history,
    )


def test_discrete_import_rejects_bad_count_shape_without_mutation() -> None:
    """A failed discrete-state import preserves state, tick, and History."""
    population = _build_population("discrete", "raw", "discrete_atomic_import")
    population.record_snapshot()
    initial_count = population.state.individual_count.copy()
    initial_tick = population.tick
    initial_history = population.history.individual_count

    with pytest.raises(ValueError, match="individual_count shape"):
        population.import_state(
            {
                "n_tick": 42,
                "individual_count": np.zeros((1,), dtype=np.float64),
            }
        )

    np.testing.assert_array_equal(population.state.individual_count, initial_count)
    assert population.tick == initial_tick
    assert population.history.ticks == (initial_tick,)
    np.testing.assert_array_equal(
        population.history.individual_count,
        initial_history,
    )
