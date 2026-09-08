"""Exercise the ownership and validation boundary of native history storage."""

import sys

import numpy as np
import pytest
from numpy.typing import NDArray

from natal._engine_rs import HistoryStore, ParameterLog, project_observation


def test_scalar_parameter_log_noops_rollback_and_clear() -> None:
    """Legacy scalar entries preserve provenance and transaction positions."""
    log = ParameterLog()
    log.append((0, "capacity", 10.0, 10.0))
    log.append_detail((0, "capacity", 10.0, 10.0), "early", 2)
    assert log.mark() == 0
    log.append((1, "capacity", 10.0, 20.0))
    mark = log.mark()
    log.append_detail((2, "capacity", 20.0, 30.0), "late", 2)
    assert log.details() == [
        (1, "update", 0, "capacity", 10.0, 20.0),
        (2, "late", 2, "capacity", 20.0, 30.0),
    ]
    exported = log.snapshot()
    exported.clear()
    assert log.mark() == 2
    log.rollback(mark)
    assert log.snapshot() == [(1, "capacity", 10.0, 20.0)]
    log.clear()
    assert log.mark() == 0 and log.snapshot() == [] and log.details() == []
    log.append((3, "capacity", 30.0, 40.0))
    assert log.snapshot() == [(3, "capacity", 30.0, 40.0)]


@pytest.mark.parametrize("width,limit", [(0, None), (2, 0)])
def test_native_history_requires_positive_storage_limits(width: int, limit: int | None) -> None:
    """An invalid ring cannot be constructed with a zero row width or capacity."""
    with pytest.raises(ValueError, match="must be positive"):
        HistoryStore(width, (1, 1, 1, 1), True, limit)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[2.0, 20.0, 99.0]]),
        np.array([[2.0, 20.0], [np.nan, 30.0]]),
        np.array([[2.0, 20.0], [np.inf, 30.0]]),
        np.array([[2.0, 20.0], [2.5, 30.0]]),
        np.array([[2.0, 20.0], [2.0, 30.0]]),
        np.array([[2.0, 20.0], [0.0, 30.0]]),
    ],
)
def test_native_append_validates_whole_batch_before_eviction(invalid: NDArray[np.float64]) -> None:
    """A malformed second row cannot append or evict the prior checkpoint."""
    store = HistoryStore(2, (1, 1, 1, 1), True, 1)
    store.append(np.array([[1.0, 10.0]]))
    with pytest.raises(ValueError):
        store.append(invalid)
    np.testing.assert_array_equal(store.query(), [[1.0, 10.0]])
    assert store.boundaries() == [(1, 0, "Ready")]
    store.restore_timeline(1)
    assert store.ticks() == [1]


def test_native_continuation_deduplicates_only_equal_boundary() -> None:
    """Continuation may repeat the prior state but cannot rewrite its payload."""
    store = HistoryStore(2, (1, 1, 1, 1), True)
    store.append(np.array([[1.0, 10.0]]))
    store.append(np.array([[1.0, 10.0], [2.0, 20.0]]), continuation=True)
    with pytest.raises(ValueError, match="payload does not match"):
        store.append(np.array([[2.0, 999.0], [3.0, 30.0]]), continuation=True)
    np.testing.assert_array_equal(store.query(), [[1.0, 10.0], [2.0, 20.0]])
    assert store.boundaries() == [(1, 0, "Ready"), (2, 0, "Ready")]


def test_native_history_exports_own_buffers_and_exact_empty_shapes() -> None:
    """Strided inputs and write-enabled exports cannot alias the native ring."""
    store = HistoryStore(2, (1, 1, 1, 1), True)
    assert store.query().shape == (0, 2)
    source = np.array([[1.0, -1.0, 10.0], [2.0, -2.0, 20.0]])
    store.append(source[:, ::2])
    source.fill(-100)
    queried = store.query(1, 2)
    exact = store.row(2)
    for exported in (queried, exact):
        exported.fill(-200)
        exported.setflags(write=False)
        try:
            exported.setflags(write=True)
        except ValueError:
            assert not exported.flags.writeable
        else:
            exported.fill(-200)
    np.testing.assert_array_equal(store.query(), [[1.0, 10.0], [2.0, 20.0]])
    assert store.query(1, 1).shape == (2, 0)
    for start, end in [(2, 1), (0, 3)]:
        with pytest.raises(ValueError, match="out of bounds"):
            store.query(start, end)
    with pytest.raises(ValueError, match="not found"):
        store.row(3)
    store.clear()
    store.append(np.array([[0.0, 5.0]]))
    assert store.ticks() == [0] and store.boundaries() == [(0, 0, "Ready")]
    np.testing.assert_array_equal(exact, [-200.0, -200.0])


def test_native_retention_and_restore_use_each_bound_log_cursor() -> None:
    """Restoring retained history rewinds each log to its own commit count."""
    store = HistoryStore(2, (1, 1, 1, 1), True)
    main, first, second = ParameterLog(), ParameterLog(), ParameterLog()
    store.bind_log(main)
    store.bind_logs([first, second])
    main.append((0, "capacity", 1.0, 2.0))
    first.append((0, "capacity", 3.0, 4.0))
    first.append((0, "capacity", 4.0, 5.0))
    store.append(np.array([[0.0, 10.0]]))
    second.append((1, "capacity", 5.0, 6.0))
    store.append(np.array([[1.0, 20.0]]))
    main.append((2, "capacity", 2.0, 3.0))
    first.append((2, "capacity", 5.0, 6.0))
    second.append((2, "capacity", 6.0, 7.0))
    store.append(np.array([[2.0, 30.0]]))
    with pytest.raises(ValueError, match="must be >= 1"):
        store.max_rows = 0
    assert store.max_rows is None and len(store) == 3
    store.max_rows = 2
    assert store.ticks() == [1, 2]
    assert store.boundaries() == [(1, 0, "Ready"), (2, 0, "Ready")]
    with pytest.raises(ValueError, match="not found"):
        store.restore_timeline(0)
    assert [log.mark() for log in (main, first, second)] == [2, 3, 2]
    store.restore_timeline(1)
    assert store.ticks() == [1]
    assert [log.mark() for log in (main, first, second)] == [1, 2, 1]
    store.max_rows = None
    store.append(np.array([[2.0, 30.0], [3.0, 40.0]]))
    assert len(store) == 3
    with pytest.raises(ValueError, match="No records"):
        store.truncate(0)
    assert store.ticks() == [1, 2, 3]
    store.truncate(2)
    assert store.ticks() == [1, 2]
    assert store.boundaries() == [(1, 0, "Ready"), (2, 0, "Ready")]


@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize("aggregate", [False, True])
def test_native_projection_axes_and_configure_failure_are_atomic(collapse: bool, aggregate: bool) -> None:
    """Genotype selectors preserve requested deme order through both reductions."""
    dimensions = (2, 1, 2, 2)
    individuals = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    mask = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    # Groups select genotype 0 and 1 respectively; deme 1 precedes deme 0.
    expected = np.array([[[[10.0, 30.0]], [[1.0, 3.0]]], [[[20.0, 40.0]], [[2.0, 4.0]]]])
    if collapse:
        expected = expected.sum(axis=3, keepdims=True)
    if aggregate:
        expected = expected.sum(axis=1, keepdims=True)
    result = project_observation(individuals, mask, dimensions, [1, 0], collapse, aggregate)
    np.testing.assert_array_equal(result, expected.ravel())
    source = HistoryStore(9, dimensions, True)
    source.append(np.concatenate(([0.0], individuals))[None, :])
    target = HistoryStore(1 + expected.size, dimensions, False)
    target.configure_observation(mask, [1, 0], collapse, aggregate)
    with pytest.raises(ValueError, match="dimensions or deme selection"):
        target.configure_observation(mask[:-1], [0], False, False)
    source.observe(target)
    np.testing.assert_array_equal(target.query(), np.concatenate(([0.0], expected.ravel()))[None, :])


@pytest.mark.parametrize("dimensions,selected", [((1, 0, 1, 1), [0]), ((2, 1, 1, 1), [0]), ((1, 1, 1, 1), []), ((1, 1, 1, 1), [1])])
def test_native_projection_rejects_invalid_layout(dimensions: tuple[int, ...], selected: list[int]) -> None:
    """Invalid dimensions or selections must fail before indexing raw arrays."""
    with pytest.raises(ValueError, match="dimensions or deme selection"):
        project_observation(np.ones(1), np.ones(1), dimensions, selected, False, False)


def test_native_observe_requires_independent_raw_source() -> None:
    """Native observation rejects self-locking and reprojection of derived rows."""
    source = HistoryStore(2, (1, 1, 1, 1), True)
    source.append(np.array([[1.0, 10.0]]))
    with pytest.raises(ValueError, match="independent"):
        source.observe(source)
    target = HistoryStore(2, (1, 1, 1, 1), False)
    with pytest.raises(ValueError, match="raw-mode"):
        target.observe(source)
    np.testing.assert_array_equal(source.query(), [[1.0, 10.0]])
    assert len(target) == 0


@pytest.mark.parametrize("dimensions", [(1, 1, 1, 2), (1, sys.maxsize, 3, 1)])
def test_native_observe_rejects_short_raw_row_without_poisoning_storage(dimensions: tuple[int, ...]) -> None:
    """A malformed raw schema must raise a Python error, leaving both stores usable."""
    source = HistoryStore(2, dimensions, True)
    source.append(np.array([[0.0, 10.0]]))
    target = HistoryStore(2, (1, 1, 1, 1), False)
    target.configure_observation(np.ones(1), [0], False, False)
    with pytest.raises(ValueError):
        source.observe(target)
    np.testing.assert_array_equal(source.query(), [[0.0, 10.0]])
    assert target.query().shape == (0, 2)


@pytest.mark.parametrize("width,previous", [(3, None), (2, 2.0), (2, 1.0)])
def test_native_observe_rejects_incompatible_destination(width: int, previous: float | None) -> None:
    """A projected row cannot violate the destination schema or time ordering."""
    source = HistoryStore(2, (1, 1, 1, 1), True)
    source.append(np.array([[1.0, 10.0]]))
    target = HistoryStore(width, (1, 1, 1, 1), False)
    target.configure_observation(np.ones(1), [0], False, False)
    if previous is not None:
        target.append(np.array([[previous, 30.0]]))
    before = target.query()
    with pytest.raises(ValueError):
        source.observe(target)
    np.testing.assert_array_equal(target.query(), before)
    np.testing.assert_array_equal(source.query(), [[1.0, 10.0]])
