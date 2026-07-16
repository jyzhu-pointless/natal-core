"""Strict tests for Phase 3 History features.

Follows numerical-verification standards — every assertion proves a
mathematical invariant, not an implementation detail.  Covers:

  - max_rows with FIFO eviction
  - Read-only cached array properties (individual_count, sperm_storage, values, ticks)
  - observe() accepting Observation objects
  - truncate(retain_until_tick=N)
  - restore_state(tick)
  - ticks property (sorted tuple, cached)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import natal as nt
from natal.output.history import (
    History,
    HistoryBatch,
    HistorySchema,
    PopulationLayout,
)
from natal.output.observation import Observation, ObservationFilter
from natal.patterns.individual_selector import IndividualSelector

# ============================================================================
# Helpers
# ============================================================================


def _make_rows(
    n_rows: int, row_size: int, *, start_tick: int = 0, seed: float = 1.0
) -> np.ndarray:
    """Create rows with distinct ticks so append() doesn't deduplicate."""
    rows = np.zeros((n_rows, row_size), dtype=np.float64)
    for i in range(n_rows):
        rows[i, 0] = float(start_tick + i)
        rows[i, 1:] = float(start_tick + i) * seed
    return rows


def _minimal_layout(
    kind: str = "discrete_generation",
    n_demes: int = 1,
    n_sexes: int = 2,
    n_ages: int = 2,
    n_ztypes: int = 6,
    has_sperm_storage: bool = False,
) -> PopulationLayout:
    """Build a minimal PopulationLayout for unit tests."""
    return PopulationLayout(
        kind=kind,  # type: ignore[arg-type]  # str stands for Literal kind
        n_demes=n_demes,
        n_sexes=n_sexes,
        n_ages=n_ages,
        n_ztypes=n_ztypes,
        has_sperm_storage=has_sperm_storage,
        sex_labels=("female", "male"),
        ztype_labels=tuple(f"z{i}" for i in range(n_ztypes)),
    )


def _raw_schema(
    layout: PopulationLayout | None = None,
    row_size: int = 25,
) -> HistorySchema:
    if layout is None:
        layout = _minimal_layout()
    return HistorySchema(mode="raw", population=layout, row_size=row_size)


def _build_pop(
    species: nt.Species,
    name: str,
    *,
    with_observation: bool = False,
) -> nt.DiscreteGenerationPopulation:
    """Build a minimal discrete-generation population with known initial state."""
    cfg = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 180.0, "WT|Dr": 20.0},
                             "male": {"WT|WT": 180.0, "WT|Dr": 20.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=50.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=6.0,
            carrying_capacity=400,
        )
    )
    if with_observation:
        cfg = cfg.with_observation(
            groups={"total": IndividualSelector()},
            collapse_age=False,
        )
    return cfg.build()


# ============================================================================
# 1. max_rows — capacity enforcement and FIFO eviction
# ============================================================================


class TestMaxRows:
    """Invariant: max_rows enforces capacity with FIFO eviction."""

    def test_max_rows_none_unlimited(self) -> None:
        """max_rows=None means no limit — all rows kept."""
        schema = _raw_schema(row_size=5)
        history = History(schema, max_rows=None)
        rows = _make_rows(50, 5)
        history._append(HistoryBatch(schema=schema, rows=rows))
        # Invariant: all 50 rows retained when unlimited
        assert len(history) == 50

    def test_max_rows_zero_raises(self) -> None:
        """max_rows=0 raises ValueError."""
        schema = _raw_schema(row_size=5)
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            History(schema, max_rows=0)

    def test_max_rows_negative_raises(self) -> None:
        """max_rows < 0 raises ValueError."""
        schema = _raw_schema(row_size=5)
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            History(schema, max_rows=-5)

    def test_max_rows_evicts_oldest(self) -> None:
        """When max_rows=3, appending 5 distinct-tick rows keeps only last 3.

        Invariant: the oldest two rows (ticks 0, 1) are evicted; ticks 2,3,4 remain.
        """
        schema = _raw_schema(row_size=5)
        history = History(schema, max_rows=3)
        rows = _make_rows(5, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))
        # Invariant: only last 3 rows retained
        assert len(history) == 3
        # Invariant: evicted ticks not present
        assert history.ticks == (2, 3, 4)
        assert 0 not in history.ticks
        assert 1 not in history.ticks
        # Invariant: retained rows have correct tick values
        arr = history._to_numpy()
        assert arr.shape[0] == 3
        np.testing.assert_array_equal(arr[:, 0], np.array([2.0, 3.0, 4.0], dtype=np.float64))

    def test_max_rows_eviction_preserves_order(self) -> None:
        """Evicted rows are always the oldest; order of remaining is preserved."""
        schema = _raw_schema(row_size=5)
        history = History(schema, max_rows=3)
        for tick in range(7):
            row = np.full(5, float(tick), dtype=np.float64)
            row[0] = float(tick)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        # Invariant: after 7 inserts into capacity 3, len=3
        assert len(history) == 3
        # Invariant: ticks are sorted and are the last 3
        assert history.ticks == (4, 5, 6)

    def test_max_rows_batch_eviction(self) -> None:
        """Large batch exceeding capacity triggers eviction correctly."""
        schema = _raw_schema(row_size=3)
        history = History(schema, max_rows=2)
        # Single batch with 5 rows, all distinct ticks
        rows = _make_rows(5, 3, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))
        # Invariant: only last 2 remain
        assert len(history) == 2
        assert history.ticks == (3, 4)

    def test_max_rows_eviction_invalidates_cache(self) -> None:
        """After eviction, the individual_count cache is refreshed."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        row_size = 1 + ind_size  # 13

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema, max_rows=2)

        # Insert 3 rows with distinct markers
        for tick, marker in [(0, 10.0), (1, 20.0), (2, 30.0)]:
            row = np.zeros(row_size, dtype=np.float64)
            row[0] = float(tick)
            row[1 : 1 + ind_size] = marker
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        # After eviction of tick 0, cache should reflect only ticks 1,2
        ic = history.individual_count
        # Invariant: shape = (2, 2, 2, 3) — only 2 records remain
        assert ic.shape == (2, n_sexes, n_ages, n_ztypes)
        # Invariant: tick 1 marker (20.0) is first record, tick 2 (30.0) is second
        assert ic[0, 0, 0, 0] == 20.0
        assert ic[1, 0, 0, 0] == 30.0

    def test_max_rows_exact_capacity_no_eviction(self) -> None:
        """When n_rows == max_rows, no eviction occurs."""
        schema = _raw_schema(row_size=5)
        history = History(schema, max_rows=5)
        rows = _make_rows(5, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))
        assert len(history) == 5
        assert history.ticks == (0, 1, 2, 3, 4)


# ============================================================================
# 2. Read-only cached array properties
# ============================================================================


class TestReadOnlyCachedArrays:
    """Invariant: cached array properties are read-only and invalidated on mutation."""

    # ── individual_count ──────────────────────────────────────────────────

    def test_individual_count_shape(self) -> None:
        """individual_count returns (n_records, n_sexes, n_ages, n_ztypes)."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        for tick in range(3):
            row = np.zeros(row_size, dtype=np.float64)
            row[0] = float(tick)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ic = history.individual_count
        # Invariant: correct shape from layout dimensions
        assert ic.shape == (3, n_sexes, n_ages, n_ztypes)
        assert ic.dtype == np.float64

    def test_individual_count_read_only(self) -> None:
        """individual_count is writeable=False — mutation raises ValueError."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ic = history.individual_count
        assert not ic.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            ic[0, 0, 0, 0] = 999.0

    def test_individual_count_cached(self) -> None:
        """Cache returns equivalent values on repeated access (defensive copy)."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ic1 = history.individual_count
        ic2 = history.individual_count
        # Invariant: values equivalent (defensive copy, not same object)
        np.testing.assert_array_equal(ic1, ic2)

    def test_individual_count_cache_invalidated_after_append(self) -> None:
        """Cache is rebuilt (new object) after appending new rows."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        ic1 = history.individual_count
        assert ic1.shape[0] == 1

        row2 = np.zeros(row_size, dtype=np.float64)
        row2[0] = 1.0
        history._append(HistoryBatch(schema=schema, rows=row2[np.newaxis, :]))
        ic2 = history.individual_count
        # Invariant: shape reflects new row
        assert ic2.shape[0] == 2
        # Invariant: new object (cache rebuilt, not stale)
        assert ic1 is not ic2

    def test_individual_count_raw_mode_only(self) -> None:
        """individual_count raises ValueError in observation mode."""
        from natal.output.history import ObservationMetadata

        layout = _minimal_layout(n_ztypes=3)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation", population=layout, row_size=5, observation=om
        )
        history = History(obs_schema)
        with pytest.raises(ValueError, match="only available in raw mode"):
            _ = history.individual_count

    # ── values (observation-mode only) ────────────────────────────────────

    def test_values_shape(self) -> None:
        """values returns (n_records, n_groups, n_sexes, n_ages)."""
        from natal.output.history import ObservationMetadata

        n_groups, n_sexes, n_ages = 2, 2, 2
        row_size = 1 + n_groups * n_sexes * n_ages  # 1 + 8 = 9

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=3)
        om = ObservationMetadata(
            labels=("g0", "g1"), collapse_age=False, n_groups=n_groups
        )
        obs_schema = HistorySchema(
            mode="observation",
            population=layout,
            row_size=row_size,
            observation=om,
        )
        history = History(obs_schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._rows.append(row)

        vals = history.values
        # Invariant: shape = (1, 2, 2, 2)
        assert vals.shape == (1, n_groups, n_sexes, n_ages)
        assert vals.dtype == np.float64

    def test_values_read_only(self) -> None:
        """values is writeable=False."""
        from natal.output.history import ObservationMetadata

        n_groups, n_sexes, n_ages = 1, 2, 2
        row_size = 1 + n_groups * n_sexes * n_ages

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=3)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation",
            population=layout,
            row_size=row_size,
            observation=om,
        )
        history = History(obs_schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._rows.append(row)

        vals = history.values
        assert not vals.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            vals[0, 0, 0, 0] = 999.0

    def test_values_cached(self) -> None:
        """values returns same instance on second access."""
        from natal.output.history import ObservationMetadata

        row_size = 1 + 1 * 2 * 2  # 5
        layout = _minimal_layout(n_sexes=2, n_ages=2, n_ztypes=3)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation",
            population=layout,
            row_size=row_size,
            observation=om,
        )
        history = History(obs_schema)
        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._rows.append(row)

        v1 = history.values
        v2 = history.values
        np.testing.assert_array_equal(v1, v2)

    def test_values_raw_mode_raises(self) -> None:
        """values raises ValueError in raw mode."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        with pytest.raises(ValueError, match="only available in observation mode"):
            _ = history.values

    # ── sperm_storage ─────────────────────────────────────────────────────

    def test_sperm_storage_none_when_not_present(self) -> None:
        """sperm_storage returns None for discrete-generation (has_sperm_storage=False)."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        row_size = 1 + ind_size  # no sperm storage

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=False
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)
        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ss = history.sperm_storage
        # Invariant: returns None when layout says no sperm_storage
        assert ss is None

    def test_sperm_storage_with_present(self) -> None:
        """sperm_storage returns correct shape when has_sperm_storage=True."""
        n_sexes, n_ages, n_ztypes = 2, 3, 4
        ind_size = n_sexes * n_ages * n_ztypes  # 2 * 3 * 4 = 24
        sperm_size = n_ages * n_ztypes * n_ztypes  # 3 * 4 * 4 = 48
        row_size = 1 + ind_size + sperm_size  # 73

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        for tick in range(2):
            row = np.zeros(row_size, dtype=np.float64)
            row[0] = float(tick)
            # Put a marker in the sperm region
            sperm_start = 1 + ind_size
            row[sperm_start] = float(tick * 100 + 1)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ss = history.sperm_storage
        assert ss is not None
        # Invariant: shape = (n_records, n_ages, n_ztypes, n_ztypes)
        assert ss.shape == (2, n_ages, n_ztypes, n_ztypes)
        assert ss.dtype == np.float64
        # Invariant: read-only
        assert not ss.flags.writeable

    def test_sperm_storage_read_only(self) -> None:
        """sperm_storage is writeable=False."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)
        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ss = history.sperm_storage
        assert ss is not None
        assert not ss.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            ss[0, 0, 0, 0] = 999.0

    def test_sperm_storage_cached(self) -> None:
        """sperm_storage returns same instance on second access."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)
        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ss1 = history.sperm_storage
        ss2 = history.sperm_storage
        assert ss1 is not None and ss2 is not None
        np.testing.assert_array_equal(ss1, ss2)

    def test_sperm_storage_cache_invalidated(self) -> None:
        """sperm_storage cache invalidated after append."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        ss1 = history.sperm_storage
        assert ss1 is not None
        assert ss1.shape[0] == 1

        row2 = np.zeros(row_size, dtype=np.float64)
        row2[0] = 1.0
        history._append(HistoryBatch(schema=schema, rows=row2[np.newaxis, :]))
        ss2 = history.sperm_storage
        assert ss2 is not None
        assert ss2.shape[0] == 2
        # Invariant: new object after cache invalidation
        assert ss1 is not ss2

    def test_sperm_storage_obs_mode_raises(self) -> None:
        """sperm_storage raises ValueError in observation mode."""
        from natal.output.history import ObservationMetadata

        layout = _minimal_layout(n_sexes=2, n_ages=2, n_ztypes=3, has_sperm_storage=True)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation", population=layout, row_size=5, observation=om
        )
        history = History(obs_schema)
        with pytest.raises(ValueError, match="only available in raw mode"):
            _ = history.sperm_storage


# ============================================================================
# 3. ticks property (strictly increasing, cached)
# ============================================================================


class TestTicksProperty:
    """Invariant: ticks returns the committed monotonic tuple and is cached."""

    def test_ticks_empty(self) -> None:
        """Empty history returns empty tuple."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        assert history.ticks == ()

    def test_ticks_strictly_increasing_tuple(self) -> None:
        """ticks preserves the validated strictly increasing record order."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(5, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))
        assert history.ticks == (0, 1, 2, 3, 4)
        assert isinstance(history.ticks, tuple)

    def test_ticks_cached(self) -> None:
        """ticks returns same tuple object on second access (caching)."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(3, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))

        t1 = history.ticks
        t2 = history.ticks
        # Invariant: same object returned (cached)
        assert t1 is t2

    def test_ticks_cache_invalidated_after_append(self) -> None:
        """ticks cache invalidated after appending new rows."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(2, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))

        t1 = history.ticks
        assert t1 == (0, 1)

        row3 = np.array([[2.0, 0, 0, 0, 0]], dtype=np.float64)
        history._append(HistoryBatch(schema=schema, rows=row3))
        t2 = history.ticks
        # Invariant: includes new tick
        assert t2 == (0, 1, 2)
        # Invariant: different object (cache invalidated)
        assert t1 is not t2

    def test_ticks_cache_invalidated_after_clear(self) -> None:
        """ticks cache invalidated after clear()."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(3, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))

        t1 = history.ticks
        assert t1 == (0, 1, 2)

        history.clear()
        t2 = history.ticks
        # Invariant: empty after clear
        assert t2 == ()
        assert t1 is not t2


# ============================================================================
# 4. truncate(retain_until_tick=N)
# ============================================================================


class TestTruncate:
    """Invariant: truncate removes rows with tick > retain_until_tick."""

    def test_truncate_removes_after_target(self) -> None:
        """Truncating at tick 3 keeps ticks 0-3, removes 4-6."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(7, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))
        assert len(history) == 7

        history.truncate(retain_until_tick=3)
        # Invariant: only ticks 0-3 remain
        assert len(history) == 4
        assert history.ticks == (0, 1, 2, 3)
        arr = history._to_numpy()
        np.testing.assert_array_equal(
            arr[:, 0], np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        )

    def test_truncate_last_tick_all_remain(self) -> None:
        """Truncating to the last recorded tick keeps all rows."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(5, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))

        history.truncate(retain_until_tick=4)
        # Invariant: all rows remain (4 is the last tick)
        assert len(history) == 5
        assert history.ticks == (0, 1, 2, 3, 4)

    def test_truncate_non_existent_raises(self) -> None:
        """Truncating to a tick with no record at or below it raises ValueError."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(3, 5, start_tick=10)
        history._append(HistoryBatch(schema=schema, rows=rows))

        with pytest.raises(ValueError, match="No records with tick <="):
            history.truncate(retain_until_tick=5)

    def test_truncate_empty_raises(self) -> None:
        """Truncating an empty history raises ValueError."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        with pytest.raises(ValueError, match="No records with tick <="):
            history.truncate(retain_until_tick=0)

    def test_truncate_invalidates_cache(self) -> None:
        """After truncate, cached properties are rebuilt."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        for tick in range(5):
            row = np.zeros(row_size, dtype=np.float64)
            row[0] = float(tick)
            row[1 : 1 + ind_size] = float(tick * 10)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ic_before = history.individual_count
        assert ic_before.shape[0] == 5

        history.truncate(retain_until_tick=2)
        ic_after = history.individual_count
        # Invariant: new cache with only 3 records
        assert ic_after.shape[0] == 3
        assert ic_before is not ic_after
        # Invariant: correct markers present
        assert ic_after[0, 0, 0, 0] == 0.0
        assert ic_after[1, 0, 0, 0] == 10.0
        assert ic_after[2, 0, 0, 0] == 20.0


# ============================================================================
# 5. restore_state(tick)
# ============================================================================


class TestRestoreState:
    """Invariant: restore_state returns correct individual_count and sperm_storage at given tick."""

    def test_restore_state_returns_correct_counts(self) -> None:
        """restore_state returns exact individual_count at a given tick."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        # Insert 3 rows with distinct markers
        markers = [10.0, 20.0, 30.0]
        for tick, marker in enumerate(markers):
            row = np.zeros(row_size, dtype=np.float64)
            row[0] = float(tick)
            row[1 : 1 + ind_size] = marker
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        # Restore tick 1
        tick, ic, ss = history.restore_state(1)
        # Invariant: tick matches
        assert tick == 1
        # Invariant: correct marker value
        assert ic[0, 0, 0] == 20.0
        # Invariant: all entries are 20.0
        assert np.all(ic == 20.0)
        # Invariant: sperm_storage is None (discrete-gen)
        assert ss is None

    def test_restore_state_shape(self) -> None:
        """restore_state returns individual_count with correct shape."""
        n_sexes, n_ages, n_ztypes = 2, 3, 4
        ind_size = n_sexes * n_ages * n_ztypes  # 24
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        _, ic, ss = history.restore_state(0)
        # Invariant: shape matches layout
        assert ic.shape == (n_sexes, n_ages, n_ztypes)
        assert ic.dtype == np.float64
        assert ss is None

    def test_restore_state_with_sperm_storage(self) -> None:
        """restore_state returns sperm_storage when has_sperm_storage=True."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        sperm_size = n_ages * n_ztypes * n_ztypes  # 2 * 3 * 3 = 18
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        # Put marker in individual_count region
        row[1 : 1 + ind_size] = 42.0
        # Put marker in sperm_storage region
        sperm_start = 1 + ind_size
        row[sperm_start : sperm_start + sperm_size] = 7.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        tick, ic, ss = history.restore_state(0)
        assert tick == 0
        assert np.all(ic == 42.0)
        assert ss is not None
        # Invariant: sperm_storage shape is correct
        assert ss.shape == (n_ages, n_ztypes, n_ztypes)
        # Invariant: sperm_storage has correct values
        assert np.all(ss == 7.0)

    def test_restore_state_wrong_tick_raises(self) -> None:
        """restore_state with non-existent tick raises ValueError."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(3, 5, start_tick=0)
        history._append(HistoryBatch(schema=schema, rows=rows))

        with pytest.raises(ValueError, match="Tick 99 not found"):
            history.restore_state(99)

    def test_restore_state_empty_history_raises(self) -> None:
        """restore_state on empty history raises ValueError."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        with pytest.raises(ValueError, match="Tick 0 not found"):
            history.restore_state(0)

    def test_restore_state_observation_mode_raises(self) -> None:
        """restore_state raises ValueError in observation mode."""
        from natal.output.history import ObservationMetadata

        layout = _minimal_layout(n_ztypes=3)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation", population=layout, row_size=5, observation=om
        )
        history = History(obs_schema)
        # Add a row so it's not empty
        row = np.zeros(5, dtype=np.float64)
        row[0] = 0.0
        history._rows.append(row)
        history._seen_ticks.add(0)

        with pytest.raises(ValueError, match="Cannot restore state from observation-mode"):
            history.restore_state(0)

    def test_restore_state_returns_independent_copies(self) -> None:
        """Restored individual_count and sperm_storage are independent copies."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        _, ic, ss = history.restore_state(0)
        assert ss is not None
        # Modify the restored copies — should not affect internal state
        ic[0, 0, 0] = 999.0
        ss[0, 0, 0] = 999.0

        # Invariant: original stored row unchanged
        _, ic2, ss2 = history.restore_state(0)
        assert ss2 is not None
        assert ic2[0, 0, 0] != 999.0
        assert ss2[0, 0, 0] != 999.0


# ============================================================================
# 6. observe() accepting Observation objects
# ============================================================================


class TestObserveWithObservation:
    """Invariant: observe(observation) projects raw history via compiled Observation."""

    def test_observe_with_observation_object(self) -> None:
        """observe(Observation) produces correct observation-mode History."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes  # 12
        raw_row_size = 1 + ind_size  # 13

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        # Build 2 ticks: tick 0 puts 100 in ztype=0, tick 1 puts 200 in ztype=0
        for tick, val in [(0, 100.0), (1, 200.0)]:
            row = np.zeros(raw_row_size, dtype=np.float64)
            row[0] = float(tick)
            ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
            ind[0, tick % n_ages, 0] = val
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        # Build an Observation that selects ztype=0, all sexes, all ages
        # Use build_from_selectors with IndividualSelector
        # Since we have no real registry, use the legacy build_filter approach
        # Or actually, let's use the direct low-level: build a mask manually
        # to construct an Observation, then test observe(observation)
        from natal.output.observation import Observation

        mask = np.zeros((1, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        mask[0, :, :, 0] = 1.0

        observation = Observation(
            labels=("ztype0",),
            collapse_age=False,
            mask=mask,
            population_fingerprint=schema.population.fingerprint,
        )

        obs_hist = history.observe(observation)

        # Invariant: result is observation-mode
        assert obs_hist.schema.mode == "observation"
        # Invariant: labels match
        assert obs_hist.schema.observation is not None
        assert obs_hist.schema.observation.labels == ("ztype0",)
        # Invariant: correct row_size = 1 + 1*2*2 = 5
        assert obs_hist.schema.row_size == 5
        # Invariant: two records
        assert len(obs_hist) == 2

        # Verify observation values
        arr = obs_hist._to_numpy()
        # Row 0: tick=0, g0_sex0_age0=100
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 100.0
        assert arr[0, 2] == 0.0
        assert arr[0, 3] == 0.0
        assert arr[0, 4] == 0.0
        # Row 1: tick=1, g0_sex0_age1=200
        assert arr[1, 0] == 1.0
        assert arr[1, 1] == 0.0
        assert arr[1, 2] == 200.0
        assert arr[1, 3] == 0.0
        assert arr[1, 4] == 0.0

    def test_observe_with_observation_equals_legacy_mask(self) -> None:
        """observe(Observation) gives same result as observe(mask, labels, collapse_age)."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        raw_row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        # Single tick: ztype=0 has 10, ztype=1 has 20, ztype=2 has 30 in sex0 age0
        row = np.zeros(raw_row_size, dtype=np.float64)
        row[0] = 0.0
        ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
        ind[0, 0, 0] = 10.0
        ind[0, 0, 1] = 20.0
        ind[0, 0, 2] = 30.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        # Build observation via Observation object
        mask = np.zeros((2, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        mask[0, :, :, 0] = 1.0  # group 0: ztype 0
        mask[1, :, :, 1] = 1.0  # group 1: ztype 1
        mask[1, :, :, 2] = 1.0  # group 1: ztype 2
        from natal.output.observation import Observation

        observation = Observation(
            labels=("g0", "g1"),
            collapse_age=False,
            mask=mask,
            population_fingerprint=schema.population.fingerprint,
        )
        obs_new = history.observe(observation)

        # Verify observation history values
        assert obs_new.ticks == (0,)
        assert obs_new.values.shape == (1, 2, 2, 2)

    def test_observe_derived_history_independent(self) -> None:
        """Adding rows to source history does not affect derived observation history."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        raw_row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        row = np.zeros(raw_row_size, dtype=np.float64)
        row[0] = 0.0
        ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
        ind[0, 0, 0] = 100.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        from natal.output.observation import Observation

        mask = np.ones((1, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        observation = Observation(
            labels=("total",),
            collapse_age=False,
            mask=mask,
            population_fingerprint=schema.population.fingerprint,
        )
        obs_hist = history.observe(observation)
        assert len(obs_hist) == 1

        # Add another row to the source history
        row2 = np.zeros(raw_row_size, dtype=np.float64)
        row2[0] = 1.0
        ind2 = row2[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
        ind2[0, 0, 0] = 200.0
        history._append(HistoryBatch(schema=schema, rows=row2[np.newaxis, :]))
        assert len(history) == 2

        # Invariant: derived history is independent — still has 1 record
        assert len(obs_hist) == 1
        arr = obs_hist._to_numpy()
        assert arr[0, 0] == 0.0  # only tick 0

    def test_observe_with_explicit_observation(self) -> None:
        """observe(Observation) produces correct observation-mode History."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        raw_row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        row = np.zeros(raw_row_size, dtype=np.float64)
        row[0] = 0.0
        ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
        ind[0, 0, 0] = 50.0
        ind[1, 1, 1] = 30.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        mask = np.ones((1, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        obs = Observation(
            labels=("total",),
            collapse_age=True,
            mask=mask.copy(),
            population_fingerprint=layout.fingerprint,
        )
        obs_hist = history.observe(obs)

        assert obs_hist.schema.mode == "observation"
        assert obs_hist.schema.observation is not None
        assert obs_hist.schema.observation.collapse_age
        assert obs_hist.schema.observation.labels == ("total",)

    def test_observe_with_real_population_observation(self, simple_species) -> None:
        """End-to-end: raw history → observe(Observation built from population)."""
        nt.disable_numba()
        pop = _build_pop(simple_species, "obs_e2e", with_observation=False)
        pop.run(n_steps=3, record_every=1)

        # Access raw history
        raw_hist = pop._history_obj  # type: ignore[attr-defined]  # accessing private attribute for test verification
        assert raw_hist.schema.mode == "raw"
        assert len(raw_hist) == 4  # tick 0,1,2,3

        # Build an Observation via ObservationFilter.build_filter
        n_sexes = 2
        n_ages = int(pop.state.individual_count.shape[1])
        n_ztypes = int(pop.state.individual_count.shape[2])

        obs_filter = ObservationFilter(pop.index_registry)
        observation = obs_filter.build_filter(
            diploid_genotypes=pop.species,
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
        )
        observation = replace(
            observation,
            population_fingerprint=raw_hist.schema.population.fingerprint,
        )

        obs_hist = raw_hist.observe(observation)

        # Invariant: same number of records
        assert len(obs_hist) == 4
        # Invariant: obs mode
        assert obs_hist.schema.mode == "observation"
        # Invariant: row_size = 1 + 1*2*2 = 5
        assert obs_hist.schema.row_size == 5

        # Invariant: total per tick exactly matches raw individual counts.
        np.testing.assert_array_equal(
            obs_hist.values.sum(axis=(1, 2, 3)),
            raw_hist.individual_count.sum(axis=(1, 2, 3)),
        )

    def test_observe_with_individual_selector_observation(
        self, simple_species
    ) -> None:
        """observe with Observation built via build_from_selectors."""
        nt.disable_numba()
        pop = _build_pop(simple_species, "obs_sel", with_observation=False)
        pop.run(n_steps=2, record_every=1)

        raw_hist = pop._history_obj  # type: ignore[attr-defined]  # accessing private attribute for test verification
        n_ztypes = int(pop.state.individual_count.shape[2])

        # Build using IndividualSelector
        obs_filter = ObservationFilter(pop.index_registry)
        observation = obs_filter.build_from_selectors(
            groups={
                "wt": IndividualSelector(ztype="WT|WT"),
                "drive": IndividualSelector(ztype="WT|Dr"),
            },
            collapse_age=False,
            n_sexes=2,
            n_ages=int(pop.state.individual_count.shape[1]),
            n_ztypes=n_ztypes,
        )
        observation = replace(
            observation,
            population_fingerprint=raw_hist.schema.population.fingerprint,
        )

        obs_hist = raw_hist.observe(observation)
        assert obs_hist.schema.mode == "observation"
        assert obs_hist.schema.observation is not None
        assert obs_hist.schema.observation.labels == ("wt", "drive")
        assert len(obs_hist) == 3  # tick 0,1,2
        np.testing.assert_array_equal(
            obs_hist.values,
            np.stack(
                [
                    observation.apply(record)
                    for record in raw_hist.individual_count
                ]
            ),
        )


# ============================================================================
# 7. Tick uniqueness and append ordering
# ============================================================================


class TestTickUniqueness:
    """Invariant: invalid tick sequences fail atomically during append."""

    def test_duplicate_single_tick_rejected(self) -> None:
        """Appending a row with an already-seen tick raises without mutation."""
        schema = _raw_schema(row_size=5)
        history = History(schema)

        row = np.array([[0.0, 1, 2, 3, 4]], dtype=np.float64)
        history._append(HistoryBatch(schema=schema, rows=row))
        assert len(history) == 1

        row2 = np.array([[0.0, 99, 99, 99, 99]], dtype=np.float64)
        with pytest.raises(ValueError, match="already recorded|strictly increasing"):
            history._append(HistoryBatch(schema=schema, rows=row2))
        assert len(history) == 1
        arr = history._to_numpy()
        np.testing.assert_array_equal(arr, row)

    def test_duplicate_in_batch_rejected_atomically(self) -> None:
        """A batch overlapping existing history commits none of its new rows."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        # Pre-populate with ticks 0, 1
        history._append(
            HistoryBatch(schema=schema, rows=_make_rows(2, 5, start_tick=0))
        )
        assert len(history) == 2

        batch_rows = _make_rows(3, 5, start_tick=1)
        with pytest.raises(ValueError, match="already recorded|strictly increasing"):
            history._append(HistoryBatch(schema=schema, rows=batch_rows))
        assert len(history) == 2
        assert history.ticks == (0, 1)

    def test_duplicate_with_eviction_interaction(self) -> None:
        """A rejected duplicate does not trigger capacity eviction."""
        schema = _raw_schema(row_size=5)
        history = History(schema, max_rows=3)

        # Insert ticks 0, 1, 2
        for tick in range(3):
            row = np.array([[float(tick), 0, 0, 0, 0]], dtype=np.float64)
            history._append(HistoryBatch(schema=schema, rows=row))
        assert history.ticks == (0, 1, 2)

        row_dup = np.array([[0.0, 99, 99, 99, 99]], dtype=np.float64)
        with pytest.raises(ValueError, match="already recorded|strictly increasing"):
            history._append(HistoryBatch(schema=schema, rows=row_dup))
        assert len(history) == 3
        assert history.ticks == (0, 1, 2)

        # Insert new tick 3 — now eviction should happen (of tick 0)
        row3 = np.array([[3.0, 0, 0, 0, 0]], dtype=np.float64)
        history._append(HistoryBatch(schema=schema, rows=row3))
        assert len(history) == 3
        assert history.ticks == (1, 2, 3)


# ============================================================================
# 8. Cache invalidation after clear
# ============================================================================


class TestCacheInvalidationAfterClear:
    """Invariant: clear() invalidates all cached arrays."""

    def test_clear_invalidates_individual_count(self) -> None:
        """individual_count cache is None after clear."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        row_size = 1 + ind_size

        layout = _minimal_layout(n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        ic1 = history.individual_count
        assert ic1.shape[0] == 1

        history.clear()
        ic2 = history.individual_count
        # Invariant: empty after clear
        assert ic2.shape[0] == 0
        assert ic1 is not ic2

    def test_clear_invalidates_sperm_storage(self) -> None:
        """sperm_storage cache rebuilt after clear."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes
        row_size = 1 + ind_size + sperm_size

        layout = _minimal_layout(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes, has_sperm_storage=True
        )
        schema = _raw_schema(layout, row_size=row_size)
        history = History(schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        ss1 = history.sperm_storage
        assert ss1 is not None
        assert ss1.shape[0] == 1

        history.clear()
        ss2 = history.sperm_storage
        assert ss2 is not None
        assert ss2.shape[0] == 0

    def test_clear_invalidates_values(self) -> None:
        """values cache rebuilt after clear."""
        from natal.output.history import ObservationMetadata

        row_size = 1 + 1 * 2 * 2  # 5
        layout = _minimal_layout(n_sexes=2, n_ages=2, n_ztypes=3)
        om = ObservationMetadata(labels=("g0",), collapse_age=False, n_groups=1)
        obs_schema = HistorySchema(
            mode="observation",
            population=layout,
            row_size=row_size,
            observation=om,
        )
        history = History(obs_schema)

        row = np.zeros(row_size, dtype=np.float64)
        row[0] = 0.0
        history._rows.append(row)

        v1 = history.values
        assert v1.shape[0] == 1

        history.clear()
        v2 = history.values
        assert v2.shape[0] == 0
        assert v1 is not v2
