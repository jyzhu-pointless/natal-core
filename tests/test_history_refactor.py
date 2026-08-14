"""Strict tests for the refactored history, observation, and recording modules.

Follows numerical-verification standards (every assertion proves a numerical
invariant).  Covers:

  - History / HistorySchema / HistoryBatch data model invariants
  - PopulationLayout construction and fingerprint stability
  - History.observe() post-hoc observation projection
  - RecordingPlan immutability and compile_recording_plan() row_size
  - Observation.apply / build_mask / collapse_age / lazy-rebuild
  - End-to-end regression: .with_observation() → run() → output_history()
  - Raw-history post-hoc == build-time observation history equivalence
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.output._recording import RecordingPlan, compile_recording_plan
from natal.output.history import (
    History,
    HistoryBatch,
    HistorySchema,
    ObservationMetadata,
    PopulationLayout,
    SpatialHistoryLayout,
)
from natal.output.observation import Observation, ObservationFilter, apply_rule
from natal.patterns import IndividualSelector


@pytest.fixture(autouse=True)
def _restore_numba_state() -> None:
    """Ensure Numba is re-enabled after any test that disables it."""
    yield
    nt.enable_numba()

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
        rows[i, 1:] = float(start_tick + i) * seed  # marker value
    return rows


def _minimal_layout(
    kind: str = "discrete_generation",
    n_demes: int = 1,
    n_sexes: int = 2,
    n_ages: int = 2,
    n_ztypes: int = 6,
) -> PopulationLayout:
    """Build a minimal PopulationLayout for unit tests."""
    return PopulationLayout(
        kind=kind,  # type: ignore[arg-type]  # str stands for Literal kind
        n_demes=n_demes,
        n_sexes=n_sexes,
        n_ages=n_ages,
        n_ztypes=n_ztypes,
        has_sperm_storage=False,
        sex_labels=("female", "male"),
        ztype_labels=tuple(f"z{i}" for i in range(n_ztypes)),
    )


def _raw_schema(layout: Optional[PopulationLayout] = None, row_size: int = 25) -> HistorySchema:
    """Build a raw history schema with optional layout and width overrides."""
    if layout is None:
        layout = _minimal_layout()
    return HistorySchema(mode="raw", population=layout, row_size=row_size)


def _obs_meta(
    labels: tuple[str, ...] = ("group_0",),
    collapse_age: bool = False,
) -> ObservationMetadata:
    """Build observation metadata for named test groups."""
    return ObservationMetadata(labels=labels, collapse_age=collapse_age, n_groups=len(labels))


def _obs_schema(
    layout: Optional[PopulationLayout] = None,
    row_size: int = 5,
    obs_meta_val: Optional[ObservationMetadata] = None,
) -> HistorySchema:
    """Build an observation-mode history schema for tests."""
    if layout is None:
        layout = _minimal_layout()
    if obs_meta_val is None:
        obs_meta_val = _obs_meta()
    return HistorySchema(
        mode="observation",
        population=layout,
        row_size=row_size,
        observation=obs_meta_val,
    )


# ============================================================================
# 1. HistorySchema validation invariants
# ============================================================================


class TestHistorySchemaValidation:
    """Invariant: HistorySchema enforces mode/metadata consistency."""

    def test_observation_mode_requires_metadata(self) -> None:
        """Schema in observation mode MUST have ObservationMetadata."""
        layout = _minimal_layout()
        with pytest.raises(ValueError, match="observation-mode schema requires"):
            HistorySchema(mode="observation", population=layout, row_size=5)

    def test_raw_mode_forbids_metadata(self) -> None:
        """Schema in raw mode MUST NOT have ObservationMetadata."""
        layout = _minimal_layout()
        om = _obs_meta()
        with pytest.raises(ValueError, match="raw-mode schema must not have"):
            HistorySchema(mode="raw", population=layout, row_size=25, observation=om)

    def test_positive_row_size_required(self) -> None:
        """row_size must be > 0."""
        layout = _minimal_layout()
        with pytest.raises(ValueError, match="row_size must be positive"):
            HistorySchema(mode="raw", population=layout, row_size=0)
        with pytest.raises(ValueError, match="row_size must be positive"):
            HistorySchema(mode="raw", population=layout, row_size=-1)

    def test_valid_schema_constructs(self) -> None:
        """Valid schemas construct without error and preserve fields."""
        layout = _minimal_layout()
        raw = _raw_schema(layout, row_size=25)
        assert raw.mode == "raw"
        assert raw.population is layout
        assert raw.row_size == 25
        assert raw.observation is None

        om = _obs_meta()
        obs = _obs_schema(layout, row_size=5, obs_meta_val=om)
        assert obs.mode == "observation"
        assert obs.population is layout
        assert obs.row_size == 5
        assert obs.observation is om
        assert obs.observation.labels == ("group_0",)
        assert obs.observation.n_groups == 1


# ============================================================================
# 2. HistoryBatch validation invariants
# ============================================================================


class TestHistoryBatchValidation:
    """Invariant: HistoryBatch enforces 2-D rows and row width == schema.row_size."""

    def test_valid_batch_constructs(self) -> None:
        """Accept rows whose dimensions match the schema."""
        schema = _raw_schema(row_size=5)
        rows = np.ones((3, 5), dtype=np.float64)
        batch = HistoryBatch(schema=schema, rows=rows)
        assert batch.schema is schema
        np.testing.assert_array_equal(batch.rows, rows)

    def test_1d_rows_rejected(self) -> None:
        """Reject a one-dimensional row buffer."""
        schema = _raw_schema(row_size=5)
        with pytest.raises(ValueError, match="rows must be 2-D"):
            HistoryBatch(schema=schema, rows=np.ones(5, dtype=np.float64))

    def test_row_width_mismatch_rejected(self) -> None:
        """Reject rows whose width differs from the schema."""
        schema = _raw_schema(row_size=5)
        with pytest.raises(ValueError, match="does not match schema row_size"):
            HistoryBatch(schema=schema, rows=np.ones((2, 7), dtype=np.float64))

    def test_batch_immutable(self) -> None:
        """HistoryBatch is frozen (dataclass)."""
        schema = _raw_schema(row_size=5)
        batch = HistoryBatch(schema=schema, rows=np.ones((2, 5), dtype=np.float64))
        with pytest.raises(FrozenInstanceError):
            batch.rows = np.ones((2, 5), dtype=np.float64)  # type: ignore[misc]  # testing frozen dataclass mutation guard

    def test_schema_equality_structural(self) -> None:
        """Two schemas with identical fields compare equal."""
        layout = _minimal_layout()
        s1 = _raw_schema(layout, row_size=25)
        s2 = _raw_schema(layout, row_size=25)
        assert s1 == s2
        # Different row_size → not equal
        s3 = _raw_schema(layout, row_size=99)
        assert s1 != s3


# ============================================================================
# 3. History.append rejects schema mismatch
# ============================================================================


class TestHistoryAppend:
    """Invariant: History.append rejects batches with non-matching schema."""

    def test_append_valid_batch(self) -> None:
        """Append a valid batch and retain both rows."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        # Two rows with different ticks
        rows = np.ones((2, 5), dtype=np.float64)
        rows[0, 0] = 0.0
        rows[1, 0] = 1.0
        batch = HistoryBatch(schema=schema, rows=rows)
        history._append(batch)
        # Invariant: after appending 2 rows with distinct ticks, length == 2
        assert len(history) == 2

    def test_schema_mismatch_rejected(self) -> None:
        """Reject a batch built for a different schema."""
        s1 = _raw_schema(row_size=5)
        # Construct genuinely different schemas with different row_size
        s_big = _raw_schema(row_size=10)
        history = History(s1)
        batch = HistoryBatch(schema=s_big, rows=np.ones((2, 10), dtype=np.float64))
        with pytest.raises(ValueError, match="does not match History schema"):
            history._append(batch)

    def test_append_empty_batch_noop(self) -> None:
        """Appending zero-row batch does not change history."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        batch = HistoryBatch(schema=schema, rows=np.zeros((0, 5), dtype=np.float64))
        history._append(batch)
        assert len(history) == 0  # invariant: still empty

    def test_duplicate_tick_rejected(self) -> None:
        """Appending a repeated tick raises and preserves the original row."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        # First row: tick=1
        row1 = np.zeros(5, dtype=np.float64)
        row1[0] = 1.0
        batch1 = HistoryBatch(schema=schema, rows=row1[np.newaxis, :])
        history._append(batch1)
        assert len(history) == 1

        # Second batch: tick=1 again (same tick)
        row2 = np.zeros(5, dtype=np.float64)
        row2[0] = 1.0
        batch2 = HistoryBatch(schema=schema, rows=row2[np.newaxis, :])
        with pytest.raises(ValueError, match="already recorded|strictly increasing"):
            history._append(batch2)
        assert len(history) == 1
        np.testing.assert_array_equal(history._to_numpy(), batch1.rows)

    def test_different_tick_appended(self) -> None:
        """Appending different ticks adds rows."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        for tick in range(5):
            row = np.zeros(5, dtype=np.float64)
            row[0] = float(tick)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))
        assert len(history) == 5


# ============================================================================
# 4. History.clear() preserves schema
# ============================================================================


class TestHistoryClear:
    """Invariant: clear() removes all rows but keeps the schema unchanged."""

    def test_clear_preserves_schema(self) -> None:
        """Remove all rows without replacing the schema."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(3, 5)
        history._append(HistoryBatch(schema=schema, rows=rows))
        assert len(history) == 3

        history.clear()
        # Invariant: schema unchanged
        assert history.schema is schema
        # Invariant: empty after clear
        assert len(history) == 0
        assert history.is_empty
        assert history.n_records == 0

    def test_clear_then_append(self) -> None:
        """After clear, new rows can be appended to the same schema."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        history._append(HistoryBatch(schema=schema, rows=_make_rows(2, 5)))
        history.clear()
        history._append(HistoryBatch(schema=schema, rows=_make_rows(3, 5)))
        assert len(history) == 3


# ============================================================================
# 5. History.to_numpy() returns correct dimensions
# ============================================================================


class TestHistoryToNumpy:
    """Invariant: to_numpy() shape = (n_records, row_size)."""

    def test_empty_returns_zero_row_array(self) -> None:
        """Represent empty history with a correctly typed zero-row array."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        arr = history._to_numpy()
        # Invariant: (0, row_size) for empty history
        assert arr.shape == (0, 5)
        assert arr.dtype == np.float64

    def test_n_rows_returns_n_by_row_size(self) -> None:
        """Return one flat row per stored record."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        for tick in range(3):
            row = np.arange(5, dtype=np.float64) + tick * 10
            row[0] = float(tick)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        arr = history._to_numpy()
        # Invariant: shape = (n_records, schema.row_size)
        assert arr.shape == (3, 5)
        # Invariant: tick values preserved
        np.testing.assert_array_equal(arr[:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float64))


# ============================================================================
# 6. History.__len__() and __iter__() match row count
# ============================================================================


class TestHistoryLenIter:
    """Invariant: len(history) == len(list(history)) == n_records."""

    def test_len_matches_n_records(self) -> None:
        """Keep the length protocol synchronized with ``n_records``."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows = _make_rows(7, 5)
        history._append(HistoryBatch(schema=schema, rows=rows))
        assert len(history) == 7
        assert history.n_records == 7

    def test_iter_yields_all_rows(self) -> None:
        """Yield every stored tick exactly once in order."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        for tick in range(4):
            row = np.zeros(5, dtype=np.float64)
            row[0] = float(tick)
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        ticks = [t for t, _ in history]
        assert ticks == [0, 1, 2, 3]
        # Invariant: iteration count matches len
        assert len(list(history)) == len(history) == 4


# ============================================================================
# 7. PopulationLayout.from_population() invariants
# ============================================================================


class TestPopulationLayout:
    """Invariant: from_population() generates correct ztype labels and stable fingerprint."""

    def test_from_population_ztype_labels(self, simple_species) -> None:
        """ztype_labels are derived from registry index_to_ztype."""
        # Build a minimal discrete-gen population to access its registry
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="layout_test", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        registry = pop.index_registry

        layout = PopulationLayout.from_population(
            kind="discrete_generation",
            n_demes=1,
            n_sexes=2,
            n_ages=2,
            n_ztypes=pop.state.individual_count.shape[2],
            has_sperm_storage=False,
            sex_labels=("female", "male"),
            registry=registry,
        )

        # Invariant: ztype_labels count matches n_ztypes
        n_ztypes = pop.state.individual_count.shape[2]
        assert len(layout.ztype_labels) == n_ztypes
        # Invariant: all labels are non-empty strings
        assert all(isinstance(lbl, str) and len(lbl) > 0 for lbl in layout.ztype_labels)
        # Invariant: contains expected genotypes for simple_species (3 alleles → 6 unordered)
        # Labels have slab suffix: "WT|WT[default]", etc.
        assert any("WT|WT" in lbl for lbl in layout.ztype_labels)
        assert any("WT|Dr" in lbl for lbl in layout.ztype_labels)

    def test_from_population_fingerprint_stable(self, simple_species) -> None:
        """Fingerprint is deterministic for the same inputs."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="fp_test", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        registry = pop.index_registry
        n_ztypes = pop.state.individual_count.shape[2]

        layout1 = PopulationLayout.from_population(
            kind="discrete_generation",
            n_demes=1,
            n_sexes=2,
            n_ages=2,
            n_ztypes=n_ztypes,
            has_sperm_storage=False,
            sex_labels=("female", "male"),
            registry=registry,
        )
        layout2 = PopulationLayout.from_population(
            kind="discrete_generation",
            n_demes=1,
            n_sexes=2,
            n_ages=2,
            n_ztypes=n_ztypes,
            has_sperm_storage=False,
            sex_labels=("female", "male"),
            registry=registry,
        )

        # Invariant: same inputs → same fingerprint
        assert layout1.fingerprint == layout2.fingerprint
        # Invariant: fingerprint is a 16-char hex string
        assert len(layout1.fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in layout1.fingerprint)

    def test_fingerprint_changes_with_labels(self, simple_species) -> None:
        """Different ztype_labels produce different fingerprints."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="fp_diff_test", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        registry = pop.index_registry
        n_ztypes = pop.state.individual_count.shape[2]

        layout_a = PopulationLayout.from_population(
            kind="discrete_generation",
            n_demes=1,
            n_sexes=2,
            n_ages=2,
            n_ztypes=n_ztypes,
            has_sperm_storage=False,
            sex_labels=("female", "male"),
            registry=registry,
        )
        layout_b = PopulationLayout.from_population(
            kind="discrete_generation",
            n_demes=1,
            n_sexes=2,
            n_ages=2,
            n_ztypes=n_ztypes,
            has_sperm_storage=False,
            sex_labels=("F", "M"),  # different sex labels
            registry=registry,
        )

        # Invariant: different labels → different fingerprint
        assert layout_a.fingerprint != layout_b.fingerprint

    def test_registry_ztype_mismatch_raises(self, simple_species) -> None:
        """ValueError when registry ztype count doesn't match n_ztypes."""
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="mismatch_test", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )

        with pytest.raises(ValueError, match="Registry ztype count"):
            PopulationLayout.from_population(
                kind="discrete_generation",
                n_demes=1,
                n_sexes=2,
                n_ages=2,
                n_ztypes=999,  # deliberately wrong
                has_sperm_storage=False,
                sex_labels=("female", "male"),
                registry=pop.index_registry,
            )


# ============================================================================
# 8. History.observe() creates correct observation-mode History
# ============================================================================


class TestHistoryObserve:
    """Invariant: observe() on raw history produces correct observation-mode History."""

    def test_observe_requires_raw_mode(self) -> None:
        """observe() raises ValueError on non-raw History."""
        schema = _obs_schema(row_size=5)
        history = History(schema)
        mask = np.ones((1, 2, 2, 6), dtype=np.float64)
        obs = _direct_observation(labels=("g0",), mask=mask, fingerprint=schema.population.fingerprint)
        with pytest.raises(ValueError, match="only valid on raw-mode"):
            history.observe(obs)

    def test_observe_produces_observation_mode(self) -> None:
        """Raw history → observe(Observation) → observation-mode History."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        raw_row_size = 1 + ind_size

        layout = _minimal_layout(n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        for tick, val in [(0, 100.0), (1, 200.0)]:
            row = np.zeros(raw_row_size, dtype=np.float64)
            row[0] = float(tick)
            ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
            ind[0, tick % n_ages, 0] = val
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        assert len(history) == 2

        mask = np.zeros((1, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        mask[0, :, :, 0] = 1.0
        labels: Tuple[str, ...] = ("observed_z0",)
        obs = _direct_observation(labels=labels, mask=mask, fingerprint=layout.fingerprint)

        obs_history = history.observe(obs)

        assert obs_history.schema.mode == "observation"
        assert obs_history.schema.observation.labels == labels
        assert obs_history.schema.observation.n_groups == 1
        assert obs_history.schema.row_size == 5

        arr = obs_history._to_numpy()
        assert arr.shape == (2, 5)
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 100.0
        assert arr[0, 2] == 0.0
        assert arr[0, 3] == 0.0
        assert arr[0, 4] == 0.0
        assert arr[1, 0] == 1.0
        assert arr[1, 1] == 0.0
        assert arr[1, 2] == 200.0
        assert arr[1, 3] == 0.0
        assert arr[1, 4] == 0.0

    def test_observe_multiple_groups(self) -> None:
        """observe() with multiple groups produces correct per-group output."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_size = n_sexes * n_ages * n_ztypes
        raw_row_size = 1 + ind_size

        layout = _minimal_layout(n_ztypes=n_ztypes)
        schema = _raw_schema(layout, row_size=raw_row_size)
        history = History(schema)

        row = np.zeros(raw_row_size, dtype=np.float64)
        row[0] = 0.0
        ind = row[1 : 1 + ind_size].reshape(n_sexes, n_ages, n_ztypes)
        ind[0, 0, 0] = 10.0
        ind[0, 0, 1] = 20.0
        ind[0, 0, 2] = 30.0
        history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        mask = np.zeros((2, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        mask[0, :, :, 0] = 1.0
        mask[1, :, :, 1] = 1.0
        mask[1, :, :, 2] = 1.0
        labels = ("g0", "g1_2")
        obs = _direct_observation(labels=labels, mask=mask, fingerprint=layout.fingerprint)

        obs_history = history.observe(obs)
        arr = obs_history._to_numpy()

        assert obs_history.schema.row_size == 9
        assert arr.shape == (1, 9)
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 10.0
        assert arr[0, 2] == 0.0
        assert arr[0, 3] == 0.0
        assert arr[0, 4] == 0.0
        assert arr[0, 5] == 50.0
        assert arr[0, 6] == 0.0
        assert arr[0, 7] == 0.0
        assert arr[0, 8] == 0.0


def _direct_observation(
    labels: Tuple[str, ...],
    mask: NDArray[np.float64],
    fingerprint: str,
) -> Observation:
    """Build an Observation directly from a pre-built mask for unit tests.

    Args:
        labels: Group labels.
        mask: Binary selection mask.
        fingerprint: Population layout fingerprint.

    Returns:
        Compiled Observation.
    """
    return Observation(
        labels=labels,
        collapse_age=False,
        mask=mask.copy(),
        population_fingerprint=fingerprint,
    )


# ============================================================================
# 9. RecordingPlan is frozen
# ============================================================================


class TestRecordingPlanFrozen:
    """Invariant: RecordingPlan cannot be mutated after construction."""

    def test_recording_plan_frozen(self) -> None:
        """Prevent mutation of a compiled recording plan."""
        schema = _raw_schema(row_size=25)
        plan = RecordingPlan(schema=schema)
        with pytest.raises(FrozenInstanceError):
            plan.schema = schema  # type: ignore[misc]  # testing frozen dataclass mutation guard

    def test_recording_plan_attributes(self) -> None:
        """Retain the schema and observation mask supplied at construction."""
        schema = _obs_schema(row_size=5)
        mask = np.ones((1, 2, 2, 6), dtype=np.float64)
        plan = RecordingPlan(schema=schema, observation_mask=mask)
        assert plan.schema is schema
        assert plan.observation_mask is mask


# ============================================================================
# 10. compile_recording_plan() row_size invariants
# ============================================================================


class TestCompileRecordingPlan:
    """Invariant: compile_recording_plan() computes correct row_size."""

    def test_raw_mode_row_size(self, simple_species) -> None:
        """Raw mode: row_size = 1 + ind_size * n_demes + sperm_size * n_demes."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="rp_raw", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        ind = pop.state.individual_count
        n_sexes, n_ages, n_ztypes = int(ind.shape[0]), int(ind.shape[1]), int(ind.shape[2])
        ind_size = n_sexes * n_ages * n_ztypes  # 2 * 2 * 6 = 24
        # has_sperm_storage=False → sperm_size = 0
        expected_raw_row_size = 1 + ind_size * 1 + 0  # 25

        plan = compile_recording_plan(
            pop,
            mode="raw",
            kind="discrete_generation",
            n_demes=1,
            has_sperm_storage=False,
            observation=pop.observation,
        )
        assert plan.schema.mode == "raw"
        assert plan.schema.row_size == expected_raw_row_size

    def test_observation_mode_row_size(self, simple_species) -> None:
        """Observation mode: row_size = 1 + n_groups * n_sexes * n_ages."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="rp_obs", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .with_observation(
                groups={"group_a": IndividualSelector(ztype="WT|WT")}
            )
            .build()
        )
        plan = compile_recording_plan(
            pop,
            mode="observation",
            kind="discrete_generation",
            n_demes=1,
            has_sperm_storage=False,
            observation=pop.observation,
        )
        assert plan.schema.mode == "observation"
        # row_size = 1 + 1 * 2 * 2 = 5
        assert plan.schema.row_size == 5
        assert plan.observation_mask is not None
        assert plan.schema.observation is not None
        assert plan.schema.observation.labels == ("group_a",)
        assert plan.schema.observation.n_groups == 1

    def test_observation_mode_row_size_multi_group(self, simple_species) -> None:
        """Row size scales with n_groups."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="rp_multi", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .with_observation(
                groups={
                    "wild": IndividualSelector(ztype="WT|WT"),
                    "heterozygous": IndividualSelector(ztype="WT|Dr"),
                    "drive": IndividualSelector(ztype="Dr|Dr"),
                }
            )
            .build()
        )
        plan = compile_recording_plan(
            pop,
            mode="observation",
            kind="discrete_generation",
            n_demes=1,
            has_sperm_storage=False,
            observation=pop.observation,
        )
        # row_size = 1 + n_groups * n_sexes * n_ages = 1 + 3 * 2 * 2 = 13
        assert plan.schema.row_size == 13

    def test_spatial_layout_created_for_multi_deme(self, simple_species) -> None:
        """Multi-deme raw recording includes SpatialHistoryLayout."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="rp_spatial", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        ind = pop.state.individual_count
        n_sexes, n_ages, n_ztypes = int(ind.shape[0]), int(ind.shape[1]), int(ind.shape[2])

        plan = compile_recording_plan(
            pop,
            mode="raw",
            kind="spatial_discrete_generation",
            n_demes=3,
            has_sperm_storage=False,
            observation=pop.observation,
        )
        layout = plan.schema.spatial_layout
        assert layout is not None
        assert layout.n_demes == 3
        assert layout.ind_per_deme == n_sexes * n_ages * n_ztypes
        assert layout.sperm_per_deme == 0  # no sperm storage
        # row_size = 1 + 3 * (n_sexes * n_ages * n_ztypes) = 1 + 3 * 24 = 73
        assert plan.schema.row_size == 1 + 3 * (n_sexes * n_ages * n_ztypes)


# ============================================================================
# 11. Observation.apply() with baked mask == apply_rule with explicit mask
# ============================================================================


class TestObservationApplyEquivalence:
    """Invariant: Observation.apply() == apply_rule() with the same mask."""

    def test_apply_equals_apply_rule(self, simple_species) -> None:
        """apply() with baked mask produces the same output as apply_rule()."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="apply_eq", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        ind_count = pop.state.individual_count
        n_sexes, n_ages, n_ztypes = (
            int(ind_count.shape[0]),
            int(ind_count.shape[1]),
            int(ind_count.shape[2]),
        )

        obs_filter = ObservationFilter(pop.index_registry)
        obs = obs_filter.build_filter(
            diploid_genotypes=pop.species,
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
        )

        result_via_apply = obs.apply(ind_count)
        mask = obs.build_mask(n_sexes, n_ages, n_ztypes)
        result_via_rule = apply_rule(ind_count, mask)

        # Invariant: both paths produce identical results
        np.testing.assert_array_equal(result_via_apply, result_via_rule)

        # Invariant: with "*" (all selected), sum over ztypes preserves total
        # For collapse_age=False: result shape = (1, 2, 2), sum = total pop
        assert result_via_apply.sum() == ind_count.sum()


# ============================================================================
# 12. Observation.build_mask() returns 4-D mask with correct shape
# ============================================================================


class TestObservationBuildMask:
    """Invariant: build_mask() returns a mask of shape (n_groups, n_sexes, n_ages, n_ztypes)."""

    def test_build_mask_shape(self, simple_species) -> None:
        """Build a four-dimensional mask with the configured axes."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="mask_shape", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        n_sexes, n_ages, n_ztypes = 2, 2, pop.state.individual_count.shape[2]

        obs_filter = ObservationFilter(pop.index_registry)
        obs = obs_filter.build_filter(
            diploid_genotypes=pop.species,
            groups={"g0": {"genotype": ["WT|WT"]}},
            collapse_age=False,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
        )

        mask = obs.build_mask(n_sexes, n_ages, n_ztypes)
        # Invariant: mask is 4-D
        assert mask.ndim == 4
        # Invariant: shape = (n_groups, n_sexes, n_ages, n_ztypes)
        assert mask.shape == (1, 2, 2, n_ztypes)
        # Invariant: mask values are only 0.0 or 1.0
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_build_mask_total_star(self, simple_species) -> None:
        """Wildcard '*' selects all ztypes."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="mask_star", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        n_sexes, n_ages, n_ztypes = 2, 2, pop.state.individual_count.shape[2]

        obs_filter = ObservationFilter(pop.index_registry)
        obs = obs_filter.build_filter(
            diploid_genotypes=pop.species,
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )

        mask = obs.build_mask(n_sexes, n_ages, n_ztypes)
        # Invariant: all entries are 1.0 (all selected)
        assert mask.sum() == pytest.approx(float(n_sexes * n_ages * n_ztypes))


# ============================================================================
# 13. Observation with collapse_age produces correct 2-D projection
# ============================================================================


class TestObservationCollapseAge:
    """Invariant: collapse_age=True produces (n_groups, n_sexes)-shaped output."""

    def test_collapse_age_output_shape(self, simple_species) -> None:
        """Remove the age axis when observation requests age collapse."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="collapse", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .with_observation(
                groups={"total": IndividualSelector()}, collapse_age=True
            )
            .build()
        )
        ind_count = pop.state.individual_count
        obs = pop.observation

        assert obs.collapse_age is True

        result = obs.apply(ind_count)
        expected = ind_count.sum(axis=(1, 2))[np.newaxis, :]
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, expected)

    def test_collapse_age_flag_preserved(self, simple_species) -> None:
        """Collapsed values equal the full projection summed over age."""
        nt.disable_numba()
        pop_collapsed = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="collapse_cmp", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .with_observation(
                groups={"total": IndividualSelector()}, collapse_age=True
            )
            .build()
        )
        pop_full = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="collapse_full", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .with_observation(
                groups={"total": IndividualSelector()}, collapse_age=False
            )
            .build()
        )

        collapsed = pop_collapsed.observe()
        full = pop_full.observe()
        assert collapsed.axes == ("group", "sex")
        assert full.axes == ("group", "sex", "age")
        np.testing.assert_array_equal(collapsed.values, full.values.sum(axis=-1))


# ============================================================================
# 14. Observation.mask being None triggers lazy rebuild
# ============================================================================


class TestObservationLazyRebuild:
    """Invariant: Observation with mask=None rebuilds on first use."""

    def test_mask_none_triggers_rebuild(self, simple_species) -> None:
        """Observation created via create_observation (mask=None) can still apply."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="lazy", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )

        obs_filter = ObservationFilter(pop.index_registry)
        obs = obs_filter.build_filter(
            diploid_genotypes=pop.species,
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )

        # Invariant: mask is None initially (created without n_ztypes)
        assert obs.mask is None

        # Invariant: apply() succeeds despite mask=None (lazy rebuild)
        result = obs.apply(pop.state.individual_count)
        assert result.ndim == 3
        assert result.shape == (1, 2, 2)
        assert result.sum() == pop.state.individual_count.sum()


# ============================================================================
# 15. Regression: .with_observation() → canonical pop.observe()
# ============================================================================


class TestWithObservationRegression:
    """Regression: with_observation installs only the canonical query rule."""

    def test_with_observation_output_history(self, simple_species) -> None:
        """Explicit Observation projects exactly while History remains raw."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species,
                name="reg_test",
                stochastic=False,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": 180.0, "WT|Dr": 20.0},
                    "male": {"WT|WT": 180.0, "WT|Dr": 20.0},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=50.0)
            .competition(
                juvenile_growth_mode="concave",
                low_density_growth_rate=6.0,
                carrying_capacity=400,
            )
            .with_observation(
                groups={
                    "drive_carriers": IndividualSelector(
                        ztype="WT|Dr", age=1
                    )
                    | IndividualSelector(ztype="Dr|Dr", age=1),
                    "wildtype": IndividualSelector(ztype="WT|WT", age=1),
                },
                collapse_age=True,
            )
            .build()
        )

        result = pop.observe()
        assert result.tick == 0
        assert result.axes == ("group", "sex")
        assert result.labels == {"group": ("drive_carriers", "wildtype")}
        np.testing.assert_array_equal(
            result.values,
            np.array([[20.0, 20.0], [180.0, 180.0]], dtype=np.float64),
        )
        assert pop.history.schema.mode == "raw"


# ============================================================================
# 16. Default and explicit Observation both keep default raw History
# ============================================================================


class TestRawVsObservationEquivalence:
    """Invariant: Observation choice is independent of default History mode."""

    def test_raw_posthoc_equals_observation_mode(self, simple_species) -> None:
        """Identity and explicit rules both leave unspecified History raw."""
        nt.disable_numba()

        pop_identity = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species,
                name="identity_raw",
                stochastic=False,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": 180.0, "WT|Dr": 20.0},
                    "male": {"WT|WT": 180.0, "WT|Dr": 20.0},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=50.0)
            .competition(
                juvenile_growth_mode="concave",
                low_density_growth_rate=6.0,
                carrying_capacity=400,
            )
            .build()
        )
        pop_explicit = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species,
                name="explicit_raw",
                stochastic=False,
            )
            .initial_state(
                individual_count={
                    "female": {"WT|WT": 180.0, "WT|Dr": 20.0},
                    "male": {"WT|WT": 180.0, "WT|Dr": 20.0},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=50.0)
            .competition(
                juvenile_growth_mode="concave",
                low_density_growth_rate=6.0,
                carrying_capacity=400,
            )
            .with_observation(
                groups={
                    "drive_carriers": IndividualSelector(
                        ztype="WT|Dr", age=1
                    )
                    | IndividualSelector(ztype="Dr|Dr", age=1),
                    "wildtype": IndividualSelector(ztype="WT|WT", age=1),
                },
                collapse_age=True,
            )
            .build()
        )

        identity_result = pop_identity.observe()
        explicit_result = pop_explicit.observe()
        np.testing.assert_array_equal(
            identity_result.values,
            np.moveaxis(pop_identity.state.individual_count, -1, 0),
        )
        np.testing.assert_array_equal(
            explicit_result.values,
            np.array([[20.0, 20.0], [180.0, 180.0]], dtype=np.float64),
        )
        assert pop_identity.history.schema.mode == "raw"
        assert pop_explicit.history.schema.mode == "raw"


# ============================================================================
# Bonus: spatial layout and schema helpers
# ============================================================================


class TestSpatialHistoryLayout:
    """Invariant: SpatialHistoryLayout stores correct per-deme sizes."""

    def test_construction(self) -> None:
        """Retain all per-deme layout dimensions."""
        layout = SpatialHistoryLayout(n_demes=5, ind_per_deme=24, sperm_per_deme=0)
        assert layout.n_demes == 5
        assert layout.ind_per_deme == 24
        assert layout.sperm_per_deme == 0


# ============================================================================
# Bonus: History.to_list and to_dict raw mode
# ============================================================================


class TestHistoryToListDict:
    """Invariant: to_list() returns correct (tick, row) pairs."""

    def test_to_list_matches_appended(self) -> None:
        """Return defensive rows matching the appended ticks and values."""
        schema = _raw_schema(row_size=5)
        history = History(schema)
        rows_data = []
        for tick in range(3):
            row = np.zeros(5, dtype=np.float64)
            row[0] = float(tick)
            row[1] = float(tick * 10 + 1)  # marker value
            rows_data.append(row.copy())
            history._append(HistoryBatch(schema=schema, rows=row[np.newaxis, :]))

        items = history._to_list()
        assert len(items) == 3
        for (tick, row), expected in zip(items, rows_data):
            assert tick == int(expected[0])
            np.testing.assert_array_equal(row, expected)

    def test_to_dict_raw_discrete(self, simple_species) -> None:
        """to_dict() on raw-mode history yields correct structure."""
        nt.disable_numba()
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="todict", stochastic=False
            )
            .initial_state(
                individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=10)
            .competition(low_density_growth_rate=2.0, carrying_capacity=2000)
            .build()
        )
        pop.run(n_steps=1, record_every=1)

        # Access History directly for to_dict testing
        history_obj = pop._history_obj  # type: ignore[attr-defined]  # accessing private attribute for test verification
        d = history_obj.to_dict(include_zero_counts=True)

        assert d["state_type"] == "DiscretePopulationState"
        assert d["n_snapshots"] >= 1
        assert len(d["snapshots"]) >= 1
        # Each snapshot should have individual_count
        snap = d["snapshots"][0]
        assert "individual_count" in snap
        # Individual count should have sex keys
        assert "female" in snap["individual_count"] or "male" in snap["individual_count"]
