"""Read-only numerical queries over native history with an immutable schema.

Rust sessions append and evict rows in HistoryStore. Python retains labels
and dimensions and exports independent arrays for individual counts, sperm
storage, and observations. HistoryBatch remains a validated import boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

from natal._engine_rs import HistoryStore

if TYPE_CHECKING:
    from natal.frontend.output.observation import Observation
    from natal.frontend.registry.index import IndexRegistry

__all__ = [
    "History",
    "HistoryBatch",
    "HistorySchema",
    "ObservationMetadata",
    "PopulationLayout",
    "SpatialHistoryLayout",
]


def _build_fingerprint(*components: object) -> str:  # object: any value with deterministic repr() — hashed via repr()
    """Build a stable SHA-256 hash fingerprint from any number of hashable values.

    Each component is converted via ``repr()`` and hashed, so any object
    with a deterministic ``repr()`` is accepted.

    Args:
        *components: Arbitrary values whose ``repr()`` will be hashed.

    Returns:
        The first 16 hex characters of the SHA-256 digest.
    """
    hasher = hashlib.sha256()
    for c in components:
        hasher.update(repr(c).encode("utf-8"))
    return hasher.hexdigest()[:16]


# ── Core data model ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PopulationLayout:
    """Stable description of population dimensions and labels.

    Attributes:
        kind: Population type identifier.
        n_demes: Number of demes (1 for panmictic).
        n_sexes: Number of sex axes.
        n_ages: Number of age classes.
        n_ztypes: Number of zygote-type entries.
        has_sperm_storage: Whether sperm storage arrays are present.
        sex_labels: Canonical sex labels.
        ztype_labels: Canonical zygote-type labels.
        fingerprint: Hash derived from layout fields.
    """

    kind: Literal[
        "age_structured",
        "discrete_generation",
        "spatial_age_structured",
        "spatial_discrete_generation",
    ]
    n_demes: int
    n_sexes: int
    n_ages: int
    n_ztypes: int
    has_sperm_storage: bool
    sex_labels: Tuple[str, ...]
    ztype_labels: Tuple[str, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Derive the immutable layout fingerprint from all schema fields."""
        object.__setattr__(
            self,
            "fingerprint",
            _build_fingerprint(
                self.kind,
                self.n_demes,
                self.n_sexes,
                self.n_ages,
                self.n_ztypes,
                self.has_sperm_storage,
                self.sex_labels,
                self.ztype_labels,
            ),
        )

    @classmethod
    def from_population(
        cls,
        *,
        kind: str,
        n_demes: int,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        has_sperm_storage: bool,
        sex_labels: Sequence[str],
        registry: IndexRegistry,
    ) -> PopulationLayout:
        """Build a layout from population metadata.

        Args:
            kind: One of ``"age_structured"``, ``"discrete_generation"``,
                ``"spatial_age_structured"``, ``"spatial_discrete_generation"``.
            n_demes: Number of demes.
            n_sexes: Number of sex axes.
            n_ages: Number of age classes.
            n_ztypes: Number of zygote-type entries.
            has_sperm_storage: Whether sperm storage arrays exist.
            sex_labels: Sex axis labels.
            registry: Index registry for deriving ztype labels.

        Returns:
            A frozen ``PopulationLayout``.
        """
        ztype_labels = tuple(
            f"{str(gt)}[{slab}]" if slab else str(gt)
            for gt, slab in registry.index_to_ztype
        )
        if len(ztype_labels) != n_ztypes:
            raise ValueError(
                f"Registry ztype count ({len(ztype_labels)}) does not match "
                f"n_ztypes ({n_ztypes})"
            )
        return cls(
            kind=kind,  # type: ignore[arg-type]  # kind validated by callers
            n_demes=n_demes,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            has_sperm_storage=has_sperm_storage,
            sex_labels=tuple(sex_labels),
            ztype_labels=ztype_labels,
        )


@dataclass(frozen=True)
class SpatialHistoryLayout:
    """Per-deme layout parameters for spatial raw history rows.

    Instances are derived from the co-located
    :class:`PopulationLayout`; the per-deme sizes are pure functions of
    its dimensions and are never stored independently.

    Attributes:
        n_demes: Total number of demes.
        ind_per_deme: Float64 values per deme for individual counts.
        sperm_per_deme: Float64 values per deme for sperm storage
            (0 for discrete-generation populations).
    """

    n_demes: int
    ind_per_deme: int
    sperm_per_deme: int

    @classmethod
    def from_population(cls, population: PopulationLayout) -> SpatialHistoryLayout:
        """Derive the per-deme row layout from a spatial population layout.

        Args:
            population: Population layout whose ``kind`` is spatial.

        Returns:
            A derived ``SpatialHistoryLayout``.
        """
        sperm_per_deme = (
            population.n_ages * population.n_ztypes * population.n_ztypes
            if population.has_sperm_storage
            else 0
        )
        return cls(
            n_demes=population.n_demes,
            ind_per_deme=population.n_sexes * population.n_ages * population.n_ztypes,
            sperm_per_deme=sperm_per_deme,
        )


@dataclass(frozen=True)
class ObservationMetadata:
    """Metadata describing an observation-mode recording.

    Attributes:
        labels: Observation group labels.
        collapse_age: Whether the age axis was collapsed.
        deme_indices: Ordered selected spatial demes, or ``None`` for
            non-spatial observations.
        deme_mode: Whether a spatial observation preserves or aggregates the
            selected deme axis.
    """

    labels: Tuple[str, ...]
    collapse_age: bool
    deme_indices: Optional[Tuple[int, ...]] = None
    deme_mode: Literal["preserve", "aggregate"] = "preserve"

    @property
    def n_groups(self) -> int:
        """Number of observation groups (derived from the labels)."""
        return len(self.labels)


@dataclass(frozen=True)
class HistorySchema:
    """Immutable description of the history recording format.

    Attributes:
        mode: ``"raw"`` for full-state, ``"observation"`` for aggregates.
        population: Population layout defining array dimensions.
        row_size: Float64 values per history row (including tick).
        observation: Observation metadata (``None`` for raw mode).
    """

    mode: Literal["raw", "observation"]
    population: PopulationLayout
    row_size: int
    observation: Optional[ObservationMetadata] = None

    def __post_init__(self) -> None:
        """Validate that the storage mode and metadata agree.

        Raises:
            ValueError: If observation metadata contradicts the mode or the
                row width is not positive.
        """
        if self.mode == "observation" and self.observation is None:
            raise ValueError("observation-mode schema requires ObservationMetadata")
        if self.mode == "raw" and self.observation is not None:
            raise ValueError("raw-mode schema must not have ObservationMetadata")
        if self.row_size <= 0:
            raise ValueError(f"row_size must be positive, got {self.row_size}")

    @property
    def spatial_layout(self) -> Optional[SpatialHistoryLayout]:
        """Per-deme layout for spatial raw rows, derived from the population.

        ``None`` for panmictic layouts and for observation-mode schemas —
        observation rows are never spatial raw rows regardless of the
        population kind.
        """
        if self.mode == "raw" and self.population.kind.startswith("spatial_"):
            return SpatialHistoryLayout.from_population(self.population)
        return None


@dataclass(frozen=True)
class HistoryBatch:
    """Validated batch of history rows.

    Produced by engine wrappers, consumed by :class:`History`.

    Attributes:
        schema: The schema all rows conform to.
        rows: 2-D float64 array ``(n_rows, row_size)`` with tick in col 0.
    """

    schema: HistorySchema
    rows: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the dimensionality and width of the batch rows.

        Raises:
            ValueError: If rows are not two-dimensional or do not match the
                schema width.
        """
        if self.rows.ndim != 2:
            raise ValueError(f"rows must be 2-D, got {self.rows.ndim}-D")
        if self.rows.shape[1] != self.schema.row_size:
            raise ValueError(
                f"Batch row width {self.rows.shape[1]} does not match "
                f"schema row_size {self.schema.row_size}"
            )


# ── History container ────────────────────────────────────────────────────────


class History:
    """Validated, schema-annotated time-series storage.

    Each Population owns exactly one :class:`History`.  The schema is
    fixed at construction time; :meth:`clear` removes rows but preserves
    the schema.

    When *max_rows* is set, the oldest records are evicted FIFO when
    capacity is exceeded.

    Attributes:
        schema: The immutable :class:`HistorySchema`.
        max_rows: Native FIFO capacity or ``None`` for unlimited.
    """

    def __init__(
        self,
        schema: HistorySchema,
        *,
        max_rows: Optional[int] = None,
    ) -> None:
        """Initialize empty history storage for an immutable schema.

        Args:
            schema: Schema shared by every stored batch.
            max_rows: Optional positive FIFO capacity.

        Raises:
            ValueError: If ``max_rows`` is less than one.
        """
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"max_rows must be >= 1 or None, got {max_rows}")
        self._schema = schema
        pop = schema.population
        self._checkpoint_pruner: Callable[[int], None] | None = None
        self._store = HistoryStore(
            schema.row_size,
            (pop.n_demes, pop.n_sexes, pop.n_ages, pop.n_ztypes),
            schema.mode == "raw", max_rows,
        )

    @property
    def boundary_metadata(self) -> Tuple[Tuple[int, int, str], ...]:
        """Return tick, phase cursor, and lifecycle status for each row.

        Normal complete boundaries have phase zero and status Ready. A manual
        snapshot after a mid-tick stop preserves its partial phase and status.

        Returns:
            An immutable tuple of the retained record boundaries.
        """
        return tuple(self._store.boundaries())

    def _bind_checkpoint_pruner(self, prune: Callable[[int], None]) -> None:
        """Pair capacity changes with immediate native checkpoint deallocation."""
        self._checkpoint_pruner = prune

    @property
    def max_rows(self) -> Optional[int]:
        """Native FIFO capacity; None means unlimited retention."""
        return self._store.max_rows

    @max_rows.setter
    def max_rows(self, value: Optional[int]) -> None:
        """Update the native retention limit."""
        self._store.max_rows = value
        ticks = self.ticks
        if ticks and self._checkpoint_pruner is not None:
            self._checkpoint_pruner(ticks[0])

    @property
    def _rows(self) -> List[NDArray[np.float64]]:
        """Materialize an export snapshot; never store runtime data here."""
        return list(self._store.query())

    def _configure_observation(self, observation: Observation) -> None:
        """Install a compiled selector in native storage.

        Args:
            observation: Canonical rule matching the history layout.
        """
        pop = self._schema.population
        mask = observation.build_mask(pop.n_sexes, pop.n_ages, pop.n_ztypes)
        selected = observation.deme_indices
        self._store.configure_observation(
            np.ascontiguousarray(mask, dtype=np.float64).ravel(),
            list(selected) if selected is not None else [0],
            observation.collapse_age, observation.deme_mode == "aggregate",
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def schema(self) -> HistorySchema:
        """The immutable schema describing all stored rows."""
        return self._schema

    @property
    def n_records(self) -> int:
        """Number of records retained by the native store."""
        return len(self._store)

    @property
    def is_empty(self) -> bool:
        """Whether native history contains no rows."""
        return len(self._store) == 0

    @property
    def ticks(self) -> Tuple[int, ...]:
        """Return retained tick metadata without copying numerical rows."""
        return tuple(self._store.ticks())

    @property
    def axes(self) -> Tuple[str, ...]:
        """Axis names for the primary typed array exposed by this History."""
        if self._schema.mode == "raw":
            axes: Tuple[str, ...] = ("record",)
            if self._schema.spatial_layout is not None:
                axes += ("deme",)
            return axes + ("sex", "age", "ztype")

        observation = self._schema.observation
        assert observation is not None
        axes = ("record", "group")
        if (
            observation.deme_indices is not None
            and observation.deme_mode == "preserve"
        ):
            axes += ("deme",)
        axes += ("sex",)
        if not observation.collapse_age:
            axes += ("age",)
        return axes

    @property
    def individual_count(self) -> NDArray[np.float64]:
        """Return an independent read-only raw count tensor.

        Raises:
            ValueError: If the history contains observation values.
        """
        if self._schema.mode != "raw":
            raise ValueError("individual_count is only available in raw mode")
        pop = self._schema.population
        size = pop.n_demes * pop.n_sexes * pop.n_ages * pop.n_ztypes
        shape = (len(self),)
        if self._schema.spatial_layout is not None:
            shape += (pop.n_demes,)
        shape += (pop.n_sexes, pop.n_ages, pop.n_ztypes)
        result = self._store.query(1, 1 + size).reshape(shape)
        result.flags.writeable = False
        return result

    @property
    def sperm_storage(self) -> Optional[NDArray[np.float64]]:
        """Return an independent read-only sperm tensor when recorded.

        Raises:
            ValueError: If the history contains observation values.
        """
        if self._schema.mode != "raw":
            raise ValueError("sperm_storage is only available in raw mode")
        pop = self._schema.population
        if not pop.has_sperm_storage:
            return None
        start = 1 + pop.n_demes * pop.n_sexes * pop.n_ages * pop.n_ztypes
        shape = (len(self),)
        if self._schema.spatial_layout is not None:
            shape += (pop.n_demes,)
        shape += (pop.n_ages, pop.n_ztypes, pop.n_ztypes)
        result = self._store.query(start).reshape(shape)
        result.flags.writeable = False
        return result

    @property
    def values(self) -> NDArray[np.float64]:
        """Return independent read-only observation values from native storage.

        Raises:
            ValueError: If raw counts were recorded instead.
        """
        if self._schema.mode != "observation":
            raise ValueError("values is only available in observation mode")
        pop = self._schema.population
        observation = self._schema.observation
        assert observation is not None
        shape = (len(self), observation.n_groups)
        if observation.deme_indices is not None and observation.deme_mode == "preserve":
            shape += (len(observation.deme_indices),)
        shape += (pop.n_sexes,)
        if not observation.collapse_age:
            shape += (pop.n_ages,)
        result = self._store.query(1).reshape(shape)
        result.flags.writeable = False
        return result

    # ── Mutations ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of retained native rows."""
        return len(self._store)

    def __iter__(self) -> Iterator[Tuple[int, NDArray[np.float64]]]:  # explicit Iterator for typed iteration
        """Iterate over defensive copies paired with their ticks."""
        return iter(self._to_list())



    def _append(self, batch: HistoryBatch) -> None:
        """Validate and import a batch into the native ring atomically.

        Args:
            batch: Numerical rows with the same frozen schema.

        Raises:
            ValueError: If schemas or row ordering do not agree.
        """
        if batch.schema != self._schema:
            raise ValueError("Batch schema does not match History schema")
        self._store.append(batch.rows, False)

    def _append_continuation(self, batch: HistoryBatch) -> None:
        """Import a batch allowing one identical overlapping boundary.

        Args:
            batch: Numerical rows with the same frozen schema.

        Raises:
            ValueError: If schemas, ordering, or the overlap disagree.
        """
        if batch.schema != self._schema:
            raise ValueError("Batch schema does not match History schema")
        self._store.append(batch.rows, True)

    def clear(self) -> None:
        """Clear the native history without changing the current state."""
        self._store.clear()

    def truncate(self, *, retain_until_tick: int) -> None:
        """Remove future native rows.

        Args:
            retain_until_tick: Inclusive latest retained tick.

        Raises:
            ValueError: If no retained row precedes this boundary.
        """
        self._store.truncate(retain_until_tick)

    def restore_state(
        self, tick: int
    ) -> Tuple[int, NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Return ``(tick, individual_count, sperm_storage)`` at *tick*.

        Only valid for raw-mode history.

        Args:
            tick: Exact tick to restore.

        Returns:
            ``(tick, individual_count, sperm_storage)`` tuple.
            *sperm_storage* is ``None`` when not present.

        Raises:
            ValueError: If mode is not ``"raw"`` or tick is not found.
        """
        if self._schema.mode != "raw":
            raise ValueError("Cannot restore state from observation-mode history.")
        pop = self._schema.population
        is_spatial = self._schema.spatial_layout is not None
        ind_per_deme = pop.n_sexes * pop.n_ages * pop.n_ztypes
        ind_size = ind_per_deme * pop.n_demes
        ind_shape = (
            (pop.n_demes, pop.n_sexes, pop.n_ages, pop.n_ztypes)
            if is_spatial
            else (pop.n_sexes, pop.n_ages, pop.n_ztypes)
        )

        for row in (self._store.row(tick),):
            if int(row[0]) == tick:
                ic = row[1 : 1 + ind_size].reshape(ind_shape).copy()
                ss: Optional[NDArray[np.float64]] = None
                if pop.has_sperm_storage:
                    n_ztypes = pop.n_ztypes
                    sperm_per_deme = pop.n_ages * n_ztypes * n_ztypes
                    sperm_size = sperm_per_deme * pop.n_demes
                    sperm_shape = (
                        (pop.n_demes, pop.n_ages, n_ztypes, n_ztypes)
                        if is_spatial
                        else (pop.n_ages, n_ztypes, n_ztypes)
                    )
                    ss = row[1 + ind_size : 1 + ind_size + sperm_size].reshape(
                        sperm_shape
                    ).copy()
                return (tick, ic, ss)

        raise ValueError(f"Tick {tick} not found in history.")

    # ── Post-hoc observation ──────────────────────────────────────────────

    def observe(self, observation: Observation) -> History:
        """Create a new observation-mode History from raw history.

        Projects every stored raw record through *observation*.

        Only valid when the current mode is ``"raw"``.

        Args:
            observation: Compiled :class:`Observation` whose layout fingerprint
                must match this History's population layout.

        Returns:
            New ``History`` in observation mode.

        Raises:
            ValueError: If mode is not ``"raw"`` or the Observation layout
                fingerprint does not match this History.
        """
        if self._schema.mode != "raw":
            raise ValueError(
                "observe() is only valid on raw-mode History."
            )

        expected_fingerprint = self._schema.population.fingerprint
        if observation.population_fingerprint != expected_fingerprint:
            raise ValueError(
                "Observation population layout does not match History: "
                f"expected {expected_fingerprint}, got "
                f"{observation.population_fingerprint or '<unset>'}."
            )
        return self._observe_from_observation(observation)

    def _observe_from_observation(self, observation: Observation) -> History:
        """Project raw records through a layout-validated Observation.

        Args:
            observation: Canonical projection rule for this History layout.

        Returns:
            An independent observation-mode History.
        """
        pop = self._schema.population
        n_sexes = pop.n_sexes
        n_ages = pop.n_ages
        n_groups = observation.n_groups
        age_width = 1 if observation.collapse_age else n_ages
        selected_demes = (
            len(observation.deme_indices)
            if observation.deme_indices is not None
            and observation.deme_mode == "preserve"
            else 1
        )
        obs_row_size = 1 + n_groups * selected_demes * n_sexes * age_width

        obs_meta = ObservationMetadata(
            labels=observation.labels,
            collapse_age=observation.collapse_age,
            deme_indices=observation.deme_indices,
            deme_mode=observation.deme_mode,
        )
        obs_schema = HistorySchema(
            mode="observation",
            population=pop,
            row_size=obs_row_size,
            observation=obs_meta,
        )
        obs_history = History(obs_schema)

        obs_history._configure_observation(observation)
        self._store.observe(obs_history._store)

        return obs_history

    def _to_numpy(self) -> NDArray[np.float64]:
        """Return all history rows as a single 2-D array.

        Returns:
            2-D float64 array ``(n_records, row_size)``.
            Returns ``(0, row_size)`` when empty.
        """
        return self._store.query()

    def _to_list(self) -> List[Tuple[int, NDArray[np.float64]]]:
        """Return history rows as ``(tick, flat_row)`` pairs.

        Returns:
            List of ``(tick, row)`` tuples. Each row is a defensive copy.
        """
        return [(int(row[0]), row.copy()) for row in self._rows]

    def to_dict(self, *, include_zero_counts: bool = False) -> Dict[str, Any]:  # Any: JSON-serializable nested dict  # Any: nested dicts with mixed value types (str, int, float, list, dict)
        """Translate history to a readable dictionary.

        Args:
            include_zero_counts: Whether to include zero-valued entries.

        Returns:
            Nested dict with metadata and per-snapshot entries.
        """
        mode = self._schema.mode
        population = self._schema.population
        snapshots: List[Dict[str, Any]] = []  # Any: JSON-serializable nested dict

        if mode == "raw":
            spatial = self._schema.spatial_layout
            if spatial is not None:
                # spatial raw: rows are [tick, ind_all_demes.ravel(), sperm_all_demes.ravel()]
                ind_per_deme = spatial.ind_per_deme
                sperm_per_deme = spatial.sperm_per_deme
                n_demes = spatial.n_demes
                ind_size = ind_per_deme * n_demes
                for row in self._rows:
                    tick = int(row[0])
                    record: Dict[str, Any] = {"tick": tick}
                    per_deme: list[list[float]] = []
                    payload = row[1:]
                    for di in range(n_demes):
                        start = di * ind_per_deme
                        end = start + ind_per_deme
                        per_deme.append(payload[start:end].tolist())
                    record["individual_count_per_deme"] = per_deme
                    if sperm_per_deme > 0:
                        sperm_start = ind_size
                        sperm_payload = payload[sperm_start:]
                        per_deme_sperm: list[list[float]] = []
                        for di in range(n_demes):
                            s = di * sperm_per_deme
                            e = s + sperm_per_deme
                            per_deme_sperm.append(sperm_payload[s:e].tolist())
                        record["sperm_storage_per_deme"] = per_deme_sperm
                    snapshots.append(record)
                return {
                    "state_type": "SpatialPopulation",
                    "n_snapshots": len(snapshots),
                    "snapshots": snapshots,
                }

            from natal.frontend.data import (
                DiscretePopulationState,
                PopulationState,
                parse_flattened_discrete_state,
                parse_flattened_state,
            )

            n_sexes = population.n_sexes
            n_ages = population.n_ages
            n_ztypes = population.n_ztypes
            genotype_labels = list(population.ztype_labels)
            sex_labels = list(population.sex_labels)

            parse_fn = (
                parse_flattened_discrete_state
                if population.kind == "discrete_generation"
                else parse_flattened_state
            )
            state_type = (
                DiscretePopulationState
                if population.kind == "discrete_generation"
                else PopulationState
            )

            for row in self._rows:
                tick = int(row[0])
                parsed = parse_fn(
                    row,
                    n_sexes=n_sexes,
                    n_ages=n_ages,
                    n_ztypes=n_ztypes,
                    copy=True,
                )
                snapshots.append(
                    _state_to_dict(
                        state=parsed,
                        sex_labels=sex_labels,
                        genotype_labels=genotype_labels,
                        include_zero_counts=include_zero_counts,
                    )
                )

            return {
                "state_type": state_type.__name__,
                "n_snapshots": len(snapshots),
                "snapshots": snapshots,
            }

        assert self._schema.observation is not None
        obs_meta = self._schema.observation
        labels = list(obs_meta.labels)
        n_demes = len(obs_meta.deme_indices) if obs_meta.deme_indices else 0

        for row in self._rows:
            tick = int(row[0])
            payload = row[1:]
            if n_demes > 0 and obs_meta.deme_mode == "preserve":
                if obs_meta.collapse_age:
                    observed = payload.reshape(
                        obs_meta.n_groups, n_demes, population.n_sexes,
                    )
                    obs_axes: Tuple[str, ...] = ("group", "deme", "sex")
                else:
                    observed = payload.reshape(
                        obs_meta.n_groups, n_demes, population.n_sexes,
                        population.n_ages,
                    )
                    obs_axes = ("group", "deme", "sex", "age")
            elif n_demes > 0 and obs_meta.deme_mode == "aggregate":
                if obs_meta.collapse_age:
                    observed = payload.reshape(
                        obs_meta.n_groups, population.n_sexes,
                    )
                    obs_axes = ("group", "sex")
                else:
                    observed = payload.reshape(
                        obs_meta.n_groups, population.n_sexes, population.n_ages,
                    )
                    obs_axes = ("group", "sex", "age")
            elif obs_meta.collapse_age:
                observed = payload.reshape(
                    obs_meta.n_groups, population.n_sexes,
                )
                obs_axes = ("group", "sex")
            else:
                observed = payload.reshape(
                    obs_meta.n_groups, population.n_sexes, population.n_ages,
                )
                obs_axes = ("group", "sex", "age")
            snapshots.append(
                {
                    "tick": tick,
                    "labels": labels,
                    "collapse_age": obs_meta.collapse_age,
                    "observed": _build_observation_payload(
                        observed=observed,
                        labels=labels,
                        sex_labels=list(population.sex_labels),
                        include_zero_counts=include_zero_counts,
                        axes=obs_axes,
                    ),
                }
            )

        return {
            "state_type": "ObservedHistory",
            "n_snapshots": len(snapshots),
            "labels": labels,
            "collapse_age": obs_meta.collapse_age,
            "snapshots": snapshots,
        }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _state_to_dict(
    state: Any,  # Any: accepts PopulationState or DiscretePopulationState
    sex_labels: List[str],
    genotype_labels: List[str],
    include_zero_counts: bool,
) -> Dict[str, Any]:  # Any: JSON-serializable nested dict
    """Serialize one population state to the legacy nested dictionary shape.

    Args:
        state: Age-structured or discrete population state.
        sex_labels: Labels corresponding to the state sex axis.
        genotype_labels: Labels corresponding to the state ZType axis.
        include_zero_counts: Whether zero-valued entries are retained.

    Returns:
        JSON-serializable state data keyed by tick and count category.
    """
    from natal.frontend.data import PopulationState

    n_ages = int(state.individual_count.shape[1])
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sex_idx, sex_name in enumerate(sex_labels):
        sex_block: Dict[str, Dict[str, float]] = {}
        for age_idx in range(n_ages):
            age_key = f"age_{age_idx}"
            geno_block: Dict[str, float] = {}
            for ztype_idx, genotype_name in enumerate(genotype_labels):
                value = float(
                    state.individual_count[sex_idx, age_idx, ztype_idx]
                )
                if include_zero_counts or value != 0.0:
                    geno_block[genotype_name] = value
            if include_zero_counts or geno_block:
                sex_block[age_key] = geno_block
        payload[sex_name] = sex_block

    result: Dict[str, Any] = {"tick": int(state.n_tick)}  # Any: JSON-serializable nested dict
    result["individual_count"] = payload

    sperm_storage = getattr(state, "sperm_storage", None)
    if isinstance(state, PopulationState) and sperm_storage is not None:
        sperm_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        for age_idx in range(n_ages):
            age_key = f"age_{age_idx}"
            female_block: Dict[str, Dict[str, float]] = {}
            for female_idx, female_name in enumerate(genotype_labels):
                male_block: Dict[str, float] = {}
                for male_idx, male_name in enumerate(genotype_labels):
                    value = float(
                        state.sperm_storage[age_idx, female_idx, male_idx]
                    )
                    if include_zero_counts or value != 0.0:
                        male_block[male_name] = value
                if include_zero_counts or male_block:
                    female_block[female_name] = male_block
            if include_zero_counts or female_block:
                sperm_payload[age_key] = female_block
        result["sperm_storage"] = sperm_payload

    return result


def _build_observation_payload(
    observed: np.ndarray,
    labels: List[str],
    sex_labels: List[str],
    include_zero_counts: bool,
    axes: Tuple[str, ...],
) -> Dict[str, Any]:  # Any: JSON-serializable nested dict  # Any: nested dicts with mixed value types (str, int, float, list, dict)
    """Build a nested-dict payload from an observed array with known axes.

    Args:
        observed: Observed ndarray.
        labels: Group labels for the first axis.
        sex_labels: Sex labels.
        include_zero_counts: Whether to include zero values.
        axes: Explicit axis names. Supported: ``("group", "sex")``,
            ``("group", "sex", "age")``,
            ``("group", "deme", "sex")``, ``("group", "deme", "sex", "age")``.
    """
    n_groups = len(labels)
    payload: Dict[str, Any] = {}  # Any: JSON-serializable nested dict
    for group_index in range(n_groups):
        group_payload: Any = _serialize_with_axes(  # Any: recursive JSON value
            observed[group_index],
            axes[1:],
            sex_labels,
            include_zero_counts,
        )
        payload[labels[group_index]] = group_payload
    return payload


def _serialize_with_axes(
    arr: np.ndarray,
    axes: Tuple[str, ...],
    sex_labels: List[str],
    include_zero_counts: bool,
) -> Any:  # Any: recursive JSON-serializable dict/list/float
    """Recursively serialize an ndarray using explicit axis names.

    Args:
        arr: Sub-array to serialize.
        axes: Remaining axis names from outer to inner.
        sex_labels: Sex labels used when the current axis is ``"sex"``.
        include_zero_counts: Whether to keep zero-valued entries.

    Returns:
        Nested dict, or float scalar at leaf.
    """
    if len(axes) == 0:
        return float(arr)
    axis = axes[0]
    rest = axes[1:]
    if axis == "sex":
        result: Dict[str, Any] = {}  # Any: JSON-serializable nested dict
        for si, sex_name in enumerate(sex_labels):
            value: Any = _serialize_with_axes(arr[si], rest, sex_labels, include_zero_counts)  # Any: recursive JSON-serializable
            if include_zero_counts or (isinstance(value, (int, float)) and value != 0.0) or (isinstance(value, dict) and value):
                result[sex_name] = value
        return result
    if axis == "age":
        n_ages = int(arr.shape[0])
        result = {}
        for ai in range(n_ages):
            value = _serialize_with_axes(arr[ai], rest, sex_labels, include_zero_counts)
            if include_zero_counts or (isinstance(value, (int, float)) and value != 0.0) or (isinstance(value, dict) and value):
                result[f"age_{ai}"] = value
        return result
    if axis == "deme":
        n_demes = int(arr.shape[0])
        result = {}
        for di in range(n_demes):
            value = _serialize_with_axes(arr[di], rest, sex_labels, include_zero_counts)
            if include_zero_counts or (isinstance(value, (int, float)) and value != 0.0) or (isinstance(value, dict) and value):
                result[f"deme_{di}"] = value
        return result
    return float(arr)
