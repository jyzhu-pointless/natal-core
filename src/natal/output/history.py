"""Self-describing history storage with immutable schema.

Each Population owns exactly one :class:`History` instance whose
:class:`HistorySchema` is fixed at construction time.  Engine wrappers
produce numerical :class:`HistoryBatch` rows that are validated and
stored by the History layer.

The history stores *flat* float64 rows internally for engine
compatibility while exposing typed array properties (``individual_count``,
``sperm_storage``, ``values``) that are recomputed lazily and cached.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.output.observation import Observation
    from natal.registry.index import IndexRegistry

__all__ = [
    "History",
    "HistoryBatch",
    "HistorySchema",
    "ObservationMetadata",
    "PopulationLayout",
    "SpatialHistoryLayout",
]


def _build_fingerprint(*components: object) -> str:
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

    Attributes:
        n_demes: Total number of demes.
        ind_per_deme: Float64 values per deme for individual counts.
        sperm_per_deme: Float64 values per deme for sperm storage
            (0 for discrete-generation populations).
    """

    n_demes: int
    ind_per_deme: int
    sperm_per_deme: int


@dataclass(frozen=True)
class ObservationMetadata:
    """Metadata describing an observation-mode recording.

    Attributes:
        labels: Observation group labels.
        collapse_age: Whether the age axis was collapsed.
        n_groups: Number of observation groups.
    """

    labels: Tuple[str, ...]
    collapse_age: bool
    n_groups: int


@dataclass(frozen=True)
class HistorySchema:
    """Immutable description of the history recording format.

    Attributes:
        mode: ``"raw"`` for full-state, ``"observation"`` for aggregates.
        population: Population layout defining array dimensions.
        row_size: Float64 values per history row (including tick).
        observation: Observation metadata (``None`` for raw mode).
        spatial_layout: Spatial layout parameters (``None`` for panmictic).
    """

    mode: Literal["raw", "observation"]
    population: PopulationLayout
    row_size: int
    observation: Optional[ObservationMetadata] = None
    spatial_layout: Optional[SpatialHistoryLayout] = None

    def __post_init__(self) -> None:
        if self.mode == "observation" and self.observation is None:
            raise ValueError("observation-mode schema requires ObservationMetadata")
        if self.mode == "raw" and self.observation is not None:
            raise ValueError("raw-mode schema must not have ObservationMetadata")
        if self.row_size <= 0:
            raise ValueError(f"row_size must be positive, got {self.row_size}")


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
        max_rows: Soft capacity or ``None`` for unlimited.
    """

    def __init__(
        self,
        schema: HistorySchema,
        *,
        max_rows: Optional[int] = None,
    ) -> None:
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"max_rows must be >= 1 or None, got {max_rows}")
        self._schema = schema
        self.max_rows: Optional[int] = max_rows
        self._rows: List[NDArray[np.float64]] = []
        self._seen_ticks: set[int] = set()
        # cached views — invalidated on any mutation
        self._cache_individual_count: Optional[NDArray[np.float64]] = None
        self._cache_sperm_storage: Optional[NDArray[np.float64]] = None
        self._cache_values: Optional[NDArray[np.float64]] = None
        self._cache_ticks: Optional[Tuple[int, ...]] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def schema(self) -> HistorySchema:
        """The immutable schema describing all stored rows."""
        return self._schema

    @property
    def n_records(self) -> int:
        """Number of stored history rows."""
        return len(self._rows)

    @property
    def is_empty(self) -> bool:
        """Whether the history has no rows."""
        return len(self._rows) == 0

    @property
    def ticks(self) -> Tuple[int, ...]:
        """Sorted tuple of all recorded ticks."""
        if self._cache_ticks is not None:
            return self._cache_ticks
        ticks = tuple(sorted(int(r[0]) for r in self._rows))
        self._cache_ticks = ticks
        return ticks

    @property
    def individual_count(self) -> NDArray[np.float64]:
        """Raw individual-count array ``(record, n_sexes, n_ages, n_ztypes)``.

        Returns a cached read-only view of the full count tensor for
        each recorded tick.  Non-spatial models omit the deme axis.
        Only valid when ``schema.mode == "raw"``.

        Raises:
            ValueError: If the schema mode is not ``"raw"``.
        """
        if self._schema.mode != "raw":
            raise ValueError("individual_count is only available in raw mode")
        if self._cache_individual_count is not None:
            return self._cache_individual_count
        pop = self._schema.population
        n_records = len(self._rows)
        arr = np.zeros(
            (n_records, pop.n_sexes, pop.n_ages, pop.n_ztypes),
            dtype=np.float64,
        )
        ind_size = pop.n_sexes * pop.n_ages * pop.n_ztypes
        for ri, row in enumerate(self._rows):
            arr[ri] = row[1 : 1 + ind_size].reshape(
                pop.n_sexes, pop.n_ages, pop.n_ztypes
            )
        arr.flags.writeable = False
        self._cache_individual_count = arr
        return arr

    @property
    def sperm_storage(self) -> Optional[NDArray[np.float64]]:
        """Raw sperm-storage array ``(record, n_ages, n_female, n_male)``.

        ``None`` for discrete-generation populations.  Only valid when
        ``schema.mode == "raw"``.
        """
        if self._schema.mode != "raw":
            raise ValueError("sperm_storage is only available in raw mode")
        if not self._schema.population.has_sperm_storage:
            return None
        if self._cache_sperm_storage is not None:
            return self._cache_sperm_storage
        pop = self._schema.population
        n_ztypes = pop.n_ztypes
        n_records = len(self._rows)
        arr = np.zeros(
            (n_records, pop.n_ages, n_ztypes, n_ztypes),
            dtype=np.float64,
        )
        ind_size = pop.n_sexes * pop.n_ages * pop.n_ztypes
        sperm_size = pop.n_ages * n_ztypes * n_ztypes
        for ri, row in enumerate(self._rows):
            arr[ri] = row[1 + ind_size : 1 + ind_size + sperm_size].reshape(
                pop.n_ages, n_ztypes, n_ztypes
            )
        arr.flags.writeable = False
        self._cache_sperm_storage = arr
        return arr

    @property
    def values(self) -> NDArray[np.float64]:
        """Observed values ``(record, n_groups, n_sexes[, n_ages])``.

        Only valid when ``schema.mode == "observation"``.
        """
        if self._schema.mode != "observation":
            raise ValueError("values is only available in observation mode")
        if self._cache_values is not None:
            return self._cache_values
        pop = self._schema.population
        obs_meta = self._schema.observation
        assert obs_meta is not None
        n_records = len(self._rows)
        n_groups = obs_meta.n_groups
        n_sexes = pop.n_sexes
        n_ages = pop.n_ages
        arr = np.zeros(
            (n_records, n_groups, n_sexes, n_ages), dtype=np.float64
        )
        for ri, row in enumerate(self._rows):
            arr[ri] = row[1:].reshape(n_groups, n_sexes, n_ages)
        arr.flags.writeable = False
        self._cache_values = arr
        return arr

    # ── Mutations ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> object:
        return iter(self.to_list())

    def _invalidate_cache(self) -> None:
        self._cache_individual_count = None
        self._cache_sperm_storage = None
        self._cache_values = None
        self._cache_ticks = None

    def _evict_if_needed(self) -> None:
        if self.max_rows is None:
            return
        while len(self._rows) > self.max_rows:
            evicted = self._rows.pop(0)
            self._seen_ticks.discard(int(evicted[0]))
            self._invalidate_cache()

    def append(self, batch: HistoryBatch) -> None:
        """Append rows from a validated batch.

        Args:
            batch: Batch of rows; schema must match.

        Raises:
            ValueError: If schemas mismatch or a tick is duplicated.
        """
        if batch.schema != self._schema:
            raise ValueError("Batch schema does not match History schema")
        if batch.rows.shape[0] == 0:
            return
        for ri in range(batch.rows.shape[0]):
            row = batch.rows[ri, :].copy()
            tick = int(row[0])
            if tick in self._seen_ticks:
                continue
            self._seen_ticks.add(tick)
            self._rows.append(row)
        self._invalidate_cache()
        self._evict_if_needed()

    def clear(self) -> None:
        """Remove all stored rows while preserving the schema."""
        self._rows.clear()
        self._seen_ticks.clear()
        self._invalidate_cache()

    def truncate(self, *, retain_until_tick: int) -> None:
        """Remove all rows with tick > *retain_until_tick*.

        Args:
            retain_until_tick: Inclusive upper bound — rows with tick
                greater than this value are removed.

        Raises:
            ValueError: If no record exists at or below the target tick.
        """
        new_rows = [r for r in self._rows if int(r[0]) <= retain_until_tick]
        if not new_rows:
            raise ValueError(
                f"No records with tick <= {retain_until_tick} exist."
            )
        self._rows = new_rows
        self._seen_ticks = {int(r[0]) for r in self._rows}
        self._invalidate_cache()

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
        ind_size = pop.n_sexes * pop.n_ages * pop.n_ztypes

        for row in self._rows:
            if int(row[0]) == tick:
                ic = row[1 : 1 + ind_size].reshape(
                    pop.n_sexes, pop.n_ages, pop.n_ztypes
                ).copy()
                ss: Optional[NDArray[np.float64]] = None
                if pop.has_sperm_storage:
                    n_ztypes = pop.n_ztypes
                    sperm_size = pop.n_ages * n_ztypes * n_ztypes
                    ss = row[1 + ind_size : 1 + ind_size + sperm_size].reshape(
                        pop.n_ages, n_ztypes, n_ztypes
                    ).copy()
                return (tick, ic, ss)

        raise ValueError(f"Tick {tick} not found in history.")

    # ── Post-hoc observation ──────────────────────────────────────────────

    def observe(
        self,
        observation: Observation | NDArray[np.float64],
        labels: Optional[Tuple[str, ...]] = None,
        collapse_age: bool = False,
    ) -> History:
        """Create a new observation-mode History from raw history.

        Supports two calling conventions:

        *New*: Pass a compiled :class:`Observation`::
            obs_hist = raw_hist.observe(observation)

        *Legacy*: Pass a mask + labels + collapse_age::
            obs_hist = raw_hist.observe(mask, ("g0",), collapse_age=False)

        Only valid when the current mode is ``"raw"``.

        Args:
            observation: Compiled :class:`Observation` **or** NDArray
                binary mask (legacy).
            labels: Group labels (legacy, required when *observation*
                is an ndarray).
            collapse_age: Collapse age flag (legacy).

        Returns:
            New ``History`` in observation mode.

        Raises:
            ValueError: If mode is not ``"raw"``.
        """
        if self._schema.mode != "raw":
            raise ValueError(
                "observe() is only valid on raw-mode History."
            )

        if isinstance(observation, np.ndarray):
            # legacy path
            assert labels is not None
            return self._observe_from_mask(observation, labels, collapse_age)

        return self._observe_from_observation(observation)

    def _observe_from_mask(
        self,
        observation_mask: NDArray[np.float64],
        observation_labels: Tuple[str, ...],
        collapse_age: bool,
    ) -> History:
        n_groups = len(observation_labels)
        pop = self._schema.population
        n_sexes = pop.n_sexes
        n_ages = pop.n_ages
        n_ztypes = pop.n_ztypes
        obs_row_size = 1 + n_groups * n_sexes * n_ages
        obs_meta = ObservationMetadata(
            labels=observation_labels,
            collapse_age=collapse_age,
            n_groups=n_groups,
        )
        obs_schema = HistorySchema(
            mode="observation",
            population=pop,
            row_size=obs_row_size,
            observation=obs_meta,
            spatial_layout=None,
        )
        obs_history = History(obs_schema)
        from natal.output.observation import apply_rule

        ind_size = n_sexes * n_ages * n_ztypes
        for row in self._rows:
            tick = row[0]
            ind_count = row[1 : 1 + ind_size].reshape(
                n_sexes, n_ages, n_ztypes
            )
            observed = apply_rule(ind_count, observation_mask)
            flat = np.empty(obs_row_size, dtype=np.float64)
            flat[0] = tick
            flat[1:] = observed.ravel()
            obs_history._rows.append(flat)

        return obs_history

    def _observe_from_observation(self, observation: Observation) -> History:
        pop = self._schema.population
        n_sexes = pop.n_sexes
        n_ages = pop.n_ages
        n_ztypes = pop.n_ztypes
        n_groups = observation.n_groups
        obs_row_size = 1 + n_groups * n_sexes * n_ages

        obs_meta = ObservationMetadata(
            labels=observation.labels,
            collapse_age=observation.collapse_age,
            n_groups=n_groups,
        )
        obs_schema = HistorySchema(
            mode="observation",
            population=pop,
            row_size=obs_row_size,
            observation=obs_meta,
            spatial_layout=None,
        )
        obs_history = History(obs_schema)

        mask = observation.build_mask(
            n_sexes=n_sexes, n_ages=n_ages, n_ztypes=n_ztypes
        )
        from natal.output.observation import apply_rule

        ind_size = n_sexes * n_ages * n_ztypes
        for row in self._rows:
            tick = row[0]
            ind_count = row[1 : 1 + ind_size].reshape(
                n_sexes, n_ages, n_ztypes
            )
            observed = apply_rule(ind_count, mask)
            flat = np.empty(obs_row_size, dtype=np.float64)
            flat[0] = tick
            flat[1:] = observed.ravel()
            obs_history._rows.append(flat)

        return obs_history

    def to_numpy(self) -> NDArray[np.float64]:
        """Return all history rows as a single 2-D array.

        Returns:
            2-D float64 array ``(n_records, row_size)``.
            Returns ``(0, row_size)`` when empty.
        """
        if len(self._rows) == 0:
            return np.zeros((0, self._schema.row_size), dtype=np.float64)
        return np.array(self._rows, dtype=np.float64)

    def to_list(self) -> List[Tuple[int, NDArray[np.float64]]]:
        """Return history rows as ``(tick, flat_row)`` pairs.

        Returns:
            List of ``(tick, row)`` tuples for legacy compatibility.
        """
        return [(int(row[0]), row) for row in self._rows]

    def to_dict(self, *, include_zero_counts: bool = False) -> Dict[str, Any]:
        """Translate history to a readable dictionary.

        Args:
            include_zero_counts: Whether to include zero-valued entries.

        Returns:
            Nested dict with metadata and per-snapshot entries.
        """
        mode = self._schema.mode
        population = self._schema.population
        snapshots: List[Dict[str, Any]] = []

        if mode == "raw":
            from natal.data import (
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

        for row in self._rows:
            tick = int(row[0])
            observed = row[1:].reshape(
                obs_meta.n_groups, population.n_sexes, population.n_ages
            )
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
    state: Any,
    sex_labels: List[str],
    genotype_labels: List[str],
    include_zero_counts: bool,
) -> Dict[str, Any]:
    from natal.data import PopulationState

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

    result: Dict[str, Any] = {"tick": int(state.n_tick)}
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
) -> Dict[str, Any]:
    if observed.ndim == 3:
        n_ages = int(observed.shape[2])
        payload: Dict[str, Any] = {}
        for group_idx, group_name in enumerate(labels):
            sex_age_block: Dict[str, Dict[str, float]] = {}
            for sex_idx, sex_name in enumerate(sex_labels):
                age_block: Dict[str, float] = {}
                for age_idx in range(n_ages):
                    value = float(observed[group_idx, sex_idx, age_idx])
                    if include_zero_counts or value != 0.0:
                        age_block[f"age_{age_idx}"] = value
                if include_zero_counts or age_block:
                    sex_age_block[sex_name] = age_block
            payload[group_name] = sex_age_block
        return payload

    if observed.ndim == 2:
        payload = {}
        for group_idx, group_name in enumerate(labels):
            sex_value_block: Dict[str, float] = {}
            for sex_idx, sex_name in enumerate(sex_labels):
                value = float(observed[group_idx, sex_idx])
                if include_zero_counts or value != 0.0:
                    sex_value_block[sex_name] = value
            payload[group_name] = sex_value_block
        return payload

    raise ValueError(f"Unsupported observed array ndim: {observed.ndim}")
