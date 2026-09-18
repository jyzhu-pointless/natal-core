"""Spatial population builder with fluent API and batch-setting support.

Provides ``SpatialPopulationBuilder`` for constructing ``SpatialPopulation`` instances
via a chainable API. Supports both homogeneous (all demes identical) and
heterogeneous (per-deme varying parameters via ``batch_setting``) construction.

Examples:
    >>> species = Species.from_dict(...)
    >>> pop = (SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    ...     .setup(name="demo", stochastic=False)
    ...     .initial_state(female={"WT|WT": 5000}, male={"WT|WT": 5000})
    ...     .reproduction(eggs_per_female=50)
    ...     .competition(carrying_capacity=10000)
    ...     .presets(drive)
    ...     .migration(kernel=my_kernel, migration_rate=0.1)
    ...     .build())
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence as SequenceABC
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.builder import PopulationBuilder
from natal.frontend.builder._base import normalize_observation_groups
from natal.frontend.genetics import Species
from natal.frontend.hooks.types import DemeSelector
from natal.frontend.model import ModelDraft
from natal.frontend.patterns import IndividualSelector
from natal.frontend.population.age_structured import AgeStructuredPopulation
from natal.frontend.population.discrete_generation import DiscreteGenerationPopulation
from natal.frontend.spatial.migration import (
    RateColumnDeclaration,
    normalize_migration_rate,
)
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import GridTopology

if TYPE_CHECKING:
    from natal.frontend.builder._declarations import ProjectedDeclarations
    from natal.frontend.genetics.compile import GameteList, ZygoteList
    from natal.frontend.model.definition import ModelDefinition
    from natal.frontend.model.definition_compiler import CompiledProducts
    from natal.frontend.model.initial_state import InitialDistributionDeclaration
    from natal.frontend.model.publication import IndexProjection
    from natal.frontend.presets import GeneticPreset

__all__ = [
    "BatchSetting",
    "batch_setting",
    "SpatialPopulationBuilder",
]

# Type aliases for population and builder types used throughout.
PopulationInstance = Union[AgeStructuredPopulation, DiscreteGenerationPopulation]
_HookItem = Union[
    Callable[..., object],  # object: hook callback return type varies by hook category
    Dict[str, List[Tuple[Callable[..., object], Optional[str], Optional[int]]]],
]


# ---------------------------------------------------------------------------
# BatchSetting
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class BatchSetting(Generic[_T]):
    """Deferred per-deme parameter specification.

    Wraps one of three value kinds used by ``SpatialPopulationBuilder`` to express
    parameters that vary across demes:

    - **scalar**: A Python sequence (list/tuple), one element per deme.
      Each element can be a scalar or an array (e.g. per-deme
      equilibrium distributions).
    - **array**: A 1D or 2D numpy array. 1D arrays have one element per deme
      (flat-index order); 2D arrays use ``(row, col)`` layout matching the
      topology grid and are flattened in row-major order at build time.
    - **spatial**: A callable ``(flat_idx) -> float`` or
      ``(row, col) -> float`` (auto-detected by parameter count),
      expanded at build time.

    ``SpatialPopulationBuilder`` detects ``BatchSetting`` values in builder method
    calls, stores them, and expands them during ``build()``.

    Note:
        The ``_T`` type parameter is the element type of the per-deme
        sequence.  It is inferred from the ``Sequence[T]`` input;
        ndarray/callable inputs default to ``Any`` because their element
        types cannot be statically determined.
    """

    _KIND_SCALAR = "scalar"
    _KIND_ARRAY = "array"
    _KIND_SPATIAL = "spatial"

    def __init__(
        self,
        values: Union[Sequence[_T], NDArray[np.floating[Any]], Callable[..., float]],
    ):
        """Initialize a BatchSetting from one of three value kinds.

        Args:
            values: Per-deme specification — a sequence of scalars (one per
                deme), a 1D/2D numpy array, or a callable for spatial
                expansion.
        """
        self._fn: Optional[Callable[..., float]] = None
        self._fn_param_count: Optional[int] = None
        self._values: Optional[List[_T]] = None
        self._values_array: Optional[
            NDArray[np.floating[Any]]
        ]  # Any: dtype parameter — npt.NDArray shorthand = None
        self._n_demes: Optional[int] = None

        if callable(values):
            self._kind: str = self._KIND_SPATIAL
            self._fn = values
        elif isinstance(values, np.ndarray):
            if values.ndim not in (1, 2):
                raise ValueError(
                    f"BatchSetting array must be 1D or 2D, got shape {values.shape}"
                )
            self._kind = self._KIND_ARRAY
            self._values_array = np.asarray(values)
            self._n_demes = int(self._values_array.size)
        else:
            self._kind = self._KIND_SCALAR
            self._values = list(values)
            self._n_demes = len(self._values)

    @property
    def kind(self) -> str:
        """str: The kind of value source (``"scalar"``, ``"array"``, or ``"spatial"``)."""
        return self._kind

    def __repr__(self) -> str:
        if self._kind == self._KIND_SPATIAL:
            return f"BatchSetting(kind={self._kind!r})"
        return f"BatchSetting(kind={self._kind!r}, n={self._n_demes})"

    def expand(
        self,
        n_demes: int,
        topology: Optional[GridTopology] = None,
    ) -> List[_T]:
        """Expand to a concrete list of per-deme values.

        - **scalar/array**: validate length, return as list (2D arrays are
          flattened row-major).
        - **spatial**: call the function for each deme index.  Parameter
          count is auto-detected: 1 param → ``fn(flat_idx)``,
          2 params → ``fn(row, col)``.

        Args:
            n_demes: Number of demes to expand to.
            topology: Optional ``GridTopology`` required for spatial kind.

        Returns:
            List of per-deme values (scalars or arrays).

        Raises:
            ValueError: If length mismatch or spatial kind without topology.
        """
        if self._kind == self._KIND_SCALAR:
            if self._values is None:
                raise ValueError("BatchSetting scalar values are None")
            if len(self._values) != n_demes:
                raise ValueError(
                    f"BatchSetting has {len(self._values)} values "
                    f"but {n_demes} demes are required"
                )
            return list(self._values)

        elif self._kind == self._KIND_ARRAY:
            if self._values_array is None:
                raise ValueError("BatchSetting array values are None")
            if self._n_demes != n_demes:
                raise ValueError(
                    f"BatchSetting array has {self._n_demes} values "
                    f"but {n_demes} demes are required"
                )
            arr = self._values_array
            if arr.ndim == 2:
                if topology is not None and arr.shape != (topology.rows, topology.cols):
                    raise ValueError(
                        f"BatchSetting 2D array shape {arr.shape} does not match "
                        f"topology shape ({topology.rows}, {topology.cols})"
                    )
                return cast(List[_T], arr.ravel(order="C").tolist())
            return cast(List[_T], arr.tolist())

        elif self._kind == self._KIND_SPATIAL:
            if topology is None:
                raise ValueError(
                    "Spatial BatchSetting requires topology for expansion."
                )
            fn = cast(Callable[..., float], self._fn)
            # Auto-detect parameter count: 2 → (row, col), else (flat_idx).
            if self._fn_param_count is None:
                import inspect

                try:
                    sig = inspect.signature(fn)
                    self._fn_param_count = len(sig.parameters)
                except (ValueError, TypeError):
                    self._fn_param_count = 1
            if self._fn_param_count >= 2:
                return cast(
                    List[_T],
                    [float(fn(*topology.from_index(i))) for i in range(n_demes)],
                )
            return cast(List[_T], [float(fn(i)) for i in range(n_demes)])

        raise ValueError(f"Unknown kind: {self._kind}")

    def first_value(self) -> Optional[_T]:
        """Return a single concrete element for template-builder delegation.

        ``SpatialPopulationBuilder`` holds a single-deme template builder internally.
        When a parameter is wrapped in ``batch_setting`` (a per-deme list),
        the template builder still needs one scalar value to proceed through
        ``setup() → … → build()``. This method provides that value —
        typically the first element of the list or array.

        For spatial kind (lazy callable), returns ``None`` because the value
        cannot be resolved without topology expansion at build time.

        Returns:
            The first element for scalar/array kinds, or ``None`` for spatial kind.
        """
        if self._kind == self._KIND_SCALAR:
            return self._values[0] if self._values else None
        elif self._kind == self._KIND_ARRAY:
            if self._values_array is not None and self._values_array.size > 0:
                flat = self._values_array.ravel(order="C")
                val = flat[0]
                return val.item() if hasattr(val, "item") else val
            return None
        return None  # spatial kind: deferred until expand() has topology

    def snapshot(self) -> BatchSetting[_T]:
        """Return a detached batch while preserving opaque element identity."""
        from natal.frontend.model.definition import copy_declaration_value

        if self._kind == self._KIND_SPATIAL:
            return BatchSetting(cast(Callable[..., float], self._fn))
        if self._kind == self._KIND_ARRAY:
            if self._values_array is None:
                raise ValueError("BatchSetting array values are None")
            return cast(BatchSetting[_T], BatchSetting(self._values_array.copy()))
        if self._values is None:
            raise ValueError("BatchSetting scalar values are None")
        copied = copy_declaration_value(self._values)
        return BatchSetting(cast(Sequence[_T], copied))


def batch_setting(
    values: Union[
        Sequence[_T], NDArray[np.floating[Any]], Callable[..., float], BatchSetting[_T]
    ],
) -> BatchSetting[_T]:
    """Create a ``BatchSetting`` for per-deme parameter specification.

    Args:
        values: One of:
            - A list/tuple of scalars of length ``n_demes``.
            - A 1D or 2D numpy array. 2D arrays use ``(row, col)`` layout and
              are flattened in row-major order.
            - A callable ``(flat_idx) -> float`` or ``(row, col) -> float``
              (auto-detected by parameter count).
            - An existing ``BatchSetting`` (returned as-is).

    Returns:
        A ``BatchSetting`` instance that ``SpatialPopulationBuilder`` detects and
        expands at build time.
    """
    if isinstance(values, BatchSetting):
        return cast(BatchSetting[_T], values)
    return BatchSetting(values)


# ---------------------------------------------------------------------------
# _make_hashable
# ---------------------------------------------------------------------------


def _make_hashable(
    value: Any,
) -> Any:  # Any param+return: accepts arbitrary types for dict-key conversion
    """Recursively convert *value* into a hashable form for deduplication.

    Used by the heterogeneous build to detect which demes have identical
    genetics values and can share a compiled config.  Without this, two
    numpy arrays with identical contents would be seen as different
    dict keys (ndarray is not hashable).

    Conversion rules:
    - numpy arrays → (``"__ndarray__"``, raw bytes).  Same content = same bytes.
    - dicts → sorted ``(key, hashable_value)`` tuples.  Sorting ensures
      ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce the same key.
    - lists / tuples → recursively converted element by element.
    - scalars (int, float, str) → pass through unchanged.

    Returns:
        A hashable object suitable for use as a dict key or set element.
    """
    if isinstance(value, np.ndarray):
        # ndarray.tobytes() is order-sensitive — different layouts or dtypes
        # produce different bytes, which is the desired behavior.
        return ("__ndarray__", value.tobytes())
    if isinstance(value, dict):
        d = cast(Dict[Any, Any], value)
        items = sorted(d.items(), key=lambda x: str(x[0]))
        return ("__dict__", tuple((k, _make_hashable(v)) for k, v in items))
    if isinstance(value, list):
        lst = cast(List[Any], value)
        return tuple(_make_hashable(v) for v in lst)
    if isinstance(value, tuple):
        tup = cast(tuple[Any, ...], value)
        return tuple(_make_hashable(v) for v in tup)
    # Scalar: int, float, str, bool — already hashable.
    return value


# Builder kwargs that alter the genetics section beyond the route table:
# positional batch presets and custom modifiers rewrite the meiosis /
# offspring maps just like genetics-section fitness rows do.
_PRESET_KWARG_PREFIX = "_preset_"
_MODIFIER_KWARGS = frozenset({"gamete_modifiers", "zygote_modifiers"})
_BATCH_KEY_SEPARATOR = "#"
_BATCH_KEYS_FIELD = "__spatial_batch_keys__"


def _batch_key_base(name: str) -> str:
    """Return the user-facing name encoded by an internal batch key."""
    return name.split(_BATCH_KEY_SEPARATOR, 1)[0]


def _declaration_field(method: str, name: str) -> str:
    """Resolve a declaration argument to the draft field it writes."""
    from natal.frontend.builder._routes import ROUTES_BY_METHOD

    for route in ROUTES_BY_METHOD.get(method, ()):
        if name == route.name or name in route.aliases:
            return route.config_field or route.name
    return name


def _genetics_route_names() -> frozenset[str]:
    """Genetics-section user-facing names from the route table."""
    from natal.frontend.builder._routes import ROUTES_BY_METHOD

    names: set[str] = set()
    for entries in ROUTES_BY_METHOD.values():
        for entry in entries:
            if entry.section == "genetics":
                names.add(entry.name)
                names.update(entry.aliases)
    return frozenset(names)


def _float_value(  # pyright: ignore[reportUnusedFunction]  # imported by the replay-log type-boundary test
    value: object, *, name: str
) -> (
    float
):  # object: accepts any scalar from builder replay log (int, float, np.generic)
    """Narrow a replay-log scalar before converting it to float.

    Args:
        value: Deferred scalar value.
        name: Configuration field used in error messages.

    Returns:
        The scalar converted to float.

    Raises:
        TypeError: If the replay value is not numeric.
    """
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        scalar = cast(bool | int | float, value.item())
        return float(scalar)
    raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# _clone_deme
# ---------------------------------------------------------------------------


def _values_equal(left: object, right: object) -> bool:
    """Compare two journal values, array-aware."""
    import numpy as np

    if left is right:
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(cast("Any", left), cast("Any", right)))
    try:
        return bool(left == right)
    except Exception:
        return False


def _genetics_batch_names(batch_param_names: List[str]) -> List[str]:
    """Return the batch parameter names that alter the genetics section.

    Grouping rule: only genetics content decides whether
    demes need distinct compiled configs (variant bank entries).  Ecology
    batch values (carrying capacity, survival, initial state, …) are
    filled per deme into the ecology columns instead of splitting groups.

    Args:
        batch_param_names: All accumulated batch kwarg names.

    Returns:
        The subset of names whose per-deme differences are genetic.
    """
    route_names = _genetics_route_names()
    return [
        name
        for name in batch_param_names
        if _batch_key_base(name) in route_names
        or _batch_key_base(name) in _MODIFIER_KWARGS
        or _batch_key_base(name).startswith(_PRESET_KWARG_PREFIX)
    ]



def _clone_deme(
    template: PopulationInstance,
    config: ModelDraft,
    name: str,
) -> PopulationInstance:
    """Create a lightweight functional copy of a template deme.

    Delegates to the population instance's ``_clone`` method.  The clone
    **shares** the following with the template (same object reference):

    - ``_config`` — ModelDraft (and all ndarrays within it)
    - ``_species``, ``_index_registry``, ``_registry``
    - ``compiled_hook_descriptors`` and the native HookProgram
    - ``_gamete_modifiers``, ``_zygote_modifiers``

    Only these are **independent copies**:

    - ``_state`` (individual_count, sperm_storage arrays)
    - ``_name``
    - ``_initial_population_snapshot``

    This means N clones of the same template share one copy of all compiled
    hook data and config arrays; only the mutable state arrays differ.

    Args:
        template: A fully-built population instance (``AgeStructuredPopulation``
            or ``DiscreteGenerationPopulation``).
        config: The ``ModelDraft`` for the clone (shared by reference).
        name: Unique name for the clone.

    Returns:
        A new population instance of the same type as *template*.
    """
    return template._clone(name=name, config=config)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# _replace optimization: builder-kwarg → config-field mappings
# ---------------------------------------------------------------------------
#

def _object_sequence(
    value: object, *, name: str
) -> Sequence[
    object
]:  # object: accepts any sequence from builder replay log (list, tuple, ndarray)
    """Validate a replay-log value used as positional arguments.

    Args:
        value: Deferred replay-log value.
        name: User-facing field name for error messages.

    Returns:
        The value narrowed to a non-string sequence.

    Raises:
        TypeError: If the value cannot be expanded as positional arguments.
    """
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, got {type(value).__name__}")
    return cast(Sequence[object], value)


# ---------------------------------------------------------------------------
# SpatialPopulationBuilder
# ---------------------------------------------------------------------------


class SpatialPopulationBuilder:
    """Fluent builder for ``SpatialPopulation``.

    Wraps a single-deme ``PopulationBuilder`` as a template. All chainable
    configuration methods delegate to the template and return ``self``.

    Spatial-specific parameters (topology, migration, adjacency) are stored
    directly and forwarded to ``SpatialPopulation`` at build time.
    """

    def __init__(
        self,
        species: Species,
        n_demes: int,
        topology: Optional[GridTopology] = None,
        *,
        pop_type: Literal["age_structured", "discrete_generation"] = "age_structured",
    ):
        """Initialize the spatial builder.

        Creates a single-deme template ``PopulationBuilder`` internally and
        stores spatial parameters (topology, migration) for later use
        during ``build()``.

        Args:
            species: Genetic architecture shared by all demes.
            n_demes: Number of demes in the spatial layout.
            topology: Optional grid topology for migration routing.
            pop_type: Population model type — ``"age_structured"``
                (default) or ``"discrete_generation"``.

        Raises:
            ValueError: If ``n_demes`` is less than 1.
        """
        if n_demes < 1:
            raise ValueError(f"n_demes must be >= 1, got {n_demes}")

        self._species = species
        self._n_demes = n_demes
        self._topology = topology
        self._pop_type: Literal["age_structured", "discrete_generation"] = pop_type

        # Observation groups (optional, applied at build time).
        self._observation_groups: dict[str, IndividualSelector] | None = None
        self._observation_collapse_age: bool = False
        self._observation_demes: tuple[int, ...] = tuple(range(n_demes))
        self._observation_deme_mode: Literal["preserve", "aggregate"] = "preserve"
        self._record_history_mode: Literal["raw", "observation"] = "raw"
        self._record_history_max_rows: int | None = None

        # Create the template builder (new path).  The unified
        # PopulationBuilder serves both granularities — the flag only picks the
        # normalized draft shape.
        if pop_type == "age_structured":
            self._template: PopulationBuilder = PopulationBuilder.from_species(species)
        else:
            self._template: PopulationBuilder = PopulationBuilder.from_species(
                species, discrete=True
            )
        # Demes build through this template: declared deme selectors must
        # survive compilation so the container plan can pin per-deme hooks.
        self._template._spatial_template = True  # pyright: ignore[reportPrivateUsage]  # owning container configures its template

        # Accumulated batch settings: param_name -> BatchSetting.
        self._batch_settings: Dict[
            str, BatchSetting[Any]
        ] = {}  # Any: BatchSetting value type varies per config field

        # Declaration journal: the spatial twin of the plain
        # PopulationBuilder's _declaration_log — same entry type, plus raw
        # BatchSetting values preserved for the per-group replay.  This is
        # the SINGLE store for the spatial chain: template calls bypass the
        # @_declared wrapper (see _call_template) so no second journal
        # entry is written for the same declaration.
        self._declaration_log: List[tuple[str, Dict[str, Any]]] = []
        # Batch keys are declaration-local.  This prevents repeated positional
        # preset calls from sharing the same ``_preset_<i>`` slot.
        self._batch_bindings: List[Dict[str, str]] = []
        self._batch_key_serial: int = 0

        # Spatial migration parameters.  ``migration_rate`` keeps the raw
        # declaration: a plain rate form is normalized by the container,
        # while a ``BatchSetting`` is expanded to a per-deme column at
        # build time (see ``_resolved_migration_rate``).
        self._migration_kernel: Optional[NDArray[np.float64]] = None
        self._migration_kernel_batch: Optional[BatchSetting[Any]] = None
        self._migration_rate: RateColumnDeclaration | BatchSetting[Any] = 0.0
        self._migration_strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = (
            "auto"
        )
        self._migration_adjacency: Optional[object] = None
        self._kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None
        self._deme_kernel_ids: Optional[NDArray[np.int64]] = None
        self._kernel_include_center: bool = False
        self._adjust_migration_on_edge: bool = False

        # Parameter registry (mirrors PopulationBuilderBase._param_values).
        self._param_values: dict[str, object] = {}
        self._spatial_name: str = "SpatialPopulation"

        # Compression (set via setup()).
        self._compress: bool = False
        self._declared_zygote_types: set[str] | set[int] | None = None

    # ------------------------------------------------------------------
    # Internal: batch detection and delegation
    # ------------------------------------------------------------------

    def _call_template(
        self,
        method_name: str,
        *args: object,  # object: template methods take heterogeneous payloads
        **kwargs: object,  # object: template methods take heterogeneous payloads
    ) -> None:
        """Invoke a template method without journaling it a second time.

        The @_declared decorator on the template's chaining methods would
        append a sanitized entry to the template's own journal; the spatial
        journal above already records the same declaration (with the raw
        BatchSetting values the replay needs), so this helper calls the
        undecorated function to keep exactly one store per declaration.

        Args:
            method_name: Template method name.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
        """
        bound = getattr(self._template, method_name)
        inner = getattr(bound, "__wrapped__", None)
        if inner is not None:
            inner(self._template, *args, **kwargs)
        else:
            bound(*args, **kwargs)

    def _stage_batch_kwargs(
        self, kwargs: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, BatchSetting[Any]]]:
        """Split *kwargs* into concrete template values and staged batch specs.

        ``BatchSetting`` values are **not** written to ``_batch_settings``
        here.  The caller commits the returned staging dict only after the
        template call succeeded, so a failing declaration leaves the spatial
        batch configuration exactly as it was.

        Args:
            kwargs: Raw keyword arguments from a chaining call.

        Returns:
            ``(concrete, staged)`` — the kwargs with each ``BatchSetting``
            replaced by its first value, plus the ``BatchSetting`` objects
            keyed by their original kwarg name.
        """
        concrete: Dict[str, Any] = {}
        staged: Dict[str, BatchSetting[Any]] = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchSetting):
                batch = cast(BatchSetting[Any], value)
                staged[key] = batch
                # Feed the first element to the template builder so it can
                # proceed through setup() → … → build() without errors; the
                # full per-deme list is expanded at build().
                first = batch.first_value()
                if first is not None:
                    concrete[key] = first
            else:
                concrete[key] = value
        return concrete, staged

    def _commit_batch_settings(
        self, staged: Mapping[str, BatchSetting[Any]]
    ) -> Dict[str, str]:
        """Commit staged batches and return declaration-local key bindings."""
        bindings: Dict[str, str] = {}
        for name, batch in staged.items():
            key = name
            if key in self._batch_settings:
                self._batch_key_serial += 1
                key = f"{name}{_BATCH_KEY_SEPARATOR}{self._batch_key_serial}"
            self._batch_settings[key] = batch.snapshot()
            bindings[name] = key
        return bindings

    def _append_declaration(
        self,
        method_name: str,
        kwargs: Dict[str, Any],
        batch_bindings: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Append one journal entry and its local batch binding map."""
        from natal.frontend.model.definition import copy_declaration_value

        self._declaration_log.append(
            (method_name, cast(Dict[str, Any], copy_declaration_value(kwargs)))
        )
        self._batch_bindings.append(dict(batch_bindings or {}))

    def _detect_and_delegate(
        self,
        method_name: str,
        kwargs: Dict[str, Any],
    ) -> SpatialPopulationBuilder:
        """Detect BatchSetting values in kwargs, store them, and delegate
        concrete (non-batch) values to the template builder's method.

        **Dual-store pattern**::

            Each chainable call does two things simultaneously:

            1. **Record** the raw kwargs (including BatchSetting objects) in
               ``_declaration_log`` — used later by the group projector
               to replay the full builder pipeline for each config group.
            2. **Delegate** a sanitized version to the template builder —
               ``BatchSetting`` values are replaced with their first element
               so the single-deme builder can proceed through its build()
               pipeline without errors.

            At the end of the chain, the template builder has been fully
            configured with the *first* value of every BatchSetting.  The
            complete per-deme lists are stored in ``_batch_settings`` and
            expanded at ``build()`` time.

        Args:
            method_name: Name of the method on the template builder.
            kwargs: Keyword arguments passed by the user.

        Returns:
            Self for chaining.
        """
        concrete, staged = self._stage_batch_kwargs(kwargs)

        # Delegate sanitized kwargs to the template (single store: the
        # decorator's journaling is bypassed).  The staged batch specs and
        # the journal entry are committed only after the template call
        # succeeded: a failed call must leave no trace, neither in the
        # replayable declaration log nor in the batch configuration.
        filtered = {k: v for k, v in concrete.items() if v is not None}
        self._call_template(method_name, **filtered)
        bindings = self._commit_batch_settings(staged)
        # Record the original call with BatchSetting objects preserved,
        # for the group declaration projector.
        self._append_declaration(method_name, dict(kwargs), bindings)
        return self

    def setup(
        self,
        name: str = "SpatialPopulation",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
    ) -> SpatialPopulationBuilder:
        """Configure basic population settings.

        Args:
            name: Human-readable population name.
            stochastic: Whether to use stochastic sampling.
            continuous_sampling: If True, use Dirichlet sampling.
            fixed_egg_count: If True, egg count is fixed.
            compress: If True, enable index compression at build time.
                Compression is applied once at the spatial level (not
                per-group), producing a unified registry shared by all
                demes — safe for cross-deme migration.
            declared_zygote_types: Optional sequence of genotype
                selectors to protect from compression pruning.
                Hook genotype references are auto-collected; use this
                only for genotypes introduced by custom native hooks.

        Returns:
            Self for chaining.
        """
        self._spatial_name = name
        replay_kwargs: dict[str, object] = {
            "name": name,
            "stochastic": stochastic,
            "continuous_sampling": continuous_sampling,
            "fixed_egg_count": fixed_egg_count,
            "compress": compress,
            "declared_zygote_types": declared_zygote_types,
        }
        template_kwargs: dict[str, object] = {
            "name": name,
            "stochastic": stochastic,
            "continuous_sampling": continuous_sampling,
            "fixed_egg_count": fixed_egg_count,
            "compress": compress if not self._batch_settings else False,
            "declared_zygote_types": declared_zygote_types,
        }
        self._call_template("setup", **template_kwargs)  # type: ignore[arg-type]  # template_kwargs has mixed value types; setup validates at runtime
        # Journal only after the template call succeeded — failed calls
        # stay out of the replayable declaration log.
        self._append_declaration("setup", replay_kwargs)
        if compress:
            self._compress = True
        if declared_zygote_types is not None:
            self._declared_zygote_types = cast(
                "set[str] | set[int]", set(declared_zygote_types)
            )
        return self

    def age_structure(
        self,
        n_ages: int = 8,
        new_adult_age: int = 2,
        generation_time: Optional[float] = None,
        equilibrium_distribution: Optional[
            Union[List[float], NDArray[np.float64]]
        ] = None,
    ) -> SpatialPopulationBuilder:
        """Configure age structure (age-structured models only).

        Args:
            n_ages: Number of age classes.
            new_adult_age: Age at which individuals become adults.
            generation_time: Optional pre-computed generation time.
            equilibrium_distribution: Optional equilibrium distribution.

        Returns:
            Self for chaining.
        """
        if self._pop_type != "age_structured":
            raise TypeError("age_structure() is only valid for age_structured pop_type")
        # ``equilibrium_distribution`` is a competition-domain declaration:
        # the underlying age_structure has no such parameter, so routing it
        # through competition (the working channel) keeps the advertised
        # kwarg functional instead of dying on a TypeError.
        result = self._detect_and_delegate(
            "age_structure",
            {
                "n_ages": n_ages,
                "new_adult_age": new_adult_age,
                "generation_time": generation_time,
            },
        )
        if equilibrium_distribution is not None:
            self._detect_and_delegate(
                "competition",
                {"equilibrium_distribution": equilibrium_distribution},
            )
        return result

    def initial_state(
        self,
        individual_count: Any,  # Any: accepts nested dict, list, or ndarray — validated internally
        sperm_storage: Optional[
            Any
        ] = None,  # Any: accepts nested dict, list, or ndarray — validated internally  # Any: accepts nested dict, list, or ndarray — validated internally
    ) -> SpatialPopulationBuilder:
        """Configure the initial population state.

        Args:
            individual_count: Initial abundance mapping.
            sperm_storage: Optional initial sperm storage (age-structured only).

        Returns:
            Self for chaining.
        """
        kwargs: Dict[str, Any] = {"individual_count": individual_count}
        if sperm_storage is not None:
            kwargs["sperm_storage"] = sperm_storage
        return self._detect_and_delegate("initial_state", kwargs)

    def survival(
        self,
        # Age-structured params
        female_age_based_survival: Optional[Any] = None,
        male_age_based_survival: Optional[Any] = None,
        generation_time: Optional[float] = None,
        equilibrium_distribution: Optional[Any] = None,
        # Discrete-generation params
        female_age0_survival: Optional[float] = None,
        male_age0_survival: Optional[float] = None,
    ) -> SpatialPopulationBuilder:
        """Configure survival rates.

        Args:
            female_age_based_survival: Per-age female survival (age-structured).
            male_age_based_survival: Per-age male survival (age-structured).
            generation_time: Optional generation time override.
            equilibrium_distribution: Optional equilibrium distribution.
            female_age0_survival: Female age-0 survival (discrete-generation).
            male_age0_survival: Male age-0 survival (discrete-generation).

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            # ``equilibrium_distribution`` is a competition-domain
            # declaration: the underlying survival has no such parameter,
            # so route it through competition (the working channel).
            # NOTE: ``generation_time`` remains forwarded-and-rejected for
            # now — the template's age_structure must precede every domain
            # method, so a survival-time override has no lawful channel
            # until the structure-domain cleanup batch.
            result = self._detect_and_delegate(
                "survival",
                {
                    "female_age_based_survival": female_age_based_survival,
                    "male_age_based_survival": male_age_based_survival,
                    "generation_time": generation_time,
                },
            )
            if equilibrium_distribution is not None:
                self._detect_and_delegate(
                    "competition",
                    {"equilibrium_distribution": equilibrium_distribution},
                )
            return result
        else:
            return self._detect_and_delegate(
                "survival",
                {
                    "female_age0_survival": female_age0_survival,
                    "male_age0_survival": male_age0_survival,
                },
            )

    def reproduction(
        self,
        # Shared params (accept BatchSetting for per-deme variation)
        eggs_per_female: Union[float, BatchSetting[Any]] = 50.0,
        sex_ratio: Union[float, BatchSetting[Any]] = 0.5,
        fixed_egg_count: bool | None = None,
        # Age-structured params
        female_age_based_mating_rate: Optional[Any] = None,
        male_age_based_mating_rate: Optional[Any] = None,
        age_based_reproduction_rate: Optional[Any] = None,
        female_age_based_fertility: Optional[Any] = None,
        sperm_displacement_rate: float = 0.05,
        # Discrete-generation params
        female_adult_mating_rate: float = 1.0,
        male_adult_mating_rate: float = 1.0,
    ) -> SpatialPopulationBuilder:
        """Configure reproduction and mating parameters.

        Args:
            eggs_per_female: Expected offspring per adult female. Accepts ``BatchSetting``.
            sex_ratio: Proportion of female offspring.
            fixed_egg_count: If True, egg count is deterministic. If omitted,
                preserve the setup value (False by default).
            female_age_based_mating_rate: Female mating rates (age-structured).
            male_age_based_mating_rate: Male mating rates (age-structured).
            age_based_reproduction_rate: Reproduction participation rates.
            female_age_based_fertility: Fertility weights.
            sperm_displacement_rate: Rate of sperm displacement (age-structured).
            female_adult_mating_rate: Adult female mating rate (discrete-generation).
            male_adult_mating_rate: Adult male mating rate (discrete-generation).

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            return self._detect_and_delegate(
                "reproduction",
                {
                    "female_age_based_mating_rate": female_age_based_mating_rate,
                    "male_age_based_mating_rate": male_age_based_mating_rate,
                    "age_based_reproduction_rate": age_based_reproduction_rate,
                    "female_age_based_fertility": female_age_based_fertility,
                    "eggs_per_female": eggs_per_female,
                    "fixed_egg_count": fixed_egg_count,
                    "sex_ratio": sex_ratio,
                    "sperm_displacement_rate": sperm_displacement_rate,
                },
            )
        else:
            return self._detect_and_delegate(
                "reproduction",
                {
                    "eggs_per_female": eggs_per_female,
                    "fixed_egg_count": fixed_egg_count,
                    "sex_ratio": sex_ratio,
                    "female_adult_mating_rate": female_adult_mating_rate,
                    "male_adult_mating_rate": male_adult_mating_rate,
                },
            )

    def competition(
        self,
        # Age-structured params
        competition_strength: float | None = None,
        juvenile_growth_mode: Union[int, str, BatchSetting[Any]] = "beverton_holt",
        low_density_growth_rate: Union[float, BatchSetting[Any]] = 6.0,
        age_1_carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
        old_juvenile_carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
        expected_num_new_adult_females: Union[int, None, BatchSetting[Any]] = None,
        equilibrium_distribution: Optional[
            Union[List[float], NDArray[np.float64], BatchSetting[Any]]
        ] = None,
        # Discrete-generation params
        carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
    ) -> SpatialPopulationBuilder:
        """Configure competition and density-dependence.

        Args:
            competition_strength: Competition weight of the second juvenile
                age class (age-structured only).  Defaults to ``1.0`` — the
                same weight as age 0, so leaving it unset adds no special
                weighting.  Passing it to a model whose only juvenile age is
                age 0 is rejected.
            juvenile_growth_mode: Growth model identifier. Accepts ``BatchSetting``.
            low_density_growth_rate: Growth rate at low density. Accepts ``BatchSetting``.
            age_1_carrying_capacity: Carrying capacity at age=1 (age-structured).
                Accepts ``BatchSetting``.
            old_juvenile_carrying_capacity: Alias for ``age_1_carrying_capacity``.
            expected_num_new_adult_females: Equilibrium adult females. Accepts ``BatchSetting``.
            equilibrium_distribution: Optional equilibrium distribution.
            carrying_capacity: Carrying capacity (discrete-generation). Accepts ``BatchSetting``.

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            # Alias carrying_capacity / old_juvenile_carrying_capacity → age_1_carrying_capacity
            resolved_cc = age_1_carrying_capacity
            if resolved_cc is None:
                resolved_cc = old_juvenile_carrying_capacity
            if resolved_cc is None:
                resolved_cc = carrying_capacity

            # Unspecified means "no special weighting": the model default is
            # 1.0 for every juvenile age, so nothing is written.  An explicit
            # value on a model with only one juvenile age is rejected
            # downstream instead of silently doing nothing.
            return self._detect_and_delegate(
                "competition",
                {
                    "competition_strength": competition_strength,
                    "juvenile_growth_mode": juvenile_growth_mode,
                    "low_density_growth_rate": low_density_growth_rate,
                    "age_1_carrying_capacity": resolved_cc,
                    "expected_num_new_adult_females": expected_num_new_adult_females,
                    "equilibrium_distribution": equilibrium_distribution,
                },
            )
        else:
            return self._detect_and_delegate(
                "competition",
                {
                    "juvenile_growth_mode": juvenile_growth_mode,
                    "low_density_growth_rate": low_density_growth_rate,
                    "carrying_capacity": carrying_capacity,
                },
            )

    def presets(self, *preset_list: GeneticPreset) -> SpatialPopulationBuilder:
        """Add gene-drive presets (applied during build).

        Each positional argument may be a ``BatchSetting`` of preset objects,
        allowing different demes to receive different presets.

        Args:
            *preset_list: One or more preset objects, or ``BatchSetting``
                instances wrapping per-deme preset values.

        Returns:
            Self for chaining.
        """
        # Detect BatchSetting in positional args.  The per-deme specs are
        # staged, not stored: a failing preset recipe must not leave a batch
        # entry behind, and re-using a ``_preset_<i>`` key from an earlier
        # call must keep the earlier value until this call succeeds.
        staged: Dict[str, BatchSetting[Any]] = {}
        concrete_args: list[object] = []
        for i, item in enumerate(preset_list):
            if isinstance(item, BatchSetting):
                batch = cast(BatchSetting[Any], item)
                staged[f"_preset_{i}"] = batch
                first = batch.first_value()
                if first is not None:
                    concrete_args.append(first)
            else:
                concrete_args.append(item)

        # Template first: the batch specs and the journal entry are committed
        # only after the recipe expanded successfully, so a failed call
        # leaves neither the spatial batch configuration nor the replayable
        # declaration log modified.
        # concrete_args contains GeneticPreset instances resolved from potential
        # BatchSetting wrappers; cast needed because first_value() returns object.
        self._call_template("presets", *cast("list[GeneticPreset]", concrete_args))
        bindings = self._commit_batch_settings(staged)
        self._append_declaration("presets", {"preset_list": preset_list}, bindings)
        return self

    def fitness(
        self,
        viability: Optional[Any] = None,
        fecundity: Optional[Any] = None,
        sexual_selection: Optional[Any] = None,
        zygote_viability: Optional[Any] = None,
        mode: str = "replace",
    ) -> SpatialPopulationBuilder:
        """Configure fitness values (applied after presets).

        Args:
            viability: Genotype selectors to viability fitness values.
            fecundity: Genotype selectors to fecundity fitness values.
            sexual_selection: Mating preference mapping.
            zygote_viability: Genotype selectors to zygote viability values.
            mode: ``"replace"`` (default) or ``"multiply"``.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate(
            "fitness",
            {
                "viability": viability,
                "fecundity": fecundity,
                "sexual_selection": sexual_selection,
                "zygote_viability": zygote_viability,
                "mode": mode,
            },
        )

    def custom(
        self, **kwargs: bool | int | float | NDArray[np.float64]
    ) -> SpatialPopulationBuilder:
        """Register custom named slots on every deme's draft.

        Custom slots are container-uniform: the kwargs are replayed onto
        each group template at build time, so all demes carry the same
        values (per-deme custom slots are not a batch_setting axis).
        Values follow the panmictic ``PopulationBuilder.custom`` contract and
        reach the Rust session via ``Params.custom_slots``.

        Args:
            **kwargs: Name-value pairs for custom slots.  Values must be
                ``bool``, ``int``, ``float``, or ``NDArray[np.float64]``.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate("custom", dict(kwargs))

    def hooks(
        self,
        *hook_items: _HookItem,
        event: Optional[str] = None,
        priority: Optional[int] = None,
        deme: DemeSelector = "*",
        name: Optional[str] = None,
    ) -> SpatialPopulationBuilder:
        """Declare lifecycle hooks for every deme's build chain.

        Declaration keywords mirror the panmictic ``PopulationBuilder.hooks``:
        ``deme`` metadata rides on the compiled descriptors (demes outside
        the selector never fire the hook), not on the container.

        Args:
            *hook_items: Functions decorated with ``@hook`` or hook mappings.
            event: Default event for items that do not carry one.
            priority: Priority assigned to the op items of this call
                (packing = one shared priority); ``None`` keeps the ops'
                own values, which must then agree within a list.
            deme: Deme selector for the compiled descriptors.
            name: Optional name for grouped op declarations.

        Returns:
            Self for chaining.
        """
        # Template first: the journal entry lands only after the call
        # succeeded, so a failed declaration stays out of the replayable
        # log.  (Hook validation itself is deferred to build(), so an
        # immediate failure here can only come from the template.)
        self._call_template(
            "hooks",
            *hook_items,
            event=event,
            priority=priority,
            deme=deme,
            name=name,
        )
        self._append_declaration(
            "hooks",
            {
                "hook_items": hook_items,
                "event": event,
                "priority": priority,
                "deme": deme,
                "name": name,
            },
        )
        return self

    def modifiers(
        self,
        gamete_modifiers: Optional[
            List[Tuple[int, Optional[str], Callable[..., object]]]
        ] = None,
        zygote_modifiers: Optional[
            List[Tuple[int, Optional[str], Callable[..., object]]]
        ] = None,
    ) -> SpatialPopulationBuilder:
        """Configure custom modifier functions.

        Args:
            gamete_modifiers: Modifiers for gamete production.
            zygote_modifiers: Modifiers for zygote formation.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate(
            "modifiers",
            {
                "gamete_modifiers": gamete_modifiers,
                "zygote_modifiers": zygote_modifiers,
            },
        )

    def with_observation(
        self,
        groups: Mapping[str, IndividualSelector],
        *,
        collapse_age: bool = False,
        demes: Optional[Sequence[int]] = None,
        deme_mode: Literal["preserve", "aggregate"] = "preserve",
    ) -> SpatialPopulationBuilder:
        """Define the canonical spatial Observation at build time.

        This method only defines how ``pop.observe()`` projects population
        counts. History remains raw unless ``record_history(mode="observation")``
        is configured independently.

        Args:
            groups: Non-empty ordered mapping from labels to selectors.
            collapse_age: Whether to sum and remove the age axis.
            demes: Ordered deme indices to observe. ``None`` selects all
                demes in population order.
            deme_mode: ``"preserve"`` keeps a shared deme axis;
                ``"aggregate"`` sums and removes it.

        Returns:
            SpatialPopulationBuilder: Self for chaining.

        Raises:
            TypeError: If groups is not a mapping of selectors.
            ValueError: If groups, a group label, the deme mode, or the deme
                selection is invalid.
        """
        self._observation_groups = normalize_observation_groups(groups)
        self._observation_collapse_age = collapse_age
        if deme_mode not in ("preserve", "aggregate"):
            raise ValueError(
                f"deme_mode must be 'preserve' or 'aggregate', got {deme_mode!r}"
            )
        selected_demes = tuple(range(self._n_demes)) if demes is None else tuple(demes)
        if not selected_demes:
            raise ValueError("Observation selects no demes")
        if any(type(index) is not int for index in selected_demes):
            raise TypeError("demes must contain integer deme indices")
        if any(index < 0 or index >= self._n_demes for index in selected_demes):
            raise ValueError(
                f"demes must be within [0, {self._n_demes}), got {selected_demes!r}"
            )
        if len(set(selected_demes)) != len(selected_demes):
            raise ValueError("demes must not contain duplicate indices")
        self._observation_demes = selected_demes
        self._observation_deme_mode = deme_mode
        return self

    def record_history(
        self,
        *,
        mode: Literal["raw", "observation"] = "raw",
        max_rows: Optional[int] = None,
    ) -> SpatialPopulationBuilder:
        """Set the recording mode and capacity for spatial population history.

        Must be called during the build phase.

        When ``mode="observation"`` and no ``.with_observation()`` has been
        called, an identity observation (one group per ZType) is
        automatically generated.

        Args:
            mode: ``"raw"`` for full-state recording or ``"observation"``
                for compressed observation-aggregate recording.
            max_rows: Maximum number of records to keep (FIFO eviction).

        Returns:
            Self for chaining.

        Raises:
            ValueError: When mode is invalid or ``max_rows`` is less than one.
        """
        if mode not in ("raw", "observation"):
            raise ValueError(f"mode must be 'raw' or 'observation', got {mode!r}")
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"max_rows must be >= 1 or None, got {max_rows}")
        self._record_history_mode = mode
        self._record_history_max_rows = max_rows
        return self

    # ------------------------------------------------------------------
    # Spatial-specific methods
    # ------------------------------------------------------------------

    def migration(
        self,
        kernel: Optional[NDArray[np.float64]] = None,
        migration_rate: Union[RateColumnDeclaration, BatchSetting[Any]] = 0.0,
        strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = "auto",
        adjacency: Optional[
            object
        ] = None,  # object: adjacency matrix (NDArray, list, or None) — duck-typed
        kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None,
        deme_kernel_ids: Optional[NDArray[np.int64]] = None,
        kernel_include_center: bool = False,
        adjust_migration_on_edge: bool = False,
    ) -> SpatialPopulationBuilder:
        """Configure spatial migration parameters.

        Args:
            kernel: Odd-shaped 2D migration kernel.
            migration_rate: Fraction of each deme's adults that migrates
                each tick.  Four sugar forms share the build-time rules:
                a scalar (adult ages only, juveniles 0), an ``(n_ages,)``
                age vector, an ``(S, A)`` sex x age table, and a per-sex
                mapping such as ``{"F": 0.2, "M": 0.05}``.  A full
                ``(n_demes, S, A)`` column sets each deme directly, and an
                ``(n_demes, n_ages)`` table broadcasts each deme's age
                vector across sexes — except that a 2-D shape of exactly
                ``(S, A)`` keeps its shared per-sex reading, so when
                ``n_demes == n_sexes`` the two 2-D shapes collide and the
                per-deme reading is unavailable: pass the 3-D column or a
                ``BatchSetting`` there.  A ``BatchSetting`` gives one rate
                declaration per deme (each element follows the same
                sugar), so ``batch_setting([0.1, 0.4, 0.1])`` on a 3-deme
                chain sends the middle deme four times as much as its
                neighbors.
            strategy: Migration strategy (``"auto"``, ``"adjacency"``,
                ``"kernel"``, ``"hybrid"``).
            adjacency: Explicit adjacency matrix.  Rows are relative
                outbound weights: the container row-normalizes every
                non-empty row, so sub- and super-stochastic inputs mean
                the same distribution.  Control the amount migrated with
                ``migration_rate``, not by shrinking a row.
            kernel_bank: Optional heterogeneous kernel bank.
            deme_kernel_ids: Per-deme kernel ids into ``kernel_bank``.
            kernel_include_center: Whether kernel includes center cell.
            adjust_migration_on_edge: Legacy bit-parity flag, kept for
                compatibility.  It does not change the destination
                distribution: the fold renormalizes every emitted row
                (and every non-empty adjacency row) to a total weight of
                1, so the denominator it selects cancels up to
                floating-point rounding (~1 ulp).  Boundary demes merely
                have fewer neighbors and therefore send a larger share to
                each; their total outbound quota equals an interior
                deme's.  Use ``migration_rate`` for per-deme quotas.

        Returns:
            Self for chaining.
        """
        if isinstance(kernel, BatchSetting):
            if kernel_bank is not None or self._kernel_bank is not None:
                raise ValueError(
                    "Cannot use batch_setting for kernel when kernel_bank "
                    "is also provided. Use one or the other."
                )
            if deme_kernel_ids is not None or self._deme_kernel_ids is not None:
                raise ValueError(
                    "Cannot use batch_setting for kernel when deme_kernel_ids "
                    "is also provided — indices would conflict."
                )
            self._migration_kernel_batch = kernel
        elif kernel is not None:
            self._migration_kernel = np.asarray(kernel, dtype=np.float64)
        # Keep the raw declaration: scalar / vector / per-sex sugar and the
        # explicit per-deme columns are normalized once by the
        # SpatialPopulation constructor.  A BatchSetting stays deferred and
        # is expanded to a (n_demes, S, A) column at build time.
        self._migration_rate = migration_rate
        self._migration_strategy = strategy
        if adjacency is not None:
            self._migration_adjacency = adjacency
        if kernel_bank is not None:
            self._kernel_bank = kernel_bank
        if deme_kernel_ids is not None:
            self._deme_kernel_ids = np.asarray(deme_kernel_ids, dtype=np.int64)
        self._kernel_include_center = bool(kernel_include_center)
        self._adjust_migration_on_edge = bool(adjust_migration_on_edge)
        # Keep the raw declaration in the parameter registry: the scalar /
        # dict / vector sugar is normalized once by the SpatialPopulation
        # constructor, not here.
        self._param_values["migration.migration_rate"] = migration_rate
        return self

    # ------------------------------------------------------------------
    # Parameter introspection (mirrors PopulationBuilderBase)
    # ------------------------------------------------------------------

    def get_params(
        self,
    ) -> dict[str, object]:  # object: config field values (int, float, ndarray, bool)
        """Return all registered parameter values.

        Merges spatial-specific params with values read from the template config.
        """
        from natal.frontend.utils.parameters import ALL_PARAMETERS

        params = dict(self._param_values)
        # Read scalar config values through ParamDescriptor registry
        for key, desc in ALL_PARAMETERS.items():
            if desc.config_field is None or desc.kind == "geno_tensor":
                continue
            field: object = getattr(
                self._template.config, desc.config_field, None
            )  # object: config fields have heterogeneous types
            if field is None:
                continue
            val: object  # object: config field values are heterogeneous (int, float, ndarray)
            if desc.config_path and isinstance(field, np.ndarray):
                val = cast(object, field[desc.config_path])
            elif isinstance(field, np.ndarray) and field.ndim == 0:
                val = cast(object, field[()])
            else:
                val = field  # pyright: ignore[reportUnknownVariableType] — getattr returns unknowable types
            params[key] = val
        return params

    def get_param(
        self, domain: str, name: str
    ) -> object | None:  # object: config field value (int, float, ndarray, bool, None)
        """Look up a single registered parameter value."""
        return self.get_params().get(f"{domain}.{name}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _resolve_migration_kernels(
        self,
    ) -> tuple[Optional[Sequence[NDArray[np.float64]]], Optional[NDArray[np.int64]]]:
        """Convert batch kernel to ``(kernel_bank, deme_kernel_ids)`` if needed.

        When ``.migration(kernel=batch_setting([...]))`` was used, this expands
        the per-deme kernel list, deduplicates unique kernels into a bank, and
        builds the index mapping.

        Returns:
            ``(kernel_bank, deme_kernel_ids)`` if batch kernel was set,
            otherwise ``(self._kernel_bank, self._deme_kernel_ids)``.
        """
        if self._migration_kernel_batch is None:
            return self._kernel_bank, self._deme_kernel_ids

        kernels = self._migration_kernel_batch.expand(self._n_demes, self._topology)
        unique: list[NDArray[np.float64]] = []
        kernel_map: dict[object, int] = {}
        ids: list[int] = []
        for k in kernels:
            arr = np.asarray(k, dtype=np.float64)
            key = _make_hashable(arr)
            if key not in kernel_map:
                kernel_map[key] = len(unique)
                unique.append(arr)
            ids.append(kernel_map[key])
        return tuple(unique), np.array(ids, dtype=np.int64)

    def _resolved_migration_rate(
        self, definition: ModelDefinition
    ) -> RateColumnDeclaration:
        """Expand a per-deme ``BatchSetting`` rate into a concrete column.

        A plain declaration is returned unchanged and normalized by the
        container; a ``BatchSetting`` expands to one element per deme, each
        normalized with the same sugar rules as a homogeneous declaration,
        and stacked into the ``(n_demes, S, A)`` contract column.  The
        frozen draft supplies the rate axes so the expansion never depends
        on the mutable builder.

        Args:
            definition: Frozen declaration whose draft supplies the axes.

        Returns:
            The per-deme ``(n_demes, S, A)`` column, or the raw declaration
            when it is not batched.

        Raises:
            ValueError: If the batch has the wrong length for the deme
                count, or the frozen declaration carries no draft.
        """
        if not isinstance(self._migration_rate, BatchSetting):
            return self._migration_rate
        draft = definition.draft
        if draft is None:
            raise ValueError(
                "a batched migration_rate requires a normalized declaration draft"
            )
        per_deme = self._migration_rate.expand(self._n_demes, self._topology)
        rows = [
            normalize_migration_rate(
                value,
                int(draft.n_sexes),
                int(draft.n_ages),
                int(draft.new_adult_age),
            )
            for value in per_deme
        ]
        return np.stack(rows, axis=0)

    def build(self) -> SpatialPopulation:
        """Build and return the configured ``SpatialPopulation``.

        Single entry point: one build path handles both
        the homogeneous and the heterogeneous case.

        - **No ``batch_setting``**: build ONE template deme, clone N-1
          times.  All demes share the same config object — maximum memory
          efficiency, zero redundant work.

        - **Any ``batch_setting``**: group demes by *genetics* content
          signature, build one template per group, then fill per-deme
          ecology through cheap ``_replace`` config shells.  Deme count
          and ecology diversity therefore never inflate the number of
          full builds; the runtime variant bank (see
          ``natal.backends.rust.rust_backend``) deduplicates along the
          same genetics-only rule.

        Returns:
            A ``SpatialPopulation`` with all demes initialized.
        """
        definition = self._definition_for_compile()
        return self._build_from_definition(definition, compiled_template=self._template)

    def _definition_for_compile(self) -> ModelDefinition:
        """Freeze concrete spatial controls before creating any execution session."""
        from natal.frontend.model.definition import (
            ModelDefinition,
            SpatialInputs,
            copy_declaration_value,
        )

        base = self._template._definition_for_compile()  # pyright: ignore[reportPrivateUsage]  # shared template compiler input.
        expanded = {name: tuple(batch.expand(self._n_demes, self._topology)) for name, batch in self._batch_settings.items()}

        def normalize(value: Any) -> Any:
            # Any: group call values can include nested batches and opaque recipes.
            if isinstance(value, BatchSetting):
                first = cast("BatchSetting[Any]", value).first_value()
                return copy_declaration_value(first)
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in cast("dict[str, Any]", value).items()}
            if isinstance(value, (tuple, list)):
                return tuple(normalize(item) for item in cast("Sequence[Any]", value))
            return copy_declaration_value(value)

        kernel_bank, kernel_ids = self._resolve_migration_kernels()
        group_calls: list[tuple[str, dict[str, Any]]] = []
        for index, (name, kwargs) in enumerate(self._declaration_log):
            normalized = normalize(kwargs)
            if index < len(self._batch_bindings) and self._batch_bindings[index]:
                # Keep this metadata in SpatialInputs only.  ModelDefinition's
                # public journal remains the clean replay journal, while a
                # detached spatial compiler retains declaration identity.
                normalized[_BATCH_KEYS_FIELD] = dict(self._batch_bindings[index])
            group_calls.append((name, normalized))
        controls = SpatialInputs(
            self._n_demes, self._topology, self._pop_type, self._spatial_name,
            tuple(expanded.items()),
            tuple(group_calls),
            {
                "adjacency": self._migration_adjacency, "kernel": self._migration_kernel,
                "strategy": self._migration_strategy, "kernel_bank": kernel_bank,
                "deme_kernel_ids": kernel_ids, "kernel_include_center": self._kernel_include_center,
                "migration_rate": self._migration_rate, "adjust_migration_on_edge": self._adjust_migration_on_edge,
            },
            self._observation_groups, self._observation_collapse_age,
            self._observation_demes, self._observation_deme_mode,
            self._record_history_mode, self._record_history_max_rows, self._compress,
            None if self._declared_zygote_types is None else cast("frozenset[str] | frozenset[int]", frozenset(self._declared_zygote_types)),
        )
        return ModelDefinition(
            self._species, self._pop_type == "discrete_generation",
            tuple(self._declaration_log), self._spatial_name,
            presets=base.presets, manual_gamete=base.manual_gamete,
            manual_zygote=base.manual_zygote, compilation_key=base.compilation_key,
            observation_collapse_age=base.observation_collapse_age,
            history_mode=base.history_mode, history_max_rows=base.history_max_rows,
            compress=base.compress, declared_zygote_types=base.declared_zygote_types,
            draft=base.draft, registry=base.registry,
            fitness_base=base.fitness_base, fitness_steps=base.fitness_steps,
            hook_calls=base.hook_calls, observation_groups=base.observation_groups,
            spatial=controls,
        )

    @classmethod
    def _build_from_definition(
        cls, definition: ModelDefinition, *, compiled_template: PopulationBuilder | None = None,
    ) -> SpatialPopulation:
        """Compile a detached declaration using the existing group compiler.

        Args:
            definition: The frozen declaration carrying the template inputs
                and concrete spatial controls.
            compiled_template: The builder whose already-compiled products
                (working draft and validity marker) may be reused so group
                compiles never re-execute the same recipes per deme.
        """
        controls = definition.spatial
        if definition.draft is None or controls is None:
            raise ValueError("Spatial compilation requires normalized spatial inputs")
        compiler = cls(definition.species, controls.n_demes, controls.topology, pop_type=controls.pop_type)
        template = PopulationBuilder(definition.draft, species=definition.species)
        template._spatial_template = True  # pyright: ignore[reportPrivateUsage]  # detached spatial group template keeps per-deme selectors.
        template._registry = definition.registry  # pyright: ignore[reportPrivateUsage]  # initialize one isolated compiler candidate.
        template._presets = list(definition.presets)  # pyright: ignore[reportPrivateUsage]
        template._manual_gamete = cast("GameteList", list(definition.manual_gamete))  # pyright: ignore[reportPrivateUsage]
        template._manual_zygote = cast("ZygoteList", list(definition.manual_zygote))  # pyright: ignore[reportPrivateUsage]
        template._fitness_base = definition.fitness_base  # pyright: ignore[reportPrivateUsage]
        template._fitness_steps = list(definition.fitness_steps)  # pyright: ignore[reportPrivateUsage]
        template._compilation_key = (  # pyright: ignore[reportPrivateUsage]
            definition.compilation_key if definition.compilation_key is not None else object()
        )
        if compiled_template is not None:
            # Transfer the source builder's compile products and validity
            # marker directly: same declaration identity, so group builds
            # finalize instead of re-running the recipes.
            template._compiled_draft = compiled_template._compiled_draft  # pyright: ignore[reportPrivateUsage]
            template._cached_compilation_key = compiled_template._cached_compilation_key  # pyright: ignore[reportPrivateUsage]
        template._hook_calls = list(definition.hook_calls)  # pyright: ignore[reportPrivateUsage]
        template._observation_groups = definition.observation_groups  # pyright: ignore[reportPrivateUsage]
        template._observation_collapse_age = definition.observation_collapse_age  # pyright: ignore[reportPrivateUsage]
        template._record_history_mode = definition.history_mode  # pyright: ignore[reportPrivateUsage]
        template._record_history_max_rows = definition.history_max_rows  # pyright: ignore[reportPrivateUsage]
        template._compress = definition.compress  # pyright: ignore[reportPrivateUsage]
        template._declared_zygote_types = None if definition.declared_zygote_types is None else cast("set[str] | set[int]", set(definition.declared_zygote_types))  # pyright: ignore[reportPrivateUsage]
        # The declared initial distribution rides with the definition; a
        # rebuilt template re-derives its arrays from it, so a rebuild
        # reproduces the originally built populations.
        template._initial_distribution = definition.initial_distribution  # pyright: ignore[reportPrivateUsage]  # declaration snapshot travels with the definition.
        compiler._template = template
        compiler._batch_settings = {name: BatchSetting(values) for name, values in controls.batch_values}
        compiler._declaration_log = list(controls.group_calls)
        compiler._batch_bindings = [
            cast(Dict[str, str], kwargs.get(_BATCH_KEYS_FIELD, {}))
            for _, kwargs in controls.group_calls
        ]
        template._declaration_log = compiler._resolved_group_journal({})  # pyright: ignore[reportPrivateUsage]  # preserve provenance without replaying recipes.
        compiler._spatial_name = controls.name
        compiler._observation_groups = None if controls.observation_groups is None else dict(controls.observation_groups)
        compiler._observation_collapse_age = controls.observation_collapse_age
        compiler._observation_demes = controls.observation_demes
        compiler._observation_deme_mode = controls.observation_deme_mode
        compiler._record_history_mode = controls.history_mode
        compiler._record_history_max_rows = controls.history_max_rows
        compiler._compress = controls.compress
        compiler._declared_zygote_types = None if controls.declared_zygote_types is None else cast("set[str] | set[int]", set(controls.declared_zygote_types))
        compiler.migration(**controls.migration)  # pyright: ignore[reportArgumentType]  # validated normalized migration keyword schema.
        return compiler._build_normalized(definition)

    def _build_normalized(self, definition: ModelDefinition) -> SpatialPopulation:
        """Execute group compilation and native construction from frozen inputs."""
        if not self._batch_settings:
            demes = self._build_homogeneous_demes()
        else:
            demes = self._build_heterogeneous_demes()

        kernel_bank, deme_kernel_ids = self._resolve_migration_kernels()

        spatial = SpatialPopulation(
            demes=demes,
            topology=self._topology,
            adjacency=self._migration_adjacency,
            migration_kernel=self._migration_kernel,
            migration_strategy=self._migration_strategy,
            kernel_bank=kernel_bank,
            deme_kernel_ids=deme_kernel_ids,
            kernel_include_center=self._kernel_include_center,
            migration_rate=self._resolved_migration_rate(definition),
            adjust_migration_on_edge=self._adjust_migration_on_edge,
            name=self._spatial_name,
        )
        spatial._definition = definition  # pyright: ignore[reportPrivateUsage]  # attach the actual input consumed by this compilation.
        # The Rust engine is the ONLY execution backend: the
        # spatial population builds its session here with the default seed
        # 0, and a missing extension is a hard error — no silent fallback
        # to the Python tick orchestration.
        from natal.backends.rust.rust_backend import rust_backend_available

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; the Rust engine is the "
                "only execution backend. Build it with `maturin develop` "
                "before constructing populations."
            )
        spatial._initialize_session(seed=0)  # pyright: ignore[reportPrivateUsage]  # build owns session initialization.
        self._compile_recording_plan(spatial)
        return spatial

    def _build_homogeneous_demes(self) -> List[PopulationInstance]:
        """Build one template deme and clone it N-1 times."""
        template = self._template.build(name=self._spatial_name)
        tpl_config = template._config  # pyright: ignore[reportPrivateUsage]  # share immutable published genetic products.

        assert tpl_config is not None
        demes: List[PopulationInstance] = [template]
        for i in range(1, self._n_demes):
            clone = _clone_deme(
                template,
                config=tpl_config,
                name=f"{self._spatial_name}_deme_{i}",
            )
            demes.append(clone)
        return demes

    def _projected_group(self, values: Mapping[str, object]) -> tuple[list[tuple[str, Dict[str, Any]]], ProjectedDeclarations]:
        """Project one deme's concrete declarations (no method execution).

        Each deme's config is the projection of its resolved declaration
        journal through the single interpreter, onto a fresh granularity
        baseline (§4.4: groups go straight to the model compiler — the
        template's consumed first values and builder-method replay are
        both out of the picture).
        """
        from natal.frontend.builder._base import PopulationBuilder
        from natal.frontend.builder._declarations import project_declaration_record

        baseline = (
            PopulationBuilder.for_age_structured(self._species)
            if self._pop_type == "age_structured"
            else PopulationBuilder.for_discrete(self._species)
        )
        journal = self._resolved_group_journal(values)
        projected = project_declaration_record(
            self._species, journal, base_draft=baseline._config,  # pyright: ignore[reportPrivateUsage]  # fresh local baseline.
        )
        return journal, projected

    def _projected_variant_config(
        self,
        group_config: ModelDraft,
        deme_values: Mapping[str, object],
        group_values: Mapping[str, object],
        *,
        initial_distribution: InitialDistributionDeclaration | None = None,
    ) -> ModelDraft:
        """Project one deme's *differing* declarations onto the group config.

        The delta carries each journaled call whose explicit values differ
        between the two value maps (whole-call kwargs, deme values
        substituted).  Undiffering calls — including derived-parameter
        declarations — are already reflected in *group_config* and stay
        untouched, preserving the variant contract that derived scalars
        freeze at the group's computation.
        """
        from natal.frontend.builder._declarations import project_declaration_record

        group_journal = self._resolved_group_journal(group_values)
        deme_journal = self._resolved_group_journal(deme_values)
        delta: list[tuple[str, Dict[str, Any]]] = []
        for declaration_index, ((name, group_kwargs), (_, deme_kwargs)) in enumerate(
            zip(group_journal, deme_journal)
        ):
            differing = {
                key for key, value in deme_kwargs.items()
                if key not in group_kwargs or not _values_equal(group_kwargs[key], value)
            }
            superseded_keys: set[str] = set()
            if differing:
                # A later uniform declaration supersedes an earlier value when
                # it is uniform across the group and variant.  The full group
                # projection already applied that later write; carrying the
                # earlier difference into the variant delta would incorrectly
                # write it back afterwards.
                for later_index in range(declaration_index + 1, len(group_journal)):
                    later_name, later_group = group_journal[later_index]
                    _, later_deme = deme_journal[later_index]
                    for key in tuple(deme_kwargs):
                        if key in {_BATCH_KEYS_FIELD, "__args__"}:
                            continue
                        field = _declaration_field(name, key)
                        superseded = any(
                            later_value is not None
                            and later_key in later_deme
                            and _declaration_field(later_name, later_key) == field
                            and _values_equal(later_value, later_deme[later_key])
                            for later_key, later_value in later_group.items()
                            if later_key != _BATCH_KEYS_FIELD
                        )
                        if superseded:
                            superseded_keys.add(key)
                differing.difference_update(superseded_keys)
            if differing:
                # Preserve the complete declaration so coupled fields and
                # derived values are recomputed in their original order.
                # Only fields proven superseded by a later uniform write are
                # removed; equal fields remain part of the same declaration.
                delta.append(
                    (name, {
                        key: value
                        for key, value in deme_kwargs.items()
                        if key not in superseded_keys
                    })
                )
        return project_declaration_record(
            self._species, delta, base_draft=group_config,
            initial_distribution=initial_distribution,
        ).draft

    def _carrier_from_projection(
        self,
        journal: list[tuple[str, Dict[str, Any]]],
        projected: ProjectedDeclarations,
        *,
        inherit_cache_from: PopulationBuilder | None = None,
    ) -> PopulationBuilder:
        """Restore an unpublished carrier builder from projected declarations.

        A data restore in the shape of ``_build_from_definition``: fields
        are assigned from the projection, never re-declared through
        builder methods.  ``inherit_cache_from`` transfers the template's
        compile cache when the group's genetics match it, so group-0
        builds reuse recipe products instead of re-running them.
        """
        from natal.frontend.builder._base import PopulationBuilder

        carrier = PopulationBuilder(projected.draft, species=self._species)
        carrier._spatial_template = True  # pyright: ignore[reportPrivateUsage]  # spatial hook selectors are retained.
        carrier._presets = list(projected.presets)  # pyright: ignore[reportPrivateUsage]
        carrier._manual_gamete = list(projected.manual_gamete)  # pyright: ignore[reportPrivateUsage]
        carrier._manual_zygote = list(projected.manual_zygote)  # pyright: ignore[reportPrivateUsage]
        carrier._fitness_steps = list(projected.fitness_steps)  # pyright: ignore[reportPrivateUsage]
        carrier._hook_calls = list(projected.hook_calls)  # pyright: ignore[reportPrivateUsage]
        carrier._initial_distribution = projected.initial_distribution  # pyright: ignore[reportPrivateUsage]
        carrier._custom_kwargs = dict(projected.draft.custom)  # pyright: ignore[reportPrivateUsage]
        carrier._compress = projected.compress  # pyright: ignore[reportPrivateUsage]
        carrier._declared_zygote_types = projected.declared_zygote_types  # pyright: ignore[reportPrivateUsage]
        carrier._observation_groups = projected.observation_groups  # pyright: ignore[reportPrivateUsage]
        carrier._observation_collapse_age = projected.observation_collapse_age  # pyright: ignore[reportPrivateUsage]
        carrier._record_history_mode = cast('Literal["raw", "observation"]', projected.history_mode)  # pyright: ignore[reportPrivateUsage]
        carrier._record_history_max_rows = projected.history_max_rows  # pyright: ignore[reportPrivateUsage]
        carrier._declaration_log = list(journal)  # pyright: ignore[reportPrivateUsage]
        if inherit_cache_from is not None:
            # Same declaration identity as the template: transfer the
            # compile cache AND the compiled modifier lists the cache's
            # fast path does not rebuild (a plain builder copy carried
            # them as instance state).
            carrier._compilation_key = inherit_cache_from._compilation_key  # pyright: ignore[reportPrivateUsage]
            carrier._compiled_draft = inherit_cache_from._compiled_draft  # pyright: ignore[reportPrivateUsage]
            carrier._cached_compilation_key = inherit_cache_from._cached_compilation_key  # pyright: ignore[reportPrivateUsage]
            carrier.gamete_modifiers = list(inherit_cache_from.gamete_modifiers)
            carrier.zygote_modifiers = list(inherit_cache_from.zygote_modifiers)
            carrier._registry = inherit_cache_from._registry  # pyright: ignore[reportPrivateUsage]  # unpublished template registry, same species catalog.
        return carrier

    def _build_heterogeneous_demes(self) -> List[PopulationInstance]:
        """Compile complete group candidates before choosing one spatial layout.

        Genetics recipes run once per genetics signature. Every initial-state
        declaration is resolved on unpublished full axes, including ecology-only
        variants. Reachability then sees every group's edges and every deme's
        seeds before any population or native session is constructed.
        """
        from copy import copy


        expanded = {
            name: batch.expand(self._n_demes, self._topology)
            for name, batch in self._batch_settings.items()
        }
        names = sorted(expanded)
        genetic_names = _genetics_batch_names(names)
        groups: defaultdict[tuple[tuple[str, Any], ...], List[int]] = defaultdict(list)
        values = [{name: expanded[name][i] for name in names} for i in range(self._n_demes)]
        for i, value in enumerate(values):
            signature = tuple((name, _make_hashable(value[name])) for name in genetic_names)
            groups[signature].append(i)

        candidates: dict[int, tuple[PopulationBuilder, CompiledProducts]] = {}
        group_products: list[CompiledProducts] = []
        for indices in groups.values():
            first = indices[0]
            journal, projected = self._projected_group(values[first])
            # The template consumed the first batch values during
            # declaration; when this group's genetics match, inherit its
            # compile cache so recipes never re-run for the same content.
            builder = self._carrier_from_projection(
                journal, projected,
                inherit_cache_from=self._template if first == 0 else None,
            )
            products = builder._compile_products()  # pyright: ignore[reportPrivateUsage]  # unpublished spatial candidate.
            group_products.append(products)
            candidates[first] = (builder, products)
            variants = {tuple((name, _make_hashable(values[first][name])) for name in names): candidates[first]}
            for i in indices[1:]:
                signature = tuple((name, _make_hashable(values[i][name])) for name in names)
                if signature in variants:
                    candidates[i] = variants[signature]
                    continue
                # An ecology variant projects only the declarations that
                # differ from the group's values, through the same
                # interpreter, onto the group's compiled config (genetics
                # within a group are identical by signature).  Derived
                # scalars (e.g. the Champer egg override) were computed on
                # the group's declaration and stay frozen unless the user
                # re-declares them — the established variant contract.
                variant = self._projected_variant_config(
                    products.config, values[i], values[first],
                    initial_distribution=builder._initial_distribution,  # pyright: ignore[reportPrivateUsage]  # group projection carries the final raw initial declaration.
                )
                variant_builder = copy(builder)
                variant_builder._config = variant  # pyright: ignore[reportPrivateUsage]  # complete unpublished axes.
                variant_builder._declaration_log = self._resolved_group_journal(values[i])  # pyright: ignore[reportPrivateUsage]
                candidates[i] = (variant_builder, products._replace(config=variant))
                variants[signature] = candidates[i]

        projection = self._spatial_projection(group_products, list(candidates.values()))
        demes: List[PopulationInstance] = [None] * self._n_demes  # type: ignore[list-item]  # filled before return.
        for indices in groups.values():
            published: dict[tuple[tuple[str, Any], ...], PopulationInstance] = {}
            genetic_template: CompiledProducts | None = None
            for i in indices:
                signature = tuple((name, _make_hashable(values[i][name])) for name in names)
                if signature in published:
                    template = published[signature]
                    config = template._config  # pyright: ignore[reportPrivateUsage]  # published template shell.
                    assert config is not None
                    demes[i] = _clone_deme(
                        template, config=config, name=f"{self._spatial_name}_deme_{i}",
                    )
                else:
                    builder, products = candidates[i]
                    deme = builder._publish_and_build(  # pyright: ignore[reportPrivateUsage]  # sole complete-to-published boundary.
                        products, name=f"{self._spatial_name}_deme_{i}", projection=projection,
                        genetic_template=genetic_template,
                    )
                    if genetic_template is None:
                        config = deme._config  # pyright: ignore[reportPrivateUsage]  # share published products, not an exported copy.
                        assert config is not None
                        genetic_template = products._replace(config=config, registry=deme.index_registry)
                    published[signature] = deme
                    demes[i] = deme
        return demes

    def _spatial_projection(
        self,
        groups: list[CompiledProducts],
        candidates: list[tuple[PopulationBuilder, CompiledProducts]],
    ) -> IndexProjection:
        """Plan the common layout using all unpublished maps and initial states."""
        from natal.frontend.builder._base import collect_hook_genotype_refs
        from natal.frontend.builder._registry_builder import resolve_declared_ztypes
        from natal.frontend.model.publication import IndexProjection, plan_projection

        first = groups[0]
        if not self._compress:
            return IndexProjection.identity(first.registry)
        seeds: set[int] = set()
        for builder, products in candidates:
            config = products.config
            seeds.update(int(i) for i in np.flatnonzero(np.any(config.initial_individual_count > 0, axis=(0, 1))))
            sperm = config.initial_sperm_storage
            if sperm.size:
                seeds.update(int(i) for i in np.flatnonzero(np.any(sperm > 0, axis=(0, 2))))
                seeds.update(int(i) for i in np.flatnonzero(np.any(sperm > 0, axis=(0, 1))))
            seeds.update(resolve_declared_ztypes(self._species, products.registry, self._declared_zygote_types))
            seeds.update(resolve_declared_ztypes(
                self._species, products.registry,
                collect_hook_genotype_refs(builder._hook_calls),  # pyright: ignore[reportPrivateUsage]  # declarations share the full catalog.
            ))
        # Merge the per-group route tables with an elementwise maximum:
        # the maps hold 0/1 routing flags, so the maximum is the union of
        # every group's routes.  The unified projection registry must
        # cover them all — a missing union entry would silently drop a
        # genotype from the projection.
        z2g = np.zeros_like(first.config.zygotes_to_gametes_map)
        g2z = np.zeros_like(first.config.gametes_to_zygotes_map)
        for products in groups:
            np.maximum(z2g, products.config.zygotes_to_gametes_map, out=z2g)
            np.maximum(g2z, products.config.gametes_to_zygotes_map, out=g2z)
        combined = first._replace(config=first.config._replace(
            zygotes_to_gametes_map=z2g, gametes_to_zygotes_map=g2z,
        ))
        return plan_projection(combined, full_ztype_indices=seeds)

    def _resolved_group_journal(self, values: Mapping[str, object]) -> list[tuple[str, dict[str, Any]]]:
        """Record concrete group inputs without executing their declarations again."""
        # Any: journal arguments include opaque recipe objects and nested selectors.
        result: list[tuple[str, dict[str, Any]]] = []
        for declaration_index, (method, kwargs) in enumerate(self._declaration_log):
            bindings: Mapping[str, str] = {}
            if _BATCH_KEYS_FIELD in kwargs:
                bindings = cast(Mapping[str, str], kwargs[_BATCH_KEYS_FIELD])
            elif declaration_index < len(self._batch_bindings):
                bindings = self._batch_bindings[declaration_index]
            resolved = {
                key: values.get(bindings[key], value) if key in bindings else value
                for key, value in kwargs.items()
                if key != _BATCH_KEYS_FIELD
            }
            if method in ("presets", "hooks"):
                source = "preset_list" if method == "presets" else "hook_items"
                items = _object_sequence(resolved.pop(source, ()), name=source)
                # Resolve only positional items belonging to this declaration.
                # A global ``_preset_<i>`` lookup would let a preset batch
                # replace an unrelated hooks item at the same position.
                journal_items: list[object] = list(cast("list[object]", items)) + list(cast("list[object]", resolved.pop("__args__", ())))
                expanded_items: list[object] = []
                for index, item in enumerate(journal_items):
                    batch_name = bindings.get(f"_preset_{index}")
                    if batch_name is not None:
                        expanded_items.append(values.get(batch_name, item))
                    elif isinstance(item, BatchSetting):
                        # Compatibility for a manually inserted live journal
                        # entry that predates declaration-local bindings.
                        legacy_name = f"_preset_{index}"
                        fallback = values.get(
                            legacy_name, cast(object, item.first_value())
                        )
                        expanded_items.append(fallback)
                    elif item is not None:
                        expanded_items.append(item)
                resolved["__args__"] = tuple(
                    expanded_items[i] for i in range(len(expanded_items)) if expanded_items[i] is not None
                )
            result.append((method, resolved))
        return result

    def _compile_recording_plan(self, spatial: SpatialPopulation) -> None:
        """Compile and freeze the :class:`RecordingPlan` on the spatial population."""
        from natal.frontend.output._recording import compile_recording_plan
        from natal.frontend.output.history import History
        from natal.frontend.output.observation import (
            ObservationFilter,
            build_identity_observation,
        )

        ref_deme = spatial._deme_object(0)  # pyright: ignore[reportPrivateUsage]  # typed internal consumer
        config = ref_deme.config
        if config.discrete_generation:
            kind = "spatial_discrete_generation"
            has_sperm = False
        else:
            kind = "spatial_age_structured"
            has_sperm = True

        record_mode = self._record_history_mode
        max_rows = self._record_history_max_rows

        n_ztypes = ref_deme.index_registry.n_ztypes
        if self._observation_groups is None:
            observation = build_identity_observation(
                ref_deme.index_registry,
                n_ztypes=n_ztypes,
                n_sexes=ref_deme.config.n_sexes,
                n_ages=ref_deme.config.n_ages,
                deme_indices=self._observation_demes,
                deme_mode=self._observation_deme_mode,
            )
        else:
            observation = ObservationFilter(
                ref_deme.index_registry
            ).build_from_selectors(
                groups=self._observation_groups,
                collapse_age=self._observation_collapse_age,
                n_sexes=ref_deme.config.n_sexes,
                n_ages=ref_deme.config.n_ages,
                n_ztypes=n_ztypes,
                deme_indices=self._observation_demes,
                deme_mode=self._observation_deme_mode,
            )
        spatial._observation = observation  # type: ignore[reportPrivateUsage]  # build-time installation of the immutable canonical rule

        plan = compile_recording_plan(
            ref_deme,
            mode=record_mode,
            kind=kind,
            n_demes=spatial.n_demes,
            has_sperm_storage=has_sperm,
            observation=observation,
        )
        from dataclasses import replace

        observation = replace(
            observation,
            population_fingerprint=plan.schema.population.fingerprint,
        )
        spatial._observation = observation  # type: ignore[reportPrivateUsage]  # bind canonical rule to the frozen PopulationLayout
        # The native store compiles its selector from this frozen policy when
        # bound; no per-tick raw batch is transported to the Python wrapper.
        spatial._observation_mask = None  # type: ignore[reportPrivateUsage]  # raw engine transport; container commits the configured History mode
        spatial._recording_plan = plan  # type: ignore[reportPrivateUsage]  # builder sets private attr on spatial
        spatial._history_obj = History(plan.schema, max_rows=max_rows)  # type: ignore[reportPrivateUsage]  # builder sets private attr
