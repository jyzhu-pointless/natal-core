"""Spatial population builder with fluent API and batch-setting support.

Provides ``SpatialConfigurator`` for constructing ``SpatialPopulation`` instances
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

from natal.frontend.configurator import Configurator
from natal.frontend.configurator._base import normalize_observation_groups
from natal.frontend.configurator._factory import (
    InitialIndividualCountInput,
    InitialSpermStorageInput,
)
from natal.frontend.data import ModelDraft
from natal.frontend.genetics import Species
from natal.frontend.genetics.structures._helpers import build_compression_mask
from natal.frontend.patterns import IndividualSelector
from natal.frontend.population.age_structured import AgeStructuredPopulation
from natal.frontend.population.discrete_generation import DiscreteGenerationPopulation
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.spatial.migration import RateDeclaration
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import GridTopology

if TYPE_CHECKING:
    from natal.frontend.data.definition import ModelDefinition
    from natal.frontend.genetics.compile import GameteList, ZygoteList
    from natal.frontend.genetics.definition_compiler import CompiledModel
    from natal.frontend.presets import GeneticPreset

__all__ = [
    "BatchSetting",
    "batch_setting",
    "SpatialConfigurator",
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

    Wraps one of three value kinds used by ``SpatialConfigurator`` to express
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

    ``SpatialConfigurator`` detects ``BatchSetting`` values in builder method
    calls, stores them, and expands them during ``build()``.

    Type Parameter:
        _T: The element type of the per-deme sequence.  Inferred from the
        ``Sequence[T]`` input; defaults to ``Any`` for ndarray/callable inputs
        where element types cannot be statically determined.
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

        ``SpatialConfigurator`` holds a single-deme template builder internally.
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
        A ``BatchSetting`` instance that ``SpatialConfigurator`` detects and
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


def _genetics_route_names() -> frozenset[str]:
    """Genetics-section user-facing names from the route table."""
    from natal.frontend.configurator._routes import ROUTES_BY_METHOD

    names: set[str] = set()
    for entries in ROUTES_BY_METHOD.values():
        for entry in entries:
            if entry.section == "genetics":
                names.add(entry.name)
                names.update(entry.aliases)
    return frozenset(names)


def _genetics_batch_names(batch_param_names: List[str]) -> List[str]:
    """Return the batch parameter names that alter the genetics section.

    Slice-5 stage-2 grouping rule: only genetics content decides whether
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
        if name in route_names
        or name in _MODIFIER_KWARGS
        or name.startswith(_PRESET_KWARG_PREFIX)
    ]


def _float_value(
    value: object, *, name: str
) -> (
    float
):  # object: accepts any scalar from configurator replay log (int, float, np.generic)
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
# ``_build_heterogeneous`` uses ``ModelDraft._replace`` to share heavy
# arrays across groups.  Most builder kwargs map directly to a same-named
# config field; only the exceptions below need explicit mappings.
#
# The dispatch in ``_build_variant_config`` works like this:
#
#   1. *array kwarg* (individual_count, sperm_storage)
#      → convert dict → array via PopulationConfigBuilder, then _replace.
#   2. *multi-field kwarg* (carrying_capacity variants)
#      → _replace into both base_carrying_capacity and the scaled
#        carrying_capacity.
#   3. *rename kwarg* (eggs_per_female → eggs_per_female, etc.)
#      → _replace under the renamed config field.
#   4. *any other kwarg*
#      → try ``hasattr(base_config, kwarg)``; if the config field exists,
#        _replace directly.  If not, fall back to full builder replay.
#
# This means adding a new batch-able scalar parameter typically requires
# zero changes here — as long as the builder kwarg and config field share
# the same name.
# ---------------------------------------------------------------------------

# Builder kwarg names that require dict → numpy-array conversion.
# The output array replaces the named config field.
_ARRAY_KWARGS: frozenset[str] = frozenset({"individual_count", "sperm_storage"})

# Builder kwarg → config field renames.
# Kwargs not listed here are tried directly with ``hasattr(base_config, name)``.
_KWARG_RENAMES: dict[str, str] = {
    "eggs_per_female": "eggs_per_female",
}

# Discrete-generation builder kwargs → (unified vector field, cell index).
# The draft schema has no per-scalar discrete fields; each kwarg writes one
# cell of a **copied** (2, n_ages) vector so variants never alias the base.
_DISCRETE_VECTOR_CELLS: dict[str, tuple[str, tuple[int, ...]]] = {
    "female_age0_survival": ("age_based_survival_rates", (0, 0)),
    "male_age0_survival": ("age_based_survival_rates", (1, 0)),
    "female_adult_mating_rate": ("age_based_mating_rates", (0, 1)),
    "male_adult_mating_rate": ("age_based_mating_rates", (1, 1)),
}


def _is_0d_field(config: ModelDraft, name: str) -> bool:
    """Return True if the config field *name* is a 0-d ndarray."""
    val = getattr(config, name, None)
    return isinstance(val, np.ndarray) and val.ndim == 0


def _object_sequence(
    value: object, *, name: str
) -> Sequence[
    object
]:  # object: accepts any sequence from configurator replay log (list, tuple, ndarray)
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
# SpatialConfigurator
# ---------------------------------------------------------------------------


class SpatialConfigurator:
    """Fluent builder for ``SpatialPopulation``.

    Wraps a single-deme ``Configurator`` as a template. All chainable
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
        """Initialize the spatial configurator.

        Creates a single-deme template ``Configurator`` internally and
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

        # Runtime population reference (None at build time; set by for_population).
        self._pop_ref: Optional[Any] = (
            None  # Any: stores a Population reference; concrete type varies
        )

        # Create the template configurator (new path).  The unified
        # Configurator serves both granularities — the flag only picks the
        # normalized draft shape.
        if pop_type == "age_structured":
            self._template: Configurator = Configurator.from_species(species)
        else:
            self._template: Configurator = Configurator.from_species(
                species, discrete=True
            )

        # Accumulated batch settings: param_name -> BatchSetting.
        self._batch_settings: Dict[
            str, BatchSetting[Any]
        ] = {}  # Any: BatchSetting value type varies per config field

        # Declaration journal (plan 5.1): the spatial twin of the plain
        # Configurator's _declaration_log — same entry type, plus raw
        # BatchSetting values preserved for the per-group replay.  This is
        # the SINGLE store for the spatial chain: template calls bypass the
        # @_declared wrapper (see _call_template) so no second journal
        # entry is written for the same declaration.
        self._declaration_log: List[tuple[str, Dict[str, Any]]] = []

        # Spatial migration parameters.
        self._migration_kernel: Optional[NDArray[np.float64]] = None
        self._migration_kernel_batch: Optional[BatchSetting[Any]] = None
        self._migration_rate: RateDeclaration = 0.0
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

    def _compress_once(self, expanded: Dict[str, List[Any]]) -> set[int]:
        """Compute the union of ztype indices reachable anywhere in the system.

        Builds the first group's template to obtain a resolved
        ``initial_individual_count`` array, then collects reachable ztype
        seeds from all groups' initial states, hook genotype refs, and
        user-declared types.  Runs a single BFS on combined modifier maps
        and returns ztype indices that must be protected from compression
        pruning across ALL groups.

        Returns:
            ``set[int]`` of ztype indices (pre-compression) to protect.
        """
        seeds: set[int] = set()

        # ── Step 1: Build first group template (no compress) ──────────
        # Must happen BEFORE seed collection because the resolved
        # initial_individual_count array provides correct ztype indices
        # — raw dict keys in `expanded` may be unresolved patterns,
        # Genotype objects, or tuples.
        batch_param_names = sorted(expanded.keys())
        first_sig: Dict[str, Any] = {
            name: expanded[name][0] for name in batch_param_names
        }
        full_template = self._build_template_for_group(
            first_sig,
            compress=False,
        )
        full_config = full_template.export_config()
        full_registry = full_template.index_registry

        # ── Step 2: Seeds from initial_individual_count ────────────────
        # Non-zero positions in the resolved array are ztype indices.
        if "individual_count" in expanded:
            n_demes = len(expanded["individual_count"])
            seen: set[tuple[tuple[str, Any], ...]] = set()
            from natal.frontend.configurator._factory import PopulationConfigBuilder

            n_ages = int(full_config.n_ages)
            new_adult_age = int(full_config.new_adult_age)

            for i in range(n_demes):
                sig_key = tuple(
                    (name, _make_hashable(expanded[name][i]))
                    for name in batch_param_names
                )
                if sig_key in seen:
                    continue
                seen.add(sig_key)

                ind_cnt = expanded["individual_count"][i]
                if not isinstance(ind_cnt, dict):
                    continue

                dist = cast(InitialIndividualCountInput, ind_cnt)

                if self._pop_type == "age_structured":
                    array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
                        species=self._species,
                        distribution=dist,
                        n_ages=n_ages,
                        new_adult_age=new_adult_age,
                    )
                else:
                    array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
                        species=self._species,
                        distribution=dist,
                    )
                if array.size > 0:
                    nz = np.nonzero(
                        array.sum(axis=(0, 1)) if array.ndim == 3 else array
                    )
                    seeds.update(int(z) for z in nz[0])

        # ── Step 3: Seeds from hook genotype refs ──────────────────────
        from natal.frontend.configurator._base import collect_hook_genotype_refs

        hook_strs: set[str] = set()
        for method_name, kwargs in self._declaration_log:
            if method_name == "hooks":
                hook_items = kwargs.get("hook_items", ())
                if hook_items:
                    hook_strs.update(
                        collect_hook_genotype_refs([(tuple(hook_items), {})])
                    )
        resolved = self._resolve_declared_to_ints(
            hook_strs,
            full_registry,
            full_config.n_slabs,
        )
        if resolved:
            seeds.update(resolved)

        # ── Step 4: Seeds from user-declared zygote types ──────────────
        user_decl = self._declared_zygote_types
        if user_decl is not None:
            str_decl: set[str] = set()
            for item in user_decl:
                if isinstance(item, str):
                    str_decl.add(item)
                else:
                    seeds.add(item)  # int — already a ztype index
            if str_decl:
                resolved_decl = self._resolve_declared_to_ints(
                    str_decl,
                    full_registry,
                    full_config.n_slabs,
                )
                if resolved_decl:
                    seeds.update(resolved_decl)

        # ── Step 5: Build combined modifier maps & BFS ─────────────────
        if not _genetics_batch_names(batch_param_names):
            # Ecology variation cannot add genetic edges. The already compiled
            # uncompressed template supplies the complete reachability graph.
            combined_z2g = full_config.zygotes_to_gametes_map
            combined_g2z = full_config.gametes_to_zygotes_map
        else:
            combined_z2g, combined_g2z = self._build_combined_modifier_maps(expanded, full_config)
        _, _, ztype_mask, _ = build_compression_mask(
            combined_z2g,
            combined_g2z,
            full_config.initial_individual_count,
            declared_zygote_types=seeds if seeds else None,
        )

        # ── Step 6: Add BFS survivors to seeds ─────────────────────────
        for old_idx in range(len(ztype_mask)):
            if ztype_mask[old_idx] >= 0:
                seeds.add(old_idx)

        return seeds

    @staticmethod
    def _resolve_declared_to_ints(
        declared: set[str],
        registry: IndexRegistry,
        n_slabs: np.integer | int,
    ) -> set[int] | None:
        """Convert declared genotype strings to slab-expanded ZType indices."""
        if not declared:
            return None
        result: set[int] = set()
        dips = registry.index_to_genotype
        n_slabs_int = int(n_slabs)
        for dg in declared:
            for gt in dips:
                if str(gt) == dg:
                    for s in range(n_slabs_int):
                        result.add(registry.ztype_index(gt, registry.slab_labels[s]))
        return result

    def _build_combined_modifier_maps(
        self,
        expanded: Dict[str, List[Any]],
        full_config: ModelDraft,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sum modifier-applied gamete/zygote maps across all config groups.

        Each group's modifiers are applied independently to the Mendelian
        baseline, and the resulting probability tensors are summed.  The
        combined map's non-zero entries represent all gamete/zygote
        productions possible anywhere in the spatial system — this is the
        adjacency matrix for the unified BFS.
        """
        batch_param_names = sorted(expanded.keys())
        n_demes = len(expanded[batch_param_names[0]])

        # Collect unique group signatures.
        seen: set[tuple[tuple[str, Any], ...]] = set()
        sigs: list[Dict[str, Any]] = []
        for i in range(n_demes):
            sig_key = tuple(
                (name, _make_hashable(expanded[name][i])) for name in batch_param_names
            )
            if sig_key not in seen:
                seen.add(sig_key)
                sigs.append({name: expanded[name][i] for name in batch_param_names})

        # Mendelian baseline from species cache.
        bp = self._species.get_config_blueprint()
        baseline_z2g = bp["zygotes_to_gametes_map"]
        baseline_g2z = bp["gametes_to_zygotes_map"]

        combined_z2g: NDArray[np.float64] = np.zeros_like(baseline_z2g)
        combined_g2z: NDArray[np.float64] = np.zeros_like(baseline_g2z)

        from natal.frontend.configurator._registry_builder import build_registry

        registry = build_registry(self._species)

        for sig in sigs:
            # Replay presets/modifiers to get modifier lists for this group.
            gamete_mods: list[tuple[int, str | None, Any]] = []
            zygote_mods: list[tuple[int, str | None, Any]] = []

            # Do a lightweight replay — only need modifiers.
            cfg = Configurator.for_age_structured(self._species)
            for method_name, kwargs in self._declaration_log:
                if method_name in (
                    "hooks",
                    "initial_state",
                    "setup",
                    "reproduction",
                    "competition",
                    "age_structure",
                    "survival",
                    "fitness",
                    "custom",
                    "with_observation",
                    "migration",
                ):
                    continue  # irrelevant for modifier collection

                # Substitute batch values.
                resolved: Dict[str, Any] = {}
                for key, value in kwargs.items():
                    if key in sig:
                        resolved[key] = sig[key]
                    elif isinstance(value, BatchSetting):
                        first: Any = (
                            cast(BatchSetting[Any], value).first_value()
                        )  # Any: BatchSetting value type is unknown until expansion
                        if first is not None:
                            resolved[key] = first
                    else:
                        resolved[key] = value

                # Apply to temporary configurator.
                method = getattr(cfg, method_name, None)
                if method is None:
                    continue

                if method_name == "presets":
                    raw_list = _object_sequence(
                        resolved.pop("preset_list", ()), name="preset_list"
                    )
                    expanded_presets: list[object] = []
                    for i_p, item in enumerate(raw_list):
                        key = f"_preset_{i_p}"
                        val = sig.get(key)
                        if val is not None:
                            expanded_presets.append(val)
                        elif isinstance(item, BatchSetting):
                            first = cast(BatchSetting[Any], item).first_value()
                            if first is not None:
                                expanded_presets.append(first)
                        else:
                            expanded_presets.append(item)
                    filtered = {k: v for k, v in resolved.items() if v is not None}
                    method(*expanded_presets, **filtered)
                elif method_name == "modifiers":
                    filtered = {k: v for k, v in resolved.items() if v is not None}
                    method(**filtered)
                else:
                    filtered = {k: v for k, v in resolved.items() if v is not None}
                    method(**filtered)

            gamete_mods = cfg.gamete_modifiers
            zygote_mods = cfg.zygote_modifiers

            # Apply the group's recipes through the unified compiler; the
            # derived offspring tensor is not needed for BFS seeding, but
            # routing here keeps exactly one application spelling.
            from natal.frontend.genetics.compile import compile_modifier_maps

            z2g_copy, g2z_copy, _unused_tensor = compile_modifier_maps(
                baseline_z2g,
                baseline_g2z,
                gamete_modifiers=list(gamete_mods),
                zygote_modifiers=list(zygote_mods),
                registry=registry,
                population=None,
            )

            combined_z2g += z2g_copy
            combined_g2z += g2z_copy

        return combined_z2g, combined_g2z

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

    def _detect_and_delegate(
        self,
        method_name: str,
        kwargs: Dict[str, Any],
    ) -> SpatialConfigurator:
        """Detect BatchSetting values in kwargs, store them, and delegate
        concrete (non-batch) values to the template builder's method.

        **Dual-store pattern**::

            Each chainable call does two things simultaneously:

            1. **Record** the raw kwargs (including BatchSetting objects) in
               ``_declaration_log`` — used later by ``_build_template_for_group``
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
        concrete: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchSetting):
                # Store the full per-deme spec for later expansion.
                self._batch_settings[key] = value
                # Feed the first element to the template builder so it
                # can proceed through setup() → … → build() without
                # errors.  The full per-deme list is expanded at build().
                first = cast(BatchSetting[Any], value).first_value()
                if first is not None:
                    concrete[key] = first
            else:
                concrete[key] = value

        # Delegate sanitized kwargs to the template (single store: the
        # decorator's journaling is bypassed).  The journal entry lands
        # only after the template call succeeded: a failed call must not
        # pollute the replayable declaration log (plan 5.1 step 5).
        filtered = {k: v for k, v in concrete.items() if v is not None}
        self._call_template(method_name, **filtered)
        # Record the original call with BatchSetting objects preserved,
        # for full replay in _build_template_for_group.
        self._declaration_log.append((method_name, dict(kwargs)))
        return self

    def _delegate_positional(
        self,
        method_name: str,
        args: tuple[object, ...],
        kwargs: Dict[str, Any],
    ) -> SpatialConfigurator:
        """Like ``_detect_and_delegate`` but accepts positional args.

        Positional args are assumed to never be BatchSetting; only kwargs
        are checked.
        """
        concrete_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchSetting):
                self._batch_settings[key] = value
                # Template builder only understands scalar values —
                # feed it the first element so it can proceed through
                # its own build() pipeline. The full per-deme list is
                # stored in _batch_settings for later expansion.
                first = cast(BatchSetting[Any], value).first_value()
                if first is not None:
                    concrete_kwargs[key] = first
            else:
                concrete_kwargs[key] = value

        filtered = {k: v for k, v in concrete_kwargs.items() if v is not None}
        self._call_template(method_name, *args, **filtered)
        # Journal only after the template call succeeded — failed calls
        # stay out of the replayable declaration log (plan 5.1 step 5).
        self._declaration_log.append((method_name, dict(kwargs)))
        return self

    # ------------------------------------------------------------------
    # Chainable configuration methods
    # ------------------------------------------------------------------

    def setup(
        self,
        name: str = "SpatialPopulation",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
    ) -> SpatialConfigurator:
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
        # stay out of the replayable declaration log (plan 5.1 step 5).
        self._declaration_log.append(("setup", replay_kwargs))
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
    ) -> SpatialConfigurator:
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
    ) -> SpatialConfigurator:
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
    ) -> SpatialConfigurator:
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
        fixed_egg_count: bool = False,
        # Age-structured params
        female_age_based_mating_rate: Optional[Any] = None,
        male_age_based_mating_rate: Optional[Any] = None,
        age_based_reproduction_rate: Optional[Any] = None,
        female_age_based_fertility: Optional[Any] = None,
        sperm_displacement_rate: float = 0.05,
        # Discrete-generation params
        female_adult_mating_rate: float = 1.0,
        male_adult_mating_rate: float = 1.0,
    ) -> SpatialConfigurator:
        """Configure reproduction and mating parameters.

        Args:
            eggs_per_female: Expected offspring per adult female. Accepts ``BatchSetting``.
            sex_ratio: Proportion of female offspring.
            fixed_egg_count: If True, egg count is deterministic.
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
                    "sex_ratio": sex_ratio,
                    "female_adult_mating_rate": female_adult_mating_rate,
                    "male_adult_mating_rate": male_adult_mating_rate,
                },
            )

    def competition(
        self,
        # Age-structured params
        competition_strength: float = 5.0,
        juvenile_growth_mode: Union[int, str, BatchSetting[Any]] = "logistic",
        low_density_growth_rate: Union[float, BatchSetting[Any]] = 6.0,
        age_1_carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
        old_juvenile_carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
        expected_num_new_adult_females: Union[int, None, BatchSetting[Any]] = None,
        equilibrium_distribution: Optional[
            Union[List[float], NDArray[np.float64], BatchSetting[Any]]
        ] = None,
        # Discrete-generation params
        carrying_capacity: Union[int, None, BatchSetting[Any]] = None,
    ) -> SpatialConfigurator:
        """Configure competition and density-dependence.

        Args:
            competition_strength: Relative competition factor for age-1 juveniles
                (age-structured only).
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

    def presets(self, *preset_list: GeneticPreset) -> SpatialConfigurator:
        """Add gene-drive presets (applied during build).

        Each positional argument may be a ``BatchSetting`` of preset objects,
        allowing different demes to receive different presets.

        Args:
            *preset_list: One or more preset objects, or ``BatchSetting``
                instances wrapping per-deme preset values.

        Returns:
            Self for chaining.
        """
        # Detect BatchSetting in positional args.
        concrete_args: list[object] = []
        for i, item in enumerate(preset_list):
            if isinstance(item, BatchSetting):
                self._batch_settings[f"_preset_{i}"] = item
                first = cast(BatchSetting[Any], item).first_value()
                if first is not None:
                    concrete_args.append(first)
            else:
                concrete_args.append(item)

        self._declaration_log.append(("presets", {"preset_list": preset_list}))
        # concrete_args contains GeneticPreset instances resolved from potential
        # BatchSetting wrappers; cast needed because first_value() returns object.
        self._call_template("presets", *cast("list[GeneticPreset]", concrete_args))
        return self

    def fitness(
        self,
        viability: Optional[Any] = None,
        fecundity: Optional[Any] = None,
        sexual_selection: Optional[Any] = None,
        zygote_viability: Optional[Any] = None,
        mode: str = "replace",
    ) -> SpatialConfigurator:
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
    ) -> SpatialConfigurator:
        """Register custom named slots on every deme's draft.

        Custom slots are container-uniform: the kwargs are replayed onto
        each group template at build time, so all demes carry the same
        values (per-deme custom slots are not a batch_setting axis).
        Values follow the panmictic ``Configurator.custom`` contract and
        reach the Rust session via ``Params.custom_slots``.

        Args:
            **kwargs: Name-value pairs for custom slots.  Values must be
                ``bool``, ``int``, ``float``, or ``NDArray[np.float64]``.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate("custom", dict(kwargs))

    def hooks(self, *hook_items: _HookItem) -> SpatialConfigurator:
        """Register lifecycle hooks.

        Args:
            *hook_items: Functions decorated with ``@hook`` or hook mappings.

        Returns:
            Self for chaining.
        """
        self._declaration_log.append(("hooks", {"hook_items": hook_items}))
        self._call_template("hooks", *hook_items)
        return self

    def modifiers(
        self,
        gamete_modifiers: Optional[
            List[Tuple[int, Optional[str], Callable[..., object]]]
        ] = None,
        zygote_modifiers: Optional[
            List[Tuple[int, Optional[str], Callable[..., object]]]
        ] = None,
    ) -> SpatialConfigurator:
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
    ) -> SpatialConfigurator:
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
            SpatialConfigurator: Self for chaining.

        Raises:
            RuntimeError: When called on a runtime Configurator.
            TypeError: If groups is not a mapping of selectors.
            ValueError: If groups, a group label, the deme mode, or the deme
                selection is invalid.
        """
        if self._pop_ref is not None:
            raise RuntimeError(
                "with_observation() is only valid during the build phase. "
                "Observation rules cannot change after the Population has been built."
            )
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
    ) -> SpatialConfigurator:
        """Set the recording mode and capacity for spatial population history.

        Must be called during the build phase.  Calling this on a runtime
        Configurator raises ``RuntimeError``.

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
            RuntimeError: When called on a runtime Configurator.
            ValueError: When mode is invalid or ``max_rows`` is less than one.
        """
        if self._pop_ref is not None:
            raise RuntimeError(
                "record_history() is only valid during the build phase. "
                "Recording settings cannot change after the Population has "
                "been built."
            )
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
        migration_rate: float = 0.0,
        strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = "auto",
        adjacency: Optional[
            object
        ] = None,  # object: adjacency matrix (NDArray, list, or None) — duck-typed
        kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None,
        deme_kernel_ids: Optional[NDArray[np.int64]] = None,
        kernel_include_center: bool = False,
        adjust_migration_on_edge: bool = False,
    ) -> SpatialConfigurator:
        """Configure spatial migration parameters.

        Args:
            kernel: Odd-shaped 2D migration kernel.
            migration_rate: Fraction of each deme that migrates.
            strategy: Migration strategy (``"auto"``, ``"adjacency"``,
                ``"kernel"``, ``"hybrid"``).
            adjacency: Explicit adjacency matrix.
            kernel_bank: Optional heterogeneous kernel bank.
            deme_kernel_ids: Per-deme kernel ids into ``kernel_bank``.
            kernel_include_center: Whether kernel includes center cell.
            adjust_migration_on_edge: Whether to adjust migration rates on
                boundaries. When False (default), boundary demes migrate less
                due to fewer valid neighbors. When True, all demes have the
                same total migration rate regardless of position.

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
        # Keep the raw declaration: scalar / vector / per-sex mapping are
        # normalized once by the SpatialPopulation constructor (slice 5).
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
        # constructor (slice 5), not here.
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

    def build(self) -> SpatialPopulation:
        """Build and return the configured ``SpatialPopulation``.

        Single entry point (slice-5 stage 2): one build path handles both
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
        return self._build_from_definition(definition, cached_template=self._template._compiled_model)  # pyright: ignore[reportPrivateUsage]  # avoid re-executing recipes already compiled by the chain.

    def _definition_for_compile(self) -> ModelDefinition:
        """Freeze concrete spatial controls before creating any execution session."""
        from dataclasses import replace

        from natal.frontend.data.definition import (
            ModelDefinition,
            SpatialInputs,
            copy_declaration_value,
        )

        inputs = self._template._definition_for_compile().normalized  # pyright: ignore[reportPrivateUsage]  # shared template compiler input.
        assert inputs is not None
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
        controls = SpatialInputs(
            self._n_demes, self._topology, self._pop_type, self._spatial_name,
            tuple(expanded.items()),
            tuple((name, normalize(kwargs)) for name, kwargs in self._declaration_log),
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
        return ModelDefinition(self._species, self._pop_type == "discrete_generation", tuple(self._declaration_log), self._spatial_name, normalized=replace(inputs, spatial=controls))

    @classmethod
    def _build_from_definition(
        cls, definition: ModelDefinition, *, cached_template: CompiledModel | None = None,
    ) -> SpatialPopulation:
        """Compile a detached normalized definition using the existing group compiler."""
        inputs = definition.normalized
        if inputs is None or inputs.spatial is None:
            raise ValueError("Spatial compilation requires normalized spatial inputs")
        controls = inputs.spatial
        compiler = cls(definition.species, controls.n_demes, controls.topology, pop_type=controls.pop_type)
        template = Configurator(inputs.settings, species=definition.species)
        template._registry = inputs.registry  # pyright: ignore[reportPrivateUsage]  # initialize one isolated compiler candidate.
        template._presets = list(inputs.presets)  # pyright: ignore[reportPrivateUsage]
        template._manual_gamete = cast("GameteList", list(inputs.manual_gamete))  # pyright: ignore[reportPrivateUsage]
        template._manual_zygote = cast("ZygoteList", list(inputs.manual_zygote))  # pyright: ignore[reportPrivateUsage]
        template._fitness_base = inputs.fitness_base  # pyright: ignore[reportPrivateUsage]
        template._fitness_steps = list(inputs.fitness_steps)  # pyright: ignore[reportPrivateUsage]
        template._compilation_key = inputs.compilation_key  # pyright: ignore[reportPrivateUsage]
        template._compiled_model = cached_template  # pyright: ignore[reportPrivateUsage]
        template._hook_calls = list(inputs.hook_calls)  # pyright: ignore[reportPrivateUsage]
        template._observation_groups = inputs.observation_groups  # pyright: ignore[reportPrivateUsage]
        template._observation_collapse_age = inputs.observation_collapse_age  # pyright: ignore[reportPrivateUsage]
        template._record_history_mode = inputs.history_mode  # pyright: ignore[reportPrivateUsage]
        template._record_history_max_rows = inputs.history_max_rows  # pyright: ignore[reportPrivateUsage]
        template._compress = inputs.compress  # pyright: ignore[reportPrivateUsage]
        template._declared_zygote_types = None if inputs.declared_zygote_types is None else cast("set[str] | set[int]", set(inputs.declared_zygote_types))  # pyright: ignore[reportPrivateUsage]
        compiler._template = template
        compiler._batch_settings = {name: BatchSetting(values) for name, values in controls.batch_values}
        compiler._declaration_log = list(controls.group_calls)
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
            migration_rate=self._migration_rate,
            adjust_migration_on_edge=self._adjust_migration_on_edge,
            name=self._spatial_name,
        )
        spatial._definition = definition  # pyright: ignore[reportPrivateUsage]  # attach the actual input consumed by this compilation.
        # The Rust engine is the ONLY execution backend (plan S6): the
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
        tpl_config = template.export_config()

        demes: List[PopulationInstance] = [template]
        for i in range(1, self._n_demes):
            clone = _clone_deme(
                template,
                config=tpl_config,
                name=f"{self._spatial_name}_deme_{i}",
            )
            demes.append(clone)
        return demes

    def _build_heterogeneous_demes(self) -> List[PopulationInstance]:
        """Group demes by *genetics* signature, build one template per
        group, then derive per-deme ecology via ``_replace`` shells.

        **Grouping algorithm**::

            1. Expand every ``BatchSetting`` → per-deme value list.
            2. For each deme, build a hashable signature from its
               *genetics-section* batch values (fitness rows, presets,
               modifiers — see ``_genetics_batch_names``).
            3. Demes with identical genetics signatures share one group
               template.  Ecology batch values are excluded from the
               signature: a K gradient across 2601 demes produces ONE
               group, not 2601.

            4. Within a group, non-first demes receive their own ecology
               through ``_build_variant_config`` (``_replace`` shallow
               copies sharing all heavy arrays) or a full builder replay
               when a value cannot be applied by ``_replace``.
        """
        # 1. Expand every BatchSetting → concrete per-deme list.
        #    e.g. K=batch_setting([10000,5000,5000,8000]) → [10000, 5000, 5000, 8000]
        expanded: Dict[str, List[Any]] = {}
        for param_name, batch in self._batch_settings.items():
            expanded[param_name] = batch.expand(self._n_demes, self._topology)

        # 1a. If compression is enabled, compute union declared ztype indices.
        union_declared: set[int] | None = None
        if self._compress:
            union_declared = self._compress_once(expanded)

        # 2. Hash each deme's values into two signatures: a *genetics*
        #    signature deciding group membership (ndarray values → bytes;
        #    dict values → sorted kv tuples) and the full-value signature
        #    deciding whether a member can plain-clone its group template.
        all_param_names = sorted(expanded.keys())
        genetics_param_names = _genetics_batch_names(all_param_names)
        genetics_signatures: List[tuple[tuple[str, Any], ...]] = []
        full_signatures: List[tuple[tuple[str, Any], ...]] = []
        for i in range(self._n_demes):
            genetics_signatures.append(
                tuple(
                    (name, _make_hashable(expanded[name][i]))
                    for name in genetics_param_names
                )
            )
            full_signatures.append(
                tuple(
                    (name, _make_hashable(expanded[name][i]))
                    for name in all_param_names
                )
            )

        # 3. Group deme indices by genetics signature.
        #    [1,1,1,...,2,...,1] → 2 groups, not n_demes groups.
        groups: defaultdict[tuple[tuple[str, Any], ...], List[int]] = defaultdict(list)
        for idx, sig in enumerate(genetics_signatures):
            groups[sig].append(idx)

        # 4. Build one template per genetics group.  The first group always
        #    runs the full builder pipeline.  Every deme whose full value
        #    map differs derives its ecology through a ``_replace`` shell
        #    (heavy ndarrays stay shared) or a full replay fallback.
        demes: List[PopulationInstance] = [None] * self._n_demes  # type: ignore[list-item]  # None placeholder; each slot filled before return
        base_config: ModelDraft | None = None
        base_template: Optional[PopulationInstance] = (
            None  # template deme from first group — cloned via _clone_deme
        )

        for _sig, indices in groups.items():
            first_idx = indices[0]
            sig_map: Dict[str, Any] = {
                name: expanded[name][first_idx] for name in all_param_names
            }

            if base_config is None:
                group_template = self._build_template_for_group(
                    sig_map,
                    extra_declared=union_declared,
                )
                base_config = group_template.export_config()
                base_template = group_template
            elif self._can_use_replace(sig_map, base_config):
                # Fast path: only scalar / known-array fields differ from
                # the first group's template.
                assert base_template is not None  # set in first-group branch above
                group_template = self._deme_from_replace(
                    sig_map,
                    base_config,
                    base_template,
                    first_idx,
                )
            else:
                # Fallback: parameter not recognized by _can_use_replace
                # (e.g. fitness dict, custom modifier). Full builder replay —
                # all arrays freshly allocated, no sharing with base_config.
                group_template = self._build_template_for_group(
                    sig_map,
                    extra_declared=union_declared,
                )

            demes[first_idx] = group_template

            # Remaining demes of this genetics group share its genetics;
            # each derives its own ecology (K gradients, initial states, …)
            # without inflating the number of full template builds.
            # base_config was assigned in the first-group branch above, so
            # every member of a later group can derive from it.
            group_config = group_template.export_config()
            for idx in indices[1:]:
                if full_signatures[idx] == full_signatures[first_idx]:
                    # Identical values: plain clone sharing the group's
                    # config by reference (only state arrays are copies).
                    demes[idx] = _clone_deme(
                        group_template,
                        config=group_config,
                        name=f"{self._spatial_name}_deme_{idx}",
                    )
                elif self._can_use_replace(
                    {name: expanded[name][idx] for name in all_param_names},
                    base_config,
                ):
                    demes[idx] = self._deme_from_replace(
                        {name: expanded[name][idx] for name in all_param_names},
                        base_config,
                        cast(PopulationInstance, base_template),
                        idx,
                    )
                else:
                    demes[idx] = self._build_template_for_group(
                        {name: expanded[name][idx] for name in all_param_names},
                        extra_declared=union_declared,
                    )

        return demes

    def _deme_from_replace(
        self,
        value_map: Dict[str, Any],
        base_config: ModelDraft,
        base_template: PopulationInstance,
        deme_idx: int,
    ) -> PopulationInstance:
        """Derive one deme from the base template via a ``_replace`` shell.

        The shell shares every unmodified ndarray with *base_config*;
        only the deme-specific values (initial state, ecology scalars)
        are new.  State arrays and the reset snapshot carry the deme's
        own initial values.

        Args:
            value_map: The deme's concrete batch values.
            base_config: The first group's template config.
            base_template: The first group's template deme.
            deme_idx: Deme index used for the deme name.

        Returns:
            A new deme population instance.
        """
        variant_config = self._build_variant_config(
            value_map,
            base_config,
            species=self._species,
            pop_type=self._pop_type,
        )
        deme = _clone_deme(
            base_template,
            config=variant_config,
            name=f"{self._spatial_name}_deme_{deme_idx}",
        )
        # _clone_deme copies state arrays from base_template; overwrite
        # them with the deme's own initial values.
        state = deme._live_state()  # pyright: ignore[reportPrivateUsage]  # live container: overwrite reaches the engine
        if "individual_count" in value_map:
            state.individual_count[:] = variant_config.initial_individual_count
        if "sperm_storage" in value_map:
            ss = getattr(state, "sperm_storage", None)
            if ss is not None:
                ss[:] = variant_config.initial_sperm_storage
        # Update snapshot so reset() restores this deme's initial state.
        ss_snap = getattr(state, "sperm_storage", None)
        object.__setattr__(
            deme,
            "_initial_population_snapshot",
            (
                state.individual_count.copy(),
                ss_snap.copy() if ss_snap is not None else None,
                None,
            ),
        )
        return deme

    @staticmethod
    def _can_use_replace(sig_map: Dict[str, object], base_config: ModelDraft) -> bool:
        """Return True if every kwarg in *sig_map* can be applied via ``_replace``.

        ``_replace`` is a NamedTuple shallow copy — it creates a new config
        where only the specified fields differ; all other fields (including
        heavy ndarrays like genotype maps, fitness tensors, survival vectors)
        share the same memory as *base_config*.

        This check gates whether a group can use the fast ``_replace`` path
        or must fall back to a full builder replay.  A kwarg qualifies if it
        appears in ``_ARRAY_KWARGS``, ``_KWARG_MULTI_FIELD``,
        ``_KWARG_RENAMES``, or exists as a direct field name on
        ``ModelDraft``.
        """
        for name in sig_map:
            if name in _ARRAY_KWARGS:
                continue
            if name in _KWARG_RENAMES:
                continue
            # Dynamic: try direct field name match on the config object
            if hasattr(base_config, name):
                continue
            return False
        return True

    @staticmethod
    def _build_variant_config(
        sig_map: Dict[str, object],
        base_config: ModelDraft,
        *,
        species: Species,
        pop_type: str = "age_structured",
    ) -> ModelDraft:
        """Create a variant config via ``_replace``, sharing all heavy arrays.

        ``ModelDraft`` is a NamedTuple.  ``_replace(**kwargs)`` creates
        a **shallow copy**: fields named in *kwargs* get new values; every
        other field keeps its original reference.  This means genotype maps,
        fitness tensors, survival vectors, and all other unchanging ndarrays
        are shared between *base_config* and the returned variant — no copy,
        no extra memory.

        Dispatch order (only fields in *sig_map* are touched):

        1. **Array kwargs** (individual_count, sperm_storage) —
           convert the per-group dict to a **new ndarray** (this array
           genuinely differs between groups), then ``_replace`` it.
        2. **Multi-field kwargs** (carrying_capacity variants) —
           ``_replace`` both the base and population-scale fields.
        3. **Rename kwargs** (eggs_per_female → eggs_per_female) —
           ``_replace`` under the config-side field name.
        4. **Any other kwarg** — direct ``_replace`` by field name
           (pre-validated by ``_can_use_replace``).

        Equilibrium metrics are recomputed when capacity / eggs / sex-ratio
        change, since these affect the equilibrium competition strength.

        Args:
            sig_map: Mapping from batch kwarg name to group's concrete value.
            base_config: The base ``ModelDraft`` to derive from.
            species: ``Species`` instance, needed for genotype resolution.
            pop_type: ``"age_structured"`` or ``"discrete_generation"``.

        Returns:
            A new ``ModelDraft`` sharing all unchanged array references
            with *base_config*.
        """
        from natal.frontend.configurator import PopulationConfigBuilder

        replace_kwargs: Dict[
            str, Any
        ] = {}  # Any: config field values (int, float, ndarray, bool)

        for kwarg, raw_val in sig_map.items():
            # sig_map values are genuinely polymorphic (float, int, dict, …);
            # their correctness is pre-validated by _can_use_replace.
            val = raw_val

            # --- 1. array-valued: dict → array conversion ---
            if kwarg == "individual_count":
                distribution = cast(InitialIndividualCountInput, val)
                if pop_type == "age_structured":
                    array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
                        species=species,
                        distribution=distribution,
                        n_ages=int(base_config.n_ages),
                        new_adult_age=int(base_config.new_adult_age),
                    )
                else:
                    array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
                        species=species,
                        distribution=distribution,
                    )
                replace_kwargs["initial_individual_count"] = array
                continue

            if kwarg == "sperm_storage":
                if pop_type == "age_structured":
                    sperm_storage = cast(InitialSpermStorageInput, val)
                    array = PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage(
                        species=species,
                        sperm_storage=sperm_storage,
                        n_ages=int(base_config.n_ages),
                        new_adult_age=int(base_config.new_adult_age),
                    )
                    replace_kwargs["initial_sperm_storage"] = array
                continue

            # --- 1b. discrete scalars: one cell of a copied unified vector ---
            if kwarg in _DISCRETE_VECTOR_CELLS:
                field_name, cell = _DISCRETE_VECTOR_CELLS[kwarg]
                arr = np.array(getattr(base_config, field_name), dtype=np.float64)
                # sig_map values are pre-validated scalars (see _can_use_replace).
                arr[cell] = float(val)  # type: ignore[reportArgumentType]  # BatchSetting already expanded upstream
                replace_kwargs[field_name] = arr
                continue

            # --- 2. rename ---
            config_field = _KWARG_RENAMES.get(kwarg, kwarg)
            # Wrap scalar values for 0-d ndarray config fields.
            if _is_0d_field(base_config, config_field) and not isinstance(
                val, np.ndarray
            ):
                replace_kwargs[config_field] = np.array(
                    _float_value(val, name=config_field)
                )
            else:
                replace_kwargs[config_field] = val

        variant = base_config._replace(**replace_kwargs)

        return variant

    def _resolved_group_journal(self, values: Mapping[str, object]) -> list[tuple[str, dict[str, Any]]]:
        """Record concrete group inputs without executing their declarations again."""
        # Any: journal arguments include opaque recipe objects and nested selectors.
        result: list[tuple[str, dict[str, Any]]] = []
        for method, kwargs in self._declaration_log:
            resolved = {key: values.get(key, value) for key, value in kwargs.items()}
            if method in ("presets", "hooks"):
                source = "preset_list" if method == "presets" else "hook_items"
                resolved["__args__"] = tuple(_object_sequence(resolved.pop(source, ()), name=source))
            result.append((method, resolved))
        return result

    def _build_template_for_group(
        self,
        sig_map: Dict[str, object],
        *,
        extra_declared: set[int] | None = None,
        compress: bool | None = None,
    ) -> PopulationInstance:
        """Build a single template deme for one config-signature group.

        Creates a fresh panmictic builder and replays every method call
        recorded in ``_declaration_log``, substituting ``BatchSetting`` values
        with the group-specific concrete values from *sig_map*.

        Args:
            sig_map: Mapping from batch parameter name to the group's
                concrete value.
            extra_declared: Additional ztype indices to protect from
                compression pruning (union seeds across all demes).
        """
        # Ecology-only variants reuse the template's compiled genetics,
        # including the uncompressed pass used to collect global BFS seeds.
        # Cloning the configurator preserves opaque recipe identities without
        # repeating their effects; build snapshots every owned array itself.
        if not _genetics_batch_names(list(sig_map)) and self._can_use_replace(sig_map, self._template.config):
            from copy import copy

            template_cfg = copy(self._template)
            template_cfg._declaration_log = self._resolved_group_journal(sig_map)  # pyright: ignore[reportPrivateUsage]  # cached products still retain the concrete group declaration.
            template_cfg._config = self._build_variant_config(  # pyright: ignore[reportPrivateUsage]  # isolated group candidate.
                sig_map, self._template.config, species=self._species, pop_type=self._pop_type,
            )
            if compress is not None:
                template_cfg._compress = compress  # pyright: ignore[reportPrivateUsage]
            if extra_declared:
                template_cfg._declared_zygote_types = set(extra_declared)  # pyright: ignore[reportPrivateUsage]
            result = template_cfg.build(name=f"{self._spatial_name}_group")
            # A cold compile creates products once; subsequent ecology groups
            # and the post-BFS build can reuse them under the same input key.
            self._template._compiled_model = template_cfg._compiled_model  # pyright: ignore[reportPrivateUsage]
            return result
        if self._pop_type == "age_structured":
            template_cfg = Configurator.for_age_structured(self._species)
        else:
            template_cfg = Configurator.for_discrete(self._species)

        for method_name, kwargs in self._declaration_log:
            method = getattr(template_cfg, method_name, None)
            if method is None:
                continue

            resolved: Dict[str, object] = {}
            for key, value in kwargs.items():
                if key in sig_map:
                    resolved[key] = sig_map[key]
                elif isinstance(value, BatchSetting):
                    first = cast(BatchSetting[Any], value).first_value()
                    if first is not None:
                        resolved[key] = first
                else:
                    resolved[key] = value

            # Merge union seeds into the setup call's declared_zygote_types.
            # extra_declared already includes user-declared types (resolved by
            # _compress_once), hook refs, and BFS survivors — it is the complete
            # seed set.  Replace the replay log's declared_zygote_types outright.
            if extra_declared and method_name == "setup":
                resolved["declared_zygote_types"] = list(extra_declared)

            # Override compress flag (used by _compress_once to build
            # without compression).
            if compress is not None and method_name == "setup":
                resolved["compress"] = compress

            # Handle positional args (presets, hooks).
            if method_name == "presets":
                raw_preset_list = _object_sequence(
                    resolved.pop("preset_list", ()), name="preset_list"
                )
                expanded_presets: list[object] = []
                for i, item in enumerate(raw_preset_list):
                    key = f"_preset_{i}"
                    preset_val = sig_map.get(key)
                    if preset_val is not None:
                        expanded_presets.append(preset_val)
                    elif isinstance(item, BatchSetting):
                        first = cast(BatchSetting[Any], item).first_value()
                        if first is not None:
                            expanded_presets.append(first)
                    elif item is not None:
                        expanded_presets.append(item)
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(*expanded_presets, **filtered)
            elif method_name == "hooks":
                hook_items = _object_sequence(
                    resolved.pop("hook_items", ()), name="hook_items"
                )
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(*hook_items, **filtered)
            else:
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(**filtered)

        return template_cfg.build(name=f"{self._spatial_name}_group")

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
        spatial._recording_plan = plan  # type: ignore[reportPrivateUsage]  # configurator sets private attr on spatial
        spatial._history_obj = History(plan.schema, max_rows=max_rows)  # type: ignore[reportPrivateUsage]  # configurator sets private attr
