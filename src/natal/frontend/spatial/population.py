"""Composition-based spatial population container.

`SpatialPopulation` intentionally does NOT inherit from ``BasePopulation``.
Each deme is represented by one concrete ``BasePopulation`` subclass instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.contracts.blueprint import Blueprint
from natal.contracts.materialize import SpatialMigration, materialize
from natal.contracts.params import Params
from natal.frontend.data import (
    DiscretePopulationState,
    ModelDefinition,
    ModelDraft,
    PopulationState,
)
from natal.frontend.genetics import Species
from natal.frontend.hooks import (
    CompiledHookDescriptor,
    DemeSelector,
    HookProgram,
    OpType,
)
from natal.frontend.population.base import BasePopulation
from natal.frontend.spatial.migration import (
    MigrationCSR,
    RateDeclaration,
    csr_dense_row,
    fold_migration_csr,
    normalize_migration_rate,
    resolve_migration_mode,
)
from natal.frontend.spatial.topology import (
    GridTopology,
    build_adjacency_matrix,
)

if TYPE_CHECKING:
    from natal.backends.rust.rust_backend import (
        RustHeterogeneousSpatialLifecycleBackend,
    )
    from natal.frontend.output.history import History
    from natal.frontend.output.observation import Observation, ObservationResult
    from natal.frontend.spatial.configurator import SpatialConfigurator

__all__ = ["SpatialPopulation"]

ConfigObject: TypeAlias = object
SpatialStateTuple: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64], int]
DemePopulation: TypeAlias = (
    BasePopulation[PopulationState] | BasePopulation[DiscretePopulationState]
)


def _coerce_adjacency_dense(
    adjacency: object,
    n_demes: int,
) -> NDArray[np.float64]:
    """Coerce dense or sparse-like adjacency input to a dense float64 matrix.

    Supported forms:

    - Dense ``np.ndarray`` with shape ``(n_demes, n_demes)``.
    - CSR tuple ``(indptr, indices, data)``.
    - Objects exposing ``toarray()`` (for example scipy sparse matrices).

    Args:
        adjacency: User-provided adjacency input.
        n_demes: Number of demes expected on each matrix axis.

    Returns:
        A dense ``float64`` adjacency matrix.

    Raises:
        TypeError: If input type is unsupported.
        ValueError: If shapes or sparse indices are invalid.
    """
    # Normalize user input early so downstream code can assume one concrete
    # ndarray representation regardless of original input form.
    adjacency_obj = adjacency

    if isinstance(adjacency_obj, np.ndarray):
        # Dense mode is interpreted as a square matrix.
        dense_arr = cast(np.ndarray, adjacency_obj)
        if dense_arr.shape != (n_demes, n_demes):
            raise ValueError(f"adjacency array must be {n_demes}x{n_demes}")
        dense = np.asarray(dense_arr, dtype=np.float64)
    elif isinstance(adjacency_obj, tuple):
        # Tuple mode is interpreted as CSR triplet: (indptr, indices, data).
        csr_items = cast(tuple[object, ...], adjacency_obj)
        if len(csr_items) != 3:
            raise TypeError("adjacency tuple input must be CSR (indptr, indices, data)")
        csr_tuple = csr_items
        indptr = np.asarray(csr_tuple[0], dtype=np.int64)
        indices = np.asarray(csr_tuple[1], dtype=np.int64)
        data = np.asarray(csr_tuple[2], dtype=np.float64)

        if indptr.ndim != 1 or indices.ndim != 1 or data.ndim != 1:
            raise ValueError("CSR adjacency tuple entries must be 1D arrays")
        if indptr.shape[0] != n_demes + 1:
            raise ValueError(
                f"CSR indptr length mismatch: expected {n_demes + 1}, got {indptr.shape[0]}"
            )
        if indices.shape[0] != data.shape[0]:
            raise ValueError(
                f"CSR indices/data length mismatch: {indices.shape[0]} vs {data.shape[0]}"
            )
        if int(indptr[0]) != 0 or int(indptr[-1]) != indices.shape[0]:
            raise ValueError("CSR indptr must start at 0 and end at nnz")
        for pos in range(indptr.shape[0] - 1):
            if int(indptr[pos + 1]) < int(indptr[pos]):
                raise ValueError("CSR indptr must be non-decreasing")

        # Here, we rebuild the dense matrix from CSR.
        # This should be efficient enough for small matrices, but may be a bottleneck
        # for very large grids with complex migration patterns.
        # TODO(spatial-migration/sparse): Add a direct sparse path.
        dense = np.zeros((n_demes, n_demes), dtype=np.float64)
        for src in range(n_demes):
            start = int(indptr[src])
            end = int(indptr[src + 1])
            for item_idx in range(start, end):
                # CSR rows may contain repeated destinations; accumulate.
                dst = int(indices[item_idx])
                if dst < 0 or dst >= n_demes:
                    raise ValueError(
                        f"CSR destination index out of range at position {item_idx}: {dst}"
                    )
                dense[src, dst] += data[item_idx]
    else:
        # Sparse-matrix compatibility path (e.g. scipy.sparse).
        # TODO(spatial-migration/sparse): Add a direct sparse path.
        toarray_fn = getattr(adjacency_obj, "toarray", None)
        if not callable(toarray_fn):
            raise TypeError(
                "adjacency must be a dense ndarray, a CSR tuple (indptr, indices, data), "
                "or an object exposing toarray()"
            )
        dense = np.asarray(toarray_fn(), dtype=np.float64)

    if dense.shape != (n_demes, n_demes):
        raise ValueError(
            f"adjacency shape mismatch: expected ({n_demes}, {n_demes}), got {dense.shape}"
        )

    return dense


class DemeSlice:
    """Compatible view of one deme over the spatial SoA contract.

    Slice-5 stage 3: ``spatial.deme(i)`` returns this view instead of the
    raw per-deme population.  Reads are fully compatible — every missing
    attribute (``config``, ``state``, ``registry``, ``name``, …) is
    delegated to the underlying deme object, so UI code and the ~15 test
    files that consume the deep ``.config`` / ``.state`` interfaces keep
    working unchanged.

    Writes follow the stage-3 data plane:

    - ``write_ecology`` writes the deme's ecology column entry AND the
      deme's draft (per-field clone-on-write), so the Rust session columns
      and every Python reader see the same value.
    - ``write_genetics`` forks the deme's genetics variant (Rust bank) and
      clones the draft arrays, so divergence at one deme never leaks into
      the demes that previously shared its tables.

    ``params`` and ``update()`` commit through the owning spatial session.
    Lifecycle, state import, and history controls belong to the container;
    a managed deme cannot start, restore, reset, or finish a separate run.
    """

    def __init__(self, pop: SpatialPopulation, index: int) -> None:
        """Bind the slice to one deme of a spatial population.

        Args:
            pop: The owning spatial population.
            index: Zero-based deme index.
        """
        self._pop = pop
        self._index = index

    @property
    def index(self) -> int:
        """int: Zero-based deme index of this slice."""
        return self._index

    def __setattr__(self, name: str, value: object) -> None:
        """Forward attribute writes to the underlying deme.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ("_pop", "_index"):
            object.__setattr__(self, name, value)
            return
        deme = object.__getattribute__(self, "_pop")._demes[  # pyright: ignore[reportPrivateUsage]  # delegated write channel
            object.__getattribute__(self, "_index")
        ]
        setattr(deme, name, value)

    def __delattr__(self, name: str) -> None:
        """Forward attribute deletion to the underlying deme.

        Args:
            name: Attribute name.
        """
        if name in ("_pop", "_index"):
            object.__delattr__(self, name)
            return
        deme = object.__getattribute__(self, "_pop")._demes[  # pyright: ignore[reportPrivateUsage]  # delegated write channel
            object.__getattribute__(self, "_index")
        ]
        delattr(deme, name)

    # -- delegated reads (full backward compatibility) ----------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying deme.

        The return type is ``Any`` on purpose: the slice's attribute
        surface is exactly the wrapped population's (config, state,
        registry, hooks, ...), which is statically unknowable from the
        view; consumers keep their existing concrete types through the
        delegation.

        Args:
            name: Attribute name.

        Returns:
            The deme attribute value.
        """
        population = object.__getattribute__(self, "_pop")
        population._ensure_rust_states_fresh()
        deme = population._demes[object.__getattribute__(self, "_index")]
        value = getattr(deme, name)
        if callable(value):
            # A user may retain a bound export/count method across a run.
            # Refresh when it is called, not only when the method is fetched.
            def delegated(*args: Any, **kwargs: Any) -> Any:  # Any: preserve the dynamically delegated population method signature
                population._ensure_rust_states_fresh()
                return value(*args, **kwargs)
            return delegated
        return value

    @property
    def config(self) -> ModelDraft:
        """ModelDraft: The deme's live draft (read projection; see class notes)."""
        return self._pop._demes[self._index].config  # pyright: ignore[reportPrivateUsage]  # delegated read channel

    @property
    def state(self) -> PopulationState | DiscretePopulationState:
        """The deme's state snapshot (plan S3 snapshot discipline).

        When the container Rust backend is enabled, the session owns the
        authoritative stacked state; reading refreshes the per-deme caches
        from one session snapshot, then returns an independent copy so a
        retained reference cannot mutate the real run state.  State
        modifications belong to initial-state declarations or a callback's
        ``TickContext.state`` transaction.
        """
        pop = self._pop
        pop._ensure_rust_states_fresh()  # pyright: ignore[reportPrivateUsage]  # container lazy refresh
        deme = pop._demes[self._index]  # pyright: ignore[reportPrivateUsage]  # slot access implies a constructed deme
        assert deme is not None  # constructed demes always carry a container
        state = deme._live_state()  # pyright: ignore[reportPrivateUsage]  # fresh after the refresh above
        assert state is not None
        # Independent copy: arrays handed out must not alias the caches.
        replacements: dict[str, NDArray[np.float64]] = {
            name: np.array(getattr(state, name), copy=True)
            for name in ("individual_count", "sperm_storage")
            if getattr(state, name, None) is not None
        }
        return state._replace(**replacements)

    # -- stage-3 write path --------------------------------------------------

    def write_ecology(self, field: str, value: float | NDArray[np.float64]) -> None:
        """Write one ecology value for this deme (column + draft + Rust).

        Args:
            field: Contract ecology field name (``carrying_capacity``,
                ``eggs_per_female``, ``sex_ratio``, ``survival_rates``, …).
            value: New scalar or array in the contract's logical shape.

        Raises:
            KeyError: If *field* is not an ecology field.
        """
        self._pop._write_deme_ecology(self._index, field, value)  # pyright: ignore[reportPrivateUsage]  # single validated write channel

    def write_genetics(self, field: str, values: NDArray[np.float64]) -> None:
        """Write one genetics tensor for this deme, forking the variant.

        The deme's bank variant is cloned first (Rust side) and the draft
        tables are detached, so demes that previously shared this genetics
        keep their numerics bitwise unchanged.

        Args:
            field: Contract genetics tensor name (one of the eight tables).
            values: New contents in the table's logical shape.
        """
        self._pop._write_deme_genetics(self._index, field, values)  # pyright: ignore[reportPrivateUsage]  # single validated write channel

    def __repr__(self) -> str:
        """Return a debug representation pointing at the deme index."""
        return f"DemeSlice(index={self._index}, pop={self._pop.name!r})"


def _minimal_contract(
    *,
    n_demes: int,
    n_sexes: int,
    n_ages: int,
    migration_csr: MigrationCSR,
    rate3d: NDArray[np.float64],
) -> tuple[Blueprint, Params]:
    """Build a minimal spatial contract without a reference draft.

    Only used for lightweight deme doubles that predate the contract
    era (test fixtures): dimensions come from the state shapes and the
    genetics section stays empty.

    Args:
        n_demes: Number of demes.
        n_sexes: Number of sexes.
        n_ages: Number of age classes.
        migration_csr: Folded migration CSR.
        rate3d: ``(n_demes, n_sexes, n_ages)`` rate column.

    Returns:
        A ``(Blueprint, Params)`` pair with placeholder genetics.
    """
    from natal.contracts.blueprint import frozen

    blueprint = Blueprint(
        n_sexes=n_sexes,
        n_ages=n_ages,
        n_ztypes=0,
        n_gtypes=0,
        n_glabs=0,
        new_adult_age=0,
        adult_ages=frozen(np.zeros(0, dtype=np.int64)),
        stochastic=False,
        continuous_sampling=False,
        fixed_egg_count=False,
        has_sex_chromosomes=False,
        extreme_speed_mode=0,
        ztype_names=(),
        gtype_names=(),
        female_only_by_sex_chrom=frozen(np.zeros(0, dtype=np.bool_)),
        male_only_by_sex_chrom=frozen(np.zeros(0, dtype=np.bool_)),
        initial_individual_count=frozen(np.zeros((0,), dtype=np.float64)),
        initial_sperm_storage=frozen(np.zeros((0,), dtype=np.float64)),
        n_demes=n_demes,
        # np.array (not asarray): the CSR inputs are already int64/float64,
        # so asarray would return the CALLER's buffers — freezing those
        # would both pollute the caller and leave the blueprint mutable
        # through any holder that re-enables the write flag.
        migration_indptr=frozen(np.array(migration_csr.indptr, dtype=np.int64)),
        migration_dest_idx=frozen(np.array(migration_csr.dest_idx, dtype=np.int64)),
        migration_weights=frozen(np.array(migration_csr.weights, dtype=np.float64)),
    )
    params = Params(
        carrying_capacity=0.0,
        eggs_per_female=0.0,
        sex_ratio=0.5,
        sperm_displacement_rate=0.0,
        low_density_growth_rate=0.0,
        growth_mode=0,
        external_expected_eggs=-1.0,
        survival_rates=np.zeros((n_sexes, n_ages), dtype=np.float64),
        mating_rates=np.zeros((n_sexes, n_ages), dtype=np.float64),
        reproduction_rates=np.zeros(n_ages, dtype=np.float64),
        fertility=np.zeros(n_ages, dtype=np.float64),
        competition_weights=np.zeros(n_ages, dtype=np.float64),
        equilibrium_distribution=np.zeros((0, 0), dtype=np.float64),
        viability_fitness=np.zeros((0,), dtype=np.float64),
        fecundity_fitness=np.zeros((0,), dtype=np.float64),
        sexual_selection_fitness=np.zeros((0,), dtype=np.float64),
        zygote_viability_fitness=np.zeros((0,), dtype=np.float64),
        offspring_tensor=np.zeros((0,), dtype=np.float64),
        meiosis_map=np.zeros((0,), dtype=np.float64),
        female_ztype_compatibility=np.zeros((0,), dtype=np.float64),
        male_ztype_compatibility=np.zeros((0,), dtype=np.float64),
        migration_rate=rate3d,
    )
    return blueprint, params


# Draft field -> contract column name for the runtime ecology fields a
# Python hook may write through its TickContext (migration_rate has its
# own dedicated channel and is excluded).
_ECO_DRAFT_TO_COLUMN: tuple[tuple[str, str], ...] = (
    ("carrying_capacity", "carrying_capacity"),
    ("eggs_per_female", "eggs_per_female"),
    ("sex_ratio", "sex_ratio"),
    ("sperm_displacement_rate", "sperm_displacement_rate"),
    ("low_density_growth_rate", "low_density_growth_rate"),
    ("juvenile_growth_mode", "growth_mode"),
    ("external_expected_eggs", "external_expected_eggs"),
    ("age_based_survival_rates", "survival_rates"),
    ("age_based_mating_rates", "mating_rates"),
    ("age_based_reproduction_rates", "reproduction_rates"),
    ("female_age_based_fertility", "fertility"),
    ("age_based_relative_competition_strength", "competition_weights"),
)

_ECOLOGY_COLUMN_FIELDS: frozenset[str] = frozenset(
    {
        "carrying_capacity",
        "eggs_per_female",
        "sex_ratio",
        "sperm_displacement_rate",
        "low_density_growth_rate",
        "growth_mode",
        "external_expected_eggs",
        "survival_rates",
        "mating_rates",
        "reproduction_rates",
        "fertility",
        "competition_weights",
        "equilibrium_distribution",
        "migration_rate",
    }
)

# Contract ecology name -> draft field (scalar / vector columns).
_ECOLOGY_DRAFT_FIELDS: dict[str, str] = {
    "carrying_capacity": "carrying_capacity",
    "eggs_per_female": "eggs_per_female",
    "sex_ratio": "sex_ratio",
    "sperm_displacement_rate": "sperm_displacement_rate",
    "low_density_growth_rate": "low_density_growth_rate",
    "growth_mode": "juvenile_growth_mode",
    "external_expected_eggs": "external_expected_eggs",
    "survival_rates": "age_based_survival_rates",
    "mating_rates": "age_based_mating_rates",
    "reproduction_rates": "age_based_reproduction_rates",
    "fertility": "female_age_based_fertility",
    "competition_weights": "age_based_relative_competition_strength",
    "equilibrium_distribution": "equilibrium_individual_distribution",
}

# Contract genetics tensor name -> draft field (stage-3 fork write path).
_GENETICS_DRAFT_FIELDS: dict[str, str] = {
    "viability_fitness": "viability_fitness",
    "fecundity_fitness": "fecundity_fitness",
    "sexual_selection_fitness": "sexual_selection_fitness",
    "zygote_viability_fitness": "zygote_viability_fitness",
    "offspring_tensor": "offspring_tensor",
    "meiosis_map": "zygotes_to_gametes_map",
    "female_ztype_compatibility": "female_ztype_compatibility",
    "male_ztype_compatibility": "male_ztype_compatibility",
}


class SpatialParamsView:
    """Validated spatial runtime-parameter surface (stage-3 columns).

    Reads return write-protected views of the live ``(n_demes, ...)``
    ecology columns (engines see updates immediately).  Writes go through
    :meth:`SpatialParamsView.tensor_write`, which validates the shape and
    then routes the per-deme values through the deme write channel, so the
    columns, the deme drafts, and the Rust session's ecology columns stay
    in lockstep.
    """

    def __init__(self, pop: SpatialPopulation) -> None:
        """Bind the view to its spatial population.

        Args:
            pop: The owning spatial population.
        """
        self._pop = pop

    def __getattr__(self, name: str) -> NDArray[np.float64]:
        """Read-protected view of one ecology column (or raise).

        Args:
            name: Contract ecology field name.

        Returns:
            A write-protected ``(n_demes, ...)`` column view.
        """
        columns = self._pop._ecology_columns  # pyright: ignore[reportPrivateUsage]  # column read channel
        if name in columns:
            backend = getattr(self._pop, "_rust_spatial_backend", None)
            if backend is not None:
                values = np.asarray(backend.ecology_columns_snapshot()[name], dtype=np.float64)
                arr = values if name == "equilibrium_distribution" else values.reshape(columns[name].shape)
            else:
                arr = columns[name].copy()
            readonly = arr.view()
            readonly.flags.writeable = False
            return readonly
        if name in _ECOLOGY_COLUMN_FIELDS:
            raise KeyError(f"ecology column {name!r} is not materialized")
        raise AttributeError(name)

    @property
    def migration_rate(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Write-protected ``(n_demes, S, A)`` rate view."""
        arr = self._pop._params.migration_rate  # pyright: ignore[reportPrivateUsage]  # view over the owned contract array
        readonly = arr.view()
        readonly.flags.writeable = False
        return readonly

    def tensor_write(self, field: str, values: RateDeclaration) -> None:
        """Write one ecology column (validated, in place, per deme).

        ``"migration_rate"`` keeps its build-time sugar: a scalar (adult
        ages of all sexes, juveniles 0), a per-sex mapping, an
        ``(n_ages,)`` vector, an ``(S, A)`` table, or the full
        ``(n_demes, S, A)`` column.  Smaller shapes broadcast across demes.

        Every other ecology field accepts the full ``(n_demes, ...)``
        column or a per-deme shape broadcast across demes; the per-deme
        values are routed through the shared write channel so the deme
        drafts and the Rust session columns update with the column.

        Args:
            field: Contract ecology field name.
            values: New column contents.

        Raises:
            ValueError: On an unknown field or a shape mismatch.
        """
        if field == "migration_rate":
            params = self._pop._params  # pyright: ignore[reportPrivateUsage]  # single validated write channel
            live = params.migration_rate
            n_demes, n_sexes, n_ages = live.shape
            if isinstance(values, dict):
                bp = self._pop._blueprint  # pyright: ignore[reportPrivateUsage]  # frozen adult-age anchor
                rate_2d = normalize_migration_rate(
                    values, n_sexes, n_ages, int(bp.new_adult_age)
                )
                new_live = np.tile(rate_2d, (n_demes, 1, 1))
            else:
                arr = np.asarray(values, dtype=np.float64)
                if arr.ndim == 3:
                    if arr.shape != live.shape:
                        raise ValueError(
                            f"migration_rate shape {arr.shape} does not match {live.shape}"
                        )
                    new_live = arr
                elif arr.ndim == 2:
                    if arr.shape != (n_sexes, n_ages):
                        raise ValueError(
                            f"migration_rate shape {arr.shape} does not match "
                            f"(n_sexes={n_sexes}, n_ages={n_ages})"
                        )
                    new_live = np.tile(arr, (n_demes, 1, 1))
                else:
                    bp = self._pop._blueprint  # pyright: ignore[reportPrivateUsage]  # frozen adult-age anchor
                    rate_2d = normalize_migration_rate(
                        values, n_sexes, n_ages, int(bp.new_adult_age)
                    )
                    new_live = np.tile(rate_2d, (n_demes, 1, 1))
            live[...] = new_live
            # The session owns the rate the migration stage consumes; a
            # runtime write must reach it or Rust ticks silently keep the
            # enable-time column.
            backend = getattr(self._pop, "_rust_spatial_backend", None)  # pyright: ignore[reportPrivateUsage]
            if backend is not None:
                backend.set_migration_rate(np.asarray(live, dtype=np.float64).ravel())
            return

        if field not in _ECOLOGY_COLUMN_FIELDS:
            raise ValueError(f"unknown spatial params field {field!r}")
        columns = self._pop._ecology_columns  # pyright: ignore[reportPrivateUsage]  # column write channel
        column = columns.get(field)
        if column is None:
            raise ValueError(f"ecology column {field!r} is not materialized")
        column_arr = np.asarray(column)
        per_deme_shape = column_arr.shape[1:]
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape == column_arr.shape:
            per_deme_values: list[NDArray[np.float64] | float] = [
                arr[i] for i in range(column_arr.shape[0])
            ]
        elif arr.ndim == 0:
            per_deme_values = [float(arr) for _ in range(column_arr.shape[0])]
        elif arr.shape == per_deme_shape:
            per_deme_values = [arr for _ in range(column_arr.shape[0])]
        else:
            raise ValueError(
                f"{field} shape {arr.shape} does not match the column "
                f"{column_arr.shape} or the per-deme shape {per_deme_shape}"
            )
        for deme_index, per_deme_value in enumerate(per_deme_values):
            self._pop._write_deme_ecology(deme_index, field, per_deme_value)  # pyright: ignore[reportPrivateUsage]  # single validated write channel


class SpatialPopulation:
    """Spatial container composed of per-deme population objects.

    This class models spatial structure via composition: every deme is one
    already-initialized ``BasePopulation`` subclass instance.

    Since slice 5 the spatial domain carries its own frozen contract pair:
    ``blueprint`` holds the deme count and the folded migration CSR;
    ``params`` holds the ``(n_demes, n_sexes, n_ages)`` migration-rate
    column.  Runtime migration is ``rate column x fixed CSR`` — topology,
    adjacency, kernel selection, and edge normalization exist only at
    build time; changing any of them rebuilds the model.

    Attributes:
        name (str): Human-readable name for the spatial container.
        demes (Sequence[DemePopulation]): Immutable view of managed demes.
        n_demes (int): Number of demes in the spatial system.
        species (object): Shared species object used by all demes.
        topology (GridTopology | None): Spatial topology used by the landscape.
        blueprint (Blueprint): Frozen spatial contract (dimensions, flags,
            migration CSR).
        params (SpatialParamsView): Validated runtime-parameter surface;
            the migration rate lives at ``params.migration_rate``.
        migration_row (callable): Normalized outbound weight readout for
            one source deme (derived from the CSR).
        tick (int): Current shared simulation tick across all demes.
    """

    _tick: int

    @classmethod
    def builder(
        cls,
        species: Species,
        n_demes: int,
        topology: Optional[GridTopology] = None,
        *,
        pop_type: Literal["age_structured", "discrete_generation"] = "age_structured",
    ) -> SpatialConfigurator:
        """Create a ``SpatialConfigurator`` for fluent spatial population construction.

        Args:
            species: Genetic architecture shared by all demes.
            n_demes: Number of demes in the spatial layout.
            topology: Optional grid topology for migration.
            pop_type: ``"age_structured"`` (default) or ``"discrete_generation"``.

        Returns:
            A ``SpatialConfigurator`` instance ready for chaining.

        Examples:
            >>> pop = SpatialPopulation.builder(species, n_demes=100) \\
            ...     .setup(name="demo") \\
            ...     .initial_state(...) \\
            ...     .competition(carrying_capacity=batch_setting([...])) \\
            ...     .build()
        """
        from natal.frontend.spatial.configurator import SpatialConfigurator

        return SpatialConfigurator(
            species=species,
            n_demes=n_demes,
            topology=topology,
            pop_type=pop_type,
        )

    def __init__(
        self,
        demes: Sequence[DemePopulation],
        *,
        topology: Optional[GridTopology] = None,
        adjacency: Optional[object] = None,
        migration_kernel: Optional[NDArray[np.float64]] = None,
        migration_strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = "auto",
        kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None,
        deme_kernel_ids: Optional[NDArray[np.int64]] = None,
        kernel_include_center: bool = False,
        migration_rate: RateDeclaration = 0.0,
        adjust_migration_on_edge: bool = False,
        name: str = "SpatialPopulation",
    ) -> None:
        """Initialize a spatial population container from existing demes.

        Args:
            demes: Sequence of already-initialized deme populations.
            topology: Optional grid topology used to derive adjacency when
                ``adjacency`` is not provided.
            adjacency: Optional explicit migration matrix with shape
                ``(n_demes, n_demes)``. Supports dense ``ndarray``, CSR tuple
                ``(indptr, indices, data)``, or sparse-like objects exposing
                ``toarray()``.
            migration_kernel: Optional odd-shaped 2D kernel used for topology-
                aware migration. When provided, ``topology`` is required and
                migration runs in kernel mode.
            migration_strategy: Migration mode policy. ``"auto"`` keeps
                existing behavior (kernel when ``migration_kernel`` is set,
                otherwise adjacency). ``"hybrid"`` is accepted as a forward-
                compatible alias of ``"auto"`` for now.
            kernel_bank: Optional kernel bank reserved for future per-deme
                heterogeneous-kernel routing.
            deme_kernel_ids: Optional per-deme kernel id array reserved for
                future heterogeneous-kernel routing.
            kernel_include_center: Whether kernel migration includes the kernel
                center as an outbound target for the source deme.
            migration_rate: Fraction of each deme that migrates each tick.
                Scalar applies only to adult ages (>= new_adult_age from
                config); juvenile ages default to 0.  ``(n_ages,)`` arrays
                are used as-is; a per-sex mapping such as
                ``{"F": 0.2, "M": 0.05}`` applies the scalar/vector rules
                per sex.  The normalized column lands on
                ``pop.params.migration_rate`` with shape
                ``(n_demes, n_sexes, n_ages)``.
            adjust_migration_on_edge: Whether to adjust migration rates on
                boundaries. When False (default), boundary demes migrate less
                due to fewer valid neighbors. When True, all demes have the
                same total migration rate regardless of position.
            name: Human-readable container name.

        Raises:
            ValueError: If ``demes`` is empty, demes do not share the same
                species object, topology size does not match the number of
                demes, migration strategy is invalid, adjacency input is
                invalid, migration kernel is invalid, or deme ticks do not
                match.
        """
        if not demes:
            raise ValueError("demes must contain at least one BasePopulation instance")

        # Keep a stable list internally; public accessor returns an immutable
        # tuple view to prevent accidental external mutation.
        self._demes: List[DemePopulation] = list(demes)
        # Frozen declaration snapshot, attached by
        # SpatialConfigurator.build() (plan 5.1 slice 3).
        self._definition: ModelDefinition | None = None
        # Genetics draft tables start out shared by every deme; in-place
        # genetics writes would leak across demes, so the per-deme params
        # view refuses them and routes through write_genetics (which forks
        # the variant first).  See ParamsView.tensor_write.
        for _d in self._demes:
            _d._shares_genetics_draft = True  # pyright: ignore[reportPrivateUsage]  # the marker is defined on BasePopulation for ParamsView to read

        # Stamp each deme with its live index so hooks see the same
        # pop.deme_id the Rust per-deme kernel reports (0 for panmictic,
        # the deme index here).  One stamping point covers every build
        # path (homogeneous template, heterogeneous groups, clones).
        for _deme_index, _deme in enumerate(self._demes):
            _deme._deme_id = _deme_index  # pyright: ignore[reportPrivateUsage]  # SpatialPopulation owns its demes; stamping the live index is the sanctioned write.

        # Spatial container expects all demes to share one Species object so
        # genotype indexing and config semantics are globally consistent.
        first_species = self._demes[0].species
        for idx, deme in enumerate(self._demes[1:], start=1):
            if deme.species is not first_species:
                raise ValueError(
                    f"deme[{idx}] species does not match deme[0]; all demes must share the same Species object"
                )

        n_demes = len(self._demes)
        if topology is not None and topology.n_demes != n_demes:
            raise ValueError(
                f"topology.n_demes ({topology.n_demes}) must match number of demes ({n_demes})"
            )

        if migration_strategy not in {"auto", "adjacency", "kernel", "hybrid"}:
            raise ValueError(
                "migration_strategy must be one of: auto, adjacency, kernel, hybrid"
            )

        # Resolve strategy-level policy into one concrete backend mode.  The
        # strategy is a build-time materialization choice only — it is
        # consumed here by the CSR fold and is not kept on the object.
        # ``auto`` and ``hybrid`` share runtime behavior (see the historical
        # hybrid-dispatch note in the migration module).
        if migration_kernel is not None:
            # Kernels are centered on one source cell; odd dimensions are
            # required so a unique center index exists.
            migration_kernel = np.asarray(migration_kernel, dtype=np.float64)
            if (
                migration_kernel.ndim != 2
                or migration_kernel.shape[0] % 2 == 0
                or migration_kernel.shape[1] % 2 == 0
            ):
                raise ValueError(
                    "migration_kernel must be a 2D array with odd dimensions"
                )

        if adjacency is None:
            # Default adjacency:
            # - no topology: identity matrix (no migration unless diagonal used)
            # - with topology: topology-derived neighborhood matrix
            if topology is None:
                adjacency = np.eye(n_demes, dtype=np.float64)
            else:
                adjacency = build_adjacency_matrix(topology)

        adjacency_dense = _coerce_adjacency_dense(adjacency, n_demes=n_demes)

        normalized_kernel_bank: tuple[NDArray[np.float64], ...] | None = None
        if kernel_bank is not None:
            if len(kernel_bank) == 0:
                raise ValueError("kernel_bank must not be empty when provided")
            kernels: List[NDArray[np.float64]] = []
            for kernel_idx, kernel_value in enumerate(kernel_bank):
                kernel_arr = np.asarray(kernel_value, dtype=np.float64)
                if (
                    kernel_arr.ndim != 2
                    or kernel_arr.shape[0] % 2 == 0
                    or kernel_arr.shape[1] % 2 == 0
                ):
                    raise ValueError(
                        "kernel_bank entries must be 2D arrays with odd dimensions "
                        f"(invalid at index {kernel_idx})"
                    )
                kernels.append(kernel_arr)
            normalized_kernel_bank = tuple(kernels)

        normalized_deme_kernel_ids: NDArray[np.int64] | None = None
        if deme_kernel_ids is not None:
            if normalized_kernel_bank is None:
                raise ValueError("deme_kernel_ids requires kernel_bank to be provided")
            normalized_deme_kernel_ids = np.asarray(deme_kernel_ids, dtype=np.int64)
            if normalized_deme_kernel_ids.shape != (n_demes,):
                raise ValueError(
                    "deme_kernel_ids shape mismatch: expected "
                    f"({n_demes},), got {normalized_deme_kernel_ids.shape}"
                )
            for deme_idx in range(n_demes):
                kernel_id = int(normalized_deme_kernel_ids[deme_idx])
                if kernel_id < 0 or kernel_id >= len(normalized_kernel_bank):
                    raise ValueError(
                        f"deme_kernel_ids[{deme_idx}]={kernel_id} out of range for kernel_bank size "
                        f"{len(normalized_kernel_bank)}"
                    )

        # Spatial hooks are local-to-deme by design, so container-level hooks
        # must always be rebuilt from all demes.
        self._hooks = self._compile_spatial_hooks_from_demes()
        # Session structure staleness flag (plan S2/S3): set by hook
        # registration and consumed by the next rust run boundary.
        self._rust_needs_rebuild = False

        migration_mode = resolve_migration_mode(
            strategy=migration_strategy,
            migration_kernel=migration_kernel,
            kernel_bank=normalized_kernel_bank,
            deme_kernel_ids=normalized_deme_kernel_ids,
        )

        self._name = name
        self._topology = topology
        # -- slice-5 contract fold ------------------------------------------
        # Topology, adjacency, kernel selection, kernel-center handling, and
        # edge normalization are resolved once here; runtime migration is the
        # frozen CSR multiplied by the Params rate column.  Nothing spatial
        # about migration survives on the object besides the CSR and the
        # contract pair.
        n_sexes, n_ages, adult_start_age = self._rate_axes(n_demes)
        rate_2d = normalize_migration_rate(
            migration_rate, n_sexes, n_ages, adult_start_age
        )
        migration_csr = fold_migration_csr(
            n_demes=n_demes,
            topology=topology,
            adjacency_dense=adjacency_dense,
            migration_kernel=migration_kernel,
            kernel_bank=normalized_kernel_bank,
            deme_kernel_ids=normalized_deme_kernel_ids,
            kernel_include_center=bool(kernel_include_center),
            adjust_on_edge=bool(adjust_migration_on_edge),
            mode=migration_mode,
        )
        self._migration_csr = migration_csr
        rate3d = np.tile(rate_2d, (n_demes, 1, 1))
        try:
            draft = self._export_reference_draft()
        except TypeError:
            # Lightweight test doubles may not implement ``export_config``;
            # the contract pair degrades to the shape-level minimum while
            # the CSR and rate column stay fully functional.
            self._blueprint, self._params = _minimal_contract(
                n_demes=n_demes,
                n_sexes=rate_2d.shape[0],
                n_ages=rate_2d.shape[1],
                migration_csr=migration_csr,
                rate3d=rate3d,
            )
        else:
            self._blueprint, self._params = materialize(
                draft,
                SpatialMigration(
                    indptr=migration_csr.indptr,
                    dest_idx=migration_csr.dest_idx,
                    weights=migration_csr.weights,
                    rate=rate3d,
                ),
            )
        # Spatial container and all demes share one logical tick counter.
        self._tick = int(self._demes[0].tick)

        # -- stage-3 ecology columns (contract name -> column array) --------
        # Every ecology field lives as an (n_demes, ...) column on this
        # container.  The migration-rate entry shares the live Params array
        # (never a copy), so engines reading pop._params.migration_rate see
        # column writes zero-copy.  Deme-slice ecology writes keep the
        # columns and the deme drafts in lockstep.
        self._ecology_columns = self._collect_ecology_columns()

        # Observation-based history recording.
        self._observation: Optional[Observation] = None
        self._observation_mask: Optional[NDArray[np.float64]] = None

        # Self-describing history model and recording plan (frozen at build time).
        self._history_obj: Optional[History] = None
        self._recording_plan: Optional[object] = None

        # History config
        self.max_history: int = 5000  # Default rolling window size

        for idx, deme in enumerate(self._demes[1:], start=1):
            if int(deme.tick) != self._tick:
                raise ValueError(
                    f"deme[{idx}] tick ({deme.tick}) does not match deme[0] tick ({self._tick})"
                )
        self._initialize_default_output_policy()

    def _rate_axes(self, n_demes: int) -> tuple[int, int, int]:
        """Resolve ``(n_sexes, n_ages, adult_start_age)`` for the rate column.

        Prefers the reference deme's draft (authoritative build-time
        axes); falls back to the state tensor shape for lightweight test
        doubles without a draft.

        Args:
            n_demes: Number of demes (unused by the resolution itself;
                accepted for call-site readability).

        Returns:
            The ``(n_sexes, n_ages, new_adult_age)`` triple.

        Raises:
            ValueError: If *n_demes* is not positive.
        """
        if n_demes < 1:
            raise ValueError("n_demes must be >= 1")
        export_fn = getattr(self._demes[0], "export_config", None)
        if callable(export_fn):
            draft = cast(ModelDraft, export_fn())
            return int(draft.n_sexes), int(draft.n_ages), int(draft.new_adult_age)
        ind = self._demes[0]._live_state().individual_count  # pyright: ignore[reportPrivateUsage]  # orchestration consumer
        if ind.ndim == 3:
            return int(ind.shape[0]), int(ind.shape[1]), (1 if ind.shape[1] > 1 else 0)
        return 2, 1, 0

    def _export_reference_draft(self) -> ModelDraft:
        """Return deme 0's exported draft as the spatial contract reference.

        The spatial Blueprint/Params pair materializes from the first
        deme's draft (all demes share dimensions and execution flags);
        per-deme ecology differences stay on the deme drafts until the
        variant-bank slice.

        Returns:
            The reference ``ModelDraft``.

        Raises:
            TypeError: If deme 0 does not implement ``export_config``.
        """
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        return cast(ModelDraft, export_fn())

    def _export_deme_drafts(self, *, compact: bool = False) -> list[ModelDraft]:
        """Export every deme's draft, in deme order.

        Used by the Rust backend wiring: the columnized ecology and the
        genetics variant bank are gathered from the per-deme drafts.

        Args:
            compact: Share identical detached arrays only within this returned
                list, for read-only native handoffs. Public snapshots are unchanged.

        Returns:
            One ``ModelDraft`` per deme.

        Raises:
            TypeError: If any deme does not implement ``export_config``.
        """
        drafts: list[ModelDraft] = []
        shared_arrays: dict[tuple[str, tuple[int, ...], bytes], NDArray[np.generic]] = {}
        for idx, deme in enumerate(self._demes):
            export_fn = getattr(deme, "export_config", None)
            if not callable(export_fn):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            draft = cast(ModelDraft, export_fn())
            if compact:
                # Compact as snapshots arrive, before retaining D copies of
                # identical genetics merely to deduplicate the variant bank.
                replacements: dict[str, object] = {}
                for name, value in zip(draft._fields, draft, strict=True):
                    if isinstance(value, np.ndarray):
                        array = np.asarray(value)
                        key = (array.dtype.str, array.shape, array.tobytes())
                        replacements[name] = shared_arrays.setdefault(key, array)
                draft = draft._replace(**replacements)
            drafts.append(draft)
        return drafts

    def _initialize_default_output_policy(self) -> None:
        """Install identity Observation and raw History for direct construction.

        ``SpatialConfigurator`` replaces these defaults with its explicitly
        compiled policy after construction. The defaults keep the public
        ``SpatialPopulation(demes, ...)`` constructor fully usable on its own.
        """
        from natal.frontend.output.history import (
            History,
            HistorySchema,
            PopulationLayout,
            SpatialHistoryLayout,
        )
        from natal.frontend.output.observation import Observation

        state = self._demes[0].state
        counts = state.individual_count
        n_sexes, n_ages, n_ztypes = map(int, counts.shape)
        has_sperm = getattr(state, "sperm_storage", None) is not None
        kind: Literal["spatial_age_structured", "spatial_discrete_generation"] = (
            "spatial_discrete_generation"
            if isinstance(state, DiscretePopulationState)
            else "spatial_age_structured"
        )
        registry = getattr(self._demes[0], "_index_registry", None)
        if registry is not None and len(registry.index_to_ztype) == n_ztypes:
            ztype_labels = tuple(
                f"{genotype}@{slab}" for genotype, slab in registry.index_to_ztype
            )
        else:
            ztype_labels = tuple(f"ztype_{index}" for index in range(n_ztypes))
        layout = PopulationLayout(
            kind=kind,
            n_demes=self.n_demes,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            has_sperm_storage=has_sperm,
            sex_labels=("female", "male")[:n_sexes],
            ztype_labels=ztype_labels,
        )
        self._observation = Observation(
            labels=ztype_labels,
            collapse_age=False,
            population_fingerprint=layout.fingerprint,
            deme_indices=tuple(range(self.n_demes)),
            deme_mode="preserve",
            _is_identity=True,
            _identity_map=np.arange(n_ztypes, dtype=np.int32),
        )
        ind_per_deme = n_sexes * n_ages * n_ztypes
        sperm_per_deme = n_ages * n_ztypes * n_ztypes if has_sperm else 0
        schema = HistorySchema(
            mode="raw",
            population=layout,
            row_size=(1 + self.n_demes * (ind_per_deme + sperm_per_deme)),
            spatial_layout=SpatialHistoryLayout(
                n_demes=self.n_demes,
                ind_per_deme=ind_per_deme,
                sperm_per_deme=sperm_per_deme,
            ),
        )
        self._history_obj = History(schema, max_rows=self.max_history)

    @property
    def name(self) -> str:
        """str: Human-readable name for the spatial container."""
        return self._name

    @property
    def demes(self) -> Sequence[DemeSlice]:
        """Sequence[DemeSlice]: Immutable view of all managed deme slices."""
        return tuple(DemeSlice(self, i) for i in range(len(self._demes)))

    @property
    def n_demes(self) -> int:
        """int: Number of demes in the spatial system."""
        return len(self._demes)

    @property
    def species(self) -> Species:
        """Species: Shared species object used by all demes."""
        return self._demes[0].species

    @property
    def topology(self) -> GridTopology | None:
        """GridTopology | None: Landscape topology used by the spatial model.

        Read-only build-time metadata (deme coordinates, neighbor
        queries); it no longer participates in runtime migration, which
        consumes the frozen Blueprint CSR.
        """
        return self._topology

    @property
    def blueprint(self) -> Blueprint:
        """Blueprint: Frozen spatial contract (dimensions, flags, migration CSR)."""
        return self._blueprint

    @property
    def params(self) -> SpatialParamsView:
        """SpatialParamsView: Validated spatial runtime-parameter surface.

        The migration rate lives at ``params.migration_rate`` with shape
        ``(n_demes, n_sexes, n_ages)``; writes go through
        :meth:`SpatialParamsView.tensor_write`.
        """
        return SpatialParamsView(self)

    @property
    def migration_csr(self) -> MigrationCSR:
        """MigrationCSR: Frozen outbound migration routing table."""
        return self._migration_csr

    def deme(self, idx: int) -> DemeSlice:
        """Return one deme slice by positional index.

        The slice delegates every read to the underlying deme population
        (``config``, ``state``, ``registry``, …), so deep read interfaces
        stay fully compatible.  Writes go through the stage-3 channels:
        :meth:`DemeSlice.write_ecology` (column + draft + Rust session)
        and :meth:`DemeSlice.write_genetics` (variant fork).

        Args:
            idx: Zero-based deme index.

        Returns:
            The deme slice at ``idx``.
        """
        return DemeSlice(self, idx)

    def _deme_object(self, idx: int) -> DemePopulation:
        """Return the raw deme object (typed internal consumers only).

        Public readers should use :meth:`deme`; this accessor exists for
        the few internal consumers (output translation, recording-plan
        compilation) whose signatures demand a ``BasePopulation``.

        Args:
            idx: Zero-based deme index.

        Returns:
            The underlying deme population.
        """
        return self._demes[idx]

    # -- stage-3 write channels (ecology columns + genetics fork) --------

    def _collect_ecology_columns(self) -> dict[str, NDArray[np.float64]]:
        """Gather the (n_demes, ...) ecology columns from the deme drafts.

        The migration-rate entry shares the live Params array (zero-copy),
        and every other column is a fresh gather.  Lightweight deme
        doubles without ``export_config`` degrade to the migration-rate
        column only.

        Returns:
            The ecology column mapping keyed by contract field name.
        """
        n_demes = len(self._demes)
        columns: dict[str, NDArray[np.float64]] = {}
        try:
            from natal.backends.rust.rust_backend import ecology_columns_from_drafts

            drafts = self._export_deme_drafts(compact=True)
            columns.update(
                {
                    name: np.asarray(column_values, dtype=np.float64)
                    for name, column_values in ecology_columns_from_drafts(
                        drafts
                    ).items()
                }
            )
            # Reshape the session-boundary flat columns to their logical
            # ``(n_demes, *per_deme)`` shapes so reads and writes see the
            # contract's dimensionality.
            for name, draft_field in _ECOLOGY_DRAFT_FIELDS.items():
                column = columns.get(name)
                if column is None or column.ndim != 1:
                    continue
                first = getattr(drafts[0], draft_field, None)
                if first is None or np.asarray(first).ndim == 0:
                    continue
                per_deme: tuple[int, ...] = np.asarray(first).shape
                if column.size == n_demes * int(np.prod(per_deme)):
                    columns[name] = column.reshape((n_demes,) + per_deme)
        except TypeError:
            # Lightweight test doubles: keep only the shared rate column.
            pass
        columns["migration_rate"] = np.asarray(
            self._params.migration_rate, dtype=np.float64
        )
        return columns

    def _rust_spatial_session(self) -> object | None:
        """Return the enabled Rust spatial backend, or ``None``.

        Returns:
            The active ``RustHeterogeneousSpatialLifecycleBackend`` when
            the Rust backend is enabled for age-structured demes.
        """
        backend = getattr(self, "_rust_spatial_backend", None)
        if isinstance(backend, object) and type(backend).__name__ == (
            "RustHeterogeneousSpatialLifecycleBackend"
        ):
            return backend
        return None

    def _detach_deme_field(
        self, deme_index: int, draft_field: str
    ) -> tuple[ModelDraft, NDArray[np.generic] | float | int | bool | None]:
        """Return the deme's draft with a private copy of one mutable field.

        Clone-on-write per array field: when any other deme shares the array
        object, the target deme gets a ``_replace`` shell with a copy so an
        in-place write cannot penetrate to the sharing demes.  Plain Python
        scalar fields are immutable NamedTuple slots — a write goes through
        ``_replace`` and needs no detach.

        Args:
            deme_index: Zero-based deme index.
            draft_field: ``ModelDraft`` field name to detach.

        Returns:
            The (possibly replaced) draft and its field value (array or
            scalar).
        """
        target = self._demes[deme_index]
        config = target.config
        field_value: NDArray[np.generic] | float | int | bool | None = getattr(
            config, draft_field
        )
        if isinstance(field_value, np.ndarray):
            typed_value = field_value
            shared = any(
                getattr(other.config, draft_field, None) is typed_value
                for j, other in enumerate(self._demes)
                if j != deme_index
            )
            if shared:
                config = config._replace(**{draft_field: typed_value.copy()})
                target.set_config(config)
                field_value: NDArray[np.generic] | float | int | bool | None = getattr(
                    config, draft_field
                )
        return config, field_value

    def _write_deme_ecology(
        self, deme_index: int, field: str, value: float | NDArray[np.float64]
    ) -> None:
        """Write one ecology value for one deme (column + draft + Rust).

        The write lands in three synchronized places: the deme's draft
        (per-field clone-on-write, equilibrium metrics re-synced), the
        container's ecology column, and — when the Rust backend is
        enabled — the session's per-deme ecology column.

        Args:
            deme_index: Zero-based deme index.
            field: Contract ecology field name.
            value: New scalar or array in the contract's logical shape.

        Raises:
            KeyError: If *field* is not an ecology field.
        """
        if field not in _ECOLOGY_DRAFT_FIELDS:
            raise KeyError(f"unknown ecology field {field!r}")
        draft_field = _ECOLOGY_DRAFT_FIELDS[field]
        config, field_value = self._detach_deme_field(deme_index, draft_field)
        if isinstance(field_value, np.ndarray):
            field_array: NDArray[np.float64] = cast("NDArray[np.float64]", field_value)
            if field_array.ndim == 0:
                field_array[()] = value
            else:
                field_array[...] = np.asarray(value, dtype=np.float64).reshape(
                    field_array.shape
                )
        else:
            # Scalar NamedTuple slot: rebuild the draft; the value is
            # float or int according to the ecology field's declared type.
            config = config._replace(**{draft_field: value})
            self._demes[deme_index].set_config(config)

        # The equilibrium metrics are derived on read (slice 2 retired
        # the stored copies), so no post-write refresh is needed here.

        column = self._ecology_columns.get(field)
        if column is not None and field != "migration_rate":
            column_arr = np.asarray(column)
            if column_arr.ndim == 1 and column_arr.size == self.n_demes:
                column_arr[deme_index] = value
            else:
                flat = column_arr.reshape(self.n_demes, -1)
                flat[deme_index] = np.asarray(value, dtype=np.float64).ravel()

        backend = self._rust_spatial_session()
        if backend is not None and field != "migration_rate":
            from natal.contracts.materialize import materialize

            contracts = materialize(config)
            refresh = getattr(backend, "refresh_deme_ecology", None)
            if callable(refresh):
                refresh(deme_index, [field], contracts.params)

    def _write_deme_genetics(
        self, deme_index: int, field: str, values: NDArray[np.float64]
    ) -> None:
        """Write one genetics tensor for one deme, forking the variant.

        The Rust bank variant of the deme is cloned first (other demes
        keep sharing the original tables) and the draft arrays are
        detached the same way, so the fork is atomic across backends.
        A ``meiosis_map`` write also recomputes the derived offspring
        tensor on the forked tables — the engines only consume the
        derived tensor, so leaving it stale would be silently inert
        (audit finding C3).

        Args:
            deme_index: Zero-based deme index.
            field: Contract genetics tensor name (one of the eight tables).
            values: New contents in the table's logical shape.

        Raises:
            KeyError: If *field* is not a genetics tensor.
            ValueError: If a meiosis table's rows are not distributions.
        """
        from natal.frontend.configurator._writers import (
            recompute_offspring_tensor,
            validate_meiosis_table,
        )

        if field not in _GENETICS_DRAFT_FIELDS:
            raise KeyError(f"unknown genetics tensor {field!r}")
        draft_field = _GENETICS_DRAFT_FIELDS[field]
        config, field_value = self._detach_deme_field(deme_index, draft_field)
        if not isinstance(field_value, np.ndarray):
            raise TypeError(
                f"{field!r} expects a tensor-backed draft field, got a scalar"
            )
        field_array: NDArray[np.float64] = cast("NDArray[np.float64]", field_value)
        candidate = np.asarray(values, dtype=np.float64).reshape(field_array.shape)
        if field == "meiosis_map":
            validate_meiosis_table(candidate)
        field_array[...] = candidate

        refresh_fields = [field]
        if field == "meiosis_map":
            # Detach the derived tensor too (clone-on-write, same as the
            # meiosis table) so the recompute cannot leak into the demes
            # still sharing the original offspring table.
            config, offspring_value = self._detach_deme_field(
                deme_index, "offspring_tensor"
            )
            offspring_array = cast("NDArray[np.float64]", offspring_value)
            offspring_array[...] = recompute_offspring_tensor(
                field_array, config.gametes_to_zygotes_map
            )
            refresh_fields.append("offspring_tensor")

        backend = self._rust_spatial_session()
        if backend is not None:
            fork = getattr(backend, "fork_variant", None)
            refresh = getattr(backend, "refresh_variant_tensors", None)
            if callable(fork) and callable(refresh):
                variant_id = fork(deme_index)
                from natal.contracts.materialize import materialize

                contracts = materialize(config)
                refresh(variant_id, refresh_fields, contracts.params)

    @property
    def tick(self) -> int:
        """int: Shared simulation tick across all demes."""
        return self._tick

    @property
    def definition(self) -> ModelDefinition:
        """The frozen declaration snapshot this spatial population was
        built from.

        Returns:
            The :class:`~natal.frontend.data.ModelDefinition` captured at
            build time (the wrapper journal with raw BatchSetting
            declarations preserved).

        Raises:
            AttributeError: If the population was not built through
                ``SpatialConfigurator.build()``.
        """
        if self._definition is None:
            raise AttributeError(
                "This spatial population has no declaration snapshot; it "
                "was not built through SpatialConfigurator.build()."
            )
        return self._definition

    @property
    def history(self) -> History:
        """Return the self-describing spatial History container.

        Returns:
            The History created from the build-time recording policy.

        Raises:
            RuntimeError: If the spatial population is not fully built.
        """
        if self._history_obj is None:
            raise RuntimeError("History is not initialized for this population.")
        return self._history_obj

    @property
    def observation(self) -> Observation:
        """Return the immutable canonical spatial Observation.

        Returns:
            The Observation created at build time.

        Raises:
            RuntimeError: If the spatial population is not fully built.
        """
        if self._observation is None:
            raise RuntimeError("Observation is not initialized for this population.")
        return self._observation

    def observe(self) -> ObservationResult:
        """Project all deme states through the canonical Observation.

        Returns:
            An ObservationResult with group-first values. ``preserve`` keeps
            the selected deme axis; ``aggregate`` sums and removes it.

        Raises:
            RuntimeError: If the canonical Observation is not initialized.
        """
        from types import MappingProxyType

        from natal.frontend.output.observation import ObservationResult

        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is None:
            self._initialize_session()
            backend = self._rust_spatial_backend
        observation = self.observation
        layout = self.history.schema.population
        mask = observation.build_mask(layout.n_sexes, layout.n_ages, layout.n_ztypes)
        selected = list(observation.deme_indices or ())
        tick, values = backend.observe_current(mask, selected, observation.collapse_age, observation.deme_mode == "aggregate")
        shape = (observation.n_groups,)
        if observation.deme_mode == "preserve":
            shape += (len(selected),)
        shape += (layout.n_sexes,)
        if not observation.collapse_age:
            shape += (layout.n_ages,)
        return ObservationResult(
            tick=tick, _values=values.reshape(shape), axes=observation.axes,
            _labels=MappingProxyType({"group": observation.labels}),
        )

    def clear_history(self) -> None:
        """Clear all recorded history and the session checkpoints."""
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is not None:
            history_obj.clear()
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is not None:
            backend.clear_checkpoints()  # pyright: ignore[reportAttributeAccessIssue]  # session backend surface

    # ========================================================================
    # Observation infrastructure
    # ========================================================================

    def _record_snapshot(self, *, allow_existing: bool) -> None:
        """Manually record the current stacked spatial state as a history entry.

        The frozen History schema selects either the complete stacked raw state
        or canonical observation values. All demes are committed atomically.

        Args:
            allow_existing: Whether an automatic run boundary may reuse the
                already-recorded current tick without writing a second row.

        Raises:
            RuntimeError: If spatial History or Observation is not initialized.
            ValueError: If a strict snapshot repeats or precedes the latest
                tick, or an automatic boundary is stale or has a different
                payload.
        """
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is None:
            self._initialize_session()
            backend = self._rust_spatial_backend
        history_obj = self.history
        if history_obj.schema.mode == "observation":
            history_obj._configure_observation(self.observation)  # pyright: ignore[reportPrivateUsage]  # container binds recording selector
        backend.bind_history(history_obj._store, [deme._params_log for deme in self._demes])  # pyright: ignore[reportPrivateUsage]  # share native ownership
        history_obj._bind_checkpoint_pruner(backend.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.
        backend.record_history(allow_existing)

    def record_snapshot(self) -> None:
        """Record the current stable state across all demes into history.

        Must only be called when the engine is not running. Records
        all demes atomically in one snapshot. Duplicate ticks are rejected.

        Raises:
            RuntimeError: If the population has finished simulation
                or is currently running.
            ValueError: If the current tick is already recorded.
        """
        if getattr(self, "_running", False):
            raise RuntimeError(
                "Cannot record snapshot while the population is running."
            )
        self._record_snapshot(allow_existing=False)

    def restore_checkpoint(self, tick: int) -> None:
        """Restore one exact retained raw checkpoint, including execution status.

        State, RNG, ecology, phase, and logs return to the recorded boundary.
        Future records are removed. A missing native checkpoint is rejected;
        a counts-only payload cannot restore a reproducible execution state.

        Args:
            tick: Exact retained tick to restore.

        Raises:
            ValueError: If raw history or the complete native checkpoint is absent.
        """
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is None or history_obj.is_empty:
            raise ValueError("No history available for checkpoint restore.")
        if history_obj.schema.mode != "raw":
            raise ValueError(
                "Cannot restore population state from observation-mode "
                "history.  Record raw history to enable checkpoint "
                "restoration."
            )
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is not None and self._restore_from_rust_checkpoint(backend, tick):
            history_obj.truncate(retain_until_tick=tick)
            return
        raise ValueError(f"No complete native checkpoint exists at tick {tick}.")

    def _restore_from_rust_checkpoint(
        self, backend: RustHeterogeneousSpatialLifecycleBackend, tick: int
    ) -> bool:
        """Restore the full runtime from the session checkpoint store.

        Args:
            backend: The live spatial backend.
            tick: Target tick.

        Returns:
            ``True`` when a checkpoint covered *tick* (the runtime, the
            ecology columns, the deme drafts, and the caches are rewound);
            ``False`` when no checkpoint covers *tick* (untouched).
        """
        restored_tick = backend.restore_from_checkpoint(int(tick))
        if restored_tick is None:
            return False
        # Roll the Python-side mirrors back to the checkpoint: container
        # columns first, then the per-deme drafts that project them.  The
        # snapshot columns are flat while the container columns carry their
        # logical (n_demes, ...) shape — roll by size and refuse unknown
        # shapes loudly rather than silently leaving a stale column.
        columns = backend.ecology_columns_snapshot()
        for name, column in columns.items():
            if name == "migration_rate":
                continue  # dedicated live contract array, rolled below
            local = self._ecology_columns.get(name)
            if local is None:
                continue  # not a materialized container column
            target = np.asarray(local)
            flat = np.asarray(column, dtype=np.float64)
            if name == "equilibrium_distribution":
                self._ecology_columns[name] = flat.copy()
                continue
            if target.size != flat.size:
                raise ValueError(
                    f"checkpoint rollback: ecology column {name!r} size "
                    f"{flat.size} does not fit the container column size "
                    f"{target.size}"
                )
            local[...] = flat.reshape(target.shape)
        rate = np.asarray(columns["migration_rate"], dtype=np.float64)
        if self._params.migration_rate.size == rate.size:
            self._params.migration_rate[...] = rate.reshape(
                self._params.migration_rate.shape
            )
        else:
            raise ValueError(
                "checkpoint rollback: migration_rate size "
                f"{rate.size} does not fit the contract array "
                f"{self._params.migration_rate.size}"
            )
        n_demes = len(self._demes)
        for deme_id, deme in enumerate(self._demes):
            self._rollback_deme_draft(deme, columns, deme_id, restored_tick)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
            status, _ = backend.execution_state()
            deme._finished = status == "Stopped"  # pyright: ignore[reportPrivateUsage]  # restore exact lifecycle marker
            deme._failed = status == "Failed"  # pyright: ignore[reportPrivateUsage]  # partial snapshots retain failure metadata
        # The session state was rewound: refresh the deme caches from it.
        self._rust_states_dirty = True
        self._ensure_rust_states_fresh()
        assert self._tick == restored_tick  # noqa: S101 — the refresh installs the restored tick
        assert n_demes == len(self._demes)  # noqa: S101 — deme set is fixed at build
        return True

    def _rollback_deme_draft(
        self,
        deme: BasePopulation[Any],
        columns: dict[str, NDArray[Any]],
        deme_id: int,
        tick: int,
    ) -> None:
        """Write the checkpoint ecology values back into one deme draft."""
        n_demes = len(self._demes)
        draft = deme.config
        updates: dict[str, object] = {}
        for draft_field, column_name in _ECO_DRAFT_TO_COLUMN:
            column = np.asarray(columns[column_name])
            if column.ndim == 1 and column.size == n_demes:
                updates[draft_field] = float(column[deme_id])
                continue
            per_deme = column.reshape(n_demes, -1)[deme_id]
            # Reshape back to the draft field's logical extent.
            current = cast(
                "NDArray[np.float64]", np.asarray(getattr(draft, draft_field))
            )
            if current.size == per_deme.size:
                updates[draft_field] = per_deme.reshape(current.shape)
        new_draft = draft
        for field, value in updates.items():
            current_value = cast(
                "NDArray[np.float64] | float | None",
                getattr(new_draft, field, None),
            )
            if isinstance(current_value, np.ndarray) and isinstance(value, np.ndarray):
                new_draft = new_draft._replace(
                    **{field: value.reshape(current_value.shape)}
                )
            else:
                new_draft = new_draft._replace(**{field: value})
        deme.set_config(new_draft)
        deme._tick = int(tick)  # pyright: ignore[reportPrivateUsage]  # container restores the shared clock

    @property
    def hooks(self) -> HookProgram:
        """Return the aggregate native hook program for all demes."""
        return self._hooks

    def register_hooks(
        self,
        *items: object,
        event: Optional[str] = None,
        priority: int = 0,
        deme: DemeSelector = "*",
        name: Optional[str] = None,
    ) -> None:
        """Register hooks on the demes selected by *deme* — the single entry.

        Mirrors :meth:`natal.frontend.population.base.BasePopulation.
        register_hooks` at the container level: the deme selector chooses
        target demes, then registration is forwarded to each target with
        panmictic selector semantics (the compiler-level selector fields
        are transport-only metadata).

        When multiple demes share the same hook storage (as happens with
        homogeneous clones), the hook is registered **once** on a
        representative and all other targeted owners see the change
        through shared storage.  This avoids accidental duplicate
        descriptors.

        Args:
            *items: Hook registrations (``Op`` objects, ``@hook``-
                decorated functions, single-parameter callables).
            event: Default event for items that do not carry one.
            priority: Execution priority — lower values run first.
            deme: Deme selector (``"*"`` = all demes).
            name: Optional name for grouped op registrations.
        """
        # Determine which demes are targeted by the selector.
        target_ids = [
            i for i in range(self.n_demes) if self._selector_matches_deme(deme, i)
        ]

        if not target_ids:
            return

        # Group targeted demes by the identity of their hook storage.
        # Demes that share the same ``compiled_hook_descriptors`` list (via
        # clone sharing) should only have the hook compiled and appended
        # once.
        storage_groups = self._group_demes_by_hook_storage(target_ids)

        for storage_key, group_ids in storage_groups.items():
            # Find ALL demes (targeted + non-targeted) sharing this storage.
            all_owners = self._demes_sharing_storage(storage_key)

            if set(all_owners).issubset(set(target_ids)):
                # All owners are targeted — register once on a
                # representative; other owners see the change through the
                # shared list.
                self._demes[group_ids[0]].register_hooks(
                    *items, event=event, priority=priority, deme="*", name=name
                )
            else:
                # Only a subset of owners are targeted — copy-on-write so
                # the non-targeted owners keep their existing hook storage
                # intact.
                self._copy_hook_storage_for_demes(group_ids)
                self._demes[group_ids[0]].register_hooks(
                    *items, event=event, priority=priority, deme="*", name=name
                )

        # Invalidate the hook executor on all targeted demes so they pick
        # up the recompiled aggregate hooks.
        for deme_id in target_ids:
            self._demes[deme_id].invalidate_hook_dispatch()

        # Rebuild aggregate hooks once after all per-deme mutations.
        self._refresh_spatial_hooks()

    def _group_demes_by_hook_storage(
        self,
        deme_ids: list[int],
    ) -> dict[int, list[int]]:
        """Group deme indices by the identity of their hook storage.

        Demes that share the same ``compiled_hook_descriptors`` list object
        (typically clones sharing template storage) are placed in the same
        group.

        Args:
            deme_ids: Indices of demes to group.

        Returns:
            A dict mapping ``id(compiled_hook_descriptors)`` to the list of
            deme indices sharing that storage.
        """
        groups: dict[int, list[int]] = {}
        for deme_id in deme_ids:
            deme = self._demes[deme_id]
            key = id(deme.compiled_hook_descriptors)
            groups.setdefault(key, []).append(deme_id)
        return groups

    def _demes_sharing_storage(
        self,
        storage_key: int,
    ) -> list[int]:
        """Return all deme indices that share the given hook storage identity.

        Args:
            storage_key: The ``id(compiled_hook_descriptors)`` of a shared
                descriptor list.

        Returns:
            List of deme indices across the entire population whose storage
            matches *storage_key*.
        """
        owners: list[int] = []
        for deme_id in range(self.n_demes):
            deme = self._demes[deme_id]
            if id(deme.compiled_hook_descriptors) == storage_key:
                owners.append(deme_id)
        return owners

    def _copy_hook_storage_for_demes(
        self,
        deme_ids: list[int],
    ) -> None:
        """Copy-on-write hook storage for a subset of demes.

        Creates a copy of ``compiled_hook_descriptors`` from the first deme
        in *deme_ids* so they no longer share storage with non-targeted
        peers; future mutations on targeted demes do not leak to
        untargeted owners.

        Args:
            deme_ids: Indices of demes to receive the new private storage.
                All listed demes will point to the same new copies.
        """
        ref = self._demes[deme_ids[0]]

        compiled_copy = list(ref.compiled_hook_descriptors)

        for deme_id in deme_ids:
            deme = self._demes[deme_id]
            object.__setattr__(deme, "compiled_hook_descriptors", compiled_copy)

    @staticmethod
    def _selector_matches_deme(selector: DemeSelector, deme_id: int) -> bool:
        """Return whether one deme selector targets a concrete deme id.

        Args:
            selector: Deme selector in any supported hook form:
                ``"*"``, integer id, ``range``, or explicit id collection.
            deme_id: Concrete deme index to test.

        Returns:
            ``True`` when ``deme_id`` is selected by ``selector``; otherwise
            ``False``.
        """
        if selector == "*":
            return True
        if isinstance(selector, int):
            return selector == deme_id
        if isinstance(selector, range):
            return deme_id in selector
        return deme_id in selector

    def _effective_compiled_hook_sequences(self) -> list[list[CompiledHookDescriptor]]:
        """Collect per-deme effective hook sequences filtered by selector.

        Each inner list contains the actual descriptor objects (not copies)
        for hooks whose ``deme_selector`` matches the owning deme. Descriptors
        are kept in their original registration order (sorted by priority).

        Returns:
            List of length ``n_demes``, one sequence per deme.

        Note:
            This method returns references to the original descriptor objects,
            which allows callers to compare sequences by descriptor identity.
        """
        sequences: list[list[CompiledHookDescriptor]] = []
        for deme_id, deme in enumerate(self._demes):
            try:
                hooks = deme.get_compiled_hooks()
            except AttributeError:
                sequences.append([])
                continue
            effective = [
                desc
                for desc in hooks
                if self._selector_matches_deme(desc.deme_selector, deme_id)
            ]
            sequences.append(effective)
        return sequences

    def _collect_compact_spatial_hooks(self) -> list[CompiledHookDescriptor]:
        """Build a compact hook descriptor list by grouping demes with identical hook sequences.

        Demes that share the exact same sequence of descriptors (compared by
        Python object identity) are folded into one set of descriptors with
        an expanded ``deme_selector`` covering all demes in that group.

        This eliminates redundant static call sites in the spatial lifecycle
        wrapper — without this compaction, N identical demes produce N static
        calls to the same dispatcher, which can trigger native instability
        (SIGSEGV) under prange execution.

        Sequence ordering, repeats, and descriptor identity are preserved:
        ``[A, A]`` stays distinct from ``[A]``, and ``[A, B]`` stays distinct
        from ``[B, A]``.  Independent-but-equivalent descriptors built at
        different times are NOT merged — identity-based grouping is
        deliberately conservative.

        Returns:
            List of ``CompiledHookDescriptor`` with compacted
            ``deme_selector`` values.
        """
        sequences = self._effective_compiled_hook_sequences()
        n_demes = self.n_demes

        # Group demes by descriptor-identity key of their full hook sequence.
        # Using ``id()`` avoids merging independently-built descriptors that
        # happen to have equivalent content but different semantics.
        key_to_demes: dict[tuple[int, ...], list[int]] = {}
        for deme_id, seq in enumerate(sequences):
            key = tuple(id(desc) for desc in seq)
            key_to_demes.setdefault(key, []).append(deme_id)

        compact: list[CompiledHookDescriptor] = []
        for key, deme_ids in key_to_demes.items():
            if not key:
                # Empty hook sequence — no descriptors to produce.
                continue

            # Determine compact selector for this group.
            if len(deme_ids) == n_demes:
                selector: DemeSelector = "*"
            elif len(deme_ids) == 1:
                selector = deme_ids[0]
            else:
                selector = tuple(sorted(deme_ids))

            # Clone each descriptor from the reference deme's sequence with
            # the compact selector, preserving per-deme execution semantics:
            # every deme in the group still executes every slot once.  The
            # callback identity (``desc.callback``) is preserved so the
            # callback bridges can map each callback slot back to every
            # deme's own runner index: per-deme selectors may filter
            # differently, so the filtered slot order and a deme's
            # unfiltered registration order need not coincide.
            ref_seq = sequences[deme_ids[0]]
            for desc in ref_seq:
                compact.append(replace(desc, deme_selector=selector))

        return compact

    def _collect_effective_compiled_hooks(self) -> list[CompiledHookDescriptor]:
        """Collect hooks from each deme and pin each one to its owner deme.

        Local spatial hook semantics are per-deme: ordering and execution scope
        are only defined inside each deme. This method lifts per-deme hook
        descriptors into one aggregate list while forcing ``deme_selector`` to
        the owning deme id.

        This method returns the **expanded** (per-deme pinned) view for public
        introspection via :meth:`get_compiled_hooks`. The compact execution
        plan is built separately by :meth:`_collect_compact_spatial_hooks`.

        Returns:
            A flat list of hook descriptors safe for aggregate spatial
            execution. Every descriptor in the returned list has
            ``deme_selector`` rewritten to one concrete integer deme id.

        Note:
            Rewriting selectors here avoids accidental cross-deme execution
            after flattening all demes into a single compiled registry.
        """
        compiled_hooks: list[CompiledHookDescriptor] = []
        for deme_id, deme in enumerate(self._demes):
            try:
                hooks = deme.get_compiled_hooks()
            except AttributeError:
                # Lightweight test doubles may not implement compiled-hook APIs.
                continue

            for desc in hooks:
                # Keep only descriptors that actually apply to this owning deme.
                if not self._selector_matches_deme(desc.deme_selector, deme_id):
                    continue
                # Pin selector to concrete owner deme so aggregate execution
                # preserves local-only hook semantics.
                compiled_hooks.append(replace(desc, deme_selector=int(deme_id)))
        return compiled_hooks

    @staticmethod
    def _build_hook_program(
        compiled_hooks: list[CompiledHookDescriptor],
    ) -> HookProgram:
        """Build one CSR ``HookProgram`` from aggregate compiled descriptors.

        Args:
            compiled_hooks: Flattened descriptor list that already encodes final
                per-hook ``deme_selector`` routing.

        Returns:
            HookProgram: Plain-data CSR payload consumed by hook execution
            kernels and the native callback bridge.

        Note:
            This function packs all declarative operation arrays into contiguous
            buffers to keep downstream execution loops vectorizable and
            allocation-free during runtime dispatch.
        """
        from natal.frontend.hooks import EVENT_NAMES

        events = EVENT_NAMES
        n_events = len(events)

        hook_offsets: list[int] = [0]
        hook_list_by_event: list[list[CompiledHookDescriptor]] = []
        callback_slots: list[int] = []
        for event_name in events:
            hooks = [h for h in compiled_hooks if h.event == event_name]
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))
            # Number the event's callback slots globally in compact order
            # (groups concatenated); the callback bridges are registered in
            # this same order, so slot j addresses bridge j regardless of
            # which compact group the slot came from.
            next_callback_index = 0
            for hook in hooks:
                if hook.callback is not None:
                    callback_slots.append(next_callback_index)
                    next_callback_index += 1
                else:
                    callback_slots.append(-1)
        n_hooks = hook_offsets[-1]
        all_op_types: list[int] = []
        all_zidx_offsets: list[int] = [0]
        all_zidx_data: list[int] = []
        all_age_offsets: list[int] = [0]
        all_age_data: list[int] = []
        all_sex_masks: list[bool] = []
        all_params: list[float] = []
        all_cond_offsets: list[int] = [0]
        all_cond_types: list[int] = []
        all_cond_params: list[int] = []
        # OP_SET_PARAM / OP_CONVERT flattened data area (rebasing per
        # plan so offsets stay global).
        all_sp_param_ids: list[int] = []
        all_sp_every: list[int] = []
        all_sp_start: list[int] = []
        all_rpn_offsets: list[int] = [0]
        all_rpn_kinds: list[int] = []
        all_rpn_payload: list[int] = []
        all_sp_literals: list[float] = []
        all_convert_source_z: list[int] = []
        all_convert_target_z: list[int] = []
        has_set_param = False
        all_deme_sel_types: list[int] = []
        all_deme_sel_offsets: list[int] = [0]
        all_deme_sel_data: list[int] = []
        n_ops_list: list[int] = []
        op_offsets: list[int] = [0]

        for hooks in hook_list_by_event:
            for hook in hooks:
                plan = hook.plan
                if plan is None or plan.n_ops == 0:
                    # Keep offset arrays aligned even for hooks without
                    # declarative operations (e.g. pure python descriptors):
                    # the deme selector must still be packed, or the Rust
                    # matcher indexes an empty array for this hook slot.
                    n_ops_list.append(0)
                    op_offsets.append(op_offsets[-1])
                    sel = hook.deme_selector
                    if sel == "*":
                        all_deme_sel_types.append(0)
                    elif isinstance(sel, int):
                        all_deme_sel_types.append(1)
                        all_deme_sel_data.append(int(sel))
                    elif isinstance(sel, range):
                        all_deme_sel_types.append(2)
                        all_deme_sel_data.append(int(sel.start))
                        all_deme_sel_data.append(int(sel.stop))
                    else:
                        all_deme_sel_types.append(3)
                        all_deme_sel_data.extend([int(x) for x in sel])
                    all_deme_sel_offsets.append(len(all_deme_sel_data))
                    continue

                n_ops_list.append(plan.n_ops)
                all_op_types.extend(plan.op_types.tolist())
                has_set_param = has_set_param or bool(
                    (plan.op_types == int(OpType.SET_PARAM)).any()
                )

                # Offsets are rebased to flattened buffers as each hook's plan
                # payload is appended.
                zidx_offset_base = len(all_zidx_data)
                for i in range(plan.n_ops):
                    all_zidx_offsets.append(
                        zidx_offset_base
                        + plan.zidx_offsets[i + 1]
                        - plan.zidx_offsets[0]
                    )
                all_zidx_data.extend(plan.zidx_data.tolist())

                age_offset_base = len(all_age_data)
                for i in range(plan.n_ops):
                    all_age_offsets.append(
                        age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0]
                    )
                all_age_data.extend(plan.age_data.tolist())

                all_sex_masks.extend(plan.sex_masks.flatten().tolist())
                all_params.extend(plan.params.tolist())

                cond_offset_base = len(all_cond_types)
                for i in range(plan.n_ops):
                    all_cond_offsets.append(
                        cond_offset_base
                        + plan.condition_offsets[i + 1]
                        - plan.condition_offsets[0]
                    )
                all_cond_types.extend(plan.condition_types.tolist())
                all_cond_params.extend(plan.condition_params.tolist())

                # set_param / convert payload columns and rebased RPN
                # token stream (mirrors the panmictic builder).
                all_sp_param_ids.extend(plan.sp_param_ids.tolist())
                all_sp_every.extend(plan.sp_every.tolist())
                all_sp_start.extend(plan.sp_start.tolist())
                rpn_offset_base = len(all_rpn_kinds)
                for i in range(plan.n_ops):
                    all_rpn_offsets.append(
                        rpn_offset_base + plan.rpn_offsets[i + 1] - plan.rpn_offsets[0]
                    )
                # Literal payloads are indices into the program-wide shared
                # pool: rebase each hook's local indices onto the current
                # pool length before appending, or a later set_param op
                # would silently read an earlier hook's literal (the
                # panmictic compiler rebases the same way).
                literal_base = len(all_sp_literals)
                is_literal = plan.rpn_kinds.tolist()
                payload = plan.rpn_payload.tolist()
                from natal.frontend.hooks.entry.declarative import RPN_LITERAL

                all_rpn_payload.extend(
                    p + literal_base if k == RPN_LITERAL else p
                    for k, p in zip(is_literal, payload)
                )
                all_rpn_kinds.extend(is_literal)
                all_sp_literals.extend(plan.sp_literals.tolist())
                all_convert_source_z.extend(plan.convert_source_z.tolist())
                all_convert_target_z.extend(plan.convert_target_z.tolist())
                op_offsets.append(len(all_op_types))

                # Persist the selector in compact integer encoding for native
                # hook execution.
                sel = hook.deme_selector
                if sel == "*":
                    all_deme_sel_types.append(0)
                elif isinstance(sel, int):
                    all_deme_sel_types.append(1)
                    all_deme_sel_data.append(int(sel))
                elif isinstance(sel, range):
                    all_deme_sel_types.append(2)
                    all_deme_sel_data.append(int(sel.start))
                    all_deme_sel_data.append(int(sel.stop))
                else:
                    all_deme_sel_types.append(3)
                    all_deme_sel_data.extend([int(x) for x in sel])
                all_deme_sel_offsets.append(len(all_deme_sel_data))

        return HookProgram(
            n_events=np.int32(n_events),
            n_hooks=np.int32(n_hooks),
            hook_offsets=np.array(hook_offsets, dtype=np.int32),
            n_ops_list=np.array(n_ops_list, dtype=np.int32),
            op_offsets=np.array(op_offsets, dtype=np.int32),
            op_types_data=np.array(all_op_types, dtype=np.int32),
            zidx_offsets_data=np.array(all_zidx_offsets, dtype=np.int32),
            zidx_data=np.array(all_zidx_data, dtype=np.int32),
            age_offsets_data=np.array(all_age_offsets, dtype=np.int32),
            age_data=np.array(all_age_data, dtype=np.int32),
            sex_masks_data=np.array(all_sex_masks, dtype=np.bool_),
            params_data=np.array(all_params, dtype=np.float64),
            condition_offsets_data=np.array(all_cond_offsets, dtype=np.int32),
            condition_types_data=np.array(all_cond_types, dtype=np.int32),
            condition_params_data=np.array(all_cond_params, dtype=np.int32),
            sp_param_ids=np.array(all_sp_param_ids, dtype=np.int32),
            sp_every=np.array(all_sp_every, dtype=np.int32),
            sp_start=np.array(all_sp_start, dtype=np.int32),
            rpn_offsets=np.array(all_rpn_offsets, dtype=np.int32),
            rpn_kinds=np.array(all_rpn_kinds, dtype=np.int32),
            rpn_payload=np.array(all_rpn_payload, dtype=np.int32),
            sp_literals=np.array(all_sp_literals, dtype=np.float64),
            convert_source_z=np.array(all_convert_source_z, dtype=np.int32),
            convert_target_z=np.array(all_convert_target_z, dtype=np.int32),
            has_set_param=has_set_param,
            deme_selector_types=np.array(all_deme_sel_types, dtype=np.int32),
            deme_selector_offsets=np.array(all_deme_sel_offsets, dtype=np.int32),
            deme_selector_data=np.array(all_deme_sel_data, dtype=np.int32),
            python_callback_slots=np.array(callback_slots, dtype=np.int32),
        )

    def _compile_spatial_hooks_from_demes(self) -> HookProgram:
        """Compile one aggregate hook bundle from current per-deme hooks.

        Uses the **compact** execution plan so that demes sharing identical
        hook sequences produce a single wildcard descriptor instead of one
        per-deme descriptor.  The aggregate bundle carries the CSR registry
        handed to the Rust session at enable/rebuild time.

        Returns:
            HookProgram consumed by the native engine session.

        Implementation detail:
            This function is the single rebuild entrypoint used by
            initialization and ``register_hooks(...)`` so all hook mutation
            paths stay behaviorally consistent.
        """
        compiled_hooks = self._collect_compact_spatial_hooks()
        registry = self._build_hook_program(compiled_hooks)
        return registry

    def _refresh_spatial_hooks(self) -> None:
        """Rebuild the aggregate compiled hooks (single rebuild entrypoint)."""
        self._hooks = self._compile_spatial_hooks_from_demes()
        # Hook structure is session state (plan S3): the changed descriptors
        # must reach the HookProgram and the callback bridges, so the live
        # session rebuilds before the next run — the container-level twin
        # of the panmictic ``_rust_needs_rebuild`` semantics.
        self._rust_needs_rebuild = True

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Trigger an event and execute all registered hooks for a specific deme.

        Args:
            event_name: Event name to trigger.
            deme_id: Deme ID (default: 0).

        Returns:
            int: RESULT_CONTINUE (0) to continue, RESULT_STOP (1) to stop.
        """
        if 0 <= deme_id < self.n_demes:
            return self._demes[deme_id].trigger_event(event_name, deme_id)
        return 0  # RESULT_CONTINUE

    def get_total_count(self) -> int:
        """Return the total count across all demes."""
        self._ensure_rust_states_fresh()
        return int(sum(deme.get_total_count() for deme in self._demes))

    def get_female_count(self) -> int:
        """Return the total female count across all demes."""
        self._ensure_rust_states_fresh()
        return int(sum(deme.get_female_count() for deme in self._demes))

    def get_male_count(self) -> int:
        """Return the total male count across all demes."""
        self._ensure_rust_states_fresh()
        return int(sum(deme.get_male_count() for deme in self._demes))

    def reset(self) -> None:
        """Reset the owning spatial session to initial state and random streams."""
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is None:
            for deme in self._demes:
                deme.reset()
            self._tick = int(self._demes[0].tick)
        else:
            initial = [deme._initial_population_snapshot for deme in self._demes]  # pyright: ignore[reportPrivateUsage]  # immutable initial declarations belong to the container
            ind_all = np.stack([snapshot[0] for snapshot in initial])
            _, _, n_ages, n_ztypes = ind_all.shape
            sperm_all = np.stack([
                snapshot[1] if snapshot[1] is not None else np.zeros((n_ages, n_ztypes, n_ztypes))
                for snapshot in initial
            ])
            backend.set_state(ind_all, sperm_all, 0)
            backend.reseed(int(self._rust_spatial_seed or 0))
            backend.clear_checkpoints()
            self._tick = 0
            self._rust_states_dirty = True
            for deme in self._demes:
                deme._tick = 0  # pyright: ignore[reportPrivateUsage]  # metadata follows the shared native clock
                deme._finished = False  # pyright: ignore[reportPrivateUsage]
                deme._failed = False  # pyright: ignore[reportPrivateUsage]
        self._failed = False
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is not None:
            history_obj.clear()

    def aggregate_individual_count(self) -> NDArray[np.float64]:
        """Return the total individual-count tensor summed over all demes."""
        self._ensure_rust_states_fresh()
        return np.sum(
            np.stack([deme.state.individual_count for deme in self._demes], axis=0),
            axis=0,
        )

    def aggregate_state(self) -> PopulationState:
        """Build one aggregate state for global summaries across all demes."""
        ind_all, sperm_all = self._stack_deme_state_arrays()
        return PopulationState(
            n_tick=int(self._tick),
            individual_count=np.sum(ind_all, axis=0),
            sperm_storage=np.sum(sperm_all, axis=0),
        )

    def compute_allele_frequencies(self) -> dict[str, float]:
        """Compute allele frequencies from the aggregate multi-deme state."""
        allele_counts: dict[str, float] = {}
        locus_totals: dict[str, float] = {}
        genotype_counts = self.aggregate_individual_count().sum(axis=(0, 1))
        registry = self._demes[0].registry

        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                locus_totals[locus.name] = 0.0
                for gene in locus.alleles:
                    allele_counts[gene.name] = 0.0

        for z_idx, (genotype, _slab) in enumerate(registry.index_to_ztype):
            count = genotype_counts[z_idx]
            if count <= 0:
                continue
            for chromosome in self.species.chromosomes:
                for locus in chromosome.loci:
                    mat, pat = genotype.get_alleles_at_locus(locus)
                    for allele in (mat, pat):
                        if allele is not None:
                            allele_counts[allele.name] += float(count)
                            locus_totals[locus.name] += float(count)

        frequencies: dict[str, float] = {}
        for allele_name, count in allele_counts.items():
            gene = self.species.gene_index.get(allele_name)
            if gene is None:
                frequencies[allele_name] = 0.0
                continue
            total = locus_totals[gene.locus.name]
            frequencies[allele_name] = 0.0 if total <= 0.0 else count / total
        return frequencies

    def migration_row(self, source_idx: int) -> NDArray[np.float64]:
        """Return normalized outbound migration weights for one source deme.

        The weights are scattered from the frozen Blueprint CSR and
        renormalized to sum to one (border demes included), matching the
        historical readout semantics.

        Args:
            source_idx: Source deme index.

        Returns:
            A dense float64 vector of length ``n_demes`` with outbound weights
            from ``source_idx``.
        """
        weights = csr_dense_row(self._migration_csr, source_idx, self.n_demes)
        total = float(weights.sum())
        if total > 0.0:
            weights /= total
        return weights

    def _stack_deme_state_arrays(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stack per-deme state arrays along a new deme axis.

        Returns:
            A tuple ``(ind_all, sperm_all)`` where each array has deme as its
            leading axis.

        Note:
            Discrete-generation demes may not expose sperm storage. In that
            case this method synthesizes zero-valued storage arrays with a
            shape compatible with the deme's age/genotype dimensions.
        """
        self._ensure_rust_states_fresh()
        ind_all = np.stack(
            [deme.state.individual_count for deme in self._demes], axis=0
        )

        # Handle potential absence of sperm_storage (e.g. DiscreteGenerationPopulation)
        sperm_list: List[NDArray[np.float64]] = []
        for deme in self._demes:
            s = getattr(deme.state, "sperm_storage", None)
            if s is None:
                # The stacked state already carries the exact active layout;
                # querying a full native config would copy unrelated tensors.
                s = np.zeros(
                    (ind_all.shape[2], ind_all.shape[3], ind_all.shape[3]),
                    dtype=np.float64,
                )
            sperm_list.append(s)

        sperm_all = np.stack(sperm_list, axis=0)
        return ind_all, sperm_all

    def _apply_stacked_state(
        self, ind_all: NDArray[np.float64], sperm_all: NDArray[np.float64], tick: int
    ) -> None:
        """Write one stacked spatial state back into each managed deme.

        Args:
            ind_all: Stacked individual-count array with deme as the first axis.
            sperm_all: Stacked sperm-storage array with deme as the first axis.
            tick: Tick value to assign to each deme and this container.

        Note:
            This method is the only write-back point from stacked kernel state
            into per-deme objects. Keeping it centralized helps preserve tick
            synchronization invariants.
        """
        for deme_id, deme in enumerate(self._demes):
            new_fields = {
                "n_tick": int(tick),
                "individual_count": ind_all[deme_id],
            }
            if hasattr(deme.state, "sperm_storage"):
                new_fields["sperm_storage"] = sperm_all[deme_id]

            # Replace immutable state tuple and keep mirror tick fields aligned.
            deme._state = deme._live_state()._replace(**new_fields)  # type: ignore[attr-defined]  # duck-typed deme doubles carry the same container shape
            deme._tick = int(tick)  # pyright: ignore[reportPrivateUsage]  # owning container publishes snapshot metadata
        self._tick = int(tick)

    def _shared_config(self) -> ConfigObject:
        """Return one shared config for spatial kernels.

        Current spatial kernel wrappers expect equivalent config values for
        all demes.

        Returns:
            The shared exported config object used by every deme.

        Raises:
            TypeError: If a deme does not implement ``export_config``.
            ValueError: If demes export different config values.
        """
        # Spatial kernels assume equivalent config values across demes to avoid
        # per-deme config branching inside native paths.
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        cfg = export_fn()
        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            if not self._configs_match(cfg, deme_export()):
                raise ValueError(
                    f"deme[{idx}] exports different config values; current spatial runner requires equivalent configs"
                )
        return cfg

    def _has_heterogeneous_configs(self) -> bool:
        """Return whether demes export non-equivalent config values."""
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        reference_cfg = export_fn()

        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            if not self._configs_match(reference_cfg, deme_export()):
                return True
        return False

    @staticmethod
    def _configs_match(
        reference_cfg: ConfigObject, candidate_cfg: ConfigObject
    ) -> bool:
        """Return whether two exported configs are equivalent by value.

        Args:
            reference_cfg: Reference config object.
            candidate_cfg: Candidate config object.

        Returns:
            ``True`` when both configs expose the same field layout and equal
            values; otherwise ``False``.
        """
        if reference_cfg is candidate_cfg:
            return True

        field_names = getattr(reference_cfg, "_fields", None)
        candidate_fields = getattr(candidate_cfg, "_fields", None)
        if field_names is not None and candidate_fields is not None:
            if field_names != candidate_fields:
                return False

            for field_name in field_names:
                reference_value = getattr(reference_cfg, field_name)
                candidate_value = getattr(candidate_cfg, field_name)

                if isinstance(reference_value, np.ndarray) or isinstance(
                    candidate_value, np.ndarray
                ):
                    if not isinstance(reference_value, np.ndarray) or not isinstance(
                        candidate_value, np.ndarray
                    ):
                        return False
                    reference_array = cast(NDArray[np.generic], reference_value)
                    candidate_array = cast(NDArray[np.generic], candidate_value)
                    if reference_array.shape != candidate_array.shape:
                        return False
                    if not np.array_equal(reference_array, candidate_array):
                        return False
                    continue

                if reference_value != candidate_value:
                    return False

            return True

        try:
            return bool(reference_cfg == candidate_cfg)
        except Exception:
            return False

    def _is_discrete_demes(self) -> bool:
        """Return whether all demes are discrete-generation (no sperm storage).

        Checks the first deme's state; all demes in a SpatialPopulation are
        expected to share the same population model type.
        """
        if not self._demes:
            return False
        return not hasattr(self._demes[0].state, "sperm_storage")

    def _has_python_hooks(self) -> bool:
        """Return whether any managed deme currently owns Python-layer hooks.

        Returns:
            ``True`` if at least one deme has a Python-callback hook
            registered; otherwise ``False``.
        """
        return any(deme.has_python_callbacks() for deme in self._demes)

    def get_compiled_hooks(
        self,
        event: Optional[str] = None,
    ) -> list[CompiledHookDescriptor]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of ``CompiledHookDescriptor`` sorted by priority.
        """
        hooks = self._collect_effective_compiled_hooks()
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def _has_compiled_hooks(self) -> bool:
        """Return whether any managed deme has compiled native hooks.

        Returns:
            ``True`` if at least one deme reports a non-empty compiled hook
            list; otherwise ``False``.
        """
        for deme in self._demes:
            try:
                if len(deme.get_compiled_hooks()) > 0:
                    return True
            except AttributeError:
                # Some test suites do not implement compiled-hook APIs.
                continue
        return False

    def _ensure_demes_runnable(self, *, context: str) -> None:
        """Raise if any deme is already finished before execution."""
        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot {context}")

    def _assert_consistent_migration_flags(self) -> None:
        """Raise when demes disagree on migration-relevant sampling flags.

        Migration kernels only consume ``stochastic`` and
        ``continuous_sampling``; a mixed set of flags would migrate under
        the first deme's mode while the demes' lifecycles ran under their
        own modes.

        Raises:
            ValueError: When a non-leading deme exports different
                ``stochastic`` or ``continuous_sampling`` values.
        """
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            return
        cfg = export_fn()
        stochastic = bool(getattr(cfg, "stochastic", False))
        continuous = bool(getattr(cfg, "continuous_sampling", False))
        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                continue
            other = deme_export()
            if bool(getattr(other, "stochastic", False)) != stochastic:
                raise ValueError(
                    f"deme[{idx}] has different stochastic; migration requires "
                    "consistent stochastic mode across demes"
                )
            if bool(getattr(other, "continuous_sampling", False)) != continuous:
                raise ValueError(
                    f"deme[{idx}] has different continuous_sampling; migration "
                    "requires consistent sampling mode across demes"
                )

    def _mark_all_demes_stopped(self) -> None:
        """Mark all demes finished and emit the finish event."""
        for deme in self._demes:
            deme._finished = True  # type: ignore[attr-defined]
            deme.trigger_event("finish", deme_id=deme._deme_id)  # pyright: ignore[reportPrivateUsage]  # SpatialPopulation owns its demes; finish hooks must observe the firing deme's own index.

    def _initialize_session(self, seed: int = 0) -> SpatialPopulation:
        """Enable the Rust spatial backend for subsequent runs.

        Both age-structured and discrete-generation spatial populations are
        supported: the session-owned Rust kernel runs each deme's lifecycle
        (declarative hooks and event-level ``set_param`` included) and then
        the CSR migration stage, with one persistent RNG stream per deme.

        Args:
            seed: Base seed; deme *d* uses ``seed ^ d`` for its persistent
                stream.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the Rust extension is unavailable.
        """
        from natal.backends.rust.rust_backend import (
            RustHeterogeneousSpatialLifecycleBackend,
            ecology_columns_from_drafts,
            genetics_variant_bank,
            rust_backend_available,
        )

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "before enabling the Rust backend."
            )

        model = "discrete_generation" if self._is_discrete_demes() else "age_structured"
        # Variant bank: per-deme ecology columns plus a genetics bank
        # deduplicated by tensor content.  Bank size follows genetics
        # diversity only — ecological batch differences never clone
        # variants, and identical genetics never clone ecology.  Both
        # models share this session (plan S3: one Program, per-deme RNG
        # banks, no per-config-bank execution sessions).
        deme_drafts = self._export_deme_drafts(compact=True)
        columns = ecology_columns_from_drafts(deme_drafts)
        columns["migration_rate"] = np.asarray(
            self._params.migration_rate, dtype=np.float64
        ).ravel()
        tensor_bank, deme_variant_ids = genetics_variant_bank(deme_drafts)
        custom_slots = [draft.custom for draft in deme_drafts]
        # Columns and the variant bank own their buffers. Release the full
        # per-deme snapshots before allocating the stacked handoff state.
        del deme_drafts
        compiled_hooks = self._collect_compact_spatial_hooks()
        hook_program = self._build_hook_program(compiled_hooks)
        # One-time build handoff: the session owns the stacked state and
        # the per-deme RNG streams from here on (plan S3).
        ind_all, sperm_all = self._stack_deme_state_arrays()
        self._rust_spatial_backend = RustHeterogeneousSpatialLifecycleBackend(
            self._blueprint,
            columns,
            tensor_bank,
            deme_variant_ids,
            ind_all,
            sperm_all,
            int(self._tick),
            model=model,
            stay_after_send=bool(self._migration_csr.stay_after_send),
            hook_program=hook_program,
            seed=seed,
        )
        from natal.backends.rust.rust_backend import RustDemeParameters

        for index, deme in enumerate(self._demes):
            channel = RustDemeParameters(self._rust_spatial_backend, index, self._invalidate_rust_states, self._prepare_explicit_event)
            channel.set_custom_slots(custom_slots[index])
            deme._runtime_state_reader = self._ensure_rust_states_fresh  # pyright: ignore[reportPrivateUsage]  # retained deme objects share the owning session's lazy read boundary
            deme._runtime_config_reader = channel.config_snapshot  # pyright: ignore[reportPrivateUsage]  # owning container binds the native read projection
            deme._runtime_parameter_writer = channel  # pyright: ignore[reportPrivateUsage]  # owning container binds the native write channel
            deme._rust_lifecycle_backend = None  # pyright: ignore[reportPrivateUsage]  # ownership was transferred; no second standalone session may remain
        self._rust_spatial_seed = seed
        self._rust_states_dirty = False
        self._rust_needs_rebuild = False
        # The fresh session already carries every deme's hook structure;
        # stale per-deme panmictic flags must not re-trigger rebuilds.
        for deme in self._demes:
            deme._rust_needs_rebuild = False  # pyright: ignore[reportPrivateUsage]  # demes are same-package engine hosts; the flag is the shared rebuild contract
        if self._has_python_hooks():
            self._register_spatial_rust_callbacks(self._rust_spatial_backend)
        return self

    def _register_spatial_rust_callbacks(
        self, backend: RustHeterogeneousSpatialLifecycleBackend
    ) -> None:
        """Bridge the demes' Python callbacks into the spatial session.

        Each callback has a separate transaction and a stable deme identity.
        Its parameter, state, and random draws commit together before the
        next callback or lifecycle stage. State arrays are requested only
        when the callback accesses state or metrics.

        Args:
            backend: The freshly constructed spatial backend.
        """
        # Fresh per-deme runners: cloned demes share the template's cached
        # runner (bound to deme 0), so a cached runner would route every
        # fire's ctx.update to the wrong draft.
        from natal.frontend.hooks._transaction import EventTransaction
        from natal.frontend.hooks.tick_context import HookRunner
        from natal.frontend.hooks.types import (
            EVENT_EARLY,
            EVENT_FINISH,
            EVENT_FIRST,
            EVENT_LATE,
            EVENT_NAMES,
        )

        runners = [HookRunner(deme) for deme in self._demes]
        per_deme_adapters = {
            event_id: [runner.rust_callbacks(event_id) for runner in runners]
            for event_id in (EVENT_FIRST, EVENT_EARLY, EVENT_LATE, EVENT_FINISH)
        }
        # One bridge per callback slot of the compact program, in the same
        # order ``_build_hook_program`` numbers them.  Per-deme selectors may
        # filter differently, so a slot's compact position is translated to
        # each deme's own runner index by callback identity — never by
        # positional coincidence.
        compact = self._collect_compact_spatial_hooks()
        bridges: list[list[Callable[..., int]]] = []
        for event_id, event_name in enumerate(EVENT_NAMES):
            event_bridges: list[Callable[..., int]] = []
            for slot_desc in compact:
                if slot_desc.event != event_name or slot_desc.callback is None:
                    continue
                mapping = [
                    runner.callback_index(event_id, slot_desc.callback)
                    for runner in runners
                ]

                def bridge(
                    ind: NDArray[np.float64] | None,
                    sperm: NDArray[np.float64] | None,
                    tick: int,
                    deme_id: int,
                    transaction: EventTransaction,
                    mapping: list[int | None] = mapping,
                    adapters: list[list[Callable[..., int]]] = per_deme_adapters[event_id],
                ) -> int:
                    """Dispatch one callback transaction to its stable deme owner."""
                    index = mapping[int(deme_id)]
                    if index is None:
                        return 0
                    return int(adapters[int(deme_id)][index](ind, sperm, tick, deme_id, transaction))

                bridge.__natal_transaction__ = True  # pyright: ignore[reportFunctionMemberAccess]  # native event transaction ABI
                event_bridges.append(bridge)
            bridges.append(event_bridges)
        backend.set_python_callbacks(*bridges)  # pyright: ignore[reportAttributeAccessIssue]  # native spatial session callback surface

    def _rebuild_stale_spatial_session(self) -> None:
        """Install changed programs without replacing native state or RNG streams."""
        dirty = bool(getattr(self, "_rust_needs_rebuild", False)) or any(
            getattr(deme, "_rust_needs_rebuild", False) for deme in self._demes
        )
        if not dirty:
            return
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is None:
            self._initialize_session(seed=int(self._rust_spatial_seed or 0))
            return
        compiled = self._collect_compact_spatial_hooks()
        backend.configure_program(self._build_hook_program(compiled))
        self._register_spatial_rust_callbacks(backend)
        self._rust_needs_rebuild = False
        for deme in self._demes:
            deme._rust_needs_rebuild = False  # pyright: ignore[reportPrivateUsage]  # owning container installs the shared program

    def _prepare_explicit_event(self) -> None:
        """Bind native event audit ownership even before the first run or snapshot."""
        self._rebuild_stale_spatial_session()
        backend = self._rust_spatial_backend
        assert backend is not None
        backend.bind_history(self.history._store, [deme._params_log for deme in self._demes])  # pyright: ignore[reportPrivateUsage]  # explicit events use the same native stores as run.

    def _invalidate_rust_states(self) -> None:
        """Invalidate every deme snapshot after a native explicit event."""
        self._rust_states_dirty = True

    def _ensure_rust_states_fresh(self) -> None:
        """Refresh the per-deme state caches from the session snapshot.

        One bulk snapshot per boundary: the session owns the authoritative
        stacked state, and the tick boundary (or an explicit reader guard)
        pulls it once instead of Python shuttling arrays through every
        tick.  Idempotent: a no-op when the caches already match.
        """
        backend = getattr(self, "_rust_spatial_backend", None)
        if backend is None or not getattr(self, "_rust_states_dirty", False):
            return
        tick, ind_flat, sperm_flat = backend.state_snapshot()
        n_demes = len(self._demes)
        n_ages = int(self._blueprint.n_ages)
        n_ztypes = int(self._blueprint.n_ztypes)
        ind_all = np.asarray(ind_flat, dtype=np.float64).reshape(
            n_demes, 2, n_ages, n_ztypes
        )
        sperm_all = np.asarray(sperm_flat, dtype=np.float64).reshape(
            n_demes, n_ages, n_ztypes, n_ztypes
        )
        self._rust_states_dirty = False
        self._apply_stacked_state(ind_all, sperm_all, int(tick))

    def _run_rust_spatial_steps(
        self,
        n_steps: int,
        record_every: int,
        clear_history_on_start: bool,
    ) -> bool:
        """Run multiple spatial ticks through the Rust backend with recording."""
        if clear_history_on_start:
            self.clear_history()
        if getattr(self, "_rust_spatial_backend", None) is None:
            self._initialize_session(seed=int(getattr(self, "_rust_spatial_seed", None) or 0))
        self._rebuild_stale_spatial_session()
        backend = self._rust_spatial_backend
        assert backend is not None
        history = self.history
        if history.schema.mode == "observation":
            history._configure_observation(self.observation)  # pyright: ignore[reportPrivateUsage]  # configure native recording once per run
        backend.bind_history(history._store, [deme._params_log for deme in self._demes])  # pyright: ignore[reportPrivateUsage]  # share native history and log ownership
        history._bind_checkpoint_pruner(backend.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.
        tick, stopped = backend.run_steps(n_steps, record_every)
        self._tick = tick
        for deme in self._demes:
            deme._tick = tick  # pyright: ignore[reportPrivateUsage]  # metadata only; arrays remain native
        self._rust_states_dirty = True
        return stopped

    def run_tick(self) -> SpatialPopulation:
        """Run one spatial tick through the session-owned Rust kernel.

        Returns:
            This spatial population instance after in-place state update.

        Raises:
            RuntimeError: If any deme has already finished or the native
                engine extension is unavailable (the session is created by
                ``SpatialConfigurator.build()`` or lazily at the first
                tick).
        """
        return self.run(1, record_every=0)

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> SpatialPopulation:
        """Run multiple spatial ticks through the session-owned Rust kernel.

        Args:
            n_steps: Number of ticks to execute.
            record_every: History recording interval forwarded to the compiled
                spatial kernel.
            finish: Whether to mark all demes finished when the run completes
                without an early stop event.
            clear_history_on_start: Whether to clear existing history before
                appending new snapshots.

        Returns:
            This spatial population instance after in-place state update.

        Raises:
            ValueError: If ``n_steps`` is negative.
            RuntimeError: If any deme has already finished or the native
                engine extension is unavailable (the session is created by
                ``SpatialConfigurator.build()`` or lazily at the first
                tick).
        """
        if getattr(self, "_running", False):
            raise RuntimeError("Nested run is forbidden")
        if getattr(self, "_failed", False):
            raise RuntimeError("Population has failed; restore or reset before run")
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")

        self._ensure_demes_runnable(context="run spatial simulation")
        self._assert_consistent_migration_flags()

        self._running = True
        try:
            if clear_history_on_start:
                self.clear_history()

            was_stopped = self._run_rust_spatial_steps(
                n_steps,
                record_every=record_every,
                clear_history_on_start=False,
            )
            if bool(was_stopped):
                self._mark_all_demes_stopped()
            elif finish:
                assert self._rust_spatial_backend is not None
                self._rust_spatial_backend.stop()
                self._mark_all_demes_stopped()

            return self
        except BaseException:
            self._failed = True
            self._rust_states_dirty = True
            raise
        finally:
            self._running = False
