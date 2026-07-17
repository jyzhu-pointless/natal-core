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
    Protocol,
    Tuple,
    TypeAlias,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.engine.lifecycle_wrappers import (
    LifecycleWrappers,
    compile_lifecycle_wrappers,
)
from natal.engine.spatial_simulator import (
    run_spatial_migration,
)
from natal.genetics import Species
from natal.hooks import (
    CompiledHookDescriptor,
    DemeSelector,
    HookProgram,
)
from natal.numba.utils import is_numba_enabled
from natal.population.base import BasePopulation
from natal.spatial.topology import (
    GridTopology,
    HeterogeneousKernelParams,
    MigrationParams,
    SpatialTopology,
    build_adjacency_matrix,
)

if TYPE_CHECKING:
    from natal.configurator import Configurator
    from natal.modifiers.module import GameteModifier, ZygoteModifier
    from natal.output.history import History
    from natal.output.observation import Observation, ObservationResult
    from natal.presets import GeneticPreset
    from natal.spatial.configurator import SpatialConfigurator

__all__ = ["SpatialPopulation"]

ConfigObject: TypeAlias = object
SpatialStateTuple: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64], int]
DemePopulation: TypeAlias = BasePopulation[PopulationState] | BasePopulation[DiscretePopulationState]

class _ConfigBankProtocol(Protocol):
    """Minimal mutable config bank interface for heterogeneous dispatch."""

    def append(self, value: ConfigObject) -> None:
        """Append one config object."""

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
            raise TypeError(
                "adjacency tuple input must be CSR (indptr, indices, data)"
            )
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


def _normalize_migration_rate(
    rate: float | NDArray[np.float64] | Sequence[float],
    n_ages: int,
    adult_start_age: int,
) -> NDArray[np.float64]:
    """Normalize migration rate to an age-indexed array.

    Scalar inputs: the rate applies only to adult ages (>= adult_start_age);
    juvenile ages (< adult_start_age) default to 0.  In discrete populations
    (n_ages == 1) the single age gets the full rate regardless of
    adult_start_age.  Explicit arrays are used as-is.
    """
    arr = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    if arr.shape == (1,):
        if n_ages > 1 and adult_start_age > 0:
            result = np.full(n_ages, arr[0], dtype=np.float64)
            result[:adult_start_age] = 0.0
            return result
        return np.full(n_ages, arr[0], dtype=np.float64)
    if arr.shape == (n_ages,):
        return arr
    raise ValueError(
        f"migration_rate shape {arr.shape} does not match n_ages={n_ages}"
    )


class _SpatialUpdate:
    """Wrapper returned by :meth:`SpatialPopulation.update`.

    Each chainable method accepts the same parameters as ``Configurator``,
    plus ``BatchSetting`` values for per-deme parameter variation — same API
    as ``SpatialConfigurator`` at build time.
    """

    def __init__(
        self, spatial_pop: SpatialPopulation, *, deme: int | None = None,
    ) -> None:
        """Initialize the per-deme or multi-deme updater.

        Args:
            spatial_pop: The spatial population whose demes will be
                updated.
            deme: When set, updates target only the given deme index;
                when ``None``, updates target all demes.
        """
        self._pop = spatial_pop
        self._deme = deme

    def competition(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._apply_batch_or_scalar("competition", kwargs)
        return self

    def reproduction(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._apply_batch_or_scalar("reproduction", kwargs)
        return self

    def survival(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._apply_batch_or_scalar("survival", kwargs)
        return self

    def fitness(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._dispatch_scalar("fitness", kwargs)
        return self

    def custom(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._apply_batch_or_scalar("custom", kwargs)
        return self

    def setup(
        self, **kwargs: object,  # object: Python **kwargs convention; values validated at call site
    ) -> _SpatialUpdate:
        self._apply_batch_or_scalar("setup", kwargs)
        return self

    def modifiers(
        self,
        gamete_modifiers: list[GameteModifier] | None = None,
        zygote_modifiers: list[ZygoteModifier] | None = None,
    ) -> _SpatialUpdate:
        """Register gamete / zygote modifiers on target deme(s).

        Args:
            gamete_modifiers: Sequence of ``GameteModifier`` instances
                affecting meiosis (genotype → gamete mapping).
            zygote_modifiers: Sequence of ``ZygoteModifier`` instances
                affecting fertilisation (gamete → zygote mapping).

        Returns:
            Self for chaining.
        """
        if self._deme is not None:
            self._pop.update_deme(self._deme).modifiers(
                gamete_modifiers=gamete_modifiers,
                zygote_modifiers=zygote_modifiers,
            )
            return self

        updated_configs: dict[int, PopulationConfig | DiscretePopulationConfig] = {}
        for d in self._pop.demes:
            old_config = d.config
            cid = id(old_config)

            if gamete_modifiers:
                for mod in gamete_modifiers:
                    d.add_gamete_modifier(mod, refresh=False)
            if zygote_modifiers:
                for mod in zygote_modifiers:
                    d.add_zygote_modifier(mod, refresh=False)

            if cid in updated_configs:
                d.set_config(updated_configs[cid])
                continue
            if gamete_modifiers or zygote_modifiers:
                d.refresh_modifier_maps()
            updated_configs[cid] = d.config
        return self

    def presets(self, *presets: GeneticPreset) -> _SpatialUpdate:
        if self._deme is not None:
            self._pop.update_deme(self._deme).presets(*presets)
            return self

        updated_configs: dict[int, PopulationConfig | DiscretePopulationConfig] = {}
        # Modifier callables built by the representative deme of each
        # config-sharing group.  Followers share these by reference so
        # that all demes in a group have identical _gamete_modifiers /
        # _zygote_modifiers — otherwise preset.gamete_modifier(d) would
        # create a fresh callable per deme, and semantically equivalent
        # but numerically different closures produce diverging maps when
        # any follower later calls refresh_modifiers() (F3×Refresh).
        shared_gamete: list[Any] = []
        shared_zygote: list[Any] = []

        for d in self._pop.demes:
            old_config = d.config
            cid = id(old_config)

            # add_preset has identity-based persistent dedup, so calling
            # it with the same object twice (or across two presets()
            # calls) is a no-op.  No per-call seen_presets needed.
            for p in presets:
                d.add_preset(p)

            if cid in updated_configs:
                d._gamete_modifiers = list(shared_gamete)  # pyright: ignore[reportPrivateUsage]
                d._zygote_modifiers = list(shared_zygote)  # pyright: ignore[reportPrivateUsage]
                d.set_config(updated_configs[cid])
                continue
            d.refresh_modifiers()
            shared_gamete = list(d._gamete_modifiers)  # pyright: ignore[reportPrivateUsage]
            shared_zygote = list(d._zygote_modifiers)  # pyright: ignore[reportPrivateUsage]
            d.reapply_preset_fitness()
            updated_configs[cid] = d.config
        return self

    def reconfigure_preset(
        self, preset: GeneticPreset, **changes: object,
    ) -> _SpatialUpdate:
        """Reconfigure a preset parameter on target deme(s).

        Single-deme delegates to ``update_deme()`` → ``reconfigure_preset()``.
        All-deme applies changes once, then rebuilds derived modifier lists
        per deme and recomputes maps per shared config group.

        Validation happens entirely before any mutation (validate-commit
        two-phase): if any target deme lacks the preset registration or an
        attribute name is invalid, the exception is raised and neither the
        preset object nor any deme's config is changed.

        A preset object registered on more than one deme cannot be
        reconfigured on a single deme — the shared object would be mutated
        for all demes while only the target deme's maps are rebuilt,
        leaving the others in an inconsistent state.  Use all-deme
        ``update().reconfigure_preset()`` instead, or register distinct
        preset instances per deme.
        """

        # ── Single-deme path ──
        if self._deme is not None:
            # Forbid reconfiguring a preset shared across multiple demes:
            # setattr mutates the shared preset object in place, affecting
            # all demes that hold a reference, but only the target deme's
            # maps are rebuilt — the others would run with stale tensors
            # while their preset metadata reflects the new parameter.
            sharing_demes = [
                i for i, d in enumerate(self._pop.demes)
                if preset in d._presets  # pyright: ignore[reportPrivateUsage]
            ]
            if len(sharing_demes) > 1:
                raise ValueError(
                    f"Preset {preset.name!r} is registered on "
                    f"{len(sharing_demes)} demes (indices {sharing_demes}). "
                    f"Single-deme reconfigure_preset would mutate the shared "
                    f"preset object but only rebuild deme {self._deme}'s "
                    f"maps, leaving the others inconsistent. Use "
                    f"update().reconfigure_preset() to reconfigure all "
                    f"demes, or register distinct preset instances per deme."
                )
            self._pop.update_deme(self._deme).reconfigure_preset(preset, **changes)
            return self

        # ── All-deme path: validate phase (zero side effects) ──
        for d in self._pop.demes:
            if preset not in d._presets:  # pyright: ignore[reportPrivateUsage]
                raise ValueError(
                    f"Preset {preset.name!r} is not registered on deme "
                    f"{d._name!r}. Use presets() to register it first."  # pyright: ignore[reportPrivateUsage]
                )
        for attr in changes:
            if not hasattr(preset, attr):
                raise AttributeError(
                    f"{type(preset).__name__} {preset.name!r} has no "
                    f"attribute {attr!r}."
                )

        # ── Commit phase ──
        for attr, value in changes.items():
            setattr(preset, attr, value)

        from natal.engine.simulation.age_structured import sync_equilibrium_metrics

        updated_configs: dict[int, PopulationConfig | DiscretePopulationConfig] = {}
        for d in self._pop.demes:
            old_config = d.config
            cid = id(old_config)

            d.refresh_modifiers(rebuild_maps=False)

            if cid in updated_configs:
                d.set_config(updated_configs[cid])
                continue

            d.refresh_modifier_maps()
            d.reapply_preset_fitness()
            sync_equilibrium_metrics(d.config)
            updated_configs[cid] = d.config
        return self

    def _apply_batch_or_scalar(
        self, method_name: str,         kwargs: dict[str, object],  # object: config field values (int, float, ndarray)
    ) -> None:
        """Dispatch kwargs: BatchSetting values → per-deme; scalars → all demes."""
        from natal.spatial.configurator import BatchSetting

        batch_keys = [k for k, v in kwargs.items() if isinstance(v, BatchSetting)]
        if not batch_keys:
            self._dispatch_scalar(method_name, kwargs)
            return

        n_demes = len(self._pop.demes)
        # Expand all batch keys into per-deme value lists
        expanded: dict[str, list[object]] = {}  # object: config field values per deme
        for batch_key in batch_keys:
            batch = cast(BatchSetting[Any], kwargs.pop(batch_key))  # Any: batch setting value type varies per config field
            expanded[batch_key] = batch.expand(n_demes, self._pop.topology)

        # Apply per-deme: each deme gets its slice of batch values + shared scalars.
        # None values in batch lists mean "skip this deme for this parameter".
        for i in range(n_demes):
            per_deme_kwargs: dict[str, object] = dict(kwargs)  # object: config field values
            all_none = True
            for batch_key, vals in expanded.items():
                val = vals[i]
                if val is not None:
                    per_deme_kwargs[batch_key] = val
                    all_none = False
            # Skip demes where every batch value is None (no modification needed)
            if all_none and not kwargs:
                continue
            cfg = self._pop.update_deme(i)
            getattr(cfg, method_name)(**per_deme_kwargs)

    def _dispatch_scalar(
        self, method_name: str,         kwargs: dict[str, object],  # object: config field values (int, float, ndarray)
    ) -> None:
        """Apply scalar kwargs to target deme(s) via the full Configurator method.

        For ``fitness`` — the only method with non-idempotent in-place
        writes (``mode='multiply'``) — deduplicates by the identity of the
        4 fitness tensor arrays rather than by config shell identity.  This
        prevents multiply from being applied multiple times to the same
        shared array when heterogeneous config shells share fitness tensors
        (e.g., ``_build_variant_config``'s shallow copy shares all
        non-replaced arrays across variant groups).

        For all other methods, in-place writes are idempotent (absolute
        assignment via ``field[()] = value``), so shell-identity dedup is
        safe: even if the same array is written twice, the result is the
        same.  Shell-identity dedup also correctly broadcasts ``_replace``
        changes (e.g., ``fixed_egg_count``) to all demes sharing a shell.
        """
        from natal.configurator import Configurator

        if self._deme is not None:
            cfg = self._pop.update_deme(self._deme)
            getattr(cfg, method_name)(**kwargs)
            return

        # Fitness tensors are the write-set for the array-identity dedup.
        # They are shared as a block across variant config shells (all 4
        # come from the same build_config_maps call), so checking all 4
        # identities correctly identifies the sharing group.
        _FITNESS_FIELDS = (
            "viability_fitness", "fecundity_fitness",
            "sexual_selection_fitness", "zygote_viability_fitness",
        )
        use_array_dedup = method_name == "fitness"

        updated_configs: dict[int, PopulationConfig | DiscretePopulationConfig] = {}
        written_arrays: set[int] = set()

        for d in self._pop.demes:
            old_config = d.config
            cid = id(old_config)

            if use_array_dedup:
                # Skip if all fitness arrays have already been written
                # in-place — the write has propagated through the shared
                # arrays to this deme already.
                array_ids = {
                    id(getattr(old_config, f))
                    for f in _FITNESS_FIELDS if hasattr(old_config, f)
                }
                if array_ids <= written_arrays:
                    if cid in updated_configs:
                        d.set_config(updated_configs[cid])
                    continue
                cfg = Configurator.for_population(d)
                getattr(cfg, method_name)(**kwargs)
                written_arrays.update(array_ids)
                updated_configs[cid] = d.config
            else:
                if cid in updated_configs:
                    d.set_config(updated_configs[cid])
                    continue
                cfg = Configurator.for_population(d)
                getattr(cfg, method_name)(**kwargs)
                updated_configs[cid] = d.config


_DETACH_FIELDS: tuple[str, ...] = (
    "carrying_capacity", "eggs_per_female", "sex_ratio",
    "sperm_displacement_rate", "low_density_growth_rate",
    "juvenile_growth_mode", "expected_competition_strength",
    "expected_survival_rate", "generation_time",
    "viability_fitness", "fecundity_fitness",
    "sexual_selection_fitness", "zygote_viability_fitness",
    "custom",
    "age_based_survival_rates", "age_based_mating_rates",
    "age_based_reproduction_rates", "female_age_based_fertility",
    "age_based_relative_competition_strength",
)


class SpatialPopulation:
    """Spatial container composed of per-deme population objects.

    This class models spatial structure via composition: every deme is one
    already-initialized ``BasePopulation`` subclass instance.

    Attributes:
        name (str): Human-readable name for the spatial container.
        demes (Sequence[DemePopulation]): Immutable view of managed demes.
        n_demes (int): Number of demes in the spatial system.
        species (object): Shared species object used by all demes.
        topology (GridTopology | None): Spatial topology used by the landscape.
        adjacency (NDArray[np.float64]): Outbound migration matrix between demes
            when migration mode is ``"adjacency"``.
        migration_strategy (Literal["auto", "adjacency", "kernel", "hybrid"]):
            Strategy selector for migration backend. ``"hybrid"`` is reserved
            for future mixed routing and currently follows ``"auto"`` runtime
            behavior.
        migration_mode (Literal["adjacency", "kernel"]): Active migration backend.
        migration_kernel (NDArray[np.float64] | None): Migration kernel used when
            ``migration_mode`` is ``"kernel"``.
        kernel_bank (tuple[NDArray[np.float64], ...] | None): Optional bank of
            per-pattern kernels reserved for future per-deme kernel routing.
        deme_kernel_ids (NDArray[np.int64] | None): Optional per-deme kernel id
            mapping into ``kernel_bank`` reserved for future mixed routing.
        migration_rate (float): Fraction of each deme that participates in
            migration on each tick.
        tick (int): Current shared simulation tick across all demes.
    """

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
        from natal.spatial.configurator import SpatialConfigurator
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
        migration_rate: float | NDArray[np.float64] | Sequence[float] = 0.0,
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
            migration_strategy: Backend selection policy. ``"auto"`` keeps
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
                config); juvenile ages default to 0.  Array indexed by age.
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

        # Resolve strategy-level policy into one concrete backend mode.
        if migration_strategy == "adjacency":
            migration_mode: Literal["adjacency", "kernel"] = "adjacency"
        elif migration_strategy == "kernel":
            migration_mode = "kernel"
        else:
            # ``auto`` and ``hybrid`` currently share runtime behavior.
            # TODO(spatial-migration/hybrid-dispatch): Implement true hybrid
            # backend selection.
            # Scope:
            # - Add runtime branch policy for mixed routing (not alias to auto).
            # - Allow per-run/per-tick decision between adjacency and kernel.
            # Definition of done:
            # - `migration_strategy="hybrid"` yields behavior distinguishable
            #   from `auto` in at least one tested scenario.
            migration_mode = "kernel" if migration_kernel is not None else "adjacency"

        if migration_mode == "kernel":
            if topology is None:
                raise ValueError("topology is required when migration_kernel is provided")
            has_heterogeneous_kernels = kernel_bank is not None and deme_kernel_ids is not None
            if migration_kernel is None and not has_heterogeneous_kernels:
                raise ValueError(
                    "migration_kernel is required in kernel mode unless kernel_bank and "
                    "deme_kernel_ids are both provided"
                )
            if migration_kernel is not None:
                # Kernels are centered on one source cell; odd dimensions are
                # required so a unique center index exists.
                migration_kernel = np.asarray(migration_kernel, dtype=np.float64)
                if (
                    migration_kernel.ndim != 2
                    or migration_kernel.shape[0] % 2 == 0
                    or migration_kernel.shape[1] % 2 == 0
                ):
                    raise ValueError("migration_kernel must be a 2D array with odd dimensions")

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

        # Heterogeneous kernels use kernel mode directly (no dense pre-build).
        # The kernel migration function selects per-deme kernels on-the-fly.
        if migration_mode == "adjacency" and normalized_kernel_bank is not None and normalized_deme_kernel_ids is not None:
            migration_mode = "kernel"

        self._name = name
        self._topology = topology
        self._adjacency = adjacency_dense
        self._migration_strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = migration_strategy
        self._migration_mode: Literal["adjacency", "kernel"] = migration_mode
        self._migration_kernel = migration_kernel
        self._kernel_bank = normalized_kernel_bank
        self._deme_kernel_ids = normalized_deme_kernel_ids
        self._kernel_include_center = bool(kernel_include_center)
        self._migration_mode_code = 0 if migration_mode == "adjacency" else 1
        ind = self._demes[0].state.individual_count
        n_ages = int(ind.shape[1]) if ind.ndim == 3 else 1
        adult_start_age = self._adult_start_age(n_ages)
        self._migration_rate = _normalize_migration_rate(migration_rate, n_ages, adult_start_age)
        self._adjust_migration_on_edge = bool(adjust_migration_on_edge)
        # Spatial container and all demes share one logical tick counter.
        self._tick = int(self._demes[0].tick)

        # Pre-built parameter bundles for kernel dispatch.
        self._spatial_topo = SpatialTopology(
            rows=0 if topology is None else int(topology.rows),
            cols=0 if topology is None else int(topology.cols),
            wrap=False if topology is None else bool(topology.wrap),
        )
        self._migration_params = MigrationParams(
            kernel=self._migration_kernel_array(),
            include_center=bool(kernel_include_center),
            rate=self._migration_rate,
            adjust_on_edge=bool(adjust_migration_on_edge),
            adjacency=adjacency_dense,
            mode_code=0 if migration_mode == "adjacency" else 1,
        )

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

    def _initialize_default_output_policy(self) -> None:
        """Install identity Observation and raw History for direct construction.

        ``SpatialConfigurator`` replaces these defaults with its explicitly
        compiled policy after construction. The defaults keep the public
        ``SpatialPopulation(demes, ...)`` constructor fully usable on its own.
        """
        from natal.output.history import (
            History,
            HistorySchema,
            PopulationLayout,
            SpatialHistoryLayout,
        )
        from natal.output.observation import Observation

        state = self._demes[0].state
        counts = state.individual_count
        n_sexes, n_ages, n_ztypes = map(int, counts.shape)
        has_sperm = getattr(state, "sperm_storage", None) is not None
        kind: Literal[
            "spatial_age_structured", "spatial_discrete_generation"
        ] = (
            "spatial_discrete_generation"
            if isinstance(state, DiscretePopulationState)
            else "spatial_age_structured"
        )
        registry = getattr(self._demes[0], "_index_registry", None)
        if registry is not None and len(registry.index_to_ztype) == n_ztypes:
            ztype_labels = tuple(
                f"{genotype}@{slab}"
                for genotype, slab in registry.index_to_ztype
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
            row_size=(
                1 + self.n_demes * (ind_per_deme + sperm_per_deme)
            ),
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
    def demes(self) -> Sequence[DemePopulation]:
        """Sequence[DemePopulation]: Immutable view of all managed demes."""
        return tuple(self._demes)

    @property
    def n_demes(self) -> int:
        """int: Number of demes in the spatial system."""
        return len(self._demes)

    @property
    def species(self) -> Species:
        """Species: Shared species object used by all demes."""
        return self._demes[0].species

    @property
    def adjacency(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Outbound migration matrix between demes."""
        return self._adjacency

    @property
    def topology(self) -> GridTopology | None:
        """GridTopology | None: Landscape topology used by the spatial model."""
        return self._topology

    @property
    def migration_mode(self) -> Literal["adjacency", "kernel"]:
        """Literal["adjacency", "kernel"]: Active migration backend."""
        return self._migration_mode

    @property
    def migration_strategy(self) -> Literal["auto", "adjacency", "kernel", "hybrid"]:
        """Literal["auto", "adjacency", "kernel", "hybrid"]: Strategy policy."""
        return self._migration_strategy

    @property
    def migration_kernel(self) -> NDArray[np.float64] | None:
        """NDArray[np.float64] | None: Kernel used by topology-aware migration."""
        return self._migration_kernel

    @property
    def kernel_bank(self) -> tuple[NDArray[np.float64], ...] | None:
        """tuple[NDArray[np.float64], ...] | None: Reserved heterogeneous kernels."""
        return self._kernel_bank

    @property
    def deme_kernel_ids(self) -> NDArray[np.int64] | None:
        """NDArray[np.int64] | None: Reserved per-deme kernel ids."""
        return self._deme_kernel_ids

    def _adult_start_age(self, n_ages: int) -> int:
        """Resolve adult start age from the first deme's config.

        Falls back to 1 for age-structured populations when config is
        unavailable (e.g. test fixtures).
        """
        try:
            return int(self._demes[0].config.new_adult_age)
        except AttributeError:
            return 1 if n_ages > 1 else 0

    @property
    def migration_rate(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Age-specific migration rate per tick."""
        return self._migration_rate

    @migration_rate.setter
    def migration_rate(self, value: float | NDArray[np.float64] | Sequence[float]) -> None:
        ind = self._demes[0].state.individual_count
        n_ages = int(ind.shape[1]) if ind.ndim == 3 else 1
        adult_start_age = self._adult_start_age(n_ages)
        self._migration_rate = _normalize_migration_rate(value, n_ages, adult_start_age)
        # Rebuild MigrationParams so the updated rate reaches both
        # the Python dispatch path and the codegen wrapper path.
        self._migration_params = self._migration_params._replace(rate=self._migration_rate)

    @property
    def adjust_migration_on_edge(self) -> bool:
        """bool: Whether to adjust migration rates on boundaries."""
        return self._adjust_migration_on_edge

    def deme(self, idx: int) -> DemePopulation:
        """Return one deme by positional index.

        Args:
            idx: Zero-based deme index.

        Returns:
            The deme population at ``idx``.
        """
        return self._demes[idx]

    def update(self, deme: int | None = None) -> _SpatialUpdate:
        """Return an updater for modifying this population's config.

        Supports both scalar and ``batch_setting`` values in chain calls,
        matching the ``SpatialConfigurator`` API::

            # Modify all demes simultaneously
            pop.update().competition(carrying_capacity=5000)

            # Modify a specific deme (auto-detaches shared config)
            pop.update(deme=3).competition(carrying_capacity=8000)

            # Batch per-deme modification (same API as build time)
            from natal.spatial.configurator import batch_setting
            pop.update().competition(
                carrying_capacity=batch_setting([100, 200, 300, 400])
            )
        """
        return _SpatialUpdate(self, deme=deme)

    def update_deme(self, demi: int) -> Configurator:
        """Get a Configurator for a specific deme with clone-on-write.

        Per-field detach: for each mutable config array in
        ``_DETACH_FIELDS``, check whether any other deme shares the same
        array object.  Only shared fields are copied into a private
        config; unshared fields stay by reference.

        This replaces the old ``carrying_capacity`` identity proxy, which
        used a single field to infer the sharing state of the entire
        config.  That inference breaks when different config shells share
        some arrays but not others — for example, when a user calls
        ``set_config(config._replace(carrying_capacity=new_K))`` on one
        deme, producing a shell with a private K but shared fitness
        arrays.  The K-proxy would see "K is unique → no sharing → no
        detach", and a subsequent in-place fitness write would penetrate
        to every deme sharing those arrays.
        """
        from natal.configurator import Configurator

        target = self._demes[demi]
        config = target.config

        copied: dict[str, Any] = {}
        for field in _DETACH_FIELDS:
            if not hasattr(config, field):
                continue
            field_array = getattr(config, field)
            # Detach if any other deme shares this exact array object.
            if any(
                getattr(d.config, field, None) is field_array
                for j, d in enumerate(self._demes) if j != demi
            ):
                copied[field] = field_array.copy()

        if copied:
            private = config._replace(**copied)
            object.__setattr__(target, '_config', private)
            config = private

        return Configurator.for_population(target)

    @property
    def tick(self) -> int:
        """int: Shared simulation tick across all demes."""
        return self._tick

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

        from natal.output.observation import ObservationResult

        ind_all, _ = self._stack_deme_state_arrays()
        values = self.observation.apply(ind_all)
        return ObservationResult(
            tick=self._tick,
            _values=values,
            axes=self.observation.axes,
            _labels=MappingProxyType({"group": self.observation.labels}),
        )

    def clear_history(self) -> None:
        """Clear all recorded history."""
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is not None:
            history_obj.clear()

    def _process_kernel_history(
        self,
        history_new: Optional[NDArray[np.float64]],
        clear_history_on_start: bool,
    ) -> None:
        """Process and append history array returned from spatial simulation engine.

        Args:
            history_new: Engine rows with tick in the first column, or ``None``.
            clear_history_on_start: Whether to discard the existing spatial
                timeline before committing the engine rows.

        Raises:
            RuntimeError: If spatial History is not initialized.
            ValueError: If row shape, schema, ordering, or boundary payload is
                inconsistent with the existing History.
        """
        if history_new is None or history_new.shape[0] == 0:
            return

        if clear_history_on_start:
            self.clear_history()

        history_obj = self.history
        rows = history_new
        if history_obj.schema.mode == "observation":
            pop = history_obj.schema.population
            ind_size = pop.n_demes * pop.n_sexes * pop.n_ages * pop.n_ztypes
            projected_rows = np.empty(
                (rows.shape[0], history_obj.schema.row_size), dtype=np.float64
            )
            projected_rows[:, 0] = rows[:, 0]
            for row_index in range(rows.shape[0]):
                counts = rows[row_index, 1 : 1 + ind_size].reshape(
                    pop.n_demes, pop.n_sexes, pop.n_ages, pop.n_ztypes
                )
                values = self.observation.apply(counts)
                projected_rows[row_index, 1:] = values.ravel()
            rows = projected_rows
        elif rows.shape[1] != history_obj.schema.row_size:
            # Discrete spatial kernels carry a zero-valued sperm placeholder
            # for a uniform engine signature. Raw History intentionally omits
            # that transport-only payload.
            rows = rows[:, : history_obj.schema.row_size]

        from natal.output.history import HistoryBatch

        history_obj._append_continuation(  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation
            HistoryBatch(schema=history_obj.schema, rows=rows)
        )

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
        ind_all, sperm_all = self._stack_deme_state_arrays()
        history_obj = self.history
        if history_obj.schema.mode == "observation":
            values = self.observation.apply(ind_all)
            flat = np.empty(history_obj.schema.row_size, dtype=np.float64)
            flat[0] = float(self._tick)
            flat[1:] = values.ravel()
        else:
            sperm_size = (
                sperm_all.size
                if history_obj.schema.population.has_sperm_storage
                else 0
            )
            flat = np.empty(1 + ind_all.size + sperm_size, dtype=np.float64)
            flat[0] = float(self._tick)
            flat[1:1 + ind_all.size] = ind_all.ravel()
            if sperm_size:
                flat[1 + ind_all.size:] = sperm_all.ravel()
        from natal.output.history import HistoryBatch

        batch = HistoryBatch(schema=history_obj.schema, rows=flat[np.newaxis, :])
        if allow_existing:
            history_obj._append_continuation(  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation
                batch
            )
        else:
            if self._tick in history_obj.ticks:
                raise ValueError(f"History already contains tick {self._tick}.")
            history_obj._append(batch)  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation

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
        """Restore spatial population state from a raw-history record.

        Only valid for raw-mode history.  Restores individual counts and
        sperm storage for all demes.  All records after *tick* are removed.

        Args:
            tick: Exact tick to restore.

        Raises:
            ValueError: If mode is not ``"raw"`` or tick is not found.
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
        restored_tick, ic, ss = history_obj.restore_state(tick)
        if ic.ndim >= 3:
            n_demes_restored = ic.shape[0] if ic.ndim == 4 else 1
            for di in range(min(n_demes_restored, len(self._demes))):
                deme = self._demes[di]
                if ic.ndim == 4:
                    deme.state.individual_count[:] = ic[di]
                else:
                    deme.state.individual_count[:] = ic
                if ss is not None:
                    sp = getattr(deme.state, "sperm_storage", None)
                    if sp is not None:
                        sp[:] = ss[di] if ss.ndim == 4 else ss
                deme._state = deme.state._replace(n_tick=restored_tick)  # type: ignore[attr-defined]  # checkpoint must synchronize the immutable state tick
                deme._tick = restored_tick  # type: ignore[attr-defined]  # private attr on base population
        self._tick = restored_tick
        history_obj.truncate(retain_until_tick=tick)

    def _write_history_obj(self, tick: int, flat_row: NDArray[np.float64]) -> None:
        """Append a single (tick, flat_row) to ``_history_obj`` if available."""
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is None:
            return
        from natal.output.history import HistoryBatch

        row = np.empty(1 + len(flat_row), dtype=np.float64)
        row[0] = tick
        row[1:] = flat_row
        batch = HistoryBatch(
            schema=history_obj.schema,
            rows=row[np.newaxis, :],
        )
        history_obj._append(batch)

    @property
    def hooks(self) -> LifecycleWrappers:
        """LifecycleWrappers: Compiled hooks and lifecycle loop functions."""
        return self._hooks

    def set_hook(
        self,
        event_name: str,
        func: Callable[..., None],
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None,
        compile: bool = True,
        deme_selector: Optional[DemeSelector] = None,
    ) -> None:
        """Register an event hook for selected demes.

        If the function carries ``@hook(deme=...)`` metadata and no explicit
        ``deme_selector`` is given, the metadata value is used automatically
        — you don't need to repeat the selector.

        Args:
            event_name: Event name (must exist in ALLOWED_EVENTS).
            func: Callback function.
            hook_id: Numeric execution priority (optional, auto-assigned if omitted).
            hook_name: Optional human-readable name for debugging.
            compile: Whether to try compiling @hook-decorated functions.
            deme_selector: Optional deme selector.  If omitted, reads from
                ``@hook`` metadata automatically.

        Note:
            This API is a spatial convenience entrypoint. ``deme_selector`` is
            interpreted here to choose target demes, then forwarded hooks are
            registered on selected demes with panmictic selector semantics.
            Compiler-level selector fields are transport-only metadata.

            When multiple demes share the same hook storage (as happens with
            homogeneous clones), the hook is registered **once** on a
            representative and all other targeted owners see the change
            through shared storage.  This avoids accidental duplicate
            descriptors that would produce redundant static call sites.
        """
        # Auto-read deme from @hook metadata when no explicit selector given.
        if deme_selector is None:
            meta = getattr(func, 'meta', None)
            if meta is not None:
                demean_meta = meta.get('deme_selector')
                if demean_meta is not None and demean_meta != "*":
                    deme_selector = demean_meta

        # Determine which demes are targeted by the selector.
        target_ids = [
            i for i in range(self.n_demes)
            if deme_selector is None or self._selector_matches_deme(deme_selector, i)
        ]

        if not target_ids:
            return

        # Group targeted demes by the identity of their hook storage.
        # Demes that share the same ``compiled_hook_descriptors`` list (via clone
        # sharing) should only have the hook compiled and appended once.
        storage_groups = self._group_demes_by_hook_storage(target_ids)

        for storage_key, group_ids in storage_groups.items():
            # Find ALL demes (targeted + non-targeted) sharing this storage.
            all_owners = self._demes_sharing_storage(storage_key)

            if set(all_owners).issubset(set(target_ids)):
                # All owners are targeted — register once on a representative;
                # other owners see the change through the shared list.
                self._demes[group_ids[0]].set_hook(
                    event_name, func, hook_id, hook_name, compile, None,
                )
            else:
                # Only a subset of owners are targeted — copy-on-write so the
                # non-targeted owners keep their existing hook storage intact.
                self._copy_hook_storage_for_demes(group_ids)
                self._demes[group_ids[0]].set_hook(
                    event_name, func, hook_id, hook_name, compile, None,
                )

        # Invalidate the hook executor on all targeted demes so they pick up
        # the recompiled aggregate hooks.
        for deme_id in target_ids:
            self._demes[deme_id].hook_executor = None

        # Rebuild aggregate hooks once after all per-deme mutations.
        self._hooks = self._compile_spatial_hooks_from_demes()

    def _group_demes_by_hook_storage(
        self, deme_ids: list[int],
    ) -> dict[tuple[int, int], list[int]]:
        """Group deme indices by the identity of their hook storage.

        Demes that share the same ``compiled_hook_descriptors`` and ``hook_entries`` objects
        (typically clones sharing template storage) are placed in the same
        group.

        Args:
            deme_ids: Indices of demes to group.

        Returns:
            A dict mapping ``(id(compiled_hook_descriptors), id(hook_entries))`` to the list
            of deme indices sharing that storage.
        """
        groups: dict[tuple[int, int], list[int]] = {}
        for deme_id in deme_ids:
            deme = self._demes[deme_id]
            key = (id(deme.compiled_hook_descriptors), id(deme.hook_entries))
            groups.setdefault(key, []).append(deme_id)
        return groups

    def _demes_sharing_storage(
        self, storage_key: tuple[int, int],
    ) -> list[int]:
        """Return all deme indices that share the given hook storage identity.

        Args:
            storage_key: A ``(id(compiled_hook_descriptors), id(hook_entries))`` pair.

        Returns:
            List of deme indices across the entire population whose storage
            matches *storage_key*.
        """
        owners: list[int] = []
        for deme_id in range(self.n_demes):
            deme = self._demes[deme_id]
            if (id(deme.compiled_hook_descriptors), id(deme.hook_entries)) == storage_key:
                owners.append(deme_id)
        return owners

    def _copy_hook_storage_for_demes(
        self, deme_ids: list[int],
    ) -> None:
        """Copy-on-write hook storage for a subset of demes.

        Creates copies of ``compiled_hook_descriptors`` and ``hook_entries`` from the first
        deme in *deme_ids* so they no longer share storage with non-targeted
        peers.  Each event list inside ``hook_entries`` is also copied so that
        future mutations on targeted demes do not leak to untargeted owners.

        Args:
            deme_ids: Indices of demes to receive the new private storage.
                All listed demes will point to the same new copies.
        """
        ref = self._demes[deme_ids[0]]

        compiled_copy = list(ref.compiled_hook_descriptors)
        hooks_copy = {
            event_name: list(entries)
            for event_name, entries in ref.hook_entries.items()
        }

        for deme_id in deme_ids:
            deme = self._demes[deme_id]
            object.__setattr__(deme, 'compiled_hook_descriptors', compiled_copy)
            object.__setattr__(deme, 'hook_entries', hooks_copy)

    def remove_hook(self, event_name: str, hook_id: int) -> bool:
        """Remove a specific hook from all demes.

        Args:
            event_name: Event name.
            hook_id: Hook ID.

        Returns:
            True if removed successfully from all demes, otherwise False.

        Note:
            Hook removal follows the same consistency rule as registration:
            mutate each deme first, then rebuild the aggregate compiled hooks.
        """
        success = True
        for deme in self._demes:
            if not deme.remove_hook(event_name, hook_id):
                success = False

        # Keep aggregate compiled hooks synchronized with deme-local state.
        self._hooks = self._compile_spatial_hooks_from_demes()

        return success

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
            # every deme in the group still executes every slot once.
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
    def _build_hook_program(compiled_hooks: list[CompiledHookDescriptor]) -> HookProgram:
        """Build one CSR ``HookProgram`` from aggregate compiled descriptors.

        Args:
            compiled_hooks: Flattened descriptor list that already encodes final
                per-hook ``deme_selector`` routing.

        Returns:
            HookProgram: Plain-data CSR payload consumed by hook execution
            kernels and the Python ``HookExecutor`` path.

        Note:
            This function packs all declarative operation arrays into contiguous
            buffers to keep downstream execution loops vectorizable and
            allocation-free during runtime dispatch.
        """
        from natal.hooks import EVENT_NAMES

        events = EVENT_NAMES
        n_events = len(events)

        hook_offsets: list[int] = [0]
        hook_list_by_event: list[list[CompiledHookDescriptor]] = []
        for event_name in events:
            hooks = [h for h in compiled_hooks if h.event == event_name]
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))

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
                    # declarative operations (e.g. pure njit/python descriptors).
                    n_ops_list.append(0)
                    op_offsets.append(op_offsets[-1])
                    continue

                n_ops_list.append(plan.n_ops)
                all_op_types.extend(plan.op_types.tolist())

                # Offsets are rebased to flattened buffers as each hook's plan
                # payload is appended.
                zidx_offset_base = len(all_zidx_data)
                for i in range(plan.n_ops):
                    all_zidx_offsets.append(zidx_offset_base + plan.zidx_offsets[i + 1] - plan.zidx_offsets[0])
                all_zidx_data.extend(plan.zidx_data.tolist())

                age_offset_base = len(all_age_data)
                for i in range(plan.n_ops):
                    all_age_offsets.append(age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0])
                all_age_data.extend(plan.age_data.tolist())

                all_sex_masks.extend(plan.sex_masks.flatten().tolist())
                all_params.extend(plan.params.tolist())

                cond_offset_base = len(all_cond_types)
                for i in range(plan.n_ops):
                    all_cond_offsets.append(
                        cond_offset_base + plan.condition_offsets[i + 1] - plan.condition_offsets[0]
                    )
                all_cond_types.extend(plan.condition_types.tolist())
                all_cond_params.extend(plan.condition_params.tolist())
                op_offsets.append(len(all_op_types))

                # Persist compiled selector in compact integer encoding expected
                # by njit-side selector matching helpers.
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
            deme_selector_types=np.array(all_deme_sel_types, dtype=np.int32),
            deme_selector_offsets=np.array(all_deme_sel_offsets, dtype=np.int32),
            deme_selector_data=np.array(all_deme_sel_data, dtype=np.int32),
        )

    def _compile_spatial_hooks_from_demes(self) -> LifecycleWrappers:
        """Compile one aggregate hook bundle from current per-deme hooks.

        Uses the **compact** execution plan so that demes sharing identical
        hook sequences produce a single wildcard descriptor instead of one
        per-deme descriptor.  This keeps the generated lifecycle wrapper
        call graph minimal and avoids redundant static call sites that can
        trigger native instability under prange.

        Returns:
            LifecycleWrappers: Event call chains plus CSR registry and
            pre-compiled lifecycle loop functions.  Used by both generated
            wrappers and Python dispatch fallback.

        Implementation detail:
            This function is the single rebuild entrypoint used by
            initialization, ``set_hook(...)``, and ``remove_hook(...)`` so all
            hook mutation paths stay behaviorally consistent.
        """
        compiled_hooks = self._collect_compact_spatial_hooks()
        registry = self._build_hook_program(compiled_hooks)
        return compile_lifecycle_wrappers(
            compiled_hooks,
            registry=registry,
            include_spatial_wrappers=True,
        )

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
        return int(sum(deme.get_total_count() for deme in self._demes))

    def get_female_count(self) -> int:
        """Return the total female count across all demes."""
        return int(sum(deme.get_female_count() for deme in self._demes))

    def get_male_count(self) -> int:
        """Return the total male count across all demes."""
        return int(sum(deme.get_male_count() for deme in self._demes))

    def reset(self) -> None:
        """Reset all demes and synchronize the container tick.

        This resets each underlying deme using its own reset logic and then
        updates the spatial container tick to match the demes.
        """
        for deme in self._demes:
            deme.reset()
        self._tick = int(self._demes[0].tick)
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is not None:
            history_obj.clear()

    def aggregate_individual_count(self) -> NDArray[np.float64]:
        """Return the total individual-count tensor summed over all demes."""
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

        Args:
            source_idx: Source deme index.

        Returns:
            A dense float64 vector of length ``n_demes`` with outbound weights
            from ``source_idx``.
        """
        if self._migration_mode == "adjacency":
            weights = self._adjacency[source_idx].astype(np.float64, copy=True)
            total = float(weights.sum())
            if total > 0.0:
                weights /= total
            return weights

        assert self._topology is not None, "topology is required for kernel migration"

        # Select kernel (single or from bank).
        if self._deme_kernel_ids is not None and self._kernel_bank is not None:
            kernel = self._kernel_bank[int(self._deme_kernel_ids[source_idx])]
        else:
            assert self._migration_kernel is not None, "migration_kernel required"
            kernel = self._migration_kernel

        weights = np.zeros(self.n_demes, dtype=np.float64)
        src_coord = self._topology.from_index(source_idx)
        kr = kernel.shape[0] // 2
        kc = kernel.shape[1] // 2

        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                if not self._kernel_include_center and row == kr and col == kc:
                    continue
                weight = float(kernel[row, col])
                if weight <= 0.0:
                    continue
                mapped = self._topology.normalize_coord(
                    src_coord[0] + row - kr,
                    src_coord[1] + col - kc,
                )
                if mapped is None:
                    continue
                weights[self._topology.to_index(mapped)] += weight

        total = float(weights.sum())
        if total > 0.0:
            weights /= total
        return weights

    def _stack_deme_state_arrays(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stack per-deme state arrays along a new deme axis.

        Returns:
            A tuple ``(ind_all, sperm_all)`` where each array has deme as its
            leading axis.

        Note:
            Discrete-generation demes may not expose sperm storage. In that
            case this method synthesizes zero-valued storage arrays with a
            shape compatible with the deme's age/genotype dimensions.
        """
        ind_all = np.stack([deme.state.individual_count for deme in self._demes], axis=0)

        # Handle potential absence of sperm_storage (e.g. DiscreteGenerationPopulation)
        sperm_list: List[NDArray[np.float64]] = []
        for deme in self._demes:
            s = getattr(deme.state, "sperm_storage", None)
            if s is None:
                # Create a dummy array if storage is missing
                cfg = getattr(deme, "config", None)
                if cfg is not None and hasattr(cfg, "n_ages") and hasattr(cfg, "n_ztypes"):
                    s = np.zeros((cfg.n_ages, cfg.n_ztypes, cfg.n_ztypes), dtype=np.float64)
                else:
                    # Conservative fallback derived from state tensor shape.
                    ind_shape = deme.state.individual_count.shape
                    s = np.zeros((ind_shape[1], ind_shape[2], ind_shape[2]), dtype=np.float64)
            sperm_list.append(s)

        sperm_all = np.stack(sperm_list, axis=0)
        return ind_all, sperm_all

    def _apply_stacked_state(self, ind_all: NDArray[np.float64], sperm_all: NDArray[np.float64], tick: int) -> None:
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
            deme._state = deme.state._replace(**new_fields)  # type: ignore[attr-defined]
            deme.tick = int(tick)
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
        # per-deme config branching inside njit paths.
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

    def _migration_config(self) -> ConfigObject:
        """Return one config object that carries migration runtime flags.

        Migration kernels only use ``stochastic`` and
        ``continuous_sampling``. Heterogeneous deme configs are supported
        as long as these migration-relevant flags are consistent when
        migration is enabled.

        Returns:
            One exported config object to feed migration kernels.

        Raises:
            TypeError: If a deme does not implement ``export_config``.
            ValueError: If migration is enabled and migration flags differ
                across demes.
        """
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        cfg = export_fn()

        if np.all(self._migration_rate <= 0.0):
            return cfg

        cfg_is_stochastic = bool(getattr(cfg, "stochastic", False))
        cfg_continuous_sampling = bool(getattr(cfg, "continuous_sampling", False))

        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            other_cfg = deme_export()
            if bool(getattr(other_cfg, "stochastic", False)) != cfg_is_stochastic:
                raise ValueError(
                    f"deme[{idx}] has different stochastic; migration requires consistent stochastic mode across demes"
                )
            if bool(getattr(other_cfg, "continuous_sampling", False)) != cfg_continuous_sampling:
                raise ValueError(
                    f"deme[{idx}] has different continuous_sampling; migration requires consistent sampling mode "
                    "across demes"
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
    def _configs_match(reference_cfg: ConfigObject, candidate_cfg: ConfigObject) -> bool:
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

                if isinstance(reference_value, np.ndarray) or isinstance(candidate_value, np.ndarray):
                    if not isinstance(reference_value, np.ndarray) or not isinstance(candidate_value, np.ndarray):
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

    def _migration_kernel_array(self) -> NDArray[np.float64]:
        """Return the migration kernel array expected by compiled kernels."""
        if self._migration_kernel is not None:
            return self._migration_kernel
        # Adjacency mode ignores this argument, but wrapper signatures require
        # one ndarray for all call sites.
        return np.zeros((1, 1), dtype=np.float64)

    def _build_heterogeneous_kernel_arrays(self) -> Optional[HeterogeneousKernelParams]:
        """Build per-kernel offset tables for heterogeneous kernel routing.

        Returns:
            ``HeterogeneousKernelParams`` when ``kernel_bank`` and
            ``deme_kernel_ids`` are both set; ``None`` otherwise.
        """
        from natal.engine.migration.kernel import (
            build_kernel_offset_table,
        )

        if self._kernel_bank is None or self._deme_kernel_ids is None:
            return None

        n_kernels = len(self._kernel_bank)
        max_kernel_size = max(k.shape[0] * k.shape[1] for k in self._kernel_bank)

        kernel_d_row = np.zeros((n_kernels, max_kernel_size), dtype=np.int64)
        kernel_d_col = np.zeros((n_kernels, max_kernel_size), dtype=np.int64)
        kernel_weights = np.zeros((n_kernels, max_kernel_size), dtype=np.float64)
        kernel_nnzs = np.zeros(n_kernels, dtype=np.int64)
        kernel_total_sums = np.zeros(n_kernels, dtype=np.float64)
        max_nnz = 0
        for k in range(n_kernels):
            d_r, d_c, w, nnz, total_sum = build_kernel_offset_table(
                migration_kernel=self._kernel_bank[k],
                kernel_include_center=bool(self._kernel_include_center),
            )
            kernel_d_row[k, :nnz] = d_r[:nnz]
            kernel_d_col[k, :nnz] = d_c[:nnz]
            kernel_weights[k, :nnz] = w[:nnz]
            kernel_nnzs[k] = nnz
            kernel_total_sums[k] = total_sum
            if nnz > max_nnz:
                max_nnz = nnz

        return HeterogeneousKernelParams(
            deme_kernel_ids=self._deme_kernel_ids,
            d_row=kernel_d_row,
            d_col=kernel_d_col,
            weights=kernel_weights,
            nnzs=kernel_nnzs,
            total_sums=kernel_total_sums,
            max_nnz=max_nnz,
        )

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
            ``True`` if at least one deme has hooks registered through the
            legacy Python callback map; otherwise ``False``.
        """
        return any(deme.has_python_hooks() for deme in self._demes)

    def _has_mixed_hook_types(self) -> bool:
        """Return whether any managed deme mixes hook types in one event.

        Returns:
            ``True`` if any deme has an event containing multiple hook payload
            categories (declarative/njit/python); otherwise ``False``.
        """
        return any(deme.has_mixed_hook_types() for deme in self._demes)

    def get_compiled_hooks(self, event: Optional[str] = None) -> list[Any]:
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
        """Return whether any managed deme has compiled (CSR/njit) hooks.

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

    def _should_use_python_dispatch(self) -> bool:
        """Return whether spatial runtime must use Python event dispatch.

        Returns:
            ``True`` if local hook execution is required for this spatial run.
            ``False`` when the simulation can use the compiled njit fast
            path end-to-end (including hooks and heterogeneous configs).

        Note:
            The spatial lifecycle wrapper now supports per-deme hook execution
            (CSR registry) and heterogeneous configs (config bank) inside njit
            with ``prange``. Python dispatch is only needed when Numba is
            disabled or when legacy Python hook callbacks are present.
        """
        if not is_numba_enabled():
            return True
        if self._has_python_hooks():
            return True
        return False

    def _config_equivalence_groups(self) -> list[tuple[ConfigObject, list[int]]]:
        """Group deme indices by value-equivalent exported configs.

        Returns:
            A list of ``(config, deme_indices)`` groups where each group shares
            one value-equivalent config.

        Raises:
            TypeError: If any deme does not implement ``export_config``.
        """
        groups: list[tuple[ConfigObject, list[int]]] = []

        for deme_idx, deme in enumerate(self._demes):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{deme_idx}] does not implement export_config()")
            cfg = deme_export()

            assigned = False
            for group_idx, (group_cfg, group_deme_indices) in enumerate(groups):
                if self._configs_match(group_cfg, cfg):
                    group_deme_indices.append(deme_idx)
                    groups[group_idx] = (group_cfg, group_deme_indices)
                    assigned = True
                    break
            if not assigned:
                groups.append((cfg, [deme_idx]))

        return groups

    def _heterogeneous_config_bank_and_ids(self) -> tuple[object, NDArray[np.int64]]:
        """Build a Numba-typed config bank and per-deme config ids.

        Returns:
            A tuple ``(config_bank, deme_config_ids)`` where ``config_bank`` is
            a numba.typed.List of unique configs and ``deme_config_ids`` maps
            each deme to one config index.

        TODO(spatial-config/flattened-config-bank): Consider replacing the
            typed-list config bank with flattened config matrices plus index
            vectors to stabilize kernel signatures and simplify heterogeneous
            dispatch ABI.
        """
        import importlib

        groups = self._config_equivalence_groups()
        deme_config_ids = np.empty(self.n_demes, dtype=np.int64)

        numba_typed = importlib.import_module("numba.typed")
        config_bank_factory = cast(Callable[[], _ConfigBankProtocol], numba_typed.List)
        config_bank = config_bank_factory()
        for group_id, (group_cfg, group_deme_indices) in enumerate(groups):
            config_bank.append(group_cfg)
            for deme_idx in group_deme_indices:
                deme_config_ids[deme_idx] = np.int64(group_id)

        return cast(object, config_bank), deme_config_ids

    def _ensure_demes_runnable(self, *, context: str) -> None:
        """Raise if any deme is already finished before execution."""
        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot {context}")

    def _mark_all_demes_stopped(self) -> None:
        """Mark all demes finished and emit the finish event."""
        for deme in self._demes:
            deme._finished = True  # type: ignore[attr-defined]
            deme.trigger_event("finish")

    def _run_python_dispatch_tick(self) -> bool:
        """Run one tick via per-deme lifecycle and shared migration.

        Spatial History belongs to this container, so the delegated deme
        lifecycle must not record a second, pre-migration snapshot.

        Returns:
            ``True`` when a deme stops the simulation during its lifecycle;
            otherwise ``False`` after migration is applied.
        """
        for deme in self._demes:
            had_record_every = hasattr(deme, "record_every")
            previous_record_every = getattr(deme, "record_every", 0)
            try:
                deme.record_every = 0
                deme.run_tick()
            finally:
                if had_record_every:
                    deme.record_every = previous_record_every
                else:
                    delattr(deme, "record_every")
            if bool(getattr(deme, "_finished", False)):
                return True

        self._tick = int(self._demes[0].tick)

        config = self._migration_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()

        # Build heterogeneous kernel arrays if needed, else pass None values.
        het = self._build_heterogeneous_kernel_arrays()

        ind_all, sperm_all = run_spatial_migration(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            adjacency=self._migration_params.adjacency,
            migration_mode=self._migration_params.mode_code,
            topology_rows=self._spatial_topo.rows,
            topology_cols=self._spatial_topo.cols,
            topology_wrap=self._spatial_topo.wrap,
            migration_kernel=self._migration_params.kernel,
            kernel_include_center=self._migration_params.include_center,
            config=cast(PopulationConfig, config),
            migration_rate=self._migration_params.rate,
            adjust_migration_on_edge=self._migration_params.adjust_on_edge,
            deme_kernel_ids=het.deme_kernel_ids if het is not None else None,
            kernel_d_row=het.d_row if het is not None else None,
            kernel_d_col=het.d_col if het is not None else None,
            kernel_weights=het.weights if het is not None else None,
            kernel_nnzs=het.nnzs if het is not None else None,
            kernel_total_sums=het.total_sums if het is not None else None,
            max_nnz=het.max_nnz if het is not None else 0,
        )
        self._apply_stacked_state(ind_all, sperm_all, int(self._tick))
        return False

    def _run_codegen_wrapper_tick(self) -> bool:
        """Run one tick through the njit spatial lifecycle wrapper.

        Uses the pre-compiled spatial lifecycle wrapper from
        ``CompiledEventHooks``, which handles per-deme hook execution inside
        ``prange`` followed by migration — all in compiled Numba code.
        """
        ind_all, sperm_all = self._stack_deme_state_arrays()
        config_bank, deme_config_ids = self._heterogeneous_config_bank_and_ids()

        if self._is_discrete_demes():
            tick_fn = self._hooks.spatial_discrete_tick_fn
        else:
            tick_fn = self._hooks.spatial_tick_fn
        assert tick_fn is not None, "spatial lifecycle wrapper not compiled (Numba disabled?)"
        assert self._hooks.hooks.registry is not None, "spatial hooks must have a compiled registry"  # type: ignore[unreachable-code]

        het = self._build_heterogeneous_kernel_arrays()
        ind, sperm, tick, was_stopped = tick_fn(
            ind_all, sperm_all,
            config_bank, deme_config_ids,
            self._hooks.hooks.registry, int(self._tick),
            self._spatial_topo,
            self._migration_params,
            het,
        )
        self._apply_stacked_state(ind, sperm, int(tick))
        return bool(was_stopped)

    def _run_codegen_wrapper_steps(self, n_steps: int, *, record_every: int, clear_history_on_start: bool = True) -> bool:
        """Run multiple ticks through the njit spatial lifecycle wrapper.

        Uses the pre-compiled spatial lifecycle ``run`` function, which handles
        per-deme hook execution, migration, and optional history recording
        entirely in compiled Numba code.
        """
        ind_all, sperm_all = self._stack_deme_state_arrays()
        config_bank, deme_config_ids = self._heterogeneous_config_bank_and_ids()

        if self._is_discrete_demes():
            run_fn = self._hooks.spatial_discrete_run_fn
        else:
            run_fn = self._hooks.spatial_run_fn
        assert run_fn is not None, "spatial lifecycle run wrapper not compiled (Numba disabled?)"
        assert self._hooks.hooks.registry is not None, "spatial hooks must have a compiled registry"  # pyright: ignore[reportUnreachable]

        het = self._build_heterogeneous_kernel_arrays()
        final_state_tuple, history_new, was_stopped = run_fn(
            ind_all, sperm_all,
            config_bank, deme_config_ids,
            self._hooks.hooks.registry, int(self._tick), int(n_steps),
            self._spatial_topo,
            self._migration_params,
            het,
            record_interval=int(record_every),
        )
        self._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))
        self._process_kernel_history(history_new, clear_history_on_start)
        return bool(was_stopped)

    def run_tick(self) -> SpatialPopulation:
        """Run one spatial tick via the spatial kernel.

        Returns:
            This spatial population instance after in-place state update.

        Raises:
            RuntimeError: If any deme has already finished.
        """
        self._ensure_demes_runnable(context="run spatial tick")

        if self._should_use_python_dispatch():
            # Hook-aware fallback: preserve per-deme local hook semantics.
            was_stopped = self._run_python_dispatch_tick()
        else:
            # Global Numba path: run spatial kernel for one full spatial tick.
            was_stopped = self._run_codegen_wrapper_tick()
        if was_stopped:
            self._mark_all_demes_stopped()
        return self

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> SpatialPopulation:
        """Run multiple spatial ticks via the spatial kernel.

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
            RuntimeError: If any deme has already finished.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")

        self._ensure_demes_runnable(context="run spatial simulation")

        self._running = True
        try:
            if clear_history_on_start:
                self.clear_history()

            if self._should_use_python_dispatch():
                # Hook-aware fallback: keep local hook timeline semantics.
                was_stopped = False
                if record_every > 0 and (self._tick % record_every == 0):
                    self._record_snapshot(allow_existing=True)
                for _ in range(n_steps):
                    if self._run_python_dispatch_tick():
                        was_stopped = True
                        break
                    if record_every > 0 and (self._tick % record_every == 0):
                        self._record_snapshot(allow_existing=True)
            else:
                # Global Numba path: run batched spatial kernel.
                was_stopped = self._run_codegen_wrapper_steps(
                    n_steps,
                    record_every=int(record_every),
                    clear_history_on_start=clear_history_on_start,
                )
            if bool(was_stopped):
                self._mark_all_demes_stopped()
            elif finish:
                for deme in self._demes:
                    deme.finish_simulation()

            return self
        finally:
            self._running = False
