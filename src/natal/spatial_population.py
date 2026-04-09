"""Composition-based spatial population container.

`SpatialPopulation` intentionally does NOT inherit from ``BasePopulation``.
Each deme is represented by one concrete ``BasePopulation`` subclass instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable, List, Literal, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.hook_dsl import DemeSelector
from natal.population_state import PopulationState
from natal.spatial_topology import (
    GridTopology,
    build_adjacency_matrix,
)

__all__ = ["SpatialPopulation"]


def _coerce_adjacency_dense(
    adjacency: Any,
    n_demes: int,
) -> NDArray[np.float64]:
    """Coerce dense or sparse-like adjacency input to a dense float64 matrix.

    Supported forms:

    - Dense ``np.ndarray`` with shape ``(n_demes, n_demes)``.
    - CSR tuple ``(indptr, indices, data)``.
    - Any object exposing ``toarray()`` (for example scipy sparse matrices).

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
    adjacency_obj = cast(object, adjacency)

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


def _build_heterogeneous_kernel_adjacency(
    topology: GridTopology,
    kernel_bank: tuple[NDArray[np.float64], ...],
    deme_kernel_ids: NDArray[np.int64],
    kernel_include_center: bool,
) -> NDArray[np.float64]:
    """Build one dense effective adjacency from per-deme kernel assignments.

    Each source deme selects one kernel by ``deme_kernel_ids[src]``. The
    selected kernel is projected onto topology coordinates to build one
    normalized outbound row for that source.

    Args:
        topology: Grid topology for coordinate/index conversion.
        kernel_bank: Available kernels.
        deme_kernel_ids: Per-source kernel id mapping.
        kernel_include_center: Whether the center cell is included.

    Returns:
        A dense row-normalized adjacency matrix.
    """
    n_demes = topology.n_demes
    adjacency = np.zeros((n_demes, n_demes), dtype=np.float64)

    # Build one outbound row per source deme, then row-normalize.
    for src in range(n_demes):
        kernel_id = int(deme_kernel_ids[src])
        kernel = kernel_bank[kernel_id]
        center_row = kernel.shape[0] // 2
        center_col = kernel.shape[1] // 2
        src_coord = topology.from_index(src)

        row_total = 0.0
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                if (not kernel_include_center) and row == center_row and col == center_col:
                    continue
                weight = float(kernel[row, col])
                if weight <= 0.0:
                    continue

                mapped = topology.normalize_coord(
                    src_coord[0] + row - center_row,
                    src_coord[1] + col - center_col,
                )
                if mapped is None:
                    continue
                dst = topology.to_index(mapped)
                adjacency[src, dst] += weight
                row_total += weight

        # Normalize each source row so migration code can treat it as
        # probability weights directly.
        if row_total > 0.0:
            adjacency[src, :] /= row_total

    return adjacency


class SpatialPopulation:
    """Spatial container composed of per-deme population objects.

    This class models spatial structure via composition: every deme is one
    already-initialized ``BasePopulation`` subclass instance.

    Attributes:
        name (str): Human-readable name for the spatial container.
        demes (Sequence[BasePopulation[Any]]): Immutable view of managed demes.
        n_demes (int): Number of demes in the spatial system.
        species (Any): Shared species object used by all demes.
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

    def __init__(
        self,
        demes: Sequence[BasePopulation[Any]],
        *,
        topology: Optional[GridTopology] = None,
        adjacency: Optional[object] = None,
        migration_kernel: Optional[NDArray[np.float64]] = None,
        migration_strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = "auto",
        kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None,
        deme_kernel_ids: Optional[NDArray[np.int64]] = None,
        kernel_include_center: bool = False,
        migration_rate: float = 0.0,
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
        self._demes: List[BasePopulation[Any]] = list(demes)

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

        # Initialize hooks system
        if all(hasattr(deme, "_hooks_obj") for deme in self._demes):
            # Use the first deme's hooks_obj for compatibility with tests.
            self._hooks: Any = self._demes[0]._hooks_obj  # type: ignore[attr-defined]
        else:
            self._hooks = self._compile_spatial_hooks_from_demes()

        heterogeneous_kernel_adjacency: NDArray[np.float64] | None = None
        if topology is not None and normalized_kernel_bank is not None and normalized_deme_kernel_ids is not None:
            # Pre-build dense effective routing matrix once so runtime kernels
            # can stay simple even under per-deme heterogeneous kernels.
            heterogeneous_kernel_adjacency = _build_heterogeneous_kernel_adjacency(
                topology=topology,
                kernel_bank=normalized_kernel_bank,
                deme_kernel_ids=normalized_deme_kernel_ids,
                kernel_include_center=bool(kernel_include_center),
            )

        self._name = name
        self._topology = topology
        self._adjacency = adjacency_dense
        self._migration_strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = migration_strategy
        self._migration_mode: Literal["adjacency", "kernel"] = migration_mode
        self._migration_kernel = migration_kernel
        # Effective adjacency compiled from per-deme kernels. When present,
        # run_tick/run dispatch migration through adjacency mode so the
        # heterogeneous kernel assignment is active end-to-end.
        self._heterogeneous_kernel_adjacency = heterogeneous_kernel_adjacency
        # TODO(spatial-migration/heterogeneous-kernel-kernelpath): Replace
        # adjacency-mode fallback with a dedicated heterogeneous-kernel kernel
        # execution path to avoid dense materialization for very large grids.
        self._kernel_bank = normalized_kernel_bank
        self._deme_kernel_ids = normalized_deme_kernel_ids
        self._kernel_include_center = bool(kernel_include_center)
        self._migration_mode_code = 0 if migration_mode == "adjacency" else 1
        self._migration_rate = float(migration_rate)
        # Spatial container and all demes share one logical tick counter.
        self._tick = int(self._demes[0].tick)

        for idx, deme in enumerate(self._demes[1:], start=1):
            if int(deme.tick) != self._tick:
                raise ValueError(
                    f"deme[{idx}] tick ({deme.tick}) does not match deme[0] tick ({self._tick})"
                )

    @property
    def name(self) -> str:
        """str: Human-readable name for the spatial container."""
        return self._name

    @property
    def demes(self) -> Sequence[BasePopulation[Any]]:
        """Sequence[BasePopulation[Any]]: Immutable view of all managed demes."""
        return tuple(self._demes)

    @property
    def n_demes(self) -> int:
        """int: Number of demes in the spatial system."""
        return len(self._demes)

    @property
    def species(self) -> Any:
        """Any: Shared species object used by all demes."""
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

    @property
    def migration_rate(self) -> float:
        """float: Fraction of each deme that migrates on each tick."""
        return self._migration_rate

    @migration_rate.setter
    def migration_rate(self, value: float) -> None:
        self._migration_rate = float(value)

    def deme(self, idx: int) -> BasePopulation[Any]:
        """Return one deme by positional index.

        Args:
            idx: Zero-based deme index.

        Returns:
            The deme population at ``idx``.
        """
        return self._demes[idx]

    @property
    def tick(self) -> int:
        """int: Shared simulation tick across all demes."""
        return self._tick

    @property
    def hooks(self) -> Any:
        """Any: Global event hooks shared across all demes."""
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
        """Register an event hook for all demes.

        Args:
            event_name: Event name (must exist in ALLOWED_EVENTS).
            func: Callback function.
            hook_id: Numeric execution priority (optional, auto-assigned if omitted).
            hook_name: Optional human-readable name for debugging.
            compile: Whether to try compiling @hook-decorated functions.
            deme_selector: Optional deme selector.

        Note:
            This API is a spatial convenience entrypoint: it forwards the same
            hook registration call to every deme, then rebuilds one aggregate
            compiled hook bundle.
        """
        # Register hook in all demes
        for deme in self._demes:
            deme.set_hook(event_name, func, hook_id, hook_name, compile, deme_selector)

        # Rebuild aggregate hooks once after all per-deme mutations.
        self._hooks = self._compile_spatial_hooks_from_demes()

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

    def _collect_effective_compiled_hooks(self) -> list[Any]:
        """Collect hooks from each deme and pin each one to its owner deme.

        Local spatial hook semantics are per-deme: ordering and execution scope
        are only defined inside each deme. This method lifts per-deme hook
        descriptors into one aggregate list while forcing ``deme_selector`` to
        the owning deme id.

        Returns:
            A flat list of hook descriptors safe for aggregate spatial
            execution. Every descriptor in the returned list has
            ``deme_selector`` rewritten to one concrete integer deme id.

        Note:
            Rewriting selectors here avoids accidental cross-deme execution
            after flattening all demes into a single compiled registry.
        """
        compiled_hooks: list[Any] = []
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
    def _build_hook_program(compiled_hooks: list[Any]) -> Any:
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
        from natal.hook_dsl import EVENT_NAMES, HookProgram

        events = EVENT_NAMES
        n_events = len(events)

        hook_offsets: list[int] = [0]
        hook_list_by_event: list[list[Any]] = []
        for event_name in events:
            hooks = [h for h in compiled_hooks if h.event == event_name]
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))

        n_hooks = hook_offsets[-1]
        all_op_types: list[int] = []
        all_gidx_offsets: list[int] = [0]
        all_gidx_data: list[int] = []
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
                gidx_offset_base = len(all_gidx_data)
                for i in range(plan.n_ops):
                    all_gidx_offsets.append(gidx_offset_base + plan.gidx_offsets[i + 1] - plan.gidx_offsets[0])
                all_gidx_data.extend(plan.gidx_data.tolist())

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
            gidx_offsets_data=np.array(all_gidx_offsets, dtype=np.int32),
            gidx_data=np.array(all_gidx_data, dtype=np.int32),
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

    def _compile_spatial_hooks_from_demes(self) -> Any:
        """Compile one aggregate hook bundle from current per-deme hooks.

        Returns:
            CompiledEventHooks: Event call chains plus CSR registry used by
            both generated wrappers and Python dispatch fallback.

        Implementation detail:
            This function is the single rebuild entrypoint used by
            initialization, ``set_hook(...)``, and ``remove_hook(...)`` so all
            hook mutation paths stay behaviorally consistent.
        """
        from natal.hook_dsl import CompiledEventHooks

        compiled_hooks = self._collect_effective_compiled_hooks()
        registry = self._build_hook_program(compiled_hooks)
        return CompiledEventHooks.from_compiled_hooks(compiled_hooks, registry=registry)

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

        for genotype_idx, count in enumerate(genotype_counts):
            if count <= 0:
                continue
            genotype = registry.index_to_genotype[genotype_idx]
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
        if self._heterogeneous_kernel_adjacency is not None:
            # Fast path: precomputed dense matrix for heterogeneous kernels.
            return self._heterogeneous_kernel_adjacency[source_idx].astype(np.float64, copy=True)

        if self._migration_mode == "adjacency":
            weights = self._adjacency[source_idx].astype(np.float64, copy=True)
            total = float(weights.sum())
            if total > 0.0:
                weights /= total
            return weights

        assert self._topology is not None, "topology is required for kernel migration"
        assert self._migration_kernel is not None, "migration_kernel is required for kernel migration"

        weights = np.zeros(self.n_demes, dtype=np.float64)
        src_coord = self._topology.from_index(source_idx)
        kernel = self._migration_kernel
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
                if cfg is not None and hasattr(cfg, "n_ages") and hasattr(cfg, "n_genotypes"):
                    s = np.zeros((cfg.n_ages, cfg.n_genotypes, cfg.n_genotypes), dtype=np.float64)
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

    def _shared_config(self) -> Any:
        """Return one shared config for spatial kernels.

        Current spatial kernel wrappers expect one config object for all demes.

        Returns:
            The shared exported config object used by every deme.

        Raises:
            TypeError: If a deme does not implement ``export_config``.
            ValueError: If demes export different config objects.
        """
        # Spatial kernels assume one shared config object to avoid per-deme
        # config branching inside njit paths.
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        cfg = export_fn()
        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            # Identity check is intentional: semantic equality is not enough
            # for kernels that capture config fields through one object.
            if deme_export() is not cfg:
                raise ValueError(
                    f"deme[{idx}] uses a different config object; current spatial runner requires a shared config"
                )
        return cfg

    def _migration_kernel_array(self) -> NDArray[np.float64]:
        """Return the migration kernel array expected by compiled kernels."""
        if self._migration_kernel is not None:
            return self._migration_kernel
        # Adjacency mode ignores this argument, but wrapper signatures require
        # one ndarray for all call sites.
        return np.zeros((1, 1), dtype=np.float64)

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
                # Some test doubles do not implement compiled-hook APIs.
                continue
        return False

    def _should_use_python_dispatch(self) -> bool:
        """Return whether spatial runtime must use Python event dispatch.

        Returns:
            ``True`` if local hook execution is required for this spatial run.
            ``False`` when the simulation can use the hook-free compiled fast
            path end-to-end.

        Note:
            Current compiled spatial wrappers intentionally skip local hook
            insertion. Therefore, presence of either Python or compiled hooks
            forces execution through the per-deme Python dispatch timeline.
        """
        return self._has_python_hooks() or self._has_compiled_hooks()

    def run_tick(self) -> SpatialPopulation:
        """Run one spatial tick via the generated spatial wrapper.

        Returns:
            This spatial population instance after in-place state update.

        Raises:
            RuntimeError: If any deme has already finished.
        """
        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot run spatial tick")

        if self._should_use_python_dispatch():
            from natal.hooks.executor import run_spatial_tick_with_hooks

            # Hook-aware path: preserve per-deme local hook semantics.
            was_stopped = bool(run_spatial_tick_with_hooks(self))
            if was_stopped:
                for deme in self._demes:
                    deme._finished = True  # type: ignore[attr-defined]
                    deme.trigger_event("finish")
            return self

        # Hook-free fast path: one compiled spatial wrapper call.
        assert self._hooks.run_spatial_tick_fn is not None, "hooks.run_spatial_tick_fn should always be initialized"
        assert self._hooks.registry is not None, "hooks.registry should always be initialized"

        run_tick_fn = cast(Callable[..., Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], int]], self._hooks.run_spatial_tick_fn)
        registry = self._hooks.registry
        config = self._shared_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()
        effective_adjacency = self._adjacency
        effective_migration_mode_code = self._migration_mode_code
        if self._heterogeneous_kernel_adjacency is not None:
            effective_adjacency = self._heterogeneous_kernel_adjacency
            effective_migration_mode_code = 0

        final_state_tuple, result = run_tick_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=int(self._tick),
            adjacency=effective_adjacency,
            # TODO(spatial-migration/wrapper-signature-run-tick): Extend
            # wrapper signature to pass heterogeneous routing payloads.
            # Scope:
            # - Add explicit args for sparse cache and per-deme kernel routing.
            # - Keep backward compatibility for existing generated wrappers.
            migration_mode=effective_migration_mode_code,
            topology_rows=0 if self._topology is None else int(self._topology.rows),
            topology_cols=0 if self._topology is None else int(self._topology.cols),
            topology_wrap=False if self._topology is None else bool(self._topology.wrap),
            migration_kernel=self._migration_kernel_array(),
            kernel_include_center=bool(self._kernel_include_center),
            migration_rate=float(self._migration_rate),
        )

        self._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))

        if int(result) != 0:
            for deme in self._demes:
                deme._finished = True  # type: ignore[attr-defined]
                deme.trigger_event("finish")
        return self

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
    ) -> SpatialPopulation:
        """Run multiple spatial ticks via the generated spatial wrapper.

        Args:
            n_steps: Number of ticks to execute.
            record_every: History recording interval forwarded to the compiled
                spatial kernel.
            finish: Whether to mark all demes finished when the run completes
                without an early stop event.

        Returns:
            This spatial population instance after in-place state update.

        Raises:
            ValueError: If ``n_steps`` is negative.
            RuntimeError: If any deme has already finished.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")

        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot run spatial simulation")

        if self._should_use_python_dispatch():
            from natal.hooks.executor import run_spatial_tick_with_hooks

            # Hook-aware execution iterates tick-by-tick so stop signals can
            # short-circuit immediately after each local event timeline.
            was_stopped = False
            for _ in range(n_steps):
                if bool(run_spatial_tick_with_hooks(self)):
                    was_stopped = True
                    break
            if was_stopped:
                for deme in self._demes:
                    deme._finished = True  # type: ignore[attr-defined]
                    deme.trigger_event("finish")
            elif finish:
                for deme in self._demes:
                    deme.finish_simulation()
            return self

        # Hook-free fast path delegates the whole batch to one compiled wrapper
        # call, which may still early-stop if runtime result requests it.
        assert self._hooks.run_spatial_fn is not None, "hooks.run_spatial_fn should always be initialized"
        assert self._hooks.registry is not None, "hooks.registry should always be initialized"

        run_fn = cast(Callable[..., Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64], bool]], self._hooks.run_spatial_fn)
        registry = self._hooks.registry
        config = self._shared_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()
        effective_adjacency = self._adjacency
        effective_migration_mode_code = self._migration_mode_code
        if self._heterogeneous_kernel_adjacency is not None:
            effective_adjacency = self._heterogeneous_kernel_adjacency
            effective_migration_mode_code = 0

        final_state_tuple, _history, was_stopped = run_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=int(self._tick),
            n_ticks=int(n_steps),
            adjacency=effective_adjacency,
            # TODO(spatial-migration/wrapper-signature-run): Extend wrapper
            # signature to pass heterogeneous routing payloads.
            # Scope:
            # - Mirror run_tick signature additions for batch run wrapper.
            # - Ensure generated template and runtime call sites stay aligned.
            migration_mode=effective_migration_mode_code,
            topology_rows=0 if self._topology is None else int(self._topology.rows),
            topology_cols=0 if self._topology is None else int(self._topology.cols),
            topology_wrap=False if self._topology is None else bool(self._topology.wrap),
            migration_kernel=self._migration_kernel_array(),
            kernel_include_center=bool(self._kernel_include_center),
            migration_rate=float(self._migration_rate),
            record_interval=int(record_every),
        )

        self._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))

        if bool(was_stopped):
            for deme in self._demes:
                deme._finished = True  # type: ignore[attr-defined]
                deme.trigger_event("finish")
        elif finish:
            for deme in self._demes:
                deme.finish_simulation()

        return self
