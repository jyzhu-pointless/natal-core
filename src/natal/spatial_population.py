"""Composition-based spatial population container.

`SpatialPopulation` intentionally does NOT inherit from ``BasePopulation``.
Each deme is represented by one concrete ``BasePopulation`` subclass instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, List, Literal, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.population_state import PopulationState
from natal.spatial_topology import (
    GridTopology,
    build_adjacency_matrix,
)

__all__ = ["SpatialPopulation"]


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
        migration_mode (Literal["adjacency", "kernel"]): Active migration backend.
        migration_kernel (NDArray[np.float64] | None): Migration kernel used when
            ``migration_mode`` is ``"kernel"``.
        migration_rate (float): Fraction of each deme that participates in
            migration on each tick.
        tick (int): Current shared simulation tick across all demes.
    """

    def __init__(
        self,
        demes: Sequence[BasePopulation[Any]],
        *,
        topology: Optional[GridTopology] = None,
        adjacency: Optional[NDArray[np.float64]] = None,
        migration_kernel: Optional[NDArray[np.float64]] = None,
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
                ``(n_demes, n_demes)``.
            migration_kernel: Optional odd-shaped 2D kernel used for topology-
                aware migration. When provided, ``topology`` is required and
                migration runs in kernel mode.
            kernel_include_center: Whether kernel migration includes the kernel
                center as an outbound target for the source deme.
            migration_rate: Fraction of each deme that migrates each tick.
            name: Human-readable container name.

        Raises:
            ValueError: If ``demes`` is empty, demes do not share the same
                species object, topology size does not match the number of
                demes, adjacency shape is invalid, migration kernel is invalid,
                or deme ticks do not match.
        """
        if not demes:
            raise ValueError("demes must contain at least one BasePopulation instance")

        self._demes: List[BasePopulation[Any]] = list(demes)

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

        migration_mode: Literal["adjacency", "kernel"] = "kernel" if migration_kernel is not None else "adjacency"
        if migration_mode == "kernel":
            if topology is None:
                raise ValueError("topology is required when migration_kernel is provided")
            migration_kernel = np.asarray(migration_kernel, dtype=np.float64)
            if migration_kernel.ndim != 2 or migration_kernel.shape[0] % 2 == 0 or migration_kernel.shape[1] % 2 == 0:
                raise ValueError("migration_kernel must be a 2D array with odd dimensions")

        if adjacency is None:
            if topology is None:
                adjacency = np.eye(n_demes, dtype=np.float64)
            else:
                adjacency = build_adjacency_matrix(topology)
        else:
            adjacency = np.asarray(adjacency, dtype=np.float64)

        if adjacency.shape != (n_demes, n_demes):
            raise ValueError(
                f"adjacency shape mismatch: expected ({n_demes}, {n_demes}), got {adjacency.shape}"
            )

        self._name = name
        self._topology = topology
        self._adjacency = adjacency
        self._migration_mode: Literal["adjacency", "kernel"] = migration_mode
        self._migration_kernel = migration_kernel
        self._kernel_include_center = bool(kernel_include_center)
        self._migration_mode_code = 0 if migration_mode == "adjacency" else 1
        self._migration_rate = float(migration_rate)
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
    def migration_kernel(self) -> NDArray[np.float64] | None:
        """NDArray[np.float64] | None: Kernel used by topology-aware migration."""
        return self._migration_kernel

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
        """Return normalized outbound migration weights for one source deme."""
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
        """
        for deme_id, deme in enumerate(self._demes):
            new_fields = {
                "n_tick": int(tick),
                "individual_count": ind_all[deme_id],
            }
            if hasattr(deme.state, "sperm_storage"):
                new_fields["sperm_storage"] = sperm_all[deme_id]

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
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        cfg = export_fn()
        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            if deme_export() is not cfg:
                raise ValueError(
                    f"deme[{idx}] uses a different config object; current spatial runner requires a shared config"
                )
        return cfg

    def _migration_kernel_array(self) -> NDArray[np.float64]:
        """Return the migration kernel array expected by compiled kernels."""
        if self._migration_kernel is not None:
            return self._migration_kernel
        return np.zeros((1, 1), dtype=np.float64)

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

        hooks = self._demes[0].get_compiled_event_hooks()
        assert hooks.run_spatial_tick_fn is not None, "hooks.run_spatial_tick_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_tick_fn = cast(Callable[..., Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], int]], hooks.run_spatial_tick_fn)
        registry = hooks.registry
        config = self._shared_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()

        final_state_tuple, result = run_tick_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=int(self._tick),
            adjacency=self._adjacency,
            migration_mode=self._migration_mode_code,
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

        hooks = self._demes[0].get_compiled_event_hooks()
        assert hooks.run_spatial_fn is not None, "hooks.run_spatial_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_fn = cast(Callable[..., Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64], bool]], hooks.run_spatial_fn)
        registry = hooks.registry
        config = self._shared_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()

        final_state_tuple, _history, was_stopped = run_fn(
            ind_count_all=ind_all,
            sperm_store_all=sperm_all,
            config=config,
            registry=registry,
            tick=int(self._tick),
            n_ticks=int(n_steps),
            adjacency=self._adjacency,
            migration_mode=self._migration_mode_code,
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
