"""Population query and lifecycle mixin for BasePopulation.

Extracted from :mod:`natal.population.base`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

import numpy as np

from natal.population._mixins._modifiers import ModifierPresetMixin

if TYPE_CHECKING:
    from natal.output.history import History
    from natal.output.observation import Observation
    from natal.population.base import BasePopulation


class OutputMixin(ModifierPresetMixin):
    """Mixin providing population queries, lifecycle management,
    and allele frequency computation.

    Inherits from :class:`ModifierPresetMixin` since ``reapply_preset_fitness``
    is called during ``reset``.
    """

    # Declared here for pyright visibility — host BasePopulation subclass
    # provides these at runtime.  Any is required because pyright does not
    # allow mixin attribute declarations to shadow base-class @property.
    _finished: bool  # type: ignore[assignment]  # host provides at runtime
    _state: Any  # type: ignore[assignment]  # host BasePopulation supplies the generic state
    _registry: Any  # type: ignore[assignment]  # host provides at runtime
    _observation: Observation | None  # type: ignore[assignment]  # host owns mutable policy
    _history_obj: History | None  # type: ignore[assignment]  # host owns mutable row storage
    _tick: int  # type: ignore[assignment]  # host owns mutable lifecycle state
    max_history: int  # type: ignore[assignment]  # host provides at runtime
    name: str  # type: ignore[assignment]  # host provides at runtime
    species: Any  # type: ignore[assignment]  # host provides at runtime

    # ── Abstract query methods (from base.py:944-957) ──────────────

    @abstractmethod
    def get_total_count(self) -> int:
        """Return the total number of individuals in the population."""
        ...

    @abstractmethod
    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        ...

    @abstractmethod
    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        ...

    # ── History management ─────────────────────────────────────────
    # From base.py:552-591

    def _record_current_snapshot(self, *, allow_existing: bool) -> None:
        """Commit the current state to the unique History container.

        Args:
            allow_existing: Whether an automatic run boundary may reuse the
                already-recorded current tick without writing a second row.

        Raises:
            RuntimeError: If History, Population state, or Observation is not
                initialized.
            ValueError: If a strict snapshot repeats or precedes the latest
                tick, or an automatic boundary is stale or has a different
                payload.
        """
        history_obj = self._history_obj
        if history_obj is None:
            raise RuntimeError("History is not initialized for this population.")
        tick = int(self._tick)
        if self._state is None:
            raise RuntimeError("Population state is not initialized.")
        state = self._state
        if history_obj.schema.mode == "observation":
            observation = self._observation
            if observation is None:
                raise RuntimeError("Observation is not initialized for this population.")
            values = observation.apply(state.individual_count)
            row = np.empty(history_obj.schema.row_size, dtype=np.float64)
            row[0] = float(tick)
            row[1:] = values.ravel()
        else:
            row = state.flatten_all()
        from natal.output.history import HistoryBatch

        batch = HistoryBatch(schema=history_obj.schema, rows=row[np.newaxis, :])
        if allow_existing:
            history_obj._append_continuation(  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation
                batch
            )
        else:
            if tick in history_obj.ticks:
                raise ValueError(f"History already contains tick {tick}.")
            history_obj._append(batch)  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation

    def clear_history(self) -> None:
        """Remove all rows while preserving the frozen History schema."""
        if self._history_obj is not None:
            self._history_obj.clear()

    def record_snapshot(self) -> None:
        """Record the current stable state into history.

        Must only be called when the engine is not running (between
        ``run()`` calls). Duplicate ticks are rejected.

        Raises:
            RuntimeError: If the population is currently running.
            ValueError: If the current tick is already recorded.
        """
        if getattr(self, "_running", False):
            raise RuntimeError(
                "Cannot record snapshot while the population is running."
            )
        self._record_current_snapshot(allow_existing=False)

    def restore_checkpoint(self, tick: int) -> None:
        """Restore population state from a raw-history record at *tick*.

        Only valid for raw-mode history.  Restores individual counts and
        (when applicable) sperm storage.  All records after *tick* are
        removed.

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
        state = self._state
        state.individual_count[:] = ic.reshape(state.individual_count.shape)
        if ss is not None and hasattr(state, "sperm_storage"):
            state.sperm_storage[:] = ss.reshape(state.sperm_storage.shape)
        self._state = state._replace(n_tick=restored_tick)
        self._tick = restored_tick
        history_obj.truncate(retain_until_tick=tick)

    def _process_kernel_history(
        self,
        history_new: Optional[np.ndarray],
        clear_history_on_start: bool
    ) -> None:
        """Process and append history array returned from simulation engine.

        Args:
            history_new: Engine rows with tick in the first column, or ``None``.
            clear_history_on_start: Whether to discard the existing timeline
                before committing the engine rows.

        Raises:
            RuntimeError: If the population History is not initialized.
            ValueError: If row shape, schema, ordering, or boundary payload is
                inconsistent with the existing History.
        """
        if history_new is None or history_new.shape[0] == 0:
            return

        if clear_history_on_start:
            self.clear_history()

        history_obj = self._history_obj
        if history_obj is None:
            raise RuntimeError("History is not initialized for this population.")
        from natal.output.history import HistoryBatch

        rows = history_new
        schema = history_obj.schema
        observation = schema.observation
        if (
            schema.mode == "observation"
            and observation is not None
            and observation.collapse_age
            and rows.shape[1] != schema.row_size
        ):
            pop = schema.population
            values = rows[:, 1:].reshape(
                rows.shape[0],
                observation.n_groups,
                pop.n_sexes,
                pop.n_ages,
            ).sum(axis=-1)
            collapsed = np.empty(
                (rows.shape[0], schema.row_size), dtype=np.float64
            )
            collapsed[:, 0] = rows[:, 0]
            collapsed[:, 1:] = values.reshape(rows.shape[0], -1)
            rows = collapsed

        history_obj._append_continuation(  # pyright: ignore[reportPrivateUsage]  # History owns flattened boundary validation
            HistoryBatch(schema=schema, rows=rows)
        )

    # ── Population queries ─────────────────────────────────────────
    # From base.py:963-982

    @property
    def total_population_size(self) -> int:
        """Total population size (alias of ``get_total_count``)."""
        return self.get_total_count()

    @property
    def total_females(self) -> int:
        """Total number of females (alias of ``get_female_count``)."""
        return self.get_female_count()

    @property
    def total_males(self) -> int:
        """Total number of males (alias of ``get_male_count``)."""
        return self.get_male_count()

    @property
    def sex_ratio(self) -> float:
        """Return the female-to-male ratio, or ``np.inf`` when male count is zero."""
        males = self.get_male_count()
        return self.get_female_count() / males if males > 0 else np.inf

    # ── Simulation lifecycle ───────────────────────────────────────
    # From base.py:1088-1133

    @property
    def is_finished(self) -> bool:
        """Whether the population is marked as finished (``finish=True``)."""
        return self._finished

    def finish_simulation(self) -> None:
        """
        End simulation, trigger the ``finish`` event, and lock the population.

        This method may be called by hooks for early termination.
        After calling it, ``step()``, ``run_tick()``, and ``run()`` cannot run again.

        Raises:
            RuntimeError: If the population is already finished.

        Examples:
            >>> def check_extinction(pop):
            ...     if pop.get_total_count() == 0:
            ...         print("Population extinct, finishing simulation.")
            ...         pop.finish_simulation()
            >>> pop.set_hook('late', check_extinction)
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has already finished."
            )

        self._finished = True
        self.trigger_event("finish")

    @abstractmethod
    def run(
        self,
        n_steps: int,
        record_every: Optional[int] = None,
        finish: bool = False
    ) -> BasePopulation[Any]:  # Any: mixin does not own the generic T_State parameter
        """
        Run multi-step evolution.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the population to its initial state."""
        pass

    # ── Allele frequency computation ───────────────────────────────
    # From base.py:1135-1187

    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        Compute frequencies of all alleles in the population, normalized per locus.

        Returns:
            Dict[str, float]: Mapping ``{allele_name: frequency}``.
            Frequencies are per-locus proportions in the range ``[0.0, 1.0]``.
        """
        if self._state is None or self._registry is None:
            return {}

        # 1. Initialize counters.
        allele_counts: Dict[str, float] = {}
        locus_totals: Dict[str, float] = {}  # locus_name -> total_count

        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                locus_totals[locus.name] = 0.0
                for gene in locus.alleles:
                    allele_counts[gene.name] = 0.0

        # 2. Aggregate genotype counts.
        # individual_count shape: (n_sexes, n_ages, n_genotypes)
        # Sum over sex and age to get total count per genotype.
        genotype_counts = self._state.individual_count.sum(axis=(0, 1))

        registry = self._registry
        for z_idx, (genotype, _slab) in enumerate(registry.index_to_ztype):
            count = genotype_counts[z_idx]
            if count <= 0:
                continue

            for chrom in self.species.chromosomes:
                for locus in chrom.loci:
                    mat, pat = genotype.get_alleles_at_locus(locus)
                    for allele in (mat, pat):
                        if allele is not None:
                            allele_counts[allele.name] += count
                            locus_totals[locus.name] += count

        # 3. Compute frequencies.
        frequencies: Dict[str, float] = {}
        for allele_name, count in allele_counts.items():
            # Lookup the locus total for this allele.
            # We do not keep a direct fast gene->locus reverse index here,
            # so we safely resolve via species.gene_index.
            gene = self.species.gene_index.get(allele_name)
            if gene and locus_totals[gene.locus.name] > 0:
                frequencies[allele_name] = count / locus_totals[gene.locus.name]
            else:
                frequencies[allele_name] = 0.0

        return frequencies
