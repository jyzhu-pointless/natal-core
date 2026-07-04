"""Population query and lifecycle mixin for BasePopulation.

Extracted from :mod:`natal.population.base`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from natal.population._mixins._modifiers import ModifierPresetMixin

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


class OutputMixin(ModifierPresetMixin):
    """Mixin providing population queries, lifecycle management,
    and allele frequency computation.

    Inherits from :class:`ModifierPresetMixin` since ``reapply_preset_fitness``
    is called during ``reset``.
    """

    # Declared here so pyright knows these come from the host class.
    _finished: bool  # type: ignore[assignment]
    _state: Any  # type: ignore[assignment]
    _registry: Any  # type: ignore[assignment]
    _history: List[Tuple[int, np.ndarray]]  # type: ignore[assignment]
    max_history: int  # type: ignore[assignment]
    name: str  # type: ignore[assignment]
    species: Any  # type: ignore[assignment]

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

    def _enforce_history_limit(self) -> None:
        """Ensure history size does not exceed max_history by dropping oldest entries."""
        if self.max_history > 0:
            excess = len(self._history) - self.max_history
            if excess > 0:
                self._history = self._history[excess:]

    @abstractmethod
    def clear_history(self) -> None:
        """Clear all recorded history states.

        Subclasses must implement this to reset their history storage
        (e.g., ``_history`` list and any subclass-specific history buffers).
        """
        pass

    def _process_kernel_history(
        self,
        history_new: Optional[np.ndarray],
        clear_history_on_start: bool
    ) -> None:
        """Process and append history array returned from simulation engine.

        Handles duplication checking (overlapping start/end ticks) and enforces limit.
        """
        if history_new is None or history_new.shape[0] == 0:
            return

        if clear_history_on_start:
            self.clear_history()

        for row_idx in range(history_new.shape[0]):
            row = history_new[row_idx, :]
            tick = int(row[0])
            # Skip duplicate entry if continuing history (overlap check)
            if not clear_history_on_start and self._history and self._history[-1][0] == tick:
                continue
            self._history.append((tick, row.copy()))

        self._enforce_history_limit()

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
    ) -> BasePopulation[Any]:
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
