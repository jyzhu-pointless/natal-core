"""Haploid-level pattern elements: GameteTypePattern, HaploidGenomePattern.

Provides :class:`GameteTypePattern` (a haplotype pattern paired with an
optional gamete-label constraint) and :class:`HaploidGenomePattern`
(a complete haploid genome pattern across all chromosomes).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from natal.genetics import HaploidGenome

from .atom import LabPattern
from .chromosome import HaplotypePath


class GameteTypePattern:
    """Pattern for a gamete (haploid genome) with optional label constraint.

    A gamete type pairs a :class:`HaplotypePath` (the genetic content across
    all chromosomes) with an optional :class:`LabPattern` parsed from the
    ``@lab`` suffix (e.g. ``A1/B1; C1@cas9_deposited``).

    Label matching is the caller's responsibility — this class simply stores
    both components so the parser doesn't silently discard the label.
    """

    def __init__(
        self,
        haplotype_path: HaplotypePath,
        lab: Optional[LabPattern] = None,
    ):
        """Initialize a GameteTypePattern.

        Args:
            haplotype_path: HaplotypePath pattern for the genetic content.
            lab: Optional gamete-label constraint.
        """
        self.haplotype_path = haplotype_path
        self.lab: Optional[LabPattern] = lab

    def __repr__(self) -> str:
        """Return a string representation of this gamete type pattern."""
        base = f"GameteTypePattern({self.haplotype_path!r})"
        return f"{base}@{self.lab}" if self.lab else base


class HaploidGenomePattern:
    """Pattern for a HaploidGenome, optionally filtered by gamete label."""

    def __init__(
        self,
        haplotype_patterns: List[Optional[HaplotypePath]],
        lab: Optional[LabPattern] = None,
    ):
        """Initialize a haploid genome pattern.

        Args:
            haplotype_patterns: List of HaplotypePath for each chromosome.
            lab: Optional gamete-label constraint (parsed from ``@lab``).
        """
        self.haplotype_patterns = haplotype_patterns
        self.lab: Optional[LabPattern] = lab

    def matches(self, haploid_genome: HaploidGenome) -> bool:
        """Check if a haploid genome matches this pattern.

        Args:
            haploid_genome: The HaploidGenome to match.

        Returns:
            True if the haploid genome matches all specified patterns.
        """
        species = haploid_genome.species

        for i, haplotype_pattern in enumerate(self.haplotype_patterns):
            if haplotype_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype for this chromosome
            chromosome = species.chromosomes[i]
            try:
                haplotype = haploid_genome.get_haplotype_for_chromosome(chromosome)
            except (AttributeError, KeyError, IndexError):
                return False

            if not haplotype_pattern.matches(haplotype):
                return False

        return True

    def to_filter(self) -> Callable[[HaploidGenome], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a HaploidGenome and returns bool.
        """
        return lambda genome: self.matches(genome)

    def __repr__(self) -> str:
        """Return a string representation of this haploid genome pattern."""
        base = f"HaploidGenomePattern([{', '.join(str(hp) if hp else 'None' for hp in self.haplotype_patterns)}])"
        return f"{base}@{self.lab}" if self.lab else base
