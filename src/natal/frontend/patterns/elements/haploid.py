"""Haploid-level pattern elements: GameteTypePattern, HaploidGenomePattern.

Provides :class:`GameteTypePattern` (a complete haploid genome pattern paired
with an optional gamete-label constraint) and :class:`HaploidGenomePattern`
(a complete haploid genome pattern across all chromosomes).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from natal.frontend.genetics import HaploidGenome

from .._groups import chromosome_groups, group_haplotype
from .atom import LabPattern
from .chromosome import HaplotypePath


class GameteTypePattern:
    """Pattern for a gamete (haploid genome) with optional label constraint.

    A gamete type pairs a complete :class:`HaploidGenomePattern` (the genetic
    content across all chromosomes) with an optional :class:`LabPattern`
    parsed from the ``@glab`` suffix (e.g. ``A1/B1; C1@cas9_deposited``).
    Keeping the content a genome pattern rather than a flattened
    :class:`HaplotypePath` means a gamete selector describes each chromosome
    the same way a content-only haploid pattern does.

    Label matching is the caller's responsibility — this class simply stores
    both components so the parser doesn't silently discard the label.
    """

    def __init__(
        self,
        genome: HaploidGenomePattern,
        glab: Optional[LabPattern] = None,
    ):
        """Initialize a GameteTypePattern.

        Args:
            genome: Complete haploid genome pattern for the genetic content.
            glab: Optional gamete-label constraint.
        """
        self.genome = genome
        self.glab: Optional[LabPattern] = glab

    def __repr__(self) -> str:
        """Return a string representation of this gamete type pattern."""
        base = f"GameteTypePattern({self.genome!r})"
        return f"{base}@{self.glab}" if self.glab else base


class HaploidGenomePattern:
    """Complete haploid genome pattern across all chromosome groups.

    Matches genetic content only.  A ``HaploidGenome`` has no label, so this
    pattern carries none: the ``@glab`` suffix belongs to
    :class:`GameteTypePattern`, which composes a genome pattern with a label
    pattern.
    """

    def __init__(
        self,
        haplotype_patterns: List[Optional[HaplotypePath]],
    ):
        """Initialize a haploid genome pattern.

        Args:
            haplotype_patterns: List of HaplotypePath for each chromosome.
        """
        self.haplotype_patterns = haplotype_patterns

    def matches(self, haploid_genome: HaploidGenome) -> bool:
        """Check if a haploid genome matches this pattern.

        Args:
            haploid_genome: The HaploidGenome to match.

        Returns:
            True if the haploid genome matches all specified patterns.
        """
        groups = chromosome_groups(haploid_genome.species)

        for i, haplotype_pattern in enumerate(self.haplotype_patterns):
            if haplotype_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype for this chromosome
            try:
                haplotype = group_haplotype(haploid_genome, groups[i])
            except (AttributeError, KeyError, IndexError, ValueError):
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
        return f"HaploidGenomePattern([{', '.join(str(hp) if hp else 'None' for hp in self.haplotype_patterns)}])"
