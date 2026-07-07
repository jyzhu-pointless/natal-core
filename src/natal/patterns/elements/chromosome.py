"""Chromosome-level pattern elements: HaplotypePath, ChromosomePairPattern.

Provides :class:`HaplotypePath` (pattern for a single haplotype covering
all loci on a chromosome) and :class:`ChromosomePairPattern` (pattern for
a pair of homologous chromosomes with optional unordered matching).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Tuple

from natal.genetics import Haplotype

from ._base import PatternElement


class HaplotypePath:
    """Pattern for a single Haplotype (one DNA strand of one chromosome).

    The ``@lab`` suffix (e.g. ``A/B@cas9_deposited``) is parsed and stored
    on the containing :class:`~natal.patterns.elements.diploid.GenotypePattern`,
    NOT on HaplotypePath — a chromosomal haplotype has no intrinsic label.

    Attributes:
        locus_patterns (Sequence[PatternElement]): Pattern elements, one
            per locus in chromosome order.
    """

    def __init__(
        self,
        locus_patterns: Sequence[PatternElement],
    ):
        """Initialize a haplotype path pattern.

        Args:
            locus_patterns: Pattern elements, one per locus in chromosome
                order.

        Note:
            The ``@lab`` suffix is stripped by the parser and stored on
            the containing GenotypePattern, NOT on HaplotypePath.
            A chromosomal haplotype has no intrinsic gamete label.
        """
        self.locus_patterns = locus_patterns

    def matches(self, haplotype: Haplotype) -> bool:
        """Check if a haplotype matches this pattern.

        Args:
            haplotype: The Haplotype to match.

        Returns:
            True if all loci match.
        """
        # Get loci from the haplotype's chromosome
        loci = haplotype.chromosome.loci

        if len(self.locus_patterns) != len(loci):
            return False

        for pattern_elem, locus in zip(self.locus_patterns, loci):
            gene = haplotype.get_gene_at_locus(locus)
            if not pattern_elem.matches(gene):
                return False

        return True

    def to_filter(self) -> Callable[[Haplotype], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a Haplotype and returns bool.
        """
        return lambda haplotype: self.matches(haplotype)

    def __repr__(self) -> str:
        """Return a string representation of this HaplotypePath."""
        return f"HaplotypePath([{', '.join(str(lp) for lp in self.locus_patterns)}])"


class ChromosomePairPattern:
    """Pattern for a pair of homologous chromosomes.

    Matches one chromosome pair (maternal and paternal haplotypes).
    Supports ordered (``|``) and unordered (``::``) matching.

    Attributes:
        maternal_pattern (HaplotypePath): Pattern for the maternal haplotype.
        paternal_pattern (HaplotypePath): Pattern for the paternal haplotype.
        unordered (bool): If True, maternal/paternal order is ignored.
        explicit_grouping (bool): If True, this pattern was explicitly
            grouped with ``()`` in the pattern string.
    """

    def __init__(
        self,
        maternal_pattern: HaplotypePath,
        paternal_pattern: HaplotypePath,
        unordered: bool = False,
        explicit_grouping: bool = False
    ):
        """Initialize a chromosome pair pattern.

        Args:
            maternal_pattern: HaplotypePath for maternal haplotype.
            paternal_pattern: HaplotypePath for paternal haplotype.
            unordered: If True, use :: ordering (match either order).
            explicit_grouping: If True, this pattern was explicitly grouped with ().
        """
        self.maternal_pattern = maternal_pattern
        self.paternal_pattern = paternal_pattern
        self.unordered = unordered
        self.explicit_grouping = explicit_grouping

    def matches(self, haplotype_pair: Tuple[Haplotype, Haplotype]) -> bool:
        """Check if a pair of haplotypes (one chromosome pair) matches.

        Args:
            haplotype_pair: Tuple of (maternal_haplotype, paternal_haplotype).

        Returns:
            True if the haplotype pair matches.
        """
        mat_hap, pat_hap = haplotype_pair

        if self.unordered:
            # Try both orderings
            match_straight = (
                self.maternal_pattern.matches(mat_hap) and
                self.paternal_pattern.matches(pat_hap)
            )
            match_reversed = (
                self.maternal_pattern.matches(pat_hap) and
                self.paternal_pattern.matches(mat_hap)
            )
            return match_straight or match_reversed
        else:
            # Strict ordering: maternal | paternal
            return (
                self.maternal_pattern.matches(mat_hap) and
                self.paternal_pattern.matches(pat_hap)
            )

    def to_filter(self) -> Callable[[Tuple[Haplotype, Haplotype]], bool]:
        """Convert to a filter function.

        Returns:
            A callable that takes a haplotype pair and returns bool.
        """
        return lambda pair: self.matches(pair)

    def __repr__(self) -> str:
        """Return a string representation of this chromosome pair pattern."""
        sep = "::" if self.unordered else "|"
        return f"ChromosomePair({self.maternal_pattern} {sep} {self.paternal_pattern})"
