"""Diploid-level pattern elements: GenotypePattern, ZygoteTypePattern.

Provides :class:`GenotypePattern` (a complete diploid genotype pattern across
all chromosomes) and :class:`ZygoteTypePattern` (a genotype pattern paired
with an optional somatic-label constraint).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from natal.frontend.genetics import Genotype

from .._groups import chromosome_groups, group_haplotype
from .atom import LabPattern
from .chromosome import ChromosomePairPattern


class GenotypePattern:
    """Complete genotype pattern across all chromosome groups.

    Matches genetic content only.  A ``Genotype`` has no label, so this
    pattern carries none: the ``@slab`` suffix belongs to
    :class:`ZygoteTypePattern` (and to ``IndividualSelector(ztype=...)``),
    the types that compose a genotype pattern with a label pattern.
    """

    def __init__(
        self,
        chromosome_patterns: List[Optional[ChromosomePairPattern]],
    ):
        """Initialize a complete genotype pattern.

        Args:
            chromosome_patterns: List of ChromosomePairPattern (or None for
                omitted chromosomes).
        """
        self.chromosome_patterns = chromosome_patterns

    def matches(self, genotype: Genotype) -> bool:
        """Check if a genotype matches this pattern.

        Args:
            genotype: The Genotype to match.

        Returns:
            True if the genotype matches all specified chromosome patterns.
        """
        groups = chromosome_groups(genotype.species)

        for i, chr_pattern in enumerate(self.chromosome_patterns):
            if chr_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype pair for this chromosome
            try:
                mat_hap = group_haplotype(genotype.maternal, groups[i])
                pat_hap = group_haplotype(genotype.paternal, groups[i])
            except (AttributeError, KeyError, IndexError, ValueError):
                return False

            if not chr_pattern.matches((mat_hap, pat_hap)):
                return False

        return True

    def to_filter(self) -> Callable[[Genotype], bool]:
        """Convert to a filter function for use in rules.

        Returns:
            A callable that takes a Genotype and returns bool.
        """
        return lambda genotype: self.matches(genotype)

    def __repr__(self) -> str:
        """Return a string representation of this genotype pattern."""
        return f"GenotypePattern([{', '.join(str(cp) if cp else 'None' for cp in self.chromosome_patterns)}])"


class ZygoteTypePattern:
    """Pattern for a zygote (diploid genotype) with a slab (somatic) label.

    A zygote type pairs a :class:`GenotypePattern` with an optional
    :class:`LabPattern` parsed from the ``@slab`` suffix (e.g.
    ``A|a@infected``).  This is the slab-aware equivalent of
    ``GenotypePattern`` — it resolves to a ``(genotype_index, slab_index)``
    pair used for ZType indexing in config arrays.

    Built by the unified selector entry::

        parse_selector("A|a@infected", species=species)   # kind="ztype"
    """

    def __init__(
        self,
        genotype: GenotypePattern,
        slab: Optional[LabPattern] = None,
        source_text: Optional[str] = None,
    ):
        """Initialize a ZygoteTypePattern.

        Args:
            genotype: The genotype pattern to match.
            slab: Optional somatic-label pattern parsed from the ``@slab``
                suffix.
        """
        self.genotype = genotype
        self.slab: Optional[LabPattern] = slab
        # Retain the grammar spelling so a structured selector can later be
        # consumed as a conversion target without round-tripping ``repr``.
        self.source_text = source_text

    def matches(self, genotype: Genotype, slab_label: str = "default") -> bool:
        """Check if this pattern matches a (genotype, slab_label) pair."""
        if not self.genotype.matches(genotype):
            return False
        if self.slab is not None:
            return self.slab.matches(slab_label)
        return True

    def __repr__(self) -> str:
        """Return a string representation of this zygote type pattern."""
        base = f"ZygoteTypePattern({self.genotype!r})"
        return f"{base}@{self.slab}" if self.slab else base
