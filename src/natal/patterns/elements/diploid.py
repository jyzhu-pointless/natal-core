"""Diploid-level pattern elements: GenotypePattern, ZygoteTypePattern.

Provides :class:`GenotypePattern` (a complete diploid genotype pattern across
all chromosomes) and :class:`ZygoteTypePattern` (a genotype pattern paired
with an optional somatic-label constraint).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from natal.genetics import Genotype, Species

from .atom import LabPattern
from .chromosome import ChromosomePairPattern


class GenotypePattern:
    """Complete genotype pattern, optionally filtered by somatic label.

    The ``@lab`` suffix (e.g. ``A|a@cas9_high``) is parsed and stored in
    *lab* but is NOT checked by :meth:`matches` — label filtering is the
    caller's responsibility (e.g. ``GenotypeSelector``).  When *lab* is
    ``None`` the pattern effectively matches any label.
    """

    def __init__(
        self,
        chromosome_patterns: List[Optional[ChromosomePairPattern]],
        lab: Optional[LabPattern] = None,
    ):
        """Initialize a complete genotype pattern.

        Args:
            chromosome_patterns: List of ChromosomePairPattern (or None for
                omitted chromosomes).
            lab: Optional somatic-label constraint (parsed from ``@lab``).
        """
        self.chromosome_patterns = chromosome_patterns
        self.lab: Optional[LabPattern] = lab

    def matches(self, genotype: Genotype) -> bool:
        """Check if a genotype matches this pattern.

        Args:
            genotype: The Genotype to match.

        Returns:
            True if the genotype matches all specified chromosome patterns.
        """
        species = genotype.species

        for i, chr_pattern in enumerate(self.chromosome_patterns):
            if chr_pattern is None:
                # Omitted chromosome - no constraint
                continue

            # Get the haplotype pair for this chromosome
            chromosome = species.chromosomes[i]
            try:
                mat_hap = genotype.maternal.get_haplotype_for_chromosome(chromosome)
                pat_hap = genotype.paternal.get_haplotype_for_chromosome(chromosome)
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
        base = f"GenotypePattern([{', '.join(str(cp) if cp else 'None' for cp in self.chromosome_patterns)}])"
        return f"{base}@{self.lab}" if self.lab else base


class ZygoteTypePattern:
    """Pattern for a zygote (diploid genotype) with a slab (somatic) label.

    A zygote type pairs a :class:`GenotypePattern` with an optional
    :class:`LabPattern` parsed from the ``@slab`` suffix (e.g.
    ``A|a@infected``).  This is the slab-aware equivalent of
    ``GenotypePattern`` — it resolves to a ``(genotype_index, slab_index)``
    pair used for ZType indexing in config arrays.

    Supports both string and tuple construction::

        ZygoteTypePattern.parse("A|a@infected", species)
        ZygoteTypePattern.from_pair(genotype_obj, "infected", species)
    """

    def __init__(
        self,
        genotype: GenotypePattern,
        slab: Optional[LabPattern] = None,
    ):
        """Initialize a ZygoteTypePattern.

        Args:
            genotype: The genotype pattern to match.
            slab: Optional somatic-label pattern parsed from the ``@slab``
                suffix.
        """
        self.genotype = genotype
        self.slab: Optional[LabPattern] = slab

    @staticmethod
    def parse(pattern_str: str, species: Species) -> ZygoteTypePattern:
        """Parse a ZType pattern string like ``"A|a@infected"``.

        The ``@slab`` suffix is extracted; everything before it is parsed
        as a :class:`GenotypePattern`.
        """
        from natal.patterns.parser import GenotypePatternParser

        parser = GenotypePatternParser(species)
        # Inline _strip_lab logic to avoid protected-access warning.
        lab: Optional[LabPattern] = None
        base = pattern_str
        if "@" in pattern_str:
            idx = pattern_str.rindex("@")
            base = pattern_str[:idx].strip()
            suffix = pattern_str[idx + 1:].strip()
            if suffix:
                lab = LabPattern.parse(suffix)
        genotype = parser.parse(base)
        return ZygoteTypePattern(genotype, lab)

    @staticmethod
    def from_pair(
        genotype: Genotype,
        slab: str,
        species: Species,
    ) -> ZygoteTypePattern:
        """Build a ZygoteTypePattern from a (Genotype, slab_name) pair.

        Args:
            genotype: A Genotype instance.
            slab: Somatic label name.
            species: Species for genotype-string resolution.

        Returns:
            A ZygoteTypePattern matching the given genotype and slab.
        """
        from natal.patterns.parser import GenotypePatternParser

        parser = GenotypePatternParser(species)
        pattern = parser.parse(str(genotype))
        return ZygoteTypePattern(pattern, LabPattern(lab=slab))

    @staticmethod
    def from_slab_key(key: str, species: Species) -> ZygoteTypePattern:
        """Parse a genotype key that may include an ``@slab`` suffix.

        Splits at ``@``, canonicalizes the genotype part via
        ``species.get_genotype_from_str``, and creates a ZygoteTypePattern
        with the resolved slab suffix.

        Args:
            key: Genotype key like ``"A|a"`` or ``"A|a@infected"``.
            species: Species definition for genotype resolution.

        Returns:
            A ZygoteTypePattern with the canonical genotype and slab.
        """
        if "@" in key:
            idx = key.rindex("@")
            gt_part = key[:idx]
            slab_part = key[idx:]
        else:
            gt_part = key
            slab_part = ""
        gt = species.get_genotype_from_str(gt_part)
        return ZygoteTypePattern.parse(str(gt) + slab_part, species)

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
