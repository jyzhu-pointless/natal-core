"""
Pattern matching system for genotypes and haploid genomes.

Provides regex-like pattern matching for genetic sequences:
- PatternElement: Base class for allele-level matching
- HaplotypePath: Pattern for a single DNA strand of one chromosome
- ChromosomePairPattern: Pattern for a pair of homologous chromosomes
- GenotypePattern: Pattern for a complete diploid genotype
- HaploidGenomePattern: Pattern for a complete haploid genome
- parse_selector / parse_target: The two semantic entries — one for
  matching, one for keep-or-replace conversion targeting
"""

from .elements._base import PatternParseError
from .elements.atom import LabPattern
from .elements.diploid import ZygoteTypePattern
from .elements.haploid import GameteTypePattern
from .entries import SelectorKind, parse_selector, parse_target
from .individual_selector import IndividualSelector
from .selector import resolve_zygote_type

__all__ = [
    "GameteTypePattern",
    "IndividualSelector",
    "LabPattern",
    "PatternParseError",
    "SelectorKind",
    "ZygoteTypePattern",
    "parse_selector",
    "parse_target",
    "resolve_zygote_type",
]
