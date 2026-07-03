"""
Pattern matching system for genotypes and haploid genomes.

Provides regex-like pattern matching for genetic sequences:
- PatternElement: Base class for allele-level matching
- HaplotypePath: Pattern for a single DNA strand of one chromosome
- ChromosomePairPattern: Pattern for a pair of homologous chromosomes
- GenotypePattern: Pattern for a complete diploid genotype
- HaploidGenomePattern: Pattern for a complete haploid genome
- GenotypePatternParser: Parser for pattern syntax strings
- GenotypeSelector: Unified genotype selector for observation/filtering
"""

from .elements._base import PatternParseError
from .elements.atom import LabPattern
from .elements.diploid import ZygoteTypePattern
from .elements.haploid import GameteTypePattern
from .parser import GenotypePatternParser
from .selector import GenotypeSelector, resolve_zygote_type

__all__ = [
    "GameteTypePattern",
    "GenotypePatternParser",
    "GenotypeSelector",
    "LabPattern",
    "PatternParseError",
    "ZygoteTypePattern",
    "resolve_zygote_type",
]
