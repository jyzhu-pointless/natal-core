"""Genetics subpackage — genetic structures and entities.

Re-exports all public symbols from the structures/ and entities/ subdirectories.
Formerly known as genetic_structures.py and genetic_entities.py.
"""

from .entities.gene import Gene
from .entities.genotype import (
    Genotype,
    compute_recombinant_haplotypes,
    compute_recombinant_haplotypes_with_alleles,
)
from .entities.haplotype import (
    HaploidGenotype,
    Haplotype,
    create_chromosome_from_allele_names,
    create_haplotype_from_allele_names,
)
from .structures._helpers import build_compression_mask
from .structures._types import SexChromosomeType
from .structures.chromosome import Chromosome
from .structures.chromosome_map import RecombinationMap
from .structures.locus import Locus
from .structures.species import Species, SpeciesConfigBlueprint

# Module-level aliases (backward compatibility)
Allele = Gene
HaploidGenome = HaploidGenotype
Genome = Genotype
DiploidGenome = Genotype
DiploidGenotype = Genotype
Linkage = Chromosome
GenomeTemplate = Species
Karyotype = Species

__all__ = [
    "SexChromosomeType",
    "Species",
    "SpeciesConfigBlueprint",
    "Chromosome",
    "Linkage",
    "RecombinationMap",
    "Locus",
    "Gene",
    "Allele",
    "Haplotype",
    "HaploidGenotype",
    "HaploidGenome",
    "Genotype",
    "Genome",
    "DiploidGenome",
    "DiploidGenotype",
    "GenomeTemplate",
    "Karyotype",
    "create_haplotype_from_allele_names",
    "create_chromosome_from_allele_names",
    "compute_recombinant_haplotypes",
    "compute_recombinant_haplotypes_with_alleles",
    "build_compression_mask",
]
