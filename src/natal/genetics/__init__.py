"""Genetics subpackage — genetic structures and entities.

This subpackage defines the complete genetic architecture modelling layer:
- **Structures**: Blueprint types (:class:`~natal.genetics.structures.species.Species`,
  :class:`~natal.genetics.structures.chromosome.Chromosome`,
  :class:`~natal.genetics.structures.locus.Locus`) that define the
  hierarchical organisation of a species' genome.
- **Entities**: Runtime instances (:class:`~natal.genetics.entities.gene.Gene`,
  :class:`~natal.genetics.entities.haplotype.Haplotype`,
  :class:`~natal.genetics.entities.haplotype.HaploidGenotype`,
  :class:`~natal.genetics.entities.genotype.Genotype`) that are bound
  to structures and support operations such as gamete production with
  recombination.

Re-exports all public symbols from the ``structures/`` and ``entities/``
subdirectories.  Formerly known as ``genetic_structures.py`` and
``genetic_entities.py``.
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
